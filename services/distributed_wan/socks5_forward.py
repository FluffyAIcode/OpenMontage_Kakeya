"""Local TCP -> SOCKS5 -> remote forwarder (stdlib only).

Why this exists: on the vast H200 box, Tailscale runs in *userspace-networking*
mode (no /dev/net/tun in the container), so the kernel cannot route the 100.x
tailnet. tailscaled instead exposes a SOCKS5 proxy (default localhost:1055).
gRPC's Python stack speaks HTTP/2 but not SOCKS5, so we bridge:

    orchestrator -> localhost:<listen> -> [this forwarder] -> SOCKS5(1055) -> mac:50051

Every accepted connection performs a SOCKS5 CONNECT handshake to the target and
then pumps bytes both ways. No third-party deps (the vast box has neither socat
nor ncat). This is real transport, not a stub.

Run:
    python socks5_forward.py --listen 127.0.0.1:55051 \
        --socks 127.0.0.1:1055 --target 100.78.64.43:50051
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading


def _socks5_connect(socks_host: str, socks_port: int, dst_host: str, dst_port: int) -> socket.socket:
    s = socket.create_connection((socks_host, socks_port), timeout=30)
    # greeting: VER=5, 1 method, NO AUTH (0x00)
    s.sendall(b"\x05\x01\x00")
    resp = s.recv(2)
    if len(resp) != 2 or resp[0] != 0x05 or resp[1] != 0x00:
        s.close()
        raise OSError(f"SOCKS5 greeting failed: {resp!r}")
    # CONNECT request, ATYP=domain so tailscaled resolves the tailnet name/ip itself
    host_b = dst_host.encode()
    req = b"\x05\x01\x00\x03" + struct.pack("!B", len(host_b)) + host_b + struct.pack("!H", dst_port)
    s.sendall(req)
    rep = s.recv(4)
    if len(rep) < 4 or rep[1] != 0x00:
        code = rep[1] if len(rep) > 1 else -1
        s.close()
        raise OSError(f"SOCKS5 CONNECT refused (code={code})")
    # drain bound address (we ignore it): ATYP-dependent length
    atyp = rep[3]
    if atyp == 0x01:
        s.recv(4 + 2)
    elif atyp == 0x04:
        s.recv(16 + 2)
    elif atyp == 0x03:
        ln = s.recv(1)[0]
        s.recv(ln + 2)
    return s


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sk in (src, dst):
            try:
                sk.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle(client: socket.socket, socks: tuple[str, int], target: tuple[str, int]) -> None:
    try:
        upstream = _socks5_connect(socks[0], socks[1], target[0], target[1])
    except OSError as exc:
        print(f"[fwd] upstream connect failed: {exc}", flush=True)
        client.close()
        return
    threading.Thread(target=_pump, args=(client, upstream), daemon=True).start()
    _pump(upstream, client)


def _split(hp: str) -> tuple[str, int]:
    host, port = hp.rsplit(":", 1)
    return host, int(port)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:55051")
    ap.add_argument("--socks", default="127.0.0.1:1055")
    ap.add_argument("--target", required=True, help="tailnet host:port, e.g. 100.78.64.43:50051")
    args = ap.parse_args()

    lhost, lport = _split(args.listen)
    socks = _split(args.socks)
    target = _split(args.target)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lhost, lport))
    srv.listen(64)
    print(f"[fwd] {lhost}:{lport} -> socks5://{socks[0]}:{socks[1]} -> {target[0]}:{target[1]}", flush=True)
    while True:
        client, _ = srv.accept()
        threading.Thread(target=_handle, args=(client, socks, target), daemon=True).start()


if __name__ == "__main__":
    main()
