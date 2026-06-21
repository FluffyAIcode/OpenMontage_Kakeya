"""Offline smoke tests for the OpenMontage Agent Gateway.

No GPU and no real orchestrator: we point the gateway at a FAKE orchestrator script that
emits the same progress/ORCH_DONE protocol and writes a tiny mp4, so the full job lifecycle
(submit -> run -> poll -> download) is exercised deterministically. Run:

    pip install fastapi httpx pytest
    pytest tests/tools/test_agent_gateway.py -q
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Fake orchestrator: prints progress + ORCH_DONE and writes the --out file.
    fake = tmp_path / "fake_orch.py"
    fake.write_text(
        "import argparse,json,pathlib\n"
        "ap=argparse.ArgumentParser()\n"
        "for a in ['--prompt','--out']: ap.add_argument(a)\n"
        "for a in ['--frames','--fw-width','--fw-height','--fw-frames','--proposer-steps','--refine-steps','--seed']:\n"
        "    ap.add_argument(a)\n"
        "x=ap.parse_args()\n"
        "print('[orch]   framework:    5%',flush=True)\n"
        "print('[orch]   framework:  100%',flush=True)\n"
        "print('[orch]   tile(0, 0)@cuda:   50%',flush=True)\n"
        "p=pathlib.Path(x.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'FAKEMP4')\n"
        "print('ORCH_DONE '+json.dumps({'out':x.out,'canvas_px':[768,1472]}),flush=True)\n"
    )
    monkeypatch.setenv("WAN_WORKERS", "127.0.0.1:50051,127.0.0.1:55051")
    monkeypatch.setenv("AGENT_GATEWAY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv("AGENT_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_RUNTIME_CMD", raising=False)
    sys.path.insert(0, str(REPO / "services" / "agent_gateway"))
    import server  # noqa: WPS433
    importlib.reload(server)
    server.ORCH = fake
    return TestClient(server.app), server


def _wait(c, jid, timeout=15):
    for _ in range(int(timeout / 0.2)):
        j = c.get(f"/v1/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_healthz(client):
    c, _ = client
    h = c.get("/healthz").json()
    assert h["status"] == "ok"
    assert h["orchestrator"] is True


def test_video_job_end_to_end(client):
    c, _ = client
    r = c.post("/v1/videos", json={"prompt": "a red fox in snow"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    j = _wait(c, jid)
    assert j["status"] == "done", j
    assert j["pct"] == 1.0
    assert j["result"] == f"/v1/jobs/{jid}/video"
    vid = c.get(f"/v1/jobs/{jid}/video")
    assert vid.status_code == 200
    assert vid.content == b"FAKEMP4"


def test_progress_stages_parsed(client):
    c, _ = client
    jid = c.post("/v1/videos", json={"prompt": "x y z"}).json()["job_id"]
    j = _wait(c, jid)
    # log should contain both framework and tile lines we emitted
    joined = "\n".join(j["log"])
    assert "framework" in joined and "tile" in joined


def test_agent_mode_without_runtime_is_honest(client):
    c, _ = client
    jid = c.post("/v1/videos", json={"prompt": "make me a film", "mode": "agent"}).json()["job_id"]
    j = _wait(c, jid)
    assert j["status"] == "error"
    assert "agent runtime" in (j["error"] or "")


def test_api_key_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("WAN_WORKERS", "127.0.0.1:50051")
    monkeypatch.setenv("AGENT_GATEWAY_API_KEY", "secret123")
    sys.path.insert(0, str(REPO / "services" / "agent_gateway"))
    import server
    importlib.reload(server)
    c = TestClient(server.app)
    assert c.post("/v1/videos", json={"prompt": "blocked"}).status_code == 401
    ok = c.post("/v1/videos", json={"prompt": "allowed"}, headers={"X-API-Key": "secret123"})
    assert ok.status_code == 200


def test_job_404(client):
    c, _ = client
    assert c.get("/v1/jobs/nope").status_code == 404
