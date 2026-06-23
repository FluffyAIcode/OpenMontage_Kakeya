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
        "for a in ['--frames','--fw-width','--fw-height','--fw-frames','--proposer-steps','--refine-steps','--seed','--out-width','--out-height','--fps','--interpolate','--interp-method','--refine-mode','--seconds','--chunks','--chunk-frames','--chunk-overlap']:\n"
        "    ap.add_argument(a)\n"
        "ap.add_argument('--no-refine',action='store_true')\n"
        "ap.add_argument('--single-refine',action='store_true')\n"
        "ap.add_argument('--refine-spread')\n"
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


def _mode_fixture(tmp_path, monkeypatch, *, workers, mode_env):
    """Spin up the gateway with a fake orch that records the flags it was called with."""
    fake = tmp_path / "fake_orch.py"
    fake.write_text(
        "import argparse,json,os,pathlib\n"
        "ap=argparse.ArgumentParser()\n"
        "for a in ['--prompt','--out','--frames','--fw-width','--fw-height','--fw-frames','--proposer-steps','--refine-steps','--seed','--out-width','--out-height','--refine-spread','--fps','--interpolate','--interp-method','--refine-mode','--seconds','--chunks','--chunk-frames','--chunk-overlap']: ap.add_argument(a)\n"
        "ap.add_argument('--no-refine',action='store_true')\n"
        "ap.add_argument('--single-refine',action='store_true')\n"
        "x=ap.parse_args()\n"
        "flags={'single_refine':x.single_refine,'no_refine':x.no_refine,'refine_spread':x.refine_spread,'workers':os.environ.get('WAN_WORKERS',''),'fps':x.fps,'interpolate':x.interpolate,'interp_method':x.interp_method,'refine_mode':x.refine_mode,'out_width':x.out_width,'frames':x.frames,'seconds':x.seconds,'refine_steps':x.refine_steps,'chunks':x.chunks,'chunk_frames':x.chunk_frames,'chunk_overlap':x.chunk_overlap}\n"
        "print('[orch]   refine:  100%',flush=True)\n"
        "p=pathlib.Path(x.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'MODEMP4')\n"
        "print('FLAGS '+json.dumps(flags),flush=True)\n"
        "print('ORCH_DONE '+json.dumps({'out':x.out,'mode':'x'}),flush=True)\n"
    )
    monkeypatch.setenv("WAN_WORKERS", workers)
    monkeypatch.delenv("AGENT_GATEWAY_WORKER_POOL", raising=False)
    if mode_env is None:
        monkeypatch.delenv("AGENT_GATEWAY_MODE", raising=False)
    else:
        monkeypatch.setenv("AGENT_GATEWAY_MODE", mode_env)
    monkeypatch.setenv("AGENT_GATEWAY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv("AGENT_GATEWAY_API_KEY", raising=False)
    sys.path.insert(0, str(REPO / "services" / "agent_gateway"))
    import server
    importlib.reload(server)
    server.ORCH = fake
    return TestClient(server.app), server


def _job_flags(c, jid):
    j = _wait(c, jid)
    assert j["status"] == "done", j
    line = next(l for l in j["log"] if l.startswith("FLAGS "))
    import json as _json
    return _json.loads(line[len("FLAGS "):])


def test_auto_mode_lets_orchestrator_decide(tmp_path, monkeypatch):
    """Default (no mode env), >1 worker -> 'auto': pass ALL workers, NO single/no-refine flag."""
    c, server = _mode_fixture(tmp_path, monkeypatch, workers="10.0.0.1:50051,10.0.0.2:50051", mode_env=None)
    assert server.GW_MODE == "auto"
    assert c.get("/healthz").json()["mode"] == "auto"
    jid = c.post("/v1/videos", json={"prompt": "auto"}).json()["job_id"]
    f = _job_flags(c, jid)
    assert f["single_refine"] is False and f["no_refine"] is False and f["refine_spread"] is None
    assert "," in f["workers"]


def test_pipeline_mode_forces_single_refine(tmp_path, monkeypatch):
    """AGENT_GATEWAY_MODE=pipeline -> head=proposer + one refiner via --single-refine."""
    c, server = _mode_fixture(tmp_path, monkeypatch, workers="10.0.0.1:50051,10.0.0.2:50051", mode_env="pipeline")
    assert server.GW_MODE == "pipeline" and server.PIPELINE_MODE is True
    assert c.get("/healthz").json()["mode"] == "pipeline"
    f = _job_flags(c, c.post("/v1/videos", json={"prompt": "pipe"}).json()["job_id"])
    assert f["single_refine"] is True and f["no_refine"] is False


def test_distributed_mode_spreads_tiles(tmp_path, monkeypatch):
    """AGENT_GATEWAY_MODE=distributed -> head=proposer, ALL refiners share tiles (roundrobin)."""
    c, server = _mode_fixture(tmp_path, monkeypatch,
                              workers="10.0.0.1:50051,10.0.0.2:50051,10.0.0.3:50051", mode_env="distributed")
    assert server.GW_MODE == "distributed"
    assert c.get("/healthz").json()["mode"] == "distributed"
    f = _job_flags(c, c.post("/v1/videos", json={"prompt": "dist"}).json()["job_id"])
    assert f["refine_spread"] == "roundrobin" and f["single_refine"] is False and f["no_refine"] is False


def test_quality_presets(tmp_path, monkeypatch):
    """quality preset maps to orchestrator knobs; 'high' is seam-free (refine_mode=single, hi-res)."""
    c, server = _mode_fixture(tmp_path, monkeypatch, workers="10.0.0.1:50051,10.0.0.2:50051", mode_env="distributed")
    # standard (default): no forced refine_mode, baseline res/fps
    fs = _job_flags(c, c.post("/v1/videos", json={"prompt": "standard clip"}).json()["job_id"])
    assert fs["out_width"] == "832" and fs["fps"] == "12" and fs["interpolate"] == "1"
    assert fs["refine_mode"] is None  # let the gateway mode/orchestrator decide
    # high: seam-free full-frame generative refine on the best refiner, hi-res, smoother
    fh = _job_flags(c, c.post("/v1/videos", json={"prompt": "hero clip", "quality": "high"}).json()["job_id"])
    assert fh["out_width"] == "1280" and fh["fps"] == "24" and fh["interpolate"] == "2"
    assert fh["refine_mode"] == "single" and fh["refine_steps"] == "24"
    assert fh["interp_method"] == "mci"   # high uses optical-flow interpolation
    # draft: proposer-only direct
    fd = _job_flags(c, c.post("/v1/videos", json={"prompt": "draft clip", "quality": "draft"}).json()["job_id"])
    assert fd["refine_mode"] == "direct"


def test_duration_seconds_and_overrides(tmp_path, monkeypatch):
    """seconds>0 drives frame count (= round(seconds*fps)); explicit knobs override the preset."""
    c, server = _mode_fixture(tmp_path, monkeypatch, workers="10.0.0.1:50051,10.0.0.2:50051", mode_env="auto")
    f = _job_flags(c, c.post("/v1/videos", json={"prompt": "long clip", "seconds": 4, "fps": 16,
                                                 "interpolate": 2}).json()["job_id"])
    assert f["fps"] == "16" and f["interpolate"] == "2"
    assert f["seconds"] == "4.0" and f["frames"] == "64"  # 4s * 16fps


def test_longform_chunks_explicit_and_from_seconds(tmp_path, monkeypatch):
    """chunks>1 enables long-form; seconds beyond one chunk auto-derives the chunk count."""
    c, server = _mode_fixture(tmp_path, monkeypatch, workers="10.0.0.1:50051,10.0.0.2:50051", mode_env="auto")
    # explicit chunks
    f = _job_flags(c, c.post("/v1/videos", json={"prompt": "a long pan over mountains",
                                                 "chunks": 3, "chunk_frames": 25, "chunk_overlap": 4}).json()["job_id"])
    assert f["chunks"] == "3" and f["chunk_frames"] == "25" and f["chunk_overlap"] == "4"
    # longform + seconds -> derived chunks: 8s * 12fps = 96 frames; net=25-4=21 -> ceil((96-4)/21)=5
    f2 = _job_flags(c, c.post("/v1/videos", json={"prompt": "an 8 second flythrough",
                                                  "longform": True, "seconds": 8}).json()["job_id"])
    assert f2["chunks"] == "5"


def test_dynamic_workers_file_membership(tmp_path, monkeypatch):
    """AGENT_GATEWAY_WORKERS_FILE -> live membership: an ephemeral vast refiner can be added/removed
    by rewriting the file, with NO gateway restart. Falls back to WAN_WORKERS when unreadable."""
    wf = tmp_path / "wan_workers"
    wf.write_text("127.0.0.1:50051,192.168.68.51:50051\n")  # 2-Mac fallback to start
    monkeypatch.setenv("AGENT_GATEWAY_WORKERS_FILE", str(wf))  # set BEFORE the fixture's reload
    c, server = _mode_fixture(tmp_path, monkeypatch, workers="127.0.0.1:50051", mode_env="distributed")
    assert server.current_workers() == ["127.0.0.1:50051", "192.168.68.51:50051"]
    assert c.get("/healthz").json()["dynamic_workers"] is True
    # vast comes online: supervisor appends it -> gateway sees 3 workers on the NEXT job, no restart
    wf.write_text("127.0.0.1:50051,192.168.68.51:50051,127.0.0.1:50052\n")
    f = _job_flags(c, c.post("/v1/videos", json={"prompt": "vast online"}).json()["job_id"])
    assert f["workers"] == "127.0.0.1:50051,192.168.68.51:50051,127.0.0.1:50052"
    # vast released: supervisor drops it -> next job is back to 2-Mac, still no restart
    wf.write_text("127.0.0.1:50051,192.168.68.51:50051\n")
    f2 = _job_flags(c, c.post("/v1/videos", json={"prompt": "vast gone"}).json()["job_id"])
    assert f2["workers"] == "127.0.0.1:50051,192.168.68.51:50051"


def test_pool_mode_two_macs(tmp_path, monkeypatch):
    """Two Thunderbolt-bridged Macs -> pool mode: 2 jobs run + complete in parallel."""
    fake = tmp_path / "fake_orch.py"
    fake.write_text(
        "import argparse,json,pathlib,time\n"
        "ap=argparse.ArgumentParser()\n"
        "for a in ['--prompt','--out','--frames','--fw-width','--fw-height','--fw-frames','--proposer-steps','--refine-steps','--seed','--out-width','--out-height','--fps','--interpolate','--interp-method','--refine-mode','--seconds','--chunks','--chunk-frames','--chunk-overlap']: ap.add_argument(a)\n"
        "ap.add_argument('--no-refine',action='store_true')\n"
        "ap.add_argument('--single-refine',action='store_true')\n"
        "ap.add_argument('--refine-spread')\n"
        "x=ap.parse_args()\n"
        "assert x.no_refine, 'pool mode must pass --no-refine'\n"
        "print('[orch]   generate:  100%',flush=True); time.sleep(0.5)\n"
        "p=pathlib.Path(x.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'POOLMP4')\n"
        "print('ORCH_DONE '+json.dumps({'out':x.out,'mode':'direct'}),flush=True)\n"
    )
    monkeypatch.setenv("WAN_WORKERS", "10.0.0.1:50051,10.0.0.2:50051")
    monkeypatch.setenv("AGENT_GATEWAY_WORKER_POOL", "1")
    monkeypatch.setenv("AGENT_GATEWAY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv("AGENT_GATEWAY_API_KEY", raising=False)
    sys.path.insert(0, str(REPO / "services" / "agent_gateway"))
    import server
    importlib.reload(server)
    server.ORCH = fake
    assert server.POOL_MODE is True
    c = TestClient(server.app)
    h = c.get("/healthz").json()
    assert h["pool_mode"] is True and h["parallel"] == 2
    j1 = c.post("/v1/videos", json={"prompt": "fox one"}).json()["job_id"]
    j2 = c.post("/v1/videos", json={"prompt": "fox two"}).json()["job_id"]
    out = {}
    for jid in (j1, j2):
        out[jid] = _wait(c, jid)
    assert all(o["status"] == "done" for o in out.values()), out
    assert c.get(f"/v1/jobs/{j1}/video").content == b"POOLMP4"
