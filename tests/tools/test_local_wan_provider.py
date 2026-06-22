"""The WAN cluster is a registered `video_generation` provider tool (ADR 0013 re-anchor).

Verifies that `wan_video` (discoverable by `video_selector`, usable by pipelines/the agent)
routes through the unified local-video seam to the distributed-WAN cluster, in both Mac-only
DIRECT mode and vast-attached refine mode. Uses a FAKE orchestrator (no GPU). Run:

    pytest tests/tools/test_local_wan_provider.py -q
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def fake_orch(tmp_path):
    p = tmp_path / "fake_orch.py"
    p.write_text(
        "import argparse,json,pathlib\n"
        "ap=argparse.ArgumentParser()\n"
        "for a in ['--prompt','--out','--frames','--seed','--fw-width','--fw-height','--fw-frames']: ap.add_argument(a)\n"
        "ap.add_argument('--no-refine',action='store_true')\n"
        "x=ap.parse_args()\n"
        "pathlib.Path(x.out).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(x.out).write_bytes(b'MP4'+ (b'D' if x.no_refine else b'R'))\n"
        "print('ORCH_DONE '+json.dumps({'out':x.out,'mode':'direct' if x.no_refine else 'refine'}))\n"
    )
    return p


def _wan_tool(monkeypatch, fake_orch, **env):
    monkeypatch.setenv("DISTWAN_ORCH_PATH", str(fake_orch))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import tools.video._shared as shared
    importlib.reload(shared)
    import tools.video.wan_video as wv
    importlib.reload(wv)
    return wv.WanVideo()


def test_wan_video_is_video_generation_provider(monkeypatch, fake_orch):
    tool = _wan_tool(monkeypatch, fake_orch, WAN_WORKERS="127.0.0.1:50051")
    assert tool.capability == "video_generation"
    assert tool.provider == "wan"
    # configured cluster -> tool reports available without torch/diffusers
    from tools.base_tool import ToolStatus
    assert tool.get_status() == ToolStatus.AVAILABLE


def test_mac_only_direct(monkeypatch, fake_orch, tmp_path):
    tool = _wan_tool(monkeypatch, fake_orch, WAN_WORKERS="127.0.0.1:50051")
    out = tmp_path / "v.mp4"
    res = tool.execute({"prompt": "a red fox in snow", "width": 1280, "height": 720,
                        "num_frames": 25, "output_path": str(out)})
    assert res.success, res.error
    assert res.data["mode"] == "local_mlx_direct"
    assert res.data["refine"] is False
    assert out.read_bytes() == b"MP4D"  # --no-refine was passed


def test_vast_attached_refine(monkeypatch, fake_orch, tmp_path):
    tool = _wan_tool(monkeypatch, fake_orch, WAN_WORKERS="127.0.0.1:50051",
                     VAST_REFINE_WORKER="vast:50051")
    out = tmp_path / "v.mp4"
    res = tool.execute({"prompt": "a red fox in snow", "output_path": str(out)})
    assert res.success, res.error
    assert res.data["mode"] == "distributed_refine"
    assert res.data["refine"] is True
    assert "vast:50051" in res.data["workers"]
    assert out.read_bytes() == b"MP4R"  # refine path (no --no-refine)
