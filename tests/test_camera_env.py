"""Camera index names must agree across the SFT and RLToken clients."""

import pytest

from sft.camera_env import camera_setting
from windows.deploy_rlt import configure


def test_accepts_documented_camera_names_and_existing_aliases():
    cfg = {"cameras": {"front": {"index_or_path": 9}, "side": {"index_or_path": 9}}}
    configure(cfg, {"FRONT_CAMERA": "0", "CAM_FRONT": "0", "SIDE_CAMERA": "1"})
    assert cfg["cameras"]["front"]["index_or_path"] == 0
    assert cfg["cameras"]["side"]["index_or_path"] == 1
    assert camera_setting({"CAM_FRONT": "2"}, "FRONT_CAMERA", "CAM_FRONT") == ("CAM_FRONT", "2")


def test_rejects_conflicting_camera_assignments():
    with pytest.raises(ValueError, match="Conflicting camera assignments"):
        camera_setting({"FRONT_CAMERA": "0", "CAM_FRONT": "1"}, "FRONT_CAMERA", "CAM_FRONT")
    with pytest.raises(ValueError, match="Conflicting camera assignments"):
        configure({"cameras": {"front": {}, "side": {}}},
                  {"FRONT_CAMERA": "0", "CAM_FRONT": "1"})
