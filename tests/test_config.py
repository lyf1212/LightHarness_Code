import json

import pytest

from lightagent.config import load_config
from lightagent.io import ROOT


def test_paths_survive_a_different_working_directory(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"blender": "${BLENDER_BIN}",
                               "sam3": {"repo": "third_party/sam3", "python": "python3"}}))
    monkeypatch.setenv("BLENDER_BIN", str(tmp_path / "blender"))
    monkeypatch.chdir(tmp_path)
    config = load_config(path)
    assert config["sam3"]["repo"] == str(ROOT / "third_party/sam3")
    assert config["blender"] == str(tmp_path / "blender")


def test_unset_path_variable_fails_before_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("LIGHTPROXY_MISSING_PATH", raising=False)
    path = tmp_path / "config.json"
    path.write_text('{"blender": "${LIGHTPROXY_MISSING_PATH}/blender"}')
    with pytest.raises(ValueError, match="Unresolved"):
        load_config(path)
