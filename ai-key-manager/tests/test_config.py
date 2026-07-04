from pathlib import Path

from aikeys import config


def test_config_path_uses_appdata_on_windows(monkeypatch, tmp_path):
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(config.os, "name", "nt")

    assert config.config_path() == appdata / "aikeys" / "config.yaml"
