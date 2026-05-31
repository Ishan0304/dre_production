from pathlib import Path

import pytest

from core import ConfigLoader, LoadedConfig, PathConfig


def test_config_loader_loads_valid_yaml(tmp_path) -> None:
    config_path = tmp_path / "example.yaml"
    config_path.write_text("project:\n  name: dre_production\n", encoding="utf-8")

    loaded = ConfigLoader().load_yaml(config_path)

    assert loaded["project"]["name"] == "dre_production"


def test_config_loader_raises_for_invalid_yaml(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("project:\n  name: [broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML config file"):
        ConfigLoader().load_yaml(config_path)


def test_config_loader_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        ConfigLoader().load_yaml(tmp_path / "missing.yaml")


def test_build_path_config_uses_sensible_defaults(tmp_path) -> None:
    path_config = ConfigLoader().build_path_config({}, project_root=tmp_path)

    assert path_config == PathConfig(
        project_root=str(tmp_path.resolve()),
        output_root=str((tmp_path / "outputs").resolve()),
        artifacts_dir=str((tmp_path / "outputs" / "artifacts").resolve()),
        reports_dir=str((tmp_path / "outputs" / "reports").resolve()),
        data_dir=str((tmp_path / "data").resolve()),
    )


def test_resolve_config_includes_resolved_paths(tmp_path) -> None:
    raw_config = {
        "project": {"name": "dre_production"},
        "paths": {"output_root": "custom_outputs"},
    }

    loaded = ConfigLoader().resolve_config(raw_config, project_root=tmp_path)

    assert isinstance(loaded, LoadedConfig)
    assert loaded.config_name == "dre_production"
    assert loaded.resolved_config["paths"]["project_root"] == str(tmp_path.resolve())
    assert loaded.resolved_config["paths"]["output_root"] == str(
        (tmp_path / "custom_outputs").resolve()
    )
    assert loaded.path_config.output_root == str((tmp_path / "custom_outputs").resolve())


def test_resolve_config_does_not_mutate_raw_config(tmp_path) -> None:
    raw_config = {"paths": {"output_root": "outputs"}}
    original = {"paths": {"output_root": "outputs"}}

    ConfigLoader().resolve_config(raw_config, project_root=tmp_path)

    assert raw_config == original


def test_load_and_resolve_uses_file_stem_as_config_name(tmp_path) -> None:
    config_path = tmp_path / "ehr_example.yaml"
    config_path.write_text("paths:\n  data_dir: data/example\n", encoding="utf-8")

    loaded = ConfigLoader().load_and_resolve(config_path)

    assert loaded.config_name == "ehr_example"
    assert Path(loaded.path_config.data_dir).name == "example"
