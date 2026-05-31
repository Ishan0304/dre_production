"""Lightweight YAML configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathConfig:
    """Standardized project paths for pipeline runs and artifacts."""

    project_root: str
    output_root: str
    artifacts_dir: str
    reports_dir: str
    data_dir: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class LoadedConfig:
    """Loaded configuration document with resolved path values."""

    config_name: str
    raw_config: dict[str, Any]
    resolved_config: dict[str, Any]
    path_config: PathConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class ConfigLoader:
    """Load YAML config files and resolve project-relative paths."""

    def load_yaml(self, config_path: str | Path) -> dict[str, Any]:
        """Load a UTF-8 YAML file and return a dictionary."""

        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config file: {path}") from exc
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError(f"YAML config must contain a mapping at the top level: {path}")
        return parsed

    def resolve_project_root(self, start_path: str | Path | None = None) -> Path:
        """Resolve the repository root from an optional anchor path."""

        anchor = Path(start_path).expanduser() if start_path is not None else Path(__file__)
        anchor = anchor.resolve()
        search_start = anchor if anchor.is_dir() else anchor.parent
        for path in [search_start, *search_start.parents]:
            if (path / "pyproject.toml").exists() and (path / "src").exists():
                return path
        raise RuntimeError("Could not resolve project root containing pyproject.toml and src/.")

    def build_path_config(
        self,
        raw_config: dict[str, Any],
        project_root: str | Path | None = None,
    ) -> PathConfig:
        """Build standardized paths from raw config values and defaults."""

        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else self.resolve_project_root()
        )
        paths = raw_config.get("paths", {})
        if paths is None:
            paths = {}
        if not isinstance(paths, dict):
            raise ValueError("Config field 'paths' must be a mapping when provided.")

        return PathConfig(
            project_root=str(root),
            output_root=str(self._resolve_path(root, paths.get("output_root", "outputs"))),
            artifacts_dir=str(
                self._resolve_path(root, paths.get("artifacts_dir", "outputs/artifacts"))
            ),
            reports_dir=str(
                self._resolve_path(root, paths.get("reports_dir", "outputs/reports"))
            ),
            data_dir=str(self._resolve_path(root, paths.get("data_dir", "data"))),
            notes=paths.get("notes"),
        )

    def resolve_config(
        self,
        raw_config: dict[str, Any],
        project_root: str | Path | None = None,
    ) -> LoadedConfig:
        """Resolve a raw config dictionary without mutating the caller input."""

        raw_copy = deepcopy(raw_config)
        resolved = deepcopy(raw_config)
        path_config = self.build_path_config(raw_copy, project_root=project_root)
        resolved_paths = path_config.to_dict()
        existing_paths = resolved.get("paths", {})
        if existing_paths is None:
            existing_paths = {}
        if not isinstance(existing_paths, dict):
            raise ValueError("Config field 'paths' must be a mapping when provided.")
        resolved["paths"] = {
            **existing_paths,
            **resolved_paths,
        }

        project_config = raw_copy.get("project", {})
        if project_config is None:
            project_config = {}
        if not isinstance(project_config, dict):
            raise ValueError("Config field 'project' must be a mapping when provided.")
        config_name = str(
            raw_copy.get("config_name")
            or project_config.get("name")
            or "dre_production_config"
        )
        return LoadedConfig(
            config_name=config_name,
            raw_config=raw_copy,
            resolved_config=resolved,
            path_config=path_config,
        )

    def load_and_resolve(self, config_path: str | Path) -> LoadedConfig:
        """Load a YAML config file and resolve project-relative paths."""

        raw_config = self.load_yaml(config_path)
        if "config_name" not in raw_config:
            raw_config = {**raw_config, "config_name": Path(config_path).stem}
        return self.resolve_config(raw_config)

    @staticmethod
    def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (project_root / path).resolve()
