from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "regime_features.toml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    _validate_config(config, config_path)
    return config


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    required_sections = {"features", "hmm", "wasserstein"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"{config_path} is missing sections: {sorted(missing)}")

    if not config["hmm"].get("features"):
        raise ValueError(f"{config_path} [hmm].features must not be empty")

    if not config["wasserstein"].get("features"):
        raise ValueError(f"{config_path} [wasserstein].features must not be empty")

    h1 = int(config["wasserstein"].get("h1", 0))
    h2 = int(config["wasserstein"].get("h2", 0))
    if h1 <= h2:
        raise ValueError(f"{config_path} requires [wasserstein].h1 > h2")

    methods = set(config["wasserstein"].get("methods", []))
    supported_methods = {"full", "sliced"}
    unsupported = methods.difference(supported_methods)
    if unsupported:
        raise ValueError(
            f"{config_path} has unsupported wasserstein methods: {sorted(unsupported)}"
        )

    final_method = config["wasserstein"].get("final_method", "sliced")
    if final_method not in supported_methods:
        raise ValueError(
            f"{config_path} has unsupported wasserstein final_method: {final_method}"
        )
