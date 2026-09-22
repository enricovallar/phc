"""Hydra configuration loading and dictionary normalization helpers.

Enables programmatic configuration composition for headless execution, automated testing,
and parameter sweeps without requiring the @hydra.main CLI wrapper.
"""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Recursively converts an OmegaConf DictConfig or dict into a pure Python dictionary.

    Args:
        cfg: DictConfig or standard dictionary.

    Returns:
        Pure Python dictionary with all internal interpolations resolved.
    """
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def load_simulation_config(
    config_name: str = "config",
    config_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Programmatically loads and composes a Hydra configuration.

    Args:
        config_name: Base YAML configuration filename without '.yaml' (default: 'config').
        config_dir: Directory containing config files. If None, resolves the workspace
            root 'configs/' directory.
        overrides: List of CLI-style override strings (e.g. `["simulation.resolution=16"]`).

    Returns:
        Composed DictConfig instance ready for pipeline execution.

    Raises:
        FileNotFoundError: If the specified config_dir does not exist.
    """
    if config_dir is None:
        # Resolve to workspace configs/ directory by searching upward
        current = Path(__file__).resolve()
        resolved_config_dir = None
        for parent in current.parents:
            candidate = parent / "configs"
            if candidate.is_dir() and (parent / "pyproject.toml").is_file():
                resolved_config_dir = candidate
                break
        if resolved_config_dir is None:
            resolved_config_dir = Path("configs").resolve()
    else:
        resolved_config_dir = Path(config_dir).resolve()

    if not resolved_config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {resolved_config_dir}")

    # Initialize hydra context and compose config
    with initialize_config_dir(
        version_base=None,
        config_dir=str(resolved_config_dir),
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg
