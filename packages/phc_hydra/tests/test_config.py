"""Unit tests for configuration helpers in phc_hydra."""

from omegaconf import DictConfig, OmegaConf
from phc_hydra.config import load_simulation_config, to_plain_dict


def test_to_plain_dict():
    """Verifies that to_plain_dict resolves DictConfigs into native dictionaries."""
    cfg = OmegaConf.create(
        {
            "base": 10,
            "derived": "${base}",
            "nested": {"key": "val"},
        }
    )
    plain = to_plain_dict(cfg)
    assert isinstance(plain, dict)
    assert plain["derived"] == 10
    assert plain["nested"]["key"] == "val"


def test_load_simulation_config():
    """Verifies programmatic composition of Hydra configs."""
    cfg = load_simulation_config(
        config_name="config",
        overrides=["simulation.resolution=16", "geometry=c4v_primitive"],
    )
    assert isinstance(cfg, DictConfig)
    assert cfg.simulation.resolution == 16
    assert cfg.geometry.name == "c4v_primitive"
