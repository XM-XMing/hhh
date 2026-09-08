"""Loading for checked-in experiment configuration profiles.

The loader owns path and type validation only.  Domain modules remain the
owners of algorithm defaults; this module makes experiment-level values
available from one installed YAML profile instead of shell defaults.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from planning.common.paths import planning_package_root


PRE_BC_CONFIG_RELATIVE_PATH = Path("config") / "pre_bc.yaml"

_TRUE_BOOL_TOKENS = frozenset({"1", "true", "yes", "y", "pass", "on"})
_FALSE_BOOL_TOKENS = frozenset({"0", "false", "no", "n", "fail", "off"})


def parse_bool(value: Any) -> bool:
    """Parse the one accepted boolean vocabulary for config/artifact fields."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    token = str(value).strip().lower()
    if token in _TRUE_BOOL_TOKENS:
        return True
    if token in _FALSE_BOOL_TOKENS:
        return False
    raise ValueError("invalid boolean value: {!r}".format(value))


@lru_cache(maxsize=8)
def _load_yaml(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping: {}".format(path))
    return payload


def load_pre_bc_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return a copy of the canonical Pre-BC experiment profile."""

    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else (planning_package_root() / PRE_BC_CONFIG_RELATIVE_PATH).resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError("Pre-BC experiment config not found: {}".format(config_path))
    return deepcopy(_load_yaml(str(config_path)))


def pre_bc_value(section: str, name: str) -> Any:
    """Resolve one required value without adding a second default owner."""

    config = load_pre_bc_config()
    values = config.get(str(section))
    if not isinstance(values, Mapping) or str(name) not in values:
        raise KeyError("missing Pre-BC config value {}.{}".format(section, name))
    return values[str(name)]


__all__ = [
    "PRE_BC_CONFIG_RELATIVE_PATH",
    "load_pre_bc_config",
    "parse_bool",
    "pre_bc_value",
]
