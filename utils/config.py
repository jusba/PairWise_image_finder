"""TOML configuration helpers for command-line scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


_SCRIPT_SECTIONS = {
    "process",
    "process_pairs",
    "sampler",
    "semantic_sampler",
    "random_semantic_sampler",
}
_SPECIAL_KEYS = {"access_token_env"}


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared --config flag to a parser."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (default: ./config.toml if it exists).",
    )


def _load_toml(path: Path, *, explicit: bool) -> dict[str, Any]:
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"Config file does not exist: {path}")
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a TOML table: {path}")
    return data


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overriding real env vars."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key.startswith("#"):
                continue
            os.environ.setdefault(key, _unquote_env_value(value))


def _load_dotenv_files(config_path: Path) -> None:
    """Load local .env files from cwd and the config directory."""
    paths = [Path.cwd() / ".env"]
    config_env = config_path.parent / ".env"
    if config_env not in paths:
        paths.append(config_env)
    for path in paths:
        _load_dotenv(path)


def _flatten_config(
    data: dict[str, Any],
    *,
    active_script_sections: set[str],
) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def visit_table(table: dict[str, Any], prefix: str | None = None) -> None:
        for key, value in table.items():
            normalized = key.replace("-", "_")
            if isinstance(value, dict):
                if prefix is None and normalized in _SCRIPT_SECTIONS:
                    if normalized in active_script_sections:
                        visit_table(value, normalized)
                    continue
                visit_table(value, normalized)
            else:
                flat[normalized] = value

    visit_table(data)
    return flat


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS and action.dest != "help"
    }


def _resolve_access_token_env(defaults: dict[str, Any]) -> None:
    env_name = defaults.pop("access_token_env", None)
    if "access_token" in defaults or not env_name:
        return
    token = os.environ.get(str(env_name))
    if token:
        defaults["access_token"] = token


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    *,
    argv: list[str] | None = None,
    default_config_path: str | Path = "config.toml",
    script_sections: Iterable[str] = (),
) -> argparse.Namespace:
    """
    Parse CLI args after applying defaults from TOML.

    The default config path is optional. If the user passes --config explicitly,
    that file must exist. CLI flags override values loaded from TOML.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=None)
    config_args, _ = probe.parse_known_args(argv)

    explicit_config = config_args.config is not None
    config_path = Path(config_args.config) if explicit_config else Path(default_config_path)
    _load_dotenv_files(config_path)
    data = _load_toml(config_path, explicit=explicit_config)

    if data:
        active_sections = {section.replace("-", "_") for section in script_sections}
        defaults = _flatten_config(data, active_script_sections=active_sections)
        _resolve_access_token_env(defaults)

        destinations = _parser_destinations(parser)
        accepted = {key: value for key, value in defaults.items() if key in destinations}
        parser.set_defaults(**accepted)

    return parser.parse_args(argv)
