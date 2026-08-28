"""Loader for ``config.yaml``.

A missing or ``TBD`` value is an error naming the exact key. Nothing in this
testbed substitutes a default model, region or sampling parameter: a silent
default is how a run ends up reporting numbers produced by a model nobody
chose.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

#: Oldest interpreter this testbed is supported on. The whole suite passes on
#: 3.10, 3.11 and 3.12, and all three produce byte-identical ``facts.jsonl`` for
#: every seed. 3.9 also passes today but is past end of life, so it is not
#: claimed. ``tests/test_docs.py`` checks that the README states this same
#: version, so the number cannot drift out of the documentation.
MINIMUM_PYTHON = (3, 10)

#: Placeholder values that must never be used as if they were configured.
UNSET_MARKERS = {"", "TBD", "tbd", "TODO", "todo", None}


class ConfigError(RuntimeError):
    """Raised when a required configuration key is missing or unset."""


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config. Raises :class:`ConfigError` if it is not readable."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"config file {config_path} did not parse to a mapping")
    return data


def require(config: dict, dotted_key: str):
    """Fetch ``dotted_key`` from ``config``, refusing missing and TBD values."""
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(
                f"config key {dotted_key!r} is missing from config.yaml; this run "
                f"cannot proceed without it")
        node = node[part]
    if isinstance(node, str) and node.strip() in UNSET_MARKERS:
        raise ConfigError(
            f"config key {dotted_key!r} is set to the placeholder {node!r}; set a "
            f"real value in config.yaml")
    if node is None:
        raise ConfigError(f"config key {dotted_key!r} is null in config.yaml")
    return node


def aws_region(config: dict) -> str:
    """Resolve the AWS region: ``AWS_REGION`` wins, then ``aws.region``."""
    env_region = os.environ.get("AWS_REGION", "").strip()
    if env_region:
        return env_region
    return require(config, "aws.region")
