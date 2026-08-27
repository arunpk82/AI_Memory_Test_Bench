"""Shared fixtures.

The artifact fixture builds a complete pipeline run into a temporary directory
rather than reading the artifacts committed in the repository. Tests that read
committed artifacts pass even when the generator has drifted away from them,
which is the failure mode where a stale fixture certifies a broken build.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from questions.instantiate import build_questions, write_questions
from scenarios.renderer import render_seed
from universe.generator import generate, write_universe

SEED = 42
ALL_SEEDS = (42, 43, 44)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory) -> SimpleNamespace:
    """A full deterministic pipeline run for seed 42 in a temp directory."""
    root: Path = tmp_path_factory.mktemp("mem-testbed")
    universe_root = root / "universe"
    scenario_root = root / "scenarios"
    question_root = root / "questions"

    universe = generate(SEED)
    write_universe(universe, universe_root)
    render = render_seed(SEED, deterministic=True, out_root=scenario_root,
                         universe_root=universe_root)
    questions = build_questions(universe.facts, universe.events, SEED)
    write_questions(SEED, questions, question_root)

    return SimpleNamespace(
        seed=SEED,
        root=root,
        universe_root=universe_root,
        scenario_root=scenario_root,
        question_root=question_root,
        universe=universe,
        facts=universe.facts,
        events=universe.events,
        facts_by_id={fact["fact_id"]: fact for fact in universe.facts},
        quotas=universe.quotas,
        questions=questions,
        render=render,
    )


@pytest.fixture(scope="session")
def all_universes() -> dict[int, object]:
    """One generated universe per seed, for cross-seed assertions."""
    return {seed: generate(seed) for seed in ALL_SEEDS}
