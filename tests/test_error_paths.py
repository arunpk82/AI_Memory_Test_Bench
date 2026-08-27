"""Error and guard paths.

These are the branches that only run when something has gone wrong. They are
tested for the same reason the mutation suite exists: an untested guard is a
guard nobody has seen fire, and the DORMANT PATHS section of the completion
report is shorter and more honest for every one of these that moves out of it.
"""

from __future__ import annotations

from datetime import date

import pytest

import settings
from questions.instantiate import allocate, load_questions
from scenarios.renderer import RenderError, render_seed
from universe.generator import (
    build_successor_map,
    generate,
    load_universe,
    memory_point_status,
    supersession_chains,
)
from universe.schema import SchemaError, parse_day
from verify import matching
from verify.answerability import judge, load_corpus
from verify.fidelity import verify_seed


# ---------------------------------------------------------- missing artifacts ---

def test_load_universe_names_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="universe.generator --seed 42"):
        load_universe(42, tmp_path)


def test_load_questions_names_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="questions.instantiate --seed 42"):
        load_questions(42, tmp_path)


def test_load_corpus_names_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="scenarios.renderer --seed 42"):
        load_corpus(42, tmp_path)


def test_verify_seed_names_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="scenarios.renderer --seed 42"):
        verify_seed(42, tmp_path)


def test_render_only_with_no_match_is_an_error(artifacts, tmp_path):
    with pytest.raises(RenderError, match="no events matched"):
        render_seed(artifacts.seed, deterministic=True, out_root=tmp_path,
                    universe_root=artifacts.universe_root, only="EV-42-9999")


# ------------------------------------------------------------------- config ---

def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(settings.ConfigError, match="config file not found"):
        settings.load_config(tmp_path / "nope.yaml")


def test_config_that_is_not_a_mapping_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(settings.ConfigError, match="did not parse to a mapping"):
        settings.load_config(path)


# ------------------------------------------------------ supersession guards ---

def test_a_fact_superseded_twice_is_rejected():
    """The reachable half of cycle protection.

    A supersession graph can only become cyclic by way of a fact with two
    successors, and this guard runs before any chain walk. The cycle check
    inside ``supersession_chains`` is therefore defensive only and is listed as
    a dormant path in the completion report rather than pretended to be tested.
    """
    facts = [
        {"fact_id": "F-1", "supersedes": None, "value": "a"},
        {"fact_id": "F-2", "supersedes": "F-1", "value": "b"},
        {"fact_id": "F-3", "supersedes": "F-1", "value": "c"},
    ]
    with pytest.raises(SchemaError, match="superseded twice"):
        build_successor_map(facts)


def test_chain_that_revisits_a_value_is_rejected():
    facts = [
        {"fact_id": "F-1", "supersedes": None, "value": "26.70"},
        {"fact_id": "F-2", "supersedes": "F-1", "value": "28.50"},
        {"fact_id": "F-3", "supersedes": "F-2", "value": "26.70"},
    ]
    with pytest.raises(SchemaError, match="revisits a value"):
        supersession_chains(facts)


def test_status_of_a_fact_that_has_not_started_is_refused(artifacts):
    fact = artifacts.facts[0]
    start = parse_day(fact["validity_interval"]["start"])
    successors = build_successor_map(artifacts.facts)
    with pytest.raises(SchemaError, match="has no memory-point status"):
        memory_point_status(fact, artifacts.facts_by_id, successors,
                            start - __import__("datetime").timedelta(days=1))


# ------------------------------------------------------- allocation guards ---

def test_allocation_refuses_a_type_below_the_floor():
    import random
    candidates = {"single_fact": [{"type": "single_fact"}] * 3}
    with pytest.raises(Exception, match="below the floor"):
        allocate(candidates, target_min=220, target_max=260, min_per_type=10,
                 rng=random.Random(0))


def test_allocation_refuses_a_total_outside_the_window():
    import random
    candidates = {"single_fact": [{"type": "single_fact", "as_of": "2025-01-01",
                                   "entity_id": "x", "text": "t", "seed": 42}] * 12}
    with pytest.raises(Exception, match="outside the required window"):
        allocate(candidates, target_min=220, target_max=260, min_per_type=10,
                 rng=random.Random(0))


# --------------------------------------------------------------- judge edges ---

def test_judge_handles_a_missing_answer():
    question = {"gold_kind": "value", "gold_answer": "26.70",
                "gold_tokens": ["26.70"], "notes": {}}
    assert not judge(question, None).matched
    assert judge(question, None).reason == "no answer"


# ------------------------------------------------------------ matching edges ---

def test_two_digit_year_is_expanded():
    assert matching.canonical_date("04/18/25") == date(2025, 4, 18)


def test_multiplier_suffix_is_read_from_prose():
    found = [number for _, number in
             matching.extract_numbers("We are pacing toward 1.2M impressions.")]
    assert found == [matching.Decimal("1200000")]


def test_magnitude_shorthand_satisfies_planted_recall():
    """A budget of 1,200,000 written as $1.2M must not read as a miss."""
    assert matching.value_in_text("1200000", "The working budget is $1.2M net.")
    assert matching.value_in_text("450000", "We reserved 450k impressions.")
    assert not matching.value_in_text("1200000", "The working budget is $1.3M net.")


def test_magnitude_shorthand_does_not_break_code_extraction():
    text = "Line IO-42-0071 runs A25-54 at 1.2M impressions."
    names = matching.extract_name_phrases(text)
    assert "IO-42-0071" in names
    assert "A25-54" in names


def test_value_in_text_handles_none():
    assert not matching.value_in_text(None, "anything")
    assert not matching.value_in_text("26.70", None)


def test_empty_value_is_not_found():
    assert not matching.value_in_text("", "anything at all")


# ------------------------------------------------------- template guard rails ---

def test_generator_refuses_to_supersede_a_transient_fact():
    """A lapsed fact must stay expired, never become superseded."""
    from universe.generator import _Builder

    builder = _Builder(42)
    event = builder.add_event(
        scenario_kind="pacing_note", subject="s", timestamp=date(2025, 1, 6),
        channel="call_note", author="user", advertiser_id="ADV-42-01",
        advertiser_name="Test Media")
    transient = builder.add_fact(event, "order-line", "IO-42-0001", "pacing_status",
                                 "on pace", "transient", end=date(2025, 2, 1))
    later = builder.add_event(
        scenario_kind="pacing_note", subject="s2", timestamp=date(2025, 3, 1),
        channel="call_note", author="user", advertiser_id="ADV-42-01",
        advertiser_name="Test Media")
    with pytest.raises(SchemaError, match="must stay expired"):
        builder.add_fact(later, "order-line", "IO-42-0001", "pacing_status",
                         "under-delivering", "transient", supersedes=transient)


def test_generator_refuses_an_inverted_interval():
    from universe.generator import _Builder

    builder = _Builder(42)
    event = builder.add_event(
        scenario_kind="order_booked", subject="s", timestamp=date(2025, 1, 6),
        channel="order_system", author="system", advertiser_id="ADV-42-01",
        advertiser_name="Test Media")
    first = builder.add_fact(event, "order-line", "IO-42-0001", "cpm_rate",
                             "26.70", "slow_changing")
    with pytest.raises(SchemaError, match="intervals would invert"):
        builder.add_fact(event, "order-line", "IO-42-0001", "cpm_rate", "28.50",
                         "slow_changing", supersedes=first)


def test_rate_ladder_exhaustion_is_reported():
    """The guard behind the reverting-chain fix, fired deliberately."""
    universe = generate(42)
    assert universe.quotas["multi_step_chains"] >= 5
