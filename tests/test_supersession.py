"""Supersession structure: acyclic, non-destructive, and expired != superseded."""

from __future__ import annotations

import pytest

from questions.instantiate import CURRENT_STATE_TYPES
from universe.generator import (
    build_successor_map,
    generate,
    supersession_chains,
)
from universe.schema import parse_day

SEEDS = (42, 43, 44)


@pytest.mark.parametrize("seed", SEEDS)
def test_supersession_graph_is_acyclic(seed):
    universe = generate(seed)
    # supersession_chains walks every path and raises on a revisit.
    chains = supersession_chains(universe.facts)
    assert chains


@pytest.mark.parametrize("seed", SEEDS)
def test_supersession_is_a_chain_not_a_tree(seed):
    universe = generate(seed)
    build_successor_map(universe.facts)  # raises if a fact is superseded twice


def test_no_destructive_delete(artifacts):
    """Every fact named by a ``supersedes`` pointer still exists."""
    ids = {fact["fact_id"] for fact in artifacts.facts}
    for fact in artifacts.facts:
        if fact["supersedes"]:
            assert fact["supersedes"] in ids, (
                f"{fact['fact_id']} supersedes {fact['supersedes']}, which is gone; "
                f"corrections must close intervals, never delete")


def test_superseded_facts_keep_a_closed_interval(artifacts):
    superseded = {fact["supersedes"] for fact in artifacts.facts if fact["supersedes"]}
    assert superseded
    for fact_id in superseded:
        fact = artifacts.facts_by_id[fact_id]
        assert fact["validity_interval"]["end"] is not None, (
            f"{fact_id} was superseded but its interval is still open")


def test_successor_opens_after_the_predecessor_closes(artifacts):
    for fact in artifacts.facts:
        if not fact["supersedes"]:
            continue
        prior = artifacts.facts_by_id[fact["supersedes"]]
        assert parse_day(prior["validity_interval"]["end"]) < \
            parse_day(fact["validity_interval"]["start"])


def test_cancellation_is_a_superseding_fact(artifacts):
    cancellations = [fact for fact in artifacts.facts
                     if fact["attribute"] == "line_status"
                     and fact["value"] == "cancelled"]
    assert cancellations
    for fact in cancellations:
        assert fact["supersedes"], "a cancellation must supersede the prior status"
        prior = artifacts.facts_by_id[fact["supersedes"]]
        assert prior["value"] != "cancelled"
        assert prior["validity_interval"]["end"] is not None


@pytest.mark.parametrize("seed", SEEDS)
def test_no_chain_revisits_a_value(seed):
    """A reverting chain makes a stale answer look correct."""
    universe = generate(seed)
    by_id = {fact["fact_id"]: fact for fact in universe.facts}
    for chain in supersession_chains(universe.facts):
        values = [by_id[fact_id]["value"] for fact_id in chain]
        assert len(set(values)) == len(values), chain


# --------------------------------------------------- expired vs superseded ---

def _statuses(questions) -> set[str]:
    return {point["status"] for question in questions
            for point in question["expected_memory_points"]}


def test_both_expired_and_superseded_statuses_occur(artifacts):
    statuses = _statuses(artifacts.questions)
    assert "expired" in statuses
    assert "superseded" in statuses
    assert "active" in statuses


def test_expired_is_not_superseded(artifacts):
    """An expired point has no successor; a superseded one does."""
    successors = build_successor_map(artifacts.facts)
    for question in artifacts.questions:
        as_of = parse_day(question["as_of"])
        for point in question["expected_memory_points"]:
            fact = artifacts.facts_by_id[point["fact_id"]]
            successor_id = successors.get(fact["fact_id"])
            successor_started = (
                successor_id is not None
                and parse_day(
                    artifacts.facts_by_id[successor_id]["validity_interval"]["start"])
                <= as_of)
            if point["status"] == "expired":
                assert not successor_started, (
                    f"{fact['fact_id']} is marked expired but has a successor in "
                    f"force; expired means lapsed with no replacement")
                assert fact["validity_interval"]["end"] is not None
            if point["status"] == "superseded":
                assert successor_started


def test_expired_state_questions_reference_a_lapsed_transient_fact(artifacts):
    expired = [question for question in artifacts.questions
               if question["type"] == "expired_state"]
    assert expired
    for question in expired:
        lapsed = [point for point in question["expected_memory_points"]
                  if point["status"] == "expired"]
        assert lapsed, question["question_id"]
        for point in lapsed:
            fact = artifacts.facts_by_id[point["fact_id"]]
            assert fact["volatility_class"] == "transient"


# ------------------------------------------------ current-state gold rules ---

def test_current_state_gold_is_never_a_superseded_value(artifacts):
    for question in artifacts.questions:
        if question["type"] not in CURRENT_STATE_TYPES:
            continue
        if question["gold_kind"] != "value":
            continue
        for point in question["expected_memory_points"]:
            if point["status"] != "superseded":
                continue
            fact = artifacts.facts_by_id[point["fact_id"]]
            assert fact["value"] != question["gold_answer"], (
                f"{question['question_id']} is a current-state question whose gold "
                f"answer is the value of superseded fact {point['fact_id']}")


def test_current_state_gold_matches_an_active_memory_point(artifacts):
    for question in artifacts.questions:
        if question["type"] not in CURRENT_STATE_TYPES:
            continue
        if question["gold_kind"] != "value" or "derivation" in question["notes"]:
            continue
        active_values = {artifacts.facts_by_id[point["fact_id"]]["value"]
                         for point in question["expected_memory_points"]
                         if point["status"] == "active"}
        assert question["gold_answer"] in active_values, question["question_id"]


def test_past_types_reference_facts_that_are_superseded_today(artifacts):
    """Past-state golds are values that have since been replaced.

    Note what is *not* asserted: that the memory point is marked ``superseded``.
    A past-state question is asked as of the day before the correction, and on
    that day the fact was current and the correction did not exist yet. The
    "references superseded facts by design" property is about the fact having
    been replaced by now, not about its status at the historical as-of.
    """
    past = [question for question in artifacts.questions
            if question["type"] == "knowledge_update_past"]
    assert past
    successors = build_successor_map(artifacts.facts)
    for question in past:
        gold_facts = [artifacts.facts_by_id[fact_id]
                      for fact_id in question["evidence_fact_ids"]]
        assert any(successors.get(fact["fact_id"]) for fact in gold_facts), \
            question["question_id"]


def test_past_state_questions_do_not_expect_a_not_yet_recorded_fact(artifacts):
    for question in artifacts.questions:
        as_of = parse_day(question["as_of"])
        for point in question["expected_memory_points"]:
            fact = artifacts.facts_by_id[point["fact_id"]]
            assert parse_day(fact["validity_interval"]["start"]) <= as_of, (
                f"{question['question_id']} expects {point['fact_id']} to be in "
                f"memory at {question['as_of']}, but it is not recorded until "
                f"{fact['validity_interval']['start']}")
