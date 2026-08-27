"""Schema validation, the trust derivation table, and quota satisfaction."""

from __future__ import annotations

import pytest

from universe import schema
from universe.generator import failed_quotas, generate
from universe.schema import (
    QUERY_KIND_BY_TYPE,
    QUERY_KINDS,
    QUESTION_TYPES,
    SchemaError,
    derive_trust_class,
    validate_event,
    validate_fact,
    validate_question,
)

SEEDS = (42, 43, 44)


# ------------------------------------------------------ trust derivation ---

def test_every_trust_table_pair_derives():
    for (channel, author), expected in schema.TRUST_TABLE.items():
        assert derive_trust_class(channel, author) == expected
        assert expected in schema.TRUST_CLASSES


def test_unknown_channel_author_pair_is_an_error_not_a_default():
    with pytest.raises(SchemaError, match="no trust_class derivation"):
        derive_trust_class("carrier_pigeon", "user")


def test_hand_set_trust_class_is_rejected(artifacts):
    fact = dict(artifacts.facts[0])
    wrong = next(value for value in schema.TRUST_CLASSES
                 if value != fact["trust_class"])
    fact["trust_class"] = wrong
    with pytest.raises(SchemaError, match="derived, never hand-set"):
        validate_fact(fact)


def test_every_generated_fact_carries_the_derived_trust_class(artifacts):
    for fact in artifacts.facts:
        assert fact["trust_class"] == derive_trust_class(fact["channel"],
                                                         fact["author"])


# ------------------------------------------------------------- validation ---

@pytest.mark.parametrize("seed", SEEDS)
def test_every_fact_and_event_validates(seed):
    universe = generate(seed)
    for fact in universe.facts:
        validate_fact(fact)
    for event in universe.events:
        validate_event(event)


def test_every_question_validates(artifacts):
    for question in artifacts.questions:
        validate_question(question)


def test_event_timestamp_is_mandatory(artifacts):
    event = dict(artifacts.events[0])
    event["timestamp"] = None
    with pytest.raises(SchemaError, match="timestamp"):
        validate_event(event)


def test_plausibility_is_non_null_exactly_on_injections(artifacts):
    for fact in artifacts.facts:
        assert (fact["plausibility"] is not None) == fact["injection"]


def test_plausibility_on_a_non_injection_is_rejected(artifacts):
    fact = dict(next(f for f in artifacts.facts if not f["injection"]))
    fact["plausibility"] = "high"
    with pytest.raises(SchemaError, match="plausibility must be null"):
        validate_fact(fact)


def test_never_memorize_facts_are_ephemeral_and_same_day(artifacts):
    probes = [fact for fact in artifacts.facts if fact["never_memorize"]]
    assert probes
    for fact in probes:
        assert fact["volatility_class"] == "ephemeral"
        interval = fact["validity_interval"]
        assert interval["end"] == interval["start"]
        assert fact["channel"] == "order_system"
        assert fact["author"] == "system"


def test_never_memorize_with_an_open_interval_is_rejected(artifacts):
    fact = dict(next(f for f in artifacts.facts if f["never_memorize"]))
    fact["validity_interval"] = {"start": fact["validity_interval"]["start"],
                                 "end": None}
    with pytest.raises(SchemaError, match="closed same-day"):
        validate_fact(fact)


# ----------------------------------------------------------------- quotas ---

@pytest.mark.parametrize("seed", SEEDS)
def test_all_quotas_are_satisfied(seed):
    universe = generate(seed)
    assert failed_quotas(universe.quotas) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_quota_headline_numbers(seed):
    quotas = generate(seed).quotas
    assert 10 <= quotas["advertisers"] <= 12
    assert quotas["scenarios"] >= 100
    assert quotas["timeline_span_days"] >= 365
    assert quotas["deals_with_correction_chains_pct"] >= 40.0
    assert quotas["multi_step_chains"] >= 5
    assert quotas["injections"] >= 8
    assert set(quotas["injection_plausibility"]) == {"low", "medium", "high"}
    assert quotas["injections_targeting_corrected_fact"] >= 2
    assert quotas["never_memorize_probes"] >= 5
    assert quotas["transient_facts_with_end_dates"] >= 10
    assert quotas["events_with_permanent_and_volatile_facts"] >= 3
    assert quotas["duplicate_active_facts"] == []


# -------------------------------------------------------- question typing ---

@pytest.mark.parametrize("seed", SEEDS)
def test_all_thirteen_types_are_present_above_the_floor(seed):
    universe = generate(seed)
    from questions.instantiate import build_questions
    questions = build_questions(universe.facts, universe.events, seed)
    counts: dict[str, int] = {}
    for question in questions:
        counts[question["type"]] = counts.get(question["type"], 0) + 1
    assert set(counts) == set(QUESTION_TYPES)
    assert len(QUESTION_TYPES) == 13
    below = {qtype: count for qtype, count in counts.items() if count < 10}
    assert not below, below
    assert 220 <= len(questions) <= 260


def test_every_question_carries_exactly_one_query_kind(artifacts):
    for question in artifacts.questions:
        assert question["query_kind"] in QUERY_KINDS
        assert question["query_kind"] == QUERY_KIND_BY_TYPE[question["type"]]


def test_all_four_query_kinds_are_exercised(artifacts):
    kinds = {question["query_kind"] for question in artifacts.questions}
    assert kinds == set(QUERY_KINDS)


def test_mismatched_query_kind_is_rejected(artifacts):
    question = dict(artifacts.questions[0])
    question["query_kind"] = next(kind for kind in QUERY_KINDS
                                  if kind != question["query_kind"])
    with pytest.raises(SchemaError, match="exactly one query_kind"):
        validate_question(question)


def test_as_of_horizons_cover_three_depths_plus_current(artifacts):
    horizons = {question["as_of_horizon"] for question in artifacts.questions}
    assert {"early", "mid", "late", "current"} <= horizons


def test_order_line_questions_name_their_io_id(artifacts):
    scoped = [question for question in artifacts.questions if question["io_id"]]
    assert scoped
    for question in scoped:
        assert question["io_id"] in question["text"]


def test_sentinel_questions_use_the_sentinel_vocabulary(artifacts):
    sentinels = [question for question in artifacts.questions
                 if question["gold_kind"] == "sentinel"]
    assert sentinels
    for question in sentinels:
        assert question["gold_answer"] in schema.SENTINELS
        assert question["gold_tokens"] == [question["gold_answer"]]
    assert {question["gold_answer"] for question in sentinels} == set(schema.SENTINELS)


def test_schema_document_lists_every_enum():
    document = schema.schema_document()
    assert document["enums"]["question_type"] == list(QUESTION_TYPES)
    assert len(document["trust_table"]) == len(schema.TRUST_TABLE)
    assert document["invariants"]
