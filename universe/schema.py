"""Schemas for facts, events and questions, plus the trust derivation table.

Two rules in here are load-bearing:

1. ``trust_class`` is **derived** from ``(channel, author)`` through
   :data:`TRUST_TABLE` and is never hand-set at a call site. The validator
   re-derives it and rejects any fact whose stored value disagrees, so a
   hand-written trust class cannot survive a schema pass.
2. An event's ``timestamp`` is part of the schema and is written into both the
   render manifest and ``meta.json``. A previous build omitted it from
   ``meta.json`` and scenario ordering broke downstream.
"""

from __future__ import annotations

import re
from datetime import date, datetime

__all__ = [
    "ENTITY_TYPES", "VOLATILITY_CLASSES", "CHANNELS", "AUTHORS", "TRUST_CLASSES",
    "PLAUSIBILITY_GRADES", "TRUST_TABLE", "derive_trust_class", "FACT_FIELDS",
    "EVENT_FIELDS", "QUESTION_FIELDS", "QUESTION_TYPES", "QUERY_KINDS",
    "QUERY_KIND_BY_TYPE", "MEMORY_POINT_STATUSES", "SENTINELS",
    "SchemaError", "validate_fact", "validate_event", "validate_question",
    "schema_document", "parse_day", "iso_day",
]


class SchemaError(ValueError):
    """Raised when a fact, event or question violates the schema."""


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

ENTITY_TYPES = ("advertiser", "agency", "deal", "order-line", "contact")

VOLATILITY_CLASSES = ("permanent", "durable", "slow_changing", "transient", "ephemeral")

CHANNELS = ("email_received", "email_sent", "order_system", "call_note")

AUTHORS = ("user", "counterparty", "system")

TRUST_CLASSES = (
    "system_of_record",
    "first_party_sent",
    "first_party_note",
    "automated_notification",
    "counterparty_asserted",
    "third_party_unverified",
)

PLAUSIBILITY_GRADES = ("low", "medium", "high")

#: The one and only mapping from (channel, author) to trust_class.
#:
#: An unknown pair is an error, not a default. Silently defaulting an unmodelled
#: pair to the weakest class would let a fabricated channel masquerade as a
#: legitimate low-trust source.
TRUST_TABLE: dict[tuple[str, str], str] = {
    ("order_system", "system"): "system_of_record",
    ("order_system", "user"): "first_party_sent",
    ("email_sent", "user"): "first_party_sent",
    ("email_sent", "system"): "automated_notification",
    ("email_received", "counterparty"): "counterparty_asserted",
    ("email_received", "system"): "automated_notification",
    ("email_received", "user"): "third_party_unverified",
    ("call_note", "user"): "first_party_note",
    ("call_note", "counterparty"): "counterparty_asserted",
}


def derive_trust_class(channel: str, author: str) -> str:
    """Derive ``trust_class`` from ``(channel, author)``.

    This is the only way a trust class is ever produced.
    """
    try:
        return TRUST_TABLE[(channel, author)]
    except KeyError:
        raise SchemaError(
            f"no trust_class derivation for channel={channel!r} author={author!r}; "
            f"add the pair to TRUST_TABLE rather than defaulting it"
        ) from None


# --------------------------------------------------------------------------
# Field sets
# --------------------------------------------------------------------------

FACT_FIELDS = (
    "fact_id", "entity", "entity_id", "event_id", "attribute", "value",
    "validity_interval", "volatility_class", "channel", "author", "trust_class",
    "supersedes", "plausibility", "injection", "never_memorize",
)

EVENT_FIELDS = (
    "event_id", "seed", "scenario_kind", "subject", "timestamp", "channel",
    "author", "advertiser_id", "advertiser_name", "agency_name", "contact_name",
    "deal_id", "io_id", "fact_ids",
)

QUESTION_FIELDS = (
    "question_id", "seed", "type", "query_kind", "text", "as_of",
    "as_of_horizon", "gold_kind", "gold_answer", "gold_tokens",
    "expected_memory_points", "evidence_event_ids", "evidence_fact_ids",
    "entity", "entity_id", "io_id", "touches_injection",
    "touches_never_memorize", "notes",
)

QUESTION_TYPES = (
    "single_fact",
    "inference_only",
    "episodic",
    "temporal",
    "multi_session",
    "knowledge_update_current",
    "knowledge_update_past",
    "expired_state",
    "conflict_resolution",
    "order_state",
    "mapping_lookup",
    "abstention_false_premise",
    "injection_probe",
)

QUERY_KINDS = ("exact", "similarity", "temporal_range", "multi_hop")

#: Exactly one query_kind per question type. The question builder reads this
#: table; it does not accept a query_kind argument, so a question can never
#: carry two kinds or none.
QUERY_KIND_BY_TYPE: dict[str, str] = {
    "single_fact": "exact",
    "inference_only": "multi_hop",
    "episodic": "similarity",
    "temporal": "temporal_range",
    "multi_session": "multi_hop",
    "knowledge_update_current": "exact",
    "knowledge_update_past": "temporal_range",
    "expired_state": "temporal_range",
    "conflict_resolution": "multi_hop",
    "order_state": "exact",
    "mapping_lookup": "exact",
    "abstention_false_premise": "similarity",
    "injection_probe": "similarity",
}

#: A memory point is *superseded* when a later fact replaced it and *expired*
#: when its validity lapsed with no successor. Collapsing the two loses the
#: distinction between "we know the new value" and "we know nothing current".
MEMORY_POINT_STATUSES = ("active", "superseded", "expired")

AS_OF_HORIZONS = ("early", "mid", "late", "current")

SENTINELS = (
    "INSUFFICIENT_EVIDENCE",
    "FALSE_PREMISE",
    "DO_NOT_STORE",
    "REJECT_UNVERIFIED_THIRD_PARTY_CLAIM",
)

GOLD_KINDS = ("value", "sentinel")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


# --------------------------------------------------------------------------
# Date helpers
# --------------------------------------------------------------------------

def parse_day(value: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` string."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def iso_day(value: date) -> str:
    return value.isoformat()


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------

def _require_exact_fields(record: dict, fields: tuple[str, ...], label: str) -> None:
    missing = [name for name in fields if name not in record]
    extra = [name for name in record if name not in fields]
    if missing or extra:
        raise SchemaError(
            f"{label} field mismatch: missing={sorted(missing)} extra={sorted(extra)}")


def _require_id(value, label: str) -> None:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise SchemaError(f"{label} must be an identifier string, got {value!r}")


def _require_choice(value, allowed: tuple[str, ...], label: str) -> None:
    if value not in allowed:
        raise SchemaError(f"{label} must be one of {allowed}, got {value!r}")


def _validate_interval(interval, label: str) -> tuple[date, date | None]:
    if not isinstance(interval, dict) or set(interval) != {"start", "end"}:
        raise SchemaError(f"{label} must be a dict with exactly start and end, "
                          f"got {interval!r}")
    try:
        start = parse_day(interval["start"])
    except (TypeError, ValueError):
        raise SchemaError(f"{label}.start must be YYYY-MM-DD, "
                          f"got {interval.get('start')!r}") from None
    end = interval["end"]
    if end is None:
        return start, None
    try:
        end_day = parse_day(end)
    except (TypeError, ValueError):
        raise SchemaError(f"{label}.end must be YYYY-MM-DD or null, got {end!r}") from None
    if end_day < start:
        raise SchemaError(f"{label}.end {end} precedes start {interval['start']}")
    return start, end_day


def validate_fact(fact: dict) -> dict:
    """Validate one fact record, returning it unchanged.

    Raises :class:`SchemaError` with a message naming the offending field.
    """
    if not isinstance(fact, dict):
        raise SchemaError(f"fact must be a dict, got {type(fact).__name__}")
    _require_exact_fields(fact, FACT_FIELDS, "fact")
    _require_id(fact["fact_id"], "fact.fact_id")
    _require_id(fact["entity_id"], "fact.entity_id")
    _require_id(fact["event_id"], "fact.event_id")
    _require_choice(fact["entity"], ENTITY_TYPES, "fact.entity")
    _require_choice(fact["volatility_class"], VOLATILITY_CLASSES, "fact.volatility_class")
    _require_choice(fact["channel"], CHANNELS, "fact.channel")
    _require_choice(fact["author"], AUTHORS, "fact.author")

    if not isinstance(fact["attribute"], str) or not fact["attribute"]:
        raise SchemaError(f"fact.attribute must be a non-empty string, "
                          f"got {fact['attribute']!r}")
    if not isinstance(fact["value"], str) or not fact["value"].strip():
        raise SchemaError(f"fact.value must be a non-empty string (canonical "
                          f"surface form), got {fact['value']!r}")

    start, end = _validate_interval(fact["validity_interval"], "fact.validity_interval")

    derived = derive_trust_class(fact["channel"], fact["author"])
    if fact["trust_class"] != derived:
        raise SchemaError(
            f"fact.trust_class {fact['trust_class']!r} disagrees with the value "
            f"derived from (channel={fact['channel']}, author={fact['author']}), "
            f"which is {derived!r}; trust_class is derived, never hand-set")

    if fact["supersedes"] is not None:
        _require_id(fact["supersedes"], "fact.supersedes")

    for flag in ("injection", "never_memorize"):
        if not isinstance(fact[flag], bool):
            raise SchemaError(f"fact.{flag} must be a bool, got {fact[flag]!r}")

    if fact["injection"]:
        _require_choice(fact["plausibility"], PLAUSIBILITY_GRADES, "fact.plausibility")
    elif fact["plausibility"] is not None:
        raise SchemaError("fact.plausibility must be null on non-injection facts, "
                          f"got {fact['plausibility']!r}")

    if fact["never_memorize"]:
        if fact["volatility_class"] != "ephemeral":
            raise SchemaError("never_memorize facts must be ephemeral, got "
                              f"{fact['volatility_class']!r}")
        if end is None or end != start:
            raise SchemaError("never_memorize facts must carry a closed same-day "
                              f"validity interval, got {fact['validity_interval']!r}")
    return fact


def validate_event(event: dict) -> dict:
    """Validate one event record, returning it unchanged."""
    if not isinstance(event, dict):
        raise SchemaError(f"event must be a dict, got {type(event).__name__}")
    _require_exact_fields(event, EVENT_FIELDS, "event")
    _require_id(event["event_id"], "event.event_id")
    if not isinstance(event["seed"], int):
        raise SchemaError(f"event.seed must be an int, got {event['seed']!r}")
    _require_choice(event["channel"], CHANNELS, "event.channel")
    _require_choice(event["author"], AUTHORS, "event.author")
    if not isinstance(event["subject"], str) or not event["subject"].strip():
        raise SchemaError(f"event.subject must be a non-empty string, "
                          f"got {event['subject']!r}")
    try:
        parse_day(event["timestamp"])
    except (TypeError, ValueError):
        raise SchemaError("event.timestamp must be YYYY-MM-DD and must be present; "
                          "a missing timestamp broke scenario ordering once. Got "
                          f"{event.get('timestamp')!r}") from None
    if not isinstance(event["fact_ids"], list) or not event["fact_ids"]:
        raise SchemaError(f"event.fact_ids must be a non-empty list, "
                          f"got {event['fact_ids']!r}")
    for fact_id in event["fact_ids"]:
        _require_id(fact_id, "event.fact_ids[]")
    if not isinstance(event["scenario_kind"], str) or not event["scenario_kind"]:
        raise SchemaError(f"event.scenario_kind must be a non-empty string, "
                          f"got {event['scenario_kind']!r}")
    _require_id(event["advertiser_id"], "event.advertiser_id")
    for optional in ("agency_name", "contact_name", "deal_id", "io_id"):
        value = event[optional]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SchemaError(f"event.{optional} must be a non-empty string or null, "
                              f"got {value!r}")
    return event


def validate_question(question: dict) -> dict:
    """Validate one question record, returning it unchanged."""
    if not isinstance(question, dict):
        raise SchemaError(f"question must be a dict, got {type(question).__name__}")
    _require_exact_fields(question, QUESTION_FIELDS, "question")
    _require_id(question["question_id"], "question.question_id")
    if not isinstance(question["seed"], int):
        raise SchemaError(f"question.seed must be an int, got {question['seed']!r}")
    _require_choice(question["type"], QUESTION_TYPES, "question.type")

    expected_kind = QUERY_KIND_BY_TYPE[question["type"]]
    if question["query_kind"] != expected_kind:
        raise SchemaError(
            f"question.query_kind {question['query_kind']!r} disagrees with the "
            f"kind assigned to type {question['type']!r}, which is "
            f"{expected_kind!r}; every question carries exactly one query_kind")

    if not isinstance(question["text"], str) or len(question["text"].strip()) < 12:
        raise SchemaError(f"question.text must be a real question string, "
                          f"got {question['text']!r}")
    try:
        parse_day(question["as_of"])
    except (TypeError, ValueError):
        raise SchemaError(f"question.as_of must be YYYY-MM-DD, "
                          f"got {question.get('as_of')!r}") from None
    _require_choice(question["as_of_horizon"], AS_OF_HORIZONS, "question.as_of_horizon")
    _require_choice(question["gold_kind"], GOLD_KINDS, "question.gold_kind")

    if question["gold_kind"] == "sentinel":
        _require_choice(question["gold_answer"], SENTINELS, "question.gold_answer")
        if question["gold_tokens"] != [question["gold_answer"]]:
            raise SchemaError("a sentinel question's gold_tokens must be exactly "
                              f"[gold_answer], got {question['gold_tokens']!r}")
    else:
        if not isinstance(question["gold_answer"], str) or not question["gold_answer"].strip():
            raise SchemaError(f"question.gold_answer must be a non-empty string, "
                              f"got {question['gold_answer']!r}")
        if question["gold_answer"] in SENTINELS:
            raise SchemaError("a value question must not use a sentinel as its "
                              f"gold_answer, got {question['gold_answer']!r}")
        tokens = question["gold_tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise SchemaError(f"question.gold_tokens must be a non-empty list, "
                              f"got {tokens!r}")
        for token in tokens:
            if not isinstance(token, str) or not token.strip():
                raise SchemaError(f"question.gold_tokens[] must be non-empty "
                                  f"strings, got {token!r}")

    points = question["expected_memory_points"]
    if not isinstance(points, list):
        raise SchemaError(f"question.expected_memory_points must be a list, "
                          f"got {points!r}")
    for point in points:
        if not isinstance(point, dict) or set(point) != {"fact_id", "status"}:
            raise SchemaError("each expected_memory_point must have exactly "
                              f"fact_id and status, got {point!r}")
        _require_id(point["fact_id"], "expected_memory_point.fact_id")
        _require_choice(point["status"], MEMORY_POINT_STATUSES,
                        "expected_memory_point.status")

    for list_field in ("evidence_event_ids", "evidence_fact_ids"):
        value = question[list_field]
        if not isinstance(value, list):
            raise SchemaError(f"question.{list_field} must be a list, got {value!r}")
        for item in value:
            _require_id(item, f"question.{list_field}[]")

    _require_choice(question["entity"], ENTITY_TYPES, "question.entity")
    for flag in ("touches_injection", "touches_never_memorize"):
        if not isinstance(question[flag], bool):
            raise SchemaError(f"question.{flag} must be a bool, got {question[flag]!r}")
    if question["io_id"] is not None:
        _require_id(question["io_id"], "question.io_id")
    return question


# --------------------------------------------------------------------------
# Machine-readable schema document
# --------------------------------------------------------------------------

def schema_document() -> dict:
    """The schema as data, written to ``universe/out/<seed>/schema.json``."""
    return {
        "version": 2,
        "fact_fields": list(FACT_FIELDS),
        "event_fields": list(EVENT_FIELDS),
        "question_fields": list(QUESTION_FIELDS),
        "enums": {
            "entity": list(ENTITY_TYPES),
            "volatility_class": list(VOLATILITY_CLASSES),
            "channel": list(CHANNELS),
            "author": list(AUTHORS),
            "trust_class": list(TRUST_CLASSES),
            "plausibility": list(PLAUSIBILITY_GRADES),
            "question_type": list(QUESTION_TYPES),
            "query_kind": list(QUERY_KINDS),
            "memory_point_status": list(MEMORY_POINT_STATUSES),
            "as_of_horizon": list(AS_OF_HORIZONS),
            "sentinel": list(SENTINELS),
            "gold_kind": list(GOLD_KINDS),
        },
        "trust_table": [
            {"channel": channel, "author": author, "trust_class": trust}
            for (channel, author), trust in sorted(TRUST_TABLE.items())
        ],
        "query_kind_by_type": dict(sorted(QUERY_KIND_BY_TYPE.items())),
        "invariants": [
            "trust_class is derived from (channel, author) via trust_table and is "
            "never hand-set",
            "plausibility is non-null exactly on injection facts",
            "never_memorize facts are ephemeral with a closed same-day validity "
            "interval",
            "cancellation supersedes with line_status='cancelled'; the prior fact "
            "is retained with its interval closed, never deleted",
            "expired (lapsed, no successor) is not superseded (replaced)",
            "expected_memory_points exclude injection and never_memorize facts",
            "every question carries exactly one query_kind, assigned by type",
            "event.timestamp is required and appears in both the render manifest "
            "and meta.json",
        ],
    }
