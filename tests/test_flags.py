"""Flag propagation: injection and never_memorize, fact to manifest to question.

Both flags have to survive the whole pipeline. If ``injection`` is lost between
the fact store and the render manifest, the renderer has no way to put the claim
in claim voice; if it is lost between the fact store and the question, the audit
cannot tell a deceived answer from a correct one.
"""

from __future__ import annotations

import json


def _manifests(artifacts) -> list[dict]:
    root = artifacts.scenario_root / str(artifacts.seed)
    return [json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            for directory in sorted(root.iterdir()) if directory.is_dir()]


def test_injection_flag_reaches_the_manifest(artifacts):
    injected = {fact["fact_id"] for fact in artifacts.facts if fact["injection"]}
    assert injected
    seen = set()
    for manifest in _manifests(artifacts):
        for fact in manifest["facts"]:
            if fact["fact_id"] in injected:
                assert fact["injection"] is True
                assert fact["plausibility"] in ("low", "medium", "high")
                seen.add(fact["fact_id"])
            else:
                assert fact["injection"] is False
    assert seen == injected


def test_never_memorize_flag_reaches_the_manifest(artifacts):
    probes = {fact["fact_id"] for fact in artifacts.facts if fact["never_memorize"]}
    assert probes
    seen = set()
    for manifest in _manifests(artifacts):
        for fact in manifest["facts"]:
            if fact["fact_id"] in probes:
                assert fact["never_memorize"] is True
                assert fact["volatility_class"] == "ephemeral"
                seen.add(fact["fact_id"])
    assert seen == probes


def test_flags_reach_question_metadata(artifacts):
    injected = {fact["fact_id"] for fact in artifacts.facts if fact["injection"]}
    probes = {fact["fact_id"] for fact in artifacts.facts if fact["never_memorize"]}

    flagged_injection = [question for question in artifacts.questions
                         if question["touches_injection"]]
    flagged_probe = [question for question in artifacts.questions
                     if question["touches_never_memorize"]]
    assert flagged_injection
    assert flagged_probe

    for question in flagged_injection:
        assert injected & set(question["evidence_fact_ids"])
        assert question["type"] == "injection_probe"
        assert question["notes"]["probe_kind"] == "injection"
    for question in flagged_probe:
        assert probes & set(question["evidence_fact_ids"])
        assert question["notes"]["probe_kind"] == "never_memorize"

    # The flag is set exactly when the evidence warrants it.
    for question in artifacts.questions:
        expected = bool(injected & set(question["evidence_fact_ids"]))
        assert question["touches_injection"] is expected


def test_injected_and_never_memorize_facts_are_never_memory_points(artifacts):
    forbidden = {fact["fact_id"] for fact in artifacts.facts
                 if fact["injection"] or fact["never_memorize"]}
    assert forbidden
    for question in artifacts.questions:
        listed = {point["fact_id"] for point in question["expected_memory_points"]}
        overlap = listed & forbidden
        assert not overlap, (
            f"{question['question_id']} lists {overlap} as expected memory points")


def test_injection_probes_expect_the_true_competing_fact(artifacts):
    probes = [question for question in artifacts.questions
              if question["notes"].get("probe_kind") == "injection"]
    assert probes
    with_truth = 0
    for question in probes:
        for point in question["expected_memory_points"]:
            fact = artifacts.facts_by_id[point["fact_id"]]
            assert not fact["injection"]
            assert point["status"] == "active"
            with_truth += 1
    assert with_truth, "at least one injection probe should have a true counterpart"


def test_never_memorize_probes_expect_nothing_to_be_stored(artifacts):
    probes = [question for question in artifacts.questions
              if question["notes"].get("probe_kind") == "never_memorize"]
    assert probes
    for question in probes:
        assert question["expected_memory_points"] == []
        assert question["gold_answer"] == "DO_NOT_STORE"


def test_injection_plausibility_grades_all_appear_in_questions(artifacts):
    grades = {question["notes"].get("plausibility")
              for question in artifacts.questions
              if question["notes"].get("probe_kind") == "injection"}
    assert grades == {"low", "medium", "high"}
