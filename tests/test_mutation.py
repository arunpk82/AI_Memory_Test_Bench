"""Mutation suite: answer-key discipline.

A checker that never fails is indistinguishable from no checker. Each test here
introduces one deliberate defect and asserts that the corresponding check
catches it. Any surviving mutant is a failed build, because it means a real
defect of that shape would ship silently.

Three mutants, one per check:

(a) corrupt a fact value        -> two-sided fidelity flags it
(b) flip a gold answer          -> dry-run gold-to-evidence traceability flags it
(c) delete an expected memory point -> dry-run consistency flags it
"""

from __future__ import annotations

import copy
import json

import pytest

from questions.instantiate import write_questions
from verify import fidelity
from verify.answerability import dry_run


# ----------------------------------------------------------------- mutant a ---

def _first_manifest(artifacts) -> tuple[dict, str]:
    root = artifacts.scenario_root / str(artifacts.seed)
    directory = next(path for path in sorted(root.iterdir()) if path.is_dir())
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    text = (directory / "rendered.md").read_text(encoding="utf-8")
    return manifest, text


def test_baseline_render_is_clean(artifacts):
    manifest, text = _first_manifest(artifacts)
    report = fidelity.verify_render(manifest, text)
    assert report.ok, report.as_dict()


def test_mutant_a_corrupted_fact_value_is_flagged_by_planted_recall(artifacts):
    manifest, text = _first_manifest(artifacts)
    mutated = copy.deepcopy(manifest)
    target = mutated["facts"][0]
    target["value"] = target["value"] + " Zylotronic"
    report = fidelity.verify_render(mutated, text)
    assert not report.ok
    assert any(fact["fact_id"] == target["fact_id"]
               for fact in report.missing_facts), report.as_dict()


def test_mutant_a2_invented_detail_in_the_text_is_flagged_by_precision(artifacts):
    manifest, text = _first_manifest(artifacts)
    mutated_text = text.replace(
        "Let me know if anything looks off.",
        "We also agreed a 17% discount with Zylotronic Media on May 4, 2031.")
    assert mutated_text != text
    report = fidelity.verify_render(manifest, mutated_text)
    assert not report.ok
    assert report.unsupported["numbers"], report.as_dict()
    assert report.unsupported["dates"], report.as_dict()
    assert report.unsupported["names"], report.as_dict()


def test_fidelity_failure_summary_is_usable_as_retry_feedback(artifacts):
    manifest, text = _first_manifest(artifacts)
    mutated = copy.deepcopy(manifest)
    mutated["facts"][0]["value"] = "definitely-not-in-the-text"
    summary = fidelity.verify_render(mutated, text).failure_summary()
    assert "definitely-not-in-the-text" in summary
    assert "missing from the draft" in summary


# ----------------------------------------------------------------- mutants b/c ---

def _dry_run_with(artifacts, tmp_path, mutate) -> dict:
    """Write mutated questions to a temp root and run the network-free audit."""
    questions = copy.deepcopy(artifacts.questions)
    mutate(questions)
    write_questions(artifacts.seed, questions, tmp_path)
    return dry_run(artifacts.seed,
                   scenario_root=artifacts.scenario_root,
                   universe_root=artifacts.universe_root,
                   question_root=tmp_path)


def test_baseline_dry_run_is_clean(artifacts):
    report = dry_run(artifacts.seed,
                     scenario_root=artifacts.scenario_root,
                     universe_root=artifacts.universe_root,
                     question_root=artifacts.question_root)
    assert report["clean"], report["findings"][:10]


def test_mutant_b_flipped_gold_answer_is_flagged_by_traceability(artifacts, tmp_path):
    def mutate(questions):
        target = next(question for question in questions
                      if question["gold_kind"] == "value"
                      and "derivation" not in question["notes"])
        target["gold_answer"] = "99.99"
        target["gold_tokens"] = ["99.99"]
        mutate.question_id = target["question_id"]

    report = _dry_run_with(artifacts, tmp_path, mutate)
    assert not report["clean"]
    findings = [finding for finding in report["findings"]
                if finding["question_id"] == mutate.question_id]
    assert any(finding["check"] == "gold_traceability" for finding in findings), findings


def test_mutant_c_deleted_memory_point_is_flagged_by_consistency(artifacts, tmp_path):
    def mutate(questions):
        target = next(question for question in questions
                      if len(question["expected_memory_points"]) >= 1
                      and question["evidence_fact_ids"])
        target["expected_memory_points"] = [
            point for point in target["expected_memory_points"]
            if point["fact_id"] != target["evidence_fact_ids"][0]]
        mutate.question_id = target["question_id"]

    report = _dry_run_with(artifacts, tmp_path, mutate)
    assert not report["clean"]
    findings = [finding for finding in report["findings"]
                if finding["question_id"] == mutate.question_id]
    assert any(finding["check"] == "memory_point_missing" for finding in findings), \
        findings


def test_mutant_d_corrupted_memory_point_status_is_flagged(artifacts, tmp_path):
    def mutate(questions):
        target = next(question for question in questions
                      if any(point["status"] == "active"
                             for point in question["expected_memory_points"]))
        for point in target["expected_memory_points"]:
            if point["status"] == "active":
                point["status"] = "expired"
                break
        mutate.question_id = target["question_id"]

    report = _dry_run_with(artifacts, tmp_path, mutate)
    assert not report["clean"]
    findings = [finding for finding in report["findings"]
                if finding["question_id"] == mutate.question_id]
    assert any(finding["check"] == "memory_point" for finding in findings), findings


def test_mutant_e_injected_fact_as_memory_point_is_flagged(artifacts, tmp_path):
    injected = next(fact for fact in artifacts.facts if fact["injection"])

    def mutate(questions):
        target = questions[0]
        target["expected_memory_points"] = list(target["expected_memory_points"]) + [
            {"fact_id": injected["fact_id"], "status": "active"}]
        mutate.question_id = target["question_id"]

    report = _dry_run_with(artifacts, tmp_path, mutate)
    assert not report["clean"]
    findings = [finding for finding in report["findings"]
                if finding["question_id"] == mutate.question_id]
    assert any("never-memorize" in finding["detail"]
               or "injected" in finding["detail"] for finding in findings), findings


def test_mutant_f_dropped_io_id_from_question_text_is_flagged(artifacts, tmp_path):
    def mutate(questions):
        target = next(question for question in questions if question["io_id"])
        target["text"] = target["text"].replace(target["io_id"], "that order line")
        mutate.question_id = target["question_id"]

    report = _dry_run_with(artifacts, tmp_path, mutate)
    assert not report["clean"]
    findings = [finding for finding in report["findings"]
                if finding["question_id"] == mutate.question_id]
    assert any(finding["check"] == "io_naming" for finding in findings), findings


@pytest.mark.parametrize("check", ["gold_traceability", "memory_point",
                                   "memory_point_missing", "io_naming"])
def test_every_dry_run_check_has_a_mutant(check):
    """The dry run must not contain a check that no mutant exercises."""
    source = (__file__)
    with open(source, encoding="utf-8") as handle:
        body = handle.read()
    assert check in body
