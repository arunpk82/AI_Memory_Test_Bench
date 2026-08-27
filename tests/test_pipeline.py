"""End-to-end properties of the deterministic pipeline."""

from __future__ import annotations

import json
import re

import pytest

from questions.instantiate import build_questions, write_questions
from scenarios.renderer import build_manifest, deal_name_map, render_seed
from scenarios.templates import business_email
from universe.generator import generate, write_universe
from verify.answerability import dry_run, load_corpus
from verify.fidelity import verify_render, verify_seed

SEEDS = (42, 43, 44)


def _scenario_dirs(artifacts):
    root = artifacts.scenario_root / str(artifacts.seed)
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


# ---------------------------------------------------------------- fidelity ---

def test_every_scenario_passes_both_fidelity_sides(artifacts):
    assert artifacts.render["fidelity_failed"] == 0
    assert artifacts.render["scenarios"] == len(artifacts.events)
    summary = verify_seed(artifacts.seed, artifacts.scenario_root)
    assert summary["failed"] == 0
    assert summary["missing_fact_total"] == 0
    assert summary["unsupported_total"] == 0
    assert summary["scenarios"] == len(artifacts.events)


def test_deterministic_render_is_reproducible(artifacts, tmp_path):
    again = render_seed(artifacts.seed, deterministic=True, out_root=tmp_path,
                        universe_root=artifacts.universe_root)
    assert again["fidelity_failed"] == 0
    for directory in _scenario_dirs(artifacts):
        first = (directory / "rendered.md").read_text(encoding="utf-8")
        second = (tmp_path / str(artifacts.seed) / directory.name /
                  "rendered.md").read_text(encoding="utf-8")
        assert first == second


# --------------------------------------------------------------- artifacts ---

def test_meta_json_carries_the_timestamp(artifacts):
    """A missing timestamp in meta.json broke scenario ordering once."""
    for directory in _scenario_dirs(artifacts):
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert meta["timestamp"] == manifest["timestamp"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", meta["timestamp"])


def test_meta_json_has_every_required_field(artifacts):
    required = {"event_id", "seed", "timestamp", "scenario_kind", "channel",
                "author", "render_mode", "model", "attempts", "fidelity",
                "fidelity_missing_facts", "fidelity_unsupported",
                "template_version", "attempt_history"}
    for directory in _scenario_dirs(artifacts):
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        assert required <= set(meta), required - set(meta)
        assert meta["fidelity"] == "pass"
        assert meta["render_mode"] == "deterministic"
        assert meta["attempts"] == 1


def test_scenarios_can_be_ordered_by_meta_timestamp(artifacts):
    stamps = []
    for directory in _scenario_dirs(artifacts):
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        stamps.append((meta["timestamp"], meta["event_id"]))
    assert stamps == sorted(stamps)


# ------------------------------------------------------------ prose quality ---

_LABEL_RE = re.compile(r"^\s*[A-Z][A-Za-z ]{2,20}:\s*\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s+\S", re.MULTILINE)


def test_rendered_text_is_prose_not_a_field_list(artifacts):
    for directory in _scenario_dirs(artifacts):
        text = (directory / "rendered.md").read_text(encoding="utf-8")
        body = text.split("\n\n", 1)[1]
        assert not _BULLET_RE.search(body), f"{directory.name} has a bullet list"
        labels = _LABEL_RE.findall(body)
        assert not labels, f"{directory.name} has key-value fields: {labels}"


def test_only_subject_and_date_appear_as_headers(artifacts):
    for directory in _scenario_dirs(artifacts):
        lines = (directory / "rendered.md").read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("Subject: ")
        assert lines[1].startswith("Date: ")


def test_signoff_is_a_role_never_a_personal_name_or_placeholder(artifacts):
    for directory in _scenario_dirs(artifacts):
        text = (directory / "rendered.md").read_text(encoding="utf-8")
        assert "[" not in text and "]" not in text
        assert "Your Name" not in text
        tail = text.strip().splitlines()[-1]
        assert tail == "ad sales ops" or tail.startswith("the ")


def test_targeting_codes_appear_character_exact(artifacts):
    corpus = load_corpus(artifacts.seed, artifacts.scenario_root)
    text = "\n".join(corpus.values())
    assert "Connected TV" not in text
    assert "adults 25" not in text
    formats = {fact["value"] for fact in artifacts.facts
               if fact["attribute"] == "creative_format"}
    for value in formats:
        assert value in text


# -------------------------------------------------------- template coverage ---

@pytest.mark.parametrize("seed", SEEDS)
def test_every_attribute_has_a_prose_clause(seed):
    universe = generate(seed)
    attributes = {fact["attribute"] for fact in universe.facts}
    missing = attributes - set(business_email.CLAUSES)
    assert not missing, f"attributes with no template clause: {sorted(missing)}"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_attribute_has_a_question_phrase(seed):
    from questions.instantiate import ATTRIBUTE_PHRASE
    universe = generate(seed)
    attributes = {fact["attribute"] for fact in universe.facts}
    missing = attributes - set(ATTRIBUTE_PHRASE)
    assert not missing, f"attributes with no question phrasing: {sorted(missing)}"


def test_every_scenario_kind_has_an_opener(artifacts):
    kinds = {event["scenario_kind"] for event in artifacts.events}
    missing = kinds - set(business_email.OPENERS)
    assert not missing, f"scenario kinds falling back to the default opener: {missing}"


def test_unknown_attribute_raises_rather_than_dropping_a_fact(artifacts):
    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(artifacts.events[0], artifacts.facts_by_id, deal_names)
    manifest["facts"][0]["attribute"] = "some_unmodelled_attribute"
    with pytest.raises(business_email.TemplateError, match="no prose clause"):
        business_email.render_deterministic(manifest)


def test_injected_facts_render_in_claim_voice(artifacts):
    injected = [fact for fact in artifacts.facts if fact["injection"]]
    assert injected
    deal_names = deal_name_map(artifacts.facts)
    events_by_id = {event["event_id"]: event for event in artifacts.events}
    for fact in injected:
        if fact["attribute"] not in business_email.INJECTION_CLAUSES:
            continue
        manifest = build_manifest(events_by_id[fact["event_id"]],
                                  artifacts.facts_by_id, deal_names)
        text = business_email.render_deterministic(manifest)
        neutral = business_email.CLAUSES[fact["attribute"]](fact["value"])
        assert neutral.capitalize() not in text
        assert verify_render(manifest, text).ok


# ----------------------------------------------------- all seeds end to end ---

@pytest.mark.parametrize("seed", SEEDS)
def test_full_pipeline_is_clean_for_every_seed(seed, tmp_path):
    universe_root = tmp_path / "universe"
    scenario_root = tmp_path / "scenarios"
    question_root = tmp_path / "questions"

    universe = generate(seed)
    write_universe(universe, universe_root)
    render = render_seed(seed, deterministic=True, out_root=scenario_root,
                         universe_root=universe_root)
    assert render["fidelity_failed"] == 0

    questions = build_questions(universe.facts, universe.events, seed)
    write_questions(seed, questions, question_root)

    report = dry_run(seed, scenario_root=scenario_root, universe_root=universe_root,
                     question_root=question_root)
    assert report["clean"], report["findings"][:10]
    assert 220 <= report["questions"] <= 260
