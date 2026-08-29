"""Resume/skip and --force behaviour of the renderer.

A passing LLM (or deterministic) meta.json is money already spent. Re-rendering
it on every call is the defect that made a rate-limit kill into a full re-spend.
"""

from __future__ import annotations

import json
import re

import providers
import settings
from scenarios.renderer import render_seed, system_prompt
from scenarios.templates import long_date, render_deterministic
from tests.test_providers import _FakeResponse, _groq_empty, _groq_ok


SENTINEL = "SENTINEL-DO-NOT-TOUCH\n"


def _groq_config():
    config = settings.load_config()
    config["renderer"] = dict(config["renderer"], provider="groq",
                                model="llama-3.3-70b-versatile")
    return config


def _write_passed_llm(directory, event_id, seed):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rendered.md").write_text(SENTINEL, encoding="utf-8")
    (directory / "meta.json").write_text(
        json.dumps({"event_id": event_id, "seed": seed,
                    "render_mode": "llm", "fidelity": "pass",
                    "model": "llama-3.3-70b-versatile"},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def test_passed_llm_scenario_is_skipped_without_a_network_call(
        monkeypatch, artifacts, tmp_path):
    event = artifacts.events[0]
    out_dir = tmp_path / str(artifacts.seed) / event["event_id"]
    _write_passed_llm(out_dir, event["event_id"], artifacts.seed)

    def forbidden(request, timeout=None):
        raise AssertionError("skip path must not open a socket")

    monkeypatch.setattr(providers.urllib.request, "urlopen", forbidden)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    results = render_seed(
        artifacts.seed, deterministic=False, out_root=tmp_path,
        universe_root=artifacts.universe_root, only=event["event_id"],
        config=_groq_config())

    assert results["skipped"] == 1
    assert results["scenarios"] == 1
    assert results["fidelity_pass"] == 1
    assert results["fidelity_failed"] == 0
    assert (out_dir / "rendered.md").read_text(encoding="utf-8") == SENTINEL


def test_force_re_renders_despite_existing_pass_llm_meta(
        monkeypatch, artifacts, tmp_path):
    from scenarios.renderer import build_manifest, deal_name_map

    event = artifacts.events[0]
    out_dir = tmp_path / str(artifacts.seed) / event["event_id"]
    _write_passed_llm(out_dir, event["event_id"], artifacts.seed)

    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
    good = render_deterministic(manifest)
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _FakeResponse(_groq_ok(good))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    results = render_seed(
        artifacts.seed, deterministic=False, out_root=tmp_path,
        universe_root=artifacts.universe_root, only=event["event_id"],
        config=_groq_config(), force=True)

    assert calls["n"] == 1
    assert results["skipped"] == 0
    assert results["scenarios"] == 1
    assert results["fidelity_failed"] == 0
    text = (out_dir / "rendered.md").read_text(encoding="utf-8")
    assert text != SENTINEL
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["render_mode"] == "llm"
    assert meta["fidelity"] == "pass"


def test_force_flag_is_wired_on_the_cli(monkeypatch):
    """--force must reach render_seed; a missing flag makes resume un-overrideable."""
    from scenarios import renderer

    captured = {}

    def fake_render_seed(*args, **kwargs):
        captured.update(kwargs)
        return {"seed": 42, "render_mode": "llm", "scenarios": 0,
                "fidelity_pass": 0, "fidelity_failed": 0, "skipped": 0,
                "empty": 0, "attempt_histogram": {}}

    monkeypatch.setattr(renderer, "render_seed", fake_render_seed)
    assert renderer.main(["--force"]) == 0
    assert captured["force"] is True
    captured.clear()
    assert renderer.main([]) == 0
    assert captured["force"] is False


def test_provider_and_model_overrides_beat_config_and_reach_meta(
        monkeypatch, artifacts, tmp_path):
    """--provider/--model win over config.yaml and are recorded verbatim."""
    from scenarios.renderer import build_manifest, deal_name_map

    # The committed config defaults to Bedrock, so a groq result here can only
    # come from the override taking precedence.
    assert settings.load_config()["renderer"]["provider"] == "bedrock"

    event = artifacts.events[0]
    out_dir = tmp_path / str(artifacts.seed) / event["event_id"]
    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
    good = render_deterministic(manifest)

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_groq_ok(good))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    results = render_seed(
        artifacts.seed, deterministic=False, out_root=tmp_path,
        universe_root=artifacts.universe_root, only=event["event_id"],
        provider="groq", model="openai/gpt-oss-120b")

    assert results["fidelity_failed"] == 0
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["provider"] == "groq"
    assert meta["model"] == "openai/gpt-oss-120b"


def test_override_does_not_mutate_the_caller_config(monkeypatch, artifacts, tmp_path):
    """Overrides must not leak back into the shared config dict."""
    from scenarios.renderer import build_manifest, deal_name_map

    event = artifacts.events[0]
    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
    good = render_deterministic(manifest)

    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda request, timeout=None: _FakeResponse(_groq_ok(good)))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    config = _groq_config()
    original_model = config["renderer"]["model"]
    render_seed(
        artifacts.seed, deterministic=False, out_root=tmp_path,
        universe_root=artifacts.universe_root, only=event["event_id"],
        config=config, model="openai/gpt-oss-120b")

    assert config["renderer"]["model"] == original_model


def test_empty_completion_is_retried_and_can_recover(
        monkeypatch, artifacts, tmp_path):
    """An intermittent empty reply is retried, not fatal; a later draft wins."""
    from scenarios.renderer import build_manifest, deal_name_map

    event = artifacts.events[0]
    out_dir = tmp_path / str(artifacts.seed) / event["event_id"]
    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
    good = render_deterministic(manifest)
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(_groq_empty())
        return _FakeResponse(_groq_ok(good))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    results = render_seed(
        artifacts.seed, deterministic=False, out_root=tmp_path,
        universe_root=artifacts.universe_root, only=event["event_id"],
        config=_groq_config())

    assert calls["n"] == 2
    assert results["fidelity_failed"] == 0
    assert results["fidelity_pass"] == 1
    assert results["empty"] == 0
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["fidelity"] == "pass"
    assert meta["attempts"] == 2
    assert meta["attempt_history"][0]["status"] == "empty"


def test_persistent_empty_does_not_kill_the_seed_run(
        monkeypatch, artifacts, tmp_path):
    """One always-empty scenario fails on its own; the rest of the seed still renders."""
    from scenarios.renderer import build_manifest, deal_name_map

    events = artifacts.events[:2]
    bad_event, good_event = events[0], events[1]
    deal_names = deal_name_map(artifacts.facts)
    drafts = {}
    for event in events:
        manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
        drafts[manifest["subject"]] = render_deterministic(manifest)

    bad_subject = build_manifest(
        bad_event, artifacts.facts_by_id, deal_names)["subject"]

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        prompt = body["messages"][1]["content"]
        subject = prompt.split("Subject: ", 1)[1].split("\n", 1)[0]
        if subject == bad_subject:
            return _FakeResponse(_groq_empty())
        return _FakeResponse(_groq_ok(drafts[subject]))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # No exception must escape: a dead scenario cannot abort the batch.
    results = render_seed(
        artifacts.seed, deterministic=False, out_root=tmp_path,
        universe_root=artifacts.universe_root, limit=2, config=_groq_config())

    assert results["scenarios"] == 2
    assert results["fidelity_pass"] == 1
    assert results["fidelity_failed"] == 1
    assert results["empty"] == 1

    bad_meta = json.loads(
        (tmp_path / str(artifacts.seed) / bad_event["event_id"] / "meta.json")
        .read_text(encoding="utf-8"))
    assert bad_meta["fidelity"] != "pass"
    assert "empty_completion_diagnostic" in bad_meta
    assert "finish_reason='length'" in bad_meta["empty_completion_diagnostic"]

    good_meta = json.loads(
        (tmp_path / str(artifacts.seed) / good_event["event_id"] / "meta.json")
        .read_text(encoding="utf-8"))
    assert good_meta["fidelity"] == "pass"


def _manifest_with_a_date_fact(artifacts):
    """First manifest whose fact list carries a bare ISO date value."""
    from scenarios.renderer import build_manifest, deal_name_map

    deal_names = deal_name_map(artifacts.facts)
    for event in artifacts.events:
        manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
        for fact in manifest["facts"]:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", fact["value"]):
                return manifest, fact
    raise AssertionError("fixture has no ISO-date fact to exercise")


def test_iso_date_fact_value_is_naturalized_in_the_prompt(artifacts):
    """The model must never see a raw ISO token in the fact list."""
    from scenarios.renderer import user_prompt

    manifest, date_fact = _manifest_with_a_date_fact(artifacts)
    iso = date_fact["value"]
    natural = long_date(iso)
    assert natural != iso

    prompt = user_prompt(manifest)
    assert f"- {date_fact['attribute']}: {natural}" in prompt
    assert f"- {date_fact['attribute']}: {iso}" not in prompt


def test_non_date_fact_values_pass_through_unchanged():
    """Only the exact ISO shape is touched; everything else is verbatim."""
    from scenarios.renderer import _naturalize_fact_value

    assert _naturalize_fact_value("2025-04-24") == "April 24, 2025"
    # Integers, names, codes and targeting strings must not be reshaped.
    assert _naturalize_fact_value("342000") == "342000"
    assert _naturalize_fact_value("Holiday Brand Awareness") == "Holiday Brand Awareness"
    assert _naturalize_fact_value("A18-34") == "A18-34"
    assert _naturalize_fact_value("DEAL-42-001") == "DEAL-42-001"
    # Near-miss shapes stay raw rather than being mis-parsed as dates.
    assert _naturalize_fact_value("2025-04") == "2025-04"
    assert _naturalize_fact_value("2025-04-24T10:00") == "2025-04-24T10:00"


def test_naturalized_date_still_passes_fidelity_against_iso_manifest(artifacts):
    """Canonicalization holds: the long-form date verifies against the ISO fact."""
    from verify import fidelity

    manifest, date_fact = _manifest_with_a_date_fact(artifacts)
    natural = long_date(date_fact["value"])

    draft = render_deterministic(manifest)
    # The draft carries the naturalized date, not the raw ISO the manifest stores.
    assert natural in draft
    assert date_fact["value"] not in draft

    report = fidelity.verify_render(manifest, draft)
    assert report.ok
    assert date_fact["fact_id"] not in {f["fact_id"] for f in report.missing_facts}


def test_system_prompt_forbids_lists_and_placeholders_and_demands_role_signoff():
    """Prompt-text contract; no boto3/LLM needed, so it lives with the renderer tests."""
    prompt = system_prompt({"author": "user", "channel": "email_sent"})
    for requirement in ("bullet lists", "character-exact", "ad sales ops",
                        "[Your Name]", "CTV"):
        assert requirement in prompt


def test_system_prompt_bans_raw_attribute_labels_in_prose():
    """Attribute names are database labels, not vocabulary; rate_revision emails
    must say "the CPM", never echo "cpm_rate" from the fact line."""
    prompt = system_prompt({"author": "user", "channel": "email_sent"})
    assert "natural business term" in prompt
    assert "raw field label" in prompt
    # The instruction names the offending labels as counter-examples.
    for label in ("cpm_rate", "line_status", "flight_start"):
        assert label in prompt


def test_provider_and_model_flags_are_wired_on_the_cli(monkeypatch):
    from scenarios import renderer

    captured = {}

    def fake_render_seed(*args, **kwargs):
        captured.update(kwargs)
        return {"seed": 42, "render_mode": "llm", "scenarios": 0,
                "fidelity_pass": 0, "fidelity_failed": 0, "skipped": 0,
                "empty": 0, "attempt_histogram": {}}

    monkeypatch.setattr(renderer, "render_seed", fake_render_seed)
    assert renderer.main(["--provider", "groq", "--model", "openai/gpt-oss-120b"]) == 0
    assert captured["provider"] == "groq"
    assert captured["model"] == "openai/gpt-oss-120b"
    captured.clear()
    assert renderer.main([]) == 0
    assert captured["provider"] is None
    assert captured["model"] is None
