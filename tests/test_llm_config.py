"""LLM-path configuration tests, hermetic.

These tests exercise the Bedrock code paths without a network call and without
credentials. What they pin is the set of fail-fast behaviours: every failure
names the exact thing that is missing, because "render failed" with no cause is
a bug report nobody can act on.
"""

from __future__ import annotations

import pytest

import providers
import settings
from scenarios import renderer
from verify import answerability

boto3 = pytest.importorskip("boto3")


class _FakeSession:
    def __init__(self, credentials, **_kwargs):
        self._credentials = credentials

    def get_credentials(self):
        return self._credentials

    def client(self, name):
        return f"client:{name}"


@pytest.fixture
def base_config():
    return settings.load_config()


# --------------------------------------------------------------- config keys ---

def test_missing_model_key_names_the_key():
    with pytest.raises(settings.ConfigError, match="renderer.model"):
        settings.require({"renderer": {}}, "renderer.model")


def test_tbd_model_is_rejected():
    with pytest.raises(settings.ConfigError, match="placeholder"):
        settings.require({"renderer": {"model": "TBD"}}, "renderer.model")


def test_null_config_value_is_rejected():
    with pytest.raises(settings.ConfigError, match="null"):
        settings.require({"renderer": {"model": None}}, "renderer.model")


def test_tbd_model_is_rejected_through_the_renderer():
    with pytest.raises(settings.ConfigError, match="renderer.model"):
        renderer.renderer_spec({"renderer": {"model": "TBD"}})


def test_anthropic_model_without_inference_profile_prefix_is_rejected(base_config):
    config = dict(base_config)
    config["renderer"] = dict(base_config["renderer"],
                              model="anthropic.claude-sonnet-4-6")
    with pytest.raises(providers.ProviderError, match="inference-profile prefix"):
        renderer.renderer_spec(config)


def test_configured_anthropic_model_carries_the_prefix(base_config):
    spec = renderer.renderer_spec(base_config)
    assert spec.model.startswith("us.")
    assert spec.provider == "bedrock"


# --------------------------------------------------------------- credentials ---

def test_missing_credentials_error_names_aws_credentials(monkeypatch, base_config):
    monkeypatch.setattr(boto3, "Session",
                        lambda **kwargs: _FakeSession(None, **kwargs))
    with pytest.raises(providers.ProviderError, match="AWS credentials"):
        renderer.build_client(base_config)


def test_present_credentials_yield_a_client(monkeypatch, base_config):
    monkeypatch.setattr(boto3, "Session",
                        lambda **kwargs: _FakeSession(object(), **kwargs))
    assert renderer.build_client(base_config) == "client:bedrock-runtime"


def test_oracle_missing_credentials_error_names_aws_credentials(monkeypatch,
                                                               base_config):
    monkeypatch.setattr(boto3, "Session",
                        lambda **kwargs: _FakeSession(None, **kwargs))
    with pytest.raises(providers.ProviderError, match="AWS credentials"):
        answerability.oracle_client(base_config)


def test_missing_boto3_error_names_boto3(monkeypatch, base_config):
    """One of the four fail-fast causes the spec requires by name."""
    import sys
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(providers.ProviderError, match="requires boto3"):
        renderer.build_client(base_config)
    with pytest.raises(providers.ProviderError, match="requires boto3"):
        answerability.oracle_client(base_config)


def test_missing_region_error_names_the_region_key(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(settings.ConfigError, match="aws.region"):
        settings.aws_region({"aws": {}})


def test_region_comes_from_the_environment_when_set(monkeypatch, base_config):
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    assert settings.aws_region(base_config) == "eu-west-2"


def test_region_falls_back_to_config(monkeypatch, base_config):
    monkeypatch.delenv("AWS_REGION", raising=False)
    assert settings.aws_region(base_config) == "us-east-1"


# ------------------------------------------------------------ inference args ---

class _CapturingClient:
    def __init__(self, text="Subject: x\nDate: y\n\nbody\n\nThanks,\nad sales ops"):
        self.calls = []
        self._text = text

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": self._text}]}}}


def test_bedrock_sends_temperature_and_never_top_p():
    """Claude on Bedrock rejects temperature and topP together."""
    client = _CapturingClient()
    spec = providers.ModelSpec("bedrock", "us.anthropic.claude-sonnet-4-6",
                               0.7, 1024, "renderer")
    providers.complete(spec, "sys", "msg", client=client)
    config = client.calls[0]["inferenceConfig"]
    assert config == {"temperature": 0.7, "maxTokens": 1024}
    assert "topP" not in config
    assert "top_p" not in config


def test_bedrock_rejects_an_empty_completion():
    client = _CapturingClient(text="   ")
    spec = providers.ModelSpec("bedrock", "amazon.nova-pro-v1:0", 0.7, 64,
                               "renderer")
    with pytest.raises(providers.ProviderError, match="empty completion"):
        providers.complete(spec, "sys", "msg", client=client)


def test_oracle_call_sends_temperature_zero_and_one_question(base_config):
    client = _CapturingClient(text="INSUFFICIENT_EVIDENCE")
    spec = answerability.oracle_spec(base_config)
    answer = answerability.ask_oracle(client, spec, "CORPUS", "What is the rate?")
    assert answer == "INSUFFICIENT_EVIDENCE"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["inferenceConfig"] == {"temperature": 0.0, "maxTokens": 512}
    assert len(call["messages"]) == 1
    assert "What is the rate?" in call["messages"][0]["content"][0]["text"]


# ------------------------------------------------------------- cross-family ---

def test_oracle_model_must_be_cross_family(base_config):
    spec = answerability.oracle_spec(base_config)
    assert spec.model == "amazon.nova-pro-v1:0"
    assert spec.provider == "bedrock"


def test_same_family_oracle_is_rejected(base_config):
    config = dict(base_config)
    config["answerability"] = dict(base_config["answerability"],
                                   oracle_model="us.anthropic.claude-haiku-4-5")
    with pytest.raises(answerability.OracleError, match="same model family"):
        answerability.oracle_spec(config)


# ------------------------------------------------------------- retry prompts ---

def test_retry_prompt_carries_the_previous_draft_and_the_failure_list():
    """Blind retries failed 4/4 on the same omission; the draft is mandatory."""
    prompt = renderer.RETRY_PROMPT.format(previous_draft="THE DRAFT",
                                          failures="THE FAILURES")
    assert "THE DRAFT" in prompt
    assert "THE FAILURES" in prompt
    assert "smallest possible set of edits" in prompt


def test_user_prompt_lists_every_fact(artifacts):
    from scenarios.renderer import build_manifest, deal_name_map
    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(artifacts.events[0], artifacts.facts_by_id, deal_names)
    prompt = renderer.user_prompt(manifest)
    for fact in manifest["facts"]:
        assert fact["value"] in prompt
        assert fact["attribute"] in prompt


def test_llm_render_retries_until_fidelity_passes(monkeypatch, artifacts, base_config):
    """The loop must stop retrying the moment fidelity passes."""
    from scenarios.renderer import build_manifest, deal_name_map
    from scenarios.templates import render_deterministic

    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(artifacts.events[0], artifacts.facts_by_id, deal_names)
    good = render_deterministic(manifest)

    drafts = iter(["Subject: nope\nDate: nope\n\nnothing useful here.", good])
    calls = []

    def fake_complete(spec, system, message, *, client=None):
        calls.append(message)
        return next(drafts)

    monkeypatch.setattr(renderer, "complete", fake_complete)
    outcome = renderer.render_llm(manifest, client=None, config=base_config)
    assert outcome["attempts"] == 2
    assert outcome["report"].ok
    # The second call must have carried the first draft back to the model.
    assert "nothing useful here" in calls[1]


def test_llm_render_seed_writes_the_same_artifacts_as_the_template_path(
        monkeypatch, artifacts, tmp_path):
    """The whole LLM path, orchestration included, with the network stubbed.

    Only the Bedrock call itself is replaced. Manifest construction, the fidelity
    loop, the retry accounting and every artifact are the real code, so this is
    the check that the LLM path is not a stub.
    """
    from scenarios.renderer import build_manifest, deal_name_map
    from scenarios.templates import render_deterministic

    deal_names = deal_name_map(artifacts.facts)
    by_event = {}
    for event in artifacts.events:
        manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
        by_event[manifest["subject"]] = render_deterministic(manifest)

    def fake_complete(spec, system, message, *, client=None):
        subject = message.split("Subject: ", 1)[1].split("\n", 1)[0]
        return by_event[subject]

    monkeypatch.setattr(renderer, "complete", fake_complete)
    monkeypatch.setattr(renderer, "build_client", lambda config: "stub-client")

    results = renderer.render_seed(artifacts.seed, deterministic=False,
                                   out_root=tmp_path,
                                   universe_root=artifacts.universe_root, limit=4)
    assert results["render_mode"] == "llm"
    assert results["scenarios"] == 4
    assert results["fidelity_failed"] == 0
    assert results["attempt_histogram"] == {"1": 4}

    import json
    for directory in sorted(path for path in
                            (tmp_path / str(artifacts.seed)).iterdir()
                            if path.is_dir()):
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        assert meta["render_mode"] == "llm"
        assert meta["model"] == "us.anthropic.claude-sonnet-4-6"
        assert meta["template_version"] is None
        assert meta["fidelity"] == "pass"
        assert meta["timestamp"]
        assert meta["attempt_history"] == [
            {"attempt": 1, "status": "pass", "missing_facts": 0, "unsupported": 0}]
        assert (directory / "rendered.md").read_text(encoding="utf-8").strip()
        assert (directory / "manifest.json").exists()


def test_llm_render_marks_fidelity_failed_when_retries_are_exhausted(
        monkeypatch, artifacts, base_config):
    """Exhausted retries are recorded as failed, never silently shipped."""
    from scenarios.renderer import build_manifest, deal_name_map

    deal_names = deal_name_map(artifacts.facts)
    manifest = build_manifest(artifacts.events[0], artifacts.facts_by_id, deal_names)

    monkeypatch.setattr(renderer, "complete",
                        lambda *args, **kwargs: "Subject: x\nDate: y\n\nuseless.")
    outcome = renderer.render_llm(manifest, client=None, config=base_config)
    assert outcome["attempts"] == int(base_config["renderer"]["max_retries"]) + 1
    assert not outcome["report"].ok
    assert outcome["report"].status == "failed"
