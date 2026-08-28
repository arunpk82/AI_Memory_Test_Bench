"""Provider layer tests, hermetic.

Groq and Gemini are plain HTTPS calls, so the wire format is the whole contract:
the URL, the auth header, the body shape, and how a reply is parsed. All of it
is pinned here against a fake ``urlopen``. Nothing in this file touches a
network or needs a key.

The one property asserted for every provider is that a request carries
``temperature`` and never ``top_p``. That rule started as a Claude-on-Bedrock
constraint; applying it everywhere is what stops it from becoming a surprise on
the day someone switches provider.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

import providers
from providers import ModelSpec, ProviderError


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Recorder:
    """Stands in for urllib.request.urlopen and records what was sent."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append({
            "url": request.full_url,
            "method": request.method,
            "headers": {key.lower(): value for key, value in request.headers.items()},
            "body": json.loads(request.data.decode("utf-8")),
            "timeout": timeout,
        })
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.payload)


def _groq_ok(text="the rate is 26.70"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _gemini_ok(text="the rate is 26.70"):
    return {"candidates": [{"content": {"parts": [{"text": text}]},
                            "finishReason": "STOP"}]}


GROQ = ModelSpec("groq", "llama-3.3-70b-versatile", 0.7, 1024, "renderer")
GEMINI = ModelSpec("gemini", "gemini-2.5-flash", 0.0, 512, "oracle")


# ------------------------------------------------------------------- specs ---

def test_unknown_provider_is_rejected_and_lists_the_valid_ones():
    with pytest.raises(ProviderError, match="bedrock, groq, gemini"):
        ModelSpec("openai", "gpt-4", 0.7, 100, "renderer")


def test_empty_model_id_is_rejected():
    with pytest.raises(ProviderError, match="model id is empty"):
        ModelSpec("groq", "  ", 0.7, 100, "renderer")


def test_bedrock_prefix_on_a_non_bedrock_provider_is_rejected():
    spec = ModelSpec("groq", "us.anthropic.claude-sonnet-4-6", 0.7, 100, "renderer")
    with pytest.raises(ProviderError, match="drop the prefix"):
        providers._validate_model_id(spec)


@pytest.mark.parametrize("provider,model,expected", [
    ("bedrock", "us.anthropic.claude-sonnet-4-6", "anthropic"),
    ("bedrock", "anthropic.claude-haiku-4-5", "anthropic"),
    ("bedrock", "amazon.nova-pro-v1:0", "amazon"),
    ("bedrock", "meta.llama3-70b-instruct-v1:0", "meta"),
    ("groq", "llama-3.3-70b-versatile", "groq:llama"),
    ("gemini", "gemini-2.5-flash", "gemini:gemini"),
])
def test_family_detection(provider, model, expected):
    assert ModelSpec(provider, model, 0.5, 10, "renderer").family == expected


def test_the_same_anthropic_model_is_one_family_across_prefixes():
    left = ModelSpec("bedrock", "us.anthropic.claude-sonnet-4-6", 0.5, 10, "renderer")
    right = ModelSpec("bedrock", "anthropic.claude-haiku-4-5", 0.5, 10, "oracle")
    assert left.family == right.family


def test_different_providers_are_different_families():
    left = ModelSpec("groq", "llama-3.3-70b-versatile", 0.5, 10, "renderer")
    right = ModelSpec("gemini", "gemini-2.5-flash", 0.5, 10, "oracle")
    assert left.family != right.family


def test_spec_from_config_defaults_to_bedrock():
    config = {"renderer": {"model": "us.anthropic.claude-sonnet-4-6",
                           "temperature": 0.7, "max_tokens": 1024}}
    spec = providers.spec_from_config(config, "renderer", model_key="model")
    assert spec.provider == "bedrock"


def test_spec_from_config_reads_an_explicit_provider():
    config = {"renderer": {"provider": "groq", "model": "llama-3.3-70b-versatile",
                           "temperature": 0.4, "max_tokens": 900}}
    spec = providers.spec_from_config(config, "renderer", model_key="model")
    assert (spec.provider, spec.model, spec.temperature, spec.max_tokens) == \
        ("groq", "llama-3.3-70b-versatile", 0.4, 900)


# ------------------------------------------------------------------- keys ---

def test_missing_groq_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        providers.api_key("groq")


def test_missing_gemini_key_names_both_accepted_variables(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        providers.api_key("gemini")


def test_gemini_falls_back_to_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")
    assert providers.api_key("gemini") == "fallback-key"


def test_build_client_resolves_the_key_eagerly(monkeypatch):
    """A missing key must fail before any scenario is rendered, not on call 1."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        providers.build_client(GROQ, {})
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert providers.build_client(GROQ, {}) == "k"


# ------------------------------------------------------------------- groq ---

def test_groq_request_shape(monkeypatch):
    recorder = _Recorder(_groq_ok())
    monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
    assert providers.complete(GROQ, "SYSTEM", "USER", client="secret-key") == \
        "the rate is 26.70"

    sent = recorder.requests[0]
    assert sent["url"] == providers.GROQ_ENDPOINT
    assert sent["method"] == "POST"
    assert sent["headers"]["authorization"] == "Bearer secret-key"
    assert sent["headers"]["content-type"] == "application/json"
    assert sent["body"]["model"] == "llama-3.3-70b-versatile"
    assert sent["body"]["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    assert sent["body"]["temperature"] == 0.7
    assert sent["body"]["max_tokens"] == 1024
    assert "top_p" not in sent["body"]


def test_groq_malformed_response_is_reported(monkeypatch):
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        _Recorder({"choices": []}))
    with pytest.raises(ProviderError, match="no message content"):
        providers.complete(GROQ, "s", "u", client="k")


def test_groq_empty_completion_is_reported(monkeypatch):
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        _Recorder(_groq_ok("   ")))
    with pytest.raises(ProviderError, match="empty completion"):
        providers.complete(GROQ, "s", "u", client="k")


# ----------------------------------------------------------------- gemini ---

def test_gemini_request_shape(monkeypatch):
    recorder = _Recorder(_gemini_ok())
    monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
    assert providers.complete(GEMINI, "SYSTEM", "USER", client="secret-key") == \
        "the rate is 26.70"

    sent = recorder.requests[0]
    assert sent["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert sent["headers"]["x-goog-api-key"] == "secret-key"
    assert sent["body"]["systemInstruction"] == {"parts": [{"text": "SYSTEM"}]}
    assert sent["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "USER"}]}]
    assert sent["body"]["generationConfig"] == {"temperature": 0.0,
                                                "maxOutputTokens": 512}
    assert "topP" not in sent["body"]["generationConfig"]
    assert "top_p" not in sent["body"]["generationConfig"]


def test_gemini_key_never_travels_in_the_url(monkeypatch):
    """A key in a query string ends up in logs and proxies."""
    recorder = _Recorder(_gemini_ok())
    monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
    providers.complete(GEMINI, "s", "u", client="secret-key")
    assert "secret-key" not in recorder.requests[0]["url"]


def test_gemini_blocked_prompt_is_reported(monkeypatch):
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        _Recorder({"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(ProviderError, match="no candidates"):
        providers.complete(GEMINI, "s", "u", client="k")


def test_gemini_truncated_candidate_suggests_raising_max_tokens(monkeypatch):
    monkeypatch.setattr(providers.urllib.request, "urlopen", _Recorder(
        {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}))
    with pytest.raises(ProviderError, match="raise max_tokens"):
        providers.complete(GEMINI, "s", "u", client="k")


# ------------------------------------------------------------ http errors ---

def _http_error(code):
    return urllib.error.HTTPError("https://x", code, "boom", {}, None)


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failure_names_the_key_variable(monkeypatch, code):
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        _Recorder(error=_http_error(code)))
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        providers.complete(GROQ, "s", "u", client="k")


def test_unknown_model_names_the_config_key(monkeypatch):
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        _Recorder(error=_http_error(404)))
    with pytest.raises(ProviderError, match="config.yaml"):
        providers.complete(GROQ, "s", "u", client="k")


def test_non_retryable_error_is_raised_immediately(monkeypatch):
    recorder = _Recorder(error=_http_error(400))
    monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
    with pytest.raises(ProviderError, match="HTTP 400"):
        providers.complete(GROQ, "s", "u", client="k")
    assert len(recorder.requests) == 1


def test_rate_limit_is_retried_then_reported(monkeypatch):
    recorder = _Recorder(error=_http_error(429))
    monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    with pytest.raises(ProviderError, match="unreachable"):
        providers.complete(GROQ, "s", "u", client="k")
    assert len(recorder.requests) == providers.HTTP_RETRIES


def test_a_transient_failure_recovers_on_retry(monkeypatch):
    attempts = {"n": 0}

    def flaky(request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(503)
        return _FakeResponse(_groq_ok())

    monkeypatch.setattr(providers.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    assert providers.complete(GROQ, "s", "u", client="k") == "the rate is 26.70"
    assert attempts["n"] == 2


def test_non_json_body_is_reported(monkeypatch):
    class _Garbage:
        def read(self):
            return b"<html>gateway</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda request, timeout=None: _Garbage())
    with pytest.raises(ProviderError, match="non-JSON"):
        providers.complete(GROQ, "s", "u", client="k")


def test_requests_carry_a_timeout(monkeypatch):
    """No provider call may hang forever; a stalled audit looks like a slow one."""
    recorder = _Recorder(_groq_ok())
    monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
    providers.complete(GROQ, "s", "u", client="k")
    assert recorder.requests[0]["timeout"] == providers.HTTP_TIMEOUT_SECONDS


# ------------------------------------------------------- no top_p, anywhere ---

def test_no_provider_ever_sends_top_p(monkeypatch):
    for spec, payload in ((GROQ, _groq_ok()), (GEMINI, _gemini_ok())):
        recorder = _Recorder(payload)
        monkeypatch.setattr(providers.urllib.request, "urlopen", recorder)
        providers.complete(spec, "s", "u", client="k")
        body = json.dumps(recorder.requests[0]["body"])
        assert "top_p" not in body and "topP" not in body, spec.provider


def test_bedrock_needs_a_client():
    spec = ModelSpec("bedrock", "amazon.nova-pro-v1:0", 0.0, 10, "oracle")
    with pytest.raises(ProviderError, match="needs a client"):
        providers.complete(spec, "s", "u", client=None)


# ------------------------------------------ end to end through the callers ---

def test_renderer_can_be_driven_by_groq(monkeypatch, artifacts, tmp_path):
    """A whole seed rendered through Groq, with only the socket replaced."""
    from scenarios.renderer import build_manifest, deal_name_map, render_seed
    from scenarios.templates import render_deterministic

    deal_names = deal_name_map(artifacts.facts)
    by_subject = {}
    for event in artifacts.events:
        manifest = build_manifest(event, artifacts.facts_by_id, deal_names)
        by_subject[manifest["subject"]] = render_deterministic(manifest)

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        prompt = body["messages"][1]["content"]
        subject = prompt.split("Subject: ", 1)[1].split("\n", 1)[0]
        return _FakeResponse(_groq_ok(by_subject[subject]))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    import settings
    config = settings.load_config()
    config["renderer"] = dict(config["renderer"], provider="groq",
                              model="llama-3.3-70b-versatile")

    results = render_seed(artifacts.seed, deterministic=False, out_root=tmp_path,
                          universe_root=artifacts.universe_root, limit=3,
                          config=config)
    assert results["fidelity_failed"] == 0
    assert results["scenarios"] == 3

    meta_path = next((tmp_path / str(artifacts.seed)).iterdir()) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["provider"] == "groq"
    assert meta["model"] == "llama-3.3-70b-versatile"
    assert meta["fidelity"] == "pass"


def test_cross_family_rule_is_satisfied_by_mixing_providers():
    from verify.answerability import oracle_spec

    config = {
        "renderer": {"provider": "gemini", "model": "gemini-2.5-flash",
                     "temperature": 0.7, "max_tokens": 1024},
        "answerability": {"provider": "groq", "oracle_model": "llama-3.3-70b-versatile",
                          "temperature": 0.0, "max_tokens": 512},
    }
    assert oracle_spec(config).provider == "groq"


def test_cross_family_rule_still_blocks_one_provider_grading_itself():
    from verify.answerability import OracleError, oracle_spec

    config = {
        "renderer": {"provider": "gemini", "model": "gemini-2.5-flash",
                     "temperature": 0.7, "max_tokens": 1024},
        "answerability": {"provider": "gemini", "oracle_model": "gemini-2.5-pro",
                          "temperature": 0.0, "max_tokens": 512},
    }
    with pytest.raises(OracleError, match="same model family"):
        oracle_spec(config)
