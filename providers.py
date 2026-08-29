"""One way to ask a language model a question.

The renderer and the answerability oracle both need a completion. They get it
from here, and only from here. This is the same rule the matching policy lives
under: two implementations of a shared operation disagree silently, and the
disagreement propagates into every number downstream. If the renderer sent
``top_p`` and the oracle did not, "the model" would mean two different things in
one report.

Three providers are supported. Bedrock is the default because it is what the
build specification pins; Groq and Gemini exist so the testbed is usable without
an AWS account.

* ``bedrock`` uses the ``converse`` API through boto3.
* ``groq`` uses the OpenAI-compatible chat-completions endpoint.
* ``gemini`` uses ``generateContent``.

Groq and Gemini are plain HTTPS calls made with the standard library, so
choosing either one adds no dependency at all. Only Bedrock needs boto3.

**No provider ever sends ``top_p``.** Claude on Bedrock rejects a request
carrying both ``temperature`` and ``top_p``, and rather than make that a
Bedrock-only special case -- the kind of asymmetry that becomes a confusing bug
on the day someone switches providers -- every provider here sends temperature
alone.

Every failure names the exact thing that is missing: the package, the
credential, the environment variable, or the model id. "Completion failed" with
no cause is a bug report nobody can act on.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import settings

PROVIDERS = ("bedrock", "groq", "gemini")
DEFAULT_PROVIDER = "bedrock"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: urllib.request adds ``User-Agent: Python-urllib/3.x`` unless we set one.
#: Groq sits behind Cloudflare, which rejects that signature with HTTP 403
#: error 1010 (browser_signature_banned) even when Authorization is a valid
#: ``Bearer <key>``. Invoke-WebRequest succeeds with the same key because
#: PowerShell sends a different User-Agent. The fake urlopen in tests never
#: sees the default, which is why this stayed green.
HTTP_USER_AGENT = "mem-testbed/2"

#: Environment variables that carry each provider's credential, in priority
#: order. Bedrock is absent because boto3 resolves credentials itself.
API_KEY_VARS = {
    "groq": ("GROQ_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

#: Anthropic model ids on Bedrock need an inference-profile prefix.
BEDROCK_PROFILE_PREFIXES = ("us.", "eu.", "apac.")

HTTP_TIMEOUT_SECONDS = 120
HTTP_RETRIES = 3
#: 429 is handled separately: Groq's per-minute ceiling is not a blip, and
#: retrying it on the same cadence as a 503 just re-hits the ceiling.
RETRYABLE_STATUS = frozenset({408, 500, 502, 503, 504})
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BACKOFF_START_SECONDS = 8
RATE_LIMIT_MAX_WAIT_SECONDS = 300


class ProviderError(RuntimeError):
    """Raised when a completion cannot be requested or cannot be parsed."""


class EmptyCompletionError(ProviderError):
    """The model returned a 200 with no usable text.

    A distinct type because it is *recoverable*: unlike an auth or model-id
    error, an empty completion is often intermittent (a reasoning model such as
    ``openai/gpt-oss-120b`` can spend its whole ``max_tokens`` budget on
    chain-of-thought and return an empty ``content`` with
    ``finish_reason: "length"``). The render loop retries it instead of letting
    one scenario kill the whole seed run.
    """


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to ask one provider for one completion."""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    #: "renderer" or "oracle". Only used to make error messages say which of the
    #: two configurations is wrong.
    role: str

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ProviderError(
                f"{self.role} provider {self.provider!r} is not supported; choose "
                f"one of {', '.join(PROVIDERS)}")
        if not self.model or not str(self.model).strip():
            raise ProviderError(f"{self.role} model id is empty")

    @property
    def family(self) -> str:
        """A coarse model family, used to enforce cross-family auditing.

        The provider is part of the family: a Gemini model and a Llama model
        served by Groq are not the same family regardless of how their ids are
        spelled, and two Anthropic models are the same family even if one is
        reached through Bedrock and one is not.
        """
        model = self.model
        for prefix in BEDROCK_PROFILE_PREFIXES:
            if model.startswith(prefix):
                model = model[len(prefix):]
                break
        vendor = model.split(".", 1)[0] if "." in model else model.split("/", 1)[0]
        vendor = vendor.split("-", 1)[0].casefold()
        if self.provider == "bedrock":
            return vendor
        return f"{self.provider}:{vendor}"


def spec_from_config(config: dict, role: str, *, model_key: str) -> ModelSpec:
    """Build a :class:`ModelSpec` from a config section.

    ``provider`` is optional and defaults to Bedrock, so a configuration written
    before providers existed keeps working unchanged.
    """
    section = "renderer" if role == "renderer" else "answerability"
    provider = str(config.get(section, {}).get("provider", DEFAULT_PROVIDER)).strip()
    model = str(settings.require(config, f"{section}.{model_key}")).strip()
    spec = ModelSpec(
        provider=provider,
        model=model,
        temperature=float(settings.require(config, f"{section}.temperature")),
        max_tokens=int(settings.require(config, f"{section}.max_tokens")),
        role=role,
    )
    _validate_model_id(spec)
    return spec


def _validate_model_id(spec: ModelSpec) -> None:
    if spec.provider == "bedrock" and "anthropic." in spec.model \
            and not spec.model.startswith(BEDROCK_PROFILE_PREFIXES):
        raise ProviderError(
            f"{spec.role} model {spec.model!r} is an Anthropic model id without an "
            f"inference-profile prefix; Bedrock requires 'us.{spec.model}' (or the "
            f"prefix for your region)")
    if spec.provider != "bedrock" and spec.model.startswith(BEDROCK_PROFILE_PREFIXES):
        raise ProviderError(
            f"{spec.role} model {spec.model!r} carries a Bedrock inference-profile "
            f"prefix but the provider is {spec.provider!r}; drop the prefix")


# --------------------------------------------------------------------------
# Credentials and clients
# --------------------------------------------------------------------------

def api_key(provider: str) -> str:
    """The API key for an HTTP provider, or an error naming the variable."""
    variables = API_KEY_VARS[provider]
    for name in variables:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    joined = " or ".join(variables)
    raise ProviderError(
        f"no API key found for provider {provider!r}; set the {joined} environment "
        f"variable")


def build_client(spec: ModelSpec, config: dict):
    """Prepare whatever the provider needs, failing fast and by name.

    Bedrock returns a boto3 client. The HTTP providers have no client object, so
    they return their resolved API key -- resolved *now*, so a missing key is an
    error before any work is done rather than on the first call.
    """
    if spec.provider == "bedrock":
        return _bedrock_client(spec, config)
    return api_key(spec.provider)


def _bedrock_client(spec: ModelSpec, config: dict):
    try:
        import boto3
    except ImportError as exc:
        raise ProviderError(
            f"the {spec.role} is configured for Bedrock, which requires boto3; "
            f"run `pip install -r requirements-bedrock.txt`, or set "
            f"provider: groq / provider: gemini in config.yaml, or use the "
            f"network-free path") from exc

    region = settings.aws_region(config)
    if not region:
        raise ProviderError(
            "no AWS region configured: set the AWS_REGION environment variable or "
            "aws.region in config.yaml")
    session = boto3.Session(region_name=region)
    if session.get_credentials() is None:
        raise ProviderError(
            f"no AWS credentials found for the {spec.role}; configure AWS "
            f"credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / "
            f"AWS_SESSION_TOKEN, or an instance role), or switch to a provider "
            f"that uses an API key (groq, gemini)")
    return session.client("bedrock-runtime")


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------

def complete(spec: ModelSpec, system: str, user: str, *, client=None) -> str:
    """One completion. The single entry point for every model call."""
    if spec.provider == "bedrock":
        text = _complete_bedrock(spec, system, user, client)
    elif spec.provider == "groq":
        text = _complete_groq(spec, system, user, client)
    elif spec.provider == "gemini":
        text = _complete_gemini(spec, system, user, client)
    else:  # pragma: no cover - ModelSpec rejects this at construction
        raise ProviderError(f"unsupported provider {spec.provider!r}")

    text = (text or "").strip()
    if not text:
        raise EmptyCompletionError(
            f"{spec.provider} returned an empty completion for model {spec.model}")
    return text


def _complete_bedrock(spec: ModelSpec, system: str, user: str, client) -> str:
    if client is None:
        raise ProviderError("the Bedrock provider needs a client; call "
                            "build_client() first")
    response = client.converse(
        modelId=spec.model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"temperature": float(spec.temperature),
                         "maxTokens": int(spec.max_tokens)},
    )
    blocks = response["output"]["message"]["content"]
    return "".join(block.get("text", "") for block in blocks)


def _complete_groq(spec: ModelSpec, system: str, user: str, key) -> str:
    payload = {
        "model": spec.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": float(spec.temperature),
        "max_tokens": int(spec.max_tokens),
    }
    headers = {"Authorization": f"Bearer {key or api_key('groq')}",
               "Content-Type": "application/json"}
    body = _post_json(GROQ_ENDPOINT, payload, headers, spec)
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"groq response for {spec.model} had no message content: "
            f"{json.dumps(body)[:400]}") from exc
    content = (message.get("content") or "").strip()
    if not content:
        raise EmptyCompletionError(_empty_groq_detail(spec, choice, message, body))
    return content


def _empty_groq_detail(spec: ModelSpec, choice: dict, message: dict,
                       body: dict) -> str:
    """A diagnostic that shows the raw body: finish_reason, content, reasoning.

    ``openai/gpt-oss-120b`` returns its chain-of-thought in a separate
    ``reasoning`` field. When the answer field is empty, this is almost always
    ``finish_reason == "length"`` with a non-empty ``reasoning`` -- the budget
    was spent thinking. Surfacing the raw fields turns the old "empty
    completion" crash into an evidence trail.
    """
    finish = choice.get("finish_reason")
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    usage = body.get("usage") or {}
    parts = [
        f"{spec.provider} returned an empty completion for model {spec.model}",
        f"finish_reason={finish!r}",
        f"content={message.get('content')!r}",
        f"reasoning_chars={len(reasoning)}",
    ]
    if usage:
        parts.append(f"usage={json.dumps(usage, sort_keys=True)}")
    if finish == "length":
        parts.append(
            "the token budget was exhausted before a final answer -- this is a "
            "reasoning model spending max_tokens on chain-of-thought; raising "
            "renderer.max_tokens or lowering reasoning effort avoids it")
    if reasoning:
        parts.append(f"reasoning_snippet={reasoning[:200]!r}")
    parts.append(f"raw={json.dumps(body)[:600]}")
    return "; ".join(parts)


def _complete_gemini(spec: ModelSpec, system: str, user: str, key) -> str:
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": float(spec.temperature),
                             "maxOutputTokens": int(spec.max_tokens)},
    }
    headers = {"x-goog-api-key": key or api_key("gemini"),
               "Content-Type": "application/json"}
    url = GEMINI_ENDPOINT.format(model=spec.model)
    body = _post_json(url, payload, headers, spec)

    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback", {})
        raise ProviderError(
            f"gemini returned no candidates for {spec.model}; this usually means "
            f"the prompt was blocked. promptFeedback={json.dumps(feedback)[:300]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        reason = candidates[0].get("finishReason", "unknown")
        raise ProviderError(
            f"gemini returned an empty candidate for {spec.model} "
            f"(finishReason={reason}); if this is MAX_TOKENS, raise max_tokens")
    return text


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Seconds to wait from a Retry-After header, or None if none/unusable."""
    headers = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        wait = float(str(raw).strip())
    except ValueError:
        return None
    if wait < 0:
        return None
    return min(wait, RATE_LIMIT_MAX_WAIT_SECONDS)


def _post_json(url: str, payload: dict, headers: dict, spec: ModelSpec) -> dict:
    """POST JSON with bounded retries on transient failures.

    HTTP 429 is not lumped in with 503. A rate limit needs a wait measured in
    seconds (Retry-After, else exponential from
    ``RATE_LIMIT_BACKOFF_START_SECONDS``) and a distinct error when patience
    runs out, so it cannot be read as an auth or network failure.
    """
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    request_headers = dict(headers)
    request_headers.setdefault("User-Agent", HTTP_USER_AGENT)
    transient_attempts = 0
    rate_limit_hits = 0

    while True:
        request = urllib.request.Request(url, data=data, headers=request_headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request,
                                        timeout=HTTP_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _read_error(exc)
            if exc.code == 401 or exc.code == 403:
                variables = " or ".join(API_KEY_VARS[spec.provider])
                raise ProviderError(
                    f"{spec.provider} rejected the credentials for {spec.model} "
                    f"(HTTP {exc.code}); check {variables}. {detail}") from exc
            if exc.code == 404:
                raise ProviderError(
                    f"{spec.provider} does not know the model {spec.model!r} "
                    f"(HTTP 404); check the model id in config.yaml. "
                    f"{detail}") from exc
            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits > RATE_LIMIT_RETRIES:
                    raise ProviderError(
                        f"{spec.provider} rate-limited {spec.model} (HTTP 429) "
                        f"and ran out of patience after {RATE_LIMIT_RETRIES} "
                        f"retries. {detail}") from exc
                wait = _retry_after_seconds(exc)
                if wait is None:
                    wait = RATE_LIMIT_BACKOFF_START_SECONDS * (
                        2 ** (rate_limit_hits - 1))
                    wait = min(wait, RATE_LIMIT_MAX_WAIT_SECONDS)
                time.sleep(wait)
                continue
            if exc.code not in RETRYABLE_STATUS:
                raise ProviderError(
                    f"{spec.provider} returned HTTP {exc.code} for {spec.model}. "
                    f"{detail}") from exc
            last_error = exc
            transient_attempts += 1
            if transient_attempts >= HTTP_RETRIES:
                break
            time.sleep(2 ** transient_attempts)
            continue
        except urllib.error.URLError as exc:
            last_error = exc
            transient_attempts += 1
            if transient_attempts >= HTTP_RETRIES:
                break
            time.sleep(2 ** transient_attempts)
            continue
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{spec.provider} returned a non-JSON body for "
                f"{spec.model}") from exc

    raise ProviderError(
        f"{spec.provider} was unreachable for {spec.model} after {HTTP_RETRIES} "
        f"attempts: {last_error}")


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:300]
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""
