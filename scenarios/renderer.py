"""Render universe events into business correspondence.

Two paths, both real:

``--deterministic``
    Template rendering from :mod:`scenarios.templates`. Zero network calls.
    This is what CI runs.

default (LLM)
    Bedrock ``converse`` with the model from ``config.yaml``. This path is
    written out in full here, not stubbed. A stub in this position once let a
    "complete" increment ship that could not render anything.

Both paths write the same three artifacts per scenario and both are verified by
:mod:`verify.fidelity`. The LLM path additionally retries on fidelity failure,
feeding the previous draft and the exact failure list back to the model and
asking for minimal edits. Blind retries -- re-asking with the original prompt --
failed 4 out of 4 times on the same omission, which is why the previous draft is
mandatory in the retry.

Usage::

    python3 -m scenarios.renderer --seed 42 --deterministic
    python3 -m scenarios.renderer --seed 42                    # LLM path
    python3 -m scenarios.renderer --seed 42 --limit 5          # cost control
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import settings
from universe.generator import load_universe
from universe.schema import derive_trust_class
from verify import fidelity

from .templates import TEMPLATE_VERSION, long_date, render_deterministic

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------

def deal_name_map(facts: list[dict]) -> dict[str, str]:
    return {fact["entity_id"]: fact["value"] for fact in facts
            if fact["attribute"] == "deal_name"}


def build_manifest(event: dict, facts_by_id: dict[str, dict],
                   deal_names: dict[str, str]) -> dict:
    """The complete, self-contained render input for one event.

    ``timestamp`` is part of the manifest and is also copied into ``meta.json``.
    A previous build kept it only in the manifest, and every consumer that
    ordered scenarios by ``meta.json`` silently got insertion order instead.
    """
    facts = [facts_by_id[fact_id] for fact_id in event["fact_ids"]]
    return {
        "event_id": event["event_id"],
        "seed": event["seed"],
        "scenario_kind": event["scenario_kind"],
        "subject": event["subject"],
        "timestamp": event["timestamp"],
        "channel": event["channel"],
        "author": event["author"],
        "trust_class": derive_trust_class(event["channel"], event["author"]),
        "advertiser_id": event["advertiser_id"],
        "advertiser_name": event["advertiser_name"],
        "agency_name": event["agency_name"],
        "contact_name": event["contact_name"],
        "deal_id": event["deal_id"],
        "deal_name": deal_names.get(event["deal_id"]) if event["deal_id"] else None,
        "io_id": event["io_id"],
        "facts": [
            {
                "fact_id": fact["fact_id"],
                "entity": fact["entity"],
                "entity_id": fact["entity_id"],
                "attribute": fact["attribute"],
                "value": fact["value"],
                "volatility_class": fact["volatility_class"],
                "trust_class": fact["trust_class"],
                "validity_interval": fact["validity_interval"],
                "supersedes": fact["supersedes"],
                "injection": fact["injection"],
                "plausibility": fact["plausibility"],
                "never_memorize": fact["never_memorize"],
            }
            for fact in facts
        ],
    }


# --------------------------------------------------------------------------
# LLM path
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You write a single piece of realistic, informal ad-sales correspondence.

Voice: you are writing as the {author} on the {channel} channel. Sound like a
working ad-sales operations person mid-thread, not like a report.

Absolute requirements:
1. Output exactly one message. No preamble, no explanation, no commentary about
   these instructions, no markdown headings beyond the Subject and Date lines
   you are given.
2. Weave every supplied fact into flowing prose. Do not use bullet lists,
   numbered lists, tables, or "Label: value" fields. If a fact would naturally
   be a field, say it in a sentence instead.
3. Company names, person names, order-line ids, and targeting codes must appear
   character-exact as supplied. Write CTV, not "Connected TV". Write A25-54, not
   "adults 25 to 54". Write CA/NY/TX, not "California, New York and Texas".
4. Introduce nothing that is not supplied. No extra dates, no extra dollar
   amounts, no percentages, no invented people, no invented agencies, no
   "as we discussed on Tuesday". The subject line and the date are part of what
   you were supplied.
5. Sign off with the role "ad sales ops". Never sign with a personal name and
   never leave a placeholder such as [Your Name].

Begin the output with the Subject line, then the Date line, then a blank line,
then the message.
"""

USER_PROMPT = """\
Subject: {subject}
Date: {date_long}
Channel: {channel}
Author: {author}
This message is about: {about}

Facts that must all appear, woven into the prose:
{fact_lines}

Write the message now.
"""

RETRY_PROMPT = """\
The draft below failed verification. Make the smallest possible set of edits to
fix exactly the problems listed, and keep everything else word for word.

Previous draft:
---
{previous_draft}
---

{failures}

Rewrite the full message with those fixes applied. Same rules as before: one
message, prose only, no invented details, sign off as ad sales ops.
"""


class RenderError(RuntimeError):
    """Raised when the LLM render path cannot run or cannot succeed."""


def _about_line(manifest: dict) -> str:
    parts = [f"advertiser {manifest['advertiser_name']}"]
    if manifest.get("agency_name"):
        parts.append(f"agency {manifest['agency_name']}")
    if manifest.get("contact_name"):
        parts.append(f"contact {manifest['contact_name']}")
    if manifest.get("deal_name"):
        parts.append(f"deal {manifest['deal_name']}")
    if manifest.get("io_id"):
        parts.append(f"order line {manifest['io_id']}")
    return "; ".join(parts)


def user_prompt(manifest: dict) -> str:
    fact_lines = "\n".join(
        f"- {fact['attribute']}: {fact['value']}" for fact in manifest["facts"])
    return USER_PROMPT.format(
        subject=manifest["subject"],
        date_long=long_date(manifest["timestamp"]),
        channel=manifest["channel"],
        author=manifest["author"],
        about=_about_line(manifest),
        fact_lines=fact_lines,
    )


def system_prompt(manifest: dict) -> str:
    return SYSTEM_PROMPT.format(author=manifest["author"], channel=manifest["channel"])


def bedrock_client(config: dict):
    """Build a Bedrock runtime client, failing fast and by name.

    Every failure mode names the exact thing that is missing: boto3, the region,
    the credentials, or the model id. "Could not render" with no cause is a bug
    report nobody can act on.
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, NoCredentialsError  # noqa: F401
    except ImportError as exc:
        raise RenderError(
            "the LLM render path requires boto3, which is not installed; "
            "`pip install boto3` or use --deterministic") from exc

    region = settings.aws_region(config)
    if not region:
        raise RenderError(
            "no AWS region configured: set the AWS_REGION environment variable "
            "or aws.region in config.yaml")

    session = boto3.Session(region_name=region)
    if session.get_credentials() is None:
        raise RenderError(
            "no AWS credentials found for the Bedrock render path; configure AWS "
            "credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / "
            "AWS_SESSION_TOKEN, or an instance role) or use --deterministic")
    return session.client("bedrock-runtime")


def resolve_renderer_model(config: dict) -> str:
    """Read and sanity-check ``renderer.model``."""
    model = str(settings.require(config, "renderer.model")).strip()
    if "anthropic." in model and not model.startswith(("us.", "eu.", "apac.")):
        raise RenderError(
            f"renderer.model {model!r} is an Anthropic model id without an "
            f"inference-profile prefix; Bedrock requires 'us.{model}' (or the "
            f"prefix for your region)")
    return model


def converse(client, model: str, system: str, message: str, *,
             temperature: float, max_tokens: int) -> str:
    """One Bedrock ``converse`` call.

    ``topP`` is deliberately absent: Claude on Bedrock rejects a request that
    carries both ``temperature`` and ``topP``, so this testbed sends temperature
    only, for every model, on every call.
    """
    response = client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": message}]}],
        inferenceConfig={"temperature": float(temperature),
                         "maxTokens": int(max_tokens)},
    )
    blocks = response["output"]["message"]["content"]
    text = "".join(block.get("text", "") for block in blocks).strip()
    if not text:
        raise RenderError(f"Bedrock returned an empty message for model {model}")
    return text


def render_llm(manifest: dict, client, config: dict) -> dict:
    """Render one scenario through Bedrock with a fidelity-driven retry loop."""
    model = resolve_renderer_model(config)
    temperature = settings.require(config, "renderer.temperature")
    max_tokens = settings.require(config, "renderer.max_tokens")
    max_retries = int(settings.require(config, "renderer.max_retries"))

    system = system_prompt(manifest)
    message = user_prompt(manifest)
    attempts = 0
    draft = ""
    report = None
    history: list[dict] = []

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        draft = converse(client, model, system, message,
                         temperature=temperature, max_tokens=max_tokens)
        report = fidelity.verify_render(manifest, draft)
        history.append({"attempt": attempts, "status": report.status,
                        "missing_facts": len(report.missing_facts),
                        "unsupported": sum(len(v) for v in report.unsupported.values())})
        if report.ok:
            break
        message = RETRY_PROMPT.format(previous_draft=draft,
                                      failures=report.failure_summary())

    return {"text": draft, "attempts": attempts, "report": report,
            "history": history, "model": model}


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_scenario(out_dir: Path, manifest: dict, text: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "rendered.md").write_text(text, encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_seed(seed: int, *, deterministic: bool, out_root: Path | None = None,
                universe_root: Path | None = None, limit: int | None = None,
                only: str | None = None, config: dict | None = None) -> dict:
    """Render every (or a limited slice of) scenario for ``seed``."""
    facts, events = load_universe(seed, universe_root)
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    deal_names = deal_name_map(facts)
    root = Path(out_root or DEFAULT_OUT_ROOT) / str(seed)

    selected = events
    if only:
        wanted = {part.strip() for part in only.split(",") if part.strip()}
        selected = [event for event in events if event["event_id"] in wanted]
        if not selected:
            raise RenderError(f"no events matched --only {only!r}")
    if limit is not None:
        selected = selected[:limit]

    client = None
    if not deterministic:
        config = config or settings.load_config()
        client = bedrock_client(config)

    results = {"seed": seed, "render_mode": "deterministic" if deterministic else "llm",
               "scenarios": 0, "fidelity_pass": 0, "fidelity_failed": 0,
               "attempt_histogram": {}}

    for event in selected:
        manifest = build_manifest(event, facts_by_id, deal_names)
        if deterministic:
            text = render_deterministic(manifest)
            report = fidelity.verify_render(manifest, text)
            attempts, model = 1, None
            history: list[dict] = []
        else:
            outcome = render_llm(manifest, client, config)
            text = outcome["text"]
            report = outcome["report"]
            attempts = outcome["attempts"]
            model = outcome["model"]
            history = outcome["history"]

        meta = {
            "event_id": manifest["event_id"],
            "seed": seed,
            "timestamp": manifest["timestamp"],
            "scenario_kind": manifest["scenario_kind"],
            "channel": manifest["channel"],
            "author": manifest["author"],
            "render_mode": results["render_mode"],
            "model": model,
            "template_version": TEMPLATE_VERSION if deterministic else None,
            "attempts": attempts,
            "attempt_history": history,
            "fidelity": report.status,
            "fidelity_missing_facts": report.missing_facts,
            "fidelity_unsupported": report.unsupported,
        }
        write_scenario(root / manifest["event_id"], manifest, text, meta)

        results["scenarios"] += 1
        key = str(attempts)
        results["attempt_histogram"][key] = results["attempt_histogram"].get(key, 0) + 1
        if report.ok:
            results["fidelity_pass"] += 1
        else:
            results["fidelity_failed"] += 1

    root.mkdir(parents=True, exist_ok=True)
    (root / "render_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render universe events as prose.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="template path, zero network calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="render only the first N scenarios (cost control)")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated event ids to render")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--universe-root", type=Path, default=None)
    args = parser.parse_args(argv)

    results = render_seed(args.seed, deterministic=args.deterministic,
                          out_root=args.out_root, universe_root=args.universe_root,
                          limit=args.limit, only=args.only)
    print(f"seed {results['seed']} [{results['render_mode']}]: "
          f"{results['scenarios']} scenarios, "
          f"{results['fidelity_pass']} fidelity pass, "
          f"{results['fidelity_failed']} failed")
    print(f"  attempts: {json.dumps(results['attempt_histogram'], sort_keys=True)}")
    return 0 if results["fidelity_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
