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
    python3 -m scenarios.renderer --seed 42 --force            # ignore existing passes
    python3 -m scenarios.renderer --seed 42 --provider groq --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import providers
import settings
from providers import EmptyCompletionError, complete
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
6. Write monetary amounts, dates, and quantities in natural business style:
   "$342,000" or "$342K" (not "342000"), "April 24" or "late April 2025"
   (not "2025-04-24"). Reformatting a supplied number or date into natural
   prose is expected and is NOT "introducing" anything — requirement 4
   forbids new facts, not natural formatting of supplied ones. Only company
   names, person names, order-line ids, and targeting codes must appear
   character-exact (requirement 3).
7. Refer to each fact by its natural business term, never the raw field label
   from the fact list. Those labels are database column names, not vocabulary
   for the email: "cpm_rate" is "the CPM", "line_status" is "the status" (or
   just state it: "the line is live"), "flight_start" and "flight_end" are
   "the flight" ("runs from X to Y", "flighting April 24 through June 23"),
   "budget_usd" is "the budget". Say what the field means in plain ad-sales
   language; never write the underscore label itself. This is about attribute
   NAMES only — the supplied values still follow requirements 3 and 6.

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

EMPTY_RETRY_PROMPT = """\
Your previous reply came back with an empty message body -- the answer field was
blank, most likely because the response ran past its length budget. Write the
message directly and concisely this time: do not deliberate at length before
answering, just produce the message.

{original}
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


#: A value that is *exactly* an ISO date and nothing else. Detection is by
#: shape, never by attribute name: any fact value shaped ``YYYY-MM-DD`` is a
#: date regardless of what it is called, and no other value shape is.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _naturalize_fact_value(value: str) -> str:
    """Long-form a bare ISO date so the model never copies a raw token.

    A raw ``2025-04-24`` in the fact list fights the system prompt: rule 4
    ("introduce nothing not supplied") plus a visible ISO token beats rule 6's
    permission to naturalize, so the model echoes the ISO string. Rewriting the
    date to ``April 24, 2025`` here removes the token it was tempted to copy.
    Fidelity is unaffected: matching canonicalizes both forms, and only date
    values are touched -- integers, names, codes and targeting strings, none of
    which match the exact ISO shape, pass through verbatim.
    """
    if _ISO_DATE_RE.match(value):
        return long_date(value)
    return value


def user_prompt(manifest: dict) -> str:
    fact_lines = "\n".join(
        f"- {fact['attribute']}: {_naturalize_fact_value(fact['value'])}"
        for fact in manifest["facts"])
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


def renderer_spec(config: dict) -> providers.ModelSpec:
    """Read and sanity-check the renderer's provider and model."""
    return providers.spec_from_config(config, "renderer", model_key="model")


def build_client(config: dict, spec: providers.ModelSpec | None = None):
    """Prepare the configured provider, failing fast and by name."""
    return providers.build_client(spec or renderer_spec(config), config)


def render_llm(manifest: dict, client, config: dict) -> dict:
    """Render one scenario through a model, with a fidelity-driven retry loop.

    Two failure kinds share the loop. A draft that fails fidelity is re-asked
    with the previous draft and the exact failures. An *empty* completion --
    common with reasoning models that spend the token budget thinking -- is
    also retried rather than raised, so one bad scenario does not abort the
    whole seed. If every attempt comes back empty, the empty draft fails
    fidelity naturally and the scenario is recorded as a fidelity failure.
    """
    spec = renderer_spec(config)
    max_retries = int(settings.require(config, "renderer.max_retries"))

    system = system_prompt(manifest)
    original = user_prompt(manifest)
    message = original
    attempts = 0
    draft = ""
    report = None
    history: list[dict] = []
    empty_note = None

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            draft = complete(spec, system, message, client=client)
        except EmptyCompletionError as exc:
            empty_note = str(exc)
            draft = ""
            report = fidelity.verify_render(manifest, draft)
            history.append({"attempt": attempts, "status": "empty",
                            "missing_facts": len(report.missing_facts),
                            "unsupported": sum(len(v) for v in
                                               report.unsupported.values())})
            message = EMPTY_RETRY_PROMPT.format(original=original)
            continue

        report = fidelity.verify_render(manifest, draft)
        history.append({"attempt": attempts, "status": report.status,
                        "missing_facts": len(report.missing_facts),
                        "unsupported": sum(len(v) for v in report.unsupported.values())})
        if report.ok:
            break
        message = RETRY_PROMPT.format(previous_draft=draft,
                                      failures=report.failure_summary())

    return {"text": draft, "attempts": attempts, "report": report,
            "history": history, "model": spec.model, "provider": spec.provider,
            "empty_note": empty_note}


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


def _already_complete(out_dir: Path, render_mode: str) -> bool:
    """True when this scenario already passed in the requested render mode."""
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("render_mode") == render_mode and meta.get("fidelity") == "pass"


def _override_config(config: dict, provider: str | None, model: str | None) -> dict:
    """Return a config copy with the renderer's provider and/or model replaced.

    Overrides come from the CLI and take precedence over ``config.yaml``. The
    committed defaults are never mutated: a shallow copy of the renderer section
    is enough because that is the only section touched.
    """
    if provider is None and model is None:
        return config
    merged = dict(config)
    merged["renderer"] = dict(config.get("renderer", {}))
    if provider is not None:
        merged["renderer"]["provider"] = provider
    if model is not None:
        merged["renderer"]["model"] = model
    return merged


def render_seed(seed: int, *, deterministic: bool, out_root: Path | None = None,
                universe_root: Path | None = None, limit: int | None = None,
                only: str | None = None, config: dict | None = None,
                force: bool = False, provider: str | None = None,
                model: str | None = None) -> dict:
    """Render every (or a limited slice of) scenario for ``seed``.

    Scenarios that already have a passing ``meta.json`` in the requested
    ``render_mode`` are skipped, so a killed LLM run can resume without
    re-spending. ``force=True`` re-renders everything.

    ``provider`` and ``model`` override ``config.yaml`` for this run when given,
    and flow through the spec into each scenario's ``meta.json`` verbatim.
    """
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

    requested_mode = "deterministic" if deterministic else "llm"
    client = None
    if not deterministic:
        config = config or settings.load_config()
        config = _override_config(config, provider, model)
        client = build_client(config)

    results = {"seed": seed, "render_mode": requested_mode,
               "scenarios": 0, "fidelity_pass": 0, "fidelity_failed": 0,
               "skipped": 0, "empty": 0, "attempt_histogram": {}}

    for event in selected:
        out_dir = root / event["event_id"]
        if not force and _already_complete(out_dir, requested_mode):
            results["scenarios"] += 1
            results["fidelity_pass"] += 1
            results["skipped"] += 1
            continue

        manifest = build_manifest(event, facts_by_id, deal_names)
        empty_note = None
        if deterministic:
            text = render_deterministic(manifest)
            report = fidelity.verify_render(manifest, text)
            attempts, model, provider = 1, None, None
            history: list[dict] = []
        else:
            outcome = render_llm(manifest, client, config)
            text = outcome["text"]
            report = outcome["report"]
            attempts = outcome["attempts"]
            model = outcome["model"]
            provider = outcome["provider"]
            history = outcome["history"]
            empty_note = outcome.get("empty_note")

        meta = {
            "event_id": manifest["event_id"],
            "seed": seed,
            "timestamp": manifest["timestamp"],
            "scenario_kind": manifest["scenario_kind"],
            "channel": manifest["channel"],
            "author": manifest["author"],
            "render_mode": results["render_mode"],
            "provider": provider,
            "model": model,
            "template_version": TEMPLATE_VERSION if deterministic else None,
            "attempts": attempts,
            "attempt_history": history,
            "fidelity": report.status,
            "fidelity_missing_facts": report.missing_facts,
            "fidelity_unsupported": report.unsupported,
        }
        if empty_note:
            meta["empty_completion_diagnostic"] = empty_note
        write_scenario(out_dir, manifest, text, meta)

        results["scenarios"] += 1
        key = str(attempts)
        results["attempt_histogram"][key] = results["attempt_histogram"].get(key, 0) + 1
        if report.ok:
            results["fidelity_pass"] += 1
        else:
            results["fidelity_failed"] += 1
            if empty_note and not text.strip():
                results["empty"] += 1

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
    parser.add_argument("--force", action="store_true",
                        help="re-render even when a passing meta.json already exists")
    parser.add_argument("--provider", type=str, default=None,
                        help="override renderer.provider from config.yaml for this run")
    parser.add_argument("--model", type=str, default=None,
                        help="override renderer.model from config.yaml for this run")
    args = parser.parse_args(argv)

    results = render_seed(args.seed, deterministic=args.deterministic,
                          out_root=args.out_root, universe_root=args.universe_root,
                          limit=args.limit, only=args.only, force=args.force,
                          provider=args.provider, model=args.model)
    print(f"seed {results['seed']} [{results['render_mode']}]: "
          f"{results['scenarios']} scenarios, "
          f"{results['fidelity_pass']} fidelity pass, "
          f"{results['fidelity_failed']} failed, "
          f"{results['skipped']} skipped, "
          f"{results['empty']} empty")
    print(f"  attempts: {json.dumps(results['attempt_histogram'], sort_keys=True)}")
    return 0 if results["fidelity_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
