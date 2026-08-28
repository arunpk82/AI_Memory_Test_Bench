"""The completion report.

Two things about this module are deliberate.

First, **it computes the hashes itself, in this run, twice per seed.** A report
that narrates a hash somebody typed in earlier has shipped once; the number
looked authoritative and was stale. Here each seed is generated fresh twice
in-process, both payloads are hashed, and the result is cross-checked against the
bytes actually on disk. A mismatch is reported as a failure of the artifacts, not
smoothed over.

Second, **the dormant-paths section is curated, not just scanned.** A scanner
finds ``NotImplementedError`` and ``TBD``; it cannot find a guarded branch that
no test exercises. An empty dormant-paths section is a claim about the build, so
the claim is written down by hand and kept true.

Usage::

    python3 -m report.completion
    python3 -m report.completion --seeds 42 --skip-tests
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import settings
from questions.instantiate import build_questions, load_questions
from universe.generator import generate
from universe.schema import QUESTION_TYPES

REPO_ROOT = Path(__file__).resolve().parent.parent
ACROSS_SEEDS = REPO_ROOT / "out" / "_across_seeds"

#: Comment markers that mean unfinished work.
COMMENT_MARKERS = ("TODO", "FIXME", "XXX", "HACK")

#: Paths that exist, are reachable, and are not exercised by the test suite.
#: Each entry says what it is and why it is dormant. Keeping this honest is the
#: point: the section is a claim about the build.
KNOWN_DORMANT: tuple[dict[str, str], ...] = (
    {
        "location": "providers.py :: _post_json, against a real endpoint",
        "kind": "live network path",
        "detail": "The Groq and Gemini transports are exercised against a fake "
                  "urlopen that pins the URL, headers, body shape, retry "
                  "behaviour and every error branch. No socket has been opened "
                  "to either service from this repository.",
    },
    {
        "location": "providers.py :: _bedrock_client, live session",
        "kind": "live network path",
        "detail": "Credential resolution and client construction are driven with "
                  "a fake boto3 session. No live Bedrock call is made in CI or in "
                  "this run.",
    },
    {
        "location": "providers.py :: complete, unsupported-provider fallback",
        "kind": "defensive, unreachable",
        "detail": "The final else in the dispatch cannot be reached because "
                  "ModelSpec rejects an unknown provider at construction. It is "
                  "kept so that adding a provider to PROVIDERS without adding a "
                  "transport fails loudly instead of silently.",
    },
    {
        "location": "scenarios/renderer.py :: render_seed(deterministic=False), "
                    "against live Bedrock",
        "kind": "live network path",
        "detail": "The LLM render of a whole seed is human-triggered for cost "
                  "control, so no seed in this run was rendered by a model. The "
                  "orchestration is not dormant: a test drives render_seed in llm "
                  "mode with only the converse call stubbed and asserts every "
                  "artifact, and separate tests cover the fidelity retry loop and "
                  "the exhausted-retry outcome. What has never executed here is "
                  "the HTTP call.",
    },
    {
        "location": "verify/answerability.py :: run_audit, against a live model",
        "kind": "live network path",
        "detail": "The oracle audit is human-triggered. ask_oracle is tested "
                  "against a capturing client; run_audit's verdict logic is tested "
                  "through classify(), which the calibration suite drives directly.",
    },
    {
        "location": "providers.py :: BEDROCK_PROFILE_PREFIXES, eu. and apac.",
        "kind": "unreachable with the shipped config",
        "detail": "Only the us. inference profile is used in practice because "
                  "that is what config.yaml sets; the eu. and apac. prefixes are "
                  "accepted but never taken.",
    },
    {
        "location": "universe/generator.py :: supersession_chains, cycle guard",
        "kind": "defensive, provably unreachable",
        "detail": "A supersession graph can only become cyclic via a fact with "
                  "two successors, and build_successor_map rejects that first. "
                  "The double-supersession guard is tested; this one cannot be "
                  "reached through any public shape and is kept as a backstop.",
    },
    {
        "location": "universe/generator.py :: bump_cpm, ladder-exhausted raise",
        "kind": "guard never fired",
        "detail": "The rate ladder has twelve rungs and no chain uses more than "
                  "three, so the exhaustion raise never fires on seeds 42, 43 or "
                  "44.",
    },
    {
        "location": "universe/generator.py :: _distinct_names, "
                    "wordlist-exhausted raise",
        "kind": "guard never fired",
        "detail": "44 advertiser roots against at most 12 advertisers, and 18 "
                  "agency roots against at most 7 agencies. The raise protects a "
                  "future increase in entity counts.",
    },
    {
        "location": "universe/generator.py :: current_value / "
                    "UniverseIndex.active_at, multi-active raise",
        "kind": "guard never fired",
        "detail": "Both raise if an (entity, attribute) pair has two active facts. "
                  "The duplicate_active_facts quota now makes that impossible at "
                  "generation time, so the runtime raise is redundant protection. "
                  "It fired once during development, which is why the quota "
                  "exists.",
    },
    {
        "location": "universe/schema.py :: TRUST_TABLE, five of nine rows",
        "kind": "table entry with no generated fact",
        "detail": "The generator produces facts on (order_system, system), "
                  "(email_received, counterparty), (email_sent, user) and "
                  "(call_note, user). The other five rows -- (order_system, user), "
                  "(email_sent, system), (email_received, system), "
                  "(email_received, user) and (call_note, counterparty) -- are "
                  "derived correctly and tested directly, but no fact in any of "
                  "the three universes carries them.",
    },
    {
        "location": "questions/instantiate.py :: UniverseIndex.label, "
                    "agency entity branch",
        "kind": "branch with no data",
        "detail": "The schema allows an 'agency' entity but the generator records "
                  "agency relationships as advertiser attributes "
                  "(agency_of_record), so no fact has entity == 'agency' and the "
                  "label branch is never taken.",
    },
    {
        "location": "questions/instantiate.py :: "
                    "build_abstention_false_premise, fake-id collision guard",
        "kind": "guard never fired",
        "detail": "Fabricated order-line ids use the 9xxx block and real ones "
                  "never reach it, so the collision check never skips.",
    },
    {
        "location": "questions/instantiate.py :: allocate, starvation break",
        "kind": "branch never taken",
        "detail": "The 'no progress' break covers a universe with fewer "
                  "candidates than the target total. All three seeds have "
                  "surplus on every horizon, so allocation always reaches the "
                  "240-question mix (140 current / 33 early / 33 mid / 34 late) "
                  "and the break is not used.",
    },
    {
        "location": "verify/matching.py :: _is_abbreviation_boundary, "
                    "single-initial branch",
        "kind": "branch with no data",
        "detail": "Sentence splitting declines to split after a single capital "
                  "letter (an initial such as 'J. Whitfield'). The generator "
                  "never produces initials, so only the abbreviation-word half of "
                  "the guard runs against the corpus. Both halves are covered by "
                  "the matching self-tests.",
    },
    {
        "location": "providers.py :: _bedrock_client, empty-region guard",
        "kind": "defensive, unreachable",
        "detail": "It raises if the resolved region is empty, but "
                  "settings.aws_region raises a ConfigError naming aws.region "
                  "before it can return an empty value. The ConfigError path is "
                  "tested; this guard cannot be reached.",
    },
    {
        "location": "verify/fidelity.py :: main, failure detail printing",
        "kind": "branch reached only on failure",
        "detail": "The per-scenario failure printout is exercised by the CLI "
                  "failure test but never by a clean run, because the "
                  "deterministic corpus has no fidelity failures.",
    },
)


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------

def payload(records: list[dict]) -> str:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False) + "\n" for record in records)


def sha256_of(records: list[dict]) -> str:
    return hashlib.sha256(payload(records).encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_section(seed: int) -> dict:
    """Counts and hashes for one seed, all computed in this run."""
    first = generate(seed)
    second = generate(seed)
    first_sha, second_sha = sha256_of(first.facts), sha256_of(second.facts)

    on_disk = REPO_ROOT / "universe" / "out" / str(seed) / "facts.jsonl"
    disk_sha = sha256_of_file(on_disk)

    questions = build_questions(first.facts, first.events, seed)
    try:
        shipped_questions = load_questions(seed)
    except FileNotFoundError:
        shipped_questions = None

    per_type = {qtype: 0 for qtype in QUESTION_TYPES}
    for question in questions:
        per_type[question["type"]] += 1

    horizons: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for question in questions:
        horizons[question["as_of_horizon"]] = horizons.get(question["as_of_horizon"], 0) + 1
        kinds[question["query_kind"]] = kinds.get(question["query_kind"], 0) + 1

    render_summary_path = (REPO_ROOT / "scenarios" / "out" / str(seed)
                           / "render_summary.json")
    fidelity_path = REPO_ROOT / "scenarios" / "out" / str(seed) / "fidelity_report.json"
    render_summary = (json.loads(render_summary_path.read_text(encoding="utf-8"))
                      if render_summary_path.exists() else None)
    fidelity_report = (json.loads(fidelity_path.read_text(encoding="utf-8"))
                       if fidelity_path.exists() else None)
    dryrun_path = REPO_ROOT / "verify" / "out" / str(seed) / "dryrun_report.json"
    dryrun = (json.loads(dryrun_path.read_text(encoding="utf-8"))
              if dryrun_path.exists() else None)

    return {
        "seed": seed,
        "facts": len(first.facts),
        "events": len(first.events),
        "scenarios": first.quotas["scenarios"],
        "advertisers": first.quotas["advertisers"],
        "deals": first.quotas["deals"],
        "order_lines": first.quotas["order_lines"],
        "timeline": {
            "start": first.quotas["timeline_start"],
            "end": first.quotas["timeline_end"],
            "span_days": first.quotas["timeline_span_days"],
            "current_as_of": first.quotas["current_as_of"],
        },
        "quotas": first.quotas,
        "questions_total": len(questions),
        "questions_by_type": per_type,
        "questions_by_query_kind": dict(sorted(kinds.items())),
        "questions_by_as_of_horizon": dict(sorted(horizons.items())),
        "shipped_questions_total": len(shipped_questions) if shipped_questions else None,
        "facts_sha256": {
            "fresh_generation_1": first_sha,
            "fresh_generation_2": second_sha,
            "identical": first_sha == second_sha,
            "on_disk_facts_jsonl": disk_sha,
            "matches_on_disk": disk_sha == first_sha,
        },
        "events_sha256": sha256_of(first.events),
        "questions_sha256": sha256_of(questions),
        "render": render_summary,
        "fidelity": fidelity_report and {
            "scenarios": fidelity_report["scenarios"],
            "passed": fidelity_report["passed"],
            "failed": fidelity_report["failed"],
            "missing_fact_total": fidelity_report["missing_fact_total"],
            "unsupported_total": fidelity_report["unsupported_total"],
        },
        "dry_run": dryrun and {
            "questions": dryrun["questions"],
            "corpus_scenarios": dryrun["corpus_scenarios"],
            "clean": dryrun["clean"],
            "findings_by_check": dryrun["findings_by_check"],
        },
    }


def scan_dormant_markers() -> list[dict]:
    """Scan the source for unfinished work, via the AST and comment tokens.

    Line-level grepping cannot be used here. This module necessarily contains
    the literal strings it searches for -- in its own docstring, in the marker
    list, and in the prose it prints -- so a substring scan reports the scanner
    as unfinished work and makes the section's own closing sentence false.
    Parsing means string literals cannot trigger a finding at all.
    """
    found: list[dict] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if ".venv" in path.parts or path.name.startswith("."):
            continue
        relative = path.relative_to(REPO_ROOT)
        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            found.append({"location": f"{relative}:{exc.lineno}",
                          "kind": "unparseable source",
                          "detail": str(exc)})
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                name = _raised_name(node)
                if name == "NotImplementedError":
                    found.append({
                        "location": f"{relative}:{node.lineno}",
                        "kind": "NotImplementedError",
                        "detail": "raises NotImplementedError",
                    })
            # Functions only. A class whose body is just a docstring is a
            # complete definition -- every exception type in this codebase looks
            # like that -- whereas a function whose body is just a docstring is
            # unimplemented.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and _is_stub_body(node):
                found.append({
                    "location": f"{relative}:{node.lineno}",
                    "kind": "stub body",
                    "detail": f"{node.name} has no implementation",
                })

        with path.open("rb") as handle:
            try:
                tokens = list(tokenize.tokenize(handle.readline))
            except (tokenize.TokenError, IndentationError):
                tokens = []
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            for marker in COMMENT_MARKERS:
                if re.search(rf"\b{marker}\b", token.string):
                    found.append({"location": f"{relative}:{token.start[0]}",
                                  "kind": "source marker",
                                  "detail": f"{marker} comment: "
                                            f"{token.string.strip()}"})
    return found


def _raised_name(node: ast.Raise) -> str | None:
    exc = node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


def _is_stub_body(node) -> bool:
    body = [statement for statement in node.body
            if not (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str))]
    return not body or all(isinstance(statement, ast.Pass) for statement in body)


def scan_unset_config() -> list[dict]:
    """Any config key still set to a placeholder."""
    config = settings.load_config()
    found: list[dict] = []

    def walk(node, prefix: str) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                walk(node[key], f"{prefix}.{key}" if prefix else key)
        elif isinstance(node, str) and node.strip() in settings.UNSET_MARKERS:
            found.append({"location": f"config.yaml :: {prefix}",
                          "kind": "unset config key",
                          "detail": f"value is the placeholder {node!r}"})
        elif node is None:
            found.append({"location": f"config.yaml :: {prefix}",
                          "kind": "unset config key",
                          "detail": "value is null"})

    walk(config, "")
    return found


def deviation_count() -> dict:
    path = REPO_ROOT / "DEVIATIONS.md"
    if not path.exists():
        return {"file": "DEVIATIONS.md", "exists": False, "entries": 0}
    text = path.read_text(encoding="utf-8")
    entries = re.findall(r"^##\s+D-\d+", text, re.MULTILINE)
    return {"file": "DEVIATIONS.md", "exists": True, "entries": len(entries),
            "ids": [entry.split()[-1] for entry in entries]}


def run_pytest() -> dict:
    """Run the suite and capture its full output, with the command echoed."""
    command = [sys.executable, "-m", "pytest", "-q", "tests/"]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    echoed = " ".join(command) + "\n" + completed.stdout
    return {"exit_code": completed.returncode,
            "stdout": echoed,
            "stderr": completed.stderr}


def cross_seed_table(sections: list[dict]) -> dict:
    """Comparison of the three universes, for out/_across_seeds/."""
    universes = {section["seed"]: generate(section["seed"]) for section in sections}
    names = {seed: sorted({advertiser.name for advertiser in universe.advertisers})
             for seed, universe in universes.items()}
    overlaps = {}
    seeds = sorted(universes)
    for left_index, left in enumerate(seeds):
        for right in seeds[left_index + 1:]:
            overlaps[f"{left}x{right}"] = sorted(set(names[left]) & set(names[right]))
    return {
        "seeds": seeds,
        "per_seed": {
            section["seed"]: {
                "facts": section["facts"],
                "events": section["events"],
                "scenarios": section["scenarios"],
                "advertisers": section["advertisers"],
                "deals": section["deals"],
                "order_lines": section["order_lines"],
                "questions": section["questions_total"],
                "timeline_span_days": section["timeline"]["span_days"],
                "current_as_of": section["timeline"]["current_as_of"],
                "facts_sha256": section["facts_sha256"]["fresh_generation_1"],
            }
            for section in sections
        },
        "advertiser_names": names,
        "advertiser_name_overlap": overlaps,
        "deal_count_vectors": {
            seed: [len([deal for deal in universe.deals
                        if deal.advertiser is advertiser])
                   for advertiser in universe.advertisers]
            for seed, universe in universes.items()
        },
        "totals": {
            "facts": sum(section["facts"] for section in sections),
            "events": sum(section["events"] for section in sections),
            "scenarios": sum(section["scenarios"] for section in sections),
            "questions": sum(section["questions_total"] for section in sections),
        },
    }


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    lines: list[str] = ["# mem-testbed v2 — completion report", ""]
    lines.append(f"Generated by `python3 -m report.completion` in this run. Every "
                 f"hash below was computed by this step, not transcribed.")
    lines.append("")

    # 1. Counts
    lines += ["## 1. Counts", ""]
    header = ("| seed | facts | events | scenarios | advertisers | deals | "
              "order lines | questions | timeline span |")
    lines += [header, "|---|---|---|---|---|---|---|---|---|"]
    for section in report["seeds"]:
        lines.append(
            f"| {section['seed']} | {section['facts']} | {section['events']} | "
            f"{section['scenarios']} | {section['advertisers']} | "
            f"{section['deals']} | {section['order_lines']} | "
            f"{section['questions_total']} | "
            f"{section['timeline']['span_days']}d |")
    totals = report["across_seeds"]["totals"]
    lines.append(f"| **total** | **{totals['facts']}** | **{totals['events']}** | "
                 f"**{totals['scenarios']}** | | | | **{totals['questions']}** | |")
    lines.append("")

    lines += ["### Questions per type, per seed", ""]
    seeds = [section["seed"] for section in report["seeds"]]
    lines.append("| type | query_kind | " + " | ".join(str(seed) for seed in seeds)
                 + " | total |")
    lines.append("|---|---|" + "---|" * (len(seeds) + 1))
    from universe.schema import QUERY_KIND_BY_TYPE
    for qtype in QUESTION_TYPES:
        counts = [section["questions_by_type"][qtype] for section in report["seeds"]]
        lines.append(f"| {qtype} | {QUERY_KIND_BY_TYPE[qtype]} | "
                     + " | ".join(str(count) for count in counts)
                     + f" | {sum(counts)} |")
    lines.append("")

    lines += ["### As-of horizon distribution", ""]
    lines.append("| seed | " + " | ".join(("current", "early", "mid", "late")) + " |")
    lines.append("|---|---|---|---|---|")
    for section in report["seeds"]:
        horizons = section["questions_by_as_of_horizon"]
        lines.append(f"| {section['seed']} | "
                     + " | ".join(str(horizons.get(name, 0))
                                  for name in ("current", "early", "mid", "late"))
                     + " |")
    lines.append("")

    # 2. Hashes
    lines += ["## 2. facts.jsonl sha256, computed in this run", "",
              "Each seed was generated fresh twice in-process. Both payloads were "
              "hashed and compared to the bytes on disk.", "",
              "| seed | fresh generation 1 | fresh generation 2 | identical | "
              "matches on-disk facts.jsonl |",
              "|---|---|---|---|---|"]
    for section in report["seeds"]:
        hashes = section["facts_sha256"]
        lines.append(
            f"| {section['seed']} | `{hashes['fresh_generation_1']}` | "
            f"`{hashes['fresh_generation_2']}` | "
            f"{'yes' if hashes['identical'] else 'NO'} | "
            f"{'yes' if hashes['matches_on_disk'] else 'NO'} |")
    lines.append("")

    # 3. Fidelity
    lines += ["## 3. Fidelity (deterministic render)", "",
              "The LLM render is human-triggered after review, so no LLM fidelity "
              "numbers are reported here.", "",
              "| seed | scenarios | pass | fail | missing facts | unsupported "
              "details | attempt distribution |",
              "|---|---|---|---|---|---|---|"]
    for section in report["seeds"]:
        fidelity = section["fidelity"]
        render = section["render"]
        if not fidelity:
            lines.append(f"| {section['seed']} | not rendered | | | | | |")
            continue
        attempts = (json.dumps(render["attempt_histogram"], sort_keys=True)
                    if render else "n/a")
        lines.append(
            f"| {section['seed']} | {fidelity['scenarios']} | {fidelity['passed']} | "
            f"{fidelity['failed']} | {fidelity['missing_fact_total']} | "
            f"{fidelity['unsupported_total']} | `{attempts}` |")
    lines.append("")

    lines += ["### Dry-run answerability audit (network-free)", "",
              "| seed | questions | corpus scenarios | clean | findings |",
              "|---|---|---|---|---|"]
    for section in report["seeds"]:
        dry = section["dry_run"]
        if not dry:
            lines.append(f"| {section['seed']} | not audited | | | |")
            continue
        lines.append(f"| {section['seed']} | {dry['questions']} | "
                     f"{dry['corpus_scenarios']} | "
                     f"{'yes' if dry['clean'] else 'NO'} | "
                     f"`{json.dumps(dry['findings_by_check'], sort_keys=True)}` |")
    lines.append("")

    # 4. pytest
    lines += ["## 4. Full pytest output", ""]
    tests = report["pytest"]
    if tests is None:
        lines += ["Skipped with `--skip-tests`.", ""]
    else:
        lines += [f"Exit code: `{tests['exit_code']}`", "", "```"]
        lines.append(tests["stdout"].rstrip())
        if tests["stderr"].strip():
            lines += ["", "--- stderr ---", tests["stderr"].rstrip()]
        lines += ["```", ""]

    # 5. Dormant paths
    dormant = report["dormant_paths"]
    lines += ["## 5. Dormant paths", "",
              f"{len(dormant)} entries. An empty section here would be a claim, so "
              f"the list is curated by hand as well as scanned. Every entry is a "
              f"path that exists in the shipped code and is not exercised by the "
              f"test suite.", ""]
    for entry in dormant:
        lines.append(f"- **{entry['location']}** — _{entry['kind']}_. "
                     f"{entry['detail']}")
    lines.append("")
    curated_kinds = {entry["location"] for entry in KNOWN_DORMANT}
    scanned = [entry for entry in dormant if entry["location"] not in curated_kinds]
    if scanned:
        lines.append(f"Of those, {len(scanned)} came from the automated scan; the "
                     f"other {len(dormant) - len(scanned)} are curated.")
    else:
        lines.append(f"All {len(dormant)} entries are curated. The automated scan "
                     f"found nothing: the source contains no unimplemented raises, "
                     f"no stub bodies, no unfinished-work comments, and no unset or "
                     f"placeholder configuration keys. The scan reads the syntax "
                     f"tree and the comment tokens rather than grepping lines, so "
                     f"a marker word inside a string literal cannot produce a false "
                     f"clean or a false finding.")
    lines.append("")

    # 6. Deviations
    deviations = report["deviations"]
    lines += ["## 6. Deviations", "",
              f"`DEVIATIONS.md` contains **{deviations['entries']}** entries: "
              + ", ".join(deviations.get("ids", [])) + ".", ""]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_report(seeds: list[int], *, skip_tests: bool) -> dict:
    sections = [seed_section(seed) for seed in seeds]
    dormant = list(KNOWN_DORMANT) + scan_dormant_markers() + scan_unset_config()
    return {
        "seeds": sections,
        "across_seeds": cross_seed_table(sections),
        "pytest": None if skip_tests else run_pytest(),
        "dormant_paths": dormant,
        "deviations": deviation_count(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the completion report.")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    report = build_report(seeds, skip_tests=args.skip_tests)

    out_root = Path(args.out_root) if args.out_root else ACROSS_SEEDS
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "completion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "comparison.json").write_text(
        json.dumps(report["across_seeds"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    markdown = render_markdown(report)
    (out_root / "completion_report.md").write_text(markdown, encoding="utf-8")

    print(markdown)

    failures = [section["seed"] for section in report["seeds"]
                if not section["facts_sha256"]["identical"]
                or not section["facts_sha256"]["matches_on_disk"]]
    if failures:
        print(f"HASH MISMATCH for seeds {failures}: the artifacts on disk do not "
              f"match a fresh generation. Regenerate before reporting.",
              file=sys.stderr)
        return 1
    if report["pytest"] and report["pytest"]["exit_code"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
