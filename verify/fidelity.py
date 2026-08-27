"""Two-sided fidelity verification for rendered scenarios.

Side 1, **planted recall**: every fact in the manifest must be findable in the
rendered text. A renderer that drops the second of two dates has produced a
corpus in which a question about that date is unanswerable, and the question
builder has no way to know.

Side 2, **unsupported precision**: every number, date and name in the rendered
text must be traceable to the manifest. A renderer that invents a discount
percentage or a second agency has produced a corpus where a correct-looking
answer is fabricated, and the judge has no way to know.

Both sides call :mod:`verify.matching`. Neither implements matching itself.

Usage::

    python3 -m verify.fidelity --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import matching

#: Manifest header fields that legitimately appear in the prose. The renderer is
#: instructed to address the counterparty by name and to name the order line, so
#: these are manifest content for the purposes of the precision side. See
#: DEVIATIONS.md.
HEADER_SUPPORT_FIELDS = ("advertiser_name", "agency_name", "contact_name",
                         "deal_name", "deal_id", "io_id", "advertiser_id")

DEFAULT_SCENARIO_ROOT = Path(__file__).resolve().parent.parent / "scenarios" / "out"


@dataclass
class FidelityReport:
    event_id: str
    fact_count: int
    missing_facts: list[dict] = field(default_factory=list)
    unsupported: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_facts and not any(self.unsupported.values())

    @property
    def status(self) -> str:
        return "pass" if self.ok else "failed"

    def as_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status
        return data

    def failure_summary(self) -> str:
        """Human-readable failure list, reused verbatim in the LLM retry prompt."""
        lines: list[str] = []
        if self.missing_facts:
            lines.append("Facts from the manifest that are missing from the draft:")
            for fact in self.missing_facts:
                lines.append(f"  - {fact['attribute']} = {fact['value']}")
        unsupported = {key: values for key, values in self.unsupported.items() if values}
        if unsupported:
            lines.append("Details in the draft that are not in the manifest and "
                         "must be removed:")
            for kind, values in sorted(unsupported.items()):
                for value in values:
                    lines.append(f"  - {kind[:-1]}: {value}")
        return "\n".join(lines)


def support_for_manifest(manifest: dict) -> matching.Support:
    """The set of values the rendered text is allowed to contain."""
    support = matching.Support()
    for fact in manifest["facts"]:
        support.add_value(fact["value"])
    support.add_value(manifest["timestamp"])
    support.add_text(manifest["subject"])
    for field_name in HEADER_SUPPORT_FIELDS:
        value = manifest.get(field_name)
        if value:
            support.add_value(str(value))
    return support


def verify_render(manifest: dict, text: str) -> FidelityReport:
    """Run both sides of fidelity for one rendered scenario."""
    missing = [
        {"fact_id": fact["fact_id"], "attribute": fact["attribute"],
         "value": fact["value"]}
        for fact in manifest["facts"]
        if not matching.value_in_text(fact["value"], text)
    ]
    support = support_for_manifest(manifest)
    unsupported = matching.unsupported_items(text, support)
    return FidelityReport(event_id=manifest["event_id"],
                          fact_count=len(manifest["facts"]),
                          missing_facts=missing,
                          unsupported=unsupported)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def verify_seed(seed: int, scenario_root: Path | None = None) -> dict:
    """Verify every rendered scenario for ``seed`` that exists on disk."""
    root = Path(scenario_root or DEFAULT_SCENARIO_ROOT) / str(seed)
    if not root.exists():
        raise FileNotFoundError(
            f"no rendered scenarios at {root}; run "
            f"`python3 -m scenarios.renderer --seed {seed} --deterministic`")

    reports: list[FidelityReport] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = directory / "manifest.json"
        rendered_path = directory / "rendered.md"
        if not manifest_path.exists() or not rendered_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        text = rendered_path.read_text(encoding="utf-8")
        reports.append(verify_render(manifest, text))

    failures = [report for report in reports if not report.ok]
    summary = {
        "seed": seed,
        "scenarios": len(reports),
        "passed": len(reports) - len(failures),
        "failed": len(failures),
        "missing_fact_total": sum(len(report.missing_facts) for report in reports),
        "unsupported_total": sum(
            sum(len(values) for values in report.unsupported.values())
            for report in reports),
        "reports": [report.as_dict() for report in reports],
    }
    (root / "fidelity_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify rendered scenarios against their manifests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-root", type=Path, default=None)
    parser.add_argument("--show", type=int, default=8,
                        help="how many failing scenarios to print in detail")
    args = parser.parse_args(argv)

    summary = verify_seed(args.seed, args.scenario_root)
    print(f"seed {summary['seed']}: {summary['passed']}/{summary['scenarios']} "
          f"scenarios pass fidelity "
          f"({summary['missing_fact_total']} missing facts, "
          f"{summary['unsupported_total']} unsupported details)")
    shown = 0
    for report in summary["reports"]:
        if report["status"] == "pass" or shown >= args.show:
            continue
        shown += 1
        print(f"\n{report['event_id']}:")
        for fact in report["missing_facts"]:
            print(f"  missing {fact['attribute']} = {fact['value']}")
        for kind, values in sorted(report["unsupported"].items()):
            for value in values:
                print(f"  unsupported {kind[:-1]}: {value}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
