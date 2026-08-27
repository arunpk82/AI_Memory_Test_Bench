"""The completion report is shipped code, so it is tested like shipped code.

The dormant-paths scan is the load-bearing part: the report closes that section
with a claim that the automated scan found nothing. If the scan is broken, the
claim is worthless and reads as if it had been verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report import completion

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- the scan ---

def _scan(tmp_path: Path, source: str, monkeypatch) -> list[dict]:
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(completion, "REPO_ROOT", tmp_path)
    return completion.scan_dormant_markers()


def test_scan_finds_a_not_implemented_raise(tmp_path, monkeypatch):
    findings = _scan(tmp_path, "def f():\n    raise NotImplementedError\n",
                     monkeypatch)
    assert [finding["kind"] for finding in findings] == ["NotImplementedError"]


def test_scan_finds_a_not_implemented_raise_with_a_message(tmp_path, monkeypatch):
    findings = _scan(tmp_path,
                     'def f():\n    raise NotImplementedError("later")\n',
                     monkeypatch)
    assert [finding["kind"] for finding in findings] == ["NotImplementedError"]


def test_scan_finds_a_todo_comment(tmp_path, monkeypatch):
    findings = _scan(tmp_path, "x = 1  # TODO: finish this\n", monkeypatch)
    assert [finding["kind"] for finding in findings] == ["source marker"]
    assert "TODO" in findings[0]["detail"]


def test_scan_finds_a_docstring_only_function(tmp_path, monkeypatch):
    findings = _scan(tmp_path, 'def f():\n    """Nothing here yet."""\n',
                     monkeypatch)
    assert [finding["kind"] for finding in findings] == ["stub body"]


def test_scan_finds_a_pass_only_function(tmp_path, monkeypatch):
    findings = _scan(tmp_path, "def f():\n    pass\n", monkeypatch)
    assert [finding["kind"] for finding in findings] == ["stub body"]


def test_scan_ignores_marker_words_inside_string_literals(tmp_path, monkeypatch):
    """The reason the scan parses instead of grepping."""
    source = (
        'MARKERS = ("TODO", "FIXME", "XXX")\n'
        'MESSAGE = "raise NotImplementedError is what a stub looks like"\n'
        'def described():\n'
        '    """Mentions TODO and FIXME and NotImplementedError in prose."""\n'
        '    return MARKERS, MESSAGE\n'
    )
    assert _scan(tmp_path, source, monkeypatch) == []


def test_scan_ignores_a_docstring_only_exception_class(tmp_path, monkeypatch):
    source = 'class Boom(ValueError):\n    """A real, complete definition."""\n'
    assert _scan(tmp_path, source, monkeypatch) == []


def test_scan_reports_unparseable_source(tmp_path, monkeypatch):
    findings = _scan(tmp_path, "def broken(:\n", monkeypatch)
    assert findings[0]["kind"] == "unparseable source"


def test_the_shipped_source_has_no_scan_findings():
    assert completion.scan_dormant_markers() == []


def test_the_shipped_config_has_no_unset_keys():
    assert completion.scan_unset_config() == []


# ------------------------------------------------------------ curated list ---

def test_every_curated_dormant_entry_points_at_a_real_file():
    for entry in completion.KNOWN_DORMANT:
        relative = entry["location"].split(" ::")[0].split(":")[0].strip()
        assert (REPO_ROOT / relative).exists(), entry["location"]
        assert entry["kind"]
        assert len(entry["detail"]) > 40, entry["location"]


def test_curated_dormant_entries_are_unique():
    locations = [entry["location"] for entry in completion.KNOWN_DORMANT]
    assert len(locations) == len(set(locations))


# ------------------------------------------------------------- assembly ---

def test_deviation_count_matches_the_file():
    counted = completion.deviation_count()
    assert counted["exists"]
    assert counted["entries"] == len(counted["ids"])
    assert counted["entries"] >= 1
    assert counted["ids"][0] == "D-001"


def test_hashes_are_recomputed_per_call():
    """Two calls must agree, and must agree with the bytes on disk."""
    section = completion.seed_section(42)
    hashes = section["facts_sha256"]
    assert hashes["identical"]
    assert hashes["matches_on_disk"], (
        "the committed universe/out/42/facts.jsonl does not match a fresh "
        "generation; regenerate before reporting")


def test_report_renders_markdown_without_running_the_suite(tmp_path):
    report = completion.build_report([42], skip_tests=True)
    assert report["pytest"] is None
    markdown = completion.render_markdown(report)
    for heading in ("## 1. Counts", "## 2. facts.jsonl sha256",
                    "## 3. Fidelity", "## 4. Full pytest output",
                    "## 5. Dormant paths", "## 6. Deviations"):
        assert heading in markdown
    assert "Skipped with `--skip-tests`." in markdown
    # The report must be JSON-serialisable; it is written to disk as JSON.
    json.dumps(report)


def test_report_cli_writes_both_artifacts(tmp_path):
    assert completion.main(["--seeds", "42", "--skip-tests",
                            "--out-root", str(tmp_path)]) == 0
    assert (tmp_path / "completion_report.md").exists()
    assert (tmp_path / "completion_report.json").exists()
    assert (tmp_path / "comparison.json").exists()
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["seeds"] == [42]


def test_cross_seed_comparison_records_zero_name_overlap():
    sections = [completion.seed_section(seed) for seed in (42, 43, 44)]
    table = completion.cross_seed_table(sections)
    assert table["seeds"] == [42, 43, 44]
    for pair, overlap in table["advertiser_name_overlap"].items():
        assert overlap == [], f"{pair} shares advertiser names: {overlap}"
    vectors = list(table["deal_count_vectors"].values())
    assert len({tuple(vector) for vector in vectors}) == len(vectors)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_shipped_artifacts_are_current(seed):
    """The committed artifacts must match what the generator produces now."""
    section = completion.seed_section(seed)
    assert section["facts_sha256"]["matches_on_disk"]
    assert section["fidelity"] is not None, "seed has no committed fidelity report"
    assert section["fidelity"]["failed"] == 0
    assert section["dry_run"] is not None, "seed has no committed dry-run report"
    assert section["dry_run"]["clean"]
    assert section["shipped_questions_total"] == section["questions_total"]
