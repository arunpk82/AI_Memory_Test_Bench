"""The documentation is checked against the code.

The README carries a per-seed count table because it is genuinely useful to a
reader. A hardcoded table is also exactly the "stale narrated number" defect the
completion report exists to prevent, so it is verified here against a fresh
generation. If the generator changes, this test fails and the table gets updated
rather than quietly becoming fiction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from questions.instantiate import build_questions
from universe.generator import generate
from universe.schema import QUERY_KIND_BY_TYPE, QUESTION_TYPES, SENTINELS

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)


@pytest.fixture(scope="module")
def readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _readme_row(readme: str, label: str) -> list[str]:
    match = re.search(rf"^\|\s*{re.escape(label)}\s*\|(.+?)\|\s*$", readme,
                      re.MULTILINE)
    assert match, f"README has no '{label}' row in the counts table"
    return [cell.strip() for cell in match.group(1).split("|")]


@pytest.mark.parametrize("label,getter", [
    ("facts", lambda u: len(u.facts)),
    ("scenarios (events)", lambda u: len(u.events)),
    ("advertisers", lambda u: u.quotas["advertisers"]),
    ("deals", lambda u: u.quotas["deals"]),
    ("order lines", lambda u: u.quotas["order_lines"]),
])
def test_readme_counts_match_a_fresh_generation(readme, label, getter,
                                                all_universes):
    stated = _readme_row(readme, label)
    actual = [str(getter(all_universes[seed])) for seed in SEEDS]
    assert stated == actual, (
        f"README '{label}' row says {stated} but a fresh generation gives {actual}")


def test_readme_question_counts_match(readme, all_universes):
    stated = _readme_row(readme, "questions")
    actual = [str(len(build_questions(all_universes[seed].facts,
                                      all_universes[seed].events, seed)))
              for seed in SEEDS]
    assert stated == actual


def test_readme_timeline_spans_match(readme, all_universes):
    stated = _readme_row(readme, "timeline span")
    actual = [f"{all_universes[seed].quotas['timeline_span_days']} days"
              for seed in SEEDS]
    assert stated == actual


def test_running_interpreter_meets_the_declared_floor():
    import sys

    import settings
    assert sys.version_info[:2] >= settings.MINIMUM_PYTHON, (
        f"this interpreter is {sys.version_info.major}.{sys.version_info.minor}, "
        f"below the declared floor of "
        f"{settings.MINIMUM_PYTHON[0]}.{settings.MINIMUM_PYTHON[1]}")


def test_readme_states_the_declared_python_floor(readme):
    """The floor lives in one place and the README has to agree with it."""
    import settings
    major, minor = settings.MINIMUM_PYTHON
    assert f"Python {major}.{minor} or newer" in readme, (
        f"README does not state the declared floor of Python {major}.{minor}")


def test_readme_documents_the_current_as_of_rule(readme):
    """The spec requires the "current" as-of definition to be in the README."""
    assert "max(fact.validity_interval.start) + 365 days" in readme
    assert "current_offset_days" in readme


def test_readme_lists_every_question_type_and_query_kind(readme):
    for qtype in QUESTION_TYPES:
        assert qtype in readme, f"README does not mention question type {qtype}"
    for kind in set(QUERY_KIND_BY_TYPE.values()):
        assert kind in readme


def test_readme_lists_every_sentinel(readme):
    for sentinel in SENTINELS:
        assert sentinel in readme


def test_readme_lists_the_configured_models(readme):
    import settings
    config = settings.load_config()
    assert config["renderer"]["model"] in readme
    assert config["answerability"]["oracle_model"] in readme


def test_deviations_entries_are_numbered_consecutively():
    text = (REPO_ROOT / "DEVIATIONS.md").read_text(encoding="utf-8")
    ids = re.findall(r"^##\s+(D-\d+)", text, re.MULTILINE)
    assert ids, "DEVIATIONS.md has no entries"
    assert ids == [f"D-{index:03d}" for index in range(1, len(ids) + 1)], ids


def test_every_deviation_states_decision_alternative_and_why():
    text = (REPO_ROOT / "DEVIATIONS.md").read_text(encoding="utf-8")
    sections = re.split(r"^##\s+D-\d+", text, flags=re.MULTILINE)[1:]
    assert sections
    for index, section in enumerate(sections, start=1):
        for required in ("**Decision.**", "**Why.**"):
            assert required in section, f"D-{index:03d} is missing {required}"
        assert "**Alternative.**" in section or "Alternative" in section, index


def test_required_documents_exist():
    for name in ("README.md", "LICENSE", "DEVIATIONS.md", "PRINCIPLES.md",
                 "config.yaml", "requirements.txt", "requirements-bedrock.txt",
                 "decisions/ADR-000-rebuild.md"):
        assert (REPO_ROOT / name).exists(), name


def test_base_requirements_do_not_pull_boto3():
    """The default install must stay small enough not to time out.

    boto3 is only needed for the two Bedrock paths and drags in a 15 MB
    botocore wheel. Putting it in the base requirements made a first-time setup
    fail on a slow link, for a dependency the failing commands never import.
    """
    base = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirements = [line.split("#")[0].strip() for line in base.splitlines()]
    assert not any(line.startswith("boto3") for line in requirements if line)

    optional = (REPO_ROOT / "requirements-bedrock.txt").read_text(encoding="utf-8")
    assert any(line.split("#")[0].strip().startswith("boto3")
               for line in optional.splitlines())


def test_bedrock_imports_are_lazy():
    """Nothing may import boto3 at module scope, or the skip would not work."""
    import ast
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(name.split(".")[0] in ("boto3", "botocore")
                           for name in names), (
                f"{path.relative_to(REPO_ROOT)} imports boto3 at module scope; it "
                f"must be imported inside the function that needs it")


def test_license_is_apache_2():
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text


def test_gitignore_covers_the_required_entries():
    entries = set((REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split())
    assert {".venv/", "__pycache__/", ".pytest_cache/", ".DS_Store", "~$*",
            "*.db"} <= entries


def test_every_package_has_an_init():
    for package in ("universe", "scenarios", "scenarios/templates", "questions",
                    "verify", "report", "tests"):
        assert (REPO_ROOT / package / "__init__.py").exists(), package
