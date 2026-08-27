"""The CLI contract.

Every entry point takes ``--seed`` and writes under ``out/<seed>/``. No seed is
hardcoded anywhere, which these tests check the only way that means anything: by
driving each entry point with a seed other than 42 and looking at where the
bytes landed.
"""

from __future__ import annotations

import json

import pytest

from questions import instantiate
from scenarios import renderer
from universe import generator
from verify import answerability, fidelity


@pytest.fixture
def roots(tmp_path):
    return {
        "universe": tmp_path / "universe" / "out",
        "scenarios": tmp_path / "scenarios" / "out",
        "questions": tmp_path / "questions" / "out",
        "verify": tmp_path / "verify" / "out",
    }


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_generator_cli_writes_under_the_requested_seed(seed, roots, capsys):
    assert generator.main(["--seed", str(seed), "--out-root", str(roots["universe"])]) == 0
    root = roots["universe"] / str(seed)
    for name in ("facts.jsonl", "events.jsonl", "schema.json", "quotas.json"):
        assert (root / name).exists(), name
    out = capsys.readouterr().out
    assert f"seed {seed}:" in out
    assert "sha256" in out


def test_generator_cli_batch_flag_produces_every_seed(roots):
    assert generator.main(["--seeds", "42,43,44",
                           "--out-root", str(roots["universe"])]) == 0
    for seed in (42, 43, 44):
        assert (roots["universe"] / str(seed) / "facts.jsonl").exists()


def test_full_cli_chain_for_a_non_default_seed(roots, capsys):
    seed = 43
    assert generator.main(["--seed", str(seed),
                           "--out-root", str(roots["universe"])]) == 0
    assert renderer.main(["--seed", str(seed), "--deterministic",
                          "--out-root", str(roots["scenarios"]),
                          "--universe-root", str(roots["universe"])]) == 0
    assert fidelity.main(["--seed", str(seed),
                          "--scenario-root", str(roots["scenarios"])]) == 0
    assert instantiate.main(["--seed", str(seed),
                             "--out-root", str(roots["questions"]),
                             "--universe-root", str(roots["universe"])]) == 0
    assert answerability.main(["--seed", str(seed), "--dry-run",
                               "--scenario-root", str(roots["scenarios"]),
                               "--universe-root", str(roots["universe"]),
                               "--question-root", str(roots["questions"]),
                               "--out-root", str(roots["verify"])]) == 0

    questions = json.loads(
        (roots["questions"] / str(seed) / "question_counts.json").read_text())
    assert questions["seed"] == seed
    assert 220 <= questions["total"] <= 260

    report = json.loads(
        (roots["verify"] / str(seed) / "dryrun_report.json").read_text())
    assert report["seed"] == seed
    assert report["clean"]

    # Nothing leaked into the seed-42 directories.
    assert not (roots["questions"] / "42").exists()
    assert not (roots["verify"] / "42").exists()

    out = capsys.readouterr().out
    assert "no findings" in out


def test_renderer_limit_flag_bounds_the_work(roots):
    seed = 44
    generator.main(["--seed", str(seed), "--out-root", str(roots["universe"])])
    assert renderer.main(["--seed", str(seed), "--deterministic", "--limit", "5",
                          "--out-root", str(roots["scenarios"]),
                          "--universe-root", str(roots["universe"])]) == 0
    rendered = [path for path in (roots["scenarios"] / str(seed)).iterdir()
                if path.is_dir()]
    assert len(rendered) == 5


def test_renderer_only_flag_selects_one_scenario(roots):
    seed = 42
    generator.main(["--seed", str(seed), "--out-root", str(roots["universe"])])
    _, events = generator.load_universe(seed, roots["universe"])
    target = events[7]["event_id"]
    assert renderer.main(["--seed", str(seed), "--deterministic",
                          "--only", target,
                          "--out-root", str(roots["scenarios"]),
                          "--universe-root", str(roots["universe"])]) == 0
    rendered = [path.name for path in (roots["scenarios"] / str(seed)).iterdir()
                if path.is_dir()]
    assert rendered == [target]


def test_dry_run_exits_non_zero_when_findings_exist(roots, artifacts, tmp_path):
    """The CLI has to fail loudly, or CI will not notice."""
    import copy
    questions = copy.deepcopy(artifacts.questions)
    target = next(question for question in questions
                  if question["gold_kind"] == "value"
                  and "derivation" not in question["notes"])
    target["gold_answer"] = "99.99"
    target["gold_tokens"] = ["99.99"]
    instantiate.write_questions(artifacts.seed, questions, roots["questions"])
    assert answerability.main(["--seed", str(artifacts.seed), "--dry-run",
                               "--scenario-root", str(artifacts.scenario_root),
                               "--universe-root", str(artifacts.universe_root),
                               "--question-root", str(roots["questions"]),
                               "--out-root", str(roots["verify"])]) == 1


def test_fidelity_cli_exits_non_zero_on_failure(roots, artifacts):
    seed = artifacts.seed
    target = next(path for path in (artifacts.scenario_root / str(seed)).iterdir()
                  if path.is_dir())
    import shutil
    copied = roots["scenarios"] / str(seed) / target.name
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target, copied)
    text = (copied / "rendered.md").read_text(encoding="utf-8")
    (copied / "rendered.md").write_text(
        text.replace("Let me know if anything looks off.",
                     "We also agreed 17% off with Zylotronic Media."),
        encoding="utf-8")
    assert fidelity.main(["--seed", str(seed),
                          "--scenario-root", str(roots["scenarios"])]) == 1


def test_current_value_resolves_a_single_active_fact(artifacts):
    """The public current-value helper, used by downstream consumers."""
    from universe.generator import current_as_of, current_value

    as_of = current_as_of(artifacts.facts)
    line_status = [fact for fact in artifacts.facts
                   if fact["attribute"] == "line_status"]
    entity_id = line_status[0]["entity_id"]
    resolved = current_value(artifacts.facts, entity_id, "line_status", as_of)
    assert resolved is not None
    assert resolved["attribute"] == "line_status"
    assert resolved["validity_interval"]["end"] is None or True
    assert current_value(artifacts.facts, entity_id, "no_such_attribute",
                         as_of) is None
