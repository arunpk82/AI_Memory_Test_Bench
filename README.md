# mem-testbed v2

A deterministic testbed for evaluating agent memory on an ad-sales
(Intent-Extractor domain) corpus.

Given a seed, it generates a self-consistent universe of facts and the events
that carry them, renders those events as business correspondence, verifies the
correspondence is faithful to the facts **in both directions**, instantiates
questions mechanically from the facts, and audits whether those questions are
actually answerable from the resulting corpus.

The point of the last step is worth stating plainly: a benchmark that quietly
contains unanswerable questions reports model failures that are really its own
authoring defects. This testbed measures itself before it measures anything else.

Three independent universes are produced, for seeds **42**, **43** and **44**.
The three seeds are the replication unit. Questions inside one universe are
correlated — a single generator quirk hits dozens at once — so a statistical
claim rests on agreement *across* seeds, never on question count within one.

## What it produces

Per seed, in this build:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| facts | 816 | 820 | 850 |
| scenarios (events) | 208 | 209 | 214 |
| questions | 240 | 240 | 240 |
| advertisers | 11 | 10 | 10 |
| deals | 39 | 41 | 41 |
| order lines | 71 | 71 | 76 |
| timeline span | 412 days | 412 days | 408 days |

Authoritative, in-run numbers live in
[`out/_across_seeds/completion_report.md`](out/_across_seeds/completion_report.md),
which recomputes every hash from a fresh generation rather than quoting one.

## Install and run

Python 3.10 or newer. The suite is run on 3.10, 3.11 and 3.12, and all three
produce byte-identical `facts.jsonl` for every seed — determinism holds across
interpreter versions, not just across runs. Every module runs as
`python3 -m <package>.<module>` from the repository root; relative imports
require it.

```bash
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r requirements.txt

# 1. Generate the universes (deterministic, no network).
python3 -m universe.generator --seeds 42,43,44

# 2. Render scenarios from templates (deterministic, no network).
python3 -m scenarios.renderer --seed 42 --deterministic

# 3. Verify two-sided fidelity of the rendered corpus.
python3 -m verify.fidelity --seed 42

# 4. Instantiate questions from the facts.
python3 -m questions.instantiate --seeds 42,43,44

# 5. Audit answerability with no network calls.
python3 -m verify.answerability --seed 42 --dry-run

# 6. Build the completion report (runs the test suite).
python3 -m report.completion

# Tests: fully green with zero network access.
python3 -m pytest tests/ -q
```

Every entry point takes `--seed` and writes under `out/<seed>/` inside its own
package. No seed is hardcoded anywhere; `tests/test_cli.py` checks that by
driving each entry point with a seed other than 42 and looking at where the bytes
landed. Cross-seed artifacts live in `out/_across_seeds/`.

`requirements.txt` is deliberately just PyYAML and pytest. boto3 lives in
`requirements-bedrock.txt` because it is only needed for the two paths below, it
pulls a 15 MB botocore wheel, and nothing else in the repository imports it.
Without it the suite still runs in full and the Bedrock tests skip themselves.

### The paths that cost money

Two paths call Bedrock and are **human-triggered**, never part of a test run:

```bash
python3 -m pip install -r requirements-bedrock.txt

# LLM render. --limit and --only exist for cost control.
python3 -m scenarios.renderer --seed 42 --limit 5

# Oracle answerability audit. One model call per question, never batched.
python3 -m verify.answerability --seed 42 --limit 20
```

They need AWS credentials and Bedrock access in the configured region
(`AWS_REGION`, falling back to `aws.region` in `config.yaml`). Every failure mode
names the exact thing that is missing — boto3, the region, the credentials, or
the model id — because "render failed" with no cause is a bug report nobody can
act on.

## Layout

```
universe/      generator.py, schema.py   -> out/<seed>/{facts,events}.jsonl, schema.json
scenarios/     renderer.py, templates/   -> out/<seed>/<event_id>/{manifest,meta}.json, rendered.md
questions/     instantiate.py            -> out/<seed>/questions.jsonl
verify/        matching.py, fidelity.py, answerability.py
report/        completion.py             -> out/_across_seeds/
tests/         full suite, zero network
decisions/     ADR-000-rebuild.md
settings.py    config.yaml loader
DEVIATIONS.md  every resolved spec gap: decision, alternative, why
PRINCIPLES.md  the methodology the build is justified by
```

## Core concepts

### Facts, not text

Facts are the only authority. The renderer reads facts and writes prose; the
question builder reads facts and writes questions. **The question builder never
opens a rendered scenario.** A question read out of the prose grades the
renderer's paraphrase instead of the ground truth, and its answer key cannot be
audited because there is nothing to audit it against.

Every fact carries `entity`, `attribute`, `value`, a `validity_interval`, a
`volatility_class` (permanent, durable, slow_changing, transient, ephemeral), the
`channel` and `author` it arrived on, a derived `trust_class`, a `supersedes`
pointer, and the `injection` / `never_memorize` flags.

`trust_class` is **derived** from `(channel, author)` through one table in
`universe/schema.py` and is never hand-set. The validator re-derives it and
rejects any record that disagrees.

### Corrections supersede; nothing is deleted

A correction is a new fact whose `supersedes` points at the old one; the old
fact stays, with its validity interval closed the day before the new one opens. A
cancellation is a superseding `line_status = 'cancelled'` fact, not a removal.

**Expired is not superseded.** A lapsed fact with no successor is `expired`; a
replaced fact is `superseded`. The first says memory holds a newer value, the
second says it holds none. A fact whose validity has not begun at the as-of date
is neither, and has no status at all.

Two invariants sit on top, both added after the corresponding defect appeared in
this build: a chain may not revisit a value (or a system that never applied the
correction answers correctly), and no `(entity, attribute)` pair may have two
active facts at the current as-of date (or "the current value" is undefined).

### The "current" as-of date

`current = max(fact.validity_interval.start) + 365 days`, from
`universe.current_offset_days` in `config.yaml`. Every current-state question is
asked as of that date, which is far enough past the last event that all transient
facts have lapsed and all corrections have landed.

### Matching: formats flexible, identity exact

`verify/matching.py` is the single implementation of "does this value appear in
this text". Fidelity and the judge both import it; neither has its own copy,
because two copies disagree silently and every downstream verdict inherits the
disagreement.

- **Numbers** compare numerically after currency, grouping and magnitude
  stripping: `$12,500` = `12,500 USD` = `12500` = `12500.0`, and `26.7` = `$26.70`
  because the comparison is on `Decimal`, not on strings.
- **Dates** compare as calendar dates: `April 18, 2025` = `18 Apr 2025` =
  `2025-04-18`. Both sides are canonicalized.
- **Names, codes and targeting strings** compare exactly after
  punctuation-stripping, whitespace-collapsing and case-folding. There is no
  semantic equivalence, ever: `New York` is not `NY`, `CTV` is not
  `Connected TV`.
- Name phrases never cross sentence punctuation or a newline.
- Judge containment is word-boundary: `active` is not contained in `inactive`.

### Two-sided fidelity

**Planted recall:** every manifest fact must be findable in the rendered text.
**Unsupported precision:** every number, date and name in the text must be
traceable to the manifest. Recall alone passes a renderer that includes all the
facts and adds a discount percentage.

All 208 seed-42 scenarios pass both sides on the deterministic path.

### Questions

Thirteen types, each with exactly one `query_kind` assigned by a table:

| type | query_kind | | type | query_kind |
|---|---|---|---|---|
| single_fact | exact | | expired_state | temporal_range |
| inference_only | multi_hop | | conflict_resolution | multi_hop |
| episodic | similarity | | order_state | exact |
| temporal | temporal_range | | mapping_lookup | exact |
| multi_session | multi_hop | | abstention_false_premise | similarity |
| knowledge_update_current | exact | | injection_probe | similarity |
| knowledge_update_past | temporal_range | | | |

As-of variants are asked at three horizon depths (early, mid, late) as well as at
the current date.

Rules that each encode a shipped defect:

- Every order-line-scoped question **names its IO id** in the text.
  "Northwind's order line" is unanswerable when Northwind has four of them.
- `expected_memory_points` exclude injected and never-memorize facts. Those
  facts stay in `evidence_fact_ids` — the audit has to confirm the temptation is
  in the corpus — but a system that stored them would be wrong.
- A superseded fact is never the gold answer for a current-state question. Past
  and temporal types reference superseded facts by design.

Refusal questions use sentinel gold answers, which the judge treats specially:
`INSUFFICIENT_EVIDENCE`, `FALSE_PREMISE`, `DO_NOT_STORE`,
`REJECT_UNVERIFIED_THIRD_PARTY_CLAIM`.

### The four verdicts

The oracle answers one question per call — never batched, because batching lets
context leak between questions and the audit stops measuring per-question
answerability. The deterministic judge then assigns one verdict, in this
precedence:

| verdict | meaning | disposition |
|---|---|---|
| `answered` | the judge matched the gold | — |
| `deceived` | the answer contains a planted lie string | model failure |
| `oracle_failed` | evidence **is** in the corpus, oracle missed it | question kept, logged as hard |
| `unanswerable` | evidence is **not** in the corpus | testbed defect, human review |

The split between the last two is the one that matters. Collapsing them lets
authoring defects be reported as model failures.

## Models

Set in `config.yaml`. A missing or `TBD` value is a hard error naming the exact
key; nothing substitutes a default model.

| role | model | sampling |
|---|---|---|
| renderer | `us.anthropic.claude-sonnet-4-6` | temperature 0.7, max_tokens 1024 |
| oracle | `amazon.nova-pro-v1:0` | temperature 0.0, max_tokens 512 |

Two constraints are enforced in code rather than documented and hoped for:

- **Cross-family auditing.** `resolve_oracle_model` refuses a configuration where
  the oracle shares a model family with the renderer. An oracle grading text
  written in its own idiom measures the wrong thing.
- **Temperature only, never `top_p`.** Claude on Bedrock rejects a request
  carrying both, so no code path in this repository ever sends `topP`. Anthropic
  model ids also require the `us.` inference-profile prefix, and a bare
  `anthropic.*` id is rejected with a message saying so.

## Testing

```bash
python3 -m pytest tests/ -q     # whole suite, no network, no credentials
```

No credentials, no network access and no boto3 are required. With boto3 absent,
`tests/test_llm_config.py` skips as a module via `pytest.importorskip` and
everything else runs unchanged.

Notable groups:

- **`test_matching.py`** — the matching policy, case by case. Trailing-period
  names match, cross-sentence tokens are not extracted, `26.7` = `$26.70`,
  `New York` ≠ `NY`, `active` ∉ `inactive`.
- **`test_determinism.py`** — each seed is generated **twice inside the test** and
  the payload hashes compared. Comparing against a stored fixture would prove
  only that the fixture is old. Seed independence is asserted on disjoint entity
  names, per-advertiser deal structure, correction placements, injection targets
  and question text.
- **`test_judge_calibration.py`** — 17 must-pass and 17 must-fail cases plus
  verdict precedence. No judged number is reportable until this is green.
- **`test_mutation.py`** — six deliberate defects, one per check: corrupt a fact
  value, invent a detail in the text, flip a gold answer, delete an expected
  memory point, corrupt a memory-point status, drop an IO id. A surviving mutant
  is a failed build, because it means a real defect of that shape would ship
  silently.
- **`test_error_paths.py`** — the guards. Testing these is what surfaced the
  magnitude-shorthand bug where `$1.2M` read as the number 1.

## Reading the artifacts

A rendered scenario is three files:

- `manifest.json` — the complete render input: subject, timestamp, channel,
  author, and every fact with its flags. Self-contained.
- `rendered.md` — the correspondence. Prose, never labelled fields.
- `meta.json` — event id, **timestamp**, render mode, model, attempt count and
  fidelity outcome. The timestamp is here as well as in the manifest because a
  previous build omitted it and every consumer that ordered scenarios by
  `meta.json` silently got insertion order instead.

## License

Apache-2.0. See [LICENSE](LICENSE).
