# ADR-000 — mem-testbed v2 rebuild

- **Status:** accepted
- **Date:** 2026-08-27
- **Authority:** build specification, Arun
- **Supersedes:** any prior mem-testbed implementation

This record is backfilled from the rebuild specification. It captures the
decisions that shaped the code, including the ones that look odd until you know
what they are protecting against.

## Context

mem-testbed evaluates agent memory against an ad-sales corpus. A previous
implementation existed and is superseded wholesale. The rebuild targets open
source, so the artifacts have to be inspectable without credentials and the
reasoning has to be legible to a reader who was not in the room.

The measurement chain is: generate a deterministic universe of facts → render
those facts as business correspondence → verify the correspondence is faithful to
the facts in both directions → instantiate questions mechanically from the facts
→ audit whether those questions are answerable from the corpus. Every link can
fail silently, and most of them have.

## Decisions

### 1. Facts are the authority; text and questions are both derived

The generator produces `facts.jsonl` and `events.jsonl` first. The renderer reads
facts and writes prose. The question builder reads facts and writes questions.
The question builder never opens a rendered scenario.

*Consequence:* a rendering defect can never silently rewrite an answer key. It
shows up as a fidelity failure instead, which is attributable.

### 2. `trust_class` is derived, never assigned

One table maps `(channel, author)` to a trust class, and the schema validator
re-derives it and rejects any record whose stored value disagrees. An unmodelled
pair is an error, not a default — defaulting an unknown channel to the weakest
class would let a fabricated channel masquerade as a legitimate low-trust
source.

### 3. Corrections supersede; nothing is ever deleted

A correction is a new fact with `supersedes` pointing at the old one, and the old
fact's validity interval is closed the day before the new one opens. A
cancellation is a superseding `line_status = 'cancelled'` fact.

Two invariants sit on top of this, both added after the corresponding defect was
observed in this build:

- A chain may not revisit a value. If the current value equals a superseded one,
  a system that never applied the correction answers correctly.
- No `(entity, attribute)` pair may have two active facts at the current as-of
  date. "The current value" has to be single-valued or every current-state
  question about that pair is quietly broken.

### 4. Expired is not superseded

A lapsed fact with no successor is `expired`; a replaced fact is `superseded`. A
fact whose validity has not begun at the as-of date is neither, and has no
memory-point status at all — the status function raises rather than guessing.

### 5. Matching is one module, imported by everything

`verify/matching.py` is written first and is the only implementation of "does
this value appear in this text". Fidelity and the judge both import it. The
policy is: numeric comparison for numbers (so `26.7` equals `$26.70`), calendar
comparison for dates (so `April 18, 2025` equals `2025-04-18`), and exact
comparison after punctuation-stripping, whitespace-collapsing and case-folding
for names, codes and targeting strings — with no semantic equivalence, ever.

Name phrases never cross sentence punctuation or a newline, because
`Fabrikam Studios. The` was once extracted as a three-word company name.

### 6. Fidelity is verified in both directions

Planted recall asks whether every manifest fact is in the text. Unsupported
precision asks whether anything in the text is untraceable to the manifest.
Recall alone passes a renderer that invents a discount.

### 7. Both render paths are real

The deterministic template path drives CI with zero network calls. The Bedrock
path is fully implemented, with a fidelity-driven retry loop that feeds the
previous draft and the exact failure list back to the model. A stub in this
position once let a "complete" increment ship that could not render anything.

Retries carry the previous draft because blind retries — re-asking with the
original prompt — failed four times out of four on the same omission.

### 8. Questions are mechanical, and their invariants are checked on the output

Thirteen types, one `query_kind` per type taken from a table, as-of variants at
three horizon depths. Order-line-scoped questions name their IO id in the text,
because "Northwind's order line" is unanswerable when Northwind has four.
`expected_memory_points` exclude injected and never-memorize facts — those facts
stay in `evidence_fact_ids`, because the audit has to confirm the temptation is
in the corpus, but a system that stored them would be wrong.

The builder re-checks these rules on the finished set rather than trusting itself
to have followed them. That check is what caught the reverting-value chain.

### 9. The judge is deterministic and calibrated before it is read

The judge only asks whether a known gold token is present in an answer. It never
runs entity extraction over an answer — extraction is tuned for corpus prose, and
running it on answers produced roughly 28 false failures.

It ships with a calibration suite of must-pass and must-fail cases and does not
report a number until that suite is green. Sentinel abstentions are recognised
from ordinary prose as well as from the literal sentinel string, because the
oracle has only ever been told about one of the four sentinels.

### 10. Four verdicts, and the defect list is separated from the model result

`answered` → `deceived` → `oracle_failed` → `unanswerable`, in that precedence.
The split that matters is the last two: if the evidence is in the corpus, an
oracle miss is a model result and the question is kept and logged as hard; if the
evidence is absent, the question is a testbed defect and goes to human review.
Collapsing them lets authoring defects be reported as model failures.

### 11. Cross-family auditing is enforced in code

The renderer is a Claude model and the oracle is an Amazon model, and
`resolve_oracle_model` refuses a configuration where they share a family. An
oracle grading text written in its own idiom is measuring the wrong thing.

### 12. Three seeds are the replication unit

Statistical claims come from agreement across seeds 42, 43 and 44, not from
question count within one seed, because questions inside one universe are
correlated. Independence is asserted, not assumed: disjoint entity-name sets,
different per-advertiser deal structures, different correction placements,
different injection targets, disjoint question text.

### 13. Dormant paths are declared and hashes are recomputed

The completion report lists every unreachable branch and unfired guard, and
computes every hash itself from fresh generations, cross-checked against the
bytes on disk. A narrated hash shipped stale once.

## Consequences

- The generator asserts its own quotas and refuses to write a universe that fails
  them. Three quota failures were caught this way during the build.
- CI needs no credentials: the deterministic render and the `--dry-run` audit
  cover the whole structural pipeline.
- Live LLM rendering and the live oracle audit are human-triggered per seed, for
  cost control. `--limit` and `--only` exist for that.
- Some paths are unavoidably dormant, notably the live Bedrock calls and five of
  the nine trust-table rows. They are listed rather than hidden.

## Alternatives considered

- **Trust-precedence resolution of conflicting facts** instead of a `_claimed`
  attribute. Rejected: it makes every current-state gold answer depend on a
  ranking function. See [D-006](../DEVIATIONS.md).
- **A language-model adjudicator for judge disputes.** Rejected explicitly by the
  specification, and rightly: it would make the judge non-deterministic. Disputes
  are tagged for blind human adjudication instead. See
  [D-012](../DEVIATIONS.md).
- **Literal-only sentinel matching in the judge.** Rejected: it would put all 35
  refusal questions per seed on the defect list. See [D-012](../DEVIATIONS.md).
