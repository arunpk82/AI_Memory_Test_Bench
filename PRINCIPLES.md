# Principles

> **Provenance note.** The build specification says of this file: *"Arun
> supplies; commit as given."* No authored text was supplied with the
> specification. What follows is therefore **not** the authored document. It is a
> placeholder that records the principles this build actually operates under,
> every one of them taken from the specification's own methodology statements, so
> that the reasoning the code is justified by is written down somewhere.
>
> **Replace this file verbatim when the authored text arrives.** Do not merge the
> two; the entries below are a reconstruction and should not be mistaken for the
> original. Recorded as [D-002](DEVIATIONS.md#d-002-principlesmd-was-not-supplied).

---

## Ground Truth First — facts before text

The universe of facts is generated first and is the sole authority. Rendered
prose is derived from facts; questions are derived from facts. Nothing is ever
derived from the rendered text.

A question read out of the prose grades the renderer's paraphrase instead of the
ground truth, and its answer key cannot be audited, because there is nothing to
audit it against.

## Every rule encodes a defect already paid for once

The constraints in this codebase are not style preferences. Each one exists
because its absence produced a specific, expensive failure. Where a rule looks
arbitrary, the comment next to it says what broke. Removing a rule means
re-buying the defect.

## Determinism is a property, not an aspiration

Same seed, same bytes, forever. No clock, no network, no dictionary package, no
language model anywhere in the generation path. Determinism is verified by
generating twice inside the test and comparing hashes, never by comparing against
a stored fixture — a fixture comparison proves only that the fixture is old.

## Replication comes from seeds, not from question count

Questions inside one universe are correlated: a single generator quirk hits
dozens of them at once. The replication unit is therefore the universe. A
statistical claim rests on agreement across seeds 42, 43 and 44; it never rests
on the number of questions inside one of them.

Independence has to be verified rather than assumed. Different seeds must produce
genuinely different entity names, deal structures, correction placements and
injection targets — a generator that quietly ignores its seed passes every other
test in the suite.

## One implementation of any judgement

Matching is implemented once, in `verify/matching.py`, and every checker imports
it. Two copies of a judgement rule disagree silently, and every verdict
downstream inherits the disagreement.

The same applies to trust: `trust_class` is derived from `(channel, author)`
through one table and is never hand-set at a call site.

## Formats are flexible; identity is exact

`$12,500` and `12500` are the same number. `April 18, 2025` and `2025-04-18` are
the same day. `New York` is not `NY`, and `CTV` is not `Connected TV`. Format
tolerance prevents false misses; semantic tolerance would let a wrong answer
pass, so there is none.

## No destructive delete

A correction is a new fact that supersedes the old one. The old fact stays, with
its validity interval closed. A cancellation is a superseding
`line_status = 'cancelled'` fact, not a removal. History that has been deleted
cannot be asked about, and a memory testbed that cannot ask about history is
measuring nothing interesting.

*Expired* — lapsed, with no successor — is not *superseded* — replaced. The first
says memory holds a newer value; the second says it holds none.

## A checker that never fails is not a checker

Every consistency check has a mutant in the test suite that makes it fire.
Corrupt a fact value, flip a gold answer, delete an expected memory point: if the
mutant survives, the build has failed, because a real defect of that shape would
ship silently.

## Verification is two-sided

Checking that every planted fact appears in the text catches omission. It says
nothing about invention. A renderer that includes all the facts and adds a
discount percentage has produced a corpus in which a fabricated answer looks
correct, so precision is checked as well as recall.

## An instrument is calibrated before it is read

The judge ships with its calibration suite, and no judged number is reportable
until that suite is green. Known judge limitations are written down and left
visible rather than patched over — and specifically not patched over with a
language model, because a non-deterministic judge is not a judge.

## Dormant paths are declared

Every unreachable branch, every guard no test has fired, every configuration key
that is set but never exercised, is listed in the completion report. An empty
dormant-paths section is a claim about the build, and claims have to be true.

## Report what this run computed

Numbers in a report are computed by the reporting step, in the reporting run.
Hashes are recomputed from fresh generations and cross-checked against the bytes
on disk. A transcribed number looks exactly as authoritative as a live one and is
worth nothing.
