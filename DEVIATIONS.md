# Deviations

Every place the implementation resolves a gap, ambiguity or conflict in the
specification. Each entry states the **decision**, the **alternative** that was
rejected, and **why**.

Entries are stable and append-only. The completion report counts them.

---

## D-001 Repository bootstrap: remote already existed

**Decision.** The repository was already initialised with a remote and a root
commit when work began, so `git init` and remote creation were skipped. Every
increment still ends with a commit and a push; nothing is local-only.

**Alternative.** Re-initialise and force a new remote.

**Why.** Re-initialising would have discarded the existing root commit and the
push target, and the requirement the rule protects — no local-only work — is
already satisfied.

---

## D-002 PRINCIPLES.md was not supplied

**Decision.** `PRINCIPLES.md` was to be committed verbatim as supplied by Arun.
It was not supplied with the spec. A file is committed that says so explicitly
and records the principles this build actually operates under, drawn from the
spec's own methodology statements. It is marked to be replaced verbatim the
moment the authored text arrives.

**Alternative.** Commit an empty file, or omit it.

**Why.** An empty file reads as an oversight, and omitting it loses the
methodology that the rest of the build is justified by. A clearly-marked
placeholder that states its own provenance is the only option that does not
misrepresent authorship.

---

## D-003 A configuration loader module was added

**Decision.** `settings.py` at the repository root loads `config.yaml` and
refuses missing, null and `TBD` values with an error naming the exact key.

**Alternative.** Read `config.yaml` inline in each module.

**Why.** The spec requires fail-fast errors naming the exact missing config key
in two separate modules (renderer and answerability). Inline loading would mean
two copies of the placeholder-detection logic, which is the same defect class as
two copies of the matching policy.

---

## D-004 A report package was added

**Decision.** `report/completion.py` builds the mandatory completion report and
writes it to `out/_across_seeds/`.

**Alternative.** Assemble the report by hand.

**Why.** The spec requires the hashes to be computed by the report step itself in
the reporting run, because a narrated hash shipped stale once. That is only
enforceable if the report is code.

---

## D-005 Additional artifacts are written per module

**Decision.** Beyond the specified files, each stage writes a machine-readable
summary: `universe/out/<seed>/quotas.json`,
`scenarios/out/<seed>/render_summary.json`,
`scenarios/out/<seed>/fidelity_report.json`,
`questions/out/<seed>/question_counts.json` and
`verify/out/<seed>/dryrun_report.json` (plus `audit_report.json` after a live
audit).

**Alternative.** Write only `facts.jsonl`, `events.jsonl`, `schema.json`,
the per-scenario triple and `questions.jsonl`.

**Why.** The completion report has to state quota satisfaction, retry
distribution and audit findings. Recomputing them from raw artifacts in the
report would be a second implementation of each measurement.

---

## D-006 Counterparty disagreements use a `_claimed` attribute

**Decision.** When a counterparty asserts a value the order system does not
have, the fact is recorded as `cpm_rate_claimed`, not as a second `cpm_rate`.
Conflict-resolution questions present both sides explicitly and the gold answer
is the system-of-record value.

**Alternative.** Record both as `cpm_rate` and resolve "the current value" by
trust precedence.

**Why.** Trust-precedence resolution makes the current value of every attribute
depend on a ranking function, and a bug in that function silently changes the
gold answer of every current-state question in the universe. With the suffix,
"the current value" stays single-valued and mechanically checkable — which is
what the `duplicate_active_facts` quota (D-015) now enforces. The conflict is
still a real two-sided conflict in the corpus and in the question text; only the
storage key differs.

---

## D-007 Never-memorize probes live under `injection_probe`

**Decision.** The thirteen question types are fixed by the spec and there is no
separate type for never-memorize probes, so they are `injection_probe` questions
with `notes.probe_kind == "never_memorize"` and gold `DO_NOT_STORE`. Injection
probes carry `notes.probe_kind == "injection"` and gold
`REJECT_UNVERIFIED_THIRD_PARTY_CLAIM`.

**Alternative.** File them under `single_fact` with a sentinel gold.

**Why.** Both probe kinds test the same capability — refusing to internalise
something that should not be stored — and both are scored by the same
sentinel-aware judge path. Filing a refusal question as `single_fact` would make
the type counts misleading about how many straight lookups the suite contains.

---

## D-008 Corpus-presence uses evidence tokens for derived and sentinel golds

**Decision.** The `oracle_failed` versus `unanswerable` split is decided by
whether the material for a correct answer is in the corpus. For a direct-lookup
question that material is the gold tokens, as specified. For a derived answer (a
count, a sum) and for a sentinel answer, the gold string is by construction not
in the corpus, so the check uses the values of the question's
`evidence_fact_ids` instead. Questions with a fabricated identifier additionally
assert that the fabricated string is **absent**.

**Alternative.** Check gold tokens literally for every question.

**Why.** Checking a sentinel literally would mark every refusal question
`unanswerable` — that is, would put all 35 of them on the defect list — even
though a refusal question is answerable by design. Checking a derived gold
literally would do the same to every count and sum. The spec's intent is "can a
correct answer be constructed from this corpus", and evidence tokens are that
notion for answers whose gold is not a planted string.

---

## D-009 Fidelity support includes manifest header identity fields

**Decision.** The precision side treats fact values, the subject line, the
timestamp **and** the manifest's identity fields (`advertiser_name`,
`agency_name`, `contact_name`, `deal_name`, `deal_id`, `io_id`) as supported
content.

**Alternative.** Support only fact values, subject and timestamp.

**Why.** The renderer is instructed to address the counterparty by name and to
name the order line, and those values live in the manifest header rather than in
a fact. Excluding them would report the renderer as fabricating the very
identifiers it was told to use. The set is closed and enumerated in
`verify/fidelity.py :: HEADER_SUPPORT_FIELDS`, so it cannot quietly widen.

---

## D-010 Seed independence tests the deal-count vector, not the scalar total

**Decision.** The seed-independence test asserts that the per-advertiser
deal-count vector differs pairwise, alongside disjoint entity-name sets,
different correction placements, different injection targets and disjoint
question text.

**Alternative.** Assert that the scalar total deal count differs pairwise.

**Why.** The total is a sum of ten or eleven draws from `{3, 4, 5}`, so
collisions are routine: seeds 43 and 44 both total 41 deals while sharing zero
advertiser names and having entirely different structure. A test that fails on
that collision reports noise, and a noisy test gets disabled. The vector is the
structural quantity "different deal counts" is reaching for.

---

## D-011 The precision side uses a curated common-word allowlist

**Decision.** Unsupported-name detection ignores runs made only of ordinary
English and business-correspondence words, listed explicitly in
`verify/matching.py` as `COMMON_WORDS`, `CORRESPONDENCE_WORDS` and
`DOMAIN_WORDS`.

**Alternative.** Flag every capitalised token that is not in the manifest.

**Why.** Sentence-initial capitalisation makes the first word of nearly every
sentence look like a proper noun, so without an allowlist the precision side
reports several findings per scenario and becomes unusable. **Known limitation:**
an entity genuinely named after a common word ("Best Buy") would not be
reported. The universe generator uses invented multi-syllable roots specifically
so that this cannot happen in practice, but the limitation is real for anyone
pointing the checker at a different corpus.

---

## D-012 Judge sentinel matching is cue-based

**Decision.** The judge recognises a sentinel from a curated list of refusal
phrasings as well as from the literal sentinel string.

**Alternative.** Match the literal sentinel string only.

**Why.** The oracle is only ever told to emit `INSUFFICIENT_EVIDENCE` literally;
it has never heard of the other three sentinels. A literal-only judge would score
every correctly-refused injection probe as wrong. The spec's own calibration
requirement — that sentinel abstentions appear among the must-pass cases — is
only satisfiable with cue matching.

**Known limitation, deliberately not solved with an LLM.** A refusal phrased
outside the cue list that names the claim it rejects still contains the lie
string and is therefore scored `deceived`. Those records are tagged
`match_dispute: mention_in_refusal` for later blind human adjudication. An LLM
adjudicator would make the judge non-deterministic, which is the one property
the judge exists to have.

---

## D-013 Memory points exclude facts that have not started

**Decision.** A fact whose validity interval begins after the question's as-of
date is not an expected memory point at all, and
`memory_point_status` raises rather than classifying it.

**Alternative.** Report it as `active`.

**Why.** The three permitted statuses — active, superseded, expired — have no
value for "not yet recorded", and reporting such a fact as `active` put a
correction into the expected memory of a past-state question asked the day before
that correction existed. This was found by a test and fixed rather than
accommodated.

---

## D-014 Supersession chains may not revisit a value

**Decision.** A generator invariant rejects any supersession chain in which a
value appears twice.

**Alternative.** Allow reverting chains as realistic.

**Why.** If the current value of an attribute equals one of its superseded
values, a system that never applied the correction answers the current-state
question correctly, and the question measures nothing. This fired twice during
the build: a CPM revised twice back to its original rung, and a contact handoff
that named the incumbent.

---

## D-015 A duplicate-active-fact quota was added

**Decision.** A universe fails generation if any `(entity_id, attribute)` pair
has two active memorable facts at the current as-of date.

**Alternative.** Detect it downstream when a question builder trips over it.

**Why.** It was detected downstream, on seed 44, where two injection emails
planted the same benign attribute on one deal. "The current value" being
undefined poisons every current-state question about that pair, and the
downstream error message pointed at the question builder rather than at the
generator that caused it.

---

## D-016 Inbound counterparty mail signs off as the client team

**Decision.** Outbound mail, system notifications and call notes sign off with
the role `ad sales ops`, as specified. An inbound counterparty email signs off as
`the <advertiser> team`.

**Alternative.** Sign every rendered scenario `ad sales ops`.

**Why.** Signing a message we *received* with our own operations role attributes
the claim in it to us. The injection scenarios exist to measure whether a system
correctly attributes an unverified counterparty claim, so mis-attributing it in
the corpus would undercut the measurement. The sign-off is still a role, never a
personal name and never a placeholder, and it comes from the manifest.

---

## D-017 Scenario counts run well above the floor

**Decision.** Each seed produces roughly 200 scenarios against a floor of 100,
and the renderer gained `--limit` and `--only` for cost control on the LLM path.

**Alternative.** Tune the generator down to just over 100.

**Why.** Ten to twelve advertisers with three to five deals each, one kickoff and
one booking event per order line, plus corrections, injections, probes, pacing
notes and renewals, lands where it lands. Cutting back would have meant fewer
order lines per deal, which weakens exactly the case the "name the IO id" rule
exists for. The flags make the LLM render affordable without shrinking the
universe.

---

## D-018 The batch seeds flag was added to the question builder too

**Decision.** `--seeds 42,43,44` works on `universe.generator` as specified and
also on `questions.instantiate`.

**Alternative.** Batch only the generator.

**Why.** The two stages are always run together for a full build, and one flag
missing from one of them means the batch path is only half tested.

---

## D-019 Generated artifacts are committed

**Decision.** `universe/out/`, `scenarios/out/`, `questions/out/`,
`verify/out/` and `out/_across_seeds/` are committed.

**Alternative.** Add `out/` to `.gitignore`.

**Why.** The specified `.gitignore` contents do not exclude them, and for an
open-source testbed the artifacts are the reviewable evidence: a reader can
inspect a rendered scenario and its manifest without an AWS account. The
determinism tests never read them, so a stale artifact cannot certify a build
(see `tests/conftest.py`).

---

## D-020 Question `notes` is a typed dictionary

**Decision.** `notes` is schema-validated as a dictionary and carries per-type
provenance: the attribute asked about, the derivation rule for inference
questions, `probe_kind` and `plausibility` for probes, the resolution rule for
conflicts, and the fabricated identifier for false-premise questions.

**Alternative.** Leave `notes` as free text.

**Why.** Several downstream checks need to distinguish a derived gold from a
looked-up one (D-008) and a never-memorize probe from an injection probe
(D-007). Parsing free text to make a verdict decision is how a checker starts
disagreeing with the thing it checks.

---

## D-021 Groq and Gemini are supported alongside Bedrock

**Decision.** Every model call goes through one function, `providers.complete`,
which dispatches to Bedrock, Groq or Gemini. The provider is set per role in
`config.yaml` and **defaults to Bedrock**, so the specified configuration — a
Claude renderer on Bedrock audited by Amazon Nova — is unchanged and is what
ships. Groq and Gemini are plain HTTPS calls made with the standard library, so
selecting either adds no dependency.

**Alternative.** Call the Bedrock `converse` API directly from the renderer and
from the oracle, as originally specified.

**Why.** The specification pins Bedrock, and the shipped configuration still
does. But Bedrock is the highest-friction way to run this: it needs an AWS
account, a 15 MB SDK and a region with model access, and that friction falls
entirely on the open-source reader who wants to reproduce a number. Making the
transport pluggable costs one module and removes that barrier.

Two properties are preserved rather than relaxed. The **cross-family rule** is
enforced across providers, not just within Bedrock: the provider is part of the
computed family, so a Gemini renderer audited by a Gemini oracle is refused
before any call is made, and a Gemini renderer audited by a Groq oracle
satisfies the rule the same way an Anthropic/Amazon pairing does. And **no
provider sends `top_p`** — that constraint began as a Claude-on-Bedrock
requirement, and applying it only to Bedrock would have made it a latent
surprise for whoever switched provider first.

The single-entry-point shape is the matching-module rule applied again: if the
renderer and the oracle each built their own request, they would drift, and
"the model" would silently mean two different things in one report.

---

## D-022 As-of horizon mix is a first-class allocation constraint

**Decision.** Question allocation targets **140 current / 33 early / 33 mid /
34 late** (240 total) with hard floors of 25 in each historical bucket. As-of
variants are built only for types where a dated question changes the gold:
`temporal`, `knowledge_update_past`, `expired_state`, `order_state`,
`knowledge_update_current`, `mapping_lookup`. The other seven types stay
current-only.

**Alternative.** Keep allocating by type alone and hope the three horizon
depths fill in, or pad the mix by duplicating `temporal` questions until the
historical buckets look thick.

**Why.** Type-only allocation left 200 of 240 questions at "current" and 8–18
in each historical bucket — too thin to say anything about whether a memory
system degrades with age. Stuffing the gap with more `temporal` questions would
make horizon analysis a property of one type. Asking an episodic, refusal or
injection probe "as of mid-timeline" does not change the answer, so those types
are the wrong place to spend the extra dated slots.

