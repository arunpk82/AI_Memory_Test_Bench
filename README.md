# mem-testbed v2

A deterministic testbed for evaluating agent memory on an ad-sales
(Intent-Extractor domain) corpus. It generates a synthetic universe of facts and
events from a seed, renders those events as business emails and call notes,
mechanically instantiates questions from the facts (never from the rendered
text), verifies that the rendered text is faithful to the facts in both
directions, and audits whether the questions are actually answerable from the
corpus.

Status: **stub — under construction.** See `decisions/ADR-000-rebuild.md` for the
design and `DEVIATIONS.md` for every place the implementation departs from the
spec.

## License

Apache-2.0. See [LICENSE](LICENSE).
