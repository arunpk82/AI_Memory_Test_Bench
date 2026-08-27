"""Determinism and seed independence.

Determinism is checked by generating the same seed twice *inside the test* and
comparing hashes of the exact bytes that would be written. Comparing against a
cached fixture proves only that the fixture is old.

Seed independence matters more than it looks: a generator that quietly ignores
its seed passes every other test in this suite, and the three-universe design
then reports one universe three times while claiming cross-seed agreement.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import pytest

from questions.instantiate import build_questions
from universe.generator import generate

SEEDS = (42, 43, 44)
PAIRS = list(itertools.combinations(SEEDS, 2))


def payload(records: list[dict]) -> str:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False) + "\n" for record in records)


def sha256(records: list[dict]) -> str:
    return hashlib.sha256(payload(records).encode("utf-8")).hexdigest()


# ------------------------------------------------------------- determinism ---

@pytest.mark.parametrize("seed", SEEDS)
def test_facts_are_byte_identical_across_two_fresh_generations(seed):
    first, second = generate(seed), generate(seed)
    assert sha256(first.facts) == sha256(second.facts)


@pytest.mark.parametrize("seed", SEEDS)
def test_events_are_byte_identical_across_two_fresh_generations(seed):
    first, second = generate(seed), generate(seed)
    assert sha256(first.events) == sha256(second.events)


@pytest.mark.parametrize("seed", SEEDS)
def test_questions_are_byte_identical_across_two_fresh_generations(seed):
    first, second = generate(seed), generate(seed)
    left = build_questions(first.facts, first.events, seed)
    right = build_questions(second.facts, second.events, seed)
    assert sha256(left) == sha256(right)


def test_different_seeds_produce_different_fact_hashes(all_universes):
    hashes = {seed: sha256(universe.facts) for seed, universe in all_universes.items()}
    assert len(set(hashes.values())) == len(SEEDS), hashes


# -------------------------------------------------------- seed independence ---

def advertiser_names(universe) -> set[str]:
    return {advertiser.name for advertiser in universe.advertisers}


def deal_count_vector(universe) -> list[int]:
    """Deals per advertiser, in advertiser order.

    The spec asks for "different deal counts". The scalar total is a poor test:
    it is a sum of ten or eleven draws from {3, 4, 5}, so two seeds collide
    routinely (43 and 44 both total 41) without the universes being remotely
    alike. The per-advertiser vector is the structural quantity that "different
    deal counts" is trying to capture. See DEVIATIONS.md.
    """
    return [len([deal for deal in universe.deals if deal.advertiser is advertiser])
            for advertiser in universe.advertisers]


def correction_placements(universe) -> set[tuple[str, str]]:
    return {(deal.deal_id, deal.plan) for deal in universe.deals
            if deal.plan != "clean"}


def injection_targets(universe) -> set[tuple[str, str, str]]:
    return {(fact["entity_id"], fact["attribute"], fact["value"])
            for fact in universe.facts if fact["injection"]}


@pytest.mark.parametrize("left,right", PAIRS)
def test_entity_name_sets_differ(all_universes, left, right):
    assert advertiser_names(all_universes[left]) != advertiser_names(all_universes[right])


@pytest.mark.parametrize("left,right", PAIRS)
def test_entity_name_sets_do_not_even_overlap(all_universes, left, right):
    overlap = (advertiser_names(all_universes[left])
               & advertiser_names(all_universes[right]))
    assert not overlap, f"seeds {left} and {right} share advertisers: {overlap}"


@pytest.mark.parametrize("left,right", PAIRS)
def test_deal_structure_differs(all_universes, left, right):
    assert deal_count_vector(all_universes[left]) != deal_count_vector(all_universes[right])


@pytest.mark.parametrize("left,right", PAIRS)
def test_correction_placements_differ(all_universes, left, right):
    assert correction_placements(all_universes[left]) != \
        correction_placements(all_universes[right])


@pytest.mark.parametrize("left,right", PAIRS)
def test_injection_targets_differ(all_universes, left, right):
    assert injection_targets(all_universes[left]) != injection_targets(all_universes[right])


@pytest.mark.parametrize("left,right", PAIRS)
def test_question_texts_are_not_identical(all_universes, left, right):
    left_texts = {question["text"] for question in
                  build_questions(all_universes[left].facts,
                                  all_universes[left].events, left)}
    right_texts = {question["text"] for question in
                   build_questions(all_universes[right].facts,
                                   all_universes[right].events, right)}
    assert left_texts != right_texts
    assert not (left_texts & right_texts), "seeds share verbatim question text"


def test_advertiser_counts_are_not_all_equal(all_universes):
    counts = {seed: len(universe.advertisers)
              for seed, universe in all_universes.items()}
    assert len(set(counts.values())) > 1, counts
