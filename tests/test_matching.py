"""Self-tests for the shared matching policy.

Every case here is a defect that was shipped once. The module docstring in
``verify/matching.py`` explains the policy; this file pins it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from verify import matching as m


# ---------------------------------------------------------------- numbers ----

@pytest.mark.parametrize("left,right", [
    ("$12,500", "12500"),
    ("$12,500", "12,500 USD"),
    ("12500", "12500.0"),
    ("26.7", "$26.70"),
    ("26.70", "26.7"),
    ("1,250,000", "1250000"),
    ("0.5", ".50"),
    ("15%", "15"),
])
def test_number_format_equivalence(left, right):
    assert m.values_match(left, right)
    assert m.values_match(right, left)


def test_trailing_zero_equality_is_numeric_not_string():
    # A string comparison here shipped a false miss.
    assert m.canonical_number("26.70") == m.canonical_number("26.7")
    assert m.canonical_number("$26.70") == Decimal("26.7")


@pytest.mark.parametrize("left,right", [
    ("12500", "12501"),
    ("26.7", "26.8"),
    ("-500", "500"),
])
def test_different_numbers_do_not_match(left, right):
    assert not m.values_match(left, right)


def test_multipliers_canonicalize():
    assert m.canonical_number("1.2M") == Decimal("1200000")
    assert m.canonical_number("450k") == Decimal("450000")


def test_non_numbers_are_not_numbers():
    for value in ["IO-42-0071", "cancelled", "A25-54", "", "CTV"]:
        assert m.canonical_number(value) is None


# ------------------------------------------------------------------ dates ----

@pytest.mark.parametrize("surface", [
    "April 18, 2025", "18 Apr 2025", "2025-04-18", "Apr 18, 2025",
    "April 18th, 2025", "04/18/2025", "18 April 2025",
])
def test_date_format_equivalence(surface):
    assert m.canonical_date(surface) == date(2025, 4, 18)
    assert m.values_match(surface, "2025-04-18")
    assert m.values_match("2025-04-18", surface)


def test_different_dates_do_not_match():
    assert not m.values_match("April 18, 2025", "April 19, 2025")
    assert not m.values_match("2025-04-18", "2024-04-18")


def test_impossible_date_is_rejected():
    assert m.canonical_date("2025-02-30") is None


def test_date_in_text_any_format():
    text = "We locked the flight to start on April 18, 2025 as discussed."
    assert m.value_in_text("2025-04-18", text)
    assert not m.value_in_text("2025-04-19", text)


# ------------------------------------------------------- names and codes ----

def test_trailing_period_name_matches():
    # "Studios." must match "Studios".
    text = "That was booked under Fabrikam Studios. The paperwork follows."
    assert m.value_in_text("Fabrikam Studios", text)


def test_cross_sentence_tokens_are_not_extracted_as_a_name():
    # "Fabrikam Studios. The" was once extracted as a single name phrase.
    text = "That was booked under Fabrikam Studios. The paperwork follows."
    phrases = m.extract_name_phrases(text)
    assert "Fabrikam Studios" in phrases
    assert not any("Studios The" in p or "Studios. The" in p for p in phrases)


def test_name_phrase_does_not_cross_newline():
    text = "Contoso Media\nGroup Holdings signed."
    assert not m.value_in_text("Contoso Media Group", text)
    assert m.value_in_text("Contoso Media", text)


def test_no_semantic_equivalence_for_names():
    assert not m.values_match("New York", "NY")
    assert not m.values_match("CTV", "Connected TV")
    assert not m.values_match("California", "CA")


def test_targeting_codes_are_separator_flexible_but_token_exact():
    assert m.values_match("CA/NY/TX", "CA, NY, TX")
    assert m.values_match("A25-54", "a25 54")
    assert not m.values_match("CA/NY/TX", "CA/NY/FL")
    assert not m.values_match("A25-54", "W25-54")


def test_case_and_punctuation_folding():
    assert m.values_match("Northwind Traders", "northwind traders")
    assert m.values_match("Northwind Traders,", "Northwind Traders")


# --------------------------------------------------- judge containment ----

def test_active_is_not_contained_in_inactive():
    assert not m.token_in_text("active", "The line is inactive right now.")
    assert m.token_in_text("active", "The line is active right now.")


def test_token_in_text_is_word_boundary_not_substring():
    assert not m.token_in_text("cancel", "The order was cancelled last week.")
    assert m.token_in_text("cancelled", "The order was cancelled last week.")


def test_token_in_text_handles_numbers_and_dates_numerically():
    assert m.token_in_text("26.7", "We settled at a $26.70 CPM.")
    assert m.token_in_text("12500", "Budget came in at $12,500 net.")
    assert m.token_in_text("2025-04-18", "Flight starts April 18, 2025.")


def test_token_in_text_multiword_phrase():
    text = "Please copy Dana Whitfield on the revision."
    assert m.token_in_text("Dana Whitfield", text)
    assert not m.token_in_text("Dana Whitmore", text)


# ------------------------------------------------------------ extraction ----

def test_iso_date_does_not_leak_numbers():
    numbers = [n for _, n in m.extract_numbers("Start 2025-04-18 confirmed.")]
    assert numbers == []


def test_io_id_does_not_leak_numbers():
    numbers = [n for _, n in m.extract_numbers("Line IO-42-0071 is live.")]
    assert numbers == []
    assert "IO-42-0071" in m.extract_name_phrases("Line IO-42-0071 is live.")


def test_currency_and_grouping_extracted_once():
    found = [n for _, n in m.extract_numbers("Net was $12,500 for the flight.")]
    assert found == [Decimal("12500")]


def test_possessive_suffix_is_not_a_token():
    text = "Northwind's revision landed today."
    assert m.extract_name_phrases(text) == ["Northwind"]
    assert m.value_in_text("Northwind", text)


def test_edge_punct_strip_preserves_interior():
    assert m.strip_edge_punct('"A25-54",') == "A25-54"
    assert m.strip_edge_punct("Studios.") == "Studios"


# --------------------------------------------------------------- support ----

def test_unsupported_number_is_flagged():
    support = m.build_support(["26.7", "Northwind Traders"], ["Rate confirmation"])
    findings = m.unsupported_items(
        "Northwind Traders is locked at 26.7 with a 15% discount.", support)
    assert findings["numbers"] == ["15%"]


def test_unsupported_date_is_flagged():
    support = m.build_support(["2025-04-18"], [])
    findings = m.unsupported_items(
        "Flight starts April 18, 2025 and bills on May 2, 2025.", support)
    assert findings["dates"] == ["May 2, 2025"]


def test_unsupported_name_is_flagged():
    support = m.build_support(["Northwind Traders", "CTV"], ["Booking recap"])
    findings = m.unsupported_items(
        "Northwind Traders on CTV, routed through Halcyon Buying.", support)
    assert findings["names"] == ["Halcyon Buying"]


def test_supported_content_is_clean():
    support = m.build_support(
        ["Northwind Traders", "IO-42-0071", "26.7", "2025-04-18", "CA/NY/TX"],
        ["Northwind rate confirmation"])
    text = ("Hi there, quick confirmation on Northwind Traders. Line IO-42-0071 "
            "is holding at a 26.7 CPM, flight opens 2025-04-18, and the geo stays "
            "CA/NY/TX. Thanks, ad sales ops")
    findings = m.unsupported_items(text, support)
    assert findings == {"numbers": [], "dates": [], "names": []}


def test_partial_date_supported_by_month_and_year():
    support = m.build_support(["2025-04-18"], [])
    assert m.unsupported_items("We start in April 2025.", support)["dates"] == []
    assert m.unsupported_items("We start in May 2025.", support)["dates"] == ["May 2025"]


def test_sentence_segments_respect_abbreviations():
    segments = m.sentence_segments("Approx. 4 lines remain. Next week we bill.")
    assert len(segments) == 2
    assert segments[0].text.startswith("Approx. 4 lines remain")
