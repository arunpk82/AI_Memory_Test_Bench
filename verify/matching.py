"""Shared matching policy for mem-testbed.

This is the single implementation of "does this value appear in this text" and
"does this answer contain this gold token". Fidelity checking (both sides) and
the answerability judge import it. Neither is permitted to implement its own
matching: divergent copies of a matching policy are the single most expensive
class of defect this testbed has produced, because the two copies disagree
silently and every downstream verdict inherits the disagreement.

Policy, in one sentence: **formats are flexible, identity is exact.**

* Numbers compare numerically after currency, grouping and unit stripping, so
  ``$12,500``, ``12,500 USD``, ``12500`` and ``12500.0`` are one value, and
  ``26.7`` equals ``$26.70`` because the comparison is on ``Decimal``, not on
  strings. A string comparison here once shipped a false miss on a trailing
  zero.
* Dates compare as calendar dates, so ``April 18, 2025``, ``18 Apr 2025`` and
  ``2025-04-18`` are one value. Both sides are canonicalized; neither side is
  assumed to already be ISO.
* Names, codes and targeting strings (``CA/NY/TX``, ``A25-54``, ``CTV``,
  company and person names) compare exactly after punctuation-stripping,
  whitespace-collapsing and case-folding. There is no semantic equivalence,
  ever: ``New York`` is not ``NY``, and ``CTV`` is not ``Connected TV``.
* Tokens are stripped of leading and trailing punctuation before matching, and
  a name phrase never crosses sentence punctuation or a newline. Without the
  second rule, ``Fabrikam Studios. The`` gets extracted as a three-word company
  name.
* Judge containment is :func:`token_in_text`: word-boundary containment of the
  gold token in the normalized answer text. ``active`` is not contained in
  ``inactive``. The judge checks tokens *in* the answer; it never runs entity
  extraction over an answer. Extraction-on-answers produced roughly 28 false
  failures before it was removed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

__all__ = [
    "ValueKind",
    "Extraction",
    "TextIndex",
    "classify_value",
    "canonical_number",
    "canonical_date",
    "canonical_partial_date",
    "normalize_name",
    "name_tokens",
    "sentence_segments",
    "extract_numbers",
    "extract_dates",
    "extract_name_phrases",
    "extract_all",
    "values_match",
    "value_in_text",
    "token_in_text",
    "any_value_in_text",
    "unsupported_items",
    "build_support",
    "Support",
]


# --------------------------------------------------------------------------
# Normalization primitives
# --------------------------------------------------------------------------

_QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2026": "...",
}

#: Characters stripped from a token's edges before it is classified or matched.
EDGE_PUNCT = ".,;:!?()[]{}<>\"'`*_~/\\|"


def _unify(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(_QUOTE_MAP.get(ch, ch) for ch in text)


def strip_edge_punct(token: str) -> str:
    """Strip leading and trailing punctuation, preserving the interior.

    ``"Studios."`` becomes ``"Studios"``; ``"A25-54"`` is unchanged, because the
    hyphen is interior.
    """
    return _unify(token).strip().strip(EDGE_PUNCT).strip()


def normalize_name(value: str) -> str:
    """Punctuation-strip, whitespace-collapse and case-fold a name or code.

    Every non-alphanumeric character becomes a space, so separators are
    interchangeable (``CA/NY/TX``, ``CA, NY, TX`` and ``CA NY TX`` normalize
    alike) while token identity is preserved (``New York`` stays two tokens and
    therefore never equals ``NY``).
    """
    text = _unify(value).casefold()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_tokens(value: str) -> list[str]:
    """Normalized token list for a name or code."""
    normalized = normalize_name(value)
    return normalized.split() if normalized else []


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_MULTIPLIERS = {"k": Decimal(1000), "m": Decimal(10) ** 6, "mm": Decimal(10) ** 6,
                "b": Decimal(10) ** 9, "bn": Decimal(10) ** 9}

_NUM_BODY = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"

_NUM_FULL_RE = re.compile(
    r"""^\s*
        (?:(?:\$|us\$|usd|eur|€)\s*)?      # leading currency
        (?P<sign>-)?
        (?P<body>""" + _NUM_BODY + r""")
        \s*(?P<mult>k|mm|m|bn|b)?
        \s*(?:%|usd|dollars?|impressions?)?  # trailing unit
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_NUM_TEXT_RE = re.compile(
    r"""(?<![0-9A-Za-z.])
        (?P<cur>\$|us\$)?\s*
        (?P<sign>-)?
        (?P<body>""" + _NUM_BODY + r""")
        (?:\s*(?P<mult>k|mm|m|bn|b)\b)?
        (?P<pct>\s*%)?
        (?![0-9])""",
    re.IGNORECASE | re.VERBOSE,
)


def canonical_number(value: str) -> Decimal | None:
    """Canonicalize a complete numeric string, or return ``None``.

    Currency symbols, thousands separators, trailing units and ``k``/``m``/``b``
    multipliers are absorbed. The whole string must be consumed, so ``"IO-42"``
    and ``"cancelled"`` are not numbers.
    """
    if not value:
        return None
    match = _NUM_FULL_RE.match(_unify(value))
    if not match:
        return None
    return _to_decimal(match.group("body"), match.group("sign"), match.group("mult"))


def _to_decimal(body: str, sign: str | None, mult: str | None) -> Decimal | None:
    try:
        number = Decimal(body.replace(",", ""))
    except InvalidOperation:
        return None
    if mult:
        number *= _MULTIPLIERS[mult.lower()]
    if sign:
        number = -number
    return number


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_RE = re.compile(r"(?<![\d/-])(\d{4})-(\d{2})-(\d{2})(?![\d-])")
_SLASH_RE = re.compile(r"(?<![\d/-])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])")
_MDY_RE = re.compile(
    r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b",
    re.IGNORECASE)
_DMY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\.?,?\s+(\d{4})\b",
    re.IGNORECASE)
_MY_RE = re.compile(r"\b(" + _MONTH_ALT + r")\.?\s+(\d{4})\b", re.IGNORECASE)

_DATE_PATTERNS = (
    (_ISO_RE, lambda m: _mkdate(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
    (_MDY_RE, lambda m: _mkdate(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))),
    (_DMY_RE, lambda m: _mkdate(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))),
    (_SLASH_RE, lambda m: _mkdate(_expand_year(m.group(3)), int(m.group(1)), int(m.group(2)))),
)


def _mkdate(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _expand_year(raw: str) -> int:
    return int(raw) if len(raw) == 4 else 2000 + int(raw)


def canonical_date(value: str) -> date | None:
    """Canonicalize a complete date string in any supported format."""
    if not value:
        return None
    text = _unify(value).strip().strip(EDGE_PUNCT).strip()
    for pattern, build in _DATE_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            result = build(match)
            if result is not None:
                return result
    return None


def canonical_partial_date(value: str) -> tuple[int, int] | None:
    """Canonicalize a month-and-year string such as ``April 2025``."""
    if not value:
        return None
    text = _unify(value).strip().strip(EDGE_PUNCT).strip()
    match = _MY_RE.fullmatch(text)
    if not match:
        return None
    return int(match.group(2)), _MONTHS[match.group(1).lower()]


# --------------------------------------------------------------------------
# Value classification
# --------------------------------------------------------------------------

class ValueKind:
    NUMBER = "number"
    DATE = "date"
    NAME = "name"


def classify_value(value: str) -> str:
    """Classify a fact value or gold token as a date, a number or a name/code."""
    if canonical_date(value) is not None:
        return ValueKind.DATE
    if canonical_number(value) is not None:
        return ValueKind.NUMBER
    return ValueKind.NAME


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

#: Words that may precede a period without ending a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "corp", "co",
    "ltd", "llc", "lp", "dept", "est", "approx", "no", "vs", "etc", "eg", "ie",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "am", "pm", "ext", "attn", "fig", "cf", "al",
}

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?;])(\s+)(?=["\'(\[]?[A-Z0-9])')


@dataclass(frozen=True)
class Segment:
    text: str
    offset: int


def sentence_segments(text: str) -> list[Segment]:
    """Split text into segments that a name phrase may not cross.

    Newlines always separate segments. Within a line, sentence-final
    punctuation separates segments unless the preceding word is a known
    abbreviation or a single capital letter (an initial).
    """
    unified = _unify(text)
    segments: list[Segment] = []
    line_start = 0
    for raw_line in unified.split("\n"):
        for segment in _split_line(raw_line, line_start):
            if segment.text.strip():
                segments.append(segment)
        line_start += len(raw_line) + 1
    return segments


def _split_line(line: str, base: int) -> list[Segment]:
    pieces: list[Segment] = []
    start = 0
    for match in _SENT_SPLIT_RE.finditer(line):
        cut = match.start()
        preceding = line[start:cut]
        if _is_abbreviation_boundary(preceding):
            continue
        pieces.append(Segment(preceding, base + start))
        start = match.end()
    pieces.append(Segment(line[start:], base + start))
    return pieces


def _is_abbreviation_boundary(preceding: str) -> bool:
    if not preceding.endswith("."):
        return False
    word_match = re.search(r"([A-Za-z]+)\.$", preceding)
    if not word_match:
        return False
    word = word_match.group(1)
    return len(word) == 1 or word.lower() in _ABBREVIATIONS


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

_CODE_TOKEN_RE = re.compile(r"(?<![\w-])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
                            r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![\w-])")

#: Runs made only of these words are never reported as names. Sentence-initial
#: capitalization makes almost any word look like a proper noun, so the
#: precision side needs an explicit list of words that are not entity names.
#: This list is a documented limitation, not a semantic model: a real company
#: literally named "Thanks" would be missed.
COMMON_WORDS = frozenset("""
a about above absolutely across add added adding additional adjust adjusted
adjustment after again against ago agree agreed ahead all already also always
am an and another any anything approval approve approved approx are around as
ask asked asking at attached available back based be because been before began
being below best better between big billed billing book booked booking both
brief bring budget but buy buying by call called came can cancel cancelled
cancelling cannot capacity care carry case catch caught change changed
changes changing channel check checked checking clear client close closed
come comes coming confirm confirmed confirming confirmation copy correct
corrected could counting course cover covered creative current currently cut
date dated dates day days deal deals dear decided deck delivery detail details
did different discuss discussed do does doing done down draft drop dropped due
during each earlier early either else end ended ending enough entire even
ever every everything exact exactly expect expected extend extended extra eye
face fact fair fall far feel few figure file final finally find fine first fit
five fix fixed flag flagged flat flight flights folks follow following for
form forward found four free from front full further gave get gets getting
give given go goes going gone good got great group had half hand happen
happened happy hard has have having he head hear heard held hello help her
here hey hi high him his hit hold holding home hope hoping how however i
if impression impressions in included including inside instead into invoice
invoiced is issue it item its just keep keeping kept key kind know known
last late later latest lead learn least leave left less let letting level
like line lines list little live locked long look looked looking looks lot
made mail main make making many market may maybe me mean means meant media
meet meeting mention mentioned message might mind mine minute miss missing
moment money month months more morning most move moved moving much must my
name need needed needs never new next nice no none nor not note noted nothing
now number numbers of off offer office often oh ok okay old on once one only
onto open opened opening or order orders other others our out over own pace
pacing page paid part pass past pause paused pay payment people per percent
period person phone pick place plan planned planning please point post
pretty price pricing prior probably problem process program pull push put
question questions quick quickly quote rate rates read ready real really
reason receive received recent recorded reference remaining remember remove
removed report request requested reserve reserved rest result results return
review reviewed revised revision right run running said same saw say saying
schedule scheduled second see seeing seem seen send sending sent series
service set setting several share shared she shift shifted short should show
side sign signed since single sir sit six slot small so social some something
soon sorry sound speak spend spent split spot spots start started starting
state status step still stop stopped straight sub such sudden sum sure switch
subject switched sync system take taken taking talk talked team tell ten term
terms
than thank thanks that the their them then there these they thing things
think third this those though three through thursday thus time times to today
together told tomorrow too took top total touch track trade traffic
trafficked true try trying turn turned two under understand unit units unless
until up update updated upon us use used using very via view volume wait
waiting walk want wanted was watch way we week weeks well went were what
when where whether which while who whole why will win window wish with
within without word work worked working works would wrap write written wrong
yes yesterday yet you your yours
""".split())

#: Ordinary business-email words that turn up sentence-initially. Kept separate
#: from COMMON_WORDS so the reason each block exists stays visible: this block
#: grew from real precision-side false positives ("Reach them at ..." reported
#: "Reach" as an unsupported company name).
CORRESPONDENCE_WORDS = frozenset("""
afternoon apologies appreciate attaching cheers circling confirmations
correction corrections details evening everybody everyone flagging follows
greetings heads hello howdy including logging looping meanwhile morning
neither noting otherwise overall passing quickly reach reaching recap
recapping regarding regards reminder revisions sending sharing shortly
sounds summary thanking tomorrow understood updates weekend welcome
""".split())

#: Signature terms of the domain. These are vocabulary, not entity names, so a
#: run consisting only of these is not reported as unsupported.
DOMAIN_WORDS = frozenset("""
ad ads adserver advertiser advertisers agency campaign cpm cpms flight
insertion io ios ops sales targeting impressions makegood makegoods
""".split())

_SAFE_WORDS = (COMMON_WORDS | CORRESPONDENCE_WORDS | DOMAIN_WORDS
               | {"of", "and", "de", "la"})


def _split_on_safe_words(tokens: list[str]) -> list[list[str]]:
    pieces: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SAFE_WORDS:
            if current:
                pieces.append(current)
                current = []
        else:
            current.append(token)
    if current:
        pieces.append(current)
    return pieces


@dataclass
class Extraction:
    numbers: list[tuple[str, Decimal]] = field(default_factory=list)
    dates: list[tuple[str, date]] = field(default_factory=list)
    partial_dates: list[tuple[str, tuple[int, int]]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = " "
    return "".join(chars)


def _extract_dates_segment(segment: str) -> tuple[list[tuple[str, date]],
                                                  list[tuple[str, tuple[int, int]]],
                                                  list[tuple[int, int]]]:
    found: list[tuple[str, date]] = []
    spans: list[tuple[int, int]] = []
    for pattern, build in _DATE_PATTERNS:
        for match in pattern.finditer(segment):
            if any(start <= match.start() < end for start, end in spans):
                continue
            value = build(match)
            if value is None:
                continue
            found.append((match.group(0), value))
            spans.append((match.start(), match.end()))
    partial: list[tuple[str, tuple[int, int]]] = []
    for match in _MY_RE.finditer(segment):
        if any(start < match.end() and match.start() < end for start, end in spans):
            continue
        partial.append((match.group(0), (int(match.group(2)),
                                         _MONTHS[match.group(1).lower()])))
        spans.append((match.start(), match.end()))
    return found, partial, spans


_POSSESSIVE_RE = re.compile(r"'s?$")


def bare_token(token: str) -> str:
    """A token stripped of edge punctuation and of a possessive suffix.

    ``Northwind's`` yields ``Northwind``. Without this, the possessive ``s``
    survives normalization as a standalone one-letter token and every
    possessive in the corpus reads as an unsupported name.
    """
    return _POSSESSIVE_RE.sub("", strip_edge_punct(token))


def _is_nameish(token: str) -> bool:
    bare = bare_token(token)
    if not bare:
        return False
    if _CODE_TOKEN_RE.fullmatch(bare):
        return True
    if bare.isupper() and bare.isalpha() and len(bare) >= 2:
        return True
    return bool(re.match(r"^[A-Z][A-Za-z'&-]*$", bare))


_CONNECTORS = {"of", "and", "&", "the", "for", "de", "la"}


def _extract_names_segment(segment: str) -> list[str]:
    tokens = segment.split()
    names: list[str] = []
    run: list[str] = []

    def flush() -> None:
        while run and bare_token(run[-1]).lower() in _CONNECTORS:
            run.pop()
        # Sentence-initial capitalization makes ordinary words look like the
        # first token of a name ("Line IO-42-0071"). Leading words that are not
        # entity names are dropped; trailing ones are kept, because company
        # names legitimately end in ordinary words ("Contoso Media").
        while run and bare_token(run[0]).casefold() in _SAFE_WORDS:
            run.pop(0)
        if run:
            phrase = " ".join(bare_token(token) for token in run).strip()
            if phrase:
                names.append(phrase)
        run.clear()

    for token in tokens:
        bare = bare_token(token)
        if _is_nameish(token):
            run.append(token)
        elif run and bare.lower() in _CONNECTORS:
            run.append(token)
        else:
            flush()
        # Sentence-final punctuation inside a segment (an abbreviation boundary
        # that survived splitting) still terminates a name phrase.
        if token.rstrip(")\"'").endswith((".", ",", ";", ":", "!", "?")) and run:
            flush()
    flush()
    return names


def extract_all(text: str) -> Extraction:
    """Extract numbers, dates and name phrases, segment by segment.

    Extraction order matters: dates are consumed first so that ``2025-04-18``
    does not also yield the numbers 2025, 4 and 18; then code-like tokens, so
    that ``IO-42-0071`` and ``A25-54`` do not yield 42, 71, 25 and 54; numbers
    come last.
    """
    result = Extraction()
    for segment in sentence_segments(text):
        body = segment.text
        dates, partials, date_spans = _extract_dates_segment(body)
        result.dates.extend(dates)
        result.partial_dates.extend(partials)
        without_dates = _mask(body, date_spans)

        code_spans = [(m.start(), m.end()) for m in _CODE_TOKEN_RE.finditer(without_dates)]
        result.names.extend(_extract_names_segment(without_dates))

        without_codes = _mask(without_dates, code_spans)
        for match in _NUM_TEXT_RE.finditer(without_codes):
            number = _to_decimal(match.group("body"), match.group("sign"),
                                 match.group("mult"))
            if number is not None:
                result.numbers.append((match.group(0).strip(), number))
    return result


def extract_numbers(text: str) -> list[tuple[str, Decimal]]:
    return extract_all(text).numbers


def extract_dates(text: str) -> list[tuple[str, date]]:
    return extract_all(text).dates


def extract_name_phrases(text: str) -> list[str]:
    return extract_all(text).names


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def values_match(left: str, right: str) -> bool:
    """Compare two values under the matching policy."""
    left_date, right_date = canonical_date(left), canonical_date(right)
    if left_date is not None and right_date is not None:
        return left_date == right_date
    left_num, right_num = canonical_number(left), canonical_number(right)
    if left_num is not None and right_num is not None:
        return left_num == right_num
    if (left_date is None) != (right_date is None):
        return False
    if (left_num is None) != (right_num is None):
        return False
    return normalize_name(left) == normalize_name(right)


class TextIndex:
    """A text prepared once for repeated containment queries.

    Segmentation and extraction are the expensive part of a containment check.
    The answerability audit asks hundreds of questions against one 80 KB corpus,
    so the corpus is indexed once. :func:`value_in_text` is a thin wrapper over
    this class rather than a second implementation -- there is one containment
    code path, and it is this one.
    """

    def __init__(self, text: str) -> None:
        self._segments = [name_tokens(segment.text)
                          for segment in sentence_segments(str(text))]
        extraction = extract_all(str(text))
        self._numbers = {number for _, number in extraction.numbers}
        self._dates = {found for _, found in extraction.dates}

    def contains(self, value: str) -> bool:
        if value is None:
            return False
        value = str(value)
        kind = classify_value(value)
        if kind == ValueKind.DATE:
            return canonical_date(value) in self._dates
        if kind == ValueKind.NUMBER:
            target = canonical_number(value)
            return any(target == number for number in self._numbers)
        wanted = name_tokens(value)
        if not wanted:
            return False
        width = len(wanted)
        for tokens in self._segments:
            for index in range(len(tokens) - width + 1):
                if tokens[index:index + width] == wanted:
                    return True
        return False


def value_in_text(value: str, text: str) -> bool:
    """Is ``value`` present in ``text`` under the matching policy?

    Numbers and dates match any equivalent surface form. Names and codes match
    as a contiguous normalized token run that does not cross a sentence
    boundary or a newline.
    """
    if value is None or text is None:
        return False
    return TextIndex(text).contains(value)


def token_in_text(token: str, text: str) -> bool:
    """Judge containment: is ``token`` present in ``text`` with word boundaries?

    This is the judge's only text operation. It never extracts entities from an
    answer; it asks whether a known gold token appears in it. Word boundaries
    come from normalized tokenization, which is why ``active`` is not contained
    in ``inactive``.
    """
    return value_in_text(token, text)


def any_value_in_text(values, text: str) -> bool:
    return any(value_in_text(value, text) for value in values)


# --------------------------------------------------------------------------
# Unsupported-content detection (fidelity side 2)
# --------------------------------------------------------------------------

@dataclass
class Support:
    """The set of values a rendered scenario is allowed to contain."""

    numbers: set[Decimal] = field(default_factory=set)
    dates: set[date] = field(default_factory=set)
    name_phrases: list[list[str]] = field(default_factory=list)

    def add_value(self, value: str) -> None:
        value = str(value)
        kind = classify_value(value)
        if kind == ValueKind.DATE:
            self.dates.add(canonical_date(value))
        elif kind == ValueKind.NUMBER:
            self.numbers.add(canonical_number(value))
        self.add_phrase(value)

    def add_phrase(self, phrase: str) -> None:
        """Register a free-text phrase (a subject line, a note body) as support."""
        for sub in extract_all(str(phrase)).names:
            tokens = name_tokens(sub)
            if tokens:
                self.name_phrases.append(tokens)
        tokens = name_tokens(str(phrase))
        if tokens:
            self.name_phrases.append(tokens)

    def add_text(self, text: str) -> None:
        """Register every number, date and name in ``text`` as support."""
        extraction = extract_all(str(text))
        self.numbers.update(number for _, number in extraction.numbers)
        self.dates.update(found for _, found in extraction.dates)
        self.add_phrase(text)

    def supports_name(self, phrase: str) -> bool:
        wanted = name_tokens(phrase)
        if not wanted:
            return True
        if all(token in _SAFE_WORDS for token in wanted):
            return True
        if self._contains(wanted):
            return True
        # A run may have swept two supported names into one phrase across a
        # conjunction ("Northwind and Fabrikam Studios"). Split on words that
        # are not entity names and require every remaining piece to be
        # supported, so a genuinely invented name is still reported.
        pieces = _split_on_safe_words(wanted)
        if not pieces or (len(pieces) == 1 and pieces[0] == wanted):
            return False
        return all(self._contains(piece) for piece in pieces)

    def _contains(self, wanted: list[str]) -> bool:
        width = len(wanted)
        for supported in self.name_phrases:
            for index in range(len(supported) - width + 1):
                if supported[index:index + width] == wanted:
                    return True
        return False

    def supports_number(self, number: Decimal) -> bool:
        return any(number == known for known in self.numbers)

    def supports_date(self, value: date) -> bool:
        return value in self.dates

    def supports_partial_date(self, year_month: tuple[int, int]) -> bool:
        year, month = year_month
        return any(known.year == year and known.month == month for known in self.dates)


def build_support(values, texts=()) -> Support:
    """Build a :class:`Support` set from fact values plus free-text context."""
    support = Support()
    for value in values:
        support.add_value(value)
    for text in texts:
        support.add_text(text)
    return support


def unsupported_items(text: str, support: Support) -> dict[str, list[str]]:
    """Every number, date and name in ``text`` that ``support`` cannot explain.

    This is the precision side of fidelity: a renderer that invents a discount
    percentage, a signing date or a second agency is caught here even though
    every planted fact is present.
    """
    extraction = extract_all(text)
    findings: dict[str, list[str]] = {"numbers": [], "dates": [], "names": []}
    for surface, number in extraction.numbers:
        if not support.supports_number(number):
            findings["numbers"].append(surface)
    for surface, value in extraction.dates:
        if not support.supports_date(value):
            findings["dates"].append(surface)
    for surface, year_month in extraction.partial_dates:
        if not support.supports_partial_date(year_month):
            findings["dates"].append(surface)
    for phrase in extraction.names:
        if not support.supports_name(phrase):
            findings["names"].append(phrase)
    return {key: sorted(set(values)) for key, values in findings.items()}
