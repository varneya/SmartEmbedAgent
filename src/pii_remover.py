"""
PII Removal Module
==================

Strips personally identifiable information from a corpus using a layered
approach:

  1. Regex pass for high-confidence patterns (emails, phones, SSNs, credit
     cards, IPv4 addresses).
  2. Optional region packs (currently: 'india') add region-specific
     recognizers — Aadhaar (Verhoeff-validated), PAN, Indian mobile,
     vehicle registration. Enable via `pii_settings.region_packs: ["india"]`.
  3. NER pass using a pre-trained Hugging Face model (default
     `dslim/bert-base-NER`) to catch PERSON, LOCATION, and ORGANIZATION
     entities.
  4. User-supplied custom redaction list — force-redacted regardless of
     model confidence.
  5. User-supplied whitelist — strings that are exempt from redaction even
     if they match a regex or are flagged by NER.

Backends
--------
    legacy (default)  : the regex + HF-NER pipeline above; lightweight, no
                        extra deps beyond transformers (already required).
    presidio          : opt-in, uses Microsoft Presidio
                        (`pip install smart-embed-agent[presidio]`). Brings
                        50+ additional recognizers, validated detection
                        (Luhn for credit cards), and confidence scores.
                        Region packs continue to apply.

    Select via `pii_settings.recognizer = "legacy" | "presidio"`. If
    "presidio" is requested but the package is not installed, the module
    logs a warning and falls back to legacy so privacy is never silently
    weaker than configured.

Public API
----------
    remove_pii(corpus: str, config: dict) -> (cleaned_text, redaction_report)

The redaction report is a list of dicts with `category`, `original`, and
`replacement` fields, plus a `summary` count by category. Use it for audit
logs.

Replacement tokens
------------------
    REDACTED_EMAIL, REDACTED_PHONE, REDACTED_SSN, REDACTED_CC, REDACTED_IP,
    REDACTED_NAME, REDACTED_LOCATION, REDACTED_ORG, REDACTED_CUSTOM,
    REDACTED_AADHAAR, REDACTED_PAN, REDACTED_VEHICLE
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("pii_remover")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[pii_remover] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Regex pattern catalog
# Ordered so that more specific patterns (SSN before generic credit card,
# email before generic phone) are evaluated first when ranges overlap.
# ---------------------------------------------------------------------------
REGEX_PATTERNS: Dict[str, str] = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": (
        r"\b(?:\d[ -]*?){13,19}\b"  # 13–19 digits, optional spaces/dashes
    ),
    # North American + common international phone formats:
    #   555-123-4567, (555) 123-4567, +1 555 123 4567, 555.123.4567
    "PHONE": (
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

# Mapping from category -> redaction token. Adding a new category requires
# adding it both here and (if relevant) to REGEX_PATTERNS or NER_LABEL_MAP.
REDACTION_TOKENS: Dict[str, str] = {
    "EMAIL": "REDACTED_EMAIL",
    "PHONE": "REDACTED_PHONE",
    "SSN": "REDACTED_SSN",
    "CREDIT_CARD": "REDACTED_CC",
    "IP": "REDACTED_IP",
    "NAME": "REDACTED_NAME",
    "LOCATION": "REDACTED_LOCATION",
    "ORG": "REDACTED_ORG",
    "CUSTOM": "REDACTED_CUSTOM",
    # India region pack
    "AADHAAR": "REDACTED_AADHAAR",
    "PAN": "REDACTED_PAN",
    "INDIAN_MOBILE": "REDACTED_PHONE",       # reuse — same downstream signal
    "INDIAN_VEHICLE": "REDACTED_VEHICLE",
}


# ---------------------------------------------------------------------------
# Region packs — each pack contributes additional regex patterns + an
# optional per-category validator. Validators (e.g. Verhoeff for Aadhaar)
# filter out random N-digit sequences so we don't redact timestamps and
# order numbers as PII.
# ---------------------------------------------------------------------------
INDIA_REGEX_PATTERNS: Dict[str, str] = {
    # 12-digit Aadhaar with optional spaces/dashes between blocks of 4.
    "AADHAAR": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    # PAN: 5 letters + 4 digits + 1 letter.
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    # Indian mobile: optional +91 / 0 prefix, then 10 digits starting 6-9.
    "INDIAN_MOBILE": r"(?:\+?91[-\s]?|0)?[6-9]\d{9}\b",
    # Vehicle registration: STATE(2) + RTO(1-2) + SERIES(1-3) + 4 digits.
    "INDIAN_VEHICLE": r"\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{4}\b",
}

REGION_PACKS: Dict[str, Dict[str, str]] = {
    "india": INDIA_REGEX_PATTERNS,
}


# Verhoeff checksum tables — used to validate Aadhaar so random 12-digit
# numbers (timestamps, order IDs) don't get falsely redacted.
_VERHOEFF_D = (
    (0,1,2,3,4,5,6,7,8,9), (1,2,3,4,0,6,7,8,9,5),
    (2,3,4,0,1,7,8,9,5,6), (3,4,0,1,2,8,9,5,6,7),
    (4,0,1,2,3,9,5,6,7,8), (5,9,8,7,6,0,4,3,2,1),
    (6,5,9,8,7,1,0,4,3,2), (7,6,5,9,8,2,1,0,4,3),
    (8,7,6,5,9,3,2,1,0,4), (9,8,7,6,5,4,3,2,1,0),
)
_VERHOEFF_P = (
    (0,1,2,3,4,5,6,7,8,9), (1,5,7,6,2,8,3,0,9,4),
    (5,8,0,3,7,9,6,1,4,2), (8,9,1,6,0,4,3,5,2,7),
    (9,4,5,3,1,2,6,8,7,0), (4,2,8,6,5,7,3,9,0,1),
    (2,7,9,3,8,0,6,4,1,5), (7,0,4,6,9,1,3,2,5,8),
)


def _verhoeff_check(num: str) -> bool:
    """Verhoeff checksum validator. Aadhaar's 12th digit is a Verhoeff
    check digit; this filters ~90% of random 12-digit false positives."""
    digits = [int(d) for d in reversed(num) if d.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


# Per-category extra validators. Returning False here filters the match out.
EXTRA_VALIDATORS: Dict[str, Any] = {
    "AADHAAR": lambda m: _verhoeff_check(m),
}

# NER label mapping. dslim/bert-base-NER uses PER/LOC/ORG/MISC; we normalize.
NER_LABEL_MAP: Dict[str, str] = {
    "PER": "NAME",
    "PERSON": "NAME",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    # MISC intentionally left out — too noisy to redact by default.
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RedactionEvent:
    category: str       # EMAIL, PHONE, NAME, ...
    original: str       # the text that was redacted
    replacement: str    # the token it was replaced with
    start: int          # original-document start offset
    end: int            # original-document end offset
    source: str         # "regex", "ner", "custom"


@dataclass
class RedactionReport:
    redactions: List[RedactionEvent] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.redactions:
            counts[r.category] = counts.get(r.category, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "total": len(self.redactions),
            "events": [asdict(r) for r in self.redactions],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_whitelisted(text: str, whitelist: Iterable[str]) -> bool:
    """Case-insensitive membership check. A match is allowed if `text`
    equals any whitelist entry (after stripping)."""
    normalized = text.strip().lower()
    return any(normalized == w.strip().lower() for w in whitelist)


def _resolve_overlapping_spans(
    spans: List[Tuple[int, int, str, str, str]],
) -> List[Tuple[int, int, str, str, str]]:
    """Given (start, end, category, original, source) tuples, return a
    non-overlapping subset, preferring earlier-listed (more specific)
    categories and longer spans on ties."""
    # Specificity order: explicit categories beat broader ones. Region-pack
    # entries (AADHAAR, PAN, INDIAN_*) rank above the generic equivalents
    # they could overlap with (CREDIT_CARD, PHONE) so a 12-digit Aadhaar
    # isn't redacted as a 12-digit "credit card".
    specificity = {
        "CUSTOM": 0,
        "AADHAAR": 1,
        "PAN": 1,
        "INDIAN_MOBILE": 1,
        "INDIAN_VEHICLE": 1,
        "SSN": 2,
        "EMAIL": 3,
        "CREDIT_CARD": 4,
        "IP": 5,
        "PHONE": 6,
        "NAME": 7,
        "LOCATION": 8,
        "ORG": 9,
    }
    spans = sorted(
        spans,
        key=lambda s: (
            s[0],                          # earliest start first
            specificity.get(s[2], 99),     # then most specific
            -(s[1] - s[0]),                # then longest
        ),
    )
    accepted: List[Tuple[int, int, str, str, str]] = []
    last_end = -1
    for span in spans:
        if span[0] >= last_end:
            accepted.append(span)
            last_end = span[1]
    return accepted


def _apply_replacements(
    text: str,
    spans: List[Tuple[int, int, str, str, str]],
) -> Tuple[str, List[RedactionEvent]]:
    """Apply replacements right-to-left so indices don't shift. Returns
    (new_text, redaction_events) with events ordered left-to-right."""
    events: List[RedactionEvent] = []
    spans_desc = sorted(spans, key=lambda s: s[0], reverse=True)
    out = text
    for start, end, category, original, source in spans_desc:
        replacement = REDACTION_TOKENS.get(category, f"REDACTED_{category}")
        out = out[:start] + replacement + out[end:]
        events.append(
            RedactionEvent(
                category=category,
                original=original,
                replacement=replacement,
                start=start,
                end=end,
                source=source,
            )
        )
    events.reverse()
    return out, events


# ---------------------------------------------------------------------------
# Stage 1 — regex (core + region packs)
# ---------------------------------------------------------------------------
def _regex_spans(
    text: str,
    whitelist: Iterable[str],
    region_packs: Optional[Iterable[str]] = None,
) -> List[Tuple[int, int, str, str, str]]:
    spans: List[Tuple[int, int, str, str, str]] = []

    # Build the combined catalog: core patterns + any opt-in region packs.
    catalog: Dict[str, str] = dict(REGEX_PATTERNS)
    for pack_name in (region_packs or []):
        pack = REGION_PACKS.get(pack_name)
        if pack:
            catalog.update(pack)
        else:
            logger.warning("Unknown region pack '%s'; ignoring.", pack_name)

    for category, pattern in catalog.items():
        for m in re.finditer(pattern, text):
            matched = m.group(0)
            # Filter low-quality phone matches: must contain at least 7 digits.
            if category == "PHONE" and sum(c.isdigit() for c in matched) < 7:
                continue
            # Filter credit card matches that are clearly not 13–19 digits
            # once dashes/spaces are removed.
            if category == "CREDIT_CARD":
                digits_only = re.sub(r"[ -]", "", matched)
                if not (13 <= len(digits_only) <= 19):
                    continue
            # Per-category validators (e.g. Verhoeff for AADHAAR).
            validator = EXTRA_VALIDATORS.get(category)
            if validator is not None and not validator(matched):
                continue
            if _is_whitelisted(matched, whitelist):
                continue
            spans.append((m.start(), m.end(), category, matched, "regex"))
    return spans


# ---------------------------------------------------------------------------
# Stage 2 — NER
# ---------------------------------------------------------------------------
class _NERWrapper:
    """Lazily-loaded Hugging Face NER pipeline. Kept as a class so the model
    is loaded at most once per process even if `remove_pii` is called many
    times."""

    _instance: Optional["_NERWrapper"] = None

    def __init__(self, model_name: str) -> None:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    @classmethod
    def get(cls, model_name: str) -> "_NERWrapper":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance


def _ner_spans(
    text: str,
    whitelist: Iterable[str],
    model_name: str,
) -> List[Tuple[int, int, str, str, str]]:
    """Run the NER model on `text` and return whitelist-filtered spans.
    Returns [] if the model can't be loaded — regex coverage is preserved."""
    if not text.strip():
        return []
    try:
        wrapper = _NERWrapper.get(model_name)
    except Exception as e:
        logger.warning("NER model unavailable (%s); skipping NER stage.", e)
        return []

    try:
        entities = wrapper.pipeline(text)
    except Exception as e:
        logger.warning("NER inference failed (%s); skipping NER stage.", e)
        return []

    spans: List[Tuple[int, int, str, str, str]] = []
    for ent in entities:
        raw_label = ent.get("entity_group") or ent.get("entity") or ""
        category = NER_LABEL_MAP.get(raw_label.upper())
        if category is None:
            continue
        original = ent["word"]
        if _is_whitelisted(original, whitelist):
            continue
        spans.append((int(ent["start"]), int(ent["end"]), category, original, "ner"))
    return spans


# ---------------------------------------------------------------------------
# Alternative backend — Microsoft Presidio (opt-in)
# ---------------------------------------------------------------------------
# Maps Presidio's entity-type vocabulary onto our internal categories so the
# downstream report shape and replacement tokens stay consistent across
# backends. Anything not in this map keeps its raw entity_type as the
# category (and gets a generic REDACTED_<type> token).
_PRESIDIO_ENTITY_TO_CATEGORY: Dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IP_ADDRESS": "IP",
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORG",
    "NRP": "ORG",  # Nationality / religious / political group → coarse ORG
    # India region-pack names propagate as-is.
    "AADHAAR": "AADHAAR",
    "PAN": "PAN",
    "INDIAN_MOBILE": "INDIAN_MOBILE",
    "INDIAN_VEHICLE": "INDIAN_VEHICLE",
}


class _PresidioWrapper:
    """Lazily-loaded Presidio AnalyzerEngine. Cached so the first analyze()
    pays the spaCy model load cost only once per process."""

    _instance: Optional["_PresidioWrapper"] = None

    def __init__(self, region_packs: Iterable[str]) -> None:
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern

        self.analyzer = AnalyzerEngine()
        self.region_packs = sorted(set(region_packs))

        # Register custom recognizers for each opt-in region pack.
        for pack in self.region_packs:
            for entity_type, pattern in REGION_PACKS.get(pack, {}).items():
                self.analyzer.registry.add_recognizer(
                    PatternRecognizer(
                        supported_entity=entity_type,
                        patterns=[Pattern(name=entity_type, regex=pattern, score=0.85)],
                    )
                )

    @classmethod
    def get(cls, region_packs: Iterable[str]) -> "_PresidioWrapper":
        key = tuple(sorted(set(region_packs or [])))
        if cls._instance is None or tuple(cls._instance.region_packs) != key:
            cls._instance = cls(region_packs or [])
        return cls._instance


def _presidio_spans(
    text: str,
    whitelist: Iterable[str],
    region_packs: Optional[Iterable[str]] = None,
    score_threshold: float = 0.4,
) -> Optional[List[Tuple[int, int, str, str, str]]]:
    """Run Presidio analyzer on `text`. Returns the spans or None if Presidio
    isn't installed (caller falls back to the legacy regex+NER pipeline)."""
    try:
        wrapper = _PresidioWrapper.get(region_packs or [])
    except ImportError:
        logger.warning(
            "presidio not installed; install with `pip install smart-embed-agent[presidio]` "
            "or set pii_settings.recognizer to 'legacy'. Falling back to legacy backend."
        )
        return None
    except Exception as e:
        logger.warning("Presidio init failed (%s); falling back to legacy backend.", e)
        return None

    try:
        results = wrapper.analyzer.analyze(text=text, language="en", score_threshold=score_threshold)
    except Exception as e:
        logger.warning("Presidio analysis failed (%s); falling back to legacy backend.", e)
        return None

    spans: List[Tuple[int, int, str, str, str]] = []
    for r in results:
        original = text[r.start:r.end]
        if _is_whitelisted(original, whitelist):
            continue
        # Apply the same per-category validator (e.g. Verhoeff for Aadhaar).
        category = _PRESIDIO_ENTITY_TO_CATEGORY.get(r.entity_type, r.entity_type)
        validator = EXTRA_VALIDATORS.get(category)
        if validator is not None and not validator(original):
            continue
        spans.append((r.start, r.end, category, original, "presidio"))
    return spans


# ---------------------------------------------------------------------------
# Stage 3 — custom redaction list (force, regardless of detection)
# ---------------------------------------------------------------------------
def _custom_spans(
    text: str,
    redaction_list: Iterable[str],
    whitelist: Iterable[str],
) -> List[Tuple[int, int, str, str, str]]:
    """Custom redaction list always wins. Whitelist still applies — if the
    user contradicts themselves, whitelist takes precedence so they don't
    accidentally redact preserved terms."""
    spans: List[Tuple[int, int, str, str, str]] = []
    for item in redaction_list:
        if not item:
            continue
        if _is_whitelisted(item, whitelist):
            continue
        for m in re.finditer(re.escape(item), text):
            spans.append((m.start(), m.end(), "CUSTOM", m.group(0), "custom"))
    return spans


# ---------------------------------------------------------------------------
# Edge case: consistent name redaction across occurrences
# ---------------------------------------------------------------------------
def _carry_forward_names(
    text: str,
    name_spans: List[Tuple[int, int, str, str, str]],
    whitelist: Iterable[str],
) -> List[Tuple[int, int, str, str, str]]:
    """If NER caught "Marcus Lee" once but missed a later occurrence, find
    every additional occurrence and redact them with the same NAME token.
    This makes downstream embeddings consistent — the same person never
    appears under two different masks."""
    extras: List[Tuple[int, int, str, str, str]] = []
    seen_names = {orig for _, _, cat, orig, _ in name_spans if cat == "NAME"}
    existing_ranges = [(s, e) for s, e, _, _, _ in name_spans]

    def overlaps_existing(start: int, end: int) -> bool:
        return any(start < ee and end > ss for ss, ee in existing_ranges)

    for name in seen_names:
        if _is_whitelisted(name, whitelist):
            continue
        # Word-boundary search for the canonical name string.
        for m in re.finditer(rf"\b{re.escape(name)}\b", text):
            if not overlaps_existing(m.start(), m.end()):
                extras.append((m.start(), m.end(), "NAME", m.group(0), "ner_carryforward"))
    return extras


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def remove_pii(
    corpus: str,
    config: Optional[Dict[str, Any]] = None,
    use_ner: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Strip PII from `corpus` using regex + NER + user rules (or Presidio if
    selected via `config["recognizer"] = "presidio"`).

    Parameters
    ----------
    corpus : str
        The text to clean.
    config : dict
        User configuration. Recognized keys:
            - "whitelist": list[str]      — strings preserved despite detection
            - "redaction_list": list[str] — strings always redacted
            - "ner_model": str            — override default NER model
            - "recognizer": "legacy" | "presidio"
                  Default "legacy". When "presidio", runs Microsoft Presidio
                  in place of the legacy regex+NER stages. Falls back to
                  legacy with a logged warning if presidio isn't installed.
            - "region_packs": list[str]
                  Opt-in regional recognizer packs. Currently supported:
                  "india" (Aadhaar with Verhoeff validation, PAN, Indian
                  mobile, vehicle registration).
    use_ner : bool
        If False, skip the HF NER stage in the legacy backend. (Presidio
        backend already includes its own spaCy NER.)

    Returns
    -------
    (cleaned_text, redaction_report)
        `redaction_report` is a dict with `summary`, `total`, `events`, and
        `recognizer_used` ("legacy" | "presidio").
    """
    config = config or {}
    whitelist: List[str] = list(config.get("whitelist", []) or [])
    redaction_list: List[str] = list(config.get("redaction_list", []) or [])
    ner_model: str = config.get("ner_model", "dslim/bert-base-NER")
    requested_recognizer: str = (config.get("recognizer") or "legacy").lower()
    region_packs: List[str] = list(config.get("region_packs", []) or [])

    presidio_spans: Optional[List[Tuple[int, int, str, str, str]]] = None
    if requested_recognizer == "presidio":
        presidio_spans = _presidio_spans(corpus, whitelist, region_packs)

    if presidio_spans is not None:
        # Presidio replaces the regex+NER stages. Custom-list still applies.
        recognizer_used = "presidio"
        detection_spans = presidio_spans
    else:
        recognizer_used = "legacy"
        # Stage 1: regex (core + region packs)
        regex_spans = _regex_spans(corpus, whitelist, region_packs=region_packs)

        # Stage 2: HF NER
        ner_spans: List[Tuple[int, int, str, str, str]] = []
        if use_ner:
            ner_spans = _ner_spans(corpus, whitelist, ner_model)
            if ner_spans:
                ner_spans += _carry_forward_names(corpus, ner_spans, whitelist)

        detection_spans = regex_spans + ner_spans

    # Stage 3: custom redaction (force) — same for both backends.
    custom_spans = _custom_spans(corpus, redaction_list, whitelist)

    # Custom wins on overlap, so feed it first into the resolver.
    all_spans = custom_spans + detection_spans
    resolved = _resolve_overlapping_spans(all_spans)

    cleaned, events = _apply_replacements(corpus, resolved)
    report_dict = RedactionReport(redactions=events).to_dict()
    report_dict["recognizer_used"] = recognizer_used
    report_dict["region_packs"] = region_packs

    # Transparency log — emit one INFO line per redaction.
    for ev in events:
        logger.info("Redacted %s (%s) -> %s", ev.category, ev.source, ev.replacement)
    logger.info(
        "PII removal complete: %d redactions across %d categories (recognizer=%s, region_packs=%s).",
        len(events), len(report_dict["summary"]), recognizer_used, region_packs or "[]",
    )

    return cleaned, report_dict


# Backwards-compatible alias for code that imports `clean_corpus`.
def clean_corpus(corpus: str, config: Optional[Dict[str, Any]] = None, use_ner: bool = True, **_: Any):
    """Legacy wrapper. Returns an object with .cleaned_text and .summary()."""
    cleaned, report = remove_pii(corpus, config=config, use_ner=use_ner)

    class _LegacyResult:
        def __init__(self, text: str, report: Dict[str, Any]):
            self.cleaned_text = text
            self._report = report

        def summary(self) -> Dict[str, int]:
            return self._report["summary"]

    return _LegacyResult(cleaned, report)


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Contact John Smith at john.smith@example.com or 555-123-4567. "
        "John Smith works for Acme Corp in San Francisco. "
        "Visit https://example.com?email=alice@example.com for details. "
        "His SSN is 123-45-6789 and credit card 4111 1111 1111 1111. "
        "Server IP: 192.168.1.42. Internal codename: Project Falcon."
    )
    cfg = {
        "whitelist": ["Acme Corp"],
        "redaction_list": ["Project Falcon"],
    }
    cleaned, report = remove_pii(sample, cfg, use_ner=False)
    print("--- BEFORE ---")
    print(sample)
    print("\n--- AFTER ---")
    print(cleaned)
    print("\n--- REPORT ---")
    print(json.dumps(report, indent=2))
