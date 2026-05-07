"""
PII Removal Module
==================

Strips personally identifiable information from a corpus using a layered
approach:

  1. Regex pass for high-confidence patterns (emails, phones, SSNs, credit
     cards, IPv4 addresses).
  2. NER pass using a pre-trained Hugging Face model (default
     `dslim/bert-base-NER`) to catch PERSON, LOCATION, and ORGANIZATION
     entities.
  3. User-supplied custom redaction list — force-redacted regardless of
     model confidence.
  4. User-supplied whitelist — strings that are exempt from redaction even
     if they match a regex or are flagged by NER.

Public API
----------
    remove_pii(corpus: str, config: dict) -> (cleaned_text, redaction_report)

The redaction report is a list of dicts with `category`, `original`, and
`replacement` fields, plus a `summary` count by category. Use it for audit
logs.

Replacement tokens
------------------
    REDACTED_EMAIL, REDACTED_PHONE, REDACTED_SSN, REDACTED_CC, REDACTED_IP,
    REDACTED_NAME, REDACTED_LOCATION, REDACTED_ORG, REDACTED_CUSTOM
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
    # Specificity order: explicit categories beat broader ones.
    specificity = {
        "CUSTOM": 0,
        "SSN": 1,
        "EMAIL": 2,
        "CREDIT_CARD": 3,
        "IP": 4,
        "PHONE": 5,
        "NAME": 6,
        "LOCATION": 7,
        "ORG": 8,
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
# Stage 1 — regex
# ---------------------------------------------------------------------------
def _regex_spans(text: str, whitelist: Iterable[str]) -> List[Tuple[int, int, str, str, str]]:
    spans: List[Tuple[int, int, str, str, str]] = []
    for category, pattern in REGEX_PATTERNS.items():
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
    Strip PII from `corpus` using regex + NER + user rules.

    Parameters
    ----------
    corpus : str
        The text to clean.
    config : dict
        User configuration. Recognized keys:
            - "whitelist": list[str]      — strings preserved despite detection
            - "redaction_list": list[str] — strings always redacted
            - "ner_model": str            — override default NER model
    use_ner : bool
        If False, run regex + custom only. Useful in tests / fast paths.

    Returns
    -------
    (cleaned_text, redaction_report)
        `redaction_report` is a dict with `summary`, `total`, and `events`.
    """
    config = config or {}
    whitelist: List[str] = list(config.get("whitelist", []) or [])
    redaction_list: List[str] = list(config.get("redaction_list", []) or [])
    ner_model: str = config.get("ner_model", "dslim/bert-base-NER")

    # --- Stage 1: regex
    regex_spans = _regex_spans(corpus, whitelist)

    # --- Stage 2: NER
    ner_spans: List[Tuple[int, int, str, str, str]] = []
    if use_ner:
        ner_spans = _ner_spans(corpus, whitelist, ner_model)
        if ner_spans:
            ner_spans += _carry_forward_names(corpus, ner_spans, whitelist)

    # --- Stage 3: custom redaction (force)
    custom_spans = _custom_spans(corpus, redaction_list, whitelist)

    # Custom wins on overlap with regex/NER, so feed it first into the
    # resolver (specificity table already encodes this).
    all_spans = custom_spans + regex_spans + ner_spans
    resolved = _resolve_overlapping_spans(all_spans)

    cleaned, events = _apply_replacements(corpus, resolved)
    report = RedactionReport(redactions=events)

    # Transparency log — emit one INFO line per redaction.
    for ev in events:
        logger.info(
            "Redacted %s (%s) -> %s",
            ev.category,
            ev.source,
            ev.replacement,
        )
    logger.info(
        "PII removal complete: %d redactions across %d categories.",
        len(events),
        len(report.summary()),
    )

    return cleaned, report.to_dict()


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
