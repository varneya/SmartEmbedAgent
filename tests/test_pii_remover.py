"""
Unit tests for src/pii_remover.py.

The NER stage requires a network download and a few hundred MB of RAM, so
these tests run with `use_ner=False` by default. The regex stage, the custom
redaction list, the whitelist, and the report-shape contract are exercised
without the model.

Run with:
    python -m pytest tests/test_pii_remover.py
or:
    python tests/test_pii_remover.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pii_remover import REDACTION_TOKENS, remove_pii, _verhoeff_check


class TestRegexRedaction(unittest.TestCase):
    def test_redacts_email(self):
        cleaned, report = remove_pii("Email me at alice@example.com.", {}, use_ner=False)
        self.assertIn("REDACTED_EMAIL", cleaned)
        self.assertNotIn("alice@example.com", cleaned)
        self.assertEqual(report["summary"].get("EMAIL"), 1)

    def test_redacts_phone_multiple_formats(self):
        text = (
            "Call 555-123-4567 or (555) 987-6543 or +1 555.111.2222."
        )
        cleaned, report = remove_pii(text, {}, use_ner=False)
        # Three phones in three different formats.
        self.assertEqual(report["summary"].get("PHONE"), 3)
        self.assertNotIn("555-123-4567", cleaned)

    def test_redacts_ssn(self):
        cleaned, report = remove_pii("SSN: 123-45-6789.", {}, use_ner=False)
        self.assertIn("REDACTED_SSN", cleaned)
        self.assertNotIn("123-45-6789", cleaned)

    def test_redacts_credit_card(self):
        cleaned, report = remove_pii(
            "Card: 4111 1111 1111 1111 expires soon.",
            {},
            use_ner=False,
        )
        self.assertIn("REDACTED_CC", cleaned)

    def test_redacts_ipv4(self):
        cleaned, report = remove_pii("Server is at 192.168.1.42.", {}, use_ner=False)
        self.assertIn("REDACTED_IP", cleaned)
        self.assertNotIn("192.168.1.42", cleaned)


class TestWhitelist(unittest.TestCase):
    def test_whitelisted_email_is_preserved(self):
        config = {"whitelist": ["public@example.com"]}
        text = "Public: public@example.com; private: bob@example.com."
        cleaned, report = remove_pii(text, config, use_ner=False)
        self.assertIn("public@example.com", cleaned)
        self.assertNotIn("bob@example.com", cleaned)
        self.assertEqual(report["summary"].get("EMAIL"), 1)

    def test_whitelist_overrides_custom_redaction_list(self):
        # If the user lists the same string in both, whitelist wins.
        config = {
            "whitelist": ["Acme Corp"],
            "redaction_list": ["Acme Corp"],
        }
        cleaned, _ = remove_pii("We work with Acme Corp.", config, use_ner=False)
        self.assertIn("Acme Corp", cleaned)


class TestCustomRedactionList(unittest.TestCase):
    def test_custom_string_is_force_redacted(self):
        config = {"redaction_list": ["Project Falcon", "Codename Atlas"]}
        text = "Project Falcon is on track. Codename Atlas ships next quarter."
        cleaned, report = remove_pii(text, config, use_ner=False)
        self.assertNotIn("Project Falcon", cleaned)
        self.assertNotIn("Codename Atlas", cleaned)
        self.assertEqual(report["summary"].get("CUSTOM"), 2)
        for ev in report["events"]:
            if ev["category"] == "CUSTOM":
                self.assertEqual(ev["source"], "custom")


class TestEdgeCases(unittest.TestCase):
    def test_embedded_pii_in_url(self):
        # A URL with an embedded email — the email should be redacted.
        text = "See https://example.com/lookup?email=alice@example.com for details."
        cleaned, report = remove_pii(text, {}, use_ner=False)
        self.assertNotIn("alice@example.com", cleaned)
        self.assertEqual(report["summary"].get("EMAIL"), 1)

    def test_consistent_redaction_across_occurrences(self):
        # The same email appearing multiple times gets redacted to the same
        # token every time — important for downstream consistency.
        text = "alice@example.com sent to bob, then alice@example.com again."
        cleaned, report = remove_pii(text, {}, use_ner=False)
        self.assertEqual(cleaned.count("REDACTED_EMAIL"), 2)
        self.assertNotIn("alice@example.com", cleaned)

    def test_report_has_required_fields(self):
        _, report = remove_pii("alice@example.com", {}, use_ner=False)
        self.assertIn("summary", report)
        self.assertIn("total", report)
        self.assertIn("events", report)
        self.assertGreater(report["total"], 0)
        ev = report["events"][0]
        for key in ("category", "original", "replacement", "start", "end", "source"):
            self.assertIn(key, ev)


class TestTokenContract(unittest.TestCase):
    def test_all_categories_have_redaction_tokens(self):
        for category, token in REDACTION_TOKENS.items():
            self.assertTrue(token.startswith("REDACTED_"), token)


class TestIndianRegionPack(unittest.TestCase):
    """Region pack 'india' adds Aadhaar, PAN, Indian mobile, vehicle reg.
    None of these are detected when the pack is omitted."""

    def test_aadhaar_redacted_when_pack_enabled(self):
        # Synthetic Aadhaar with a valid Verhoeff check digit (last digit).
        text = "Aadhaar: 4567 1234 5679"
        self.assertTrue(_verhoeff_check("4567 1234 5679"))
        cleaned, report = remove_pii(text, {"region_packs": ["india"]}, use_ner=False)
        self.assertIn("REDACTED_AADHAAR", cleaned)
        self.assertNotIn("4567", cleaned)
        self.assertEqual(report["summary"].get("AADHAAR"), 1)

    def test_aadhaar_NOT_detected_when_pack_disabled(self):
        text = "Aadhaar: 4567 1234 5679"
        cleaned, report = remove_pii(text, {}, use_ner=False)
        self.assertNotIn("REDACTED_AADHAAR", cleaned)
        self.assertNotIn("AADHAAR", report["summary"])

    def test_verhoeff_filters_random_12_digit(self):
        # Looks like an Aadhaar but the check digit fails Verhoeff.
        # 1234 5678 9012 has check digit 2 which is wrong.
        text = "Order ID 1234 5678 9012 was shipped."
        cleaned, report = remove_pii(text, {"region_packs": ["india"]}, use_ner=False)
        self.assertNotIn("REDACTED_AADHAAR", cleaned, "Verhoeff should reject random numbers")

    def test_pan_redacted(self):
        text = "PAN: ABCDE1234F"
        cleaned, report = remove_pii(text, {"region_packs": ["india"]}, use_ner=False)
        self.assertIn("REDACTED_PAN", cleaned)
        self.assertEqual(report["summary"].get("PAN"), 1)

    def test_indian_mobile_redacted(self):
        for number in ("+91 9876543210", "919876543210", "9876543210", "09876543210"):
            text = f"Call me at {number}"
            cleaned, report = remove_pii(text, {"region_packs": ["india"]}, use_ner=False)
            # INDIAN_MOBILE wins over generic PHONE thanks to specificity ordering.
            self.assertIn("REDACTED_PHONE", cleaned, f"failed for {number!r}")
            # Either categorized as INDIAN_MOBILE or PHONE — both ok.
            cats = report["summary"]
            self.assertTrue(cats.get("INDIAN_MOBILE", 0) + cats.get("PHONE", 0) >= 1, f"failed for {number!r}")

    def test_vehicle_registration_redacted(self):
        text = "Bike registration MH-12-AB-1234 was renewed."
        cleaned, report = remove_pii(text, {"region_packs": ["india"]}, use_ner=False)
        self.assertIn("REDACTED_VEHICLE", cleaned)
        self.assertEqual(report["summary"].get("INDIAN_VEHICLE"), 1)

    def test_unknown_region_pack_logs_but_does_not_crash(self):
        # Should warn about unknown pack and still apply core regex.
        text = "Email: alice@example.com"
        cleaned, report = remove_pii(text, {"region_packs": ["mars"]}, use_ner=False)
        self.assertIn("REDACTED_EMAIL", cleaned)


class TestRecognizerBackend(unittest.TestCase):
    """The 'recognizer' config field selects between legacy and presidio
    backends, and the report records which one was used."""

    def test_legacy_backend_default(self):
        _, report = remove_pii("Email me at alice@example.com.", {}, use_ner=False)
        self.assertEqual(report.get("recognizer_used"), "legacy")
        self.assertEqual(report.get("region_packs"), [])

    def test_legacy_backend_explicit(self):
        _, report = remove_pii(
            "Email me at alice@example.com.",
            {"recognizer": "legacy", "region_packs": ["india"]},
            use_ner=False,
        )
        self.assertEqual(report["recognizer_used"], "legacy")
        self.assertEqual(report["region_packs"], ["india"])

    def test_presidio_falls_back_to_legacy_when_unavailable(self):
        # presidio isn't installed in CI by default — the request should
        # silently fall back to legacy with the report flagging it.
        try:
            import presidio_analyzer  # noqa: F401
            self.skipTest("presidio is installed; this test is for fallback behavior.")
        except ImportError:
            pass
        _, report = remove_pii(
            "Email me at alice@example.com.",
            {"recognizer": "presidio"},
            use_ner=False,
        )
        self.assertEqual(report["recognizer_used"], "legacy", "should fall back when presidio missing")
        # Detection still works via the regex backend.
        self.assertGreater(report["total"], 0)


class TestVerhoeffChecksum(unittest.TestCase):
    def test_known_valid_aadhaar(self):
        # Synthetic test vectors — last digit is the Verhoeff check digit,
        # generated by computing Verhoeff over the first 11 digits.
        for valid in ("456712345679", "987654321096", "123456789010"):
            self.assertTrue(_verhoeff_check(valid), f"{valid} should validate")

    def test_invalid_check_digit(self):
        # Same as above with the last digit altered (off by one).
        self.assertFalse(_verhoeff_check("456712345670"))
        self.assertFalse(_verhoeff_check("987654321090"))

    def test_wrong_length(self):
        self.assertFalse(_verhoeff_check("12345"))
        self.assertFalse(_verhoeff_check("1234567890123"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
