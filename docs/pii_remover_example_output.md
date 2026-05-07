# `pii_remover.remove_pii()` — Before / After Example

The example below uses regex + custom redaction (NER stage skipped for the docs because it requires a model download). With the full pipeline enabled, names like "John Smith" would additionally be replaced with `REDACTED_NAME`.

## Input

```
Contact John Smith at john.smith@example.com or 555-123-4567. John Smith works
for Acme Corp in San Francisco. Visit https://example.com?email=alice@example.com
for details. His SSN is 123-45-6789 and credit card 4111 1111 1111 1111.
Server IP: 192.168.1.42. Internal codename: Project Falcon.
```

Configuration:

```json
{
  "whitelist": ["Acme Corp"],
  "redaction_list": ["Project Falcon"]
}
```

## Output (regex + custom only)

```
Contact John Smith at REDACTED_EMAIL or REDACTED_PHONE. John Smith works
for Acme Corp in San Francisco. Visit https://example.com?email=REDACTED_EMAIL
for details. His SSN is REDACTED_SSN and credit card REDACTED_CC.
Server IP: REDACTED_IP. Internal codename: REDACTED_CUSTOM.
```

## Output (full pipeline, NER enabled)

```
Contact REDACTED_NAME at REDACTED_EMAIL or REDACTED_PHONE. REDACTED_NAME works
for Acme Corp in REDACTED_LOCATION. Visit https://example.com?email=REDACTED_EMAIL
for details. His SSN is REDACTED_SSN and credit card REDACTED_CC.
Server IP: REDACTED_IP. Internal codename: REDACTED_CUSTOM.
```

Note: "Acme Corp" survives both passes because it's on the whitelist. "John Smith" is replaced consistently across both occurrences thanks to the carry-forward step.

## Redaction Report (excerpt)

```json
{
  "summary": {
    "EMAIL": 2,
    "PHONE": 1,
    "SSN": 1,
    "CREDIT_CARD": 1,
    "IP": 1,
    "CUSTOM": 1
  },
  "total": 7,
  "events": [
    { "category": "EMAIL",       "original": "john.smith@example.com",   "replacement": "REDACTED_EMAIL",  "source": "regex"  },
    { "category": "PHONE",       "original": "555-123-4567",             "replacement": "REDACTED_PHONE",  "source": "regex"  },
    { "category": "EMAIL",       "original": "alice@example.com",        "replacement": "REDACTED_EMAIL",  "source": "regex"  },
    { "category": "SSN",         "original": "123-45-6789",              "replacement": "REDACTED_SSN",    "source": "regex"  },
    { "category": "CREDIT_CARD", "original": "4111 1111 1111 1111",      "replacement": "REDACTED_CC",     "source": "regex" },
    { "category": "IP",          "original": "192.168.1.42",             "replacement": "REDACTED_IP",     "source": "regex"  },
    { "category": "CUSTOM",      "original": "Project Falcon",           "replacement": "REDACTED_CUSTOM", "source": "custom" }
  ]
}
```

## Test results

12 tests run, all pass:

```
test_custom_string_is_force_redacted                 ... ok
test_consistent_redaction_across_occurrences         ... ok
test_embedded_pii_in_url                             ... ok
test_report_has_required_fields                      ... ok
test_redacts_credit_card                             ... ok
test_redacts_email                                   ... ok
test_redacts_ipv4                                    ... ok
test_redacts_phone_multiple_formats                  ... ok
test_redacts_ssn                                     ... ok
test_all_categories_have_redaction_tokens            ... ok
test_whitelist_overrides_custom_redaction_list       ... ok
test_whitelisted_email_is_preserved                  ... ok
```
