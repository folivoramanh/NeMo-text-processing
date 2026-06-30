# Advanced TN Coverage

Use this after a class has basic working tests but still needs reviewer-grade coverage.

## Coverage Levels

### Level 1: Basic

- One simple input.
- One common formatted input.
- One sentence-context input.

### Level 2: Structural Variants

- Different separators.
- Optional spaces.
- Punctuation adjacency.
- Prefixes or suffixes.
- Cased input when relevant.
- Leading zero behavior.
- Multiple valid outputs when the language permits variants.

### Level 3: Boundary And Ambiguity

- Inputs that look similar but should not normalize.
- Inputs that need context words before the class should own them.
- Minimum and maximum length supported by the graph.
- Nested or neighboring semiotic classes.
- Sentence beginning, middle, and end.
- Locale-specific alternatives.
- Production/export-sensitive cases such as `preserve_order`.

## Telephone TN Checklist

For telephone-like classes, cover:

- compact local number
- hyphen-separated local number
- dot-separated local number
- parenthesized area code
- landline area code
- mobile prefix
- country code with `+`
- country code with spaces
- country code with hyphens
- country code plus parenthesized area code
- emergency/service short code with context cues
- IP-like dotted number if the class intentionally owns that behavior
- long grouped ID/card-like sequence if the class intentionally owns that behavior
- sentence context before and after the number
- punctuation immediately after the number
- ambiguous numeric forms that should stay as another class or word

Avoid treating bare short codes such as `112`, `113`, `114`, or `1900` as telephone by default. They may require context words like `call`, `hotline`, `emergency`, or locale-specific equivalents. See `class-ownership-and-boundaries.md`.

## Using The Script

Run:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang vi --class telephone
```

The report is heuristic. Use it to decide what cases to add, not as proof that the WFST is correct.

## Profiles Are Data, Not Code

The audit script reads JSON profiles from:

```text
skills/nemo-tn-itn-language-dev/coverage_profiles/
```

Bundled profiles include common TN semiotic classes:

- `generic.json`: fallback for any class.
- `cardinal.json`, `ordinal.json`, `decimal.json`, `fraction.json`
- `date.json`, `time.json`
- `money.json`, `measure.json`
- `telephone.json`, `electronic.json`, `address.json`
- `range.json`, `roman.json`, `serial.json`
- `abbreviation.json`, `whitelist.json`, `word.json`, `punctuation.json`

List available profiles:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --list-profiles
```

Use a custom profile:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py \
  --lang ja \
  --class date \
  --profile /path/to/date-profile.json
```

When moving to a new language, keep the profile focused on input shapes and coverage categories. Do not encode expected verbalizations in the profile; those belong in the language's `test_cases_<class>.txt`.

Supported detector types:

- `all`
- `any_of`
- `not`
- `any`
- `sentence_context`
- `multiple_expected`
- `regex_fullmatch`
- `regex_search`
- `startswith`
- `contains`

Use `all`, `any_of`, and `not` to model ownership boundaries without encoding language-specific verbalizations. For example, a profile can require a telephone-like separator while excluding date-like slash patterns.

## Adding Advanced Cases

For each missing scenario, add:

```text
input~expected
```

Do not add generated cases blindly. A native or domain reviewer should be able to explain why the expected verbalization is correct.

Prefer adding a few high-signal examples over dozens of repetitive variants.
