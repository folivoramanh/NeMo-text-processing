# Class Ownership And Boundaries

Use this before adding advanced test cases or broadening a tagger.

## Principle

Every semiotic class must have a clear ownership boundary. A test case should not merely be accepted by a graph; it should be appropriate for that class.

Ask:

- Why is this input this class, not another class?
- Does the input need context words to disambiguate?
- Would classifying this input broadly cause regressions in ordinary text?
- Should this be a positive test, a negative/risk note, or out of scope?

## Ambiguous Bare Numbers

Bare numbers are usually owned by `cardinal` unless there is a strong class-specific reason.

Examples:

```text
112
113
114
1900
```

These may be emergency/service/telephone-like numbers in some locales, but as standalone TN inputs they are also plausible cardinals. Prefer contextual tests unless the language has an explicit product requirement that bare service codes are telephone.

Better telephone-oriented examples:

```text
gọi 112
số khẩn cấp 113
đường dây 1900 1234
hotline 1900-xxxx
```

The exact wording and expected output are language-specific and should be reviewed by someone who understands that locale.

## Context Cues

Use context when class ownership depends on meaning:

- telephone: `call`, `phone`, `hotline`, `emergency`, locale-specific words
- money: currency symbol/name
- measure: unit symbol/name
- address: street, avenue, apartment, postal cues
- date: month name, era marker, separator pattern, date-order convention
- time: hour/minute markers, AM/PM, clock punctuation

## Ownership Matrix

Use this table as a starting point. Language-specific behavior can override it, but the final summary should say why.

| Class | Owns | Common Overlaps / Risks |
| --- | --- | --- |
| `cardinal` | Bare numeric values and language-specific number spellings | date, time, money, measure, telephone, serial, range |
| `ordinal` | Ordinal suffixes, ordinal words, ordinal markers | cardinal, roman, names, serials |
| `decimal` / `decimals` | Numeric values with decimal separator | money, measure, version, IP address, date |
| `fraction` | Fraction expressions | date, ratio, measure, slash-based IDs |
| `date` | Numeric or named dates with date evidence | cardinal, fraction, range, serial |
| `time` | Clock-like expressions with time evidence | ratio, score, duration, version, cardinal |
| `money` | Amount plus currency symbol/code/name | decimal, measure, ticker/abbreviation |
| `measure` | Amount plus unit | money, math, address abbreviations, electronic units |
| `telephone` | Phone numbers with phone-like structure or context | cardinal, serial, IP address, long ID/card numbers |
| `electronic` | Email, URL, domain, handles when supported | decimal/version, word, telephone/IP |
| `address` | Address number plus street/postal/unit cues | cardinal, measure/unit abbreviations, telephone/ZIP |
| `range` | Two values linked by range punctuation/words | negative numbers, phone numbers, IDs, dates |
| `roman` | Roman numerals with appropriate context | letters, names, ordinal, serial |
| `serial` | Alphanumeric identifiers/codes | telephone, range, cardinal, word |
| `abbreviation` | Initialisms or abbreviation lexicon entries | word, address/state code, measure units |
| `whitelist` | Explicit lexical exceptions | Any more specific class if whitelist is broad |
| `word` | Ordinary passthrough words | Any class if graph priority is too broad |
| `punctuation` | Standalone or sentence punctuation | punctuation inside dates, decimals, URLs, phone numbers |

## Positive And Risk Cases

Advanced coverage should include both:

- Positive cases that the class should own.
- Risk cases that should be discussed or avoided because they overlap another class.

Risk cases do not always belong in `test_cases_<class>.txt`. Sometimes they belong in reviewer notes or in a negative test only if the repo has a clear negative-test pattern for that behavior.

## Patch Guidance

When adding advanced cases:

1. Add context to ambiguous inputs.
2. Keep bare ambiguous inputs out unless explicitly required.
3. Document the class ownership rationale in the final summary.
4. Check class priority weights if the new graph overlaps `cardinal`, `word`, or another broad class.
