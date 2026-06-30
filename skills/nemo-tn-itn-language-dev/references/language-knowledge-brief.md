# Language Knowledge Brief

Use this as a compact checklist for building enough language understanding before coding.

## Identity

```text
language_code:
language_name:
script:
direction: TN | ITN | both
locale assumptions:
comparable repo languages:
```

## Number System

Capture:

- zero and digits
- teens and tens
- hundreds, thousands, millions, larger magnitudes
- negative marker
- decimal separator and spoken decimal marker
- digit-by-digit contexts such as ZIP, phone, IDs
- whether `1` changes by context, gender, or following noun

Examples:

```text
0 ->
1 ->
10 ->
21 ->
100 ->
1,000 ->
-5 ->
3.14 ->
ZIP/ID 101 ->
```

## Morphology And Agreement

Record only what affects TN/ITN:

- gender
- number
- case
- classifiers
- plural forms
- apocope or contextual short forms
- script variants
- optional spoken variants

Prefer encoding these as data or narrow rewrite rules. Avoid introducing arbitrary token fields without checking repo/Sparrowhawk constraints.

## Semiotic Class Plan

For each class:

```text
class:
direction:
input examples:
token schema:
output examples:
dependencies:
data files:
ambiguous cases:
out of scope:
```

## Ambiguity Rules

List forms that should not be normalized because they are common words, names, or ambiguous context.

Example:

```text
"one" alone remains a word in ITN unless this language's existing behavior converts single digits.
```

## Reviewer Notes

Before finalizing, make sure a reviewer can answer:

- Why this language pattern is correct.
- Which existing language implementation it follows.
- Where the linguistic inventory lives.
- What tests prove the behavior.
- What cases are intentionally deferred.
