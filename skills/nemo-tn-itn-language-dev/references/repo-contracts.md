# Repo Contracts

## Layout

- TN lives under `nemo_text_processing/text_normalization/<lang>/`.
- ITN lives under `nemo_text_processing/inverse_text_normalization/<lang>/`.
- A language normally has `taggers/`, `verbalizers/`, `data/`, optional `graph_utils.py`, and optional `utils.py`.
- Tests live under `tests/nemo_text_processing/<lang>/`.

## Runtime Flow

TN:

1. `Normalizer` selects language-specific `ClassifyFst` and `VerbalizeFinalFst`.
2. `ClassifyFst` composes semiotic-class taggers into sentence-level tokenization.
3. `TokenParser` parses `tokens { ... }` structures.
4. Token fields may be permuted.
5. `VerbalizeFinalFst` composes verbalizers and word handling into final text.

ITN follows the same shape through `InverseNormalizer`.

## Tagger And Verbalizer Contract

Taggers should wrap class output with `GraphFst.add_tokens()`.

Example:

```text
12 kg -> measure { cardinal { integer: "twelve" } units: "kilograms" }
```

Verbalizers should remove wrappers with `GraphFst.delete_tokens()`.

Example:

```text
measure { cardinal { integer: "twelve" } units: "kilograms" } -> twelve kilograms
```

Define this schema before editing Pynini. Reviewers should be able to inspect input -> token -> output without mentally executing the graph.

## Integration Points

For a new language or new class, check:

- `<direction>/<lang>/taggers/<class>.py`
- `<direction>/<lang>/verbalizers/<class>.py`
- `<direction>/<lang>/taggers/tokenize_and_classify.py`
- `<direction>/<lang>/verbalizers/verbalize.py`
- `<direction>/<lang>/verbalizers/verbalize_final.py` if the language has custom final handling
- `nemo_text_processing/text_normalization/normalize.py`
- `nemo_text_processing/inverse_text_normalization/inverse_normalize.py`
- `tools/text_processing_deployment/pynini_export.py`
- `tests/nemo_text_processing/<lang>/test_<class>.py`
- `tests/nemo_text_processing/<lang>/data_text_normalization/test_cases_<class>.txt`
- `tests/nemo_text_processing/<lang>/data_inverse_text_normalization/test_cases_<class>.txt`
- Sparrowhawk shell tests when deployment coverage exists

## Data Rules

- Prefer existing TSVs in the same language before adding near-duplicates.
- Prefer shared `en` address/electronic resources only when the output is intentionally code-switched or language-independent.
- Keep TSV entries normalized: no duplicate rows differing only by trailing spaces.
- Use data files for linguistic inventories such as digits, months, units, states, designators, currency names, and whitelist entries.

## Sparrowhawk Compatibility

Be conservative with token fields. `CONTRIBUTING.md` warns that arbitrary tagger properties may work in Python but fail in Sparrowhawk.

- Prefer predefined semiotic classes.
- Prefer `morphosyntactic_features` for extra grammatical features when needed.
- Use `preserve_order: true` when property order matters or permutation is unnecessary.
- Update `tools/text_processing_deployment/pynini_export.py` for new language deployment support.
