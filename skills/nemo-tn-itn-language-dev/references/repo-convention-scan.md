# Repo Convention Scan

Use this before writing or rewriting a semiotic class.

## Why

NeMo TN/ITN code is convention-heavy. A class can pass basic tests while still feeling wrong to reviewers because it ignores local patterns:

- hard-coded lexical inventories instead of TSV data
- duplicated utilities instead of shared helpers
- broad graph domains that misclassify other classes
- missing composer wiring
- missing Sparrowhawk/export/test hooks
- inconsistent token schema between tagger and verbalizer

## Workflow

Run:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction tn --lang <lang> --class <class>
```

For ITN:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction itn --lang <lang> --class <class>
```

Read the report before editing. It shows:

- target tagger/verbalizer/test files
- comparable implementations in other languages
- imports and helper usage
- data files referenced through `get_abs_path("data/...")`
- Pynini idioms used
- composer registration hints
- warning patterns such as broad `NEMO_SIGMA`, inline `string_map`, or missing tests

## Convention Rules

Prefer:

- `GraphFst`, `add_tokens()`, `delete_tokens()`
- `get_abs_path("data/...")` plus `pynini.string_file(...)` for lexical inventories
- local `graph_utils.py` only for language-specific helpers
- importing shared `en` helpers when the logic is common and already reusable
- `convert_space()` when token values can contain spaces inside quotes
- `pynutil.add_weight()` when class priority is intentional
- domain restriction with `pynini.compose(...)` when a graph could over-match
- `preserve_order: true` when verbalizer order should not be permuted

Avoid:

- large hard-coded dictionaries or lists for lexical data
- copying `en/graph_utils.py` into a language just to reuse common constants
- broad `NEMO_SIGMA` matches without tight boundaries
- class-specific TSVs that duplicate existing number/month/unit resources
- adding a test case before deciding class ownership

## Reviewer Packet

When finalizing a class change, include:

- Which comparable implementations were inspected.
- Which repo conventions were followed.
- Which hard-coded logic, if any, remains and why.
- Data files added or reused.
- Composer/test/export wiring status.
