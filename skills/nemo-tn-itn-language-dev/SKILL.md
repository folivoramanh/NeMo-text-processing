---
name: nemo-tn-itn-language-dev
description: Build, extend, or review NeMo Text Processing WFST text normalization and inverse text normalization language grammars. Use when adding a completely new language, extending a semiotic class for an existing language, creating language-specific TSV resources, designing tagger/verbalizer token schemas, adding tests, auditing language coverage, or preparing code for reviewer-friendly validation.
---

# NeMo TN/ITN Language Development

## Purpose

Help a coding agent act like a careful NeMo TN/ITN language developer: understand the language behavior, choose comparable implementations, define reviewable token schemas, implement WFST graphs incrementally, and produce enough tests and notes for a human reviewer to trust the change.

This skill is for two main tasks:

- Scale a language that does not exist in the repo yet.
- Extend or repair a semiotic class for a language that already exists.

## Triage

Before editing, identify:

1. Direction: `TN`, `ITN`, or both.
2. Language code and whether the language already exists.
3. Target semiotic classes.
4. Comparable languages in this repo.
5. Expected input -> token -> output examples.
6. Test files and export/deployment hooks affected.

If the target is unclear, inspect the repo and make a conservative assumption. Ask only when the direction, language, or class cannot be inferred.

## Repo Convention Scan

Read `references/repo-convention-scan.md` before implementing or heavily rewriting a class. Run the offline scanner to learn local patterns before coding:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction tn --lang <lang> --class <class>
```

Use the scan to identify comparable implementations, data TSV usage, shared helpers, composer wiring, tests, and likely hard-coded logic that should move into data files.

## Core Contracts

Read `references/repo-contracts.md` when touching graph structure, token fields, new language registration, or Sparrowhawk export.

Key rules:

- Taggers emit parseable token strings such as `cardinal { integer: "one" }`.
- Verbalizers delete the token structure and produce final text.
- Token fields may be permuted between tagger and verbalizer unless `preserve_order: true` is used.
- Avoid novel semiotic classes or arbitrary token fields unless compatibility is explicitly understood.
- Reuse existing data TSVs and utilities before adding duplicated lexicons.

Read `references/class-ownership-and-boundaries.md` before adding advanced cases that could overlap with another semiotic class. A test case should make clear why the target class owns the input instead of `cardinal`, `word`, or another class.

## New Language Workflow

Read `references/language-scaling-workflow.md` when adding a language from scratch or planning broad language coverage.

Implement in dependency order:

1. `word`, `punctuation`, `whitelist`
2. `cardinal`
3. `decimal`
4. `ordinal`, `fraction`
5. `date`, `time`
6. `measure`, `money`
7. `telephone`, `electronic`, `range`, `address`, language-specific extras

For each class, keep tagger, verbalizer, TSV data, composer imports, and tests in sync.

## Semiotic Class Extension Workflow

Read `references/semiotic-class-extension.md` when adding or improving one class in an existing language.

Work class-by-class. First infer the current token schema and tests. Then add missing lexical data, graph logic, composer wiring, and test cases. Keep behavior changes local unless the class depends on a shared lower-level graph.

## Advanced Test Coverage

Read `references/advanced-tn-coverage.md` when the implemented behavior passes basic tests but needs stronger TN coverage.

Use the offline coverage helper before adding more cases:

```bash
python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang <lang> --class <class>
```

The helper reads `tests/nemo_text_processing/<lang>/data_text_normalization/test_cases_<class>.txt`, reports scenario coverage, flags duplicate inputs, and suggests missing advanced cases. It does not require Pynini.

## Prompt Templates

Read `references/prompt-templates.md` when the user wants reusable prompts for auditing a language, extending a semiotic class, creating advanced coverage profiles, or adding advanced test cases.

## Language Knowledge Brief

Read `references/language-knowledge-brief.md` when the language behavior is not already documented in the task. Fill it mentally or explicitly before large changes.

Useful output for reviewers:

- Which language facts are encoded.
- Which variants are accepted or rejected.
- Which ambiguous cases intentionally remain unchanged.
- Which examples prove the tagger and verbalizer contract.

## Validation

Prefer focused tests:

```bash
python -m pytest --cpu tests/nemo_text_processing/<lang>/test_<class>.py -q
```

If `pytest`, `pynini`, or Sparrowhawk is unavailable, run what is still useful:

```bash
python -m py_compile <changed-python-files>
sh -n tests/nemo_text_processing/<lang>/test_sparrowhawk_normalization.sh
git diff --check
```

Never claim semantic WFST behavior is verified unless a Pynini-backed normalization test actually ran.

## Final Response

For implementation work, report:

- Classes and direction changed.
- Token schema examples.
- Files changed.
- Tests added or updated.
- Validation commands run.
- Known gaps and environment blockers.
