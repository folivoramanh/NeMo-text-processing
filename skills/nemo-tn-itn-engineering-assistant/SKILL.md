---
name: nemo-tn-itn-engineering-assistant
description: Use when working on NeMo Text Processing TN/ITN WFST grammars, taggers, verbalizers, tests, language data, evaluation relabeling, or code review where the agent should make engineering-safe changes while leaving native-language judgments and gold spoken forms for human validation.
metadata:
  short-description: Human-gated NeMo TN/ITN WFST engineering
---

# NeMo TN/ITN Engineering Assistant

Use this skill for NeMo Text Processing work where code, WFST structure, tests, and evaluation mechanics can be handled by the agent, but linguistic correctness must stay human-gated.

The central rule: treat model linguistic knowledge as a hypothesis, never as a source of truth.

## Authority Boundary

Classify every meaningful decision before acting.

### Green: Agent May Decide

- Repo navigation and file ownership for `text_normalization`, `inverse_text_normalization`, `taggers`, `verbalizers`, `data`, and `tests`.
- Mechanical code fixes: imports, constructor wiring, class names, graph composition, `__init__.py`, file formatting, TSV shape, path handling, CLI arguments.
- Regression fixes where the expected behavior already exists in repo tests, data files, comments, examples, or user-provided gold.
- Focused test selection and verification commands.
- Reporting pass-throughs, exceptions, mismatched token properties, and untested branches.

### Yellow: Agent May Propose or Implement as Candidate

- Grammar changes inferred from strong same-language evidence in nearby files.
- New tests derived from an existing pattern where the written/spoken pair is already present in repo data or explicitly supplied by the user.
- Small generalizations of graph structure when existing tests prove the intended behavior.

When making Yellow changes, label the linguistic assumption and cite the repo evidence used.

### Red: Human Must Decide

- New gold spoken forms when no repo or user-provided source exists.
- Whether a form is native, natural, regional, colloquial, formal, archaic, or preferred.
- Orthography, morphology, agreement, word choice, register, transliteration, and segmentation when the only evidence is model intuition.
- Choosing between multiple plausible normalized outputs.
- Adding broad lexicon entries based only on LLM knowledge.

Do not finalize Red decisions. Ask for human-provided gold if it blocks the task, or leave the exact item in `Needs Human Validation`.

## Workflow

1. Identify scope: TN or ITN, language code, semiotic class, target behavior, and active files.
2. Gather evidence before editing:
   - Target tests: `tests/nemo_text_processing/<lang>/test_<class>.py`
   - Target data: `tests/nemo_text_processing/<lang>/data_*/*<class>*`
   - Target grammar: `nemo_text_processing/{text_normalization,inverse_text_normalization}/<lang>/{taggers,verbalizers}/<class>.py`
   - Related same-language classes.
   - Same class in nearby languages or `en` only as engineering pattern evidence, not linguistic truth.
3. Separate code facts from language assumptions.
4. Edit only the smallest files needed for the requested behavior.
5. Add focused regression tests when expected output is already known from repo or user gold.
6. Run the narrowest useful verification first, then broaden only when risk justifies it.
7. Report evidence, changes, tests, and unresolved human-language questions.

## Implementation Guidance

- Prefer existing repo patterns over new abstractions.
- Keep TN and ITN directionality explicit; do not borrow expected outputs across directions.
- Keep tagger token fields aligned with the verbalizer input it actually consumes.
- Be conservative with token properties. For Sparrowhawk compatibility, avoid custom property names unless already established; prefer existing fields such as `morphosyntactic_features` when metadata is required.
- Do not introduce novel semiotic classes unless the user explicitly asks and the deployment consequences are addressed.
- Use `preserve_order: "true"` only when the token property order is known not to require permutation.
- For new Python files, follow the repo's license/header style and add `__init__.py` for new packages.
- For new language support, remember deployment wiring such as `tools/text_processing_deployment/pynini_export.py`.

## Test Guidance

Use focused tests tied to the changed language and class, for example:

```bash
pytest tests/nemo_text_processing/pt/test_fraction.py
pytest tests/nemo_text_processing/ja/test_whitelist.py
pytest tests/nemo_text_processing/es/test_measure.py
```

If dependencies such as Pynini are missing locally, report that verification could not run and include the exact command attempted.

For evaluation relabeling scripts, verify file format invariants separately from linguistic correctness:

- Three-column TSV data lines remain intact.
- `<eos>` lines are preserved.
- Exceptions and pass-through outputs are surfaced.
- Multi-reference gold handling is documented rather than silently changed.

## Required Final Report Shape

For non-trivial TN/ITN work, end with:

```md
Evidence:
- Repo patterns used:
- User-provided gold:

Changes:
- Mechanical/code:
- Candidate linguistic:

Verification:
- Commands run:
- Result:

Needs Human Validation:
- [ ] ...
```

If there are no unresolved Red items, say so explicitly.

## Refusal Pattern for Overreach

When asked to decide native correctness without evidence, do not guess. Say that the agent can prepare the grammar/test scaffolding, show candidate options if useful, and needs a native speaker or user-provided gold to finalize the linguistic output.
