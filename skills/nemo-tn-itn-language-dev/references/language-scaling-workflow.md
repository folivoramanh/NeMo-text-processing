# Language Scaling Workflow

## 1. Scope The Work

Record:

- Language code and script.
- Direction: TN, ITN, or both.
- Semiotic classes in scope.
- Accepted out-of-scope cases.
- Comparable languages in the repo.

Use `en` for base contracts, then pick closer languages by feature:

- Rich TN and ITN: `en`, `de`, `es`, `pt`, `vi`, `ko`
- Smaller scaffold: `rw`, `hy`, `mr`
- CJK or script-specific behavior: `zh`, `ja`, `ko`
- Gender/case-heavy behavior: `es`, `fr`, `de`, `ru`

## 2. Build A Coverage Matrix

Create a quick matrix before coding:

```text
class       TN status     ITN status     depends on        notes
cardinal    needed        needed         numbers TSV       core
decimal     needed        needed         cardinal          separator rules
date        later         later          cardinal/months   locale formats
```

This prevents hidden scope creep.

## 3. Define Token Schemas

For each class, write examples:

```text
TN input: 21 kg
TN token: measure { cardinal { integer: "twenty one" } units: "kilograms" }
TN output: twenty one kilograms

ITN input: twenty one kilograms
ITN token: measure { cardinal { integer: "21" } units: "kg" }
ITN output: 21 kg
```

If the schema needs gender, case, classifier, script form, or variant output, decide field names before implementation and check Sparrowhawk constraints.

## 4. Implement In Dependency Order

Start with foundational classes:

1. `word`, `punctuation`, `whitelist`
2. `cardinal`
3. `decimal`
4. `ordinal`, `fraction`
5. `date`, `time`
6. `measure`, `money`
7. `telephone`, `electronic`, `range`, `address`

Only build higher-level classes after lower-level graphs expose stable reusable attributes such as `graph`, `final_graph_wo_negative`, or `graph_no_exception`.

## 5. Keep Patch Units Reviewable

One class patch should usually include:

- Tagger.
- Verbalizer.
- TSV data files.
- Composer import and graph union.
- Unit test file.
- Test case file.
- Export/Sparrowhawk hook if applicable.

Do not combine unrelated classes unless the user asks for a full scaffold.

## 6. Test Shape

Use parameterized test files with `input~expected`.

For multiple valid outputs:

```text
200~doscientos~doscientas
```

Add edge cases:

- Single digit and zero behavior.
- Negative values.
- Leading zeros if the class can contain ZIPs, phone numbers, or IDs.
- Punctuation next to the semiotic class.
- Ambiguous forms that should remain words.
- Cased input if the language supports cased mode.

## 7. Final Review Summary

Report:

- Token schemas implemented.
- Files changed.
- New tests and representative cases.
- Known gaps.
- Whether Pynini tests, syntax checks, and Sparrowhawk checks ran.
