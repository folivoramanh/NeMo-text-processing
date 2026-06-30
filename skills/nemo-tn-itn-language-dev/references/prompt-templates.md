# Prompt Templates

Use these prompts from the repository root. Replace values in angle brackets.

## Audit Existing Language

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: <LANGUAGE_NAME> <TN|ITN|TN and ITN>, path: nemo_text_processing/<text_normalization|inverse_text_normalization>/<lang>.

Goal:
1. Read the existing implementation.
2. Build a semiotic class coverage matrix.
3. Compare with relevant languages in this repo.
4. Identify complete, missing, and weak classes.
5. For each gap, list tagger/verbalizer/data/test files likely affected.
6. Do not modify files yet.
```

## Extend Existing Semiotic Class

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: <LANGUAGE_NAME> <TN|ITN>, semiotic class: <class>.

Goal:
1. Inspect current <lang> implementation and tests for <class>.
2. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction <tn|itn> --lang <lang> --class <class>
3. Compare with 1-2 relevant languages.
4. Define input -> token -> output examples before editing.
5. Implement the missing behavior in tagger/verbalizer/data/composer files while following repo conventions.
6. Add focused tests in test_cases_<class>.txt and test_<class>.py if needed.
7. Run focused validation if available.
8. Summarize token schema, repo conventions followed, files changed, tests, and known gaps.
```

## Plan A New Language

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: new language <LANGUAGE_NAME> (<lang>), direction: <TN|ITN|both>.

Goal:
1. Build a language knowledge brief.
2. Choose comparable languages from the repo.
3. Propose class implementation order.
4. Create a coverage matrix for required semiotic classes.
5. Define initial token schemas for core classes: cardinal, decimal, date, time, measure, money.
6. List files that need to be created or registered.
7. Do not modify files yet.
```

## Japanese TN Scaling Prompt Set

Use these prompts when scaling `ja` text normalization in this repo. Run them from the repository root and handle one class at a time unless the prompt explicitly says review-only.

### 1. Audit Japanese TN Coverage

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: Japanese TN (`ja`), path: nemo_text_processing/text_normalization/ja.

Review only. Do not modify files.

Goal:
1. Inspect existing Japanese TN taggers, verbalizers, data files, composer files, and tests.
2. Build a class coverage matrix for:
   word, punctuation, whitelist, cardinal, decimal, ordinal, fraction, date, time, money, measure, telephone, electronic, address, range, roman, serial, abbreviation.
3. For each existing class, run if a profile exists:
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang ja --class <class>
4. For each class, run:
   python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction tn --lang ja --class <class>
5. Compare Japanese with relevant implementations in this repo, especially `en`, `zh` if present, `ko`, and mature European implementations when useful.
6. Identify:
   - classes that are implemented and acceptable
   - implemented but weak classes
   - missing classes
   - class-overlap risks
   - missing tests or Sparrowhawk hooks
7. Produce a recommended implementation order. Do not code yet.
```

### 2. Harden An Existing Japanese TN Class

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: Japanese TN (`ja`), semiotic class: <class>.

Goal:
1. Inspect the current Japanese implementation and tests for <class>.
2. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction tn --lang ja --class <class>
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang ja --class <class>
3. Read:
   - references/semiotic-class-extension.md
   - references/advanced-tn-coverage.md
   - references/class-ownership-and-boundaries.md
4. Define Japanese-specific input -> token -> output examples before editing.
5. Pay attention to Japanese-specific behavior:
   - Arabic digits vs Kanji numerals
   - half-width vs full-width punctuation or symbols when relevant
   - era/year/month/day formats for dates
   - counters, suffixes, and readings only if the current class owns them
   - ambiguous numeric strings that should remain cardinal, date, time, serial, or word
6. Implement only the missing behavior that belongs to <class>.
7. Prefer TSV-backed lexical inventories and existing Japanese number/date/time resources over hard-coded lists.
8. Add high-signal tests in tests/nemo_text_processing/ja/data_text_normalization/test_cases_<class>.txt.
9. Run focused validation:
   python3 -m pytest --cpu tests/nemo_text_processing/ja/test_<class>.py -q
   python3 -m py_compile <changed Python files>
   git diff --check
10. Summarize token schema, class ownership decisions, files changed, tests, and known gaps.
```

### 3. Add Or Finish Japanese TN Money

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: Japanese TN (`ja`), semiotic class: money.

Goal:
1. Inspect Japanese cardinal, decimal, and existing money files if present.
2. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction tn --lang ja --class money
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang ja --class money
3. Compare with `en`, `ko`, and another mature implementation that has major/minor currency handling.
4. Define input -> token -> output examples for:
   - yen symbol and yen name
   - integer yen
   - decimal currency when supported
   - major/minor currency units when supported
   - currency code if supported
   - punctuation adjacency and sentence context
5. Decide what Japanese money should not own, especially bare decimals and bare numbers.
6. Implement tagger, verbalizer, TSV data, composer wiring, tests, and Sparrowhawk hook if this repo convention requires it.
7. Run focused validation and summarize reviewer-facing decisions.
```

### 4. Add A Missing Japanese TN Class

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: Japanese TN (`ja`), new semiotic class: <class>.

Goal:
1. Confirm that <class> is missing or only partially wired.
2. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction tn --lang ja --class <class>
3. Read:
   - references/language-scaling-workflow.md
   - references/semiotic-class-extension.md
   - references/class-ownership-and-boundaries.md
4. Compare with 2-3 existing implementations for the same class.
5. Before coding, produce a small Japanese language brief for this class:
   - valid positive examples
   - ambiguous examples to reject or require context
   - required TSV inventories
   - token schema
   - tests and composer files affected
6. Implement the smallest reviewable version first.
7. Add tests that prove ownership boundaries, not only happy paths.
8. Run focused validation and report gaps clearly.
```

### 5. Final Japanese TN PR Review

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Review my current Japanese TN (`ja`) changes before PR.

Review only. Do not modify files.

Focus on:
1. Token schema consistency between Japanese taggers and verbalizers.
2. Composer registration in Japanese classify/verbalize files.
3. TSV layout, trailing whitespace, duplicate lexical data, and hard-coded lists.
4. Coverage quality by semiotic class, using:
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang ja --class <class>
5. Class ownership risks involving cardinal, date, time, money, serial, electronic, and word.
6. Sparrowhawk/export hooks and generated grammar paths if affected.
7. Validation commands that should be run before PR.

Report findings first with file/line references, then give a compact merge-readiness summary.
```

## Add Advanced TN Cases

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: <LANGUAGE_NAME> TN, semiotic class: <class>.

Goal:
1. Inspect existing basic test cases.
2. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang <lang> --class <class>
3. Read references/class-ownership-and-boundaries.md.
4. If no class-specific profile exists, propose or create coverage_profiles/<class>.json.
5. Compare against relevant languages' test_cases_<class>.txt.
6. Propose advanced scenarios before editing, including which ambiguous inputs require context.
7. Add only high-signal test cases whose expected output is supported by the graph or by the intended implementation.
8. Run focused checks and summarize missing coverage and risky class-overlap cases that remain.
```

## Review, Fix, And Enhance Existing Class

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: <LANGUAGE_NAME> <TN|ITN>, semiotic class: <class>.

Files likely involved:
- nemo_text_processing/<text_normalization|inverse_text_normalization>/<lang>/taggers/<class>.py
- nemo_text_processing/<text_normalization|inverse_text_normalization>/<lang>/verbalizers/<class>.py
- nemo_text_processing/<text_normalization|inverse_text_normalization>/<lang>/taggers/tokenize_and_classify.py
- nemo_text_processing/<text_normalization|inverse_text_normalization>/<lang>/verbalizers/verbalize.py
- tests/nemo_text_processing/<lang>/test_<class>.py
- tests/nemo_text_processing/<lang>/data_text_normalization/test_cases_<class>.txt
- tests/nemo_text_processing/<lang>/data_inverse_text_normalization/test_cases_<class>.txt
- tests/nemo_text_processing/<lang>/test_sparrowhawk_normalization.sh
- tests/nemo_text_processing/<lang>/test_sparrowhawk_inverse_text_normalization.sh

Goal:
1. Review the existing implementation and tests for <class>.
2. Read:
   - references/semiotic-class-extension.md
   - references/advanced-tn-coverage.md
   - references/class-ownership-and-boundaries.md
   - references/repo-convention-scan.md
3. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/repo_convention_scan.py --direction <tn|itn> --lang <lang> --class <class>
4. If this is TN, run:
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang <lang> --class <class>
5. Identify:
   - valid positive cases the class should own
   - risky or ambiguous cases that may belong to another class
   - missing advanced cases
   - duplicate, low-value, or misleading tests
   - hard-coded lexical inventories that should move to TSV data
   - deviations from comparable repo implementations
   - graph priority or domain-restriction risks
6. Before editing, produce a short plan:
   - which test cases should be removed, moved, rewritten with context, or added
   - whether tagger/verbalizer logic needs to change
   - which data should live in TSV files
   - whether composer, export, or Sparrowhawk tests need updates
7. Implement the fix/enhancement.
8. Preserve class ownership. Do not broaden the class to capture ambiguous inputs unless context or product requirements justify it.
9. Add or update tests.
10. Run focused validation if available:
   python3 -m pytest --cpu tests/nemo_text_processing/<lang>/test_<class>.py -q
   python3 -m py_compile <changed Python files>
   git diff --check
11. Final summary must include:
   - token schema examples
   - repo conventions followed
   - cases removed or rewritten because of class ownership risk
   - advanced cases added
   - files changed
   - validation status
   - remaining known gaps
```

## Review Existing Class Only

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target: <LANGUAGE_NAME> <TN|ITN>, semiotic class: <class>.

Review only. Do not modify files.

Goal:
1. Inspect the existing implementation and tests for <class>.
2. If this is TN, run:
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang <lang> --class <class>
3. Apply class ownership rules from references/class-ownership-and-boundaries.md.
4. Identify risky or misclassified cases, missing advanced coverage, duplicate tests, and graph-overlap risks.
5. Separate findings into:
   - must fix
   - should enhance
   - optional coverage
   - needs native/product decision
6. Include file/line references where possible.
```

## Create Coverage Profile

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Target semiotic class: <class>.

Goal:
1. Read references/advanced-tn-coverage.md.
2. Inspect existing test cases for this class across several languages.
3. Create or update skills/nemo-tn-itn-language-dev/coverage_profiles/<class>.json.
4. Keep the profile language-agnostic and focused on input shapes.
5. Do not encode expected normalized outputs in the profile.
6. Run:
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --list-profiles
   python3 skills/nemo-tn-itn-language-dev/scripts/tn_coverage_audit.py --lang <sample_lang> --class <class>
```

## Review Before PR

```text
Use the skill at skills/nemo-tn-itn-language-dev/SKILL.md.

Review my current changes for NeMo TN/ITN language development.

Focus on:
1. Token schema consistency between tagger and verbalizer.
2. Composer registration.
3. TSV duplication or trailing-space issues.
4. Test coverage by semiotic class.
5. Sparrowhawk/export compatibility.
6. Whether the final summary is reviewer-friendly.

Report findings first with file/line references. Do not edit files unless I ask.
```
