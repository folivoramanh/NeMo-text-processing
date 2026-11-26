# English TN comparison: raw cardinal versus raw cardinal plus lookahead

The complete guarded runtime was validated on November 26, 2026, against the personal fork main base `255bbf8b`. The baseline uses the ordinary tagger/verbalizer. Raw and raw-plus-lookahead use paired raw grammars and retry the original pipeline when competing best readings occur.

**Reporting scope:** inputs with no numeric span longer than 12 digits. Digits across commas and decimal points count together; separate numbers are not added across a sentence or serial separators. Larger spans are outside this table and route directly to the baseline at runtime.

| Corpus | Eligible inputs | Removed by digit filter | Raw differences | Raw + lookahead differences | Fallbacks in each optimized mode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Written regression | 505 | 2 | 0 | 0 | 116 |
| Selected counterexamples | 51 | 0 | 0 | 0 | 22 |
| Generated boundaries and contexts | 2,371 | 0 | 0 | 0 | 181 |

Both complete runtimes match the baseline in these corpora. Every historical difference below now triggers `ambiguous_rewrite` fallback and returns the baseline column exactly. These rows remain regression fixtures and document why the guard is necessary.

The remaining table records the **unguarded prototype measured on November 22**, before baseline routing was implemented. Its changed-output column is historical, rather than the output of the current runtime.

JSON-style escapes make tabs, newlines, backslashes, and invisible characters explicit. Outputs are complete.

| ID | Corpus | Input | Baseline and current guarded modes | Unguarded raw + lookahead (historical) | Common path cost |
| --- | --- | --- | --- | --- | ---: |
| D01 | regression | 31/31/100 | thirty one/thirty one/one hundred | thirty one slash thirty one slash one hundred | 1.12010002 |
| D02 | regression | 123/261788/2021 | one hundred twenty three/two six one seven eight eight/two thousand twenty one | one hundred twenty three slash two six one seven eight eight slash two thousand twenty one | 1.12010002 |
| D03 | counterexamples | https://a.b/1\\u202f133-ABC\\u202fa@b.com\\u202fO'Reilly\\u202fhello | HTTPS: slash/a. b/1\\u202f133 - ABC\\u202fa@b.com\\u202fO'Reilly\\u202fhello | HTTPS: slash slash a. b/1\\u202f133 - ABC\\u202fa@b.com\\u202fO'Reilly\\u202fhello | 208.430008 |
| D04 | counterexamples | 12:30/https://a.b/1/hello/hai/ngày/12kg | twelve: thirty/https: slash/a. b/1/hello/hai/ngày/12kg | twelve: thirty/https: slash slash a. b/1/hello/hai/ngày/12kg | 109.640205 |
| D05 | counterexamples | https://a.b/1\\u202f\\" | HTTPS: slash/a. b/1\\u202f \\" | HTTPS: slash slash a. b/1\\u202f \\" | 108.430099 |
| D06 | counterexamples | 12:30\\u200b한국어\\u200bhttps://a.b/1\\u200b\\"\\u200bO'Reilly | twelve: 30\\u200b한국어\\u200bhttps: slash/a. b/1\\u200b \\" \\u200bO'Reilly | twelve: 30\\u200b한국어\\u200bhttps: slash slash a. b/1\\u200b \\" \\u200bO'Reilly | 310.620026 |
| D07 | counterexamples | a@b.com\\u202f\$2.50\\u202fhttps://a.b/1\\u202fa@b.com\\u202f1st\\u202f\$2.50\\u202fhai\\u202f한국어 | a at b. com\\u202f\$2. 50\\u202fhttps: slash/a. b/1\\u202fa@b. com\\u202f1st\\u202f\$2. 50\\u202fhai\\u202f한국어 | a at b. com\\u202f\$2. 50\\u202fhttps: slash slash a. b/1\\u202fa@b. com\\u202f1st\\u202f\$2. 50\\u202fhai\\u202f한국어 | 514.84021 |
| D08 | counterexamples | một/https://a.b/1/hai/https://a.b/1/한국어 | một/https://a. b/one/hai/https: slash/a. b/1/한국어 | một/https://a. b/one/hai/https: slash slash a. b/1/한국어 | 208.5401 |
| D09 | counterexamples | hai\\u202fhello\\u202f\$2.50\\u202fO'Reilly\\u202f12\\u202fhttps://a.b/1\\u202f\\"\\u202f\[0\] | hai\\u202fhello\\u202f\$2. 50\\u202fO ' Reilly\\u202f12\\u202fhttps: slash/a. b/1\\u202f \\" \\u202f \[zero\] | hai\\u202fhello\\u202f\$2. 50\\u202fO ' Reilly\\u202f12\\u202fhttps: slash slash a. b/1\\u202f \\" \\u202f \[zero\] | 516.920044 |
| D10 | counterexamples | https://a.b/1/中文 | HTTPS: slash/a. b/1/中文 | HTTPS: slash slash a. b/1/中文 | 106.330101 |

Each combined tag was constrained through the original composed lattice, after rendering top-level raw-cardinal fields with the original number graph where needed. All ten candidate paths are accepted at the same minimum cost as the baseline. Equal costs still permit different strings and do not remove the output-parity failure.

The first two rows previously failed existing serial assertions; both optimized modes now pass those assertions. The eight selected replay inputs have no independent output oracle. All current modes share three pre-existing Roman-numeral expectation failures in the written corpus; these are not candidate-vs-baseline differences. Passing the English tests does not imply perfect linguistic accuracy.

Use `benchmark_deferred_cardinal.py` with all non-audio `test_cases_*.txt` files and `--max-digits 12` for the first corpus. Use `--cases-jsonl tests/nemo_text_processing/en/data_text_normalization/deferred_cardinal_counterexamples.jsonl` for the second. `generate_deferred_cardinal_cases.py --output /tmp/boundaries.jsonl` reproduces the third corpus using seed `20261126`, including powers-of-ten boundaries, signs, zeros, commas, and mixed-class contexts. Its JSONL SHA-256 is `01dee6fb2b977be65c8e137b7f1fe439e4043b2fad399ba864c439efd41a1e1a`. These generated inputs have no independent expected-output oracle. The proposal contains complete invocation examples, current timing results, and the validation gates.
