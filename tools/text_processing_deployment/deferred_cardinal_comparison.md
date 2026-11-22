# English TN comparison: raw cardinal versus raw cardinal plus lookahead

Measured on November 22, 2026, against the personal fork main base `255bbf8b` with this prototype applied. The baseline uses the unchanged ordinary tagger/verbalizer. Raw and raw-plus-lookahead use the same paired raw grammars.

**Scope:** inputs with no numeric span longer than 12 digits. Digits across commas and decimal points count together; separate numbers are not added across a sentence or serial separators. This is an evaluation filter, not a runtime limit.

| Corpus | Eligible inputs | Removed by digit filter | Raw differences | Raw + lookahead differences |
| --- | ---: | ---: | ---: | ---: |
| Written regression | 505 | 2 | 0 | 2 |
| Selected counterexamples | 51 | 0 | 0 | 8 |

Raw cardinal matches the baseline in both corpora. The combined configuration differs on the ten inputs below. The baseline output is therefore also the raw-cardinal output for these rows.

JSON-style escapes make tabs, newlines, backslashes, and invisible characters explicit. Outputs are complete.

| ID | Corpus | Input | Baseline and raw cardinal | Raw cardinal + lookahead | Common path cost |
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

The first two rows also fail existing serial assertions. The eight selected replay inputs have no independent output oracle. Baseline and raw mode separately share three pre-existing Roman-numeral expectation failures in the written corpus; these are not introduced by the prototype and are not candidate-vs-baseline differences.

Use `benchmark_deferred_cardinal.py` with all non-audio `test_cases_*.txt` files and `--max-digits 12` for the first corpus. Use `--cases-jsonl tests/nemo_text_processing/en/data_text_normalization/deferred_cardinal_counterexamples.jsonl` for the second. The proposal contains complete invocation examples and the validation gates.
