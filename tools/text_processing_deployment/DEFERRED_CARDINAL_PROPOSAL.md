# English TN: guarded deferred cardinal rendering

**Status:** implemented, experimental, and opt-in; the ordinary runtime remains
the default. Both deferred configurations retry a complete native baseline pair
when the optimized result is ambiguous or outside scope.

The change moves top-level cardinal expansion from tagging to verbalization,
using ordinary composition or OpenFst label lookahead. It does not establish
equivalence for every input or readiness for production.

## Grammar change and supported scope

```text
Ordinary: 125 -> cardinal { integer: "one hundred and twenty five" }
Deferred: 125 -> cardinal { integer: "125" }
```

Only deterministic English TN without an LM is supported.
Embedded numeric graphs in money, measures, decimals, dates, ordinals, serials,
and other classes keep their existing spoken outputs. Signs, grouping, and
leading zeros retain their existing grammar rules.

The top-level recognizer is the tropical input projection of the original
weighted number graph, followed by optimization:

```python
raw_number_graph = pynini.project(original_number_graph, 'input').optimize()
```

For spelling `x`, projection keeps `x -> x` at cost `min_y c(x, y)`. A plain
unweighted digits expression would lose classification preferences. The paired
verbalizer uses the original weighted number graph after tagging selects the
cardinal. Preserving a span's minimum cost alone cannot preserve tied winners.

The optimized numeric window is at most **12 digits per numeric span**.
The runtime counts ASCII digits across commas and decimal points, including
leading zeros, using `[0-9][0-9,.]*`. Separate numbers across words or serial
separators are counted separately. Any longer span routes the whole input to
the native baseline; digits are neither truncated nor rejected.

## Guarded selection and baseline retry

Both native deferred and lookahead deferred runtimes guard the tagging lattice
and every selected verbalization lattice before relying on their best paths.
The guard excludes the selected output language and computes the cheapest
remaining output. It does not enumerate all paths, introduce a beam, cap states,
or change the selected path when the guard passes.

For tags, equivalence includes only ASCII spaces ignored by `TokenParser`.
The exclusion acceptor preserves the parsed structure, literal keys and
delimiters, field order, and every character inside quoted values. It permits
`SPACE*` at parser whitespace sites and requires `SPACE+` after closing quoted
values, matching the parser's terminator rule. Embedded quotes stay literal.
If parsing or reconstruction fails, or the acceptor misses the complete chosen
tag, the guard conservatively uses the exact literal output instead.

The verbalizer guard compares exact output strings. Duplicate paths with the
same output are harmless. A distinct equal-cost or near-optimal output raises
`AmbiguousRewrite`. The comparison tolerance is
`max(1e-6, 8 * 2**-23 * max(1, abs(best_cost), abs(alternate_cost)))` to account
conservatively for float32 tropical weights. A strictly cheaper selected output
may therefore still request fallback.

On ambiguity, `Normalizer.normalize()` retries the **whole original input** with
the paired native baseline tagger and baseline verbalizer, retaining the call's
punctuation settings and current post-processor. It never sends baseline spoken
cardinal tags through the deferred verbalizer. `last_fallback_reason` is reset
for each call: `None`, `ambiguous_rewrite`, or `numeric_scope`.

Default normalization, grammar weights, global token/byte defaults, and the
existing parser/permutation/post-processing behavior remain unchanged. ITN,
audio/non-deterministic decoding, and Riva integration are outside this change.

## Public API and CLI

```python
from nemo_text_processing.text_normalization.normalize import Normalizer

raw = Normalizer(input_case='cased', defer_cardinal=True, cache_dir='build/tn-cache')
raw.export_runtime('build/deferred-lookahead')  # Default: lookahead=True.
raw.export_runtime('build/deferred-native', lookahead=False)

loaded = Normalizer(input_case='cased', runtime_dir='build/deferred-lookahead')
print(loaded.normalize('31/31/100'))
```

`runtime_dir` loads the complete pair without compiling grammars. Its language
and input case must match the bundle. A bundle without a post-processor requires
`post_process=False`. Whitelists must be baked in before export; do not combine
`runtime_dir` with a whitelist or `lookahead_dir`; deferred mode requires complete
paired bundles.

```bash
python -m nemo_text_processing.text_normalization.normalize \
  --language en --input_case cased --defer_cardinal \
  --cache_dir build/tn-cache --export_runtime build/deferred-cli
python -m nemo_text_processing.text_normalization.normalize \
  --language en --input_case cased --runtime_dir build/deferred-cli \
  --text '31/31/100'
```

CLI export creates a lookahead bundle. Native `export_runtime(..., lookahead=False)`
does not require the plugin. This flag cannot convert a loaded lookahead matcher
back to a native tagger.
Lookahead conversion needs `fstconvert` and a compatible `ilabel_lookahead`
plugin; ordinary Pynini pip wheels do not provide that plugin.

## Pairing, integrity, caches, and workers

```text
manifest.json
raw/verbalizer.fst
raw/lookahead/{tagger.fst,relabel.pairs,metadata.json}  # Lookahead backend.
raw/tagger.fst                                      # Native backend instead.
baseline/{tagger.fst,verbalizer.fst}
post_processor.fst                                 # When exported.
```

Export validates the complete bundle in a temporary sibling directory and
publishes it with one directory rename. Existing destinations are not reused.
The manifest fixes schema version 1, grammar ABI `english_deferred_cardinal_v1`,
English TN, deterministic mode, input case, backend, the 12-digit window,
post-processing presence, whitelist identity, and SHA-256 hashes for the exact
artifact set. Loading checks these fields, hashes, graph types, and graph
validity. Hashes detect changed contents; they are not artifact signatures or a
promise of binary compatibility across arbitrary OpenFst installations.

Custom whitelist contents are hashed, recorded in the bundle, and separated
in raw and native fallback caches. Deferred public constructors additionally
namespace all their caches by the grammar ABI, Pynini version, and English
`.py`, `.tsv`, and `.far` source contents. Ordinary cache behavior is preserved.
The bundled baseline is independent of subsequently installed grammar sources.

Lookahead retains its immutable OpenFst matcher and complete relabel table,
including collision-avoidance entries. Process workers reload graphs from the
same bundle and revalidate its manifest digest and contents; serialization does
not silently turn the matcher into a mutable vector FST. Keep the bundle path
available and unchanged for `normalize_list(..., n_jobs=...)` workers.

## Validation evidence and interpretation

Measured November 26, 2026, on macOS arm64, Python 3.12.2, Pynini 2.1.6.post1,
OpenFst 1.8.3, against personal fork base `255bbf8b` plus this change.

| Check | Baseline | Raw digits | Raw digits + lookahead |
| --- | ---: | ---: | ---: |
| Existing English pytest suite (audio excluded; includes ITN) | 1,555 passed | 1,555 passed | 1,555 passed |
| Written regression, ≤12 digits | Reference | 0/505 differences | 0/505 differences |
| Selected counterexample replay | Reference | 0/51 differences | 0/51 differences |
| Generated boundaries and contexts | Reference | 0/2,371 differences | 0/2,371 differences |

The default English suite plus component/API tests passed 1,983 tests in the
first full run. After final artifact/cache hardening, all 438 component/API
tests passed again. The two historical serial failures and eight historical
replay differences now return the baseline output through fallback; see the
[complete historical difference table](deferred_cardinal_comparison.md).

Latency cells show **mean / p95 milliseconds**, including guards and retries.
The corpora use three repetitions; each sentence uses 100 repetitions.
Fallback counts describe unique inputs from the first repetition, in each
optimized mode, rather than the number of timed calls.

| Corpus | Baseline | Raw digits | Raw digits + lookahead | Fallback inputs |
| --- | ---: | ---: | ---: | ---: |
| Written regression (505; two longer rows filtered) | 5.08 / 13.50 | 4.87 / 12.53 | 3.48 / 8.97 | 116/505 |
| Selected replay (51) | 8.20 / 14.67 | 10.35 / 24.61 | 7.99 / 20.68 | 22/51 |
| Boundaries and contexts (2,371) | 4.26 / 10.04 | 4.71 / 10.93 | 3.90 / 9.22 | 181/2,371 |
| Mixed `sentence` (29 words) | 41.70 / 51.99 | 69.98 / 84.04 | 56.53 / 70.72 | 1/1 |
| Simple `cardinal-sentence` (14 words) | 14.54 / 15.32 | 16.19 / 17.10 | 11.51 / 11.85 | 0/1 |

Combined-mode means improve about 31% on written regression, 8% on boundaries,
and 21% on the simple cardinal sentence. The mixed sentence is about 36% slower
because of fallback. Replay mean is similar, while its p95 worsens. Raw digits
alone also regresses on several workloads. These measurements support testing
the combined runtime as an opt-in choice, rather than a universal speedup claim.

Reports distinguish baseline parity, expected-output failures, execution
errors, and fallback counts. The written regression corpus includes the shared
Roman-numeral expectation failures `Sam II`, `Chapter IV`, and `PART XL`.
Their deterministic assertions are commented out in the stock Roman pytest;
passing that suite does not mean those written expectations pass.

The selected replay has no independent output oracle. The generated boundary
corpus likewise compares baseline outputs: 2,371 deterministic cases from 401
core integers, seeded with 20261126, covering every width from 1 to 12 digits,
signs, zero/grouping boundaries, and mixed semiotic contexts.

The `sentence` benchmark is the 29-word mixed input containing `$2,450.75`.
It always falls back: `dollars` and `us dollars` remain genuine equal-cost
alternatives after structural tag spaces are ignored. Guards and retry add work
on every call. Judge guarded end-to-end performance with its fallback frequency.
The separate `cardinal-sentence` benchmark is a simple cardinal control with no
observed fallback; tables distinguish these two sentence inputs.

Exact sentence inputs, passed with the runner's `--text` and `--repeats 100`:

```text
sentence: On November 20, 2026, we shipped 125 packages weighing 12.5 kg each to 3 offices, and the total cost was $2,450.75, including a 7.5% delivery charge for overnight service.
cardinal-sentence: We counted 125 packages and 42 boxes before sending 103 items to 3 offices.
```

## Reproduce regression, replay, and boundary comparisons

Run these Bash commands from the repository root in a compatible Pynini/OpenFst
environment. They exclude audio data and report only the 12-digit window;
runtime routing remains active independently of this reporting filter.

```bash
WORK_DIR=$(mktemp -d)
REGRESSION_FILES=()
for path in tests/nemo_text_processing/en/data_text_normalization/test_cases_*.txt; do
  [[ "$path" == *normalize_with_audio* ]] || REGRESSION_FILES+=("$path")
done
python -m tools.text_processing_deployment.benchmark_deferred_cardinal \
  --work-dir "$WORK_DIR/regression" --report "$WORK_DIR/regression.json" \
  --max-digits 12 --repeats 3 --test-files "${REGRESSION_FILES[@]}"
python -m tools.text_processing_deployment.benchmark_deferred_cardinal \
  --work-dir "$WORK_DIR/replay" --report "$WORK_DIR/replay.json" \
  --max-digits 12 --repeats 3 --cases-jsonl \
  tests/nemo_text_processing/en/data_text_normalization/deferred_cardinal_counterexamples.jsonl
python -m tools.text_processing_deployment.generate_deferred_cardinal_cases \
  --output "$WORK_DIR/boundaries.jsonl"
python -m tools.text_processing_deployment.benchmark_deferred_cardinal \
  --work-dir "$WORK_DIR/boundaries" --report "$WORK_DIR/boundaries-report.json" \
  --max-digits 12 --repeats 3 --cases-jsonl "$WORK_DIR/boundaries.jsonl"
```

The runner uses the public guarded API, rotates query order, preserves money and
punctuation test settings, and excludes construction/loading/conversion from
timings. `--without-lookahead` omits the optional plugin. Nonzero status means a
parity, expectation, or execution-error gate failed; inspect each separately.

Run stock English tests sequentially, retaining their constructor/call settings:

```bash
python -m pytest tests/nemo_text_processing/en \
  --ignore=tests/nemo_text_processing/en/test_normalization_with_audio.py \
  --cpu --use_local_test_data --tn_cache_dir "$WORK_DIR/stock-cache"
for variant in raw raw-lookahead; do
  python -m pytest tests/nemo_text_processing/en \
    --ignore=tests/nemo_text_processing/en/test_normalization_with_audio.py \
    --cpu --use_local_test_data --tn_cache_dir "$WORK_DIR/stock-cache" \
    -p tools.text_processing_deployment.deferred_cardinal_pytest \
    --deferred-runtime "$variant" --deferred-work-dir "$WORK_DIR/$variant-tests"
done

# Component, pairing, public API, ambiguity, and worker tests.
python -m pytest tests/nemo_text_processing/test_*.py --cpu --use_local_test_data
```

## Remaining acceptance gates

Require zero new baseline differences and review shared linguistic failures
separately. Add representative held-out English text and adversarial mixed
inputs; finite regression and boundary corpora are not an all-input proof.
Measure the exact guarded build on Linux and in the intended Riva/service path:
latency distributions, throughput, cold load, bundle/RSS memory, fallback rate,
and process/concurrency overhead. Local results do not establish those targets.
Activate an explicit configuration only after those gates pass, preserving the
ordinary runtime and paired native baseline as the operational rollback path.
