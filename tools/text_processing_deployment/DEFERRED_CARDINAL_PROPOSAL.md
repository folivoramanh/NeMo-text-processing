# Proposal: Defer English cardinal rendering to the verbalizer

**Status:** experimental, opt-in. **Scope:** deterministic English TN, initially
evaluated on numeric literals containing at most 12 digits.

Move number-to-words expansion for top-level cardinal tokens from the tagger to
the verbalizer. Compare two implementations against ordinary NeMo: deferred
cardinal rendering alone, and the same grammar with OpenFst lookahead tagging.
Raw cardinal is the initial candidate: it matches the measured baseline outputs
and existing English assertions on this fork. The combined mode is faster but
has two known serial regressions and remains a comparison configuration. Keep
the default pipeline unchanged until the chosen configuration meets the
output-parity and serving-latency gates below.

## Problem and proposed behavior

The current tagger generates spoken forms while it is still searching possible
token boundaries and semiotic classes. For a top-level cardinal, the verbalizer
then mainly removes the field labels. This carries spoken-number output paths
through the expensive tagging search before the tokenization has been selected.

```text
Current:
125 -> cardinal { integer: "one hundred and twenty five" }
    -> one hundred and twenty five

Deferred:
125 -> cardinal { integer: "125" }
    -> one hundred and twenty five
```

The proposed tagger recognizes and scores the original numeric spelling, and
the verbalizer renders the selected cardinal using the existing number graph.
Leading zeros, comma grouping, and the separate negative field retain their
existing grammar rules. Embedded numbers in money, measure, decimal, date,
ordinal, and other classes continue using their current graphs in this first
prototype.

## Two configurations

| Configuration | Tagging | Verbalization |
| --- | --- | --- |
| Reference | Ordinary tagger, producing spoken cardinal fields | Existing verbalizer |
| Raw cardinal | Ordinary composition with the deferred-cardinal tagger | Paired deferred-cardinal verbalizer |
| Raw cardinal + lookahead | Convert that same deferred tagger to `ilabel_lookahead`, relabel the input, and compose | The same paired deferred-cardinal verbalizer |

Both candidate configurations use the existing parser, field permutations,
punctuation handling, and post-processing. The prototype does not change serial
weights, token-type defaults, decoder pruning, or epsilon-removal order. ITN,
audio/non-deterministic normalization, and serving deployment are separate work.

## Preserve classification costs

Build the raw-number recognizer by tropical input projection of the original
number-to-words transducer, followed by optimization:

```python
raw_number_graph = pynini.project(original_number_graph, "input").optimize()
```

For a numeric spelling `x`, this retains the original minimum classification
cost over its spoken alternatives:

```text
original: x -> y, with cost c(x, y)
deferred: x -> x, with minimum cost min_y c(x, y)
```

An unweighted digits expression would lose those preferences and could alter
competition with other classes. The verbalizer uses the original weighted
number graph to choose a reading after classification. Classification and
verbalization remain separate stages; preserving a span's minimum cost does
not guarantee an identical winner when full-sentence paths are tied.

Lookahead is applied only after building the deferred tagger. Its complete
relabel table must travel with the immutable matcher, including collision-
avoidance entries. Loading the matcher as a mutable Pynini vector FST would
discard its lookahead implementation, so the loader retains its OpenFst type.

## Implementation and artifact pairing

- `CardinalFst(defer_verbalization=True)` changes only its top-level `fst`;
  subgraphs used by other semiotic classes retain their spoken outputs.
- `ClassifyFst(defer_cardinal=True)` builds the paired English tagger.
- `VerbalizeFinalFst(defer_cardinal=True)` builds its matching verbalizer.
- Both grammar caches use distinct `_raw_cardinal` filenames. Ordinary caches
  keep their current names.
- `Normalizer(lookahead_dir=...)` provides the optional lookahead tagging path.
  The comparison helper exports the raw tagger and installs its matching raw
  verbalizer together.

A raw tagger must always be deployed with its raw verbalizer. Mixing it with
the ordinary verbalizer would leave digits unspoken. Bundle and version those
artifacts together. All new grammar switches default to `False`; the public
`Normalizer` does not silently enable deferred rendering.

The implementation is in the English cardinal/classifier/verbalizer modules
and [lookahead.py](../../nemo_text_processing/text_normalization/lookahead.py).
The [comparison runner](benchmark_deferred_cardinal.py) constructs the complete
pairs and measures both candidates against the same baseline.

## Evidence and limits

Measurements were repeated on the personal fork's `main` base `255bbf8b`, with
this prototype applied, on November 22, 2026: macOS arm64, Python 3.12.2,
Pynini 2.1.6.post1, and OpenFst 1.8.3. Query order rotates between the three
configurations. Means include `normalize()` and exclude grammar construction,
loading, and lookahead conversion.

| Corpus | Ordinary Pynini | Raw cardinal | Raw cardinal + lookahead |
| --- | ---: | ---: | ---: |
| 505 eligible English TN regression rows, 3 repetitions | 7.04 ms | 4.57 ms | 2.66 ms |
| One 29-word mixed English sentence, 100 repetitions | 75.92 ms | 44.75 ms | 19.96 ms |

Raw cardinal reduced these measured means by about **35% and 41%** relative to
the ordinary pipeline. The combined mode was approximately 42% and 55% faster
than raw cardinal alone, but its parity failures prevent recommending it for
activation. These are local single-process results; target Linux/service
latency, throughput, and memory remain acceptance work.

| Validation on the fork base | Baseline | Raw cardinal | Raw cardinal + lookahead |
| --- | --- | --- | --- |
| Existing English pytest suite | 1,555 passed | 1,555 passed | 1,553 passed, 2 failed |
| Additional cardinal/lookahead/evaluation-filter tests | 91 passed | Covered in the additional tests | Covered in the additional tests |
| Eligible written regression rows | Reference | 0/505 differences | 2/505 differences |
| Selected counterexample replay | Reference | 0/51 differences | 8/51 differences |
| Repeated 29-word sentence | Reference | 0 differences | 0 differences |

The two combined-mode regression failures are `31/31/100` and
`123/261788/2021`: the baseline preserves `/`, while the combined mode emits
`slash`. Full outputs for these and the eight replay differences are in the
[comparison tables](deferred_cardinal_comparison.md). Both tables exclude inputs
with numeric spans longer than 12 digits.

All ten combined candidate tags are accepted by the original lattice at the
same minimum cost as the baseline tags. This identifies alternative tied paths;
it does not make the output differences acceptable for a strict parity gate.

The 505-row set is the fork's complete 507-row non-audio written regression
corpus after filtering two longer-number inputs. It uses the money and
punctuation-matching settings from the repository tests. Baseline and raw mode
both match 502/505 supplied expectations; combined mode matches 500/505. The
three shared failures are existing Roman-numeral expectations whose deterministic
pytest assertions are commented out. Passing pytest is therefore not a claim
of 100% linguistic accuracy. The English test total also includes unaffected
ITN tests.

The 51-input replay is a selected counterexample set, mostly short serial and
mixed-input cases; it has no independent expected-output oracle. It does not
establish equivalence for every 12-digit input. Earlier broader testing found
long comma-grouped inputs outside this reporting window that changed span
boundaries at the same cost. Keep that known limitation visible in any rollout
decision.

The 12-digit bound is an **evaluation filter**, not a runtime restriction.
The implementation does not reject, truncate, or automatically fall back for
larger numbers. Any production policy limiting this optimization to that window
must first define and validate the corresponding baseline fallback.

## Reproduce the comparison

Run in a Pynini/OpenFst environment with the compatible lookahead plugin, from
the repository root:

```bash
python -m tools.text_processing_deployment.benchmark_deferred_cardinal \
  --work-dir /tmp/deferred-cardinal --report /tmp/deferred-cardinal.json \
  --max-digits 12 --repeats 3 \
  --test-files tests/nemo_text_processing/en/data_text_normalization/test_cases_cardinal.txt \
               tests/nemo_text_processing/en/data_text_normalization/test_cases_money.txt
```

Add the other non-audio regression files for the complete written corpus.
For the selected replay, use
`--cases-jsonl tests/nemo_text_processing/en/data_text_normalization/deferred_cardinal_counterexamples.jsonl`.
Use `--without-lookahead` when evaluating raw mode in an environment without the
optional OpenFst lookahead plugin.
The runner applies the repository's money/post-processing and punctuation-
matching settings, reports filtered inputs explicitly, records every changed
output and shared error, and separates expected-output checks from parity.
Nonzero exit status means a parity, expected-output, or execution-error gate
failed. Setup and conversion are excluded from timing, and query order rotates
between the three configurations.

Run the existing English pytest assertions with each candidate while retaining
the constructor and call settings in the tests:

```bash
python -m pytest tests/nemo_text_processing/en \
  --ignore=tests/nemo_text_processing/en/test_normalization_with_audio.py \
  --cpu --use_local_test_data --tn_cache_dir=/tmp/deferred-test-native \
  -p tools.text_processing_deployment.deferred_cardinal_pytest \
  --deferred-runtime raw --deferred-work-dir /tmp/deferred-test-raw
```

Repeat with `--deferred-runtime raw-lookahead`. These commands intentionally
exclude the separate audio decoder; deferred mode rejects non-deterministic TN.

## Acceptance and rollout

1. Establish output oracles and baseline comparisons for powers-of-ten
   boundaries, zero/leading zeros, signs, grouping, and mixed-class contexts
   throughout the 1–12-digit window. Add a representative held-out English corpus.
2. Require zero regressions against expected outputs and review every baseline
   difference for each candidate separately. Resolve weighted ties with an
   agreed policy before enabling a configuration requiring exact parity.
3. Measure the exact candidate build on Linux at sentence length, recording
   latency distributions, throughput, memory, and dispatch overhead under the
   intended service concurrency.
4. Define the policy for larger or ambiguous inputs, including a validated
   baseline fallback if rollout is restricted to a numeric window.
5. Enable one explicit configuration only after those gates pass. Retain the
   ordinary pipeline as the rollback path and package the raw tagger/verbalizer
   as one versioned unit.

The requested decision is to retain and evaluate both opt-in prototypes.
Production activation and a general accuracy-preservation claim depend on the
remaining gates above.
