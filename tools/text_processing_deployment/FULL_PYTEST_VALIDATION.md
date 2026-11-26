# Full pytest validation for deferred English cardinals

Validation date: November 26, 2026. The tested environment is macOS arm64,
Python 3.12.2, pytest 8.4.0, Pynini 2.1.6.post1, and OpenFst 1.8.3 with the
lookahead plugin available. The unchanged personal-fork baseline is
`255bbf8b584dd869e95e8eb0abe0f4c10bfbf938`.

The full suite has existing failures. These are documented here; repairing other
languages is outside this proposal. No expected outputs, assertions, or audio
candidate limits were changed to obtain a passing result.

## What the earlier 1,555 result covered

The earlier English-only command excluded `en/test_normalization_with_audio.py`.
Its 1,555 collected items consisted of:

| Coverage | Test items |
| --- | ---: |
| Deterministic English TN assertions | 509 |
| English ITN | 1,041 |
| Roman tests with inactive deterministic assertions | 4 |
| Sentence splitting | 1 |
| Total | 1,555 |

The dedicated English audio module adds 58 items and is included in the full
runs. The full inventory is **9,913 items: 9,475 stock tests plus 438 component
and API tests**. There are no file exclusions or marker filters.

The standard CPU profile omits `--run_audio_based`. Consequently, 26 stock
items execute without active assertions: 19 German ordinal items, four English
Roman items, and three Swedish dedicated audio items. These are passing items,
not pytest skips. Other tests can also have additional audio assertions behind
that flag. The dedicated English, German, and Spanish audio modules and Russian
TN still run in the standard profile.

## Full-suite protocol and mode scope

The [pytest adapter](deferred_cardinal_pytest.py) enables raw or raw-plus-lookahead
normalization for the 22 applicable deterministic English `Normalizer` instances
in the stock tests. English ITN, audio subclasses, and other languages retain
their native behavior. Component and public API tests exercise their explicit
modes independently. A full-suite total therefore does not mean every test calls
the optimized English TN path.

With `--deferred-isolate-caches`, constructors explicitly given the resolved
`--tn_cache_dir` root use `<root>/isolated/<language>`. This includes audio parent
initializations and inverse normalizers. Kinyarwanda uses `cache_dir=None` to
avoid its existing FAR-key mismatch. Independent component-test caches are
unchanged. This is a test-adapter configuration, with no production grammar or
test-oracle changes.

| Mode | Collected | Passed | Failed | Skipped / errors | New failures versus baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 9,913 | 9,707 | 206 | 0 / 0 | Reference |
| Raw digits | 9,913 | 9,707 | 206 | 0 / 0 | 0 |
| Raw digits + lookahead | 9,913 | 9,707 | 206 | 0 / 0 | 0 |

Every mode passes all **1,613 English tests** and all **438 component/API tests**.
All 206 failed test IDs and literal failure messages are identical across modes.
The [complete failure inventory](deferred_cardinal_pytest_baseline_failures.json)
records every test ID and assertion message, together with environment and
baseline-comparison metadata.
Each full pytest command exits **1**; the repository-wide suite is not entirely
green. These existing failures are documented under the agreed proposal scope.
No test files, assertions, or candidate limits were changed to obtain these results.

## Why the first full attempts were not acceptance results

The first shared-cache attempt stopped with three Kinyarwanda collection errors,
before tests executed. Its unchanged
[verbalizer](../../nemo_text_processing/text_normalization/rw/verbalizers/verbalize_final.py)
writes FAR entries `ALL` and `REDUP`, then requests `verbalize` when restoring
the same file. Subsequent class-level constructors raise `KeyError: 'verbalize'`.
The writer/reader mismatch also exists at the unchanged baseline commit.

For diagnosis only, a temporary cached FAR received a `verbalize` alias of the
original `ALL` graph, retaining both original entries. A subsequent shared-cache
run collected all 9,913 tests: **9,456 passed, 457 failed, zero skipped**. This
alias is not part of the final isolated-cache protocol.

Of those shared-cache failures, 432 occurred in the areas affected by two
existing cache filename collisions: the
[Hindi verbalizer](../../nemo_text_processing/text_normalization/hi/verbalizers/verbalize_final.py)
uses an English verbalizer filename, while the
[Japanese TN tagger](../../nemo_text_processing/text_normalization/ja/taggers/tokenize_and_classify.py)
and [Chinese TN tagger](../../nemo_text_processing/text_normalization/zh/taggers/tokenize_and_classify.py)
use the same Chinese tagger filename. The shared directory produced 140 Hindi
and 292 Chinese failures. Separate language directories address this test setup
problem without changing those production filenames. The resulting full-suite
failure set must be measured again; it cannot be inferred by subtracting those
432 cases, because the corrected grammar selection changes the test context.

## Independently reproduced baseline failures

A clean checkout of `255bbf8b` ran the **entire 9,475-item stock suite** with the
same per-language cache isolation and uncached Kinyarwanda configuration.
Its normalizer and inverse-normalizer imports were verified to come from that
clean checkout. It produced **9,269 passed, 206 failed, zero skipped or errors**.
Every failed test ID and literal JUnit failure message exactly matches the
feature checkout's full-suite baseline and both optimized modes.

An earlier four-language subset reproduced 25 failures across 1,033 items
(1,008 passed). That subset result was diagnostic, not the complete failure
inventory. The full stock run establishes the 206-case baseline below.

| Language | Failures | Observed symptom |
| --- | ---: | --- |
| Italian | 4 | Terminal-period spacing differs from the expected electronic readings. |
| Russian | 5 | Expected cardinal, decimal, date, money, or electronic reading is absent from the returned candidates. |
| Swedish | 2 | Whitelist outputs contain literal serialized-token fragments. |
| Vietnamese | 49 | ITN outputs contain unconverted spoken text or serialized field fragments. |
| Chinese | 146 | Date and money ITN outputs contain serialized field fragments; the date corpus is reused by several stock test modules. |

The Russian tests request 50 tagged candidates for the failing cardinal case
and 100 for the other cases. These observations do not establish whether a
missing reading is absent from the grammar or outside the returned candidates.
Candidate limits were not increased, and no nondeterministic grammar repairs
were made. The full-suite and subset failure counts differ, so this report uses
the complete unchanged-base run as its comparison oracle.

## Reproduction

Run from the repository root in an environment with `fstconvert` and its
compatible lookahead plugin configured. Keep the local test archive at
`tests/.data/test_data.tar.gz`. Run the modes sequentially:
[conftest](../../tests/conftest.py) deletes and re-extracts that directory.
Use the explicit test path below because [setup.cfg](../../setup.cfg) includes
`nemo_text_processing` in `norecursedirs`.

```bash
WORK_DIR=/tmp/nemo-deferred-pytest
mkdir -p "$WORK_DIR"
export NEMO_DEFERRED_CACHE="$WORK_DIR/component-cache"
python -m pytest tests/nemo_text_processing --cpu --use_local_test_data \
  --tn_cache_dir "$WORK_DIR/cache" -p tools.text_processing_deployment.deferred_cardinal_pytest \
  --deferred-isolate-caches -o addopts='' -q -ra --tb=short \
  --junitxml "$WORK_DIR/baseline.xml" > "$WORK_DIR/baseline.log" 2>&1
printf '%s\n' "$?" > "$WORK_DIR/baseline.exitcode"
for mode in raw raw-lookahead; do
  python -m pytest tests/nemo_text_processing --cpu --use_local_test_data \
    --tn_cache_dir "$WORK_DIR/cache" -p tools.text_processing_deployment.deferred_cardinal_pytest \
    --deferred-isolate-caches --deferred-runtime "$mode" --deferred-work-dir "$WORK_DIR/$mode" \
    -o addopts='' -q -ra --tb=short --junitxml "$WORK_DIR/$mode.xml" \
    > "$WORK_DIR/$mode.log" 2>&1
  printf '%s\n' "$?" > "$WORK_DIR/$mode.exitcode"
done
```

Inspect each exit code, failure ID, and skip reason. A missing-plugin skip cannot
validate lookahead mode. The local logs and JUnit artifacts are retained for
comparison; large machine-specific artifacts are not committed. The separate
unchanged-base reproduction excluded conftest to avoid re-extracting shared test
data and set the existing grammar cache through the test utilities before
calling pytest; it was a baseline diagnostic, not a full-suite acceptance run.
