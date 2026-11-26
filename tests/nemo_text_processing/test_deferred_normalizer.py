# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public deferred-cardinal API coverage using the complete English grammars."""

import copy
import hashlib
import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pynini
import pytest

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on('CPU')]

SERIAL_CASES = [
    ('31/31/100', 'thirty one/thirty one/one hundred'),
    ('123/261788/2021', 'one hundred twenty three/two six one seven eight eight/two thousand twenty one'),
]

# D03--D10 in deferred_cardinal_comparison.md. These inputs previously changed
# their reading when only the raw tagger was converted to a lookahead matcher.
LOOKAHEAD_COUNTEREXAMPLES = [
    "https://a.b/1\u202f133-ABC\u202fa@b.com\u202fO'Reilly\u202fhello",
    '12:30/https://a.b/1/hello/hai/ngày/12kg',
    'https://a.b/1\u202f"',
    '12:30\u200b한국어\u200bhttps://a.b/1\u200b"\u200bO\'Reilly',
    'a@b.com\u202f$2.50\u202fhttps://a.b/1\u202fa@b.com\u202f1st\u202f$2.50\u202fhai\u202f한국어',
    'mo\u0323\u0302t/https://a.b/1/hai/https://a.b/1/한국어',
    'hai\u202fhello\u202f$2.50\u202fO\'Reilly\u202f12\u202fhttps://a.b/1\u202f"\u202f[0]',
    'https://a.b/1/中文',
]

REPLAY_FILE = Path(__file__).parent / 'en/data_text_normalization/deferred_cardinal_counterexamples.jsonl'
REPLAY_CASES = [json.loads(line) for line in REPLAY_FILE.read_text(encoding='utf-8').splitlines() if line]


@pytest.fixture(scope='module')
def grammar_cache(tmp_path_factory):
    # A shared cache can be supplied for local verification. Otherwise build
    # these expensive graphs once, without depending on a developer's files.
    return os.environ.get('NEMO_DEFERRED_CACHE') or str(tmp_path_factory.mktemp('deferred-grammars'))


@pytest.fixture(scope='module')
def normalizers(grammar_cache):
    ordinary = Normalizer(input_case='cased', lang='en', cache_dir=grammar_cache)
    deferred = Normalizer(input_case='cased', lang='en', cache_dir=grammar_cache, defer_cardinal=True)
    return ordinary, deferred


@pytest.fixture(scope='module')
def custom_whitelist_normalizer(grammar_cache, tmp_path_factory):
    source = tmp_path_factory.mktemp('deferred-whitelist') / 'words.tsv'
    contents = b'NVToken\tcaptured replacement\n1000000000000\tcaptured large number\n'
    source.write_bytes(contents)
    normalizer = Normalizer(input_case='cased', cache_dir=grammar_cache, whitelist=str(source), defer_cardinal=True)
    return normalizer, source, contents


@pytest.fixture(scope='module')
def baseline_output(normalizers):
    @lru_cache(maxsize=None)
    def normalize(text, punct_pre_process=False, punct_post_process=False):
        return normalizers[0].normalize(
            text, punct_pre_process=punct_pre_process, punct_post_process=punct_post_process
        )

    return normalize


def _export_runtime(normalizer, directory, *, lookahead):
    try:
        normalizer.export_runtime(directory, lookahead=lookahead)
    except RuntimeError as error:
        if lookahead and ('fstconvert is required' in str(error) or 'Unknown FST type ilabel_lookahead' in str(error)):
            pytest.skip('OpenFst lookahead plugin is not installed')
        raise
    return directory


@pytest.fixture(scope='module')
def native_bundle(normalizers, tmp_path_factory):
    return _export_runtime(normalizers[1], tmp_path_factory.mktemp('deferred-native') / 'runtime', lookahead=False)


@pytest.fixture(scope='module')
def lookahead_bundle(normalizers, tmp_path_factory):
    return _export_runtime(normalizers[1], tmp_path_factory.mktemp('deferred-lookahead') / 'runtime', lookahead=True)


@pytest.fixture(scope='module', params=['raw', 'native', 'lookahead'])
def deferred(request, normalizers):
    if request.param == 'raw':
        return normalizers[1]
    directory = request.getfixturevalue(request.param + '_bundle')
    return Normalizer(input_case='cased', lang='en', runtime_dir=str(directory))


@pytest.fixture(scope='module', params=['native_bundle', 'lookahead_bundle'], ids=['native', 'lookahead'])
def runtime_directory(request):
    return request.getfixturevalue(request.param)


def test_constructor_pairs_raw_cardinal_grammars_without_changing_the_default(normalizers):
    ordinary, deferred = normalizers
    native_tag = Normalizer.select_tag(ordinary.find_tags('42'))
    raw_tag = Normalizer.select_tag(deferred.find_tags('42'))
    assert 'integer: "forty two"' in native_tag
    assert 'integer: "42"' in raw_tag
    assert ordinary.normalize('42') == deferred.normalize('42') == 'forty two'
    assert deferred.last_fallback_reason is None


@pytest.mark.parametrize('options', [{'lang': 'vi'}, {'deterministic': False}, {'lm': True}])
def test_deferred_constructor_rejects_unsupported_grammars(options):
    with pytest.raises(ValueError, match='deterministic English'):
        Normalizer(input_case='cased', defer_cardinal=True, **options)


def test_deferred_constructor_rejects_a_legacy_tagger_only_directory(tmp_path):
    with pytest.raises(ValueError, match='runtime_dir'):
        Normalizer(input_case='cased', defer_cardinal=True, lookahead_dir=str(tmp_path))


@pytest.mark.parametrize('defer_cardinal', [False, True])
def test_legacy_lookahead_argument_rejects_a_deferred_bundle_tagger(lookahead_bundle, grammar_cache, defer_cardinal):
    with pytest.raises(ValueError, match='runtime_dir'):
        Normalizer(
            input_case='cased',
            cache_dir=grammar_cache,
            lookahead_dir=str(lookahead_bundle / 'raw/lookahead'),
            defer_cardinal=defer_cardinal,
        )


@pytest.mark.parametrize(
    'options',
    [{'lang': 'vi'}, {'deterministic': False}, {'lm': True}, {'lookahead_dir': 'tagger'}, {'whitelist': 'words.tsv'}],
)
def test_runtime_constructor_rejects_incompatible_options(tmp_path, options):
    with pytest.raises(ValueError):
        Normalizer(input_case='cased', runtime_dir=str(tmp_path), **options)


@pytest.mark.parametrize('text,expected', SERIAL_CASES)
def test_serial_ties_preserve_the_exact_ordinary_reading(deferred, baseline_output, text, expected):
    assert baseline_output(text) == expected
    assert deferred.normalize(text) == expected
    assert deferred.last_fallback_reason == 'ambiguous_rewrite'


@pytest.mark.parametrize('text', LOOKAHEAD_COUNTEREXAMPLES, ids=[f'D{i:02d}' for i in range(3, 11)])
def test_lookahead_counterexamples_fall_back_to_the_ordinary_reading(deferred, baseline_output, text):
    assert deferred.normalize(text) == baseline_output(text)
    assert deferred.last_fallback_reason == 'ambiguous_rewrite'


@pytest.mark.parametrize('case', REPLAY_CASES, ids=[f'replay-{i:02d}' for i in range(len(REPLAY_CASES))])
def test_replay_preserves_unicode_literals_and_punctuation(deferred, baseline_output, case):
    options = {name: case[name] for name in ('punct_pre_process', 'punct_post_process')}
    assert deferred.normalize(case['text'], **options) == baseline_output(case['text'], **options)


@pytest.mark.parametrize('punct_pre_process', [False, True])
@pytest.mark.parametrize('punct_post_process', [False, True])
def test_fallback_keeps_the_requested_punctuation_processing(
    deferred, baseline_output, punct_pre_process, punct_post_process
):
    text = '[31/31/100], 42!'
    options = dict(punct_pre_process=punct_pre_process, punct_post_process=punct_post_process)
    assert deferred.normalize(text, **options) == baseline_output(text, **options)


@pytest.mark.parametrize('text', ['1000000000000', '1,000,000,000,000', '12345678.12345'])
def test_numbers_over_twelve_digits_use_the_native_fallback(deferred, baseline_output, text):
    assert deferred.normalize(text) == baseline_output(text)
    assert deferred.last_fallback_reason == 'numeric_scope'


@pytest.mark.parametrize('text', ['999999999999', '12345678 and 87654321', '12345678-87654321'])
def test_numeric_scope_is_counted_per_span(deferred, baseline_output, text):
    assert deferred.normalize(text) == baseline_output(text)
    assert deferred.last_fallback_reason != 'numeric_scope'


def test_fallback_reason_is_reset_for_each_call(deferred):
    deferred.normalize('1000000000000')
    assert deferred.last_fallback_reason == 'numeric_scope'
    assert deferred.normalize('42') == 'forty two'
    assert deferred.last_fallback_reason is None
    deferred.normalize('31/31/100')
    assert deferred.last_fallback_reason == 'ambiguous_rewrite'
    assert deferred.normalize('  ') == ''
    assert deferred.last_fallback_reason is None


@pytest.mark.parametrize('stage', ['tagger', 'verbalizer'])
def test_either_ambiguous_stage_restarts_the_ordinary_pipeline(normalizers, stage):
    # Force each stage independently: the real serial fixtures primarily cover
    # tagger ambiguity, while verbalizer ambiguity must also escape its handler.
    normalizer = copy.copy(normalizers[1])
    normalizer.parser = TokenParser()
    tag = 'tokens { name: "candidate" }'
    normalizer.parser(tag)
    serialized = next(normalizer.generate_permutations(normalizer.parser.parse()))
    tagger = pynini.cross('42', pynini.escape(tag))
    verbalizer = pynini.cross(pynini.escape(serialized), 'candidate')
    if stage == 'tagger':
        tagger |= pynini.cross('42', 'tokens { name: "other" }')
    else:
        verbalizer |= pynini.cross(pynini.escape(serialized), 'other')
    normalizer.tagger = SimpleNamespace(fst=tagger)
    normalizer.verbalizer = SimpleNamespace(fst=verbalizer)
    assert normalizer.normalize('42') == 'forty two'
    assert normalizer.last_fallback_reason == 'ambiguous_rewrite'


@pytest.mark.parametrize('post_process', [False, True])
def test_runtime_loads_the_complete_pair_without_building_grammars(
    runtime_directory, normalizers, monkeypatch, post_process
):
    def unexpected_build(*args, **kwargs):
        pytest.fail('Loading an exported runtime must not rebuild source grammars')

    monkeypatch.setattr(
        'nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify.ClassifyFst', unexpected_build
    )
    monkeypatch.setattr(
        'nemo_text_processing.text_normalization.en.verbalizers.verbalize_final.VerbalizeFinalFst', unexpected_build
    )
    monkeypatch.setattr(
        'nemo_text_processing.text_normalization.en.verbalizers.post_processing.PostProcessingFst', unexpected_build
    )
    loaded = Normalizer(
        input_case='cased', runtime_dir=str(runtime_directory), defer_cardinal=True, post_process=post_process
    )
    baseline = copy.copy(normalizers[0])
    if not post_process:
        baseline.post_processor = None
    for text in ['42', '31/31/100', '1000000000000', 'hello, 42!']:
        assert loaded.normalize(text) == baseline.normalize(text)


def test_runtime_rejects_an_input_case_mismatch(native_bundle):
    with pytest.raises(ValueError, match='input_case mismatch'):
        Normalizer(input_case='lower_cased', runtime_dir=str(native_bundle))


@pytest.mark.parametrize('source_state', ['modified', 'deleted'])
def test_whitelist_export_preserves_the_compiled_contents_after_source_changes(
    custom_whitelist_normalizer, tmp_path, source_state
):
    normalizer, source, contents = custom_whitelist_normalizer
    changed_contents = b'NVToken\tchanged replacement\n1000000000000\tchanged large number\n'
    source.write_bytes(contents)
    if source_state == 'modified':
        source.write_bytes(changed_contents)
    else:
        source.unlink()

    directory = normalizer.export_runtime(tmp_path / 'runtime', lookahead=False)
    expected_whitelist = {'kind': 'custom', 'sha256': hashlib.sha256(contents).hexdigest()}
    assert json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['whitelist'] == expected_whitelist

    # Changing the file again after export must not change either the paired
    # grammars or the provenance carried into a re-export from a loaded bundle.
    if source.exists():
        source.unlink()
    else:
        source.write_bytes(changed_contents)
    restored = Normalizer(input_case='cased', runtime_dir=str(directory))
    for candidate in (normalizer, restored):
        assert candidate.normalize('NVToken') == 'captured replacement'
        assert candidate.last_fallback_reason is None
        assert candidate.normalize('1000000000000') == 'captured large number'
        assert candidate.last_fallback_reason == 'numeric_scope'

    reexported = restored.export_runtime(tmp_path / 'reexported', lookahead=False)
    assert json.loads((reexported / 'manifest.json').read_text(encoding='utf-8'))['whitelist'] == expected_whitelist
    reloaded = Normalizer(input_case='cased', runtime_dir=str(reexported))
    assert reloaded.normalize('NVToken') == 'captured replacement'
    assert reloaded.normalize('1000000000000') == 'captured large number'


def test_ordinary_pipeline_cannot_export_a_deferred_runtime(normalizers, tmp_path):
    with pytest.raises(ValueError, match='defer_cardinal'):
        normalizers[0].export_runtime(tmp_path / 'ordinary')
    assert not (tmp_path / 'ordinary').exists()


def test_lookahead_runtime_survives_pickle_and_process_workers(lookahead_bundle, baseline_output):
    normalizer = Normalizer(input_case='cased', runtime_dir=str(lookahead_bundle))
    restored = pickle.loads(pickle.dumps(normalizer))
    assert restored.tagger.fst.fst_type() == 'ilabel_lookahead'
    texts = ['42', '31/31/100', '1000000000000', 'https://a.b/1/中文']
    expected = [baseline_output(text) for text in texts]
    assert [restored.normalize(text) for text in texts] == expected
    assert restored.normalize_list(texts, n_jobs=2, batch_size=1) == expected
