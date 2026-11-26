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

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import pynini
import pytest

from nemo_text_processing.text_normalization.deferred_runtime import (
    deferred_cache_directory,
    deferred_cache_suffix,
    export_deferred_runtime,
    load_deferred_runtime,
)
from nemo_text_processing.text_normalization.lookahead import LookaheadTagger

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on('CPU')]


def _graph(source, target):
    return SimpleNamespace(fst=pynini.cross(pynini.escape(source), pynini.escape(target)))


@pytest.fixture
def source_normalizer():
    return SimpleNamespace(
        lang='en',
        input_case='cased',
        defer_cardinal=True,
        whitelist=None,
        tagger=_graph('12', 'raw12'),
        verbalizer=_graph('raw12', 'twelve'),
        post_processor=_graph('twelve', 'TWELVE'),
        _baseline_normalizer=SimpleNamespace(
            tagger=_graph('12', 'spoken12'), verbalizer=_graph('spoken12', 'baseline twelve')
        ),
    )


@pytest.fixture(params=[False, True], ids=['native', 'lookahead'])
def bundle(request, source_normalizer, tmp_path):
    directory = tmp_path / 'runtime'
    try:
        export_deferred_runtime(source_normalizer, directory, lookahead=request.param)
    except RuntimeError as error:
        if request.param and (
            'fstconvert is required' in str(error) or 'Unknown FST type ilabel_lookahead' in str(error)
        ):
            pytest.skip('OpenFst lookahead plugin is not installed')
        raise
    return directory


def _rewrite(tagger, text):
    lattice = tagger.compose(pynini.escape(text)) if isinstance(tagger, LookaheadTagger) else text @ tagger.fst
    return pynini.shortestpath(lattice).string()


def _load(directory):
    return load_deferred_runtime(directory, language='en', input_case='cased')


def test_bundle_roundtrip_contains_both_complete_pairs(bundle):
    runtime = _load(bundle)
    assert _rewrite(runtime.tagger, '12') == 'raw12'
    assert _rewrite(runtime.verbalizer, 'raw12') == 'twelve'
    assert _rewrite(runtime.baseline_tagger, '12') == 'spoken12'
    assert _rewrite(runtime.baseline_verbalizer, 'spoken12') == 'baseline twelve'
    assert _rewrite(runtime.post_processor, 'twelve') == 'TWELVE'
    assert runtime.max_numeric_digits == 12
    assert runtime.manifest_sha256 == hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest()


def test_bundle_graphs_pickle_and_retain_matcher(bundle):
    before = _load(bundle)
    restored = pickle.loads(pickle.dumps(before))
    assert _rewrite(restored.tagger, '12') == 'raw12'
    assert _rewrite(restored.baseline_tagger, '12') == 'spoken12'
    assert _rewrite(restored.verbalizer, 'raw12') == 'twelve'
    if isinstance(before.tagger, LookaheadTagger):
        assert restored.tagger.fst.fst_type() == 'ilabel_lookahead'


def test_nested_matcher_retains_the_deferred_pair_contract(bundle):
    metadata_path = bundle / 'raw/lookahead/metadata.json'
    if not metadata_path.exists():
        assert json.loads((bundle / 'manifest.json').read_text())['backend'] == 'native'
        return
    metadata = json.loads(metadata_path.read_text())
    assert metadata['defer_cardinal'] is True
    assert metadata['grammar_abi'] == 'english_deferred_cardinal_v1'


@pytest.mark.parametrize(
    'field,value',
    [('defer_cardinal', False), ('defer_cardinal', None), ('grammar_abi', 'other'), ('grammar_abi', None)],
)
def test_bundle_rejects_missing_or_mismatched_nested_cardinal_contract(bundle, field, value):
    metadata_path = bundle / 'raw/lookahead/metadata.json'
    if not metadata_path.exists():
        assert json.loads((bundle / 'manifest.json').read_text())['backend'] == 'native'
        return
    metadata = json.loads(metadata_path.read_text())
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256']['raw/lookahead/metadata.json'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='matching deferred-cardinal ABI'):
        _load(bundle)


@pytest.mark.parametrize(
    'component', ['tagger', 'verbalizer', 'baseline_tagger', 'baseline_verbalizer', 'post_processor']
)
def test_each_pickled_graph_rejects_replacement_bundle(bundle, component):
    serialized = pickle.dumps(getattr(_load(bundle), component))
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['provenance'] = 'a different valid manifest'
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    # A new caller can load the newly identified bundle; an existing worker
    # must retain the exact identity selected by its original parent process.
    _load(bundle)
    with pytest.raises(ValueError, match='manifest changed'):
        pickle.loads(serialized)


@pytest.mark.parametrize('filename', ['raw/verbalizer.fst', 'baseline/tagger.fst', 'post_processor.fst'])
def test_corrupt_graphs_fail_before_returning_a_runtime(bundle, filename):
    (bundle / filename).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        _load(bundle)


def test_missing_graphs_fail_with_bundle_context(bundle):
    (bundle / 'baseline/verbalizer.fst').unlink()
    with pytest.raises(ValueError, match='Cannot read deferred runtime bundle'):
        _load(bundle)


@pytest.mark.parametrize(
    'field,value',
    [('schema_version', 2), ('grammar_abi', 'other'), ('defer_cardinal', False), ('max_numeric_digits', 13)],
)
def test_unsupported_contract_is_rejected(bundle, field, value):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest[field] = value
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        _load(bundle)


@pytest.mark.parametrize('language,input_case', [('de', 'cased'), ('en', 'lower_cased')])
def test_bundle_must_match_requested_configuration(bundle, language, input_case):
    with pytest.raises(ValueError, match='mismatch'):
        load_deferred_runtime(bundle, language=language, input_case=input_case)


def test_existing_bundle_is_not_overwritten(bundle, source_normalizer):
    original = (bundle / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        export_deferred_runtime(source_normalizer, bundle, lookahead=False)
    assert (bundle / 'manifest.json').read_bytes() == original


def test_failed_export_never_publishes_a_partial_pair(source_normalizer, tmp_path, monkeypatch):
    import nemo_text_processing.text_normalization.deferred_runtime as deferred_runtime

    def fail(*args, **kwargs):
        raise RuntimeError('conversion failed')

    monkeypatch.setattr(deferred_runtime, 'export_lookahead_tagger', fail)
    with pytest.raises(RuntimeError, match='conversion failed'):
        export_deferred_runtime(source_normalizer, tmp_path / 'runtime')
    assert list(tmp_path.iterdir()) == []


def test_native_export_without_postprocessor(source_normalizer, tmp_path):
    source_normalizer.post_processor = None
    runtime = _load(export_deferred_runtime(source_normalizer, tmp_path / 'runtime', lookahead=False))
    assert runtime.post_processor is None


def test_custom_whitelist_is_recorded_by_contents(source_normalizer, tmp_path):
    whitelist = tmp_path / 'custom.tsv'
    whitelist.write_text('12\tdozen\n')
    source_normalizer.whitelist = str(whitelist)
    directory = export_deferred_runtime(source_normalizer, tmp_path / 'runtime', lookahead=False)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['whitelist'] == {'kind': 'custom', 'sha256': hashlib.sha256(whitelist.read_bytes()).hexdigest()}


@pytest.mark.parametrize('delete_source', [False, True])
def test_export_keeps_the_whitelist_fingerprint_captured_at_construction(source_normalizer, tmp_path, delete_source):
    whitelist = tmp_path / 'custom.tsv'
    whitelist.write_text('12\tdozen\n')
    original_digest = hashlib.sha256(whitelist.read_bytes()).hexdigest()
    source_normalizer.whitelist = str(whitelist)
    source_normalizer._whitelist_sha256 = original_digest
    if delete_source:
        whitelist.unlink()
    else:
        whitelist.write_text('12\tthe edited file\n')
    directory = export_deferred_runtime(source_normalizer, tmp_path / 'runtime', lookahead=False)
    assert _load(directory).whitelist == {'kind': 'custom', 'sha256': original_digest}


@pytest.mark.parametrize('lookahead', [False, True])
def test_loaded_pair_can_be_reexported_without_the_original_whitelist(source_normalizer, tmp_path, lookahead):
    whitelist = tmp_path / 'custom.tsv'
    whitelist.write_text('12\tdozen\n')
    source_normalizer.whitelist = str(whitelist)
    first = export_deferred_runtime(source_normalizer, tmp_path / 'first', lookahead=lookahead)
    runtime = _load(first)
    whitelist.unlink()
    loaded = SimpleNamespace(
        lang='en',
        input_case='cased',
        defer_cardinal=True,
        whitelist=None,
        tagger=runtime.tagger,
        verbalizer=runtime.verbalizer,
        post_processor=runtime.post_processor,
        _baseline_normalizer=SimpleNamespace(tagger=runtime.baseline_tagger, verbalizer=runtime.baseline_verbalizer),
        _runtime_bundle=runtime,
    )
    second = export_deferred_runtime(loaded, tmp_path / 'second', lookahead=lookahead)
    assert _load(second).whitelist == runtime.whitelist
    assert _rewrite(_load(second).tagger, '12') == 'raw12'


def test_cache_identity_tracks_whitelist_contents_not_basename(tmp_path):
    first = tmp_path / 'one' / 'custom.tsv'
    second = tmp_path / 'two' / 'custom.tsv'
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('a\tone\n')
    second.write_text('a\ttwo\n')
    original = deferred_cache_suffix(first)
    assert original != deferred_cache_suffix(second)
    shutil.copyfile(first, second)
    assert original == deferred_cache_suffix(second)
    first.write_text('a\tthree\n')
    assert original != deferred_cache_suffix(first)


def test_deferred_cache_directory_tracks_source_contents_and_paths(tmp_path, monkeypatch):
    import nemo_text_processing.text_normalization.deferred_runtime as deferred_runtime

    sources = tmp_path / 'english'
    sources.mkdir()
    graph = sources / 'graph.py'
    graph.write_text('original graph')
    monkeypatch.setattr(deferred_runtime, '_ENGLISH_GRAMMAR_DIRECTORY', sources)
    cache = tmp_path / 'cache'
    original = deferred_cache_directory(cache)
    assert original == deferred_cache_directory(cache)
    assert not cache.exists()
    graph.write_text('edited graph')
    assert original != deferred_cache_directory(cache)
    graph.write_text('original graph')
    assert original == deferred_cache_directory(cache)
    graph.rename(sources / 'renamed.py')
    assert original != deferred_cache_directory(cache)
    assert deferred_cache_directory(None) is None
    assert deferred_cache_directory('None') == 'None'


@pytest.mark.parametrize('cache_at_source_root', [False, True])
def test_deferred_cache_inside_sources_does_not_fingerprint_generated_fars(
    tmp_path, monkeypatch, cache_at_source_root
):
    import nemo_text_processing.text_normalization.deferred_runtime as deferred_runtime

    sources = tmp_path / 'english'
    sources.mkdir()
    source_graph = sources / 'number.far'
    source_graph.write_bytes(b'original source FAR')
    monkeypatch.setattr(deferred_runtime, '_ENGLISH_GRAMMAR_DIRECTORY', sources)
    cache = sources if cache_at_source_root else sources / 'cache'
    original_namespace = deferred_cache_directory(cache)
    generated_cache = Path(original_namespace) if cache_at_source_root else cache
    generated_cache.mkdir(parents=True)
    generated_graph = generated_cache / 'graph.far'
    generated_graph.write_bytes(b'generated compiled FAR')
    assert original_namespace == deferred_cache_directory(cache)
    generated_graph.write_bytes(b'edited generated compiled FAR')
    assert original_namespace == deferred_cache_directory(cache)
    source_graph.write_bytes(b'changed source FAR')
    assert original_namespace != deferred_cache_directory(cache)


def _legacy_files(tmp_path, pairs):
    (tmp_path / 'tagger.fst').write_bytes(b'not read for invalid table tests')
    (tmp_path / 'relabel.pairs').write_text(pairs)
    metadata = {
        'version': 1,
        'language': 'en',
        'input_case': 'cased',
        'mode': 'tn',
        'sha256': {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ('tagger.fst', 'relabel.pairs')
        },
    }
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize('pairs', ['1 2\n1 3\n', 'not integers\n', '1 2 3\n', '-1 2\n', '0 2\n'])
def test_legacy_lookahead_rejects_invalid_relabel_tables_before_loading_matcher(tmp_path, pairs):
    _legacy_files(tmp_path, pairs)
    with pytest.raises(ValueError, match='relabel table'):
        LookaheadTagger(tmp_path, language='en', input_case='cased', mode='tn')


@pytest.mark.parametrize('metadata', ['[]', '{broken', '{"version": 1}', 'null'])
def test_legacy_lookahead_reports_malformed_metadata(tmp_path, metadata):
    (tmp_path / 'metadata.json').write_text(metadata)
    with pytest.raises(ValueError, match='metadata|mismatch'):
        LookaheadTagger(tmp_path, language='en', input_case='cased', mode='tn')


@pytest.mark.parametrize('filename', ['metadata.json', 'tagger.fst', 'relabel.pairs'])
def test_legacy_lookahead_reports_missing_artifacts(tmp_path, filename):
    _legacy_files(tmp_path, '1 2\n')
    (tmp_path / filename).unlink()
    with pytest.raises(ValueError, match='Cannot read lookahead'):
        LookaheadTagger(tmp_path, language='en', input_case='cased', mode='tn')
