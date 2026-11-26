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

"""Versioned, paired artifacts for the opt-in English deferred-cardinal runtime."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pynini

from nemo_text_processing.text_normalization.lookahead import LookaheadTagger, export_lookahead_tagger

SCHEMA_VERSION = 1
GRAMMAR_ABI = 'english_deferred_cardinal_v1'
MAX_NUMERIC_DIGITS = 12
_KIND = 'nemo_deferred_cardinal_runtime'
_ENGLISH_GRAMMAR_DIRECTORY = Path(__file__).parent / 'en'
_BASE_FILES = {'raw/verbalizer.fst', 'baseline/tagger.fst', 'baseline/verbalizer.fst'}
_LOOKAHEAD_FILES = {
    'raw/lookahead/tagger.fst',
    'raw/lookahead/relabel.pairs',
    'raw/lookahead/metadata.json',
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def deferred_cache_suffix(whitelist=None):
    """Keep deferred cache ABI and whitelist contents separate from legacy caches."""
    suffix = '_raw_cardinal_v1'
    if whitelist:
        suffix += '_whitelist_' + _sha256(whitelist)
    return suffix


def deferred_cache_directory(cache_dir):
    """Keep raw, baseline, and post-processing caches on one source revision.

    Only deferred public runtimes use this namespace; ordinary grammar cache
    locations and filenames retain their established behavior.
    """
    if cache_dir is None or cache_dir == 'None':
        return cache_dir
    source_root = _ENGLISH_GRAMMAR_DIRECTORY.resolve()
    cache_root = Path(cache_dir).resolve()
    try:
        cache_root.relative_to(source_root)
    except ValueError:
        excluded_root = None
    else:
        # A cache inside the grammar package must not fingerprint its own
        # generated FARs. When both roots coincide, retain the source files
        # and exclude only the namespace this helper creates.
        excluded_root = source_root / 'deferred_cardinal_v1' if cache_root == source_root else cache_root
    digest = hashlib.sha256()
    digest.update((GRAMMAR_ABI + '\0' + str(pynini.__version__) + '\0').encode('utf-8'))
    sources = sorted(
        path
        for path in source_root.rglob('*')
        if path.is_file()
        and path.suffix in ('.py', '.tsv', '.far')
        and (excluded_root is None or excluded_root not in path.parents)
    )
    for path in sources:
        digest.update(path.relative_to(source_root).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(_sha256(path).encode('ascii'))
        digest.update(b'\n')
    return str(Path(cache_dir) / 'deferred_cardinal_v1' / digest.hexdigest())


def _read_manifest(directory, *, language=None, input_case=None, expected_sha256=None):
    directory = Path(directory)
    try:
        manifest_path = directory / 'manifest.json'
        manifest_sha256 = _sha256(manifest_path)
        if expected_sha256 is not None and manifest_sha256 != expected_sha256:
            raise ValueError('Deferred runtime manifest changed since the bundle was loaded')
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(metadata, dict):
            raise ValueError('Deferred runtime manifest must be a JSON object')
        required = {
            'kind': _KIND,
            'schema_version': SCHEMA_VERSION,
            'grammar_abi': GRAMMAR_ABI,
            'language': 'en',
            'mode': 'tn',
            'deterministic': True,
            'defer_cardinal': True,
            'max_numeric_digits': MAX_NUMERIC_DIGITS,
        }
        for key, expected in required.items():
            value = metadata.get(key)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f'Unsupported deferred runtime {key}: {value!r}; expected {expected!r}')
        if metadata.get('input_case') not in ('cased', 'lower_cased'):
            raise ValueError('Unsupported deferred runtime input_case')
        for key, expected in (('language', language), ('input_case', input_case)):
            if expected is not None and metadata.get(key) != expected:
                raise ValueError(
                    f'Deferred runtime {key} mismatch: expected {expected!r}, found {metadata.get(key)!r}'
                )
        backend = metadata.get('backend')
        if backend not in ('native', 'lookahead'):
            raise ValueError(f'Unsupported deferred runtime backend: {backend!r}')
        if type(metadata.get('post_process')) is not bool:
            raise ValueError('Deferred runtime manifest must specify post_process')
        whitelist = metadata.get('whitelist')
        if not isinstance(whitelist, dict) or whitelist.get('kind') not in ('default', 'custom'):
            raise ValueError('Invalid deferred runtime whitelist metadata')
        if whitelist['kind'] == 'custom':
            digest = whitelist.get('sha256')
            if not _is_digest(digest):
                raise ValueError('Invalid deferred runtime whitelist checksum')
        files = _BASE_FILES | (_LOOKAHEAD_FILES if backend == 'lookahead' else {'raw/tagger.fst'})
        if metadata['post_process']:
            files |= {'post_processor.fst'}
        checksums = metadata.get('sha256')
        if not isinstance(checksums, dict) or set(checksums) != files:
            raise ValueError('Deferred runtime artifact list does not match its configuration')
        for name in sorted(files):
            expected = checksums[name]
            if not _is_digest(expected) or _sha256(directory / name) != expected:
                raise ValueError(f'Deferred runtime artifact checksum mismatch: {name}')
        return metadata, manifest_sha256
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read deferred runtime bundle {directory}: {error}') from error


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


class _BundleGraph:
    """A native graph whose worker reload remains bound to its complete bundle."""

    def __init__(self, directory, filename, manifest_sha256):
        self.directory = str(Path(directory).resolve())
        self.filename = filename
        self.manifest_sha256 = manifest_sha256
        self._load()

    def _load(self):
        metadata, _ = _read_manifest(self.directory, expected_sha256=self.manifest_sha256)
        if self.filename not in metadata['sha256'] or not self.filename.endswith('.fst'):
            raise ValueError(f'Graph is not part of the deferred runtime bundle: {self.filename}')
        try:
            self.fst = pynini.Fst.read(str(Path(self.directory) / self.filename))
        except Exception as error:
            raise ValueError(f'Cannot load deferred runtime graph {self.filename}: {error}') from error
        if self.fst.arc_type() != 'standard' or not self.fst.verify():
            raise ValueError(f'Invalid deferred runtime graph: {self.filename}')

    def __getstate__(self):
        return {
            'directory': self.directory,
            'filename': self.filename,
            'manifest_sha256': self.manifest_sha256,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()


class _BundleLookaheadTagger(LookaheadTagger):
    """Retain OpenFst's matcher and bind every reload to the paired bundle."""

    def __init__(self, directory, *, language, input_case, manifest_sha256):
        self.bundle_directory = str(Path(directory).resolve())
        self.bundle_manifest_sha256 = manifest_sha256
        super().__init__(Path(directory) / 'raw/lookahead', language=language, input_case=input_case, mode='tn')

    def _load(self):
        _read_manifest(self.bundle_directory, expected_sha256=self.bundle_manifest_sha256)
        super()._load()
        if self.metadata.get('defer_cardinal') is not True or self.metadata.get('grammar_abi') != GRAMMAR_ABI:
            raise ValueError('Deferred runtime lookahead metadata must identify the matching deferred-cardinal ABI')
        if self.fst.arc_type() != 'standard' or not self.fst.verify():
            raise ValueError('Invalid deferred runtime lookahead graph')

    def __getstate__(self):
        state = super().__getstate__()
        state.update(
            bundle_directory=self.bundle_directory,
            bundle_manifest_sha256=self.bundle_manifest_sha256,
        )
        return state


def load_deferred_runtime(directory, *, language, input_case):
    """Validate and load an immutable raw/baseline pair without building grammars.

    The baseline pair is part of the same bundle so inputs outside the evaluated
    numeric window never depend on whichever source grammars happen to be installed.
    """
    directory = Path(directory).resolve()
    metadata, manifest_sha256 = _read_manifest(directory, language=language, input_case=input_case)

    def graph(filename):
        return _BundleGraph(directory, filename, manifest_sha256)

    tagger = (
        _BundleLookaheadTagger(directory, language=language, input_case=input_case, manifest_sha256=manifest_sha256)
        if metadata['backend'] == 'lookahead'
        else graph('raw/tagger.fst')
    )
    return SimpleNamespace(
        tagger=tagger,
        verbalizer=graph('raw/verbalizer.fst'),
        baseline_tagger=graph('baseline/tagger.fst'),
        baseline_verbalizer=graph('baseline/verbalizer.fst'),
        post_processor=graph('post_processor.fst') if metadata['post_process'] else None,
        max_numeric_digits=metadata['max_numeric_digits'],
        manifest_sha256=manifest_sha256,
        whitelist=metadata['whitelist'],
    )


def export_deferred_runtime(normalizer, output_dir, *, lookahead=True):
    """Atomically publish the raw graphs, original baseline, and their contract.

    ``normalizer`` must be a deferred English Normalizer with its original
    ``_baseline_normalizer`` retained. Existing output directories are never reused.
    """
    if not getattr(normalizer, 'defer_cardinal', False) or normalizer.lang != 'en':
        raise ValueError('Export requires a deterministic English deferred-cardinal Normalizer')
    baseline = getattr(normalizer, '_baseline_normalizer', None)
    if baseline is None:
        raise ValueError('Export requires the original paired baseline Normalizer')
    if normalizer.input_case not in ('cased', 'lower_cased'):
        raise ValueError('Unsupported deferred runtime input_case')
    if type(lookahead) is not bool:
        raise ValueError('lookahead must be a boolean')
    if isinstance(baseline.tagger, LookaheadTagger):
        raise ValueError('The paired baseline must use the original native tagger')
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Deferred runtime export already exists: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.deferred-runtime-', dir=output_dir.parent) as temporary:
        temporary = Path(temporary)
        (temporary / 'raw').mkdir()
        (temporary / 'baseline').mkdir()
        files = set(_BASE_FILES)
        if lookahead:
            matcher_dir = temporary / 'raw/lookahead'
            if isinstance(normalizer.tagger, LookaheadTagger):
                if isinstance(normalizer.tagger, _BundleLookaheadTagger):
                    _read_manifest(
                        normalizer.tagger.bundle_directory,
                        expected_sha256=normalizer.tagger.bundle_manifest_sha256,
                    )
                matcher_dir.mkdir()
                for name in ('tagger.fst', 'relabel.pairs', 'metadata.json'):
                    shutil.copyfile(Path(normalizer.tagger.directory) / name, matcher_dir / name)
            else:
                export_lookahead_tagger(
                    normalizer.tagger.fst,
                    matcher_dir,
                    language=normalizer.lang,
                    input_case=normalizer.input_case,
                    mode='tn',
                )
            matcher_metadata_path = matcher_dir / 'metadata.json'
            matcher_metadata = json.loads(matcher_metadata_path.read_text(encoding='utf-8'))
            matcher_metadata.update(defer_cardinal=True, grammar_abi=GRAMMAR_ABI)
            matcher_metadata_path.write_text(json.dumps(matcher_metadata, indent=2) + '\n', encoding='utf-8')
            files |= _LOOKAHEAD_FILES
        else:
            if isinstance(normalizer.tagger, LookaheadTagger):
                raise ValueError('A loaded lookahead matcher cannot be exported as an ordinary native tagger')
            normalizer.tagger.fst.write(str(temporary / 'raw/tagger.fst'))
            files.add('raw/tagger.fst')
        normalizer.verbalizer.fst.write(str(temporary / 'raw/verbalizer.fst'))
        baseline.tagger.fst.write(str(temporary / 'baseline/tagger.fst'))
        baseline.verbalizer.fst.write(str(temporary / 'baseline/verbalizer.fst'))
        post_processor = getattr(normalizer, 'post_processor', None)
        if post_processor is not None:
            post_processor.fst.write(str(temporary / 'post_processor.fst'))
            files.add('post_processor.fst')
        whitelist = getattr(normalizer, 'whitelist', None)
        runtime_bundle = getattr(normalizer, '_runtime_bundle', None)
        if runtime_bundle is not None:
            whitelist_metadata = runtime_bundle.whitelist
        else:
            whitelist_sha256 = getattr(normalizer, '_whitelist_sha256', None)
            if whitelist:
                if whitelist_sha256 is None:
                    whitelist_sha256 = _sha256(whitelist)
                whitelist_metadata = {'kind': 'custom', 'sha256': whitelist_sha256}
            else:
                whitelist_metadata = {'kind': 'default'}
        metadata = {
            'kind': _KIND,
            'schema_version': SCHEMA_VERSION,
            'grammar_abi': GRAMMAR_ABI,
            'language': normalizer.lang,
            'input_case': normalizer.input_case,
            'mode': 'tn',
            'deterministic': True,
            'defer_cardinal': True,
            'max_numeric_digits': MAX_NUMERIC_DIGITS,
            'backend': 'lookahead' if lookahead else 'native',
            'post_process': post_processor is not None,
            'whitelist': whitelist_metadata,
            'sha256': {name: _sha256(temporary / name) for name in sorted(files)},
        }
        (temporary / 'manifest.json').write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        # Loading validates graph types and the pair before one directory rename
        # makes any of the artifacts visible at the destination.
        load_deferred_runtime(temporary, language=normalizer.lang, input_case=normalizer.input_case)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f'Deferred runtime export already exists: {output_dir}')
        temporary.rename(output_dir)
    return output_dir
