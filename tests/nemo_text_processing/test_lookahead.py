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

import pickle
import random
import subprocess
from types import SimpleNamespace

import pynini
import pytest
from joblib import Parallel, delayed
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.lookahead import LookaheadTagger, export_lookahead_tagger
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on('CPU')]


@pytest.fixture
def exported(tmp_path):
    mappings = [('12', 'twelve'), ('é', 'accent'), ('hà nội', 'Hanoi'), ('[12]', '[twelve]'), ('', 'empty')]
    graph = pynini.union(*(pynini.cross(pynini.escape(src), pynini.escape(dst)) for src, dst in mappings))
    graph |= pynutil.add_weight(pynini.cross('12', 'wrong'), 1.0)
    graph.optimize()
    directory = tmp_path / 'lookahead'
    try:
        export_lookahead_tagger(graph, directory, language='en', input_case='cased', mode='tn')
    except RuntimeError as error:
        if 'fstconvert is required' in str(error) or 'Unknown FST type ilabel_lookahead' in str(error):
            pytest.skip('OpenFst lookahead plugin is not installed')
        raise
    return graph, directory


@pytest.mark.parametrize('text', ['12', 'é', 'hà nội', '[12]', ''])
def test_composition_preserves_weighted_outputs_and_original_input(exported, text):
    graph, directory = exported
    tagger = LookaheadTagger(directory, language='en', input_case='cased', mode='tn')
    escaped = pynini.escape(text)
    lattice = tagger.compose(escaped)
    assert tagger.fst.fst_type() == 'ilabel_lookahead'
    assert pynini.shortestpath(lattice).string() == pynini.shortestpath(escaped @ graph).string()
    assert pynini.shortestpath(lattice.copy().project('input')).string() == text


@pytest.mark.parametrize('text', ['unknown', '\x01', '\x02', '\x03', '\x04', 'è'])
def test_unseen_labels_cannot_collide_with_relabelled_labels(exported, text):
    _, directory = exported
    tagger = LookaheadTagger(directory, language='en', input_case='cased', mode='tn')
    assert tagger.compose(pynini.escape(text)).num_states() == 0


def test_worker_serialization_retains_lookahead_matcher(exported):
    _, directory = exported
    tagger = LookaheadTagger(directory, language='en', input_case='cased', mode='tn')
    restored = pickle.loads(pickle.dumps(tagger))
    assert restored.fst.fst_type() == 'ilabel_lookahead'
    assert pynini.shortestpath(restored.compose('12')).string() == 'twelve'


def test_lookahead_composes_in_worker_processes(exported):
    _, directory = exported
    tagger = LookaheadTagger(directory, language='en', input_case='cased', mode='tn')

    def tag(text):
        return tagger.fst.fst_type(), pynini.shortestpath(tagger.compose(pynini.escape(text))).string()

    assert Parallel(n_jobs=2)(delayed(tag)(text) for text in ['12', 'é']) == [
        ('ilabel_lookahead', 'twelve'),
        ('ilabel_lookahead', 'accent'),
    ]


@pytest.mark.parametrize(
    'settings',
    [
        {'language': 'vi', 'input_case': 'cased', 'mode': 'tn'},
        {'language': 'en', 'input_case': 'lower_cased', 'mode': 'tn'},
        {'language': 'en', 'input_case': 'cased', 'mode': 'itn'},
    ],
)
def test_rejects_wrong_grammar_configuration(exported, settings):
    _, directory = exported
    with pytest.raises(ValueError, match='mismatch'):
        LookaheadTagger(directory, **settings)


def test_rejects_changed_relabel_table(exported):
    _, directory = exported
    (directory / 'relabel.pairs').write_text('1 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        LookaheadTagger(directory, language='en', input_case='cased', mode='tn')


def test_failed_conversion_does_not_publish_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, '', 'missing plugin'))
    directory = tmp_path / 'lookahead'
    with pytest.raises(RuntimeError, match='missing plugin'):
        export_lookahead_tagger(
            pynini.accep('text'), directory, language='en', input_case='cased', mode='tn', fstconvert='fstconvert'
        )
    assert not directory.exists()
    assert not list(tmp_path.iterdir())


def test_existing_export_is_not_overwritten(exported):
    graph, directory = exported
    original = (directory / 'tagger.fst').read_bytes()
    with pytest.raises(FileExistsError):
        export_lookahead_tagger(graph, directory, language='en', input_case='cased', mode='tn')
    assert (directory / 'tagger.fst').read_bytes() == original


def _export_graph(graph, directory):
    export_lookahead_tagger(graph, directory, language='en', input_case='cased', mode='tn')
    return LookaheadTagger(directory, language='en', input_case='cased', mode='tn')


def _output_weights(lattice):
    outputs = {}
    for _, output, weight in lattice.paths().items():
        outputs[output] = min(outputs.get(output, float('inf')), float(weight))
    return outputs


@pytest.mark.parametrize('text', ['', 'a', 'é'])
def test_empty_tagger_returns_an_empty_lattice(exported, tmp_path, text):
    tagger = _export_graph(pynini.Fst(), tmp_path / 'empty')
    assert tagger.compose(pynini.escape(text)).num_states() == 0


@pytest.mark.parametrize('lookahead', [False, True])
@pytest.mark.parametrize('post_process', [False, True])
def test_full_pipeline_preserves_unicode(exported, tmp_path, lookahead, post_process):
    value = 'hà nội é 🧑🏽‍💻'
    tagged = f'tokens {{ name: "{value}" }}'
    normalizer = Normalizer.__new__(Normalizer)
    normalizer.lang = 'en'
    normalizer.max_number_of_permutations_per_split = 729
    normalizer.parser = TokenParser()
    normalizer.parser(tagged)
    serialized = next(normalizer.generate_permutations(normalizer.parser.parse()))
    tagger = pynini.cross(pynini.escape(value), pynini.escape(tagged))
    normalizer.tagger = (
        _export_graph(tagger, tmp_path / 'unicode-pipeline') if lookahead else SimpleNamespace(fst=tagger)
    )
    normalizer.verbalizer = SimpleNamespace(fst=pynini.cross(pynini.escape(serialized), pynini.escape(value)))
    normalizer.post_processor = (
        SimpleNamespace(fst=pynini.cross(pynini.escape(value), pynini.escape(value + '!'))) if post_process else None
    )
    assert normalizer.normalize(value) == value + ('!' if post_process else '')


@pytest.mark.parametrize('loop_weight', [0.0, 0.5, 2.0])
def test_epsilon_cycles_preserve_the_best_path(exported, tmp_path, loop_weight):
    graph = pynini.cross('', 'prefix ') + pynini.cross('a', 'A') + pynini.cross('', ' suffix')
    graph.add_arc(graph.start(), pynini.Arc(0, 0, pynini.Weight('tropical', loop_weight), graph.start()))
    tagger = _export_graph(graph, tmp_path / 'cycle')
    for text in ['', 'a', 'aa', 'é']:
        baseline = pynini.accep(text) @ graph
        candidate = tagger.compose(text)
        assert (candidate.num_states() == 0) == (baseline.num_states() == 0)
        if candidate.num_states():
            assert pynini.shortestpath(candidate).string() == 'prefix A suffix'
            assert float(pynini.shortestdistance(candidate, reverse=True)[candidate.start()]) == pytest.approx(0)


def test_epsilon_only_tagger_and_final_weights(exported, tmp_path):
    graph = pynutil.add_weight(pynini.cross('', 'inserted'), -0.5)
    tagger = _export_graph(graph, tmp_path / 'epsilon')
    assert _output_weights(tagger.compose('')) == {'inserted': -0.5}
    for codepoint in range(1, 128):
        assert tagger.compose(pynini.escape(chr(codepoint))).num_states() == 0


def test_unreachable_and_nonfinal_states_do_not_create_matches(exported, tmp_path):
    graph = pynini.cross('a', 'A')
    dead = graph.add_state()
    graph.add_arc(graph.start(), pynini.Arc(ord('b'), ord('B'), pynini.Weight.one('tropical'), dead))
    unreachable = graph.add_state()
    graph.add_arc(unreachable, pynini.Arc(255, 255, pynini.Weight.one('tropical'), unreachable))
    tagger = _export_graph(graph, tmp_path / 'unreachable')
    for codepoint in range(1, 128):
        text = pynini.escape(chr(codepoint))
        assert _output_weights(tagger.compose(text)) == _output_weights(text @ graph)


@pytest.mark.parametrize(
    'text',
    [
        'é',
        'e\u0301',
        'hà nội',
        'ha\u0300 no\u0323\u0302i',
        '日本語',
        '한국어',
        '中文１２３',
        '🧑🏽‍💻',
        '\u200b\u200d\ufeff',
        '\t\n\r\u00a0\u202f',
        '[0] [EPS] [1234] [<eps>]',
        '" } tokens { name: "injected" } \\ []',
        'x' * 4096,
    ],
)
def test_literal_input_and_original_labels_are_preserved(exported, tmp_path, text):
    escaped = pynini.escape(text)
    tagger = _export_graph(pynini.cross(escaped, escaped), tmp_path / 'literal')
    lattice = tagger.compose(escaped)
    assert pynini.shortestpath(lattice).string() == text
    assert pynini.shortestpath(lattice.copy().project('input')).string() == text


@pytest.mark.parametrize('seed', range(20))
def test_seeded_weighted_relations_preserve_every_output(exported, tmp_path, seed):
    rng = random.Random(seed)
    alphabet = ['a', 'b', '1', 'é', '中', '[', ']', '\\', '\x01']
    inputs = ['', 'a', 'ab', 'é']
    inputs += [''.join(rng.choices(alphabet, k=rng.randint(0, 6))) for _ in range(20)]
    graph = pynini.Fst()
    for index, text in enumerate(inputs):
        output = '' if index % 4 == 0 else f'out {index} é'
        # Binary fractions avoid conflating relabeling bugs with floating-point
        # cancellation. Duplicate inputs exercise weighted alternatives and ties.
        path = pynini.cross(pynini.escape(text), output)
        graph |= pynutil.add_weight(path, rng.choice([-0.5, 0.0, 0.125, 0.5, 2.0]))
    tagger = _export_graph(graph, tmp_path / f'random-{seed}')
    inputs += [''.join(rng.choices(alphabet, k=rng.randint(0, 7))) for _ in range(25)]
    for text in inputs:
        escaped = pynini.escape(text)
        baseline = _output_weights(escaped @ graph)
        candidate = _output_weights(tagger.compose(escaped))
        assert candidate == pytest.approx(baseline), (seed, repr(text), baseline, candidate)


def test_collision_avoidance_rows_are_required(exported, tmp_path):
    tagger = _export_graph(pynini.cross('a', 'A'), tmp_path / 'collision')
    assert tagger.compose('\x01').num_states() == 0
    # A deliberately broken implementation that keeps only a -> 1 would accept
    # the unrelated control byte 1. This checks that the test detects that bug.
    missing_rows = [(old, new) for old, new in tagger.relabel_pairs if old == ord('a')]
    text = pynini.accep('\x01').relabel_pairs(opairs=missing_rows)
    import pywrapfst

    broken = pynini.Fst.from_pywrapfst(pywrapfst.compose(text, tagger.fst))
    assert pynini.shortestpath(broken).string() == 'A'
