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

import pynini
import pytest

from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst as CardinalTagger
from nemo_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst as CardinalVerbalizer

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on('CPU')]


def _best(graph):
    if graph.start() == pynini.NO_STATE_ID:
        return None
    return (
        pynini.shortestpath(graph).string(),
        float(pynini.shortestdistance(graph, reverse=True)[graph.start()]),
    )


@pytest.fixture(scope='module')
def cardinal_graphs():
    native = CardinalTagger()
    deferred = CardinalTagger(defer_verbalization=True)
    return native, deferred, CardinalVerbalizer().fst, CardinalVerbalizer(number_fst=native.final_graph).fst


@pytest.mark.parametrize(
    'text',
    ['0', '-0', '00', '0001', '12', '-123', '125', '999', '1,234', '12345', '1,000,000', '12345678901234567890'],
)
def test_deferred_cardinal_keeps_classification_costs_and_readings(cardinal_graphs, text):
    native, deferred, verbalizer, raw_verbalizer = cardinal_graphs
    native_tags = _best(text @ native.fst)
    raw_tags = _best(text @ deferred.fst)
    assert native_tags is not None and raw_tags is not None
    assert raw_tags[1] == pytest.approx(native_tags[1], abs=1e-6)
    assert (
        _best(pynini.escape(raw_tags[0]) @ raw_verbalizer)[0] == _best(pynini.escape(native_tags[0]) @ verbalizer)[0]
    )
    raw_digits = text[1:] if text.startswith('-') else text
    assert f'integer: "{raw_digits}"' in raw_tags[0]


@pytest.mark.parametrize('text', ['', '-', '1,23', '1.25', 'hello', '１２３'])
def test_deferred_cardinal_does_not_broaden_accepted_inputs(cardinal_graphs, text):
    native, deferred, _, _ = cardinal_graphs
    assert bool((pynini.escape(text) @ deferred.fst).num_states()) == bool(
        (pynini.escape(text) @ native.fst).num_states()
    )


def test_deferred_cardinal_leaves_embedded_numeric_graphs_unchanged(cardinal_graphs):
    native, deferred, _, _ = cardinal_graphs
    for text in ['12', '123', '12345', '0001']:
        for attribute in ['graph', 'long_numbers', 'final_graph']:
            before = _best(text @ getattr(native, attribute))
            after = _best(text @ getattr(deferred, attribute))
            assert before == after


def test_deferred_cardinal_rejects_nondeterministic_mode():
    with pytest.raises(ValueError, match='deterministic'):
        CardinalTagger(deterministic=False, defer_verbalization=True)
