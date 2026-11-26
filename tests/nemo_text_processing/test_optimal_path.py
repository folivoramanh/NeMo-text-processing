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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.optimal_path import AmbiguousRewrite, guarded_best_path

pytestmark = pytest.mark.unit


def test_different_optimal_outputs_require_fallback():
    lattice = pynini.cross('input', 'one') | pynini.cross('input', 'two')
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(lattice)


def test_duplicate_paths_with_the_same_output_are_unambiguous():
    direct = pynini.cross('input', 'same')
    with_epsilon = pynini.cross('', '') + pynini.cross('input', 'same')
    lattice = direct | with_epsilon
    before = lattice.write_to_string()
    assert guarded_best_path(lattice).string() == 'same'
    assert lattice.write_to_string() == before


def test_strictly_more_expensive_output_does_not_change_the_selected_path():
    lattice = pynutil.add_weight(pynini.cross('input', 'one'), -0.5)
    lattice |= pynutil.add_weight(pynini.cross('input', 'two'), 0.5)
    original = pynini.shortestpath(lattice, nshortest=1, unique=True)
    selected = guarded_best_path(lattice)
    assert selected.write_to_string() == original.write_to_string()


@pytest.mark.parametrize('other_cost,ambiguous', [(-3.0, True), (-2.0, False)])
def test_epsilon_arcs_negative_weights_and_empty_output(other_cost, ambiguous):
    lattice = pynutil.add_weight(pynini.cross('input', '') + pynini.cross('', ''), -3.0)
    lattice |= pynutil.add_weight(pynini.cross('input', '') + pynini.cross('', 'other'), other_cost)
    if ambiguous:
        with pytest.raises(AmbiguousRewrite):
            guarded_best_path(lattice)
    else:
        assert guarded_best_path(lattice).string() == ''


@pytest.mark.parametrize('token_type', ['byte', 'utf8'])
@pytest.mark.parametrize('text', ['một 한국어 🧑🏽\u200d💻', '[0] [EPS] [1234] \\ []'])
def test_unicode_and_literal_labels_are_preserved(token_type, text):
    output = pynini.accep(pynini.escape(text), token_type=token_type)
    lattice = pynini.cross('input', output)
    lattice |= pynutil.add_weight(pynini.cross('input', 'other'), 1.0)
    selected = guarded_best_path(lattice)
    assert selected.string(token_type=token_type) == text
    assert pynini.project(selected, 'input').string() == 'input'


@pytest.mark.parametrize('token_type', ['byte', 'utf8'])
def test_tied_unicode_and_literal_outputs_require_fallback(token_type):
    lattice = pynini.cross('input', pynini.accep(pynini.escape('một [1234]'), token_type=token_type))
    lattice |= pynini.cross('input', pynini.accep(pynini.escape('한국어 [EPS]'), token_type=token_type))
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(lattice)


@pytest.mark.parametrize(
    'best_cost,other_cost,ambiguous',
    [(0.0, 0.0000005, True), (1000.0, 1000.0005, True), (1000.0, 1000.1, False)],
)
def test_near_ties_use_conservative_float32_tolerance(best_cost, other_cost, ambiguous):
    lattice = pynutil.add_weight(pynini.cross('input', 'one'), best_cost)
    lattice |= pynutil.add_weight(pynini.cross('input', 'two'), other_cost)
    if ambiguous:
        with pytest.raises(AmbiguousRewrite):
            guarded_best_path(lattice)
    else:
        assert guarded_best_path(lattice).string() == 'one'


@pytest.mark.parametrize('loop_weight', [0.0, 0.5])
def test_epsilon_cycle_does_not_create_an_alternate_output(loop_weight):
    lattice = pynini.cross('input', 'same')
    lattice.add_arc(lattice.start(), pynini.Arc(0, 0, pynini.Weight('tropical', loop_weight), lattice.start()))
    assert guarded_best_path(lattice).string() == 'same'


def test_zero_weight_output_cycle_is_ambiguous():
    lattice = pynini.cross('input', 'same')
    lattice.add_arc(lattice.start(), pynini.Arc(0, ord('x'), pynini.Weight.one('tropical'), lattice.start()))
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(lattice)


def test_empty_lattice_returns_an_empty_copy():
    lattice = pynini.Fst()
    selected = guarded_best_path(lattice)
    assert selected.start() == -1
    assert selected is not lattice


def test_lattice_without_a_successful_path_returns_empty():
    lattice = pynini.Fst()
    lattice.set_start(lattice.add_state())
    assert guarded_best_path(lattice).start() == -1


def test_infinite_weight_alternate_output_is_not_ambiguous():
    lattice = pynini.cross('input', 'one')
    lattice |= pynutil.add_weight(pynini.cross('input', 'two'), float('inf'))
    assert guarded_best_path(lattice).string() == 'one'


def test_requested_absolute_tolerance_is_respected():
    lattice = pynini.cross('input', 'one')
    lattice |= pynutil.add_weight(pynini.cross('input', 'two'), 0.1)
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(lattice, delta=0.2)


def _tag_lattice(*tags):
    return pynini.union(*(pynini.cross('input', pynini.escape(tag)) for tag in tags))


def test_tagger_ignores_only_structural_ascii_spaces():
    lattice = _tag_lattice('tokens{name:"same" }', '  tokens { name : "same"   }   ')
    original = pynini.shortestpath(lattice, nshortest=1, unique=True)
    assert guarded_best_path(lattice, tagger=True).write_to_string() == original.write_to_string()
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(lattice)


def test_tagger_preserves_nested_fields_and_unquoted_preserve_order():
    lattice = _tag_lattice(
        'tokens { date { month: "november" day: "twenty" preserve_order: true } } tokens { name: "." }',
        ' tokens{date{month :"november"  day:"twenty" preserve_order :true}}tokens{name:"."   } ',
    )
    guarded_best_path(lattice, tagger=True)


@pytest.mark.parametrize('value', ['"', 'a"b', 'a""b', ''])
def test_tagger_keeps_embedded_quotes_and_empty_values_literal(value):
    lattice = _tag_lattice(f'tokens {{ name: "{value}" }}', f' tokens{{name : "{value}"   }}  ')
    guarded_best_path(lattice, tagger=True)


@pytest.mark.parametrize('token_type', ['byte', 'utf8'])
def test_tagger_equivalence_respects_existing_unicode_token_context(token_type):
    with pynini.default_token_type(token_type):
        value = 'một 한국어 [0] [EPS] [1234] \\ []'
        lattice = _tag_lattice(f'tokens {{ name: "{value}" }}', f'tokens{{name:"{value}"  }}')
        selected = guarded_best_path(lattice, tagger=True)
        assert value in selected.string()


@pytest.mark.parametrize(
    'first,second',
    [
        ('tokens { name: "one" }', 'tokens { name: "two" }'),
        ('tokens { name: "a b" }', 'tokens { name: "a  b" }'),
        ('tokens { name: "a b" }', 'tokens { name: "a\tb" }'),
        ('tokens { name: "a"b" }', 'tokens { name: "ab" }'),
        (
            'tokens { date { month: "november" day: "twenty" } }',
            'tokens { date { day: "twenty" month: "november" } }',
        ),
    ],
)
def test_tagger_semantic_or_field_order_ties_still_require_fallback(first, second):
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(_tag_lattice(first, second), tagger=True)


@pytest.mark.parametrize(
    'first,second',
    [
        ('one', 'two'),
        ('tokens { name: "same" } suffix', 'tokens { name: "same" }  suffix'),
        ('tokens{name:"same"}', 'tokens{name:"same" }'),
        ('tokens { name: "same" name: "same" }', 'tokens { name: "same"  name: "same" }'),
        ('tokens { name: "same" }', 'tokens\t{ name: "same" }'),
    ],
)
def test_tagger_unparseable_or_unmatched_structure_uses_literal_exclusion(first, second):
    with pytest.raises(AmbiguousRewrite):
        guarded_best_path(_tag_lattice(first, second), tagger=True)
