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

"""Detect output ties before relying on a transformed grammar's shortest path."""

import pynini

from nemo_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser


class AmbiguousRewrite(RuntimeError):
    """Distinct output strings have equal or near-equal minimum path costs."""


def _tagger_output_equivalence(best_path, literal_output):
    """Accept the chosen ordered token structure with parser-ignored spacing."""
    try:
        parser = TokenParser()
        parser(best_path.string())
        tokens = parser.parse()
        space = pynini.closure(' ')
        required_space = pynini.closure(' ', 1)

        def literal(value):
            return pynini.accep(pynini.escape(value))

        def fields(items):
            graph = space.copy()
            for key, value in items:
                graph += literal(key) + space
                if key == PRESERVE_ORDER_KEY:
                    graph += literal(':') + space + literal('true')
                elif isinstance(value, dict):
                    graph += literal('{') + fields(value.items()) + literal('}')
                else:
                    # TokenParser returns None for an empty quoted value. It
                    # recognizes the closing quote only before ASCII space;
                    # quotes inside the exact value must remain literal.
                    graph += literal(':') + space + literal('"')
                    graph += literal(value if value is not None else '') + literal('"') + required_space
                graph += space
            return graph

        ordered_fields = (item for token in tokens for item in token.items())
        equivalent = pynini.determinize(fields(ordered_fields).rmepsilon()).arcsort(sort_type='ilabel')
        # Parsing is intentionally permissive and can stop before malformed
        # trailing text. Require the reconstructed language to contain the
        # entire original output before using this broader exclusion.
        if pynini.intersect(literal_output, equivalent).start() != -1:
            return equivalent
    except Exception:
        # Decoding, parsing, or recompiling unfamiliar labels must never weaken
        # the literal guard. In particular, do not override token-type context.
        pass
    return literal_output


def guarded_best_path(lattice, delta=1e-6, *, tagger=False):
    """Return the usual best path, raising when another output is near-optimal.

    A linear acceptor built from the chosen path excludes its complete output
    string. Shortest distance over the remaining lattice then detects alternate
    outputs without enumerating paths or determinizing the output lattice.
    Duplicate paths spelling the same output do not count as ambiguity. With
    ``tagger=True``, TokenParser-ignored ASCII spaces are also equivalent;
    quoted values and field order remain exact. Unparseable tags retain the
    conservative literal comparison.

    Tropical weights use float32 arithmetic. To conservatively include rounding
    around a tie, the comparison uses the larger of ``delta`` and eight float32
    machine-epsilon units relative to the path-cost scale (at least one).
    Near-ties may therefore request a baseline fallback even when the selected
    output is strictly cheaper. Input/output labels and the original selected
    path are retained. The default literal comparison never decodes strings;
    tagger equivalence uses the caller's existing string-token context.
    """
    if lattice.start() == -1:
        return lattice.copy()

    best_path = pynini.shortestpath(lattice, delta=delta, nshortest=1, unique=True)
    if best_path.start() == -1:
        return best_path

    chosen_output = pynini.arcmap(pynini.project(best_path, 'output'), map_type='rmweight')
    # A single path is already deterministic after epsilon removal. General
    # optimization adds avoidable work, especially for long tagged strings.
    chosen_output.rmepsilon().arcsort(sort_type='ilabel')
    if tagger:
        chosen_output = _tagger_output_equivalence(best_path, chosen_output)
    alternatives = pynini.difference(pynini.project(lattice, 'output'), chosen_output)
    if alternatives.start() == -1:
        return best_path

    best_cost = float(pynini.shortestdistance(best_path, reverse=True, delta=delta)[best_path.start()])
    alternate_cost = float(pynini.shortestdistance(alternatives, reverse=True, delta=delta)[alternatives.start()])
    if alternate_cost == float('inf'):
        return best_path
    tolerance = max(delta, 8 * 2**-23 * max(1.0, abs(best_cost), abs(alternate_cost)))
    if alternate_cost <= best_cost + tolerance:
        raise AmbiguousRewrite(
            'Distinct output strings have equal or near-equal path costs '
            f'(best={best_cost}, alternative={alternate_cost}, tolerance={tolerance}).'
        )
    return best_path
