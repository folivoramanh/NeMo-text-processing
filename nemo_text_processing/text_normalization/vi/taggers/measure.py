# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_COMMA,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path
from nemo_text_processing.text_normalization.vi.vietnamese_phonetic_rules import VietnamesePhoneticRules


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure for Vietnamese, e.g.
        12kg -> measure { cardinal { integer: "mười hai" } units: "ki lô gam" }
        1kg -> measure { cardinal { integer: "một" } units: "ki lô gam" }
        0.5kg -> measure { decimal { fractional_part: "năm" } units: "ki lô gam" }
        -12kg -> measure { negative: "true" cardinal { integer: "mười hai" } units: "ki lô gam" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def _create_measure_subgraph(self, measure_type: str, number_graph, optional_negative, graph_unit):
        """Helper to create measure subgraph pattern - reduces duplication"""
        return (
            optional_negative
            + pynutil.insert(f"{measure_type} {{ ")
            + number_graph
            + pynutil.insert(" }")
            + delete_space
            + pynutil.insert(" units: \"")
            + graph_unit
            + pynutil.insert('"')
        )

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        # Initialize phonetic rules for non-deterministic alternatives
        if not deterministic:
            self.phonetic_rules = VietnamesePhoneticRules()

        # Use enhanced cardinal graph for non-deterministic mode
        if deterministic:
            cardinal_graph = cardinal.graph
        else:
            cardinal_graph = self._create_enhanced_cardinal_graph(cardinal.graph)

        # Load minimal measurement files (massive redundancy removed via subfst)
        measurements_path = get_abs_path("data/measure/measurements_minimal.tsv")
        prefixes_path = get_abs_path("data/measure/prefixes.tsv")
        base_units_path = get_abs_path("data/measure/base_units.tsv")

        # Create subfst for metric units: prefix + space + base_unit
        graph_prefixes = pynini.string_file(prefixes_path)
        graph_base_units = pynini.string_file(base_units_path)
        space = pynutil.insert(NEMO_SPACE)
        graph_metric_units = graph_prefixes + space + graph_base_units

        # Load non-metric and special units
        graph_special_units = pynini.string_file(measurements_path)

        # Also allow base units without prefixes (e.g., 'g' not just 'kg')
        graph_standalone_units = graph_base_units

        # Combine all unit mappings
        base_graph_unit = graph_metric_units | graph_special_units | graph_standalone_units
        
        # Add enhanced unit alternatives for non-deterministic mode
        graph_unit = base_graph_unit

        # Add compound unit support with alternatives (unit/unit patterns like km/h)
        if not deterministic:
            # Use enhanced compound alternatives: "trên", "một", "mỗi"
            graph_unit_compound = self._create_compound_unit_alternatives(graph_unit)
        else:
            # Standard "trên" only
            graph_unit_compound = (
                graph_unit +
                pynini.cross("/", " trên ") +
                pynutil.insert(NEMO_SPACE) +
                graph_unit
            )

        optional_graph_unit_compound = pynini.closure(
            pynutil.insert(NEMO_SPACE) + graph_unit_compound,
            0,
            1,
        )

        # Update unit graph to include compound units
        graph_unit = graph_unit + optional_graph_unit_compound | graph_unit_compound

        # Create unit symbol pattern using FST operations (no loops needed)
        prefix_symbols = pynini.project(graph_prefixes, "input")  # Extract prefix symbols
        base_symbols = pynini.project(graph_base_units, "input")  # Extract base symbols
        special_symbols = pynini.project(graph_special_units, "input")  # Extract special symbols

        # Build unit pattern: metric combinations | standalone bases | special units
        metric_pattern = prefix_symbols + base_symbols  # All prefix+base combinations
        simple_unit_pattern = metric_pattern | base_symbols | special_symbols

        # Add compound unit patterns to recognition
        compound_pattern = simple_unit_pattern + "/" + simple_unit_pattern
        unit_pattern = simple_unit_pattern | compound_pattern

        number = pynini.closure(NEMO_DIGIT, 1)
        decimal_number = number + NEMO_COMMA + pynini.closure(NEMO_DIGIT, 1)

        # Optional negative sign handling for Vietnamese
        optional_graph_negative = pynini.closure(
            pynini.cross("-", "negative: \"true\" "),
            0,
            1,
        )

        # Domain restriction patterns - only match core number+unit patterns
        # Remove punctuation handling to let punctuation tagger handle it separately
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        optional_negative_sign = pynini.closure("-" + optional_space, 0, 1)

        integer_measure_domain = optional_negative_sign + number + optional_space + unit_pattern
        decimal_measure_domain = optional_negative_sign + decimal_number + optional_space + unit_pattern
        fraction_measure_domain = optional_negative_sign + number + "/" + number + optional_space + unit_pattern

        cardinal_number_graph = pynutil.insert('integer: "') + (number @ cardinal_graph) + pynutil.insert('"')

        subgraph_cardinal = self._create_measure_subgraph(
            "cardinal", cardinal_number_graph, optional_graph_negative, graph_unit
        )
        subgraph_decimal = self._create_measure_subgraph(
            "decimal", decimal.final_graph_wo_negative, optional_graph_negative, graph_unit
        )
        subgraph_fraction = self._create_measure_subgraph(
            "fraction", fraction.final_graph_wo_negative, optional_graph_negative, graph_unit
        )

        # Apply domain restrictions to ensure we only match complete number+unit patterns
        subgraph_cardinal = pynini.compose(integer_measure_domain, subgraph_cardinal)
        subgraph_decimal = pynini.compose(decimal_measure_domain, subgraph_decimal)
        subgraph_fraction = pynini.compose(fraction_measure_domain, subgraph_fraction)

        # Final graph combining main patterns
        final_graph = subgraph_cardinal | subgraph_decimal | subgraph_fraction

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def _create_enhanced_cardinal_graph(self, base_cardinal_graph):
        """
        Create enhanced cardinal graph with phonetic alternatives for non-deterministic mode.
        For measures, common numbers like weights, distances need alternatives.
        """
        alternatives = []
        
        # Add base cardinal graph
        alternatives.append(base_cardinal_graph)
        
        # Add phonetic alternatives for common measurement numbers
        for num in range(1, 1001):  # Common measurement ranges 1-1000
            num_str = str(num)
            phonetic_alts = self.phonetic_rules.generate_alternatives(num_str, "general")
            
            for alt in phonetic_alts:
                if alt != num_str:  # Don't duplicate base form
                    alternatives.append(pynini.cross(num_str, alt))
        
        return pynini.union(*alternatives)

    def _create_compound_unit_alternatives(self, graph_unit):
        """
        Create compound unit alternatives for non-deterministic mode.
        Handle patterns like km/h with different readings.
        """
        alternatives = []
        
        # Alternative 1: "trên" (standard)
        # km/h -> "ki lô mét trên giờ"
        compound_with_tren = (
            graph_unit +
            pynini.cross("/", " trên ") +
            pynutil.insert(NEMO_SPACE) +
            graph_unit
        )
        alternatives.append(compound_with_tren)
        
        # Alternative 2: "một"
        # km/h -> "ki lô mét một giờ"
        compound_with_mot = (
            graph_unit +
            pynini.cross("/", " một ") +
            pynutil.insert(NEMO_SPACE) +
            graph_unit
        )
        alternatives.append(compound_with_mot)
        
        # Alternative 3: "mỗi"
        # km/h -> "ki lô mét mỗi giờ"
        compound_with_moi = (
            graph_unit +
            pynini.cross("/", " mỗi ") +
            pynutil.insert(NEMO_SPACE) +
            graph_unit
        )
        alternatives.append(compound_with_moi)
        
        return pynini.union(*alternatives)
