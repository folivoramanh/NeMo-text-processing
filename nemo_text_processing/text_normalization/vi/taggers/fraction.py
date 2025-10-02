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

from nemo_text_processing.text_normalization.vi.graph_utils import GraphFst
from nemo_text_processing.text_normalization.vi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.vi.vietnamese_phonetic_rules import VietnamesePhoneticRules


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese fraction numbers, e.g.
        23 1/5 -> fraction { integer_part: "hai mươi ba" numerator: "một" denominator: "năm" }
        3/9 -> fraction { numerator: "ba" denominator: "chín" }
        1/4 -> fraction { numerator: "một" denominator: "tư" }

    Args:
        cardinal: CardinalFst for converting numbers to Vietnamese words
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        digit = pynini.union(*[str(i) for i in range(10)])
        number = pynini.closure(digit, 1)

        # Load denominator exceptions
        denominator_exceptions = {
            row[0]: row[1] for row in load_labels(get_abs_path("data/fraction/denominator_exceptions.tsv"))
        }

        # Enhanced non-deterministic support
        if not deterministic:
            self.phonetic_rules = VietnamesePhoneticRules()
            enhanced_denominator_graph = self._create_enhanced_denominator_graph(
                cardinal_graph, denominator_exceptions
            )
            enhanced_numerator_graph = self._create_enhanced_numerator_graph(cardinal_graph)
        else:
            # Standard deterministic behavior
            denominator_exception_patterns = [pynini.cross(k, v) for k, v in denominator_exceptions.items()]
            denominator_exception_graph = (
                pynini.union(*denominator_exception_patterns) if denominator_exception_patterns else None
            )
            enhanced_denominator_graph = (
                pynini.union(denominator_exception_graph, cardinal_graph)
                if denominator_exception_graph
                else cardinal_graph
            )
            enhanced_numerator_graph = cardinal_graph

        # Build fraction components
        numerator = (
            pynutil.insert("numerator: \"") 
            + (number @ enhanced_numerator_graph) 
            + pynutil.insert("\" ") 
            + pynutil.delete("/")
        )
        denominator = (
            pynutil.insert("denominator: \"") 
            + (number @ enhanced_denominator_graph) 
            + pynutil.insert("\"")
        )
        integer_part = (
            pynutil.insert("integer_part: \"") 
            + (number @ enhanced_numerator_graph) 
            + pynutil.insert("\" ")
        )

        simple_fraction = numerator + denominator
        mixed_fraction = integer_part + pynutil.delete(" ") + numerator + denominator

        # Create graph without negative for reuse in other FSTs (like measure)
        fraction_wo_negative = simple_fraction | mixed_fraction
        self.final_graph_wo_negative = fraction_wo_negative.optimize()

        optional_graph_negative = (pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")).ques

        self.fst = self.add_tokens(optional_graph_negative + (simple_fraction | mixed_fraction)).optimize()
    
    def _create_enhanced_denominator_graph(self, cardinal_graph, base_exceptions):
        """Create enhanced denominator graph using systematic rules"""
        try:
            # Enhanced denominator exceptions with systematic alternatives
            enhanced_exceptions = dict(base_exceptions)
            
            # Use systematic rule-based alternatives instead of hardcoded dict
            for i in range(2, 101):  # Common denominators 2-100
                number_str = str(i)
                if number_str not in enhanced_exceptions:  # Don't override existing exceptions
                    denom_alts = self.phonetic_rules.generate_fraction_denominator_alternatives(number_str)
                    if len(denom_alts) > 1:  # Only add if there are alternatives
                        enhanced_exceptions[number_str] = denom_alts
            
            # Create patterns
            exception_patterns = []
            for digit, alternatives in enhanced_exceptions.items():
                if isinstance(alternatives, list):
                    for alt in alternatives:
                        exception_patterns.append(pynini.cross(digit, alt))
                else:
                    exception_patterns.append(pynini.cross(digit, alternatives))
            
            # Use systematic range generation for larger denominators
            range_alternatives = self.phonetic_rules.get_systematic_range_alternatives(101, 1000, "general")
            rule_based_patterns = []
            
            for number_str, alternatives in range_alternatives.items():
                if number_str not in enhanced_exceptions:  # Don't override exceptions
                    for alt in alternatives:
                        rule_based_patterns.append(pynini.cross(number_str, alt))
            
            # Combine all patterns
            all_patterns = exception_patterns + rule_based_patterns
            if all_patterns:
                enhanced_graph = pynini.union(*all_patterns)
                combined_graph = pynini.union(enhanced_graph, cardinal_graph)
            else:
                combined_graph = cardinal_graph
                
            return combined_graph.optimize()
            
        except Exception as e:
            print(f"Warning: Could not create enhanced denominator graph: {e}")
            # Fallback to standard behavior
            denominator_exception_patterns = [pynini.cross(k, v) for k, v in base_exceptions.items()]
            denominator_exception_graph = (
                pynini.union(*denominator_exception_patterns) if denominator_exception_patterns else None
            )
            return (
                pynini.union(denominator_exception_graph, cardinal_graph)
                if denominator_exception_graph
                else cardinal_graph
            )
    
    def _create_enhanced_numerator_graph(self, cardinal_graph):
        """Create enhanced numerator graph using systematic rules"""
        try:
            # Use systematic range generation instead of hardcoded loops
            range_alternatives = self.phonetic_rules.get_systematic_range_alternatives(1, 100, "general")
            rule_based_patterns = []
            
            for number_str, alternatives in range_alternatives.items():
                for alt in alternatives:
                    rule_based_patterns.append(pynini.cross(number_str, alt))
            
            if rule_based_patterns:
                enhanced_graph = pynini.union(*rule_based_patterns)
                combined_graph = pynini.union(enhanced_graph, cardinal_graph)
            else:
                combined_graph = cardinal_graph
                
            return combined_graph.optimize()
            
        except Exception as e:
            print(f"Warning: Could not create enhanced numerator graph: {e}")
            return cardinal_graph
