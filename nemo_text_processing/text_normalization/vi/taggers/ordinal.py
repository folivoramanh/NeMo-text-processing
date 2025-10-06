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

import os
import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.vi.vietnamese_phonetic_rules import VietnamesePhoneticRules


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese ordinals, e.g.
        thứ 1 -> ordinal { integer: "nhất" }
        thứ 4 -> ordinal { integer: "tư" }
        thứ 15 -> ordinal { integer: "mười lăm" }
    Args:
        cardinal: CardinalFst for number conversion
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Vietnamese ordinal prefixes
        ordinal_prefixes = pynini.union("thứ ", "hạng ")
        number_pattern = pynini.closure(NEMO_DIGIT, 1)

        # Load ordinal exceptions
        ordinal_exceptions = {
            row[0]: row[1] for row in load_labels(get_abs_path("data/ordinal/ordinal_exceptions.tsv"))
        }

        # Enhanced non-deterministic support
        if not deterministic:
            self.phonetic_rules = VietnamesePhoneticRules()
            enhanced_exceptions = self._create_enhanced_ordinal_exceptions(ordinal_exceptions)
            combined_graph = self._create_enhanced_ordinal_graph(cardinal, enhanced_exceptions)
        else:
            # Standard deterministic behavior
            exception_patterns = []
            for digit, word in ordinal_exceptions.items():
                exception_patterns.append(pynini.cross(digit, word))

            exception_graph = pynini.union(*exception_patterns) if exception_patterns else None
            combined_graph = cardinal.graph
            if exception_graph:
                combined_graph = pynini.union(exception_graph, cardinal.graph)

        self.graph = (
            pynutil.delete(ordinal_prefixes)
            + pynutil.insert("integer: \"")
            + pynini.compose(number_pattern, combined_graph)
            + pynutil.insert("\"")
        )

        self.fst = self.add_tokens(self.graph).optimize()
    
    def _create_enhanced_ordinal_exceptions(self, base_exceptions):
        """Create enhanced ordinal exceptions using systematic rules"""
        enhanced = dict(base_exceptions)
        
        # Use systematic rule-based alternatives instead of hardcoded dict
        for i in range(1, 100):  # Cover common ordinals
            number_str = str(i)
            if number_str not in enhanced:  # Don't override existing exceptions
                ordinal_alts = self.phonetic_rules.generate_ordinal_alternatives(number_str)
                if len(ordinal_alts) > 1:  # Only add if there are alternatives
                    enhanced[number_str] = ordinal_alts
        
        return enhanced
    
    def _create_enhanced_ordinal_graph(self, cardinal, enhanced_exceptions):
        """Create enhanced ordinal graph with systematic rule-based alternatives"""
        try:
            # Create exception patterns with alternatives
            exception_patterns = []
            for digit, alternatives in enhanced_exceptions.items():
                if isinstance(alternatives, list):
                    for alt in alternatives:
                        exception_patterns.append(pynini.cross(digit, alt))
                else:
                    exception_patterns.append(pynini.cross(digit, alternatives))
            
            # Use systematic range generation instead of hardcoded loops
            range_alternatives = self.phonetic_rules.get_systematic_range_alternatives(100, 999, "general")
            rule_based_patterns = []
            
            for number_str, alternatives in range_alternatives.items():
                if number_str not in enhanced_exceptions:  # Don't override exceptions
                    for alt in alternatives:
                        rule_based_patterns.append(pynini.cross(number_str, alt))
            
            # Combine all patterns with weights to prefer enhanced alternatives
            all_patterns = exception_patterns + rule_based_patterns
            if all_patterns:
                enhanced_graph = pynini.union(*all_patterns)
                # Give enhanced alternatives lower weight (higher priority) than standard cardinal
                combined_graph = pynini.union(
                    pynutil.add_weight(enhanced_graph, 0.9),  # Enhanced alternatives get priority
                    pynutil.add_weight(cardinal.graph, 1.1)   # Standard alternatives get lower priority
                )
            else:
                combined_graph = cardinal.graph
                
            return combined_graph.optimize()
            
        except Exception as e:
            print(f"Warning: Could not create enhanced ordinal graph: {e}")
            # Fallback to standard behavior
            exception_patterns = []
            for digit, alternatives in enhanced_exceptions.items():
                if isinstance(alternatives, list):
                    exception_patterns.append(pynini.cross(digit, alternatives[0]))
                else:
                    exception_patterns.append(pynini.cross(digit, alternatives))
            
            exception_graph = pynini.union(*exception_patterns) if exception_patterns else None
            combined_graph = cardinal.graph
            if exception_graph:
                combined_graph = pynini.union(exception_graph, cardinal.graph)
            return combined_graph
