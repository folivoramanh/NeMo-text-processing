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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_COMMA, NEMO_DIGIT, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.vi.vietnamese_phonetic_rules import VietnamesePhoneticRules


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese decimal numbers, e.g.
        -12,5 tỷ -> decimal { negative: "true" integer_part: "mười hai" fractional_part: "năm" quantity: "tỷ" }
        12.345,67 -> decimal { integer_part: "mười hai nghìn ba trăm bốn mươi lăm" fractional_part: "sáu bảy" }
        1tr2 -> decimal { integer_part: "một triệu hai trăm nghìn" }
        818,303 -> decimal { integer_part: "tám trăm mười tám" fractional_part: "ba không ba" }
        0,2 triệu -> decimal { integer_part: "không" fractional_part: "hai" quantity: "triệu" }
    Args:
        cardinal: CardinalFst instance for processing integer parts
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_with_and
        self.graph = cardinal.single_digits_graph.optimize()
        
        # Enhanced non-deterministic support with rule-based alternatives
        if not deterministic:
            self.graph = self.graph | cardinal_graph
            # Add rule-based alternatives for fractional parts
            self.phonetic_rules = VietnamesePhoneticRules()
            self._enhanced_fractional_graph = self._create_enhanced_fractional_graph()
        else:
            self._enhanced_fractional_graph = None

        # Load data
        digit_labels = load_labels(get_abs_path("data/numbers/digit.tsv"))
        zero_labels = load_labels(get_abs_path("data/numbers/zero.tsv"))
        magnitude_labels = load_labels(get_abs_path("data/numbers/magnitudes.tsv"))
        quantity_abbr_labels = load_labels(get_abs_path("data/numbers/quantity_abbr.tsv"))

        # Enhanced digit mapping for non-deterministic mode
        if not deterministic:
            single_digit_map = self._create_enhanced_digit_map(digit_labels, zero_labels)
        else:
            single_digit_map = pynini.union(*[pynini.cross(k, v) for k, v in digit_labels + zero_labels])
            
        quantity_units = pynini.union(*[v for _, v in magnitude_labels])
        one_to_three_digits = NEMO_DIGIT + pynini.closure(NEMO_DIGIT, 0, 2)

        # Enhanced building blocks
        integer_part = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        
        # Enhanced fractional part with alternatives
        if not deterministic and self._enhanced_fractional_graph:
            fractional_part = (
                pynutil.insert("fractional_part: \"")
                + self._enhanced_fractional_graph
                + pynutil.insert("\"")
            )
        else:
            fractional_part = (
                pynutil.insert("fractional_part: \"")
                + single_digit_map
                + pynini.closure(pynutil.insert(NEMO_SPACE) + single_digit_map)
                + pynutil.insert("\"")
            )
            
        optional_quantity = (
            pynutil.delete(NEMO_SPACE).ques + pynutil.insert(" quantity: \"") + quantity_units + pynutil.insert("\"")
        ).ques

        patterns = []

        # 1. Enhanced basic decimal patterns with alternatives
        basic_decimal = (
            (integer_part + pynutil.insert(NEMO_SPACE)).ques
            + pynutil.delete(NEMO_COMMA)
            + pynutil.insert(NEMO_SPACE)
            + fractional_part
        )
        patterns.append(basic_decimal)
        patterns.append(basic_decimal + optional_quantity)

        # 2. Thousand-separated decimals: 12.345,67 and 12.345,67 tỷ
        integer_with_dots = (
            NEMO_DIGIT + pynini.closure(NEMO_DIGIT, 0, 2) + pynini.closure(pynutil.delete(".") + NEMO_DIGIT**3, 1)
        )
        separated_integer_part = (
            pynutil.insert("integer_part: \"")
            + pynini.compose(integer_with_dots, cardinal_graph)
            + pynutil.insert("\"")
        )
        separated_decimal = (
            separated_integer_part
            + pynutil.insert(NEMO_SPACE)
            + pynutil.delete(NEMO_COMMA)
            + pynutil.insert(NEMO_SPACE)
            + fractional_part
        )
        patterns.append(separated_decimal)
        patterns.append(separated_decimal + optional_quantity)

        # 3. Integer with quantity: 100 triệu
        integer_with_quantity = (
            integer_part
            + pynutil.delete(NEMO_SPACE).ques
            + pynutil.insert(" quantity: \"")
            + quantity_units
            + pynutil.insert("\"")
        )
        patterns.append(integer_with_quantity)

        # 4. Standard abbreviations: 1k, 100tr, etc.
        for abbr, full_name in quantity_abbr_labels:
            abbr_pattern = pynini.compose(
                one_to_three_digits + pynutil.delete(abbr),
                pynutil.insert("integer_part: \"")
                + pynini.compose(one_to_three_digits, cardinal_graph)
                + pynutil.insert(f"\" quantity: \"{full_name}\""),
            )
            patterns.append(abbr_pattern)

        # 5. Enhanced decimal with abbreviations: 2,5tr
        measure_prefix_labels = load_labels(get_abs_path("data/measure/prefixes.tsv"))
        measure_prefixes = {prefix.lower() for prefix, _ in measure_prefix_labels}

        # Filter quantity abbreviations to avoid measure conflicts
        safe_quantity_abbrs = [
            (abbr, full) for abbr, full in quantity_abbr_labels if abbr.lower() not in measure_prefixes
        ]

        for abbr, full_name in safe_quantity_abbrs:
            decimal_abbr_pattern = (
                (integer_part + pynutil.insert(NEMO_SPACE)).ques
                + pynutil.delete(NEMO_COMMA)
                + pynutil.insert(NEMO_SPACE)
                + fractional_part
                + pynutil.insert(f" quantity: \"{full_name}\"")
                + pynutil.delete(abbr)
            )
            patterns.append(decimal_abbr_pattern)

        # 6. Compound abbreviations: 1tr2 -> một triệu hai trăm nghìn, 2t3 -> hai tỷ ba trăm triệu
        compound_expansions = {
            "tr": ("triệu", "trăm nghìn"),  # 1tr2 -> một triệu hai trăm nghìn
            "t": ("tỷ", "trăm triệu"),  # 2t3 -> hai tỷ ba trăm triệu
        }

        for abbr, (major_unit, minor_suffix) in compound_expansions.items():
            pattern = one_to_three_digits + pynini.cross(abbr, "") + NEMO_DIGIT
            expansion = (
                pynutil.insert("integer_part: \"")
                + pynini.compose(one_to_three_digits, cardinal_graph)
                + pynutil.insert(f" {major_unit} ")
                + pynini.compose(NEMO_DIGIT, cardinal_graph)
                + pynutil.insert(f" {minor_suffix}\"")
            )
            patterns.append(pynini.compose(pattern, expansion))

        # 7. Non-deterministic: Add digit-by-digit reading for long decimals
        if not deterministic:
            digit_by_digit_pattern = self._create_digit_by_digit_decimal_pattern(single_digit_map)
            if digit_by_digit_pattern:
                patterns.append(digit_by_digit_pattern)

        # Combine all patterns
        self._final_graph_wo_negative = pynini.union(*patterns).optimize()

        # Add optional negative prefix
        negative = (pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")).ques
        final_graph = negative + self._final_graph_wo_negative

        self.fst = self.add_tokens(final_graph).optimize()

    @property
    def final_graph_wo_negative(self):
        """Graph without negative prefix, used by MoneyFst"""
        return self._final_graph_wo_negative
    
    def _create_enhanced_digit_map(self, digit_labels, zero_labels):
        """Create enhanced digit mapping using systematic rules"""
        basic_map = pynini.union(*[pynini.cross(k, v) for k, v in digit_labels + zero_labels])
        
        # Use systematic rule-based alternatives instead of hardcoded patterns
        alternatives = []
        
        # Generate alternatives for all digits systematically
        for digit in "0123456789":
            digit_alts = self.phonetic_rules.generate_decimal_digit_alternatives(digit, "fractional")
            for alt in digit_alts:
                alternatives.append(pynini.cross(digit, alt))
        
        # Combine basic and alternative mappings
        enhanced_map = basic_map
        if alternatives:
            enhanced_map = basic_map | pynini.union(*alternatives)
            
        return enhanced_map.optimize()
    
    def _create_enhanced_fractional_graph(self):
        """Create enhanced fractional part graph with alternatives"""
        try:
            # Load basic digit mappings
            digit_labels = load_labels(get_abs_path("data/numbers/digit.tsv"))
            zero_labels = load_labels(get_abs_path("data/numbers/zero.tsv"))
            
            # Create enhanced digit map
            enhanced_digit_map = self._create_enhanced_digit_map(digit_labels, zero_labels)
            
            # Build fractional graph with enhanced alternatives
            fractional_graph = enhanced_digit_map + pynini.closure(
                pynutil.insert(NEMO_SPACE) + enhanced_digit_map
            )
            
            return fractional_graph.optimize()
            
        except Exception as e:
            print(f"Warning: Could not create enhanced fractional graph: {e}")
            # Fallback to basic mapping
            basic_map = pynini.union(*[pynini.cross(k, v) for k, v in digit_labels + zero_labels])
            return basic_map + pynini.closure(pynutil.insert(NEMO_SPACE) + basic_map)
    
    def _create_digit_by_digit_decimal_pattern(self, single_digit_map):
        """Create digit-by-digit reading pattern for long decimals"""
        try:
            # Pattern for long decimal numbers (e.g., 3.14159 -> "ba phẩy một bốn một năm chín")
            # This is useful for scientific numbers, coordinates, etc.
            
            integer_digits = pynini.closure(NEMO_DIGIT, 1, 4)  # 1-4 digits before decimal
            fractional_digits = pynini.closure(NEMO_DIGIT, 3, 10)  # 3+ digits after decimal (long)
            
            # Create digit-by-digit reading for both parts
            integer_digit_by_digit = pynini.compose(integer_digits, 
                single_digit_map + pynini.closure(pynutil.insert(NEMO_SPACE) + single_digit_map))
            
            fractional_digit_by_digit = pynini.compose(fractional_digits,
                single_digit_map + pynini.closure(pynutil.insert(NEMO_SPACE) + single_digit_map))
            
            # Combine with decimal point handling
            digit_by_digit_pattern = (
                pynutil.insert("integer_part: \"")
                + integer_digit_by_digit
                + pynutil.insert("\"")
                + pynutil.insert(NEMO_SPACE)
                + pynutil.delete(NEMO_COMMA)
                + pynutil.insert(NEMO_SPACE)
                + pynutil.insert("fractional_part: \"")
                + fractional_digit_by_digit
                + pynutil.insert("\"")
            )
            
            # Add weight to make it lower priority than standard decimal reading
            return pynutil.add_weight(digit_by_digit_pattern, 0.1)
            
        except Exception as e:
            print(f"Warning: Could not create digit-by-digit decimal pattern: {e}")
            return None
