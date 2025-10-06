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
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels

class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "10,5$" -> money { integer_part: "mười" currency_maj: "đô la" fractional_part: "năm mươi" currency_min: "xu" preserve_order: true }
        "10đ" -> money { integer_part: "mười" currency_maj: "đồng" }
        "10 triệu đồng" -> money { integer_part: "mười" quantity: "triệu" currency_maj: "đồng" }

    Args:
        cardinal: CardinalFst instance for processing integer parts
        decimal: DecimalFst instance for processing fractional parts
        deterministic: if True will provide a single transduction option, for False multiple transduction are generated.
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        # Load data
        currency_major_labels = load_labels(get_abs_path("data/money/currency.tsv"))
        currency_minor_labels = load_labels(get_abs_path("data/money/currency_minor.tsv"))
        quantity_graph = pynini.string_file(get_abs_path("data/numbers/quantity_abbr.tsv"))

        # Load optimized per_unit files using subfst approach
        per_unit_non_metric_path = get_abs_path("data/money/per_unit_non_metric.tsv")
        per_unit_prefixes_path = get_abs_path("data/money/per_unit_prefixes.tsv")
        per_unit_bases_path = get_abs_path("data/money/per_unit_bases.tsv")

        # Create subfst for metric per_unit patterns
        graph_prefixes = pynini.string_file(per_unit_prefixes_path)
        graph_bases = pynini.string_file(per_unit_bases_path)

        # Build metric combinations: "/kg" -> "một ki lô gam"
        slash = pynutil.delete("/")
        one_space = pynutil.insert("một ")
        space = pynutil.insert(NEMO_SPACE)

        graph_metric_per_units = slash + one_space + graph_prefixes + space + graph_bases
        graph_standalone_per_units = slash + one_space + graph_bases

        # Load non-metric per_unit entries
        graph_non_metric_per_units = pynini.string_file(per_unit_non_metric_path)

        # Combine all per_unit mappings
        per_unit_graph = graph_metric_per_units | graph_standalone_per_units | graph_non_metric_per_units

        # Basic components
        cardinal_graph = cardinal.graph
        currency_major_graph = pynini.string_map(currency_major_labels)
        currency_minor_map = dict(currency_minor_labels)
        decimal_graph = decimal.final_graph_wo_negative

        # Common patterns
        integer_part = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        preserve_order = pynutil.insert(" preserve_order: true")
        optional_space = pynini.closure(delete_space, 0, 1)

        # Fractional part conversion for cents
        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )
        fractional_conversion = two_digits_fractional_part @ cardinal_graph
        fractional_part = pynutil.insert('fractional_part: "') + fractional_conversion + pynutil.insert('"')

        all_patterns = []

        # 1. Symbol-based patterns 
        symbol_patterns = []
        reverse_symbol_patterns = []
        minor_only_patterns = []

        for symbol, major_name in currency_major_labels:
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')

            # Forward pattern: 10$ or 10US$ -> mười đô la 
            simple_pattern = integer_part + pynutil.delete(symbol) + insert_space + maj_tag
            symbol_patterns.append(simple_pattern)
            
            # Reverse pattern: $10 or US$10 -> mười đô la (symbol before number)
            # Apply to all symbols, not just single character ones
            reverse_pattern = pynutil.delete(symbol) + integer_part + insert_space + maj_tag
            reverse_symbol_patterns.append(reverse_pattern)
            
            # Reverse pattern with optional space: $ 10 or US$ 10 -> mười đô la
            reverse_pattern_with_space = pynutil.delete(symbol) + optional_space + integer_part + insert_space + maj_tag
            reverse_symbol_patterns.append(reverse_pattern_with_space)

            # Patterns with minor currency (cents/xu)
            if symbol in currency_minor_map:
                minor_name = currency_minor_map[symbol]
                min_tag = pynutil.insert(f' currency_min: "{minor_name}"')

                # Minor-only pattern: 0,5$ or 0,5US$ -> năm mươi xu (highest priority)
                minor_only = (
                    pynutil.delete("0")
                    + pynutil.delete(NEMO_COMMA)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + pynutil.delete(symbol)
                    + preserve_order
                )
                minor_only_patterns.append(minor_only)
                
                # Reverse minor-only pattern: $0,5 or US$0,5 -> năm mươi xu
                reverse_minor_only = (
                    pynutil.delete(symbol)
                    + pynutil.delete("0")
                    + pynutil.delete(NEMO_COMMA)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + preserve_order
                )
                minor_only_patterns.append(reverse_minor_only)

                # Major + minor pattern: 10,5$ or 10,5US$ -> mười đô la năm mươi xu
                major_minor = (
                    integer_part
                    + insert_space
                    + maj_tag
                    + pynini.cross(NEMO_COMMA, NEMO_SPACE)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + pynutil.delete(symbol)
                    + preserve_order
                )
                symbol_patterns.append(major_minor)
                
                # Reverse major + minor pattern: $10,5 or US$10,5 -> mười đô la năm mươi xu
                # Apply to all symbols, not just single character ones
                reverse_major_minor = (
                    pynutil.delete(symbol)
                    + integer_part
                    + insert_space
                    + maj_tag
                    + pynini.cross(NEMO_COMMA, NEMO_SPACE)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + preserve_order
                )
                reverse_symbol_patterns.append(reverse_major_minor)
                
                # Reverse major + minor pattern with space: $ 10,5 or US$ 10,5 -> mười đô la năm mươi xu
                reverse_major_minor_with_space = (
                    pynutil.delete(symbol)
                    + optional_space
                    + integer_part
                    + insert_space
                    + maj_tag
                    + pynini.cross(NEMO_COMMA, NEMO_SPACE)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + preserve_order
                )
                reverse_symbol_patterns.append(reverse_major_minor_with_space)

        # 2. Word-based patterns
        word_patterns = []

        # Complex decimal + currency: 1tr5 vnd -> một triệu năm trăm nghìn đồng
        # Also handles cases like 321,6 tỷ USD -> ba trăm hai mươi mốt phẩy sáu tỷ đô la
        decimal_with_currency = (
            decimal_graph
            + optional_space
            + insert_space
            + pynutil.insert(' currency_maj: "')
            + convert_space(currency_major_graph)
            + pynutil.insert('"')
        )
        word_patterns.append(decimal_with_currency)

        # Quantity + currency: 10tr đồng -> mười triệu đồng
        quantity_tag = pynutil.insert(' quantity: "') + convert_space(quantity_graph) + pynutil.insert('"')
        quantity_pattern = (
            integer_part
            + optional_space
            + insert_space
            + quantity_tag
            + optional_space
            + insert_space
            + pynutil.insert(' currency_maj: "')
            + convert_space(currency_major_graph)
            + pynutil.insert('"')
        )
        word_patterns.append(quantity_pattern)

        # Simple word pattern: 10 đồng -> mười đồng
        simple_word_pattern = (
            integer_part
            + optional_space
            + insert_space
            + pynutil.insert(' currency_maj: "')
            + convert_space(currency_major_graph)
            + pynutil.insert('"')
        )
        word_patterns.append(simple_word_pattern)
        
        # Add reverse decimal patterns for symbols: $ 321,6 -> ba trăm hai mươi mốt phẩy sáu đô la
        # This should have higher priority than fractional cents patterns
        for symbol, major_name in currency_major_labels:
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')
            
            # Reverse decimal pattern: $321,6 -> ba trăm hai mươi mốt phẩy sáu đô la  
            reverse_decimal_pattern = (
                pynutil.delete(symbol)
                + optional_space
                + decimal_graph
                + insert_space
                + maj_tag
                + preserve_order
            )
            # Add with high priority to override fractional cents
            reverse_symbol_patterns.append(reverse_decimal_pattern)

        # 3. Generic combined patterns for complex currency cases
        combined_patterns = []
        
        # Define currency combinations that should be treated as single units
        currency_combinations = [
            ("$", "USD", "đô la"),
            ("US$", "USD", "đô la"), 
            ("$", "US", "đô la"),
            ("US$", "US", "đô la"),
        ]
        
        for symbol1, symbol2, unified_name in currency_combinations:
            unified_tag = pynutil.insert(f' currency_maj: "{unified_name}"')
            
            # Generic patterns for all combinations of symbol1, number, symbol2
            number_patterns = [
                (integer_part, "integer"),
                (decimal_graph, "decimal")
            ]
            
            for number_pattern, pattern_type in number_patterns:
                # Pattern 1: symbol1 number symbol2 (e.g., $ 10 USD)
                pattern1 = (
                    pynutil.delete(symbol1)
                    + optional_space
                    + number_pattern
                    + optional_space
                    + pynutil.delete(symbol2)
                    + insert_space
                    + unified_tag
                    + preserve_order
                )
                combined_patterns.append(pattern1)
                
                # Pattern 2: symbol2 symbol1 number (e.g., USD $ 10)
                pattern2 = (
                    pynutil.delete(symbol2)
                    + optional_space
                    + pynutil.delete(symbol1)
                    + optional_space
                    + number_pattern
                    + insert_space
                    + unified_tag
                    + preserve_order
                )
                combined_patterns.append(pattern2)
                
                # Pattern 3: number symbol1 symbol2 (e.g., 10 $ USD)
                pattern3 = (
                    number_pattern
                    + optional_space
                    + pynutil.delete(symbol1)
                    + optional_space
                    + pynutil.delete(symbol2)
                    + insert_space
                    + unified_tag
                    + preserve_order
                )
                combined_patterns.append(pattern3)

        # Combine patterns without weights
        # Combined patterns get highest priority (for $ digit USD cases)
        if combined_patterns:
            all_patterns.append(pynini.union(*combined_patterns))
            
        if minor_only_patterns:
            all_patterns.append(pynini.union(*minor_only_patterns))
            
        if reverse_symbol_patterns:
            all_patterns.append(pynini.union(*reverse_symbol_patterns))

        if symbol_patterns:
            all_patterns.append(pynini.union(*symbol_patterns))

        if word_patterns:
            all_patterns.append(pynini.union(*word_patterns))

        # Final graph with optional per-unit support
        final_graph = pynini.union(*all_patterns)
        per_unit_tag = pynutil.insert(' morphosyntactic_features: "') + per_unit_graph + pynutil.insert('"')
        final_graph += per_unit_tag.ques

        self.fst = self.add_tokens(final_graph.optimize())
