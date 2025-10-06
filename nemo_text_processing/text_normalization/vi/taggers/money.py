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
from nemo_text_processing.text_normalization.vi.vietnamese_phonetic_rules import VietnamesePhoneticRules


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

        # Initialize phonetic rules for non-deterministic alternatives
        if not deterministic:
            self.phonetic_rules = VietnamesePhoneticRules()

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

        # Build compound unit alternatives for non-deterministic mode
        slash = pynutil.delete("/")
        space = pynutil.insert(NEMO_SPACE)
        
        # Load non-metric per_unit entries
        graph_non_metric_per_units = pynini.string_file(per_unit_non_metric_path)
        
        if not deterministic:
            # Alternative 1: "một" (standard)
            one_connector = pynutil.insert("một ")
            graph_metric_per_units_mot = slash + one_connector + graph_prefixes + space + graph_bases
            graph_standalone_per_units_mot = slash + one_connector + graph_bases
            
            # Alternative 2: "mỗi" 
            moi_connector = pynutil.insert("mỗi ")
            graph_metric_per_units_moi = slash + moi_connector + graph_prefixes + space + graph_bases
            graph_standalone_per_units_moi = slash + moi_connector + graph_bases
            
            # Alternative 3: "trên"
            tren_connector = pynutil.insert("trên ")
            graph_metric_per_units_tren = slash + tren_connector + graph_prefixes + space + graph_bases
            graph_standalone_per_units_tren = slash + tren_connector + graph_bases
            
            # Combine all compound alternatives
            per_unit_graph = (
                graph_metric_per_units_mot | graph_standalone_per_units_mot |
                graph_metric_per_units_moi | graph_standalone_per_units_moi |
                graph_metric_per_units_tren | graph_standalone_per_units_tren |
                graph_non_metric_per_units
            )
        else:
            # Deterministic mode: only "một"
            one_connector = pynutil.insert("một ")
            graph_metric_per_units = slash + one_connector + graph_prefixes + space + graph_bases
            graph_standalone_per_units = slash + one_connector + graph_bases
            per_unit_graph = graph_metric_per_units | graph_standalone_per_units | graph_non_metric_per_units

        # Basic components - enhanced for non-deterministic mode
        if deterministic:
            cardinal_graph = cardinal.graph
            self.cardinal_alternatives = pynini.closure(pynini.union(*"0123456789"), 1)
        else:
            cardinal_graph = self._create_enhanced_cardinal_graph(cardinal.graph)
            self.cardinal_alternatives = self._create_enhanced_cardinal_graph(pynini.closure(pynini.union(*"0123456789"), 1))
            
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

        # Enhanced patterns for non-deterministic mode
        if not deterministic:
            # Add fraction alternatives (working)
            fraction_patterns = self._create_fraction_alternatives()
            all_patterns.append(fraction_patterns)
            
            # Add decimal unit alternatives (working)
            decimal_unit_patterns = self._create_decimal_unit_alternatives(currency_major_labels, currency_minor_map)
            all_patterns.append(decimal_unit_patterns)
            
            flexible_patterns = self._create_flexible_currency_patterns(currency_major_labels, currency_minor_map)
            all_patterns.append(flexible_patterns)

        # Original patterns (for deterministic mode or as fallback)
        # 1. Symbol-based patterns
        symbol_patterns = []
        reverse_symbol_patterns = []  # Higher priority for reverse patterns
        minor_only_patterns = []

        for symbol, major_name in currency_major_labels:
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')

            # Simple integer pattern: 10$ -> mười đô la
            simple_pattern = integer_part + pynutil.delete(symbol) + insert_space + maj_tag
            symbol_patterns.append(simple_pattern)
            
            # Reverse pattern: $10 -> mười đô la (symbol before number)
            # Only for actual symbols (not words)
            if len(symbol) == 1 and not symbol.isalpha():
                # Use pynini.accep for literal symbol matching
                symbol_fst = pynini.accep(symbol)
                reverse_pattern = pynutil.delete(symbol_fst) + integer_part + insert_space + maj_tag
                reverse_symbol_patterns.append(reverse_pattern)

            # Patterns with minor currency (cents/xu)
            if symbol in currency_minor_map:
                minor_name = currency_minor_map[symbol]
                min_tag = pynutil.insert(f' currency_min: "{minor_name}"')

                # Minor-only pattern: 0,5$ -> năm mươi xu (highest priority)
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

                # Major + minor pattern: 10,5$ -> mười đô la năm mươi xu
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
                
                # Reverse major + minor pattern: $10,5 -> mười đô la năm mươi xu
                # Only for actual symbols (not words)
                if len(symbol) == 1 and not symbol.isalpha():
                    # Use pynini.accep for literal symbol matching
                    symbol_fst = pynini.accep(symbol)
                    reverse_major_minor = (
                        pynutil.delete(symbol_fst)
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
                
                # Reverse major + minor pattern: $10,5 -> mười đô la năm mươi xu
                # Only for actual symbols (not words)
                if len(symbol) == 1 and not symbol.isalpha():
                    # Use pynini.accep for literal symbol matching
                    symbol_fst = pynini.accep(symbol)
                    reverse_major_minor = (
                        pynutil.delete(symbol_fst)
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

        # 2. Word-based patterns
        word_patterns = []

        # Complex decimal + currency: 1tr5 vnd -> một triệu năm trăm nghìn đồng
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

        # Combine patterns with priorities
        # Reverse symbol patterns get highest priority (for $10 patterns)
        if reverse_symbol_patterns:
            all_patterns.append(pynutil.add_weight(pynini.union(*reverse_symbol_patterns), -0.0002))
            
        # Minor-only patterns get high priority (negative weight)
        # Reverse symbol patterns get highest priority (for $10 patterns)
        if reverse_symbol_patterns:
            all_patterns.append(pynutil.add_weight(pynini.union(*reverse_symbol_patterns), -0.0002))
            
        # Minor-only patterns get high priority (negative weight)
        if minor_only_patterns:
            all_patterns.append(pynutil.add_weight(pynini.union(*minor_only_patterns), -0.0001))

        # Symbol patterns get normal priority
        if symbol_patterns:
            all_patterns.append(pynini.union(*symbol_patterns))

        # Word patterns get lowest priority
        if word_patterns:
            all_patterns.append(pynutil.add_weight(pynini.union(*word_patterns), 0.1))

        # Final graph with optional per-unit support
        final_graph = pynini.union(*all_patterns)
        per_unit_tag = pynutil.insert(' morphosyntactic_features: "') + per_unit_graph + pynutil.insert('"')
        final_graph += per_unit_tag.ques

        self.fst = self.add_tokens(final_graph.optimize())

    def _create_enhanced_cardinal_graph(self, base_cardinal_graph):
        """
        Create enhanced cardinal graph with phonetic alternatives for non-deterministic mode.
        """
        # For numbers that commonly appear in money contexts, add alternatives
        alternatives = []
        
        # Add base cardinal graph
        alternatives.append(base_cardinal_graph)
        
        # Add phonetic alternatives for common money amounts
        for num in range(1, 101):  # Common money amounts 1-100
            num_str = str(num)
            phonetic_alts = self.phonetic_rules.generate_alternatives(num_str, "general")
            
            for alt in phonetic_alts:
                if alt != num_str:  # Don't duplicate base form
                    alternatives.append(pynini.cross(num_str, alt))
        
        return pynini.union(*alternatives)

    def _create_flexible_currency_patterns(self, currency_major_labels, currency_minor_map):
        """
        Create patterns that support flexible currency symbol positions:
        $10, $ 10, 10$, 10 $
        """
        all_patterns = []
        
        for symbol, major_name in currency_major_labels:
            # Skip word-based currencies (only process symbols)
            if len(symbol) > 3 or symbol.isalpha():
                continue
                
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')
            optional_space = pynini.closure(delete_space, 0, 1)
            
            # Pattern 1: $10 (symbol before, no space)
            pattern1 = (
                pynutil.delete(symbol) +
                pynutil.insert('integer_part: "') + 
                self._get_cardinal_alternatives() + 
                pynutil.insert('"') +
                insert_space + maj_tag
            )
            all_patterns.append(pattern1)
            
            # Pattern 2: $ 10 (symbol before, with space)  
            pattern2 = (
                pynutil.delete(symbol) +
                optional_space +
                pynutil.insert('integer_part: "') + 
                self._get_cardinal_alternatives() + 
                pynutil.insert('"') +
                insert_space + maj_tag
            )
            all_patterns.append(pattern2)
            
            # Pattern 3: 10$ (symbol after, no space)
            pattern3 = (
                pynutil.insert('integer_part: "') + 
                self._get_cardinal_alternatives() + 
                pynutil.insert('"') +
                pynutil.delete(symbol) +
                insert_space + maj_tag
            )
            all_patterns.append(pattern3)
            
            # Pattern 4: 10 $ (symbol after, with space)
            pattern4 = (
                pynutil.insert('integer_part: "') + 
                self._get_cardinal_alternatives() + 
                pynutil.insert('"') +
                optional_space +
                pynutil.delete(symbol) +
                insert_space + maj_tag
            )
            all_patterns.append(pattern4)
            
            # Decimal patterns with flexible positions
            if symbol in currency_minor_map:
                minor_name = currency_minor_map[symbol]
                min_tag = pynutil.insert(f' currency_min: "{minor_name}"')
                
                # Add decimal patterns for each position
                for base_pattern in [pattern1, pattern2, pattern3, pattern4]:
                    # Base patterns already handle decimal cases
                    all_patterns.append(base_pattern)
        
        return pynini.union(*all_patterns)

    def _get_cardinal_alternatives(self):
        """Get cardinal graph with alternatives."""
        return self.cardinal_alternatives

    def _create_fraction_alternatives(self):
        """
        Create fraction patterns with "trên" and "một" alternatives:
        a/b -> "a trên b" OR "a một b"
        """
        alternatives = []
        
        # Basic fraction pattern: number/number
        numerator = pynutil.insert('integer_part: "') + self._get_cardinal_alternatives() + pynutil.insert('"')
        denominator = pynutil.insert('fractional_part: "') + self._get_cardinal_alternatives() + pynutil.insert('"')
        
        # Alternative 1: "a trên b"
        fraction_with_tren = (
            numerator +
            pynini.cross("/", " trên ") +
            denominator
        )
        alternatives.append(fraction_with_tren)
        
        # Alternative 2: "a một b" (less common)
        fraction_with_mot = (
            numerator +
            pynini.cross("/", " một ") +
            denominator
        )
        alternatives.append(fraction_with_mot)
        
        return pynini.union(*alternatives)

    def _create_decimal_unit_alternatives(self, currency_major_labels, currency_minor_map, name = "money"):
        """
        Create decimal unit reading alternatives:
        10,5$ -> "mười phẩy năm đô la" OR "mười đô la năm mười xu"
        """
        alternatives = []
        
        for symbol, major_name in currency_major_labels:
            if symbol not in currency_minor_map:
                continue
                
            minor_name = currency_minor_map[symbol]
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')
            min_tag = pynutil.insert(f' currency_min: "{minor_name}"')
            preserve_order = pynutil.insert(" preserve_order: true")
            
            # Pattern: integer,fractional + currency
            integer_part = pynutil.insert('integer_part: "') + self._get_cardinal_alternatives() + pynutil.insert('"')
            fractional_part = pynutil.insert('fractional_part: "') + self._get_cardinal_alternatives() + pynutil.insert('"')
            
            # Alternative 1: "mười phẩy năm đô la" (decimal reading)
            decimal_reading = (
                pynutil.insert(name + " { ") +
                integer_part +
                pynini.cross(NEMO_COMMA, ' ') +
                pynutil.insert('fractional_part: "phẩy ') +
                self._get_cardinal_alternatives() +
                pynutil.insert('" ') +
                pynutil.delete(symbol) +
                maj_tag +
                pynutil.insert(" }")
            )
            alternatives.append(decimal_reading)
            
            # Alternative 2: "mười đô la năm mười xu" (major + minor reading)
            # Use existing fractional_conversion logic instead of custom method
            minor_reading = (
                pynutil.insert(name + " { ") +
                integer_part +
                maj_tag +
                pynini.cross(NEMO_COMMA, ' ') +
                fractional_part +  # Use existing fractional_part logic
                pynutil.delete(symbol) +
                min_tag +
                preserve_order +
                pynutil.insert(" }")
            )
            alternatives.append(minor_reading)
        
        return pynini.union(*alternatives) if alternatives else pynini.accep("")
