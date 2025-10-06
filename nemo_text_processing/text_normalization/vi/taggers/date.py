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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.vi.vietnamese_phonetic_rules import VietnamesePhoneticRules


class DateFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese dates, e.g.
        15/01/2024 -> date { day: "mười lăm" month: "một" year: "hai nghìn hai mươi tư" }
        tháng 4 2024 -> date { month: "tư" year: "hai nghìn hai mươi tư" }
        ngày 15/01/2024 -> date { day: "mười lăm" month: "một" year: "hai nghìn hai mươi tư" }
        ngày 12 tháng 5 năm 2025 -> date { day: "mười hai" month: "năm" year: "hai nghìn hai mươi lăm" }
        năm 20 SCN -> date { year: "hai mươi" era: "sau công nguyên" }
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        # Initialize phonetic rules for non-deterministic alternatives
        if not deterministic:
            self.phonetic_rules = VietnamesePhoneticRules()

        # Vietnamese date keywords
        DAY_WORD = "ngày"
        MONTH_WORD = "tháng"
        YEAR_WORD = "năm"
        ORDINAL_YEAR_WORD = "năm thứ"

        # Prebuilt patterns for common usage
        # Expand day_prefix to include all possible day prefixes
        day_prefix = pynini.union(
            pynini.accep("ngày "),
            pynini.accep("mùng "),
            pynini.accep("mồng ")
        )
        month_prefix = pynini.accep(MONTH_WORD + NEMO_SPACE)
        year_prefix = pynini.accep(YEAR_WORD + NEMO_SPACE)
        ordinal_year_prefix = pynini.accep(ORDINAL_YEAR_WORD + NEMO_SPACE)

        delete_day_prefix = pynutil.delete(DAY_WORD + NEMO_SPACE)
        delete_month_prefix = pynutil.delete(MONTH_WORD + NEMO_SPACE)
        delete_year_prefix = pynutil.delete(YEAR_WORD + NEMO_SPACE)
        delete_ordinal_year_prefix = pynutil.delete(ORDINAL_YEAR_WORD + NEMO_SPACE)

        day_mappings = load_labels(get_abs_path("data/date/days.tsv"))
        month_mappings = load_labels(get_abs_path("data/date/months.tsv"))
        era_mappings = load_labels(get_abs_path("data/date/year_suffix.tsv"))

        day_digit = pynini.closure(NEMO_DIGIT, 1, 2)
        month_digit = pynini.closure(NEMO_DIGIT, 1, 2)
        year_digit = pynini.closure(NEMO_DIGIT, 1, 4)
        separator = pynini.union("/", "-", ".")

        day_convert = pynini.string_map([(k, v) for k, v in day_mappings])
        month_convert = pynini.string_map([(k, v) for k, v in month_mappings])
        year_convert = pynini.compose(year_digit, cardinal.graph)

        # Enhanced converters for non-deterministic mode
        if not deterministic:
            day_convert = self._create_enhanced_day_converter(day_mappings)
            month_convert = self._create_enhanced_month_converter(month_mappings)
            year_convert = self._create_enhanced_year_converter(year_digit, cardinal.graph)

        era_to_full = {}
        for abbr, full_form in era_mappings:
            era_to_full[abbr.lower()] = full_form
            era_to_full[abbr.upper()] = full_form

        era_convert = pynini.string_map([(k, v) for k, v in era_to_full.items()])

        # Create day/month/year parts - enhanced for non-deterministic mode
        if not deterministic:
            # For non-deterministic mode, day_convert already contains all prefix alternatives
            day_part = pynutil.insert("day: \"") + day_convert + pynutil.insert("\" ")
            month_part = pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ")
            year_part = pynutil.insert("year: \"") + year_convert + pynutil.insert("\"")
            month_final = pynutil.insert("month: \"") + month_convert + pynutil.insert("\"")
            
            # Create converters without prefixes for cases where input already has prefix
            day_convert_no_prefix = self._create_enhanced_day_converter_no_prefix(day_mappings)
            month_convert_no_prefix = self._create_enhanced_month_converter_no_prefix(month_mappings)
            year_convert_no_prefix = self._create_enhanced_year_converter_no_prefix(year_digit, cardinal.graph)
            
            day_part_no_prefix = pynutil.insert("day: \"") + day_convert_no_prefix + pynutil.insert("\" ")
            month_part_no_prefix = pynutil.insert("month: \"") + month_convert_no_prefix + pynutil.insert("\" ")
            year_part_no_prefix = pynutil.insert("year: \"") + year_convert_no_prefix + pynutil.insert("\"")
            month_final_no_prefix = pynutil.insert("month: \"") + month_convert_no_prefix + pynutil.insert("\"")
        else:
            # Standard deterministic mode
            day_part = pynutil.insert("day: \"") + day_convert + pynutil.insert("\" ")
            month_part = pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ")
            year_part = pynutil.insert("year: \"") + year_convert + pynutil.insert("\"")
            month_final = pynutil.insert("month: \"") + month_convert + pynutil.insert("\"")
            
            # No-prefix versions same as regular for deterministic mode
            day_part_no_prefix = day_part
            month_part_no_prefix = month_part  
            year_part_no_prefix = year_part
            month_final_no_prefix = month_final
            day_convert_no_prefix = day_convert
            month_convert_no_prefix = month_convert
            year_convert_no_prefix = year_convert
        era_part = pynutil.insert("era: \"") + era_convert + pynutil.insert("\"")

        patterns = []

        # DD/MM/YYYY format (Vietnamese standard)
        date_sep = day_part + pynutil.delete(separator) + month_part + pynutil.delete(separator) + year_part
        patterns.append(pynini.compose(day_digit + separator + month_digit + separator + year_digit, date_sep))
        
        # Higher priority for patterns that preserve input prefixes
        prefix_preserving_pattern = pynini.compose(
            day_prefix + day_digit + separator + month_digit + separator + year_digit,
            pynutil.insert("day: \"") + day_prefix + day_convert_no_prefix + pynutil.insert("\" ") + pynutil.delete(separator) + pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ") + pynutil.delete(separator) + pynutil.insert("year: \"") + year_convert + pynutil.insert("\""),
        )
        patterns.append(pynutil.add_weight(prefix_preserving_pattern, -0.1))  # Higher priority

        # YYYY/MM/DD format (ISO standard) - output in Vietnamese order
        iso_year_part = pynutil.insert("year: \"") + year_convert + pynutil.insert("\" ")
        iso_month_part = pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ")
        iso_day_part = pynutil.insert("day: \"") + day_convert + pynutil.insert("\"")

        iso_date_sep = (
            iso_year_part + pynutil.delete(separator) + iso_month_part + pynutil.delete(separator) + iso_day_part
        )
        patterns.append(pynini.compose(year_digit + separator + month_digit + separator + day_digit, iso_date_sep))

        for sep in [separator, pynini.accep(NEMO_SPACE)]:
            month_prefix_pattern = pynini.compose(
                month_prefix + month_digit + sep + year_digit,
                pynutil.insert("month: \"") + month_prefix + month_convert_no_prefix + pynutil.insert("\" ") + pynutil.delete(sep) + pynutil.insert("year: \"") + year_convert + pynutil.insert("\""),
            )
            patterns.append(pynutil.add_weight(month_prefix_pattern, -0.1))  # Higher priority

        day_month_prefix_pattern = pynini.compose(day_prefix + day_digit + separator + month_digit, pynutil.insert("day: \"") + day_prefix + day_convert_no_prefix + pynutil.insert("\" ") + pynutil.delete(separator) + pynutil.insert("month: \"") + month_convert + pynutil.insert("\""))
        patterns.append(pynutil.add_weight(day_month_prefix_pattern, -0.1))  # Higher priority

        day_month_word_prefix_pattern = pynini.compose(
            day_prefix + day_digit + pynini.accep(NEMO_SPACE + MONTH_WORD + NEMO_SPACE) + month_digit,
            pynutil.insert("day: \"") + day_prefix + day_convert_no_prefix + pynutil.insert("\" ") + pynutil.delete(NEMO_SPACE + MONTH_WORD + NEMO_SPACE) + pynutil.insert("month: \"") + month_convert + pynutil.insert("\""),
        )
        patterns.append(pynutil.add_weight(day_month_word_prefix_pattern, -0.1))  # Higher priority

        patterns.append(
            pynini.compose(
                day_prefix
                + day_digit
                + pynini.accep(NEMO_SPACE + MONTH_WORD + NEMO_SPACE)
                + month_digit
                + pynini.accep(NEMO_SPACE + YEAR_WORD + NEMO_SPACE)
                + year_digit,
                delete_day_prefix
                + day_part
                + pynutil.delete(NEMO_SPACE + MONTH_WORD + NEMO_SPACE)
                + month_part
                + pynutil.delete(NEMO_SPACE + YEAR_WORD + NEMO_SPACE)
                + year_part,
            )
        )

        patterns.append(pynini.compose(year_prefix + year_digit, delete_year_prefix + year_part))

        era_abbrs = list(era_to_full.keys())
        for era_abbr in era_abbrs:
            patterns.append(
                pynini.compose(
                    year_prefix + year_digit + pynini.accep(NEMO_SPACE) + pynini.accep(era_abbr),
                    delete_year_prefix + year_part + pynutil.delete(NEMO_SPACE) + era_part,
                )
            )

            patterns.append(
                pynini.compose(
                    ordinal_year_prefix + year_digit + pynini.accep(NEMO_SPACE) + pynini.accep(era_abbr),
                    delete_ordinal_year_prefix
                    + pynutil.insert("ordinal: \"")
                    + year_convert
                    + pynutil.insert("\" ")
                    + pynutil.delete(NEMO_SPACE)
                    + era_part,
                )
            )

        self.fst = self.add_tokens(pynini.union(*patterns))

    def _create_enhanced_day_converter(self, base_mappings):
        """Create enhanced day converter WITHOUT prefixes - used by standard patterns"""
        alternatives = []
        
        # Add base mappings (without prefix) - for backward compatibility
        for k, v in base_mappings:
            alternatives.append(pynini.cross(k, v))
        
        # Optimize: Generate alternatives for all days at once to reduce redundant calls
        day_alternatives = {}
        
        # Generate alternatives for days 1-31 (WITHOUT prefixes only)
        for day in range(1, 32):
            day_str = str(day)
            day_alts_no_prefix = self.phonetic_rules.generate_date_day_alternatives(day_str, include_prefix=False)
            
            # Store for reuse
            day_alternatives[day_str] = day_alts_no_prefix
        
        # Add alternatives WITHOUT prefixes to FST
        for day_str, alts in day_alternatives.items():
            for alt in alts:
                alternatives.append(pynini.cross(day_str, alt))
                
        # Add zero-padded versions (reuse cached alternatives)
        for day in range(1, 10):
            padded_day_str = f"0{day}"
            day_str = str(day)
            if day_str in day_alternatives:
                # Without prefixes only
                for alt in day_alternatives[day_str]:
                    alternatives.append(pynini.cross(padded_day_str, alt))
        
        return pynini.union(*alternatives).optimize() if alternatives else pynini.string_map([(k, v) for k, v in base_mappings])

    def _create_enhanced_month_converter(self, base_mappings):
        """Create enhanced month converter with alternatives including tháng prefix"""
        alternatives = []
        
        # Add base mappings (without prefix)
        for k, v in base_mappings:
            alternatives.append(pynini.cross(k, v))
        
        # Optimize: Generate alternatives for all months at once
        month_alternatives = {}
        
        # Generate alternatives for months 1-12
        for month in range(1, 13):
            month_str = str(month)
            month_alts_with_prefix = self.phonetic_rules.generate_date_month_alternatives(month_str, include_prefix=True)
            if len(month_alts_with_prefix) > 0:
                month_alternatives[month_str] = month_alts_with_prefix
        
        # Add alternatives to FST
        for month_num, alts in month_alternatives.items():
            for alt in alts:
                alternatives.append(pynini.cross(month_num, alt))
                
        # Add zero-padded versions (reuse cached alternatives)
        for month in range(1, 10):
            padded_month_str = f"0{month}"
            month_str = str(month)
            if month_str in month_alternatives:
                for alt in month_alternatives[month_str]:
                    alternatives.append(pynini.cross(padded_month_str, alt))
        
        return pynini.union(*alternatives).optimize() if alternatives else pynini.string_map([(k, v) for k, v in base_mappings])

    def _create_enhanced_year_converter(self, year_digit, cardinal_graph):
        """Create enhanced year converter with alternatives including năm prefix"""
        alternatives = []
        
        # Add base cardinal composition (without prefix) for backward compatibility
        base_year_converter = pynini.compose(year_digit, cardinal_graph)
        alternatives.append(base_year_converter)
        
        # Add alternatives with "năm" prefix using phonetic rules
        if hasattr(self, 'phonetic_rules'):
            year_alternatives = {}
            
            # Generate alternatives for common years (1900-2100)
            for year in range(1900, 2101):
                year_str = str(year)
                year_alts_with_prefix = self.phonetic_rules.generate_date_year_alternatives(year_str, include_prefix=True)
                if len(year_alts_with_prefix) > 0:
                    year_alternatives[year_str] = year_alts_with_prefix
            
            # Add alternatives to FST
            for year_num, alts in year_alternatives.items():
                for alt in alts:
                    alternatives.append(pynini.cross(year_num, alt))
        
        return pynini.union(*alternatives).optimize() if len(alternatives) > 1 else base_year_converter

    def _create_enhanced_day_converter_no_prefix(self, base_mappings):
        """Create enhanced day converter without prefixes - for cases where input already has prefix"""
        alternatives = []
        
        # Add base mappings (without prefix)
        for k, v in base_mappings:
            alternatives.append(pynini.cross(k, v))
        
        # Add alternatives without prefixes only
        if hasattr(self, 'phonetic_rules'):
            day_alternatives = {}
            
            # Generate alternatives for days 1-31 (no prefixes)
            for day in range(1, 32):
                day_str = str(day)
                day_alts_no_prefix = self.phonetic_rules.generate_date_day_alternatives(day_str, include_prefix=False)
                day_alternatives[day_str] = day_alts_no_prefix
            
            # Add alternatives to FST
            for day_str, alts in day_alternatives.items():
                for alt in alts:
                    alternatives.append(pynini.cross(day_str, alt))
                    
            # Add zero-padded versions (reuse cached alternatives)
            for day in range(1, 10):
                padded_day_str = f"0{day}"
                day_str = str(day)
                if day_str in day_alternatives:
                    for alt in day_alternatives[day_str]:
                        alternatives.append(pynini.cross(padded_day_str, alt))
        
        return pynini.union(*alternatives).optimize() if alternatives else pynini.string_map([(k, v) for k, v in base_mappings])

    def _create_enhanced_month_converter_no_prefix(self, base_mappings):
        """Create enhanced month converter without prefixes - for cases where input already has prefix"""
        alternatives = []
        
        # Add base mappings (without prefix)
        for k, v in base_mappings:
            alternatives.append(pynini.cross(k, v))
        
        # Add alternatives without prefixes only
        if hasattr(self, 'phonetic_rules'):
            month_alternatives = {}
            
            # Generate alternatives for months 1-12 (no prefixes)
            for month in range(1, 13):
                month_str = str(month)
                month_alts_no_prefix = self.phonetic_rules.generate_date_month_alternatives(month_str, include_prefix=False)
                if len(month_alts_no_prefix) > 0:
                    month_alternatives[month_str] = month_alts_no_prefix
            
            # Add alternatives to FST
            for month_num, alts in month_alternatives.items():
                for alt in alts:
                    alternatives.append(pynini.cross(month_num, alt))
                    
            # Add zero-padded versions (reuse cached alternatives)
            for month in range(1, 10):
                padded_month_str = f"0{month}"
                month_str = str(month)
                if month_str in month_alternatives:
                    for alt in month_alternatives[month_str]:
                        alternatives.append(pynini.cross(padded_month_str, alt))
        
        return pynini.union(*alternatives).optimize() if alternatives else pynini.string_map([(k, v) for k, v in base_mappings])

    def _create_enhanced_year_converter_no_prefix(self, year_digit, cardinal_graph):
        """Create enhanced year converter without prefixes - for cases where input already has prefix"""
        alternatives = []
        
        # Add base cardinal composition (without prefix)
        base_year_converter = pynini.compose(year_digit, cardinal_graph)
        alternatives.append(base_year_converter)
        
        # Add alternatives without "năm" prefix using phonetic rules
        if hasattr(self, 'phonetic_rules'):
            year_alternatives = {}
            
            # Generate alternatives for common years (1900-2100) without prefix
            for year in range(1900, 2101):
                year_str = str(year)
                year_alts_no_prefix = self.phonetic_rules.generate_date_year_alternatives(year_str, include_prefix=False)
                if len(year_alts_no_prefix) > 0:
                    year_alternatives[year_str] = year_alts_no_prefix
            
            # Add alternatives to FST
            for year_num, alts in year_alternatives.items():
                for alt in alts:
                    alternatives.append(pynini.cross(year_num, alt))
        
        return pynini.union(*alternatives).optimize() if len(alternatives) > 1 else base_year_converter
