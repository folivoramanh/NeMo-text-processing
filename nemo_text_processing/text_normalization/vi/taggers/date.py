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
        day_prefix = pynini.accep(DAY_WORD + NEMO_SPACE)
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
        else:
            # Standard deterministic mode
            day_part = pynutil.insert("day: \"") + day_convert + pynutil.insert("\" ")
            month_part = pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ")
            year_part = pynutil.insert("year: \"") + year_convert + pynutil.insert("\"")
            month_final = pynutil.insert("month: \"") + month_convert + pynutil.insert("\"")
            
        era_part = pynutil.insert("era: \"") + era_convert + pynutil.insert("\"")

        patterns = []

        # DD/MM/YYYY format (Vietnamese standard)
        date_sep = day_part + pynutil.delete(separator) + month_part + pynutil.delete(separator) + year_part
        patterns.append(pynini.compose(day_digit + separator + month_digit + separator + year_digit, date_sep))
        patterns.append(
            pynini.compose(
                day_prefix + day_digit + separator + month_digit + separator + year_digit,
                delete_day_prefix + date_sep,
            )
        )

        # YYYY/MM/DD format (ISO standard) - output in Vietnamese order
        iso_year_part = pynutil.insert("year: \"") + year_convert + pynutil.insert("\" ")
        iso_month_part = pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ")
        iso_day_part = pynutil.insert("day: \"") + day_convert + pynutil.insert("\"")

        iso_date_sep = (
            iso_year_part + pynutil.delete(separator) + iso_month_part + pynutil.delete(separator) + iso_day_part
        )
        patterns.append(pynini.compose(year_digit + separator + month_digit + separator + day_digit, iso_date_sep))

        for sep in [separator, pynini.accep(NEMO_SPACE)]:
            patterns.append(
                pynini.compose(
                    month_prefix + month_digit + sep + year_digit,
                    delete_month_prefix + month_part + pynutil.delete(sep) + year_part,
                )
            )

        day_month_sep = day_part + pynutil.delete(separator) + month_final
        patterns.append(
            pynini.compose(day_prefix + day_digit + separator + month_digit, delete_day_prefix + day_month_sep)
        )

        patterns.append(
            pynini.compose(
                day_prefix + day_digit + pynini.accep(NEMO_SPACE + MONTH_WORD + NEMO_SPACE) + month_digit,
                delete_day_prefix + day_part + pynutil.delete(NEMO_SPACE + MONTH_WORD + NEMO_SPACE) + month_final,
            )
        )

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
        """Create enhanced day converter with alternatives including ngày/mùng/mồng prefixes"""
        alternatives = []
        
        # Add base mappings (without prefix) - for backward compatibility
        for k, v in base_mappings:
            alternatives.append(pynini.cross(k, v))
        
        # Add alternatives with Vietnamese day prefixes using phonetic rules
        for day in range(1, 32):  # Days 1-31
            day_str = str(day)
            # Get alternatives with prefixes (ngày/mùng/mồng)
            day_alts_with_prefix = self.phonetic_rules.generate_date_day_alternatives(day_str, include_prefix=True)
            for alt in day_alts_with_prefix:
                alternatives.append(pynini.cross(day_str, alt))
                
        # Add zero-padded versions
        for day in range(1, 10):
            day_str = f"0{day}"
            day_alts_with_prefix = self.phonetic_rules.generate_date_day_alternatives(str(day), include_prefix=True)
            for alt in day_alts_with_prefix:
                alternatives.append(pynini.cross(day_str, alt))
        
        # Also add alternatives without prefixes (for number-only formats)
        for day in range(1, 32):
            day_str = str(day)
            day_alts_no_prefix = self.phonetic_rules.generate_date_day_alternatives(day_str, include_prefix=False)
            for alt in day_alts_no_prefix:
                alternatives.append(pynini.cross(day_str, alt))
                
        # Add zero-padded versions without prefixes
        for day in range(1, 10):
            day_str = f"0{day}"
            day_alts_no_prefix = self.phonetic_rules.generate_date_day_alternatives(str(day), include_prefix=False)
            for alt in day_alts_no_prefix:
                alternatives.append(pynini.cross(day_str, alt))
        
        return pynini.union(*alternatives).optimize() if alternatives else pynini.string_map([(k, v) for k, v in base_mappings])

    def _create_enhanced_month_converter(self, base_mappings):
        """Create enhanced month converter with alternatives including tháng prefix"""
        alternatives = []
        
        # Add base mappings (without prefix)
        for k, v in base_mappings:
            alternatives.append(pynini.cross(k, v))
        
        # Add alternatives with Vietnamese month prefixes using phonetic rules
        month_alternatives = {}
        for month in range(1, 13):  # Months 1-12
            month_str = str(month)
            # Get alternatives with "tháng" prefix
            month_alts_with_prefix = self.phonetic_rules.generate_date_month_alternatives(month_str, include_prefix=True)
            if len(month_alts_with_prefix) > 0:
                month_alternatives[month_str] = month_alts_with_prefix
                
        # Add zero-padded versions
        for month in range(1, 10):
            month_str = f"0{month}"
            month_alts_with_prefix = self.phonetic_rules.generate_date_month_alternatives(str(month), include_prefix=True)
            if len(month_alts_with_prefix) > 0:
                month_alternatives[month_str] = month_alts_with_prefix
        
        # Create alternative patterns
        for month_num, alts in month_alternatives.items():
            for alt in alts:
                alternatives.append(pynini.cross(month_num, alt))
        
        return pynini.union(*alternatives).optimize() if alternatives else pynini.string_map([(k, v) for k, v in base_mappings])

    def _create_enhanced_year_converter(self, year_digit, cardinal_graph):
        """Create enhanced year converter with alternatives"""
        # Use the existing cardinal graph which already has alternatives
        # The cardinal FST already handles number alternatives
        return pynini.compose(year_digit, cardinal_graph)
