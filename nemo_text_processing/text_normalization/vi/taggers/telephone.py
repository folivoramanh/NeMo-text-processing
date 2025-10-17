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
    NEMO_ALPHA,
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese telephone numbers
    
    Vietnamese phone conventions:
        - Most popular format: 4-3-3 (e.g., 0912-345-678)
        - Plain 10-digit numbers starting with 0x (x ≠ 0): treated as telephone
        - International: +XX read digit-by-digit
        
    Args:
        cardinal: CardinalFst for digit conversion
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        add_separator = pynutil.insert(", ")
        
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        non_zero_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        digit = zero | non_zero_digit

        # Plain 10-digit Vietnamese phone numbers: 0[1-9]XXXXXXXX in 4-3-3 format
        non_zero_input = non_zero_digit.project("input")
        plain_10digit_pattern = pynini.accep("0") + non_zero_input + NEMO_DIGIT ** 8
        
        plain_10digit_graph = (
            digit + insert_space +
            digit + insert_space +
            digit + insert_space +
            digit + add_separator +
            digit + insert_space +
            digit + insert_space +
            digit + add_separator +
            digit + insert_space +
            digit + insert_space +
            digit
        )
        plain_10digit_graph = pynini.compose(plain_10digit_pattern, plain_10digit_graph)
        plain_10digit_graph = pynutil.insert("number_part: \"") + plain_10digit_graph + pynutil.insert("\"")
        
        # International format: +XX XXXXXXXXX (country code digit-by-digit, number in 3-3-3)
        country_code_digits = (
            pynutil.insert("country_code: \"") +
            pynini.cross("+", "cộng ") +
            pynini.closure(digit + insert_space, 1, 3) +
            digit +
            pynutil.insert(",\"") +
            insert_space
        )
        
        plain_intl_pattern = pynini.accep("+") + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT ** 9
        plain_intl_number = (
            digit + insert_space +
            digit + insert_space +
            digit + add_separator +
            digit + insert_space +
            digit + insert_space +
            digit + add_separator +
            digit + insert_space +
            digit + insert_space +
            digit
        )
        plain_intl_number = pynini.compose(NEMO_DIGIT ** 9, plain_intl_number)
        plain_intl_number = pynutil.insert("number_part: \"") + plain_intl_number + pynutil.insert("\"")
        
        plain_intl_graph = pynini.compose(
            plain_intl_pattern,
            country_code_digits + plain_intl_number
        )
        
        # Numbers with separators (hyphens, dots, parentheses, spaces)
        area_part_3digit = pynini.closure(digit + insert_space, 2, 2) + digit
        area_part_4digit = pynini.closure(digit + insert_space, 3, 3) + digit
        area_part_1800 = pynini.cross("1800", "một tám không không")
        
        area_part = area_part_1800 | area_part_3digit | area_part_4digit
        
        area_with_sep = (
            (area_part + (pynutil.delete("-") | pynutil.delete(".") | pynutil.delete(" ")))
            | (
                pynutil.delete("(") +
                area_part +
                ((pynutil.delete(")") + pynini.closure(pynutil.delete(" "), 0, 1)) | pynutil.delete(")-"))
            )
        ) + add_separator
        
        del_separator = pynini.closure(pynini.union("-", " ", "."), 0, 1)
        number_length = ((NEMO_DIGIT + del_separator) | (NEMO_ALPHA + del_separator)) ** (3, 8)
        
        number_words = pynini.closure(
            (NEMO_DIGIT @ digit) + (insert_space | (pynini.cross("-", ', ')))
            | NEMO_ALPHA
            | (NEMO_ALPHA + pynini.cross("-", ' '))
        )
        number_words |= pynini.closure(
            (NEMO_DIGIT @ digit) + (insert_space | (pynini.cross(".", ', ')))
            | NEMO_ALPHA
            | (NEMO_ALPHA + pynini.cross(".", ' '))
        )
        number_words |= pynini.closure(
            (NEMO_DIGIT @ digit) + (insert_space | (pynini.cross(" ", ', ')))
            | NEMO_ALPHA
            | (NEMO_ALPHA + pynini.cross(" ", ' '))
        )
        number_words = pynini.compose(number_length, number_words)
        
        number_part_with_sep = area_with_sep + number_words
        number_part_with_sep = pynutil.insert("number_part: \"") + number_part_with_sep + pynutil.insert("\"")
        
        country_with_sep = (
            country_code_digits +
            pynini.closure(pynutil.delete("-") | pynutil.delete(" "), 0, 1)
        )
        
        extension = (
            pynutil.insert("extension: \"") + 
            pynini.closure(digit + insert_space, 0, 3) + 
            digit + 
            pynutil.insert("\"")
        )
        extension = pynini.closure(insert_space + extension, 0, 1)
        
        def priority_union(high_priority, low_priority):
            return pynini.union(
                pynutil.add_weight(high_priority, -0.0001),
                low_priority
            ).optimize()
        
        graph = plain_10digit_graph
        graph = priority_union(plain_intl_graph, graph)
        graph = priority_union(country_with_sep + number_part_with_sep, graph)
        graph = priority_union(number_part_with_sep, graph)
        graph = priority_union(country_with_sep + number_part_with_sep + extension, graph)
        graph = priority_union(number_part_with_sep + extension, graph)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
