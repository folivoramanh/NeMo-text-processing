# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese telephone-like digit strings, e.g.
        093-655-5449 -> telephone { number_part: "không chín ba sáu năm năm năm bốn bốn chín" preserve_order: true }
        +84 093-655-5449 -> telephone { country_code: "tám mươi tư" number_part: "không chín ba sáu năm năm năm bốn bốn chín" preserve_order: true }
        192.168.0.1 -> telephone { number_part: "một chín hai chấm một sáu tám chấm không chấm một" preserve_order: true }

    Args:
        cardinal: CardinalFst for country code verbalization
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        zero = pynini.cross("0", "không")
        digit = (zero | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()

        def digits(count: int):
            graph = pynini.accep("") + digit
            for _ in range(count - 1):
                graph = graph + insert_space + digit
            return graph

        def zero_prefixed_digits(count: int):
            graph = pynini.accep("") + zero
            for _ in range(count - 1):
                graph = graph + insert_space + digit
            return graph

        delete_sep = pynutil.delete("-") | pynutil.delete(".") | pynutil.delete(NEMO_SPACE)
        optional_sep = pynini.closure(delete_sep, 0, 1) + insert_space
        required_sep = pynini.closure(delete_sep, 1) + insert_space
        optional_deleted_sep = pynini.closure(delete_sep, 0, 1)

        three_digits = digits(3)
        four_digits = digits(4)
        zero_three_digits = zero_prefixed_digits(3)
        zero_four_digits = zero_prefixed_digits(4)

        area_three = three_digits | (pynutil.delete("(") + three_digits + pynutil.delete(")"))
        zero_area_three = zero_three_digits | (pynutil.delete("(") + zero_three_digits + pynutil.delete(")"))

        # National Vietnamese-looking numbers may be written compactly; require a leading 0
        # to avoid swallowing ordinary long cardinals.
        national_number = (
            zero_area_three + optional_sep + three_digits + optional_sep + four_digits
            | zero_area_three + optional_sep + four_digits + optional_sep + four_digits
            | zero_four_digits + optional_sep + three_digits + optional_sep + four_digits
        )

        # Non-zero area codes are accepted only with explicit grouping, or after a country code below.
        grouped_number = area_three + required_sep + (three_digits | four_digits) + required_sep + four_digits

        international_number = (
            area_three + optional_sep + three_digits + optional_sep + four_digits
            | area_three + optional_sep + four_digits + optional_sep + four_digits
            | three_digits + optional_sep + three_digits + optional_sep + three_digits
            | four_digits + optional_sep + three_digits + optional_sep + four_digits
        )

        local_number = national_number | grouped_number

        country_code = (
            pynutil.delete("+")
            + pynutil.insert('country_code: "')
            + pynini.compose(pynini.closure(NEMO_DIGIT, 1, 3), cardinal.graph)
            + pynutil.insert('" ')
            + optional_deleted_sep
        )

        number_part = pynutil.insert('number_part: "') + local_number + pynutil.insert('"')
        international_number_part = pynutil.insert('number_part: "') + international_number + pynutil.insert('"')

        emergency = pynini.compose(pynini.union("112", "113", "114", "115"), three_digits)
        hotline = pynini.compose(pynini.union("1900"), four_digits)
        short_service = emergency | hotline
        context_cue = pynini.union(
            pynini.cross("gọi", "gọi"),
            pynini.cross("số khẩn cấp", "số khẩn cấp"),
            pynini.cross("đường dây", "đường dây"),
            pynini.cross("hotline", "hotline"),
        )
        contextual_short_number = (
            pynutil.insert('number_part: "')
            + context_cue
            + pynutil.delete(NEMO_SPACE)
            + insert_space
            + short_service
            + pynutil.insert('"')
        )

        octet = digit | digits(2) | digits(3)
        ip_address = octet + pynini.cross(".", " chấm ") + octet + pynini.cross(".", " chấm ") + octet
        ip_address += pynini.cross(".", " chấm ") + octet
        ip_number = pynutil.add_weight(pynutil.insert('number_part: "') + ip_address + pynutil.insert('"'), 5.0)

        card_sep = pynini.closure(pynutil.delete("-") | pynutil.delete(NEMO_SPACE), 1) + insert_space
        credit_card = four_digits + card_sep + four_digits + card_sep + four_digits + card_sep + four_digits
        card_number = pynutil.add_weight(
            pynutil.insert('number_part: "') + credit_card + pynutil.insert('"'), -1.0
        )

        preserve_order = pynutil.insert(" preserve_order: true")
        graph = (
            country_code + international_number_part
            | number_part
            | contextual_short_number
            | ip_number
            | card_number
        ) + preserve_order

        self.fst = self.add_tokens(graph.optimize()).optimize()
