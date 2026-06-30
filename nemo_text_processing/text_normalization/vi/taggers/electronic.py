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

import string

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese electronic expressions, e.g.
        abc@abc.com -> electronic { username: "a b c" domain: "a b c chấm com" preserve_order: true }
        nvidia.com -> electronic { domain: "nvidia chấm com" preserve_order: true }
        https://www.nvidia.com -> electronic { protocol: "h t t p s hai chấm sẹc sẹc" domain: "w w w chấm nvidia chấm com" preserve_order: true }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        digit = (pynini.cross("0", "không") | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
        symbol = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()
        known_word = pynutil.add_weight(pynini.string_file(get_abs_path("data/electronic/words.tsv")), -1.0)
        common_domain = pynutil.add_weight(pynini.string_file(get_abs_path("data/electronic/domain.tsv")), -1.0)

        alpha_to_lower = pynini.string_map(
            [(char, char) for char in string.ascii_lowercase]
            + [(char, char.lower()) for char in string.ascii_uppercase]
        )

        username_symbol = pynini.string_map(
            [
                (".", "chấm"),
                ("-", "gạch"),
                ("_", "gạch dưới"),
                ("&", "và"),
                ("+", "cộng"),
            ]
        )
        username_item = (alpha_to_lower | digit | username_symbol).optimize()
        username = username_item + pynini.closure(insert_space + username_item)

        label_char = (alpha_to_lower | digit | pynini.cross("-", "gạch")).optimize()
        label_item = (known_word | label_char).optimize()
        label_start = (known_word | alpha_to_lower).optimize()
        label = label_start + pynini.closure(insert_space + label_item)

        dot_label = common_domain | (pynini.cross(".", "chấm") + insert_space + label)
        domain = label + pynini.closure(insert_space + dot_label, 1)

        path_item = (known_word | alpha_to_lower | digit | symbol).optimize()
        path = insert_space + pynini.cross("/", "sẹc") + pynini.closure(insert_space + path_item, 1)
        domain_with_path = domain + pynini.closure(path, 0, 1)

        protocol = (
            pynini.cross("http://", "h t t p hai chấm sẹc sẹc")
            | pynini.cross("https://", "h t t p s hai chấm sẹc sẹc")
            | pynini.cross("file:///", "file hai chấm sẹc sẹc sẹc")
        )

        username_part = pynutil.insert('username: "') + username + pynutil.insert('"')
        domain_part = pynutil.insert('domain: "') + domain_with_path + pynutil.insert('"')
        protocol_part = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        preserve_order = pynutil.insert(" preserve_order: true")

        email = username_part + pynutil.delete("@") + insert_space + domain_part
        url = protocol_part + insert_space + domain_part
        bare_domain = domain_part

        self.fst = self.add_tokens((email | url | bare_domain) + preserve_order).optimize()
