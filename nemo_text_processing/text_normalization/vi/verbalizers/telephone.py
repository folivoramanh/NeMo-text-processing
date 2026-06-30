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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese telephone tokens, e.g.
        telephone { number_part: "không chín ba sáu năm năm năm bốn bốn chín" preserve_order: true }
        -> không chín ba sáu năm năm năm bốn bốn chín

        telephone { country_code: "tám mươi tư" number_part: "không chín ba sáu năm năm năm bốn bốn chín" preserve_order: true }
        -> cộng tám mươi tư không chín ba sáu năm năm năm bốn bốn chín

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        country_code = (
            pynutil.delete('country_code: "')
            + pynutil.insert("cộng ")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        number_part = pynutil.delete('number_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        graph = (country_code + delete_space + insert_space + number_part) | number_part
        graph += delete_preserve_order

        self.fst = self.delete_tokens(graph).optimize()
