# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal, e.g.
        cardinal { negative: "true" integer: "23" } -> minus twenty three

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        number_fst: optional raw-number-to-words graph for the experimental top-level cardinal branch.
    """

    def __init__(self, deterministic: bool = True, number_fst: 'pynini.Fst' = None):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        self.optional_sign = pynini.cross("negative: \"true\"", "minus ")
        if not deterministic:
            self.optional_sign |= pynini.cross("negative: \"true\"", "negative ")
            self.optional_sign |= pynini.cross("negative: \"true\"", "dash ")

        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)

        integer = pynini.closure(NEMO_NOT_QUOTE) if number_fst is None else number_fst

        self.integer = delete_space + pynutil.delete("\"") + integer + pynutil.delete("\"")
        integer = pynutil.delete("integer:") + self.integer

        self.numbers = self.optional_sign + integer
        delete_tokens = self.delete_tokens(self.numbers)
        self.fst = delete_tokens.optimize()
