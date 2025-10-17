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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_UPPER, GraphFst, insert_space


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for classifying abbreviations, e.g.:
        "ABC" -> tokens { abbreviation { value: "A B C" } }
        "A.B.C." -> tokens { abbreviation { value: "A B C" } }
    
    Args:
        whitelist: whitelist FST to exclude known words
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, whitelist: GraphFst, deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")
        # A.B.C. -> A. B. C. (with spaces)
        graph = NEMO_UPPER + dot + pynini.closure(insert_space + NEMO_UPPER + dot, 1)
        # A.B.C. -> A.B.C. (without spaces)
        graph |= NEMO_UPPER + dot + pynini.closure(NEMO_UPPER + dot, 1)
        # ABC -> A B C (uppercase letters)
        graph |= NEMO_UPPER + pynini.closure(insert_space + NEMO_UPPER, 1)

        # Remove dots from output for cleaner verbalization
        graph = pynini.compose(graph, pynini.cdrewrite(pynutil.delete("."), "", "", pynini.closure(pynini.union(NEMO_UPPER, " ", "."))))

        graph = pynutil.insert("value: \"") + graph.optimize() + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()

