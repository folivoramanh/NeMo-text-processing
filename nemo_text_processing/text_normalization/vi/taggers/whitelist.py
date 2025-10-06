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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist for Vietnamese, e.g.
        Dr. -> tokens { name: "bác sĩ" }
        $ -> tokens { name: "đô la" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from data files.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str = "cased", deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(file):
            whitelist = load_labels(file)
            graph = pynini.string_map(whitelist)
            return graph

        # Load symbol mappings
        graph = _get_whitelist_graph(get_abs_path("data/whitelist/symbol.tsv"))
        
        # Load TTS mappings  
        graph |= _get_whitelist_graph(get_abs_path("data/whitelist/tts.tsv"))

        # Compose with non-slash filter like English does
        graph = pynini.compose(
            pynini.difference(NEMO_SIGMA, pynini.accep("/")).optimize(),
            graph,
        ).optimize()

        self.fst = (pynutil.insert("name: \"") + graph + pynutil.insert("\"")).optimize()