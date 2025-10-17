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
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial (alphanumeric sequences).
    The serial is a combination of digits, letters and dashes, e.g.:
        c325b -> tokens { serial { name: "c ba hai năm b" } }
        covid-19 -> tokens { serial { name: "covid mười chín" } }
    
    Args:
        cardinal: cardinal FST for number verbalization
        ordinal: ordinal FST to exclude ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        # For numbers: use cardinal graph for verbalization
        # For serial numbers, we want to verbalize as full numbers when they're standalone,
        # but as individual digits when mixed with letters
        # Build digit-by-digit verbalization that preserves all digits including leading zeros
        single_digit_graph = pynini.compose(NEMO_DIGIT, cardinal.single_digits_graph).optimize()
        
        if deterministic:
            # Numbers 1-2 digits NOT starting with zero: use full cardinal (e.g., "19" -> "mười chín")
            num_graph = pynini.compose(
                pynini.union("1", "2", "3", "4", "5", "6", "7", "8", "9") + pynini.closure(NEMO_DIGIT, 0, 1),
                cardinal.graph
            ).optimize()
            # Numbers 3+ digits: use single digit verbalization with spaces
            num_graph |= pynini.compose(NEMO_DIGIT ** (3, 20), pynini.closure(single_digit_graph + pynutil.insert(" "))).optimize()
            # Numbers starting with zero: always use single digit verbalization with spaces
            zero_start = pynini.accep("0") + pynini.closure(NEMO_DIGIT)
            num_graph |= pynini.compose(zero_start, pynini.closure(single_digit_graph + pynutil.insert(" "))).optimize()
        else:
            num_graph = cardinal.graph | cardinal.single_digits_graph

        # Load symbols and create symbol graph
        symbols_labels = load_labels(get_abs_path("data/whitelist/symbol.tsv"))
        symbols_graph = pynini.string_map(symbols_labels).optimize()
        num_graph |= symbols_graph

        if not deterministic:
            num_graph |= cardinal.single_digits_graph
            # Remove "trăm" from verbalization in serial context
            num_graph |= pynini.compose(num_graph, NEMO_SIGMA + pynutil.delete("trăm ") + NEMO_SIGMA)

        # Get all symbol characters for delimiter logic
        symbols = pynini.union(*[x[0] for x in symbols_labels])
        digit_symbol = NEMO_DIGIT | symbols

        # Add space between letter and digit/symbol, and between digit/symbol and letter
        graph_with_space = pynini.compose(
            pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA | symbols, digit_symbol, NEMO_SIGMA),
            pynini.cdrewrite(pynutil.insert(" "), digit_symbol, NEMO_ALPHA | symbols, NEMO_SIGMA),
        )

        # Define delimiters - always convert to space for clean output
        delimiter = pynini.cross("-", " ") | pynini.cross("/", " ") | pynini.accep(" ")
        if not deterministic:
            delimiter |= pynini.cross("-", " gạch ") | pynini.cross("/", " sẹc ")

        # Build serial patterns
        alphas = pynini.closure(NEMO_ALPHA, 1)
        
        # For direct alphanumeric without delimiters, use digit-by-digit with spaces
        # Create a version that verbalizes each digit individually
        single_digit_with_space = pynini.compose(NEMO_DIGIT, cardinal.single_digits_graph) + pynutil.insert(" ")
        digit_num_graph = pynini.closure(single_digit_with_space, 1).optimize()
        
        # Patterns with delimiters: use num_graph (can be full cardinal for short numbers)
        letter_num = alphas + delimiter + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alphas
        next_alpha_or_num = pynini.closure(delimiter + (alphas | num_graph))
        next_alpha_or_num |= pynini.closure(delimiter + num_graph + pynutil.insert(" ") + alphas)

        serial_graph = letter_num + next_alpha_or_num
        serial_graph |= num_letter + next_alpha_or_num
        # Numbers only with 2+ delimiters - use regular num_graph for pure number sequences
        serial_graph |= (
            num_graph + delimiter + num_graph + delimiter + num_graph + pynini.closure(delimiter + num_graph)
        )
        # 2+ symbols
        serial_graph |= pynini.compose(NEMO_SIGMA + symbols + NEMO_SIGMA, num_graph + delimiter + num_graph)
        
        # Direct alphanumeric adjacency (no delimiters): use digit-by-digit
        # e.g., "c325b", "ABC123", "iphone12"
        # Add spaces between components
        alpha_component = alphas + pynutil.insert(" ")
        serial_graph |= alpha_component + digit_num_graph + pynini.closure(alpha_component | digit_num_graph)
        serial_graph |= digit_num_graph + alpha_component + pynini.closure(alpha_component | digit_num_graph)

        # Note: Ordinal numbers will naturally have higher priority than serial
        # in tokenize_and_classify.py (both have weight 1.1, but ordinal is listed first)
        # No need to explicitly exclude ordinals here
        # Handle superscripts (^2, ^3)
        serial_graph |= (
            pynini.closure(NEMO_NOT_SPACE, 1)
            + (pynini.cross("^2", " bình phương") | pynini.cross("^3", " lập phương")).optimize()
        )

        # At least one serial graph with alphanumeric value and optional additional serial/num/alpha values
        serial_graph = (
            pynini.closure((serial_graph | num_graph | alphas) + delimiter)
            + serial_graph
            + pynini.closure(delimiter + (serial_graph | num_graph | alphas))
        )

        serial_graph |= pynini.compose(graph_with_space, serial_graph.optimize()).optimize()
        # At least 2 characters to be considered serial
        serial_graph = pynini.compose(pynini.closure(NEMO_NOT_SPACE, 2), serial_graph).optimize()

        # Note: Pure alpha/alpha patterns like "import/export" will be handled by word class
        # which has higher priority (weight 100 vs serial's 1.1)
        
        self.graph = serial_graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()

