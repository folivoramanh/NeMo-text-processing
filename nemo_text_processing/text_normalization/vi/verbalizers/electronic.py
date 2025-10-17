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
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> c d f một a còng a b c chấm e d u

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        # Load digit mappings from data file (no invert needed - we want written -> spoken)
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
        graph_digit = graph_digit | graph_zero

        # Load symbol mappings from data file
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        # Load domain mappings from data file
        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        # Load server name mappings from data file
        server_names = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))

        NEMO_NOT_BRACKET = pynini.difference(NEMO_CHAR, pynini.union("{", "}")).optimize()

        # Spell out each letter individually with spaces
        graph_letters = pynini.union(
            *[pynini.cross(c, f" {c.lower()} ") for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        ).optimize()

        # Add spaces around digits and symbols for consistency
        graph_digit_with_spaces = pynutil.insert(" ") + graph_digit + pynutil.insert(" ")
        graph_symbols_with_spaces = pynutil.insert(" ") + graph_symbols + pynutil.insert(" ")

        # Combine with higher priority for server names, then symbols, digits, and finally letters
        char_conversion = pynutil.add_weight(pynutil.insert(" ") + server_names + pynutil.insert(" "), -10) | graph_digit_with_spaces | graph_symbols_with_spaces | graph_letters

        # Convert each character/digit/symbol/server name to spoken form
        default_chars_symbols = pynini.cdrewrite(
            char_conversion,
            "",
            "",
            NEMO_SIGMA,
        )
        default_chars_symbols = pynini.compose(
            pynini.closure(NEMO_NOT_BRACKET), default_chars_symbols.optimize()
        ).optimize()

        # Username: extract and verbalize (ALWAYS spell out letter-by-letter)
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete('"')
            + default_chars_symbols
            + pynutil.delete('"')
        )

        # Domain processing - TWO versions:
        # 1. For EMAILS (with username): spell out unknown words
        # 2. For URLS/paths (no username): keep words intact
        
        # Version 1: Email domain - spell out unknown words
        domain_email_conversion = (
            pynutil.add_weight(domain_common, -100)  # .com → chấm com
            | pynutil.add_weight(pynutil.insert(" ") + server_names + pynutil.insert(" "), -10)  # gmail → g mail
            | graph_letters  # a → a (spell out)
            | graph_digit_with_spaces  # 1 → một
            | graph_symbols_with_spaces  # . → chấm
        )
        
        domain_email_processing = pynini.cdrewrite(
            domain_email_conversion,
            "",
            "",
            NEMO_SIGMA,
        )
        domain_email_processing = pynini.compose(
            pynini.closure(NEMO_NOT_BRACKET), domain_email_processing.optimize()
        ).optimize()
        
        # Version 2: URL/path domain - keep words intact
        # Need to add spaces around domain_common too
        domain_common_with_spaces = pynutil.insert(" ") + domain_common + pynutil.insert(" ")
        
        domain_url_conversion = (
            pynutil.add_weight(domain_common_with_spaces, -100)  # .com → chấm com  
            | pynutil.add_weight(pynutil.insert(" ") + server_names + pynutil.insert(" "), -10)  # gmail → g mail
            | graph_digit_with_spaces  # 1 → một
            | graph_symbols_with_spaces  # . → chấm, / → sẹc
            # NO graph_letters - keep words intact
        )
        
        domain_url_processing = pynini.cdrewrite(
            domain_url_conversion,
            "",
            "",
            NEMO_SIGMA,
        )
        domain_url_processing = pynini.compose(
            pynini.closure(NEMO_NOT_BRACKET), domain_url_processing.optimize()
        ).optimize()

        # Domain for emails (with username)
        domain_email = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + domain_email_processing
            + delete_space
            + pynutil.delete('"')
        ).optimize()
        
        # Domain for URLs (no username, or with protocol)
        domain_url = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + domain_url_processing
            + delete_space
            + pynutil.delete('"')
        ).optimize()

        # Protocol: keep as is (already verbalized in tagger)
        protocol = pynutil.delete("protocol: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Final graph:
        # 1. Email: username + @ + domain_email
        # 2. URL with protocol: protocol + domain_url
        # 3. Domain only: domain_url
        graph_email = user_name + delete_space + pynutil.insert(" a còng ") + delete_space + domain_email
        graph_url = protocol + delete_space + domain_url
        graph_domain_only = domain_url
        
        graph = (
            graph_email | graph_url | graph_domain_only
        ).optimize() @ pynini.cdrewrite(delete_extra_space, "", "", NEMO_SIGMA)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

