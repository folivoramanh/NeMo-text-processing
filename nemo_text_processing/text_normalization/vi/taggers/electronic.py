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
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. cdf1@abc.edu -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # Load accepted symbols from data file
        accepted_symbols = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input"
        )

        # Load common domains from data file
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )

        # Load server names from data file
        server_names = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/server_name.tsv")), "input"
        )

        # Define numbers (digits)
        numbers = NEMO_DIGIT

        # Define accepted characters for username and domain
        accepted_characters = NEMO_ALPHA | numbers | accepted_symbols

        # Username: starts with alpha, can contain alphanumeric and symbols
        username = (NEMO_ALPHA | server_names) + pynini.closure(accepted_characters)
        username = pynutil.insert('username: "') + username + pynutil.insert('"') + pynini.cross("@", " ")

        # Domain: alphanumeric and symbols
        domain_graph = accepted_characters + pynini.closure(accepted_characters | accepted_common_domains)

        domain_graph_with_tags = (
            pynutil.insert('domain: "')
            + pynini.compose(
                NEMO_ALPHA + pynini.closure(NEMO_NOT_SPACE) + (NEMO_ALPHA | NEMO_DIGIT | pynini.accep("/")),
                domain_graph,
            ).optimize()
            + pynutil.insert('"')
        )

        # Email: username@domain
        graph = pynini.compose(
            NEMO_SIGMA + pynini.accep("@") + NEMO_SIGMA + pynini.accep(".") + NEMO_SIGMA,
            username + domain_graph_with_tags,
        )

        # Domain only: test.com, nvidia.com, etc.
        # Must have at least one dot and end with alpha
        full_stop_accep = pynini.accep(".")
        domain_component = full_stop_accep + pynini.closure(accepted_characters, 2)
        graph_domain = (
            pynutil.insert('domain: "')
            + (pynini.closure(accepted_characters, 1) + pynini.closure(domain_component, 1))
            + pynutil.insert('"')
        ).optimize()

        graph |= graph_domain

        # Protocol-based URLs: http://, https://, www.
        # IMPORTANT: https must come before http to match correctly!
        protocol_start = (
            pynini.cross("https://", "https hai chấm sẹc sẹc ")
            | pynini.cross("http://", "http hai chấm sẹc sẹc ")
        )

        protocol_end = pynini.cross("www.", "www chấm ")

        protocol = (
            protocol_start + pynini.closure(protocol_end, 0, 1)
            | protocol_end
        )

        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')

        # URL: protocol + domain
        graph |= protocol + pynutil.insert(" ") + domain_graph_with_tags

        # Slash-separated paths: update/upgrade
        slash_string = (
            pynini.accep(" ").ques + pynini.accep("/") + pynini.accep(" ").ques + pynini.closure(NEMO_ALPHA, 1)
        )

        graph |= (
            pynutil.insert('domain: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynini.closure(slash_string, 1)
            + pynutil.insert('"')
        ).optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()

