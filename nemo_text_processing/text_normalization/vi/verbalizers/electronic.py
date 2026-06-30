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


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese electronic tokens, e.g.
        electronic { username: "a b c" domain: "a b c chấm com" preserve_order: true }
        -> a b c a móc a b c chấm com
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        username = (
            pynutil.delete('username: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        domain = pynutil.delete('domain: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        protocol = pynutil.delete('protocol: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        email = username + delete_space + pynutil.insert(" a móc ") + domain
        url = protocol + delete_space + insert_space + domain
        graph = (email | url | domain) + delete_preserve_order

        self.fst = self.delete_tokens(graph).optimize()
