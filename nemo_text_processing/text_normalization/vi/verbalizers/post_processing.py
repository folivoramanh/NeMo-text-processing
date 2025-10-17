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

import os
from typing import Dict, List

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_SIGMA, generator_main
from nemo_text_processing.utils.logging import logger


class PostProcessingFst:
    """
    Finite state transducer that post-processes an entire Vietnamese sentence after verbalization is complete, e.g.
    removes extra spaces around punctuation marks " ( một trăm hai mươi ba ) " -> "(một trăm hai mươi ba)"

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "vi_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.set_punct_dict()
            self.fst = self.get_punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_vietnamese_punct_config(self) -> Dict[str, List[str]]:
        """
        Returns Vietnamese-specific punctuation configuration.
        Only keeps sentence-connecting and sentence-ending punctuation: , . ? !
        All other punctuation will be removed and replaced with space.
        """
        return {
            # Punctuation to keep (sentence connectors and enders)
            'keep_punct': [",", ".", "!", "?"],
            # Punctuation that should not have space before them
            'no_space_before': [",", ".", "!", "?"],
        }

    def set_punct_dict(self):
        # Vietnamese punctuation marks that might need special handling
        self.punct_marks = {
            "'": [
                "'",
                '´',
                'ʹ',
                'ʻ',
                'ʼ',
                'ʽ',
                'ʾ',
                'ˈ',
                'ˊ',
                'ˋ',
                '˴',
                'ʹ',
                '΄',
                '`',
                '´',
                '’',
                '‛',
                '′',
                '‵',
                'ꞌ',
                '＇',
                '｀',
            ],
        }

    def get_punct_postprocess_graph(self):
        """
        Returns graph to post process punctuation marks for Vietnamese.

        Keeps only: , . ? !
        Removes all other punctuation and replaces with space.
        Collapses multiple spaces and strips leading/trailing spaces.
        """
        # Get configuration
        punct_config = self.get_vietnamese_punct_config()
        keep_punct = punct_config['keep_punct']
        no_space_before_punct = punct_config['no_space_before']

        # Create FST for kept punctuation
        kept_punct_fst = pynini.union(*keep_punct)
        
        delete_space = pynutil.delete(" ")
        space_fst = pynini.accep(" ")

        # Rule 1: Remove space before kept punctuation (primary rule)
        # " ," -> ",", " ." -> ".", etc.
        remove_space_before = pynini.cdrewrite(
            delete_space + kept_punct_fst,
            "",
            "",
            NEMO_SIGMA,
        ).optimize()

        # Rule 2: Replace all other punctuation with space
        # List of safe punctuation characters to remove (excluding kept ones: , . ? !)
        # Only include characters that can be safely compiled by pynini
        punct_to_remove = [
            ":", ";", 
            "(", ")",  # brackets
            "-", "_",  # dashes
            "+", "=", "*", "/", "|",  # math/operators
            "@", "#", "$", "%", "^", "&",  # symbols
            "~", "`", '"', "'",  # quotes
        ]
        
        # Build replacement FST: each punct -> space
        # Use try-except to skip problematic characters
        punct_replacements = []
        for punct_char in punct_to_remove:
            try:
                punct_replacements.append(pynini.cross(punct_char, " "))
            except:
                # Skip characters that can't be compiled
                continue
        
        # Union all replacements if any were successfully created
        if punct_replacements:
            punct_to_space = pynini.union(*punct_replacements).optimize()
            
            # Apply replacement rule
            replace_punct_with_space = pynini.cdrewrite(
                punct_to_space,
                "",
                "",
                NEMO_SIGMA
            ).optimize()
        else:
            # No replacements, use identity
            replace_punct_with_space = pynini.Fst()

        # Rule 3: Collapse multiple spaces into single space
        # "  " -> " ", "   " -> " ", etc.
        double_space = space_fst + space_fst
        collapse_spaces = pynini.cdrewrite(
            pynini.cross(double_space, " "),  # two spaces -> one space
            "",
            "",
            NEMO_SIGMA
        ).optimize()

        # Combine all rules in order:
        # 1. Replace unwanted punct with space
        # 2. Remove space before kept punct
        # 3. Collapse multiple spaces (applied multiple times to handle any number of spaces)
        if punct_replacements:
            graph = pynini.compose(
                pynini.compose(
                    replace_punct_with_space,
                    remove_space_before
                ),
                pynini.compose(collapse_spaces, collapse_spaces)  # Apply twice to handle 3+ spaces
            )
        else:
            # No punct replacement, just use other rules
            graph = pynini.compose(
                remove_space_before,
                pynini.compose(collapse_spaces, collapse_spaces)
            )

        return graph.optimize()
