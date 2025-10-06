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
import time

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    NEMO_NOT_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.vi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.vi.taggers.date import DateFst
from nemo_text_processing.text_normalization.vi.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.vi.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.vi.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.vi.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.vi.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.vi.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.vi.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.vi.taggers.word import WordFst
from nemo_text_processing.text_normalization.vi.verbalizers.cardinal import CardinalFst as VCardinalFst
from nemo_text_processing.text_normalization.vi.verbalizers.date import DateFst as VDateFst
from nemo_text_processing.text_normalization.vi.verbalizers.decimal import DecimalFst as VDecimalFst
from nemo_text_processing.text_normalization.vi.verbalizers.fraction import FractionFst as VFractionFst
from nemo_text_processing.text_normalization.vi.verbalizers.measure import MeasureFst as VMeasureFst
from nemo_text_processing.text_normalization.vi.verbalizers.money import MoneyFst as VMoneyFst
from nemo_text_processing.text_normalization.vi.verbalizers.ordinal import OrdinalFst as VOrdinalFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars for Vietnamese audio-based text normalization.
    This class can process an entire sentence including punctuation.
    
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(
                cache_dir,
                f"vi_tn_audio_{deterministic}_deterministic_{input_case}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating Vietnamese Audio-based ClassifyFst grammars.")

            start_time = time.time()
            
            # TAGGERS - Non-deterministic for audio-based TN (like English approach)
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            logger.debug(f"cardinal: {time.time() - start_time: .2f}s -- {cardinal_graph.num_states()} nodes")

            start_time = time.time()
            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst
            logger.debug(f"punct: {time.time() - start_time: .2f}s -- {punct_graph.num_states()} nodes")

            start_time = time.time()
            whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic)
            whitelist_graph = whitelist.fst
            logger.debug(f"whitelist: {time.time() - start_time: .2f}s -- {whitelist_graph.num_states()} nodes")

            start_time = time.time()
            word_graph = WordFst(deterministic=deterministic).fst
            logger.debug(f"word: {time.time() - start_time: .2f}s -- {word_graph.num_states()} nodes")

            start_time = time.time()
            date = DateFst(cardinal=cardinal, deterministic=deterministic)
            date_graph = date.fst
            logger.debug(f"date: {time.time() - start_time: .2f}s -- {date_graph.num_states()} nodes")

            start_time = time.time()
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            logger.debug(f"decimal: {time.time() - start_time: .2f}s -- {decimal_graph.num_states()} nodes")

            start_time = time.time()
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst
            logger.debug(f"ordinal: {time.time() - start_time: .2f}s -- {ordinal_graph.num_states()} nodes")

            start_time = time.time()
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            fraction_graph = fraction.fst
            logger.debug(f"fraction: {time.time() - start_time: .2f}s -- {fraction_graph.num_states()} nodes")

            start_time = time.time()
            money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            money_graph = money.fst
            logger.debug(f"money: {time.time() - start_time: .2f}s -- {money_graph.num_states()} nodes")

            start_time = time.time()
            measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            measure_graph = measure.fst
            logger.debug(f"measure: {time.time() - start_time: .2f}s -- {measure_graph.num_states()} nodes")

            # VERBALIZERS - Compose with taggers like English
            start_time = time.time()
            v_cardinal = VCardinalFst(deterministic=deterministic)
            v_cardinal_graph = v_cardinal.fst
            logger.debug(f"v_cardinal: {time.time() - start_time: .2f}s -- {v_cardinal_graph.num_states()} nodes")

            start_time = time.time()
            v_date = VDateFst(deterministic=deterministic)
            v_date_graph = v_date.fst
            logger.debug(f"v_date: {time.time() - start_time: .2f}s -- {v_date_graph.num_states()} nodes")

            start_time = time.time()
            v_decimal = VDecimalFst(cardinal=v_cardinal, deterministic=deterministic)
            v_decimal_graph = v_decimal.fst
            logger.debug(f"v_decimal: {time.time() - start_time: .2f}s -- {v_decimal_graph.num_states()} nodes")

            start_time = time.time()
            v_ordinal = VOrdinalFst(deterministic=deterministic)
            v_ordinal_graph = v_ordinal.fst
            logger.debug(f"v_ordinal: {time.time() - start_time: .2f}s -- {v_ordinal_graph.num_states()} nodes")

            start_time = time.time()
            v_fraction = VFractionFst(deterministic=deterministic)
            v_fraction_graph = v_fraction.fst
            logger.debug(f"v_fraction: {time.time() - start_time: .2f}s -- {v_fraction_graph.num_states()} nodes")

            start_time = time.time()
            v_money = VMoneyFst(deterministic=deterministic)
            v_money_graph = v_money.fst
            logger.debug(f"v_money: {time.time() - start_time: .2f}s -- {v_money_graph.num_states()} nodes")

            start_time = time.time()
            v_measure = VMeasureFst(cardinal=v_cardinal, decimal=v_decimal, fraction=v_fraction, deterministic=deterministic)
            v_measure_graph = v_measure.fst
            logger.debug(f"v_measure: {time.time() - start_time: .2f}s -- {v_measure_graph.num_states()} nodes")

            # COMPOSE TAGGERS + VERBALIZERS (like English approach)
            start_time = time.time()
            classify_and_verbalize = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(pynini.compose(cardinal_graph, v_cardinal_graph), 1.1)
                | pynutil.add_weight(pynini.compose(date_graph, v_date_graph), 1.09)
                | pynutil.add_weight(pynini.compose(decimal_graph, v_decimal_graph), 1.08)
                | pynutil.add_weight(pynini.compose(ordinal_graph, v_ordinal_graph), 1.07)
                | pynutil.add_weight(pynini.compose(fraction_graph, v_fraction_graph), 1.06)
                | pynutil.add_weight(pynini.compose(money_graph, v_money_graph), 1.05)
                | pynutil.add_weight(pynini.compose(measure_graph, v_measure_graph), 1.04)
                | pynutil.add_weight(word_graph, 100)
            ).optimize()
            logger.debug(f"classify_and_verbalize: {time.time() - start_time: .2f}s -- {classify_and_verbalize.num_states()} nodes")

            # PUNCTUATION handling
            punct_only = pynutil.add_weight(punct_graph, 2.1)
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct_only),
                1,
            )

            # TOKEN + PUNCTUATION composition (like English)
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + classify_and_verbalize
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            # FINAL GRAPH construction
            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph |= punct_only + pynini.closure(punct)
            graph = delete_space + graph + delete_space

            # Simple remove extra spaces using existing constants
            remove_extra_spaces = pynini.closure(NEMO_NOT_SPACE, 1) + pynini.closure(
                delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1)
            )
            remove_extra_spaces |= (
                pynini.closure(pynutil.delete(" "), 1)
                + pynini.closure(NEMO_NOT_SPACE, 1)
                + pynini.closure(delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1))
            )

            graph = pynini.compose(graph.optimize(), remove_extra_spaces).optimize()
            self.fst = graph

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
