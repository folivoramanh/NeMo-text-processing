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
    NEMO_DIGIT,
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
from nemo_text_processing.text_normalization.vi.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.vi.taggers.time import TimeFst
from nemo_text_processing.text_normalization.vi.taggers.range import RangeFst
# from nemo_text_processing.text_normalization.vi.taggers.telephone import TelephoneFst
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
from nemo_text_processing.text_normalization.vi.verbalizers.roman import RomanFst as VRomanFst
from nemo_text_processing.text_normalization.vi.verbalizers.time import TimeFst as VTimeFst
from nemo_text_processing.text_normalization.vi.verbalizers.word import WordFst as VWordFst
from nemo_text_processing.text_normalization.vi.verbalizers.range import RangeFst as VRangeFst
from nemo_text_processing.text_normalization.vi.verbalizers.whitelist import WhiteListFst as VWhiteListFst


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

            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            punctuation = PunctuationFst(deterministic=True)
            punct_graph = punctuation.graph
            whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic)
            whitelist_graph = whitelist.fst
            word_graph = WordFst(deterministic=deterministic, punctuation=punctuation).graph
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            fraction_graph = fraction.fst
            date = DateFst(cardinal=cardinal, deterministic=deterministic)
            date_graph = date.fst
            roman = RomanFst(cardinal=cardinal, deterministic=deterministic)
            roman_graph = roman.fst
            time_fst = TimeFst(cardinal=cardinal, deterministic=deterministic)
            time_graph = time_fst.fst
            money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            money_graph = money.fst
            measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            measure_graph = measure.fst
            v_deterministic_cardinal = VCardinalFst(deterministic=True)
            v_deterministic_date = VDateFst(deterministic=True)
            date_final = pynini.compose(date_graph, v_deterministic_date.fst)

            v_decimal = VDecimalFst(v_deterministic_cardinal, deterministic=deterministic)
            decimal_final = pynini.compose(decimal_graph, v_decimal.fst)

            v_deterministic_time = VTimeFst(deterministic=True)
            time_final = pynini.compose(time_graph, v_deterministic_time.fst)

            v_deterministic_money = VMoneyFst(deterministic=True)
            money_final = pynini.compose(money_graph, v_deterministic_money.fst)

            v_deterministic_fraction = VFractionFst(deterministic=True)
            v_deterministic_measure = VMeasureFst(
                decimal=v_decimal, cardinal=v_deterministic_cardinal, fraction=v_deterministic_fraction, deterministic=deterministic
            )
            measure_final = pynini.compose(measure_graph, v_deterministic_measure.fst)

            # Create range graph
            range_fst = RangeFst(
                time=time_final,
                date=date_final,
                decimal=decimal_final,
                money=money_final,
                measure=measure_final,
                deterministic=deterministic,
            )
            range_graph = range_fst.fst

            v_cardinal = VCardinalFst(deterministic=deterministic)
            v_cardinal_graph = v_cardinal.fst
            v_date = VDateFst(deterministic=deterministic)
            v_date_graph = v_date.fst
            v_decimal = VDecimalFst(cardinal=v_cardinal, deterministic=deterministic)
            v_decimal_graph = v_decimal.fst
            v_ordinal = VOrdinalFst(deterministic=deterministic)
            v_ordinal_graph = v_ordinal.fst
            v_fraction = VFractionFst(deterministic=deterministic)
            v_fraction_graph = v_fraction.fst
            v_money = VMoneyFst(deterministic=deterministic)
            v_money_graph = v_money.fst
            v_measure = VMeasureFst(cardinal=v_cardinal, decimal=v_decimal, fraction=v_fraction, deterministic=deterministic)
            v_measure_graph = v_measure.fst
            v_word = VWordFst(deterministic=deterministic)
            v_word_graph = v_word.fst
            v_roman = VRomanFst(deterministic=deterministic)
            v_roman_graph = v_roman.fst
            v_range = VRangeFst(deterministic=deterministic)
            v_range_graph = v_range.fst
            v_time = VTimeFst(deterministic=deterministic)
            v_time_graph = v_time.fst
            v_whitelist = VWhiteListFst(deterministic=deterministic)
            v_whitelist_graph = v_whitelist.fst
            sem_w = 1
            word_w = 100
            punct_w = 2
            classify_and_verbalize = (
                pynutil.add_weight(pynini.compose(whitelist_graph, v_whitelist_graph), sem_w)
                | pynutil.add_weight(pynini.compose(time_graph, v_time_graph), sem_w) 
                | pynutil.add_weight(pynini.compose(decimal_graph, v_decimal_graph), sem_w)  
                | pynutil.add_weight(pynini.compose(measure_graph, v_measure_graph), sem_w)
                | pynutil.add_weight(pynini.compose(cardinal_graph, v_cardinal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(ordinal_graph, v_ordinal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(fraction_graph, v_fraction_graph), sem_w)
                | pynutil.add_weight(pynini.compose(money_graph, v_money_graph), sem_w)
                | pynutil.add_weight(word_graph, word_w)
                | pynutil.add_weight(pynini.compose(date_graph, v_date_graph), sem_w)
                | pynutil.add_weight(pynini.compose(roman_graph, v_roman_graph), word_w)
                | pynutil.add_weight(pynini.compose(range_graph, v_range_graph), sem_w)
            ).optimize()

            punct_only = pynutil.add_weight(punct_graph, weight=punct_w)
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct_only),
                1,
            )

            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + classify_and_verbalize
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph |= punct_only + pynini.closure(punct)
            graph = delete_space + graph + delete_space

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

        # to remove normalization options that still contain digits and some special symbols
        # e.g., "P&E" -> {P and E, P&E}, "P & E" will be removed from the list of normalization options
        no_digits = pynini.closure(pynini.difference(NEMO_CHAR, pynini.union(NEMO_DIGIT, "&")))
        self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
