# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Opt-in English regression adapter for raw and raw-plus-lookahead grammars."""

import copy
import tempfile
from pathlib import Path

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser
from tools.text_processing_deployment.benchmark_deferred_cardinal import deferred_normalizer


def pytest_addoption(parser):
    parser.addoption('--deferred-runtime', choices=['raw', 'raw-lookahead'])
    parser.addoption('--deferred-work-dir', type=Path)


def pytest_collection_modifyitems(config, items):
    variant = config.getoption('--deferred-runtime')
    if variant is None:
        return
    work = config.getoption('--deferred-work-dir')
    if work is None:
        raise ValueError('--deferred-work-dir is required with --deferred-runtime')
    work.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f'pytest-{variant}-', dir=work))
    pairs = {}
    seen = set()
    for item in items:
        if item.cls is None or item.path.parent.name != 'en':
            continue
        for normalizer in vars(item.cls).values():
            if type(normalizer) is not Normalizer or normalizer.lang != 'en' or id(normalizer) in seen:
                continue
            if not normalizer.tagger.deterministic:
                raise ValueError('This adapter supports the stock deterministic English suite only.')
            seen.add(id(normalizer))
            key = normalizer.input_case
            # Reuse built pairs while retaining each test instance's processing
            # settings and independent parser. Install the complete runtime,
            # including its native fallback, rather than swapping two FSTs.
            if key not in pairs:
                directory = artifacts / key if variant == 'raw-lookahead' else None
                pairs[key] = deferred_normalizer(key, work / 'grammars', directory)
            source = pairs[key]
            normalizer.tagger, normalizer.verbalizer = source.tagger, source.verbalizer
            normalizer.defer_cardinal = True
            normalizer._runtime_bundle = source._runtime_bundle
            normalizer._max_numeric_digits = source._max_numeric_digits
            normalizer._baseline_normalizer = copy.copy(source._baseline_normalizer)
            normalizer._baseline_normalizer.parser = TokenParser()
    config._deferred_normalizer_count = len(seen)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if hasattr(config, '_deferred_normalizer_count'):
        terminalreporter.write_line(
            f"{config.getoption('--deferred-runtime')} enabled for {config._deferred_normalizer_count} English TN instances."
        )
