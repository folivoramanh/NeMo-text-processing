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

import pytest

from tools.text_processing_deployment.benchmark_deferred_cardinal import longest_number

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on('CPU')]


@pytest.mark.parametrize(
    'text,expected',
    [
        ('999999999999', 12),
        ('1000000000000', 13),
        ('$123,456,789.012', 12),
        ('-000,123.45', 8),
        ('1-413-te-b-1-5', 3),
        ('12345678 items and 12345678 more', 8),
    ],
)
def test_evaluation_digit_filter_counts_per_numeric_span(text, expected):
    assert longest_number(text) == expected
