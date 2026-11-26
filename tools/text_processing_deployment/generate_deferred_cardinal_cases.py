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

"""Generate deterministic English cardinal boundary inputs for parity comparisons.

The JSONL corpus is accepted by benchmark_deferred_cardinal.py --cases-jsonl.
It covers numeric spans of at most 12 digits and supplies no expected outputs;
the benchmark compares each candidate with the ordinary baseline. Generation
uses only the Python standard library and does not load Pynini or grammars.
"""

import hashlib
import json
import random
import re
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

SEED = 20261126
MAX_DIGITS = 12


def longest_number(text):
    """Count digits across commas and decimal points within each numeric span."""
    return max(
        (sum(char.isdigit() for char in span) for span in re.findall(r'[0-9][0-9,.]*', text)),
        default=0,
    )


def cardinal_boundaries():
    """Return sorted boundaries, spelling patterns, and seeded integer samples."""
    numbers = set(range(32))
    for tens in range(10, 100, 10):
        numbers.update(tens + offset for offset in (-1, 0, 1))
    for hundred in range(100, 1000, 100):
        numbers.update(hundred + offset for offset in (-1, 0, 1, 9, 10, 11, 19, 20, 21, 99))
    for exponent in range(2, MAX_DIGITS):
        numbers.update(
            10**exponent + offset
            for offset in (-101, -100, -99, -11, -10, -9, -2, -1, 0, 1, 2, 9, 10, 11, 99, 100, 101)
        )

    rng = random.Random(SEED)
    for width in range(1, MAX_DIGITS + 1):
        numbers.update(int(digit * width) for digit in '1259')
        numbers.update(int((pattern * width)[:width]) for pattern in ('10', '12', '90'))
        numbers.add(10 ** (width - 1) + 1)
        numbers.add(10**width - 1)
        numbers.update(rng.randrange(10 ** (width - 1), 10**width) for _ in range(6))
    return sorted(number for number in numbers if 0 <= number < 10**MAX_DIGITS)


def representatives(numbers, width, count):
    """Choose evenly spaced sorted examples for a digit width."""
    values = [number for number in numbers if len(str(number)) == width]
    if len(values) <= count:
        return values
    return [values[index * (len(values) - 1) // (count - 1)] for index in range(count)]


def generate_cases():
    """Return deduplicated cases in stable order, retaining their source labels."""
    numbers = cardinal_boundaries()
    cases = {}

    def add(text, category, **settings):
        if longest_number(text) > MAX_DIGITS:
            raise ValueError(f'Numeric span exceeds {MAX_DIGITS} digits in {category}: {text!r}')
        key = (text, tuple(sorted(settings.items())))
        cases.setdefault(key, dict(text=text, source='boundaries/' + category, **settings))

    for number in numbers:
        add(str(number), 'plain')
        add('-' + str(number), 'negative')
        if number >= 1000:
            add(f'{number:,}', 'comma_groups')
            add(f'-{number:,}', 'comma_groups_negative')
        if len(str(number)) < MAX_DIGITS:
            add('0' + str(number), 'leading_zero')

    for width in range(1, MAX_DIGITS + 1):
        for number in representatives(numbers, width, 3):
            add('+' + str(number), 'explicit_plus')
            add('-' + str(number).zfill(MAX_DIGITS), 'leading_zeros_negative')
            add(f'We counted {number} packages.', 'sentence')
            add(f'{number} kg', 'measure_kg')
            add(f'{number} m', 'measure_m')
            add(f'A-{number}', 'serial_alpha')
            add(f'{number}/31/100', 'serial_slash')
            add(f'{number}/261788/2021', 'serial_mixed_width')
        for number in representatives(numbers, width, 2):
            add(str(number).zfill(MAX_DIGITS), 'leading_zeros_fixed_width')
            digits = str(number).zfill(MAX_DIGITS)
            grouped_zeros = ','.join(digits[offset : offset + 3] for offset in range(0, MAX_DIGITS, 3))
            add(grouped_zeros, 'leading_zeros_grouped')
            add(f'There are {number} boxes and 12 items.', 'multiple_cardinals')
            add(f'{number} kg costs $12.50.', 'mixed_measure_money')

    # Leave two digits for fractional parts inside the same numeric span.
    for width in range(1, MAX_DIGITS - 1):
        for number in representatives(numbers, width, 3):
            add(f'${number}.50', 'money_usd', post_process=False)
            add(f'${number:,}.05', 'money_usd_grouped')
            add(f'€{number}.50', 'money_eur', post_process=False)
            add(f'{number}.05', 'decimal')
            add(f'-{number}.01', 'decimal_negative')
            add(f'{number}.50 kg', 'measure_decimal')
        for number in representatives(numbers, width, 2):
            add(f'{number}.5%', 'percent')
            add(f'${number}-$25', 'money_range')

    for year in (999, 1000, 1001, 1899, 1900, 1999, 2000, 2001, 2010, 2020, 2021, 2026, 2099, 2100, 2999, 9999):
        add(f'01/01/{year}', 'date_slash')
        add(f'November 20, {year}', 'date_written_month')
        add(f'{year}-12-31', 'date_iso_or_serial')

    for number in (0, 1, 9, 10, 11, 12, 13, 19, 20, 21, 23, 29, 30, 31, 59, 60, 99, 100):
        add(f'{number}:30', 'time_boundary')
        suffix = 'th' if number % 100 in (11, 12, 13) else {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        add(f'{number}{suffix}', 'ordinal_suffix_boundary')

    for text in (
        '31/31/100',
        '123/261788/2021',
        '000000000000',
        '-000000000000',
        '000,000,000,000',
        '999,999,999,999',
        '-999,999,999,999',
        '9999999999.99',
        '$9,999,999,999.99',
        '0.00000000000',
        '1.00000000000',
        '1,000,000,000.00',
    ):
        add(text, 'targeted_edges')
    return list(cases.values())


def write_cases(output, cases):
    """Write the JSONL corpus and a sibling summary, returning the summary."""
    payload = ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding='utf-8')
    summary = {
        'path': str(output),
        'cases': len(cases),
        'distinct_core_integers': sum(case['source'] == 'boundaries/plain' for case in cases),
        'random_seed': SEED,
        'max_digits_per_numeric_span': max(longest_number(case['text']) for case in cases),
        'numeric_span_width_counts': dict(sorted(Counter(longest_number(case['text']) for case in cases).items())),
        'category_counts': dict(sorted(Counter(case['source'].split('/', 1)[1] for case in cases).items())),
        'sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
        'baseline_expected_outputs_supplied': False,
        'benchmark_run': False,
    }
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, required=True, help='Destination JSONL file; also writes a .summary.json'
    )
    args = parser.parse_args()
    print(json.dumps(write_cases(args.output, generate_cases()), indent=2))


if __name__ == '__main__':
    main()
