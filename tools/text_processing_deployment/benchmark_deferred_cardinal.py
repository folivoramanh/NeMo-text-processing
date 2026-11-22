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

"""Compare English deferred cardinal rendering with and without lookahead."""

import hashlib
import json
import re
import statistics
import tempfile
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pynini

from nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.en.verbalizers.verbalize_final import VerbalizeFinalFst
from nemo_text_processing.text_normalization.lookahead import LookaheadTagger, export_lookahead_tagger
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.utils.logging import logger


@dataclass(frozen=True)
class Case:
    text: str
    expected: tuple = ()
    post_process: bool = True
    punct_pre_process: bool = False
    punct_post_process: bool = False
    source: str = 'text'


def longest_number(text):
    """Count digits per comma/decimal-separated numeric span, including zeros."""
    return max((sum(char.isdigit() for char in span) for span in re.findall(r'[0-9][0-9,.]*', text)), default=0)


def deferred_components(input_case, cache_dir, lookahead_dir=None):
    """Build the paired raw grammars and optionally export their lookahead matcher."""
    tagger = ClassifyFst(input_case=input_case, cache_dir=str(cache_dir), defer_cardinal=True)
    verbalizer = VerbalizeFinalFst(cache_dir=str(cache_dir), defer_cardinal=True)
    if lookahead_dir is not None:
        export_lookahead_tagger(tagger.fst, lookahead_dir, language='en', input_case=input_case, mode='tn')
        tagger = LookaheadTagger(lookahead_dir, language='en', input_case=input_case, mode='tn')
    return tagger, verbalizer


def outcome(normalizer, case):
    try:
        return {
            'output': normalizer.normalize(
                case.text, punct_pre_process=case.punct_pre_process, punct_post_process=case.punct_post_process
            )
        }
    except Exception as error:
        return {'error': type(error).__name__, 'message': str(error)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--input-case', choices=['cased', 'lower_cased'], default='cased')
    parser.add_argument('--test-files', type=Path, nargs='*', default=[])
    parser.add_argument('--cases-jsonl', type=Path, help='One object per line: text and optional normalize settings')
    parser.add_argument('--text', action='append', default=[])
    parser.add_argument('--max-digits', type=int, help='Evaluation filter only; does not limit the grammar/runtime')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--without-lookahead', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.max_digits is not None and args.max_digits < 1:
        parser.error('repeats/max-digits must be positive and warmup must be nonnegative')
    cases = []
    for path in args.test_files:
        for line in path.read_text(encoding='utf-8').splitlines():
            text, *expected = line.split('~')
            if not expected:
                parser.error(f'Expected input~expected in {path}: {line!r}')
            cases.append(
                Case(
                    text,
                    tuple(expected),
                    post_process=path.stem != 'test_cases_money',
                    punct_post_process=path.stem == 'test_cases_punctuation_match_input',
                    source=str(path),
                )
            )
    if args.cases_jsonl:
        for line in args.cases_jsonl.read_text(encoding='utf-8').splitlines():
            data = json.loads(line)
            data['expected'] = tuple(data.get('expected', ()))
            cases.append(Case(**data))
    cases += [Case(text) for text in args.text]
    supplied = len(cases)
    if args.max_digits is not None:
        cases = [case for case in cases if longest_number(case.text) <= args.max_digits]
    if not cases:
        parser.error('No evaluation cases remain; supply text, regression files, or a cases JSONL file')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix='deferred-', dir=args.work_dir))
    cache = args.cache_dir or args.work_dir / 'grammars'
    native = Normalizer(input_case=args.input_case, lang='en', cache_dir=str(cache))
    raw = Normalizer(input_case=args.input_case, lang='en', cache_dir=str(cache))
    raw.tagger, raw.verbalizer = deferred_components(args.input_case, cache)
    variants = {'baseline': native, 'raw': raw}
    if not args.without_lookahead:
        raw_lookahead = Normalizer(input_case=args.input_case, lang='en', cache_dir=str(cache))
        raw_lookahead.tagger, raw_lookahead.verbalizer = deferred_components(
            args.input_case, cache, artifacts / 'lookahead'
        )
        variants['raw-lookahead'] = raw_lookahead
    processors = {name: normalizer.post_processor for name, normalizer in variants.items()}
    timings = {name: [] for name in variants}
    outputs = {name: [] for name in variants}
    was_disabled = logger.disabled
    logger.disabled = True
    try:
        for case in cases[: args.warmup]:
            for name, normalizer in variants.items():
                normalizer.post_processor = processors[name] if case.post_process else None
                outcome(normalizer, case)
        names = list(variants)
        for iteration in range(args.repeats):
            for index, case in enumerate(cases):
                offset = (iteration + index) % len(names)
                for name in names[offset:] + names[:offset]:
                    normalizer = variants[name]
                    normalizer.post_processor = processors[name] if case.post_process else None
                    started = perf_counter()
                    result = outcome(normalizer, case)
                    elapsed_ms = (perf_counter() - started) * 1000
                    if 'output' in result:
                        timings[name].append(elapsed_ms)
                    if iteration == 0:
                        outputs[name].append(result)
    finally:
        logger.disabled = was_disabled
    report = {
        'status': 'complete',
        'input_case': args.input_case,
        'supplied_cases': supplied,
        'filtered_cases': supplied - len(cases),
        'inputs': len(cases),
        'max_digits': args.max_digits,
        'repeats': args.repeats,
        'warmup_inputs': args.warmup,
        'pynini_version': pynini.__version__,
        'corpus_sha256': hashlib.sha256(
            json.dumps([asdict(case) for case in cases], sort_keys=True).encode()
        ).hexdigest(),
        'timing': 'Successful normalize calls, interleaved order; loading and conversion excluded.',
        'artifacts': str(artifacts),
        'variants': {},
    }
    for name in variants:
        ordered = sorted(timings[name])
        differences = []
        expected_failures = []
        shared_errors = []
        for case, baseline, candidate in zip(cases, outputs['baseline'], outputs[name]):
            if baseline != candidate:
                differences.append({'case': asdict(case), 'baseline': baseline, 'candidate': candidate})
            elif 'error' in candidate:
                shared_errors.append({'case': asdict(case), 'result': candidate})
            if case.expected and candidate.get('output') not in case.expected:
                expected_failures.append({'case': asdict(case), 'result': candidate})
        report['variants'][name] = {
            'successful_timed_calls': len(ordered),
            'mean_ms': statistics.mean(ordered) if ordered else None,
            'p50_ms': statistics.median(ordered) if ordered else None,
            'p95_ms': ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else None,
            'differences': differences,
            'shared_errors': shared_errors,
            'scored': sum(bool(case.expected) for case in cases),
            'expected_failures': expected_failures,
        }
    report['parity_passed'] = all(not entry['differences'] for entry in report['variants'].values())
    report['expected_outputs_passed'] = all(not entry['expected_failures'] for entry in report['variants'].values())
    report['error_free'] = all('error' not in result for results in outputs.values() for result in results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for name, entry in report['variants'].items():
        print(
            name,
            json.dumps(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in ['differences', 'expected_failures', 'shared_errors']
                }
            ),
            'diffs',
            len(entry['differences']),
            'expected failures',
            len(entry['expected_failures']),
            'shared errors',
            len(entry['shared_errors']),
        )
    return int(not report['parity_passed'] or not report['expected_outputs_passed'] or not report['error_free'])


if __name__ == '__main__':
    raise SystemExit(main())
