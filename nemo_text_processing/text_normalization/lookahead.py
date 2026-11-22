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

"""Optional OpenFst label-lookahead tagging; requires OpenFst's lookahead plugin."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pynini
import pywrapfst


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_lookahead_tagger(fst, output_dir, *, language, input_case, mode, fstconvert=None):
    """Convert a tagger once, saving its matcher and matching relabel table together.

    ``output_dir`` must not already exist. The ordinary grammar FARs are retained
    separately for use with the default runtime and Sparrowhawk.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Lookahead export already exists: {output_dir}')
    if mode not in ('tn', 'itn'):
        raise ValueError('mode must be tn or itn')
    converter = fstconvert or shutil.which('fstconvert')
    if converter is None:
        candidate = Path(sys.prefix) / 'bin' / 'fstconvert'
        if candidate.is_file():
            converter = str(candidate)
        else:
            raise RuntimeError(
                'fstconvert is required; install Pynini and OpenFst with lookahead plugins from conda-forge.'
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lookahead-', dir=output_dir.parent) as temporary:
        temporary = Path(temporary)
        source = temporary / 'source.fst'
        matcher = temporary / 'tagger.fst'
        relabel = temporary / 'relabel.pairs'
        fst.write(str(source))
        result = subprocess.run(
            [
                str(converter),
                '--fst_type=ilabel_lookahead',
                f'--save_relabel_ipairs={relabel}',
                str(source),
                str(matcher),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                'Lookahead conversion failed. Use an OpenFst build with the ilabel_lookahead plugin '
                f'(the Pynini pip wheels do not supply it).\n{result.stderr.strip()}'
            )
        metadata = {
            'version': 1,
            'language': language,
            'input_case': input_case,
            'mode': mode,
            'sha256': {path.name: _sha256(path) for path in (matcher, relabel)},
        }
        (temporary / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        source.unlink()
        # Publish the matcher and table together; a failed conversion leaves no artifact.
        temporary.rename(output_dir)
    return output_dir


class LookaheadTagger:
    """A read-only matcher and the input relabeling required by that matcher."""

    def __init__(self, directory, *, language, input_case, mode):
        self.directory = str(Path(directory).resolve())
        self.expected = {'language': language, 'input_case': input_case, 'mode': mode}
        self._load()

    def _load(self):
        directory = Path(self.directory)
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('version') != 1:
            raise ValueError('Unsupported lookahead export version')
        for key, value in self.expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'Lookahead {key} mismatch: expected {value!r}, found {metadata.get(key)!r}')
        for name in ('tagger.fst', 'relabel.pairs'):
            if _sha256(directory / name) != metadata['sha256'].get(name):
                raise ValueError(f'Lookahead artifact checksum mismatch: {name}')
        self.relabel_pairs = [
            tuple(map(int, line.split()))
            for line in (directory / 'relabel.pairs').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        if any(len(pair) != 2 or min(pair) < 0 or (0 in pair and pair != (0, 0)) for pair in self.relabel_pairs):
            raise ValueError('Invalid lookahead relabel table')
        try:
            # pynini.Fst.read() silently converts this to a vector FST, losing the matcher.
            self.fst = pywrapfst.Fst.read(str(directory / 'tagger.fst'))
        except pywrapfst.FstIOError as error:
            raise RuntimeError(
                'Cannot load the lookahead tagger; install the matching OpenFst lookahead plugin.'
            ) from error
        if self.fst.fst_type() != 'ilabel_lookahead':
            raise ValueError('Expected an ilabel_lookahead tagger')

    def compose(self, escaped_text):
        """Tag an escaped Pynini string, preserving its original input labels."""
        text = pynini.accep(escaped_text)
        # Only the composition interface (the output labels) is relabeled. The
        # extra collision-avoidance pairs emitted by fstconvert must be retained.
        if self.relabel_pairs:
            text.relabel_pairs(opairs=self.relabel_pairs)
        lattice = pywrapfst.compose(text, self.fst, compose_filter='auto')
        return pynini.Fst.from_pywrapfst(lattice)

    def __getstate__(self):
        # Reload the immutable matcher in joblib worker processes rather than
        # allowing Pynini serialization to turn it into an ordinary vector FST.
        return {'directory': self.directory, 'expected': self.expected}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()
