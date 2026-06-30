#!/usr/bin/env python3
"""Scan NeMo TN/ITN class implementations for repo conventions.

The script is offline and regex-based. It does not import Pynini. Use it before
editing a language/class to find comparable implementations, data-file usage,
helper patterns, and wiring gaps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HELPER_PATTERNS = {
    "GraphFst": r"\bGraphFst\b",
    "add_tokens": r"\.add_tokens\(",
    "delete_tokens": r"\.delete_tokens\(",
    "string_file": r"pynini\.string_file\(",
    "string_map": r"pynini\.string_map\(",
    "get_abs_path": r"get_abs_path\(",
    "convert_space": r"\bconvert_space\(",
    "add_weight": r"pynutil\.add_weight\(",
    "compose": r"pynini\.compose\(",
    "cdrewrite": r"pynini\.cdrewrite\(",
    "preserve_order": r"preserve_order",
    "NEMO_SIGMA": r"\bNEMO_SIGMA\b",
}


def direction_dir(direction: str) -> str:
    return "text_normalization" if direction == "tn" else "inverse_text_normalization"


def class_name(class_id: str) -> str:
    return "".join(part.capitalize() for part in class_id.split("_")) + "Fst"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def extract_imports(text: str) -> list[str]:
    imports = []
    for line in text.splitlines():
        if line.startswith("from nemo_text_processing") or line.startswith("import pynini"):
            imports.append(line)
    return imports


def extract_data_refs(text: str) -> list[str]:
    refs = set()
    for match in re.finditer(r"get_abs_path\([\"']([^\"']+)[\"']\)", text):
        refs.add(match.group(1))
    return sorted(refs)


def helper_hits(text: str) -> list[str]:
    return [name for name, pattern in HELPER_PATTERNS.items() if re.search(pattern, text)]


def warning_hits(text: str) -> list[str]:
    warnings = []
    inline_maps = len(re.findall(r"pynini\.string_map\(\s*\[", text))
    if inline_maps >= 2:
        warnings.append(f"{inline_maps} inline string_map lists; consider TSV data if this is lexical inventory")
    if re.search(r"NEMO_SIGMA\s*\+", text) or re.search(r"\+\s*NEMO_SIGMA", text):
        warnings.append("NEMO_SIGMA appears in a broad concatenation; check over-matching risk")
    if "preserve_order" in text and "delete_preserve_order" not in text:
        warnings.append("preserve_order emitted here; verify verbalizer deletes or handles it")
    if "data/" not in text and "string_map" in text:
        warnings.append("no data/ references found but string_map is used; check for hard-coded lexicon")
    return warnings


def find_implementations(root: Path, direction: str, class_id: str) -> list[Path]:
    base = root / "nemo_text_processing" / direction_dir(direction)
    paths = []
    for subdir in ("taggers", "verbalizers"):
        paths.extend(base.glob(f"*/{subdir}/{class_id}.py"))
    return sorted(paths)


def target_paths(root: Path, direction: str, lang: str, class_id: str) -> dict[str, Path]:
    base = root / "nemo_text_processing" / direction_dir(direction) / lang
    test_base = root / "tests" / "nemo_text_processing" / lang
    data_kind = "data_text_normalization" if direction == "tn" else "data_inverse_text_normalization"
    return {
        "tagger": base / "taggers" / f"{class_id}.py",
        "verbalizer": base / "verbalizers" / f"{class_id}.py",
        "tagger_composer": base / "taggers" / "tokenize_and_classify.py",
        "verbalizer_composer": base / "verbalizers" / "verbalize.py",
        "test_py": test_base / f"test_{class_id}.py",
        "test_cases": test_base / data_kind / f"test_cases_{class_id}.txt",
        "sparrowhawk": test_base / (
            "test_sparrowhawk_normalization.sh"
            if direction == "tn"
            else "test_sparrowhawk_inverse_text_normalization.sh"
        ),
    }


def print_file_summary(root: Path, label: str, path: Path) -> None:
    text = read(path)
    exists = path.exists()
    print(f"### {label}: `{rel(path, root)}`")
    print(f"- exists: {exists}")
    if not exists:
        print()
        return
    print(f"- lines: {len(text.splitlines())}")
    data_refs = extract_data_refs(text)
    if data_refs:
        print("- data refs:")
        for item in data_refs:
            print(f"  - `{item}`")
    hits = helper_hits(text)
    if hits:
        print(f"- helper patterns: {', '.join(hits)}")
    warnings = warning_hits(text)
    if warnings:
        print("- warnings:")
        for item in warnings:
            print(f"  - {item}")
    print()


def composer_status(text: str, class_id: str) -> str:
    expected_class = class_name(class_id)
    if expected_class in text or f".{class_id} import" in text:
        return "appears registered"
    return "not found"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--direction", choices=["tn", "itn"], required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--class", dest="class_id", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    paths = target_paths(root, args.direction, args.lang, args.class_id)

    print(f"# Repo Convention Scan: {args.direction} / {args.lang} / {args.class_id}")
    print()

    print("## Target Files")
    for label, path in paths.items():
        print_file_summary(root, label, path)

    print("## Composer Wiring")
    tagger_status = composer_status(read(paths["tagger_composer"]), args.class_id)
    verbalizer_status = composer_status(read(paths["verbalizer_composer"]), args.class_id)
    print(f"- tagger composer: {tagger_status}")
    print(f"- verbalizer composer: {verbalizer_status}")
    print()

    print("## Comparable Implementations")
    implementations = [path for path in find_implementations(root, args.direction, args.class_id) if path not in paths.values()]
    if not implementations:
        print("- No comparable implementations found.")
    for path in implementations:
        text = read(path)
        print(f"- `{rel(path, root)}`")
        refs = extract_data_refs(text)
        hits = helper_hits(text)
        if refs:
            print(f"  - data refs: {', '.join(refs[:8])}")
        if hits:
            print(f"  - helper patterns: {', '.join(hits)}")
    print()

    print("## Convention Checklist")
    print("- Prefer TSV-backed lexical inventories over inline string_map lists.")
    print("- Reuse existing language number/month/unit resources before adding duplicates.")
    print("- Define input -> token -> output before changing the graph.")
    print("- Check class ownership and risk profile before broadening the accepted domain.")
    print("- Keep tagger, verbalizer, composer, tests, and Sparrowhawk hooks aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
