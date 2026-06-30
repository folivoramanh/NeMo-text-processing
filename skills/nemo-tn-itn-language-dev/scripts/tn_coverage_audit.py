#!/usr/bin/env python3
"""Audit TN test-case coverage for a NeMo semiotic class.

This script is intentionally offline: it does not import Pynini or run the
normalizer. It reads test case files and a JSON coverage profile, then reports
which input-shape scenarios are covered.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    detector: dict[str, Any]
    suggested_input: str


def contains_sentence_context(value: str) -> bool:
    return bool(re.search(r"[^\W\d_]\s+", value, flags=re.UNICODE) and re.search(r"\d", value))


def detector_matches(detector: dict[str, Any], value: str, case: tuple[str, list[str]]) -> bool:
    kind = detector.get("type")
    flags = re.IGNORECASE if detector.get("ignore_case") else 0

    if kind == "all":
        return all(detector_matches(item, value, case) for item in detector["detectors"])
    if kind == "any_of":
        return any(detector_matches(item, value, case) for item in detector["detectors"])
    if kind == "not":
        return not detector_matches(detector["detector"], value, case)
    if kind == "any":
        return bool(value.strip())
    if kind == "sentence_context":
        return contains_sentence_context(value)
    if kind == "multiple_expected":
        return len(case[1]) > 1
    if kind == "regex_fullmatch":
        return bool(re.fullmatch(detector["pattern"], value, flags=flags))
    if kind == "regex_search":
        return bool(re.search(detector["pattern"], value, flags=flags))
    if kind == "startswith":
        prefixes = detector["prefix"]
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        return any(value.strip().startswith(prefix) for prefix in prefixes)
    if kind == "contains":
        needles = detector["value"]
        if isinstance(needles, str):
            needles = [needles]
        return any(needle in value for needle in needles)

    raise ValueError(f"Unsupported detector type: {kind}")


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_profile_path(class_name: str) -> Path:
    specific = skill_root() / "coverage_profiles" / f"{class_name}.json"
    if specific.exists():
        return specific
    return skill_root() / "coverage_profiles" / "generic.json"


def merge_items(base: list[dict[str, Any]], child: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {item["key"]: item for item in base}
    for item in child:
        merged[item["key"]] = item
    return list(merged.values())


def load_profile(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular coverage profile inheritance involving {path}")
    seen.add(path)

    with path.open(encoding="utf-8") as f:
        profile = json.load(f)

    parent_name = profile.get("extends")
    if parent_name:
        parent_path = path.parent / f"{parent_name}.json"
        parent = load_profile(parent_path, seen)
        profile = {
            **parent,
            **profile,
            "scenarios": merge_items(parent.get("scenarios", []), profile.get("scenarios", [])),
            "risk_scenarios": merge_items(parent.get("risk_scenarios", []), profile.get("risk_scenarios", [])),
        }
    if "scenarios" not in profile or not isinstance(profile["scenarios"], list):
        raise ValueError(f"{path}: expected top-level 'scenarios' list or 'extends'")
    return profile


def parse_scenarios(profile: dict[str, Any]) -> list[Scenario]:
    scenarios = []
    for item in profile["scenarios"]:
        scenarios.append(
            Scenario(
                key=item["key"],
                description=item["description"],
                detector=item["detector"],
                suggested_input=item.get("suggested_input", ""),
            )
        )
    return scenarios


def parse_cases(path: Path) -> list[tuple[str, list[str]]]:
    cases: list[tuple[str, list[str]]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split("~")
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_no}: expected input~expected format")
        cases.append((parts[0], parts[1:]))
    return cases


def scenario_report(scenarios: Iterable[Scenario], cases: list[tuple[str, list[str]]]) -> list[dict[str, object]]:
    report = []
    for scenario in scenarios:
        matches = [case[0] for case in cases if detector_matches(scenario.detector, case[0], case)]
        report.append(
            {
                "key": scenario.key,
                "description": scenario.description,
                "covered": bool(matches),
                "examples": matches[:3],
                "suggested_input": scenario.suggested_input,
            }
        )
    return report


def build_report(repo_root: Path, lang: str, class_name: str, profile_path: Path) -> dict[str, object]:
    test_path = (
        repo_root
        / "tests"
        / "nemo_text_processing"
        / lang
        / "data_text_normalization"
        / f"test_cases_{class_name}.txt"
    )
    profile = load_profile(profile_path)
    scenarios = parse_scenarios(profile)
    risk_scenarios = parse_scenarios({"scenarios": profile.get("risk_scenarios", [])})

    if not test_path.exists():
        return {
            "lang": lang,
            "class": class_name,
            "test_file": str(test_path),
            "profile": str(profile_path),
            "exists": False,
            "error": "test case file not found",
        }

    cases = parse_cases(test_path)
    inputs = [case[0] for case in cases]
    duplicates = [item for item, count in Counter(inputs).items() if count > 1]
    multiple_expected = [case[0] for case in cases if len(case[1]) > 1]
    scenario_items = scenario_report(scenarios, cases)
    risk_items = [item for item in scenario_report(risk_scenarios, cases) if item["covered"]]
    missing = [item for item in scenario_items if not item["covered"]]

    return {
        "lang": lang,
        "class": class_name,
        "test_file": str(test_path),
        "profile": str(profile_path),
        "exists": True,
        "case_count": len(cases),
        "duplicate_inputs": duplicates,
        "multiple_expected_inputs": multiple_expected,
        "covered_scenarios": [item for item in scenario_items if item["covered"]],
        "missing_scenarios": missing,
        "risk_scenarios": risk_items,
    }


def print_markdown(report: dict[str, object]) -> None:
    print(f"# TN Coverage Audit: {report['lang']} / {report['class']}")
    print()
    print(f"Test file: `{report['test_file']}`")
    print(f"Profile: `{report['profile']}`")
    if not report.get("exists"):
        print()
        print(f"Error: {report['error']}")
        return

    print()
    print(f"Cases: {report['case_count']}")
    duplicates = report["duplicate_inputs"]
    multiple = report["multiple_expected_inputs"]
    print(f"Duplicate inputs: {len(duplicates)}")
    print(f"Multiple-expected inputs: {len(multiple)}")

    if duplicates:
        print()
        print("## Duplicate Inputs")
        for item in duplicates:
            print(f"- `{item}`")

    print()
    print("## Covered Scenarios")
    for item in report["covered_scenarios"]:
        examples = ", ".join(f"`{example}`" for example in item["examples"])
        print(f"- `{item['key']}`: {item['description']} ({examples})")

    print()
    print("## Risky Or Ambiguous Inputs")
    if not report["risk_scenarios"]:
        print("- None detected by the selected profile.")
    else:
        for item in report["risk_scenarios"]:
            examples = ", ".join(f"`{example}`" for example in item["examples"])
            print(f"- `{item['key']}`: {item['description']} ({examples})")

    print()
    print("## Missing Scenarios")
    if not report["missing_scenarios"]:
        print("- None detected by the selected profile.")
    else:
        for item in report["missing_scenarios"]:
            print(f"- `{item['key']}`: {item['description']}")
            if item["suggested_input"]:
                print(f"  Suggested input shape: `{item['suggested_input']}`")


def list_profiles() -> None:
    profile_dir = skill_root() / "coverage_profiles"
    for path in sorted(profile_dir.glob("*.json")):
        profile = load_profile(path)
        label = profile.get("name", path.stem)
        print(f"{path.stem}: {label} ({path})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--lang", help="Language code, e.g. vi, ja, es.")
    parser.add_argument("--class", dest="class_name", help="Semiotic class, e.g. telephone.")
    parser.add_argument("--profile", help="Path to a JSON coverage profile. Defaults to coverage_profiles/<class>.json.")
    parser.add_argument("--list-profiles", action="store_true", help="List bundled coverage profiles and exit.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    args = parser.parse_args()

    if args.list_profiles:
        list_profiles()
        return 0

    if not args.lang or not args.class_name:
        parser.error("--lang and --class are required unless --list-profiles is used")

    profile_path = Path(args.profile).resolve() if args.profile else default_profile_path(args.class_name)
    report = build_report(Path(args.repo_root).resolve(), args.lang, args.class_name, profile_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_markdown(report)
    return 0 if report.get("exists") else 1


if __name__ == "__main__":
    raise SystemExit(main())
