"""Report generation for QAQ result artifacts."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from qaq.results import (
    ResultValidationError,
    build_report_rows,
    group_result_artifacts,
    load_result_artifact,
    validate_comparison,
)


def build_report(result_paths: list[str | Path]) -> dict[str, Any]:
    """Load result artifacts and build grouped comparison rows."""

    artifacts = tuple(load_result_artifact(path) for path in result_paths)
    comparisons: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for key, group in group_result_artifacts(artifacts).items():
        validation = validate_comparison(group)
        comparisons.append(
            {
                "key": key.as_dict(),
                "validation": validation.as_dict(),
                "result_count": len(group),
            }
        )
        rows.extend(build_report_rows(group, validation=validation))
    return {
        "schema_version": "qaq.report.v1",
        "comparisons": comparisons,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a QAQ comparison report.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Result artifact JSON files to compare.",
    )
    parser.add_argument("--output", help="Optional report JSON output path.")
    parser.add_argument("--print-json", action="store_true", help="Print report JSON.")
    args = parser.parse_args(argv)

    try:
        report = build_report([Path(path) for path in args.results])
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.print_json:
            print(json.dumps(report, indent=2, sort_keys=True))
    except ResultValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
