"""Minimal evaluation entry point for QAQ runtime checks."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser

from qaq.config import ConfigValidationError, load_config_file
from qaq.config import QAQ_MODES
from qaq.results import ResultValidationError, build_result_artifact, save_result_artifact
from qaq.runtime.adaptive import run_adaptive_runtime
from qaq.runtime.common import RuntimeError
from qaq.runtime.static import load_artifact_index, run_static_runtime


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run a QAQ evaluation smoke pass.")
    parser.add_argument("--config", required=True, help="Path to a JSON or TOML config.")
    parser.add_argument(
        "--artifact-index",
        default=None,
        help="Optional JSON mapping block IDs and bit-widths to bit-plane artifact paths.",
    )
    parser.add_argument(
        "--skip-output-dir-check",
        action="store_true",
        help="Skip existing output directory reuse validation.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the runtime output bundle as JSON.",
    )
    parser.add_argument(
        "--result-output",
        default=None,
        help="Optional path for the machine-readable result artifact JSON.",
    )
    parser.add_argument(
        "--print-result-json",
        action="store_true",
        help="Print the result artifact as JSON.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config_file(
            args.config,
            validate_output=not args.skip_output_dir_check,
        )
        artifact_refs = (
            load_artifact_index(args.artifact_index) if args.artifact_index else None
        )
        if config.mode in QAQ_MODES:
            result = run_adaptive_runtime(config, artifact_refs=artifact_refs)
        else:
            result = run_static_runtime(config, artifact_refs=artifact_refs)
        result_artifact = build_result_artifact(config, result)
        if args.result_output:
            save_result_artifact(result_artifact, args.result_output)
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except ResultValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.print_json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    if args.print_result_json:
        print(json.dumps(result_artifact.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
