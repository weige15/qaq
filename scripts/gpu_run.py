#!/usr/bin/env python3
"""Run a command on free physical GPUs selected from nvidia-smi output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_PHYSICAL_IDS = "0-7"
DEFAULT_GPU_NAME_CONTAINS = "RTX 3090"
GPU_QUERY_ARGS = (
    "--query-gpu=index,name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
)


@dataclass(frozen=True)
class GpuInfo:
    physical_id: int
    name: str
    memory_total_mb: int
    memory_free_mb: int


class GpuRunError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _parse_mb(value: str) -> int:
    normalized = value.strip().replace("MiB", "").strip()
    try:
        return int(float(normalized))
    except ValueError as exc:
        raise GpuRunError(
            "invalid_nvidia_smi_output",
            f"could not parse memory value {value!r}",
        ) from exc


def parse_nvidia_smi_csv(output: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            raise GpuRunError(
                "invalid_nvidia_smi_output",
                f"expected at least 4 CSV columns, got {line!r}",
            )
        try:
            physical_id = int(parts[0])
        except ValueError as exc:
            raise GpuRunError(
                "invalid_nvidia_smi_output",
                f"could not parse GPU index from {line!r}",
            ) from exc
        name = ",".join(parts[1:-2]).strip()
        gpus.append(
            GpuInfo(
                physical_id=physical_id,
                name=name,
                memory_total_mb=_parse_mb(parts[-2]),
                memory_free_mb=_parse_mb(parts[-1]),
            )
        )
    if not gpus:
        raise GpuRunError("no_gpus_detected", "nvidia-smi did not report any GPUs")
    return gpus


def parse_physical_ids(value: str) -> tuple[int, ...]:
    ids: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise GpuRunError(
                    "invalid_physical_ids",
                    f"could not parse physical GPU id range {part!r}",
                ) from exc
            if start > end:
                raise GpuRunError(
                    "invalid_physical_ids",
                    f"physical GPU id range {part!r} is descending",
                )
            ids.extend(range(start, end + 1))
        else:
            try:
                ids.append(int(part))
            except ValueError as exc:
                raise GpuRunError(
                    "invalid_physical_ids",
                    f"could not parse physical GPU id {part!r}",
                ) from exc
    if not ids:
        raise GpuRunError("invalid_physical_ids", "no physical GPU ids were provided")
    if len(ids) != len(set(ids)):
        raise GpuRunError("invalid_physical_ids", "physical GPU ids must be unique")
    return tuple(ids)


def query_gpus(nvidia_smi: str) -> list[GpuInfo]:
    try:
        completed = subprocess.run(
            [nvidia_smi, *GPU_QUERY_ARGS],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GpuRunError(
            "nvidia_smi_not_found",
            "nvidia-smi was not found; refusing to run without GPU status",
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GpuRunError(
            "nvidia_smi_failed",
            f"nvidia-smi failed with exit code {completed.returncode}: {detail}",
        )
    return parse_nvidia_smi_csv(completed.stdout)


def select_gpus(
    gpus: Sequence[GpuInfo],
    *,
    count: int,
    min_free_mb: int,
    physical_ids: Sequence[int],
    gpu_name_contains: str,
) -> list[GpuInfo]:
    if count <= 0:
        raise GpuRunError("invalid_count", "--count must be positive")
    if min_free_mb < 0:
        raise GpuRunError("invalid_min_free_mb", "--min-free-mb must be non-negative")
    allowed_ids = set(physical_ids)
    required_name = gpu_name_contains.strip().lower()
    eligible = [
        gpu
        for gpu in gpus
        if gpu.physical_id in allowed_ids
        and gpu.memory_free_mb >= min_free_mb
        and (not required_name or required_name in gpu.name.lower())
    ]
    eligible.sort(key=lambda gpu: (-gpu.memory_free_mb, gpu.physical_id))
    if len(eligible) < count:
        raise GpuRunError(
            "no_suitable_gpus",
            (
                f"requested {count} GPU(s) with at least {min_free_mb} MiB free "
                f"and name containing {gpu_name_contains!r}, but only "
                f"{len(eligible)} eligible GPU(s) were found"
            ),
        )
    return eligible[:count]


def build_status(
    *,
    status: str,
    command: Sequence[str],
    requested_count: int,
    min_free_mb: int,
    physical_ids: Sequence[int],
    gpu_name_contains: str,
    detected_gpus: Sequence[GpuInfo],
    selected_gpus: Sequence[GpuInfo],
    message: str | None = None,
    code: str | None = None,
    dry_run: bool = False,
) -> dict:
    selected_ids = [gpu.physical_id for gpu in selected_gpus]
    payload = {
        "status": status,
        "code": code,
        "message": message,
        "requested_count": requested_count,
        "min_free_mb": min_free_mb,
        "eligible_physical_ids": list(physical_ids),
        "gpu_name_contains": gpu_name_contains,
        "selected_physical_gpu_ids": selected_ids,
        "cuda_visible_devices": ",".join(str(gpu_id) for gpu_id in selected_ids),
        "pytorch_logical_mapping": {
            f"cuda:{logical_id}": physical_id
            for logical_id, physical_id in enumerate(selected_ids)
        },
        "detected_gpus": [asdict(gpu) for gpu in detected_gpus],
        "command": list(command),
        "dry_run": dry_run,
    }
    return {key: value for key, value in payload.items() if value is not None}


def emit_status(payload: dict, status_file: str | None) -> None:
    text = json.dumps(payload, sort_keys=True)
    print(text, file=sys.stderr, flush=True)
    if status_file:
        path = Path(status_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_cli(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Select free physical GPUs with nvidia-smi, set CUDA_VISIBLE_DEVICES, "
            "and run a command."
        )
    )
    parser.add_argument("--count", type=int, required=True, help="Number of GPUs to select.")
    parser.add_argument(
        "--min-free-mb",
        type=int,
        required=True,
        help="Minimum free memory in MiB required on each selected GPU.",
    )
    parser.add_argument(
        "--physical-ids",
        default=DEFAULT_PHYSICAL_IDS,
        help="Comma-separated physical GPU ids or ranges eligible for selection.",
    )
    parser.add_argument(
        "--gpu-name-contains",
        default=DEFAULT_GPU_NAME_CONTAINS,
        help=(
            "Required substring in the nvidia-smi GPU name. Use an empty string "
            "only when intentionally targeting non-RTX-3090 hardware."
        ),
    )
    parser.add_argument(
        "--status-file",
        help="Optional JSON file where selected physical GPU ids and command metadata are recorded.",
    )
    parser.add_argument(
        "--nvidia-smi",
        default="nvidia-smi",
        help="Path to nvidia-smi. Intended for tests or unusual environments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected GPUs without running the command.",
    )
    if "--" not in argv:
        parser.error("command separator '--' is required before the command")
    separator_index = list(argv).index("--")
    args = parser.parse_args(list(argv[:separator_index]))
    command = list(argv[separator_index + 1 :])
    if not command:
        parser.error("a command is required after '--'")
    return args, command


def main(argv: Sequence[str] | None = None) -> int:
    args, command = parse_cli(sys.argv[1:] if argv is None else argv)
    detected_gpus: list[GpuInfo] = []
    try:
        physical_ids = parse_physical_ids(args.physical_ids)
        detected_gpus = query_gpus(args.nvidia_smi)
        selected_gpus = select_gpus(
            detected_gpus,
            count=args.count,
            min_free_mb=args.min_free_mb,
            physical_ids=physical_ids,
            gpu_name_contains=args.gpu_name_contains,
        )
    except GpuRunError as exc:
        payload = build_status(
            status="failed",
            code=exc.code,
            message=exc.message,
            command=command,
            requested_count=args.count,
            min_free_mb=args.min_free_mb,
            physical_ids=parse_physical_ids(args.physical_ids)
            if exc.code != "invalid_physical_ids"
            else (),
            gpu_name_contains=args.gpu_name_contains,
            detected_gpus=detected_gpus,
            selected_gpus=(),
            dry_run=args.dry_run,
        )
        emit_status(payload, args.status_file)
        return 2

    cuda_visible_devices = ",".join(str(gpu.physical_id) for gpu in selected_gpus)
    payload = build_status(
        status="selected",
        command=command,
        requested_count=args.count,
        min_free_mb=args.min_free_mb,
        physical_ids=physical_ids,
        gpu_name_contains=args.gpu_name_contains,
        detected_gpus=detected_gpus,
        selected_gpus=selected_gpus,
        dry_run=args.dry_run,
    )
    emit_status(payload, args.status_file)
    if args.dry_run:
        return 0

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    child_env["QAQ_SELECTED_PHYSICAL_GPUS"] = cuda_visible_devices
    child_env["QAQ_GPU_RUN_STATUS"] = json.dumps(payload, sort_keys=True)
    try:
        completed = subprocess.run(command, check=False, env=child_env)
    except FileNotFoundError as exc:
        error_payload = dict(payload)
        error_payload.update(
            {
                "status": "failed",
                "code": "command_not_found",
                "message": str(exc),
            }
        )
        emit_status(error_payload, args.status_file)
        return 127
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
