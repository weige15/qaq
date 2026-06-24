import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_run.py"


def _write_fake_nvidia_smi(tmp_path: Path, output: str, exit_code: int = 0) -> dict[str, str]:
    fake = tmp_path / "nvidia-smi"
    fake.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                f"sys.stdout.write({output!r})",
                f"sys.exit({exit_code})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    return env


def _first_status(stderr: str) -> dict:
    return json.loads(stderr.splitlines()[0])


def test_gpu_run_selects_free_physical_gpus_and_maps_cuda_visible_devices(
    tmp_path: Path,
) -> None:
    env = _write_fake_nvidia_smi(
        tmp_path,
        "\n".join(
            [
                "0, NVIDIA GeForce RTX 3090, 24576, 1000",
                "3, NVIDIA GeForce RTX 3090, 24576, 24000",
                "5, NVIDIA GeForce RTX 3090, 24576, 23000",
            ]
        )
        + "\n",
    )
    command = (
        "import os, sys; "
        "ok = os.environ.get('CUDA_VISIBLE_DEVICES') == '3,5' "
        "and os.environ.get('QAQ_SELECTED_PHYSICAL_GPUS') == '3,5'; "
        "sys.exit(0 if ok else 9)"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count",
            "2",
            "--min-free-mb",
            "20000",
            "--",
            sys.executable,
            "-c",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    status = _first_status(completed.stderr)
    assert status["selected_physical_gpu_ids"] == [3, 5]
    assert status["cuda_visible_devices"] == "3,5"
    assert status["pytorch_logical_mapping"] == {"cuda:0": 3, "cuda:1": 5}


def test_gpu_run_stops_before_command_when_no_gpu_has_required_memory(
    tmp_path: Path,
) -> None:
    env = _write_fake_nvidia_smi(
        tmp_path,
        "\n".join(
            [
                "0, NVIDIA GeForce RTX 3090, 24576, 1000",
                "1, NVIDIA GeForce RTX 3090, 24576, 1500",
            ]
        )
        + "\n",
    )
    marker = tmp_path / "should_not_exist"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count",
            "1",
            "--min-free-mb",
            "20000",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert not marker.exists()
    status = _first_status(completed.stderr)
    assert status["code"] == "no_suitable_gpus"
    assert status["selected_physical_gpu_ids"] == []


def test_gpu_run_rejects_non_lab_gpu_name_by_default(tmp_path: Path) -> None:
    env = _write_fake_nvidia_smi(
        tmp_path,
        "0, NVIDIA GeForce RTX 4050 Laptop GPU, 6144, 5000\n",
    )
    marker = tmp_path / "should_not_run_on_local_gpu"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count",
            "1",
            "--min-free-mb",
            "1000",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert not marker.exists()
    status = _first_status(completed.stderr)
    assert status["code"] == "no_suitable_gpus"
    assert status["gpu_name_contains"] == "RTX 3090"


def test_gpu_run_writes_status_file_for_selected_physical_ids(tmp_path: Path) -> None:
    env = _write_fake_nvidia_smi(
        tmp_path,
        "2, NVIDIA GeForce RTX 3090, 24576, 24000\n",
    )
    status_file = tmp_path / "run" / "gpu_status.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count",
            "1",
            "--min-free-mb",
            "20000",
            "--status-file",
            str(status_file),
            "--dry-run",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(99)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status["selected_physical_gpu_ids"] == [2]
    assert status["cuda_visible_devices"] == "2"
    assert status["dry_run"] is True
