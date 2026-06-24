from pathlib import Path

import pytest

from qaq.config import RunConfig
from qaq.runtime import adaptive as adaptive_runtime
from qaq.runtime import static as static_runtime


def _cuda_config(tmp_path: Path, mode: str = "fp16") -> RunConfig:
    return RunConfig.from_mapping(
        {
            "model": "fake-qaq-smoke-model",
            "tokenizer": "fake-qaq-smoke-tokenizer",
            "dataset": "fake_smoke",
            "split": "validation",
            "mode": mode,
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cuda",
            "gpu_ids": [0],
            "seed": 0,
            "output_dir": str(tmp_path / mode),
            "logging": {"console": False},
            "router_diagnostic": mode.startswith("qaq_"),
        },
        validate_output=False,
    )


class _CudaWithInvalidDeviceStats:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def reset_peak_memory_stats(index: int) -> None:
        raise RuntimeError("Invalid device argument")

    @staticmethod
    def max_memory_allocated(index: int) -> int:
        raise RuntimeError("Invalid device argument")


class _TorchWithInvalidDeviceStats:
    cuda = _CudaWithInvalidDeviceStats()


def test_static_cuda_memory_stats_ignore_invalid_device_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cuda_config(tmp_path)
    monkeypatch.setattr(static_runtime, "_try_import_torch", lambda: _TorchWithInvalidDeviceStats)

    static_runtime._reset_cuda_peak_memory_if_available(config)

    assert static_runtime._peak_gpu_memory_gb(config) == 0.0


def test_adaptive_cuda_memory_stats_ignore_invalid_device_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cuda_config(tmp_path, mode="qaq_on_demand_off")
    monkeypatch.setattr(adaptive_runtime, "_try_import_torch", lambda: _TorchWithInvalidDeviceStats)

    adaptive_runtime._reset_cuda_peak_memory_if_available(config)

    assert adaptive_runtime._peak_gpu_memory_gb(config) == 0.0


def test_cuda_memory_stats_reraise_other_runtime_errors() -> None:
    class BrokenCuda:
        @staticmethod
        def reset_peak_memory_stats(index: int) -> None:
            raise RuntimeError("driver unavailable")

        @staticmethod
        def max_memory_allocated(index: int) -> int:
            raise RuntimeError("driver unavailable")

    class BrokenTorch:
        cuda = BrokenCuda()

    with pytest.raises(RuntimeError, match="driver unavailable"):
        static_runtime._safe_reset_peak_memory_stats(BrokenTorch, 0)
    with pytest.raises(RuntimeError, match="driver unavailable"):
        adaptive_runtime._safe_max_memory_allocated(BrokenTorch, 0)
