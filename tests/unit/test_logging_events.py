import io
import json

from qaq.logging import JsonlLogWriter, LogEvent
from qaq.progress import ConsoleProgressMonitor, TimingMeasurement
from qaq.status import EventType, RunStatus


def test_log_event_serializes_structured_fields() -> None:
    event = LogEvent.progress(
        run_id="run-1",
        module="trainer",
        step=3,
        epoch=1,
        loss=0.25,
        learning_rate=1e-4,
        elapsed_seconds=12.5,
        selected_gpu_ids=(0, 1),
        details={"phase": "distill"},
    )

    data = event.as_dict()

    assert data["event_type"] == EventType.PROGRESS.value
    assert data["run_id"] == "run-1"
    assert data["module"] == "trainer"
    assert data["status"] == RunStatus.RUNNING.value
    assert data["step"] == 3
    assert data["epoch"] == 1
    assert data["loss"] == 0.25
    assert data["learning_rate"] == 1e-4
    assert data["elapsed_seconds"] == 12.5
    assert data["selected_gpu_ids"] == [0, 1]
    assert data["details"] == {"phase": "distill"}


def test_jsonl_writer_persists_and_flushes_events(tmp_path) -> None:
    log_path = tmp_path / "events.jsonl"
    writer = JsonlLogWriter(log_path)

    writer.record(
        LogEvent(
            event_type=EventType.METRIC.value,
            run_id="run-1",
            module="eval",
            mode="static_8bit",
            benchmark="toy",
            latency_seconds=0.12,
            peak_gpu_memory_gb=1.5,
        )
    )
    writer.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event_type"] == EventType.METRIC.value
    assert data["mode"] == "static_8bit"
    assert data["benchmark"] == "toy"
    assert data["latency_seconds"] == 0.12
    assert data["peak_gpu_memory_gb"] == 1.5


def test_console_monitor_updates_training_fields() -> None:
    stream = io.StringIO()
    monitor = ConsoleProgressMonitor(run_id="run-1", stream=stream)

    state = monitor.handle(
        LogEvent.progress(
            run_id="run-1",
            module="router_train",
            step=10,
            epoch=2,
            loss=0.5,
            learning_rate=2e-4,
            checkpoint_path="runs/checkpoint.json",
        )
    )

    assert state.step == 10
    assert state.epoch == 2
    assert state.loss == 0.5
    assert state.learning_rate == 2e-4
    assert state.last_checkpoint_path == "runs/checkpoint.json"
    assert "step=10" in stream.getvalue()
    assert "loss=0.5" in stream.getvalue()


def test_console_monitor_updates_inference_fields_and_failures() -> None:
    monitor = ConsoleProgressMonitor(run_id="run-2", enabled=False)

    state = monitor.handle(
        LogEvent.progress(
            run_id="run-2",
            module="eval",
            mode="qaq_on_demand_on",
            benchmark="wikitext2",
            processed_examples=4,
            total_examples=10,
            latency_seconds=1.25,
            peak_gpu_memory_gb=7.5,
            routing_summary={"4": 3, "8": 1},
            loader_summary={"loads": 4},
        )
    )

    assert state.mode == "qaq_on_demand_on"
    assert state.benchmark == "wikitext2"
    assert state.processed_examples == 4
    assert state.total_examples == 10
    assert state.latency_seconds == 1.25
    assert state.peak_gpu_memory_gb == 7.5
    assert state.routing_summary == {"4": 3, "8": 1}
    assert state.loader_summary == {"loads": 4}

    state = monitor.handle(
        LogEvent.error(
            run_id="run-2",
            module="eval",
            code="controlled_failure",
            message="boom",
        )
    )

    assert state.status == RunStatus.FAILED.value
    assert state.failure == {"code": "controlled_failure", "message": "boom"}


def test_timing_measurement_is_separate_from_console_progress() -> None:
    times = iter([10.0, 10.25])
    stream = io.StringIO()
    monitor = ConsoleProgressMonitor(run_id="run-3", stream=stream)

    with TimingMeasurement("inference", clock=lambda: next(times)) as timing:
        pass

    assert timing.elapsed_seconds == 0.25
    assert stream.getvalue() == ""
    monitor.handle(
        LogEvent.progress(
            run_id="run-3",
            module="eval",
            latency_seconds=timing.elapsed_seconds,
        )
    )
    assert "latency_s=0.25" in stream.getvalue()
