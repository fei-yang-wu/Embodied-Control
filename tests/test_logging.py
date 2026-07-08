"""The configurable base logger: human log format, structured JSONL events,
child-logger hierarchy/propagation, and the null-logger default (default env;
no mujoco/docker needed)."""

from __future__ import annotations

import json

from embodied_control.logging.config import LogConfig
from embodied_control.logging.logger import EcLogger


def test_create_writes_human_and_jsonl_logs(tmp_path):
    logger = EcLogger.create(
        LogConfig(level="INFO", log_dir=str(tmp_path)), run_id="run1"
    )
    logger.info("hello", foo="bar")
    logger.event("thing.happened", phase="init", severity="warning", count=3)
    logger.close()

    human = (tmp_path / "orchestrator.log").read_text()
    assert "hello" in human and "foo=bar" in human
    assert "thing.happened" in human and "phase=init" in human and "count=3" in human

    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run1"
    assert rows[0]["message"] == "hello"
    assert rows[0]["foo"] == "bar"
    assert rows[1]["event"] == "thing.happened"
    assert rows[1]["phase"] == "init"
    assert rows[1]["level"] == "WARNING"
    assert rows[1]["count"] == 3


def test_child_logger_propagates_to_root_handlers(tmp_path):
    logger = EcLogger.create(LogConfig(log_dir=str(tmp_path)), run_id="run2")
    child = logger.child("sim")
    grandchild = child.child("render")

    child.info("from child")
    grandchild.warning("from grandchild")
    logger.close()

    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    loggers = [r["logger"] for r in rows]
    assert any(l.endswith(".sim") for l in loggers)
    assert any(l.endswith(".sim.render") for l in loggers)
    # both rows still carry the same run_id -- children share the root's identity
    assert all(r["run_id"] == "run2" for r in rows)


def test_level_filters_debug_by_default(tmp_path):
    logger = EcLogger.create(LogConfig(level="INFO", log_dir=str(tmp_path)), run_id="run3")
    logger.debug("should not appear")
    logger.info("should appear")
    logger.close()

    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    messages = [r["message"] for r in rows]
    assert "should appear" in messages
    assert "should not appear" not in messages


def test_debug_level_captures_debug_events(tmp_path):
    logger = EcLogger.create(LogConfig(level="DEBUG", log_dir=str(tmp_path)), run_id="run4")
    logger.debug("verbose detail")
    logger.close()

    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows[0]["message"] == "verbose detail"
    assert rows[0]["level"] == "DEBUG"


def test_null_logger_never_raises_and_writes_nothing(tmp_path):
    logger = EcLogger.null()
    # None of these should raise or touch the filesystem.
    logger.debug("x")
    logger.info("y", a=1)
    logger.warning("z")
    logger.error("w")
    logger.event("noop.event", phase="p")
    child = logger.child("sub")
    child.info("still silent")
    assert list(tmp_path.iterdir()) == []


def test_two_runs_do_not_share_or_duplicate_handlers(tmp_path):
    """Distinct run_ids must not leak handlers/files into each other (regression
    guard for the logger-name-collision class of bug the planner's run_id
    dedupe also guards against)."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    logger_a = EcLogger.create(LogConfig(log_dir=str(dir_a)), run_id="run_a")
    logger_b = EcLogger.create(LogConfig(log_dir=str(dir_b)), run_id="run_b")

    logger_a.info("only in a")
    logger_b.info("only in b")
    logger_a.close()
    logger_b.close()

    rows_a = [json.loads(l) for l in (dir_a / "events.jsonl").read_text().splitlines()]
    rows_b = [json.loads(l) for l in (dir_b / "events.jsonl").read_text().splitlines()]
    assert len(rows_a) == 1 and rows_a[0]["message"] == "only in a"
    assert len(rows_b) == 1 and rows_b[0]["message"] == "only in b"


def test_create_reused_run_id_does_not_duplicate_handlers(tmp_path):
    """If a run_id were ever reused within a process, create() must not pile up
    duplicate handlers (which would duplicate every log line written after)."""
    logger1 = EcLogger.create(LogConfig(log_dir=str(tmp_path)), run_id="dup")
    logger1.info("first")
    logger1.close()

    logger2 = EcLogger.create(LogConfig(log_dir=str(tmp_path)), run_id="dup")
    logger2.info("second")
    logger2.close()

    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    # Exactly one row per info() call -- not two-for-one from a duplicated handler.
    assert [r["message"] for r in rows] == ["first", "second"]
