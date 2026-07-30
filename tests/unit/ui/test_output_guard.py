from __future__ import annotations

from io import StringIO
import logging
import sys

import pytest

from litecoder.ui.output_guard import TUIOutputGuard


def test_guard_discards_stdout_stderr_and_existing_logging_handlers(
    monkeypatch,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    logger = logging.getLogger("litecoder-test-existing-output")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stderr)
    logger.handlers = [handler]

    with TUIOutputGuard():
        print("hidden stdout")
        print("hidden stderr", file=sys.stderr)
        logger.error("hidden log")

    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert handler.stream is stderr
    logger.handlers.clear()


def test_guard_restores_handlers_created_while_active(monkeypatch) -> None:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    logger = logging.getLogger("litecoder-test-new-output")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    with TUIOutputGuard():
        handler = logging.StreamHandler()
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.addHandler(stdout_handler)
        logger.error("hidden during tui")

    assert stderr.getvalue() == ""
    assert handler.stream is stderr
    assert stdout_handler.stream is stdout
    logger.error("visible after tui")
    assert "visible after tui" in stderr.getvalue()
    logger.handlers.clear()


def test_guard_restores_process_output_after_exception(monkeypatch) -> None:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(RuntimeError, match="boom"):
        with TUIOutputGuard():
            raise RuntimeError("boom")

    assert sys.stdout is stdout
    assert sys.stderr is stderr


def test_guard_does_not_redirect_non_terminal_stream_handlers(monkeypatch) -> None:
    stdout = StringIO()
    stderr = StringIO()
    diagnostic = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    logger = logging.getLogger("litecoder-test-diagnostic-output")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(diagnostic)
    logger.handlers = [handler]

    with TUIOutputGuard():
        logger.error("kept diagnostic")

    assert "kept diagnostic" in diagnostic.getvalue()
    assert handler.stream is diagnostic
    logger.handlers.clear()


def test_guard_supports_dynamic_read_only_stderr_handler(monkeypatch) -> None:
    class DynamicStderrHandler(logging.StreamHandler):
        def __init__(self) -> None:
            logging.Handler.__init__(self)

        @property
        def stream(self):
            return sys.stderr

    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    logger = logging.getLogger("litecoder-test-read-only-stderr")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = DynamicStderrHandler()
    logger.handlers = [handler]

    with TUIOutputGuard():
        logger.error("hidden dynamic log")

    assert stderr.getvalue() == ""
    assert handler.stream is stderr
    logger.error("visible dynamic log")
    assert "visible dynamic log" in stderr.getvalue()
    logger.handlers.clear()
