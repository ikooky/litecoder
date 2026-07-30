"""Windows inline terminal driver."""

from __future__ import annotations

import asyncio

from textual import events
from textual.drivers import win32
from textual.drivers._writer_thread import WriterThread
from textual.drivers.windows_driver import WindowsDriver


class WindowsInlineDriver(WindowsDriver):
    """Windows Textual driver that renders in the normal terminal buffer."""

    @property
    def is_inline(self) -> bool:
        """Return whether the inline condition holds."""
        return True

    def start_application_mode(self) -> None:
        """Start the application mode."""
        loop = asyncio.get_running_loop()

        self._restore_console = win32.enable_application_mode()
        self._writer_thread = WriterThread(self._file)
        self._writer_thread.start()

        self._enable_mouse_support()
        self.write("\x1b[?25l")
        self.write("\x1b[?1004h")
        self.write("\x1b[>1u")
        self._enable_bracketed_paste()
        self.write("\n" * self._app.INLINE_PADDING)
        self.flush()

        def process_event(event: events.Event) -> None:
            if isinstance(event, events.CursorPosition):
                self.cursor_origin = (event.x, event.y)
                return
            self.process_message(event)

        self._event_thread = win32.EventMonitor(
            loop,
            self._app,
            self.exit_event,
            process_event,
        )
        self._event_thread.start()

    def stop_application_mode(self) -> None:
        """Stop the application mode."""
        self._disable_bracketed_paste()
        self.disable_input()
        self.write("\x1b[<u")
        self.write("\x1b[?25h")
        self.write("\x1b[?1004l")
        self.write("\x1b[J")
        self.flush()
