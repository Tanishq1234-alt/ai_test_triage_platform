"""Specialized test logging engine producing realistic wireless hardware and stack logs."""

import datetime
import io
import logging
import os
from typing import List, Optional


class TestLogger:
    """Manages scoped log collection for test runs and hardware simulation."""

    def __init__(self, test_name: str = "general", log_file: Optional[str] = None):
        self.test_name = test_name
        self.log_file = log_file
        self.memory_buffer: List[str] = []
        self._logger = logging.getLogger(f"wireless.{test_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # Clear existing handlers to prevent duplicate messages across re-instantiations
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        # In-memory stream handler
        self._stream = io.StringIO()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(subsystem)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(self._stream)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _log(self, level: int, subsystem: str, message: str, **kwargs):
        extra = {"subsystem": subsystem}
        extra.update(kwargs)
        self._logger.log(level, message, extra=extra)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = logging.getLevelName(level)
        formatted_entry = f"{now_str} [{level_name}] [{subsystem}] {message}"
        self.memory_buffer.append(formatted_entry)

    def debug(self, subsystem: str, message: str):
        self._log(logging.DEBUG, subsystem, message)

    def info(self, subsystem: str, message: str):
        self._log(logging.INFO, subsystem, message)

    def warning(self, subsystem: str, message: str):
        self._log(logging.WARNING, subsystem, message)

    def error(self, subsystem: str, message: str):
        self._log(logging.ERROR, subsystem, message)

    def critical(self, subsystem: str, message: str):
        self._log(logging.CRITICAL, subsystem, message)

    def get_logs(self) -> List[str]:
        """Returns the in-memory log list for the current test scope."""
        return list(self.memory_buffer)

    def clear(self):
        """Clears accumulated in-memory logs."""
        self.memory_buffer.clear()
        self._stream.truncate(0)
        self._stream.seek(0)


def get_test_logger(test_name: str, log_file: Optional[str] = None) -> TestLogger:
    """Factory helper to obtain a test logger instance."""
    return TestLogger(test_name=test_name, log_file=log_file)
