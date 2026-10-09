"""Custom exceptions for simulated wireless device testing."""

from src.models.schemas import FailureCategory


class WirelessError(Exception):
    """Base exception for wireless device test errors."""
    category: FailureCategory = FailureCategory.NONE

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class WirelessTimeoutError(WirelessError):
    """Raised when an operation times out (e.g. handshake, probe response, keep-alive ACK)."""
    category = FailureCategory.TIMEOUT


class WirelessAssertionError(WirelessError):
    """Raised when device telemetry breaches service level agreements (e.g. latency, packet loss)."""
    category = FailureCategory.ASSERTION


class WirelessConfigError(WirelessError):
    """Raised when configuration parameters are invalid or mismatched (SSID, security, frequency band)."""
    category = FailureCategory.CONFIGURATION


class WirelessConnectionError(WirelessError):
    """Raised when the physical or link-layer connection fails or drops unexpectedly."""
    category = FailureCategory.CONNECTION


class WirelessDependencyError(WirelessError):
    """Raised when an external dependency or low-level subsystem is unavailable (HAL, DHCP, kernel driver)."""
    category = FailureCategory.DEPENDENCY
