"""Wireless device test automation suite covering connectivity, reconnection, latency, packet loss, and signal threshold."""

import pytest
from src.logging_utils.logger import get_test_logger
from src.models.schemas import FailureCategory, TestStatus
from src.simulator.device import WirelessDevice
from src.simulator.exceptions import (
    WirelessAssertionError,
    WirelessConfigError,
    WirelessConnectionError,
    WirelessDependencyError,
    WirelessTimeoutError,
)
from src.simulator.runner import TestRunner


@pytest.fixture
def clean_device():
    """Provides a fresh wireless device with fixed seed for reproducible behavior."""
    logger = get_test_logger("test_clean_device")
    return WirelessDevice(device_id="DUT-FIXTURE-01", seed=42, logger=logger)


class TestConnectivity:
    """Connectivity test suite."""

    def test_connectivity_nominal_success(self, clean_device):
        result = clean_device.connect(ssid="Office_WiFi_5G", auth_key="Passphrase123#")
        assert result["status"] == "CONNECTED"
        assert result["ssid"] == "Office_WiFi_5G"
        assert clean_device.ip_address.startswith("192.168.1.")
        logs = clean_device.logger.get_logs()
        assert any("Associated successfully" in log for log in logs)
        assert any("DHCP lease acquired" in log for log in logs)

    def test_connectivity_config_empty_ssid(self, clean_device):
        with pytest.raises(WirelessConfigError) as exc_info:
            clean_device.connect(ssid="", auth_key="Passphrase123#")
        assert "SSID parameter cannot be empty" in str(exc_info.value)
        logs = clean_device.logger.get_logs()
        assert any("[CONFIG]" in log for log in logs)

    def test_connectivity_config_short_password(self, clean_device):
        with pytest.raises(WirelessConfigError) as exc_info:
            clean_device.connect(ssid="Office_WiFi_5G", auth_key="short")
        assert "Authentication key must be at least 8 characters" in str(exc_info.value)

    def test_connectivity_handshake_timeout(self, clean_device):
        clean_device.set_injected_fault("timeout")
        with pytest.raises(WirelessTimeoutError) as exc_info:
            clean_device.connect(ssid="Office_WiFi_5G", auth_key="Passphrase123#", timeout_s=4.0)
        assert "handshake timed out" in str(exc_info.value)
        logs = clean_device.logger.get_logs()
        assert any("[WIFI-STACK]" in log and "Association timeout" in log for log in logs)

    def test_connectivity_radio_link_failure(self, clean_device):
        clean_device.set_injected_fault("connection")
        with pytest.raises(WirelessConnectionError) as exc_info:
            clean_device.connect(ssid="Office_WiFi_5G", auth_key="Passphrase123#")
        assert "Beacon frame loss" in str(exc_info.value)
        logs = clean_device.logger.get_logs()
        assert any("[RADIO]" in log and "Radio link failure" in log for log in logs)

    def test_connectivity_hal_driver_dependency_failure(self, clean_device):
        clean_device.set_injected_fault("dependency")
        with pytest.raises(WirelessDependencyError) as exc_info:
            clean_device.connect(ssid="Office_WiFi_5G", auth_key="Passphrase123#")
        assert "HAL driver failed" in str(exc_info.value)
        logs = clean_device.logger.get_logs()
        assert any("[HAL]" in log and "Fatal hardware driver failure" in log for log in logs)


class TestReconnection:
    """Reconnection test suite."""

    def test_reconnection_nominal_success(self, clean_device):
        clean_device.connect(ssid="IoT_Backbone", auth_key="IndustrialKey456")
        result = clean_device.reconnect(max_retries=3, backoff_s=0.01)
        assert result["reconnected"] is True
        assert clean_device.status == "CONNECTED"

    def test_reconnection_probe_timeout(self, clean_device):
        clean_device.connect(ssid="IoT_Backbone", auth_key="IndustrialKey456")
        clean_device.set_injected_fault("timeout")
        with pytest.raises(WirelessTimeoutError) as exc_info:
            clean_device.reconnect(max_retries=2, backoff_s=0.01)
        assert "exceeded maximum retry count" in str(exc_info.value)
        logs = clean_device.logger.get_logs()
        assert any("probe response timer expired" in log for log in logs)

    def test_reconnection_carrier_drop(self, clean_device):
        clean_device.connect(ssid="IoT_Backbone", auth_key="IndustrialKey456")
        clean_device.set_injected_fault("connection")
        with pytest.raises(WirelessConnectionError) as exc_info:
            clean_device.reconnect(max_retries=2, backoff_s=0.01)
        assert "continuous carrier drop" in str(exc_info.value)


class TestLatency:
    """Latency SLA test suite."""

    def test_latency_within_sla(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        res = clean_device.ping(target_ip="192.168.1.1", count=5, max_latency_sla_ms=50.0)
        assert res["latency_ms"] <= 50.0
        assert res["latency_ms"] > 0.0

    def test_latency_sla_breached(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        clean_device.set_injected_fault("assertion")
        with pytest.raises(WirelessAssertionError) as exc_info:
            clean_device.ping(target_ip="192.168.1.1", count=5, max_latency_sla_ms=50.0)
        assert "Latency SLA violation" in str(exc_info.value)

    def test_latency_ping_timeout(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        clean_device.set_injected_fault("timeout")
        with pytest.raises(WirelessTimeoutError) as exc_info:
            clean_device.ping(target_ip="192.168.1.1", count=5, timeout_s=2.0)
        assert "timed out after 2.00s" in str(exc_info.value)


class TestPacketLoss:
    """Packet loss threshold test suite."""

    def test_packet_loss_nominal(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        res = clean_device.transmit_packets(packet_count=50, max_loss_pct_threshold=2.0)
        assert res["packet_loss_pct"] <= 2.0

    def test_packet_loss_threshold_breached(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        clean_device.set_injected_fault("assertion")
        with pytest.raises(WirelessAssertionError) as exc_info:
            clean_device.transmit_packets(packet_count=50, max_loss_pct_threshold=2.0)
        assert "Packet loss threshold breach" in str(exc_info.value)

    def test_packet_loss_dma_dependency_failure(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        clean_device.set_injected_fault("dependency")
        with pytest.raises(WirelessDependencyError) as exc_info:
            clean_device.transmit_packets(packet_count=50)
        assert "DMA TX queue ring buffer overflow" in str(exc_info.value)


class TestSignalThreshold:
    """Signal strength and RF threshold test suite."""

    def test_signal_strength_nominal(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        res = clean_device.measure_signal(min_rssi_threshold_dbm=-75.0)
        assert res["rssi_dbm"] >= -75.0
        assert res["snr_db"] > 20.0

    def test_signal_strength_below_threshold(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        clean_device.set_injected_fault("assertion")
        with pytest.raises(WirelessAssertionError) as exc_info:
            clean_device.measure_signal(min_rssi_threshold_dbm=-70.0)
        assert "Signal threshold breached" in str(exc_info.value)

    def test_signal_rf_sensor_bus_dependency_failure(self, clean_device):
        clean_device.connect(ssid="Lab_WiFi", auth_key="SecureTestingKey99")
        clean_device.set_injected_fault("dependency")
        with pytest.raises(WirelessDependencyError) as exc_info:
            clean_device.measure_signal(min_rssi_threshold_dbm=-70.0)
        assert "HAL RF sensor bus error" in str(exc_info.value)


class TestSuiteExecutionRunner:
    """Verifies that the automated test runner executes the full suite generating all 5 failure categories."""

    def test_full_suite_runner_generates_all_failure_categories(self):
        runner = TestRunner(seed=42)
        summary = runner.run_suite("wireless_test_suite")

        assert summary.total_tests == 14
        assert summary.passed_count == 5  # nominal connectivity, reconn, latency, packet_loss, signal
        assert summary.failed_count == 9  # injected faults

        # Verify all 5 categories are present among failed tests
        categories_found = {r.failure_category for r in summary.results if r.status == TestStatus.FAILED}
        expected_categories = {
            FailureCategory.TIMEOUT,
            FailureCategory.ASSERTION,
            FailureCategory.CONFIGURATION,
            FailureCategory.CONNECTION,
            FailureCategory.DEPENDENCY,
        }
        assert expected_categories == categories_found, f"Expected {expected_categories}, got {categories_found}"

        # Verify logs are populated for each test
        for res in summary.results:
            assert len(res.raw_logs) > 0
            assert res.duration_ms >= 0.0
