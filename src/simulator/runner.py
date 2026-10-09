"""Automated test execution runner for wireless device test suites."""

import datetime
import time
import uuid
from typing import Callable, Dict, List, Optional

from src.logging_utils.logger import TestLogger, get_test_logger
from src.models.schemas import FailureCategory, TestCaseResult, TestSessionSummary, TestStatus
from src.parser.log_parser import LogParser
from src.simulator.device import WirelessDevice
from src.simulator.exceptions import (
    WirelessAssertionError,
    WirelessConfigError,
    WirelessConnectionError,
    WirelessDependencyError,
    WirelessTimeoutError,
)


class TestRunner:
    """Executes a suite of wireless test cases, captures logs and telemetry, and outputs structured session results."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.parser = LogParser()

    def run_single_test(
        self,
        test_name: str,
        test_fn: Callable[[WirelessDevice, TestLogger], None],
        fault_type: Optional[str] = None,
        fault_details: Optional[Dict[str, any]] = None,
        test_id: Optional[str] = None,
    ) -> TestCaseResult:
        """Execute a single test function with an isolated logger and device."""
        t_id = test_id or f"TC-{uuid.uuid4().hex[:8]}"
        logger = get_test_logger(test_name=test_name)
        device = WirelessDevice(device_id=f"DUT-{test_name}", seed=self.seed, logger=logger)

        if fault_type:
            device.set_injected_fault(fault_type, fault_details)

        start_time = time.perf_counter()
        status = TestStatus.PASSED
        category = FailureCategory.NONE
        error_type: Optional[str] = None
        error_msg: Optional[str] = None
        stack_trace: Optional[str] = None

        logger.info("TEST", f"Starting test case: {test_name} (injected_fault={fault_type})")

        try:
            test_fn(device, logger)
            logger.info("TEST", f"Test case {test_name} PASSED successfully.")
        except WirelessTimeoutError as e:
            status = TestStatus.FAILED
            category = FailureCategory.TIMEOUT
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = f"{type(e).__name__}: {e}\n  at wireless_device_test"
            logger.error("TEST", f"Test FAILED with timeout: {error_msg}")
        except WirelessAssertionError as e:
            status = TestStatus.FAILED
            category = FailureCategory.ASSERTION
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = f"{type(e).__name__}: {e}\n  at wireless_device_test"
            logger.error("TEST", f"Test FAILED with assertion breach: {error_msg}")
        except WirelessConfigError as e:
            status = TestStatus.FAILED
            category = FailureCategory.CONFIGURATION
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = f"{type(e).__name__}: {e}\n  at wireless_device_test"
            logger.error("TEST", f"Test FAILED with configuration error: {error_msg}")
        except WirelessConnectionError as e:
            status = TestStatus.FAILED
            category = FailureCategory.CONNECTION
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = f"{type(e).__name__}: {e}\n  at wireless_device_test"
            logger.error("TEST", f"Test FAILED with connection error: {error_msg}")
        except WirelessDependencyError as e:
            status = TestStatus.FAILED
            category = FailureCategory.DEPENDENCY
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = f"{type(e).__name__}: {e}\n  at wireless_device_test"
            logger.error("TEST", f"Test FAILED with dependency error: {error_msg}")
        except Exception as e:
            status = TestStatus.ERROR
            category = FailureCategory.NONE
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = f"Unhandled {type(e).__name__}: {e}"
            logger.critical("TEST", f"Test ERROR: {error_msg}")
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

        logs = logger.get_logs()
        result = self.parser.parse_test_result(
            test_id=t_id,
            test_name=test_name,
            status=status.value,
            duration_ms=elapsed_ms,
            raw_logs=logs,
            stack_trace=stack_trace,
            explicit_category=category if status == TestStatus.FAILED else FailureCategory.NONE,
            explicit_error_type=error_type,
            explicit_error_message=error_msg,
        )
        return result

    def get_standard_test_definitions(self) -> List[Dict[str, any]]:
        """Returns standard wireless test definitions with passing tests and 5 failure categories."""
        def case_conn_pass(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            assert d.status == "CONNECTED"

        def case_conn_cfg_fail(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="", auth_key="short")

        def case_conn_timeout_fail(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123", timeout_s=3.0)

        def case_conn_link_fail(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")

        def case_conn_hal_fail(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")

        def case_reconn_pass(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            res = d.reconnect(max_retries=3, backoff_s=0.01)
            assert res["reconnected"] is True

        def case_reconn_timeout(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            d.reconnect(max_retries=2, backoff_s=0.01)

        def case_latency_pass(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            res = d.ping(target_ip="8.8.8.8", count=5, max_latency_sla_ms=50.0)
            assert res["latency_ms"] <= 50.0

        def case_latency_sla_breach(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            d.ping(target_ip="8.8.8.8", count=5, max_latency_sla_ms=40.0)

        def case_packet_loss_pass(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            res = d.transmit_packets(packet_count=50, max_loss_pct_threshold=3.0)
            assert res["packet_loss_pct"] <= 3.0

        def case_packet_loss_breach(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            d.transmit_packets(packet_count=50, max_loss_pct_threshold=1.5)

        def case_signal_pass(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            res = d.measure_signal(min_rssi_threshold_dbm=-75.0)
            assert res["rssi_dbm"] >= -75.0

        def case_signal_weak_breach(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            d.measure_signal(min_rssi_threshold_dbm=-70.0)

        def case_signal_rf_dep_fail(d: WirelessDevice, l: TestLogger):
            d.connect(ssid="IoT_Lab_5G", auth_key="SuperSecurePass123")
            d.measure_signal(min_rssi_threshold_dbm=-70.0)

        return [
            {"name": "test_connectivity_nominal_success", "fn": case_conn_pass, "fault": None},
            {"name": "test_connectivity_config_invalid", "fn": case_conn_cfg_fail, "fault": "configuration"},
            {"name": "test_connectivity_handshake_timeout", "fn": case_conn_timeout_fail, "fault": "timeout"},
            {"name": "test_connectivity_radio_link_failure", "fn": case_conn_link_fail, "fault": "connection"},
            {"name": "test_connectivity_hal_driver_dependency_failure", "fn": case_conn_hal_fail, "fault": "dependency"},
            {"name": "test_reconnection_nominal_success", "fn": case_reconn_pass, "fault": None},
            {"name": "test_reconnection_probe_timeout", "fn": case_reconn_timeout, "fault": "timeout"},
            {"name": "test_latency_nominal_within_sla", "fn": case_latency_pass, "fault": None},
            {"name": "test_latency_sla_exceeded", "fn": case_latency_sla_breach, "fault": "assertion"},
            {"name": "test_packet_loss_nominal", "fn": case_packet_loss_pass, "fault": None},
            {"name": "test_packet_loss_threshold_breached", "fn": case_packet_loss_breach, "fault": "assertion"},
            {"name": "test_signal_strength_nominal", "fn": case_signal_pass, "fault": None},
            {"name": "test_signal_strength_below_threshold", "fn": case_signal_weak_breach, "fault": "assertion"},
            {"name": "test_signal_rf_sensor_bus_dependency_failure", "fn": case_signal_rf_dep_fail, "fault": "dependency"},
        ]

    def run_suite(self, suite_name: str = "wireless_full_suite") -> TestSessionSummary:
        """Run all test cases in the standard wireless suite and return a session summary."""
        definitions = self.get_standard_test_definitions()
        results: List[TestCaseResult] = []
        start_session = time.perf_counter()

        for idx, item in enumerate(definitions, start=1):
            res = self.run_single_test(
                test_name=item["name"],
                test_fn=item["fn"],
                fault_type=item["fault"],
                test_id=f"TEST-{idx:03d}",
            )
            results.append(res)

        total_duration = time.perf_counter() - start_session
        passed_count = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_count = sum(1 for r in results if r.status in [TestStatus.FAILED, TestStatus.ERROR])

        return TestSessionSummary(
            session_id=f"SESSION-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            duration_seconds=round(total_duration, 3),
            results=results,
        )
