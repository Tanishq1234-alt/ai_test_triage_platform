"""Unit tests for the WirelessDevice simulator."""

import pytest
from src.logging_utils.logger import get_test_logger
from src.simulator.device import WirelessDevice
from src.simulator.exceptions import (
    WirelessAssertionError,
    WirelessConfigError,
    WirelessConnectionError,
    WirelessDependencyError,
    WirelessTimeoutError,
)


def test_device_initialization():
    dev = WirelessDevice(device_id="TEST-DEV", mac_address="AA:BB:CC:DD:EE:FF", firmware_version="v1.0.0")
    assert dev.device_id == "TEST-DEV"
    assert dev.status == "DISCONNECTED"
    assert dev.mac_address == "AA:BB:CC:DD:EE:FF"
    assert dev.firmware_version == "v1.0.0"


def test_device_reproducibility_with_seed():
    dev1 = WirelessDevice(seed=12345)
    res1 = dev1.connect("SSID_A", "AuthKey1234")

    dev2 = WirelessDevice(seed=12345)
    res2 = dev2.connect("SSID_A", "AuthKey1234")

    assert res1["ip_address"] == res2["ip_address"]
    assert res1["rssi_dbm"] == res2["rssi_dbm"]
    assert res1["snr_db"] == res2["snr_db"]


def test_device_disconnect():
    dev = WirelessDevice(seed=42)
    dev.connect("SSID_B", "AuthKey1234")
    assert dev.status == "CONNECTED"
    dev.disconnect()
    assert dev.status == "DISCONNECTED"
    assert dev.current_ssid is None


def test_ping_when_disconnected_raises_connection_error():
    dev = WirelessDevice(seed=42)
    with pytest.raises(WirelessConnectionError):
        dev.ping(target_ip="8.8.8.8")


def test_transmit_when_disconnected_raises_connection_error():
    dev = WirelessDevice(seed=42)
    with pytest.raises(WirelessConnectionError):
        dev.transmit_packets(packet_count=20)


def test_clear_fault_injection():
    dev = WirelessDevice(seed=42)
    dev.set_injected_fault("timeout")
    dev.clear_injected_fault()
    res = dev.connect("SSID_C", "AuthKey1234")
    assert res["status"] == "CONNECTED"
