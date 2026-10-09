"""Simulated Wireless Device with realistic telemetry, state machine, and fault injection."""

import random
import time
from typing import Dict, Optional

from src.logging_utils.logger import TestLogger, get_test_logger
from src.simulator.exceptions import (
    WirelessAssertionError,
    WirelessConfigError,
    WirelessConnectionError,
    WirelessDependencyError,
    WirelessTimeoutError,
)


class WirelessDevice:
    """Simulates an embedded 802.11ax (Wi-Fi 6) wireless device under test (DUT)."""

    def __init__(
        self,
        device_id: str = "DEV-WIFI6-01",
        mac_address: str = "00:1A:2B:3C:4D:5E",
        firmware_version: str = "v2.4.12-rc3",
        seed: Optional[int] = 42,
        logger: Optional[TestLogger] = None,
    ):
        self.device_id = device_id
        self.mac_address = mac_address
        self.firmware_version = firmware_version
        self.rng = random.Random(seed if seed is not None else 42)
        self.logger = logger or get_test_logger(test_name=device_id)

        # Operational state
        self.status = "DISCONNECTED"
        self.current_ssid: Optional[str] = None
        self.current_bssid: Optional[str] = None
        self.ip_address: Optional[str] = None
        self.channel: int = 36
        self.band: str = "5GHz"

        # Telemetry metrics
        self.rssi_dbm: float = -55.0
        self.snr_db: float = 38.0
        self.latency_ms: float = 16.5
        self.packet_loss_pct: float = 0.0

        # Fault injection hook
        self._injected_fault: Optional[str] = None
        self._fault_details: Dict[str, any] = {}

    def set_injected_fault(self, fault_type: Optional[str], details: Optional[Dict[str, any]] = None):
        """Inject a specific fault category: 'timeout', 'assertion', 'configuration', 'connection', 'dependency', or None."""
        self._injected_fault = fault_type
        self._fault_details = details or {}

    def clear_injected_fault(self):
        """Remove any active fault injection."""
        self._injected_fault = None
        self._fault_details = {}

    def connect(
        self,
        ssid: str,
        auth_key: str,
        band: str = "5GHz",
        channel: int = 36,
        timeout_s: float = 5.0,
    ) -> Dict[str, any]:
        """Attempt association and 4-way handshake with target Wi-Fi Access Point."""
        self.logger.info("RADIO", f"Initiating Wi-Fi association to SSID '{ssid}' on band {band} ch {channel}")
        self.status = "CONNECTING"

        # Check configuration faults
        if self._injected_fault == "configuration":
            err_msg = self._fault_details.get(
                "message",
                f"Configuration reject: Invalid WPA3 security key or mismatched cipher suite for SSID '{ssid}'"
            )
            self.status = "ERROR"
            self.logger.error("CONFIG", f"WPA3-SAE negotiation rejected: {err_msg}")
            raise WirelessConfigError(err_msg, details={"ssid": ssid, "auth_key_len": len(auth_key)})

        if not ssid or len(ssid.strip()) == 0:
            self.status = "ERROR"
            self.logger.error("CONFIG", "Empty SSID provided in test profile")
            raise WirelessConfigError("SSID parameter cannot be empty", details={"ssid": ssid})

        if not auth_key or len(auth_key) < 8:
            self.status = "ERROR"
            self.logger.error("CONFIG", f"Authentication key length {len(auth_key)} is below 8-character WPA2/WPA3 requirement")
            raise WirelessConfigError("Authentication key must be at least 8 characters", details={"auth_key_len": len(auth_key)})

        # Check dependency faults (HAL driver, DHCP service)
        if self._injected_fault == "dependency":
            subsystem = self._fault_details.get("subsystem", "HAL")
            err_msg = self._fault_details.get(
                "message",
                "Firmware HAL driver failed to respond: err_code=0xDEAD8002 kernel ring buffer corrupted"
            )
            self.status = "ERROR"
            self.logger.critical(subsystem, f"Fatal hardware driver failure: {err_msg}")
            raise WirelessDependencyError(err_msg, details={"subsystem": subsystem})

        # Check connection faults (Physical radio drop, beacon lost)
        if self._injected_fault == "connection":
            err_msg = self._fault_details.get(
                "message",
                f"Beacon frame loss detected: AP beacon timeout on BSSID 00:11:22:33:44:55 for SSID '{ssid}'"
            )
            self.status = "ERROR"
            self.logger.error("RADIO", f"Radio link failure: {err_msg}")
            raise WirelessConnectionError(err_msg, details={"ssid": ssid, "channel": channel})

        # Check timeout faults
        if self._injected_fault == "timeout":
            elapsed = self._fault_details.get("elapsed", timeout_s + 0.5)
            err_msg = self._fault_details.get(
                "message",
                f"Connection handshake timed out after {elapsed:.2f}s waiting for EAPOL M2 frame"
            )
            self.status = "ERROR"
            self.logger.error("WIFI-STACK", f"Association timeout: {err_msg}")
            raise WirelessTimeoutError(err_msg, details={"timeout_s": timeout_s, "elapsed": elapsed})

        # Successful nominal connection
        self.status = "CONNECTED"
        self.current_ssid = ssid
        self.current_bssid = "00:11:22:33:44:55"
        self.band = band
        self.channel = channel
        self.ip_address = f"192.168.1.{self.rng.randint(100, 200)}"
        self.rssi_dbm = round(-52.0 - self.rng.uniform(0.0, 10.0), 1)
        self.snr_db = round(35.0 + self.rng.uniform(0.0, 8.0), 1)

        self.logger.info("RADIO", f"Associated successfully with BSSID {self.current_bssid} RSSI={self.rssi_dbm} dBm")
        self.logger.info("DHCP", f"DHCP lease acquired: IP {self.ip_address} subnet 255.255.255.0 gw 192.168.1.1")
        return {
            "status": self.status,
            "ssid": self.current_ssid,
            "bssid": self.current_bssid,
            "ip_address": self.ip_address,
            "rssi_dbm": self.rssi_dbm,
            "snr_db": self.snr_db,
        }

    def reconnect(self, max_retries: int = 3, backoff_s: float = 0.05) -> Dict[str, any]:
        """Simulate a reconnection sequence after link drop."""
        self.logger.info("WIFI-STACK", f"Executing link reconnect procedure with max_retries={max_retries}")
        self.status = "RECONNECTING"

        for attempt in range(1, max_retries + 1):
            self.logger.info("WIFI-STACK", f"Reconnection attempt {attempt}/{max_retries}...")
            time.sleep(backoff_s)

            if self._injected_fault == "timeout":
                self.logger.warning("RADIO", f"Attempt {attempt}: probe response timer expired (timeout)")
                if attempt == max_retries:
                    self.status = "ERROR"
                    err_msg = f"Reconnection procedure exceeded maximum retry count ({max_retries}) with probe timeout"
                    self.logger.error("WIFI-STACK", err_msg)
                    raise WirelessTimeoutError(err_msg, details={"retries": attempt})

            elif self._injected_fault == "connection":
                self.logger.warning("RADIO", f"Attempt {attempt}: carrier wave degraded, SNR below lock threshold")
                if attempt == max_retries:
                    self.status = "ERROR"
                    err_msg = f"Reconnection aborted: continuous carrier drop across {max_retries} attempts"
                    self.logger.error("RADIO", err_msg)
                    raise WirelessConnectionError(err_msg, details={"retries": attempt})

            elif self._injected_fault == "dependency":
                err_msg = "Reconnection failed: driver kernel lockup detected during interface reset"
                self.status = "ERROR"
                self.logger.critical("HAL", err_msg)
                raise WirelessDependencyError(err_msg, details={"attempt": attempt})

            else:
                # Nominal reconnection succeeds on 1st or 2nd try
                self.status = "CONNECTED"
                self.logger.info("RADIO", f"Reconnection established on attempt {attempt}")
                return {"reconnected": True, "attempts": attempt, "status": self.status}

        return {"reconnected": True, "attempts": 1, "status": self.status}

    def measure_signal(self, min_rssi_threshold_dbm: Optional[float] = None) -> Dict[str, float]:
        """Measure current RSSI and SNR. Asserts threshold if specified."""
        self.logger.info("RADIO", "Sampling RF receiver signal metrics...")

        if self._injected_fault == "dependency":
            err_msg = "HAL RF sensor bus error: I2C telemetry register read returned 0xFF (device unresponsive)"
            self.logger.critical("HAL", err_msg)
            raise WirelessDependencyError(err_msg)

        if self._injected_fault == "connection":
            self.rssi_dbm = -98.0
            self.snr_db = 4.0
            self.logger.error("RADIO", f"Radio signal critically degraded: RSSI={self.rssi_dbm} dBm SNR={self.snr_db} dB")
            raise WirelessConnectionError(f"Radio link lost: RSSI {self.rssi_dbm} dBm below physical sensitivity limit")

        if self._injected_fault == "assertion":
            self.rssi_dbm = -88.5
            self.snr_db = 11.0
            self.logger.warning("RADIO", f"Weak RF signal measured: RSSI={self.rssi_dbm} dBm SNR={self.snr_db} dB")
        else:
            self.rssi_dbm = round(-58.0 + self.rng.uniform(-3.0, 3.0), 1)
            self.snr_db = round(34.0 + self.rng.uniform(-2.0, 2.0), 1)

        self.logger.info("METRIC", f"telemetry: rssi_dbm={self.rssi_dbm} snr_db={self.snr_db}")

        if min_rssi_threshold_dbm is not None and self.rssi_dbm < min_rssi_threshold_dbm:
            err_msg = f"Signal threshold breached: measured RSSI {self.rssi_dbm} dBm < minimum required {min_rssi_threshold_dbm} dBm"
            self.logger.error("TEST", err_msg)
            raise WirelessAssertionError(err_msg, details={"measured": self.rssi_dbm, "threshold": min_rssi_threshold_dbm})

        return {"rssi_dbm": self.rssi_dbm, "snr_db": self.snr_db}

    def ping(
        self,
        target_ip: str = "8.8.8.8",
        count: int = 5,
        max_latency_sla_ms: Optional[float] = None,
        timeout_s: float = 2.0,
    ) -> Dict[str, float]:
        """Perform ICMP ping latency measurement."""
        self.logger.info("STACK", f"Executing ICMP ping to {target_ip} with count={count}")

        if self.status != "CONNECTED":
            err_msg = f"Cannot ping {target_ip}: Device is not connected (status={self.status})"
            self.logger.error("WIFI-STACK", err_msg)
            raise WirelessConnectionError(err_msg)

        if self._injected_fault == "timeout":
            err_msg = f"ICMP echo request timed out after {timeout_s:.2f}s waiting for reply from {target_ip}"
            self.logger.error("STACK", err_msg)
            raise WirelessTimeoutError(err_msg, details={"target": target_ip, "timeout_s": timeout_s})

        if self._injected_fault == "assertion":
            self.latency_ms = round(165.0 + self.rng.uniform(20.0, 80.0), 2)
            self.logger.warning("STACK", f"High round-trip delay observed: {self.latency_ms} ms")
        else:
            self.latency_ms = round(14.0 + self.rng.uniform(2.0, 8.0), 2)

        self.logger.info("METRIC", f"telemetry: latency_ms={self.latency_ms} target={target_ip}")

        if max_latency_sla_ms is not None and self.latency_ms > max_latency_sla_ms:
            err_msg = f"Latency SLA violation: measured RTT {self.latency_ms} ms exceeds maximum allowed {max_latency_sla_ms} ms"
            self.logger.error("TEST", err_msg)
            raise WirelessAssertionError(err_msg, details={"measured": self.latency_ms, "sla": max_latency_sla_ms})

        return {"latency_ms": self.latency_ms, "target_ip": target_ip, "packets_sent": count}

    def transmit_packets(
        self,
        packet_count: int = 100,
        max_loss_pct_threshold: Optional[float] = 2.0,
    ) -> Dict[str, float]:
        """Transmit a burst of packets and evaluate packet drop rate."""
        self.logger.info("RADIO", f"Transmitting {packet_count} UDP throughput test packets...")

        if self.status != "CONNECTED":
            err_msg = "Packet transmission aborted: Radio interface not in CONNECTED state"
            self.logger.error("RADIO", err_msg)
            raise WirelessConnectionError(err_msg)

        if self._injected_fault == "dependency":
            err_msg = "DMA TX queue ring buffer overflow: host controller interface (HCI) dropped packets"
            self.logger.critical("HAL", err_msg)
            raise WirelessDependencyError(err_msg)

        if self._injected_fault == "assertion":
            self.packet_loss_pct = round(12.5 + self.rng.uniform(2.0, 15.0), 2)
            self.logger.warning("RADIO", f"Elevated frame retries detected. Packet loss rate: {self.packet_loss_pct}%")
        else:
            self.packet_loss_pct = round(self.rng.uniform(0.0, 0.8), 2)

        self.logger.info("METRIC", f"telemetry: packet_loss_pct={self.packet_loss_pct} transmitted={packet_count}")

        if max_loss_pct_threshold is not None and self.packet_loss_pct > max_loss_pct_threshold:
            err_msg = f"Packet loss threshold breach: {self.packet_loss_pct}% dropped packets exceeds limit {max_loss_pct_threshold}%"
            self.logger.error("TEST", err_msg)
            raise WirelessAssertionError(err_msg, details={"packet_loss_pct": self.packet_loss_pct, "threshold": max_loss_pct_threshold})

        return {"packet_loss_pct": self.packet_loss_pct, "packet_count": packet_count}

    def disconnect(self):
        """Cleanly disconnect wireless interface."""
        self.logger.info("RADIO", f"Disassociating from SSID '{self.current_ssid}'")
        self.status = "DISCONNECTED"
        self.current_ssid = None
        self.current_bssid = None
        self.ip_address = None
