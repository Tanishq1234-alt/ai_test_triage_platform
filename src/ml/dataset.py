"""Dataset preparation, label normalization, and synthetic generation for ML triage."""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical failure categories required for Phase 3
CANONICAL_CATEGORIES = [
    "timeout",
    "assertion_failure",
    "connection_error",
    "configuration_error",
    "dependency_error",
    "unknown",
]

CATEGORY_NORMALIZATION_MAP = {
    "timeout": "timeout",
    "timeoutexception": "timeout",
    "assertion": "assertion_failure",
    "assertion_failure": "assertion_failure",
    "assert": "assertion_failure",
    "connection": "connection_error",
    "connection_error": "connection_error",
    "configuration": "configuration_error",
    "configuration_error": "configuration_error",
    "config": "configuration_error",
    "dependency": "dependency_error",
    "dependency_error": "dependency_error",
    "unknown": "unknown",
}


def normalize_failure_category(raw_category: Optional[str]) -> str:
    """Normalize Phase 2 categories into canonical Phase 3 categories."""
    if not raw_category:
        return "unknown"
    cat_lower = str(raw_category).strip().lower()
    return CATEGORY_NORMALIZATION_MAP.get(cat_lower, "unknown")


class DatasetLoader:
    """Loads, cleans, and validates failure datasets from Phase 2 normalized records."""

    def __init__(self, min_samples_per_class: int = 5):
        self.min_samples_per_class = min_samples_per_class

    def load_from_json(self, file_path: str) -> pd.DataFrame:
        """Load records from Phase 2 normalized JSON export."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list of records in {file_path}")

        return self.process_records(data, source_name=file_path)

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load records from Phase 2 normalized CSV export."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        return self.process_records(records, source_name=file_path)

    def process_records(self, records: List[Dict[str, Any]], source_name: str = "memory") -> pd.DataFrame:
        """Filter out non-failures, deduplicate, clean missing fields, and normalize labels."""
        filtered: List[Dict[str, Any]] = []

        for r in records:
            status = str(r.get("status", "")).lower().strip()
            # Exclude passed and skipped tests from failure classifier
            if status in ["passed", "skipped"]:
                continue

            raw_cat = r.get("failure_category")
            canonical_cat = normalize_failure_category(raw_cat)

            rec_clean = {
                "test_id": r.get("test_id", ""),
                "test_name": r.get("test_name", ""),
                "test_suite": r.get("test_suite") or "default_suite",
                "timestamp": r.get("timestamp") or "",
                "status": status,
                "duration_seconds": float(r["duration_seconds"]) if r.get("duration_seconds") is not None else 0.0,
                "error_type": r.get("error_type") or "",
                "error_message": r.get("error_message") or "",
                "stack_trace": r.get("stack_trace") or "",
                "log_excerpt": r.get("log_excerpt") or "",
                "original_category": str(raw_cat) if raw_cat else "unknown",
                "canonical_category": canonical_cat,
                "source_file": r.get("source_file") or source_name,
                "is_synthetic": r.get("is_synthetic", False),
                "template_group": r.get("template_group", r.get("test_name", "generic_group")),
            }
            filtered.append(rec_clean)

        df = pd.DataFrame(filtered)
        if df.empty:
            return df

        # Deduplicate records by error message, stack trace, and test name
        df = df.drop_duplicates(subset=["test_name", "error_message", "stack_trace"]).reset_index(drop=True)

        return df

    def check_class_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect and report class distribution and underrepresented categories."""
        if df.empty or "canonical_category" not in df.columns:
            return {"counts": {}, "underrepresented": CANONICAL_CATEGORIES, "total": 0}

        counts = df["canonical_category"].value_counts().to_dict()
        underrepresented = [
            cat for cat in CANONICAL_CATEGORIES
            if counts.get(cat, 0) < self.min_samples_per_class
        ]

        return {
            "counts": counts,
            "underrepresented": underrepresented,
            "total": len(df),
        }


class SyntheticDatasetGenerator:
    """Generates varied, reproducible synthetic wireless failure records for training and benchmarking."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_dataset(self, samples_per_category: int = 40) -> pd.DataFrame:
        """Generate labeled synthetic examples across all canonical failure categories."""
        templates = self._get_failure_templates()
        records: List[Dict[str, Any]] = []

        for category, cat_templates in templates.items():
            for i in range(samples_per_category):
                tmpl = cat_templates[i % len(cat_templates)]
                rec = self._render_template(category, tmpl, i)
                records.append(rec)

        df = pd.DataFrame(records)
        return df

    def _render_template(self, category: str, tmpl: Dict[str, Any], index: int) -> Dict[str, Any]:
        dut_id = f"DUT-WIFI6-{self.rng.randint(10, 99)}"
        bssid = f"00:11:22:33:44:{self.rng.randint(10, 99):02x}"
        ssid = self.rng.choice(["IoT_Core_5G", "Enterprise_Mesh_WiFi", "SmartMeter_Net", "Industrial_WLAN"])
        channel = self.rng.choice([36, 40, 44, 48, 149, 153])

        code = f"{self.rng.randint(1000, 9999):X}"
        key_len = self.rng.choice([3, 4, 5, 6, 7])
        band = self.rng.choice(["2.4GHz", "5GHz", "6GHz"])

        fmt_kwargs = {
            "dut_id": dut_id,
            "bssid": bssid,
            "ssid": ssid,
            "channel": channel,
            "band": band,
            "code": code,
            "key_len": key_len,
            "latency": round(self.rng.uniform(120.0, 320.0), 2),
            "packet_loss": round(self.rng.uniform(8.0, 35.0), 2),
            "rssi": round(self.rng.uniform(-95.0, -82.0), 1),
            "timeout_s": round(self.rng.uniform(3.0, 8.0), 1),
        }

        error_msg = tmpl["error_msg_fmt"].format(**fmt_kwargs)
        fmt_kwargs["error_msg"] = error_msg
        log_excerpt = tmpl["log_excerpt_fmt"].format(**fmt_kwargs)

        trace = tmpl.get("stack_trace_fmt", "Traceback (most recent call last):\n  File 'test.py', in run\n    raise {error_type}('{error_msg}')").format(
            error_type=tmpl["error_type"],
            error_msg=error_msg,
        )

        duration = round(self.rng.uniform(0.1, 5.5), 3)

        return {
            "test_id": f"SYN-{category[:3].upper()}-{index:04d}",
            "test_name": tmpl["test_name"],
            "test_suite": tmpl["test_suite"],
            "timestamp": f"2026-10-10 0{self.rng.randint(0, 9)}:{self.rng.randint(10, 59)}:{self.rng.randint(10, 59)}",
            "status": "failed" if category != "unknown" else self.rng.choice(["failed", "error"]),
            "duration_seconds": duration,
            "error_type": tmpl["error_type"],
            "error_message": error_msg,
            "stack_trace": trace,
            "log_excerpt": log_excerpt,
            "original_category": category,
            "canonical_category": category,
            "source_file": "synthetic_generator",
            "is_synthetic": True,
            "template_group": tmpl["group"],
        }

    def _get_failure_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "timeout": [
                {
                    "group": "timeout_handshake",
                    "test_name": "test_eapol_handshake_timeout",
                    "test_suite": "tests.test_connectivity",
                    "error_type": "WirelessTimeoutError",
                    "error_msg_fmt": "EAPOL 4-Way handshake timed out after {timeout_s}s waiting for Message 2 from AP {bssid} on DUT {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [WIFI-STACK] Association request sent to {ssid}\n2026-10-10 [ERROR] [WIFI-STACK] Handshake timeout: EAPOL timer expired after {timeout_s}s\nERROR: {error_msg}",
                },
                {
                    "group": "timeout_ping",
                    "test_name": "test_icmp_probe_deadline",
                    "test_suite": "tests.test_latency",
                    "error_type": "WirelessTimeoutError",
                    "error_msg_fmt": "ICMP echo probe timed out after {timeout_s}s waiting for reply from default gateway on ch {channel}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [STACK] Ping probe transmitted count=5\n2026-10-10 [ERROR] [STACK] ICMP response timer expired: host unresponsive\nERROR: {error_msg}",
                },
                {
                    "group": "timeout_dhcp",
                    "test_name": "test_dhcp_lease_timeout",
                    "test_suite": "tests.test_connectivity",
                    "error_type": "WirelessTimeoutError",
                    "error_msg_fmt": "DHCP lease acquisition timed out after {timeout_s}s waiting for DHCPACK on ch {channel} for {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [DHCP] DHCPDISCOVER broadcast transmitted\n2026-10-10 [ERROR] [DHCP] No DHCPOFFER received within timeout window\nERROR: {error_msg}",
                },
            ],
            "assertion_failure": [
                {
                    "group": "assertion_latency",
                    "test_name": "test_latency_sla_breach",
                    "test_suite": "tests.test_latency",
                    "error_type": "WirelessAssertionError",
                    "error_msg_fmt": "Latency SLA violation: measured RTT {latency} ms exceeds maximum allowed 50.0 ms on {ssid}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [METRIC] telemetry: latency_ms={latency} target=8.8.8.8\n2026-10-10 [ERROR] [TEST] Latency SLA breach: {latency} ms > 50.0 ms\nERROR: {error_msg}",
                },
                {
                    "group": "assertion_packet_loss",
                    "test_name": "test_packet_loss_breach",
                    "test_suite": "tests.test_packet_loss",
                    "error_type": "WirelessAssertionError",
                    "error_msg_fmt": "Packet loss threshold breach: {packet_loss}% dropped packets exceeds limit 2.0% on channel {channel}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [METRIC] telemetry: packet_loss_pct={packet_loss}\n2026-10-10 [ERROR] [TEST] Packet loss rate exceeded maximum threshold\nERROR: {error_msg}",
                },
                {
                    "group": "assertion_signal",
                    "test_name": "test_signal_threshold_breach",
                    "test_suite": "tests.test_signal",
                    "error_type": "WirelessAssertionError",
                    "error_msg_fmt": "Signal threshold breached: measured RSSI {rssi} dBm < minimum required -75.0 dBm on {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [RADIO] Sampling RF receiver telemetry RSSI={rssi} dBm\n2026-10-10 [ERROR] [TEST] Signal power below acceptable operating boundary\nERROR: {error_msg}",
                },
            ],
            "connection_error": [
                {
                    "group": "connection_beacon_loss",
                    "test_name": "test_beacon_frame_loss",
                    "test_suite": "tests.test_connectivity",
                    "error_type": "WirelessConnectionError",
                    "error_msg_fmt": "Beacon frame loss detected: AP beacon timeout on BSSID {bssid} for SSID '{ssid}' on channel {channel}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [RADIO] Scanning channel {channel}\n2026-10-10 [ERROR] [RADIO] Carrier lock lost: Beacon frames absent\nERROR: {error_msg}",
                },
                {
                    "group": "connection_carrier_drop",
                    "test_name": "test_carrier_drop_reconnect",
                    "test_suite": "tests.test_reconnection",
                    "error_type": "WirelessConnectionError",
                    "error_msg_fmt": "Radio link lost: carrier drop detected across reconnection attempts for {dut_id} on {band}",
                    "log_excerpt_fmt": "2026-10-10 [WARNING] [RADIO] Carrier wave degraded, SNR below lock threshold\n2026-10-10 [ERROR] [RADIO] Link dropped: interface physical layer disassociated\nERROR: {error_msg}",
                },
            ],
            "configuration_error": [
                {
                    "group": "config_invalid_ssid",
                    "test_name": "test_invalid_ssid_config",
                    "test_suite": "tests.test_connectivity",
                    "error_type": "WirelessConfigError",
                    "error_msg_fmt": "Configuration reject: Invalid SSID profile '{ssid}' or empty SSID provided on band {band} ch {channel}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [CONFIG] Loading Wi-Fi profile parameters for {ssid}\n2026-10-10 [ERROR] [CONFIG] Validation error: SSID cannot be empty or exceed 32 octets\nERROR: {error_msg}",
                },
                {
                    "group": "config_auth_key",
                    "test_name": "test_wpa3_key_length_invalid",
                    "test_suite": "tests.test_connectivity",
                    "error_type": "WirelessConfigError",
                    "error_msg_fmt": "Authentication key length {key_len} is below 8-character WPA2/WPA3 requirement for SSID '{ssid}'",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [CONFIG] Initializing SAE authentication profile on {dut_id}\n2026-10-10 [ERROR] [CONFIG] Passphrase rejected: key length does not satisfy WPA3 security policy\nERROR: {error_msg}",
                },
                {
                    "group": "config_cipher_mismatch",
                    "test_name": "test_cipher_suite_mismatch",
                    "test_suite": "tests.test_connectivity",
                    "error_type": "WirelessConfigError",
                    "error_msg_fmt": "Cipher suite mismatch: AP {bssid} requires CCMP-256 but DUT {dut_id} configured for TKIP on ch {channel}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [CONFIG] Negotiating security cipher suite with {bssid}\n2026-10-10 [ERROR] [CONFIG] Rejected: cipher suite unsupported by peer\nERROR: {error_msg}",
                },
            ],
            "dependency_error": [
                {
                    "group": "dep_hal_hang",
                    "test_name": "test_firmware_driver_hang",
                    "test_suite": "tests.test_hardware",
                    "error_type": "WirelessDependencyError",
                    "error_msg_fmt": "Firmware HAL driver failed to respond: err_code=0xDEAD{code} kernel ring buffer corrupted on {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [HAL] Initializing driver hardware bus on {dut_id}\n2026-10-10 [CRITICAL] [HAL] Fatal hardware driver failure: ring buffer fault\nERROR: {error_msg}",
                },
                {
                    "group": "dep_sensor_bus",
                    "test_name": "test_rf_telemetry_sensor_bus",
                    "test_suite": "tests.test_hardware",
                    "error_type": "WirelessDependencyError",
                    "error_msg_fmt": "HAL RF sensor bus error: I2C telemetry register read returned 0x{code} on device {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [HAL] Polling RF power sensor over I2C\n2026-10-10 [CRITICAL] [HAL] Sensor bus timeout: peripheral controller unresponsive\nERROR: {error_msg}",
                },
                {
                    "group": "dep_dma_overflow",
                    "test_name": "test_dma_queue_overflow",
                    "test_suite": "tests.test_packet_loss",
                    "error_type": "WirelessDependencyError",
                    "error_msg_fmt": "DMA TX queue ring buffer overflow: host controller interface (HCI) dropped packets on channel {channel} for {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [HAL] Burst packet transmission via DMA channel 3\n2026-10-10 [CRITICAL] [HAL] Host controller buffer overrun\nERROR: {error_msg}",
                },
            ],
            "unknown": [
                {
                    "group": "unknown_anomaly",
                    "test_name": "test_system_anomaly",
                    "test_suite": "tests.test_system",
                    "error_type": "RuntimeError",
                    "error_msg_fmt": "Unrecognized system anomaly in thread {dut_id}: diagnostic code 0x{code}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [SYS] Running background monitoring thread on {dut_id}\n2026-10-10 [ERROR] [SYS] General exception raised with unmapped diagnostic code 0x{code}\nERROR: {error_msg}",
                },
                {
                    "group": "unknown_memory",
                    "test_name": "test_memory_heap_unmapped",
                    "test_suite": "tests.test_system",
                    "error_type": "MemoryError",
                    "error_msg_fmt": "Memory allocation failure in test harness buffer at address 0x{code} for {dut_id}",
                    "log_excerpt_fmt": "2026-10-10 [INFO] [SYS] Allocating scratch memory block\n2026-10-10 [ERROR] [SYS] Buffer allocation returned NULL pointer\nERROR: {error_msg}",
                },
            ],
        }
