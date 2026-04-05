"""
Example: Wireless Security
============================
Wi-Fi protocol analysis, WPA2 4-way handshake model, evil twin
detection, and wireless channel analyzer.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Wi-Fi Encryption Standards
# ---------------------------------------------------------------------------

class WifiEncryption(Enum):
    OPEN = ("Open", 0, "No encryption")
    WEP = ("WEP", 1, "Broken — crackable in minutes")
    WPA_TKIP = ("WPA-TKIP", 2, "Deprecated — vulnerable to TKIP attacks")
    WPA2_PSK = ("WPA2-PSK", 3, "Vulnerable to offline dictionary attack")
    WPA2_ENT = ("WPA2-Enterprise", 4, "Strong with proper 802.1X config")
    WPA3_SAE = ("WPA3-SAE", 5, "Resistant to offline attacks (Dragonfly)")

    @property
    def security_level(self) -> int:
        return self.value[1]

    @property
    def notes(self) -> str:
        return self.value[2]


# ---------------------------------------------------------------------------
# Access Point Model
# ---------------------------------------------------------------------------

@dataclass
class AccessPoint:
    ssid: str
    bssid: str
    channel: int
    encryption: WifiEncryption
    signal_dbm: int
    clients: int = 0

    @property
    def signal_quality(self) -> str:
        if self.signal_dbm > -50:
            return "excellent"
        if self.signal_dbm > -60:
            return "good"
        if self.signal_dbm > -70:
            return "fair"
        return "weak"


# ---------------------------------------------------------------------------
# WPA2 4-Way Handshake Model
# ---------------------------------------------------------------------------

@dataclass
class HandshakeMessage:
    msg_number: int
    direction: str  # "AP->Client" or "Client->AP"
    contains_anonce: bool
    contains_snonce: bool
    contains_mic: bool
    installs_key: bool


WPA2_HANDSHAKE = [
    HandshakeMessage(1, "AP->Client", True, False, False, False),
    HandshakeMessage(2, "Client->AP", False, True, True, False),
    HandshakeMessage(3, "AP->Client", True, False, True, True),
    HandshakeMessage(4, "Client->AP", False, False, True, False),
]


def validate_handshake(captured: list[int]) -> dict:
    """Check if captured messages form a crackable handshake."""
    has_m1 = 1 in captured
    has_m2 = 2 in captured
    has_m3 = 3 in captured
    has_m4 = 4 in captured

    crackable = (has_m1 and has_m2) or (has_m2 and has_m3)
    complete = has_m1 and has_m2 and has_m3 and has_m4
    return {
        "messages": captured,
        "crackable": crackable,
        "complete": complete,
        "missing": [m for m in [1, 2, 3, 4] if m not in captured],
        "method": "aircrack-ng or hashcat -m 22000" if crackable else "N/A",
    }


# ---------------------------------------------------------------------------
# Evil Twin Detection
# ---------------------------------------------------------------------------

@dataclass
class APObservation:
    timestamp: str
    ssid: str
    bssid: str
    channel: int
    signal_dbm: int


def detect_evil_twin(observations: list[APObservation]) -> dict:
    """Detect potential evil twin APs from observation history."""
    ssid_bssids: dict[str, set[str]] = {}
    ssid_signals: dict[str, list[tuple[str, int]]] = {}

    for obs in observations:
        ssid_bssids.setdefault(obs.ssid, set()).add(obs.bssid)
        ssid_signals.setdefault(obs.ssid, []).append(
            (obs.bssid, obs.signal_dbm))

    alerts = []
    for ssid, bssids in ssid_bssids.items():
        if len(bssids) > 1:
            signals = ssid_signals[ssid]
            alerts.append({
                "ssid": ssid,
                "bssid_count": len(bssids),
                "bssids": list(bssids),
                "indicator": "Multiple BSSIDs for same SSID",
                "risk": "high" if len(bssids) > 2 else "medium",
            })

    return {"alerts": alerts, "evil_twin_suspected": len(alerts) > 0}


# ---------------------------------------------------------------------------
# Channel Utilization
# ---------------------------------------------------------------------------

WIFI_24_CHANNELS = list(range(1, 14))
NON_OVERLAPPING_24 = [1, 6, 11]


def channel_congestion(aps: list[AccessPoint]) -> dict[int, int]:
    """Count APs per channel to identify congestion."""
    counts: dict[int, int] = {ch: 0 for ch in WIFI_24_CHANNELS}
    for ap in aps:
        if ap.channel in counts:
            counts[ap.channel] += 1
    return counts


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Wireless Security Examples")
    print("=" * 50)

    # Encryption comparison
    print("\nWi-Fi Encryption Standards:")
    for enc in WifiEncryption:
        print(f"  [{enc.security_level}] {enc.value[0]:20s} {enc.notes}")

    # Handshake validation
    print("\nWPA2 Handshake Validation:")
    for captured in [[1, 2], [2, 3], [1, 2, 3, 4], [1, 3]]:
        result = validate_handshake(captured)
        status = "CRACKABLE" if result["crackable"] else "incomplete"
        print(f"  Messages {captured}: {status}")

    # Evil twin detection
    print("\nEvil Twin Detection:")
    observations = [
        APObservation("10:00", "CorpWiFi", "AA:BB:CC:DD:EE:01", 6, -45),
        APObservation("10:05", "CorpWiFi", "AA:BB:CC:DD:EE:01", 6, -47),
        APObservation("10:10", "CorpWiFi", "11:22:33:44:55:66", 6, -30),
    ]
    result = detect_evil_twin(observations)
    for alert in result["alerts"]:
        print(f"  ALERT: {alert['ssid']} has {alert['bssid_count']} BSSIDs")
        print(f"    BSSIDs: {alert['bssids']}")

    # Channel congestion
    print("\nChannel Congestion (2.4 GHz):")
    aps = [
        AccessPoint("Net1", "AA:01", 1, WifiEncryption.WPA2_PSK, -50),
        AccessPoint("Net2", "AA:02", 1, WifiEncryption.WPA2_PSK, -60),
        AccessPoint("Net3", "AA:03", 6, WifiEncryption.WPA2_ENT, -55),
        AccessPoint("Net4", "AA:04", 6, WifiEncryption.WPA2_PSK, -65),
        AccessPoint("Net5", "AA:05", 6, WifiEncryption.WPA2_PSK, -70),
        AccessPoint("Net6", "AA:06", 11, WifiEncryption.OPEN, -60),
    ]
    congestion = channel_congestion(aps)
    for ch in NON_OVERLAPPING_24:
        count = congestion[ch]
        bar = "#" * count
        print(f"  Ch {ch:2d}: {bar} ({count} APs)")


if __name__ == "__main__":
    demo()
