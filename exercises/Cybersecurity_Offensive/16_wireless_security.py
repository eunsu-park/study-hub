"""
Exercises for Lesson 16: Wireless Security
Topic: Cybersecurity_Offensive

Practice problems covering Wi-Fi protocol analysis, WPA2 cracking
concepts, evil twin detection, and wireless recon.
"""


# === Exercise 1: Wi-Fi Protocol Analyzer ===
# Problem: Given captured beacon frame data, extract AP information.

def exercise_1():
    """
    beacon_frames = [
        {"ssid": "CorpWiFi", "bssid": "AA:BB:CC:DD:EE:01",
         "channel": 6, "encryption": "WPA2-Enterprise", "signal": -45,
         "cipher": "CCMP", "auth": "802.1X"},
        {"ssid": "GuestNet", "bssid": "AA:BB:CC:DD:EE:02",
         "channel": 11, "encryption": "WPA2-Personal", "signal": -60,
         "cipher": "TKIP", "auth": "PSK"},
        {"ssid": "", "bssid": "AA:BB:CC:DD:EE:03",
         "channel": 1, "encryption": "Open", "signal": -70,
         "cipher": "None", "auth": "None"},
        {"ssid": "CorpWiFi", "bssid": "AA:BB:CC:DD:EE:04",
         "channel": 6, "encryption": "WPA2-Enterprise", "signal": -80,
         "cipher": "CCMP", "auth": "802.1X"},
    ]
    Analyze:
      1. Which networks use weak encryption?
      2. Are there hidden SSIDs?
      3. Are there potential rogue APs (same SSID, different BSSID)?
    Return analysis dict.
    """
    # TODO: Analyze beacon frames for security issues
    pass


# === Exercise 2: WPA2 Handshake Validator ===
# Problem: Verify if captured EAPOL frames constitute a complete
# 4-way handshake for WPA2 cracking.

def exercise_2():
    """
    captured_eapol = [
        {"msg_num": 1, "src": "AP", "dst": "Client", "has_anonce": True,
         "has_snonce": False, "has_mic": False},
        {"msg_num": 2, "src": "Client", "dst": "AP", "has_anonce": False,
         "has_snonce": True, "has_mic": True},
        {"msg_num": 3, "src": "AP", "dst": "Client", "has_anonce": True,
         "has_snonce": False, "has_mic": True},
    ]
    Determine:
      1. Is this a complete 4-way handshake? (need at least msg 1+2 or 2+3)
      2. Can we attempt offline cracking?
      3. What minimum messages are needed?
    Return {"complete": bool, "crackable": bool, "messages_captured": list,
            "missing": list, "recommendation": str}
    """
    # TODO: Validate WPA2 handshake completeness
    pass


# === Exercise 3: Evil Twin Detector ===
# Problem: Given a list of APs seen over time, detect potential
# evil twin attacks by analyzing anomalies.

def exercise_3():
    """
    ap_history = [
        {"time": "10:00", "ssid": "CorpWiFi", "bssid": "AA:BB:CC:DD:EE:01",
         "channel": 6, "signal": -45},
        {"time": "10:05", "ssid": "CorpWiFi", "bssid": "AA:BB:CC:DD:EE:01",
         "channel": 6, "signal": -47},
        {"time": "10:10", "ssid": "CorpWiFi", "bssid": "11:22:33:44:55:66",
         "channel": 6, "signal": -30},  # suspicious: new BSSID, stronger
        {"time": "10:15", "ssid": "CorpWiFi", "bssid": "AA:BB:CC:DD:EE:01",
         "channel": 6, "signal": -80},  # original weakened
    ]
    Detect evil twin indicators:
      - New BSSID for known SSID
      - Unusually strong signal
      - Original AP signal degradation (deauth attack?)
    Return {"evil_twin_detected": bool, "indicators": list[str],
            "suspicious_bssid": str, "confidence": str}
    """
    # TODO: Detect evil twin attack indicators
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Wi-Fi Protocol Analyzer ===")
    print(exercise_1())
    print("\n=== Exercise 2: WPA2 Handshake Validator ===")
    print(exercise_2())
    print("\n=== Exercise 3: Evil Twin Detector ===")
    print(exercise_3())
