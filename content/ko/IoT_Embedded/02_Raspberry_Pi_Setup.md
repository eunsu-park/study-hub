# 02. 라즈베리파이 설정

**이전**: [IoT 개요](./01_IoT_Overview.md) | **다음**: [Python GPIO 제어](./03_Python_GPIO_Control.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 주어진 IoT 프로젝트에 적합한 Raspberry Pi 모델을 선택할 수 있다
2. Raspberry Pi Imager를 사용하여 Raspberry Pi OS를 설치할 수 있다
3. SSH 접속을 설정하고 초기 시스템 구성을 수행할 수 있다
4. GPIO 핀 배치를 파악하고 BCM과 BOARD 번호 체계를 구분할 수 있다
5. 공통 IoT 라이브러리로 Python 가상 환경을 구성할 수 있다
6. 애플리케이션 자동 시작을 위한 systemd 서비스를 생성할 수 있다

---

모든 IoT 프로젝트는 실행할 물리적 플랫폼이 필요하며, Raspberry Pi는 임베디드 개발에 입문하는 가장 접근하기 쉬운 관문입니다. 올바른 모델 선택, OS 설정, GPIO 핀 배치 이해를 통해 안정적인 기반을 마련하면, 이후 이 과정에서 다루는 센서, 네트워킹, AI 프로젝트를 모두 원활하게 진행할 수 있습니다.

---

## 1. Raspberry Pi 모델 소개

### 1.1 모델 비교

| 모델 | CPU | RAM | 특징 | 권장 용도 |
|------|-----|-----|------|-----------|
| **Pi 5** | Cortex-A76 2.4GHz | 4-8GB | PCIe, USB 3.0 | AI, 데스크탑 |
| **Pi 4B** | Cortex-A72 1.8GHz | 1-8GB | USB 3.0, 듀얼 HDMI | 범용, IoT 게이트웨이 |
| **Pi 3B+** | Cortex-A53 1.4GHz | 1GB | WiFi, BLE 내장 | 교육, 간단한 IoT |
| **Pi Zero 2W** | Cortex-A53 1GHz | 512MB | 소형, 저전력 | 임베디드, 웨어러블 |
| **Pi Pico** | RP2040 133MHz | 264KB | 마이크로컨트롤러 | 센서 노드 |

### 1.2 IoT 프로젝트별 권장 모델

```python
# 프로젝트별 모델 권장 가이드
recommendations = {
    "Edge AI (TFLite, ONNX)": {
        "model": "Pi 4B (4GB+) 또는 Pi 5",
        "reason": "AI 모델 실행에 충분한 RAM과 CPU 필요"
    },
    "IoT 게이트웨이": {
        "model": "Pi 4B (2GB+)",
        "reason": "다중 프로토콜 처리, 안정적인 네트워킹"
    },
    "단순 센서 수집": {
        "model": "Pi Zero 2W 또는 Pi 3B+",
        "reason": "저전력, 소형, 비용 효율"
    },
    "카메라/영상 분석": {
        "model": "Pi 4B (4GB+) 또는 Pi 5",
        "reason": "영상 처리에 높은 처리 능력 필요"
    },
    "교육/프로토타이핑": {
        "model": "Pi 4B (2GB)",
        "reason": "범용성, 풍부한 자료"
    }
}
```

### 1.3 필수 액세서리

```
┌─────────────────────────────────────────────────────────────┐
│                    권장 액세서리 목록                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [필수]                                                      │
│  ├── microSD 카드 (32GB+, Class 10/A2)                      │
│  ├── 전원 어댑터 (Pi 4: 5V 3A USB-C, Pi 3: 5V 2.5A micro)   │
│  └── 히트싱크/냉각팬 (Pi 4/5 권장)                           │
│                                                              │
│  [권장]                                                      │
│  ├── 케이스 (GPIO 접근 가능한 모델)                          │
│  ├── HDMI 케이블/모니터 (초기 설정용)                        │
│  └── USB 키보드/마우스 (초기 설정용)                         │
│                                                              │
│  [IoT 프로젝트용]                                            │
│  ├── 브레드보드 + 점퍼 와이어                                │
│  ├── LED, 저항, 버튼                                         │
│  ├── 센서 키트 (DHT11, PIR 등)                              │
│  └── Pi Camera Module                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. OS 설치

### 2.1 Raspberry Pi Imager 사용

가장 쉬운 방법은 공식 Raspberry Pi Imager를 사용하는 것입니다.

```bash
# macOS
brew install --cask raspberry-pi-imager

# Windows
# https://www.raspberrypi.com/software/ 에서 다운로드

# Linux (Ubuntu/Debian)
sudo apt install rpi-imager
```

### 2.2 설치 과정

```
┌─────────────────────────────────────────────────────────────┐
│                  OS 설치 단계                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Raspberry Pi Imager 실행                                │
│     ▼                                                        │
│  2. OS 선택                                                  │
│     └── Raspberry Pi OS (64-bit) 권장                       │
│     ▼                                                        │
│  3. 저장장치 선택 (microSD 카드)                            │
│     ▼                                                        │
│  4. 고급 옵션 설정 (톱니바퀴 아이콘)                        │
│     ├── 호스트네임 설정 (예: raspberrypi)                   │
│     ├── SSH 활성화                                          │
│     ├── 사용자명/비밀번호 설정                              │
│     ├── WiFi 설정 (SSID, 비밀번호)                         │
│     └── 지역 설정 (시간대, 키보드 레이아웃)                 │
│     ▼                                                        │
│  5. 쓰기 버튼 클릭                                          │
│     ▼                                                        │
│  6. 완료 후 SD카드를 Pi에 삽입                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 OS 선택 가이드

| OS 버전 | 특징 | 권장 용도 |
|---------|------|-----------|
| **Raspberry Pi OS (64-bit)** | 데스크탑 환경, 권장 | 범용, AI |
| **Raspberry Pi OS Lite** | CLI만, 경량 | 서버, 헤드리스 IoT |
| **Ubuntu Server** | 표준 Ubuntu | 서버, Docker |
| **Home Assistant OS** | 스마트홈 전용 | 홈 자동화 |

### 2.4 수동 헤드리스 설정 (모니터 없이)

```bash
# SD카드의 boot 파티션에서 작업

# 1. SSH 활성화
touch /Volumes/bootfs/ssh  # macOS
# touch /media/$USER/bootfs/ssh  # Linux

# 2. WiFi 설정 (wpa_supplicant.conf 생성)
cat > /Volumes/bootfs/wpa_supplicant.conf << 'EOF'
country=KR
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YOUR_WIFI_SSID"
    psk="YOUR_WIFI_PASSWORD"
    key_mgmt=WPA-PSK
}
EOF
```

---

## 3. SSH 설정

### 3.1 첫 연결

```bash
# Pi가 네트워크에 연결된 후

# 호스트네임으로 연결 (mDNS)
ssh pi@raspberrypi.local

# 또는 IP 주소로 연결
ssh pi@192.168.1.100

# 기본 사용자: pi (Imager에서 설정한 경우 해당 사용자)
# 기본 비밀번호: Imager에서 설정한 비밀번호
```

### 3.2 SSH 키 설정 (권장)

```bash
# 1. 로컬 PC에서 SSH 키 생성 (없는 경우)
ssh-keygen -t ed25519 -C "your-email@example.com"

# 2. 공개키를 Pi에 복사
ssh-copy-id pi@raspberrypi.local

# 또는 수동으로
cat ~/.ssh/id_ed25519.pub | ssh pi@raspberrypi.local 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'

# 3. 비밀번호 없이 연결 확인
ssh pi@raspberrypi.local
```

### 3.3 SSH 보안 강화

```bash
# Pi에서 /etc/ssh/sshd_config 수정
sudo nano /etc/ssh/sshd_config

# 권장 설정:
# PermitRootLogin no
# PasswordAuthentication no  # 키 설정 후
# PubkeyAuthentication yes

# SSH 서비스 재시작
sudo systemctl restart ssh
```

### 3.4 VS Code Remote SSH

```
┌─────────────────────────────────────────────────────────────┐
│              VS Code Remote SSH 설정                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. VS Code 확장 설치                                        │
│     └── "Remote - SSH" 검색 후 설치                         │
│                                                              │
│  2. SSH 호스트 추가                                          │
│     └── F1 > "Remote-SSH: Add New SSH Host"                 │
│     └── ssh pi@raspberrypi.local 입력                       │
│                                                              │
│  3. 연결                                                     │
│     └── F1 > "Remote-SSH: Connect to Host"                  │
│     └── raspberrypi.local 선택                              │
│                                                              │
│  4. 폴더 열기                                                │
│     └── "Open Folder" > /home/pi/projects                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 기본 설정 (raspi-config)

### 4.1 raspi-config 실행

```bash
sudo raspi-config
```

### 4.2 주요 설정 항목

```
┌─────────────────────────────────────────────────────────────┐
│                    raspi-config 메뉴                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1 System Options                                            │
│     ├── S1 Wireless LAN      : WiFi 설정                    │
│     ├── S3 Password          : 비밀번호 변경                 │
│     ├── S4 Hostname          : 호스트네임 변경               │
│     └── S5 Boot / Auto Login : 부팅 옵션                    │
│                                                              │
│  3 Interface Options                                         │
│     ├── I1 Legacy Camera     : 레거시 카메라 (Pi 4 이전)    │
│     ├── I2 SSH               : SSH 활성화/비활성화           │
│     ├── I3 VNC               : VNC 원격 데스크탑             │
│     ├── I4 SPI               : SPI 인터페이스                │
│     ├── I5 I2C               : I2C 인터페이스                │
│     ├── I6 Serial Port       : 시리얼 포트                   │
│     └── I7 1-Wire            : 1-Wire 프로토콜               │
│                                                              │
│  5 Localisation Options                                      │
│     ├── L1 Locale            : 언어/지역 설정                │
│     ├── L2 Timezone          : 시간대 (Asia/Seoul)          │
│     └── L4 WLAN Country      : 무선 국가 코드 (KR)          │
│                                                              │
│  6 Advanced Options                                          │
│     ├── A1 Expand Filesystem : SD 카드 전체 사용            │
│     └── A3 Memory Split      : GPU 메모리 할당              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 명령줄로 설정

```bash
# 인터페이스 활성화
sudo raspi-config nonint do_ssh 0       # SSH 활성화
sudo raspi-config nonint do_i2c 0       # I2C 활성화
sudo raspi-config nonint do_spi 0       # SPI 활성화
sudo raspi-config nonint do_serial 0    # Serial 활성화

# 시간대 설정
sudo timedatectl set-timezone Asia/Seoul

# 호스트네임 변경
sudo hostnamectl set-hostname my-iot-pi

# 파일시스템 확장
sudo raspi-config --expand-rootfs
```

### 4.4 필수 패키지 설치

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 개발 도구
sudo apt install -y python3-pip python3-venv git

# GPIO 라이브러리
sudo apt install -y python3-gpiozero python3-rpi.gpio

# 센서 라이브러리
sudo apt install -y python3-smbus i2c-tools

# 네트워킹 도구
sudo apt install -y curl wget net-tools
```

---

## 5. GPIO 핀아웃

### 5.1 GPIO 핀 맵 (40핀 헤더)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Raspberry Pi GPIO 핀아웃                       │
│                      (40핀 헤더, Pi 2B+)                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│            3V3  (1)  (2)  5V                                      │
│   GPIO2 (SDA1)  (3)  (4)  5V                                      │
│   GPIO3 (SCL1)  (5)  (6)  GND                                     │
│          GPIO4  (7)  (8)  GPIO14 (TXD)                            │
│            GND  (9) (10)  GPIO15 (RXD)                            │
│         GPIO17 (11) (12)  GPIO18 (PWM0)                           │
│         GPIO27 (13) (14)  GND                                     │
│         GPIO22 (15) (16)  GPIO23                                  │
│            3V3 (17) (18)  GPIO24                                  │
│  GPIO10 (MOSI) (19) (20)  GND                                     │
│   GPIO9 (MISO) (21) (22)  GPIO25                                  │
│  GPIO11 (SCLK) (23) (24)  GPIO8 (CE0)                             │
│            GND (25) (26)  GPIO7 (CE1)                             │
│   GPIO0 (ID_SD)(27) (28)  GPIO1 (ID_SC)                           │
│          GPIO5 (29) (30)  GND                                     │
│          GPIO6 (31) (32)  GPIO12 (PWM0)                           │
│  GPIO13 (PWM1) (33) (34)  GND                                     │
│  GPIO19 (MISO) (35) (36)  GPIO16                                  │
│         GPIO26 (37) (38)  GPIO20 (MOSI)                           │
│            GND (39) (40)  GPIO21 (SCLK)                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

범례:
  - 3V3/5V  : 전원 핀
  - GND     : 그라운드
  - GPIOxx  : 범용 입출력 핀
  - SDA/SCL : I2C 통신
  - MOSI/MISO/SCLK/CE : SPI 통신
  - TXD/RXD : UART 시리얼 통신
  - PWM     : 하드웨어 PWM
```

### 5.2 핀 기능 분류

```python
# GPIO 핀 분류
gpio_pins = {
    "power": {
        "3.3V": [1, 17],
        "5V": [2, 4],
        "GND": [6, 9, 14, 20, 25, 30, 34, 39]
    },
    "i2c": {
        "SDA": 3,   # GPIO2
        "SCL": 5    # GPIO3
    },
    "spi0": {
        "MOSI": 19,  # GPIO10
        "MISO": 21,  # GPIO9
        "SCLK": 23,  # GPIO11
        "CE0": 24,   # GPIO8
        "CE1": 26    # GPIO7
    },
    "uart": {
        "TXD": 8,    # GPIO14
        "RXD": 10    # GPIO15
    },
    "pwm": {
        "PWM0": [12, 32],  # GPIO18, GPIO12
        "PWM1": [33, 35]   # GPIO13, GPIO19
    },
    "general_purpose": [
        7, 11, 12, 13, 15, 16, 18, 22,
        29, 31, 32, 33, 35, 36, 37, 38, 40
    ]
}
```

### 5.3 pinout 명령어

```bash
# pinout 명령어로 핀아웃 확인
pinout

# 출력 예시:
# +--]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+--+
# |                   Raspberry Pi 4B          |
# |    USB  USB     ETH                        |
# +]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+]+--+
```

### 5.4 GPIO 확인 Python 스크립트

```python
#!/usr/bin/env python3
"""GPIO 핀 상태 확인 스크립트"""

import subprocess

def check_gpio_status():
    """현재 GPIO 상태 확인"""
    try:
        result = subprocess.run(['gpio', 'readall'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("wiringPi가 설치되지 않았습니다.")
        print("대안: gpiozero 사용")
        check_with_gpiozero()

def check_with_gpiozero():
    """gpiozero로 핀 정보 출력"""
    from gpiozero import Device
    from gpiozero.pins.rpigpio import RPiGPIOFactory

    print("\n=== GPIO 핀 정보 ===")
    Device.pin_factory = RPiGPIOFactory()

    # 사용 가능한 GPIO 핀
    available_pins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 26, 27]

    for pin in available_pins:
        print(f"GPIO{pin}: 사용 가능")

if __name__ == "__main__":
    check_gpio_status()
```

---

## 6. 시스템 관리

### 6.1 유용한 명령어

```bash
# 시스템 정보
cat /etc/os-release          # OS 버전
uname -a                      # 커널 정보
vcgencmd measure_temp         # CPU 온도
free -h                       # 메모리 사용량
df -h                         # 디스크 사용량

# 네트워크
ip addr                       # IP 주소
iwconfig                      # WiFi 상태
ping google.com               # 인터넷 연결 확인

# 서비스 관리
sudo systemctl status ssh     # SSH 상태
sudo systemctl restart ssh    # SSH 재시작
sudo systemctl enable myapp   # 부팅 시 자동 시작
```

### 6.2 Python 가상환경 설정

```bash
# 프로젝트 디렉토리 생성
mkdir -p ~/projects/iot-demo
cd ~/projects/iot-demo

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install gpiozero paho-mqtt flask

# 가상환경 비활성화
deactivate
```

### 6.3 자동 시작 서비스 설정

```bash
# systemd 서비스 파일 생성
sudo nano /etc/systemd/system/my-iot-app.service
```

```ini
# /etc/systemd/system/my-iot-app.service
[Unit]
Description=My IoT Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/projects/iot-demo
ExecStart=/home/pi/projects/iot-demo/venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable my-iot-app
sudo systemctl start my-iot-app
sudo systemctl status my-iot-app
```

---

## 연습 문제

### 연습 1: 모델 선택 근거 설명

세 가지 IoT 배포 시나리오에 대해 가장 적합한 Raspberry Pi 모델을 선택하고 이유를 설명하세요:

1. 체온을 모니터링하여 스마트폰으로 전송하는 휴대용 웨어러블(Wearable). 크기와 전력 소비가 중요합니다.
2. 20개의 센서에서 데이터를 수집하고, 로컬 MQTT 브로커(broker)를 실행하며, 클라우드(cloud) 서버로 영상을 스트리밍(streaming)하는 게이트웨이(gateway) 장치.
3. GPIO, Python, 데스크탑 GUI를 실험하고 싶지만 예산이 제한적인 학습 프로젝트.

각 시나리오에 대해 짧은 단락으로 작성하고, 섹션 1의 모델 비교 표를 참조하세요.

### 연습 2: 헤드리스(Headless) 설정 및 SSH 보안 강화

모니터나 키보드 없이 헤드리스 모드로 Raspberry Pi를 설정하고 SSH 접근을 강화하세요:

1. Raspberry Pi Imager를 사용하여 SSH 활성화 및 WiFi 사전 설정과 함께 Raspberry Pi OS Lite를 SD 카드에 씁니다.
2. Pi를 부팅하고 `nmap` 또는 `arp`로 IP 주소를 찾아 SSH로 접속합니다.
3. 호스트 머신에서 Ed25519 키 쌍을 생성하고 `ssh-copy-id`를 사용하여 공개 키를 Pi에 복사합니다.
4. `/etc/ssh/sshd_config`를 편집하여 비밀번호 인증(Password Authentication)을 비활성화(`PasswordAuthentication no`)하고 SSH 서비스를 재시작합니다.
5. 키로 로그인이 가능하고 비밀번호 로그인이 거부되는지 확인합니다.

### 연습 3: GPIO 핀 매핑(Pin Mapping)

섹션 4.1의 40핀 GPIO 헤더 배치를 학습하고 어떤 참고 자료도 보지 않고 다음에 답하세요:

1. 어느 물리적 핀 번호가 5V 전원을 공급하나요? 3.3V는?
2. BCM 번호 체계에서 GPIO17의 물리적(BOARD) 핀 번호는 무엇인가요?
3. 하드웨어 PWM을 지원하는 GPIO 핀은 어느 것인가요?
4. I2C 통신에 사용되는 핀은 어느 것이며, BCM 번호는 무엇인가요?

그런 다음 물리적 핀 11에 연결된 LED를 BOARD 모드에서 `RPi.GPIO`를 사용하여 깜빡이는 짧은 Python 코드를 작성하세요.

### 연습 4: 자동 시작을 위한 systemd 서비스

부팅 시 Python 센서 스크립트를 자동으로 실행하는 systemd 서비스를 만드세요:

1. `vcgencmd measure_temp`로 CPU 온도를 읽고, 타임스탬프(timestamp)와 함께 `/home/pi/iot_project/temp_log.csv`에 기록하며, 루프에서 30초간 슬립(sleep)하는 Python 스크립트(`/home/pi/iot_project/temperature_logger.py`)를 작성합니다.
2. 해당 스크립트를 참조하는 systemd 서비스 파일을 `/etc/systemd/system/temp_logger.service`에 만듭니다.
3. 서비스를 활성화하고 시작한 후, `systemctl status`로 실행 중임을 확인합니다.
4. Pi를 재부팅하고 부팅 후 로그 파일을 확인하여 서비스가 자동으로 시작되는지 검증합니다.

### 연습 5: 네트워크 모니터링 및 자동 재연결

인터넷 연결을 모니터링하고 연결이 끊기면 자동으로 네트워크를 재시작하는 Python 스크립트를 작성하세요:

1. `subprocess`를 사용하여 60초마다 `8.8.8.8`에 ping을 보냅니다.
2. ping이 실패하면 연결 끊김 시각을 파일에 기록하고 `sudo systemctl restart networking`으로 네트워킹 서비스를 재시작합니다.
3. 재연결 후 재연결 시각을 기록하고 중단 지속 시간을 계산합니다.
4. 스크립트를 systemd 서비스로 감싸 백그라운드에서 지속적으로 실행되도록 합니다. `sudo ip link set wlan0 down`으로 WiFi 인터페이스를 임시로 비활성화하여 테스트하세요.

---

## 다음 단계

- [Python GPIO 제어](03_Python_GPIO_Control.md): Python으로 GPIO 제어 시작
- [WiFi 네트워킹](04_WiFi_Networking.md): 네트워크 프로그래밍

---

*최종 업데이트: 2026-02-01*
