# 04. WiFi 네트워킹

**이전**: [Python GPIO 제어](./03_Python_GPIO_Control.md) | **다음**: [BLE 연결](./05_BLE_Connectivity.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. wpa_supplicant를 사용하여 Raspberry Pi에서 WiFi 네트워크를 설정할 수 있다
2. Python으로 TCP 서버와 클라이언트 통신을 구현할 수 있다
3. 실시간 데이터 전송을 위한 UDP 서버와 클라이언트 통신을 구현할 수 있다
4. IoT API와의 HTTP 통신에 requests 라이브러리를 활용할 수 있다
5. 안정적인 데이터 전송을 위한 재시도 및 배치 전송 로직을 구축할 수 있다
6. 로컬 네트워크를 스캔하여 IoT 장치와 열린 포트를 탐색할 수 있다

---

통신할 수 없는 IoT 장치는 그저 아무데도 기록하지 않는 센서에 불과합니다. WiFi 네트워킹은 Raspberry Pi를 고립된 데이터 수집기에서 측정값을 보고하고, 명령을 받고, 더 큰 시스템에 참여할 수 있는 연결된 노드로 변환시킵니다. 소켓 프로그래밍과 HTTP 통신을 마스터하면 이후 레슨에서 다루는 모든 프로토콜의 기반을 갖추게 됩니다.

---

## 1. 라즈베리파이 WiFi 설정

### 1.1 명령줄 WiFi 설정

```bash
# 현재 WiFi 상태 확인
iwconfig wlan0

# 사용 가능한 네트워크 스캔
sudo iwlist wlan0 scan | grep -E "ESSID|Quality"

# WiFi 연결 (nmcli 사용)
sudo nmcli dev wifi connect "SSID이름" password "비밀번호"

# 연결 상태 확인
nmcli connection show

# IP 주소 확인
ip addr show wlan0
```

### 1.2 wpa_supplicant 설정

```bash
# /etc/wpa_supplicant/wpa_supplicant.conf 편집
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

```conf
# /etc/wpa_supplicant/wpa_supplicant.conf
country=KR
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

# 기본 WPA2 네트워크
network={
    ssid="MyNetwork"
    psk="MyPassword"
    key_mgmt=WPA-PSK
}

# 숨겨진 네트워크
network={
    ssid="HiddenNetwork"
    scan_ssid=1
    psk="Password"
}

# 우선순위 설정 (높은 값 = 우선)
network={
    ssid="PreferredNetwork"
    psk="Password"
    priority=10
}
```

### 1.3 Python으로 WiFi 정보 조회

```python
#!/usr/bin/env python3
"""WiFi 연결 정보 조회"""

import subprocess
import re

def get_wifi_info() -> dict:
    """현재 WiFi 연결 정보 반환"""
    info = {}

    try:
        # SSID 조회
        result = subprocess.run(
            ['iwgetid', '-r'],
            capture_output=True,
            text=True
        )
        info['ssid'] = result.stdout.strip()

        # IP 주소 조회
        result = subprocess.run(
            ['hostname', '-I'],
            capture_output=True,
            text=True
        )
        ips = result.stdout.strip().split()
        info['ip_addresses'] = ips

        # 신호 강도 조회
        result = subprocess.run(
            ['iwconfig', 'wlan0'],
            capture_output=True,
            text=True
        )
        match = re.search(r'Signal level=(-?\d+)', result.stdout)
        if match:
            info['signal_dbm'] = int(match.group(1))

        # MAC 주소
        result = subprocess.run(
            ['cat', '/sys/class/net/wlan0/address'],
            capture_output=True,
            text=True
        )
        info['mac_address'] = result.stdout.strip()

    except Exception as e:
        info['error'] = str(e)

    return info

def get_wifi_networks() -> list:
    """주변 WiFi 네트워크 스캔"""
    networks = []

    try:
        result = subprocess.run(
            ['sudo', 'iwlist', 'wlan0', 'scan'],
            capture_output=True,
            text=True
        )

        current_network = {}
        for line in result.stdout.split('\n'):
            if 'ESSID:' in line:
                ssid = re.search(r'ESSID:"(.+)"', line)
                if ssid and current_network:
                    networks.append(current_network)
                current_network = {'ssid': ssid.group(1) if ssid else ''}

            elif 'Quality=' in line:
                quality = re.search(r'Quality=(\d+)/(\d+)', line)
                if quality:
                    current_network['quality'] = f"{quality.group(1)}/{quality.group(2)}"

                signal = re.search(r'Signal level=(-?\d+)', line)
                if signal:
                    current_network['signal_dbm'] = int(signal.group(1))

        if current_network:
            networks.append(current_network)

    except Exception as e:
        print(f"스캔 실패: {e}")

    return networks

if __name__ == "__main__":
    print("=== WiFi 연결 정보 ===")
    info = get_wifi_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n=== 주변 WiFi 네트워크 ===")
    networks = get_wifi_networks()
    for net in networks[:10]:  # 상위 10개만
        print(f"  {net.get('ssid', 'Unknown')}: {net.get('signal_dbm', 'N/A')} dBm")
```

---

## 2. Python 소켓 프로그래밍

### 2.1 소켓 기초

```
┌─────────────────────────────────────────────────────────────┐
│                    소켓 통신 흐름                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   클라이언트                              서버                │
│   ┌─────────┐                        ┌─────────┐            │
│   │ socket()│                        │ socket()│            │
│   └────┬────┘                        └────┬────┘            │
│        │                                  │                 │
│        │                             ┌────┴────┐            │
│        │                             │  bind() │            │
│        │                             └────┬────┘            │
│        │                             ┌────┴────┐            │
│        │                             │ listen()│            │
│        │                             └────┬────┘            │
│   ┌────┴────┐      연결 요청         ┌────┴────┐            │
│   │connect()│ ──────────────────────▶│ accept()│            │
│   └────┬────┘                        └────┬────┘            │
│        │                                  │                 │
│   ┌────┴────┐      데이터 송수신     ┌────┴────┐            │
│   │  send() │ ◀────────────────────▶│  recv() │            │
│   │  recv() │                        │  send() │            │
│   └────┬────┘                        └────┬────┘            │
│        │                                  │                 │
│   ┌────┴────┐                        ┌────┴────┐            │
│   │ close() │                        │ close() │            │
│   └─────────┘                        └─────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 TCP 서버

```python
#!/usr/bin/env python3
"""TCP 서버 - 센서 데이터 수신"""

import socket
import json
from datetime import datetime

HOST = '0.0.0.0'  # 모든 인터페이스
PORT = 9999

def start_tcp_server():
    """TCP 서버 시작"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # 주소 재사용 허용
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server.bind((HOST, PORT))
        server.listen(5)

        print(f"TCP 서버 시작: {HOST}:{PORT}")

        while True:
            client, address = server.accept()
            print(f"클라이언트 연결: {address}")

            with client:
                while True:
                    data = client.recv(1024)
                    if not data:
                        break

                    try:
                        # JSON 데이터 파싱
                        message = json.loads(data.decode('utf-8'))
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] 수신: {message}")

                        # 응답 전송
                        response = {
                            "status": "ok",
                            "received": message.get("sensor_id")
                        }
                        client.sendall(json.dumps(response).encode('utf-8'))

                    except json.JSONDecodeError:
                        print(f"잘못된 JSON: {data}")

            print(f"클라이언트 연결 종료: {address}")

if __name__ == "__main__":
    start_tcp_server()
```

### 2.3 TCP 클라이언트

```python
#!/usr/bin/env python3
"""TCP 클라이언트 - 센서 데이터 전송"""

import socket
import json
import time
import random

SERVER_HOST = '192.168.1.100'  # 서버 IP
SERVER_PORT = 9999

def send_sensor_data():
    """센서 데이터를 서버로 전송"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((SERVER_HOST, SERVER_PORT))
        print(f"서버 연결: {SERVER_HOST}:{SERVER_PORT}")

        sensor_id = "temp_sensor_01"

        try:
            while True:
                # 센서 데이터 생성
                data = {
                    "sensor_id": sensor_id,
                    "temperature": round(random.uniform(20, 30), 1),
                    "humidity": round(random.uniform(40, 70), 1),
                    "timestamp": time.time()
                }

                # 전송
                message = json.dumps(data).encode('utf-8')
                client.sendall(message)
                print(f"전송: {data}")

                # 응답 수신
                response = client.recv(1024)
                if response:
                    print(f"응답: {response.decode('utf-8')}")

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n연결 종료")

if __name__ == "__main__":
    send_sensor_data()
```

### 2.4 UDP 소켓

```python
#!/usr/bin/env python3
"""UDP 소켓 통신 (빠른 센서 데이터 전송)"""

import socket
import json
import time

# === UDP 서버 ===
def udp_server(port: int = 9998):
    """UDP 서버"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))

    print(f"UDP 서버 시작: 포트 {port}")

    while True:
        data, addr = sock.recvfrom(1024)
        message = json.loads(data.decode('utf-8'))
        print(f"[{addr}] {message}")

# === UDP 클라이언트 ===
def udp_client(server_ip: str, port: int = 9998):
    """UDP 클라이언트"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sensor_data = {
        "sensor_id": "motion_01",
        "motion_detected": True,
        "timestamp": time.time()
    }

    message = json.dumps(sensor_data).encode('utf-8')
    sock.sendto(message, (server_ip, port))
    print(f"전송 완료: {sensor_data}")
    sock.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        udp_server()
    else:
        udp_client('192.168.1.100')
```

---

## 3. ESP32 WiFi 개요

### 3.1 ESP32와 라즈베리파이 비교

| 특성 | ESP32 | Raspberry Pi |
|------|-------|--------------|
| **프로세서** | Xtensa 240MHz | ARM 1.5GHz |
| **RAM** | 520KB | 1-8GB |
| **OS** | FreeRTOS/없음 | Linux |
| **언어** | C/C++, MicroPython | Python, 모든 언어 |
| **WiFi** | 내장 | 내장 (Pi 3+) |
| **전력** | 낮음 (80mA) | 높음 (700mA+) |
| **용도** | 센서 노드 | 게이트웨이, 엣지 |

### 3.2 ESP32 MicroPython WiFi 예제

```python
# ESP32용 MicroPython 코드
# (참고용 - 라즈베리파이에서는 실행 불가)

import network
import time

def connect_wifi(ssid: str, password: str) -> str:
    """ESP32 WiFi 연결"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if not wlan.isconnected():
        print(f'WiFi 연결 중: {ssid}')
        wlan.connect(ssid, password)

        # 연결 대기
        timeout = 10
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f'연결됨! IP: {ip}')
        return ip
    else:
        print('연결 실패')
        return None

# 사용
# ip = connect_wifi("MySSID", "MyPassword")
```

### 3.3 라즈베리파이 - ESP32 통신 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│              라즈베리파이 - ESP32 통신                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐         WiFi         ┌──────────────┐    │
│   │              │◀────────────────────▶│              │    │
│   │  Raspberry   │                      │    ESP32     │    │
│   │     Pi       │                      │   센서 노드  │    │
│   │              │                      │              │    │
│   │  - MQTT      │       TCP/UDP        │  - 온도 센서 │    │
│   │    Broker    │◀────────────────────▶│  - 습도 센서 │    │
│   │  - 데이터    │                      │  - 모션 센서 │    │
│   │    수집      │       HTTP           │              │    │
│   │  - 분석      │◀────────────────────▶│  저전력      │    │
│   │              │                      │  동작        │    │
│   └──────────────┘                      └──────────────┘    │
│        │                                      │             │
│        │                                      │             │
│        ▼                                      ▼             │
│   ┌──────────────┐                      ┌──────────────┐    │
│   │   클라우드   │                      │   배터리     │    │
│   │   AWS/GCP    │                      │   동작 가능  │    │
│   └──────────────┘                      └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 네트워크 스캔 및 모니터링

### 4.1 네트워크 장치 스캔

```python
#!/usr/bin/env python3
"""네트워크 장치 스캔"""

import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
import socket

def get_local_network() -> str:
    """로컬 네트워크 주소 반환"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    # 네트워크 주소 추출 (예: 192.168.1.0/24)
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def ping_host(ip: str) -> dict | None:
    """단일 호스트 핑"""
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', ip],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return {'ip': ip, 'status': 'up'}
    except:
        pass
    return None

def scan_network(network: str = None) -> list:
    """네트워크 전체 스캔"""
    if network is None:
        network = get_local_network()

    # IP 범위 생성
    base = '.'.join(network.split('.')[:-1])
    ips = [f"{base}.{i}" for i in range(1, 255)]

    print(f"스캔 중: {network}")

    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for result in executor.map(ping_host, ips):
            if result:
                results.append(result)
                print(f"  발견: {result['ip']}")

    return results

def get_hostname(ip: str) -> str:
    """IP 주소에서 호스트명 조회"""
    try:
        return socket.gethostbyaddr(ip)[0]
    except:
        return "Unknown"

if __name__ == "__main__":
    devices = scan_network()
    print(f"\n=== 발견된 장치: {len(devices)}개 ===")

    for device in devices:
        hostname = get_hostname(device['ip'])
        print(f"  {device['ip']:15} - {hostname}")
```

### 4.2 포트 스캔

```python
#!/usr/bin/env python3
"""간단한 포트 스캔"""

import socket
from concurrent.futures import ThreadPoolExecutor

COMMON_PORTS = {
    22: 'SSH',
    80: 'HTTP',
    443: 'HTTPS',
    1883: 'MQTT',
    3306: 'MySQL',
    5432: 'PostgreSQL',
    8080: 'HTTP-Alt',
    8883: 'MQTT-TLS'
}

def check_port(target: str, port: int) -> dict | None:
    """포트 열림 확인"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)

    try:
        result = sock.connect_ex((target, port))
        if result == 0:
            return {
                'port': port,
                'status': 'open',
                'service': COMMON_PORTS.get(port, 'unknown')
            }
    except:
        pass
    finally:
        sock.close()

    return None

def scan_ports(target: str, ports: list = None) -> list:
    """여러 포트 스캔"""
    if ports is None:
        ports = list(COMMON_PORTS.keys())

    print(f"포트 스캔: {target}")

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_port, target, port): port for port in ports}
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
                print(f"  포트 {result['port']} ({result['service']}): OPEN")

    return results

if __name__ == "__main__":
    target = input("스캔할 IP 주소: ")
    scan_ports(target)
```

---

## 5. HTTP 클라이언트

### 5.1 requests 라이브러리

```python
#!/usr/bin/env python3
"""HTTP 클라이언트 - 센서 데이터 전송"""

import requests
import time
import json

API_BASE = "http://192.168.1.100:5000/api"

def send_sensor_data(sensor_id: str, data: dict) -> bool:
    """센서 데이터 POST 전송"""
    url = f"{API_BASE}/sensors/{sensor_id}/data"

    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )

        if response.status_code == 201:
            print(f"데이터 전송 성공: {data}")
            return True
        else:
            print(f"전송 실패: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"네트워크 오류: {e}")
        return False

def get_sensor_config(sensor_id: str) -> dict | None:
    """센서 설정 조회"""
    url = f"{API_BASE}/sensors/{sensor_id}/config"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"조회 실패: {e}")

    return None

def periodic_reporting(sensor_id: str, interval: int = 10):
    """주기적 데이터 리포팅"""
    import random

    print(f"센서 {sensor_id} 리포팅 시작 (간격: {interval}초)")

    while True:
        data = {
            "temperature": round(random.uniform(20, 30), 1),
            "humidity": round(random.uniform(40, 70), 1),
            "timestamp": int(time.time())
        }

        send_sensor_data(sensor_id, data)
        time.sleep(interval)

if __name__ == "__main__":
    periodic_reporting("sensor_001", 10)
```

### 5.2 비동기 HTTP 클라이언트

```python
#!/usr/bin/env python3
"""비동기 HTTP 클라이언트 (aiohttp)"""

import asyncio
import aiohttp
import time

API_BASE = "http://192.168.1.100:5000/api"

async def send_data_async(session: aiohttp.ClientSession,
                          sensor_id: str,
                          data: dict) -> bool:
    """비동기 데이터 전송"""
    url = f"{API_BASE}/sensors/{sensor_id}/data"

    try:
        async with session.post(url, json=data) as response:
            if response.status == 201:
                return True
    except aiohttp.ClientError as e:
        print(f"오류: {e}")

    return False

async def batch_send(sensors: list, data_list: list):
    """여러 센서 데이터 동시 전송"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_data_async(session, sensor, data)
            for sensor, data in zip(sensors, data_list)
        ]

        results = await asyncio.gather(*tasks)
        success = sum(results)
        print(f"전송 완료: {success}/{len(results)}")

if __name__ == "__main__":
    sensors = ["sensor_001", "sensor_002", "sensor_003"]
    data_list = [
        {"temperature": 25.5, "timestamp": time.time()},
        {"temperature": 26.0, "timestamp": time.time()},
        {"temperature": 24.8, "timestamp": time.time()}
    ]

    asyncio.run(batch_send(sensors, data_list))
```

### 5.3 HTTP 클라이언트 with 재시도

```python
#!/usr/bin/env python3
"""재시도 로직이 있는 HTTP 클라이언트"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

def create_session_with_retry(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 503, 504)
) -> requests.Session:
    """재시도 설정이 된 세션 생성"""
    session = requests.Session()

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

class IoTHttpClient:
    """IoT용 HTTP 클라이언트"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = create_session_with_retry()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'IoT-Sensor/1.0'
        })

    def send_data(self, endpoint: str, data: dict) -> dict:
        """데이터 전송"""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.post(url, json=data, timeout=10)
            response.raise_for_status()
            return {"success": True, "data": response.json()}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_config(self, endpoint: str) -> dict:
        """설정 조회"""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return {"success": True, "data": response.json()}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def close(self):
        """세션 종료"""
        self.session.close()

# 사용 예
if __name__ == "__main__":
    client = IoTHttpClient("http://192.168.1.100:5000/api")

    result = client.send_data("sensors/001/data", {
        "temperature": 25.5,
        "timestamp": time.time()
    })

    print(result)
    client.close()
```

---

## 연습 문제

### 연습 1: WiFi 신호 모니터

WiFi 신호 강도와 연결 품질을 지속적으로 모니터링하는 Python 스크립트를 작성하세요:

1. `subprocess`를 사용하여 10초마다 `iwconfig wlan0`을 호출하고, `re`를 사용하여 출력에서 신호 레벨(dBm)과 링크 품질(link quality)을 파싱합니다.
2. 각 읽기를 CSV 파일에 기록합니다. 열 구성: `timestamp`, `ssid`, `signal_dbm`, `quality`.
3. 신호가 -70 dBm 아래로 떨어지면 콘솔에 경고를 출력합니다.
4. 네트워크가 완전히 끊기면(SSID가 반환되지 않으면) 끊김 시각을 기록합니다. 재연결되면 재연결 시각과 중단 지속 시간을 기록합니다.

### 연습 2: 다중 클라이언트 TCP 센서 서버

Raspberry Pi에서 여러 센서 클라이언트를 동시에 받는 TCP 서버를 만드세요:

1. 섹션 2.1의 `TCPServer` 클래스를 구현하고, `threading.Thread`를 사용하여 동시 클라이언트를 지원하도록 확장합니다.
2. 각 클라이언트는 `{"sensor_id": "...", "value": ..., "unit": "..."}` 형식의 JSON 메시지를 전송합니다.
3. 서버는 `sensor_id`별 최신 읽기를 메모리 딕셔너리에 유지하고 30초마다 상태 요약을 출력합니다.
4. 2초 간격으로 시뮬레이션된 온도 읽기 10개를 전송한 후 정상적으로 연결을 끊는 `TCPClient` 스크립트를 작성합니다.
5. Pi에서 서버를 실행하고 최소 두 개의 클라이언트 인스턴스로 테스트합니다.

### 연습 3: 재시도 로직이 포함된 HTTP 센서 리포터

섹션 3.2의 `SensorDataSender` 클래스를 프로덕션(production) 수준으로 확장하세요:

1. 최대 500개 항목의 `collections.deque`를 사용하여 영구 로컬 큐(queue)를 추가합니다.
2. HTTP POST가 실패하면(예외 또는 200이 아닌 상태) 데이터를 버리는 대신 큐에 다시 넣습니다.
3. 지수 백오프(Exponential Backoff) 재시도를 구현합니다: 첫 번째 재시도는 5초 후, 그 다음은 10초, 20초, 최대 5분까지.
4. 연결이 복원되면 10개씩 배치(batch)로 큐에 쌓인 읽기를 주기적으로 플러시(flush)하는 별도 스레드(thread)를 추가합니다.
5. Python의 `logging.handlers.RotatingFileHandler`를 사용하여 모든 재시도 시도와 배치 전송을 로테이션 로그 파일에 기록합니다.

### 연습 4: 서비스 감지 기능이 있는 네트워크 장치 스캐너

섹션 4의 `NetworkScanner`를 완전한 IoT 장치 탐색 도구로 확장하세요:

1. 병렬 `socket` 연결을 사용하여 서브넷(기본값: `192.168.1.0/24`)에서 활성 호스트를 스캔합니다.
2. 각 활성 호스트에 대해 섹션 4에 나열된 일반적인 IoT 포트를 스캔합니다: 22(SSH), 80(HTTP), 443(HTTPS), 1883(MQTT), 8883(MQTT-TLS), 8080(HTTP-Alt), 23(Telnet).
3. `socket.gethostbyaddr()`를 사용하여 각 발견된 호스트의 역방향 DNS 조회를 시도합니다.
4. 형식화된 테이블을 출력합니다: IP 주소, 호스트명, 열린 서비스의 쉼표로 구분된 목록.
5. 5분마다 스캔을 실행하도록 예약하고, 이전 스캔에 없던 새 장치가 네트워크에 나타나면 경고(콘솔 출력)를 표시합니다.

### 연습 5: Raspberry Pi vs ESP32 프로토콜 벤치마크

TCP와 UDP 통신 오버헤드(overhead)를 실용적으로 비교하세요:

1. Raspberry Pi에서 UDP 서버(포트 9999)를 설정합니다. 수신된 메시지를 서버 측 타임스탬프와 함께 발신자에게 에코(echo)합니다.
2. 100바이트 메시지 1000개를 전송하고, 각 메시지의 왕복 시간(RTT: Round-Trip Time)을 기록하며, 최소/최대/평균/표준편차 RTT를 계산하는 UDP 클라이언트를 작성합니다.
3. 동등한 TCP 서버와 클라이언트를 만들어 실험을 반복합니다.
4. 결과를 비교합니다: 어느 프로토콜이 평균 RTT가 낮은가요? 어느 것이 더 일관된 지연(낮은 표준편차)을 보이나요? 어떤 IoT 시나리오에서 각각을 선호하겠나요?
5. 결과를 설명하고 섹션 5의 Raspberry Pi vs ESP32 비교와 연결하는 200단어 요약을 작성합니다.

---

## 다음 단계

- [BLE 연결](05_BLE_Connectivity.md): BLE 통신으로 저전력 센서 연결
- [MQTT 프로토콜](06_MQTT_Protocol.md): MQTT로 효율적인 IoT 메시징

---

*최종 업데이트: 2026-02-01*
