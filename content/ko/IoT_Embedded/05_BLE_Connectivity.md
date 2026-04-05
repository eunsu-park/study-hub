# 05. BLE 연결

**이전**: [WiFi 네트워킹](./04_WiFi_Networking.md) | **다음**: [MQTT 프로토콜](./06_MQTT_Protocol.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. BLE와 클래식 Bluetooth를 비교하고 각각의 적합한 사용 시점을 설명할 수 있다
2. BLE 프로토콜 스택과 GATT 계층 구조를 설명할 수 있다
3. bleak 라이브러리를 사용하여 BLE 장치를 스캔하고 연결할 수 있다
4. BLE 특성(characteristics)을 프로그래밍으로 읽고, 쓰고, 구독할 수 있다
5. 표준 BLE 서비스와 해당 UUID를 식별할 수 있다
6. 자동 재연결 로직을 갖춘 BLE 센서 모니터를 구축할 수 있다

---

피트니스 트래커, 환경 모니터, 스마트홈 기기 등 많은 IoT 센서는 WiFi가 아닌 Bluetooth Low Energy(저전력 블루투스)로 통신합니다. BLE는 전력을 훨씬 적게 사용하고, 밀리초 단위로 연결되며, 별도의 네트워크 인프라 없이 동작합니다. BLE 프로그래밍을 익히면 배터리로 구동되는 방대한 무선 센서 생태계를 IoT 프로젝트에 통합할 수 있습니다.

---

## 1. BLE 프로토콜 개요

### 1.1 BLE vs 클래식 Bluetooth

| 특성 | BLE (Bluetooth Low Energy) | 클래식 Bluetooth |
|------|---------------------------|------------------|
| **전력 소비** | 매우 낮음 | 높음 |
| **데이터 전송률** | 1-2 Mbps | 1-3 Mbps |
| **범위** | ~100m | ~100m |
| **지연 시간** | ~6ms | ~100ms |
| **페어링** | 간단/자동 | 복잡 |
| **용도** | IoT 센서, 웨어러블 | 오디오, 파일 전송 |

### 1.2 BLE 프로토콜 스택

```
┌─────────────────────────────────────────────────────────────┐
│                    BLE 프로토콜 스택                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Application                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                GAP (Generic Access Profile)          │    │
│  │           디바이스 검색, 연결, 보안                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             GATT (Generic Attribute Profile)         │    │
│  │              서비스, 특성, 데이터 교환                │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                 ATT (Attribute Protocol)             │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    L2CAP                             │    │
│  │           논리 링크 제어 및 적응 프로토콜             │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Link Layer + Physical Layer             │    │
│  │                  무선 통신 처리                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 BLE 역할

```python
# BLE 역할 정의
ble_roles = {
    "Central (Master)": {
        "description": "다른 장치를 스캔하고 연결을 시작",
        "example": "스마트폰, 라즈베리파이",
        "behavior": ["스캔", "연결 요청", "데이터 요청"]
    },
    "Peripheral (Slave)": {
        "description": "광고하고 연결을 기다림",
        "example": "센서, 비콘, 웨어러블",
        "behavior": ["광고", "연결 대기", "데이터 제공"]
    },
    "Observer": {
        "description": "광고 패킷만 수신 (연결 없음)",
        "example": "비콘 리더",
        "behavior": ["스캔만"]
    },
    "Broadcaster": {
        "description": "광고 패킷만 송신 (연결 없음)",
        "example": "비콘",
        "behavior": ["광고만"]
    }
}
```

---

## 2. GATT 구조

### 2.1 GATT 계층 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    GATT 계층 구조                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   GATT Server (Peripheral)                                   │
│   │                                                          │
│   ├── Profile                                                │
│   │   │                                                      │
│   │   ├── Service (UUID: 0x180F - Battery Service)          │
│   │   │   │                                                  │
│   │   │   └── Characteristic (UUID: 0x2A19 - Battery Level)│
│   │   │       ├── Value: 85 (0-100%)                        │
│   │   │       ├── Properties: Read, Notify                  │
│   │   │       └── Descriptors                               │
│   │   │           └── CCCD (Client Config Descriptor)       │
│   │   │                                                      │
│   │   └── Service (UUID: 0x181A - Environmental Sensing)    │
│   │       │                                                  │
│   │       ├── Characteristic: Temperature (0x2A6E)          │
│   │       │   └── Value: 25.5°C                             │
│   │       │                                                  │
│   │       └── Characteristic: Humidity (0x2A6F)             │
│   │           └── Value: 60%                                 │
│   │                                                          │
│   └── ...                                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 표준 UUID

```python
# 표준 BLE 서비스 UUID (16-bit)
standard_services = {
    "0x1800": "Generic Access",
    "0x1801": "Generic Attribute",
    "0x180A": "Device Information",
    "0x180F": "Battery Service",
    "0x181A": "Environmental Sensing",
    "0x180D": "Heart Rate",
}

# 표준 특성 UUID
standard_characteristics = {
    "0x2A00": "Device Name",
    "0x2A19": "Battery Level",
    "0x2A6E": "Temperature",
    "0x2A6F": "Humidity",
    "0x2A37": "Heart Rate Measurement",
}

# 16-bit UUID를 128-bit로 변환
def uuid_16_to_128(uuid_16: str) -> str:
    """16-bit UUID를 128-bit BLE 기본 UUID로 변환"""
    base_uuid = "00000000-0000-1000-8000-00805f9b34fb"
    uuid_16_clean = uuid_16.replace("0x", "").lower()
    return f"0000{uuid_16_clean}{base_uuid[8:]}"

# 예: 0x180F -> 0000180f-0000-1000-8000-00805f9b34fb
```

### 2.3 특성 속성

```python
# 특성 속성 (Properties)
characteristic_properties = {
    "Broadcast": 0x01,       # 광고에 포함 가능
    "Read": 0x02,            # 읽기 가능
    "Write No Response": 0x04,  # 응답 없이 쓰기
    "Write": 0x08,           # 응답 있는 쓰기
    "Notify": 0x10,          # 알림 (응답 없음)
    "Indicate": 0x20,        # 표시 (응답 있음)
}

def parse_properties(props: int) -> list:
    """속성 비트마스크를 리스트로 변환"""
    result = []
    for name, value in characteristic_properties.items():
        if props & value:
            result.append(name)
    return result

# 예: parse_properties(0x12) -> ['Read', 'Notify']
```

---

## 3. bleak 라이브러리

### 3.1 설치 및 설정

```bash
# bleak 설치
pip install bleak

# Linux에서 추가 설정 (bluetoothctl 접근 권한)
sudo usermod -a -G bluetooth $USER

# D-Bus 서비스 확인
sudo systemctl status bluetooth
```

### 3.2 BLE 장치 스캔

```python
#!/usr/bin/env python3
"""BLE 장치 스캔 (bleak)"""

import asyncio
from bleak import BleakScanner

async def scan_devices(timeout: float = 10.0):
    """주변 BLE 장치 스캔"""
    print(f"BLE 장치 스캔 중... ({timeout}초)")

    devices = await BleakScanner.discover(timeout=timeout)

    print(f"\n발견된 장치: {len(devices)}개\n")

    for device in devices:
        rssi = device.rssi if hasattr(device, 'rssi') else 'N/A'
        print(f"  이름: {device.name or 'Unknown'}")
        print(f"  주소: {device.address}")
        print(f"  RSSI: {rssi} dBm")
        print()

    return devices

async def scan_with_filter(name_filter: str = None):
    """이름 필터로 스캔"""
    devices = await BleakScanner.discover()

    if name_filter:
        devices = [d for d in devices if d.name and name_filter.lower() in d.name.lower()]

    return devices

async def continuous_scan(callback=None, duration: float = 30.0):
    """연속 스캔 (장치 발견 시 콜백)"""
    def detection_callback(device, advertisement_data):
        print(f"발견: {device.name} ({device.address})")
        if callback:
            callback(device, advertisement_data)

    scanner = BleakScanner(detection_callback=detection_callback)

    print(f"연속 스캔 시작 ({duration}초)")
    await scanner.start()
    await asyncio.sleep(duration)
    await scanner.stop()

if __name__ == "__main__":
    asyncio.run(scan_devices(10))
```

### 3.3 BLE 장치 연결

```python
#!/usr/bin/env python3
"""BLE 장치 연결 및 서비스 탐색"""

import asyncio
from bleak import BleakClient, BleakScanner

async def connect_and_explore(address: str):
    """장치 연결 후 서비스/특성 탐색"""
    print(f"연결 중: {address}")

    async with BleakClient(address) as client:
        print(f"연결됨! MTU: {client.mtu_size}")

        # 서비스 탐색
        for service in client.services:
            print(f"\n서비스: {service.uuid}")
            print(f"  설명: {service.description}")

            # 특성 탐색
            for char in service.characteristics:
                print(f"    특성: {char.uuid}")
                print(f"      속성: {char.properties}")

                # 읽기 가능하면 값 읽기
                if "read" in char.properties:
                    try:
                        value = await client.read_gatt_char(char.uuid)
                        print(f"      값: {value}")
                    except Exception as e:
                        print(f"      읽기 실패: {e}")

async def find_and_connect(name_filter: str):
    """이름으로 장치 찾아 연결"""
    print(f"장치 검색: '{name_filter}'")

    device = await BleakScanner.find_device_by_name(name_filter)

    if device:
        print(f"장치 발견: {device.address}")
        await connect_and_explore(device.address)
    else:
        print("장치를 찾을 수 없습니다.")

if __name__ == "__main__":
    # MAC 주소로 직접 연결
    # asyncio.run(connect_and_explore("AA:BB:CC:DD:EE:FF"))

    # 이름으로 검색 후 연결
    asyncio.run(find_and_connect("Temperature"))
```

---

## 4. 센서 데이터 수신

### 4.1 특성 값 읽기

```python
#!/usr/bin/env python3
"""BLE 특성 값 읽기"""

import asyncio
from bleak import BleakClient
import struct

# 표준 UUID
BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
TEMPERATURE_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"

async def read_battery_level(address: str) -> int | None:
    """배터리 레벨 읽기"""
    async with BleakClient(address) as client:
        try:
            data = await client.read_gatt_char(BATTERY_LEVEL_UUID)
            # 배터리 레벨은 1바이트 (0-100%)
            return data[0]
        except Exception as e:
            print(f"읽기 실패: {e}")
            return None

async def read_temperature(address: str) -> float | None:
    """온도 읽기 (IEEE 11073 형식)"""
    async with BleakClient(address) as client:
        try:
            data = await client.read_gatt_char(TEMPERATURE_UUID)
            # 온도는 16-bit 부호있는 정수 (0.01도 단위)
            temp_raw = struct.unpack('<h', data[:2])[0]
            return temp_raw * 0.01
        except Exception as e:
            print(f"읽기 실패: {e}")
            return None

async def read_multiple_chars(address: str, char_uuids: list) -> dict:
    """여러 특성 한 번에 읽기"""
    results = {}

    async with BleakClient(address) as client:
        for uuid in char_uuids:
            try:
                data = await client.read_gatt_char(uuid)
                results[uuid] = data
            except Exception as e:
                results[uuid] = None
                print(f"UUID {uuid} 읽기 실패: {e}")

    return results

if __name__ == "__main__":
    address = "AA:BB:CC:DD:EE:FF"

    # 배터리 레벨
    level = asyncio.run(read_battery_level(address))
    if level is not None:
        print(f"배터리: {level}%")

    # 온도
    temp = asyncio.run(read_temperature(address))
    if temp is not None:
        print(f"온도: {temp}°C")
```

### 4.2 알림(Notification) 수신

```python
#!/usr/bin/env python3
"""BLE 알림 수신 (실시간 센서 데이터)"""

import asyncio
from bleak import BleakClient
from datetime import datetime

# 예시 UUID (장치에 따라 다름)
HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

def notification_handler(sender, data):
    """알림 수신 콜백"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 수신 ({sender}): {data.hex()}")

    # 데이터 파싱 (예: Heart Rate Measurement)
    flags = data[0]
    if flags & 0x01:  # 16-bit heart rate
        hr = int.from_bytes(data[1:3], 'little')
    else:  # 8-bit heart rate
        hr = data[1]

    print(f"  심박수: {hr} bpm")

async def subscribe_notifications(address: str, char_uuid: str, duration: float = 60):
    """알림 구독"""
    async with BleakClient(address) as client:
        print(f"연결됨: {address}")

        # 알림 시작
        await client.start_notify(char_uuid, notification_handler)
        print(f"알림 구독 시작: {char_uuid}")

        # 지정된 시간 동안 수신
        await asyncio.sleep(duration)

        # 알림 중지
        await client.stop_notify(char_uuid)
        print("알림 구독 종료")

async def subscribe_multiple(address: str, char_uuids: list, duration: float = 60):
    """여러 특성 알림 구독"""
    async with BleakClient(address) as client:
        print(f"연결됨: {address}")

        for uuid in char_uuids:
            await client.start_notify(uuid, notification_handler)
            print(f"구독: {uuid}")

        await asyncio.sleep(duration)

        for uuid in char_uuids:
            await client.stop_notify(uuid)

if __name__ == "__main__":
    asyncio.run(subscribe_notifications("AA:BB:CC:DD:EE:FF", HEART_RATE_UUID, 30))
```

### 4.3 특성 값 쓰기

```python
#!/usr/bin/env python3
"""BLE 특성에 값 쓰기"""

import asyncio
from bleak import BleakClient

async def write_characteristic(address: str, char_uuid: str, data: bytes):
    """특성에 값 쓰기 (응답 있음)"""
    async with BleakClient(address) as client:
        await client.write_gatt_char(char_uuid, data, response=True)
        print(f"쓰기 완료: {data.hex()}")

async def write_without_response(address: str, char_uuid: str, data: bytes):
    """특성에 값 쓰기 (응답 없음 - 빠름)"""
    async with BleakClient(address) as client:
        await client.write_gatt_char(char_uuid, data, response=False)
        print(f"쓰기 전송: {data.hex()}")

async def toggle_led(address: str, led_uuid: str, state: bool):
    """LED 제어 예제"""
    data = bytes([0x01 if state else 0x00])
    await write_characteristic(address, led_uuid, data)

async def set_sensor_interval(address: str, config_uuid: str, interval_ms: int):
    """센서 측정 주기 설정"""
    # 2바이트 리틀엔디안
    data = interval_ms.to_bytes(2, 'little')
    await write_characteristic(address, config_uuid, data)
    print(f"측정 주기 설정: {interval_ms}ms")

if __name__ == "__main__":
    # LED 토글
    asyncio.run(toggle_led("AA:BB:CC:DD:EE:FF", "custom-led-uuid", True))
```

---

## 5. 종합 예제: BLE 센서 모니터

```python
#!/usr/bin/env python3
"""BLE 환경 센서 모니터"""

import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime
import struct

class BLESensorMonitor:
    """BLE 환경 센서 모니터링 클래스"""

    # 표준 Environmental Sensing 서비스
    ENV_SENSING_SERVICE = "0000181a-0000-1000-8000-00805f9b34fb"
    TEMPERATURE_CHAR = "00002a6e-0000-1000-8000-00805f9b34fb"
    HUMIDITY_CHAR = "00002a6f-0000-1000-8000-00805f9b34fb"

    def __init__(self, device_address: str = None, device_name: str = None):
        self.device_address = device_address
        self.device_name = device_name
        self.client = None
        self.data_buffer = []

    async def find_device(self) -> str | None:
        """장치 검색"""
        if self.device_address:
            return self.device_address

        if self.device_name:
            print(f"장치 검색: {self.device_name}")
            device = await BleakScanner.find_device_by_name(self.device_name)
            if device:
                self.device_address = device.address
                return device.address

        return None

    def _handle_temperature(self, sender, data):
        """온도 데이터 핸들러"""
        # 0.01도 단위 16-bit 정수
        temp = struct.unpack('<h', data[:2])[0] * 0.01
        timestamp = datetime.now()

        self.data_buffer.append({
            'type': 'temperature',
            'value': temp,
            'unit': '°C',
            'timestamp': timestamp
        })

        print(f"[{timestamp.strftime('%H:%M:%S')}] 온도: {temp:.2f}°C")

    def _handle_humidity(self, sender, data):
        """습도 데이터 핸들러"""
        # 0.01% 단위 16-bit 정수
        humidity = struct.unpack('<H', data[:2])[0] * 0.01
        timestamp = datetime.now()

        self.data_buffer.append({
            'type': 'humidity',
            'value': humidity,
            'unit': '%',
            'timestamp': timestamp
        })

        print(f"[{timestamp.strftime('%H:%M:%S')}] 습도: {humidity:.2f}%")

    async def start_monitoring(self, duration: float = 60):
        """모니터링 시작"""
        address = await self.find_device()
        if not address:
            print("장치를 찾을 수 없습니다.")
            return

        print(f"연결 중: {address}")

        async with BleakClient(address) as client:
            self.client = client
            print("연결됨!")

            # 서비스 확인
            services = client.services
            has_env_sensing = any(
                self.ENV_SENSING_SERVICE in str(s.uuid)
                for s in services
            )

            if not has_env_sensing:
                print("Environmental Sensing 서비스를 찾을 수 없습니다.")
                print("사용 가능한 서비스:")
                for s in services:
                    print(f"  - {s.uuid}")
                return

            # 알림 구독
            try:
                await client.start_notify(self.TEMPERATURE_CHAR, self._handle_temperature)
                print("온도 알림 구독 시작")
            except Exception as e:
                print(f"온도 구독 실패: {e}")

            try:
                await client.start_notify(self.HUMIDITY_CHAR, self._handle_humidity)
                print("습도 알림 구독 시작")
            except Exception as e:
                print(f"습도 구독 실패: {e}")

            print(f"\n모니터링 중... ({duration}초)")
            await asyncio.sleep(duration)

            # 정리
            await client.stop_notify(self.TEMPERATURE_CHAR)
            await client.stop_notify(self.HUMIDITY_CHAR)

        print("\n=== 모니터링 종료 ===")
        print(f"수집된 데이터: {len(self.data_buffer)}개")

    async def read_once(self) -> dict:
        """한 번 읽기"""
        address = await self.find_device()
        if not address:
            return {}

        async with BleakClient(address) as client:
            result = {}

            try:
                data = await client.read_gatt_char(self.TEMPERATURE_CHAR)
                result['temperature'] = struct.unpack('<h', data[:2])[0] * 0.01
            except:
                pass

            try:
                data = await client.read_gatt_char(self.HUMIDITY_CHAR)
                result['humidity'] = struct.unpack('<H', data[:2])[0] * 0.01
            except:
                pass

            return result

    def get_summary(self) -> dict:
        """수집된 데이터 요약"""
        if not self.data_buffer:
            return {}

        temps = [d['value'] for d in self.data_buffer if d['type'] == 'temperature']
        humids = [d['value'] for d in self.data_buffer if d['type'] == 'humidity']

        summary = {}

        if temps:
            summary['temperature'] = {
                'min': min(temps),
                'max': max(temps),
                'avg': sum(temps) / len(temps),
                'count': len(temps)
            }

        if humids:
            summary['humidity'] = {
                'min': min(humids),
                'max': max(humids),
                'avg': sum(humids) / len(humids),
                'count': len(humids)
            }

        return summary

if __name__ == "__main__":
    # 장치 이름으로 검색
    monitor = BLESensorMonitor(device_name="EnvSensor")

    # 또는 MAC 주소로 직접 지정
    # monitor = BLESensorMonitor(device_address="AA:BB:CC:DD:EE:FF")

    try:
        asyncio.run(monitor.start_monitoring(duration=30))
    except KeyboardInterrupt:
        print("\n사용자 중단")

    # 요약 출력
    summary = monitor.get_summary()
    if summary:
        print("\n=== 데이터 요약 ===")
        for sensor, stats in summary.items():
            print(f"{sensor}:")
            print(f"  최소: {stats['min']:.2f}")
            print(f"  최대: {stats['max']:.2f}")
            print(f"  평균: {stats['avg']:.2f}")
```

---

## 6. BLE 5.0 기능과 BLE Mesh(BLE 5.0 Features and BLE Mesh)

Bluetooth Low Energy는 4.0 사양 이후 크게 발전해 왔습니다. BLE 5.0(2016년 출시)은 속도, 범위, 브로드캐스트 용량에서 주요 개선을 도입했으며, BLE Mesh(Mesh Profile 1.0, 2017년 출시)는 다대다(many-to-many) 통신 토폴로지를 추가했습니다. 이러한 발전을 이해하는 것은 넓은 영역을 커버하거나 수백 대의 기기를 연결해야 하는 현대 IoT 시스템 설계에 필수적입니다.

### BLE 5.0 핵심 개선 사항

BLE 5.0은 BLE 4.2 대비 세 가지 주요 개선을 도입했습니다:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BLE 5.0 개선 사항                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  2배 속도    │    │  4배 범위    │    │ 8배 브로드   │              │
│  │             │    │             │    │ 캐스트 데이터 │              │
│  │  2 Mbps PHY │    │ Coded PHY   │    │ Extended    │              │
│  │  (기존 1 Mbps)│   │ (LE Coded)  │    │ Advertising │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                      │
│  참고: 이들은 트레이드오프 — 한 번에 하나의 PHY 모드 선택            │
│  - 고속 (2M PHY): 짧은 범위, 높은 처리량                             │
│  - 장거리 (Coded PHY): 낮은 속도, 훨씬 먼 거리                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 장거리 모드(Long Range Mode, Coded PHY)

Coded PHY는 전방 오류 정정(Forward Error Correction, FEC)을 사용하여 데이터 전송률을 희생하고 범위를 확장합니다:

```
┌────────────────────────────────────────────────────────────────┐
│  PHY 모드            │ 데이터 전송률 │ 범위         │ 사용 사례  │
├────────────────────────────────────────────────────────────────┤
│  LE 1M (BLE 4.x)    │ 1 Mbps       │ ~50-100m    │ 일반 IoT   │
│  LE 2M (BLE 5.0)    │ 2 Mbps       │ ~30-50m     │ 높은 처리량 │
│  LE Coded S=2        │ 500 kbps     │ ~200m       │ 중거리     │
│  LE Coded S=8        │ 125 kbps     │ ~400m+      │ 장거리     │
└────────────────────────────────────────────────────────────────┘

S=2: 각 비트를 2개 심볼로 인코딩 (중간 수준의 FEC)
S=8: 각 비트를 8개 심볼로 인코딩 (최대 FEC, 최대 범위)
```

**실무적 의미:**
- 옥외 센서 네트워크가 중계기 없이 건물 캠퍼스를 커버 가능
- 개방 필드의 농업 IoT 센서가 300-400m 도달 가능
- 벽과 장애물을 통과하는 실내 범위 개선

### 확장 광고(Extended Advertising)

BLE 4.x 광고 패킷은 31바이트로 제한되었습니다. BLE 5.0 확장 광고(Extended Advertising)는 패킷당 최대 **255바이트**, 체인 광고로 총 최대 **1650바이트**를 지원합니다:

```
BLE 4.x 광고:
┌────────────────────────────┐
│  31 bytes payload          │  ← 단일 패킷, 제한된 데이터
└────────────────────────────┘

BLE 5.0 확장 광고:
┌────────────────────────────┐  ┌────────────────────────────┐
│  255 bytes (패킷 1)        │──│  255 bytes (패킷 2)        │──...
└────────────────────────────┘  └────────────────────────────┘
  체인: 총 최대 1650 bytes

이점:
- 전체 기기 이름 + 서비스 UUID + 제조사 데이터를 광고 가능
- 연결 없이 센서 측정값 브로드캐스트
- 다른 PHY에서 동시에 여러 광고 세트 운용
```

### BLE 버전 비교

| 기능 | BLE 4.2 | BLE 5.0 | BLE 5.3 |
|------|---------|---------|---------|
| **최대 데이터 전송률** | 1 Mbps | 2 Mbps | 2 Mbps |
| **범위 (이론적)** | ~100m | ~400m (Coded PHY) | ~400m (Coded PHY) |
| **광고 페이로드** | 31 bytes | 255 bytes (확장) | 255 bytes (확장) |
| **체인 광고** | 미지원 | 지원 (최대 1650 bytes) | 지원 |
| **PHY 옵션** | LE 1M만 | LE 1M, LE 2M, LE Coded | LE 1M, LE 2M, LE Coded |
| **브로드캐스트 채널** | 3 | 3 기본 + 37 보조 | 3 기본 + 37 보조 |
| **연결 서브레이팅(Subrating)** | 미지원 | 미지원 | 지원 (전력 절약) |
| **채널 분류(Classification)** | 미지원 | 미지원 | 지원 (공존 개선) |
| **AdvDataInfo (ADI)** | 미지원 | 지원 (중복 제거) | 지원 |
| **주기적 광고(Periodic Adv)** | 미지원 | 지원 | 향상됨 |

### BLE Mesh 아키텍처

BLE Mesh는 점대점(point-to-point) BLE 토폴로지를 **관리형 플러드(Managed Flood)** 방식을 사용하여 다대다(many-to-many) 네트워크로 변환합니다:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BLE Mesh 네트워크 아키텍처                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────┐         ┌──────┐         ┌──────┐         ┌──────┐       │
│  │노드 A│◄───────►│노드 B│◄───────►│노드 C│◄───────►│노드 D│       │
│  │(Relay)│        │(Relay)│        │      │         │      │       │
│  └──┬───┘         └──┬───┘         └──────┘         └──────┘       │
│     │                │                                              │
│     │    ┌──────┐    │    ┌──────┐                                  │
│     └───►│노드 E│◄───┘    │노드 F│  (Low-Power Node)                │
│          │      │         │(LPN) │──── Friend ────► 노드 B          │
│          └──────┘         └──────┘                                  │
│                                                                      │
│  주요 역할:                                                          │
│  - Relay 노드: 범위 확장을 위해 메시지 전달                           │
│  - Proxy 노드: GATT 기반 기기를 메시에 연결                           │
│  - Friend 노드: 저전력 노드(LPN)를 위해 메시지 저장                   │
│  - Low Power 노드(LPN): 대부분의 시간을 슬립 상태로 보냄              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Mesh 핵심 개념

**노드(Node), 요소(Element), 모델(Model):**
- **노드(Node)**: 메시에 프로비저닝된 장치 (예: 스마트 전구)
- **요소(Element)**: 노드 내의 기능 단위 (예: 이중 색상 전구는 2개의 요소를 가짐)
- **모델(Model)**: 동작을 정의 (예: Generic OnOff Model, Light Lightness Model)
  - **서버 모델(Server Model)**: 상태를 보유 (예: 조명이 켜짐/꺼짐)
  - **클라이언트 모델(Client Model)**: 상태 변경 요청 전송

**주소 지정(Addressing):**
- **유니캐스트(Unicast)**: 일대일 (요소별 고유 주소)
- **그룹(Group)**: 일대다 (예: "거실 전체 조명")
- **가상(Virtual)**: 128비트 UUID를 사용한 레이블 기반 주소 지정

### Mesh 프로비저닝(Provisioning)

프로비저닝(Provisioning)은 새 기기를 메시 네트워크에 추가하는 과정입니다:

```
┌───────────────┐                    ┌───────────────┐
│ 프로비저닝 전  │                    │  프로비저너    │
│    기기        │                    │  (폰 앱)      │
├───────────────┤                    ├───────────────┤
│ 1. 비콘       │────── Beacon ────►│               │
│               │                    │ 2. 초대       │
│               │◄──── Invite ──────│               │
│ 3. 키 교환    │◄──── Public Keys ►│               │
│               │                    │               │
│ 4. 인증       │◄──── OOB/Confirm ►│ 5. 검증       │
│ 6. 데이터     │◄──── Provisioning ─│               │
│    수신       │      Data          │               │
│               │  (Network Key,     │               │
│               │   Unicast Addr,    │               │
│               │   IV Index)        │               │
├───────────────┤                    ├───────────────┤
│  프로비저닝   │ ═══ Mesh Network ══│  프로비저너    │
│    완료 노드   │                    │               │
└───────────────┘                    └───────────────┘
```

### BLE Mesh 사용 사례

| 사용 사례 | BLE Mesh를 사용하는 이유 | 규모 |
|----------|------------------------|------|
| **스마트 조명** | 그룹 제어, 디밍, 방/층 단위 색상 제어 | 50-200개 전구 |
| **빌딩 자동화** | HVAC, 재실 센서, 출입 제어 | 100-1000개 노드 |
| **산업용 IoT** | 자산 추적, 공장 설비 상태 모니터링 | 100-500개 노드 |
| **물류 창고 관리** | 비콘 기반 위치 추적, 재고 관리 | 200-1000개 노드 |
| **의료** | 환자 추적, 장비 모니터링 | 50-500개 노드 |

### BLE Mesh vs 기타 메시 프로토콜

| 기능 | BLE Mesh | Zigbee | Thread (Matter) | Wi-Fi Mesh |
|------|----------|--------|-----------------|------------|
| **전력** | 매우 낮음 | 낮음 | 낮음 | 높음 |
| **홉당 범위** | ~30m 실내 | ~30m 실내 | ~30m 실내 | ~50m 실내 |
| **최대 노드 수** | ~32,000 | ~65,000 | ~250 | ~250 |
| **데이터 전송률** | 1 Mbps | 250 kbps | 250 kbps | 100+ Mbps |
| **폰 직접 접속** | 가능 (Proxy 경유) | 불가 (게이트웨이 필요) | 불가 (Border Router 필요) | 가능 |
| **라우팅** | 관리형 플러드(Managed Flood) | 라우팅 테이블 | 라우팅 테이블 | 라우팅 테이블 |
| **프로비저닝** | 보안 (ECDH) | Trust Center | Thread Commissioner | WPA |

BLE Mesh의 핵심 장점은 스마트폰이 프록시 노드(Proxy Node)를 통해 메시와 직접 상호작용할 수 있어, 전용 게이트웨이(Gateway)나 허브(Hub)가 필요 없다는 점입니다.

---

## 연습 문제

### 연습 1: RSSI 순위가 있는 BLE 장치 스캐너

`bleak`을 사용하여 주변 BLE 장치를 탐색하고 신호 강도로 순위를 매기는 Python 스크립트를 작성하세요:

1. `BleakScanner.discover()`로 10초간 스캔합니다.
2. 발견된 장치를 RSSI 내림차순(신호 강도 강한 순서)으로 정렬합니다.
3. 각 장치에 대해 순위, 장치 이름(`None`이면 "Unknown"), MAC 주소, dBm 단위 RSSI를 출력합니다.
4. 30초마다 스캔을 다시 실행하고, 새로 나타난 장치, 사라진 장치, 지속된 장치의 RSSI 변화를 표시합니다.

### 연습 2: GATT 서비스 탐색기

실제 또는 시뮬레이션된 BLE 주변 장치(Peripheral)에 연결하여 전체 GATT 프로파일을 열거하세요:

1. `BleakClient`를 사용하여 BLE 장치(BLE 주변 장치 앱을 실행하는 스마트폰, Arduino Nano 33 BLE, 또는 접근 가능한 BLE 센서)에 연결합니다.
2. 모든 서비스와 특성(Characteristic)을 반복합니다. 각 특성에 대해 서비스 UUID(Universally Unique Identifier), 특성 UUID, 속성(읽기/쓰기/알림/표시), 핸들 번호를 출력합니다.
3. `read` 속성이 있는 모든 특성에 대해 값 읽기를 시도하고 원시 바이트를 16진수 문자열로 표시합니다.
4. 섹션 3.1의 표준 GATT 서비스 중 어떤 것이 장치에 있는지, 그 특성이 무엇을 보고하는지 식별합니다.

### 연습 3: 실시간 알림(Notification) 로거

BLE 주변 장치에서 알림을 구독하고 들어오는 데이터를 파일에 기록하세요:

1. 최소 하나의 알림 가능한 특성(Notifiable Characteristic)을 노출하는 BLE 장치(예: 심박수 센서, 환경 센서, 또는 Arduino의 커스텀 특성)에 연결합니다.
2. 각 알림에 타임스탬프를 붙이고 CSV 파일에 추가하는 `notification_handler` 콜백을 등록합니다: `timestamp`, `characteristic_uuid`, `raw_hex`, `parsed_value`.
3. GATT 사양에 따라 원시 바이트를 파싱합니다: 심박수(0x2A37), 온도(0x2A1C, ×0.01°C), 또는 배터리 레벨(0x2A19, uint8).
4. 60초간 로거를 실행한 후 요약을 출력합니다: 총 수신된 알림 수, 평균값, 최소값, 최대값.

### 연습 4: 자동 재연결이 있는 BLE 센서 모니터

섹션 4의 `BLESensorMonitor` 클래스를 실세계 연결 문제를 처리하도록 확장하세요:

1. `max_retries` 파라미터를 구현합니다. `max_retries` 번의 시도 내에 장치를 찾거나 연결할 수 없으면 설명적인 메시지와 함께 `RuntimeError`를 발생시킵니다.
2. 재연결 시도 사이에 지수 백오프(Exponential Backoff)를 추가합니다: 2초에서 시작하여 최대 60초까지 두 배씩 늘어납니다.
3. 연결 가동 시간(Uptime)과 총 연결 끊김 시간을 추적합니다. 모니터가 중지되면 연결 품질 보고서를 출력합니다.
4. 연결되어 있는데 30초 이상 알림이 수신되지 않으면 경고를 기록하고 특성 재구독을 시도하는 데이터 품질 검사를 추가합니다.

### 연습 5: BLE vs WiFi IoT 설계 트레이드오프 분석

세 가지 특정 IoT 사용 사례에 대해 BLE와 WiFi를 비교하는 구조적 분석(약 300단어)을 작성하세요:

1. **웨어러블 피트니스 트래커(Wearable Fitness Tracker)**: 야외 운동 중 착용하며 5분마다 체크인.
2. **산업용 진동 센서(Industrial Vibration Sensor)**: 공장 기계에 부착하여 초당 1000개 샘플을 지속적으로 스트리밍.
3. **스마트 도어락(Smart Door Lock)**: 500ms 이내에 잠금 해제 명령에 응답해야 하며 가정용 WiFi 라우터가 다운되어도 안정적으로 동작해야 함.

각 사용 사례에서 선택할 무선 기술(BLE, WiFi, 또는 혼합)을 명시하고, 이 레슨의 개념(전력 소비, 데이터 전송률, 범위, 연결 시간, GATT vs 소켓 모델)을 사용하여 이유를 설명하며, 트레이드오프를 기술합니다. 섹션 1.1의 비교 표와 섹션 6.4의 BLE 5.0 개선 사항을 논거로 활용하세요.

---

## 다음 단계

- [MQTT 프로토콜](06_MQTT_Protocol.md): BLE 데이터를 MQTT로 전송
- [홈 자동화 프로젝트](10_Home_Automation_Project.md): BLE 스마트홈 프로젝트

---

*최종 업데이트: 2026-02-01*
