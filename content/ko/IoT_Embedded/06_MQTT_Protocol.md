# 06. MQTT 프로토콜

**이전**: [BLE 연결](./05_BLE_Connectivity.md) | **다음**: [HTTP REST for IoT](./07_HTTP_REST_for_IoT.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. MQTT 발행/구독 아키텍처와 IoT에서 HTTP 대비 갖는 장점을 설명할 수 있다
2. 인증을 포함한 Mosquitto MQTT 브로커를 설치하고 설정할 수 있다
3. 단일 레벨 및 다중 레벨 와일드카드를 활용한 계층적 토픽 구조를 설계할 수 있다
4. QoS 레벨 0, 1, 2를 비교하고 사용 사례에 맞는 레벨을 선택할 수 있다
5. paho-mqtt를 사용하여 Python으로 MQTT 발행자와 구독자를 구현할 수 있다
6. 보존 메시지(retained messages)와 유언 메시지(Last Will and Testament, LWT)를 활용하여 장치 상태를 안정적으로 추적할 수 있다

---

MQTT는 IoT에서 지배적인 메시징 프로토콜입니다. IoT 장치가 직면한 제약 — 제한된 대역폭, 불안정한 네트워크, 수천 개의 센서에서 중앙 시스템으로 데이터를 푸시해야 하는 필요성 — 에 맞게 설계되었기 때문입니다. HTTP가 웹 브라우저의 페이지 요청에 적합하다면, MQTT는 경량 장치가 데이터를 지속적으로 스트리밍하거나 실시간으로 명령을 받아야 할 때 탁월합니다.

---

## 1. MQTT 프로토콜 개요

### 1.1 MQTT란?

**MQTT (Message Queuing Telemetry Transport)**는 경량 메시징 프로토콜로, IoT 환경에 최적화되어 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    MQTT 아키텍처                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Publisher                    Broker                        │
│   (센서)                       (중계자)                       │
│   ┌─────────┐                 ┌─────────┐                   │
│   │ 온도    │ ──PUBLISH────▶ │         │                   │
│   │ 센서    │    (topic:     │ Mosquitto│                   │
│   └─────────┘   home/temp)   │         │                   │
│                               │         │                   │
│   ┌─────────┐                 │         │    ┌─────────┐    │
│   │ 습도    │ ──PUBLISH────▶ │         │ ──▶│ 모바일  │    │
│   │ 센서    │    (topic:     │         │    │   앱    │    │
│   └─────────┘   home/humid)  │         │    └─────────┘    │
│                               │         │    Subscriber     │
│                               │         │                   │
│                               │         │    ┌─────────┐    │
│                               │         │ ──▶│ 웹      │    │
│                               │         │    │ 대시보드│    │
│                               └─────────┘    └─────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 MQTT 특징

| 특징 | 설명 |
|------|------|
| **경량** | 최소 2바이트 고정 헤더(1바이트: 패킷 타입+플래그, 1바이트: 잔여 길이; 대형 패킷은 잔여 길이에 가변 길이 인코딩으로 최대 4바이트 사용 가능하나, 고정 부분은 항상 2바이트 -- HTTP의 ~800바이트 헤더와 비교) |
| **Pub/Sub** | 발행/구독 패턴 (느슨한 결합) |
| **QoS** | 3가지 메시지 전달 보장 수준 |
| **Last Will** | 비정상 연결 종료 시 알림 |
| **Retained** | 마지막 메시지 저장 |
| **Keep Alive** | 연결 상태 모니터링 |

### 1.3 MQTT vs HTTP 비교

```python
# 프로토콜 비교
comparison = {
    "Header Size": {
        "MQTT": "2 bytes (최소)",
        "HTTP": "~800 bytes (평균)"
    },
    "Pattern": {
        "MQTT": "Pub/Sub (비동기)",
        "HTTP": "Request/Response (동기)"
    },
    "Connection": {
        "MQTT": "지속 연결",
        "HTTP": "비연결 (HTTP/1.1) 또는 지속 (HTTP/2)"
    },
    "Bidirectional": {
        "MQTT": "지원 (양방향)",
        "HTTP": "서버 푸시 제한적"
    },
    "Use Case": {
        "MQTT": "실시간 센서, 저전력, 저대역폭",
        "HTTP": "웹 API, 대용량 데이터"
    }
}
```

---

## 2. Mosquitto 브로커

### 2.1 설치

```bash
# Ubuntu/Debian (라즈베리파이)
sudo apt update
sudo apt install mosquitto mosquitto-clients

# 서비스 시작 및 활성화
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# 상태 확인
sudo systemctl status mosquitto
```

### 2.2 기본 설정

```bash
# 설정 파일 편집
sudo nano /etc/mosquitto/mosquitto.conf
```

```conf
# /etc/mosquitto/mosquitto.conf

# 기본 설정
pid_file /run/mosquitto/mosquitto.pid

# 리스너 설정
listener 1883
protocol mqtt

# 익명 접속 (테스트용)
allow_anonymous true

# 로그 설정
log_dest file /var/log/mosquitto/mosquitto.log
log_type all

# 지속성 (메시지 저장)
persistence true
persistence_location /var/lib/mosquitto/

# 추가 설정 파일 포함
include_dir /etc/mosquitto/conf.d
```

### 2.3 인증 설정

```bash
# 비밀번호 파일 생성
sudo mosquitto_passwd -c /etc/mosquitto/passwd iotuser

# 추가 사용자
sudo mosquitto_passwd /etc/mosquitto/passwd anotheruser
```

```conf
# /etc/mosquitto/conf.d/auth.conf

# 익명 접속 비활성화
allow_anonymous false

# 비밀번호 파일
password_file /etc/mosquitto/passwd
```

```bash
# 설정 적용
sudo systemctl restart mosquitto
```

### 2.4 TLS 설정 (보안 연결)

```bash
# 인증서 생성 (자체 서명)
mkdir -p ~/mqtt-certs && cd ~/mqtt-certs

# CA 키 및 인증서
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -key ca.key -out ca.crt

# 서버 키 및 인증서
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365
```

```conf
# /etc/mosquitto/conf.d/tls.conf

listener 8883
protocol mqtt

cafile /home/pi/mqtt-certs/ca.crt
certfile /home/pi/mqtt-certs/server.crt
keyfile /home/pi/mqtt-certs/server.key

require_certificate false
```

### 2.5 CLI 테스트

```bash
# 터미널 1: 구독
mosquitto_sub -h localhost -t "test/topic" -v

# 터미널 2: 발행
mosquitto_pub -h localhost -t "test/topic" -m "Hello MQTT!"

# 인증 포함
mosquitto_pub -h localhost -t "test/topic" -m "Hello" -u iotuser -P password

# QoS 지정
mosquitto_pub -h localhost -t "test/topic" -m "QoS 1" -q 1
```

---

## 3. Topic과 QoS

### 3.1 Topic 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    MQTT Topic 구조                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   계층적 구조 (슬래시로 구분)                                 │
│                                                              │
│   home/                                                      │
│   ├── living-room/                                          │
│   │   ├── temperature      → 거실 온도                      │
│   │   ├── humidity         → 거실 습도                      │
│   │   └── light            → 거실 조명                      │
│   ├── bedroom/                                              │
│   │   ├── temperature                                       │
│   │   └── motion           → 침실 모션 센서                 │
│   └── kitchen/                                              │
│       └── smoke            → 주방 연기 감지                 │
│                                                              │
│   와일드카드:                                                │
│   • + (단일 레벨): home/+/temperature                       │
│     → home/living-room/temperature, home/bedroom/temperature│
│                                                              │
│   • # (다중 레벨): home/#                                    │
│     → home 아래 모든 토픽                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Topic 설계 가이드

```python
# 좋은 Topic 설계 예시
topic_examples = {
    # 위치/장치/센서
    "home/living-room/temperature": "거실 온도",
    "office/floor1/room101/hvac/status": "사무실 HVAC 상태",

    # 장치ID/데이터타입
    "sensor/abc123/data": "센서 데이터",
    "sensor/abc123/status": "센서 상태",

    # 명령 및 응답
    "device/led001/command": "LED 명령",
    "device/led001/response": "LED 응답",

    # 클라우드 연동
    "aws/things/sensor001/shadow/update": "AWS IoT 섀도우",
}

# 피해야 할 패턴
bad_patterns = [
    "/leading/slash",     # 선행 슬래시 불필요
    "space in topic",     # 공백 피하기
    "UpperCase/Mixed",    # 소문자 권장
    "too/deep/hierarchy/a/b/c/d/e",  # 과도한 깊이
]
```

### 3.3 QoS (Quality of Service)

```
┌─────────────────────────────────────────────────────────────┐
│                    MQTT QoS 레벨                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  QoS 0: At most once (최대 1회)                             │
│  ┌────────┐         ┌────────┐                              │
│  │Publisher│──PUBLISH──▶│Broker│                             │
│  └────────┘         └────────┘                              │
│  • 전송 확인 없음                                            │
│  • 가장 빠름, 메시지 손실 가능                               │
│  • 용도: 센서 데이터 (손실 허용)                             │
│                                                              │
│  QoS 1: At least once (최소 1회)                            │
│  ┌────────┐         ┌────────┐                              │
│  │Publisher│──PUBLISH──▶│Broker│                             │
│  │        │◀──PUBACK───│      │                             │
│  └────────┘         └────────┘                              │
│  • 확인 응답, 재전송 가능                                    │
│  • 중복 가능, 손실 없음                                      │
│  • 용도: 중요 알림                                          │
│                                                              │
│  QoS 2: Exactly once (정확히 1회)                           │
│  ┌────────┐         ┌────────┐                              │
│  │Publisher│──PUBLISH──▶│Broker│                             │
│  │        │◀──PUBREC───│      │                             │
│  │        │──PUBREL──▶│      │                             │
│  │        │◀──PUBCOMP──│      │                             │
│  └────────┘         └────────┘                              │
│  • 4-way handshake                                          │
│  • 가장 느림, 정확한 전달 보장                               │
│  • 용도: 결제, 중요 명령                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**QoS 2에 4-메시지 핸드셰이크가 필요한 이유:** QoS 1은 단순한 PUBLISH + PUBACK 쌍을 사용하지만, PUBACK이 손실되면 발신자가 재전송하여 수신자가 중복 메시지를 받게 됩니다. QoS 2는 프로세스를 두 단계로 분리하여 이를 해결합니다: (1) PUBLISH/PUBREC 단계에서 수신자가 메시지를 정확히 한 번 저장합니다 -- 수신자가 패킷 ID를 기록하고 PUBREC으로 응답; (2) PUBREL/PUBCOMP 단계에서 메시지를 전달용으로 해제합니다 -- 발신자가 "수신 확인, 전달 진행하라"(PUBREL)고 알리고, 수신자가 완료를 확인(PUBCOMP)하고 패킷 ID를 폐기합니다. 어떤 메시지가 손실되더라도, 현재 단계 메시지의 재전송은 멱등적(idempotent)입니다 -- 수신자가 동일한 패킷 ID로 인식할 수 있기 때문입니다. 이 2단계 커밋(two-phase commit)이 QoS 1의 2배 왕복 비용으로 정확히 한 번 전달을 보장합니다.

### 3.4 Retained 메시지

```python
# Retained 메시지 개념
"""
Retained Message:
- 브로커가 마지막 메시지를 저장
- 새 구독자가 연결 시 즉시 수신
- 센서 현재 상태 전달에 유용

예:
1. 센서가 온도 25도를 retain=True로 발행
2. 브로커가 메시지 저장
3. 새 구독자 연결 시 즉시 25도 수신
4. 센서 오프라인이어도 마지막 값 유지
"""

# 사용 예
retained_use_cases = {
    "장치 상태": "device/sensor01/status (online/offline)",
    "현재 값": "home/temperature (마지막 측정값)",
    "설정": "device/config (현재 설정)",
}
```

---

## 4. paho-mqtt 라이브러리

### 4.1 설치

```bash
pip install paho-mqtt
```

### 4.2 기본 Publisher

```python
#!/usr/bin/env python3
"""MQTT Publisher 기본 예제"""

import paho.mqtt.client as mqtt
import json
import time

# 브로커 설정
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "sensor/temperature"

def on_connect(client, userdata, flags, rc):
    """연결 콜백"""
    if rc == 0:
        print("브로커 연결 성공")
    else:
        print(f"연결 실패: {rc}")

def on_publish(client, userdata, mid):
    """발행 완료 콜백"""
    print(f"메시지 발행됨: mid={mid}")

def publish_sensor_data():
    """센서 데이터 발행"""
    client = mqtt.Client(client_id="temperature_sensor_01")
    client.on_connect = on_connect
    client.on_publish = on_publish

    # 연결
    # keepalive=60 이유: 이 시간(초) 내에 다른 패킷을 보내지 않으면 클라이언트가
    # PINGREQ를 전송하여 브로커에게 살아있음을 증명. 트레이드오프:
    #   - 너무 낮으면 (예: 10초): 잦은 PING이 제한된 네트워크에서 대역폭 낭비
    #   - 너무 높으면 (예: 300초): 브로커가 죽은 클라이언트를 감지하는 데
    #     최대 1.5배 소요 (MQTT 사양), 즉 300초면 ~450초까지 LWT 미발동
    # 60초는 대역폭과 감지 속도를 균형 있게 맞추는 일반적인 기본값.
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    try:
        while True:
            # 센서 데이터 생성
            data = {
                "sensor_id": "temp_01",
                "temperature": round(20 + (time.time() % 10), 1),
                "timestamp": int(time.time())
            }

            payload = json.dumps(data)

            # 발행 (QoS 1, Retained 사용)
            result = client.publish(TOPIC, payload, qos=1, retain=True)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"발행: {payload}")
            else:
                print(f"발행 실패: {result.rc}")

            # sleep(5) 이유: 발행 주기를 제어. 트레이드오프:
            #   - 짧은 주기 (예: 0.5초): 세밀한 데이터이나 브로커 부하 증가,
            #     대역폭 소모, 로깅 시 SD 카드 마모 가속
            #   - 긴 주기 (예: 60초): 부하 감소이나 데이터 해상도 저하,
            #     이상 감지 지연. 온도 모니터링에 5초는 적정 수준.
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n종료")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    publish_sensor_data()
```

### 4.3 기본 Subscriber

```python
#!/usr/bin/env python3
"""MQTT Subscriber 기본 예제"""

import paho.mqtt.client as mqtt
import json

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPICS = [
    ("sensor/temperature", 1),
    ("sensor/humidity", 1),
]

def on_connect(client, userdata, flags, rc):
    """연결 콜백"""
    if rc == 0:
        print("브로커 연결 성공")
        # 토픽 구독
        for topic, qos in TOPICS:
            client.subscribe(topic, qos)
            print(f"구독: {topic} (QoS {qos})")
    else:
        print(f"연결 실패: {rc}")

def on_message(client, userdata, msg):
    """메시지 수신 콜백"""
    try:
        payload = json.loads(msg.payload.decode())
        print(f"[{msg.topic}] {payload}")
    except json.JSONDecodeError:
        print(f"[{msg.topic}] {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    """연결 해제 콜백"""
    print(f"연결 해제: {rc}")

def subscribe_sensors():
    """센서 데이터 구독"""
    client = mqtt.Client(client_id="sensor_monitor_01")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # keepalive=60 이유: Publisher 예제의 트레이드오프 설명 참조
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n종료")
        client.disconnect()

if __name__ == "__main__":
    subscribe_sensors()
```

### 4.4 인증 사용

```python
#!/usr/bin/env python3
"""MQTT 인증 연결"""

import paho.mqtt.client as mqtt
import ssl

BROKER_HOST = "mqtt.example.com"
BROKER_PORT = 8883  # TLS

def create_secure_client(username: str, password: str) -> mqtt.Client:
    """보안 MQTT 클라이언트 생성"""
    client = mqtt.Client(client_id="secure_client_01")

    # 인증 설정
    client.username_pw_set(username, password)

    # TLS 설정
    client.tls_set(
        ca_certs="/path/to/ca.crt",
        certfile="/path/to/client.crt",  # 클라이언트 인증서 (옵션)
        keyfile="/path/to/client.key",   # 클라이언트 키 (옵션)
        tls_version=ssl.PROTOCOL_TLS
    )

    # 호스트명 검증 비활성화 (자체 서명 인증서용)
    # client.tls_insecure_set(True)

    return client

def connect_secure():
    """보안 연결"""
    client = create_secure_client("iotuser", "password123")

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("보안 연결 성공")
        else:
            print(f"연결 실패: {rc}")

    client.on_connect = on_connect
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()
```

### 4.5 Last Will and Testament (LWT)

```python
#!/usr/bin/env python3
"""MQTT Last Will (비정상 종료 알림)"""

import paho.mqtt.client as mqtt
import time

def create_client_with_lwt(client_id: str) -> mqtt.Client:
    """LWT가 설정된 클라이언트 생성"""
    client = mqtt.Client(client_id=client_id)

    # Last Will 설정 -- 반드시 connect() 호출 전에 설정해야 함.
    # 순서가 중요한 이유: will_set()은 CONNECT 패킷의 Will 필드를 구성합니다.
    # connect()가 호출되면 CONNECT 패킷이 이미 브로커에 전송되므로,
    # connect() 이후에 will_set()을 호출하면 효과가 없습니다 -- 브로커가
    # Will 메시지를 수신하지 못해 "offline" 알림이 조용히 발동되지 않습니다.
    # 비정상 연결 종료 시 이 메시지가 발행됨
    client.will_set(
        topic=f"device/{client_id}/status",
        payload="offline",
        qos=1,
        retain=True
    )

    return client

def run_sensor_with_lwt():
    """LWT가 있는 센서 실행"""
    client_id = "sensor_with_lwt"
    client = create_client_with_lwt(client_id)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("연결됨")
            # 온라인 상태 발행
            client.publish(
                f"device/{client_id}/status",
                "online",
                qos=1,
                retain=True
            )

    client.on_connect = on_connect
    client.connect("localhost", 1883, keepalive=60)
    client.loop_start()

    try:
        while True:
            client.publish(f"device/{client_id}/data", "sensor data")
            time.sleep(5)
    except KeyboardInterrupt:
        # 정상 종료 시 오프라인 상태 발행
        client.publish(f"device/{client_id}/status", "offline", qos=1, retain=True)
        client.disconnect()
```

---

## 5. 고급 패턴

### 5.1 메시지 라우팅

```python
#!/usr/bin/env python3
"""토픽 기반 메시지 라우팅"""

import paho.mqtt.client as mqtt
import json
from typing import Callable

class MQTTRouter:
    """MQTT 메시지 라우터"""

    def __init__(self, broker_host: str, broker_port: int = 1883):
        self.client = mqtt.Client()
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.routes: dict[str, Callable] = {}

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def route(self, topic_pattern: str):
        """라우트 데코레이터"""
        def decorator(func: Callable):
            self.routes[topic_pattern] = func
            return func
        return decorator

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("라우터 연결됨")
            for topic in self.routes.keys():
                client.subscribe(topic)
                print(f"라우트 등록: {topic}")

    def _on_message(self, client, userdata, msg):
        # 매칭되는 핸들러 찾기
        for pattern, handler in self.routes.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    payload = json.loads(msg.payload.decode())
                except json.JSONDecodeError:
                    payload = msg.payload.decode()

                handler(msg.topic, payload)
                break

    def run(self):
        """라우터 실행"""
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_forever()

# 사용 예
router = MQTTRouter("localhost")

@router.route("sensor/+/temperature")
def handle_temperature(topic: str, payload: dict):
    sensor_id = topic.split('/')[1]
    print(f"온도 [{sensor_id}]: {payload}")

@router.route("sensor/+/humidity")
def handle_humidity(topic: str, payload: dict):
    sensor_id = topic.split('/')[1]
    print(f"습도 [{sensor_id}]: {payload}")

@router.route("device/+/command")
def handle_command(topic: str, payload: dict):
    device_id = topic.split('/')[1]
    print(f"명령 [{device_id}]: {payload}")

if __name__ == "__main__":
    router.run()
```

### 5.2 비동기 MQTT (asyncio)

```python
#!/usr/bin/env python3
"""비동기 MQTT 클라이언트 (asyncio-mqtt)"""

import asyncio
import aiomqtt  # pip install aiomqtt
import json

async def publish_sensor_data():
    """비동기 발행"""
    async with aiomqtt.Client("localhost") as client:
        while True:
            data = {
                "temperature": 25.5,
                "timestamp": asyncio.get_event_loop().time()
            }
            await client.publish("sensor/temp", json.dumps(data))
            print(f"발행: {data}")
            await asyncio.sleep(5)

async def subscribe_sensor_data():
    """비동기 구독"""
    async with aiomqtt.Client("localhost") as client:
        async with client.messages() as messages:
            await client.subscribe("sensor/#")

            async for message in messages:
                print(f"[{message.topic}] {message.payload.decode()}")

async def main():
    """발행과 구독 동시 실행"""
    await asyncio.gather(
        publish_sensor_data(),
        subscribe_sensor_data()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.3 재연결 로직

```python
#!/usr/bin/env python3
"""자동 재연결 MQTT 클라이언트"""

import paho.mqtt.client as mqtt
import time

class RobustMQTTClient:
    """자동 재연결을 지원하는 MQTT 클라이언트"""

    def __init__(self, broker_host: str, broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.connected = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("연결됨")
            self.connected = True
            self.reconnect_delay = 1  # 리셋
        else:
            print(f"연결 실패: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"연결 해제: {rc}")
        self.connected = False

        if rc != 0:
            self._reconnect()

    def _reconnect(self):
        """재연결 시도"""
        while not self.connected:
            try:
                print(f"재연결 시도... ({self.reconnect_delay}초 후)")
                time.sleep(self.reconnect_delay)
                self.client.reconnect()
            except Exception as e:
                print(f"재연결 실패: {e}")
                # 지수 백오프
                self.reconnect_delay = min(
                    self.reconnect_delay * 2,
                    self.max_reconnect_delay
                )

    def connect(self):
        """초기 연결"""
        self.client.connect(self.broker_host, self.broker_port)

    def run(self):
        """클라이언트 실행"""
        self.connect()
        self.client.loop_forever()

if __name__ == "__main__":
    client = RobustMQTTClient("localhost")
    client.run()
```

## 6. CoAP 프로토콜

MQTT가 푸시 기반 IoT 메시징을 지배하는 반면, 많은 IoT 시나리오에서는 HTTP와 유사한 요청/응답(request/response) 패턴이 필요합니다 -- 단, HTTP의 오버헤드 없이 말입니다. CoAP(Constrained Application Protocol)는 제약된 장치와 손실이 있는 네트워크를 위해 특별히 설계된 경량 RESTful 프로토콜로 이 gap을 채웁니다.

### 6.1 CoAP vs HTTP 비교

| 특성 | CoAP | HTTP |
|------|------|------|
| **전송 계층** | UDP (DTLS 선택 가능) | TCP (TLS 포함) |
| **헤더 크기** | 4바이트 (고정) | 가변 (평균 ~800바이트) |
| **메서드** | GET, POST, PUT, DELETE | GET, POST, PUT, DELETE, PATCH, ... |
| **검색** | 내장 (`/.well-known/core`) | 없음 (문서화 필요) |
| **관찰(Observe)** | 네이티브 (구독과 유사) | 폴링 또는 WebSocket 필요 |
| **멀티캐스트** | 지원 (UDP 멀티캐스트) | 미지원 |
| **신뢰성** | Confirmable / Non-confirmable | TCP가 전달 보장 |
| **적합 환경** | 제약 장치, 센서 쿼리 | 웹 애플리케이션, 리치 클라이언트 |

> **CoAP vs MQTT 선택 기준.** 장치가 지속적으로 데이터를 푸시할 때(센서 텔레메트리, 이벤트 스트림)는 MQTT를 사용합니다. 장치 상태를 온디맨드로 조회해야 할 때(센서 값 읽기, 설정 변경), 특히 장치가 지속적 TCP 연결을 유지하기 어려울 만큼 제약이 많을 때는 CoAP를 사용합니다. 많은 시스템이 양쪽을 함께 사용합니다: 텔레메트리에는 MQTT, 장치 관리에는 CoAP.

### 6.2 CoAP 메시지 형식

```
┌─────────────────────────────────────────────────────────────┐
│                    CoAP 메시지 형식                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0                   1                   2                   │
│  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3           │
│ ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐       │
│ │Ver│ T │ TKL │     Code      │      Message ID     │       │
│ └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘       │
│ │              Token (0-8 bytes)               │             │
│ ├──────────────────────────────────────────────┤             │
│ │              Options (가변)                  │             │
│ ├──────────────────────────────────────────────┤             │
│ │         0xFF (페이로드 마커)                  │             │
│ ├──────────────────────────────────────────────┤             │
│ │              Payload (가변)                  │             │
│ └──────────────────────────────────────────────┘             │
│                                                             │
│  Ver: 버전 (1 = CoAP v1)                                   │
│  T:   타입 (CON=0, NON=1, ACK=2, RST=3)                   │
│  TKL: 토큰 길이 (0-8)                                      │
│  Code: 메서드 (0.01=GET, 0.02=POST, 0.03=PUT, 0.04=DELETE)│
│        또는 응답 (2.05=Content, 4.04=Not Found 등)         │
│                                                             │
│  메시지 타입:                                               │
│  • CON (Confirmable): ACK 필요, 타임아웃 시 재전송         │
│  • NON (Non-confirmable): 전송 후 잊음(fire and forget)    │
│  • ACK (Acknowledgement): CON 수신 확인                    │
│  • RST (Reset): 메시지 처리 오류 표시                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 CoAP Observe 패턴

Observe 옵션을 사용하면 클라이언트가 리소스에 관심을 등록하고 변경 시 알림을 받을 수 있습니다 -- MQTT 구독과 유사하지만 RESTful 프레임워크 안에서 동작합니다:

```
Client                          Server
  │                               │
  │──GET /temperature (Observe)──>│   관찰 등록
  │<──2.05 Content (temp=23.5)────│   초기값
  │                               │
  │       ... 시간 경과 ...       │
  │                               │
  │<──2.05 Content (temp=24.1)────│   값 변경, 알림
  │<──2.05 Content (temp=24.8)────│   값 변경, 알림
  │                               │
  │──GET /temperature (no Observe)│   관찰 취소
  │<──2.05 Content (temp=24.8)────│
```

### 6.4 Python aiocoap 예제

```python
# pip install aiocoap

# --- CoAP 서버 ---
import asyncio
import aiocoap
import aiocoap.resource as resource

class TemperatureResource(resource.ObservableResource):
    """CoAP 관찰 가능 온도 리소스."""

    def __init__(self):
        super().__init__()
        self.temperature = 22.5
        self._notify_task = None

    async def start_updates(self):
        """온도 변화 시뮬레이션 및 관찰자에게 알림."""
        import random
        while True:
            await asyncio.sleep(10)
            self.temperature = round(self.temperature + random.uniform(-0.5, 0.5), 1)
            self.updated_state()  # 모든 관찰자에게 알림

    async def render_get(self, request):
        """GET 요청 처리."""
        payload = f"{self.temperature}".encode('ascii')
        return aiocoap.Message(payload=payload, content_format=0)

class DeviceInfoResource(resource.Resource):
    """정적 장치 정보 리소스."""

    async def render_get(self, request):
        payload = '{"device": "sensor-01", "firmware": "1.2.0"}'.encode()
        return aiocoap.Message(payload=payload, content_format=50)

async def main():
    root = resource.Site()
    temp_resource = TemperatureResource()
    root.add_resource(['temperature'], temp_resource)
    root.add_resource(['info'], DeviceInfoResource())

    # CoAP 서버 시작
    await aiocoap.Context.create_server_context(root, bind=('::', 5683))

    # 온도 업데이트 시뮬레이션 시작
    await temp_resource.start_updates()

if __name__ == "__main__":
    asyncio.run(main())

# --- CoAP 클라이언트 ---
# import asyncio
# import aiocoap
#
# async def main():
#     context = await aiocoap.Context.create_client_context()
#
#     # 단순 GET 요청
#     request = aiocoap.Message(code=aiocoap.GET,
#                               uri='coap://localhost/temperature')
#     response = await context.request(request).response
#     print(f"온도: {response.payload.decode()}")
#
#     # Observe (변경 구독)
#     request = aiocoap.Message(code=aiocoap.GET,
#                               uri='coap://localhost/temperature',
#                               observe=0)
#     requester = context.request(request)
#     async for response in requester.observation:
#         print(f"업데이트: {response.payload.decode()}")
#
# asyncio.run(main())
```

### 6.5 CoAP 보안: DTLS

CoAP는 암호화와 인증을 위해 DTLS(Datagram Transport Layer Security) -- TLS의 UDP 버전 -- 를 사용합니다. DTLS는 TLS와 동일한 보안 보장을 제공하면서 UDP의 문제(패킷 재정렬, 손실)를 처리합니다:

- **Pre-Shared Key (PSK)**: 가장 간단하며 제약 장치에 적합
- **Raw Public Key (RPK)**: CA 없는 인증서, 더 작은 풋프린트
- **X.509 Certificates**: 완전한 PKI, 최고 수준의 보안

---

## 7. OTA 펌웨어 업데이트

OTA(Over-the-Air) 업데이트는 물리적 접근이 비실용적인 현장에 배치된 IoT 장치에 필수적입니다. 견고한 OTA 시스템은 장치가 수명 주기 전반에 걸쳐 보안을 유지하고, 버그 수정을 받으며, 새로운 기능을 확보할 수 있도록 보장합니다.

### 7.1 OTA가 필요한 이유

- **원격 장치**: 농장, 공장, 도시에 배치된 수천 개의 센서를 수동으로 업데이트할 수 없음
- **보안 패치**: 전체 플릿에 걸쳐 취약점을 신속하게 패치해야 함
- **기능 제공**: 하드웨어 교체 없이 새로운 기능 추가
- **규정 준수**: 일부 산업에서는 적시 보안 업데이트를 요구

### 7.2 OTA 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    OTA 업데이트 아키텍처                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                                         │
│  │  빌드 서버    │  CI/CD 파이프라인으로 펌웨어 빌드        │
│  │  (CI/CD)      │  비공개 키로 바이너리 서명               │
│  └───────┬───────┘                                         │
│          │ 서명된 펌웨어 업로드                              │
│          ▼                                                  │
│  ┌───────────────┐                                         │
│  │  OTA 서버     │  펌웨어 버전 저장                        │
│  │  + CDN        │  롤아웃 관리 (단계적/카나리)             │
│  └───────┬───────┘                                         │
│          │ 장치 알림 (MQTT / CoAP / HTTP)                   │
│          ▼                                                  │
│  ┌───────────────────────────────────────────────────┐     │
│  │              IoT 장치 플릿                          │     │
│  │                                                    │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │     │
│  │  │ 장치 1   │  │ 장치 2   │  │ 장치 N   │        │     │
│  │  │          │  │          │  │          │        │     │
│  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │        │     │
│  │  │ │슬롯 A│ │  │ │슬롯 A│ │  │ │슬롯 A│ │        │     │
│  │  │ │(활성)│ │  │ │(활성)│ │  │ │(활성)│ │        │     │
│  │  │ ├──────┤ │  │ ├──────┤ │  │ ├──────┤ │        │     │
│  │  │ │슬롯 B│ │  │ │슬롯 B│ │  │ │슬롯 B│ │        │     │
│  │  │ │(대기)│ │  │ │(대기)│ │  │ │(대기)│ │        │     │
│  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │        │     │
│  │  └──────────┘  └──────────┘  └──────────┘        │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 업데이트 전략

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A/B 파티션** | 두 펌웨어 슬롯; 비활성에 쓰고 재부팅 시 교체 | 즉각 롤백, 원자적 | 2배 스토리지 필요 |
| **델타 업데이트** | 버전 간 바이너리 diff만 전송 | 작은 다운로드 크기 | 복잡한 diff 처리, 취약성 |
| **전체 이미지** | 완전한 펌웨어 이미지 전송 | 간단하고 신뢰성 높음 | 큰 다운로드 |
| **컨테이너 업데이트** | OS가 아닌 앱 컨테이너를 업데이트 | 빠르고 격리됨 | 컨테이너 런타임 필요 |

### 7.4 OTA 보안

```
┌─────────────────────────────────────────────────────────────┐
│                    OTA 보안 체크리스트                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 코드 서명 (Code Signing)                               │
│     • 비공개 키로 펌웨어 서명 (Ed25519 또는 RSA)           │
│     • 장치가 내장된 공개 키로 서명 검증                     │
│     • 미서명 또는 변조된 펌웨어 거부                        │
│                                                             │
│  2. 안전한 전송 (Secure Transport)                          │
│     • 펌웨어 다운로드에 TLS/DTLS 사용                      │
│     • MITM 방지를 위한 인증서 피닝(pinning)                │
│                                                             │
│  3. 롤백 보호 (Rollback Protection)                        │
│     • 보안 스토리지에 단조 증가 버전 카운터                 │
│     • 취약한 이전 펌웨어로의 다운그레이드 방지              │
│                                                             │
│  4. 무결성 검사 (Integrity Check)                          │
│     • 플래싱 전 SHA-256 해시 검증                          │
│     • 플래싱 후 검증 (부팅 시 CRC 체크)                    │
│                                                             │
│  5. 복구 메커니즘 (Recovery Mechanism)                     │
│     • 워치독 타이머: 새 펌웨어 실패 시 자동 롤백           │
│     • 부트로더가 양호한 파티션으로 폴백                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.5 클라우드 OTA 서비스

| 서비스 | 프로토콜 | 기능 |
|--------|----------|------|
| **AWS IoT Jobs** | MQTT + HTTPS | 작업 타겟팅, 롤아웃 설정, 장치 섀도우(device shadows) |
| **Azure IoT Hub** | MQTT + HTTPS | 장치 트윈(device twins), 자동 장치 관리 |
| **Mender.io** | HTTPS | 오픈소스, A/B 파티션, 델타 업데이트 |
| **balena** | HTTPS | 컨테이너 기반 OTA, 플릿 관리 |

### 7.6 실전 OTA 흐름

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  1. 빌드  │───>│ 2. 서명  │───>│ 3. 업로드│───>│ 4. 알림  │
│  펌웨어   │    │ (Ed25519)│    │ 서버에   │    │ (MQTT)   │
└──────────┘    └──────────┘    └──────────┘    └─────┬────┘
                                                      │
      ┌──────────────────────────────────────────────┘
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│5. 장치가 │───>│6. 서명 + │───>│7. 슬롯 B │───>│ 8. 재부팅│
│ 펌웨어   │    │ 해시 검증│    │ 에 플래싱│    │ & 검증   │
│ 다운로드 │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └─────┬────┘
                                                      │
                                              ┌───────┴───────┐
                                              │  헬스 체크     │
                                              │  통과?         │
                                              ├───────┬───────┤
                                              │ 예    │  아니오│
                                              ▼       ▼       │
                                         ┌────────┐ ┌────────┐│
                                         │슬롯 B를│ │슬롯 A로││
                                         │부팅으로│ │롤백    ││
                                         │확정    │ │        ││
                                         └────────┘ └────────┘│
```

---

## 연습 문제

### 연습 1: 토픽 계층 구조 설계와 와일드카드(Wildcard) 구독

스마트 빌딩 시스템을 위한 완전한 MQTT 토픽 계층 구조를 설계하고 테스트하세요:

1. 층당 5개의 방이 있는 3층 건물을 위한 토픽 구조를 설계합니다. 각 방에는 온도, 습도, 모션, 조명 센서와 제어 가능한 HVAC(냉난방공조) 유닛이 있습니다.
2. 섹션 3.1의 모범 사례(소문자, 선행 슬래시 없음, 3~5 레벨 깊이)를 따르는 구체적인 토픽 경로를 10개 이상 작성합니다.
3. 다음에 해당하는 와일드카드 구독 문자열을 작성합니다: (a) 2층에서만 모든 온도 읽기를 수신, (b) 센서 유형에 관계없이 101호의 모든 센서 데이터를 수신, (c) 건물의 모든 방의 모션 알림을 수신.
4. 명령줄에서 `mosquitto_pub`과 `mosquitto_sub`를 사용하여 로컬 Mosquitto 브로커에서 토픽 경로 3개 이상을 테스트합니다.

### 연습 2: QoS(Quality of Service) 레벨 비교

QoS 레벨 간의 실질적 차이를 보여주는 Python 실험을 구현하세요:

1. 로컬 Mosquitto 브로커를 설정합니다. 초당 10개 메시지 속도로 세 개의 병렬 토픽에 100개의 메시지를 발행하는 발행자를 작성합니다 -- 하나는 QoS 0, 하나는 QoS 1, 하나는 QoS 2.
2. 수신된 메시지를 세고 도착 타임스탬프를 기록하는 세 개의 구독자(QoS 토픽 하나당 하나)를 작성합니다.
3. 실험 중간(약 50개 메시지 후)에 브로커를 임시로 종료하고 재시작하여 패킷 손실을 시뮬레이션합니다. 각 구독자가 총 몇 개의 메시지를 수신했는지 기록합니다.
4. 150단어 요약으로 설명합니다: (a) 빈번하고 중요하지 않은 센서 읽기, (b) 결제나 청구 이벤트, (c) 정확히 한 번만 실행되어야 하는 장치 명령에 어느 QoS 레벨을 사용해야 하는가.

### 연습 3: LWT(Last Will and Testament)와 보존 메시지로 장치 상태 추적

LWT와 보존 메시지(Retained Messages)를 사용하여 장치 플릿(Fleet) 상태 대시보드를 만드세요:

1. (섹션 6.1 기반) 연결 전에 `devices/<client_id>/status`에 `"offline"` LWT 메시지(보존, QoS 1)를 설정하고, 성공적으로 연결되면 `"online"`(보존, QoS 1)을 발행하는 `SensorPublisher` 클래스를 작성합니다.
2. `devices/+/status`를 와일드카드로 구독하고 `{device_id: status}` 딕셔너리를 유지하는 `FleetMonitor` 구독자를 만듭니다. 10초마다 실시간 플릿 상태 테이블을 출력합니다.
3. 서로 다른 클라이언트 ID로 3개의 발행자 인스턴스를 동시에 시작합니다. 하나를 갑자기 종료(`kill -9` 또는 네트워크 케이블 분리)합니다. 플릿 모니터가 LWT `"offline"` 메시지를 수신하는지 확인합니다.
4. 모든 발행자가 실행된 후 새 구독자를 시작하여 모든 활성 장치에 대한 보존된 `"online"` 상태를 즉시 수신하는지 확인합니다.

### 연습 4: MQTT 기반 원격 GPIO 제어

네트워크의 모든 장치에서 MQTT 메시지로 Raspberry Pi GPIO 핀을 제어하는 시스템을 만드세요:

1. `gpio/+/command`를 수신하는 Pi의 MQTT 구독자를 작성합니다. `+` 레벨은 GPIO 핀 번호(예: `gpio/17/command`)이며 페이로드(payload)는 `"on"`, `"off"`, 또는 `"toggle"`입니다.
2. 명령을 수신하면 `gpiozero`를 사용하여 해당 GPIO 핀에 적용합니다. 새 핀 상태를 `gpio/<pin>/state`에 `"1"` 또는 `"0"`으로 발행합니다(보존, QoS 1).
3. 표준 입력에서 명령을 읽고 적절한 토픽에 발행하는 클라이언트 스크립트를 작성합니다(Pi 또는 다른 컴퓨터에서 실행 가능).
4. 노트북에서 LED를 제어하여 테스트합니다: `mosquitto_pub -t "gpio/17/command" -m "toggle"`. LED가 토글되고 상태 토픽이 업데이트되는지 확인합니다.

### 연습 5: 완전한 IoT 파이프라인 -- 센서에서 대시보드까지

센서에서 MQTT 브로커, 로거, 간단한 대시보드로 이어지는 엔드투엔드(End-to-End) 파이프라인을 만드세요:

1. 5초마다 `sensors/<device_id>/temperature`와 `sensors/<device_id>/humidity`에 무작위 온도(20~30°C)와 습도(40~70%)를 발행하는 `SensorPublisher`를 작성합니다(QoS 1, 보존 메시지 사용).
2. `sensors/#`를 구독하고 모든 수신된 읽기를 CSV 파일에 추가하는 `DataLogger` 구독자를 작성합니다. 열: `timestamp`, `device_id`, `metric`, `value`.
3. 동일한 토픽을 구독하고 온도가 27°C를 초과하면 `alerts/<device_id>/high_temp`에 알림을 발행하는 `AlertMonitor` 구독자를 작성합니다(QoS 2, 보존).
4. `sensors/#`와 `alerts/#` 모두를 구독하고, 장치별·지표별 최신 값을 메모리에 유지하며, 현재 읽기와 활성 알람을 보여주는 형식화된 상태 테이블을 15초마다 터미널에 출력하는 `Dashboard` 구독자를 작성합니다.

---

## 다음 단계

- [HTTP/REST for IoT](07_HTTP_REST_for_IoT.md): REST API와 MQTT 통합
- [홈 자동화 프로젝트](10_Home_Automation_Project.md): MQTT 기반 스마트홈

---

*최종 업데이트: 2026-02-01*
