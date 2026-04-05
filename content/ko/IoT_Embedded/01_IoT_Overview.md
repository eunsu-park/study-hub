# 01. IoT 개요

**다음**: [Raspberry Pi 설정](./02_Raspberry_Pi_Setup.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. IoT(사물인터넷)를 정의하고 핵심 개념을 설명할 수 있다
2. 3계층 IoT 시스템 아키텍처의 구성 요소를 식별할 수 있다
3. 엣지 컴퓨팅과 클라우드 컴퓨팅의 트레이드오프를 비교할 수 있다
4. 주요 IoT 프로토콜과 적합한 사용 사례를 설명할 수 있다
5. 일반적인 IoT 보안 위협과 모범 사례를 파악할 수 있다

---

IoT는 농업, 의료, 제조업 등 모든 산업을 재편하고 있습니다. 연결된 장치나 센서 네트워크를 구축하기 전에 IoT 기초를 이해하는 것은 필수적입니다. 초기에 내리는 아키텍처 결정(어떤 프로토콜을 사용할지, 어디서 데이터를 처리할지, 어떻게 장치를 보호할지)이 프로젝트 전 단계에 걸쳐 연쇄적으로 영향을 미치기 때문입니다.

---

## 1. IoT란 무엇인가?

### 1.1 정의

**IoT(Internet of Things, 사물인터넷)**는 센서, 소프트웨어, 네트워크 연결을 갖춘 물리적 장치들이 데이터를 수집하고 교환하는 시스템입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                        IoT 생태계                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│   │  센서   │    │ 게이트  │    │ 클라우드 │    │  사용자 │  │
│   │ 디바이스│───▶│  웨이   │───▶│  서버   │───▶│   앱    │  │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│        │              │              │              │       │
│   온도, 습도      데이터 집계    저장, 분석     시각화,      │
│   움직임 감지     프로토콜 변환   ML/AI 처리    제어 명령    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 IoT의 핵심 요소

| 요소 | 설명 | 예시 |
|------|------|------|
| **Things** | 센서/액추에이터가 장착된 물리적 장치 | 온도 센서, 스마트 조명 |
| **Connectivity** | 장치 간 데이터 전송을 위한 네트워크 | WiFi, BLE, LoRa, 5G |
| **Data Processing** | 수집된 데이터의 처리 및 분석 | 엣지 처리, 클라우드 분석 |
| **User Interface** | 사용자와 시스템 간 상호작용 | 모바일 앱, 웹 대시보드 |

### 1.3 IoT 활용 분야

```python
# IoT 활용 분야 예시
iot_applications = {
    "스마트홈": ["온도 조절", "조명 제어", "보안 카메라", "음성 비서"],
    "스마트시티": ["교통 관리", "가로등 제어", "쓰레기통 모니터링"],
    "산업 IoT": ["예측 정비", "자산 추적", "품질 관리"],
    "헬스케어": ["웨어러블 기기", "원격 모니터링", "약물 관리"],
    "농업": ["토양 센서", "자동 관개", "드론 모니터링"],
}

for sector, applications in iot_applications.items():
    print(f"\n{sector}:")
    for app in applications:
        print(f"  - {app}")
```

---

## 2. IoT 시스템 아키텍처

### 2.1 3계층 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     3계층 IoT 아키텍처                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 응용 계층 (Application Layer)          │  │
│  │  • 데이터 시각화                                        │  │
│  │  • 비즈니스 로직                                        │  │
│  │  • 사용자 인터페이스                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ▲                                 │
│                            │                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  네트워크 계층 (Network Layer)         │  │
│  │  • 데이터 전송                                         │  │
│  │  • 프로토콜 변환                                       │  │
│  │  • 게이트웨이                                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ▲                                 │
│                            │                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  인식 계층 (Perception Layer)          │  │
│  │  • 센서                                                │  │
│  │  • 액추에이터                                          │  │
│  │  • 임베디드 시스템                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 계층별 역할

```python
# IoT 아키텍처 계층 정의
class IoTArchitecture:
    """IoT 3계층 아키텍처 모델"""

    layers = {
        "perception": {
            "name": "인식 계층",
            "components": ["센서", "액추에이터", "RFID", "GPS"],
            "function": "물리적 환경에서 데이터 수집 및 동작 수행",
            "devices": ["Raspberry Pi", "Arduino", "ESP32"]
        },
        "network": {
            "name": "네트워크 계층",
            "components": ["게이트웨이", "라우터", "프로토콜 변환기"],
            "function": "데이터 전송 및 라우팅",
            "protocols": ["WiFi", "BLE", "LoRa", "Zigbee", "MQTT", "HTTP"]
        },
        "application": {
            "name": "응용 계층",
            "components": ["클라우드 서버", "데이터베이스", "분석 엔진"],
            "function": "데이터 저장, 분석, 시각화",
            "services": ["AWS IoT", "Azure IoT", "Google Cloud IoT"]
        }
    }

    @classmethod
    def describe_layer(cls, layer_name: str):
        layer = cls.layers.get(layer_name)
        if layer:
            print(f"계층: {layer['name']}")
            print(f"기능: {layer['function']}")
            print(f"구성 요소: {', '.join(layer['components'])}")

# 사용 예
IoTArchitecture.describe_layer("perception")
```

---

## 3. 엣지 vs 클라우드 컴퓨팅

### 3.1 개념 비교

```
┌────────────────────────────────────────────────────────────────┐
│                  엣지 vs 클라우드 컴퓨팅                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                      ┌─────────────────────┐ │
│   │   센서      │                      │      클라우드       │ │
│   │  디바이스   │                      │       서버         │ │
│   └──────┬──────┘                      └──────────┬──────────┘ │
│          │                                        │            │
│          ▼                                        │            │
│   ┌─────────────┐      ┌──────────┐              │            │
│   │   엣지      │─────▶│ 네트워크 │──────────────┘            │
│   │  디바이스   │      └──────────┘                           │
│   └─────────────┘                                              │
│          │                                                     │
│   로컬 처리 수행                                               │
│   - 필터링                                                     │
│   - 집계                                                       │
│   - 간단한 분석                                                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 비교표

| 특성 | 엣지 컴퓨팅 | 클라우드 컴퓨팅 |
|------|------------|----------------|
| **처리 위치** | 데이터 소스 근처 | 원격 데이터센터 |
| **지연 시간** | 매우 낮음 (< 10ms) | 높음 (100ms+) |
| **대역폭** | 낮은 요구량 | 높은 요구량 |
| **오프라인** | 동작 가능 | 연결 필요 |
| **비용** | 초기 투자 높음 | 운영 비용 증가 |
| **처리 능력** | 제한적 | 무제한 확장 |
| **예시** | 라즈베리파이 | AWS, GCP, Azure |

### 3.3 하이브리드 접근

```python
# 엣지-클라우드 하이브리드 아키텍처 예시
class HybridIoTSystem:
    """엣지와 클라우드를 결합한 IoT 시스템"""

    def __init__(self):
        self.edge_buffer = []
        self.cloud_threshold = 100  # 100개 데이터마다 클라우드 전송

    def process_at_edge(self, sensor_data: dict) -> dict:
        """엣지에서 즉시 처리할 작업"""
        # 1. 이상치 감지 (즉각 대응 필요)
        if sensor_data.get("temperature", 0) > 50:
            self.trigger_local_alarm()

        # 2. 데이터 필터링/정제
        cleaned_data = self.filter_noise(sensor_data)

        # 3. 로컬 저장 및 집계
        self.edge_buffer.append(cleaned_data)

        return cleaned_data

    def should_send_to_cloud(self) -> bool:
        """클라우드 전송 조건 확인"""
        return len(self.edge_buffer) >= self.cloud_threshold

    def send_to_cloud(self):
        """집계된 데이터를 클라우드로 전송"""
        if self.should_send_to_cloud():
            aggregated = self.aggregate_data(self.edge_buffer)
            # cloud_client.publish(aggregated)
            self.edge_buffer.clear()
            return aggregated

    def filter_noise(self, data: dict) -> dict:
        """노이즈 제거 (엣지 처리)"""
        return {k: v for k, v in data.items() if v is not None}

    def aggregate_data(self, buffer: list) -> dict:
        """데이터 집계"""
        if not buffer:
            return {}
        temps = [d.get("temperature", 0) for d in buffer]
        return {
            "avg_temperature": sum(temps) / len(temps),
            "max_temperature": max(temps),
            "min_temperature": min(temps),
            "count": len(buffer)
        }

    def trigger_local_alarm(self):
        """로컬 알람 트리거 (저지연 응답)"""
        print("WARNING: High temperature detected!")
```

---

## 4. IoT 프로토콜 개요

### 4.1 프로토콜 스택

```
┌─────────────────────────────────────────────────────────────┐
│                    IoT 프로토콜 스택                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────┐   │
│  │  HTTP   │  MQTT   │  CoAP   │  AMQP   │  WebSocket  │   │
│  │         │         │         │         │             │   │
│  ├─────────┴─────────┴─────────┴─────────┴─────────────┤   │
│  │                    TCP / UDP                         │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                       IP                             │   │
│  ├─────────┬─────────┬─────────┬─────────┬─────────────┤   │
│  │  WiFi   │   BLE   │  LoRa   │ Zigbee  │  Cellular   │   │
│  │ 802.11  │ 802.15.1│         │ 802.15.4│   4G/5G     │   │
│  └─────────┴─────────┴─────────┴─────────┴─────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 주요 애플리케이션 프로토콜

#### MQTT (Message Queuing Telemetry Transport)

```python
# MQTT 프로토콜 특성
mqtt_characteristics = {
    "type": "Pub/Sub 메시징",
    "transport": "TCP",
    "port": 1883,  # TLS: 8883
    "qos_levels": [0, 1, 2],  # At most once, At least once, Exactly once
    "use_cases": ["센서 데이터", "실시간 모니터링", "알림"],
    "advantages": ["경량", "저대역폭", "신뢰성 옵션"],
    "broker": ["Mosquitto", "HiveMQ", "EMQX"]
}

# MQTT 토픽 구조 예시
topics = [
    "home/living-room/temperature",
    "home/living-room/humidity",
    "home/+/temperature",  # + : 단일 레벨 와일드카드
    "home/#"               # # : 다중 레벨 와일드카드
]
```

#### HTTP/REST

```python
# REST API IoT 패턴
rest_patterns = {
    "GET /sensors": "모든 센서 목록 조회",
    "GET /sensors/{id}": "특정 센서 정보 조회",
    "GET /sensors/{id}/data": "센서 데이터 조회",
    "POST /sensors/{id}/data": "새 센서 데이터 전송",
    "PUT /devices/{id}/config": "디바이스 설정 변경",
}

# RESTful IoT 요청 예시
import requests

def get_sensor_data(sensor_id: str, api_base: str = "http://iot-server:8080"):
    """센서 데이터 조회"""
    response = requests.get(f"{api_base}/sensors/{sensor_id}/data")
    return response.json()

def post_sensor_reading(sensor_id: str, data: dict, api_base: str = "http://iot-server:8080"):
    """센서 데이터 전송"""
    response = requests.post(
        f"{api_base}/sensors/{sensor_id}/data",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    return response.status_code == 201
```

#### CoAP (Constrained Application Protocol)

```python
# CoAP 프로토콜 특성
coap_characteristics = {
    "type": "RESTful (UDP 기반)",
    "transport": "UDP",
    "port": 5683,  # DTLS: 5684
    "features": ["경량 HTTP 대안", "멀티캐스트 지원", "비동기 메시징"],
    "use_cases": ["저전력 디바이스", "제한된 네트워크"],
    "message_types": ["CON", "NON", "ACK", "RST"]
}
```

### 4.3 프로토콜 선택 가이드

| 요구사항 | 권장 프로토콜 |
|----------|--------------|
| 실시간 양방향 통신 | MQTT, WebSocket |
| 저전력 디바이스 | CoAP, MQTT-SN |
| 기존 웹 인프라 활용 | HTTP/REST |
| 대규모 메시지 처리 | AMQP, Kafka |
| 단순 데이터 수집 | MQTT |

---

## 5. IoT 보안 고려사항

### 5.1 보안 위협

```
┌─────────────────────────────────────────────────────────────┐
│                    IoT 보안 위협 유형                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  디바이스   │    │   네트워크  │    │   클라우드  │     │
│  │   공격      │    │    공격     │    │    공격     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        │                  │                  │              │
│  • 물리적 접근      • 중간자 공격      • 인증 우회          │
│  • 펌웨어 추출      • 도청             • API 취약점         │
│  • 사이드채널      • 스푸핑           • 데이터 유출         │
│  • 기본 자격증명    • DDoS            • 권한 상승           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 보안 모범 사례

```python
# IoT 보안 체크리스트
security_checklist = {
    "device_security": [
        "기본 비밀번호 변경",
        "펌웨어 정기 업데이트",
        "불필요한 포트/서비스 비활성화",
        "Secure Boot 활성화",
        "물리적 접근 제한"
    ],
    "network_security": [
        "TLS/DTLS 암호화 사용",
        "네트워크 분리 (VLAN)",
        "방화벽 설정",
        "VPN 사용 고려"
    ],
    "data_security": [
        "전송 중 암호화 (TLS)",
        "저장 시 암호화 (AES)",
        "최소 권한 원칙",
        "정기적 백업"
    ],
    "authentication": [
        "강력한 인증 메커니즘",
        "인증서 기반 인증",
        "토큰 기반 인증 (JWT)",
        "API 키 관리"
    ]
}

# TLS를 사용한 MQTT 연결 예시
import ssl

def create_secure_mqtt_context():
    """보안 MQTT 연결을 위한 SSL 컨텍스트 생성"""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
        certfile="client-cert.pem",
        keyfile="client-key.pem"
    )
    context.load_verify_locations("ca-cert.pem")
    context.verify_mode = ssl.CERT_REQUIRED
    return context
```

### 5.3 데이터 암호화 예시

```python
from cryptography.fernet import Fernet
import json

class SecureDataHandler:
    """IoT 데이터 암호화 처리"""

    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_data(self, data: dict) -> bytes:
        """센서 데이터 암호화"""
        json_data = json.dumps(data).encode()
        return self.cipher.encrypt(json_data)

    def decrypt_data(self, encrypted: bytes) -> dict:
        """암호화된 데이터 복호화"""
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())

# 사용 예
handler = SecureDataHandler()
sensor_data = {"temperature": 25.5, "humidity": 60}

encrypted = handler.encrypt_data(sensor_data)
print(f"암호화된 데이터: {encrypted[:50]}...")

decrypted = handler.decrypt_data(encrypted)
print(f"복호화된 데이터: {decrypted}")
```

---

## 6. IoT 개발 환경

### 6.1 하드웨어 플랫폼

| 플랫폼 | CPU | RAM | 특징 | 용도 |
|--------|-----|-----|------|------|
| Raspberry Pi 4 | ARM Cortex-A72 | 1-8GB | Full Linux OS | 게이트웨이, 엣지 AI |
| Raspberry Pi Pico | RP2040 | 264KB | 마이크로컨트롤러 | 센서 노드 |
| ESP32 | Xtensa LX6 | 520KB | WiFi/BLE 내장 | IoT 센서 |
| Arduino | ATmega/ARM | 2-256KB | 단순, 저전력 | 프로토타이핑 |

### 6.2 개발 도구

```python
# 권장 개발 환경
dev_environment = {
    "ide": ["VS Code + Remote SSH", "Thonny", "PyCharm"],
    "languages": ["Python 3.9+", "MicroPython", "C/C++"],
    "debugging": ["print debugging", "logging", "remote debugger"],
    "testing": ["pytest", "unittest", "hardware simulation"],
    "version_control": ["Git"],
    "ci_cd": ["GitHub Actions", "GitLab CI"]
}
```

---

## 연습 문제

### 문제 1: IoT 시스템 설계
스마트 주차장 시스템을 설계하세요. 다음을 포함해야 합니다:
- 필요한 센서 종류
- 통신 프로토콜 선택 (이유 포함)
- 엣지 vs 클라우드 처리 분배

### 문제 2: 프로토콜 선택
다음 시나리오에 적합한 프로토콜을 선택하고 이유를 설명하세요:
1. 배터리로 동작하는 원격 온도 센서
2. 실시간 보안 카메라 영상 스트리밍
3. 스마트 조명 제어 시스템

### 문제 3: 보안 분석
가정용 스마트 도어락의 잠재적 보안 취약점 3가지를 나열하고, 각각에 대한 대응 방안을 제시하세요.

---

## 다음 단계

- [라즈베리파이 설정](02_Raspberry_Pi_Setup.md): 라즈베리파이 설정으로 실습 환경 구축
- [MQTT 프로토콜](06_MQTT_Protocol.md): MQTT 프로토콜 심화 학습
- [클라우드 IoT 통합](12_Cloud_IoT_Integration.md): 클라우드 IoT 서비스 연동

---

*최종 업데이트: 2026-02-01*
