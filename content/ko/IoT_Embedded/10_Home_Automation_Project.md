# 10. 홈 자동화 프로젝트

**이전**: [Edge AI (ONNX)](./09_Edge_AI_ONNX.md) | **다음**: [이미지 분석 프로젝트](./11_Image_Analysis_Project.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 게이트웨이, 센서, 액추에이터를 포함한 스마트홈 시스템 아키텍처를 설계할 수 있다
2. 릴레이 모듈과 GPIO를 통한 조명 제어를 구현할 수 있다
3. 스레드 기반 센서 모니터링 클래스로 환경 상태를 모니터링할 수 있다
4. MQTT 기반 장치 명령 및 상태 시스템을 구축할 수 있다
5. 실시간 장치 제어와 센서 표시를 위한 웹 대시보드를 개발할 수 있다
6. 모든 구성 요소를 통합하여 동작하는 스마트홈 게이트웨이를 완성할 수 있다

---

이 프로젝트 레슨은 지금까지 배운 모든 것 — GPIO 제어, 센서 읽기, MQTT 메시징, Flask 웹 서빙 — 을 하나의 통합 시스템으로 결합합니다. 홈 자동화는 가장 실질적이고 보람 있는 IoT 애플리케이션 중 하나로, 이를 처음부터 끝까지 구축함으로써 개별 기술들이 어떻게 결합되어 실제로 동작하는 제품을 만드는지 직접 경험할 수 있습니다.

---

## 1. 스마트홈 아키텍처

### 1.1 시스템 구성

```
┌─────────────────────────────────────────────────────────────┐
│                    스마트홈 시스템 아키텍처                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    사용자 인터페이스                  │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│   │  │ 웹 대시  │  │ 모바일   │  │ 음성     │          │   │
│   │  │   보드   │  │   앱     │  │ 어시스턴트│          │   │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │   │
│   └───────┼─────────────┼─────────────┼────────────────┘   │
│           │             │             │                     │
│           └─────────────┼─────────────┘                     │
│                         ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  게이트웨이 (라즈베리파이)            │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│   │  │   MQTT   │  │ REST API │  │ 데이터   │          │   │
│   │  │  Broker  │  │  Server  │  │   DB     │          │   │
│   │  └──────────┘  └──────────┘  └──────────┘          │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│        ┌──────────────────┼──────────────────┐              │
│        │                  │                  │              │
│        ▼                  ▼                  ▼              │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│   │  조명    │      │ 온습도   │      │  모션    │         │
│   │  제어    │      │  센서    │      │  센서    │         │
│   │ (릴레이) │      │ (DHT11)  │      │  (PIR)   │         │
│   └──────────┘      └──────────┘      └──────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 하드웨어 구성

| 구성요소 | 모델 | 역할 | GPIO |
|---------|------|------|------|
| **게이트웨이** | Raspberry Pi 4 | 중앙 제어, MQTT 브로커 | - |
| **릴레이 모듈** | 4채널 릴레이 | 조명/가전 제어 | 17, 27, 22, 23 |
| **온습도 센서** | DHT11 | 환경 모니터링 | 4 |
| **모션 센서** | PIR HC-SR501 | 동작 감지 | 24 |
| **조도 센서** | 포토레지스터 | 밝기 감지 | MCP3008 (SPI) |

### 1.3 프로젝트 구조

```
smart_home/
├── gateway/
│   ├── main.py              # 메인 애플리케이션
│   ├── config.py            # 설정
│   ├── mqtt_handler.py      # MQTT 처리
│   ├── device_controller.py # 장치 제어
│   ├── sensor_monitor.py    # 센서 모니터링
│   └── database.py          # 데이터 저장
├── web/
│   ├── app.py               # Flask 웹 서버
│   ├── templates/           # HTML 템플릿
│   └── static/              # CSS, JS
└── requirements.txt
```

---

## 2. 조명 제어 (릴레이)

### 2.1 릴레이 연결

```
┌─────────────────────────────────────────────────────────────┐
│                    릴레이 모듈 연결도                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Raspberry Pi              릴레이 모듈 (4채널)              │
│   ┌──────────┐              ┌──────────────┐                │
│   │ 5V (핀2) │─────────────▶│ VCC          │                │
│   │ GND(핀6) │─────────────▶│ GND          │                │
│   │GPIO17(11)│─────────────▶│ IN1 (조명1)  │                │
│   │GPIO27(13)│─────────────▶│ IN2 (조명2)  │                │
│   │GPIO22(15)│─────────────▶│ IN3 (조명3)  │                │
│   │GPIO23(16)│─────────────▶│ IN4 (조명4)  │                │
│   └──────────┘              └──────────────┘                │
│                                                              │
│   주의: 릴레이는 LOW 활성 (Active Low) 일 수 있음            │
│         GPIO.LOW = 릴레이 ON, GPIO.HIGH = 릴레이 OFF         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 릴레이 제어 클래스

```python
#!/usr/bin/env python3
"""릴레이 조명 제어"""

from gpiozero import OutputDevice
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class Light:
    """조명 장치"""
    id: str
    name: str
    gpio_pin: int
    location: str
    is_on: bool = False

class LightController:
    """조명 제어기"""

    def __init__(self, config: dict):
        self.lights: Dict[str, Light] = {}
        self.relays: Dict[str, OutputDevice] = {}

        # 조명 설정
        for light_config in config.get('lights', []):
            light = Light(**light_config)
            self.lights[light.id] = light

            # 릴레이 초기화 (active_high=False: Active Low 릴레이)
            relay = OutputDevice(
                light.gpio_pin,
                active_high=False,
                initial_value=False
            )
            self.relays[light.id] = relay

    def turn_on(self, light_id: str) -> bool:
        """조명 켜기"""
        if light_id not in self.lights:
            return False

        self.relays[light_id].on()
        self.lights[light_id].is_on = True
        return True

    def turn_off(self, light_id: str) -> bool:
        """조명 끄기"""
        if light_id not in self.lights:
            return False

        self.relays[light_id].off()
        self.lights[light_id].is_on = False
        return True

    def toggle(self, light_id: str) -> bool:
        """조명 토글"""
        if light_id not in self.lights:
            return False

        if self.lights[light_id].is_on:
            return self.turn_off(light_id)
        else:
            return self.turn_on(light_id)

    def get_status(self, light_id: str = None) -> dict:
        """상태 조회"""
        if light_id:
            light = self.lights.get(light_id)
            if light:
                return {
                    "id": light.id,
                    "name": light.name,
                    "location": light.location,
                    "is_on": light.is_on
                }
            return None

        return {
            "lights": [
                {
                    "id": l.id,
                    "name": l.name,
                    "location": l.location,
                    "is_on": l.is_on
                }
                for l in self.lights.values()
            ]
        }

    def all_off(self):
        """모든 조명 끄기"""
        for light_id in self.lights:
            self.turn_off(light_id)

    def all_on(self):
        """모든 조명 켜기"""
        for light_id in self.lights:
            self.turn_on(light_id)

    def cleanup(self):
        """정리"""
        for relay in self.relays.values():
            relay.close()

# 사용 예
if __name__ == "__main__":
    config = {
        "lights": [
            {"id": "living_room", "name": "거실 조명", "gpio_pin": 17, "location": "거실"},
            {"id": "bedroom", "name": "침실 조명", "gpio_pin": 27, "location": "침실"},
            {"id": "kitchen", "name": "주방 조명", "gpio_pin": 22, "location": "주방"},
            {"id": "bathroom", "name": "욕실 조명", "gpio_pin": 23, "location": "욕실"},
        ]
    }

    controller = LightController(config)

    print("조명 상태:", json.dumps(controller.get_status(), indent=2, ensure_ascii=False))

    controller.turn_on("living_room")
    print("거실 조명 ON")

    controller.toggle("bedroom")
    print("침실 조명 토글")

    controller.cleanup()
```

---

## 3. 온도 모니터링

### 3.1 센서 모니터링 클래스

```python
#!/usr/bin/env python3
"""환경 센서 모니터링"""

import time
from datetime import datetime
from dataclasses import dataclass
import threading
import queue

# DHT 센서 라이브러리
import adafruit_dht
import board

@dataclass
class SensorReading:
    """센서 데이터"""
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: datetime

class EnvironmentMonitor:
    """환경 모니터링"""

    def __init__(self, sensor_pin: int = 4, sensor_id: str = "env_01"):
        self.sensor_id = sensor_id
        self.dht = adafruit_dht.DHT11(getattr(board, f"D{sensor_pin}"))

        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None

        # 최근 데이터 저장
        self.latest_reading = None
        self.readings_history = []
        self.max_history = 1000

    def read_sensor(self) -> SensorReading | None:
        """센서 읽기"""
        try:
            temperature = self.dht.temperature
            humidity = self.dht.humidity

            if temperature is not None and humidity is not None:
                reading = SensorReading(
                    sensor_id=self.sensor_id,
                    temperature=temperature,
                    humidity=humidity,
                    timestamp=datetime.now()
                )
                return reading

        except RuntimeError as e:
            # DHT 센서는 가끔 읽기 실패 (정상)
            pass

        return None

    def _monitor_loop(self, interval: int):
        """모니터링 루프"""
        while self.running:
            reading = self.read_sensor()

            if reading:
                self.latest_reading = reading
                self.readings_history.append(reading)

                # 히스토리 크기 제한
                if len(self.readings_history) > self.max_history:
                    self.readings_history.pop(0)

                # 큐에 추가 (외부 구독자용)
                self.data_queue.put(reading)

            time.sleep(interval)

    def start(self, interval: int = 5):
        """모니터링 시작"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.thread.start()
        print(f"환경 모니터링 시작 (간격: {interval}초)")

    def stop(self):
        """모니터링 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.dht.exit()
        print("환경 모니터링 중지")

    def get_latest(self) -> dict | None:
        """최신 데이터"""
        if self.latest_reading:
            return {
                "sensor_id": self.latest_reading.sensor_id,
                "temperature": self.latest_reading.temperature,
                "humidity": self.latest_reading.humidity,
                "timestamp": self.latest_reading.timestamp.isoformat()
            }
        return None

    def get_stats(self) -> dict:
        """통계 데이터"""
        if not self.readings_history:
            return {}

        temps = [r.temperature for r in self.readings_history]
        humids = [r.humidity for r in self.readings_history]

        return {
            "count": len(self.readings_history),
            "temperature": {
                "min": min(temps),
                "max": max(temps),
                "avg": sum(temps) / len(temps)
            },
            "humidity": {
                "min": min(humids),
                "max": max(humids),
                "avg": sum(humids) / len(humids)
            }
        }

# 사용 예
if __name__ == "__main__":
    monitor = EnvironmentMonitor(sensor_pin=4)
    monitor.start(interval=5)

    try:
        while True:
            latest = monitor.get_latest()
            if latest:
                print(f"온도: {latest['temperature']}°C, 습도: {latest['humidity']}%")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
```

---

## 4. MQTT 기반 제어

### 4.1 MQTT 핸들러

```python
#!/usr/bin/env python3
"""MQTT 기반 스마트홈 제어"""

import paho.mqtt.client as mqtt
import json
from datetime import datetime

class SmartHomeMQTT:
    """스마트홈 MQTT 핸들러"""

    TOPICS = {
        "light_command": "home/+/light/command",
        "light_status": "home/{}/light/status",
        "sensor_data": "home/sensor/{}",
        "motion": "home/motion/{}",
        "system": "home/system/status"
    }

    def __init__(self, light_controller, env_monitor, broker: str = "localhost"):
        self.light_controller = light_controller
        self.env_monitor = env_monitor

        self.client = mqtt.Client(client_id="smart_home_gateway")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # LWT 설정
        self.client.will_set(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )

        self.client.connect(broker, 1883)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("MQTT 브로커 연결됨")

            # 토픽 구독
            client.subscribe(self.TOPICS["light_command"])

            # 온라인 상태 발행
            client.publish(
                self.TOPICS["system"],
                json.dumps({"status": "online", "timestamp": datetime.now().isoformat()}),
                qos=1,
                retain=True
            )

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            print(f"수신: {topic} = {payload}")

            # 조명 명령 처리
            if "light/command" in topic:
                self._handle_light_command(topic, payload)

        except json.JSONDecodeError:
            print(f"잘못된 JSON: {msg.payload}")
        except Exception as e:
            print(f"메시지 처리 오류: {e}")

    def _handle_light_command(self, topic: str, payload: dict):
        """조명 명령 처리"""
        # topic: home/{room}/light/command
        parts = topic.split('/')
        room = parts[1] if len(parts) >= 2 else None

        command = payload.get("command")

        if command == "on":
            result = self.light_controller.turn_on(room)
        elif command == "off":
            result = self.light_controller.turn_off(room)
        elif command == "toggle":
            result = self.light_controller.toggle(room)
        else:
            result = False

        # 상태 발행
        status = self.light_controller.get_status(room)
        if status:
            self.publish_light_status(room, status)

    def publish_light_status(self, room: str, status: dict):
        """조명 상태 발행"""
        topic = self.TOPICS["light_status"].format(room)
        self.client.publish(topic, json.dumps(status), qos=1, retain=True)

    def publish_sensor_data(self, sensor_id: str, data: dict):
        """센서 데이터 발행"""
        topic = self.TOPICS["sensor_data"].format(sensor_id)
        self.client.publish(topic, json.dumps(data), qos=0)

    def publish_motion(self, sensor_id: str, detected: bool):
        """모션 감지 발행"""
        topic = self.TOPICS["motion"].format(sensor_id)
        data = {
            "detected": detected,
            "timestamp": datetime.now().isoformat()
        }
        self.client.publish(topic, json.dumps(data), qos=1)

    def start(self):
        """MQTT 루프 시작"""
        self.client.loop_start()

    def stop(self):
        """MQTT 중지"""
        # 오프라인 상태 발행
        self.client.publish(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )
        self.client.loop_stop()
        self.client.disconnect()
```

### 4.2 통합 게이트웨이

```python
#!/usr/bin/env python3
"""스마트홈 통합 게이트웨이"""

import time
import threading
from datetime import datetime
import json

# 이전에 정의한 클래스들 import
# from light_controller import LightController
# from sensor_monitor import EnvironmentMonitor
# from mqtt_handler import SmartHomeMQTT

class SmartHomeGateway:
    """스마트홈 게이트웨이"""

    def __init__(self, config: dict):
        self.config = config

        # 조명 제어기
        self.light_controller = LightController(config)

        # 환경 모니터
        self.env_monitor = EnvironmentMonitor(
            sensor_pin=config.get('dht_pin', 4)
        )

        # MQTT 핸들러
        self.mqtt_handler = SmartHomeMQTT(
            self.light_controller,
            self.env_monitor,
            broker=config.get('mqtt_broker', 'localhost')
        )

        self.running = False

    def _sensor_publish_loop(self, interval: int):
        """센서 데이터 발행 루프"""
        while self.running:
            data = self.env_monitor.get_latest()
            if data:
                self.mqtt_handler.publish_sensor_data("env_01", data)
            time.sleep(interval)

    def start(self):
        """게이트웨이 시작"""
        print("=== 스마트홈 게이트웨이 시작 ===")

        self.running = True

        # 환경 모니터링 시작
        self.env_monitor.start(interval=5)

        # MQTT 시작
        self.mqtt_handler.start()

        # 센서 데이터 발행 스레드
        self.publish_thread = threading.Thread(
            target=self._sensor_publish_loop,
            args=(10,),
            daemon=True
        )
        self.publish_thread.start()

        print("게이트웨이 실행 중...")

    def stop(self):
        """게이트웨이 중지"""
        print("\n게이트웨이 중지 중...")

        self.running = False

        self.env_monitor.stop()
        self.mqtt_handler.stop()
        self.light_controller.all_off()
        self.light_controller.cleanup()

        print("게이트웨이 중지 완료")

    def run(self):
        """메인 루프"""
        self.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

# 메인
if __name__ == "__main__":
    config = {
        "lights": [
            {"id": "living_room", "name": "거실", "gpio_pin": 17, "location": "거실"},
            {"id": "bedroom", "name": "침실", "gpio_pin": 27, "location": "침실"},
        ],
        "dht_pin": 4,
        "mqtt_broker": "localhost"
    }

    gateway = SmartHomeGateway(config)
    gateway.run()
```

---

## 5. 웹 대시보드

### 5.1 Flask 웹 서버

```python
#!/usr/bin/env python3
"""스마트홈 웹 대시보드"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt
import json
import threading

app = Flask(__name__)
CORS(app)

# 상태 저장
state = {
    "lights": {},
    "sensors": {},
    "motion": {}
}

# MQTT 클라이언트
mqtt_client = mqtt.Client()

def on_mqtt_message(client, userdata, msg):
    """MQTT 메시지 처리"""
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic

        if "light/status" in topic:
            room = topic.split('/')[1]
            state["lights"][room] = payload

        elif "sensor" in topic:
            sensor_id = topic.split('/')[-1]
            state["sensors"][sensor_id] = payload

        elif "motion" in topic:
            sensor_id = topic.split('/')[-1]
            state["motion"][sensor_id] = payload

    except Exception as e:
        print(f"메시지 처리 오류: {e}")

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("home/#")
        print("MQTT 연결됨")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

# === 라우트 ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(state)

@app.route('/api/lights')
def get_lights():
    return jsonify(state["lights"])

@app.route('/api/lights/<room>', methods=['POST'])
def control_light(room):
    data = request.get_json()
    command = data.get('command', 'toggle')

    topic = f"home/{room}/light/command"
    mqtt_client.publish(topic, json.dumps({"command": command}))

    return jsonify({"status": "sent", "room": room, "command": command})

@app.route('/api/sensors')
def get_sensors():
    return jsonify(state["sensors"])

@app.route('/api/sensors/<sensor_id>/history')
def get_sensor_history(sensor_id):
    # 실제로는 DB에서 조회
    return jsonify({"sensor_id": sensor_id, "history": []})

def start_mqtt():
    mqtt_client.connect("localhost", 1883)
    mqtt_client.loop_forever()

if __name__ == "__main__":
    # MQTT 스레드 시작
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 5.2 HTML 템플릿

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>스마트홈 대시보드</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; margin-bottom: 20px; }

        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }

        .card {
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h2 { font-size: 1.2rem; color: #666; margin-bottom: 15px; }

        .light-control { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; }
        .light-name { font-weight: 500; }
        .light-btn {
            padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer;
            font-size: 14px; transition: all 0.3s;
        }
        .light-btn.on { background: #4CAF50; color: white; }
        .light-btn.off { background: #ddd; color: #666; }

        .sensor-value { font-size: 2rem; font-weight: bold; color: #333; }
        .sensor-label { color: #666; font-size: 0.9rem; }

        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
        .status.online { background: #4CAF50; }
        .status.offline { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>스마트홈 대시보드</h1>

        <div class="grid">
            <!-- 조명 제어 -->
            <div class="card">
                <h2>조명 제어</h2>
                <div id="lights-container">
                    <p>로딩 중...</p>
                </div>
            </div>

            <!-- 환경 센서 -->
            <div class="card">
                <h2>환경 센서</h2>
                <div id="sensor-container">
                    <div style="display: flex; gap: 30px;">
                        <div>
                            <div class="sensor-value" id="temperature">--</div>
                            <div class="sensor-label">온도 (°C)</div>
                        </div>
                        <div>
                            <div class="sensor-value" id="humidity">--</div>
                            <div class="sensor-label">습도 (%)</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 시스템 상태 -->
            <div class="card">
                <h2>시스템 상태</h2>
                <p><span class="status online"></span> 게이트웨이 온라인</p>
                <p id="last-update">마지막 업데이트: --</p>
            </div>
        </div>
    </div>

    <script>
        // 상태 업데이트
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                // 조명 업데이트
                const lightsContainer = document.getElementById('lights-container');
                lightsContainer.innerHTML = '';

                for (const [room, status] of Object.entries(data.lights)) {
                    const div = document.createElement('div');
                    div.className = 'light-control';
                    div.innerHTML = `
                        <span class="light-name">${status.name || room}</span>
                        <button class="light-btn ${status.is_on ? 'on' : 'off'}"
                                onclick="toggleLight('${room}')">
                            ${status.is_on ? 'ON' : 'OFF'}
                        </button>
                    `;
                    lightsContainer.appendChild(div);
                }

                // 센서 업데이트
                const sensorData = Object.values(data.sensors)[0];
                if (sensorData) {
                    document.getElementById('temperature').textContent =
                        sensorData.temperature?.toFixed(1) || '--';
                    document.getElementById('humidity').textContent =
                        sensorData.humidity?.toFixed(1) || '--';
                }

                document.getElementById('last-update').textContent =
                    '마지막 업데이트: ' + new Date().toLocaleTimeString();

            } catch (error) {
                console.error('업데이트 실패:', error);
            }
        }

        // 조명 토글
        async function toggleLight(room) {
            try {
                await fetch(`/api/lights/${room}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'toggle' })
                });
                setTimeout(updateStatus, 500);
            } catch (error) {
                console.error('조명 제어 실패:', error);
            }
        }

        // 초기화 및 주기적 업데이트
        updateStatus();
        setInterval(updateStatus, 3000);
    </script>
</body>
</html>
```

---

## 6. 전체 시스템 실행

```python
#!/usr/bin/env python3
"""스마트홈 시스템 실행 스크립트"""

import subprocess
import time
import signal
import sys

def main():
    processes = []

    try:
        # 1. Mosquitto 브로커 확인
        print("MQTT 브로커 확인...")
        # subprocess.run(["sudo", "systemctl", "start", "mosquitto"])

        # 2. 게이트웨이 시작
        print("게이트웨이 시작...")
        gateway = subprocess.Popen(
            ["python3", "gateway/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(gateway)

        time.sleep(2)

        # 3. 웹 서버 시작
        print("웹 서버 시작...")
        web = subprocess.Popen(
            ["python3", "web/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(web)

        print("\n=== 스마트홈 시스템 실행 중 ===")
        print("웹 대시보드: http://localhost:5000")
        print("Ctrl+C로 종료\n")

        # 프로세스 모니터링
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"프로세스 종료됨: {p.pid}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n시스템 종료 중...")
    finally:
        for p in processes:
            p.terminate()
        print("시스템 종료 완료")

if __name__ == "__main__":
    main()
```

---

## 연습 문제

### 문제 1: 자동화 규칙
1. 온도가 30도 이상이면 에어컨(릴레이)을 자동으로 켜세요.
2. 모션 감지 시 조명을 자동으로 켜세요.

### 문제 2: 스케줄링
1. 특정 시간에 조명을 자동으로 제어하는 스케줄러를 구현하세요.
2. 일출/일몰 시간에 맞춰 조명을 제어하세요.

### 문제 3: 알림 시스템
1. 온도가 임계값을 초과하면 MQTT로 알림을 발행하세요.
2. 웹 대시보드에 알림을 표시하세요.

---

## 다음 단계

- [영상 분석 프로젝트](11_Image_Analysis_Project.md): AI 카메라 연동
- [클라우드 IoT 통합](12_Cloud_IoT_Integration.md): 클라우드 연동

---

*최종 업데이트: 2026-02-01*
