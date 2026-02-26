# 07. HTTP/REST for IoT

**이전**: [MQTT 프로토콜](./06_MQTT_Protocol.md) | **다음**: [Edge AI (TFLite)](./08_Edge_AI_TFLite.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Flask를 사용하여 IoT 데이터 수집 서버를 구축할 수 있다
2. 적절한 CRUD 연산을 갖춘 RESTful 센서 데이터 API를 설계할 수 있다
3. 리소스 명명 규칙과 HTTP 메서드를 포함한 RESTful 설계 원칙을 적용할 수 있다
4. 데코레이터와 Pydantic 모델을 사용하여 수신 JSON 데이터를 검증할 수 있다
5. 센서 데이터 쿼리에 페이지네이션(pagination)과 필터링을 구현할 수 있다
6. 게이트웨이 서버에서 HTTP와 MQTT를 통합할 수 있다

---

MQTT가 실시간 장치 간 메시징에 탁월하다면, HTTP/REST는 웹 대시보드, 모바일 앱 통합, 또는 기존 엔터프라이즈 시스템과의 상호 운용성이 필요한 IoT 시스템에 여전히 필수적입니다. 게이트웨이의 REST API는 브라우저, 스마트폰, 클라우드 서비스 등 어떤 클라이언트든 인터넷 전체를 구동하는 동일한 표준 웹 프로토콜로 센서 데이터를 조회하거나 명령을 보낼 수 있게 해줍니다.

---

## 1. Flask IoT 서버

### 1.1 기본 설정

```bash
# 패키지 설치
pip install flask flask-cors

# 추가 유틸리티
pip install python-dotenv
```

```python
#!/usr/bin/env python3
"""Flask IoT 서버 기본 구조"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # CORS 활성화

# 메모리 저장소 (실제 프로젝트에서는 DB 사용)
sensor_data_store = []
devices = {}

@app.route('/')
def index():
    """API 정보"""
    return jsonify({
        "name": "IoT API Server",
        "version": "1.0",
        "endpoints": {
            "/api/sensors": "GET, POST",
            "/api/sensors/<id>": "GET",
            "/api/devices": "GET, POST",
            "/api/devices/<id>": "GET, PUT, DELETE"
        }
    })

@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 1.2 프로젝트 구조

```
iot_server/
├── app.py              # 메인 애플리케이션
├── config.py           # 설정
├── requirements.txt    # 의존성
├── routes/             # 라우트 모듈
│   ├── __init__.py
│   ├── sensors.py
│   └── devices.py
├── models/             # 데이터 모델
│   ├── __init__.py
│   └── sensor.py
└── utils/              # 유틸리티
    ├── __init__.py
    └── validators.py
```

### 1.3 설정 관리

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """애플리케이션 설정"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))

    # 데이터베이스 (SQLite 예시)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///iot_data.db'
    )

    # MQTT 설정
    MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
    MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
```

---

## 2. 센서 데이터 API

### 2.1 센서 데이터 CRUD

```python
#!/usr/bin/env python3
"""센서 데이터 API"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import uuid

sensors_bp = Blueprint('sensors', __name__)

# 메모리 저장소
sensor_readings = []
sensors_registry = {}

# === 센서 등록 ===
@sensors_bp.route('/sensors', methods=['GET'])
def list_sensors():
    """등록된 센서 목록"""
    return jsonify({
        "sensors": list(sensors_registry.values()),
        "count": len(sensors_registry)
    })

@sensors_bp.route('/sensors', methods=['POST'])
def register_sensor():
    """새 센서 등록"""
    data = request.get_json()

    if not data or 'name' not in data:
        return jsonify({"error": "name is required"}), 400

    sensor_id = str(uuid.uuid4())[:8]
    sensor = {
        "id": sensor_id,
        "name": data['name'],
        "type": data.get('type', 'generic'),
        "location": data.get('location', 'unknown'),
        "registered_at": datetime.now().isoformat(),
        "status": "active"
    }

    sensors_registry[sensor_id] = sensor

    return jsonify(sensor), 201

@sensors_bp.route('/sensors/<sensor_id>', methods=['GET'])
def get_sensor(sensor_id):
    """센서 정보 조회"""
    sensor = sensors_registry.get(sensor_id)

    if not sensor:
        return jsonify({"error": "Sensor not found"}), 404

    return jsonify(sensor)

# === 센서 데이터 ===
@sensors_bp.route('/sensors/<sensor_id>/data', methods=['POST'])
def post_sensor_data(sensor_id):
    """센서 데이터 수신"""
    if sensor_id not in sensors_registry:
        # 자동 등록 (옵션)
        sensors_registry[sensor_id] = {
            "id": sensor_id,
            "name": f"auto_{sensor_id}",
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    reading = {
        "id": str(uuid.uuid4()),
        "sensor_id": sensor_id,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

    sensor_readings.append(reading)

    # 최근 1000개만 유지
    if len(sensor_readings) > 1000:
        sensor_readings.pop(0)

    return jsonify({"status": "ok", "reading_id": reading["id"]}), 201

@sensors_bp.route('/sensors/<sensor_id>/data', methods=['GET'])
def get_sensor_data(sensor_id):
    """센서 데이터 조회"""
    # 쿼리 파라미터
    limit = request.args.get('limit', 100, type=int)
    since = request.args.get('since', None)  # ISO timestamp

    # 필터링
    readings = [r for r in sensor_readings if r['sensor_id'] == sensor_id]

    if since:
        readings = [r for r in readings if r['timestamp'] > since]

    # 최신순 정렬 및 제한
    readings = sorted(readings, key=lambda x: x['timestamp'], reverse=True)[:limit]

    return jsonify({
        "sensor_id": sensor_id,
        "readings": readings,
        "count": len(readings)
    })

@sensors_bp.route('/sensors/<sensor_id>/latest', methods=['GET'])
def get_latest_reading(sensor_id):
    """최신 센서 데이터"""
    readings = [r for r in sensor_readings if r['sensor_id'] == sensor_id]

    if not readings:
        return jsonify({"error": "No data found"}), 404

    latest = max(readings, key=lambda x: x['timestamp'])
    return jsonify(latest)

# 블루프린트 등록
# app.py에서: app.register_blueprint(sensors_bp, url_prefix='/api')
```

### 2.2 집계 API

```python
@sensors_bp.route('/sensors/<sensor_id>/stats', methods=['GET'])
def get_sensor_stats(sensor_id):
    """센서 데이터 통계"""
    readings = [r for r in sensor_readings if r['sensor_id'] == sensor_id]

    if not readings:
        return jsonify({"error": "No data found"}), 404

    # 숫자 데이터 추출 (예: temperature)
    field = request.args.get('field', 'temperature')

    values = []
    for r in readings:
        if field in r.get('data', {}):
            try:
                values.append(float(r['data'][field]))
            except (ValueError, TypeError):
                pass

    if not values:
        return jsonify({"error": f"No numeric data for field: {field}"}), 404

    stats = {
        "sensor_id": sensor_id,
        "field": field,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "latest": values[-1] if values else None
    }

    return jsonify(stats)
```

---

## 3. REST API 설계

### 3.1 RESTful 원칙

```
┌─────────────────────────────────────────────────────────────┐
│                    RESTful API 원칙                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 리소스 중심 설계                                         │
│     • URL은 명사 사용: /sensors, /devices                   │
│     • 동사는 HTTP 메서드로 표현                              │
│                                                              │
│  2. HTTP 메서드                                              │
│     • GET: 조회 (멱등, 안전)                                │
│     • POST: 생성                                            │
│     • PUT: 전체 수정 (멱등)                                 │
│     • PATCH: 부분 수정                                      │
│     • DELETE: 삭제 (멱등)                                   │
│                                                              │
│  3. 상태 코드                                                │
│     • 200: 성공                                             │
│     • 201: 생성됨                                           │
│     • 204: 내용 없음 (삭제)                                 │
│     • 400: 잘못된 요청                                      │
│     • 401: 인증 필요                                        │
│     • 404: 리소스 없음                                      │
│     • 500: 서버 오류                                        │
│                                                              │
│  4. 버저닝                                                   │
│     • URL: /api/v1/sensors                                  │
│     • 헤더: Accept: application/vnd.api.v1+json             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 IoT API 설계 예시

```python
# routes/devices.py
"""장치 관리 API"""

from flask import Blueprint, jsonify, request
from datetime import datetime

devices_bp = Blueprint('devices', __name__)

# 저장소
devices = {}

@devices_bp.route('/devices', methods=['GET'])
def list_devices():
    """장치 목록 조회"""
    # 필터링
    device_type = request.args.get('type')
    status = request.args.get('status')

    result = list(devices.values())

    if device_type:
        result = [d for d in result if d.get('type') == device_type]
    if status:
        result = [d for d in result if d.get('status') == status]

    return jsonify({
        "devices": result,
        "total": len(result)
    })

@devices_bp.route('/devices', methods=['POST'])
def create_device():
    """장치 등록"""
    data = request.get_json()

    required_fields = ['id', 'name', 'type']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    device_id = data['id']
    if device_id in devices:
        return jsonify({"error": "Device already exists"}), 409

    device = {
        **data,
        "status": "offline",
        "created_at": datetime.now().isoformat(),
        "last_seen": None
    }

    devices[device_id] = device
    return jsonify(device), 201

@devices_bp.route('/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    """장치 정보 조회"""
    device = devices.get(device_id)
    if not device:
        return jsonify({"error": "Device not found"}), 404
    return jsonify(device)

@devices_bp.route('/devices/<device_id>', methods=['PUT'])
def update_device(device_id):
    """장치 정보 전체 수정"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    data = request.get_json()
    data['id'] = device_id  # ID 유지

    devices[device_id] = {
        **data,
        "updated_at": datetime.now().isoformat()
    }

    return jsonify(devices[device_id])

@devices_bp.route('/devices/<device_id>', methods=['PATCH'])
def patch_device(device_id):
    """장치 정보 부분 수정"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    data = request.get_json()

    # 일부 필드만 업데이트
    devices[device_id].update(data)
    devices[device_id]['updated_at'] = datetime.now().isoformat()

    return jsonify(devices[device_id])

@devices_bp.route('/devices/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    """장치 삭제"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    del devices[device_id]
    return '', 204

# === 장치 제어 ===
@devices_bp.route('/devices/<device_id>/commands', methods=['POST'])
def send_command(device_id):
    """장치에 명령 전송"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    data = request.get_json()

    if 'command' not in data:
        return jsonify({"error": "Command required"}), 400

    # 명령 처리 (실제로는 MQTT 발행 등)
    command = {
        "device_id": device_id,
        "command": data['command'],
        "params": data.get('params', {}),
        "sent_at": datetime.now().isoformat()
    }

    # 여기서 MQTT 발행 또는 직접 제어
    print(f"Command sent: {command}")

    return jsonify({
        "status": "sent",
        "command": command
    }), 202
```

### 3.3 페이지네이션

```python
@sensors_bp.route('/readings', methods=['GET'])
def list_all_readings():
    """모든 센서 데이터 (페이지네이션)"""
    # 쿼리 파라미터
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # 최대 100개

    # 정렬
    sort_by = request.args.get('sort', 'timestamp')
    order = request.args.get('order', 'desc')

    # 데이터 정렬
    readings = sorted(
        sensor_readings,
        key=lambda x: x.get(sort_by, ''),
        reverse=(order == 'desc')
    )

    # 페이지네이션
    total = len(readings)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = readings[start:end]

    return jsonify({
        "readings": page_data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page,
            "has_next": end < total,
            "has_prev": page > 1
        }
    })
```

---

## 4. JSON 데이터 처리

### 4.1 요청 검증

```python
# utils/validators.py
"""요청 데이터 검증"""

from functools import wraps
from flask import request, jsonify

def validate_json(required_fields: list = None, optional_fields: list = None):
    """JSON 요청 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # JSON 확인
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400

            data = request.get_json()

            if data is None:
                return jsonify({"error": "Invalid JSON"}), 400

            # 필수 필드 확인
            if required_fields:
                missing = [f for f in required_fields if f not in data]
                if missing:
                    return jsonify({
                        "error": "Missing required fields",
                        "fields": missing
                    }), 400

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# 사용 예
@app.route('/api/sensors', methods=['POST'])
@validate_json(required_fields=['name', 'type'])
def create_sensor():
    data = request.get_json()
    # ... 처리
```

### 4.2 Pydantic을 이용한 검증

```python
# models/sensor.py
"""Pydantic 모델로 데이터 검증"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class SensorReading(BaseModel):
    """센서 데이터 모델"""
    temperature: Optional[float] = Field(None, ge=-50, le=100)
    humidity: Optional[float] = Field(None, ge=0, le=100)
    pressure: Optional[float] = Field(None, ge=800, le=1200)
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('temperature', 'humidity', 'pressure', pre=True)
    def round_values(cls, v):
        if v is not None:
            return round(v, 2)
        return v

class SensorCreate(BaseModel):
    """센서 생성 요청"""
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., pattern=r'^(temperature|humidity|motion|generic)$')
    location: Optional[str] = None

class DeviceCommand(BaseModel):
    """장치 명령"""
    command: str = Field(..., pattern=r'^(on|off|toggle|set)$')
    params: Optional[dict] = None

# Flask에서 사용
from pydantic import ValidationError

@app.route('/api/sensors/<sensor_id>/data', methods=['POST'])
def post_data(sensor_id):
    try:
        reading = SensorReading(**request.get_json())
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    # 검증된 데이터 사용
    data = reading.dict()
    # ...
```

### 4.3 응답 포맷팅

```python
# utils/response.py
"""응답 헬퍼"""

from flask import jsonify
from datetime import datetime
from functools import wraps

def api_response(data=None, message=None, status_code=200):
    """표준 API 응답 생성"""
    response = {
        "success": 200 <= status_code < 300,
        "timestamp": datetime.now().isoformat()
    }

    if message:
        response["message"] = message
    if data is not None:
        response["data"] = data

    return jsonify(response), status_code

def error_response(message: str, status_code: int = 400, details=None):
    """에러 응답 생성"""
    response = {
        "success": False,
        "error": {
            "message": message,
            "code": status_code
        },
        "timestamp": datetime.now().isoformat()
    }

    if details:
        response["error"]["details"] = details

    return jsonify(response), status_code

# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return error_response("Resource not found", 404)

@app.errorhandler(500)
def internal_error(error):
    return error_response("Internal server error", 500)
```

---

## 5. 종합 예제: IoT 게이트웨이 서버

```python
#!/usr/bin/env python3
"""IoT 게이트웨이 서버"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import paho.mqtt.client as mqtt
import threading
import json

app = Flask(__name__)
CORS(app)

# 데이터 저장소
class DataStore:
    def __init__(self):
        self.sensors = {}
        self.readings = []
        self.devices = {}
        self.commands = []

    def add_reading(self, sensor_id: str, data: dict):
        reading = {
            "sensor_id": sensor_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.readings.append(reading)

        # 센서 자동 등록
        if sensor_id not in self.sensors:
            self.sensors[sensor_id] = {
                "id": sensor_id,
                "last_seen": reading["timestamp"]
            }
        else:
            self.sensors[sensor_id]["last_seen"] = reading["timestamp"]

        # 최대 10000개 유지
        if len(self.readings) > 10000:
            self.readings = self.readings[-10000:]

store = DataStore()

# === MQTT 클라이언트 ===
mqtt_client = mqtt.Client()

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT 브로커 연결됨")
        client.subscribe("sensor/#")

def on_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        # topic: sensor/{sensor_id}/data
        parts = msg.topic.split('/')
        if len(parts) >= 2:
            sensor_id = parts[1]
            store.add_reading(sensor_id, payload)
            print(f"[MQTT] {sensor_id}: {payload}")
    except Exception as e:
        print(f"MQTT 메시지 처리 오류: {e}")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

def start_mqtt():
    try:
        mqtt_client.connect("localhost", 1883)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"MQTT 연결 실패: {e}")

# === HTTP API ===
@app.route('/api/sensors', methods=['GET'])
def list_sensors():
    return jsonify({
        "sensors": list(store.sensors.values()),
        "count": len(store.sensors)
    })

@app.route('/api/sensors/<sensor_id>/data', methods=['GET'])
def get_sensor_data(sensor_id):
    limit = request.args.get('limit', 100, type=int)

    readings = [r for r in store.readings if r['sensor_id'] == sensor_id]
    readings = readings[-limit:]

    return jsonify({
        "sensor_id": sensor_id,
        "readings": readings,
        "count": len(readings)
    })

@app.route('/api/sensors/<sensor_id>/data', methods=['POST'])
def post_sensor_data(sensor_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    store.add_reading(sensor_id, data)

    # MQTT로도 발행 (다른 구독자에게)
    mqtt_client.publish(f"sensor/{sensor_id}/data", json.dumps(data))

    return jsonify({"status": "ok"}), 201

@app.route('/api/devices/<device_id>/command', methods=['POST'])
def send_device_command(device_id):
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({"error": "Command required"}), 400

    command = {
        "device_id": device_id,
        "command": data['command'],
        "params": data.get('params', {}),
        "timestamp": datetime.now().isoformat()
    }

    # MQTT로 명령 발행
    mqtt_client.publish(
        f"device/{device_id}/command",
        json.dumps(command)
    )

    store.commands.append(command)

    return jsonify({"status": "sent", "command": command}), 202

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "sensors_count": len(store.sensors),
        "readings_count": len(store.readings),
        "devices_count": len(store.devices),
        "commands_count": len(store.commands)
    })

if __name__ == "__main__":
    # MQTT 스레드 시작
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()

    # Flask 서버 시작
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## 연습 문제

### 문제 1: 센서 API
1. 센서 CRUD API를 구현하세요.
2. 데이터 검증을 추가하세요.

### 문제 2: 페이지네이션
1. 센서 데이터 목록에 페이지네이션을 구현하세요.
2. 날짜 범위 필터링을 추가하세요.

### 문제 3: MQTT 연동
1. HTTP POST로 받은 데이터를 MQTT로 발행하세요.
2. MQTT로 받은 데이터를 HTTP GET으로 조회하세요.

---

## 다음 단계

- [Edge AI - TensorFlow Lite](08_Edge_AI_TFLite.md): 센서 데이터 AI 분석
- [홈 자동화 프로젝트](10_Home_Automation_Project.md): REST API 기반 스마트홈

---

*최종 업데이트: 2026-02-01*
