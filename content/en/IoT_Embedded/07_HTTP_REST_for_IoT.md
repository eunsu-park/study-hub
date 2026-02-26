# 07. HTTP/REST for IoT

**Previous**: [MQTT Protocol](./06_MQTT_Protocol.md) | **Next**: [Edge AI - TensorFlow Lite](./08_Edge_AI_TFLite.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Build an IoT data collection server using Flask
2. Design RESTful sensor data APIs with proper CRUD operations
3. Apply RESTful design principles including resource naming and HTTP methods
4. Validate incoming JSON data using decorators and Pydantic models
5. Implement pagination and filtering for sensor data queries
6. Integrate HTTP and MQTT in a gateway server

---

While MQTT excels at real-time device-to-device messaging, HTTP/REST remains essential for IoT systems that need web dashboards, mobile app integration, or interoperability with existing enterprise systems. A REST API on your gateway lets any client -- a browser, a phone, a cloud service -- query sensor data or send commands using the same standard web protocols that power the entire internet.

---

## 1. Flask IoT Server

### 1.1 Basic Setup

```bash
# Package installation
pip install flask flask-cors

# Additional utilities
pip install python-dotenv
```

```python
#!/usr/bin/env python3
"""Flask IoT Server Basic Structure"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS

# In-memory storage (use DB in production)
sensor_data_store = []
devices = {}

@app.route('/')
def index():
    """API information"""
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
    """Health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 1.2 Project Structure

```
iot_server/
├── app.py              # Main application
├── config.py           # Configuration
├── requirements.txt    # Dependencies
├── routes/             # Route modules
│   ├── __init__.py
│   ├── sensors.py
│   └── devices.py
├── models/             # Data models
│   ├── __init__.py
│   └── sensor.py
└── utils/              # Utilities
    ├── __init__.py
    └── validators.py
```

### 1.3 Configuration Management

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))

    # Database (SQLite example)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///iot_data.db'
    )

    # MQTT settings
    MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
    MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
```

---

## 2. Sensor Data API

### 2.1 Sensor Data CRUD

```python
#!/usr/bin/env python3
"""Sensor Data API"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import uuid

sensors_bp = Blueprint('sensors', __name__)

# In-memory storage
sensor_readings = []
sensors_registry = {}

# === Sensor Registration ===
@sensors_bp.route('/sensors', methods=['GET'])
def list_sensors():
    """List registered sensors"""
    return jsonify({
        "sensors": list(sensors_registry.values()),
        "count": len(sensors_registry)
    })

@sensors_bp.route('/sensors', methods=['POST'])
def register_sensor():
    """Register new sensor"""
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
    """Get sensor information"""
    sensor = sensors_registry.get(sensor_id)

    if not sensor:
        return jsonify({"error": "Sensor not found"}), 404

    return jsonify(sensor)

# === Sensor Data ===
@sensors_bp.route('/sensors/<sensor_id>/data', methods=['POST'])
def post_sensor_data(sensor_id):
    """Receive sensor data"""
    if sensor_id not in sensors_registry:
        # Auto-register (optional)
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

    # Keep only recent 1000 entries
    if len(sensor_readings) > 1000:
        sensor_readings.pop(0)

    return jsonify({"status": "ok", "reading_id": reading["id"]}), 201

@sensors_bp.route('/sensors/<sensor_id>/data', methods=['GET'])
def get_sensor_data(sensor_id):
    """Query sensor data"""
    # Query parameters
    limit = request.args.get('limit', 100, type=int)
    since = request.args.get('since', None)  # ISO timestamp

    # Filter
    readings = [r for r in sensor_readings if r['sensor_id'] == sensor_id]

    if since:
        readings = [r for r in readings if r['timestamp'] > since]

    # Sort and limit (most recent first)
    readings = sorted(readings, key=lambda x: x['timestamp'], reverse=True)[:limit]

    return jsonify({
        "sensor_id": sensor_id,
        "readings": readings,
        "count": len(readings)
    })

@sensors_bp.route('/sensors/<sensor_id>/latest', methods=['GET'])
def get_latest_reading(sensor_id):
    """Get latest sensor data"""
    readings = [r for r in sensor_readings if r['sensor_id'] == sensor_id]

    if not readings:
        return jsonify({"error": "No data found"}), 404

    latest = max(readings, key=lambda x: x['timestamp'])
    return jsonify(latest)

# Register blueprint
# In app.py: app.register_blueprint(sensors_bp, url_prefix='/api')
```

### 2.2 Statistics API

```python
@sensors_bp.route('/sensors/<sensor_id>/stats', methods=['GET'])
def get_sensor_stats(sensor_id):
    """Sensor data statistics"""
    readings = [r for r in sensor_readings if r['sensor_id'] == sensor_id]

    if not readings:
        return jsonify({"error": "No data found"}), 404

    # Extract numeric data (e.g., temperature)
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

## 3. REST API Design

### 3.1 RESTful Principles

```
┌─────────────────────────────────────────────────────────────┐
│                    RESTful API Principles                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Resource-Oriented Design                                 │
│     • Use nouns in URLs: /sensors, /devices                  │
│     • Express actions with HTTP methods                      │
│                                                              │
│  2. HTTP Methods                                             │
│     • GET: Retrieve (idempotent, safe)                       │
│     • POST: Create                                           │
│     • PUT: Full update (idempotent)                          │
│     • PATCH: Partial update                                  │
│     • DELETE: Delete (idempotent)                            │
│                                                              │
│  3. Status Codes                                             │
│     • 200: Success                                           │
│     • 201: Created                                           │
│     • 204: No content (delete)                               │
│     • 400: Bad request                                       │
│     • 401: Authentication required                           │
│     • 404: Resource not found                                │
│     • 500: Server error                                      │
│                                                              │
│  4. Versioning                                               │
│     • URL: /api/v1/sensors                                   │
│     • Header: Accept: application/vnd.api.v1+json            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 IoT API Design Example

```python
# routes/devices.py
"""Device Management API"""

from flask import Blueprint, jsonify, request
from datetime import datetime

devices_bp = Blueprint('devices', __name__)

# Storage
devices = {}

@devices_bp.route('/devices', methods=['GET'])
def list_devices():
    """List devices"""
    # Filtering
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
    """Register device"""
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
    """Get device information"""
    device = devices.get(device_id)
    if not device:
        return jsonify({"error": "Device not found"}), 404
    return jsonify(device)

@devices_bp.route('/devices/<device_id>', methods=['PUT'])
def update_device(device_id):
    """Full device update"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    data = request.get_json()
    data['id'] = device_id  # Preserve ID

    devices[device_id] = {
        **data,
        "updated_at": datetime.now().isoformat()
    }

    return jsonify(devices[device_id])

@devices_bp.route('/devices/<device_id>', methods=['PATCH'])
def patch_device(device_id):
    """Partial device update"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    data = request.get_json()

    # Update only provided fields
    devices[device_id].update(data)
    devices[device_id]['updated_at'] = datetime.now().isoformat()

    return jsonify(devices[device_id])

@devices_bp.route('/devices/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    """Delete device"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    del devices[device_id]
    return '', 204

# === Device Control ===
@devices_bp.route('/devices/<device_id>/commands', methods=['POST'])
def send_command(device_id):
    """Send command to device"""
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404

    data = request.get_json()

    if 'command' not in data:
        return jsonify({"error": "Command required"}), 400

    # Process command (e.g., publish to MQTT)
    command = {
        "device_id": device_id,
        "command": data['command'],
        "params": data.get('params', {}),
        "sent_at": datetime.now().isoformat()
    }

    # Here: publish via MQTT or direct control
    print(f"Command sent: {command}")

    return jsonify({
        "status": "sent",
        "command": command
    }), 202
```

### 3.3 Pagination

```python
@sensors_bp.route('/readings', methods=['GET'])
def list_all_readings():
    """All sensor data (with pagination)"""
    # Query parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # Max 100

    # Sorting
    sort_by = request.args.get('sort', 'timestamp')
    order = request.args.get('order', 'desc')

    # Sort data
    readings = sorted(
        sensor_readings,
        key=lambda x: x.get(sort_by, ''),
        reverse=(order == 'desc')
    )

    # Pagination
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

## 4. JSON Data Processing

### 4.1 Request Validation

```python
# utils/validators.py
"""Request Data Validation"""

from functools import wraps
from flask import request, jsonify

def validate_json(required_fields: list = None, optional_fields: list = None):
    """JSON request validation decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check JSON
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400

            data = request.get_json()

            if data is None:
                return jsonify({"error": "Invalid JSON"}), 400

            # Check required fields
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

# Usage example
@app.route('/api/sensors', methods=['POST'])
@validate_json(required_fields=['name', 'type'])
def create_sensor():
    data = request.get_json()
    # ... process
```

### 4.2 Validation with Pydantic

```python
# models/sensor.py
"""Data Validation with Pydantic Models"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class SensorReading(BaseModel):
    """Sensor data model"""
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
    """Sensor creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., pattern=r'^(temperature|humidity|motion|generic)$')
    location: Optional[str] = None

class DeviceCommand(BaseModel):
    """Device command"""
    command: str = Field(..., pattern=r'^(on|off|toggle|set)$')
    params: Optional[dict] = None

# Use in Flask
from pydantic import ValidationError

@app.route('/api/sensors/<sensor_id>/data', methods=['POST'])
def post_data(sensor_id):
    try:
        reading = SensorReading(**request.get_json())
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    # Use validated data
    data = reading.dict()
    # ...
```

### 4.3 Response Formatting

```python
# utils/response.py
"""Response Helpers"""

from flask import jsonify
from datetime import datetime
from functools import wraps

def api_response(data=None, message=None, status_code=200):
    """Generate standard API response"""
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
    """Generate error response"""
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

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return error_response("Resource not found", 404)

@app.errorhandler(500)
def internal_error(error):
    return error_response("Internal server error", 500)
```

---

## 5. Comprehensive Example: IoT Gateway Server

```python
#!/usr/bin/env python3
"""IoT Gateway Server"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import paho.mqtt.client as mqtt
import threading
import json

app = Flask(__name__)
CORS(app)

# Data storage
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

        # Auto-register sensor
        if sensor_id not in self.sensors:
            self.sensors[sensor_id] = {
                "id": sensor_id,
                "last_seen": reading["timestamp"]
            }
        else:
            self.sensors[sensor_id]["last_seen"] = reading["timestamp"]

        # Keep max 10000 entries
        if len(self.readings) > 10000:
            self.readings = self.readings[-10000:]

store = DataStore()

# === MQTT Client ===
mqtt_client = mqtt.Client()

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT broker connected")
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
        print(f"MQTT message processing error: {e}")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

def start_mqtt():
    try:
        mqtt_client.connect("localhost", 1883)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"MQTT connection failed: {e}")

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

    # Also publish to MQTT (for other subscribers)
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

    # Publish command via MQTT
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
    # Start MQTT thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## Practice Exercises

### Exercise 1: Sensor API
1. Implement sensor CRUD API
2. Add data validation

### Exercise 2: Pagination
1. Implement pagination for sensor data list
2. Add date range filtering

### Exercise 3: MQTT Integration
1. Publish data received via HTTP POST to MQTT
2. Query data received via MQTT through HTTP GET

---

**Previous**: [MQTT Protocol](./06_MQTT_Protocol.md) | **Next**: [Edge AI - TensorFlow Lite](./08_Edge_AI_TFLite.md)
