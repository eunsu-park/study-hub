#!/usr/bin/env python3
"""
IoT Flask REST API Server
Simple sensor data collection and device control API

Reference: content/ko/IoT_Embedded/07_HTTP_REST_for_IoT.md
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import uuid
import sqlite3
import os

app = Flask(__name__)
# Why: CORS is required because IoT dashboards are usually served from a
# different origin than the API, and browsers block cross-origin requests by default.
CORS(app)

# === Database Setup ===
# Why: Supporting both SQLite and in-memory storage lets the same codebase run
# in production (persistent disk) and in quick demos/tests (volatile memory).
DB_PATH = "iot_data.db"
USE_SQLITE = True  # Set to False to use in-memory storage


def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Sensor table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT,
            location TEXT,
            status TEXT DEFAULT 'active',
            registered_at TEXT
        )
    """)

    # Sensor data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id TEXT PRIMARY KEY,
            sensor_id TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (sensor_id) REFERENCES sensors(id)
        )
    """)

    # Device table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT,
            status TEXT DEFAULT 'offline',
            created_at TEXT,
            last_seen TEXT
        )
    """)

    conn.commit()
    conn.close()


# In-memory storage (simulation mode)
memory_store = {
    'sensors': {},
    'sensor_readings': [],
    'devices': {}
}


# === Helper Functions ===
def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def dict_from_row(row):
    """Convert sqlite3.Row to dictionary"""
    return dict(zip(row.keys(), row))


# === API Endpoints ===

@app.route('/')
def index():
    """API information"""
    return jsonify({
        "name": "IoT REST API Server",
        "version": "1.0",
        "storage": "SQLite" if USE_SQLITE else "Memory",
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/api/sensors": "GET, POST - Sensor list/register",
            "/api/sensors/<id>": "GET - Sensor info query",
            "/api/sensors/<id>/data": "GET, POST - Sensor data query/submit",
            "/api/sensors/<id>/latest": "GET - Latest sensor data",
            "/api/sensors/<id>/stats": "GET - Sensor statistics",
            "/api/devices": "GET, POST - Device list/register",
            "/api/devices/<id>": "GET, PUT, DELETE - Device query/update/delete",
            "/api/devices/<id>/command": "POST - Send device command"
        }
    })


@app.route('/health')
def health():
    """Health check"""
    if USE_SQLITE:
        # Check DB connection
        try:
            conn = get_db_connection()
            conn.close()
            status = "healthy"
        except Exception as e:
            status = f"unhealthy: {str(e)}"
    else:
        status = "healthy (memory mode)"

    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "storage": "SQLite" if USE_SQLITE else "Memory"
    })


# === Sensor API ===

@app.route('/api/sensors', methods=['GET'])
def list_sensors():
    """List registered sensors"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensors")
        sensors = [dict_from_row(row) for row in cursor.fetchall()]
        conn.close()
    else:
        sensors = list(memory_store['sensors'].values())

    return jsonify({
        "sensors": sensors,
        "count": len(sensors)
    })


@app.route('/api/sensors', methods=['POST'])
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
        "status": "active",
        "registered_at": datetime.now().isoformat()
    }

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sensors (id, name, type, location, status, registered_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sensor['id'], sensor['name'], sensor['type'],
              sensor['location'], sensor['status'], sensor['registered_at']))
        conn.commit()
        conn.close()
    else:
        memory_store['sensors'][sensor_id] = sensor

    return jsonify(sensor), 201


@app.route('/api/sensors/<sensor_id>', methods=['GET'])
def get_sensor(sensor_id):
    """Get sensor information"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensors WHERE id = ?", (sensor_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Sensor not found"}), 404

        sensor = dict_from_row(row)
    else:
        sensor = memory_store['sensors'].get(sensor_id)
        if not sensor:
            return jsonify({"error": "Sensor not found"}), 404

    return jsonify(sensor)


@app.route('/api/sensors/<sensor_id>/data', methods=['POST'])
def post_sensor_data(sensor_id):
    """Receive sensor data"""
    import json

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    reading_id = str(uuid.uuid4())
    reading = {
        "id": reading_id,
        "sensor_id": sensor_id,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

    if USE_SQLITE:
        # Why: Auto-registering unknown sensors on first POST removes the manual
        # provisioning step. IoT devices can start sending data immediately after
        # flashing, which drastically simplifies fleet onboarding.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sensors WHERE id = ?", (sensor_id,))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO sensors (id, name, type, status, registered_at)
                VALUES (?, ?, ?, ?, ?)
            """, (sensor_id, f"auto_{sensor_id}", "generic",
                  "active", datetime.now().isoformat()))

        # Save sensor data
        cursor.execute("""
            INSERT INTO sensor_readings (id, sensor_id, data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (reading['id'], reading['sensor_id'],
              json.dumps(reading['data']), reading['timestamp']))
        conn.commit()
        conn.close()
    else:
        # Auto-register sensor
        if sensor_id not in memory_store['sensors']:
            memory_store['sensors'][sensor_id] = {
                "id": sensor_id,
                "name": f"auto_{sensor_id}",
                "type": "generic",
                "status": "active",
                "registered_at": datetime.now().isoformat()
            }

        memory_store['sensor_readings'].append(reading)

        # Why: Capping the in-memory list prevents unbounded memory growth when
        # sensors publish at high frequency. 1000 readings is enough for a demo
        # while keeping the server from OOM-killing itself.
        if len(memory_store['sensor_readings']) > 1000:
            memory_store['sensor_readings'].pop(0)

    return jsonify({"status": "ok", "reading_id": reading['id']}), 201


@app.route('/api/sensors/<sensor_id>/data', methods=['GET'])
def get_sensor_data(sensor_id):
    """Query sensor data"""
    import json

    # Query parameters
    limit = request.args.get('limit', 100, type=int)
    since = request.args.get('since', None)  # ISO timestamp

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM sensor_readings WHERE sensor_id = ?"
        params = [sensor_id]

        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        readings = []
        for row in rows:
            reading_dict = dict_from_row(row)
            reading_dict['data'] = json.loads(reading_dict['data'])
            readings.append(reading_dict)
    else:
        # Filtering
        readings = [r for r in memory_store['sensor_readings']
                   if r['sensor_id'] == sensor_id]

        if since:
            readings = [r for r in readings if r['timestamp'] > since]

        # Sort by latest and limit
        readings = sorted(readings, key=lambda x: x['timestamp'],
                         reverse=True)[:limit]

    return jsonify({
        "sensor_id": sensor_id,
        "readings": readings,
        "count": len(readings)
    })


@app.route('/api/sensors/<sensor_id>/latest', methods=['GET'])
def get_latest_reading(sensor_id):
    """Get latest sensor data"""
    import json

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sensor_readings
            WHERE sensor_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (sensor_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No data found"}), 404

        latest = dict_from_row(row)
        latest['data'] = json.loads(latest['data'])
    else:
        readings = [r for r in memory_store['sensor_readings']
                   if r['sensor_id'] == sensor_id]

        if not readings:
            return jsonify({"error": "No data found"}), 404

        latest = max(readings, key=lambda x: x['timestamp'])

    return jsonify(latest)


@app.route('/api/sensors/<sensor_id>/stats', methods=['GET'])
def get_sensor_stats(sensor_id):
    """Sensor data statistics (numeric fields)"""
    import json

    field = request.args.get('field', 'temperature')

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data FROM sensor_readings WHERE sensor_id = ?
        """, (sensor_id,))
        rows = cursor.fetchall()
        conn.close()

        values = []
        for row in rows:
            data = json.loads(row['data'])
            if field in data:
                try:
                    values.append(float(data[field]))
                except (ValueError, TypeError):
                    pass
    else:
        readings = [r for r in memory_store['sensor_readings']
                   if r['sensor_id'] == sensor_id]

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


# === Device API ===

@app.route('/api/devices', methods=['GET'])
def list_devices():
    """List devices"""
    # Filtering
    device_type = request.args.get('type')
    status = request.args.get('status')

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM devices WHERE 1=1"
        params = []

        if device_type:
            query += " AND type = ?"
            params.append(device_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        cursor.execute(query, params)
        devices = [dict_from_row(row) for row in cursor.fetchall()]
        conn.close()
    else:
        devices = list(memory_store['devices'].values())

        if device_type:
            devices = [d for d in devices if d.get('type') == device_type]
        if status:
            devices = [d for d in devices if d.get('status') == status]

    return jsonify({
        "devices": devices,
        "total": len(devices)
    })


@app.route('/api/devices', methods=['POST'])
def create_device():
    """Register device"""
    data = request.get_json()

    required_fields = ['id', 'name', 'type']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    device_id = data['id']

    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device already exists"}), 409
    else:
        if device_id in memory_store['devices']:
            return jsonify({"error": "Device already exists"}), 409

    device = {
        "id": device_id,
        "name": data['name'],
        "type": data['type'],
        "status": "offline",
        "created_at": datetime.now().isoformat(),
        "last_seen": None
    }

    if USE_SQLITE:
        cursor.execute("""
            INSERT INTO devices (id, name, type, status, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (device['id'], device['name'], device['type'],
              device['status'], device['created_at'], device['last_seen']))
        conn.commit()
        conn.close()
    else:
        memory_store['devices'][device_id] = device

    return jsonify(device), 201


@app.route('/api/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    """Get device information"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Device not found"}), 404

        device = dict_from_row(row)
    else:
        device = memory_store['devices'].get(device_id)
        if not device:
            return jsonify({"error": "Device not found"}), 404

    return jsonify(device)


@app.route('/api/devices/<device_id>', methods=['PUT'])
def update_device(device_id):
    """Full device update"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device not found"}), 404
    else:
        if device_id not in memory_store['devices']:
            return jsonify({"error": "Device not found"}), 404

    data = request.get_json()
    data['id'] = device_id  # Preserve ID
    data['updated_at'] = datetime.now().isoformat()

    if USE_SQLITE:
        cursor.execute("""
            UPDATE devices
            SET name = ?, type = ?, status = ?
            WHERE id = ?
        """, (data.get('name'), data.get('type'),
              data.get('status'), device_id))
        conn.commit()
        conn.close()

        # Get updated device
        return get_device(device_id)
    else:
        memory_store['devices'][device_id] = data
        return jsonify(data)


@app.route('/api/devices/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    """Delete device"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device not found"}), 404

        cursor.execute("DELETE FROM devices WHERE id = ?", (device_id,))
        conn.commit()
        conn.close()
    else:
        if device_id not in memory_store['devices']:
            return jsonify({"error": "Device not found"}), 404

        del memory_store['devices'][device_id]

    return '', 204


@app.route('/api/devices/<device_id>/command', methods=['POST'])
def send_device_command(device_id):
    """Send command to device (simulation)"""
    if USE_SQLITE:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Device not found"}), 404
        conn.close()
    else:
        if device_id not in memory_store['devices']:
            return jsonify({"error": "Device not found"}), 404

    data = request.get_json()

    if 'command' not in data:
        return jsonify({"error": "Command required"}), 400

    # Why: In production this endpoint would publish to MQTT, bridging the
    # HTTP world (user dashboards) and the MQTT world (constrained devices).
    # Keeping device commands behind a REST endpoint gives you access control for free.
    command = {
        "device_id": device_id,
        "command": data['command'],
        "params": data.get('params', {}),
        "sent_at": datetime.now().isoformat()
    }

    # Simulation: print command
    print(f"[Command sent] {device_id}: {command['command']}")

    return jsonify({
        "status": "sent",
        "command": command
    }), 202


# === Main Execution ===

if __name__ == "__main__":
    # Initialize database if SQLite mode
    if USE_SQLITE:
        init_db()
        print(f"Database initialized: {DB_PATH}")
    else:
        print("In-memory storage mode (simulation)")

    print("\n=== IoT REST API Server Starting ===")
    print("Endpoints: http://localhost:5000/")
    print("API docs: http://localhost:5000/")
    print("\nExit: Ctrl+C")

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
