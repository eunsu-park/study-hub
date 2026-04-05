"""
Exercises for Lesson 07: HTTP/REST for IoT
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates an IoT REST API server using Python's built-in http.server
and demonstrates CRUD operations, pagination, and MQTT integration.

On a real deployment, use Flask or FastAPI for the REST server and
paho-mqtt for MQTT integration.
"""

import json
import time
import random
from datetime import datetime, timedelta
from collections import defaultdict


# ---------------------------------------------------------------------------
# Simulated data store (replaces a database in production)
# ---------------------------------------------------------------------------

class InMemorySensorDB:
    """In-memory sensor data store simulating a REST API backend.

    In production, this would be backed by:
    - SQLite for edge devices with limited resources
    - PostgreSQL/TimescaleDB for cloud-based time-series data
    - InfluxDB for high-throughput sensor ingestion
    """

    def __init__(self):
        self._sensors = {}      # sensor_id -> metadata dict
        self._readings = []     # list of reading dicts
        self._next_id = 1

    def create_sensor(self, name, sensor_type, location):
        """POST /sensors -- Register a new sensor."""
        sensor_id = f"sensor_{self._next_id:03d}"
        self._next_id += 1

        sensor = {
            "id": sensor_id,
            "name": name,
            "type": sensor_type,
            "location": location,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }
        self._sensors[sensor_id] = sensor
        return sensor, 201

    def get_sensor(self, sensor_id):
        """GET /sensors/{id} -- Retrieve sensor metadata."""
        sensor = self._sensors.get(sensor_id)
        if sensor:
            return sensor, 200
        return {"error": f"Sensor {sensor_id} not found"}, 404

    def update_sensor(self, sensor_id, updates):
        """PUT /sensors/{id} -- Update sensor metadata."""
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            return {"error": f"Sensor {sensor_id} not found"}, 404

        for key, value in updates.items():
            if key in ("name", "type", "location", "status"):
                sensor[key] = value
        sensor["updated_at"] = datetime.now().isoformat()
        return sensor, 200

    def delete_sensor(self, sensor_id):
        """DELETE /sensors/{id} -- Deactivate a sensor."""
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            return {"error": f"Sensor {sensor_id} not found"}, 404

        sensor["status"] = "inactive"
        return {"message": f"Sensor {sensor_id} deactivated"}, 200

    def list_sensors(self):
        """GET /sensors -- List all sensors."""
        return list(self._sensors.values()), 200

    def add_reading(self, sensor_id, value, unit):
        """POST /sensors/{id}/data -- Add a sensor reading."""
        if sensor_id not in self._sensors:
            return {"error": f"Sensor {sensor_id} not found"}, 404

        # Validate value is numeric
        if not isinstance(value, (int, float)):
            return {"error": "value must be numeric"}, 400

        reading = {
            "sensor_id": sensor_id,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
        }
        self._readings.append(reading)
        return reading, 201

    def get_readings(self, sensor_id=None, page=1, per_page=10,
                     start_date=None, end_date=None):
        """GET /sensors/{id}/data -- Get readings with pagination and date filter.

        Pagination follows REST conventions:
        - page: 1-indexed page number
        - per_page: items per page (default 10, max 100)
        - Returns: items, total count, page info
        """
        # Filter by sensor_id
        filtered = self._readings
        if sensor_id:
            filtered = [r for r in filtered if r["sensor_id"] == sensor_id]

        # Filter by date range
        if start_date:
            filtered = [r for r in filtered
                       if r["timestamp"] >= start_date]
        if end_date:
            filtered = [r for r in filtered
                       if r["timestamp"] <= end_date]

        # Paginate
        per_page = min(per_page, 100)  # Cap at 100
        total = len(filtered)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        end = start + per_page
        items = filtered[start:end]

        return {
            "items": items,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }, 200


# ---------------------------------------------------------------------------
# Simulated MQTT bridge (for Exercise 3)
# ---------------------------------------------------------------------------

class MQTTHTTPBridge:
    """Bridge between HTTP REST API and MQTT message bus.

    Architecture:
    - HTTP POST /sensors/{id}/data -> publishes to MQTT topic sensors/{id}/{metric}
    - MQTT subscriber stores messages -> available via HTTP GET /sensors/{id}/data

    This pattern is common in IoT gateways where edge devices speak MQTT
    but cloud applications consume data via REST APIs.
    """

    def __init__(self, db):
        self.db = db
        self.mqtt_messages = []  # Simulated MQTT message store

    def http_to_mqtt(self, sensor_id, value, unit):
        """Convert HTTP POST to MQTT publish.

        In production:
            mqtt_client.publish(f"sensors/{sensor_id}/{unit}", str(value), qos=1)
        """
        topic = f"sensors/{sensor_id}/{unit}"
        payload = json.dumps({"value": value, "unit": unit,
                             "timestamp": datetime.now().isoformat()})
        self.mqtt_messages.append({"topic": topic, "payload": payload})
        return topic

    def mqtt_to_http(self, topic, payload):
        """Store MQTT message so it can be queried via HTTP GET.

        In production, the MQTT subscriber callback writes to the database,
        and the HTTP GET endpoint reads from that same database.
        """
        data = json.loads(payload)
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "sensors":
            sensor_id = parts[1]
            self.db.add_reading(sensor_id, data["value"], data["unit"])


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: Sensor CRUD API ===
# Problem: Implement sensor CRUD API with data validation.

def exercise_1():
    """Solution: Sensor CRUD REST API with validation."""

    db = InMemorySensorDB()

    print("  Sensor CRUD API\n")

    # CREATE
    print("    --- POST /sensors (Create) ---")
    sensor1, status = db.create_sensor("Living Room Temp", "DHT11", "living_room")
    print(f"    Status: {status}")
    print(f"    Response: {json.dumps(sensor1, indent=6)}\n")

    sensor2, status = db.create_sensor("Outdoor Humidity", "DHT22", "garden")
    print(f"    Created: {sensor2['id']} ({sensor2['name']})")

    sensor3, status = db.create_sensor("Garage Motion", "PIR", "garage")
    print(f"    Created: {sensor3['id']} ({sensor3['name']})")

    # READ
    print("\n    --- GET /sensors/{id} (Read) ---")
    result, status = db.get_sensor("sensor_001")
    print(f"    Status: {status}, Name: {result['name']}")

    result, status = db.get_sensor("nonexistent")
    print(f"    Status: {status}, Error: {result['error']}")

    # UPDATE
    print("\n    --- PUT /sensors/{id} (Update) ---")
    result, status = db.update_sensor("sensor_001", {"location": "bedroom"})
    print(f"    Status: {status}, New location: {result['location']}")

    # Validation: reject unknown sensor
    result, status = db.update_sensor("bad_id", {"name": "x"})
    print(f"    Status: {status}, Error: {result['error']}")

    # DELETE (soft delete -- deactivate)
    print("\n    --- DELETE /sensors/{id} (Deactivate) ---")
    result, status = db.delete_sensor("sensor_003")
    print(f"    Status: {status}, Message: {result['message']}")

    # Verify deactivation
    sensor, _ = db.get_sensor("sensor_003")
    print(f"    sensor_003 status: {sensor['status']}")

    # LIST
    print("\n    --- GET /sensors (List All) ---")
    sensors, status = db.list_sensors()
    for s in sensors:
        print(f"    {s['id']}: {s['name']} ({s['status']})")

    # DATA VALIDATION
    print("\n    --- POST /sensors/{id}/data (with validation) ---")
    result, status = db.add_reading("sensor_001", 22.5, "celsius")
    print(f"    Valid reading: status={status}")

    result, status = db.add_reading("sensor_001", "not_a_number", "celsius")
    print(f"    Invalid reading: status={status}, error={result['error']}")

    result, status = db.add_reading("nonexistent", 22.5, "celsius")
    print(f"    Unknown sensor: status={status}, error={result['error']}")


# === Exercise 2: Pagination ===
# Problem: Implement pagination for sensor data list with date range filtering.

def exercise_2():
    """Solution: Paginated sensor data API with date range filtering."""

    db = InMemorySensorDB()

    # Create a sensor and populate with readings
    db.create_sensor("Test Sensor", "DHT11", "lab")

    print("  Paginated Sensor Data API\n")

    # Add 35 readings with timestamps spread over the past week
    base_time = datetime.now() - timedelta(days=7)
    for i in range(35):
        reading_time = base_time + timedelta(hours=i * 5)
        reading = {
            "sensor_id": "sensor_001",
            "value": round(random.uniform(20, 30), 1),
            "unit": "celsius",
            "timestamp": reading_time.isoformat(),
        }
        db._readings.append(reading)

    print(f"    Total readings: {len(db._readings)}\n")

    # Page 1 (default: 10 per page)
    result, status = db.get_readings("sensor_001", page=1, per_page=10)
    pag = result["pagination"]
    print(f"    --- Page 1 ---")
    print(f"    Items: {len(result['items'])}, Total: {pag['total_items']}, "
          f"Pages: {pag['total_pages']}, Has next: {pag['has_next']}")
    for item in result["items"][:3]:
        print(f"      {item['timestamp']}: {item['value']} {item['unit']}")
    print(f"      ... ({len(result['items'])} items shown)")

    # Page 2
    result, status = db.get_readings("sensor_001", page=2, per_page=10)
    pag = result["pagination"]
    print(f"\n    --- Page 2 ---")
    print(f"    Items: {len(result['items'])}, Has prev: {pag['has_prev']}, "
          f"Has next: {pag['has_next']}")

    # Last page
    result, status = db.get_readings("sensor_001", page=4, per_page=10)
    pag = result["pagination"]
    print(f"\n    --- Page 4 (last) ---")
    print(f"    Items: {len(result['items'])}, Has next: {pag['has_next']}")

    # Date range filter: last 3 days
    three_days_ago = (datetime.now() - timedelta(days=3)).isoformat()
    result, status = db.get_readings(
        "sensor_001", page=1, per_page=100,
        start_date=three_days_ago,
    )
    print(f"\n    --- Date filter: last 3 days ---")
    print(f"    Readings in range: {result['pagination']['total_items']}")


# === Exercise 3: MQTT Integration ===
# Problem: Bridge HTTP POST to MQTT publish, and MQTT receive to HTTP GET.

def exercise_3():
    """Solution: HTTP-MQTT bidirectional bridge."""

    db = InMemorySensorDB()
    db.create_sensor("Bridge Sensor", "DHT22", "server_room")

    bridge = MQTTHTTPBridge(db)

    print("  HTTP-MQTT Integration Bridge\n")

    # Direction 1: HTTP POST -> MQTT publish
    print("    --- HTTP POST -> MQTT Publish ---\n")

    test_data = [
        ("sensor_001", 23.5, "celsius"),
        ("sensor_001", 55.2, "percent"),
        ("sensor_001", 24.1, "celsius"),
    ]

    for sensor_id, value, unit in test_data:
        # Simulate HTTP POST (adds to DB)
        db.add_reading(sensor_id, value, unit)
        # Bridge also publishes to MQTT
        topic = bridge.http_to_mqtt(sensor_id, value, unit)
        print(f"    HTTP POST /sensors/{sensor_id}/data "
              f"(value={value}) -> MQTT publish to '{topic}'")

    print(f"\n    MQTT messages published: {len(bridge.mqtt_messages)}")

    # Direction 2: MQTT receive -> HTTP GET
    print("\n    --- MQTT Receive -> HTTP GET ---\n")

    # Simulate MQTT messages arriving (from other IoT devices)
    mqtt_incoming = [
        ("sensors/sensor_001/celsius", json.dumps({"value": 25.0, "unit": "celsius",
         "timestamp": datetime.now().isoformat()})),
        ("sensors/sensor_001/percent", json.dumps({"value": 60.5, "unit": "percent",
         "timestamp": datetime.now().isoformat()})),
    ]

    for topic, payload in mqtt_incoming:
        bridge.mqtt_to_http(topic, payload)
        print(f"    MQTT received: {topic} -> stored in DB")

    # Query via HTTP GET
    result, status = db.get_readings("sensor_001")
    print(f"\n    HTTP GET /sensors/sensor_001/data:")
    print(f"    Total readings available: {result['pagination']['total_items']}")
    for item in result["items"]:
        print(f"      {item['timestamp']}: {item['value']} {item['unit']}")


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 07: HTTP/REST for IoT - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Sensor CRUD API")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Pagination")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: MQTT Integration")
    print("-" * 50)
    exercise_3()
