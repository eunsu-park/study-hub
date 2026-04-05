"""
Exercises for Lesson 12: Cloud IoT Integration
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates cloud IoT platform operations (AWS IoT Core, data analytics,
and dashboards) using Python data structures.

On a real deployment:
    pip install awsiotsdk  # AWS IoT Device SDK
    pip install boto3       # AWS SDK for Python
    pip install flask       # For dashboard web server
"""

import time
import json
import csv
import os
import random
import statistics
from datetime import datetime, timedelta
from collections import defaultdict


# ---------------------------------------------------------------------------
# Simulated AWS IoT Core
# ---------------------------------------------------------------------------

class SimulatedAWSIoTCore:
    """Simulate AWS IoT Core for device registration and messaging.

    AWS IoT Core provides:
    - Device registry (Things) with certificates and policies
    - MQTT message broker with topic-based routing
    - Rules engine for transforming and routing data to other AWS services
    - Device Shadow for persistent device state

    Real setup requires:
    1. Create a Thing in the AWS IoT Console
    2. Download certificates (device cert, private key, root CA)
    3. Attach an IoT policy allowing connect/publish/subscribe
    """

    def __init__(self):
        self._things = {}           # thing_name -> metadata
        self._shadows = {}          # thing_name -> shadow document
        self._messages = []         # all published messages
        self._rules = []            # IoT rules
        self._certificates = {}     # thing_name -> cert info

    def register_thing(self, thing_name, thing_type="sensor", attributes=None):
        """Register a device (Thing) in AWS IoT Core.

        Real AWS CLI:
            aws iot create-thing --thing-name my-sensor-01
            aws iot create-keys-and-certificate --set-as-active
            aws iot attach-thing-principal --thing-name my-sensor-01 \\
                --principal <certificate-arn>
        """
        if thing_name in self._things:
            return {"error": f"Thing '{thing_name}' already exists"}, 409

        # Simulate certificate generation
        cert_id = f"cert-{random.randint(10000, 99999)}"
        self._certificates[thing_name] = {
            "certificate_id": cert_id,
            "certificate_pem": f"-----BEGIN CERTIFICATE-----\n...{cert_id}...\n-----END CERTIFICATE-----",
            "private_key": f"-----BEGIN RSA PRIVATE KEY-----\n...{cert_id}...\n-----END RSA PRIVATE KEY-----",
            "root_ca": "AmazonRootCA1.pem",
        }

        self._things[thing_name] = {
            "thing_name": thing_name,
            "thing_type": thing_type,
            "attributes": attributes or {},
            "certificate_id": cert_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        # Initialize device shadow
        self._shadows[thing_name] = {
            "state": {
                "reported": {},
                "desired": {},
            },
            "metadata": {},
            "version": 1,
        }

        return self._things[thing_name], 201

    def publish(self, topic, payload, qos=1):
        """Publish message to AWS IoT MQTT broker.

        Real code (AWS IoT Device SDK v2):
            from awsiot import mqtt_connection_builder

            connection = mqtt_connection_builder.mtls_from_path(
                endpoint='xxxx.iot.us-east-1.amazonaws.com',
                cert_filepath='device.pem.crt',
                pri_key_filepath='private.pem.key',
                ca_filepath='AmazonRootCA1.pem',
                client_id='my-sensor-01'
            )
            connection.connect().result()
            connection.publish(topic=topic, payload=json.dumps(payload), qos=qos)
        """
        message = {
            "topic": topic,
            "payload": payload if isinstance(payload, dict) else json.loads(payload),
            "qos": qos,
            "timestamp": datetime.now().isoformat(),
        }
        self._messages.append(message)
        self._evaluate_rules(message)
        return message

    def update_shadow(self, thing_name, reported=None, desired=None):
        """Update the Device Shadow.

        Device Shadow maintains a virtual state of the device:
        - reported: current state reported by the device
        - desired: state requested by the cloud application
        - delta: difference between desired and reported (auto-computed)

        The device syncs its state on reconnection by reading the shadow.
        """
        if thing_name not in self._shadows:
            return {"error": f"Thing '{thing_name}' not found"}, 404

        shadow = self._shadows[thing_name]
        if reported:
            shadow["state"]["reported"].update(reported)
        if desired:
            shadow["state"]["desired"].update(desired)
        shadow["version"] += 1
        shadow["metadata"]["last_updated"] = datetime.now().isoformat()

        return shadow, 200

    def get_shadow(self, thing_name):
        """Get the Device Shadow."""
        if thing_name not in self._shadows:
            return {"error": f"Thing '{thing_name}' not found"}, 404
        return self._shadows[thing_name], 200

    def add_rule(self, name, sql, action):
        """Add an IoT Rule for message routing.

        AWS IoT Rules use a SQL-like syntax to filter and transform messages:
            SELECT temperature FROM 'sensors/+/data' WHERE temperature > 30

        Actions route matching messages to:
        - DynamoDB, S3, Kinesis, Lambda, SNS, SQS, etc.
        """
        self._rules.append({
            "name": name,
            "sql": sql,
            "action": action,
        })

    def _evaluate_rules(self, message):
        """Evaluate IoT rules against a message (simplified)."""
        # In real AWS, rules use SQL to filter messages
        for rule in self._rules:
            if rule["action"]:
                rule["action"](message)

    def get_messages(self, topic_filter=None, limit=10):
        """Retrieve stored messages (simulates querying DynamoDB/S3)."""
        msgs = self._messages
        if topic_filter:
            msgs = [m for m in msgs if topic_filter in m["topic"]]
        return msgs[-limit:]


# ---------------------------------------------------------------------------
# Simulated Analytics Engine
# ---------------------------------------------------------------------------

class SensorAnalytics:
    """Analytics engine for IoT sensor data.

    In production, this would use:
    - AWS Timestream or DynamoDB for storage
    - AWS Lambda for processing
    - Amazon QuickSight or Grafana for visualization
    """

    def __init__(self, data):
        self.data = data  # list of dicts with 'timestamp', 'value', 'sensor_id'

    def daily_statistics(self):
        """Calculate daily statistics: min, max, mean, std."""
        by_day = defaultdict(list)
        for reading in self.data:
            # Extract date from ISO timestamp
            date = reading["timestamp"][:10]
            by_day[date].append(reading["value"])

        stats = {}
        for date, values in sorted(by_day.items()):
            stats[date] = {
                "count": len(values),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "mean": round(statistics.mean(values), 2),
                "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
            }
        return stats

    def detect_anomalies(self, threshold_std=2.0):
        """Detect anomalies using z-score method.

        An anomaly is a data point more than threshold_std standard deviations
        from the mean. This is a simple but effective approach for stationary
        sensor data. For non-stationary data (seasonal patterns), use
        moving average or exponential smoothing based detection.
        """
        if len(self.data) < 10:
            return []

        values = [r["value"] for r in self.data]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 1.0

        anomalies = []
        for reading in self.data:
            z_score = abs(reading["value"] - mean) / std if std > 0 else 0
            if z_score > threshold_std:
                anomalies.append({
                    "timestamp": reading["timestamp"],
                    "value": reading["value"],
                    "z_score": round(z_score, 2),
                    "direction": "high" if reading["value"] > mean else "low",
                    "threshold": round(mean + threshold_std * std, 2),
                })

        return anomalies


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: AWS IoT Integration ===
# Problem: Register a device on AWS IoT Core. Publish sensor data.

def exercise_1():
    """Solution: AWS IoT Core device registration and data publishing."""

    print("  AWS IoT Core Integration\n")

    iot = SimulatedAWSIoTCore()

    # Part 1: Register a device
    print("    --- Part 1: Register Device ---\n")

    thing, status = iot.register_thing(
        thing_name="temperature-sensor-01",
        thing_type="DHT22",
        attributes={
            "location": "server_room",
            "firmware": "v1.2.3",
            "protocol": "MQTT",
        },
    )

    print(f"    Registered: {thing['thing_name']} (status: {status})")
    print(f"    Type: {thing['thing_type']}")
    print(f"    Certificate ID: {thing['certificate_id']}")

    # Show certificate files
    cert = iot._certificates["temperature-sensor-01"]
    print(f"\n    Certificate files generated:")
    print(f"      Device cert:  {cert['certificate_pem'][:50]}...")
    print(f"      Private key:  {cert['private_key'][:50]}...")
    print(f"      Root CA:      {cert['root_ca']}")

    # Part 2: Publish sensor data
    print("\n    --- Part 2: Publish Sensor Data ---\n")

    # Add an IoT Rule to route data
    stored_data = []

    def store_to_dynamodb(message):
        """Simulated IoT Rule action: store to DynamoDB."""
        stored_data.append(message)

    iot.add_rule(
        name="StoreSensorData",
        sql="SELECT * FROM 'sensors/+/data'",
        action=store_to_dynamodb,
    )

    # Publish 10 readings
    for i in range(10):
        payload = {
            "device_id": "temperature-sensor-01",
            "temperature": round(random.uniform(20.0, 26.0), 1),
            "humidity": round(random.uniform(40.0, 60.0), 1),
            "timestamp": (datetime.now() + timedelta(minutes=i * 5)).isoformat(),
        }

        msg = iot.publish(
            topic="sensors/temperature-sensor-01/data",
            payload=payload,
            qos=1,
        )

        if i < 3:  # Show first 3
            print(f"    Published: temp={payload['temperature']}C, "
                  f"humidity={payload['humidity']}%")

    print(f"    ... ({len(iot._messages)} total messages published)")
    print(f"    IoT Rule stored: {len(stored_data)} messages to DynamoDB")

    # Update device shadow
    print("\n    --- Device Shadow ---")
    iot.update_shadow(
        "temperature-sensor-01",
        reported={"temperature": 23.5, "humidity": 52.0, "status": "online"},
    )
    shadow, _ = iot.get_shadow("temperature-sensor-01")
    print(f"    Shadow state: {json.dumps(shadow['state']['reported'], indent=6)}")

    # Reference connection code
    print("""
    --- Reference Code (AWS IoT Device SDK v2) ---

    from awsiot import mqtt_connection_builder
    import json

    # Create MQTT connection using mutual TLS
    connection = mqtt_connection_builder.mtls_from_path(
        endpoint='a1b2c3d4e5f6g7.iot.us-east-1.amazonaws.com',
        cert_filepath='temperature-sensor-01.cert.pem',
        pri_key_filepath='temperature-sensor-01.private.key',
        ca_filepath='AmazonRootCA1.pem',
        client_id='temperature-sensor-01',
        clean_session=False,
        keep_alive_secs=30,
    )

    connect_future = connection.connect()
    connect_future.result()  # Wait for connection

    # Publish sensor data
    payload = {"temperature": 23.5, "humidity": 52.0}
    connection.publish(
        topic='sensors/temperature-sensor-01/data',
        payload=json.dumps(payload),
        qos=mqtt.QoS.AT_LEAST_ONCE,
    )
    """)


# === Exercise 2: Data Analysis ===
# Problem: Calculate daily statistics. Implement anomaly detection alerts.

def exercise_2():
    """Solution: Sensor data analytics with daily stats and anomaly detection."""

    print("  Sensor Data Analytics\n")

    # Generate 7 days of simulated sensor data (readings every 30 minutes)
    data = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(7 * 48):  # 48 readings per day (every 30 min)
        ts = base_time + timedelta(minutes=i * 30)

        # Normal temperature: 22-24C with daily variation
        hour = ts.hour
        # Warmer during midday, cooler at night
        daily_variation = 2.0 * (1 - abs(hour - 13) / 12)
        temp = 22.0 + daily_variation + random.gauss(0, 0.5)

        # Inject a few anomalies
        if i in (50, 150, 250):
            temp += random.choice([-8, 8])  # Spike or dip

        data.append({
            "sensor_id": "temp-01",
            "timestamp": ts.isoformat(),
            "value": round(temp, 2),
        })

    analytics = SensorAnalytics(data)

    # Part 1: Daily statistics
    print("    --- Part 1: Daily Statistics ---\n")
    daily_stats = analytics.daily_statistics()

    print(f"    {'Date':<12} {'Count':>6} {'Min':>8} {'Max':>8} "
          f"{'Mean':>8} {'Std':>8}")
    print(f"    {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for date, stats in daily_stats.items():
        print(f"    {date:<12} {stats['count']:>6} {stats['min']:>7.1f}C "
              f"{stats['max']:>7.1f}C {stats['mean']:>7.1f}C {stats['std']:>7.2f}")

    # Part 2: Anomaly detection
    print("\n    --- Part 2: Anomaly Detection ---\n")
    anomalies = analytics.detect_anomalies(threshold_std=2.0)

    if anomalies:
        print(f"    Found {len(anomalies)} anomalies (z-score > 2.0):\n")
        print(f"    {'Timestamp':<22} {'Value':>8} {'Z-Score':>9} "
              f"{'Direction':<6} {'Threshold':>10}")
        print(f"    {'-'*22} {'-'*8} {'-'*9} {'-'*6} {'-'*10}")

        for a in anomalies:
            print(f"    {a['timestamp'][:19]:<22} {a['value']:>7.1f}C "
                  f"{a['z_score']:>8.2f} {a['direction']:<6} {a['threshold']:>9.1f}C")

        # Alert output
        print(f"\n    Alerts generated:")
        for a in anomalies:
            direction_label = "above" if a["direction"] == "high" else "below"
            print(f"      [ALERT] {a['timestamp'][:19]}: Temperature {a['value']}C "
                  f"is {direction_label} normal range (z={a['z_score']})")
    else:
        print("    No anomalies detected.")

    # Analysis summary
    all_values = [d["value"] for d in data]
    mean_temp = statistics.mean(all_values)
    std_temp = statistics.stdev(all_values)
    print(f"\n    Overall: mean={mean_temp:.2f}C, std={std_temp:.2f}C, "
          f"range=[{min(all_values):.1f}, {max(all_values):.1f}]C")


# === Exercise 3: Dashboard ===
# Problem: Create a dashboard to visualize cloud data with real-time updates.

def exercise_3():
    """Solution: Terminal-based IoT dashboard with real-time data visualization.

    In production, use Flask/Dash with Chart.js or Grafana for a web dashboard.
    This exercise demonstrates the data flow and display logic.
    """

    print("  IoT Dashboard\n")

    # Simulate cloud data store (DynamoDB or Timestream)
    cloud_data = defaultdict(list)

    # Generate data for 3 devices
    devices = {
        "sensor-room-A": {"type": "DHT22", "location": "Server Room A"},
        "sensor-room-B": {"type": "DHT22", "location": "Server Room B"},
        "sensor-outdoor": {"type": "DS18B20", "location": "Outdoor"},
    }

    base_time = datetime.now() - timedelta(hours=2)
    for device_id in devices:
        for i in range(24):  # 24 readings (every 5 min for 2 hours)
            ts = base_time + timedelta(minutes=i * 5)
            base_temp = {"sensor-room-A": 22, "sensor-room-B": 23,
                        "sensor-outdoor": 15}[device_id]
            cloud_data[device_id].append({
                "timestamp": ts.isoformat(),
                "temperature": round(base_temp + random.gauss(0, 1), 1),
                "humidity": round(55 + random.gauss(0, 5), 1),
            })

    # Dashboard render function
    def render_dashboard():
        """Render the dashboard view."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"    ╔{'═' * 60}╗")
        print(f"    ║{'IoT Sensor Dashboard':^60}║")
        print(f"    ║{'Updated: ' + now:^60}║")
        print(f"    ╠{'═' * 60}╣")

        for device_id, info in devices.items():
            readings = cloud_data[device_id]
            latest = readings[-1]
            temps = [r["temperature"] for r in readings[-12:]]  # Last hour
            avg_temp = statistics.mean(temps)
            trend = "rising" if temps[-1] > temps[0] else "falling" if temps[-1] < temps[0] else "stable"

            # Status indicator
            if abs(latest["temperature"] - avg_temp) > 3:
                status = "!! ALERT"
            else:
                status = "   OK"

            print(f"    ║  {info['location']:<20} ({device_id})")
            print(f"    ║    Temp: {latest['temperature']:>6.1f}C  "
                  f"Humidity: {latest['humidity']:>5.1f}%  "
                  f"Trend: {trend:<8} {status}")

            # Mini sparkline (text-based chart of last 12 readings)
            min_t = min(temps)
            max_t = max(temps)
            range_t = max_t - min_t or 1
            sparkline = ""
            for t in temps:
                level = int((t - min_t) / range_t * 7)
                sparkline += ["_", ".", "-", "~", "+", "*", "#", "^"][level]
            print(f"    ║    Last hour: [{sparkline}] "
                  f"({min_t:.0f}-{max_t:.0f}C)")
            print(f"    ╟{'─' * 60}╢")

        # Summary stats
        all_temps = []
        for readings in cloud_data.values():
            all_temps.extend([r["temperature"] for r in readings])

        print(f"    ║  Summary: {len(devices)} devices, "
              f"{sum(len(v) for v in cloud_data.values())} total readings")
        print(f"    ║  Global temp range: "
              f"{min(all_temps):.1f}C - {max(all_temps):.1f}C")
        print(f"    ╚{'═' * 60}╝")

    # Render dashboard
    render_dashboard()

    # Simulate real-time update
    print("\n    --- Simulating real-time update ---\n")
    for device_id in devices:
        new_reading = {
            "timestamp": datetime.now().isoformat(),
            "temperature": round(cloud_data[device_id][-1]["temperature"]
                                + random.gauss(0, 0.3), 1),
            "humidity": round(cloud_data[device_id][-1]["humidity"]
                             + random.gauss(0, 1), 1),
        }
        cloud_data[device_id].append(new_reading)
        print(f"    New data: {device_id} -> "
              f"temp={new_reading['temperature']}C, "
              f"humidity={new_reading['humidity']}%")

    print("\n    --- Updated Dashboard ---\n")
    render_dashboard()

    # Flask reference
    print("""
    --- Reference: Flask Dashboard (production) ---

    from flask import Flask, render_template, jsonify
    import boto3

    app = Flask(__name__)
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('SensorData')

    @app.route('/dashboard')
    def dashboard():
        # Query latest readings from DynamoDB
        response = table.query(
            KeyConditionExpression='device_id = :did',
            ExpressionAttributeValues={':did': 'sensor-room-A'},
            ScanIndexForward=False,
            Limit=100,
        )
        return render_template('dashboard.html', data=response['Items'])

    @app.route('/api/latest')
    def api_latest():
        # REST endpoint for real-time JavaScript updates
        # Called by dashboard.html via fetch() every 5 seconds
        ...
        return jsonify(latest_readings)
    """)


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 12: Cloud IoT Integration - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: AWS IoT Integration")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Data Analysis")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: Dashboard")
    print("-" * 50)
    exercise_3()
