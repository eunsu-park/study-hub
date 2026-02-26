# 12. Cloud IoT Integration

**Previous**: [Image Analysis Project](./11_Image_Analysis_Project.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe AWS IoT Core architecture and its core components
2. Register IoT devices with certificates on AWS IoT Core
3. Publish sensor data to GCP Cloud Pub/Sub from a Raspberry Pi
4. Build an MQTT bridge that forwards local messages to cloud services
5. Implement bidirectional cloud-to-device communication
6. Store and analyze IoT time-series data using InfluxDB and dashboard APIs

---

Edge processing handles real-time decisions, but cloud integration is where IoT data becomes truly valuable at scale. Cloud services provide unlimited storage, powerful analytics, fleet management for thousands of devices, and integration with machine learning pipelines. This final lesson connects your local IoT system to the cloud, completing the full data pipeline from sensor to insight.

---

## 1. AWS IoT Core Overview

### 1.1 AWS IoT Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS IoT Core Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   IoT Devices                     AWS Cloud                  │
│   ┌─────────┐                                                │
│   │Raspberry│     MQTT/TLS      ┌──────────────────────┐    │
│   │   Pi    │───────────────────▶│    AWS IoT Core     │    │
│   └─────────┘                    │                      │    │
│                                  │  • Message Broker    │    │
│   ┌─────────┐                    │  • Rules Engine      │    │
│   │  ESP32  │───────────────────▶│  • Device Shadow     │    │
│   └─────────┘                    │  • Registry          │    │
│                                  └──────────┬───────────┘    │
│                                             │                │
│        ┌────────────────┬──────────────────┼────────────┐   │
│        │                │                  │            │   │
│        ▼                ▼                  ▼            ▼   │
│   ┌─────────┐    ┌──────────┐     ┌──────────┐  ┌────────┐ │
│   │   S3    │    │ DynamoDB │     │  Lambda  │  │Kinesis │ │
│   │(Storage)│    │  (DB)    │     │(Process) │  │(Stream)│ │
│   └─────────┘    └──────────┘     └──────────┘  └────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Description |
|----------|------|
| **Device Gateway** | Manages MQTT/HTTPS/WebSocket connections |
| **Message Broker** | Routes Pub/Sub messages |
| **Rules Engine** | Filters messages and integrates with AWS services |
| **Device Shadow** | Virtual replica of device state |
| **Registry** | Manages device IDs and authentication |

### 1.3 AWS CLI Setup

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# AWS Access Key ID: your-access-key
# AWS Secret Access Key: your-secret-key
# Default region name: ap-northeast-2
# Default output format: json

# Check IoT endpoint
aws iot describe-endpoint --endpoint-type iot:Data-ATS
```

---

## 2. AWS IoT Device Registration

### 2.1 Certificate Generation

```bash
# Create device directory
mkdir -p ~/iot-certs && cd ~/iot-certs

# Generate certificate and keys
aws iot create-keys-and-certificate \
    --set-as-active \
    --certificate-pem-outfile device.cert.pem \
    --public-key-outfile device.public.key \
    --private-key-outfile device.private.key

# Download root CA
wget https://www.amazontrust.com/repository/AmazonRootCA1.pem -O root-CA.crt
```

### 2.2 Create and Attach Policy

```bash
# Create IoT policy
cat > iot-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iot:Connect",
                "iot:Publish",
                "iot:Subscribe",
                "iot:Receive"
            ],
            "Resource": "*"
        }
    ]
}
EOF

aws iot create-policy \
    --policy-name IoTDevicePolicy \
    --policy-document file://iot-policy.json

# Attach policy to certificate
aws iot attach-policy \
    --policy-name IoTDevicePolicy \
    --target <certificate-ARN>

# Create Thing
aws iot create-thing --thing-name RaspberryPi-001

# Attach certificate to Thing
aws iot attach-thing-principal \
    --thing-name RaspberryPi-001 \
    --principal <certificate-ARN>
```

### 2.3 Python Connection Client

```python
#!/usr/bin/env python3
"""Connect to AWS IoT Core"""

from awscrt import mqtt
from awsiot import mqtt_connection_builder
import json
import time

class AWSIoTClient:
    """AWS IoT Core client"""

    def __init__(self, endpoint: str, cert_path: str, key_path: str,
                 ca_path: str, client_id: str):
        self.endpoint = endpoint
        self.client_id = client_id

        # Create MQTT connection
        self.connection = mqtt_connection_builder.mtls_from_path(
            endpoint=endpoint,
            port=8883,
            cert_filepath=cert_path,
            pri_key_filepath=key_path,
            ca_filepath=ca_path,
            client_id=client_id,
            clean_session=False,
            keep_alive_secs=30
        )

        self.connected = False

    def connect(self):
        """Connect"""
        print(f"Connecting to AWS IoT: {self.endpoint}")

        connect_future = self.connection.connect()
        connect_future.result()

        self.connected = True
        print("Connected successfully!")

    def disconnect(self):
        """Disconnect"""
        if self.connected:
            disconnect_future = self.connection.disconnect()
            disconnect_future.result()
            print("Disconnected")

    def publish(self, topic: str, payload: dict, qos: mqtt.QoS = mqtt.QoS.AT_LEAST_ONCE):
        """Publish message"""
        message = json.dumps(payload)

        publish_future, _ = self.connection.publish(
            topic=topic,
            payload=message,
            qos=qos
        )
        publish_future.result()

        print(f"Published: {topic} = {message}")

    def subscribe(self, topic: str, callback):
        """Subscribe to topic"""
        def on_message(topic, payload, **kwargs):
            message = json.loads(payload)
            callback(topic, message)

        subscribe_future, _ = self.connection.subscribe(
            topic=topic,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=on_message
        )
        subscribe_future.result()

        print(f"Subscribed: {topic}")

# Usage example
if __name__ == "__main__":
    # Certificate paths
    ENDPOINT = "your-endpoint.iot.ap-northeast-2.amazonaws.com"
    CERT_PATH = "~/iot-certs/device.cert.pem"
    KEY_PATH = "~/iot-certs/device.private.key"
    CA_PATH = "~/iot-certs/root-CA.crt"

    client = AWSIoTClient(
        endpoint=ENDPOINT,
        cert_path=CERT_PATH,
        key_path=KEY_PATH,
        ca_path=CA_PATH,
        client_id="RaspberryPi-001"
    )

    try:
        client.connect()

        # Publish sensor data
        for i in range(10):
            data = {
                "device_id": "RaspberryPi-001",
                "temperature": 25.5 + i * 0.1,
                "humidity": 60 + i,
                "timestamp": int(time.time())
            }
            client.publish("sensor/data", data)
            time.sleep(5)

    finally:
        client.disconnect()
```

---

## 3. GCP IoT (Pub/Sub)

> **Important Notice**: GCP IoT Core service was retired on August 16, 2023.
>
> Google recommends using **Cloud Pub/Sub + Cloud Functions** as a replacement.
> Existing IoT Core users should consider these alternatives:
> - **Cloud Pub/Sub**: Message broker (examples below)
> - **Clearblade IoT Core**: Google partner's IoT Core compatible service
> - **AWS IoT Core**: AWS alternative
> - **Azure IoT Hub**: Microsoft alternative
>
> This lesson covers **using Cloud Pub/Sub directly**.

### 3.1 GCP Pub/Sub Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GCP Pub/Sub Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Publisher                                 Subscriber       │
│   (IoT Devices)                            (Processing)      │
│   ┌─────────┐        ┌──────────────┐      ┌─────────┐      │
│   │Raspberry│──────▶│    Topic     │─────▶│ Cloud   │      │
│   │   Pi    │        │  (sensor)    │      │ Function│      │
│   └─────────┘        └──────────────┘      └─────────┘      │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │ Subscription │                       │
│                      │    (pull)    │                       │
│                      └──────────────┘                       │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │   BigQuery   │                       │
│                      │  (Analysis)  │                       │
│                      └──────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 GCP Setup

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable Pub/Sub API
gcloud services enable pubsub.googleapis.com

# Create topic
gcloud pubsub topics create iot-sensor-data

# Create subscription
gcloud pubsub subscriptions create iot-sensor-sub \
    --topic=iot-sensor-data

# Create service account
gcloud iam service-accounts create iot-device \
    --display-name="IoT Device Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:iot-device@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.publisher"

# Generate key
gcloud iam service-accounts keys create ~/gcp-credentials.json \
    --iam-account=iot-device@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3.3 Python Pub/Sub Client

```python
#!/usr/bin/env python3
"""GCP Pub/Sub client"""

from google.cloud import pubsub_v1
import json
import time
import os

# Set authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-credentials.json'

class GCPPubSubClient:
    """GCP Pub/Sub client"""

    def __init__(self, project_id: str, topic_id: str):
        self.project_id = project_id
        self.topic_id = topic_id
        self.topic_path = f"projects/{project_id}/topics/{topic_id}"

        self.publisher = pubsub_v1.PublisherClient()

    def publish(self, data: dict, **attributes):
        """Publish message"""
        message = json.dumps(data).encode('utf-8')

        future = self.publisher.publish(
            self.topic_path,
            message,
            **attributes
        )

        message_id = future.result()
        print(f"Published: {message_id}")
        return message_id

    def publish_batch(self, messages: list):
        """Batch publish"""
        futures = []

        for data in messages:
            message = json.dumps(data).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message)
            futures.append(future)

        # Wait for all publishes to complete
        results = [f.result() for f in futures]
        print(f"Batch publish complete: {len(results)} messages")
        return results

class GCPPubSubSubscriber:
    """GCP Pub/Sub subscriber"""

    def __init__(self, project_id: str, subscription_id: str):
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.subscription_path = f"projects/{project_id}/subscriptions/{subscription_id}"

        self.subscriber = pubsub_v1.SubscriberClient()

    def pull(self, max_messages: int = 10) -> list:
        """Pull messages"""
        response = self.subscriber.pull(
            subscription=self.subscription_path,
            max_messages=max_messages
        )

        messages = []
        ack_ids = []

        for msg in response.received_messages:
            data = json.loads(msg.message.data.decode('utf-8'))
            messages.append({
                'data': data,
                'attributes': dict(msg.message.attributes),
                'message_id': msg.message.message_id
            })
            ack_ids.append(msg.ack_id)

        # Send ACK
        if ack_ids:
            self.subscriber.acknowledge(
                subscription=self.subscription_path,
                ack_ids=ack_ids
            )

        return messages

    def subscribe(self, callback):
        """Streaming subscription"""
        def on_message(message):
            data = json.loads(message.data.decode('utf-8'))
            callback(data, message.attributes)
            message.ack()

        streaming_pull = self.subscriber.subscribe(
            self.subscription_path,
            callback=on_message
        )

        print(f"Subscription started: {self.subscription_path}")
        return streaming_pull

# Usage example
if __name__ == "__main__":
    PROJECT_ID = "your-project-id"

    # Publisher
    publisher = GCPPubSubClient(PROJECT_ID, "iot-sensor-data")

    for i in range(5):
        data = {
            "device_id": "pi-001",
            "temperature": 25.5 + i * 0.5,
            "timestamp": int(time.time())
        }
        publisher.publish(data, device_type="raspberry_pi")
        time.sleep(1)
```

---

## 4. MQTT Bridge

### 4.1 Local-to-Cloud Bridge

```python
#!/usr/bin/env python3
"""MQTT Bridge: Local -> Cloud"""

import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import json
import threading
import time
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-credentials.json'

class MQTTBridge:
    """Bridge local MQTT to cloud"""

    def __init__(self, local_broker: str, gcp_project: str, gcp_topic: str):
        # Local MQTT client
        self.local_client = mqtt.Client(client_id="mqtt_bridge")
        self.local_client.on_connect = self._on_local_connect
        self.local_client.on_message = self._on_local_message

        self.local_broker = local_broker

        # GCP Pub/Sub
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = f"projects/{gcp_project}/topics/{gcp_topic}"

        # Bridge configuration
        self.topic_mapping = {
            "sensor/#": True,
            "device/+/status": True
        }

        self.stats = {
            "messages_received": 0,
            "messages_forwarded": 0,
            "errors": 0
        }

    def _on_local_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Local broker connected")
            # Subscribe to topics
            for topic in self.topic_mapping:
                client.subscribe(topic)
                print(f"Subscribed: {topic}")

    def _on_local_message(self, client, userdata, msg):
        """Receive local message -> Forward to cloud"""
        self.stats["messages_received"] += 1

        try:
            # Parse message
            payload = msg.payload.decode('utf-8')

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"raw": payload}

            # Add metadata
            cloud_message = {
                "source_topic": msg.topic,
                "data": data,
                "timestamp": time.time()
            }

            # Publish to cloud
            self._forward_to_cloud(cloud_message, msg.topic)
            self.stats["messages_forwarded"] += 1

        except Exception as e:
            print(f"Bridge error: {e}")
            self.stats["errors"] += 1

    def _forward_to_cloud(self, message: dict, source_topic: str):
        """Forward message to cloud"""
        data = json.dumps(message).encode('utf-8')

        future = self.publisher.publish(
            self.topic_path,
            data,
            source_topic=source_topic
        )
        future.result()  # Wait for publish to complete

    def start(self):
        """Start bridge"""
        print(f"MQTT bridge starting: {self.local_broker} -> GCP")

        self.local_client.connect(self.local_broker, 1883)
        self.local_client.loop_start()

    def stop(self):
        """Stop bridge"""
        self.local_client.loop_stop()
        self.local_client.disconnect()

        print(f"\nBridge statistics:")
        print(f"  Received: {self.stats['messages_received']}")
        print(f"  Forwarded: {self.stats['messages_forwarded']}")
        print(f"  Errors: {self.stats['errors']}")

    def run(self):
        """Main loop"""
        self.start()

        try:
            while True:
                time.sleep(10)
                print(f"Bridge status: received={self.stats['messages_received']}, "
                      f"forwarded={self.stats['messages_forwarded']}")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

if __name__ == "__main__":
    bridge = MQTTBridge(
        local_broker="localhost",
        gcp_project="your-project-id",
        gcp_topic="iot-sensor-data"
    )
    bridge.run()
```

### 4.2 Bidirectional Bridge

```python
#!/usr/bin/env python3
"""Bidirectional MQTT-Cloud bridge"""

import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import json
import threading
import time

class BidirectionalBridge:
    """Bidirectional bridge"""

    def __init__(self, local_broker: str, gcp_project: str,
                 up_topic: str, down_topic: str, down_sub: str):
        # Local MQTT
        self.local_client = mqtt.Client()
        self.local_broker = local_broker

        # GCP Pub/Sub
        self.gcp_project = gcp_project
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        self.up_topic_path = f"projects/{gcp_project}/topics/{up_topic}"
        self.down_sub_path = f"projects/{gcp_project}/subscriptions/{down_sub}"

    def _on_local_message(self, client, userdata, msg):
        """Local -> Cloud"""
        try:
            payload = json.loads(msg.payload.decode())
            data = json.dumps({
                "topic": msg.topic,
                "payload": payload
            }).encode()

            self.publisher.publish(self.up_topic_path, data)
            print(f"[UP] {msg.topic}")

        except Exception as e:
            print(f"Upstream error: {e}")

    def _on_cloud_message(self, message):
        """Cloud -> Local"""
        try:
            data = json.loads(message.data.decode())
            topic = data.get("topic", "command")
            payload = data.get("payload", {})

            self.local_client.publish(topic, json.dumps(payload))
            print(f"[DOWN] {topic}")

            message.ack()

        except Exception as e:
            print(f"Downstream error: {e}")
            message.nack()

    def start(self):
        """Start bridge"""
        # Start local MQTT
        self.local_client.on_message = self._on_local_message
        self.local_client.connect(self.local_broker, 1883)
        self.local_client.subscribe("sensor/#")
        self.local_client.loop_start()

        # Start cloud subscription
        self.streaming_pull = self.subscriber.subscribe(
            self.down_sub_path,
            callback=self._on_cloud_message
        )

        print("Bidirectional bridge started")

    def stop(self):
        """Stop bridge"""
        self.streaming_pull.cancel()
        self.local_client.loop_stop()
        self.local_client.disconnect()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
```

---

## 5. Data Collection and Analysis

### 5.1 Time Series Data Storage

```python
#!/usr/bin/env python3
"""IoT data storage and analysis"""

from datetime import datetime, timedelta
import json
from typing import List, Dict
from dataclasses import dataclass

# Using InfluxDB (time series DB)
# pip install influxdb-client

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

@dataclass
class SensorData:
    device_id: str
    temperature: float
    humidity: float
    timestamp: datetime

class TimeSeriesDB:
    """Time series database client"""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org

    def write_sensor_data(self, data: SensorData):
        """Write sensor data"""
        point = Point("sensor_reading") \
            .tag("device_id", data.device_id) \
            .field("temperature", data.temperature) \
            .field("humidity", data.humidity) \
            .time(data.timestamp)

        self.write_api.write(bucket=self.bucket, record=point)

    def write_batch(self, data_list: List[SensorData]):
        """Batch write"""
        points = []
        for data in data_list:
            point = Point("sensor_reading") \
                .tag("device_id", data.device_id) \
                .field("temperature", data.temperature) \
                .field("humidity", data.humidity) \
                .time(data.timestamp)
            points.append(point)

        self.write_api.write(bucket=self.bucket, record=points)

    def query_recent(self, device_id: str, hours: int = 24) -> List[Dict]:
        """Query recent data"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
            |> filter(fn: (r) => r["_measurement"] == "sensor_reading")
        '''

        tables = self.query_api.query(query, org=self.org)

        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    "time": record.get_time(),
                    "field": record.get_field(),
                    "value": record.get_value()
                })

        return results

    def get_statistics(self, device_id: str, hours: int = 24) -> Dict:
        """Query statistics"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
            |> filter(fn: (r) => r["_field"] == "temperature")
            |> mean()
        '''

        tables = self.query_api.query(query, org=self.org)

        stats = {}
        for table in tables:
            for record in table.records:
                stats["mean_temperature"] = record.get_value()

        return stats

    def close(self):
        self.client.close()
```

### 5.2 Dashboard Data API

```python
#!/usr/bin/env python3
"""IoT dashboard API"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# Data store (in practice, connect to DB)
data_store = {
    "devices": {},
    "readings": []
}

@app.route('/api/devices')
def list_devices():
    """Device list"""
    return jsonify({
        "devices": list(data_store["devices"].values()),
        "count": len(data_store["devices"])
    })

@app.route('/api/devices/<device_id>/readings')
def get_readings(device_id):
    """Query device data"""
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 100, type=int)

    cutoff = datetime.now() - timedelta(hours=hours)

    readings = [
        r for r in data_store["readings"]
        if r["device_id"] == device_id and
           datetime.fromisoformat(r["timestamp"]) > cutoff
    ]

    readings = sorted(readings, key=lambda x: x["timestamp"], reverse=True)[:limit]

    return jsonify({
        "device_id": device_id,
        "readings": readings,
        "count": len(readings)
    })

@app.route('/api/devices/<device_id>/stats')
def get_stats(device_id):
    """Device statistics"""
    hours = request.args.get('hours', 24, type=int)
    cutoff = datetime.now() - timedelta(hours=hours)

    readings = [
        r for r in data_store["readings"]
        if r["device_id"] == device_id and
           datetime.fromisoformat(r["timestamp"]) > cutoff
    ]

    if not readings:
        return jsonify({"error": "No data"}), 404

    temps = [r["temperature"] for r in readings if "temperature" in r]
    humids = [r["humidity"] for r in readings if "humidity" in r]

    stats = {
        "device_id": device_id,
        "period_hours": hours,
        "reading_count": len(readings)
    }

    if temps:
        stats["temperature"] = {
            "min": min(temps),
            "max": max(temps),
            "avg": sum(temps) / len(temps)
        }

    if humids:
        stats["humidity"] = {
            "min": min(humids),
            "max": max(humids),
            "avg": sum(humids) / len(humids)
        }

    return jsonify(stats)

@app.route('/api/alerts')
def get_alerts():
    """Query alerts"""
    alerts = []

    # Check thresholds
    for device_id, device in data_store["devices"].items():
        last_reading = device.get("last_reading")
        if last_reading:
            if last_reading.get("temperature", 0) > 35:
                alerts.append({
                    "device_id": device_id,
                    "type": "high_temperature",
                    "value": last_reading["temperature"],
                    "threshold": 35
                })

    return jsonify({"alerts": alerts, "count": len(alerts)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

---

## 6. Comprehensive IoT Pipeline

```python
#!/usr/bin/env python3
"""Comprehensive IoT data pipeline"""

import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import json
import time
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class IoTMessage:
    device_id: str
    data: dict
    timestamp: datetime
    source: str

class IoTPipeline:
    """IoT data pipeline"""

    def __init__(self, config: dict):
        self.config = config
        self.handlers: List[Callable] = []

        # Local MQTT
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self._on_mqtt_message

        # Cloud (optional)
        if config.get("gcp_enabled"):
            self.publisher = pubsub_v1.PublisherClient()

    def add_handler(self, handler: Callable):
        """Add message handler"""
        self.handlers.append(handler)

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message processing"""
        try:
            payload = json.loads(msg.payload.decode())

            message = IoTMessage(
                device_id=payload.get("device_id", "unknown"),
                data=payload,
                timestamp=datetime.now(),
                source=msg.topic
            )

            # Execute handlers
            for handler in self.handlers:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Handler error: {e}")

        except Exception as e:
            print(f"Message processing error: {e}")

    def start(self):
        """Start pipeline"""
        broker = self.config.get("mqtt_broker", "localhost")
        topics = self.config.get("mqtt_topics", ["sensor/#"])

        self.mqtt_client.connect(broker, 1883)

        for topic in topics:
            self.mqtt_client.subscribe(topic)
            print(f"Subscribed: {topic}")

        self.mqtt_client.loop_start()
        print("IoT pipeline started")

    def stop(self):
        """Stop pipeline"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# Handler examples
def log_handler(message: IoTMessage):
    """Logging handler"""
    print(f"[{message.timestamp}] {message.device_id}: {message.data}")

def alert_handler(message: IoTMessage):
    """Alert handler"""
    temp = message.data.get("temperature")
    if temp and temp > 35:
        print(f"[ALERT] High temperature detected: {message.device_id} = {temp}°C")

def cloud_handler(message: IoTMessage, publisher, topic_path):
    """Cloud forwarding handler"""
    data = json.dumps({
        "device_id": message.device_id,
        "data": message.data,
        "timestamp": message.timestamp.isoformat()
    }).encode()

    publisher.publish(topic_path, data)

# Usage example
if __name__ == "__main__":
    config = {
        "mqtt_broker": "localhost",
        "mqtt_topics": ["sensor/#", "device/+/status"],
        "gcp_enabled": False
    }

    pipeline = IoTPipeline(config)
    pipeline.add_handler(log_handler)
    pipeline.add_handler(alert_handler)

    pipeline.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
```

---

## Exercises

### Exercise 1: AWS IoT Integration
1. Register a device on AWS IoT Core.
2. Publish sensor data to AWS IoT.

### Exercise 2: Data Analysis
1. Calculate daily statistics for collected sensor data.
2. Implement anomaly detection alerts.

### Exercise 3: Dashboard
1. Create a dashboard to visualize cloud data.
2. Implement real-time data updates.

---

**Previous**: [Image Analysis Project](./11_Image_Analysis_Project.md)
