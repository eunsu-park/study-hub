# 06. MQTT Protocol

**Previous**: [BLE Connectivity](./05_BLE_Connectivity.md) | **Next**: [HTTP/REST for IoT](./07_HTTP_REST_for_IoT.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the MQTT publish/subscribe architecture and its advantages over HTTP for IoT
2. Install and configure the Mosquitto MQTT broker with authentication
3. Design hierarchical topic structures using single-level and multi-level wildcards
4. Compare QoS levels 0, 1, and 2 and select the appropriate level for a given use case
5. Implement MQTT publishers and subscribers in Python using paho-mqtt
6. Use retained messages and Last Will and Testament (LWT) for robust device status tracking

---

MQTT is the dominant messaging protocol in IoT because it was purpose-built for the constraints that IoT devices face: limited bandwidth, unreliable networks, and the need to push data from thousands of sensors to a central system. While HTTP works well for web browsers requesting pages, MQTT excels when lightweight devices need to stream data continuously or receive commands in real time.

---

This lesson covers MQTT (Message Queuing Telemetry Transport), a lightweight messaging protocol designed for IoT devices. We'll learn MQTT broker setup, publish/subscribe patterns, QoS levels, and Python implementation using paho-mqtt.

---

> **Analogy: Radio Broadcast** -- MQTT works like a radio system. Publishers are radio stations that broadcast on specific frequencies (topics). Subscribers tune their radios to the frequencies they care about. The broker is the radio tower that receives all transmissions and relays them to the right listeners. No station needs to know who is listening, and no listener needs to know who is broadcasting. This decoupling is what makes MQTT so scalable: adding a new sensor or a new dashboard requires zero changes to existing devices.

## 1. MQTT Basics

### 1.1 What is MQTT?

**MQTT (Message Queuing Telemetry Transport)**
- Lightweight publish/subscribe messaging protocol
- Designed for constrained devices and low-bandwidth networks
- Widely used in IoT, home automation, industrial monitoring

**Key Features:**
- Minimal protocol overhead (2-byte fixed header: 1 byte for packet type + flags, 1 byte for remaining length; variable-length encoding allows up to 4 bytes for remaining length in large packets, but the fixed portion is always just 2 bytes -- compare this to HTTP's ~800 bytes of headers)
- Asynchronous bidirectional communication
- Quality of Service (QoS) levels
- Persistent sessions and retained messages
- Last Will and Testament (LWT) for disconnection handling

### 1.2 MQTT Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Publisher  │────────>│  MQTT Broker │<────────│ Subscriber  │
│  (Sensor)   │ Publish │              │Subscribe│  (Client)   │
└─────────────┘         └──────────────┘         └─────────────┘
                             │      │
                    ┌────────┘      └────────┐
                    ▼                        ▼
              ┌─────────────┐          ┌─────────────┐
              │ Subscriber  │          │ Subscriber  │
              │  (Client)   │          │  (Client)   │
              └─────────────┘          └─────────────┘
```

**Components:**
- **Broker**: Central server that receives messages and routes them to subscribers
- **Publisher**: Client that sends messages to topics
- **Subscriber**: Client that receives messages from topics
- **Topic**: Message routing path (e.g., `home/livingroom/temperature`)

### 1.3 MQTT vs HTTP

| Feature | MQTT | HTTP |
|---------|------|------|
| **Pattern** | Pub/Sub (asynchronous) | Request/Response (synchronous) |
| **Connection** | Persistent | Per-request |
| **Overhead** | Low (2-byte fixed header: type+flags + remaining length) | High (HTTP headers ~800 bytes avg) |
| **Bi-directional** | Native | Requires polling/WebSocket |
| **QoS** | 3 levels | None (TCP reliability only) |
| **Best For** | Real-time updates, constrained devices | Web APIs, file transfer |

---

## 2. Mosquitto Broker Setup

### 2.1 Mosquitto Installation

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
```

**Check status:**

```bash
sudo systemctl status mosquitto
```

**Start broker:**

```bash
sudo systemctl start mosquitto
sudo systemctl enable mosquitto  # Auto-start on boot
```

### 2.2 Mosquitto Configuration

**Configuration file:** `/etc/mosquitto/mosquitto.conf`

```conf
# Basic configuration
listener 1883
protocol mqtt

# Allow anonymous connections (disable for production)
allow_anonymous true

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type all

# Persistence
persistence true
persistence_location /var/lib/mosquitto/
```

**Apply configuration:**

```bash
sudo systemctl restart mosquitto
```

### 2.3 Mosquitto with Authentication

**Create password file:**

```bash
# Create user
sudo mosquitto_passwd -c /etc/mosquitto/passwd username

# Add more users (omit -c flag)
sudo mosquitto_passwd /etc/mosquitto/passwd username2
```

**Update configuration:**

```conf
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
```

**Restart broker:**

```bash
sudo systemctl restart mosquitto
```

### 2.4 Command Line Testing

**Terminal 1 - Subscribe:**

```bash
mosquitto_sub -h localhost -t test/topic

# With authentication
mosquitto_sub -h localhost -t test/topic -u username -P password
```

**Terminal 2 - Publish:**

```bash
mosquitto_pub -h localhost -t test/topic -m "Hello MQTT!"

# With authentication
mosquitto_pub -h localhost -t test/topic -m "Hello" -u username -P password
```

---

## 3. MQTT Topics

### 3.1 Topic Structure

Topics use hierarchical structure with `/` separator:

```
home/livingroom/temperature
home/livingroom/humidity
home/bedroom/temperature
home/kitchen/light/status
sensors/outdoor/weather/wind
```

**Best Practices:**
- Use descriptive names
- Start with general → specific
- Use lowercase
- Avoid leading `/`
- Keep depth reasonable (3-5 levels)

### 3.2 Wildcards

**Single-level wildcard (`+`)**: Matches one level

```
home/+/temperature
Matches:
  ✓ home/livingroom/temperature
  ✓ home/bedroom/temperature
  ✗ home/livingroom/sensor/temperature  (too deep)
```

**Multi-level wildcard (`#`)**: Matches all sub-levels

```
home/livingroom/#
Matches:
  ✓ home/livingroom/temperature
  ✓ home/livingroom/humidity
  ✓ home/livingroom/sensor/temp
  ✓ home/livingroom/sensor/data/raw
```

**Combined wildcards:**

```
home/+/sensor/#
Matches:
  ✓ home/livingroom/sensor/temp
  ✓ home/bedroom/sensor/data/raw
```

### 3.3 Reserved Topics

Topics starting with `$` are reserved:

- `$SYS/broker/clients/connected`: Number of connected clients
- `$SYS/broker/messages/received`: Total messages received
- `$SYS/broker/uptime`: Broker uptime

```bash
# Monitor broker statistics
mosquitto_sub -h localhost -t '$SYS/#' -v
```

---

## 4. Quality of Service (QoS)

### 4.1 QoS Levels

| Level | Name | Guarantee | Use Case |
|-------|------|-----------|----------|
| **QoS 0** | At most once | No guarantee (fire and forget) | Non-critical sensor data, high-frequency updates |
| **QoS 1** | At least once | Message delivered, duplicates possible | Most IoT applications, general telemetry |
| **QoS 2** | Exactly once | Message delivered exactly once, no duplicates | Critical commands, billing, safety systems |

### 4.2 QoS Flow Diagrams

**QoS 0:**
```
Publisher ──PUBLISH──> Broker ──PUBLISH──> Subscriber
```

**QoS 1:**
```
Publisher ──PUBLISH──> Broker ──PUBLISH──> Subscriber
         <──PUBACK───         <──PUBACK───
```

**QoS 2:**
```
Publisher ──PUBLISH──> Broker ──PUBLISH──> Subscriber
         <──PUBREC───         <──PUBREC───
          ──PUBREL──>          ──PUBREL──>
         <──PUBCOMP──         <──PUBCOMP──
```

**Why does QoS 2 need a 4-message handshake?** QoS 1 uses a simple PUBLISH + PUBACK pair, but if the PUBACK is lost, the sender retransmits and the receiver gets a duplicate. QoS 2 solves this by splitting the process into two phases: (1) PUBLISH/PUBREC ensures the message is stored by the receiver exactly once -- the receiver records the packet ID and replies with PUBREC; (2) PUBREL/PUBCOMP releases the message for delivery -- the sender says "I know you have it, go ahead and deliver" (PUBREL), and the receiver confirms completion (PUBCOMP) and discards the packet ID. If any message is lost, the retransmission of the current phase message is idempotent because the receiver can recognize it by the same packet ID. This two-phase commit guarantees exactly-once delivery at the cost of 2x the round-trips of QoS 1.

**Performance Comparison:**
- QoS 0: Fastest, lowest bandwidth
- QoS 1: Good balance (recommended)
- QoS 2: Highest overhead, use only when necessary

---

## 5. Python MQTT with paho-mqtt

### 5.1 paho-mqtt Installation

```bash
pip3 install paho-mqtt
```

### 5.2 MQTT Publisher

```python
import paho.mqtt.client as mqtt
import time
import random

# Broker configuration
BROKER = "localhost"
PORT = 1883
TOPIC = "sensors/temperature"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

def on_publish(client, userdata, mid):
    print(f"Message {mid} published")

# Create client
client = mqtt.Client()
client.on_connect = on_connect
client.on_publish = on_publish

# Connect to broker
# Why keepalive=60: The client sends a PINGREQ if no other packets are sent within
# this many seconds, proving to the broker it is still alive. Trade-off:
#   - Too low (e.g., 10s): frequent PINGs waste bandwidth on constrained networks
#   - Too high (e.g., 300s): broker takes up to 1.5x this value to detect a dead
#     client (MQTT spec), so 300s means up to ~450s before LWT fires
# 60s is a common default that balances bandwidth and detection speed.
client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

try:
    while True:
        # Simulate sensor reading
        temperature = round(random.uniform(20.0, 30.0), 2)

        # Publish message
        result = client.publish(TOPIC, str(temperature), qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published: {temperature}°C")
        else:
            print(f"Publish failed: {result.rc}")

        # Why sleep(5): Controls the publish interval. Trade-off:
        #   - Shorter interval (e.g., 0.5s): more granular data but higher broker
        #     load, more bandwidth, faster SD card wear if logging
        #   - Longer interval (e.g., 60s): less load but coarser data, slower
        #     anomaly detection. 5s is reasonable for temperature monitoring.
        time.sleep(5)

except KeyboardInterrupt:
    print("\nStopping publisher...")
finally:
    client.loop_stop()
    client.disconnect()
```

### 5.3 MQTT Subscriber

```python
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "sensors/#"  # Subscribe to all sensor topics

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe on connection
        client.subscribe(TOPIC, qos=1)
        print(f"Subscribed to: {TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")

# Create client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect and start loop
# Why keepalive=60: see publisher example for trade-off discussion
client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
```

### 5.4 MQTT Client with Authentication

```python
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
USERNAME = "iot_user"
PASSWORD = "secure_password"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully")
    elif rc == 5:
        print("Authentication failed")
    else:
        print(f"Connection failed, code {rc}")

client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect

client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
```

### 5.5 Retained Messages

Retained messages are stored by the broker and immediately delivered to new subscribers.

```python
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("localhost", 1883)

# Publish retained message
client.publish(
    "home/livingroom/light/status",
    "ON",
    qos=1,
    retain=True  # Retain message
)

print("Retained message published")
client.disconnect()
```

**When a new subscriber connects, it immediately receives the last retained message.**

### 5.6 Last Will and Testament (LWT)

LWT is a message automatically published by the broker when a client disconnects unexpectedly.

```python
import paho.mqtt.client as mqtt
import time

client = mqtt.Client()

# Set Last Will -- MUST be called BEFORE connect().
# Why ordering matters: will_set() configures the CONNECT packet's Will fields.
# Once connect() is called, the CONNECT packet is already sent to the broker.
# Calling will_set() after connect() has no effect -- the broker never receives
# the Will message, so your "offline" notification will silently never fire.
client.will_set(
    "devices/rpi001/status",
    payload="offline",
    qos=1,
    retain=True
)

client.connect("localhost", 1883)
client.loop_start()

# Publish online status
client.publish("devices/rpi001/status", "online", qos=1, retain=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Graceful shutdown: publish offline status manually
    client.publish("devices/rpi001/status", "offline", qos=1, retain=True)
    client.disconnect()
    client.loop_stop()
```

---

## 6. Practical Project: IoT Sensor System with MQTT

Complete sensor monitoring system using MQTT.

### 6.1 Sensor Publisher (Raspberry Pi)

```python
import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime
import random

class SensorPublisher:
    def __init__(self, broker, port=1883, client_id="sensor_rpi001"):
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.client = mqtt.Client(client_id)

        # Set callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish

        # Set Last Will -- must happen before connect() because will_set()
        # configures the CONNECT packet; calling it after connect() is a no-op.
        self.client.will_set(
            f"devices/{client_id}/status",
            payload="offline",
            qos=1,
            retain=True
        )

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[✓] Connected to broker: {self.broker}")
            # Publish online status
            self.client.publish(
                f"devices/{self.client_id}/status",
                "online",
                qos=1,
                retain=True
            )
        else:
            print(f"[✗] Connection failed, code {rc}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"[!] Unexpected disconnect, code {rc}")
        else:
            print("[✓] Disconnected")

    def on_publish(self, client, userdata, mid):
        print(f"  → Message {mid} published")

    def read_sensors(self):
        """Simulate sensor readings"""
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(40.0, 70.0), 2),
            "pressure": round(random.uniform(1000, 1020), 1)
        }

    def publish_sensor_data(self):
        """Read and publish sensor data"""
        data = self.read_sensors()

        # Publish to separate topics
        self.client.publish(
            f"sensors/{self.client_id}/temperature",
            data['temperature'],
            qos=1
        )
        self.client.publish(
            f"sensors/{self.client_id}/humidity",
            data['humidity'],
            qos=1
        )

        # Publish aggregated JSON
        self.client.publish(
            f"sensors/{self.client_id}/data",
            json.dumps(data),
            qos=1
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Published: {data}")

    def run(self, interval=10):
        """Connect and start publishing"""
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

        try:
            while True:
                self.publish_sensor_data()
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[!] Stopping publisher...")
        finally:
            # Graceful shutdown
            self.client.publish(
                f"devices/{self.client_id}/status",
                "offline",
                qos=1,
                retain=True
            )
            self.client.disconnect()
            self.client.loop_stop()

if __name__ == "__main__":
    publisher = SensorPublisher(broker="192.168.1.100")
    publisher.run(interval=10)
```

### 6.2 Data Subscriber and Logger

```python
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import csv
import os

class SensorLogger:
    def __init__(self, broker, port=1883, log_file="sensor_data.csv"):
        self.broker = broker
        self.port = port
        self.log_file = log_file
        self.client = mqtt.Client("logger_001")

        # Set callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Initialize CSV file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'device_id', 'temperature', 'humidity', 'pressure'])

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[✓] Connected to broker: {self.broker}")

            # Subscribe to all sensor data
            client.subscribe("sensors/+/data", qos=1)
            # Subscribe to device status
            client.subscribe("devices/+/status", qos=1)

            print("[✓] Subscribed to topics")
        else:
            print(f"[✗] Connection failed, code {rc}")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode()

        # Handle device status
        if "/status" in topic:
            device_id = topic.split('/')[1]
            print(f"[STATUS] {device_id}: {payload}")
            return

        # Handle sensor data
        if "/data" in topic:
            try:
                data = json.loads(payload)
                device_id = topic.split('/')[1]

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{device_id}: Temp={data['temperature']}°C, "
                      f"Humidity={data['humidity']}%, "
                      f"Pressure={data['pressure']}hPa")

                # Log to CSV
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        data['timestamp'],
                        device_id,
                        data['temperature'],
                        data['humidity'],
                        data['pressure']
                    ])

            except json.JSONDecodeError:
                print(f"[✗] Invalid JSON: {payload}")

    def run(self):
        """Connect and start logging"""
        self.client.connect(self.broker, self.port, keepalive=60)

        print(f"[✓] Logging to: {self.log_file}")
        print("[✓] Waiting for messages... (Ctrl+C to stop)")

        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\n[!] Stopping logger...")
            self.client.disconnect()

if __name__ == "__main__":
    logger = SensorLogger(broker="192.168.1.100")
    logger.run()
```

---

## 7. Advanced MQTT Patterns

### 7.1 Request/Response Pattern

MQTT can implement request/response using separate topics:

```python
import paho.mqtt.client as mqtt
import json
import time
import uuid

class MQTTRequestResponse:
    def __init__(self, broker, port=1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.pending_requests = {}

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        # Subscribe to response topic
        client.subscribe("response/#", qos=1)

    def on_message(self, client, userdata, msg):
        # Handle response
        request_id = msg.topic.split('/')[-1]
        if request_id in self.pending_requests:
            response = json.loads(msg.payload.decode())
            self.pending_requests[request_id] = response

    def send_request(self, request_data, timeout=5):
        """Send request and wait for response"""
        request_id = str(uuid.uuid4())
        self.pending_requests[request_id] = None

        # Publish request
        request = {
            'request_id': request_id,
            'data': request_data
        }
        self.client.publish("request", json.dumps(request), qos=1)

        # Wait for response
        start_time = time.time()
        while self.pending_requests[request_id] is None:
            if time.time() - start_time > timeout:
                del self.pending_requests[request_id]
                raise TimeoutError("Request timeout")
            time.sleep(0.1)

        response = self.pending_requests[request_id]
        del self.pending_requests[request_id]
        return response

# Server side (request handler)
def on_request(client, userdata, msg):
    request = json.loads(msg.payload.decode())
    request_id = request['request_id']

    # Process request
    result = {"status": "success", "result": "processed"}

    # Send response
    client.publish(f"response/{request_id}", json.dumps(result), qos=1)
```

### 7.2 Message Routing Pattern

```python
import paho.mqtt.client as mqtt
import json

class MessageRouter:
    def __init__(self, broker, port=1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Routing rules
        self.routes = {
            'sensors/+/temperature': self.handle_temperature,
            'sensors/+/alert': self.handle_alert,
            'devices/+/command': self.handle_command
        }

    def on_connect(self, client, userdata, flags, rc):
        # Subscribe to all routing topics
        for topic in self.routes.keys():
            client.subscribe(topic, qos=1)
            print(f"Subscribed to: {topic}")

    def on_message(self, client, userdata, msg):
        # Find matching route
        for pattern, handler in self.routes.items():
            if self.topic_matches(msg.topic, pattern):
                handler(msg)
                break

    def topic_matches(self, topic, pattern):
        """Simple topic matching with + wildcard"""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')

        if len(topic_parts) != len(pattern_parts):
            return False

        for t, p in zip(topic_parts, pattern_parts):
            if p != '+' and t != p:
                return False
        return True

    def handle_temperature(self, msg):
        device_id = msg.topic.split('/')[1]
        temp = float(msg.payload.decode())
        print(f"Temperature from {device_id}: {temp}°C")

        # Route to alerting if high temperature
        if temp > 35.0:
            self.client.publish(
                f"sensors/{device_id}/alert",
                json.dumps({"type": "high_temp", "value": temp}),
                qos=1
            )

    def handle_alert(self, msg):
        device_id = msg.topic.split('/')[1]
        alert = json.loads(msg.payload.decode())
        print(f"[ALERT] {device_id}: {alert}")

    def handle_command(self, msg):
        device_id = msg.topic.split('/')[1]
        command = msg.payload.decode()
        print(f"[COMMAND] {device_id}: {command}")

    def run(self):
        self.client.connect(self.broker, self.port)
        self.client.loop_forever()
```

## 8. CoAP Protocol

While MQTT dominates push-based IoT messaging, many IoT scenarios require a request/response pattern similar to HTTP -- but without HTTP's overhead. CoAP (Constrained Application Protocol) fills this gap as a lightweight RESTful protocol designed specifically for constrained devices and lossy networks.

### 8.1 CoAP vs HTTP

| Feature | CoAP | HTTP |
|---------|------|------|
| **Transport** | UDP (with optional DTLS) | TCP (with TLS) |
| **Header size** | 4 bytes (fixed) | Variable (~800 bytes avg) |
| **Methods** | GET, POST, PUT, DELETE | GET, POST, PUT, DELETE, PATCH, ... |
| **Discovery** | Built-in (`/.well-known/core`) | None (requires documentation) |
| **Observe** | Native (like subscription) | Requires polling or WebSocket |
| **Multicast** | Supported (UDP multicast) | Not supported |
| **Reliability** | Confirmable / Non-confirmable | TCP guarantees delivery |
| **Best for** | Constrained devices, sensor queries | Web applications, rich clients |

> **When to use CoAP vs MQTT.** Use MQTT when devices push data continuously (sensor telemetry, event streams). Use CoAP when you need to query device state on demand (read sensor value, change configuration), especially when the device is too constrained for a persistent TCP connection. Many systems use both: MQTT for telemetry and CoAP for device management.

### 8.2 CoAP Message Format

```
┌─────────────────────────────────────────────────────────────┐
│                    CoAP Message Format                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0                   1                   2                   │
│  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3           │
│ ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐       │
│ │Ver│ T │ TKL │     Code      │      Message ID     │       │
│ └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘       │
│ │              Token (0-8 bytes)               │             │
│ ├──────────────────────────────────────────────┤             │
│ │              Options (variable)              │             │
│ ├──────────────────────────────────────────────┤             │
│ │         0xFF (payload marker)                │             │
│ ├──────────────────────────────────────────────┤             │
│ │              Payload (variable)              │             │
│ └──────────────────────────────────────────────┘             │
│                                                             │
│  Ver: Version (1 = CoAP v1)                                │
│  T:   Type (CON=0, NON=1, ACK=2, RST=3)                   │
│  TKL: Token Length (0-8)                                    │
│  Code: Method (0.01=GET, 0.02=POST, 0.03=PUT, 0.04=DELETE)│
│        or Response (2.05=Content, 4.04=Not Found, etc.)    │
│                                                             │
│  Message Types:                                             │
│  • CON (Confirmable): Requires ACK, retransmits on timeout │
│  • NON (Non-confirmable): Fire and forget                  │
│  • ACK (Acknowledgement): Confirms CON receipt             │
│  • RST (Reset): Indicates message processing error         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 CoAP Observe Pattern

The Observe option lets a client register interest in a resource and receive notifications when it changes -- similar to MQTT subscription but within a RESTful framework:

```
Client                          Server
  │                               │
  │──GET /temperature (Observe)──>│   Register observation
  │<──2.05 Content (temp=23.5)────│   Initial value
  │                               │
  │       ... time passes ...     │
  │                               │
  │<──2.05 Content (temp=24.1)────│   Value changed, notify
  │<──2.05 Content (temp=24.8)────│   Value changed, notify
  │                               │
  │──GET /temperature (no Observe)│   Cancel observation
  │<──2.05 Content (temp=24.8)────│
```

### 8.4 Python aiocoap Example

```python
# pip install aiocoap

# --- CoAP Server ---
import asyncio
import aiocoap
import aiocoap.resource as resource

class TemperatureResource(resource.ObservableResource):
    """CoAP observable temperature resource."""

    def __init__(self):
        super().__init__()
        self.temperature = 22.5
        # Periodically update and notify observers
        self._notify_task = None

    async def start_updates(self):
        """Simulate temperature changes and notify observers."""
        import random
        while True:
            await asyncio.sleep(10)
            self.temperature = round(self.temperature + random.uniform(-0.5, 0.5), 1)
            self.updated_state()  # Notify all observers

    async def render_get(self, request):
        """Handle GET requests."""
        payload = f"{self.temperature}".encode('ascii')
        return aiocoap.Message(payload=payload, content_format=0)

class DeviceInfoResource(resource.Resource):
    """Static device information resource."""

    async def render_get(self, request):
        payload = '{"device": "sensor-01", "firmware": "1.2.0"}'.encode()
        return aiocoap.Message(payload=payload, content_format=50)

async def main():
    root = resource.Site()
    temp_resource = TemperatureResource()
    root.add_resource(['temperature'], temp_resource)
    root.add_resource(['info'], DeviceInfoResource())

    # Start the CoAP server
    await aiocoap.Context.create_server_context(root, bind=('::', 5683))

    # Start temperature update simulation
    await temp_resource.start_updates()

if __name__ == "__main__":
    asyncio.run(main())

# --- CoAP Client ---
# import asyncio
# import aiocoap
#
# async def main():
#     context = await aiocoap.Context.create_client_context()
#
#     # Simple GET request
#     request = aiocoap.Message(code=aiocoap.GET,
#                               uri='coap://localhost/temperature')
#     response = await context.request(request).response
#     print(f"Temperature: {response.payload.decode()}")
#
#     # Observe (subscribe to changes)
#     request = aiocoap.Message(code=aiocoap.GET,
#                               uri='coap://localhost/temperature',
#                               observe=0)
#     requester = context.request(request)
#     async for response in requester.observation:
#         print(f"Update: {response.payload.decode()}")
#
# asyncio.run(main())
```

### 8.5 CoAP Security: DTLS

CoAP uses DTLS (Datagram Transport Layer Security) -- the UDP equivalent of TLS -- for encryption and authentication. DTLS provides the same security guarantees as TLS but handles the challenges of UDP (packet reordering, loss):

- **Pre-Shared Key (PSK)**: Simplest, suitable for constrained devices
- **Raw Public Key (RPK)**: Certificates without CA, smaller footprint
- **X.509 Certificates**: Full PKI, highest security

---

## 9. OTA Firmware Updates

Over-the-Air (OTA) updates are essential for IoT devices deployed in the field where physical access is impractical. A robust OTA system ensures devices stay secure, receive bug fixes, and gain new features throughout their lifecycle.

### 9.1 Why OTA?

- **Remote devices**: Thousands of sensors in farms, factories, or cities cannot be updated manually
- **Security patches**: Vulnerabilities must be patched quickly across the entire fleet
- **Feature delivery**: New capabilities without hardware replacement
- **Regulatory compliance**: Some industries require timely security updates

### 9.2 OTA Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OTA Update Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                                         │
│  │  Build Server │  CI/CD pipeline builds firmware          │
│  │  (CI/CD)      │  Signs binary with private key           │
│  └───────┬───────┘                                         │
│          │ Upload signed firmware                            │
│          ▼                                                  │
│  ┌───────────────┐                                         │
│  │  OTA Server   │  Stores firmware versions                │
│  │  + CDN        │  Manages rollout (staged/canary)         │
│  └───────┬───────┘                                         │
│          │ Notify devices (MQTT / CoAP / HTTP)              │
│          ▼                                                  │
│  ┌───────────────────────────────────────────────────┐     │
│  │              IoT Device Fleet                      │     │
│  │                                                    │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │     │
│  │  │ Device 1 │  │ Device 2 │  │ Device N │        │     │
│  │  │          │  │          │  │          │        │     │
│  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │        │     │
│  │  │ │Slot A│ │  │ │Slot A│ │  │ │Slot A│ │        │     │
│  │  │ │(act) │ │  │ │(act) │ │  │ │(act) │ │        │     │
│  │  │ ├──────┤ │  │ ├──────┤ │  │ ├──────┤ │        │     │
│  │  │ │Slot B│ │  │ │Slot B│ │  │ │Slot B│ │        │     │
│  │  │ │(idle)│ │  │ │(idle)│ │  │ │(idle)│ │        │     │
│  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │        │     │
│  │  └──────────┘  └──────────┘  └──────────┘        │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Update Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **A/B Partition** | Two firmware slots; write to inactive, swap on reboot | Instant rollback, atomic | Requires 2x storage |
| **Delta Update** | Send only binary diff between versions | Small download size | Complex diffing, fragile |
| **Full Image** | Send complete firmware image | Simple, reliable | Large download |
| **Container Update** | Update application container, not OS | Fast, isolated | Requires container runtime |

### 9.4 OTA Security

```
┌─────────────────────────────────────────────────────────────┐
│                    OTA Security Checklist                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Code Signing                                            │
│     • Sign firmware with private key (Ed25519 or RSA)      │
│     • Device verifies signature with embedded public key    │
│     • Reject unsigned or tampered firmware                  │
│                                                             │
│  2. Secure Transport                                        │
│     • TLS/DTLS for firmware download                       │
│     • Certificate pinning to prevent MITM                  │
│                                                             │
│  3. Rollback Protection                                     │
│     • Monotonic version counter in secure storage          │
│     • Prevent downgrade to older vulnerable firmware       │
│                                                             │
│  4. Integrity Check                                         │
│     • SHA-256 hash verification before flashing            │
│     • Post-flash verification (CRC check on boot)         │
│                                                             │
│  5. Recovery Mechanism                                      │
│     • Watchdog timer: auto-rollback if new firmware fails  │
│     • Bootloader fallback to known-good partition          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.5 Cloud OTA Services

| Service | Protocol | Features |
|---------|----------|----------|
| **AWS IoT Jobs** | MQTT + HTTPS | Job targeting, rollout config, device shadows |
| **Azure IoT Hub** | MQTT + HTTPS | Device twins, automatic device management |
| **Mender.io** | HTTPS | Open-source, A/B partition, delta updates |
| **balena** | HTTPS | Container-based OTA, fleet management |

### 9.6 Practical OTA Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  1. Build │───>│ 2. Sign  │───>│ 3. Upload│───>│4. Notify │
│  firmware │    │ (Ed25519)│    │ to server│    │ (MQTT)   │
└──────────┘    └──────────┘    └──────────┘    └─────┬────┘
                                                      │
      ┌──────────────────────────────────────────────┘
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│5. Device │───>│6. Verify │───>│7. Flash  │───>│ 8. Reboot│
│ downloads│    │ signature│    │ to Slot B│    │ & verify │
│ firmware │    │ + hash   │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └─────┬────┘
                                                      │
                                              ┌───────┴───────┐
                                              │  Health check  │
                                              │  passed?       │
                                              ├───────┬───────┤
                                              │ YES   │  NO   │
                                              ▼       ▼       │
                                         ┌────────┐ ┌────────┐│
                                         │Confirm │ │Rollback││
                                         │Slot B  │ │Slot A  ││
                                         │as boot │ │as boot ││
                                         └────────┘ └────────┘│
```

---

## 10. Summary

### Completed Tasks

- ✅ **MQTT Basics**: Pub/Sub architecture, MQTT vs HTTP comparison
- ✅ **Mosquitto Broker**: Installation, configuration, authentication
- ✅ **Topics**: Hierarchical structure, wildcards, reserved topics
- ✅ **QoS Levels**: QoS 0/1/2 guarantees and use cases
- ✅ **paho-mqtt**: Publisher, subscriber, authentication, retained messages, LWT
- ✅ **Practical Project**: Complete sensor system with MQTT
- ✅ **Advanced Patterns**: Request/response, message routing

### Hands-On Exercises

1. **Multi-Sensor System**:
   - Deploy 3 virtual sensors (temp, humidity, motion)
   - Each publishes to separate topics
   - Create unified dashboard subscriber

2. **Alert System**:
   - Monitor sensor values
   - Trigger alerts on threshold violations
   - Implement escalation (email, SMS)

3. **Device Control**:
   - Create command topics for LED control
   - Implement status reporting
   - Add acknowledgment mechanism

4. **MQTT Bridge**:
   - Connect two MQTT brokers
   - Forward messages between them
   - Implement topic filtering

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Connection refused** | Broker not running | Check `systemctl status mosquitto` |
| **Authentication failed** | Wrong credentials | Verify username/password, check password file |
| **Messages not received** | Topic mismatch | Verify topic spelling, check wildcards |
| **High latency** | Network issues | Check QoS level, broker load, network quality |

---

## References

- [MQTT Protocol Specification](https://mqtt.org/mqtt-specification/)
- [Mosquitto Documentation](https://mosquitto.org/documentation/)
- [paho-mqtt Documentation](https://www.eclipse.org/paho/index.php?page=clients/python/docs/index.php)
- [HiveMQ MQTT Essentials](https://www.hivemq.com/mqtt-essentials/)
- [CoAP RFC 7252](https://datatracker.ietf.org/doc/html/rfc7252)
- [aiocoap Documentation](https://aiocoap.readthedocs.io/)
- [Mender.io OTA Documentation](https://docs.mender.io/)
- [AWS IoT Jobs](https://docs.aws.amazon.com/iot/latest/developerguide/iot-jobs.html)

---

## Exercises

### Exercise 1: Topic Hierarchy Design and Wildcard Subscriptions

Design and test a complete MQTT topic hierarchy for a smart building system:

1. Design a topic structure for a 3-floor building with 5 rooms per floor. Each room has temperature, humidity, motion, and light sensors, plus a controllable HVAC unit.
2. Write out at least 10 concrete topic paths following the best practices in Section 3.1 (lowercase, no leading slash, 3-5 levels deep).
3. Write the wildcard subscription strings that would: (a) receive all temperature readings from floor 2 only, (b) receive all sensor data from room 101 regardless of sensor type, (c) receive motion alerts from every room in the building.
4. Use `mosquitto_pub` and `mosquitto_sub` from the command line to test at least 3 of your topic paths on a local Mosquitto broker.

### Exercise 2: QoS Level Comparison

Implement a Python experiment that demonstrates the practical differences between QoS levels:

1. Set up a local Mosquitto broker. Write a publisher that sends 100 messages to three parallel topics -- one at QoS 0, one at QoS 1, one at QoS 2 -- at a rate of 10 messages per second.
2. Write three corresponding subscribers (one per QoS topic) that count received messages and log arrival timestamps.
3. Simulate packet loss by temporarily killing and restarting the broker mid-experiment (after approximately 50 messages). Record how many messages each subscriber received in total.
4. Explain in a 150-word summary: which QoS level should be used for (a) frequent non-critical sensor readings, (b) billing or payment events, (c) device commands that must be executed exactly once.

### Exercise 3: Device Status Tracking with LWT and Retained Messages

Build a device fleet status dashboard using LWT and retained messages:

1. Write a `SensorPublisher` class (based on Section 6.1) that sets an LWT message of `"offline"` (retained, QoS 1) on `devices/<client_id>/status` before connecting, and publishes `"online"` (retained, QoS 1) on successful connection.
2. Create a `FleetMonitor` subscriber that subscribes to `devices/+/status` using a wildcard and maintains a dictionary of `{device_id: status}`. Print a live fleet status table every 10 seconds.
3. Start 3 publisher instances simultaneously with different client IDs. Kill one of them abruptly (using `kill -9` or pulling the network cable). Verify the fleet monitor receives the LWT `"offline"` message.
4. Start a new subscriber after all publishers are running and verify it immediately receives the retained `"online"` status for all active devices.

### Exercise 4: MQTT-Based Remote GPIO Control

Build a system that controls Raspberry Pi GPIO pins via MQTT messages from any device on the network:

1. Write an MQTT subscriber on the Pi that listens on `gpio/+/command`. The `+` level is the GPIO pin number (e.g., `gpio/17/command`). The payload is `"on"`, `"off"`, or `"toggle"`.
2. When a command is received, apply it to the corresponding GPIO pin using `gpiozero`. Publish the new pin state as `"1"` or `"0"` to `gpio/<pin>/state` (retained, QoS 1).
3. Write a client script (can run on the Pi or any computer) that reads commands from standard input and publishes them to the appropriate topic.
4. Test by controlling an LED from your laptop: `mosquitto_pub -t "gpio/17/command" -m "toggle"`. Confirm the LED toggles and the state topic updates.

### Exercise 5: Complete IoT Pipeline -- Sensor to Dashboard

Build an end-to-end pipeline: sensor to MQTT broker to logger to simple dashboard:

1. Write a `SensorPublisher` that simulates a DHT11 by publishing random temperature (20-30 degrees C) and humidity (40-70%) to `sensors/<device_id>/temperature` and `sensors/<device_id>/humidity` every 5 seconds, using QoS 1 and retained messages.
2. Write a `DataLogger` subscriber that subscribes to `sensors/#` and appends all received readings to a CSV file with columns: `timestamp`, `device_id`, `metric`, `value`.
3. Write an `AlertMonitor` subscriber that subscribes to the same topics and publishes an alert to `alerts/<device_id>/high_temp` (QoS 2, retained) whenever temperature exceeds 27 degrees C.
4. Write a `Dashboard` subscriber that subscribes to both `sensors/#` and `alerts/#`, maintains the latest value per device per metric in memory, and prints a formatted status table to the terminal every 15 seconds showing current readings and any active alerts.

---

**Previous**: [BLE Connectivity](./05_BLE_Connectivity.md) | **Next**: [HTTP/REST for IoT](./07_HTTP_REST_for_IoT.md)
