# 01. IoT Overview

**Next**: [Raspberry Pi Setup](./02_Raspberry_Pi_Setup.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define IoT (Internet of Things) and explain its core concepts
2. Identify the components of a three-layer IoT system architecture
3. Compare edge computing and cloud computing trade-offs
4. Describe major IoT protocols and their appropriate use cases
5. Recognize common IoT security threats and best practices

---

The Internet of Things is reshaping every industry -- from agriculture to healthcare to manufacturing. Understanding IoT fundamentals is the essential first step before building any connected device or sensor network, because the architectural decisions you make early (which protocol, where to process data, how to secure devices) cascade through every stage of your project.

---

## 1. What is IoT?

### 1.1 Definition

**IoT (Internet of Things)** is a system where physical devices equipped with sensors, software, and network connectivity collect and exchange data.

```
┌─────────────────────────────────────────────────────────────┐
│                        IoT Ecosystem                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│   │ Sensor  │    │ Gateway │    │  Cloud  │    │  User   │  │
│   │ Device  │───▶│         │───▶│  Server │───▶│   App   │  │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│        │              │              │              │       │
│   Temperature,    Data            Storage,      Visualize,   │
│   Humidity,       Aggregation     Analysis      Control      │
│   Motion          Protocol        ML/AI         Commands     │
│   Detection       Conversion      Processing                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Core Elements of IoT

| Element | Description | Example |
|---------|-------------|---------|
| **Things** | Physical devices equipped with sensors/actuators | Temperature sensor, smart lights |
| **Connectivity** | Network for data transmission between devices | WiFi, BLE, LoRa, 5G |
| **Data Processing** | Processing and analysis of collected data | Edge processing, cloud analytics |
| **User Interface** | Interaction between users and system | Mobile app, web dashboard |

### 1.3 IoT Application Areas

```python
# IoT Application Examples
iot_applications = {
    "Smart Home": ["Temperature control", "Lighting control", "Security cameras", "Voice assistants"],
    "Smart City": ["Traffic management", "Street light control", "Waste bin monitoring"],
    "Industrial IoT": ["Predictive maintenance", "Asset tracking", "Quality control"],
    "Healthcare": ["Wearable devices", "Remote monitoring", "Medication management"],
    "Agriculture": ["Soil sensors", "Automated irrigation", "Drone monitoring"],
}

for sector, applications in iot_applications.items():
    print(f"\n{sector}:")
    for app in applications:
        print(f"  - {app}")
```

---

## 2. IoT System Architecture

### 2.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  3-Layer IoT Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Application Layer                        │  │
│  │  • Data visualization                                  │  │
│  │  • Business logic                                      │  │
│  │  • User interface                                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ▲                                 │
│                            │                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               Network Layer                           │  │
│  │  • Data transmission                                   │  │
│  │  • Protocol conversion                                 │  │
│  │  • Gateway                                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ▲                                 │
│                            │                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Perception Layer                         │  │
│  │  • Sensors                                             │  │
│  │  • Actuators                                           │  │
│  │  • Embedded systems                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Roles

```python
# IoT Architecture Layer Definition
class IoTArchitecture:
    """IoT 3-Layer Architecture Model"""

    layers = {
        "perception": {
            "name": "Perception Layer",
            "components": ["Sensors", "Actuators", "RFID", "GPS"],
            "function": "Collect data from physical environment and perform actions",
            "devices": ["Raspberry Pi", "Arduino", "ESP32"]
        },
        "network": {
            "name": "Network Layer",
            "components": ["Gateway", "Router", "Protocol Converter"],
            "function": "Data transmission and routing",
            "protocols": ["WiFi", "BLE", "LoRa", "Zigbee", "MQTT", "HTTP"]
        },
        "application": {
            "name": "Application Layer",
            "components": ["Cloud Server", "Database", "Analytics Engine"],
            "function": "Data storage, analysis, visualization",
            "services": ["AWS IoT", "Azure IoT", "Google Cloud IoT"]
        }
    }

    @classmethod
    def describe_layer(cls, layer_name: str):
        layer = cls.layers.get(layer_name)
        if layer:
            print(f"Layer: {layer['name']}")
            print(f"Function: {layer['function']}")
            print(f"Components: {', '.join(layer['components'])}")

# Usage example
IoTArchitecture.describe_layer("perception")
```

---

## 3. Edge vs Cloud Computing

### 3.1 Concept Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                  Edge vs Cloud Computing                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                      ┌─────────────────────┐ │
│   │   Sensor    │                      │      Cloud          │ │
│   │   Device    │                      │      Server         │ │
│   └──────┬──────┘                      └──────────┬──────────┘ │
│          │                                        │            │
│          ▼                                        │            │
│   ┌─────────────┐      ┌──────────┐              │            │
│   │   Edge      │─────▶│ Network  │──────────────┘            │
│   │   Device    │      └──────────┘                           │
│   └─────────────┘                                              │
│          │                                                     │
│   Local processing                                             │
│   - Filtering                                                  │
│   - Aggregation                                                │
│   - Simple analysis                                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Comparison Table

| Feature | Edge Computing | Cloud Computing |
|---------|----------------|-----------------|
| **Processing Location** | Near data source | Remote data center |
| **Latency** | Very low (< 10ms) | High (100ms+) |
| **Bandwidth** | Low requirement | High requirement |
| **Offline** | Can operate | Connection required |
| **Cost** | High initial investment | Increasing operational cost |
| **Processing Power** | Limited | Unlimited scaling |
| **Example** | Raspberry Pi | AWS, GCP, Azure |

### 3.3 Hybrid Approach

```python
# Edge-Cloud Hybrid Architecture Example
class HybridIoTSystem:
    """IoT system combining edge and cloud"""

    def __init__(self):
        self.edge_buffer = []
        self.cloud_threshold = 100  # Send to cloud every 100 data points

    def process_at_edge(self, sensor_data: dict) -> dict:
        """Tasks to process immediately at edge"""
        # 1. Anomaly detection (requires immediate response)
        if sensor_data.get("temperature", 0) > 50:
            self.trigger_local_alarm()

        # 2. Data filtering/cleaning
        cleaned_data = self.filter_noise(sensor_data)

        # 3. Local storage and aggregation
        self.edge_buffer.append(cleaned_data)

        return cleaned_data

    def should_send_to_cloud(self) -> bool:
        """Check cloud transmission condition"""
        return len(self.edge_buffer) >= self.cloud_threshold

    def send_to_cloud(self):
        """Send aggregated data to cloud"""
        if self.should_send_to_cloud():
            aggregated = self.aggregate_data(self.edge_buffer)
            # cloud_client.publish(aggregated)
            self.edge_buffer.clear()
            return aggregated

    def filter_noise(self, data: dict) -> dict:
        """Remove noise (edge processing)"""
        return {k: v for k, v in data.items() if v is not None}

    def aggregate_data(self, buffer: list) -> dict:
        """Aggregate data"""
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
        """Trigger local alarm (low-latency response)"""
        print("WARNING: High temperature detected!")
```

---

## 4. IoT Protocol Overview

### 4.1 Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    IoT Protocol Stack                        │
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

### 4.2 Major Application Protocols

#### MQTT (Message Queuing Telemetry Transport)

```python
# MQTT Protocol Characteristics
mqtt_characteristics = {
    "type": "Pub/Sub Messaging",
    "transport": "TCP",
    "port": 1883,  # TLS: 8883
    "qos_levels": [0, 1, 2],  # At most once, At least once, Exactly once
    "use_cases": ["Sensor data", "Real-time monitoring", "Notifications"],
    "advantages": ["Lightweight", "Low bandwidth", "Reliability options"],
    "broker": ["Mosquitto", "HiveMQ", "EMQX"]
}

# MQTT Topic Structure Example
topics = [
    "home/living-room/temperature",
    "home/living-room/humidity",
    "home/+/temperature",  # + : Single level wildcard
    "home/#"               # # : Multi-level wildcard
]
```

#### HTTP/REST

```python
# REST API IoT Pattern
rest_patterns = {
    "GET /sensors": "List all sensors",
    "GET /sensors/{id}": "Get specific sensor info",
    "GET /sensors/{id}/data": "Get sensor data",
    "POST /sensors/{id}/data": "Send new sensor data",
    "PUT /devices/{id}/config": "Update device configuration",
}

# RESTful IoT Request Example
import requests

def get_sensor_data(sensor_id: str, api_base: str = "http://iot-server:8080"):
    """Query sensor data"""
    response = requests.get(f"{api_base}/sensors/{sensor_id}/data")
    return response.json()

def post_sensor_reading(sensor_id: str, data: dict, api_base: str = "http://iot-server:8080"):
    """Send sensor data"""
    response = requests.post(
        f"{api_base}/sensors/{sensor_id}/data",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    return response.status_code == 201
```

#### CoAP (Constrained Application Protocol)

```python
# CoAP Protocol Characteristics
coap_characteristics = {
    "type": "RESTful (UDP-based)",
    "transport": "UDP",
    "port": 5683,  # DTLS: 5684
    "features": ["Lightweight HTTP alternative", "Multicast support", "Asynchronous messaging"],
    "use_cases": ["Low-power devices", "Constrained networks"],
    "message_types": ["CON", "NON", "ACK", "RST"]
}
```

### 4.3 Protocol Selection Guide

| Requirement | Recommended Protocol |
|-------------|---------------------|
| Real-time bidirectional communication | MQTT, WebSocket |
| Low-power devices | CoAP, MQTT-SN |
| Leverage existing web infrastructure | HTTP/REST |
| Large-scale message processing | AMQP, Kafka |
| Simple data collection | MQTT |

---

## 5. IoT Security Considerations

### 5.1 Security Threats

```
┌─────────────────────────────────────────────────────────────┐
│                    IoT Security Threat Types                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Device    │    │   Network   │    │    Cloud    │     │
│  │   Attack    │    │    Attack   │    │    Attack   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        │                  │                  │              │
│  • Physical access    • Man-in-middle    • Auth bypass      │
│  • Firmware extract   • Eavesdropping    • API vulnerabil   │
│  • Side-channel      • Spoofing          • Data breach      │
│  • Default creds     • DDoS              • Privilege escal  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Security Best Practices

```python
# IoT Security Checklist
security_checklist = {
    "device_security": [
        "Change default passwords",
        "Regular firmware updates",
        "Disable unnecessary ports/services",
        "Enable Secure Boot",
        "Restrict physical access"
    ],
    "network_security": [
        "Use TLS/DTLS encryption",
        "Network isolation (VLAN)",
        "Firewall configuration",
        "Consider VPN usage"
    ],
    "data_security": [
        "Encryption in transit (TLS)",
        "Encryption at rest (AES)",
        "Principle of least privilege",
        "Regular backups"
    ],
    "authentication": [
        "Strong authentication mechanisms",
        "Certificate-based authentication",
        "Token-based authentication (JWT)",
        "API key management"
    ]
}

# MQTT Connection with TLS Example
import ssl

def create_secure_mqtt_context():
    """Create SSL context for secure MQTT connection"""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
        certfile="client-cert.pem",
        keyfile="client-key.pem"
    )
    context.load_verify_locations("ca-cert.pem")
    context.verify_mode = ssl.CERT_REQUIRED
    return context
```

### 5.3 Data Encryption Example

```python
from cryptography.fernet import Fernet
import json

class SecureDataHandler:
    """IoT Data Encryption Handler"""

    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_data(self, data: dict) -> bytes:
        """Encrypt sensor data"""
        json_data = json.dumps(data).encode()
        return self.cipher.encrypt(json_data)

    def decrypt_data(self, encrypted: bytes) -> dict:
        """Decrypt encrypted data"""
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())

# Usage example
handler = SecureDataHandler()
sensor_data = {"temperature": 25.5, "humidity": 60}

encrypted = handler.encrypt_data(sensor_data)
print(f"Encrypted data: {encrypted[:50]}...")

decrypted = handler.decrypt_data(encrypted)
print(f"Decrypted data: {decrypted}")
```

---

## 6. IoT Development Environment

### 6.1 Hardware Platforms

| Platform | CPU | RAM | Features | Use Case |
|----------|-----|-----|----------|----------|
| Raspberry Pi 4 | ARM Cortex-A72 | 1-8GB | Full Linux OS | Gateway, Edge AI |
| Raspberry Pi Pico | RP2040 | 264KB | Microcontroller | Sensor node |
| ESP32 | Xtensa LX6 | 520KB | WiFi/BLE built-in | IoT sensor |
| Arduino | ATmega/ARM | 2-256KB | Simple, low-power | Prototyping |

### 6.2 Development Tools

```python
# Recommended Development Environment
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

## Practice Problems

### Problem 1: IoT System Design
Design a smart parking lot system. Include:
- Types of sensors needed
- Communication protocol selection (with reasoning)
- Edge vs cloud processing distribution

### Problem 2: Protocol Selection
Select appropriate protocols for the following scenarios and explain why:
1. Battery-powered remote temperature sensor
2. Real-time security camera video streaming
3. Smart lighting control system

### Problem 3: Security Analysis
List 3 potential security vulnerabilities for a home smart door lock and propose countermeasures for each.

---

**Next**: [Raspberry Pi Setup](./02_Raspberry_Pi_Setup.md)
