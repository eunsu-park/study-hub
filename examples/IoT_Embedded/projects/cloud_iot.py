#!/usr/bin/env python3
"""
Cloud IoT Integration - Simulation Mode
AWS IoT Core and GCP Pub/Sub Integration Simulation

Operates without actual cloud accounts or credentials
"""

import json
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import random


# ==============================================================================
# Data Models
# ==============================================================================

class CloudProvider(Enum):
    """Cloud Provider"""
    AWS_IOT = "aws_iot"
    GCP_PUBSUB = "gcp_pubsub"
    SIMULATION = "simulation"


# Why: MQTT defines three QoS levels with very different trade-offs.
# QoS 0 (fire-and-forget) saves bandwidth; QoS 1 (ACK) guarantees delivery
# at the cost of possible duplicates; QoS 2 (4-step handshake) ensures
# exactly-once but doubles round trips. IoT sensors typically use QoS 1.
class MessageQoS(Enum):
    """MQTT QoS Level"""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


@dataclass
class IoTMessage:
    """IoT Message"""
    topic: str
    payload: Dict
    timestamp: datetime
    message_id: str
    qos: MessageQoS = MessageQoS.AT_LEAST_ONCE

    def to_json(self) -> str:
        """JSON serialization"""
        data = {
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "qos": self.qos.value
        }
        return json.dumps(data)


@dataclass
class DeviceInfo:
    """Device Information"""
    device_id: str
    device_type: str
    location: str
    firmware_version: str
    registered_at: datetime


@dataclass
class TelemetryData:
    """Telemetry Data"""
    device_id: str
    temperature: float
    humidity: float
    pressure: float
    battery_level: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "device_id": self.device_id,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "battery_level": self.battery_level,
            "timestamp": self.timestamp.isoformat()
        }


# ==============================================================================
# AWS IoT Core Simulation
# ==============================================================================

class SimulatedAWSIoTClient:
    """AWS IoT Core Client (Simulation)"""

    def __init__(self, endpoint: str, cert_path: str, key_path: str,
                 ca_path: str, client_id: str):
        self.endpoint = endpoint
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.client_id = client_id
        self.connected = False
        self.subscriptions: Dict[str, Callable] = {}
        self.message_queue = queue.Queue()

        print(f"[AWS IoT Simulation] Client created")
        print(f"  - Endpoint: {endpoint}")
        print(f"  - Client ID: {client_id}")
        print(f"  - Certificate: {cert_path}")

    def connect(self):
        """Connect (simulation)"""
        print(f"\n[AWS IoT Simulation] Connecting...")
        time.sleep(0.5)  # Simulate connection delay

        # Certificate verification simulation
        print("  - TLS handshake...")
        time.sleep(0.2)
        print("  - Certificate verification...")
        time.sleep(0.2)
        print("  - MQTT connecting...")
        time.sleep(0.2)

        self.connected = True
        print("Connection successful!\n")

    def disconnect(self):
        """Disconnect"""
        if self.connected:
            print("[AWS IoT Simulation] Disconnected")
            self.connected = False

    def publish(self, topic: str, payload: Dict, qos: MessageQoS = MessageQoS.AT_LEAST_ONCE):
        """Publish message"""
        if not self.connected:
            raise RuntimeError("Not connected")

        message = IoTMessage(
            topic=topic,
            payload=payload,
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
            qos=qos
        )

        # Publish simulation
        print(f"[AWS IoT Publish] {topic}")
        print(f"  Message ID: {message.message_id[:8]}...")
        print(f"  QoS: {qos.name}")
        print(f"  Payload: {json.dumps(payload, indent=2)}")

        # Network delay simulation
        time.sleep(random.uniform(0.01, 0.05))

        return message.message_id

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        if not self.connected:
            raise RuntimeError("Not connected")

        self.subscriptions[topic] = callback
        print(f"[AWS IoT Subscribe] {topic}")

    def _simulate_incoming_message(self, topic: str, payload: Dict):
        """Simulate incoming message (for internal testing)"""
        if topic in self.subscriptions:
            callback = self.subscriptions[topic]
            callback(topic, payload)


class AWSIoTDeviceManager:
    """AWS IoT Device Manager"""

    def __init__(self, client: SimulatedAWSIoTClient):
        self.client = client
        self.device_info: Optional[DeviceInfo] = None

    def register_device(self, device_info: DeviceInfo):
        """Register device"""
        self.device_info = device_info

        print("\n[AWS IoT] Device registration")
        print(f"  - ID: {device_info.device_id}")
        print(f"  - Type: {device_info.device_type}")
        print(f"  - Location: {device_info.location}")
        print(f"  - Firmware: {device_info.firmware_version}")

        # Thing creation simulation
        print("  - Creating Thing...")
        time.sleep(0.3)

        # Certificate attachment simulation
        print("  - Attaching certificate...")
        time.sleep(0.3)

        # Policy attachment simulation
        print("  - Attaching IoT policy...")
        time.sleep(0.3)

        print("Device registration complete\n")

    def publish_telemetry(self, data: TelemetryData):
        """Publish telemetry"""
        topic = f"device/{data.device_id}/telemetry"
        payload = data.to_dict()

        self.client.publish(topic, payload)

    # Why: Device Shadow is AWS IoT's "digital twin" — a JSON document that
    # stores the last-known device state in the cloud. This lets mobile apps
    # read the device's state even when the device is offline or asleep.
    def update_device_shadow(self, state: Dict):
        """Update Device Shadow"""
        if not self.device_info:
            raise ValueError("No device info")

        topic = f"$aws/things/{self.device_info.device_id}/shadow/update"

        shadow_payload = {
            "state": {
                "reported": state
            },
            "metadata": {
                "reported": {
                    k: {"timestamp": int(time.time())}
                    for k in state.keys()
                }
            }
        }

        print(f"\n[AWS IoT] Device Shadow update")
        print(f"  State: {json.dumps(state, indent=2)}")

        self.client.publish(topic, shadow_payload)


# ==============================================================================
# GCP Pub/Sub Simulation
# ==============================================================================

class SimulatedGCPPubSubPublisher:
    """GCP Pub/Sub Publisher (Simulation)"""

    def __init__(self, project_id: str, topic_id: str):
        self.project_id = project_id
        self.topic_id = topic_id
        self.topic_path = f"projects/{project_id}/topics/{topic_id}"

        print(f"[GCP Pub/Sub Simulation] Publisher created")
        print(f"  - Project: {project_id}")
        print(f"  - Topic: {topic_id}")
        print(f"  - Path: {self.topic_path}")

    def publish(self, data: Dict, **attributes) -> str:
        """Publish message"""
        message_id = str(uuid.uuid4())

        print(f"\n[GCP Pub/Sub Publish]")
        print(f"  Topic: {self.topic_id}")
        print(f"  Message ID: {message_id[:8]}...")

        if attributes:
            print(f"  Attributes: {attributes}")

        print(f"  Data: {json.dumps(data, indent=2)}")

        # Network delay simulation
        time.sleep(random.uniform(0.01, 0.05))

        print(f"Publish complete\n")
        return message_id

    def publish_batch(self, messages: List[Dict]) -> List[str]:
        """Batch publish"""
        print(f"\n[GCP Pub/Sub Batch Publish] {len(messages)} messages")

        message_ids = []
        for i, data in enumerate(messages):
            message_id = str(uuid.uuid4())
            message_ids.append(message_id)
            print(f"  [{i+1}] {message_id[:8]}...")

        time.sleep(random.uniform(0.05, 0.1))
        print(f"Batch publish complete\n")

        return message_ids


class SimulatedGCPPubSubSubscriber:
    """GCP Pub/Sub Subscriber (Simulation)"""

    def __init__(self, project_id: str, subscription_id: str):
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.subscription_path = f"projects/{project_id}/subscriptions/{subscription_id}"
        self.message_queue = queue.Queue()

        print(f"[GCP Pub/Sub Simulation] Subscriber created")
        print(f"  - Subscription: {subscription_id}")
        print(f"  - Path: {self.subscription_path}")

    def pull(self, max_messages: int = 10) -> List[Dict]:
        """Pull messages (synchronous)"""
        print(f"\n[GCP Pub/Sub Pull] Max {max_messages} messages")

        messages = []

        # Simulation: get messages from queue
        for _ in range(min(max_messages, self.message_queue.qsize())):
            try:
                msg = self.message_queue.get_nowait()
                messages.append(msg)
            except queue.Empty:
                break

        print(f"  Received: {len(messages)} messages")

        # ACK simulation
        if messages:
            print(f"  Sending ACK...")
            time.sleep(0.1)

        return messages

    def subscribe(self, callback: Callable):
        """Streaming subscription (asynchronous)"""
        print(f"\n[GCP Pub/Sub] Streaming subscription started")

        def streaming_loop():
            while True:
                try:
                    msg = self.message_queue.get(timeout=1)
                    callback(msg, {})
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Subscription error: {e}")
                    break

        thread = threading.Thread(target=streaming_loop, daemon=True)
        thread.start()

        return thread


# ==============================================================================
# MQTT Message Format
# ==============================================================================

# Why: Standardizing message formats (telemetry, event, command, response)
# across all devices ensures that cloud-side rules, analytics, and dashboards
# can parse any device's messages without per-device customization.
class MQTTMessageFormat:
    """MQTT Message Format Standard"""

    @staticmethod
    def create_telemetry(device_id: str, sensor_data: Dict) -> Dict:
        """Create telemetry message"""
        return {
            "device_id": device_id,
            "message_type": "telemetry",
            "data": sensor_data,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }

    @staticmethod
    def create_event(device_id: str, event_type: str, event_data: Dict) -> Dict:
        """Create event message"""
        return {
            "device_id": device_id,
            "message_type": "event",
            "event_type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }

    @staticmethod
    def create_command(device_id: str, command: str, parameters: Dict) -> Dict:
        """Create command message"""
        return {
            "device_id": device_id,
            "message_type": "command",
            "command": command,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "command_id": str(uuid.uuid4()),
            "version": "1.0"
        }

    @staticmethod
    def create_response(device_id: str, command_id: str, status: str, result: Dict) -> Dict:
        """Create command response message"""
        return {
            "device_id": device_id,
            "message_type": "response",
            "command_id": command_id,
            "status": status,  # "success", "error", "timeout"
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }


# ==============================================================================
# Device Provisioning
# ==============================================================================

# Why: Provisioning bundles identity creation (Thing), credential issuance
# (X.509 certificate), and policy attachment into one atomic workflow.
# Automating this avoids the manual, error-prone process of copying certs
# onto each device, which doesn't scale beyond a handful of units.
class DeviceProvisioning:
    """Device Provisioning (Simulation)"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.provisioned_devices: Dict[str, DeviceInfo] = {}

        print(f"\n[Provisioning] Initialized ({provider.value})")

    def provision_device(self, device_id: str, device_type: str, location: str) -> DeviceInfo:
        """Provision device"""
        print(f"\n[Provisioning] Device registration started")
        print(f"  - ID: {device_id}")
        print(f"  - Type: {device_type}")
        print(f"  - Location: {location}")

        # Step 1: Create device
        print("\n  [1/4] Creating device...")
        time.sleep(0.3)

        # Step 2: Generate certificate
        print("  [2/4] Generating certificate...")
        cert_arn = f"arn:aws:iot:region:account:cert/{uuid.uuid4()}"
        print(f"    Certificate ARN: {cert_arn}")
        time.sleep(0.3)

        # Step 3: Attach policy
        print("  [3/4] Attaching policy...")
        print("    Policy: IoTDevicePolicy")
        time.sleep(0.3)

        # Step 4: Create and attach Thing
        print("  [4/4] Creating and attaching Thing...")
        time.sleep(0.3)

        device_info = DeviceInfo(
            device_id=device_id,
            device_type=device_type,
            location=location,
            firmware_version="1.0.0",
            registered_at=datetime.now()
        )

        self.provisioned_devices[device_id] = device_info

        print("\nProvisioning complete!")
        print(f"  Certificate generated: certs/{device_id}.cert.pem")
        print(f"  Private key generated: certs/{device_id}.private.key\n")

        return device_info

    def deprovision_device(self, device_id: str):
        """Deprovision device"""
        if device_id not in self.provisioned_devices:
            raise ValueError(f"Device not found: {device_id}")

        print(f"\n[Provisioning] Device deprovisioning: {device_id}")

        # Deactivate certificate
        print("  - Deactivating certificate...")
        time.sleep(0.2)

        # Delete Thing
        print("  - Deleting Thing...")
        time.sleep(0.2)

        del self.provisioned_devices[device_id]
        print("Deprovisioning complete\n")


# ==============================================================================
# Integrated IoT Client
# ==============================================================================

class CloudIoTClient:
    """Integrated Cloud IoT Client"""

    def __init__(self, provider: CloudProvider, config: Dict):
        self.provider = provider
        self.config = config

        print("\n" + "="*60)
        print(f"Cloud IoT Client Initialized ({provider.value})")
        print("="*60)

        if provider == CloudProvider.AWS_IOT:
            self.client = SimulatedAWSIoTClient(
                endpoint=config.get('endpoint', 'simulated-endpoint.iot.region.amazonaws.com'),
                cert_path=config.get('cert_path', 'certs/device.cert.pem'),
                key_path=config.get('key_path', 'certs/device.private.key'),
                ca_path=config.get('ca_path', 'certs/root-CA.crt'),
                client_id=config.get('client_id', 'device-001')
            )

        elif provider == CloudProvider.GCP_PUBSUB:
            self.publisher = SimulatedGCPPubSubPublisher(
                project_id=config.get('project_id', 'my-iot-project'),
                topic_id=config.get('topic_id', 'iot-telemetry')
            )
            self.subscriber = SimulatedGCPPubSubSubscriber(
                project_id=config.get('project_id', 'my-iot-project'),
                subscription_id=config.get('subscription_id', 'iot-telemetry-sub')
            )

        self.message_stats = {
            "published": 0,
            "received": 0,
            "errors": 0
        }

    def connect(self):
        """Connect"""
        if self.provider == CloudProvider.AWS_IOT:
            self.client.connect()

    def disconnect(self):
        """Disconnect"""
        if self.provider == CloudProvider.AWS_IOT:
            self.client.disconnect()

    def publish_telemetry(self, device_id: str, sensor_data: Dict):
        """Publish telemetry"""
        message = MQTTMessageFormat.create_telemetry(device_id, sensor_data)

        if self.provider == CloudProvider.AWS_IOT:
            topic = f"device/{device_id}/telemetry"
            self.client.publish(topic, message)
        elif self.provider == CloudProvider.GCP_PUBSUB:
            self.publisher.publish(message, device_id=device_id, message_type="telemetry")

        self.message_stats["published"] += 1

    def subscribe_commands(self, device_id: str, callback: Callable):
        """Subscribe to commands"""
        def command_handler(topic: str, payload: Dict):
            print(f"\n[Command received] {topic}")
            print(f"  Command: {payload.get('command')}")
            print(f"  Parameters: {payload.get('parameters')}")

            # Execute callback
            callback(payload)

            # Send response
            response = MQTTMessageFormat.create_response(
                device_id=device_id,
                command_id=payload.get('command_id'),
                status="success",
                result={"executed": True}
            )

            response_topic = f"device/{device_id}/response"
            self.client.publish(response_topic, response)

        if self.provider == CloudProvider.AWS_IOT:
            topic = f"device/{device_id}/command"
            self.client.subscribe(topic, command_handler)

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return self.message_stats.copy()


# ==============================================================================
# Sensor Data Simulator
# ==============================================================================

class SensorSimulator:
    """Sensor Data Simulator"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.base_temp = 25.0
        self.base_humidity = 60.0
        self.base_pressure = 1013.25
        self.battery_level = 100.0

    def generate_reading(self) -> TelemetryData:
        """Generate sensor reading"""
        # Add random variation
        temp = self.base_temp + random.uniform(-2, 2)
        humidity = self.base_humidity + random.uniform(-5, 5)
        pressure = self.base_pressure + random.uniform(-2, 2)

        # Battery drain
        self.battery_level = max(0, self.battery_level - random.uniform(0.01, 0.05))

        return TelemetryData(
            device_id=self.device_id,
            temperature=round(temp, 2),
            humidity=round(humidity, 2),
            pressure=round(pressure, 2),
            battery_level=round(self.battery_level, 2),
            timestamp=datetime.now()
        )


# ==============================================================================
# Demo Scenarios
# ==============================================================================

def demo_aws_iot():
    """AWS IoT Core Demo"""
    print("\n" + "="*60)
    print("AWS IoT Core Demo")
    print("="*60)

    # Provisioning
    provisioning = DeviceProvisioning(CloudProvider.AWS_IOT)
    device_info = provisioning.provision_device(
        device_id="raspberry-pi-001",
        device_type="sensor-hub",
        location="Seoul, Korea"
    )

    # Create client
    config = {
        'endpoint': 'a1b2c3d4e5f6g7.iot.ap-northeast-2.amazonaws.com',
        'cert_path': f'certs/{device_info.device_id}.cert.pem',
        'key_path': f'certs/{device_info.device_id}.private.key',
        'ca_path': 'certs/AmazonRootCA1.pem',
        'client_id': device_info.device_id
    }

    client = CloudIoTClient(CloudProvider.AWS_IOT, config)
    client.connect()

    # Device manager
    device_manager = AWSIoTDeviceManager(client.client)
    device_manager.register_device(device_info)

    # Sensor simulator
    sensor = SensorSimulator(device_info.device_id)

    # Subscribe to commands
    def on_command(payload: Dict):
        command = payload.get('command')
        print(f"\nExecuting command: {command}")

    client.subscribe_commands(device_info.device_id, on_command)

    # Publish telemetry
    print("\n" + "-"*60)
    print("Telemetry publishing started")
    print("-"*60)

    for i in range(5):
        print(f"\n[{i+1}/5] Publishing sensor data")

        # Sensor reading
        data = sensor.generate_reading()
        print(f"  Temperature: {data.temperature}C")
        print(f"  Humidity: {data.humidity}%")
        print(f"  Pressure: {data.pressure} hPa")
        print(f"  Battery: {data.battery_level}%")

        # Publish
        device_manager.publish_telemetry(data)

        # Update Device Shadow
        if i % 2 == 0:
            shadow_state = {
                "temperature": data.temperature,
                "humidity": data.humidity,
                "battery": data.battery_level
            }
            device_manager.update_device_shadow(shadow_state)

        time.sleep(2)

    # Statistics
    print("\n" + "="*60)
    print("AWS IoT Demo Complete")
    print("="*60)
    stats = client.get_statistics()
    print(f"Published messages: {stats['published']}")

    client.disconnect()


def demo_gcp_pubsub():
    """GCP Pub/Sub Demo"""
    print("\n" + "="*60)
    print("GCP Pub/Sub Demo")
    print("="*60)

    # Create client
    config = {
        'project_id': 'my-iot-project-123456',
        'topic_id': 'iot-telemetry',
        'subscription_id': 'iot-telemetry-sub'
    }

    client = CloudIoTClient(CloudProvider.GCP_PUBSUB, config)

    # Sensor simulator
    sensor = SensorSimulator("gcp-device-001")

    # Publish telemetry
    print("\n" + "-"*60)
    print("Telemetry publishing started")
    print("-"*60)

    messages = []
    for i in range(5):
        data = sensor.generate_reading()
        print(f"\n[{i+1}/5] Sensor data generated")
        print(f"  Temperature: {data.temperature}C")
        print(f"  Humidity: {data.humidity}%")

        # Individual publish
        sensor_data = {
            "temperature": data.temperature,
            "humidity": data.humidity,
            "pressure": data.pressure,
            "battery_level": data.battery_level
        }

        client.publish_telemetry("gcp-device-001", sensor_data)
        messages.append(data.to_dict())

        time.sleep(1)

    # Batch publish
    print("\n" + "-"*60)
    print("Batch publish")
    print("-"*60)
    client.publisher.publish_batch(messages)

    # Statistics
    print("\n" + "="*60)
    print("GCP Pub/Sub Demo Complete")
    print("="*60)
    stats = client.get_statistics()
    print(f"Published messages: {stats['published']}")


def demo_command_control():
    """Command and Control Demo"""
    print("\n" + "="*60)
    print("Command and Control Demo")
    print("="*60)

    device_id = "smart-device-001"

    # Command creation examples
    print("\n[Cloud -> Device] Sending commands")

    # 1. LED control command
    led_command = MQTTMessageFormat.create_command(
        device_id=device_id,
        command="set_led",
        parameters={"color": "red", "brightness": 80}
    )
    print(f"\n1. LED control command:")
    print(json.dumps(led_command, indent=2))

    # 2. Configuration change command
    config_command = MQTTMessageFormat.create_command(
        device_id=device_id,
        command="update_config",
        parameters={"report_interval": 60, "threshold_temp": 30}
    )
    print(f"\n2. Configuration change command:")
    print(json.dumps(config_command, indent=2))

    # 3. Firmware update command
    firmware_command = MQTTMessageFormat.create_command(
        device_id=device_id,
        command="update_firmware",
        parameters={"version": "2.0.0", "url": "https://example.com/firmware.bin"}
    )
    print(f"\n3. Firmware update command:")
    print(json.dumps(firmware_command, indent=2))

    # Response example
    print("\n[Device -> Cloud] Command response")

    response = MQTTMessageFormat.create_response(
        device_id=device_id,
        command_id=led_command["command_id"],
        status="success",
        result={"led_state": "on", "color": "red", "brightness": 80}
    )
    print(json.dumps(response, indent=2))


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main function"""
    print("Cloud IoT Integration - Simulation Mode")
    print("="*60)
    print("This program operates as a simulation without actual cloud accounts.")
    print()

    # Menu
    print("Demo scenarios:")
    print("  1. AWS IoT Core")
    print("  2. GCP Pub/Sub")
    print("  3. Command and Control")
    print("  4. Run all")
    print()

    choice = input("Select (1-4, default=4): ").strip() or "4"

    if choice == "1":
        demo_aws_iot()
    elif choice == "2":
        demo_gcp_pubsub()
    elif choice == "3":
        demo_command_control()
    elif choice == "4":
        demo_aws_iot()
        time.sleep(2)
        demo_gcp_pubsub()
        time.sleep(2)
        demo_command_control()
    else:
        print("Invalid selection")


if __name__ == "__main__":
    main()
