"""
Exercises for Lesson 06: MQTT Protocol
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates MQTT pub/sub messaging without requiring a real Mosquitto broker.

On a real Raspberry Pi with Mosquitto installed:
    pip install paho-mqtt
    sudo apt install mosquitto mosquitto-clients
Replace SimulatedMQTTBroker with actual paho.mqtt.client connections.
"""

import time
import json
import csv
import os
import random
import threading
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Simulated MQTT Broker
# ---------------------------------------------------------------------------

class SimulatedMQTTBroker:
    """In-process MQTT broker simulation for exercise purposes.

    A real MQTT broker (Mosquitto, HiveMQ, EMQX) manages:
    - Client connections with keep-alive and session state
    - Topic-based message routing with wildcard matching
    - QoS delivery guarantees (0: at-most-once, 1: at-least-once, 2: exactly-once)
    - Retained messages (last message per topic, delivered to new subscribers)
    - Last Will and Testament (LWT) messages for disconnect detection
    """

    def __init__(self):
        self._subscriptions = defaultdict(list)  # topic_pattern -> [callbacks]
        self._retained = {}  # topic -> (payload, qos)
        self._lwt = {}  # client_id -> (topic, payload, qos)
        self._connected_clients = set()
        self._lock = threading.Lock()

    def connect(self, client_id, lwt_topic=None, lwt_payload=None, lwt_qos=0):
        """Register a client connection."""
        with self._lock:
            self._connected_clients.add(client_id)
            if lwt_topic:
                self._lwt[client_id] = (lwt_topic, lwt_payload, lwt_qos)

    def disconnect(self, client_id, clean=True):
        """Disconnect a client. If not clean, publish LWT."""
        with self._lock:
            self._connected_clients.discard(client_id)
            if not clean and client_id in self._lwt:
                topic, payload, qos = self._lwt[client_id]
                # Publish LWT as retained so future subscribers see it
                self._do_publish(topic, payload, qos, retain=True)
            self._lwt.pop(client_id, None)

    def subscribe(self, topic_pattern, callback):
        """Subscribe to a topic pattern (supports + and # wildcards)."""
        with self._lock:
            self._subscriptions[topic_pattern].append(callback)

            # Deliver retained messages matching this pattern
            for topic, (payload, qos) in self._retained.items():
                if self._topic_matches(topic_pattern, topic):
                    callback(topic, payload, qos)

    def publish(self, topic, payload, qos=0, retain=False):
        """Publish a message to a topic."""
        with self._lock:
            self._do_publish(topic, payload, qos, retain)

    def _do_publish(self, topic, payload, qos, retain):
        """Internal publish (called under lock)."""
        if retain:
            self._retained[topic] = (payload, qos)

        # Route to matching subscribers
        for pattern, callbacks in self._subscriptions.items():
            if self._topic_matches(pattern, topic):
                for cb in callbacks:
                    cb(topic, payload, qos)

    @staticmethod
    def _topic_matches(pattern, topic):
        """Match MQTT topic pattern against a concrete topic.

        Wildcards:
        - '+' matches exactly one level: 'home/+/temp' matches 'home/room1/temp'
        - '#' matches zero or more levels: 'home/#' matches 'home/room1/temp'
        """
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for i, p in enumerate(pattern_parts):
            if p == "#":
                return True  # # matches everything from here
            if i >= len(topic_parts):
                return False
            if p == "+":
                continue  # + matches any single level
            if p != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)


# Shared broker instance
broker = SimulatedMQTTBroker()


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: Topic Hierarchy Design and Wildcard Subscriptions ===
# Problem: Design topic hierarchy for a 3-floor, 5-room smart building.

def exercise_1():
    """Solution: MQTT topic hierarchy for a smart building."""

    print("  MQTT Topic Hierarchy for Smart Building\n")

    # Part 1 & 2: Topic structure and concrete paths
    topic_examples = [
        "building/floor1/room101/temperature",
        "building/floor1/room101/humidity",
        "building/floor1/room101/motion",
        "building/floor1/room101/light",
        "building/floor1/room101/hvac/command",
        "building/floor1/room101/hvac/status",
        "building/floor2/room201/temperature",
        "building/floor2/room202/humidity",
        "building/floor3/room301/motion",
        "building/floor3/room305/light",
    ]

    print("    Part 1-2: Concrete topic paths (10 examples):\n")
    for t in topic_examples:
        print(f"      {t}")

    # Part 3: Wildcard subscriptions
    wildcards = {
        "(a) All temperature from floor 2": "building/floor2/+/temperature",
        "(b) All sensors from room 101": "building/floor1/room101/#",
        "(c) Motion alerts from every room": "building/+/+/motion",
    }

    print("\n    Part 3: Wildcard subscription patterns:\n")
    for desc, pattern in wildcards.items():
        print(f"      {desc}")
        print(f"        Pattern: {pattern}")

        # Test which of our example topics match
        matches = [t for t in topic_examples if broker._topic_matches(pattern, t)]
        print(f"        Matches: {matches}\n")

    # Part 4: Demonstrate with simulated broker
    print("    Part 4: Live broker test\n")

    local_broker = SimulatedMQTTBroker()
    received = []

    def on_message(topic, payload, qos):
        received.append((topic, payload))
        print(f"      Received: topic={topic}, payload={payload}")

    # Subscribe with wildcard
    local_broker.subscribe("building/floor2/+/temperature", on_message)

    # Publish test messages
    local_broker.publish("building/floor2/room201/temperature", "22.5")
    local_broker.publish("building/floor2/room202/temperature", "23.1")
    local_broker.publish("building/floor1/room101/temperature", "21.0")  # Should NOT match
    local_broker.publish("building/floor2/room203/humidity", "55")  # Should NOT match

    print(f"\n      Total received (should be 2): {len(received)}")


# === Exercise 2: QoS Level Comparison ===
# Problem: Send 100 messages at QoS 0, 1, 2; simulate packet loss; compare.

def exercise_2():
    """Solution: QoS level comparison experiment."""

    print("  QoS Level Comparison Experiment\n")

    local_broker = SimulatedMQTTBroker()
    results = {0: [], 1: [], 2: []}

    def make_counter(qos_level):
        def callback(topic, payload, qos):
            # Simulate packet loss: QoS 0 loses ~10%, QoS 1 loses ~2%, QoS 2 loses 0%
            loss_rates = {0: 0.10, 1: 0.02, 2: 0.0}
            if random.random() >= loss_rates[qos_level]:
                results[qos_level].append(payload)
        return callback

    # Subscribe one counter per QoS level
    for qos in [0, 1, 2]:
        local_broker.subscribe(f"test/qos{qos}", make_counter(qos))

    # Publish 100 messages per QoS level
    for i in range(100):
        for qos in [0, 1, 2]:
            local_broker.publish(f"test/qos{qos}", f"msg_{i}", qos=qos)

    # Results table
    print(f"    {'QoS Level':<12} {'Sent':>6} {'Received':>10} {'Lost':>6} {'Delivery Rate':>15}")
    print(f"    {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*15}")

    for qos in [0, 1, 2]:
        sent = 100
        received = len(results[qos])
        lost = sent - received
        rate = received / sent * 100
        qos_name = {0: "At most once", 1: "At least once", 2: "Exactly once"}[qos]
        print(f"    QoS {qos} ({qos_name[:13]:>13}) {sent:>6} {received:>10} {lost:>6} {rate:>13.1f}%")

    # Summary
    print("""
    Summary (150 words):

    QoS 0 (at-most-once) provides fire-and-forget delivery. The broker makes
    no attempt to confirm receipt, so messages can be lost due to network issues.
    Use QoS 0 for frequent, non-critical sensor readings (e.g., temperature
    every 5 seconds) where a missed sample is acceptable because the next
    reading will arrive shortly.

    QoS 1 (at-least-once) guarantees delivery via PUBACK acknowledgment,
    but may deliver duplicates if the ACK is lost. Use QoS 1 for important
    events like billing records or device status changes where losing a
    message is unacceptable but receiving it twice can be handled by
    idempotent processing.

    QoS 2 (exactly-once) uses a four-step handshake (PUBLISH, PUBREC,
    PUBREL, PUBCOMP) to guarantee single delivery. Use QoS 2 for commands
    that must execute exactly once, such as financial transactions or
    actuator commands where duplicate execution would cause harm (e.g.,
    dispensing medication or toggling a valve).
    """)


# === Exercise 3: Device Status with LWT and Retained Messages ===
# Problem: Fleet status dashboard using LWT for disconnect detection.

def exercise_3():
    """Solution: Device fleet status dashboard with LWT and retained messages."""

    print("  Device Fleet Status Dashboard\n")

    local_broker = SimulatedMQTTBroker()
    fleet_status = {}
    fleet_lock = threading.Lock()

    def fleet_monitor_callback(topic, payload, qos):
        """Fleet monitor: track device status from wildcard subscription."""
        # Topic format: devices/<client_id>/status
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "devices" and parts[2] == "status":
            device_id = parts[1]
            with fleet_lock:
                fleet_status[device_id] = {
                    "status": payload,
                    "updated": datetime.now().strftime("%H:%M:%S"),
                }

    def print_fleet_table():
        """Print formatted fleet status table."""
        with fleet_lock:
            print(f"\n    {'Device ID':<20} {'Status':<12} {'Last Updated'}")
            print(f"    {'-'*20} {'-'*12} {'-'*12}")
            for did, info in sorted(fleet_status.items()):
                status_marker = "OK" if info["status"] == "online" else "!!"
                print(f"    {did:<20} {info['status']:<12} {info['updated']} [{status_marker}]")

    # Fleet monitor subscribes with wildcard
    local_broker.subscribe("devices/+/status", fleet_monitor_callback)

    # Part 1: Start 3 publishers
    device_ids = ["sensor_alpha", "sensor_beta", "sensor_gamma"]

    print("    Starting 3 device publishers...")
    for did in device_ids:
        # Set LWT (published by broker if client disconnects ungracefully)
        local_broker.connect(
            did,
            lwt_topic=f"devices/{did}/status",
            lwt_payload="offline",
            lwt_qos=1,
        )
        # Publish "online" as retained (new subscribers immediately see status)
        local_broker.publish(f"devices/{did}/status", "online", qos=1, retain=True)
        print(f"      {did}: connected, status=online (retained)")

    print_fleet_table()

    # Part 3: Kill one publisher abruptly (not clean disconnect)
    print(f"\n    Killing sensor_beta abruptly (simulating crash)...")
    local_broker.disconnect("sensor_beta", clean=False)

    print_fleet_table()

    # Part 4: New subscriber joins and immediately receives retained statuses
    print(f"\n    New subscriber joining (should see retained statuses)...")
    late_received = []

    def late_subscriber(topic, payload, qos):
        late_received.append((topic, payload))

    local_broker.subscribe("devices/+/status", late_subscriber)
    print(f"    Late subscriber received {len(late_received)} retained messages:")
    for topic, payload in late_received:
        print(f"      {topic} -> {payload}")

    # Cleanup
    for did in device_ids:
        local_broker.disconnect(did, clean=True)


# === Exercise 4: MQTT-Based Remote GPIO Control ===
# Problem: Control GPIO pins via MQTT topics like gpio/<pin>/command.

def exercise_4():
    """Solution: Remote GPIO control via MQTT pub/sub.

    Topic structure:
    - gpio/<pin>/command  -- subscriber listens for "on", "off", "toggle"
    - gpio/<pin>/state    -- publisher reports current state "1" or "0" (retained)
    """

    print("  MQTT-Based Remote GPIO Control\n")

    local_broker = SimulatedMQTTBroker()

    # Simulated GPIO pin states
    gpio_states = {}

    def gpio_command_handler(topic, payload, qos):
        """Handle GPIO commands received via MQTT.

        On a real Pi, this would call gpiozero LED.on()/off()/toggle()
        and then publish the resulting state.
        """
        parts = topic.split("/")
        if len(parts) != 3 or parts[0] != "gpio" or parts[2] != "command":
            return

        pin = parts[1]
        command = payload.lower()

        # Initialize pin if not seen before
        if pin not in gpio_states:
            gpio_states[pin] = False  # Off by default

        # Apply command
        if command == "on":
            gpio_states[pin] = True
        elif command == "off":
            gpio_states[pin] = False
        elif command == "toggle":
            gpio_states[pin] = not gpio_states[pin]
        else:
            print(f"    [ERROR] Unknown command: {command}")
            return

        # Publish new state as retained
        state_str = "1" if gpio_states[pin] else "0"
        local_broker.publish(f"gpio/{pin}/state", state_str, qos=1, retain=True)
        print(f"    [GPIO] Pin {pin}: command={command} -> state={state_str}")

    # State monitor (to show state publications)
    state_log = []

    def state_monitor(topic, payload, qos):
        state_log.append((topic, payload))

    # Subscribe to commands (wildcard: all pins)
    local_broker.subscribe("gpio/+/command", gpio_command_handler)
    local_broker.subscribe("gpio/+/state", state_monitor)

    # Simulate commands from a remote client
    commands = [
        ("gpio/17/command", "on"),      # Turn LED on pin 17 ON
        ("gpio/17/command", "toggle"),   # Toggle -> OFF
        ("gpio/17/command", "toggle"),   # Toggle -> ON
        ("gpio/22/command", "on"),       # Turn LED on pin 22 ON
        ("gpio/22/command", "off"),      # Turn LED on pin 22 OFF
        ("gpio/17/command", "off"),      # Turn LED on pin 17 OFF
    ]

    print("    Sending commands:\n")
    for topic, payload in commands:
        print(f"    mosquitto_pub -t \"{topic}\" -m \"{payload}\"")
        local_broker.publish(topic, payload, qos=1)
        time.sleep(0.2)

    # Show final states
    print(f"\n    Final GPIO states:")
    for pin, state in sorted(gpio_states.items()):
        print(f"      Pin {pin}: {'ON' if state else 'OFF'}")


# === Exercise 5: Complete IoT Pipeline -- Sensor to Dashboard ===
# Problem: Sensor publisher -> Data logger -> Alert monitor -> Dashboard.

def exercise_5():
    """Solution: End-to-end IoT pipeline with MQTT."""

    print("  Complete IoT Pipeline: Sensor -> Logger -> Alerts -> Dashboard\n")

    local_broker = SimulatedMQTTBroker()
    csv_file = "/tmp/sensor_log.csv"

    # --- Component 1: Sensor Publisher ---
    def publish_sensor_data(device_id, num_readings=8):
        """Simulate DHT11 publishing temperature and humidity."""
        for _ in range(num_readings):
            temp = round(random.uniform(20, 30), 1)
            hum = round(random.uniform(40, 70), 1)
            local_broker.publish(
                f"sensors/{device_id}/temperature", str(temp), qos=1, retain=True
            )
            local_broker.publish(
                f"sensors/{device_id}/humidity", str(hum), qos=1, retain=True
            )
            time.sleep(0.1)

    # --- Component 2: Data Logger ---
    log_entries = []

    def data_logger(topic, payload, qos):
        """Log all sensor readings to CSV."""
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "sensors":
            device_id = parts[1]
            metric = parts[2]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entries.append([ts, device_id, metric, payload])

    # --- Component 3: Alert Monitor ---
    active_alerts = {}

    def alert_monitor(topic, payload, qos):
        """Monitor temperature and publish alerts when > 27C."""
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "sensors" and parts[2] == "temperature":
            device_id = parts[1]
            temp = float(payload)
            if temp > 27.0:
                alert_topic = f"alerts/{device_id}/high_temp"
                alert_payload = json.dumps({
                    "device_id": device_id,
                    "temperature": temp,
                    "threshold": 27.0,
                    "timestamp": datetime.now().isoformat(),
                })
                local_broker.publish(alert_topic, alert_payload, qos=2, retain=True)
                active_alerts[device_id] = temp

    # --- Component 4: Dashboard ---
    dashboard_data = defaultdict(dict)
    dashboard_alerts = {}

    def dashboard_sensor_cb(topic, payload, qos):
        """Track latest sensor value per device per metric."""
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "sensors":
            device_id = parts[1]
            metric = parts[2]
            dashboard_data[device_id][metric] = payload

    def dashboard_alert_cb(topic, payload, qos):
        """Track active alerts."""
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "alerts":
            device_id = parts[1]
            dashboard_alerts[device_id] = payload

    # Wire up all subscribers
    local_broker.subscribe("sensors/#", data_logger)
    local_broker.subscribe("sensors/+/temperature", alert_monitor)
    local_broker.subscribe("sensors/#", dashboard_sensor_cb)
    local_broker.subscribe("alerts/#", dashboard_alert_cb)

    # Run two sensor publishers
    print("    Publishing sensor data from 2 devices...\n")
    publish_sensor_data("RPi_001", num_readings=8)
    publish_sensor_data("RPi_002", num_readings=8)

    # Write CSV log
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "device_id", "metric", "value"])
        writer.writerows(log_entries)

    # Print dashboard
    print(f"    --- Dashboard (latest values) ---\n")
    print(f"    {'Device':<15} {'Temperature':>12} {'Humidity':>10} {'Alert'}")
    print(f"    {'-'*15} {'-'*12} {'-'*10} {'-'*20}")

    for device_id in sorted(dashboard_data.keys()):
        metrics = dashboard_data[device_id]
        temp = metrics.get("temperature", "N/A")
        hum = metrics.get("humidity", "N/A")
        alert = "HIGH TEMP" if device_id in dashboard_alerts else "None"
        print(f"    {device_id:<15} {temp:>10} C {hum:>8}% {alert}")

    print(f"\n    Data logger: {len(log_entries)} entries written to {csv_file}")
    print(f"    Active alerts: {len(active_alerts)}")

    # Show a few log entries
    print(f"\n    --- Sample Log Entries (last 5) ---")
    for entry in log_entries[-5:]:
        print(f"      {entry}")

    if os.path.exists(csv_file):
        os.remove(csv_file)


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 06: MQTT Protocol - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Topic Hierarchy Design")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: QoS Level Comparison")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: Device Status with LWT")
    print("-" * 50)
    exercise_3()

    print("\n\n>>> Exercise 4: MQTT-Based Remote GPIO Control")
    print("-" * 50)
    exercise_4()

    print("\n\n>>> Exercise 5: Complete IoT Pipeline")
    print("-" * 50)
    exercise_5()
