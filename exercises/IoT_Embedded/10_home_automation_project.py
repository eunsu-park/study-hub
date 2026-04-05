"""
Exercises for Lesson 10: Home Automation Project
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates a smart home system with gateway, sensors, actuators,
MQTT messaging, and a web dashboard using Python data structures.

On a real Raspberry Pi:
    pip install paho-mqtt     # MQTT client
    pip install flask         # Web dashboard
    pip install RPi.GPIO      # GPIO control
    pip install adafruit-dht  # DHT sensor library
"""

import time
import json
import random
import threading
from datetime import datetime, timedelta
from collections import defaultdict


# ---------------------------------------------------------------------------
# Simulated Smart Home Components
# ---------------------------------------------------------------------------

class SimulatedGPIO:
    """Simulate Raspberry Pi GPIO for relay and sensor control.

    Real GPIO usage:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
    """

    _pins = {}

    @classmethod
    def setup(cls, pin, mode):
        cls._pins[pin] = {"mode": mode, "state": False}

    @classmethod
    def output(cls, pin, state):
        if pin in cls._pins:
            cls._pins[pin]["state"] = bool(state)

    @classmethod
    def input(cls, pin):
        return cls._pins.get(pin, {}).get("state", False)


class SmartDevice:
    """Represent a controllable smart home device."""

    def __init__(self, device_id, device_type, gpio_pin, location):
        self.device_id = device_id
        self.device_type = device_type
        self.gpio_pin = gpio_pin
        self.location = location
        self.state = False
        self.brightness = 100  # For dimmable lights (0-100)
        self.last_changed = datetime.now()

        SimulatedGPIO.setup(gpio_pin, "OUT")

    def turn_on(self):
        self.state = True
        SimulatedGPIO.output(self.gpio_pin, True)
        self.last_changed = datetime.now()
        return self.status()

    def turn_off(self):
        self.state = False
        SimulatedGPIO.output(self.gpio_pin, False)
        self.last_changed = datetime.now()
        return self.status()

    def set_brightness(self, level):
        """Set brightness for dimmable lights (0-100)."""
        self.brightness = max(0, min(100, level))
        if self.brightness == 0:
            self.turn_off()
        elif not self.state:
            self.turn_on()
        self.last_changed = datetime.now()
        return self.status()

    def status(self):
        return {
            "device_id": self.device_id,
            "type": self.device_type,
            "location": self.location,
            "state": "ON" if self.state else "OFF",
            "brightness": self.brightness,
            "gpio_pin": self.gpio_pin,
            "last_changed": self.last_changed.isoformat(),
        }


class SensorMonitor:
    """Threaded sensor monitoring class.

    Reads temperature and humidity at regular intervals and stores
    the history. In production, reads from a real DHT22 sensor via
    the adafruit-dht library.

    Real sensor reading:
        import adafruit_dht
        import board
        dht = adafruit_dht.DHT22(board.D4)
        temperature = dht.temperature
        humidity = dht.humidity
    """

    def __init__(self, sensor_id, location, interval_sec=5):
        self.sensor_id = sensor_id
        self.location = location
        self.interval_sec = interval_sec
        self.history = []
        self._running = False

    def read_once(self):
        """Simulate a single sensor reading."""
        reading = {
            "sensor_id": self.sensor_id,
            "location": self.location,
            "temperature": round(random.uniform(19.0, 27.0), 1),
            "humidity": round(random.uniform(35.0, 65.0), 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(reading)
        return reading

    def start(self, num_readings=5):
        """Collect a fixed number of readings (simulates threaded loop)."""
        self._running = True
        for _ in range(num_readings):
            if not self._running:
                break
            self.read_once()
        self._running = False

    def stop(self):
        self._running = False

    def get_latest(self):
        return self.history[-1] if self.history else None

    def get_average(self, last_n=10):
        recent = self.history[-last_n:]
        if not recent:
            return None
        return {
            "avg_temperature": round(
                sum(r["temperature"] for r in recent) / len(recent), 1),
            "avg_humidity": round(
                sum(r["humidity"] for r in recent) / len(recent), 1),
            "sample_count": len(recent),
        }


class MQTTCommandBus:
    """Simulated MQTT command and status bus for device coordination.

    Real MQTT:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect("localhost", 1883)
        client.subscribe("home/+/command")
        client.publish("home/living_room/status", json.dumps(status))
    """

    def __init__(self):
        self._subscriptions = defaultdict(list)
        self._message_log = []

    def subscribe(self, topic, callback):
        self._subscriptions[topic].append(callback)

    def publish(self, topic, payload):
        message = {
            "topic": topic,
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
        }
        self._message_log.append(message)
        # Dispatch to subscribers (simple topic matching)
        for sub_topic, callbacks in self._subscriptions.items():
            if self._topic_matches(sub_topic, topic):
                for cb in callbacks:
                    cb(topic, payload)

    @staticmethod
    def _topic_matches(pattern, topic):
        """Simple MQTT wildcard matching (+ for single level)."""
        p_parts = pattern.split("/")
        t_parts = topic.split("/")
        if len(p_parts) != len(t_parts):
            return False
        return all(p == "+" or p == t for p, t in zip(p_parts, t_parts))

    def get_log(self, limit=10):
        return self._message_log[-limit:]


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: Smart Lighting System ===
# Problem: Build a lighting control system with relay modules and GPIO.
# Support on/off, brightness, and scheduling.

def exercise_1():
    """Solution: Smart lighting control with GPIO relays and scheduling."""

    print("  Smart Lighting System\n")

    # Create devices
    devices = {
        "light_living": SmartDevice("light_living", "light", 17, "Living Room"),
        "light_kitchen": SmartDevice("light_kitchen", "light", 27, "Kitchen"),
        "light_bedroom": SmartDevice("light_bedroom", "light", 22, "Bedroom"),
        "fan_living": SmartDevice("fan_living", "fan", 23, "Living Room"),
    }

    # Part 1: Basic control
    print("    --- Part 1: Device Control ---\n")

    devices["light_living"].turn_on()
    devices["light_kitchen"].set_brightness(60)
    devices["fan_living"].turn_on()

    print(f"    {'Device':<20} {'Location':<15} {'State':<6} {'Brightness':>10}")
    print(f"    {'-'*20} {'-'*15} {'-'*6} {'-'*10}")
    for dev in devices.values():
        s = dev.status()
        print(f"    {s['device_id']:<20} {s['location']:<15} {s['state']:<6} "
              f"{s['brightness']:>9}%")

    # Part 2: Scene presets
    print("\n    --- Part 2: Scene Presets ---\n")

    scenes = {
        "movie": {"light_living": 20, "light_kitchen": 0, "light_bedroom": 0},
        "cooking": {"light_living": 50, "light_kitchen": 100, "light_bedroom": 0},
        "goodnight": {"light_living": 0, "light_kitchen": 0, "light_bedroom": 10},
    }

    def apply_scene(scene_name):
        if scene_name not in scenes:
            print(f"    Unknown scene: {scene_name}")
            return
        print(f"    Applying scene: '{scene_name}'")
        for device_id, brightness in scenes[scene_name].items():
            if device_id in devices:
                devices[device_id].set_brightness(brightness)
                state = "ON" if brightness > 0 else "OFF"
                print(f"      {device_id}: {state} (brightness={brightness}%)")

    apply_scene("movie")

    # Part 3: Simple schedule
    print("\n    --- Part 3: Scheduling ---\n")

    schedule = [
        {"time": "07:00", "scene": "cooking", "description": "Morning routine"},
        {"time": "19:00", "scene": "movie", "description": "Evening relaxation"},
        {"time": "23:00", "scene": "goodnight", "description": "Bedtime"},
    ]

    print(f"    {'Time':<8} {'Scene':<12} {'Description'}")
    print(f"    {'-'*8} {'-'*12} {'-'*25}")
    for entry in schedule:
        print(f"    {entry['time']:<8} {entry['scene']:<12} {entry['description']}")


# === Exercise 2: Environmental Monitoring ===
# Problem: Monitor temperature and humidity across multiple rooms.
# Trigger alerts when thresholds are exceeded.

def exercise_2():
    """Solution: Multi-room environmental monitoring with alerts."""

    print("  Environmental Monitoring System\n")

    # Set up sensors in multiple rooms
    sensors = {
        "living_room": SensorMonitor("dht22_01", "Living Room"),
        "kitchen": SensorMonitor("dht22_02", "Kitchen"),
        "bedroom": SensorMonitor("dht22_03", "Bedroom"),
    }

    # Collect readings
    print("    --- Part 1: Sensor Readings ---\n")
    for name, sensor in sensors.items():
        sensor.start(num_readings=10)

    print(f"    {'Room':<15} {'Temp (C)':>9} {'Humidity (%)':>13} {'Avg Temp':>9} {'Avg Hum':>9}")
    print(f"    {'-'*15} {'-'*9} {'-'*13} {'-'*9} {'-'*9}")

    for name, sensor in sensors.items():
        latest = sensor.get_latest()
        avg = sensor.get_average()
        print(f"    {name:<15} {latest['temperature']:>8.1f} {latest['humidity']:>12.1f} "
              f"{avg['avg_temperature']:>8.1f} {avg['avg_humidity']:>8.1f}")

    # Part 2: Threshold alerts
    print("\n    --- Part 2: Threshold Alerts ---\n")

    thresholds = {
        "temperature_high": 26.0,
        "temperature_low": 18.0,
        "humidity_high": 60.0,
        "humidity_low": 30.0,
    }

    alerts = []
    for name, sensor in sensors.items():
        for reading in sensor.history:
            if reading["temperature"] > thresholds["temperature_high"]:
                alerts.append({
                    "room": name,
                    "type": "HIGH_TEMP",
                    "value": reading["temperature"],
                    "threshold": thresholds["temperature_high"],
                    "timestamp": reading["timestamp"],
                })
            if reading["humidity"] > thresholds["humidity_high"]:
                alerts.append({
                    "room": name,
                    "type": "HIGH_HUMIDITY",
                    "value": reading["humidity"],
                    "threshold": thresholds["humidity_high"],
                    "timestamp": reading["timestamp"],
                })

    if alerts:
        print(f"    Found {len(alerts)} alert(s):\n")
        for a in alerts[:5]:
            print(f"      [{a['type']}] {a['room']}: "
                  f"value={a['value']}, threshold={a['threshold']}")
    else:
        print("    No alerts triggered (all readings within thresholds).")

    # Part 3: History summary
    print("\n    --- Part 3: Reading History ---\n")
    for name, sensor in sensors.items():
        temps = [r["temperature"] for r in sensor.history]
        print(f"    {name}: {len(temps)} readings, "
              f"range=[{min(temps):.1f}, {max(temps):.1f}]C")


# === Exercise 3: MQTT-Based Smart Home Gateway ===
# Problem: Integrate devices and sensors into a unified MQTT gateway
# with command handling and a text-based dashboard.

def exercise_3():
    """Solution: Smart home gateway with MQTT command bus and dashboard."""

    print("  Smart Home Gateway\n")

    bus = MQTTCommandBus()

    # Register devices
    devices = {
        "light_living": SmartDevice("light_living", "light", 17, "Living Room"),
        "light_kitchen": SmartDevice("light_kitchen", "light", 27, "Kitchen"),
        "fan_bedroom": SmartDevice("fan_bedroom", "fan", 22, "Bedroom"),
    }

    sensors = {
        "living_room": SensorMonitor("dht22_01", "Living Room"),
        "kitchen": SensorMonitor("dht22_02", "Kitchen"),
    }

    # Part 1: Command handler
    print("    --- Part 1: MQTT Command Handler ---\n")

    def handle_command(topic, payload):
        """Process incoming device commands via MQTT."""
        parts = topic.split("/")
        if len(parts) >= 3:
            device_id = parts[1]
            if device_id in devices:
                action = payload.get("action", "")
                if action == "on":
                    devices[device_id].turn_on()
                elif action == "off":
                    devices[device_id].turn_off()
                elif action == "brightness":
                    devices[device_id].set_brightness(payload.get("level", 100))

                # Publish status update
                status = devices[device_id].status()
                bus.publish(f"home/{device_id}/status", status)
                print(f"    Command: {device_id} -> {action} | "
                      f"State: {status['state']}")

    bus.subscribe("home/+/command", handle_command)

    # Send commands
    bus.publish("home/light_living/command", {"action": "on"})
    bus.publish("home/light_kitchen/command",
                {"action": "brightness", "level": 75})
    bus.publish("home/fan_bedroom/command", {"action": "on"})
    bus.publish("home/light_living/command", {"action": "off"})

    # Part 2: Sensor data publishing
    print("\n    --- Part 2: Sensor Data Publishing ---\n")

    for name, sensor in sensors.items():
        sensor.start(num_readings=5)
        latest = sensor.get_latest()
        bus.publish(f"home/{name}/sensor", latest)
        print(f"    Sensor {name}: temp={latest['temperature']}C, "
              f"humidity={latest['humidity']}%")

    # Part 3: Dashboard
    print("\n    --- Part 3: Gateway Dashboard ---\n")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"    ╔{'═' * 56}╗")
    print(f"    ║{'Smart Home Gateway Dashboard':^56}║")
    print(f"    ║{'Updated: ' + now:^56}║")
    print(f"    ╠{'═' * 56}╣")
    print(f"    ║  Devices                                              ║")
    print(f"    ╟{'─' * 56}╢")

    for dev in devices.values():
        s = dev.status()
        indicator = "●" if s["state"] == "ON" else "○"
        print(f"    ║  {indicator} {s['device_id']:<22} {s['location']:<14} "
              f"{s['state']:<5} {s['brightness']:>3}% ║")

    print(f"    ╟{'─' * 56}╢")
    print(f"    ║  Sensors                                              ║")
    print(f"    ╟{'─' * 56}╢")

    for name, sensor in sensors.items():
        latest = sensor.get_latest()
        if latest:
            print(f"    ║  {name:<18} "
                  f"Temp: {latest['temperature']:>5.1f}C  "
                  f"Hum: {latest['humidity']:>5.1f}%    ║")

    print(f"    ╟{'─' * 56}╢")
    log = bus.get_log(limit=4)
    print(f"    ║  Recent MQTT ({len(log)} messages)                        ║")
    for msg in log[-3:]:
        topic_short = msg["topic"][-30:]
        print(f"    ║    {topic_short:<52}║")

    print(f"    ╚{'═' * 56}╝")

    # Show message log summary
    full_log = bus.get_log(limit=100)
    print(f"\n    Total MQTT messages: {len(full_log)}")


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 10: Home Automation Project - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Smart Lighting System")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Environmental Monitoring")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: MQTT-Based Smart Home Gateway")
    print("-" * 50)
    exercise_3()
