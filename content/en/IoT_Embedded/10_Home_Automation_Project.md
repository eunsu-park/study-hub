# 10. Home Automation Project

**Previous**: [Edge AI - ONNX Runtime](./09_Edge_AI_ONNX.md) | **Next**: [Image Analysis Project](./11_Image_Analysis_Project.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a smart home system architecture with gateway, sensors, and actuators
2. Implement lighting control through relay modules and GPIO
3. Monitor environmental conditions with a threaded sensor monitoring class
4. Build an MQTT-based device command and status system
5. Develop a web dashboard for real-time device control and sensor display
6. Integrate all components into a working smart home gateway

---

This project lesson pulls together everything you have learned so far -- GPIO control, sensor reading, MQTT messaging, and Flask web serving -- into a single cohesive system. Home automation is one of the most tangible and rewarding IoT applications, and building it end-to-end demonstrates how individual skills combine to create a real, interactive product.

---

## 1. Smart Home Architecture

### 1.1 System Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Home System Architecture            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    User Interfaces                    │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│   │  │   Web    │  │  Mobile  │  │  Voice   │          │   │
│   │  │Dashboard │  │   App    │  │Assistant │          │   │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │   │
│   └───────┼─────────────┼─────────────┼────────────────┘   │
│           │             │             │                     │
│           └─────────────┼─────────────┘                     │
│                         ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Gateway (Raspberry Pi)                  │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│   │  │   MQTT   │  │ REST API │  │   Data   │          │   │
│   │  │  Broker  │  │  Server  │  │    DB    │          │   │
│   │  └──────────┘  └──────────┘  └──────────┘          │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│        ┌──────────────────┼──────────────────┐              │
│        │                  │                  │              │
│        ▼                  ▼                  ▼              │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│   │ Lighting │      │Temp/Hum  │      │  Motion  │         │
│   │ Control  │      │  Sensor  │      │  Sensor  │         │
│   │ (Relay)  │      │ (DHT11)  │      │  (PIR)   │         │
│   └──────────┘      └──────────┘      └──────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Hardware Configuration

| Component | Model | Role | GPIO |
|-----------|-------|------|------|
| **Gateway** | Raspberry Pi 4 | Central control, MQTT broker | - |
| **Relay Module** | 4-channel relay | Lighting/appliance control | 17, 27, 22, 23 |
| **Temp/Humidity Sensor** | DHT11 | Environment monitoring | 4 |
| **Motion Sensor** | PIR HC-SR501 | Motion detection | 24 |
| **Light Sensor** | Photoresistor | Brightness detection | MCP3008 (SPI) |

### 1.3 Project Structure

```
smart_home/
├── gateway/
│   ├── main.py              # Main application
│   ├── config.py            # Configuration
│   ├── mqtt_handler.py      # MQTT handler
│   ├── device_controller.py # Device control
│   ├── sensor_monitor.py    # Sensor monitoring
│   └── database.py          # Data storage
├── web/
│   ├── app.py               # Flask web server
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS
└── requirements.txt
```

---

## 2. Lighting Control (Relay)

### 2.1 Relay Connection

```
┌─────────────────────────────────────────────────────────────┐
│                    Relay Module Wiring Diagram               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Raspberry Pi              Relay Module (4-channel)         │
│   ┌──────────┐              ┌──────────────┐                │
│   │ 5V (Pin2)│─────────────▶│ VCC          │                │
│   │ GND(Pin6)│─────────────▶│ GND          │                │
│   │GPIO17(11)│─────────────▶│ IN1 (Light1) │                │
│   │GPIO27(13)│─────────────▶│ IN2 (Light2) │                │
│   │GPIO22(15)│─────────────▶│ IN3 (Light3) │                │
│   │GPIO23(16)│─────────────▶│ IN4 (Light4) │                │
│   └──────────┘              └──────────────┘                │
│                                                              │
│   Note: Relay may be Active Low                             │
│         GPIO.LOW = Relay ON, GPIO.HIGH = Relay OFF          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Relay Control Class

```python
#!/usr/bin/env python3
"""Relay lighting control"""

from gpiozero import OutputDevice
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class Light:
    """Light device"""
    id: str
    name: str
    gpio_pin: int
    location: str
    is_on: bool = False

class LightController:
    """Light controller"""

    def __init__(self, config: dict):
        self.lights: Dict[str, Light] = {}
        self.relays: Dict[str, OutputDevice] = {}

        # Light configuration
        for light_config in config.get('lights', []):
            light = Light(**light_config)
            self.lights[light.id] = light

            # Initialize relay (active_high=False: Active Low relay)
            relay = OutputDevice(
                light.gpio_pin,
                active_high=False,
                initial_value=False
            )
            self.relays[light.id] = relay

    def turn_on(self, light_id: str) -> bool:
        """Turn on light"""
        if light_id not in self.lights:
            return False

        self.relays[light_id].on()
        self.lights[light_id].is_on = True
        return True

    def turn_off(self, light_id: str) -> bool:
        """Turn off light"""
        if light_id not in self.lights:
            return False

        self.relays[light_id].off()
        self.lights[light_id].is_on = False
        return True

    def toggle(self, light_id: str) -> bool:
        """Toggle light"""
        if light_id not in self.lights:
            return False

        if self.lights[light_id].is_on:
            return self.turn_off(light_id)
        else:
            return self.turn_on(light_id)

    def get_status(self, light_id: str = None) -> dict:
        """Get status"""
        if light_id:
            light = self.lights.get(light_id)
            if light:
                return {
                    "id": light.id,
                    "name": light.name,
                    "location": light.location,
                    "is_on": light.is_on
                }
            return None

        return {
            "lights": [
                {
                    "id": l.id,
                    "name": l.name,
                    "location": l.location,
                    "is_on": l.is_on
                }
                for l in self.lights.values()
            ]
        }

    def all_off(self):
        """Turn off all lights"""
        for light_id in self.lights:
            self.turn_off(light_id)

    def all_on(self):
        """Turn on all lights"""
        for light_id in self.lights:
            self.turn_on(light_id)

    def cleanup(self):
        """Cleanup"""
        for relay in self.relays.values():
            relay.close()

# Usage example
if __name__ == "__main__":
    config = {
        "lights": [
            {"id": "living_room", "name": "Living Room Light", "gpio_pin": 17, "location": "Living Room"},
            {"id": "bedroom", "name": "Bedroom Light", "gpio_pin": 27, "location": "Bedroom"},
            {"id": "kitchen", "name": "Kitchen Light", "gpio_pin": 22, "location": "Kitchen"},
            {"id": "bathroom", "name": "Bathroom Light", "gpio_pin": 23, "location": "Bathroom"},
        ]
    }

    controller = LightController(config)

    print("Light status:", json.dumps(controller.get_status(), indent=2))

    controller.turn_on("living_room")
    print("Living room light ON")

    controller.toggle("bedroom")
    print("Bedroom light toggled")

    controller.cleanup()
```

---

## 3. Temperature Monitoring

### 3.1 Sensor Monitoring Class

```python
#!/usr/bin/env python3
"""Environment sensor monitoring"""

import time
from datetime import datetime
from dataclasses import dataclass
import threading
import queue

# DHT sensor library
import adafruit_dht
import board

@dataclass
class SensorReading:
    """Sensor data"""
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: datetime

class EnvironmentMonitor:
    """Environment monitoring"""

    def __init__(self, sensor_pin: int = 4, sensor_id: str = "env_01"):
        self.sensor_id = sensor_id
        self.dht = adafruit_dht.DHT11(getattr(board, f"D{sensor_pin}"))

        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None

        # Store recent data
        self.latest_reading = None
        self.readings_history = []
        self.max_history = 1000

    def read_sensor(self) -> SensorReading | None:
        """Read sensor"""
        try:
            temperature = self.dht.temperature
            humidity = self.dht.humidity

            if temperature is not None and humidity is not None:
                reading = SensorReading(
                    sensor_id=self.sensor_id,
                    temperature=temperature,
                    humidity=humidity,
                    timestamp=datetime.now()
                )
                return reading

        except RuntimeError as e:
            # DHT sensor occasionally fails to read (normal)
            pass

        return None

    def _monitor_loop(self, interval: int):
        """Monitoring loop"""
        while self.running:
            reading = self.read_sensor()

            if reading:
                self.latest_reading = reading
                self.readings_history.append(reading)

                # Limit history size
                if len(self.readings_history) > self.max_history:
                    self.readings_history.pop(0)

                # Add to queue (for external subscribers)
                self.data_queue.put(reading)

            time.sleep(interval)

    def start(self, interval: int = 5):
        """Start monitoring"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.thread.start()
        print(f"Environment monitoring started (interval: {interval}s)")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.dht.exit()
        print("Environment monitoring stopped")

    def get_latest(self) -> dict | None:
        """Get latest data"""
        if self.latest_reading:
            return {
                "sensor_id": self.latest_reading.sensor_id,
                "temperature": self.latest_reading.temperature,
                "humidity": self.latest_reading.humidity,
                "timestamp": self.latest_reading.timestamp.isoformat()
            }
        return None

    def get_stats(self) -> dict:
        """Get statistics"""
        if not self.readings_history:
            return {}

        temps = [r.temperature for r in self.readings_history]
        humids = [r.humidity for r in self.readings_history]

        return {
            "count": len(self.readings_history),
            "temperature": {
                "min": min(temps),
                "max": max(temps),
                "avg": sum(temps) / len(temps)
            },
            "humidity": {
                "min": min(humids),
                "max": max(humids),
                "avg": sum(humids) / len(humids)
            }
        }

# Usage example
if __name__ == "__main__":
    monitor = EnvironmentMonitor(sensor_pin=4)
    monitor.start(interval=5)

    try:
        while True:
            latest = monitor.get_latest()
            if latest:
                print(f"Temperature: {latest['temperature']}°C, Humidity: {latest['humidity']}%")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
```

---

## 4. MQTT-based Control

### 4.1 MQTT Handler

```python
#!/usr/bin/env python3
"""MQTT-based smart home control"""

import paho.mqtt.client as mqtt
import json
from datetime import datetime

class SmartHomeMQTT:
    """Smart home MQTT handler"""

    TOPICS = {
        "light_command": "home/+/light/command",
        "light_status": "home/{}/light/status",
        "sensor_data": "home/sensor/{}",
        "motion": "home/motion/{}",
        "system": "home/system/status"
    }

    def __init__(self, light_controller, env_monitor, broker: str = "localhost"):
        self.light_controller = light_controller
        self.env_monitor = env_monitor

        self.client = mqtt.Client(client_id="smart_home_gateway")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # LWT setup
        self.client.will_set(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )

        self.client.connect(broker, 1883)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")

            # Subscribe to topics
            client.subscribe(self.TOPICS["light_command"])

            # Publish online status
            client.publish(
                self.TOPICS["system"],
                json.dumps({"status": "online", "timestamp": datetime.now().isoformat()}),
                qos=1,
                retain=True
            )

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            print(f"Received: {topic} = {payload}")

            # Handle light commands
            if "light/command" in topic:
                self._handle_light_command(topic, payload)

        except json.JSONDecodeError:
            print(f"Invalid JSON: {msg.payload}")
        except Exception as e:
            print(f"Message processing error: {e}")

    def _handle_light_command(self, topic: str, payload: dict):
        """Handle light commands"""
        # topic: home/{room}/light/command
        parts = topic.split('/')
        room = parts[1] if len(parts) >= 2 else None

        command = payload.get("command")

        if command == "on":
            result = self.light_controller.turn_on(room)
        elif command == "off":
            result = self.light_controller.turn_off(room)
        elif command == "toggle":
            result = self.light_controller.toggle(room)
        else:
            result = False

        # Publish status
        status = self.light_controller.get_status(room)
        if status:
            self.publish_light_status(room, status)

    def publish_light_status(self, room: str, status: dict):
        """Publish light status"""
        topic = self.TOPICS["light_status"].format(room)
        self.client.publish(topic, json.dumps(status), qos=1, retain=True)

    def publish_sensor_data(self, sensor_id: str, data: dict):
        """Publish sensor data"""
        topic = self.TOPICS["sensor_data"].format(sensor_id)
        self.client.publish(topic, json.dumps(data), qos=0)

    def publish_motion(self, sensor_id: str, detected: bool):
        """Publish motion detection"""
        topic = self.TOPICS["motion"].format(sensor_id)
        data = {
            "detected": detected,
            "timestamp": datetime.now().isoformat()
        }
        self.client.publish(topic, json.dumps(data), qos=1)

    def start(self):
        """Start MQTT loop"""
        self.client.loop_start()

    def stop(self):
        """Stop MQTT"""
        # Publish offline status
        self.client.publish(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )
        self.client.loop_stop()
        self.client.disconnect()
```

### 4.2 Integrated Gateway

```python
#!/usr/bin/env python3
"""Smart home integrated gateway"""

import time
import threading
from datetime import datetime
import json

# Import previously defined classes
# from light_controller import LightController
# from sensor_monitor import EnvironmentMonitor
# from mqtt_handler import SmartHomeMQTT

class SmartHomeGateway:
    """Smart home gateway"""

    def __init__(self, config: dict):
        self.config = config

        # Light controller
        self.light_controller = LightController(config)

        # Environment monitor
        self.env_monitor = EnvironmentMonitor(
            sensor_pin=config.get('dht_pin', 4)
        )

        # MQTT handler
        self.mqtt_handler = SmartHomeMQTT(
            self.light_controller,
            self.env_monitor,
            broker=config.get('mqtt_broker', 'localhost')
        )

        self.running = False

    def _sensor_publish_loop(self, interval: int):
        """Sensor data publishing loop"""
        while self.running:
            data = self.env_monitor.get_latest()
            if data:
                self.mqtt_handler.publish_sensor_data("env_01", data)
            time.sleep(interval)

    def start(self):
        """Start gateway"""
        print("=== Smart Home Gateway Starting ===")

        self.running = True

        # Start environment monitoring
        self.env_monitor.start(interval=5)

        # Start MQTT
        self.mqtt_handler.start()

        # Sensor data publishing thread
        self.publish_thread = threading.Thread(
            target=self._sensor_publish_loop,
            args=(10,),
            daemon=True
        )
        self.publish_thread.start()

        print("Gateway running...")

    def stop(self):
        """Stop gateway"""
        print("\nStopping gateway...")

        self.running = False

        self.env_monitor.stop()
        self.mqtt_handler.stop()
        self.light_controller.all_off()
        self.light_controller.cleanup()

        print("Gateway stopped")

    def run(self):
        """Main loop"""
        self.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

# Main
if __name__ == "__main__":
    config = {
        "lights": [
            {"id": "living_room", "name": "Living Room", "gpio_pin": 17, "location": "Living Room"},
            {"id": "bedroom", "name": "Bedroom", "gpio_pin": 27, "location": "Bedroom"},
        ],
        "dht_pin": 4,
        "mqtt_broker": "localhost"
    }

    gateway = SmartHomeGateway(config)
    gateway.run()
```

---

## 5. Web Dashboard

### 5.1 Flask Web Server

```python
#!/usr/bin/env python3
"""Smart home web dashboard"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt
import json
import threading

app = Flask(__name__)
CORS(app)

# State storage
state = {
    "lights": {},
    "sensors": {},
    "motion": {}
}

# MQTT client
mqtt_client = mqtt.Client()

def on_mqtt_message(client, userdata, msg):
    """MQTT message handler"""
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic

        if "light/status" in topic:
            room = topic.split('/')[1]
            state["lights"][room] = payload

        elif "sensor" in topic:
            sensor_id = topic.split('/')[-1]
            state["sensors"][sensor_id] = payload

        elif "motion" in topic:
            sensor_id = topic.split('/')[-1]
            state["motion"][sensor_id] = payload

    except Exception as e:
        print(f"Message processing error: {e}")

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("home/#")
        print("MQTT connected")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(state)

@app.route('/api/lights')
def get_lights():
    return jsonify(state["lights"])

@app.route('/api/lights/<room>', methods=['POST'])
def control_light(room):
    data = request.get_json()
    command = data.get('command', 'toggle')

    topic = f"home/{room}/light/command"
    mqtt_client.publish(topic, json.dumps({"command": command}))

    return jsonify({"status": "sent", "room": room, "command": command})

@app.route('/api/sensors')
def get_sensors():
    return jsonify(state["sensors"])

@app.route('/api/sensors/<sensor_id>/history')
def get_sensor_history(sensor_id):
    # In practice, query from database
    return jsonify({"sensor_id": sensor_id, "history": []})

def start_mqtt():
    mqtt_client.connect("localhost", 1883)
    mqtt_client.loop_forever()

if __name__ == "__main__":
    # Start MQTT thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 5.2 HTML Template

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Home Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; margin-bottom: 20px; }

        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }

        .card {
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h2 { font-size: 1.2rem; color: #666; margin-bottom: 15px; }

        .light-control { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; }
        .light-name { font-weight: 500; }
        .light-btn {
            padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer;
            font-size: 14px; transition: all 0.3s;
        }
        .light-btn.on { background: #4CAF50; color: white; }
        .light-btn.off { background: #ddd; color: #666; }

        .sensor-value { font-size: 2rem; font-weight: bold; color: #333; }
        .sensor-label { color: #666; font-size: 0.9rem; }

        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
        .status.online { background: #4CAF50; }
        .status.offline { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Smart Home Dashboard</h1>

        <div class="grid">
            <!-- Light Control -->
            <div class="card">
                <h2>Light Control</h2>
                <div id="lights-container">
                    <p>Loading...</p>
                </div>
            </div>

            <!-- Environment Sensor -->
            <div class="card">
                <h2>Environment Sensor</h2>
                <div id="sensor-container">
                    <div style="display: flex; gap: 30px;">
                        <div>
                            <div class="sensor-value" id="temperature">--</div>
                            <div class="sensor-label">Temperature (°C)</div>
                        </div>
                        <div>
                            <div class="sensor-value" id="humidity">--</div>
                            <div class="sensor-label">Humidity (%)</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- System Status -->
            <div class="card">
                <h2>System Status</h2>
                <p><span class="status online"></span> Gateway Online</p>
                <p id="last-update">Last update: --</p>
            </div>
        </div>
    </div>

    <script>
        // Update status
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                // Update lights
                const lightsContainer = document.getElementById('lights-container');
                lightsContainer.innerHTML = '';

                for (const [room, status] of Object.entries(data.lights)) {
                    const div = document.createElement('div');
                    div.className = 'light-control';
                    div.innerHTML = `
                        <span class="light-name">${status.name || room}</span>
                        <button class="light-btn ${status.is_on ? 'on' : 'off'}"
                                onclick="toggleLight('${room}')">
                            ${status.is_on ? 'ON' : 'OFF'}
                        </button>
                    `;
                    lightsContainer.appendChild(div);
                }

                // Update sensors
                const sensorData = Object.values(data.sensors)[0];
                if (sensorData) {
                    document.getElementById('temperature').textContent =
                        sensorData.temperature?.toFixed(1) || '--';
                    document.getElementById('humidity').textContent =
                        sensorData.humidity?.toFixed(1) || '--';
                }

                document.getElementById('last-update').textContent =
                    'Last update: ' + new Date().toLocaleTimeString();

            } catch (error) {
                console.error('Update failed:', error);
            }
        }

        // Toggle light
        async function toggleLight(room) {
            try {
                await fetch(`/api/lights/${room}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'toggle' })
                });
                setTimeout(updateStatus, 500);
            } catch (error) {
                console.error('Light control failed:', error);
            }
        }

        // Initialize and periodic update
        updateStatus();
        setInterval(updateStatus, 3000);
    </script>
</body>
</html>
```

---

## 6. Full System Execution

```python
#!/usr/bin/env python3
"""Smart home system execution script"""

import subprocess
import time
import signal
import sys

def main():
    processes = []

    try:
        # 1. Check Mosquitto broker
        print("Checking MQTT broker...")
        # subprocess.run(["sudo", "systemctl", "start", "mosquitto"])

        # 2. Start gateway
        print("Starting gateway...")
        gateway = subprocess.Popen(
            ["python3", "gateway/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(gateway)

        time.sleep(2)

        # 3. Start web server
        print("Starting web server...")
        web = subprocess.Popen(
            ["python3", "web/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(web)

        print("\n=== Smart Home System Running ===")
        print("Web dashboard: http://localhost:5000")
        print("Press Ctrl+C to exit\n")

        # Monitor processes
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"Process exited: {p.pid}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down system...")
    finally:
        for p in processes:
            p.terminate()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
```

---

## Practice Problems

### Problem 1: Automation Rules
1. Automatically turn on air conditioning (relay) when temperature exceeds 30°C.
2. Automatically turn on lights when motion is detected.

### Problem 2: Scheduling
1. Implement a scheduler that automatically controls lights at specific times.
2. Control lights based on sunrise/sunset times.

### Problem 3: Notification System
1. Publish MQTT notification when temperature exceeds threshold.
2. Display notifications on web dashboard.

---

**Previous**: [Edge AI - ONNX Runtime](./09_Edge_AI_ONNX.md) | **Next**: [Image Analysis Project](./11_Image_Analysis_Project.md)
