#!/usr/bin/env python3
"""
Smart Home Automation System

An integrated home automation system providing lighting control, environmental monitoring,
and MQTT-based device communication.
Supports simulation mode for operation without actual hardware.

Key Features:
- Relay-based lighting/appliance control (simulation mode supported)
- Temperature/humidity sensor monitoring (simulated data generation)
- MQTT-based device communication and control
- Automation rule engine (temperature-based, motion-based)
- Web dashboard JSON API
- State management and logging

Usage:
    # Simulation mode (no hardware required)
    python home_automation.py --simulate

    # Real hardware mode
    python home_automation.py

    # Specify MQTT broker
    python home_automation.py --broker mqtt.example.com --simulate
"""

import time
import json
import random
import threading
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from queue import Queue
import argparse

# Simulation mode flag
SIMULATION_MODE = True

# Real hardware libraries (not used in simulation mode)
try:
    from gpiozero import OutputDevice
    import adafruit_dht
    import board
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("Hardware libraries not found. Running in simulation mode.")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("paho-mqtt library not found. MQTT features will be disabled.")


# ============================================================
# Data Models
# ============================================================

@dataclass
class Light:
    """Light device data class"""
    id: str
    name: str
    gpio_pin: int
    location: str
    is_on: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SensorReading:
    """Sensor data class"""
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "sensor_id": self.sensor_id,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================
# Light Controller
# ============================================================

class LightController:
    """
    Light Control Class

    Controls lights through relays.
    In simulation mode, manages state only without actual GPIO.
    """

    def __init__(self, config: dict, simulate: bool = True):
        """
        Initialize

        Args:
            config: Light configuration (includes lights list)
            simulate: Whether to use simulation mode
        """
        self.simulate = simulate
        self.lights: Dict[str, Light] = {}
        self.relays: Dict[str, any] = {}

        # Light configuration
        for light_config in config.get('lights', []):
            light = Light(**light_config)
            self.lights[light.id] = light

            # Relay initialization
            if not simulate and HARDWARE_AVAILABLE:
                # Why: Most relay modules are active-low: the relay closes when
                # GPIO goes LOW. Setting active_high=False inverts the logic so
                # that on()/off() in Python match physical on/off.
                relay = OutputDevice(
                    light.gpio_pin,
                    active_high=False,
                    initial_value=False
                )
                self.relays[light.id] = relay
            else:
                # Simulation: store None
                self.relays[light.id] = None

        logging.info(f"LightController initialized (simulation={simulate}, lights={len(self.lights)})")

    def turn_on(self, light_id: str) -> bool:
        """
        Turn on light

        Args:
            light_id: Light ID

        Returns:
            Success status
        """
        if light_id not in self.lights:
            logging.warning(f"Light ID '{light_id}' not found")
            return False

        if self.simulate:
            # Simulation: change state only
            self.lights[light_id].is_on = True
            logging.info(f"[Sim] Light ON: {self.lights[light_id].name}")
        else:
            # Real hardware: control relay
            self.relays[light_id].on()
            self.lights[light_id].is_on = True
            logging.info(f"Light ON: {self.lights[light_id].name}")

        return True

    def turn_off(self, light_id: str) -> bool:
        """
        Turn off light

        Args:
            light_id: Light ID

        Returns:
            Success status
        """
        if light_id not in self.lights:
            logging.warning(f"Light ID '{light_id}' not found")
            return False

        if self.simulate:
            # Simulation
            self.lights[light_id].is_on = False
            logging.info(f"[Sim] Light OFF: {self.lights[light_id].name}")
        else:
            # Real hardware
            self.relays[light_id].off()
            self.lights[light_id].is_on = False
            logging.info(f"Light OFF: {self.lights[light_id].name}")

        return True

    def toggle(self, light_id: str) -> bool:
        """
        Toggle light

        Args:
            light_id: Light ID

        Returns:
            Success status
        """
        if light_id not in self.lights:
            return False

        if self.lights[light_id].is_on:
            return self.turn_off(light_id)
        else:
            return self.turn_on(light_id)

    def get_status(self, light_id: str = None) -> Optional[dict]:
        """
        Get light status

        Args:
            light_id: Light ID (None for all)

        Returns:
            Status dictionary
        """
        if light_id:
            light = self.lights.get(light_id)
            if light:
                return light.to_dict()
            return None

        # All lights status
        return {
            "lights": [light.to_dict() for light in self.lights.values()]
        }

    def all_off(self):
        """Turn off all lights"""
        for light_id in self.lights:
            self.turn_off(light_id)
        logging.info("All lights OFF")

    def all_on(self):
        """Turn on all lights"""
        for light_id in self.lights:
            self.turn_on(light_id)
        logging.info("All lights ON")

    def cleanup(self):
        """Cleanup (called on program exit)"""
        if not self.simulate:
            for relay in self.relays.values():
                if relay:
                    relay.close()
        logging.info("LightController cleanup complete")


# ============================================================
# Environment Monitor
# ============================================================

class EnvironmentMonitor:
    """
    Environmental Sensor Monitoring Class

    Periodically reads temperature/humidity from DHT11 sensor and stores history.
    In simulation mode, generates random data.
    """

    def __init__(self, sensor_pin: int = 4, sensor_id: str = "env_01", simulate: bool = True):
        """
        Initialize

        Args:
            sensor_pin: GPIO pin number
            sensor_id: Sensor ID
            simulate: Whether to use simulation mode
        """
        self.sensor_id = sensor_id
        self.sensor_pin = sensor_pin
        self.simulate = simulate

        # DHT sensor initialization
        self.dht = None
        if not simulate and HARDWARE_AVAILABLE:
            try:
                self.dht = adafruit_dht.DHT11(getattr(board, f"D{sensor_pin}"))
            except Exception as e:
                logging.error(f"DHT sensor initialization failed: {e}")
                self.simulate = True

        # Data queue and history
        self.data_queue = Queue()
        self.latest_reading: Optional[SensorReading] = None
        self.readings_history: List[SensorReading] = []
        self.max_history = 1000

        # Thread control
        self.running = False
        self.thread = None

        # Simulation current values
        self.sim_temperature = 25.0
        self.sim_humidity = 60.0

        logging.info(f"EnvironmentMonitor initialized (simulation={simulate})")

    def read_sensor(self) -> Optional[SensorReading]:
        """
        Read sensor

        Returns:
            Sensor data or None (on failure)
        """
        if self.simulate:
            # Simulation: generate random changes
            self.sim_temperature += random.uniform(-0.5, 0.5)
            self.sim_temperature = max(10, min(40, self.sim_temperature))

            self.sim_humidity += random.uniform(-2, 2)
            self.sim_humidity = max(30, min(90, self.sim_humidity))

            reading = SensorReading(
                sensor_id=self.sensor_id,
                temperature=round(self.sim_temperature, 1),
                humidity=round(self.sim_humidity, 1),
                timestamp=datetime.now()
            )
            return reading

        else:
            # Real sensor
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
                logging.debug(f"Sensor read failed (normal): {e}")
            except Exception as e:
                logging.error(f"Sensor read error: {e}")

            return None

    def _monitor_loop(self, interval: int):
        """
        Monitoring loop (background thread)

        Args:
            interval: Read interval (seconds)
        """
        while self.running:
            reading = self.read_sensor()

            if reading:
                # Update latest data
                self.latest_reading = reading

                # Store history
                self.readings_history.append(reading)
                if len(self.readings_history) > self.max_history:
                    self.readings_history.pop(0)

                # Add to queue (for external subscribers)
                self.data_queue.put(reading)

            time.sleep(interval)

    def start(self, interval: int = 5):
        """
        Start monitoring

        Args:
            interval: Read interval (seconds)
        """
        if self.running:
            logging.warning("Already monitoring")
            return

        self.running = True
        # Why: daemon=True ensures the monitoring thread doesn't prevent program
        # exit. Without it, a forgotten stop() call would hang the process
        # forever, which is a real risk on headless IoT devices.
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.thread.start()
        logging.info(f"Environment monitoring started (interval={interval}s)")

    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

        if self.dht and not self.simulate:
            self.dht.exit()

        logging.info("Environment monitoring stopped")

    def get_latest(self) -> Optional[dict]:
        """Return latest sensor data"""
        if self.latest_reading:
            return self.latest_reading.to_dict()
        return None

    def get_stats(self) -> dict:
        """
        Return statistics data

        Returns:
            Min/max/average statistics
        """
        if not self.readings_history:
            return {}

        temps = [r.temperature for r in self.readings_history]
        humids = [r.humidity for r in self.readings_history]

        return {
            "count": len(self.readings_history),
            "temperature": {
                "min": round(min(temps), 1),
                "max": round(max(temps), 1),
                "avg": round(sum(temps) / len(temps), 1)
            },
            "humidity": {
                "min": round(min(humids), 1),
                "max": round(max(humids), 1),
                "avg": round(sum(humids) / len(humids), 1)
            }
        }


# ============================================================
# MQTT Handler
# ============================================================

class SmartHomeMQTT:
    """
    MQTT-based Smart Home Control Handler

    Controls devices and publishes sensor data through an MQTT broker.
    """

    TOPICS = {
        "light_command": "home/+/light/command",
        "light_status": "home/{}/light/status",
        "sensor_data": "home/sensor/{}",
        "motion": "home/motion/{}",
        "system": "home/system/status",
        "automation": "home/automation/event"
    }

    def __init__(self, light_controller: LightController,
                 env_monitor: EnvironmentMonitor,
                 broker: str = "localhost",
                 port: int = 1883):
        """
        Initialize

        Args:
            light_controller: Light controller
            env_monitor: Environment monitor
            broker: MQTT broker address
            port: MQTT port
        """
        self.light_controller = light_controller
        self.env_monitor = env_monitor
        self.broker = broker
        self.port = port

        if not MQTT_AVAILABLE:
            logging.warning("MQTT library not found. MQTT features disabled")
            self.client = None
            return

        # Create MQTT client
        self.client = mqtt.Client(client_id="smart_home_gateway")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # Why: LWT (Last Will and Testament) is an MQTT feature where the broker
        # publishes a pre-registered message if the client disconnects ungracefully.
        # This lets other devices detect that the gateway is offline without polling.
        self.client.will_set(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )

        logging.info(f"MQTT client created (broker={broker}:{port})")

    def connect(self):
        """Connect to MQTT broker"""
        if not self.client:
            return

        try:
            self.client.connect(self.broker, self.port)
            logging.info("Attempting MQTT broker connection...")
        except Exception as e:
            logging.error(f"MQTT connection failed: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logging.info("MQTT broker connection successful")

            # Subscribe to topics
            client.subscribe(self.TOPICS["light_command"])
            logging.info(f"Topic subscribed: {self.TOPICS['light_command']}")

            # Publish online status
            client.publish(
                self.TOPICS["system"],
                json.dumps({
                    "status": "online",
                    "timestamp": datetime.now().isoformat()
                }),
                qos=1,
                retain=True
            )
        else:
            logging.error(f"MQTT connection failed (code={rc})")

    def _on_message(self, client, userdata, msg):
        """MQTT message received callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            logging.debug(f"MQTT received: {topic} = {payload}")

            # Handle light commands
            if "light/command" in topic:
                self._handle_light_command(topic, payload)

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON: {msg.payload}")
        except Exception as e:
            logging.error(f"Message processing error: {e}")

    def _handle_light_command(self, topic: str, payload: dict):
        """
        Handle light command

        Args:
            topic: MQTT topic (home/{room}/light/command)
            payload: Command data {"command": "on|off|toggle"}
        """
        # Extract room ID from topic
        parts = topic.split('/')
        room = parts[1] if len(parts) >= 2 else None

        if not room:
            logging.warning(f"No room ID: {topic}")
            return

        command = payload.get("command")

        # Execute command
        result = False
        if command == "on":
            result = self.light_controller.turn_on(room)
        elif command == "off":
            result = self.light_controller.turn_off(room)
        elif command == "toggle":
            result = self.light_controller.toggle(room)
        else:
            logging.warning(f"Unknown command: {command}")

        # Publish status
        if result:
            status = self.light_controller.get_status(room)
            if status:
                self.publish_light_status(room, status)

    def publish_light_status(self, room: str, status: dict):
        """
        Publish light status

        Args:
            room: Room ID
            status: Status data
        """
        if not self.client:
            return

        topic = self.TOPICS["light_status"].format(room)
        self.client.publish(topic, json.dumps(status), qos=1, retain=True)
        logging.debug(f"Light status published: {topic}")

    def publish_sensor_data(self, sensor_id: str, data: dict):
        """
        Publish sensor data

        Args:
            sensor_id: Sensor ID
            data: Sensor data
        """
        if not self.client:
            return

        topic = self.TOPICS["sensor_data"].format(sensor_id)
        self.client.publish(topic, json.dumps(data), qos=0)

    def publish_motion(self, sensor_id: str, detected: bool):
        """
        Publish motion detection

        Args:
            sensor_id: Sensor ID
            detected: Whether motion was detected
        """
        if not self.client:
            return

        topic = self.TOPICS["motion"].format(sensor_id)
        data = {
            "detected": detected,
            "timestamp": datetime.now().isoformat()
        }
        self.client.publish(topic, json.dumps(data), qos=1)

    def publish_automation_event(self, event_type: str, details: dict):
        """
        Publish automation event

        Args:
            event_type: Event type
            details: Detail information
        """
        if not self.client:
            return

        data = {
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.client.publish(self.TOPICS["automation"], json.dumps(data), qos=1)

    def start(self):
        """Start MQTT loop"""
        if self.client:
            self.client.loop_start()

    def stop(self):
        """Stop MQTT"""
        if not self.client:
            return

        # Publish offline status
        self.client.publish(
            self.TOPICS["system"],
            json.dumps({"status": "offline"}),
            qos=1,
            retain=True
        )

        self.client.loop_stop()
        self.client.disconnect()
        logging.info("MQTT connection closed")


# ============================================================
# Automation Rule Engine
# ============================================================

class AutomationEngine:
    """
    Automation Rule Engine

    Automatically controls lights/appliances based on sensor data.
    """

    def __init__(self, light_controller: LightController, mqtt_handler: SmartHomeMQTT):
        """
        Initialize

        Args:
            light_controller: Light controller
            mqtt_handler: MQTT handler
        """
        self.light_controller = light_controller
        self.mqtt_handler = mqtt_handler
        self.rules: List[dict] = []

        logging.info("AutomationEngine initialized")

    # Why: Separating condition and action as callables implements a lightweight
    # rule engine. New automation behaviors can be added at runtime without
    # modifying the engine itself — a plugin architecture for home automation.
    def add_rule(self, name: str, condition: Callable, action: Callable):
        """
        Add rule

        Args:
            name: Rule name
            condition: Condition function (returns True/False)
            action: Action function
        """
        self.rules.append({
            "name": name,
            "condition": condition,
            "action": action,
            "last_triggered": None
        })
        logging.info(f"Automation rule added: {name}")

    def check_rules(self, sensor_data: dict):
        """
        Check and execute rules

        Args:
            sensor_data: Sensor data
        """
        for rule in self.rules:
            try:
                if rule["condition"](sensor_data):
                    # Execute action when condition is met
                    logging.info(f"Automation rule triggered: {rule['name']}")
                    rule["action"](sensor_data)
                    rule["last_triggered"] = datetime.now()

                    # Publish MQTT event
                    self.mqtt_handler.publish_automation_event(
                        event_type=rule["name"],
                        details=sensor_data
                    )

            except Exception as e:
                logging.error(f"Rule execution error ({rule['name']}): {e}")

    def get_rules_status(self) -> List[dict]:
        """Get rules status"""
        return [
            {
                "name": rule["name"],
                "last_triggered": rule["last_triggered"].isoformat() if rule["last_triggered"] else None
            }
            for rule in self.rules
        ]


# ============================================================
# Smart Home Gateway (Integrated System)
# ============================================================

class SmartHomeGateway:
    """
    Smart Home Integrated Gateway

    Integrates all components to operate the smart home system.
    """

    def __init__(self, config: dict, simulate: bool = True):
        """
        Initialize

        Args:
            config: Configuration dictionary
            simulate: Whether to use simulation mode
        """
        self.config = config
        self.simulate = simulate

        # Light controller
        self.light_controller = LightController(config, simulate=simulate)

        # Environment monitor
        self.env_monitor = EnvironmentMonitor(
            sensor_pin=config.get('dht_pin', 4),
            sensor_id="env_01",
            simulate=simulate
        )

        # MQTT handler
        self.mqtt_handler = SmartHomeMQTT(
            self.light_controller,
            self.env_monitor,
            broker=config.get('mqtt_broker', 'localhost'),
            port=config.get('mqtt_port', 1883)
        )

        # Automation engine
        self.automation_engine = AutomationEngine(
            self.light_controller,
            self.mqtt_handler
        )

        # Thread control
        self.running = False
        self.threads = []

        # Setup automation rules
        self._setup_automation_rules()

        logging.info("SmartHomeGateway initialization complete")

    def _setup_automation_rules(self):
        """Setup automation rules"""

        # Rule 1: Turn on living room light when temperature exceeds 30C (e.g., instead of AC)
        def temp_high_condition(data):
            return data.get("temperature", 0) > 30

        def temp_high_action(data):
            self.light_controller.turn_on("living_room")
            logging.info(f"[Automation] High temperature detected ({data['temperature']}C) - Living room light ON")

        self.automation_engine.add_rule(
            "high_temperature_alert",
            temp_high_condition,
            temp_high_action
        )

        # Rule 2: Turn off all lights when temperature drops below 20C
        def temp_low_condition(data):
            return data.get("temperature", 100) < 20

        def temp_low_action(data):
            self.light_controller.all_off()
            logging.info(f"[Automation] Low temperature detected ({data['temperature']}C) - All lights OFF")

        self.automation_engine.add_rule(
            "low_temperature_save",
            temp_low_condition,
            temp_low_action
        )

        # Rule 3: Turn on bathroom light when humidity exceeds 80%
        def humidity_high_condition(data):
            return data.get("humidity", 0) > 80

        def humidity_high_action(data):
            self.light_controller.turn_on("bathroom")
            logging.info(f"[Automation] High humidity detected ({data['humidity']}%) - Bathroom light ON")

        self.automation_engine.add_rule(
            "high_humidity_ventilation",
            humidity_high_condition,
            humidity_high_action
        )

    def _sensor_publish_loop(self, interval: int):
        """
        Sensor data publish loop

        Args:
            interval: Publish interval (seconds)
        """
        while self.running:
            data = self.env_monitor.get_latest()
            if data:
                # MQTT publish
                self.mqtt_handler.publish_sensor_data("env_01", data)

                # Check automation rules
                self.automation_engine.check_rules(data)

            time.sleep(interval)

    def _status_report_loop(self, interval: int):
        """
        Status report loop

        Args:
            interval: Report interval (seconds)
        """
        while self.running:
            # Print system status
            sensor_data = self.env_monitor.get_latest()
            light_status = self.light_controller.get_status()

            logging.info("=" * 60)
            logging.info("System Status Report")
            logging.info("-" * 60)

            if sensor_data:
                logging.info(f"Temperature: {sensor_data['temperature']}C, Humidity: {sensor_data['humidity']}%")

            if light_status:
                for light in light_status["lights"]:
                    status = "ON" if light["is_on"] else "OFF"
                    logging.info(f"{light['name']} ({light['location']}): {status}")

            logging.info("=" * 60)

            time.sleep(interval)

    def start(self):
        """Start gateway"""
        if self.running:
            logging.warning("Already running")
            return

        logging.info("=" * 60)
        logging.info("Smart Home Gateway Starting")
        logging.info(f"Simulation mode: {self.simulate}")
        logging.info("=" * 60)

        self.running = True

        # Start environment monitoring
        self.env_monitor.start(interval=5)

        # MQTT connect and start
        self.mqtt_handler.connect()
        self.mqtt_handler.start()

        # Sensor data publish thread
        sensor_thread = threading.Thread(
            target=self._sensor_publish_loop,
            args=(10,),
            daemon=True
        )
        sensor_thread.start()
        self.threads.append(sensor_thread)

        # Status report thread
        status_thread = threading.Thread(
            target=self._status_report_loop,
            args=(30,),
            daemon=True
        )
        status_thread.start()
        self.threads.append(status_thread)

        logging.info("Gateway running...")

    def stop(self):
        """Stop gateway"""
        if not self.running:
            return

        logging.info("Stopping gateway...")

        self.running = False

        # Cleanup components
        self.env_monitor.stop()
        self.mqtt_handler.stop()
        self.light_controller.all_off()
        self.light_controller.cleanup()

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)

        logging.info("Gateway stopped")

    def run(self):
        """Main execution loop"""
        self.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nKeyboardInterrupt received")
        finally:
            self.stop()

    def get_dashboard_data(self) -> dict:
        """
        Provide JSON data for web dashboard

        Returns:
            Full system status
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "lights": self.light_controller.get_status(),
            "sensor": {
                "latest": self.env_monitor.get_latest(),
                "stats": self.env_monitor.get_stats()
            },
            "automation": {
                "rules": self.automation_engine.get_rules_status()
            },
            "system": {
                "running": self.running,
                "simulation_mode": self.simulate
            }
        }


# ============================================================
# Main Execution
# ============================================================

def main():
    """Main function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Smart Home Automation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python home_automation.py --simulate
  python home_automation.py --broker mqtt.example.com --simulate
  python home_automation.py --loglevel DEBUG --simulate
        """
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulation mode (no hardware required)"
    )

    parser.add_argument(
        "--broker",
        type=str,
        default="localhost",
        help="MQTT broker address (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="MQTT port (default: 1883)"
    )

    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )

    args = parser.parse_args()

    # Logging configuration
    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Simulation mode configuration
    simulate = args.simulate or not HARDWARE_AVAILABLE

    if simulate:
        logging.info("Running in simulation mode (no hardware required)")
    else:
        logging.info("Running in real hardware mode")

    # Configuration
    config = {
        "lights": [
            {
                "id": "living_room",
                "name": "Living Room Light",
                "gpio_pin": 17,
                "location": "Living Room"
            },
            {
                "id": "bedroom",
                "name": "Bedroom Light",
                "gpio_pin": 27,
                "location": "Bedroom"
            },
            {
                "id": "kitchen",
                "name": "Kitchen Light",
                "gpio_pin": 22,
                "location": "Kitchen"
            },
            {
                "id": "bathroom",
                "name": "Bathroom Light",
                "gpio_pin": 23,
                "location": "Bathroom"
            }
        ],
        "dht_pin": 4,
        "mqtt_broker": args.broker,
        "mqtt_port": args.port
    }

    # Create and run gateway
    gateway = SmartHomeGateway(config, simulate=simulate)

    # Demo: print dashboard data after 5 seconds
    def print_dashboard():
        time.sleep(5)
        dashboard_data = gateway.get_dashboard_data()
        logging.info("\n" + "=" * 60)
        logging.info("Dashboard Data (JSON API)")
        logging.info("=" * 60)
        print(json.dumps(dashboard_data, indent=2, ensure_ascii=False))
        logging.info("=" * 60)

    demo_thread = threading.Thread(target=print_dashboard, daemon=True)
    demo_thread.start()

    # Run main loop
    gateway.run()


if __name__ == "__main__":
    main()
