"""
Exercises for Lesson 02: Raspberry Pi Setup
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates Raspberry Pi hardware operations using Python.

Note: GPIO and hardware operations are simulated for environments
without actual Raspberry Pi hardware. On a real Pi, replace the
SimulatedGPIO class with RPi.GPIO.
"""

import time
import csv
import os
import subprocess
from datetime import datetime


# ---------------------------------------------------------------------------
# Simulated GPIO layer (allows running on any machine)
# On a real Raspberry Pi, replace with: import RPi.GPIO as GPIO
# ---------------------------------------------------------------------------

class SimulatedGPIO:
    """Simulate RPi.GPIO for exercise purposes.

    Real Raspberry Pi GPIO drives voltage on physical pins.
    BCM numbering maps to the Broadcom SoC channel numbers (e.g., GPIO17),
    while BOARD numbering maps to the physical 40-pin header positions.
    """
    BCM = "BCM"
    BOARD = "BOARD"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0
    PUD_UP = "PUD_UP"

    _mode = None
    _pins = {}

    @classmethod
    def setmode(cls, mode):
        cls._mode = mode
        print(f"[GPIO] Mode set to {mode}")

    @classmethod
    def setup(cls, pin, direction, pull_up_down=None):
        cls._pins[pin] = {"direction": direction, "state": cls.LOW, "pull": pull_up_down}
        pull_info = f" (pull={pull_up_down})" if pull_up_down else ""
        print(f"[GPIO] Pin {pin} configured as {direction}{pull_info}")

    @classmethod
    def output(cls, pin, state):
        if pin in cls._pins:
            cls._pins[pin]["state"] = state
            state_str = "HIGH" if state == cls.HIGH else "LOW"
            print(f"[GPIO] Pin {pin} -> {state_str}")

    @classmethod
    def input(cls, pin):
        # Simulate: return LOW (pressed) every 3rd call for button demos
        if not hasattr(cls, '_call_count'):
            cls._call_count = {}
        cls._call_count[pin] = cls._call_count.get(pin, 0) + 1
        return cls.LOW if cls._call_count[pin] % 3 == 0 else cls.HIGH

    @classmethod
    def cleanup(cls):
        cls._pins.clear()
        cls._mode = None
        print("[GPIO] Cleanup complete - all pins released")


GPIO = SimulatedGPIO


# === Exercise 1: Model Selection Justification ===
# Problem: For three IoT deployment scenarios, identify the most appropriate
# Raspberry Pi model and explain your reasoning.

def exercise_1():
    """Solution: Raspberry Pi model selection for three scenarios."""

    # Each scenario has different constraints (size, power, compute, budget).
    # The answer references the model comparison table from Section 1.1.

    scenarios = [
        {
            "scenario": "A portable wearable that monitors body temperature and "
                       "transmits to a smartphone. Size and power consumption are critical.",
            "recommended_model": "Raspberry Pi Zero 2 W",
            "reasoning": [
                "Smallest form factor in the Raspberry Pi family (65mm x 30mm), "
                "essential for a wearable enclosure",
                "Quad-core Cortex-A53 at 1 GHz is sufficient for temperature sampling "
                "and BLE transmission; no heavy compute needed",
                "512 MB RAM is adequate for a single-purpose sensor application",
                "Built-in WiFi and Bluetooth enable smartphone communication",
                "Lower power draw than Pi 3B+/4B, extending battery life",
                "Micro USB for compact connectors compatible with small batteries",
            ],
            "why_not_alternatives": {
                "Pi 4B": "Too large, too power-hungry (5V 3A) for a wearable on battery",
                "Pi Pico": "No WiFi/BLE built-in, no Linux OS (harder BLE stack); "
                          "viable if paired with external BLE module but adds complexity",
            },
        },
        {
            "scenario": "A gateway device that collects data from 20 sensors, "
                       "runs a local MQTT broker, and streams video to a cloud server.",
            "recommended_model": "Raspberry Pi 4B (4 GB or 8 GB)",
            "reasoning": [
                "Quad-core Cortex-A72 at 1.5 GHz handles concurrent tasks: "
                "MQTT broker, 20 sensor connections, video encoding",
                "4-8 GB RAM needed for Mosquitto broker, video buffer, and Python processes",
                "Gigabit Ethernet provides reliable wired uplink for video streaming "
                "(WiFi may saturate with 20 sensors + video)",
                "USB 3.0 ports allow connecting a USB camera or additional storage",
                "Dual 4K display output useful for local monitoring dashboard (optional)",
                "Full Linux OS supports Docker, systemd services, and standard networking tools",
            ],
            "why_not_alternatives": {
                "Pi 3B+": "Only 1 GB RAM may be insufficient for broker + video + 20 sensors; "
                         "USB 2.0 and 100 Mbps Ethernet limit throughput",
                "Pi Zero 2 W": "512 MB RAM cannot handle MQTT broker and video simultaneously",
            },
        },
        {
            "scenario": "A learning project where you want to experiment with GPIOs, "
                       "Python, and a desktop GUI. Budget is limited.",
            "recommended_model": "Raspberry Pi 3B+",
            "reasoning": [
                "Cheapest full-Linux model with 40-pin GPIO header for hardware experiments",
                "1 GB RAM is sufficient for lightweight desktop (Raspberry Pi OS with Desktop)",
                "Built-in WiFi and Bluetooth for networking exercises",
                "Extensive community support and tutorials targeted at Pi 3B+",
                "Quad-core 1.4 GHz runs Python, Thonny IDE, and simple GUIs smoothly",
                "Standard HDMI output for monitor; no need for micro-HDMI adapters",
                "Available used for under $25 due to large install base",
            ],
            "why_not_alternatives": {
                "Pi 4B": "Higher cost ($35-75); overkill for basic GPIO and Python learning",
                "Pi Pico": "No desktop GUI capability (microcontroller, not a full computer)",
            },
        },
    ]

    print("=== Raspberry Pi Model Selection ===\n")

    for i, s in enumerate(scenarios, 1):
        print(f"--- Scenario {i} ---")
        print(f"  Description: {s['scenario']}")
        print(f"  Recommended: {s['recommended_model']}")
        print("  Reasoning:")
        for r in s["reasoning"]:
            print(f"    - {r}")
        print("  Why not alternatives:")
        for model, reason in s["why_not_alternatives"].items():
            print(f"    {model}: {reason}")
        print()


# === Exercise 2: Headless Setup and SSH Hardening ===
# Problem: Set up a Raspberry Pi in headless mode and harden SSH access.
# (Simulated: demonstrates the commands and configuration that would be used.)

def exercise_2():
    """Solution: Headless setup and SSH hardening steps."""

    # On a real Pi, these commands run in a terminal. Here we document the
    # exact steps and explain the security rationale for each.

    steps = {
        "Step 1: Write OS with Raspberry Pi Imager": {
            "commands": [
                "# Launch Raspberry Pi Imager",
                "# Select: Raspberry Pi OS Lite (64-bit) -- no desktop needed for headless",
                "# Click gear icon for Advanced Options:",
                "#   Enable SSH: checked",
                "#   Set username: pi",
                "#   Set password: <strong-password>",
                "#   Configure WiFi: SSID=<your-ssid>, Password=<wifi-password>",
                "#   Set locale: timezone and keyboard layout",
                "# Write to SD card",
            ],
            "explanation": "Pre-configuring SSH and WiFi in the Imager means the Pi is "
                          "network-accessible on first boot without ever connecting a monitor.",
        },
        "Step 2: Find the Pi's IP and connect via SSH": {
            "commands": [
                "# Scan the local subnet for the new device",
                "sudo nmap -sn 192.168.1.0/24",
                "",
                "# Or use arp to find it",
                "arp -a | grep -i raspberry",
                "",
                "# Connect",
                "ssh pi@<ip-address>",
            ],
            "explanation": "nmap -sn performs a ping sweep without port scanning. "
                          "The Pi's hostname often appears as 'raspberrypi' in arp tables.",
        },
        "Step 3: Generate Ed25519 key pair and copy to Pi": {
            "commands": [
                "# On the host machine (laptop/desktop):",
                "ssh-keygen -t ed25519 -C 'pi-access-key'",
                "# Accept default path (~/.ssh/id_ed25519) or specify a custom one",
                "",
                "# Copy public key to the Pi:",
                "ssh-copy-id -i ~/.ssh/id_ed25519.pub pi@<ip-address>",
                "",
                "# Verify: this should log in without asking for a password",
                "ssh pi@<ip-address>",
            ],
            "explanation": "Ed25519 is preferred over RSA: shorter keys (256-bit), faster "
                          "signing, and resistant to several classes of attacks. "
                          "ssh-copy-id appends the public key to ~/.ssh/authorized_keys on the Pi.",
        },
        "Step 4: Disable password authentication": {
            "commands": [
                "# On the Pi, edit the SSH server configuration:",
                "sudo nano /etc/ssh/sshd_config",
                "",
                "# Change or add these lines:",
                "#   PasswordAuthentication no",
                "#   PubkeyAuthentication yes",
                "#   ChallengeResponseAuthentication no",
                "",
                "# Restart SSH service to apply changes:",
                "sudo systemctl restart sshd",
            ],
            "explanation": "Disabling password authentication prevents brute-force attacks. "
                          "Only holders of the matching private key can log in. "
                          "Always test key-based login BEFORE disabling passwords to avoid lockout.",
        },
        "Step 5: Verify hardened configuration": {
            "commands": [
                "# From the host machine:",
                "",
                "# Test 1: Key-based login should succeed",
                "ssh pi@<ip-address>  # Should log in without password prompt",
                "",
                "# Test 2: Password login should be rejected",
                "ssh -o PubkeyAuthentication=no pi@<ip-address>",
                "# Expected output: Permission denied (publickey).",
            ],
            "explanation": "The -o PubkeyAuthentication=no flag forces the client to attempt "
                          "password auth. If the server correctly rejects it, hardening is verified.",
        },
    }

    print("=== Headless Setup and SSH Hardening ===\n")

    for step_name, details in steps.items():
        print(f"--- {step_name} ---")
        for cmd in details["commands"]:
            print(f"  {cmd}")
        print(f"\n  Explanation: {details['explanation']}\n")


# === Exercise 3: GPIO Pin Mapping ===
# Problem: Answer GPIO pin layout questions and write a BOARD-mode blink script.

def exercise_3():
    """Solution: GPIO pin mapping knowledge test."""

    # These answers come directly from the 40-pin header layout in Section 4.1.

    answers = {
        "Q1: Which physical pin numbers carry 5V? Which carry 3.3V?": {
            "5V pins": "Physical pins 2 and 4",
            "3.3V pins": "Physical pins 1 and 17",
            "note": "5V pins are connected directly to the power input; 3.3V pins "
                   "are regulated by an on-board 3.3V regulator (max ~50 mA total draw).",
        },
        "Q2: GPIO17 BCM = which physical (BOARD) pin?": {
            "answer": "Physical pin 11",
            "mapping": "GPIO17 (BCM) == Pin 11 (BOARD)",
            "note": "The BCM number refers to the Broadcom SoC channel; the BOARD number "
                   "refers to the physical position on the 40-pin header counting from pin 1.",
        },
        "Q3: Which GPIO pins support hardware PWM?": {
            "answer": "GPIO12 (pin 32), GPIO13 (pin 33), GPIO18 (pin 12), GPIO19 (pin 35)",
            "detail": "GPIO12 and GPIO18 share PWM channel 0; GPIO13 and GPIO19 share PWM channel 1. "
                     "Hardware PWM is more precise than software PWM because it is driven by "
                     "dedicated silicon rather than CPU timing loops.",
        },
        "Q4: Which pins are used for I2C and their BCM numbers?": {
            "answer": "GPIO2 (SDA, pin 3) and GPIO3 (SCL, pin 5)",
            "detail": "These are the default I2C bus 1 pins. They have internal 1.8 kOhm "
                     "pull-up resistors to 3.3V built into the Pi. I2C is commonly used "
                     "for sensors like BMP280 (pressure) and SSD1306 (OLED display).",
        },
    }

    print("=== GPIO Pin Mapping Answers ===\n")
    for question, info in answers.items():
        print(f"  {question}")
        for key, val in info.items():
            print(f"    {key}: {val}")
        print()

    # BOARD-mode LED blink script
    print("--- BOARD-mode LED Blink Script ---\n")

    # Using simulated GPIO in BOARD mode
    # On a real Pi: import RPi.GPIO as GPIO
    LED_PIN_BOARD = 11  # Physical pin 11 = GPIO17 (BCM)

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN_BOARD, GPIO.OUT)

    print("Blinking LED on physical pin 11 (3 cycles)...\n")
    for cycle in range(3):
        GPIO.output(LED_PIN_BOARD, GPIO.HIGH)
        print(f"  Cycle {cycle + 1}: LED ON")
        time.sleep(0.5)

        GPIO.output(LED_PIN_BOARD, GPIO.LOW)
        print(f"  Cycle {cycle + 1}: LED OFF")
        time.sleep(0.5)

    GPIO.cleanup()
    print("\nBlink complete.")


# === Exercise 4: systemd Service for Auto-Start ===
# Problem: Create a temperature logger and a systemd service to run it on boot.

def exercise_4():
    """Solution: systemd service for a CPU temperature logger.

    On a real Raspberry Pi, vcgencmd measure_temp returns the SoC temperature.
    Here we simulate the temperature reading for portability.
    """

    # Part 1: The temperature logger script
    # On a real Pi, this would be saved as /home/pi/iot_project/temperature_logger.py

    log_file = "/tmp/temp_log.csv"

    def read_cpu_temperature():
        """Read CPU temperature using vcgencmd (simulated).

        On a real Raspberry Pi:
            result = subprocess.check_output(['vcgencmd', 'measure_temp'])
            temp_str = result.decode('utf-8')  # "temp=42.8'C\n"
            temp = float(temp_str.split('=')[1].split("'")[0])
        """
        import random
        return round(random.uniform(38.0, 55.0), 1)

    def log_temperature(filepath, num_readings=5, interval=2):
        """Log CPU temperature to CSV file.

        In production this would run with interval=30 in an infinite loop.
        For the exercise demo we do a limited number of readings.
        """
        file_exists = os.path.exists(filepath)

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "temperature_c"])

            for i in range(num_readings):
                temp = read_cpu_temperature()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([ts, temp])
                print(f"  [{ts}] Temperature: {temp} C")

                if i < num_readings - 1:
                    time.sleep(interval)

        print(f"\n  Logged {num_readings} readings to {filepath}")

    print("=== systemd Service for Auto-Start ===\n")

    # Part 1: Run the logger
    print("--- Part 1: Temperature Logger ---")
    log_temperature(log_file, num_readings=5, interval=1)

    # Part 2: The systemd service file
    service_content = """\
[Unit]
Description=CPU Temperature Logger
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/iot_project
ExecStart=/usr/bin/python3 /home/pi/iot_project/temperature_logger.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"""

    print("\n--- Part 2: systemd Service File ---")
    print("  File: /etc/systemd/system/temp_logger.service\n")
    for line in service_content.split("\n"):
        print(f"  {line}")

    # Part 3: Commands to enable and verify
    commands = [
        "sudo systemctl daemon-reload",
        "sudo systemctl enable temp_logger.service",
        "sudo systemctl start temp_logger.service",
        "sudo systemctl status temp_logger.service",
        "",
        "# After reboot, verify:",
        "sudo reboot",
        "# ... wait for boot ...",
        "sudo systemctl status temp_logger.service",
        "cat /home/pi/iot_project/temp_log.csv | tail -5",
    ]

    print("\n--- Part 3: Enable and Verify ---")
    for cmd in commands:
        print(f"  {cmd}")

    # Cleanup
    if os.path.exists(log_file):
        os.remove(log_file)


# === Exercise 5: Network Monitoring and Automatic Reconnection ===
# Problem: Write a Python script that monitors connectivity and auto-restarts
# the network on failure, with logging and outage duration tracking.

def exercise_5():
    """Solution: Network connectivity monitor with auto-reconnect.

    On a real Raspberry Pi this would use subprocess to ping and restart
    networking. Here we simulate the ping success/failure cycle.
    """

    class NetworkMonitor:
        """Monitor internet connectivity and auto-restart on failure.

        On a real Pi, check_internet() pings 8.8.8.8 via subprocess.
        restart_network() calls 'sudo systemctl restart networking'.
        """

        def __init__(self, log_file="/tmp/network_log.txt"):
            self.log_file = log_file
            self.disconnected_at = None
            self.total_downtime_seconds = 0.0

        def check_internet(self):
            """Ping 8.8.8.8 to test connectivity.

            Real implementation:
                try:
                    subprocess.check_call(
                        ['ping', '-c', '1', '-W', '2', '8.8.8.8'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return True
                except subprocess.CalledProcessError:
                    return False
            """
            import random
            # Simulate: 80% chance of success, 20% failure
            return random.random() < 0.8

        def restart_network(self):
            """Restart networking service.

            Real implementation:
                subprocess.call(['sudo', 'systemctl', 'restart', 'networking'])
            """
            print("    [ACTION] Restarting networking service...")
            time.sleep(0.5)  # Simulate restart delay
            print("    [ACTION] Network service restarted")

        def log_event(self, message):
            """Append an event to the log file."""
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{ts}] {message}"
            print(f"    {entry}")
            with open(self.log_file, "a") as f:
                f.write(entry + "\n")

        def run(self, check_interval=2, max_checks=10):
            """Main monitoring loop.

            In production: check_interval=60, max_checks=infinite (while True).
            For the exercise demo we limit to max_checks iterations.
            """
            print(f"  Starting network monitor (interval={check_interval}s)\n")

            for check_num in range(1, max_checks + 1):
                connected = self.check_internet()
                ts = datetime.now().strftime("%H:%M:%S")

                if connected:
                    if self.disconnected_at is not None:
                        # We just reconnected after an outage
                        outage_duration = (datetime.now() - self.disconnected_at).total_seconds()
                        self.total_downtime_seconds += outage_duration
                        self.log_event(
                            f"RECONNECTED - Outage lasted {outage_duration:.1f} seconds"
                        )
                        self.disconnected_at = None
                    print(f"  [{ts}] Check {check_num}/{max_checks}: Connected")
                else:
                    if self.disconnected_at is None:
                        # First detection of disconnection
                        self.disconnected_at = datetime.now()
                        self.log_event("DISCONNECTED - Internet unreachable")
                    else:
                        self.log_event("STILL DOWN - Attempting restart...")

                    self.restart_network()

                if check_num < max_checks:
                    time.sleep(check_interval)

            # Summary
            print(f"\n  --- Monitoring Summary ---")
            print(f"  Total checks: {max_checks}")
            print(f"  Total downtime: {self.total_downtime_seconds:.1f} seconds")

            if os.path.exists(self.log_file):
                os.remove(self.log_file)

    # systemd service for the monitor
    service_content = """\
[Unit]
Description=Network Connectivity Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /home/pi/iot_project/network_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target"""

    print("=== Network Monitoring with Auto-Reconnect ===\n")

    monitor = NetworkMonitor()
    monitor.run(check_interval=1, max_checks=10)

    print("\n--- systemd Service for Network Monitor ---")
    print("  File: /etc/systemd/system/network_monitor.service\n")
    for line in service_content.split("\n"):
        print(f"  {line}")

    print("\n  Test procedure:")
    print("    1. sudo systemctl start network_monitor")
    print("    2. sudo ip link set wlan0 down   # Simulate failure")
    print("    3. Wait 60 seconds, check log file")
    print("    4. sudo ip link set wlan0 up     # Restore")
    print("    5. Verify reconnection is logged with outage duration")


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 02: Raspberry Pi Setup - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Model Selection Justification")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Headless Setup and SSH Hardening")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: GPIO Pin Mapping")
    print("-" * 50)
    exercise_3()

    print("\n\n>>> Exercise 4: systemd Service for Auto-Start")
    print("-" * 50)
    exercise_4()

    print("\n\n>>> Exercise 5: Network Monitoring and Auto-Reconnect")
    print("-" * 50)
    exercise_5()
