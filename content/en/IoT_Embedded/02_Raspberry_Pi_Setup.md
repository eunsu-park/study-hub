# 02. Raspberry Pi Setup

**Previous**: [IoT Overview](./01_IoT_Overview.md) | **Next**: [Python GPIO Control](./03_Python_GPIO_Control.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Select the appropriate Raspberry Pi model for a given IoT project
2. Install Raspberry Pi OS using the Raspberry Pi Imager
3. Configure SSH access and perform initial system setup
4. Navigate the GPIO pin layout and distinguish BCM from BOARD numbering
5. Set up a Python virtual environment with common IoT libraries
6. Create a systemd service for automatic application startup

---

Every IoT project needs a physical platform to run on, and the Raspberry Pi is the most accessible gateway into embedded development. Setting it up properly -- choosing the right model, configuring the OS, and understanding the GPIO pin layout -- ensures a stable foundation for all the sensor, networking, and AI projects that follow in this course.

---

Raspberry Pi is a single-board computer designed for educational and project purposes, providing GPIO pins suitable for IoT development. This lesson covers Raspberry Pi models, OS installation, initial configuration, and GPIO basics.

---

## 1. Raspberry Pi Model Selection

### 1.1 Main Model Comparison

| Model | CPU | RAM | Features | Use Cases |
|-------|-----|-----|----------|-----------|
| **Raspberry Pi 4B** | Cortex-A72 (4-core, 1.5GHz) | 2/4/8GB LPDDR4 | USB 3.0, Gigabit Ethernet, Dual 4K display | General projects, AI inference, media center |
| **Raspberry Pi 3B+** | Cortex-A53 (4-core, 1.4GHz) | 1GB LPDDR2 | USB 2.0, 802.11ac Wi-Fi, Bluetooth 4.2 | Basic IoT projects, learning |
| **Raspberry Pi Zero 2 W** | Cortex-A53 (4-core, 1GHz) | 512MB LPDDR2 | Micro USB, Mini HDMI, Compact size | Wearables, compact projects |
| **Raspberry Pi Pico** | RP2040 (Cortex-M0+, dual-core) | 264KB SRAM | Microcontroller (not Linux), MicroPython | Real-time control, low power projects |

**Selection Guidelines:**
- **General Projects**: Raspberry Pi 4B (2GB or more)
- **Budget/Learning**: Raspberry Pi 3B+
- **Size-Critical Projects**: Raspberry Pi Zero 2 W
- **Real-Time Control**: Raspberry Pi Pico (requires different development approach)

### 1.2 Required Peripherals

| Component | Purpose | Recommendation |
|-----------|---------|----------------|
| **Power Supply** | Pi 4: 5V 3A USB-C, Pi 3: 5V 2.5A Micro USB | Use official adapter |
| **microSD Card** | Operating system storage | 16GB or more, Class 10 or UHS-I |
| **HDMI Cable** | Display output | Micro HDMI (Pi 4), Standard HDMI (Pi 3) |
| **Keyboard/Mouse** | Initial setup | Wired/Wireless both work |
| **Case** | Protective enclosure | Recommend active cooling for Pi 4 |

---

## 2. Raspberry Pi OS Installation

### 2.1 Raspberry Pi Imager Installation

**Download:** [https://www.raspberrypi.org/software/](https://www.raspberrypi.org/software/)

**Install Raspberry Pi Imager:**

```bash
# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install rpi-imager

# macOS (Homebrew)
brew install --cask raspberry-pi-imager

# Windows: Download installer from website
```

### 2.2 OS Image Writing

1. **Launch Raspberry Pi Imager**
2. **Select OS**: Raspberry Pi OS (64-bit) recommended
   - "Raspberry Pi OS (64-bit)": Full desktop environment
   - "Raspberry Pi OS Lite (64-bit)": Server (CLI only)
3. **Select Storage**: Choose microSD card
4. **Advanced Options** (gear icon): Configure initial settings
   - **Enable SSH**: Check
   - **Set username/password**: e.g., username `pi`, password `raspberry`
   - **Configure Wi-Fi**: SSID and password
   - **Set locale**: Timezone and keyboard layout
5. **Write**: Wait 5-10 minutes for completion

### 2.3 First Boot

1. Insert microSD card into Raspberry Pi
2. Connect power
3. Wait 1-2 minutes (first boot takes time for initialization)
4. Desktop will launch (if desktop version)

---

## 3. Initial Configuration

### 3.1 SSH Connection

**Find IP address:**

```bash
# Method 1: On the Pi itself
hostname -I

# Method 2: From another computer on same network
# Linux/Mac
arp -a | grep raspberry

# Or use nmap
sudo nmap -sn 192.168.1.0/24
```

**Connect via SSH:**

```bash
ssh pi@192.168.1.100  # Replace with your Pi's IP
```

### 3.2 System Update

```bash
# Package list update
sudo apt update

# Package upgrade (takes 10-30 minutes on first run)
sudo apt upgrade -y

# Reboot
sudo reboot
```

### 3.3 raspi-config

`raspi-config` is a configuration tool for Raspberry Pi.

```bash
sudo raspi-config
```

**Common Settings:**

| Menu | Option | Description |
|------|--------|-------------|
| **1 System Options** | S3 Password | Change password |
| **1 System Options** | S4 Hostname | Set device name |
| **3 Interface Options** | I2 SSH | Enable/disable SSH |
| **3 Interface Options** | I3 VNC | Enable VNC (remote desktop) |
| **3 Interface Options** | I4 SPI | Enable SPI interface (for sensors) |
| **3 Interface Options** | I5 I2C | Enable I2C interface (for sensors) |
| **5 Localisation Options** | L1 Locale | Set language/region |
| **5 Localisation Options** | L2 Timezone | Set timezone |
| **6 Advanced Options** | A1 Expand Filesystem | Expand to full SD card space |

After configuration, select **Finish** and reboot if prompted.

### 3.4 Useful System Commands

```bash
# System information
uname -a                    # Kernel version
cat /proc/cpuinfo           # CPU information
cat /proc/meminfo           # Memory information
vcgencmd measure_temp       # Current temperature

# Network information
ifconfig                    # Network interface info
iwconfig                    # Wireless interface info
ping -c 4 8.8.8.8          # Test internet connectivity

# Disk usage
df -h                       # Disk space usage
du -sh /home/pi/*          # Folder size

# Process monitoring
top                         # CPU/memory usage (real-time)
htop                        # Enhanced process viewer (install: sudo apt install htop)
```

---

## 4. GPIO Basics

### 4.1 GPIO Pin Layout

Raspberry Pi 4B/3B+ uses a **40-pin header**.

```
3V3  (1)  (2)  5V
GPIO2 (3)  (4)  5V
GPIO3 (5)  (6)  GND
GPIO4 (7)  (8)  GPIO14 (TXD)
GND   (9)  (10) GPIO15 (RXD)
GPIO17 (11) (12) GPIO18 (PWM0)
GPIO27 (13) (14) GND
GPIO22 (15) (16) GPIO23
3V3  (17) (18) GPIO24
GPIO10 (19) (20) GND
GPIO9 (21) (22) GPIO25
GPIO11 (23) (24) GPIO8
GND  (25) (26) GPIO7
GPIO0 (27) (28) GPIO1
GPIO5 (29) (30) GND
GPIO6 (31) (32) GPIO12 (PWM0)
GPIO13 (33) (34) GND
GPIO19 (35) (36) GPIO16
GPIO26 (37) (38) GPIO20
GND  (39) (40) GPIO21
```

**Pin Types:**
- **GPIO**: Digital input/output pins
- **3V3**: 3.3V power (max 50mA total)
- **5V**: 5V power (max depends on power supply)
- **GND**: Ground
- **Special Functions**: I2C (GPIO2/3), SPI (GPIO9/10/11), UART (GPIO14/15), PWM (GPIO12/13/18/19)

### 4.2 GPIO Numbering Schemes

There are two numbering schemes:

1. **BCM (Broadcom SOC channel)**: Uses GPIO numbers (e.g., GPIO17, GPIO27)
2. **BOARD (Physical pin)**: Uses physical pin numbers (e.g., Pin 11, Pin 13)

**Example:**
- Physical Pin 11 = GPIO17 (BCM)
- Physical Pin 13 = GPIO27 (BCM)

**Most Python libraries use BCM numbering by default.**

### 4.3 GPIO Test: LED Blink

**Circuit:**
- LED anode (+) → GPIO17 (Pin 11)
- LED cathode (-) → 220Ω resistor → GND (Pin 9)

```python
import RPi.GPIO as GPIO
import time

# Pin setup
LED_PIN = 17  # GPIO17 (Physical Pin 11)

GPIO.setmode(GPIO.BCM)           # Use BCM numbering
GPIO.setup(LED_PIN, GPIO.OUT)    # Set as output

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)  # LED ON
        print("LED ON")
        time.sleep(1)

        GPIO.output(LED_PIN, GPIO.LOW)   # LED OFF
        print("LED OFF")
        time.sleep(1)
except KeyboardInterrupt:
    print("\nProgram stopped by user")
finally:
    GPIO.cleanup()  # Reset GPIO state
```

**Run:**

```bash
python3 led_blink.py
```

---

## 5. Python Environment Setup

### 5.1 Python Version Check

Raspberry Pi OS includes Python 3 by default.

```bash
python3 --version  # Example output: Python 3.9.2
```

### 5.2 pip Installation

```bash
# Update pip
sudo apt install python3-pip

# Verify installation
pip3 --version
```

### 5.3 Virtual Environment Setup

Use virtual environments to avoid package conflicts.

```bash
# Install venv
sudo apt install python3-venv

# Create virtual environment
python3 -m venv ~/iot_env

# Activate
source ~/iot_env/bin/activate

# Deactivate
deactivate
```

### 5.4 Common Library Installation

```bash
# GPIO control
pip3 install RPi.GPIO
pip3 install gpiozero

# Sensor libraries
pip3 install adafruit-circuitpython-dht  # DHT11/22 temperature/humidity sensor
pip3 install adafruit-blinka              # CircuitPython compatibility layer

# Communication
pip3 install paho-mqtt    # MQTT client
pip3 install requests     # HTTP client

# Image processing
pip3 install opencv-python
pip3 install pillow

# Machine learning
pip3 install tflite-runtime  # TensorFlow Lite runtime
```

---

## 6. System Management

### 6.1 Automatic Startup Settings

**Using systemd:**

Create service file:

```bash
sudo nano /etc/systemd/system/iot_app.service
```

```ini
[Unit]
Description=IoT Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/iot_project
ExecStart=/usr/bin/python3 /home/pi/iot_project/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable iot_app.service
sudo systemctl start iot_app.service

# Check status
sudo systemctl status iot_app.service

# View logs
sudo journalctl -u iot_app.service -f
```

### 6.2 Automatic Reconnection Settings

**WiFi Auto-Reconnect:**

Add to `/etc/wpa_supplicant/wpa_supplicant.conf`:

```
network={
    ssid="YourSSID"
    psk="YourPassword"
    id_str="home"
    priority=1
}

network={
    ssid="BackupSSID"
    psk="BackupPassword"
    id_str="backup"
    priority=2
}
```

**Network monitoring script:**

```python
import subprocess
import time

def check_internet():
    try:
        subprocess.check_call(['ping', '-c', '1', '8.8.8.8'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def restart_network():
    print("Network disconnected. Restarting...")
    subprocess.call(['sudo', 'systemctl', 'restart', 'networking'])

while True:
    if not check_internet():
        restart_network()
    time.sleep(60)  # Check every 60 seconds
```

---

## 7. Summary

### Completed Tasks

- ✅ **Raspberry Pi Model Selection**: Understand different models and selection criteria
- ✅ **OS Installation**: Write Raspberry Pi OS to SD card using Raspberry Pi Imager
- ✅ **Initial Configuration**: SSH connection, system update, raspi-config settings
- ✅ **GPIO Basics**: Pin layout, numbering schemes (BCM/BOARD)
- ✅ **LED Blink Test**: Basic GPIO output control
- ✅ **Python Environment**: Virtual environment and library installation
- ✅ **System Management**: Systemd service setup, automatic startup

### Hands-On Exercise

1. **LED Control**:
   - Connect LED to GPIO17
   - Implement blinking pattern (0.5 second intervals)
   - Add variable speed control

2. **Button Input**:
   - Connect button to GPIO27 (with pull-down resistor)
   - Print message when button pressed
   - Toggle LED on button press

3. **Temperature Monitoring**:
   - Use `vcgencmd measure_temp` to read CPU temperature
   - Log to file every 5 minutes
   - Send alert if temperature exceeds 70°C

### Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| **Cannot SSH connect** | Incorrect IP, SSH not enabled | Check IP with `hostname -I`, enable SSH in raspi-config |
| **Insufficient power warning** | Power supply insufficient | Use official 5V 3A adapter |
| **GPIO not working** | Wrong pin number, incorrect permissions | Check BCM vs BOARD numbering, run with `sudo` |
| **SD card wear** | Frequent writes | Use log rotation, write to RAM disk (tmpfs) |

---

## References

- [Raspberry Pi Official Documentation](https://www.raspberrypi.org/documentation/)
- [GPIO Pinout Reference](https://pinout.xyz/)
- [RPi.GPIO Documentation](https://sourceforge.net/p/raspberry-gpio-python/wiki/Home/)
- [systemd Service Configuration Guide](https://www.freedesktop.org/software/systemd/man/systemd.service.html)

---

## Exercises

### Exercise 1: Model Selection Justification

You are planning three separate IoT deployments. For each scenario below, identify the most appropriate Raspberry Pi model and explain your reasoning:

1. A portable wearable that monitors body temperature and transmits to a smartphone. Size and power consumption are critical.
2. A gateway device that collects data from 20 sensors, runs a local MQTT broker, and streams video to a cloud server.
3. A learning project where you want to experiment with GPIOs, Python, and a desktop GUI — budget is limited.

Write your answers as a short paragraph for each scenario, referencing the model comparison table in Section 1.

### Exercise 2: Headless Setup and SSH Hardening

Set up a Raspberry Pi in headless mode (no monitor or keyboard) and harden SSH access:

1. Use Raspberry Pi Imager to write Raspberry Pi OS Lite to an SD card with SSH enabled and WiFi pre-configured.
2. Boot the Pi, find its IP address using `nmap` or `arp`, and connect via SSH.
3. Generate an Ed25519 key pair on your host machine and copy the public key to the Pi using `ssh-copy-id`.
4. Edit `/etc/ssh/sshd_config` to disable password authentication (`PasswordAuthentication no`) and restart the SSH service.
5. Verify that you can still log in using the key, and that password login is rejected.

### Exercise 3: GPIO Pin Mapping

Study the 40-pin GPIO header layout in Section 4.1 and answer the following without looking at any reference:

1. Which physical pin numbers carry 5V power? Which carry 3.3V?
2. GPIO17 uses BCM numbering. What is its physical (BOARD) pin number?
3. Which GPIO pins support hardware PWM?
4. Which pins are used for I2C communication, and what are their BCM numbers?

Then write a short Python snippet that uses `RPi.GPIO` in BOARD mode to blink an LED connected to physical pin 11.

### Exercise 4: systemd Service for Auto-Start

Create a systemd service that runs a Python sensor script automatically on boot:

1. Write a Python script (`/home/pi/iot_project/temperature_logger.py`) that reads the CPU temperature using `vcgencmd measure_temp`, logs it with a timestamp to `/home/pi/iot_project/temp_log.csv`, and sleeps for 30 seconds in a loop.
2. Create a systemd service file at `/etc/systemd/system/temp_logger.service` referencing that script.
3. Enable and start the service, then verify it is running with `systemctl status`.
4. Reboot the Pi and confirm the service starts automatically by checking the log file after boot.

### Exercise 5: Network Monitoring and Automatic Reconnection

Write a Python script that monitors internet connectivity and automatically restarts the network if it goes down:

1. Use `subprocess` to ping `8.8.8.8` every 60 seconds.
2. If the ping fails, log the disconnection time to a file and restart the networking service using `sudo systemctl restart networking`.
3. After reconnection, log the reconnection time and calculate how long the outage lasted.
4. Wrap the script in a systemd service so it runs continuously in the background. Test by temporarily disabling the WiFi interface with `sudo ip link set wlan0 down`.

---

**Previous**: [IoT Overview](./01_IoT_Overview.md) | **Next**: [Python GPIO Control](./03_Python_GPIO_Control.md)
