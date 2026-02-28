# IoT and Embedded Systems Learning Guide

## Introduction

This folder contains systematic learning materials for **IoT (Internet of Things) and Embedded Systems**. It covers Python-based IoT development centered around the Raspberry Pi, encompassing network connectivity, Edge AI, and cloud integration.

### Target Audience

- Developers with knowledge of basic Python syntax
- Engineers interested in building IoT systems
- Beginners starting projects with Raspberry Pi
- Developers interested in edge computing and AI integration

### Differences from C_Programming

| Category | C_Programming | IoT_Embedded |
|----------|---------------|--------------|
| **Language** | C (low-level) | Python (high-level) |
| **Platform** | Arduino, STM32 | Raspberry Pi |
| **Focus** | Hardware control, registers | Network, cloud integration |
| **Communication** | UART, I2C, SPI (low-level) | MQTT, HTTP, BLE (protocols) |
| **AI** | Not included | Edge AI (TFLite, ONNX) |
| **Projects** | Firmware development | IoT system construction |

C_Programming covers low-level hardware control of microcontrollers, while IoT_Embedded teaches how to build network-connected smart systems on Raspberry Pi using Python.

---

## Learning Roadmap

```
                    ┌─────────────────────────────────────┐
                    │          IoT Learning Roadmap        │
                    └─────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
  ┌──────────┐               ┌──────────────┐             ┌──────────────┐
  │ 01. IoT  │               │ 02. Raspberry│             │              │
  │ Overview │──────────────▶│  Pi Setup    │             │              │
  └──────────┘               └──────────────┘             │              │
                                     │                    │              │
                                     ▼                    │              │
                            ┌──────────────┐              │              │
                            │ 03. GPIO     │              │              │
                            │   Control    │              │              │
                            └──────────────┘              │              │
                                     │                    │              │
        ┌────────────────────────────┼────────────────────┤              │
        │                            │                    │              │
        ▼                            ▼                    ▼              │
  ┌──────────┐               ┌──────────────┐      ┌──────────────┐     │
  │ 04. WiFi │               │ 05. BLE      │      │ 06. MQTT     │     │
  │Networking│               │Connectivity  │      │  Protocol    │     │
  └──────────┘               └──────────────┘      └──────────────┘     │
        │                            │                    │              │
        └────────────────────────────┼────────────────────┘              │
                                     │                                   │
                                     ▼                                   │
                            ┌──────────────┐                            │
                            │ 07. HTTP/    │                            │
                            │   REST API   │                            │
                            └──────────────┘                            │
                                     │                                   │
        ┌────────────────────────────┴────────────────────┐              │
        │                                                 │              │
        ▼                                                 ▼              │
  ┌──────────────┐                               ┌──────────────┐       │
  │ 08. Edge AI  │                               │ 09. Edge AI  │       │
  │   TFLite     │                               │    ONNX      │       │
  └──────────────┘                               └──────────────┘       │
        │                                                 │              │
        └─────────────────────┬───────────────────────────┘              │
                              │                                          │
        ┌─────────────────────┼─────────────────────┐                   │
        │                     │                     │                   │
        ▼                     ▼                     ▼                   │
  ┌──────────────┐    ┌──────────────┐     ┌──────────────┐            │
  │ 10. Home     │    │ 11. Image    │     │ 12. Cloud    │◀───────────┘
  │  Automation  │    │   Analysis   │     │  IoT Integra │
  └──────────────┘    └──────────────┘     └──────────────┘
```

---

## File List

| Filename | Difficulty | Topic | Key Content |
|----------|------------|-------|-------------|
| [01_IoT_Overview.md](01_IoT_Overview.md) | ⭐ | IoT Overview | IoT definition, architecture, protocols |
| [02_Raspberry_Pi_Setup.md](02_Raspberry_Pi_Setup.md) | ⭐ | Raspberry Pi Setup | OS installation, SSH, GPIO pinout |
| [03_Python_GPIO_Control.md](03_Python_GPIO_Control.md) | ⭐⭐ | GPIO Control | RPi.GPIO, gpiozero, sensors |
| [04_WiFi_Networking.md](04_WiFi_Networking.md) | ⭐⭐ | WiFi Networking | Sockets, HTTP client |
| [05_BLE_Connectivity.md](05_BLE_Connectivity.md) | ⭐⭐⭐ | BLE Connectivity | GATT, bleak library |
| [06_MQTT_Protocol.md](06_MQTT_Protocol.md) | ⭐⭐ | MQTT Protocol | Mosquitto, paho-mqtt |
| [07_HTTP_REST_for_IoT.md](07_HTTP_REST_for_IoT.md) | ⭐⭐ | HTTP/REST | Flask server, API design |
| [08_Edge_AI_TFLite.md](08_Edge_AI_TFLite.md) | ⭐⭐⭐ | Edge AI (TFLite) | Model conversion, inference |
| [09_Edge_AI_ONNX.md](09_Edge_AI_ONNX.md) | ⭐⭐⭐ | Edge AI (ONNX) | ONNX Runtime, optimization |
| [10_Home_Automation_Project.md](10_Home_Automation_Project.md) | ⭐⭐⭐ | Home Automation | Smart home, MQTT control |
| [11_Image_Analysis_Project.md](11_Image_Analysis_Project.md) | ⭐⭐⭐ | Image Analysis | Pi Camera, object detection |
| [12_Cloud_IoT_Integration.md](12_Cloud_IoT_Integration.md) | ⭐⭐⭐ | Cloud IoT | AWS IoT, GCP Pub/Sub |
| [13_Zigbee_ZWave.md](13_Zigbee_ZWave.md) | ⭐⭐⭐ | Zigbee and Z-Wave | Mesh networking, device pairing, smart home protocols |
| [14_Multi_Sensor_Fusion.md](14_Multi_Sensor_Fusion.md) | ⭐⭐⭐ | Multi-Sensor Fusion | Kalman filter, sensor calibration, data fusion algorithms |

**Difficulty Legend**: ⭐ Beginner | ⭐⭐ Elementary | ⭐⭐⭐ Intermediate

---

## Environment Setup

### Hardware Requirements

- **Raspberry Pi 4 Model B** (recommended, 2GB+ RAM)
- microSD card (32GB or more, Class 10)
- Power adapter (5V 3A USB-C)
- (Optional) Sensor kit, Pi Camera, relay modules

### Software Setup

#### 1. Raspberry Pi OS Installation

```bash
# Use Raspberry Pi Imager (on PC)
# https://www.raspberrypi.com/software/

# Enable SSH: Create ssh file on boot partition
touch /Volumes/boot/ssh  # macOS
# or
touch /media/user/boot/ssh  # Linux
```

#### 2. Python Environment Setup

```bash
# Check Python version
python3 --version  # 3.9+ recommended

# Create virtual environment
python3 -m venv ~/iot-env
source ~/iot-env/bin/activate

# Install base packages
pip install --upgrade pip
pip install RPi.GPIO gpiozero
```

#### 3. IoT Package Installation

```bash
# MQTT
pip install paho-mqtt

# BLE
pip install bleak

# Web server
pip install flask flask-cors

# Edge AI
pip install tflite-runtime  # For Raspberry Pi
pip install onnxruntime

# Camera
pip install picamera2

# Other utilities
pip install requests numpy pillow
```

#### 4. Development Environment (PC)

You can write code directly on the Raspberry Pi or develop on PC and transfer.

```bash
# After installing VS Code Remote SSH extension
# Ctrl+Shift+P > Remote-SSH: Connect to Host
# pi@raspberrypi.local

# Or transfer files with scp
scp script.py pi@raspberrypi.local:~/projects/
```

---

## Related Resources

### Official Documentation

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [gpiozero Documentation](https://gpiozero.readthedocs.io/)
- [paho-mqtt Documentation](https://eclipse.dev/paho/files/paho.mqtt.python/html/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

### Recommended Learning Resources

- [Raspberry Pi Projects](https://projects.raspberrypi.org/)
- [AWS IoT Core Developer Guide](https://docs.aws.amazon.com/iot/)
- [MQTT.org](https://mqtt.org/)

### Related Folders

- [C_Programming](../C_Programming/): Low-level embedded programming (Arduino, C)
- [Python](../Python/): Advanced Python syntax
- [Networking](../Networking/): Network theory
- [Machine_Learning](../Machine_Learning/): Machine learning basics
- [Computer_Vision](../Computer_Vision/): OpenCV and computer vision

---

## Learning Tips

1. **Set up practice environment first**: Complete Raspberry Pi setup before starting learning
2. **Step-by-step progress**: Complete 01-03, then branch into networking (04-07) or AI (08-09)
3. **Project-oriented**: Use projects 10-12 as goals and learn required technologies backwards
4. **Use simulation**: If you don't have hardware, you can use GPIO simulators

---

*Last updated: 2026-02-01*
