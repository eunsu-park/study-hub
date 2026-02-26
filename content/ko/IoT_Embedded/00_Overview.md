# IoT와 임베디드 시스템 학습 가이드

## 소개

이 폴더는 **IoT(사물인터넷)와 임베디드 시스템**에 대한 체계적인 학습 자료를 담고 있습니다. 라즈베리파이를 중심으로 Python 기반의 IoT 개발을 다루며, 네트워크 연결, 엣지 AI, 클라우드 통합까지 포괄합니다.

### 대상 독자

- Python 기본 문법을 알고 있는 개발자
- IoT 시스템 구축에 관심 있는 엔지니어
- 라즈베리파이로 프로젝트를 시작하려는 입문자
- 엣지 컴퓨팅과 AI 통합에 관심 있는 개발자

### C_Programming과의 차별점

| 구분 | C_Programming | IoT_Embedded |
|------|---------------|--------------|
| **언어** | C (저수준) | Python (고수준) |
| **플랫폼** | Arduino, STM32 | Raspberry Pi |
| **초점** | 하드웨어 제어, 레지스터 | 네트워크, 클라우드 연동 |
| **통신** | UART, I2C, SPI (저수준) | MQTT, HTTP, BLE (프로토콜) |
| **AI** | 미포함 | Edge AI (TFLite, ONNX) |
| **프로젝트** | 펌웨어 개발 | IoT 시스템 구축 |

C_Programming은 마이크로컨트롤러의 저수준 하드웨어 제어를 다루고, IoT_Embedded는 라즈베리파이에서 Python으로 네트워크 연결된 스마트 시스템을 구축하는 방법을 다룹니다.

---

## 학습 로드맵

```
                    ┌─────────────────────────────────────┐
                    │         IoT 학습 로드맵              │
                    └─────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
  ┌──────────┐               ┌──────────────┐             ┌──────────────┐
  │ 01. IoT  │               │ 02. 라즈베리 │             │              │
  │   개요   │──────────────▶│  파이 설정   │             │              │
  └──────────┘               └──────────────┘             │              │
                                     │                    │              │
                                     ▼                    │              │
                            ┌──────────────┐              │              │
                            │ 03. GPIO     │              │              │
                            │    제어      │              │              │
                            └──────────────┘              │              │
                                     │                    │              │
        ┌────────────────────────────┼────────────────────┤              │
        │                            │                    │              │
        ▼                            ▼                    ▼              │
  ┌──────────┐               ┌──────────────┐      ┌──────────────┐     │
  │ 04. WiFi │               │ 05. BLE      │      │ 06. MQTT     │     │
  │ 네트워킹 │               │   연결       │      │  프로토콜    │     │
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
  │ 10. 홈       │    │ 11. 영상     │     │ 12. 클라우드 │◀───────────┘
  │  자동화      │    │   분석       │     │    IoT 통합  │
  └──────────────┘    └──────────────┘     └──────────────┘
```

---

## 파일 목록

| 파일명 | 난이도 | 주제 | 핵심 내용 |
|--------|--------|------|-----------|
| [IoT 개요](01_IoT_Overview.md) | ⭐ | IoT 개요 | IoT 정의, 아키텍처, 프로토콜 |
| [라즈베리파이 설정](02_Raspberry_Pi_Setup.md) | ⭐ | 라즈베리파이 설정 | OS 설치, SSH, GPIO 핀아웃 |
| [Python GPIO 제어](03_Python_GPIO_Control.md) | ⭐⭐ | GPIO 제어 | RPi.GPIO, gpiozero, 센서 |
| [WiFi 네트워킹](04_WiFi_Networking.md) | ⭐⭐ | WiFi 네트워킹 | 소켓, HTTP 클라이언트 |
| [BLE 연결](05_BLE_Connectivity.md) | ⭐⭐⭐ | BLE 연결 | GATT, bleak 라이브러리 |
| [MQTT 프로토콜](06_MQTT_Protocol.md) | ⭐⭐ | MQTT 프로토콜 | Mosquitto, paho-mqtt |
| [HTTP/REST for IoT](07_HTTP_REST_for_IoT.md) | ⭐⭐ | HTTP/REST | Flask 서버, API 설계 |
| [Edge AI - TensorFlow Lite](08_Edge_AI_TFLite.md) | ⭐⭐⭐ | Edge AI (TFLite) | 모델 변환, 추론 |
| [Edge AI - ONNX Runtime](09_Edge_AI_ONNX.md) | ⭐⭐⭐ | Edge AI (ONNX) | ONNX Runtime, 최적화 |
| [홈 자동화 프로젝트](10_Home_Automation_Project.md) | ⭐⭐⭐ | 홈 자동화 | 스마트홈, MQTT 제어 |
| [영상 분석 프로젝트](11_Image_Analysis_Project.md) | ⭐⭐⭐ | 영상 분석 | Pi Camera, 객체 검출 |
| [클라우드 IoT 통합](12_Cloud_IoT_Integration.md) | ⭐⭐⭐ | 클라우드 IoT | AWS IoT, GCP Pub/Sub |

**난이도 범례**: ⭐ 입문 | ⭐⭐ 초급 | ⭐⭐⭐ 중급

---

## 환경 설정

### 하드웨어 요구사항

- **Raspberry Pi 4 Model B** (권장, 2GB+ RAM)
- microSD 카드 (32GB 이상, Class 10)
- 전원 어댑터 (5V 3A USB-C)
- (선택) 센서 키트, Pi Camera, 릴레이 모듈

### 소프트웨어 설정

#### 1. Raspberry Pi OS 설치

```bash
# Raspberry Pi Imager 사용 (PC에서)
# https://www.raspberrypi.com/software/

# SSH 활성화: boot 파티션에 ssh 파일 생성
touch /Volumes/boot/ssh  # macOS
# 또는
touch /media/user/boot/ssh  # Linux
```

#### 2. Python 환경 설정

```bash
# Python 버전 확인
python3 --version  # 3.9+ 권장

# 가상환경 생성
python3 -m venv ~/iot-env
source ~/iot-env/bin/activate

# 기본 패키지 설치
pip install --upgrade pip
pip install RPi.GPIO gpiozero
```

#### 3. IoT 패키지 설치

```bash
# MQTT
pip install paho-mqtt

# BLE
pip install bleak

# 웹 서버
pip install flask flask-cors

# Edge AI
pip install tflite-runtime  # 라즈베리파이용
pip install onnxruntime

# 카메라
pip install picamera2

# 기타 유틸리티
pip install requests numpy pillow
```

#### 4. 개발 환경 (PC)

라즈베리파이에 직접 코드를 작성하거나, PC에서 개발 후 전송할 수 있습니다.

```bash
# VS Code Remote SSH 확장 설치 후
# Ctrl+Shift+P > Remote-SSH: Connect to Host
# pi@raspberrypi.local

# 또는 scp로 파일 전송
scp script.py pi@raspberrypi.local:~/projects/
```

---

## 관련 자료

### 공식 문서

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [gpiozero Documentation](https://gpiozero.readthedocs.io/)
- [paho-mqtt Documentation](https://eclipse.dev/paho/files/paho.mqtt.python/html/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

### 추천 학습 자료

- [Raspberry Pi Projects](https://projects.raspberrypi.org/)
- [AWS IoT Core Developer Guide](https://docs.aws.amazon.com/iot/)
- [MQTT.org](https://mqtt.org/)

### 관련 폴더

- [C_Programming](../C_Programming/): 저수준 임베디드 프로그래밍 (Arduino, C)
- [Python](../Python/): Python 고급 문법
- [Networking](../Networking/): 네트워크 이론
- [Machine_Learning](../Machine_Learning/): 머신러닝 기초
- [Computer_Vision](../Computer_Vision/): OpenCV와 컴퓨터 비전

---

## 학습 팁

1. **실습 환경 구축이 먼저**: 라즈베리파이 설정을 완료한 후 학습 시작
2. **단계별 진행**: 01-03을 완료한 후 네트워킹(04-07) 또는 AI(08-09)로 분기
3. **프로젝트 중심**: 10-12의 프로젝트를 목표로 필요한 기술을 역으로 학습
4. **시뮬레이션 활용**: 하드웨어가 없다면 GPIO 시뮬레이터 사용 가능

---

*최종 업데이트: 2026-02-01*
