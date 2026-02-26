# 03. Python GPIO 제어

**이전**: [Raspberry Pi 설정](./02_Raspberry_Pi_Setup.md) | **다음**: [WiFi 네트워킹](./04_WiFi_Networking.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 프로젝트 요구사항에 따라 RPi.GPIO와 gpiozero 라이브러리 중 적합한 것을 선택할 수 있다
2. 두 라이브러리를 사용하여 디지털 출력(LED)을 제어할 수 있다
3. 폴링과 인터럽트 방식으로 디지털 입력(버튼)을 읽을 수 있다
4. LED 밝기 조절 등 아날로그적 제어를 위한 PWM을 구현할 수 있다
5. 일반 센서(DHT11, PIR, HC-SR04)와 인터페이스할 수 있다
6. 통합 다중 센서 모니터링 시스템을 구축할 수 있다
7. GPIO 정리(cleanup), 디바운싱(debouncing), 오류 처리 모범 사례를 적용할 수 있다

---

GPIO 핀은 Raspberry Pi를 소형 컴퓨터에서 강력한 IoT 컨트롤러로 변환시켜 줍니다. Python 코드로 센서를 읽고 액추에이터를 제어하는 법을 배우는 것은 소프트웨어와 물리 세계를 연결하는 다리입니다. 이는 이후의 네트워킹, MQTT, 엣지 AI 레슨을 순전히 이론에 그치지 않고 실용적으로 만들어 주는 핵심 기술입니다.

---

## 1. GPIO 라이브러리 개요

### 1.1 라이브러리 비교

| 라이브러리 | 특징 | 난이도 | 권장 용도 |
|-----------|------|--------|-----------|
| **RPi.GPIO** | 저수준, 세밀한 제어 | 중급 | 정밀 제어, 타이밍 |
| **gpiozero** | 고수준, 직관적 API | 입문 | 교육, 빠른 프로토타이핑 |
| **pigpio** | 원격 제어, 정밀 타이밍 | 고급 | 서보, 정밀 PWM |
| **lgpio** | 최신, Pi 5 지원 | 중급 | Pi 5 프로젝트 |

### 1.2 설치

```bash
# RPi.GPIO (보통 기본 설치됨)
sudo apt install python3-rpi.gpio

# gpiozero (권장)
sudo apt install python3-gpiozero

# pigpio (정밀 타이밍 필요시)
sudo apt install pigpio python3-pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

---

## 2. RPi.GPIO 라이브러리

### 2.1 기본 구조

```python
#!/usr/bin/env python3
"""RPi.GPIO 기본 구조"""

import RPi.GPIO as GPIO
import time

# 1. 핀 번호 체계 설정
GPIO.setmode(GPIO.BCM)  # BCM 번호 사용 (GPIO 번호)
# GPIO.setmode(GPIO.BOARD)  # 물리적 핀 번호 사용

# 2. 경고 메시지 비활성화 (선택)
GPIO.setwarnings(False)

# 3. 핀 설정
LED_PIN = 17
GPIO.setup(LED_PIN, GPIO.OUT)  # 출력 핀

BUTTON_PIN = 27
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # 입력 핀 (풀업)

try:
    # 4. GPIO 사용
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(LED_PIN, GPIO.LOW)

finally:
    # 5. 정리 (필수!)
    GPIO.cleanup()
```

### 2.2 LED 제어 (디지털 출력)

```python
#!/usr/bin/env python3
"""LED 깜빡이기 (RPi.GPIO)"""

import RPi.GPIO as GPIO
import time

LED_PIN = 17

def blink_led(times: int = 5, interval: float = 0.5):
    """LED를 지정 횟수만큼 깜빡임"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)

    try:
        for i in range(times):
            print(f"Blink {i + 1}/{times}")
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(interval)
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    blink_led(10, 0.3)
```

### 2.3 PWM 출력 (밝기 조절)

```python
#!/usr/bin/env python3
"""LED 밝기 조절 - PWM (RPi.GPIO)"""

import RPi.GPIO as GPIO
import time

LED_PIN = 18  # 하드웨어 PWM 지원 핀 권장

def fade_led():
    """LED 페이드 인/아웃"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)

    # PWM 객체 생성 (핀, 주파수 Hz)
    pwm = GPIO.PWM(LED_PIN, 1000)  # 1kHz
    pwm.start(0)  # 듀티 사이클 0%로 시작

    try:
        while True:
            # 페이드 인 (0% -> 100%)
            for duty in range(0, 101, 5):
                pwm.ChangeDutyCycle(duty)
                time.sleep(0.05)

            # 페이드 아웃 (100% -> 0%)
            for duty in range(100, -1, -5):
                pwm.ChangeDutyCycle(duty)
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    fade_led()
```

### 2.4 버튼 입력 (폴링 방식)

```python
#!/usr/bin/env python3
"""버튼 입력 - 폴링 방식 (RPi.GPIO)"""

import RPi.GPIO as GPIO
import time

BUTTON_PIN = 27
LED_PIN = 17

def polling_button():
    """폴링으로 버튼 상태 확인"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED_PIN, GPIO.OUT)

    print("버튼을 누르면 LED가 켜집니다. Ctrl+C로 종료.")

    try:
        while True:
            # 풀업이므로 버튼 누르면 LOW
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("버튼 눌림!")
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.1)  # 디바운싱

    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    polling_button()
```

### 2.5 버튼 입력 (인터럽트 방식)

```python
#!/usr/bin/env python3
"""버튼 입력 - 인터럽트 방식 (RPi.GPIO)"""

import RPi.GPIO as GPIO
import time

BUTTON_PIN = 27
LED_PIN = 17
led_state = False

def button_callback(channel):
    """버튼 눌림 콜백 함수"""
    global led_state
    led_state = not led_state
    GPIO.output(LED_PIN, led_state)
    print(f"LED {'ON' if led_state else 'OFF'}")

def interrupt_button():
    """인터럽트로 버튼 감지"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED_PIN, GPIO.OUT)

    # 인터럽트 등록 (하강 에지, 디바운싱 200ms)
    GPIO.add_event_detect(
        BUTTON_PIN,
        GPIO.FALLING,
        callback=button_callback,
        bouncetime=200
    )

    print("버튼을 누르면 LED가 토글됩니다. Ctrl+C로 종료.")

    try:
        while True:
            time.sleep(1)  # 메인 루프는 다른 작업 가능
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    interrupt_button()
```

---

## 3. gpiozero 라이브러리

### 3.1 gpiozero 장점

```
┌─────────────────────────────────────────────────────────────┐
│                    gpiozero 특징                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✓ 객체 지향적 API                                          │
│  ✓ 자동 cleanup (with 문 또는 종료시)                       │
│  ✓ 다양한 장치 추상화 (LED, Button, Sensor 등)              │
│  ✓ 가독성 높은 코드                                         │
│  ✓ 원격 GPIO 지원 (다른 Pi 제어)                            │
│  ✓ Mock 핀 지원 (테스트용)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 LED 제어

```python
#!/usr/bin/env python3
"""LED 제어 (gpiozero)"""

from gpiozero import LED
from time import sleep

# LED 객체 생성 (GPIO 17)
led = LED(17)

# 기본 제어
led.on()        # 켜기
sleep(1)
led.off()       # 끄기
sleep(1)

# 토글
led.toggle()    # 상태 반전
sleep(1)
led.toggle()

# 깜빡이기 (비동기)
led.blink(on_time=0.5, off_time=0.5, n=5, background=True)
sleep(6)

# 스크립트 종료 시 자동 cleanup
```

### 3.3 PWM LED

```python
#!/usr/bin/env python3
"""PWM LED 제어 (gpiozero)"""

from gpiozero import PWMLED
from time import sleep

led = PWMLED(18)

# 밝기 직접 설정 (0.0 ~ 1.0)
led.value = 0.5  # 50% 밝기
sleep(1)

led.value = 1.0  # 100% 밝기
sleep(1)

# 페이드 효과 (pulse)
# 기본값: fade_in_time=1, fade_out_time=1
led.pulse(fade_in_time=2, fade_out_time=2)
sleep(10)

# 수동 페이드
for brightness in range(0, 101, 10):
    led.value = brightness / 100
    sleep(0.1)
```

### 3.4 버튼 처리

```python
#!/usr/bin/env python3
"""버튼 제어 (gpiozero)"""

from gpiozero import Button, LED
from signal import pause

led = LED(17)
button = Button(27, pull_up=True, bounce_time=0.2)

# 방법 1: 콜백 함수
def on_pressed():
    print("버튼 눌림!")
    led.on()

def on_released():
    print("버튼 떼짐!")
    led.off()

button.when_pressed = on_pressed
button.when_released = on_released

# 방법 2: LED와 직접 연결
# led.source = button  # 버튼 누르면 LED 켜짐

print("버튼을 누르세요. Ctrl+C로 종료.")
pause()  # 프로그램 유지
```

### 3.5 버튼으로 LED 토글

```python
#!/usr/bin/env python3
"""버튼으로 LED 토글 (gpiozero)"""

from gpiozero import Button, LED
from signal import pause

led = LED(17)
button = Button(27, bounce_time=0.2)

# 버튼 누를 때마다 LED 토글
button.when_pressed = led.toggle

print("버튼을 누르면 LED가 토글됩니다.")
pause()
```

### 3.6 다중 LED 제어

```python
#!/usr/bin/env python3
"""다중 LED 제어 (gpiozero)"""

from gpiozero import LEDBoard, LED
from time import sleep
from signal import pause

# 개별 LED
leds = [LED(pin) for pin in [17, 27, 22]]

# 순차 점등
for i, led in enumerate(leds):
    led.on()
    sleep(0.5)
    led.off()

# LEDBoard 사용
led_board = LEDBoard(17, 27, 22)

led_board.on()       # 모두 켜기
sleep(1)
led_board.off()      # 모두 끄기
sleep(1)

# 깜빡이기
led_board.blink(on_time=0.2, off_time=0.2, n=5)
sleep(3)

# 값 설정 (개별 제어)
led_board.value = (1, 0, 1)  # 첫째, 셋째만 켜기
sleep(1)
```

---

## 4. 센서 연결

### 4.1 DHT11 온습도 센서

```
┌─────────────────────────────────────────────────────────────┐
│                   DHT11 연결도                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   DHT11 핀           Raspberry Pi                            │
│   ┌─────────┐                                                │
│   │ VCC (+) │ ───────── 3.3V (핀 1)                         │
│   │ DATA    │ ───────── GPIO4 (핀 7) + 10kΩ 풀업           │
│   │ NC      │ ───────── 연결 안함                           │
│   │ GND (-) │ ───────── GND (핀 6)                          │
│   └─────────┘                                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
#!/usr/bin/env python3
"""DHT11 온습도 센서 읽기"""

import time

# adafruit-circuitpython-dht 라이브러리 사용
# pip install adafruit-circuitpython-dht
# sudo apt install libgpiod2

import adafruit_dht
import board

# DHT11 센서 초기화 (GPIO4)
dht = adafruit_dht.DHT11(board.D4)

def read_dht11():
    """DHT11 센서 데이터 읽기"""
    try:
        temperature = dht.temperature
        humidity = dht.humidity

        if humidity is not None and temperature is not None:
            return {
                "temperature": temperature,
                "humidity": humidity,
                "status": "ok"
            }
        else:
            return {"status": "error", "message": "Failed to read"}

    except RuntimeError as e:
        # DHT 센서는 가끔 읽기 실패함 (정상)
        return {"status": "error", "message": str(e)}

def monitor_environment(interval: int = 5):
    """환경 모니터링"""
    print("온습도 모니터링 시작 (Ctrl+C로 종료)")

    while True:
        data = read_dht11()

        if data["status"] == "ok":
            print(f"온도: {data['temperature']:.1f}°C, "
                  f"습도: {data['humidity']:.1f}%")
        else:
            print(f"읽기 실패: {data.get('message', 'Unknown error')}")

        time.sleep(interval)

if __name__ == "__main__":
    try:
        monitor_environment(3)
    except KeyboardInterrupt:
        print("\n모니터링 종료")
    finally:
        dht.exit()
```

### 4.2 PIR 모션 센서

```
┌─────────────────────────────────────────────────────────────┐
│                   PIR 센서 연결도                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   PIR 핀             Raspberry Pi                            │
│   ┌─────────┐                                                │
│   │ VCC     │ ───────── 5V (핀 2)                           │
│   │ OUT     │ ───────── GPIO17 (핀 11)                      │
│   │ GND     │ ───────── GND (핀 6)                          │
│   └─────────┘                                                │
│                                                              │
│   * 감도/지연 조절: 센서 뒷면 가변저항 조절                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
#!/usr/bin/env python3
"""PIR 모션 센서 (gpiozero)"""

from gpiozero import MotionSensor, LED
from datetime import datetime
from signal import pause

pir = MotionSensor(17)
led = LED(27)

def motion_detected():
    """모션 감지 시 호출"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] 모션 감지!")
    led.on()

def motion_ended():
    """모션 종료 시 호출"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] 모션 종료")
    led.off()

pir.when_motion = motion_detected
pir.when_no_motion = motion_ended

print("PIR 모션 감지 시작...")
print("센서 안정화 대기 중 (약 10초)...")
pir.wait_for_no_motion()
print("준비 완료! 모션을 감지합니다.")

pause()
```

### 4.3 초음파 거리 센서 (HC-SR04)

```python
#!/usr/bin/env python3
"""HC-SR04 초음파 거리 센서 (gpiozero)"""

from gpiozero import DistanceSensor
from time import sleep

# TRIGGER: GPIO23, ECHO: GPIO24
# 주의: ECHO 핀은 5V 출력이므로 분압 회로 필요!
sensor = DistanceSensor(echo=24, trigger=23, max_distance=4)

def measure_distance():
    """거리 측정"""
    distance_m = sensor.distance
    distance_cm = distance_m * 100
    return distance_cm

def proximity_monitor(threshold_cm: float = 30):
    """근접 감지 모니터링"""
    print(f"거리 임계값: {threshold_cm}cm")

    while True:
        distance = measure_distance()

        if distance < threshold_cm:
            print(f"근접 감지! 거리: {distance:.1f}cm")
        else:
            print(f"거리: {distance:.1f}cm")

        sleep(0.5)

if __name__ == "__main__":
    try:
        proximity_monitor()
    except KeyboardInterrupt:
        print("\n종료")
```

---

## 5. 종합 예제: 센서 모니터링 시스템

```python
#!/usr/bin/env python3
"""종합 센서 모니터링 시스템"""

from gpiozero import Button, LED, MotionSensor
from datetime import datetime
import time
import json

class IoTSensorSystem:
    """IoT 센서 모니터링 시스템"""

    def __init__(self):
        # GPIO 설정
        self.led_status = LED(17)
        self.led_alarm = LED(27)
        self.button = Button(22, bounce_time=0.2)
        self.pir = MotionSensor(23)

        # 상태
        self.is_armed = False
        self.motion_count = 0
        self.last_motion = None

        # 콜백 설정
        self.button.when_pressed = self.toggle_arm
        self.pir.when_motion = self.on_motion

    def toggle_arm(self):
        """시스템 활성화/비활성화 토글"""
        self.is_armed = not self.is_armed
        self.led_status.value = self.is_armed

        status = "활성화" if self.is_armed else "비활성화"
        print(f"[시스템] {status}")

    def on_motion(self):
        """모션 감지 핸들러"""
        self.last_motion = datetime.now()
        self.motion_count += 1

        if self.is_armed:
            print(f"[경고] 모션 감지! (총 {self.motion_count}회)")
            self.trigger_alarm()
        else:
            print(f"[정보] 모션 감지 (시스템 비활성)")

    def trigger_alarm(self):
        """알람 트리거"""
        # LED 깜빡임
        self.led_alarm.blink(on_time=0.1, off_time=0.1, n=10)

    def get_status(self) -> dict:
        """현재 상태 반환"""
        return {
            "is_armed": self.is_armed,
            "motion_count": self.motion_count,
            "last_motion": self.last_motion.isoformat() if self.last_motion else None,
            "timestamp": datetime.now().isoformat()
        }

    def run(self):
        """메인 루프"""
        print("=== IoT 센서 시스템 시작 ===")
        print("버튼을 눌러 시스템을 활성화/비활성화합니다.")

        try:
            while True:
                # 주기적으로 상태 출력
                status = self.get_status()
                print(f"\r상태: {json.dumps(status, ensure_ascii=False)}", end="")
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n시스템 종료")
            self.cleanup()

    def cleanup(self):
        """정리"""
        self.led_status.off()
        self.led_alarm.off()

if __name__ == "__main__":
    system = IoTSensorSystem()
    system.run()
```

---

## 6. 시뮬레이션 (하드웨어 없이)

### 6.1 Mock 핀 팩토리

```python
#!/usr/bin/env python3
"""하드웨어 없이 GPIO 시뮬레이션"""

from gpiozero import Device, LED, Button
from gpiozero.pins.mock import MockFactory

# Mock 핀 팩토리 설정
Device.pin_factory = MockFactory()

led = LED(17)
button = Button(27)

# LED 제어
print(f"LED 초기 상태: {led.is_lit}")
led.on()
print(f"LED on 후: {led.is_lit}")
led.off()
print(f"LED off 후: {led.is_lit}")

# 버튼 시뮬레이션
print(f"\n버튼 초기 상태: {button.is_pressed}")

# 버튼 핀을 직접 조작하여 누름 시뮬레이션
button.pin.drive_low()
print(f"버튼 누름: {button.is_pressed}")

button.pin.drive_high()
print(f"버튼 떼기: {button.is_pressed}")
```

---

## 연습 문제

### 연습 1: 교통 신호등 컨트롤러(Traffic Light Controller)

LED 3개(빨강: GPIO17, 노랑: GPIO27, 초록: GPIO22)를 사용하여 교통 신호등 시스템을 만드세요:

1. 각 LED를 220Ω 저항을 통해 GND에 연결합니다.
2. 표준 교통 신호등 순서를 구현합니다: 빨강(3초) → 빨강+노랑(1초) → 초록(3초) → 노랑(1초) → 반복.
3. `gpiozero` 라이브러리를 사용하고 `while True` 루프 내에서 순서를 구현하며, `finally` 블록에서 적절히 정리(cleanup)합니다.
4. GPIO4에 보행자 버튼(풀업(pull-up))을 추가합니다. 누르면 현재 단계를 중단하고 5초 이내에 빨강으로 전환합니다.

### 연습 2: 버튼 속도 제어 PWM 조광기(Dimmer)

GPIO18(하드웨어 PWM 핀)에 LED 하나와 버튼 두 개를 연결하세요:

1. LED를 220Ω 저항을 통해 연결합니다. 버튼 A를 GPIO27(풀업), 버튼 B를 GPIO23(풀업)에 연결합니다.
2. `RPi.GPIO` PWM 1000 Hz를 사용하여 LED를 50% 밝기로 시작합니다.
3. 폴링(polling)이 아닌 인터럽트(interrupt)를 사용하여, 버튼 A를 누르면 밝기가 10% 증가(최대 100%)하고, 버튼 B를 누르면 10% 감소(최소 0%)합니다.
4. 버튼을 누를 때마다 현재 듀티 사이클(duty cycle)을 콘솔에 출력합니다.

### 연습 3: DHT11 환경 모니터 및 알람

GPIO4에 DHT11 센서를 연결하고 모니터링 애플리케이션을 만드세요:

1. `adafruit_dht`를 사용하여 5초마다 온도와 습도를 읽습니다.
2. 재시도 로직을 구현합니다: `RuntimeError`로 읽기에 실패하면 2초 기다렸다가 최대 3번 재시도한 후 실패를 기록하고 다음으로 넘어갑니다.
3. GPIO17에 알람 LED를 연결합니다. 온도가 28°C를 초과하거나 습도가 75%를 초과하면 켭니다.
4. 각 성공적인 읽기를 CSV 파일(`env_log.csv`)에 열 포함하여 추가합니다: `timestamp`, `temperature`, `humidity`, `alert`.

### 연습 4: 초음파 주차 보조 시스템

HC-SR04 센서(TRIG → GPIO18, ECHO → GPIO24)를 연결하고 근접 알람 시스템을 만드세요:

1. `gpiozero`의 `DistanceSensor`를 사용하여 0.5초마다 거리를 측정합니다.
2. 세 개의 LED(초록: GPIO17, 노랑: GPIO27, 빨강: GPIO22)를 거리 표시기로 연결합니다:
   - 초록 켜짐: 거리 > 50 cm (안전)
   - 노랑 켜짐: 20–50 cm (주의)
   - 빨강 켜짐: < 20 cm (위험)
3. 20 cm 미만의 거리에서는 `gpiozero`의 `blink()` 메서드를 사용하여 빨강 LED를 5 Hz로 깜빡입니다.
4. 각 측정마다 형식화된 거리 읽기를 콘솔에 출력합니다.

### 연습 5: 통합 보안 시스템

PIR 센서, 초음파 센서, DHT11, LED 3개를 객체 지향(Object-Oriented) 설계로 보안 모니터링 시스템에 결합하세요:

1. 모든 장치 참조를 담는 `SecuritySystem` 클래스를 만듭니다.
2. 시스템은 버튼(GPIO27)으로 **경계(armed)** 와 **해제(disarmed)** 두 모드를 전환합니다. 초록 LED(GPIO17)는 해제 상태, 빨강 LED(GPIO22)는 경계 상태를 나타냅니다.
3. 경계 모드에서 PIR(GPIO23)이 모션을 감지하면 알람이 울립니다: 빨강 LED를 10 Hz로 깜빡이고 타임스탬프와 함께 이벤트를 `security_log.txt`에 기록합니다.
4. 모드에 관계없이 DHT11에서 60초마다 온도와 습도를 `env_log.csv`에 기록합니다.
5. 초음파 센서가 15 cm보다 가까운 물체를 감지하면, 해제 모드에서도 침입으로 처리하고 이벤트를 별도로 기록합니다.

---

## 다음 단계

- [WiFi 네트워킹](04_WiFi_Networking.md): 센서 데이터를 네트워크로 전송
- [MQTT 프로토콜](06_MQTT_Protocol.md): MQTT로 실시간 센서 데이터 발행

---

*최종 업데이트: 2026-02-01*
