# 03. Python GPIO Control

**Previous**: [Raspberry Pi Setup](./02_Raspberry_Pi_Setup.md) | **Next**: [WiFi Networking](./04_WiFi_Networking.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Choose between RPi.GPIO and gpiozero libraries based on project needs
2. Control digital outputs (LEDs) using both libraries
3. Read digital inputs (buttons) using polling and interrupt methods
4. Implement PWM for analog-like control such as LED brightness
5. Interface with common sensors (DHT11, PIR, HC-SR04)
6. Build an integrated multi-sensor monitoring system
7. Apply best practices for GPIO cleanup, debouncing, and error handling

---

GPIO pins are what transform a Raspberry Pi from a miniature computer into a powerful IoT controller. Learning to read sensors and control actuators through Python code is the bridge between software and the physical world -- it is the skill that makes every subsequent lesson on networking, MQTT, and edge AI practically useful rather than purely theoretical.

---

This lesson covers Python-based GPIO control on Raspberry Pi. We'll learn two main libraries (RPi.GPIO and gpiozero), implement digital output (LED), digital input (button), PWM control, and sensor integration (DHT11, PIR, ultrasonic).

---

## 1. GPIO Library Selection

### 1.1 RPi.GPIO vs gpiozero

| Feature | RPi.GPIO | gpiozero |
|---------|----------|----------|
| **Level** | Low-level (direct GPIO control) | High-level (abstracted device interface) |
| **Code Complexity** | More code required | Concise, intuitive |
| **Fine Control** | Precise timing, interrupt control | Abstracted (limited fine control) |
| **Learning Curve** | Steeper | Gentle |
| **Use Cases** | Direct control, custom protocols | Rapid prototyping, standard devices |

**Recommendation:**
- **Beginners**: Start with **gpiozero** (simple and intuitive)
- **Advanced Projects**: Use **RPi.GPIO** for precise control

### 1.2 Library Installation

```bash
# RPi.GPIO (usually pre-installed)
sudo apt install python3-rpi.gpio

# gpiozero (usually pre-installed)
sudo apt install python3-gpiozero

# Or via pip
pip3 install RPi.GPIO
pip3 install gpiozero
```

---

## 2. Digital Output: LED Control

### 2.1 Basic LED Blink (RPi.GPIO)

**Circuit:**
- LED anode (+) → GPIO17
- LED cathode (-) → 220Ω resistor → GND

```python
import RPi.GPIO as GPIO
import time

# Pin configuration
LED_PIN = 17  # GPIO17

# GPIO setup
GPIO.setmode(GPIO.BCM)           # Use BCM numbering
GPIO.setup(LED_PIN, GPIO.OUT)    # Set as output

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)  # LED ON
        time.sleep(1)
        GPIO.output(LED_PIN, GPIO.LOW)   # LED OFF
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()  # Reset GPIO state
```

### 2.2 Basic LED Blink (gpiozero)

```python
from gpiozero import LED
from time import sleep

led = LED(17)  # GPIO17

try:
    while True:
        led.on()
        sleep(1)
        led.off()
        sleep(1)
except KeyboardInterrupt:
    pass
```

**Even simpler with gpiozero:**

```python
from gpiozero import LED

led = LED(17)
led.blink()  # Blink automatically with 1 second intervals

# Custom intervals
led.blink(on_time=0.5, off_time=0.2)  # 0.5s ON, 0.2s OFF
```

### 2.3 PWM: LED Brightness Control

PWM (Pulse Width Modulation) controls LED brightness by varying the duty cycle.

**RPi.GPIO version:**

```python
import RPi.GPIO as GPIO
import time

LED_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Create PWM object (frequency: 1000Hz)
pwm = GPIO.PWM(LED_PIN, 1000)
pwm.start(0)  # Start with 0% duty cycle

try:
    while True:
        # Gradually brighten (0% → 100%)
        for duty in range(0, 101, 5):
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.1)

        # Gradually dim (100% → 0%)
        for duty in range(100, -1, -5):
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
```

**gpiozero version:**

```python
from gpiozero import PWMLED
from time import sleep

led = PWMLED(17)

try:
    while True:
        # Gradually brighten (0 → 1)
        for brightness in range(0, 11):
            led.value = brightness / 10
            sleep(0.1)

        # Gradually dim (1 → 0)
        for brightness in range(10, -1, -1):
            led.value = brightness / 10
            sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    led.close()
```

**gpiozero built-in pulse effect:**

```python
from gpiozero import PWMLED

led = PWMLED(17)
led.pulse()  # Automatic breathing effect
```

---

## 3. Digital Input: Button Control

### 3.1 Button Input (Polling Method)

**Circuit:**
- Button one side → GPIO27
- Button other side → GND
- Enable internal pull-up resistor (or add external 10kΩ pull-up resistor)

**RPi.GPIO version:**

```python
import RPi.GPIO as GPIO
import time

BUTTON_PIN = 27  # GPIO27

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Enable pull-up

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Button pressed
            print("Button Pressed!")
            time.sleep(0.2)  # Debounce delay
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
```

**gpiozero version:**

```python
from gpiozero import Button
from time import sleep

button = Button(27, pull_up=True)  # Enable pull-up

try:
    while True:
        if button.is_pressed:
            print("Button Pressed!")
            sleep(0.2)  # Debounce delay
        sleep(0.1)
except KeyboardInterrupt:
    pass
```

### 3.2 Button Input (Interrupt Method)

Interrupt method is CPU-efficient as it responds to events rather than constant polling.

**RPi.GPIO version:**

```python
import RPi.GPIO as GPIO
import time

BUTTON_PIN = 27

def button_callback(channel):
    print(f"Button Pressed! (Pin: {channel})")

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Register interrupt (trigger on falling edge = button press)
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING,
                     callback=button_callback,
                     bouncetime=200)  # 200ms debounce

try:
    print("Waiting for button press... (Ctrl+C to exit)")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
```

**gpiozero version:**

```python
from gpiozero import Button
from signal import pause

button = Button(27, pull_up=True, bounce_time=0.2)  # 0.2s debounce

def on_pressed():
    print("Button Pressed!")

def on_released():
    print("Button Released!")

button.when_pressed = on_pressed
button.when_released = on_released

print("Waiting for button press... (Ctrl+C to exit)")
pause()  # Wait indefinitely
```

### 3.3 LED Toggle with Button

```python
from gpiozero import LED, Button
from signal import pause

led = LED(17)
button = Button(27, pull_up=True, bounce_time=0.2)

def toggle_led():
    led.toggle()
    state = "ON" if led.is_lit else "OFF"
    print(f"LED {state}")

button.when_pressed = toggle_led

print("Press button to toggle LED (Ctrl+C to exit)")
pause()
```

---

## 4. Sensor Integration

### 4.1 DHT11 Temperature/Humidity Sensor

**Circuit:**
- DHT11 VCC → 3.3V
- DHT11 DATA → GPIO4
- DHT11 GND → GND

**Library Installation:**

```bash
pip3 install adafruit-circuitpython-dht
sudo apt install libgpiod2  # Required dependency
```

**Code:**

```python
import time
import board
import adafruit_dht

# Initialize DHT11 (GPIO4 = board.D4)
dht_device = adafruit_dht.DHT11(board.D4)

try:
    while True:
        try:
            temperature = dht_device.temperature
            humidity = dht_device.humidity

            print(f"Temperature: {temperature:.1f}°C")
            print(f"Humidity: {humidity:.1f}%")
            print("-" * 30)

        except RuntimeError as e:
            # DHT sensors occasionally fail to read
            print(f"Read error: {e}")

        time.sleep(2)  # DHT11 max sample rate: 1Hz

except KeyboardInterrupt:
    pass
finally:
    dht_device.exit()
```

**Error Handling:**
- DHT sensors can fail intermittently → Use try/except
- Wait at least 2 seconds between reads
- If errors persist, check wiring

### 4.2 PIR Motion Sensor

PIR (Passive Infrared) sensors detect human movement.

**Circuit:**
- PIR VCC → 5V
- PIR OUT → GPIO23
- PIR GND → GND

**Code (gpiozero):**

```python
from gpiozero import MotionSensor
from signal import pause

pir = MotionSensor(23)  # GPIO23

def motion_detected():
    print("Motion Detected!")

def no_motion():
    print("No motion")

pir.when_motion = motion_detected
pir.when_no_motion = no_motion

print("PIR Sensor Ready (Ctrl+C to exit)")
pause()
```

**Code with LED indicator:**

```python
from gpiozero import MotionSensor, LED
from signal import pause

pir = MotionSensor(23)
led = LED(17)

pir.when_motion = led.on
pir.when_no_motion = led.off

print("Motion detection with LED indicator")
pause()
```

### 4.3 HC-SR04 Ultrasonic Distance Sensor

Measures distance using ultrasonic waves (2cm - 400cm range).

**Circuit:**
- HC-SR04 VCC → 5V
- HC-SR04 TRIG → GPIO18
- HC-SR04 ECHO → GPIO24 (use voltage divider for 5V→3.3V or direct if 3.3V tolerant)
- HC-SR04 GND → GND

**Code (gpiozero):**

```python
from gpiozero import DistanceSensor
from time import sleep

# TRIG: GPIO18, ECHO: GPIO24
sensor = DistanceSensor(echo=24, trigger=18, max_distance=4)

try:
    while True:
        distance = sensor.distance * 100  # Convert to cm
        print(f"Distance: {distance:.1f} cm")
        sleep(0.5)
except KeyboardInterrupt:
    pass
```

**Manual calculation version (RPi.GPIO):**

```python
import RPi.GPIO as GPIO
import time

TRIG_PIN = 18
ECHO_PIN = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def measure_distance():
    # Send 10μs pulse to TRIG
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)  # 10μs
    GPIO.output(TRIG_PIN, GPIO.LOW)

    # Measure ECHO pulse duration
    while GPIO.input(ECHO_PIN) == GPIO.LOW:
        pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == GPIO.HIGH:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    # Distance = (duration × speed of sound) / 2
    # Speed of sound: 34300 cm/s
    distance = (pulse_duration * 34300) / 2

    return distance

try:
    while True:
        dist = measure_distance()
        print(f"Distance: {dist:.1f} cm")
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
```

---

## 5. Practical Project: Integrated Sensor Monitoring System

Integrate multiple sensors and display real-time data.

**Components:**
- DHT11 (Temperature/Humidity) → GPIO4
- PIR (Motion Sensor) → GPIO23
- HC-SR04 (Ultrasonic) → TRIG: GPIO18, ECHO: GPIO24
- LED (Status Indicator) → GPIO17

```python
import time
import board
import adafruit_dht
from gpiozero import MotionSensor, DistanceSensor, LED

class SensorMonitor:
    def __init__(self):
        # Initialize sensors
        self.dht = adafruit_dht.DHT11(board.D4)
        self.pir = MotionSensor(23)
        self.ultrasonic = DistanceSensor(echo=24, trigger=18, max_distance=4)
        self.led = LED(17)

        # Motion detection callbacks
        self.pir.when_motion = self.on_motion
        self.pir.when_no_motion = self.on_no_motion

        self.motion_detected = False

    def on_motion(self):
        self.motion_detected = True
        self.led.on()
        print("[ALERT] Motion Detected!")

    def on_no_motion(self):
        self.motion_detected = False
        self.led.off()
        print("[INFO] No motion")

    def read_sensors(self):
        data = {}

        # Read DHT11
        try:
            data['temperature'] = self.dht.temperature
            data['humidity'] = self.dht.humidity
        except RuntimeError as e:
            print(f"DHT11 read error: {e}")
            data['temperature'] = None
            data['humidity'] = None

        # Read ultrasonic distance
        try:
            data['distance'] = self.ultrasonic.distance * 100  # cm
        except Exception as e:
            print(f"Ultrasonic read error: {e}")
            data['distance'] = None

        # Motion status
        data['motion'] = self.motion_detected

        return data

    def display_data(self, data):
        print("\n" + "="*40)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*40)

        if data['temperature'] is not None:
            print(f"Temperature: {data['temperature']:.1f}°C")
            print(f"Humidity:    {data['humidity']:.1f}%")
        else:
            print("Temperature: Read Failed")

        if data['distance'] is not None:
            print(f"Distance:    {data['distance']:.1f} cm")
        else:
            print("Distance:    Read Failed")

        print(f"Motion:      {'Detected' if data['motion'] else 'None'}")
        print("="*40)

    def run(self):
        print("Sensor Monitoring System Started")
        print("Press Ctrl+C to stop")

        try:
            while True:
                data = self.read_sensors()
                self.display_data(data)
                time.sleep(3)  # Update every 3 seconds
        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        self.dht.exit()
        self.led.close()
        print("Cleanup complete")

if __name__ == "__main__":
    monitor = SensorMonitor()
    monitor.run()
```

**Sample Output:**

```
Sensor Monitoring System Started
Press Ctrl+C to stop

========================================
Timestamp: 2024-01-15 14:32:10
----------------------------------------
Temperature: 24.0°C
Humidity:    55.0%
Distance:    125.3 cm
Motion:      None
========================================

[ALERT] Motion Detected!

========================================
Timestamp: 2024-01-15 14:32:13
----------------------------------------
Temperature: 24.1°C
Humidity:    55.2%
Distance:    45.7 cm
Motion:      Detected
========================================
```

---

## 6. Best Practices

### 6.1 Proper GPIO Cleanup

Always call `GPIO.cleanup()` to reset GPIO state on exit:

```python
try:
    # Your code here
    pass
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()  # Essential!
```

Or use context manager:

```python
from gpiozero import LED

with LED(17) as led:
    led.on()
    # Automatically cleaned up when exiting with block
```

### 6.2 Debouncing

Mechanical buttons can produce multiple signals (bounce). Use debounce:

```python
# RPi.GPIO
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING,
                     callback=callback_func,
                     bouncetime=200)  # 200ms debounce

# gpiozero
button = Button(27, bounce_time=0.2)  # 0.2s debounce
```

### 6.3 Error Handling

Always handle sensor read errors:

```python
try:
    temperature = dht_device.temperature
except RuntimeError as e:
    print(f"Sensor error: {e}")
    temperature = None
```

### 6.4 Resource Management

Close resources properly:

```python
class SensorSystem:
    def __init__(self):
        self.sensor = SomeDevice()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sensor.close()
        return False

# Usage
with SensorSystem() as system:
    system.read_data()
# Automatically cleaned up
```

---

## 7. Summary

### Completed Tasks

- ✅ **GPIO Libraries**: Understand RPi.GPIO vs gpiozero differences and selection
- ✅ **Digital Output**: LED control, PWM brightness control
- ✅ **Digital Input**: Button input (polling and interrupt methods)
- ✅ **Sensor Integration**: DHT11, PIR, HC-SR04 sensor interfacing
- ✅ **Practical Project**: Integrated multi-sensor monitoring system
- ✅ **Best Practices**: GPIO cleanup, debouncing, error handling

### Hands-On Exercises

1. **Traffic Light System**:
   - Use 3 LEDs (Red, Yellow, Green)
   - Implement traffic light sequence
   - Add pedestrian button to trigger sequence

2. **Smart Night Light**:
   - Use PIR motion sensor
   - Turn on LED when motion detected
   - Auto-off after 30 seconds of no motion

3. **Parking Assistant**:
   - Use ultrasonic sensor
   - Show distance on LED (PWM brightness: closer = brighter)
   - Add buzzer for proximity alert

4. **Climate Monitor**:
   - Log DHT11 data every 5 minutes
   - Save to CSV file with timestamps
   - Display warning if temperature/humidity out of range

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **GPIO warnings** | Pins not cleaned up | Always call `GPIO.cleanup()` in finally block |
| **Permission denied** | Insufficient privileges | Run with `sudo` or add user to gpio group |
| **DHT read errors** | Normal intermittent failures | Add try/except, retry logic |
| **Button multiple triggers** | Contact bounce | Add debounce delay (200ms recommended) |
| **Ultrasonic timeout** | Out of range or wiring issue | Check max_distance setting, verify wiring |

---

## References

- [gpiozero Documentation](https://gpiozero.readthedocs.io/)
- [RPi.GPIO Documentation](https://sourceforge.net/p/raspberry-gpio-python/wiki/Home/)
- [Adafruit DHT Library](https://github.com/adafruit/Adafruit_CircuitPython_DHT)
- [GPIO Pin Reference](https://pinout.xyz/)

---

## Exercises

### Exercise 1: Traffic Light Controller

Build a traffic light system using three LEDs (red on GPIO17, yellow on GPIO27, green on GPIO22):

1. Wire each LED through a 220Ω resistor to GND.
2. Implement the standard UK traffic light sequence: red (3 s) → red+yellow (1 s) → green (3 s) → yellow (1 s) → repeat.
3. Use the `gpiozero` library and implement the sequence inside a `while True` loop with proper cleanup in a `finally` block.
4. Add a pedestrian button on GPIO4 (pull-up). When pressed, interrupt the current phase and jump to red within 5 seconds.

### Exercise 2: PWM-Based Dimmer with Button Speed Control

Connect a single LED to GPIO18 (a hardware PWM pin) and two buttons:

1. Wire the LED through a 220Ω resistor. Connect button A to GPIO27 (pull-up) and button B to GPIO23 (pull-up).
2. Using `RPi.GPIO` PWM at 1000 Hz, start the LED at 50% brightness.
3. Use interrupts (not polling) so that pressing button A increases brightness by 10% (clamped at 100%) and pressing button B decreases brightness by 10% (clamped at 0%).
4. Print the current duty cycle to the console each time a button is pressed.

### Exercise 3: DHT11 Environmental Monitor with Alert

Wire a DHT11 sensor to GPIO4 and build a monitoring application:

1. Read temperature and humidity every 5 seconds using `adafruit_dht`.
2. Implement retry logic: if the read fails with `RuntimeError`, wait 2 seconds and retry up to 3 times before logging the failure and moving on.
3. Connect an alert LED to GPIO17. Turn it on when temperature exceeds 28°C or humidity exceeds 75%.
4. Append each successful reading to a CSV file (`env_log.csv`) with columns: `timestamp`, `temperature`, `humidity`, `alert`.

### Exercise 4: Ultrasonic Parking Assistant

Wire an HC-SR04 sensor (TRIG → GPIO18, ECHO → GPIO24) and build a proximity alert system:

1. Measure distance every 0.5 seconds using the `gpiozero` `DistanceSensor`.
2. Connect three LEDs (green: GPIO17, yellow: GPIO27, red: GPIO22) as distance indicators:
   - Green on: distance > 50 cm (safe)
   - Yellow on: 20–50 cm (caution)
   - Red on: < 20 cm (danger)
3. At distances under 20 cm, also blink the red LED at 5 Hz using `gpiozero`'s `blink()` method.
4. Print a formatted distance reading to the console on each measurement.

### Exercise 5: Integrated Security System

Combine a PIR sensor, ultrasonic sensor, DHT11, and three LEDs into a security monitoring system using object-oriented design:

1. Create a `SecuritySystem` class that holds all device references.
2. The system has two modes toggled by a button (GPIO27): **armed** and **disarmed**. A green LED (GPIO17) indicates disarmed; a red LED (GPIO22) indicates armed.
3. In armed mode, motion detected by the PIR (GPIO23) triggers an alarm: blink the red LED at 10 Hz and log the event with a timestamp to `security_log.txt`.
4. Regardless of mode, log temperature and humidity from DHT11 every 60 seconds to `env_log.csv`.
5. If the ultrasonic sensor detects an object closer than 15 cm, treat it as an intrusion even in disarmed mode and log the event separately.

---

**Previous**: [Raspberry Pi Setup](./02_Raspberry_Pi_Setup.md) | **Next**: [WiFi Networking](./04_WiFi_Networking.md)
