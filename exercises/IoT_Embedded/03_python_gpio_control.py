"""
Exercises for Lesson 03: Python GPIO Control
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates GPIO hardware (LEDs, buttons, sensors) in pure Python.

On a real Raspberry Pi, replace the Simulated* classes with:
  - RPi.GPIO or gpiozero for digital I/O
  - adafruit_dht for DHT11 sensor
  - gpiozero.DistanceSensor for HC-SR04
"""

import time
import csv
import os
import random
import threading
from datetime import datetime


# ---------------------------------------------------------------------------
# Simulated hardware layer
# ---------------------------------------------------------------------------

class SimulatedLED:
    """Simulate gpiozero.LED -- a digital output controlling an LED.

    Hardware note: An LED is a diode that emits light when forward-biased.
    A 220-ohm current-limiting resistor prevents exceeding the GPIO's
    ~16 mA per-pin maximum on Raspberry Pi.
    """

    def __init__(self, pin):
        self.pin = pin
        self._lit = False
        self._blinking = False

    @property
    def is_lit(self):
        return self._lit

    def on(self):
        self._lit = True
        self._blinking = False

    def off(self):
        self._lit = False
        self._blinking = False

    def toggle(self):
        self._lit = not self._lit

    def blink(self, on_time=1, off_time=1, n=None):
        """Simulate blinking. In gpiozero this runs in a background thread."""
        self._blinking = True
        count = 0
        while self._blinking and (n is None or count < n):
            self._lit = True
            time.sleep(on_time)
            if not self._blinking:
                break
            self._lit = False
            time.sleep(off_time)
            count += 1

    def close(self):
        self._lit = False
        self._blinking = False


class SimulatedPWMLED:
    """Simulate gpiozero.PWMLED -- an LED with brightness control via PWM.

    Hardware note: PWM (Pulse Width Modulation) rapidly switches the pin
    between HIGH and LOW. The duty cycle (% of time spent HIGH) determines
    perceived brightness. At 1 kHz, the switching is invisible to the eye.
    """

    def __init__(self, pin):
        self.pin = pin
        self._value = 0.0  # 0.0 (off) to 1.0 (full brightness)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = max(0.0, min(1.0, v))

    def close(self):
        self._value = 0.0


class SimulatedButton:
    """Simulate gpiozero.Button -- a digital input with pull-up resistor.

    Hardware note: A pull-up resistor holds the input HIGH when the button
    is not pressed. Pressing the button connects the pin to GND (LOW).
    Debouncing filters out the electrical noise from mechanical contact bounce.
    """

    def __init__(self, pin, pull_up=True, bounce_time=0.2):
        self.pin = pin
        self.pull_up = pull_up
        self.bounce_time = bounce_time
        self._pressed = False
        self.when_pressed = None
        self.when_released = None

    @property
    def is_pressed(self):
        return self._pressed

    def simulate_press(self):
        """Simulate a button press event."""
        self._pressed = True
        if self.when_pressed:
            self.when_pressed()

    def simulate_release(self):
        """Simulate a button release event."""
        self._pressed = False
        if self.when_released:
            self.when_released()


class SimulatedDHT11:
    """Simulate adafruit_dht.DHT11 -- temperature/humidity sensor.

    Hardware note: The DHT11 uses a capacitive humidity sensor and a
    thermistor. It communicates via a custom single-wire protocol on one
    GPIO pin. Reads occasionally fail (RuntimeError) due to timing issues;
    this is normal and expected -- always use try/except.
    """

    def __init__(self, pin):
        self.pin = pin
        self._fail_rate = 0.15  # ~15% chance of read failure (realistic)

    @property
    def temperature(self):
        if random.random() < self._fail_rate:
            raise RuntimeError("DHT sensor read failed: checksum error")
        return round(random.uniform(18.0, 32.0), 1)

    @property
    def humidity(self):
        if random.random() < self._fail_rate:
            raise RuntimeError("DHT sensor read failed: timeout")
        return round(random.uniform(35.0, 80.0), 1)

    def exit(self):
        pass


class SimulatedDistanceSensor:
    """Simulate gpiozero.DistanceSensor (HC-SR04 ultrasonic).

    Hardware note: HC-SR04 sends an ultrasonic pulse via TRIG, then measures
    the echo return time on ECHO. Distance = (echo_duration * speed_of_sound) / 2.
    Speed of sound at 20 C is approximately 343 m/s.
    The ECHO pin outputs 5V but Raspberry Pi GPIOs are 3.3V tolerant --
    use a voltage divider (1 kOhm + 2 kOhm) to protect the Pi.
    """

    def __init__(self, echo, trigger, max_distance=4):
        self.echo_pin = echo
        self.trigger_pin = trigger
        self.max_distance = max_distance

    @property
    def distance(self):
        """Return distance in meters (0 to max_distance)."""
        return round(random.uniform(0.05, self.max_distance), 3)


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: Traffic Light Controller ===
# Problem: Build a traffic light using three LEDs (red=GPIO17, yellow=GPIO27,
# green=GPIO22) with UK sequence and a pedestrian interrupt button (GPIO4).

def exercise_1():
    """Solution: Traffic light controller with pedestrian button."""

    # UK traffic light sequence:
    #   RED (3s) -> RED+YELLOW (1s) -> GREEN (3s) -> YELLOW (1s) -> repeat
    # A pedestrian button interrupts the current phase and forces RED within 5s.

    red = SimulatedLED(17)
    yellow = SimulatedLED(27)
    green = SimulatedLED(22)
    ped_button = SimulatedButton(4, pull_up=True)

    pedestrian_requested = False

    def pedestrian_callback():
        nonlocal pedestrian_requested
        pedestrian_requested = True
        print("    [PEDESTRIAN] Button pressed! Will switch to RED soon.")

    ped_button.when_pressed = pedestrian_callback

    def set_lights(r, y, g, label):
        """Set traffic light state and print status."""
        red.on() if r else red.off()
        yellow.on() if y else yellow.off()
        green.on() if g else green.off()
        indicators = []
        if r:
            indicators.append("RED")
        if y:
            indicators.append("YELLOW")
        if g:
            indicators.append("GREEN")
        print(f"    Traffic Light: [{' + '.join(indicators)}] -- {label}")

    def wait_or_interrupt(seconds):
        """Wait for specified seconds, but check for pedestrian interrupt."""
        nonlocal pedestrian_requested
        for _ in range(int(seconds * 10)):
            if pedestrian_requested:
                return True  # Interrupted
            time.sleep(0.1)
        return False

    print("  Traffic Light Controller (3 cycles)\n")

    for cycle in range(3):
        print(f"  --- Cycle {cycle + 1} ---")

        # Simulate a pedestrian press during cycle 2
        if cycle == 1:
            threading.Timer(1.5, ped_button.simulate_press).start()

        # RED phase (3 seconds)
        set_lights(True, False, False, "Stop")
        pedestrian_requested = False  # Reset after red phase serves pedestrians
        if wait_or_interrupt(3):
            set_lights(True, False, False, "Pedestrian crossing")
            time.sleep(2)
            pedestrian_requested = False
            continue

        # RED + YELLOW phase (1 second) -- prepare to go
        set_lights(True, True, False, "Prepare to go")
        time.sleep(1)

        # GREEN phase (3 seconds)
        set_lights(False, False, True, "Go")
        interrupted = wait_or_interrupt(3)
        if interrupted:
            # Transition to yellow then red for pedestrian
            set_lights(False, True, False, "Stopping for pedestrian")
            time.sleep(1)
            set_lights(True, False, False, "Pedestrian crossing")
            time.sleep(2)
            pedestrian_requested = False
            continue

        # YELLOW phase (1 second) -- prepare to stop
        set_lights(False, True, False, "Prepare to stop")
        time.sleep(1)

    # Cleanup
    red.close()
    yellow.close()
    green.close()
    print("\n  Traffic light controller stopped.")


# === Exercise 2: PWM-Based Dimmer with Button Speed Control ===
# Problem: LED on GPIO18, button A (GPIO27) increases brightness by 10%,
# button B (GPIO23) decreases by 10%. Use interrupts.

def exercise_2():
    """Solution: PWM dimmer controlled by two buttons."""

    led = SimulatedPWMLED(18)
    button_a = SimulatedButton(27, pull_up=True)
    button_b = SimulatedButton(23, pull_up=True)

    # Start at 50% brightness
    led.value = 0.5

    def increase_brightness():
        """Interrupt callback: increase brightness by 10%, clamped at 100%."""
        new_val = min(1.0, led.value + 0.1)
        led.value = new_val
        print(f"    [Button A] Brightness increased -> {led.value * 100:.0f}%")

    def decrease_brightness():
        """Interrupt callback: decrease brightness by 10%, clamped at 0%."""
        new_val = max(0.0, led.value - 0.1)
        led.value = new_val
        print(f"    [Button B] Brightness decreased -> {led.value * 100:.0f}%")

    # Register interrupt callbacks (on real Pi, GPIO.add_event_detect with FALLING edge)
    button_a.when_pressed = increase_brightness
    button_b.when_pressed = decrease_brightness

    print("  PWM Dimmer with Button Control\n")
    print(f"    Initial brightness: {led.value * 100:.0f}%\n")

    # Simulate a sequence of button presses
    actions = [
        ("A", button_a),  # 60%
        ("A", button_a),  # 70%
        ("A", button_a),  # 80%
        ("B", button_b),  # 70%
        ("B", button_b),  # 60%
        ("B", button_b),  # 50%
        ("B", button_b),  # 40%
        ("B", button_b),  # 30%
        ("A", button_a),  # 40%
    ]

    for label, button in actions:
        button.simulate_press()
        time.sleep(0.3)

    print(f"\n    Final brightness: {led.value * 100:.0f}%")
    led.close()


# === Exercise 3: DHT11 Environmental Monitor with Alert ===
# Problem: Read DHT11 every 5s with retry logic (up to 3 retries),
# alert LED on GPIO17 when temp > 28C or humidity > 75%, log to CSV.

def exercise_3():
    """Solution: DHT11 monitor with retry logic and CSV logging."""

    dht = SimulatedDHT11(pin=4)
    alert_led = SimulatedLED(17)
    csv_file = "/tmp/env_log.csv"

    def read_with_retry(max_retries=3, retry_delay=2):
        """Read DHT11 with retry logic.

        DHT11 sensors use a single-wire protocol that is timing-sensitive.
        Reads fail ~10-20% of the time due to interrupt conflicts on Linux.
        Retry with a delay between attempts is the standard approach.
        """
        for attempt in range(1, max_retries + 1):
            try:
                temp = dht.temperature
                hum = dht.humidity
                return temp, hum
            except RuntimeError as e:
                print(f"    [RETRY] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)

        print(f"    [FAIL] All {max_retries} attempts failed, skipping this reading")
        return None, None

    # Initialize CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "temperature", "humidity", "alert"])

    print("  DHT11 Environmental Monitor (8 readings)\n")

    for i in range(8):
        temp, hum = read_with_retry(max_retries=3, retry_delay=0.5)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if temp is not None and hum is not None:
            # Check alert conditions
            alert = temp > 28.0 or hum > 75.0
            if alert:
                alert_led.on()
                alert_str = "ALERT"
                reason = []
                if temp > 28.0:
                    reason.append(f"temp={temp}C > 28C")
                if hum > 75.0:
                    reason.append(f"humidity={hum}% > 75%")
                print(f"    [{ts}] Temp={temp}C, Humidity={hum}% "
                      f"** ALERT: {', '.join(reason)} **")
            else:
                alert_led.off()
                alert_str = "OK"
                print(f"    [{ts}] Temp={temp}C, Humidity={hum}% -- OK")

            # Append to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, temp, hum, alert_str])
        else:
            print(f"    [{ts}] Reading failed, not logged")

        if i < 7:
            time.sleep(1)  # Shortened for demo; production uses 5s

    # Show logged data
    print(f"\n  --- CSV Log Contents ({csv_file}) ---")
    with open(csv_file, "r") as f:
        print(f.read())

    # Cleanup
    alert_led.close()
    dht.exit()
    if os.path.exists(csv_file):
        os.remove(csv_file)


# === Exercise 4: Ultrasonic Parking Assistant ===
# Problem: HC-SR04 measures distance every 0.5s; three LEDs indicate zones:
# green (>50cm safe), yellow (20-50cm caution), red (<20cm danger, blink at 5Hz).

def exercise_4():
    """Solution: Ultrasonic parking assistant with LED indicators."""

    sensor = SimulatedDistanceSensor(echo=24, trigger=18, max_distance=4)
    green_led = SimulatedLED(17)
    yellow_led = SimulatedLED(27)
    red_led = SimulatedLED(22)

    def update_leds(distance_cm):
        """Update LED indicators based on distance zones.

        The three-zone system mirrors real parking assist systems:
        - Green: safe distance, continue approaching
        - Yellow: caution zone, slow down
        - Red + blink: danger, stop immediately
        """
        green_led.off()
        yellow_led.off()
        red_led.off()

        if distance_cm > 50:
            green_led.on()
            zone = "SAFE (green)"
        elif 20 <= distance_cm <= 50:
            yellow_led.on()
            zone = "CAUTION (yellow)"
        else:
            red_led.on()
            # On real hardware: red_led.blink(on_time=0.1, off_time=0.1)
            # That creates a 5 Hz blink (1/0.2s = 5 cycles/second)
            zone = "DANGER (red, blinking at 5 Hz)"

        return zone

    # Simulate specific distances for a clear demo
    simulated_distances = [1.50, 1.20, 0.80, 0.55, 0.45, 0.30, 0.15, 0.10, 0.25, 0.60]

    print("  Ultrasonic Parking Assistant\n")
    print(f"  {'Reading':>8}  {'Distance':>10}  {'Zone':<30}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 30}")

    for i, dist_m in enumerate(simulated_distances, 1):
        distance_cm = dist_m * 100
        zone = update_leds(distance_cm)
        print(f"  {i:>8}  {distance_cm:>8.1f} cm  {zone:<30}")
        time.sleep(0.3)

    # Cleanup
    green_led.close()
    yellow_led.close()
    red_led.close()
    print("\n  Parking assistant stopped.")


# === Exercise 5: Integrated Security System ===
# Problem: Combine PIR, ultrasonic, DHT11, three LEDs, and a button into
# an OOP security system with armed/disarmed modes.

def exercise_5():
    """Solution: Integrated security monitoring system."""

    class SecuritySystem:
        """Multi-sensor security system with armed/disarmed modes.

        Architecture:
        - Button toggles between armed (red LED) and disarmed (green LED)
        - In armed mode, PIR motion triggers alarm (red LED blinks at 10 Hz)
        - DHT11 logs temperature/humidity every 60s regardless of mode
        - Ultrasonic intrusion detection (<15 cm) works in ALL modes
        """

        def __init__(self):
            # Devices
            self.green_led = SimulatedLED(17)    # Disarmed indicator
            self.yellow_led = SimulatedLED(27)   # Status indicator
            self.red_led = SimulatedLED(22)      # Armed / alarm indicator
            self.button = SimulatedButton(27, pull_up=True, bounce_time=0.2)
            self.dht = SimulatedDHT11(pin=4)
            self.ultrasonic = SimulatedDistanceSensor(echo=24, trigger=18)

            # State
            self.armed = False
            self.alarm_active = False
            self.security_log = []
            self.env_log = []

            # Register button callback
            self.button.when_pressed = self.toggle_mode

        def toggle_mode(self):
            """Toggle between armed and disarmed."""
            self.armed = not self.armed
            self.alarm_active = False

            if self.armed:
                self.green_led.off()
                self.red_led.on()
                status = "ARMED"
            else:
                self.red_led.off()
                self.green_led.on()
                status = "DISARMED"

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_security(f"Mode changed: {status}")
            print(f"      [{ts}] System {status}")

        def trigger_alarm(self, reason):
            """Trigger alarm: blink red LED at 10 Hz and log event."""
            if not self.alarm_active:
                self.alarm_active = True
                # On real hardware: self.red_led.blink(on_time=0.05, off_time=0.05)
                # 10 Hz = 1/(0.05+0.05) = 10 cycles/second
                self.red_led.on()
                self.log_security(f"ALARM: {reason}")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"      [{ts}] *** ALARM: {reason} *** (red LED blinking at 10 Hz)")

        def check_motion(self, motion_detected):
            """Check PIR sensor -- only triggers alarm in armed mode."""
            if self.armed and motion_detected:
                self.trigger_alarm("Motion detected by PIR sensor")

        def check_proximity(self, distance_cm):
            """Check ultrasonic sensor -- triggers in ANY mode if <15 cm.

            This acts as a tamper detection mechanism: someone physically
            approaching the sensor unit is suspicious regardless of mode.
            """
            if distance_cm < 15:
                self.log_security(
                    f"INTRUSION ALERT: Object detected at {distance_cm:.1f} cm "
                    f"(threshold: 15 cm)"
                )
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"      [{ts}] INTRUSION: Object at {distance_cm:.1f} cm (any mode)")

        def log_environment(self):
            """Read and log DHT11 data (runs in both modes)."""
            try:
                temp = self.dht.temperature
                hum = self.dht.humidity
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = {"timestamp": ts, "temperature": temp, "humidity": hum}
                self.env_log.append(entry)
                print(f"      [{ts}] Env: {temp}C, {hum}%")
            except RuntimeError as e:
                print(f"      [ENV] Sensor error: {e}")

        def log_security(self, message):
            """Append security event to log."""
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{ts}] {message}"
            self.security_log.append(entry)

        def print_summary(self):
            """Print all logs."""
            print("\n    --- Security Log ---")
            for entry in self.security_log:
                print(f"      {entry}")

            print("\n    --- Environment Log ---")
            for entry in self.env_log:
                print(f"      [{entry['timestamp']}] "
                      f"Temp={entry['temperature']}C, "
                      f"Humidity={entry['humidity']}%")

        def cleanup(self):
            """Release all hardware resources."""
            self.green_led.close()
            self.yellow_led.close()
            self.red_led.close()
            self.dht.exit()

    # Run the security system demo
    print("  Integrated Security System\n")

    system = SecuritySystem()

    # Initial state: disarmed
    system.green_led.on()
    print("    System initialized (DISARMED)\n")

    # Simulate a sequence of events
    events = [
        ("env_log", None),
        ("motion", True),         # Motion while disarmed -- no alarm
        ("arm", None),            # Arm the system
        ("env_log", None),
        ("motion", True),         # Motion while armed -- ALARM
        ("proximity", 12.0),      # Object too close -- intrusion (any mode)
        ("disarm", None),         # Disarm
        ("proximity", 10.0),      # Still triggers in disarmed mode
        ("env_log", None),
    ]

    for event_type, value in events:
        if event_type == "env_log":
            system.log_environment()
        elif event_type == "motion":
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"      [{ts}] PIR: Motion {'detected' if value else 'clear'}")
            system.check_motion(value)
        elif event_type == "proximity":
            system.check_proximity(value)
        elif event_type == "arm":
            system.button.simulate_press()
        elif event_type == "disarm":
            system.button.simulate_press()
        time.sleep(0.5)

    system.print_summary()
    system.cleanup()
    print("\n  Security system stopped.")


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 03: Python GPIO Control - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Traffic Light Controller")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: PWM-Based Dimmer with Button Control")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: DHT11 Environmental Monitor")
    print("-" * 50)
    exercise_3()

    print("\n\n>>> Exercise 4: Ultrasonic Parking Assistant")
    print("-" * 50)
    exercise_4()

    print("\n\n>>> Exercise 5: Integrated Security System")
    print("-" * 50)
    exercise_5()
