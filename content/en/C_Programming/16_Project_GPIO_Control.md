# Project 15: GPIO Control

**Previous**: [Advanced Bit Operations](./15_Bit_Operations.md) | **Next**: [Project 16: Serial Communication](./17_Project_Serial_Communication.md)

Control LEDs and buttons using GPIO (General Purpose Input/Output).

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure GPIO pins as digital inputs or outputs using `pinMode()`
2. Drive LEDs with `digitalWrite()` and create multi-LED lighting patterns
3. Read button states with `digitalRead()` using internal pull-up resistors
4. Explain the difference between pull-up and pull-down resistor configurations
5. Implement software debouncing to eliminate false triggers from mechanical switch bounce
6. Control GPIO pins through direct register manipulation (DDRx, PORTx, PINx) for higher performance
7. Build an interactive project that combines button input with LED pattern output

---

GPIO pins are your program's physical hands -- they let software reach into the real world to turn things on, sense button presses, and respond to the environment. This project moves beyond blinking a single LED and challenges you to build an interactive controller where buttons select patterns and speeds, combining everything you have learned about digital I/O and bit operations.

## Prerequisites
- Arduino basic structure (setup, loop)
- Bit operation basics

---

## 1. GPIO Concepts

### What is GPIO?

**GPIO (General Purpose Input/Output)** are general-purpose digital pins on an MCU that can be freely configured as input or output through programming.

```
Two modes of GPIO pins:

Output Mode (OUTPUT)
+-------------------------------------+
|  MCU outputs voltage to the pin     |
|  - HIGH: 5V (or 3.3V)               |
|  - LOW: 0V (GND)                    |
|  Example: Turn LED on/off, relay    |
+-------------------------------------+

Input Mode (INPUT)
+-------------------------------------+
|  MCU reads voltage on the pin       |
|  - HIGH: Above threshold (~3V)      |
|  - LOW: Below threshold (~1.5V)     |
|  Example: Read button, sensor       |
+-------------------------------------+
```

### Arduino Uno GPIO Pins

```
Arduino Uno Pin Layout:
+------------------------------------------------+
|                                                |
|  Digital Pins: 0 ~ 13 (14 total)               |
|  - 0, 1: Serial communication (TX, RX)         |
|  - 2, 3: External interrupt capable            |
|  - 3, 5, 6, 9, 10, 11: PWM output capable (~)  |
|  - 13: Built-in LED connected                  |
|                                                |
|  Analog Pins: A0 ~ A5 (6 total)                |
|  - Analog input capable (ADC)                  |
|  - Can also be used as digital I/O             |
|                                                |
+------------------------------------------------+
```

---

## 2. Digital Output: LED Control

### Basic Circuit Connection

```
LED Connection Methods:

Method 1: Pin -> Resistor -> LED -> GND (sinking)
+------+
| Pin 9|---[330R]---[LED]---GND
+------+
* LED turns on with HIGH output

Method 2: VCC -> LED -> Resistor -> Pin (sourcing)
          VCC---[LED]---[330R]---| Pin 9|
                                 +------+
* LED turns on with LOW output

Resistor Value Calculation:
R = (V_supply - V_led) / I_led
  = (5V - 2V) / 10mA
  = 300R -> Use 330R (standard value)
```

### Basic LED On/Off

```cpp
// led_basic.ino
// Most basic LED control

const int LED_PIN = 9;  // LED connected pin

void setup() {
    // Set pin mode to output
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Turn LED on (HIGH = 5V output)
    digitalWrite(LED_PIN, HIGH);
    delay(1000);  // Wait 1 second

    // Turn LED off (LOW = 0V output)
    digitalWrite(LED_PIN, LOW);
    delay(1000);
}
```

### Setting Up Circuit in Wokwi

Build the circuit in Wokwi (https://wokwi.com) as follows:

1. Add Arduino Uno
2. Add LED (search in parts list)
3. Add resistor (330 ohms)
4. Connections:
   - Pin 9 -> Resistor -> LED anode (+, long leg)
   - LED cathode (-, short leg) -> GND

### Sequential LED Lighting

```cpp
// led_sequence.ino
// Light multiple LEDs sequentially

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    // Set all LED pins to output
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
}

void loop() {
    // Turn on sequentially
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], HIGH);
        delay(200);
    }

    // Turn off sequentially
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], LOW);
        delay(200);
    }
}
```

### Creating LED Patterns

```cpp
// led_patterns.ino
// Various LED patterns

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
}

// Set all LED states
void setLEDs(int pattern) {
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], (pattern >> i) & 1);
    }
}

// Knight Rider pattern
void knightRider() {
    // Left
    for (int i = 0; i < NUM_LEDS; i++) {
        setLEDs(1 << i);
        delay(100);
    }
    // Right
    for (int i = NUM_LEDS - 2; i > 0; i--) {
        setLEDs(1 << i);
        delay(100);
    }
}

// Blink pattern
void blinkAll(int times, int delayMs) {
    for (int i = 0; i < times; i++) {
        setLEDs(0x0F);  // All on
        delay(delayMs);
        setLEDs(0x00);  // All off
        delay(delayMs);
    }
}

// Fill pattern
void fillPattern() {
    for (int i = 0; i < NUM_LEDS; i++) {
        setLEDs((1 << (i + 1)) - 1);  // 0001, 0011, 0111, 1111
        delay(200);
    }
    for (int i = NUM_LEDS; i > 0; i--) {
        setLEDs((1 << i) - 1);
        delay(200);
    }
}

void loop() {
    knightRider();
    delay(500);

    blinkAll(3, 200);
    delay(500);

    fillPattern();
    delay(500);
}
```

---

## 3. Digital Input: Button Reading

### Button Circuit Connection

```
Button Connection Methods:

Method 1: External pull-down resistor
        VCC (5V)
          |
        [Button]
          |
    +-----+-----+
    |           |
  [Pin]       [10k]
                |
              GND

- Button not pressed: Pin = LOW (resistor pulls to GND)
- Button pressed: Pin = HIGH (connected to VCC)

Method 2: External pull-up resistor
        VCC (5V)
          |
       [10k]
          |
    +-----+-----+
    |           |
  [Pin]       [Button]
                |
              GND

- Button not pressed: Pin = HIGH (resistor pulls to VCC)
- Button pressed: Pin = LOW (connected to GND)

Method 3: Internal pull-up resistor (recommended)
  [Pin]---[Button]---GND

- Use pinMode(pin, INPUT_PULLUP)
- No external resistor needed
- Button not pressed: HIGH
- Button pressed: LOW
```

### Basic Button Reading

```cpp
// button_basic.ino
// Control LED with button

const int BUTTON_PIN = 2;
const int LED_PIN = 13;

void setup() {
    // Use internal pull-up resistor
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);

    Serial.begin(9600);
}

void loop() {
    // Read button state
    int buttonState = digitalRead(BUTTON_PIN);

    // Turn on LED when button pressed (LOW = pressed)
    if (buttonState == LOW) {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("Button pressed!");
    } else {
        digitalWrite(LED_PIN, LOW);
    }

    delay(10);  // Short delay
}
```

### Button Toggle

```cpp
// button_toggle.ino
// Toggle LED state with each button press

const int BUTTON_PIN = 2;
const int LED_PIN = 13;

bool ledState = false;
bool lastButtonState = HIGH;  // HIGH is default with pull-up

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    bool currentButtonState = digitalRead(BUTTON_PIN);

    // When button is pressed (HIGH -> LOW transition)
    if (lastButtonState == HIGH && currentButtonState == LOW) {
        ledState = !ledState;  // Toggle LED state
        digitalWrite(LED_PIN, ledState);
    }

    lastButtonState = currentButtonState;
    delay(10);
}
```

---

## 4. Debouncing

### The Bouncing Problem

Physical buttons cause multiple contact bounces when pressed, creating noise.

```
Actual button signal (bouncing):

    Button press                    Button release
        |                             |
HIGH -----+   +-+ +-+        +-+ +--------
          |   | | | |        | | |
LOW       +---+ +-+ +--------+ +-+

         ^^^ bouncing noise ^^^

Duration: Approximately 1-50ms
```

### Software Debouncing

```cpp
// debounce_software.ino
// Software debouncing

const int BUTTON_PIN = 2;
const int LED_PIN = 13;

bool ledState = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;  // 50ms

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    bool reading = digitalRead(BUTTON_PIN);

    // Detect state change
    if (reading != lastButtonState) {
        lastDebounceTime = millis();  // Reset timer
    }

    // Confirm after stable for set time
    if ((millis() - lastDebounceTime) > debounceDelay) {
        static bool buttonState = HIGH;

        if (reading != buttonState) {
            buttonState = reading;

            // Button press confirmed (HIGH -> LOW)
            if (buttonState == LOW) {
                ledState = !ledState;
                digitalWrite(LED_PIN, ledState);
                Serial.println(ledState ? "LED ON" : "LED OFF");
            }
        }
    }

    lastButtonState = reading;
}
```

### Creating a Button Class

```cpp
// button_class.ino
// Reusable button class

class Button {
private:
    int pin;
    bool lastState;
    bool currentState;
    unsigned long lastDebounceTime;
    unsigned long debounceDelay;

public:
    Button(int p, unsigned long delay = 50) {
        pin = p;
        debounceDelay = delay;
        lastState = HIGH;
        currentState = HIGH;
        lastDebounceTime = 0;
    }

    void begin() {
        pinMode(pin, INPUT_PULLUP);
    }

    // Update and detect press
    bool pressed() {
        bool reading = digitalRead(pin);

        if (reading != lastState) {
            lastDebounceTime = millis();
        }

        if ((millis() - lastDebounceTime) > debounceDelay) {
            if (reading != currentState) {
                currentState = reading;
                if (currentState == LOW) {
                    lastState = reading;
                    return true;  // Button pressed!
                }
            }
        }

        lastState = reading;
        return false;
    }

    // Current state (is being held)
    bool isPressed() {
        return currentState == LOW;
    }
};

// Usage example
Button btn1(2);
Button btn2(3);
const int LED1 = 12;
const int LED2 = 13;

void setup() {
    btn1.begin();
    btn2.begin();
    pinMode(LED1, OUTPUT);
    pinMode(LED2, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    if (btn1.pressed()) {
        digitalWrite(LED1, !digitalRead(LED1));
        Serial.println("Button 1 pressed");
    }

    if (btn2.pressed()) {
        digitalWrite(LED2, !digitalRead(LED2));
        Serial.println("Button 2 pressed");
    }
}
```

---

## 5. Practice Project: LED Controller

### Project Overview

Control 4 LEDs with 2 buttons:
- Button 1: Change pattern
- Button 2: Change speed

```cpp
// led_controller.ino
// Control LED pattern and speed with buttons

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;
const int BTN_PATTERN = 2;  // Pattern change button
const int BTN_SPEED = 3;    // Speed change button

// State variables
int currentPattern = 0;
const int NUM_PATTERNS = 4;
int speedLevel = 1;  // 0=fast, 1=normal, 2=slow
const int SPEEDS[] = {50, 150, 300};

// Debouncing variables
bool lastBtnPattern = HIGH;
bool lastBtnSpeed = HIGH;
unsigned long lastDebouncePattern = 0;
unsigned long lastDebounceSpeed = 0;
const unsigned long debounceDelay = 50;

// Pattern timing
unsigned long lastPatternUpdate = 0;
int patternStep = 0;

void setup() {
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
    pinMode(BTN_PATTERN, INPUT_PULLUP);
    pinMode(BTN_SPEED, INPUT_PULLUP);

    Serial.begin(9600);
    Serial.println("LED Controller");
    Serial.println("BTN1: Change pattern, BTN2: Change speed");
}

void setLEDs(byte pattern) {
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], (pattern >> i) & 1);
    }
}

// Pattern 0: Sequential lighting
void pattern0() {
    setLEDs(1 << (patternStep % NUM_LEDS));
}

// Pattern 1: Knight Rider
void pattern1() {
    int pos = patternStep % (NUM_LEDS * 2 - 2);
    if (pos >= NUM_LEDS) {
        pos = NUM_LEDS * 2 - 2 - pos;
    }
    setLEDs(1 << pos);
}

// Pattern 2: Fill
void pattern2() {
    int step = patternStep % (NUM_LEDS * 2);
    if (step < NUM_LEDS) {
        setLEDs((1 << (step + 1)) - 1);
    } else {
        setLEDs((1 << (NUM_LEDS * 2 - step)) - 1);
    }
}

// Pattern 3: Alternating blink
void pattern3() {
    if (patternStep % 2 == 0) {
        setLEDs(0b0101);  // LED 0, 2
    } else {
        setLEDs(0b1010);  // LED 1, 3
    }
}

void updatePattern() {
    switch (currentPattern) {
        case 0: pattern0(); break;
        case 1: pattern1(); break;
        case 2: pattern2(); break;
        case 3: pattern3(); break;
    }
    patternStep++;
}

bool checkButton(int pin, bool& lastState, unsigned long& lastTime) {
    bool reading = digitalRead(pin);

    if (reading != lastState) {
        lastTime = millis();
    }

    if ((millis() - lastTime) > debounceDelay) {
        if (reading == LOW && lastState == HIGH) {
            lastState = reading;
            return true;
        }
    }

    lastState = reading;
    return false;
}

void loop() {
    // Button 1: Change pattern
    if (checkButton(BTN_PATTERN, lastBtnPattern, lastDebouncePattern)) {
        currentPattern = (currentPattern + 1) % NUM_PATTERNS;
        patternStep = 0;
        Serial.print("Pattern: ");
        Serial.println(currentPattern);
    }

    // Button 2: Change speed
    if (checkButton(BTN_SPEED, lastBtnSpeed, lastDebounceSpeed)) {
        speedLevel = (speedLevel + 1) % 3;
        Serial.print("Speed: ");
        Serial.println(SPEEDS[speedLevel]);
    }

    // Update pattern
    if (millis() - lastPatternUpdate >= SPEEDS[speedLevel]) {
        lastPatternUpdate = millis();
        updatePattern();
    }
}
```

### Wokwi Circuit Setup

```
Parts list:
- Arduino Uno x1
- LED x4 (red, yellow, green, blue)
- 330 ohm resistor x4
- Push button x2

Connections:
- LED: Pins 9, 10, 11, 12 -> Resistor -> LED -> GND
- Button 1: Pin 2 <-> GND
- Button 2: Pin 3 <-> GND
```

---

## 6. Direct Register Control

Controlling registers directly instead of Arduino functions is much faster.

### LED Control with Registers

```cpp
// register_gpio.ino
// Direct GPIO control with registers

void setup() {
    // DDRB: Port B Direction Register (pins 8-13)
    // Set bits 1-4 to output (pins 9-12)
    DDRB |= 0b00011110;

    // DDRD: Port D Direction Register (pins 0-7)
    // Set bits 2, 3 to input (pins 2, 3)
    DDRD &= ~0b00001100;

    // PORTD: Enable pull-up
    PORTD |= 0b00001100;

    Serial.begin(9600);
}

void loop() {
    // PIND: Port D Input Register
    // Check button 1 (pin 2)
    if (!(PIND & 0b00000100)) {  // Bit 2
        // All LEDs on
        PORTB |= 0b00011110;
        Serial.println("All ON");
    }

    // Check button 2 (pin 3)
    if (!(PIND & 0b00001000)) {  // Bit 3
        // All LEDs off
        PORTB &= ~0b00011110;
        Serial.println("All OFF");
    }

    // LED shift pattern
    static unsigned long lastUpdate = 0;
    static byte ledPattern = 0b00000010;  // Start from pin 9

    if (millis() - lastUpdate > 200) {
        lastUpdate = millis();

        PORTB = (PORTB & ~0b00011110) | ledPattern;

        // Shift left
        ledPattern <<= 1;
        if (ledPattern > 0b00010000) {
            ledPattern = 0b00000010;
        }
    }
}
```

### Port Mapping Reference

```
Arduino Uno Port Mapping:

Port B (PORTB, DDRB, PINB):
  Bit 0: Pin 8
  Bit 1: Pin 9
  Bit 2: Pin 10
  Bit 3: Pin 11
  Bit 4: Pin 12
  Bit 5: Pin 13 (Built-in LED)

Port D (PORTD, DDRD, PIND):
  Bit 0: Pin 0 (RX)
  Bit 1: Pin 1 (TX)
  Bit 2: Pin 2
  Bit 3: Pin 3
  Bit 4: Pin 4
  Bit 5: Pin 5
  Bit 6: Pin 6
  Bit 7: Pin 7

Port C (PORTC, DDRC, PINC):
  Bits 0-5: A0-A5
```

---

## 7. Practice Project: Reaction Time Game

```cpp
// reaction_game.ino
// Press button when LED turns on and measure reaction time

const int LED_PIN = 13;
const int BUTTON_PIN = 2;

enum GameState { WAITING, READY, PLAYING, RESULT };
GameState state = WAITING;

unsigned long ledOnTime = 0;
unsigned long reactionTime = 0;
unsigned long waitStart = 0;
int bestTime = 9999;

void setup() {
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    Serial.begin(9600);
    Serial.println("=== Reaction Time Game ===");
    Serial.println("Press button to start");
}

void loop() {
    bool buttonPressed = (digitalRead(BUTTON_PIN) == LOW);

    switch (state) {
        case WAITING:
            // Press button to start game
            if (buttonPressed) {
                state = READY;
                waitStart = millis();
                Serial.println("\nGet ready...");

                // Wait for button release
                while (digitalRead(BUTTON_PIN) == LOW);
            }
            break;

        case READY:
            // Turn on LED after random time
            if (millis() - waitStart > random(2000, 5000)) {
                digitalWrite(LED_PIN, HIGH);
                ledOnTime = millis();
                state = PLAYING;
                Serial.println("GO!");
            }

            // Too early press = fail
            if (buttonPressed) {
                Serial.println("Too early! Try again.");
                state = WAITING;
            }
            break;

        case PLAYING:
            // Measure time when button pressed
            if (buttonPressed) {
                reactionTime = millis() - ledOnTime;
                digitalWrite(LED_PIN, LOW);
                state = RESULT;
            }

            // Timeout (3 seconds)
            if (millis() - ledOnTime > 3000) {
                digitalWrite(LED_PIN, LOW);
                Serial.println("Too slow! (> 3 seconds)");
                state = WAITING;
            }
            break;

        case RESULT:
            Serial.print("Reaction time: ");
            Serial.print(reactionTime);
            Serial.println(" ms");

            if (reactionTime < bestTime) {
                bestTime = reactionTime;
                Serial.print("New best time: ");
                Serial.print(bestTime);
                Serial.println(" ms!");
            }

            Serial.println("\nPress button to play again");

            // Wait for button release
            while (digitalRead(BUTTON_PIN) == LOW);
            delay(500);

            state = WAITING;
            break;
    }

    delay(10);
}
```

---

## Exercises

### Exercise 1: Binary Counter
Count from 0 to 15 with 4 LEDs. Increment by 1 each button press.

### Exercise 2: Traffic Light
Implement a traffic light with red-yellow-green LEDs.
- Red: 3 seconds
- Red+Yellow: 1 second
- Green: 3 seconds
- Yellow: 1 second
- Repeat

### Exercise 3: Multiple Buttons
Control different LEDs with 3 buttons. All blink when pressed simultaneously.

### Exercise 4: Morse Code Input
Short button press = dot (.)
Long button press = dash (-)
Display entered Morse code on serial monitor

---

## Key Concepts Summary

| Function | Description |
|----------|-------------|
| `pinMode(pin, mode)` | Set pin mode (INPUT, OUTPUT, INPUT_PULLUP) |
| `digitalWrite(pin, val)` | Digital output (HIGH, LOW) |
| `digitalRead(pin)` | Read digital input |

| Concept | Description |
|---------|-------------|
| Pull-up resistor | Keep input pin HIGH, LOW when button pressed |
| Pull-down resistor | Keep input pin LOW, HIGH when button pressed |
| Debouncing | Button noise removal technique |
| Registers | DDRx (direction), PORTx (output), PINx (input) |

---

## Next Step

After mastering GPIO control, proceed to the next document:
- [17. Serial Communication](17_Project_Serial_Communication.md) - UART communication and debugging
