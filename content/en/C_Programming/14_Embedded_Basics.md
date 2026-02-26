# Embedded Programming Basics

**Previous**: [Project 12: Multithreaded Programming](./13_Project_Multithreading.md) | **Next**: [Advanced Bit Operations](./15_Bit_Operations.md)

Understand the concepts of embedded systems and set up the Arduino development environment.

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what an embedded system is and how it differs from a general-purpose computer
2. Describe the internal architecture of a microcontroller (CPU, Flash, SRAM, peripherals)
3. Compare common Arduino boards and select an appropriate one for a project
4. Set up an Arduino development environment (IDE, Wokwi simulator, or PlatformIO)
5. Write, compile, and upload an Arduino program using `setup()` and `loop()`
6. Control digital output pins to drive LEDs with various blink patterns
7. Use the Serial Monitor for debugging and variable inspection

---

Every appliance you interact with daily -- from your microwave to your car's braking system -- runs on an embedded processor. Learning embedded programming connects the abstract world of C code to physical hardware, giving you the ability to make things blink, move, and communicate. In this lesson you will set up your first embedded environment and see your code control real (or simulated) hardware.

## Prerequisites
- C language basic syntax
- Functions and variables

---

## 1. What is an Embedded System?

### Definition

An **Embedded System** is a computer system designed to perform specific functions.

```
General Computer:
┌─────────────────────────────────────┐
│  Can run various programs           │
│  Web browser, games, documents, etc.│
│  User has free control              │
└─────────────────────────────────────┘

Embedded System:
┌─────────────────────────────────────┐
│  Performs only specific purposes    │
│  Washing machine, microwave, car ECU│
│  Dedicated hardware + software      │
└─────────────────────────────────────┘
```

### Embedded Systems Around Us

| Field | Examples |
|-------|----------|
| Home Appliances | Washing machine, refrigerator, air conditioner, microwave |
| Automotive | ECU, ABS, airbag, navigation |
| Medical Devices | Blood pressure monitor, thermometer, MRI, insulin pump |
| Communication | Router, smartphone, set-top box |
| Industrial | Factory automation, robots, PLC |
| IoT | Smart home, wearables, sensors |

### Characteristics of Embedded Systems

```
1. Limited Resources
   - Small memory (KB ~ MB)
   - Slow CPU (MHz range)
   - Limited storage
   - Low power consumption

2. Real-time Requirements
   - Must respond within specific time
   - Example: Airbag must deploy within tens of ms after crash detection

3. Reliability
   - Stable operation 24/7/365
   - Errors can have critical consequences

4. Dedicated Hardware
   - Design optimized for specific purpose
```

---

## 2. Microcontroller (MCU)

### MCU vs MPU

```
MPU (Microprocessor Unit):
┌─────────────────────────────────────┐
│ Contains only CPU core              │
│ Requires external RAM, ROM, I/O     │
│ Example: Intel Core, AMD Ryzen      │
│ High performance, general computing │
└─────────────────────────────────────┘

MCU (Microcontroller Unit):
┌─────────────────────────────────────┐
│ CPU + RAM + ROM + I/O integrated    │
│ One Chip Solution                   │
│ Example: ATmega328, STM32, ESP32    │
│ Low power, specific purpose         │
└─────────────────────────────────────┘
```

### MCU Internal Structure

```
┌─────────────────────────────────────────────────┐
│                    MCU                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │   CPU   │  │  Flash  │  │  SRAM   │         │
│  │  Core   │  │(Program)│  │(Variables│         │
│  │         │  │         │  │  Stack) │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  GPIO   │  │  Timer  │  │  UART   │         │
│  │(Digital │  │(Timer/  │  │(Serial  │         │
│  │  I/O)   │  │  PWM)   │  │  Comm)  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │   ADC   │  │   I2C   │  │   SPI   │         │
│  │(Analog  │  │  (Bus   │  │ (High   │         │
│  │ Input)  │  │  Comm)  │  │ Speed)  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
└─────────────────────────────────────────────────┘
```

### Main Memory Types

| Memory | Characteristics | Usage |
|--------|-----------------|-------|
| **Flash** | Non-volatile, fast read, slow write | Program code storage |
| **SRAM** | Volatile, fast read/write | Variables, stack, heap |
| **EEPROM** | Non-volatile, byte-level write | Configuration storage |

---

## 3. Introduction to Arduino

### What is Arduino?

Arduino is an **open-source hardware platform** designed to make embedded development easy to start.

```
Arduino Components:
┌─────────────────────────────────────┐
│  1. Hardware (Board)                │
│     - ATmega328P MCU               │
│     - USB connection               │
│     - Power circuit                │
│     - Pin headers                  │
├─────────────────────────────────────┤
│  2. Software (IDE)                 │
│     - Code editor                  │
│     - Compiler                     │
│     - Upload tool                  │
├─────────────────────────────────────┤
│  3. Libraries                      │
│     - Sensors, motors, displays    │
│     - Rich examples                │
└─────────────────────────────────────┘
```

### Main Arduino Boards

| Board | MCU | Flash | SRAM | Pins | Features |
|-------|-----|-------|------|------|----------|
| **Uno** | ATmega328P | 32KB | 2KB | 14+6 | Most basic, beginner |
| **Nano** | ATmega328P | 32KB | 2KB | 14+8 | Small, breadboard |
| **Mega** | ATmega2560 | 256KB | 8KB | 54+16 | Many pins, large projects |
| **Leonardo** | ATmega32U4 | 32KB | 2.5KB | 20+12 | USB HID support |

### Arduino Uno Pin Layout

```
                    ┌─────────────────────┐
                    │     USB Port        │
                    └─────────────────────┘
    ┌───────────────────────────────────────────┐
    │  AREF  GND  13  12  11  10  9  8         │
    │  [ ]  [ ]  [ ] [ ] [ ] [ ] [ ] [ ]       │ ← Digital pins
    │                                          │
    │    ┌─────┐                               │
    │    │     │  ATmega328P                   │
    │    │     │                               │
    │    └─────┘                               │
    │                                          │
    │  [ ]  [ ]  [ ] [ ] [ ] [ ] [ ] [ ]       │
    │  RESET 3.3V 5V GND GND Vin               │ ← Power
    │                                          │
    │  [ ]  [ ]  [ ] [ ] [ ] [ ]               │
    │  A0   A1   A2  A3  A4  A5                │ ← Analog pins
    └───────────────────────────────────────────┘

Digital pins (0~13):
- 0, 1: Serial (TX, RX) - Serial communication
- 3, 5, 6, 9, 10, 11: PWM capable (~ marked)
- 13: Built-in LED connected

Analog pins (A0~A5):
- Analog input (ADC)
- Can also be used as digital pins
```

---

## 4. Development Environment Setup

### Method 1: Arduino IDE Installation (For Real Hardware)

#### Windows / macOS

1. Visit https://www.arduino.cc/en/software
2. Download version for your OS
3. Run installer

#### macOS (Homebrew)

```bash
brew install --cask arduino-ide
```

#### Linux (Ubuntu/Debian)

```bash
# Method 1: apt
sudo apt update
sudo apt install arduino

# Method 2: Flatpak
flatpak install flathub cc.arduino.IDE2
```

### Method 2: Wokwi Simulator (Learning Without Hardware)

**Wokwi** is a free tool to simulate Arduino in your browser.

1. Visit https://wokwi.com
2. Click "Start Creating"
3. Select "Arduino Uno"
4. Start coding immediately!

```
Wokwi Advantages:
- Free, no installation required
- Various component simulation (LED, button, sensors, etc.)
- Circuit diagram visualization
- Code sharing
- Real-time debugging
```

### Method 3: VS Code + PlatformIO (Advanced)

1. Install VS Code
2. Install PlatformIO extension
3. Select Arduino Uno when creating new project

```bash
# Install PlatformIO CLI (optional)
pip install platformio
```

---

## 5. First Program: Blink

### Arduino Program Structure

```cpp
// Basic Arduino program structure

// Global variables, constants
const int LED_PIN = 13;

// setup(): Runs once at program start
void setup() {
    // Initialization code
    pinMode(LED_PIN, OUTPUT);
}

// loop(): Runs infinitely after setup()
void loop() {
    // Repeatedly executed code
    digitalWrite(LED_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_PIN, LOW);
    delay(1000);
}
```

### Comparison with Standard C

```c
// Standard C program
int main(void) {
    // Initialize
    init_hardware();

    // Infinite loop
    while (1) {
        // Repeated execution
        do_something();
    }

    return 0;  // Actually never reached
}
```

```cpp
// Arduino program (same structure)
void setup() {
    // Initialize (beginning of main)
}

void loop() {
    // Same as inside while(1)
}

// Arduino framework provides main():
// int main() {
//     setup();
//     while(1) loop();
// }
```

### Blink Example Detailed Explanation

```cpp
// blink.ino - LED blinking

// Pin number where LED is connected (Arduino Uno built-in LED)
const int LED_PIN = 13;

void setup() {
    // Set pin mode
    // OUTPUT: Output mode (sends voltage)
    // INPUT: Input mode (reads voltage)
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Turn LED on
    // HIGH = 5V (or 3.3V) output
    digitalWrite(LED_PIN, HIGH);

    // Wait 1000 milliseconds (1 second)
    delay(1000);

    // Turn LED off
    // LOW = 0V (GND) output
    digitalWrite(LED_PIN, LOW);

    // Wait 1 second
    delay(1000);

    // When loop() ends, it starts again from the beginning
}
```

### Running on Wokwi

1. Visit https://wokwi.com
2. "New Project" → Select "Arduino Uno"
3. Enter code:

```cpp
void setup() {
    pinMode(LED_BUILTIN, OUTPUT);  // LED_BUILTIN = 13
}

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
    delay(1000);
}
```

4. Click green "Start Simulation" button
5. Confirm that board's LED is blinking

---

## 6. Main Arduino Functions

### Digital I/O

```cpp
// Set pin mode
pinMode(pin, mode);
// mode: INPUT, OUTPUT, INPUT_PULLUP

// Digital output
digitalWrite(pin, value);
// value: HIGH (5V), LOW (0V)

// Digital input
int value = digitalRead(pin);
// Returns: HIGH or LOW
```

### Time Related

```cpp
// Wait milliseconds
delay(ms);

// Wait microseconds
delayMicroseconds(us);

// Time elapsed since program start (milliseconds)
unsigned long time = millis();

// Time elapsed since program start (microseconds)
unsigned long time = micros();
```

### Serial Communication

```cpp
// Initialize serial (typically 9600 or 115200)
Serial.begin(baudrate);

// Output data
Serial.print("Hello");      // Without newline
Serial.println("World");    // With newline
Serial.print(123);          // Print number

// Input data
if (Serial.available() > 0) {
    char c = Serial.read();
}
```

---

## 7. Practice Project: Various Blink Patterns

### Project 1: Variable Speed Blink

```cpp
// LED that gets faster

const int LED_PIN = 13;
int delayTime = 1000;  // Starting delay

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);
    delay(delayTime);
    digitalWrite(LED_PIN, LOW);
    delay(delayTime);

    // Decrease delay (minimum 50ms)
    delayTime -= 50;
    if (delayTime < 50) {
        delayTime = 1000;  // Reset
    }
}
```

### Project 2: SOS Signal

```cpp
// Morse code SOS (... --- ...)

const int LED_PIN = 13;
const int DOT = 200;    // Dot length
const int DASH = 600;   // Dash length
const int GAP = 200;    // Gap between signals
const int LETTER_GAP = 600;  // Gap between letters

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void dot() {
    digitalWrite(LED_PIN, HIGH);
    delay(DOT);
    digitalWrite(LED_PIN, LOW);
    delay(GAP);
}

void dash() {
    digitalWrite(LED_PIN, HIGH);
    delay(DASH);
    digitalWrite(LED_PIN, LOW);
    delay(GAP);
}

void loop() {
    // S: ...
    dot(); dot(); dot();
    delay(LETTER_GAP);

    // O: ---
    dash(); dash(); dash();
    delay(LETTER_GAP);

    // S: ...
    dot(); dot(); dot();
    delay(LETTER_GAP * 3);  // Long gap between words
}
```

### Project 3: Using millis() (Asynchronous Blink)

`delay()` stops the program, but using `millis()` allows other tasks to run.

```cpp
// Blink LED without delay()

const int LED_PIN = 13;
unsigned long previousMillis = 0;
const long interval = 1000;  // 1 second
int ledState = LOW;

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    unsigned long currentMillis = millis();

    // Check if interval time has passed
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;

        // Toggle LED state
        ledState = (ledState == LOW) ? HIGH : LOW;
        digitalWrite(LED_PIN, ledState);
    }

    // Can do other tasks here!
    // Example: Read sensors, check buttons, etc.
}
```

### Project 4: Multiple LED Control

You can test by connecting external LEDs in Wokwi.

```cpp
// Sequential lighting of 3 LEDs

const int LED1 = 11;
const int LED2 = 12;
const int LED3 = 13;

void setup() {
    pinMode(LED1, OUTPUT);
    pinMode(LED2, OUTPUT);
    pinMode(LED3, OUTPUT);
}

void loop() {
    // Turn on only LED1
    digitalWrite(LED1, HIGH);
    digitalWrite(LED2, LOW);
    digitalWrite(LED3, LOW);
    delay(300);

    // Turn on only LED2
    digitalWrite(LED1, LOW);
    digitalWrite(LED2, HIGH);
    digitalWrite(LED3, LOW);
    delay(300);

    // Turn on only LED3
    digitalWrite(LED1, LOW);
    digitalWrite(LED2, LOW);
    digitalWrite(LED3, HIGH);
    delay(300);
}
```

---

## 8. Debugging with Serial Monitor

### Basic Serial Output

```cpp
void setup() {
    Serial.begin(9600);  // Start serial communication
    Serial.println("Arduino started!");
}

void loop() {
    static int count = 0;
    count++;

    Serial.print("Count: ");
    Serial.println(count);

    delay(1000);
}
```

### Using Serial Monitor in Wokwi

1. Start simulation
2. Click "Serial Monitor" tab on the right
3. Check output

### Monitoring Variable Values

```cpp
const int LED_PIN = 13;
int blinkCount = 0;

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
    Serial.println("=== Blink Counter ===");
}

void loop() {
    blinkCount++;

    digitalWrite(LED_PIN, HIGH);
    Serial.print("LED ON - Count: ");
    Serial.println(blinkCount);
    delay(500);

    digitalWrite(LED_PIN, LOW);
    Serial.println("LED OFF");
    delay(500);
}
```

---

## Exercises

### Exercise 1: Heartbeat LED
Create a pattern where the LED blinks twice quickly like a heartbeat, then pauses.

### Exercise 2: Countdown
Output a countdown from 10 to 1 on the serial monitor, and when it reaches 0, blink the LED 3 times.

### Exercise 3: Random Blink
Use the `random()` function to make the LED blink at irregular intervals.

```cpp
// Hint
int randomDelay = random(100, 1000);  // Random value between 100~999
```

### Exercise 4: Binary Counter
Use 4 LEDs to display 0~15 in binary.
- 0 = 0000 (all LEDs off)
- 5 = 0101 (LED2, LED4 on)
- 15 = 1111 (all LEDs on)

---

## Key Concepts Summary

| Term | Description |
|------|-------------|
| Embedded System | Dedicated computer system for specific functions |
| MCU | Chip integrating CPU + memory + peripherals |
| GPIO | General Purpose Input/Output pins |
| Flash | Non-volatile memory for program code |
| SRAM | Volatile memory for variables |
| setup() | Initialization code (runs once) |
| loop() | Repeatedly executed code (infinite loop) |

---

## Wokwi Project Link Example

You can run a basic Blink project directly on Wokwi:
- https://wokwi.com/projects/new/arduino-uno

---

## Next Steps

Once you've mastered Arduino basics, proceed to the next document:
- [15. Advanced Bit Operations](15_Bit_Operations.md) - Core embedded technology
