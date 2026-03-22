# Embedded Systems

**Previous**: [IPC and Signals](./13_IPC_and_Signals.md) | **Next**: [Debugging and Profiling](./15_Debugging_and_Profiling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how C's low-level features (volatile, bitwise ops, memory-mapped I/O) enable embedded programming
2. Configure GPIO pins for digital input/output and implement software debouncing
3. Implement serial communication using UART parameters and command parsing
4. Use hardware communication protocols (I2C, SPI) and ADC for sensor data
5. Apply PWM for analog output control and hardware timer configuration

---

Embedded systems are everywhere -- from microwaves and thermostats to automotive ECUs and medical devices. C dominates this domain because it provides direct hardware access with minimal overhead. This lesson consolidates the essential embedded programming concepts: GPIO control, serial communication, hardware protocols, and analog interfacing. You will learn how C's low-level features map to real hardware and how to build interactive embedded applications.

**Difficulty**: Advanced

**Prerequisites**: Pointers, bit operations, structures

---

## 1. C for Embedded -- Why C Dominates

### Embedded vs General-Purpose Computing

| Feature | General Computer | Embedded System |
|---------|-----------------|-----------------|
| Purpose | Runs various programs | Performs specific functions |
| Memory | GB of RAM | KB to MB of SRAM |
| CPU Speed | GHz range | MHz range |
| Storage | SSD/HDD (TB) | Flash (KB to MB) |
| Examples | Laptop, desktop | Washing machine, ECU, IoT sensor |

### Why C?

- **Direct hardware access**: Pointers can address memory-mapped registers
- **Minimal overhead**: No garbage collector, no runtime
- **Predictable performance**: Deterministic execution for real-time requirements
- **Small footprint**: A "Hello World" can fit in a few hundred bytes of Flash

### Cross-Compilation

Embedded targets differ from the development machine. You compile on your PC (host) for the target MCU:

```bash
# Native compilation (runs on your PC)
gcc -o program program.c

# Cross-compilation (runs on ARM MCU)
arm-none-eabi-gcc -mcpu=cortex-m4 -o firmware.elf main.c
```

### MCU Architecture

A microcontroller (MCU) integrates CPU, memory, and peripherals on one chip:

| Component | Description |
|-----------|-------------|
| **CPU Core** | Executes instructions (e.g., ARM Cortex-M, AVR) |
| **Flash** | Non-volatile program storage (32KB-2MB) |
| **SRAM** | Volatile data storage for variables/stack (2KB-512KB) |
| **GPIO** | General Purpose Input/Output pins |
| **Timer** | Hardware counters for timing/PWM |
| **UART** | Serial communication |
| **ADC** | Analog-to-digital converter |
| **I2C/SPI** | Bus communication protocols |

---

## 2. Arduino and GPIO Basics

### Arduino Program Structure

Arduino uses a C/C++ dialect with two required functions:

```c
// Standard C equivalent
int main(void) {
    init_hardware();   // setup()
    while (1) {
        do_work();     // loop()
    }
    return 0;
}
```

```cpp
// Arduino version
void setup() {
    // Runs once at startup
    pinMode(13, OUTPUT);
}

void loop() {
    // Runs repeatedly
    digitalWrite(13, HIGH);
    delay(1000);
    digitalWrite(13, LOW);
    delay(1000);
}
```

### Digital Output -- LED Control

```cpp
// Basic LED blink
const int LED_PIN = 9;

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);  // 5V output
    delay(1000);
    digitalWrite(LED_PIN, LOW);   // 0V output
    delay(1000);
}
```

### Multiple LED Patterns

```cpp
const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
}

// Set all LEDs from a bitmask
void setLEDs(int pattern) {
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], (pattern >> i) & 1);
    }
}

// Knight Rider pattern
void knightRider() {
    for (int i = 0; i < NUM_LEDS; i++) {
        setLEDs(1 << i);
        delay(100);
    }
    for (int i = NUM_LEDS - 2; i > 0; i--) {
        setLEDs(1 << i);
        delay(100);
    }
}

void loop() {
    knightRider();
}
```

---

## 3. GPIO Advanced -- Pull-ups, Debouncing, Interrupts

### Button Reading with Internal Pull-up

```cpp
const int BUTTON_PIN = 2;
const int LED_PIN = 13;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);  // Internal pull-up resistor
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // LOW = pressed (pull-up inverts logic)
    if (digitalRead(BUTTON_PIN) == LOW) {
        digitalWrite(LED_PIN, HIGH);
    } else {
        digitalWrite(LED_PIN, LOW);
    }
    delay(10);
}
```

### Software Debouncing

Mechanical buttons produce noise (bouncing) when pressed. Software debouncing waits for the signal to stabilize:

```cpp
const int BUTTON_PIN = 2;
const int LED_PIN = 13;

bool ledState = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;  // 50ms

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    bool reading = digitalRead(BUTTON_PIN);

    if (reading != lastButtonState) {
        lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > debounceDelay) {
        static bool buttonState = HIGH;

        if (reading != buttonState) {
            buttonState = reading;

            if (buttonState == LOW) {
                ledState = !ledState;
                digitalWrite(LED_PIN, ledState);
            }
        }
    }

    lastButtonState = reading;
}
```

### Interrupt-Driven Button

```cpp
const int BUTTON_PIN = 2;  // INT0
const int LED_PIN = 13;
volatile int buttonCount = 0;

void buttonISR() {
    buttonCount++;
    digitalWrite(LED_PIN, buttonCount % 2);
}

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN),
                    buttonISR, FALLING);
}

void loop() {
    Serial.println(buttonCount);
    delay(500);
}
```

### Direct Register Control

For maximum performance, manipulate GPIO registers directly:

```cpp
void setup() {
    // DDRB: Port B Direction Register (pins 8-13)
    DDRB |= 0b00011110;   // Set pins 9-12 as output

    // DDRD: Port D Direction Register (pins 0-7)
    DDRD &= ~0b00001100;  // Set pins 2,3 as input
    PORTD |= 0b00001100;  // Enable pull-ups on pins 2,3
}

void loop() {
    // Read button on pin 2
    if (!(PIND & 0b00000100)) {
        PORTB |= 0b00011110;   // All LEDs on
    }

    // Read button on pin 3
    if (!(PIND & 0b00001000)) {
        PORTB &= ~0b00011110;  // All LEDs off
    }
}
```

Port mapping (Arduino Uno):
- **Port B** (PORTB, DDRB, PINB): Pins 8-13
- **Port D** (PORTD, DDRD, PIND): Pins 0-7
- **Port C** (PORTC, DDRC, PINC): Pins A0-A5

---

## 4. Serial Communication -- UART

### UART Fundamentals

UART (Universal Asynchronous Receiver/Transmitter) is the most basic serial protocol:

| Parameter | Description |
|-----------|-------------|
| Baud Rate | Bits per second (9600, 115200 common) |
| Data Bits | 5-9 (usually 8) |
| Parity | None, Even, Odd |
| Stop Bits | 1 or 2 |

Common setting: **8N1** (8 data bits, No parity, 1 stop bit)

### Serial Output

```cpp
void setup() {
    Serial.begin(9600);
    Serial.println("=== Serial Demo ===");
}

void loop() {
    int sensorValue = analogRead(A0);
    float voltage = sensorValue * (5.0 / 1023.0);

    Serial.print("Raw: ");
    Serial.print(sensorValue);
    Serial.print(" Voltage: ");
    Serial.println(voltage, 2);

    // Formatted output
    char buffer[50];
    sprintf(buffer, "ADC=%d V=%.2f", sensorValue, voltage);
    Serial.println(buffer);

    delay(1000);
}
```

### Command Parsing

```cpp
#define MAX_INPUT 64

char inputBuffer[MAX_INPUT];
int inputIndex = 0;
const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    Serial.begin(9600);
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
    Serial.println("Commands: SET <led> <0|1>, PATTERN <0-15>, HELP");
}

void processCommand(char* input) {
    char* cmd = strtok(input, " ");
    if (cmd == NULL) return;

    for (int i = 0; cmd[i]; i++) cmd[i] = toupper(cmd[i]);

    if (strcmp(cmd, "SET") == 0) {
        char* ledStr = strtok(NULL, " ");
        char* stateStr = strtok(NULL, " ");
        if (ledStr && stateStr) {
            int led = atoi(ledStr);
            int state = atoi(stateStr);
            if (led >= 0 && led < NUM_LEDS) {
                digitalWrite(LED_PINS[led], state ? HIGH : LOW);
                Serial.print("LED ");
                Serial.print(led);
                Serial.println(state ? " ON" : " OFF");
            }
        }
    } else if (strcmp(cmd, "PATTERN") == 0) {
        char* valStr = strtok(NULL, " ");
        if (valStr) {
            int pattern = atoi(valStr) & 0x0F;
            for (int i = 0; i < NUM_LEDS; i++) {
                digitalWrite(LED_PINS[i], (pattern >> i) & 1);
            }
            Serial.print("Pattern set: ");
            Serial.println(pattern);
        }
    } else if (strcmp(cmd, "HELP") == 0) {
        Serial.println("  SET <0-3> <0|1>");
        Serial.println("  PATTERN <0-15>");
    }
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (inputIndex > 0) {
                inputBuffer[inputIndex] = '\0';
                processCommand(inputBuffer);
                inputIndex = 0;
            }
        } else if (inputIndex < MAX_INPUT - 1) {
            inputBuffer[inputIndex++] = c;
        }
    }
}
```

### Binary Protocol

```cpp
#define STX 0x02
#define ETX 0x03
#define MSG_LED_SET 0x01
#define MSG_ACK     0x10

void sendMessage(byte type, byte* data, byte length) {
    byte checksum = type ^ length;
    for (int i = 0; i < length; i++) checksum ^= data[i];

    Serial.write(STX);
    Serial.write(type);
    Serial.write(length);
    Serial.write(data, length);
    Serial.write(checksum);
    Serial.write(ETX);
}

// Receive state machine
enum RxState { WAIT_STX, WAIT_TYPE, WAIT_LENGTH, WAIT_DATA, WAIT_CHECKSUM, WAIT_ETX };
RxState rxState = WAIT_STX;
byte rxType, rxLength, rxChecksum, rxData[32], rxIndex;

void processRx(byte b) {
    switch (rxState) {
        case WAIT_STX:
            if (b == STX) rxState = WAIT_TYPE;
            break;
        case WAIT_TYPE:
            rxType = b; rxChecksum = b; rxState = WAIT_LENGTH;
            break;
        case WAIT_LENGTH:
            rxLength = b; rxChecksum ^= b; rxIndex = 0;
            rxState = (rxLength > 0) ? WAIT_DATA : WAIT_CHECKSUM;
            break;
        case WAIT_DATA:
            rxData[rxIndex++] = b; rxChecksum ^= b;
            if (rxIndex >= rxLength) rxState = WAIT_CHECKSUM;
            break;
        case WAIT_CHECKSUM:
            rxState = (b == rxChecksum) ? WAIT_ETX : WAIT_STX;
            break;
        case WAIT_ETX:
            if (b == ETX) {
                // Process valid message
                if (rxType == MSG_LED_SET && rxLength >= 1) {
                    digitalWrite(13, rxData[0] ? HIGH : LOW);
                    byte ack[] = {0x00};
                    sendMessage(MSG_ACK, ack, 1);
                }
            }
            rxState = WAIT_STX;
            break;
    }
}
```

---

## 5. I2C and SPI Protocols

### I2C (Inter-Integrated Circuit)

Two-wire synchronous protocol with addressing:

| Feature | Description |
|---------|-------------|
| Wires | 2 (SDA data, SCL clock) |
| Speed | 100kHz (standard), 400kHz (fast) |
| Addressing | 7-bit (up to 128 devices on one bus) |
| Use cases | Sensors, displays, EEPROMs |

```cpp
#include <Wire.h>

// Read temperature from LM75 sensor
const uint8_t LM75_ADDR = 0x48;

float readTemperature() {
    Wire.beginTransmission(LM75_ADDR);
    Wire.write(0x00);  // Temperature register
    Wire.endTransmission();

    Wire.requestFrom(LM75_ADDR, (uint8_t)2);
    if (Wire.available() >= 2) {
        int16_t temp = Wire.read() << 8;
        temp |= Wire.read();
        temp >>= 5;  // 11-bit data
        return temp * 0.125;
    }
    return 0.0;
}

void setup() {
    Wire.begin();
    Serial.begin(9600);
}

void loop() {
    Serial.print("Temperature: ");
    Serial.print(readTemperature());
    Serial.println(" C");
    delay(1000);
}
```

### SPI (Serial Peripheral Interface)

Four-wire high-speed protocol:

| Feature | Description |
|---------|-------------|
| Wires | 4 (MOSI, MISO, SCK, SS) |
| Speed | Up to several MHz |
| Addressing | Chip select (SS) pin per device |
| Use cases | SD cards, displays, high-speed sensors |

```cpp
#include <SPI.h>

const int SS_PIN = 10;

void setup() {
    pinMode(SS_PIN, OUTPUT);
    digitalWrite(SS_PIN, HIGH);  // Deactivate
    SPI.begin();
    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
}

uint8_t spiTransfer(uint8_t data) {
    digitalWrite(SS_PIN, LOW);
    uint8_t result = SPI.transfer(data);
    digitalWrite(SS_PIN, HIGH);
    return result;
}
```

### SD Card Data Logging (SPI)

```cpp
#include <SPI.h>
#include <SD.h>

const int CS_PIN = 4;

void setup() {
    Serial.begin(9600);
    if (!SD.begin(CS_PIN)) {
        Serial.println("SD card failed!");
        return;
    }

    File dataFile = SD.open("log.csv", FILE_WRITE);
    if (dataFile) {
        dataFile.println("timestamp,sensor");
        dataFile.close();
    }
}

void loop() {
    File dataFile = SD.open("log.csv", FILE_WRITE);
    if (dataFile) {
        int sensorValue = analogRead(A0);
        dataFile.print(millis());
        dataFile.print(",");
        dataFile.println(sensorValue);
        dataFile.close();
    }
    delay(1000);
}
```

---

## 6. PWM and Timers

### PWM Basics

PWM (Pulse Width Modulation) creates analog-like output by varying the duty cycle:

```cpp
const int LED_PIN = 9;  // PWM-capable pin

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Fade in
    for (int brightness = 0; brightness <= 255; brightness++) {
        analogWrite(LED_PIN, brightness);
        delay(10);
    }
    // Fade out
    for (int brightness = 255; brightness >= 0; brightness--) {
        analogWrite(LED_PIN, brightness);
        delay(10);
    }
}
```

### Hardware Timer Interrupt (AVR)

```c
#include <avr/io.h>
#include <avr/interrupt.h>

// Timer1 CTC mode: interrupt every 1 second
void timer1_init(void) {
    TCCR1B |= (1 << WGM12);         // CTC mode
    OCR1A = 62500 - 1;               // 16MHz / 256 / 62500 = 1Hz
    TCCR1B |= (1 << CS12);           // Prescaler 256
    TIMSK1 |= (1 << OCIE1A);        // Enable compare interrupt
    sei();                            // Enable global interrupts
}

ISR(TIMER1_COMPA_vect) {
    PORTB ^= (1 << PB0);  // Toggle LED
}

int main(void) {
    DDRB |= (1 << PB0);
    timer1_init();
    while (1) { /* Main loop */ }
    return 0;
}
```

### Servo Motor Control

```cpp
#include <Servo.h>

Servo myServo;

void setup() {
    myServo.attach(9);
}

void loop() {
    for (int angle = 0; angle <= 180; angle++) {
        myServo.write(angle);
        delay(15);
    }
    for (int angle = 180; angle >= 0; angle--) {
        myServo.write(angle);
        delay(15);
    }
}
```

---

## 7. ADC and Sensor Data

### Reading Analog Values

```cpp
const int SENSOR_PIN = A0;

void setup() {
    Serial.begin(9600);
}

void loop() {
    int rawValue = analogRead(SENSOR_PIN);  // 0-1023 (10-bit)
    float voltage = rawValue * (5.0 / 1023.0);

    Serial.print("Raw: ");
    Serial.print(rawValue);
    Serial.print(" Voltage: ");
    Serial.println(voltage, 2);

    delay(100);
}
```

### Noise Filtering

```cpp
const int SENSOR_PIN = A0;
const int NUM_SAMPLES = 10;

// Moving average
int readFiltered() {
    long sum = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        sum += analogRead(SENSOR_PIN);
        delay(1);
    }
    return sum / NUM_SAMPLES;
}

// Median filter
int readMedian() {
    int samples[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) {
        samples[i] = analogRead(SENSOR_PIN);
        delay(1);
    }

    // Bubble sort
    for (int i = 0; i < NUM_SAMPLES - 1; i++) {
        for (int j = i + 1; j < NUM_SAMPLES; j++) {
            if (samples[i] > samples[j]) {
                int temp = samples[i];
                samples[i] = samples[j];
                samples[j] = temp;
            }
        }
    }
    return samples[NUM_SAMPLES / 2];
}
```

### Temperature Sensor (TMP36)

```cpp
const int TEMP_PIN = A0;

float readTemperature() {
    int rawValue = analogRead(TEMP_PIN);
    float voltage = rawValue * (5.0 / 1023.0);
    // TMP36: 10mV/C, 500mV at 0C
    return (voltage - 0.5) * 100.0;
}
```

---

## 8. Volatile and Register Access

### The volatile Keyword

In embedded C, `volatile` tells the compiler a variable may change outside the program's control (hardware registers, ISR-modified variables):

```c
// Without volatile: compiler may optimize away the read
volatile uint8_t *PORTA_REG = (volatile uint8_t *)0x3B;

// ISR-shared variable must be volatile
volatile bool flag = false;

ISR(INT0_vect) {
    flag = true;  // Set by hardware interrupt
}

int main(void) {
    while (!flag) {
        // Without volatile, compiler might optimize this to while(1)
    }
    // Process event
}
```

### Memory-Mapped Register Access

On bare-metal systems, peripherals are controlled through memory-mapped registers:

```c
#include <stdint.h>

// STM32-style register definitions
#define GPIOA_BASE  0x40020000
#define GPIOA_MODER (*(volatile uint32_t *)(GPIOA_BASE + 0x00))
#define GPIOA_ODR   (*(volatile uint32_t *)(GPIOA_BASE + 0x14))

void gpio_init(void) {
    // Set pin 5 as output (MODER bits [11:10] = 01)
    GPIOA_MODER &= ~(3 << 10);  // Clear bits
    GPIOA_MODER |=  (1 << 10);  // Set output mode
}

void gpio_toggle_pin5(void) {
    GPIOA_ODR ^= (1 << 5);  // Toggle pin 5
}
```

### Bit-Band Access (ARM Cortex-M)

ARM Cortex-M processors support bit-banding, which maps each bit to a word address for atomic bit manipulation:

```c
// Bit-band formula for peripheral region
#define BITBAND_PERI(addr, bit) \
    (*(volatile uint32_t *)(0x42000000 + \
     ((uint32_t)(addr) - 0x40000000) * 32 + (bit) * 4))

// Atomic bit set/clear on GPIO
#define LED_BIT BITBAND_PERI(&GPIOA_ODR, 5)

LED_BIT = 1;  // Set pin 5 HIGH (atomic, no read-modify-write)
LED_BIT = 0;  // Set pin 5 LOW
```

---

## Exercises

### Exercise 1: Traffic Light Controller

Implement a traffic light with 3 LEDs (red, yellow, green):
- Red: 3 seconds
- Red+Yellow: 1 second
- Green: 3 seconds
- Yellow: 1 second
- Add a pedestrian button that triggers a walk signal

### Exercise 2: Serial Command Interface

Build a terminal interface that controls 4 LEDs with serial commands:
- `led <n> on/off` -- control individual LEDs
- `pattern <0-15>` -- set LED pattern from bitmask
- `status` -- show current LED states and uptime
- `help` -- show available commands

### Exercise 3: I2C Temperature Logger

Read temperature from an I2C sensor every 5 seconds. Display on serial and log to SD card in CSV format with timestamps.

### Exercise 4: Reaction Time Game

Build a reaction-time game using an LED and a button:
- LED turns on after a random delay (2-5 seconds)
- Player presses the button as fast as possible
- Display reaction time in milliseconds on serial
- Track the best time across rounds

### Exercise 5: Multi-Sensor Dashboard

Combine ADC (light sensor), I2C (temperature), and serial output to create a sensor dashboard that updates every second. Apply moving-average filtering to the ADC readings. Use formatted output to create a readable table.

---

## Next Steps

With embedded fundamentals covered, proceed to:
- [Debugging and Profiling](./15_Debugging_and_Profiling.md) -- Advanced tools for finding bugs and optimizing performance
