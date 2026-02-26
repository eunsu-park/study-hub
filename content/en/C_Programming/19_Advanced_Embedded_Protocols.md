# Advanced Embedded Protocols

**Previous**: [Debugging and Memory Analysis](./18_Debugging_Memory_Analysis.md) | **Next**: [Advanced C Pointers](./20_Advanced_Pointers.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Generate PWM signals to control LED brightness, servo angle, and DC motor speed
2. Configure hardware timers and write Interrupt Service Routines (ISRs) for periodic tasks
3. Communicate with I2C slave devices (sensors, displays) using the Arduino Wire library
4. Transfer data at high speed over SPI for peripherals such as SD cards
5. Read analog sensor values through the ADC and convert raw readings to physical units
6. Apply noise-filtering techniques (moving average, median filter) to improve ADC accuracy
7. Combine multiple protocols (I2C, SPI, ADC) in a practical data-logging project

---

Once you can blink an LED and read a button, the next frontier is talking to the rich ecosystem of sensors, displays, and actuators that make embedded systems useful. PWM lets you dim lights and spin motors, I2C and SPI connect you to temperature sensors and SD cards, and ADC brings the analog world into your digital program. This lesson gives you the protocol toolkit to build real embedded applications.

**Difficulty**: Advanced

**Prerequisites**: Embedded basics, bit operations, pointers

---

## Table of Contents

1. [PWM (Pulse Width Modulation)](#pwm-pulse-width-modulation)
2. [Timers and Interrupts](#timers-and-interrupts)
3. [I2C Communication](#i2c-communication)
4. [SPI Communication](#spi-communication)
5. [ADC (Analog to Digital Converter)](#adc-analog-to-digital-converter)
6. [Practical Project](#practical-project)

---

## PWM (Pulse Width Modulation)

### What is PWM?

PWM is a technique that achieves analog-like effects by adjusting the ON/OFF ratio of digital signals.

```
Duty Cycle 25%:
████________________████________________
|<-- Period -->|

Duty Cycle 50%:
████████________████████________
|<-- Period -->|

Duty Cycle 75%:
████████████____████████████____
|<-- Period -->|
```

### PWM Terminology

| Term | Description |
|------|------|
| Period | Total time of one cycle |
| Duty Cycle | Ratio of HIGH state (%) |
| Frequency | Cycles per second (Hz) |

### Arduino PWM

```cpp
// Arduino PWM basics
const int LED_PIN = 9;  // PWM-capable pin

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Brightness control with 0-255 values
    for (int brightness = 0; brightness <= 255; brightness++) {
        analogWrite(LED_PIN, brightness);
        delay(10);
    }

    for (int brightness = 255; brightness >= 0; brightness--) {
        analogWrite(LED_PIN, brightness);
        delay(10);
    }
}
```

### Servo Motor Control

```cpp
#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 9;

void setup() {
    myServo.attach(SERVO_PIN);
}

void loop() {
    // Move from 0 to 180 degrees
    for (int angle = 0; angle <= 180; angle++) {
        myServo.write(angle);
        delay(15);
    }

    // Move from 180 to 0 degrees
    for (int angle = 180; angle >= 0; angle--) {
        myServo.write(angle);
        delay(15);
    }
}
```

### DC Motor Speed Control

```cpp
const int MOTOR_PIN = 9;     // PWM pin
const int DIR_PIN = 8;       // Direction control pin
const int POT_PIN = A0;      // Potentiometer

void setup() {
    pinMode(MOTOR_PIN, OUTPUT);
    pinMode(DIR_PIN, OUTPUT);
}

void loop() {
    int potValue = analogRead(POT_PIN);  // 0-1023
    int speed = map(potValue, 0, 1023, 0, 255);

    digitalWrite(DIR_PIN, HIGH);  // Forward
    analogWrite(MOTOR_PIN, speed);

    delay(100);
}
```

### Software PWM

```c
// For pins without hardware PWM
#include <avr/io.h>
#include <util/delay.h>

void software_pwm(volatile uint8_t *port, uint8_t pin,
                  uint8_t duty, uint16_t period_us) {
    uint16_t on_time = (period_us * duty) / 100;
    uint16_t off_time = period_us - on_time;

    *port |= (1 << pin);   // HIGH
    _delay_us(on_time);
    *port &= ~(1 << pin);  // LOW
    _delay_us(off_time);
}

int main(void) {
    DDRB |= (1 << PB0);  // Set as output

    while (1) {
        software_pwm(&PORTB, PB0, 30, 1000);  // 30% duty
    }
    return 0;
}
```

---

## Timers and Interrupts

### Timer Overview

```
┌─────────────┐     Clock     ┌─────────────┐
│ Clock Source │────────────▶│   Timer     │
└─────────────┘              │  Counter    │
                             └──────┬──────┘
                                    │
                             ┌──────▼──────┐
                             │  Comparator │──▶ Interrupt/PWM
                             └─────────────┘
```

### AVR Timer Setup

```c
#include <avr/io.h>
#include <avr/interrupt.h>

// Timer1 CTC mode with interrupt every 1 second
void timer1_init(void) {
    // CTC mode setup
    TCCR1B |= (1 << WGM12);

    // Compare value (16MHz / 256 / 62500 = 1Hz)
    OCR1A = 62500 - 1;

    // Prescaler 256
    TCCR1B |= (1 << CS12);

    // Enable compare match interrupt
    TIMSK1 |= (1 << OCIE1A);

    // Enable global interrupts
    sei();
}

// Interrupt Service Routine
ISR(TIMER1_COMPA_vect) {
    PORTB ^= (1 << PB0);  // Toggle LED
}

int main(void) {
    DDRB |= (1 << PB0);
    timer1_init();

    while (1) {
        // Main loop
    }
    return 0;
}
```

### Arduino Timer Interrupt

```cpp
// Using TimerOne library
#include <TimerOne.h>

const int LED_PIN = 13;
volatile bool ledState = false;

void timerISR() {
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
}

void setup() {
    pinMode(LED_PIN, OUTPUT);

    Timer1.initialize(500000);  // 500ms (in microseconds)
    Timer1.attachInterrupt(timerISR);
}

void loop() {
    // Other tasks
}
```

### External Interrupts

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

    // FALLING: Interrupt on HIGH→LOW transition
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN),
                    buttonISR, FALLING);
}

void loop() {
    Serial.println(buttonCount);
    delay(500);
}
```

---

## I2C Communication

### I2C Overview

I2C is a 2-wire synchronous communication protocol.

```
        VCC
         │
        ┌┴┐
        │R│ Pull-up resistor
        └┬┘
         │
Master ──┼─────────────────── SDA (Data)
         │
        ┌┴┐
        │R│ Pull-up resistor
        └┬┘
         │
Master ──┼─────────────────── SCL (Clock)
         │
      ┌──┴──┐   ┌──┴──┐
      │Slave│   │Slave│
      │ 0x48│   │ 0x50│
      └─────┘   └─────┘
```

### I2C Features

| Feature | Description |
|------|------|
| Wire count | 2 (SDA, SCL) |
| Speed | 100kHz (standard), 400kHz (fast) |
| Address | 7-bit (128 devices) |
| Master/Slave | Multi-master supported |

### Arduino Wire Library

```cpp
#include <Wire.h>

// Send data as I2C master
void sendToSlave(uint8_t address, uint8_t *data, size_t len) {
    Wire.beginTransmission(address);
    Wire.write(data, len);
    Wire.endTransmission();
}

// Receive data as I2C master
void readFromSlave(uint8_t address, uint8_t *buffer, size_t len) {
    Wire.requestFrom(address, len);

    size_t i = 0;
    while (Wire.available() && i < len) {
        buffer[i++] = Wire.read();
    }
}

void setup() {
    Wire.begin();  // Initialize as master
    Serial.begin(9600);
}

void loop() {
    uint8_t data[] = {0x01, 0x02, 0x03};
    sendToSlave(0x48, data, sizeof(data));

    uint8_t buffer[4];
    readFromSlave(0x48, buffer, 4);

    for (int i = 0; i < 4; i++) {
        Serial.print(buffer[i], HEX);
        Serial.print(" ");
    }
    Serial.println();

    delay(1000);
}
```

### I2C Temperature Sensor (LM75)

```cpp
#include <Wire.h>

const uint8_t LM75_ADDR = 0x48;
const uint8_t TEMP_REG = 0x00;

float readTemperature() {
    Wire.beginTransmission(LM75_ADDR);
    Wire.write(TEMP_REG);
    Wire.endTransmission();

    Wire.requestFrom(LM75_ADDR, (uint8_t)2);

    if (Wire.available() >= 2) {
        int16_t temp = Wire.read() << 8;
        temp |= Wire.read();
        temp >>= 5;  // 11-bit data

        return temp * 0.125;  // 0.125°C units
    }

    return 0.0;
}

void setup() {
    Wire.begin();
    Serial.begin(9600);
}

void loop() {
    float temp = readTemperature();
    Serial.print("Temperature: ");
    Serial.print(temp);
    Serial.println(" C");
    delay(1000);
}
```

### I2C OLED Display

```cpp
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT,
                          &Wire, OLED_RESET);

void setup() {
    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("SSD1306 allocation failed");
        for (;;);
    }

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Hello, World!");
    display.display();
}

void loop() {
    static int count = 0;

    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Count: ");
    display.println(count++);
    display.display();

    delay(1000);
}
```

---

## SPI Communication

### SPI Overview

SPI is a 4-wire high-speed synchronous communication protocol.

```
Master              Slave
  │                   │
  ├───── MOSI ───────▶│  (Master Out, Slave In)
  │                   │
  │◀───── MISO ───────┤  (Master In, Slave Out)
  │                   │
  ├───── SCK ────────▶│  (Clock)
  │                   │
  ├───── SS ─────────▶│  (Slave Select)
  │                   │
```

### SPI vs I2C

| Feature | SPI | I2C |
|------|-----|-----|
| Wire count | 4 | 2 |
| Speed | Faster (several MHz) | Slower (400kHz) |
| Distance | Short | Medium |
| Device count | Depends on SS pins | 127 via addresses |

### Arduino SPI Usage

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
    digitalWrite(SS_PIN, LOW);   // Select slave
    uint8_t result = SPI.transfer(data);
    digitalWrite(SS_PIN, HIGH);  // Release slave
    return result;
}

void loop() {
    uint8_t response = spiTransfer(0x42);
    Serial.println(response, HEX);
    delay(100);
}
```

### SPI SD Card

```cpp
#include <SPI.h>
#include <SD.h>

const int CS_PIN = 4;

void setup() {
    Serial.begin(9600);

    if (!SD.begin(CS_PIN)) {
        Serial.println("SD card initialization failed!");
        return;
    }
    Serial.println("SD card initialized.");

    // Write file
    File dataFile = SD.open("data.txt", FILE_WRITE);
    if (dataFile) {
        dataFile.println("Hello, SD Card!");
        dataFile.close();
        Serial.println("Data written.");
    }

    // Read file
    dataFile = SD.open("data.txt");
    if (dataFile) {
        while (dataFile.available()) {
            Serial.write(dataFile.read());
        }
        dataFile.close();
    }
}

void loop() {
    // Sensor data logging
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

## ADC (Analog to Digital Converter)

### ADC Overview

```
Analog Input    ADC     Digital Output
  0V ~ 5V  ──▶ ┌───┐ ──▶ 0 ~ 1023 (10-bit)
               │ADC│
               └───┘
```

### Arduino ADC

```cpp
const int SENSOR_PIN = A0;

void setup() {
    Serial.begin(9600);
    // analogReference(DEFAULT);  // 5V reference
    // analogReference(INTERNAL); // 1.1V reference (more precise)
}

void loop() {
    int rawValue = analogRead(SENSOR_PIN);  // 0-1023

    // Convert to voltage
    float voltage = rawValue * (5.0 / 1023.0);

    Serial.print("Raw: ");
    Serial.print(rawValue);
    Serial.print(" Voltage: ");
    Serial.println(voltage);

    delay(100);
}
```

### Temperature Sensor (TMP36)

```cpp
const int TEMP_PIN = A0;

float readTemperature() {
    int rawValue = analogRead(TEMP_PIN);

    // TMP36: 10mV/°C, 500mV at 0°C
    float voltage = rawValue * (5.0 / 1023.0);
    float tempC = (voltage - 0.5) * 100.0;

    return tempC;
}

void setup() {
    Serial.begin(9600);
}

void loop() {
    float temp = readTemperature();
    Serial.print("Temperature: ");
    Serial.print(temp);
    Serial.println(" C");
    delay(1000);
}
```

### Light Sensor (LDR)

```cpp
const int LDR_PIN = A0;
const int LED_PIN = 9;

void setup() {
    pinMode(LED_PIN, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    int lightLevel = analogRead(LDR_PIN);

    // Bright → dim LED, Dark → bright LED
    int brightness = map(lightLevel, 0, 1023, 255, 0);
    analogWrite(LED_PIN, brightness);

    Serial.print("Light: ");
    Serial.print(lightLevel);
    Serial.print(" LED: ");
    Serial.println(brightness);

    delay(100);
}
```

### Reading Multiple ADC Channels

```cpp
const int NUM_CHANNELS = 4;
const int ADC_PINS[] = {A0, A1, A2, A3};

void setup() {
    Serial.begin(9600);
}

void loop() {
    for (int i = 0; i < NUM_CHANNELS; i++) {
        int value = analogRead(ADC_PINS[i]);
        Serial.print("CH");
        Serial.print(i);
        Serial.print(": ");
        Serial.print(value);
        Serial.print("\t");
    }
    Serial.println();
    delay(500);
}
```

### ADC Noise Filtering

```cpp
const int SENSOR_PIN = A0;
const int NUM_SAMPLES = 10;

// Moving average filter
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

    // Sort
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

void setup() {
    Serial.begin(9600);
}

void loop() {
    int raw = analogRead(SENSOR_PIN);
    int filtered = readFiltered();
    int median = readMedian();

    Serial.print("Raw: ");
    Serial.print(raw);
    Serial.print(" Filtered: ");
    Serial.print(filtered);
    Serial.print(" Median: ");
    Serial.println(median);

    delay(100);
}
```

---

## Practical Project

### Temperature/Humidity Data Logger

A project that combines I2C, SPI, and ADC.

```cpp
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <DHT.h>

// Pin setup
const int DHT_PIN = 2;
const int SD_CS_PIN = 4;
const int LDR_PIN = A0;

// DHT sensor
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

// Logging interval (milliseconds)
const unsigned long LOG_INTERVAL = 5000;
unsigned long lastLogTime = 0;

void setup() {
    Serial.begin(9600);

    // Initialize DHT
    dht.begin();

    // Initialize SD card
    if (!SD.begin(SD_CS_PIN)) {
        Serial.println("SD card failed!");
        while (1);
    }
    Serial.println("SD card initialized.");

    // Write CSV header
    File dataFile = SD.open("log.csv", FILE_WRITE);
    if (dataFile) {
        dataFile.println("timestamp,temperature,humidity,light");
        dataFile.close();
    }
}

void loop() {
    unsigned long currentTime = millis();

    if (currentTime - lastLogTime >= LOG_INTERVAL) {
        lastLogTime = currentTime;

        // Read sensors
        float temp = dht.readTemperature();
        float humidity = dht.readHumidity();
        int light = analogRead(LDR_PIN);

        // Validate
        if (isnan(temp) || isnan(humidity)) {
            Serial.println("DHT read failed!");
            return;
        }

        // Serial output
        Serial.print("Temp: ");
        Serial.print(temp);
        Serial.print("C, Humidity: ");
        Serial.print(humidity);
        Serial.print("%, Light: ");
        Serial.println(light);

        // Write to SD card
        File dataFile = SD.open("log.csv", FILE_WRITE);
        if (dataFile) {
            dataFile.print(currentTime);
            dataFile.print(",");
            dataFile.print(temp);
            dataFile.print(",");
            dataFile.print(humidity);
            dataFile.print(",");
            dataFile.println(light);
            dataFile.close();
        }
    }
}
```

### Wiring Diagram

```
Arduino Uno
    │
    ├── D2  ────── DHT22 DATA
    ├── D4  ────── SD CS
    ├── D11 ────── SD MOSI
    ├── D12 ────── SD MISO
    ├── D13 ────── SD SCK
    ├── A0  ────── LDR (with voltage divider)
    ├── 5V  ────── VCC (sensors)
    └── GND ────── GND (sensors)
```

---

## Practice Problems

### Problem 1: PWM LED Control

Write a program that increases LED brightness by 25% each time a button is pressed. At 100%, it returns to 0%.

<details>
<summary>Show Answer</summary>

```cpp
const int LED_PIN = 9;
const int BUTTON_PIN = 2;

int brightness = 0;
bool lastButtonState = HIGH;

void setup() {
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
}

void loop() {
    bool buttonState = digitalRead(BUTTON_PIN);

    if (lastButtonState == HIGH && buttonState == LOW) {
        brightness = (brightness + 64) % 256;  // 25% = 64
        analogWrite(LED_PIN, brightness);
        delay(50);  // Debouncing
    }

    lastButtonState = buttonState;
}
```

</details>

### Problem 2: I2C Scanner

Write a scanner that finds addresses of all connected I2C devices.

<details>
<summary>Show Answer</summary>

```cpp
#include <Wire.h>

void setup() {
    Wire.begin();
    Serial.begin(9600);
    Serial.println("I2C Scanner");
}

void loop() {
    int deviceCount = 0;

    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        uint8_t error = Wire.endTransmission();

        if (error == 0) {
            Serial.print("Found device at 0x");
            Serial.println(addr, HEX);
            deviceCount++;
        }
    }

    Serial.print("Found ");
    Serial.print(deviceCount);
    Serial.println(" device(s)");
    Serial.println();

    delay(5000);
}
```

</details>

---

## References

- [Arduino Reference](https://www.arduino.cc/reference/en/)
- [AVR Libc Reference](https://www.nongnu.org/avr-libc/user-manual/)
- [I2C Protocol](https://i2c.info/)
- [SPI Protocol](https://en.wikipedia.org/wiki/Serial_Peripheral_Interface)
