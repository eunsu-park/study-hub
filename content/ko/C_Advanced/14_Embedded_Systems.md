# 임베디드 시스템

**이전**: [프로세스 간 통신과 시그널](./13_IPC_and_Signals.md) | **다음**: [디버깅과 프로파일링](./15_Debugging_and_Profiling.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C의 저수준 기능(volatile, 비트 연산, 메모리 매핑 I/O)이 임베디드 프로그래밍을 어떻게 가능하게 하는지 설명할 수 있다
2. 디지털 입출력을 위한 GPIO 핀을 설정하고 소프트웨어 디바운싱을 구현할 수 있다
3. UART 매개변수와 명령 파싱을 사용하여 시리얼 통신을 구현할 수 있다
4. 하드웨어 통신 프로토콜(I2C, SPI)과 ADC를 사용하여 센서 데이터를 처리할 수 있다
5. PWM을 사용한 아날로그 출력 제어와 하드웨어 타이머 설정을 적용할 수 있다

---

임베디드 시스템은 어디에나 있습니다 -- 전자레인지와 온도 조절기부터 자동차 ECU와 의료 기기까지. C는 최소한의 오버헤드로 직접적인 하드웨어 접근을 제공하기 때문에 이 분야를 지배합니다. 이 레슨에서는 필수 임베디드 프로그래밍 개념인 GPIO 제어, 시리얼 통신, 하드웨어 프로토콜, 아날로그 인터페이싱을 통합합니다. C의 저수준 기능이 실제 하드웨어에 어떻게 매핑되는지, 그리고 대화형 임베디드 애플리케이션을 구축하는 방법을 배웁니다.

**난이도**: 고급

**사전 지식**: 포인터, 비트 연산, 구조체

---

## 1. 임베디드에서의 C -- C가 지배하는 이유

### 임베디드 vs 범용 컴퓨팅

| 특성 | 범용 컴퓨터 | 임베디드 시스템 |
|------|------------|----------------|
| 목적 | 다양한 프로그램 실행 | 특정 기능 수행 |
| 메모리 | GB 단위 RAM | KB~MB 단위 SRAM |
| CPU 속도 | GHz 범위 | MHz 범위 |
| 저장소 | SSD/HDD (TB) | 플래시 (KB~MB) |
| 예시 | 노트북, 데스크탑 | 세탁기, ECU, IoT 센서 |

### C를 사용하는 이유

- **직접적인 하드웨어 접근**: 포인터로 메모리 매핑된 레지스터에 접근 가능
- **최소한의 오버헤드**: 가비지 컬렉터 없음, 런타임 없음
- **예측 가능한 성능**: 실시간 요구사항을 위한 결정론적 실행
- **작은 설치 공간**: "Hello World"가 플래시 수백 바이트에 들어갈 수 있음

### 크로스 컴파일

임베디드 타겟은 개발 머신과 다릅니다. PC(호스트)에서 타겟 MCU를 위해 컴파일합니다:

```bash
# Native compilation (runs on your PC)
gcc -o program program.c

# Cross-compilation (runs on ARM MCU)
arm-none-eabi-gcc -mcpu=cortex-m4 -o firmware.elf main.c
```

### MCU 아키텍처

마이크로컨트롤러(MCU)는 CPU, 메모리, 주변장치를 하나의 칩에 통합합니다:

| 구성요소 | 설명 |
|---------|------|
| **CPU 코어** | 명령어 실행 (예: ARM Cortex-M, AVR) |
| **플래시** | 비휘발성 프로그램 저장소 (32KB-2MB) |
| **SRAM** | 변수/스택을 위한 휘발성 데이터 저장소 (2KB-512KB) |
| **GPIO** | 범용 입출력 핀 |
| **타이머** | 타이밍/PWM을 위한 하드웨어 카운터 |
| **UART** | 시리얼 통신 |
| **ADC** | 아날로그-디지털 변환기 |
| **I2C/SPI** | 버스 통신 프로토콜 |

---

## 2. Arduino와 GPIO 기초

### Arduino 프로그램 구조

Arduino는 두 개의 필수 함수를 가진 C/C++ 방언을 사용합니다:

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

### 디지털 출력 -- LED 제어

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

### 다중 LED 패턴

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

## 3. GPIO 고급 -- 풀업, 디바운싱, 인터럽트

### 내부 풀업을 사용한 버튼 읽기

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

### 소프트웨어 디바운싱

기계식 버튼은 눌릴 때 노이즈(바운싱)를 발생시킵니다. 소프트웨어 디바운싱은 신호가 안정될 때까지 기다립니다:

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

### 인터럽트 기반 버튼

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

### 직접 레지스터 제어

최대 성능을 위해 GPIO 레지스터를 직접 조작합니다:

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

포트 매핑 (Arduino Uno):
- **Port B** (PORTB, DDRB, PINB): 핀 8-13
- **Port D** (PORTD, DDRD, PIND): 핀 0-7
- **Port C** (PORTC, DDRC, PINC): 핀 A0-A5

---

## 4. 시리얼 통신 -- UART

### UART 기초

UART(Universal Asynchronous Receiver/Transmitter)는 가장 기본적인 시리얼 프로토콜입니다:

| 매개변수 | 설명 |
|---------|------|
| 보드율 | 초당 비트 수 (9600, 115200이 일반적) |
| 데이터 비트 | 5-9 (보통 8) |
| 패리티 | 없음, 짝수, 홀수 |
| 정지 비트 | 1 또는 2 |

일반적인 설정: **8N1** (8 데이터 비트, 패리티 없음, 1 정지 비트)

### 시리얼 출력

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

### 명령 파싱

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

### 바이너리 프로토콜

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

## 5. I2C와 SPI 프로토콜

### I2C (Inter-Integrated Circuit)

주소 지정을 가진 2선 동기 프로토콜:

| 특성 | 설명 |
|------|------|
| 선 수 | 2 (SDA 데이터, SCL 클록) |
| 속도 | 100kHz (표준), 400kHz (고속) |
| 주소 지정 | 7비트 (하나의 버스에 최대 128개 장치) |
| 사용 사례 | 센서, 디스플레이, EEPROM |

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

4선 고속 프로토콜:

| 특성 | 설명 |
|------|------|
| 선 수 | 4 (MOSI, MISO, SCK, SS) |
| 속도 | 수 MHz까지 |
| 주소 지정 | 장치별 칩 선택(SS) 핀 |
| 사용 사례 | SD 카드, 디스플레이, 고속 센서 |

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

### SD 카드 데이터 로깅 (SPI)

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

## 6. PWM과 타이머

### PWM 기초

PWM(Pulse Width Modulation)은 듀티 사이클을 변화시켜 아날로그와 유사한 출력을 만듭니다:

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

### 하드웨어 타이머 인터럽트 (AVR)

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

### 서보 모터 제어

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

## 7. ADC와 센서 데이터

### 아날로그 값 읽기

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

### 노이즈 필터링

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

### 온도 센서 (TMP36)

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

## 8. volatile과 레지스터 접근

### volatile 키워드

임베디드 C에서 `volatile`은 프로그램의 제어 밖에서 변수가 변경될 수 있음을 컴파일러에 알립니다 (하드웨어 레지스터, ISR에서 수정된 변수):

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

### 메모리 매핑 레지스터 접근

베어 메탈 시스템에서 주변장치는 메모리 매핑된 레지스터를 통해 제어됩니다:

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

### 비트 밴드 접근 (ARM Cortex-M)

ARM Cortex-M 프로세서는 비트 밴딩을 지원하여 각 비트를 워드 주소에 매핑해 원자적 비트 조작을 가능하게 합니다:

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

## 연습 문제

### 연습 문제 1: 신호등 컨트롤러

3개의 LED(빨강, 노랑, 초록)로 신호등을 구현하세요:
- 빨강: 3초
- 빨강+노랑: 1초
- 초록: 3초
- 노랑: 1초
- 보행자 버튼을 추가하여 보행 신호를 트리거하세요

### 연습 문제 2: 시리얼 명령 인터페이스

시리얼 명령으로 4개의 LED를 제어하는 터미널 인터페이스를 구축하세요:
- `led <n> on/off` -- 개별 LED 제어
- `pattern <0-15>` -- 비트마스크로 LED 패턴 설정
- `status` -- 현재 LED 상태와 가동 시간 표시
- `help` -- 사용 가능한 명령 표시

### 연습 문제 3: I2C 온도 로거

I2C 센서에서 5초마다 온도를 읽습니다. 시리얼에 표시하고 타임스탬프와 함께 CSV 형식으로 SD 카드에 기록하세요.

### 연습 문제 4: 반응 시간 게임

LED와 버튼을 사용하여 반응 시간 게임을 만드세요:
- LED가 랜덤 지연(2-5초) 후에 켜짐
- 플레이어가 가능한 빨리 버튼을 누름
- 시리얼에 밀리초 단위의 반응 시간 표시
- 라운드를 거치며 최고 기록 추적

### 연습 문제 5: 멀티센서 대시보드

ADC(조도 센서), I2C(온도), 시리얼 출력을 결합하여 매초 업데이트되는 센서 대시보드를 만드세요. ADC 읽기에 이동 평균 필터링을 적용하세요. 서식화된 출력을 사용하여 읽기 쉬운 표를 만드세요.

---

## 다음 단계

임베디드 기초를 다루었으니 다음으로 진행하세요:
- [디버깅과 프로파일링](./15_Debugging_and_Profiling.md) -- 버그 찾기와 성능 최적화를 위한 고급 도구
