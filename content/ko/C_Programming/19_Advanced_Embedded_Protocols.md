# 고급 임베디드 프로토콜(Advanced Embedded Protocols)

**이전**: [디버깅과 메모리 분석](./18_Debugging_Memory_Analysis.md) | **다음**: [C 언어 포인터 심화](./20_Advanced_Pointers.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PWM 신호를 생성하여 LED 밝기, 서보 각도, DC 모터 속도를 제어한다
2. 하드웨어 타이머를 설정하고 주기적 작업을 위한 인터럽트 서비스 루틴(ISR, Interrupt Service Routine)을 작성한다
3. Arduino Wire 라이브러리를 사용하여 I2C 슬레이브 장치(센서, 디스플레이)와 통신한다
4. SD 카드 등 주변 장치에 SPI로 고속 데이터를 전송한다
5. ADC를 통해 아날로그 센서 값을 읽고 원시 값을 물리 단위로 변환한다
6. 이동 평균 및 중앙값 필터 등 노이즈 제거 기법을 적용하여 ADC 정확도를 높인다
7. 실용적인 데이터 로깅 프로젝트에서 여러 프로토콜(I2C, SPI, ADC)을 결합한다

---

LED를 깜빡이고 버튼을 읽을 수 있게 됐다면, 다음 과제는 임베디드 시스템을 실질적으로 유용하게 만드는 풍부한 센서·디스플레이·액추에이터 생태계와 통신하는 것입니다. PWM은 조명을 조절하고 모터를 돌리게 해주고, I2C와 SPI는 온도 센서와 SD 카드를 연결해주며, ADC는 아날로그 세계를 디지털 프로그램으로 끌어들입니다. 이 레슨은 실제 임베디드 애플리케이션을 구축하는 데 필요한 프로토콜 도구 상자를 제공합니다.

**난이도**: 고급(Advanced)

**선수 지식**: 임베디드 기초, 비트 연산, 포인터

---

## 목차

1. [PWM (Pulse Width Modulation)](#pwm-pulse-width-modulation)
2. [타이머와 인터럽트](#타이머와-인터럽트)
3. [I2C 통신](#i2c-통신)
4. [SPI 통신](#spi-통신)
5. [ADC (Analog to Digital Converter)](#adc-analog-to-digital-converter)
6. [실전 프로젝트](#실전-프로젝트)

---

## PWM (Pulse Width Modulation)

### PWM이란?

PWM은 디지털 신호의 ON/OFF 비율을 조절하여 아날로그와 유사한 효과를 내는 기술입니다.

```
듀티 사이클 25%:
████________________████________________
|<-- Period -->|

듀티 사이클 50%:
████████________████████________
|<-- Period -->|

듀티 사이클 75%:
████████████____████████████____
|<-- Period -->|
```

### PWM 용어

| 용어 | 설명 |
|------|------|
| 주기(Period) | 한 사이클의 전체 시간 |
| 듀티 사이클(Duty Cycle) | HIGH 상태의 비율 (%) |
| 주파수(Frequency) | 초당 사이클 수 (Hz) |

### Arduino PWM

```cpp
// Arduino PWM 기본
const int LED_PIN = 9;  // PWM 지원 핀

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // 0-255 값으로 밝기 조절
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

### 서보 모터 제어

```cpp
#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 9;

void setup() {
    myServo.attach(SERVO_PIN);
}

void loop() {
    // 0도에서 180도로 이동
    for (int angle = 0; angle <= 180; angle++) {
        myServo.write(angle);
        delay(15);
    }

    // 180도에서 0도로 이동
    for (int angle = 180; angle >= 0; angle--) {
        myServo.write(angle);
        delay(15);
    }
}
```

### DC 모터 속도 제어

```cpp
const int MOTOR_PIN = 9;     // PWM 핀
const int DIR_PIN = 8;       // 방향 제어 핀
const int POT_PIN = A0;      // 가변저항

void setup() {
    pinMode(MOTOR_PIN, OUTPUT);
    pinMode(DIR_PIN, OUTPUT);
}

void loop() {
    int potValue = analogRead(POT_PIN);  // 0-1023
    int speed = map(potValue, 0, 1023, 0, 255);

    digitalWrite(DIR_PIN, HIGH);  // 정방향
    analogWrite(MOTOR_PIN, speed);

    delay(100);
}
```

### 소프트웨어 PWM

```c
// 하드웨어 PWM이 없는 핀에서 사용
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
    DDRB |= (1 << PB0);  // 출력 설정

    while (1) {
        software_pwm(&PORTB, PB0, 30, 1000);  // 30% 듀티
    }
    return 0;
}
```

---

## 타이머와 인터럽트

### 타이머 개요

```
┌─────────────┐     클럭     ┌─────────────┐
│  클럭 소스   │────────────▶│   타이머     │
└─────────────┘              │  카운터      │
                             └──────┬──────┘
                                    │
                             ┌──────▼──────┐
                             │   비교기     │──▶ 인터럽트/PWM
                             └─────────────┘
```

### AVR 타이머 설정

```c
#include <avr/io.h>
#include <avr/interrupt.h>

// Timer1 CTC 모드로 1초마다 인터럽트
void timer1_init(void) {
    // CTC 모드 설정
    TCCR1B |= (1 << WGM12);

    // 비교값 설정 (16MHz / 256 / 62500 = 1Hz)
    OCR1A = 62500 - 1;

    // 프리스케일러 256
    TCCR1B |= (1 << CS12);

    // 비교 일치 인터럽트 활성화
    TIMSK1 |= (1 << OCIE1A);

    // 전역 인터럽트 활성화
    sei();
}

// 인터럽트 서비스 루틴
ISR(TIMER1_COMPA_vect) {
    PORTB ^= (1 << PB0);  // LED 토글
}

int main(void) {
    DDRB |= (1 << PB0);
    timer1_init();

    while (1) {
        // 메인 루프
    }
    return 0;
}
```

### Arduino 타이머 인터럽트

```cpp
// TimerOne 라이브러리 사용
#include <TimerOne.h>

const int LED_PIN = 13;
volatile bool ledState = false;

void timerISR() {
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
}

void setup() {
    pinMode(LED_PIN, OUTPUT);

    Timer1.initialize(500000);  // 500ms (마이크로초 단위)
    Timer1.attachInterrupt(timerISR);
}

void loop() {
    // 다른 작업 수행
}
```

### 외부 인터럽트

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

    // FALLING: HIGH→LOW 전환 시 인터럽트
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN),
                    buttonISR, FALLING);
}

void loop() {
    Serial.println(buttonCount);
    delay(500);
}
```

---

## I2C 통신

### I2C 개요

I2C는 2선식 동기 통신 프로토콜입니다.

```
        VCC
         │
        ┌┴┐
        │R│ 풀업 저항
        └┬┘
         │
Master ──┼─────────────────── SDA (데이터)
         │
        ┌┴┐
        │R│ 풀업 저항
        └┬┘
         │
Master ──┼─────────────────── SCL (클럭)
         │
      ┌──┴──┐   ┌──┴──┐
      │Slave│   │Slave│
      │ 0x48│   │ 0x50│
      └─────┘   └─────┘
```

### I2C 특징

| 특징 | 설명 |
|------|------|
| 선 수 | 2개 (SDA, SCL) |
| 속도 | 100kHz (표준), 400kHz (고속) |
| 주소 | 7비트 (128개 장치) |
| 마스터/슬레이브 | 다중 마스터 지원 |

### Arduino Wire 라이브러리

```cpp
#include <Wire.h>

// I2C 마스터로 데이터 전송
void sendToSlave(uint8_t address, uint8_t *data, size_t len) {
    Wire.beginTransmission(address);
    Wire.write(data, len);
    Wire.endTransmission();
}

// I2C 마스터로 데이터 수신
void readFromSlave(uint8_t address, uint8_t *buffer, size_t len) {
    Wire.requestFrom(address, len);

    size_t i = 0;
    while (Wire.available() && i < len) {
        buffer[i++] = Wire.read();
    }
}

void setup() {
    Wire.begin();  // 마스터로 초기화
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

### I2C 온도 센서 (LM75)

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
        temp >>= 5;  // 11비트 데이터

        return temp * 0.125;  // 0.125°C 단위
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

### I2C OLED 디스플레이

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

## SPI 통신

### SPI 개요

SPI는 4선식 고속 동기 통신 프로토콜입니다.

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

| 특징 | SPI | I2C |
|------|-----|-----|
| 선 수 | 4개 | 2개 |
| 속도 | 더 빠름 (수 MHz) | 느림 (400kHz) |
| 거리 | 짧음 | 중간 |
| 장치 수 | SS 핀 수에 따라 | 주소로 127개 |

### Arduino SPI 사용

```cpp
#include <SPI.h>

const int SS_PIN = 10;

void setup() {
    pinMode(SS_PIN, OUTPUT);
    digitalWrite(SS_PIN, HIGH);  // 비활성화

    SPI.begin();
    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
}

uint8_t spiTransfer(uint8_t data) {
    digitalWrite(SS_PIN, LOW);   // 슬레이브 선택
    uint8_t result = SPI.transfer(data);
    digitalWrite(SS_PIN, HIGH);  // 슬레이브 해제
    return result;
}

void loop() {
    uint8_t response = spiTransfer(0x42);
    Serial.println(response, HEX);
    delay(100);
}
```

### SPI SD 카드

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

    // 파일 쓰기
    File dataFile = SD.open("data.txt", FILE_WRITE);
    if (dataFile) {
        dataFile.println("Hello, SD Card!");
        dataFile.close();
        Serial.println("Data written.");
    }

    // 파일 읽기
    dataFile = SD.open("data.txt");
    if (dataFile) {
        while (dataFile.available()) {
            Serial.write(dataFile.read());
        }
        dataFile.close();
    }
}

void loop() {
    // 센서 데이터 로깅
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

### ADC 개요

```
아날로그 입력    ADC     디지털 출력
  0V ~ 5V  ──▶ ┌───┐ ──▶ 0 ~ 1023 (10비트)
               │ADC│
               └───┘
```

### Arduino ADC

```cpp
const int SENSOR_PIN = A0;

void setup() {
    Serial.begin(9600);
    // analogReference(DEFAULT);  // 5V 기준
    // analogReference(INTERNAL); // 1.1V 기준 (더 정밀)
}

void loop() {
    int rawValue = analogRead(SENSOR_PIN);  // 0-1023

    // 전압으로 변환
    float voltage = rawValue * (5.0 / 1023.0);

    Serial.print("Raw: ");
    Serial.print(rawValue);
    Serial.print(" Voltage: ");
    Serial.println(voltage);

    delay(100);
}
```

### 온도 센서 (TMP36)

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

### 조도 센서 (LDR)

```cpp
const int LDR_PIN = A0;
const int LED_PIN = 9;

void setup() {
    pinMode(LED_PIN, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    int lightLevel = analogRead(LDR_PIN);

    // 밝으면 LED 어둡게, 어두우면 LED 밝게
    int brightness = map(lightLevel, 0, 1023, 255, 0);
    analogWrite(LED_PIN, brightness);

    Serial.print("Light: ");
    Serial.print(lightLevel);
    Serial.print(" LED: ");
    Serial.println(brightness);

    delay(100);
}
```

### 다중 ADC 채널 읽기

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

### ADC 노이즈 필터링

```cpp
const int SENSOR_PIN = A0;
const int NUM_SAMPLES = 10;

// 이동 평균 필터
int readFiltered() {
    long sum = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        sum += analogRead(SENSOR_PIN);
        delay(1);
    }
    return sum / NUM_SAMPLES;
}

// 중앙값 필터
int readMedian() {
    int samples[NUM_SAMPLES];

    for (int i = 0; i < NUM_SAMPLES; i++) {
        samples[i] = analogRead(SENSOR_PIN);
        delay(1);
    }

    // 정렬
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

## 실전 프로젝트

### 온습도 데이터 로거

I2C, SPI, ADC를 종합 활용하는 프로젝트입니다.

```cpp
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <DHT.h>

// 핀 설정
const int DHT_PIN = 2;
const int SD_CS_PIN = 4;
const int LDR_PIN = A0;

// DHT 센서
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

// 로깅 간격 (밀리초)
const unsigned long LOG_INTERVAL = 5000;
unsigned long lastLogTime = 0;

void setup() {
    Serial.begin(9600);

    // DHT 초기화
    dht.begin();

    // SD 카드 초기화
    if (!SD.begin(SD_CS_PIN)) {
        Serial.println("SD card failed!");
        while (1);
    }
    Serial.println("SD card initialized.");

    // CSV 헤더 작성
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

        // 센서 읽기
        float temp = dht.readTemperature();
        float humidity = dht.readHumidity();
        int light = analogRead(LDR_PIN);

        // 유효성 검사
        if (isnan(temp) || isnan(humidity)) {
            Serial.println("DHT read failed!");
            return;
        }

        // 시리얼 출력
        Serial.print("Temp: ");
        Serial.print(temp);
        Serial.print("C, Humidity: ");
        Serial.print(humidity);
        Serial.print("%, Light: ");
        Serial.println(light);

        // SD 카드에 기록
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

### 회로 연결도

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

## 연습 문제

### 문제 1: PWM LED 제어

버튼을 누를 때마다 LED 밝기가 25%씩 증가하는 프로그램을 작성하세요. 100%에서 다시 0%로 돌아갑니다.

<details>
<summary>정답 보기</summary>

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
        delay(50);  // 디바운싱
    }

    lastButtonState = buttonState;
}
```

</details>

### 문제 2: I2C 스캐너

연결된 모든 I2C 장치의 주소를 찾는 스캐너를 작성하세요.

<details>
<summary>정답 보기</summary>

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

## 참고 자료

- [Arduino Reference](https://www.arduino.cc/reference/en/)
- [AVR Libc Reference](https://www.nongnu.org/avr-libc/user-manual/)
- [I2C Protocol](https://i2c.info/)
- [SPI Protocol](https://en.wikipedia.org/wiki/Serial_Peripheral_Interface)

---

**이전**: [디버깅과 메모리 분석](./18_Debugging_Memory_Analysis.md) | **다음**: [C 언어 포인터 심화](./20_Advanced_Pointers.md)
