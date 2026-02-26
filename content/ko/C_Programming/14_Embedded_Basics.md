# 임베디드 프로그래밍 기초

**이전**: [프로젝트 12: 멀티스레드 프로그래밍](./13_Project_Multithreading.md) | **다음**: [비트 연산 심화](./15_Bit_Operations.md)

임베디드 시스템의 개념을 이해하고 Arduino 개발 환경을 설정합니다.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 임베디드 시스템(Embedded System)이 무엇인지, 일반 컴퓨터와 어떻게 다른지 설명할 수 있다
2. 마이크로컨트롤러(Microcontroller)의 내부 구조(CPU, Flash, SRAM, 주변장치)를 설명할 수 있다
3. 주요 Arduino 보드를 비교하고 프로젝트에 적합한 보드를 선택할 수 있다
4. Arduino 개발 환경(IDE, Wokwi 시뮬레이터, 또는 PlatformIO)을 설정할 수 있다
5. `setup()`과 `loop()`를 사용하여 Arduino 프로그램을 작성, 컴파일, 업로드할 수 있다
6. 디지털 출력 핀을 제어하여 다양한 깜빡임 패턴으로 LED를 구동할 수 있다
7. 시리얼 모니터(Serial Monitor)를 사용하여 디버깅하고 변수 값을 확인할 수 있다

---

매일 사용하는 모든 가전제품 -- 전자레인지부터 자동차 제동 시스템까지 -- 은 임베디드 프로세서로 작동합니다. 임베디드 프로그래밍을 배우면 추상적인 C 코드의 세계와 물리적인 하드웨어를 연결하여, 무언가를 깜빡이게 하고 움직이게 하고 통신하게 만드는 능력을 갖출 수 있습니다. 이 레슨에서는 첫 임베디드 환경을 구축하고, 작성한 코드가 실제(또는 시뮬레이션된) 하드웨어를 제어하는 것을 확인합니다.

## 사전 지식
- C 언어 기본 문법
- 함수와 변수

---

## 1. 임베디드 시스템이란?

### 정의

**임베디드 시스템(Embedded System)**은 특정 기능을 수행하도록 설계된 컴퓨터 시스템입니다.

```
일반 컴퓨터:
┌─────────────────────────────────────┐
│  다양한 프로그램 실행 가능          │
│  웹 브라우저, 게임, 문서 작성 등    │
│  사용자가 자유롭게 활용             │
└─────────────────────────────────────┘

임베디드 시스템:
┌─────────────────────────────────────┐
│  특정 목적만 수행                   │
│  세탁기, 전자레인지, 자동차 ECU 등  │
│  전용 하드웨어 + 전용 소프트웨어    │
└─────────────────────────────────────┘
```

### 우리 주변의 임베디드 시스템

| 분야 | 예시 |
|------|------|
| 가전제품 | 세탁기, 냉장고, 에어컨, 전자레인지 |
| 자동차 | ECU, ABS, 에어백, 내비게이션 |
| 의료기기 | 혈압계, 체온계, MRI, 인슐린 펌프 |
| 통신 | 공유기, 스마트폰, 셋톱박스 |
| 산업 | 공장 자동화, 로봇, PLC |
| IoT | 스마트홈, 웨어러블, 센서 |

### 임베디드 시스템의 특징

```
1. 제한된 자원
   - 적은 메모리 (KB ~ MB)
   - 느린 CPU (MHz 단위)
   - 제한된 저장공간
   - 낮은 전력 소모

2. 실시간성
   - 정해진 시간 내 응답 필요
   - 예: 에어백은 충돌 감지 후 수십 ms 내 동작

3. 신뢰성
   - 24시간 365일 안정적 동작
   - 오류 시 치명적 결과 가능

4. 전용 하드웨어
   - 특정 목적에 최적화된 설계
```

---

## 2. 마이크로컨트롤러 (MCU)

### MCU vs MPU

```
MPU (Microprocessor Unit):
┌─────────────────────────────────────┐
│ CPU 코어만 포함                     │
│ 외부에 RAM, ROM, I/O 필요           │
│ 예: Intel Core, AMD Ryzen           │
│ 고성능, 범용 컴퓨팅                 │
└─────────────────────────────────────┘

MCU (Microcontroller Unit):
┌─────────────────────────────────────┐
│ CPU + RAM + ROM + I/O 통합          │
│ 원칩 솔루션 (One Chip Solution)     │
│ 예: ATmega328, STM32, ESP32         │
│ 저전력, 특정 목적                   │
└─────────────────────────────────────┘
```

### MCU 내부 구조

```
┌─────────────────────────────────────────────────┐
│                    MCU                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │   CPU   │  │  Flash  │  │  SRAM   │         │
│  │  코어   │  │ (프로그 │  │ (변수,  │         │
│  │         │  │   램)   │  │  스택)  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  GPIO   │  │  Timer  │  │  UART   │         │
│  │ (디지털 │  │ (타이머 │  │ (시리얼 │         │
│  │  입출력)│  │  /PWM)  │  │  통신)  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │   ADC   │  │   I2C   │  │   SPI   │         │
│  │ (아날로 │  │ (버스   │  │ (고속   │         │
│  │ 그입력) │  │  통신)  │  │  통신)  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
└─────────────────────────────────────────────────┘
```

### 주요 메모리 종류

| 메모리 | 특징 | 용도 |
|--------|------|------|
| **Flash** | 비휘발성, 읽기 빠름, 쓰기 느림 | 프로그램 코드 저장 |
| **SRAM** | 휘발성, 읽기/쓰기 빠름 | 변수, 스택, 힙 |
| **EEPROM** | 비휘발성, 바이트 단위 쓰기 | 설정값 저장 |

---

## 3. Arduino 소개

### Arduino란?

Arduino는 **오픈소스 하드웨어 플랫폼(open-source hardware platform)**으로, 임베디드 개발을 쉽게 시작할 수 있도록 설계되었습니다.

```
Arduino의 구성:
┌─────────────────────────────────────┐
│  1. 하드웨어 (보드)                 │
│     - ATmega328P MCU               │
│     - USB 연결                     │
│     - 전원 회로                    │
│     - 핀 헤더                      │
├─────────────────────────────────────┤
│  2. 소프트웨어 (IDE)               │
│     - 코드 편집기                  │
│     - 컴파일러                     │
│     - 업로드 도구                  │
├─────────────────────────────────────┤
│  3. 라이브러리                     │
│     - 센서, 모터, 디스플레이 등    │
│     - 풍부한 예제                  │
└─────────────────────────────────────┘
```

### 주요 Arduino 보드

| 보드 | MCU | Flash | SRAM | 핀 | 특징 |
|------|-----|-------|------|-----|------|
| **Uno** | ATmega328P | 32KB | 2KB | 14+6 | 가장 기본, 입문용 |
| **Nano** | ATmega328P | 32KB | 2KB | 14+8 | 소형, 브레드보드용 |
| **Mega** | ATmega2560 | 256KB | 8KB | 54+16 | 많은 핀, 대형 프로젝트 |
| **Leonardo** | ATmega32U4 | 32KB | 2.5KB | 20+12 | USB HID 지원 |

### Arduino Uno 핀 배치

```
                    ┌─────────────────────┐
                    │     USB 포트        │
                    └─────────────────────┘
    ┌───────────────────────────────────────────┐
    │  AREF  GND  13  12  11  10  9  8         │
    │  [ ]  [ ]  [ ] [ ] [ ] [ ] [ ] [ ]       │ ← 디지털 핀
    │                                          │
    │    ┌─────┐                               │
    │    │     │  ATmega328P                   │
    │    │     │                               │
    │    └─────┘                               │
    │                                          │
    │  [ ]  [ ]  [ ] [ ] [ ] [ ] [ ] [ ]       │
    │  RESET 3.3V 5V GND GND Vin               │ ← 전원
    │                                          │
    │  [ ]  [ ]  [ ] [ ] [ ] [ ]               │
    │  A0   A1   A2  A3  A4  A5                │ ← 아날로그 핀
    └───────────────────────────────────────────┘

디지털 핀 (0~13):
- 0, 1: Serial (TX, RX) - 시리얼 통신
- 3, 5, 6, 9, 10, 11: PWM 가능 (~표시)
- 13: 내장 LED 연결

아날로그 핀 (A0~A5):
- 아날로그 입력 (ADC)
- 디지털 핀으로도 사용 가능
```

---

## 4. 개발 환경 설정

### 방법 1: Arduino IDE 설치 (실제 하드웨어용)

#### Windows / macOS

1. https://www.arduino.cc/en/software 접속
2. 운영체제에 맞는 버전 다운로드
3. 설치 프로그램 실행

#### macOS (Homebrew)

```bash
brew install --cask arduino-ide
```

#### Linux (Ubuntu/Debian)

```bash
# 방법 1: apt
sudo apt update
sudo apt install arduino

# 방법 2: Flatpak
flatpak install flathub cc.arduino.IDE2
```

### 방법 2: Wokwi 시뮬레이터 (하드웨어 없이 학습)

**Wokwi**는 브라우저에서 Arduino를 시뮬레이션할 수 있는 무료 도구입니다.

1. https://wokwi.com 접속
2. "Start Creating" 클릭
3. "Arduino Uno" 선택
4. 바로 코딩 시작!

```
Wokwi 장점:
- 무료, 설치 불필요
- 다양한 부품 시뮬레이션 (LED, 버튼, 센서 등)
- 회로도 시각화
- 코드 공유 가능
- 실시간 디버깅
```

### 방법 3: VS Code + PlatformIO (고급)

1. VS Code 설치
2. PlatformIO 확장 설치
3. 새 프로젝트 생성시 Arduino Uno 선택

```bash
# PlatformIO CLI 설치 (선택사항)
pip install platformio
```

---

## 5. 첫 프로그램: Blink

### Arduino 프로그램 구조

```cpp
// Arduino 프로그램의 기본 구조

// 전역 변수, 상수 선언
const int LED_PIN = 13;

// setup(): 프로그램 시작시 한 번 실행
void setup() {
    // 초기화 코드
    pinMode(LED_PIN, OUTPUT);
}

// loop(): setup() 후 무한 반복 실행
void loop() {
    // 반복 실행할 코드
    digitalWrite(LED_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_PIN, LOW);
    delay(1000);
}
```

### 일반 C와 비교

```c
// 일반 C 프로그램
int main(void) {
    // 초기화
    init_hardware();

    // 무한 루프
    while (1) {
        // 반복 실행
        do_something();
    }

    return 0;  // 실제로는 도달 안 함
}
```

```cpp
// Arduino 프로그램 (동일한 구조)
void setup() {
    // 초기화 (main 시작 부분)
}

void loop() {
    // while(1) 내부와 동일
}

// Arduino 프레임워크가 main()을 제공:
// int main() {
//     setup();
//     while(1) loop();
// }
```

### Blink 예제 상세 설명

```cpp
// blink.ino - LED 깜빡이기

// LED가 연결된 핀 번호 (Arduino Uno의 내장 LED)
const int LED_PIN = 13;

void setup() {
    // 핀 모드 설정
    // OUTPUT: 출력 모드 (전압을 내보냄)
    // INPUT: 입력 모드 (전압을 읽음)
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // LED 켜기
    // HIGH = 5V (또는 3.3V) 출력
    digitalWrite(LED_PIN, HIGH);

    // 1000밀리초(1초) 대기
    delay(1000);

    // LED 끄기
    // LOW = 0V (GND) 출력
    digitalWrite(LED_PIN, LOW);

    // 1초 대기
    delay(1000);

    // loop()이 끝나면 다시 처음부터 실행
}
```

### Wokwi에서 실행하기

1. https://wokwi.com 접속
2. "New Project" → "Arduino Uno" 선택
3. 코드 입력:

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

4. 녹색 "Start Simulation" 버튼 클릭
5. 보드의 LED가 깜빡이는지 확인

---

## 6. 주요 Arduino 함수

### 디지털 I/O

```cpp
// 핀 모드 설정
pinMode(pin, mode);
// mode: INPUT, OUTPUT, INPUT_PULLUP

// 디지털 출력
digitalWrite(pin, value);
// value: HIGH (5V), LOW (0V)

// 디지털 입력
int value = digitalRead(pin);
// 반환: HIGH 또는 LOW
```

### 시간 관련

```cpp
// 밀리초 대기
delay(ms);

// 마이크로초 대기
delayMicroseconds(us);

// 프로그램 시작 후 경과 시간 (밀리초)
unsigned long time = millis();

// 프로그램 시작 후 경과 시간 (마이크로초)
unsigned long time = micros();
```

### 시리얼 통신

```cpp
// 시리얼 초기화 (보통 9600 또는 115200)
Serial.begin(baudrate);

// 데이터 출력
Serial.print("Hello");      // 줄바꿈 없이
Serial.println("World");    // 줄바꿈 포함
Serial.print(123);          // 숫자 출력

// 데이터 입력
if (Serial.available() > 0) {
    char c = Serial.read();
}
```

---

## 7. 실습 프로젝트: 다양한 Blink 패턴

### 프로젝트 1: 속도 조절 Blink

```cpp
// 점점 빨라지는 LED

const int LED_PIN = 13;
int delayTime = 1000;  // 시작 딜레이

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);
    delay(delayTime);
    digitalWrite(LED_PIN, LOW);
    delay(delayTime);

    // 딜레이 감소 (최소 50ms)
    delayTime -= 50;
    if (delayTime < 50) {
        delayTime = 1000;  // 리셋
    }
}
```

### 프로젝트 2: SOS 신호

```cpp
// 모스 부호 SOS (... --- ...)

const int LED_PIN = 13;
const int DOT = 200;    // 점 길이
const int DASH = 600;   // 대시 길이
const int GAP = 200;    // 신호 사이 간격
const int LETTER_GAP = 600;  // 글자 사이 간격

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
    delay(LETTER_GAP * 3);  // 단어 사이 긴 간격
}
```

### 프로젝트 3: millis() 사용 (비동기 Blink)

`delay()`는 프로그램을 멈추지만, `millis()`를 사용하면 다른 작업도 할 수 있습니다.

```cpp
// delay() 없이 LED 깜빡이기

const int LED_PIN = 13;
unsigned long previousMillis = 0;
const long interval = 1000;  // 1초
int ledState = LOW;

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    unsigned long currentMillis = millis();

    // interval 시간이 지났는지 확인
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;

        // LED 상태 토글
        ledState = (ledState == LOW) ? HIGH : LOW;
        digitalWrite(LED_PIN, ledState);
    }

    // 여기서 다른 작업 가능!
    // 예: 센서 읽기, 버튼 확인 등
}
```

### 프로젝트 4: 여러 LED 제어

Wokwi에서 외부 LED를 연결하여 테스트할 수 있습니다.

```cpp
// 3개 LED 순차 점등

const int LED1 = 11;
const int LED2 = 12;
const int LED3 = 13;

void setup() {
    pinMode(LED1, OUTPUT);
    pinMode(LED2, OUTPUT);
    pinMode(LED3, OUTPUT);
}

void loop() {
    // LED1만 켜기
    digitalWrite(LED1, HIGH);
    digitalWrite(LED2, LOW);
    digitalWrite(LED3, LOW);
    delay(300);

    // LED2만 켜기
    digitalWrite(LED1, LOW);
    digitalWrite(LED2, HIGH);
    digitalWrite(LED3, LOW);
    delay(300);

    // LED3만 켜기
    digitalWrite(LED1, LOW);
    digitalWrite(LED2, LOW);
    digitalWrite(LED3, HIGH);
    delay(300);
}
```

---

## 8. 시리얼 모니터로 디버깅

### 기본 시리얼 출력

```cpp
void setup() {
    Serial.begin(9600);  // 시리얼 통신 시작
    Serial.println("Arduino 시작!");
}

void loop() {
    static int count = 0;
    count++;

    Serial.print("카운트: ");
    Serial.println(count);

    delay(1000);
}
```

### Wokwi에서 시리얼 모니터 사용

1. 시뮬레이션 시작
2. 화면 오른쪽의 "Serial Monitor" 탭 클릭
3. 출력 확인

### 변수 값 모니터링

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
    Serial.print("LED ON - 횟수: ");
    Serial.println(blinkCount);
    delay(500);

    digitalWrite(LED_PIN, LOW);
    Serial.println("LED OFF");
    delay(500);
}
```

---

## 연습 문제

### 연습 1: 심장 박동 LED
심장 박동처럼 LED가 두 번 빠르게 깜빡이고, 잠시 쉬는 패턴을 만드세요.

### 연습 2: 카운트다운
시리얼 모니터에 10부터 1까지 카운트다운을 출력하고, 0이 되면 LED를 3번 깜빡이세요.

### 연습 3: 랜덤 Blink
`random()` 함수를 사용하여 불규칙한 간격으로 LED를 깜빡이게 하세요.

```cpp
// 힌트
int randomDelay = random(100, 1000);  // 100~999 사이 랜덤 값
```

### 연습 4: 2진수 카운터
4개의 LED를 사용하여 0~15까지 2진수로 표시하세요.
- 0 = 0000 (모든 LED 꺼짐)
- 5 = 0101 (LED2, LED4 켜짐)
- 15 = 1111 (모든 LED 켜짐)

---

## 핵심 개념 정리

| 용어 | 설명 |
|------|------|
| 임베디드 시스템 | 특정 기능 수행을 위한 전용 컴퓨터 시스템 |
| MCU | CPU + 메모리 + 주변장치가 통합된 칩 |
| GPIO | 범용 디지털 입출력(General Purpose Input/Output) 핀 |
| Flash | 프로그램 코드 저장용 비휘발성 메모리 |
| SRAM | 변수 저장용 휘발성 메모리 |
| setup() | 초기화 코드 (1회 실행) |
| loop() | 반복 실행 코드 (무한 루프) |

---

## Wokwi 프로젝트 링크 예시

기본 Blink 프로젝트를 Wokwi에서 바로 실행해볼 수 있습니다:
- https://wokwi.com/projects/new/arduino-uno

---

## 다음 단계

Arduino 기초를 익혔다면 다음 문서로 넘어가세요:
- [15. 비트 연산 심화](15_Bit_Operations.md) - 임베디드의 핵심 기술

---

**이전**: [프로젝트 12: 멀티스레드 프로그래밍](./13_Project_Multithreading.md) | **다음**: [비트 연산 심화](./15_Bit_Operations.md)
