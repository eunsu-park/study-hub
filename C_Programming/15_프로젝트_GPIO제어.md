# 프로젝트 15: GPIO 제어

GPIO(General Purpose Input/Output)로 LED와 버튼을 제어합니다.

## 학습 목표
- GPIO 입출력 개념 이해
- LED 제어 (디지털 출력)
- 버튼 읽기 (디지털 입력)
- 풀업/풀다운 저항 이해
- 디바운싱 기법 습득

## 사전 지식
- Arduino 기본 구조 (setup, loop)
- 비트 연산 기초

---

## 1. GPIO 개념

### GPIO란?

**GPIO (General Purpose Input/Output)**는 MCU의 범용 디지털 핀으로, 프로그램에서 자유롭게 입력 또는 출력으로 설정할 수 있습니다.

```
GPIO 핀의 두 가지 모드:

출력 모드 (OUTPUT)
┌─────────────────────────────────────┐
│  MCU가 핀에 전압을 출력              │
│  - HIGH: 5V (또는 3.3V)             │
│  - LOW: 0V (GND)                    │
│  예: LED 켜기/끄기, 릴레이 제어      │
└─────────────────────────────────────┘

입력 모드 (INPUT)
┌─────────────────────────────────────┐
│  MCU가 핀의 전압을 읽음              │
│  - HIGH: 임계값 이상 (약 3V)        │
│  - LOW: 임계값 미만 (약 1.5V)       │
│  예: 버튼 상태 읽기, 센서 읽기       │
└─────────────────────────────────────┘
```

### Arduino Uno의 GPIO 핀

```
Arduino Uno 핀 배치:
┌────────────────────────────────────────────────┐
│                                                │
│  디지털 핀: 0 ~ 13 (총 14개)                   │
│  - 0, 1: Serial 통신용 (TX, RX)               │
│  - 2, 3: 외부 인터럽트 가능                    │
│  - 3, 5, 6, 9, 10, 11: PWM 출력 가능 (~)      │
│  - 13: 내장 LED 연결                          │
│                                                │
│  아날로그 핀: A0 ~ A5 (총 6개)                 │
│  - 아날로그 입력 가능 (ADC)                    │
│  - 디지털 입출력으로도 사용 가능               │
│                                                │
└────────────────────────────────────────────────┘
```

---

## 2. 디지털 출력: LED 제어

### 기본 회로 연결

```
LED 연결 방법:

방법 1: 핀 → 저항 → LED → GND (싱킹)
┌──────┐
│ 핀 9 │───[330Ω]───[LED]───GND
└──────┘
* HIGH 출력 시 LED 켜짐

방법 2: VCC → LED → 저항 → 핀 (소싱)
          VCC───[LED]───[330Ω]───│ 핀 9 │
                                 └──────┘
* LOW 출력 시 LED 켜짐

저항값 계산:
R = (V_supply - V_led) / I_led
  = (5V - 2V) / 10mA
  = 300Ω → 330Ω 사용 (표준 저항값)
```

### 기본 LED 켜기/끄기

```cpp
// led_basic.ino
// 가장 기본적인 LED 제어

const int LED_PIN = 9;  // LED 연결 핀

void setup() {
    // 핀 모드를 출력으로 설정
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // LED 켜기 (HIGH = 5V 출력)
    digitalWrite(LED_PIN, HIGH);
    delay(1000);  // 1초 대기

    // LED 끄기 (LOW = 0V 출력)
    digitalWrite(LED_PIN, LOW);
    delay(1000);
}
```

### Wokwi에서 회로 구성

Wokwi (https://wokwi.com)에서 다음과 같이 회로를 구성합니다:

1. Arduino Uno 추가
2. LED 추가 (부품 목록에서 검색)
3. 저항 추가 (330Ω)
4. 연결:
   - 핀 9 → 저항 → LED 양극(+, 긴 다리)
   - LED 음극(-, 짧은 다리) → GND

### 여러 LED 순차 점등

```cpp
// led_sequence.ino
// 여러 LED 순차적으로 켜기

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    // 모든 LED 핀을 출력으로 설정
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
}

void loop() {
    // 순차적으로 켜기
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], HIGH);
        delay(200);
    }

    // 순차적으로 끄기
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], LOW);
        delay(200);
    }
}
```

### LED 패턴 만들기

```cpp
// led_patterns.ino
// 다양한 LED 패턴

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
}

// 모든 LED 상태 설정
void setLEDs(int pattern) {
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], (pattern >> i) & 1);
    }
}

// 왕복 패턴 (나이트 라이더)
void knightRider() {
    // 왼쪽으로
    for (int i = 0; i < NUM_LEDS; i++) {
        setLEDs(1 << i);
        delay(100);
    }
    // 오른쪽으로
    for (int i = NUM_LEDS - 2; i > 0; i--) {
        setLEDs(1 << i);
        delay(100);
    }
}

// 깜빡임 패턴
void blinkAll(int times, int delayMs) {
    for (int i = 0; i < times; i++) {
        setLEDs(0x0F);  // 모두 켜기
        delay(delayMs);
        setLEDs(0x00);  // 모두 끄기
        delay(delayMs);
    }
}

// 채우기 패턴
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

## 3. 디지털 입력: 버튼 읽기

### 버튼 회로 연결

```
버튼 연결 방법:

방법 1: 외부 풀다운 저항 사용
        VCC (5V)
          │
        [버튼]
          │
    ┌─────┼─────┐
    │           │
  [핀]       [10kΩ]
                │
              GND

- 버튼 안 누름: 핀 = LOW (저항이 GND로 당김)
- 버튼 누름: 핀 = HIGH (VCC 연결)

방법 2: 외부 풀업 저항 사용
        VCC (5V)
          │
       [10kΩ]
          │
    ┌─────┼─────┐
    │           │
  [핀]       [버튼]
                │
              GND

- 버튼 안 누름: 핀 = HIGH (저항이 VCC로 당김)
- 버튼 누름: 핀 = LOW (GND 연결)

방법 3: 내부 풀업 저항 사용 (권장)
  [핀]───[버튼]───GND

- pinMode(pin, INPUT_PULLUP) 사용
- 외부 저항 불필요
- 버튼 안 누름: HIGH
- 버튼 누름: LOW
```

### 기본 버튼 읽기

```cpp
// button_basic.ino
// 버튼으로 LED 제어

const int BUTTON_PIN = 2;
const int LED_PIN = 13;

void setup() {
    // 내부 풀업 저항 사용
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);

    Serial.begin(9600);
}

void loop() {
    // 버튼 상태 읽기
    int buttonState = digitalRead(BUTTON_PIN);

    // 버튼 누르면 LED 켜기 (LOW = 눌림)
    if (buttonState == LOW) {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("Button pressed!");
    } else {
        digitalWrite(LED_PIN, LOW);
    }

    delay(10);  // 짧은 대기
}
```

### 버튼 토글

```cpp
// button_toggle.ino
// 버튼 누를 때마다 LED 상태 토글

const int BUTTON_PIN = 2;
const int LED_PIN = 13;

bool ledState = false;
bool lastButtonState = HIGH;  // 풀업이므로 HIGH가 기본

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    bool currentButtonState = digitalRead(BUTTON_PIN);

    // 버튼이 눌렸을 때 (HIGH → LOW 변화)
    if (lastButtonState == HIGH && currentButtonState == LOW) {
        ledState = !ledState;  // LED 상태 토글
        digitalWrite(LED_PIN, ledState);
    }

    lastButtonState = currentButtonState;
    delay(10);
}
```

---

## 4. 디바운싱 (Debouncing)

### 바운싱 문제

물리적 버튼을 누르면 접점이 여러 번 튀어 노이즈가 발생합니다.

```
실제 버튼 신호 (바운싱):

    버튼 누름                    버튼 뗌
        ↓                         ↓
HIGH ─────┐   ┌─┐ ┌─┐        ┌─┐ ┌────────
          │   │ │ │ │        │ │ │
LOW       └───┘ └─┘ └────────┘ └─┘

         ↑↑↑ 바운싱 노이즈 ↑↑↑

시간: 약 1~50ms 동안 발생
```

### 소프트웨어 디바운싱

```cpp
// debounce_software.ino
// 소프트웨어 디바운싱

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

    // 상태 변화 감지
    if (reading != lastButtonState) {
        lastDebounceTime = millis();  // 타이머 리셋
    }

    // 일정 시간 동안 상태가 유지되면 확정
    if ((millis() - lastDebounceTime) > debounceDelay) {
        static bool buttonState = HIGH;

        if (reading != buttonState) {
            buttonState = reading;

            // 버튼 눌림 확정 (HIGH → LOW)
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

### Bounce 클래스 만들기

```cpp
// button_class.ino
// 재사용 가능한 버튼 클래스

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

    // 업데이트 및 눌림 감지
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
                    return true;  // 버튼 눌림!
                }
            }
        }

        lastState = reading;
        return false;
    }

    // 현재 상태 (누르고 있는지)
    bool isPressed() {
        return currentState == LOW;
    }
};

// 사용 예
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

## 5. 실습 프로젝트: LED 제어기

### 프로젝트 개요

2개의 버튼으로 4개의 LED를 제어합니다:
- 버튼 1: 패턴 변경
- 버튼 2: 속도 변경

```cpp
// led_controller.ino
// 버튼으로 LED 패턴과 속도 제어

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;
const int BTN_PATTERN = 2;  // 패턴 변경 버튼
const int BTN_SPEED = 3;    // 속도 변경 버튼

// 상태 변수
int currentPattern = 0;
const int NUM_PATTERNS = 4;
int speedLevel = 1;  // 0=빠름, 1=보통, 2=느림
const int SPEEDS[] = {50, 150, 300};

// 디바운싱 변수
bool lastBtnPattern = HIGH;
bool lastBtnSpeed = HIGH;
unsigned long lastDebouncePattern = 0;
unsigned long lastDebounceSpeed = 0;
const unsigned long debounceDelay = 50;

// 패턴 타이밍
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

// 패턴 0: 순차 점등
void pattern0() {
    setLEDs(1 << (patternStep % NUM_LEDS));
}

// 패턴 1: 나이트 라이더
void pattern1() {
    int pos = patternStep % (NUM_LEDS * 2 - 2);
    if (pos >= NUM_LEDS) {
        pos = NUM_LEDS * 2 - 2 - pos;
    }
    setLEDs(1 << pos);
}

// 패턴 2: 채우기
void pattern2() {
    int step = patternStep % (NUM_LEDS * 2);
    if (step < NUM_LEDS) {
        setLEDs((1 << (step + 1)) - 1);
    } else {
        setLEDs((1 << (NUM_LEDS * 2 - step)) - 1);
    }
}

// 패턴 3: 번갈아 깜빡임
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
    // 버튼 1: 패턴 변경
    if (checkButton(BTN_PATTERN, lastBtnPattern, lastDebouncePattern)) {
        currentPattern = (currentPattern + 1) % NUM_PATTERNS;
        patternStep = 0;
        Serial.print("Pattern: ");
        Serial.println(currentPattern);
    }

    // 버튼 2: 속도 변경
    if (checkButton(BTN_SPEED, lastBtnSpeed, lastDebounceSpeed)) {
        speedLevel = (speedLevel + 1) % 3;
        Serial.print("Speed: ");
        Serial.println(SPEEDS[speedLevel]);
    }

    // 패턴 업데이트
    if (millis() - lastPatternUpdate >= SPEEDS[speedLevel]) {
        lastPatternUpdate = millis();
        updatePattern();
    }
}
```

### Wokwi 회로 구성

```
부품 목록:
- Arduino Uno x1
- LED x4 (빨강, 노랑, 초록, 파랑)
- 저항 330Ω x4
- 버튼 x2

연결:
- LED: 핀 9, 10, 11, 12 → 저항 → LED → GND
- 버튼 1: 핀 2 ↔ GND
- 버튼 2: 핀 3 ↔ GND
```

---

## 6. 직접 레지스터 제어

Arduino 함수 대신 레지스터를 직접 제어하면 훨씬 빠릅니다.

### 레지스터로 LED 제어

```cpp
// register_gpio.ino
// 레지스터로 직접 GPIO 제어

void setup() {
    // DDRB: 포트 B 방향 레지스터 (핀 8-13)
    // 비트 1~4를 출력으로 설정 (핀 9-12)
    DDRB |= 0b00011110;

    // DDRD: 포트 D 방향 레지스터 (핀 0-7)
    // 비트 2, 3을 입력으로 (핀 2, 3)
    DDRD &= ~0b00001100;

    // PORTD: 풀업 활성화
    PORTD |= 0b00001100;

    Serial.begin(9600);
}

void loop() {
    // PIND: 포트 D 입력 레지스터
    // 버튼 1 (핀 2) 확인
    if (!(PIND & 0b00000100)) {  // 비트 2
        // 모든 LED 켜기
        PORTB |= 0b00011110;
        Serial.println("All ON");
    }

    // 버튼 2 (핀 3) 확인
    if (!(PIND & 0b00001000)) {  // 비트 3
        // 모든 LED 끄기
        PORTB &= ~0b00011110;
        Serial.println("All OFF");
    }

    // LED 시프트 패턴
    static unsigned long lastUpdate = 0;
    static byte ledPattern = 0b00000010;  // 핀 9부터 시작

    if (millis() - lastUpdate > 200) {
        lastUpdate = millis();

        PORTB = (PORTB & ~0b00011110) | ledPattern;

        // 왼쪽으로 시프트
        ledPattern <<= 1;
        if (ledPattern > 0b00010000) {
            ledPattern = 0b00000010;
        }
    }
}
```

### 포트 매핑 참조

```
Arduino Uno 포트 매핑:

포트 B (PORTB, DDRB, PINB):
  비트 0: 핀 8
  비트 1: 핀 9
  비트 2: 핀 10
  비트 3: 핀 11
  비트 4: 핀 12
  비트 5: 핀 13 (내장 LED)

포트 D (PORTD, DDRD, PIND):
  비트 0: 핀 0 (RX)
  비트 1: 핀 1 (TX)
  비트 2: 핀 2
  비트 3: 핀 3
  비트 4: 핀 4
  비트 5: 핀 5
  비트 6: 핀 6
  비트 7: 핀 7

포트 C (PORTC, DDRC, PINC):
  비트 0~5: A0~A5
```

---

## 7. 실습 프로젝트: 반응 속도 게임

```cpp
// reaction_game.ino
// LED가 켜지면 버튼을 누르고 반응 시간 측정

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
            // 버튼 누르면 게임 시작
            if (buttonPressed) {
                state = READY;
                waitStart = millis();
                Serial.println("\nGet ready...");

                // 버튼 떼기 대기
                while (digitalRead(BUTTON_PIN) == LOW);
            }
            break;

        case READY:
            // 랜덤 시간 후 LED 켜기
            if (millis() - waitStart > random(2000, 5000)) {
                digitalWrite(LED_PIN, HIGH);
                ledOnTime = millis();
                state = PLAYING;
                Serial.println("GO!");
            }

            // 일찍 누르면 실패
            if (buttonPressed) {
                Serial.println("Too early! Try again.");
                state = WAITING;
            }
            break;

        case PLAYING:
            // 버튼 누르면 시간 측정
            if (buttonPressed) {
                reactionTime = millis() - ledOnTime;
                digitalWrite(LED_PIN, LOW);
                state = RESULT;
            }

            // 타임아웃 (3초)
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

            // 버튼 떼기 대기
            while (digitalRead(BUTTON_PIN) == LOW);
            delay(500);

            state = WAITING;
            break;
    }

    delay(10);
}
```

---

## 연습 문제

### 연습 1: 이진 카운터
4개의 LED로 0~15까지 카운트하세요. 버튼을 누를 때마다 1 증가.

### 연습 2: 신호등
빨강-노랑-초록 LED로 신호등을 구현하세요.
- 빨강: 3초
- 빨강+노랑: 1초
- 초록: 3초
- 노랑: 1초
- 반복

### 연습 3: 다중 버튼
3개의 버튼으로 각각 다른 LED를 제어하세요. 동시에 누르면 모두 깜빡임.

### 연습 4: 모스 부호 입력기
버튼 짧게 누르기 = 점 (.)
버튼 길게 누르기 = 대시 (-)
입력된 모스 부호를 시리얼 모니터에 출력

---

## 핵심 개념 정리

| 함수 | 설명 |
|------|------|
| `pinMode(pin, mode)` | 핀 모드 설정 (INPUT, OUTPUT, INPUT_PULLUP) |
| `digitalWrite(pin, val)` | 디지털 출력 (HIGH, LOW) |
| `digitalRead(pin)` | 디지털 입력 읽기 |

| 개념 | 설명 |
|------|------|
| 풀업 저항 | 입력 핀을 HIGH로 유지, 버튼 누르면 LOW |
| 풀다운 저항 | 입력 핀을 LOW로 유지, 버튼 누르면 HIGH |
| 디바운싱 | 버튼 노이즈 제거 기법 |
| 레지스터 | DDRx (방향), PORTx (출력), PINx (입력) |

---

## 다음 단계

GPIO 제어를 익혔다면 다음 문서로 넘어가세요:
- [16. 시리얼 통신](16_프로젝트_시리얼통신.md) - UART 통신과 디버깅
