# 비트 연산 심화

임베디드 프로그래밍의 핵심인 비트 단위 조작을 완벽하게 익힙니다.

## 학습 목표
- 비트 연산자 완벽 이해
- 비트 마스킹 기법 습득
- 레지스터 제어 개념 이해
- volatile 키워드 이해

## 사전 지식
- C 언어 기본 문법
- 2진수, 16진수 표현

---

## 1. 왜 비트 연산이 중요한가?

### 임베디드에서 비트 연산이 필수인 이유

```
1. 레지스터 제어
   - MCU의 모든 기능은 레지스터(특수 메모리)로 제어
   - 레지스터의 각 비트가 특정 기능 담당

2. 메모리 절약
   - 임베디드는 메모리가 제한적 (KB 단위)
   - 8개 플래그를 1바이트로 저장 가능

3. 통신 프로토콜
   - 데이터 패킷의 비트별 해석 필요

4. 성능 최적화
   - 비트 연산은 CPU에서 가장 빠른 연산
```

### 예시: LED 제어 레지스터

```
Arduino Uno의 포트 B 레지스터 (핀 8~13)
┌─────────────────────────────────────────────┐
│  PORTB = 0b00100000                        │
│                                             │
│  비트:  7    6    5    4    3    2    1    0│
│        [ ]  [ ]  [1]  [ ]  [ ]  [ ]  [ ]  [ ]│
│              │    │    │    │    │    │    │ │
│              │   핀13 핀12 핀11 핀10 핀9  핀8│
│              │    ↓                         │
│              │   LED ON (핀 13)             │
│              │                              │
│  비트 5 = 1 → 핀 13에 HIGH 출력            │
└─────────────────────────────────────────────┘
```

---

## 2. 비트 연산자 복습

### 기본 비트 연산자

```c
// bitwise_operators.c
#include <stdio.h>

void print_binary(unsigned char n) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
        if (i == 4) printf(" ");  // 가독성을 위한 공백
    }
    printf("\n");
}

int main(void) {
    unsigned char a = 0b11001010;  // 202
    unsigned char b = 0b10110011;  // 179

    printf("a        = "); print_binary(a);  // 1100 1010
    printf("b        = "); print_binary(b);  // 1011 0011
    printf("\n");

    // AND (&): 둘 다 1일 때만 1
    printf("a & b    = "); print_binary(a & b);   // 1000 0010

    // OR (|): 둘 중 하나라도 1이면 1
    printf("a | b    = "); print_binary(a | b);   // 1111 1011

    // XOR (^): 서로 다르면 1
    printf("a ^ b    = "); print_binary(a ^ b);   // 0111 1001

    // NOT (~): 비트 반전
    printf("~a       = "); print_binary(~a);      // 0011 0101

    // Left Shift (<<): 왼쪽으로 이동, 빈 자리는 0
    printf("a << 2   = "); print_binary(a << 2);  // 0010 1000

    // Right Shift (>>): 오른쪽으로 이동
    printf("a >> 2   = "); print_binary(a >> 2);  // 0011 0010

    return 0;
}
```

### 연산자 진리표

```
AND (&)          OR (|)           XOR (^)
A  B  A&B        A  B  A|B        A  B  A^B
0  0   0         0  0   0         0  0   0
0  1   0         0  1   1         0  1   1
1  0   0         1  0   1         1  0   1
1  1   1         1  1   1         1  1   0
```

---

## 3. 비트 마스킹 기법

### 3.1 특정 비트 읽기 (GET)

```c
// 특정 비트의 값 확인
// 방법: (value >> bit) & 1

unsigned char reg = 0b10110100;

// 비트 2의 값 읽기
int bit2 = (reg >> 2) & 1;  // 결과: 1

// 비트 3의 값 읽기
int bit3 = (reg >> 3) & 1;  // 결과: 0

// 매크로로 정의
#define GET_BIT(value, bit) (((value) >> (bit)) & 1)

// 사용 예
if (GET_BIT(reg, 5)) {
    printf("비트 5가 설정됨\n");
}
```

### 3.2 특정 비트 설정 (SET)

```c
// 특정 비트를 1로 설정
// 방법: value |= (1 << bit)

unsigned char reg = 0b10100000;

// 비트 3을 1로 설정
reg |= (1 << 3);  // 결과: 0b10101000

// 여러 비트 동시 설정
reg |= (1 << 1) | (1 << 4);  // 비트 1, 4 설정

// 매크로로 정의
#define SET_BIT(value, bit) ((value) |= (1 << (bit)))

// 사용 예
SET_BIT(reg, 6);  // 비트 6 설정
```

**동작 원리:**
```
reg       = 1010 0000
1 << 3    = 0000 1000
           ─────────── OR
결과      = 1010 1000
```

### 3.3 특정 비트 해제 (CLEAR)

```c
// 특정 비트를 0으로 해제
// 방법: value &= ~(1 << bit)

unsigned char reg = 0b11111111;

// 비트 5를 0으로 해제
reg &= ~(1 << 5);  // 결과: 0b11011111

// 매크로로 정의
#define CLEAR_BIT(value, bit) ((value) &= ~(1 << (bit)))

// 사용 예
CLEAR_BIT(reg, 2);  // 비트 2 해제
```

**동작 원리:**
```
reg       = 1111 1111
1 << 5    = 0010 0000
~(1 << 5) = 1101 1111
           ─────────── AND
결과      = 1101 1111
```

### 3.4 특정 비트 토글 (TOGGLE)

```c
// 특정 비트 반전 (0→1, 1→0)
// 방법: value ^= (1 << bit)

unsigned char reg = 0b10101010;

// 비트 4 토글
reg ^= (1 << 4);  // 결과: 0b10111010 (0→1)
reg ^= (1 << 4);  // 결과: 0b10101010 (1→0)

// 매크로로 정의
#define TOGGLE_BIT(value, bit) ((value) ^= (1 << (bit)))

// 사용 예: LED 토글
TOGGLE_BIT(PORTB, 5);  // 핀 13 LED 토글
```

### 3.5 비트 마스크 유틸리티 헤더

```c
// bit_utils.h
#ifndef BIT_UTILS_H
#define BIT_UTILS_H

// 비트 조작 매크로
#define BIT(n)                  (1 << (n))
#define SET_BIT(reg, bit)       ((reg) |= BIT(bit))
#define CLEAR_BIT(reg, bit)     ((reg) &= ~BIT(bit))
#define TOGGLE_BIT(reg, bit)    ((reg) ^= BIT(bit))
#define GET_BIT(reg, bit)       (((reg) >> (bit)) & 1)
#define CHECK_BIT(reg, bit)     ((reg) & BIT(bit))

// 여러 비트 조작
#define SET_BITS(reg, mask)     ((reg) |= (mask))
#define CLEAR_BITS(reg, mask)   ((reg) &= ~(mask))
#define TOGGLE_BITS(reg, mask)  ((reg) ^= (mask))

// 비트 필드 조작
#define GET_FIELD(reg, mask, shift)     (((reg) & (mask)) >> (shift))
#define SET_FIELD(reg, mask, shift, val) \
    ((reg) = ((reg) & ~(mask)) | (((val) << (shift)) & (mask)))

#endif
```

---

## 4. 플래그 관리

여러 상태를 하나의 변수로 관리합니다.

### 플래그 정의

```c
// flags.c
#include <stdio.h>
#include <stdbool.h>

// 각 비트에 의미 부여
#define FLAG_RUNNING    (1 << 0)  // 비트 0: 실행 중
#define FLAG_ERROR      (1 << 1)  // 비트 1: 에러 발생
#define FLAG_CONNECTED  (1 << 2)  // 비트 2: 연결됨
#define FLAG_READY      (1 << 3)  // 비트 3: 준비 완료
#define FLAG_BUSY       (1 << 4)  // 비트 4: 사용 중
#define FLAG_TIMEOUT    (1 << 5)  // 비트 5: 타임아웃

// 전역 상태 플래그
unsigned char system_flags = 0;

// 플래그 설정
void set_flag(unsigned char flag) {
    system_flags |= flag;
}

// 플래그 해제
void clear_flag(unsigned char flag) {
    system_flags &= ~flag;
}

// 플래그 확인
bool is_flag_set(unsigned char flag) {
    return (system_flags & flag) != 0;
}

// 플래그 토글
void toggle_flag(unsigned char flag) {
    system_flags ^= flag;
}

int main(void) {
    // 시스템 시작
    set_flag(FLAG_RUNNING);
    set_flag(FLAG_READY);

    printf("FLAGS: 0x%02X\n", system_flags);

    // 상태 확인
    if (is_flag_set(FLAG_RUNNING)) {
        printf("시스템 실행 중\n");
    }

    if (is_flag_set(FLAG_ERROR)) {
        printf("에러 발생!\n");
    } else {
        printf("정상 동작\n");
    }

    // 에러 발생
    set_flag(FLAG_ERROR);
    printf("에러 플래그 설정 후: 0x%02X\n", system_flags);

    // 에러 해결
    clear_flag(FLAG_ERROR);
    printf("에러 플래그 해제 후: 0x%02X\n", system_flags);

    return 0;
}
```

### Arduino에서 플래그 사용 예

```cpp
// arduino_flags.ino

// 상태 플래그
#define STATE_IDLE      0x00
#define STATE_RUNNING   0x01
#define STATE_PAUSED    0x02
#define STATE_ERROR     0x04

unsigned char deviceState = STATE_IDLE;

void setup() {
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    // 시리얼 명령 처리
    if (Serial.available()) {
        char cmd = Serial.read();

        switch (cmd) {
            case 'r':  // Run
                deviceState |= STATE_RUNNING;
                deviceState &= ~STATE_PAUSED;
                Serial.println("Running");
                break;

            case 'p':  // Pause
                deviceState |= STATE_PAUSED;
                Serial.println("Paused");
                break;

            case 's':  // Stop
                deviceState = STATE_IDLE;
                Serial.println("Stopped");
                break;

            case 'e':  // Error toggle
                deviceState ^= STATE_ERROR;
                Serial.println("Error toggled");
                break;
        }

        // 상태 출력
        Serial.print("State: 0x");
        Serial.println(deviceState, HEX);
    }

    // 상태에 따른 LED 제어
    if (deviceState & STATE_ERROR) {
        // 에러: 빠른 깜빡임
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        delay(100);
    } else if (deviceState & STATE_RUNNING) {
        if (!(deviceState & STATE_PAUSED)) {
            // 실행 중: 느린 깜빡임
            digitalWrite(LED_BUILTIN, HIGH);
            delay(500);
            digitalWrite(LED_BUILTIN, LOW);
            delay(500);
        }
    } else {
        // 대기: LED 꺼짐
        digitalWrite(LED_BUILTIN, LOW);
    }
}
```

---

## 5. 레지스터 개념

### MCU 레지스터란?

레지스터는 MCU 내부의 특수한 메모리 위치로, 하드웨어를 제어합니다.

```
Arduino Uno (ATmega328P) GPIO 관련 레지스터:

┌─────────────────────────────────────────────────────┐
│ DDRx (Data Direction Register)                      │
│ - 핀의 입/출력 방향 설정                            │
│ - 0 = 입력 (INPUT)                                  │
│ - 1 = 출력 (OUTPUT)                                 │
├─────────────────────────────────────────────────────┤
│ PORTx (Port Output Register)                        │
│ - 출력 모드: HIGH/LOW 출력                          │
│ - 입력 모드: 풀업 저항 활성화                       │
├─────────────────────────────────────────────────────┤
│ PINx (Port Input Register)                          │
│ - 핀의 현재 상태 읽기                               │
└─────────────────────────────────────────────────────┘

x = B (핀 8~13), C (아날로그 핀), D (핀 0~7)
```

### Arduino 함수 vs 직접 레지스터 제어

```cpp
// Arduino 라이브러리 사용 (편리하지만 느림)
pinMode(13, OUTPUT);
digitalWrite(13, HIGH);
digitalWrite(13, LOW);

// 직접 레지스터 제어 (빠름)
DDRB |= (1 << 5);   // 핀 13을 출력으로 설정
PORTB |= (1 << 5);  // 핀 13 HIGH
PORTB &= ~(1 << 5); // 핀 13 LOW
```

### 속도 비교

```cpp
// speed_compare.ino
// digitalWrite vs 직접 레지스터

void setup() {
    Serial.begin(9600);
    DDRB |= (1 << 5);  // 핀 13 출력 설정
}

void loop() {
    unsigned long start, end;

    // digitalWrite 속도 측정
    start = micros();
    for (long i = 0; i < 100000; i++) {
        digitalWrite(13, HIGH);
        digitalWrite(13, LOW);
    }
    end = micros();
    Serial.print("digitalWrite: ");
    Serial.print(end - start);
    Serial.println(" us");

    // 직접 레지스터 속도 측정
    start = micros();
    for (long i = 0; i < 100000; i++) {
        PORTB |= (1 << 5);
        PORTB &= ~(1 << 5);
    }
    end = micros();
    Serial.print("Direct register: ");
    Serial.print(end - start);
    Serial.println(" us");

    delay(3000);
}

// 결과 예시:
// digitalWrite: 625000 us
// Direct register: 12500 us
// → 직접 제어가 약 50배 빠름!
```

### 레지스터 직접 접근 예제

```cpp
// register_access.ino
// 레지스터로 LED 제어

void setup() {
    // DDRB: Data Direction Register B
    // 비트 5 = 1 → 핀 13 출력 모드
    DDRB |= (1 << DDB5);

    Serial.begin(9600);
    Serial.println("Register control demo");
}

void loop() {
    // PORTB 레지스터의 비트 5를 토글
    PORTB ^= (1 << PORTB5);

    // PINB 레지스터로 현재 상태 읽기
    if (PINB & (1 << PINB5)) {
        Serial.println("LED ON");
    } else {
        Serial.println("LED OFF");
    }

    delay(500);
}
```

---

## 6. volatile 키워드

### volatile이 필요한 이유

```c
// 문제 상황: 컴파일러 최적화

int flag = 0;

// 인터럽트 핸들러 (하드웨어에 의해 호출)
void interrupt_handler() {
    flag = 1;
}

int main() {
    while (flag == 0) {
        // 대기
    }
    // flag가 1이 되면 여기 실행
}
```

컴파일러는 `flag`가 루프 안에서 변경되지 않는다고 판단하여 최적화할 수 있습니다:

```c
// 컴파일러가 최적화한 코드 (문제!)
if (flag == 0) {
    while (1) { }  // 무한 루프로 변환
}
```

### volatile 사용

```c
// 해결: volatile 키워드 사용
volatile int flag = 0;

void interrupt_handler() {
    flag = 1;
}

int main() {
    while (flag == 0) {
        // 매번 메모리에서 flag 값을 읽음
    }
    // 정상 동작
}
```

### volatile의 의미

```
volatile = "변덕스러운"

컴파일러에게 알려주는 것:
1. 이 변수는 언제든 외부에서 변경될 수 있음
2. 최적화하지 말고 항상 메모리에서 읽어라
3. 레지스터에 캐싱하지 마라
```

### Arduino에서 volatile 사용

```cpp
// volatile_example.ino
// 인터럽트와 volatile

volatile bool buttonPressed = false;
const int BUTTON_PIN = 2;
const int LED_PIN = 13;

// 인터럽트 서비스 루틴 (ISR)
void buttonISR() {
    buttonPressed = true;
}

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);

    // 핀 2에 인터럽트 연결 (FALLING = HIGH→LOW)
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

    Serial.begin(9600);
}

void loop() {
    if (buttonPressed) {
        Serial.println("Button pressed!");
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));  // LED 토글

        buttonPressed = false;  // 플래그 리셋
        delay(200);  // 디바운스
    }
}
```

### 레지스터와 volatile

```c
// MCU 레지스터는 항상 volatile
// 하드웨어가 값을 변경할 수 있으므로

// 실제 Arduino 헤더 파일 정의 (avr/io.h)
#define PORTB (*(volatile uint8_t *)0x25)
#define DDRB  (*(volatile uint8_t *)0x24)
#define PINB  (*(volatile uint8_t *)0x23)

// 설명:
// 0x25 = PORTB 레지스터의 메모리 주소
// (volatile uint8_t *) = 해당 주소를 volatile 포인터로 캐스팅
// * = 해당 주소의 값에 접근
```

---

## 7. 실습: 비트 조작 유틸리티

### 비트 카운터

```c
// bit_counter.c
#include <stdio.h>

// 1인 비트 개수 세기 (popcount)
int count_ones(unsigned int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

// 더 빠른 방법: Brian Kernighan 알고리즘
int count_ones_fast(unsigned int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);  // 가장 오른쪽 1 비트 제거
        count++;
    }
    return count;
}

int main(void) {
    unsigned int test[] = {0, 1, 7, 255, 0xABCD};

    for (int i = 0; i < 5; i++) {
        printf("0x%04X (%5u): %d ones\n",
               test[i], test[i], count_ones(test[i]));
    }

    return 0;
}
```

### 비트 반전 (Reverse)

```c
// bit_reverse.c
#include <stdio.h>

// 8비트 반전
unsigned char reverse_bits(unsigned char n) {
    unsigned char result = 0;
    for (int i = 0; i < 8; i++) {
        result <<= 1;
        result |= (n & 1);
        n >>= 1;
    }
    return result;
}

int main(void) {
    unsigned char val = 0b10110001;  // 177

    printf("원본:   ");
    for (int i = 7; i >= 0; i--) printf("%d", (val >> i) & 1);
    printf(" (0x%02X)\n", val);

    unsigned char reversed = reverse_bits(val);

    printf("반전:   ");
    for (int i = 7; i >= 0; i--) printf("%d", (reversed >> i) & 1);
    printf(" (0x%02X)\n", reversed);

    return 0;
}
```

### 비트 스왑

```c
// bit_swap.c
#include <stdio.h>

// 두 비트 위치의 값 교환
unsigned char swap_bits(unsigned char n, int i, int j) {
    // i와 j 위치의 비트가 다를 때만 교환
    if (((n >> i) & 1) != ((n >> j) & 1)) {
        n ^= (1 << i) | (1 << j);  // 둘 다 토글
    }
    return n;
}

// 상위 4비트와 하위 4비트 교환
unsigned char swap_nibbles(unsigned char n) {
    return ((n & 0x0F) << 4) | ((n & 0xF0) >> 4);
}

int main(void) {
    unsigned char val = 0b11001010;

    printf("원본: 0x%02X (0b%d%d%d%d%d%d%d%d)\n", val,
           (val>>7)&1, (val>>6)&1, (val>>5)&1, (val>>4)&1,
           (val>>3)&1, (val>>2)&1, (val>>1)&1, val&1);

    // 비트 1과 6 교환
    unsigned char swapped = swap_bits(val, 1, 6);
    printf("비트 1,6 교환: 0x%02X\n", swapped);

    // 니블 교환
    unsigned char nibble_swapped = swap_nibbles(val);
    printf("니블 교환: 0x%02X\n", nibble_swapped);

    return 0;
}
```

### Arduino 비트 조작 예제

```cpp
// bit_manipulation.ino
// 비트 조작으로 여러 LED 제어

// LED 연결: 핀 8, 9, 10, 11 (PORTB 비트 0~3)
#define LED_MASK 0x0F  // 하위 4비트

void setup() {
    // 핀 8~11을 출력으로 설정
    DDRB |= LED_MASK;
    PORTB &= ~LED_MASK;  // 모든 LED 끄기

    Serial.begin(9600);
    Serial.println("Bit manipulation demo");
    Serial.println("Commands: 0-F (hex), r (rotate), i (invert)");
}

void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();

        if (cmd >= '0' && cmd <= '9') {
            // 숫자 0-9: 해당 패턴 표시
            PORTB = (PORTB & ~LED_MASK) | (cmd - '0');
        }
        else if (cmd >= 'a' && cmd <= 'f') {
            // a-f: 10-15 패턴 표시
            PORTB = (PORTB & ~LED_MASK) | (cmd - 'a' + 10);
        }
        else if (cmd >= 'A' && cmd <= 'F') {
            // A-F: 10-15 패턴 표시
            PORTB = (PORTB & ~LED_MASK) | (cmd - 'A' + 10);
        }
        else if (cmd == 'r' || cmd == 'R') {
            // 왼쪽 순환 시프트
            unsigned char leds = PORTB & LED_MASK;
            leds = ((leds << 1) | (leds >> 3)) & LED_MASK;
            PORTB = (PORTB & ~LED_MASK) | leds;
        }
        else if (cmd == 'i' || cmd == 'I') {
            // 반전
            PORTB ^= LED_MASK;
        }

        // 현재 상태 출력
        Serial.print("LED pattern: 0b");
        for (int i = 3; i >= 0; i--) {
            Serial.print((PORTB >> i) & 1);
        }
        Serial.println();
    }
}
```

---

## 연습 문제

### 연습 1: 비트 필드 추출
8비트 값에서 비트 2~5 (4비트)를 추출하는 함수를 작성하세요.

```c
unsigned char extract_bits(unsigned char value, int start, int length);
// extract_bits(0b11010110, 2, 4) → 0b0101 (5)
```

### 연습 2: 2의 거듭제곱 확인
주어진 숫자가 2의 거듭제곱인지 확인하는 함수를 비트 연산으로 작성하세요.

```c
int is_power_of_two(unsigned int n);
// is_power_of_two(8) → 1 (true)
// is_power_of_two(6) → 0 (false)
```

### 연습 3: 패리티 비트
8비트 값의 1인 비트 개수가 홀수면 1, 짝수면 0을 반환하는 함수를 작성하세요.

```c
int parity(unsigned char n);
// parity(0b10110001) → 0 (1이 4개 = 짝수)
// parity(0b10110011) → 1 (1이 5개 = 홀수)
```

### 연습 4: Arduino LED 바
4개의 LED로 0~15 값을 2진수로 표시하고, 버튼을 누를 때마다 값이 1씩 증가하도록 만드세요.

---

## 핵심 개념 정리

| 연산 | 코드 | 설명 |
|------|------|------|
| 비트 설정 | `val \|= (1 << n)` | n번째 비트를 1로 |
| 비트 해제 | `val &= ~(1 << n)` | n번째 비트를 0으로 |
| 비트 토글 | `val ^= (1 << n)` | n번째 비트 반전 |
| 비트 확인 | `(val >> n) & 1` | n번째 비트 값 |
| 하위 n비트 | `val & ((1 << n) - 1)` | 하위 n개 비트만 |

| 키워드 | 의미 |
|--------|------|
| volatile | 컴파일러 최적화 방지, 항상 메모리에서 읽기 |
| register | 레지스터에 저장 요청 (힌트) |

---

## 다음 단계

비트 연산을 익혔다면 다음 문서로 넘어가세요:
- [15. GPIO 제어](15_프로젝트_GPIO제어.md) - LED와 버튼으로 실습
