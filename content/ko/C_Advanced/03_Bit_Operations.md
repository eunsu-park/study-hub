# 고급 비트 연산

**이전**: [고급 메모리 관리](./02_Advanced_Memory_Management.md) | **다음**: [프로젝트: 동적 배열](./04_Project_Dynamic_Array.md)

시스템 프로그래밍, 임베디드 개발, 성능 최적화에 필수적인 비트 레벨 조작을 마스터합니다.

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 6가지 비트 연산자(`&`, `|`, `^`, `~`, `<<`, `>>`)를 적용하여 실용적인 문제를 해결할 수 있다
2. 마스킹 매크로를 사용하여 개별 비트를 읽기, 설정, 클리어, 토글할 수 있다
3. 메모리를 절약하기 위해 여러 불리언 플래그를 단일 바이트에 패킹할 수 있다
4. MCU 하드웨어 레지스터가 메모리 주소에 매핑되어 주변 장치를 제어하는 방식을 설명할 수 있다
5. 라이브러리 호출보다 훨씬 빠른 직접 레지스터 조작으로 GPIO를 제어할 수 있다
6. 하드웨어 레지스터와 인터럽트 공유 변수에 `volatile` 키워드가 필요한 이유를 설명할 수 있다
7. 팝카운트(population count), 비트 반전, 니블 스왑 등의 유틸리티 함수를 구현할 수 있다

---

하드웨어 레벨에서 모든 것은 비트입니다. 레지스터의 단일 비트 하나가 LED를 켜고, 통신 채널을 활성화하거나, 인터럽트를 발생시킬 수 있습니다. 비트 연산을 마스터하는 것은 고수준 C를 작성하는 것과 진정으로 하드웨어를 제어하는 것 사이의 다리입니다 -- 이것이 마이크로컨트롤러가 실제로 사용하는 언어입니다.

## 선수 조건
- C 언어 기본 문법
- 2진수와 16진수 표현

---

## 1. 비트 연산이 왜 중요한가?

### 시스템 프로그래밍에서 비트 연산이 필수적인 이유

```
1. 레지스터 제어
   - 모든 MCU 기능은 레지스터(특수 메모리)를 통해 제어됨
   - 레지스터의 각 비트가 특정 기능을 제어

2. 메모리 절약
   - 임베디드 시스템은 메모리가 제한적 (KB 단위)
   - 8개의 플래그를 1바이트에 저장 가능

3. 통신 프로토콜
   - 데이터 패킷의 비트 레벨 해석 필요

4. 성능 최적화
   - 비트 연산은 CPU에서 가장 빠른 연산
```

### 예제: LED 제어 레지스터

> **Arduino Uno Port B 레지스터 (핀 8~13)**
>
> `PORTB = 0b00100000`
>
> | 비트 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
> |------|---|---|---|---|---|---|---|---|
> | 값 | 0 | 0 | **1** | 0 | 0 | 0 | 0 | 0 |
> | 핀 | - | - | Pin13 | Pin12 | Pin11 | Pin10 | Pin9 | Pin8 |
>
> 비트 5 = 1 -> 핀 13에 HIGH 출력 (LED 켜짐)

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

    // OR (|): 하나라도 1이면 1
    printf("a | b    = "); print_binary(a | b);   // 1111 1011

    // XOR (^): 다르면 1
    printf("a ^ b    = "); print_binary(a ^ b);   // 0111 1001

    // NOT (~): 비트 반전
    printf("~a       = "); print_binary(~a);      // 0011 0101

    // 왼쪽 시프트 (<<): 왼쪽으로 이동, 0으로 채움
    printf("a << 2   = "); print_binary(a << 2);  // 0010 1000

    // 오른쪽 시프트 (>>): 오른쪽으로 이동
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

// 비트 2 읽기
int bit2 = (reg >> 2) & 1;  // 결과: 1

// 비트 3 읽기
int bit3 = (reg >> 3) & 1;  // 결과: 0

// 매크로로 정의
#define GET_BIT(value, bit) (((value) >> (bit)) & 1)

// 사용 예
if (GET_BIT(reg, 5)) {
    printf("Bit 5 is set\n");
}
```

### 3.2 특정 비트 설정 (SET)

```c
// 특정 비트를 1로 설정
// 방법: value |= (1 << bit)

unsigned char reg = 0b10100000;

// 비트 3을 1로 설정
reg |= (1 << 3);  // 결과: 0b10101000

// 여러 비트를 동시에 설정
reg |= (1 << 1) | (1 << 4);  // 비트 1, 4 설정

// 매크로로 정의
#define SET_BIT(value, bit) ((value) |= (1 << (bit)))

// 사용 예
SET_BIT(reg, 6);  // 비트 6 설정
```

**작동 원리:**
```
reg       = 1010 0000
1 << 3    = 0000 1000
           ----------- OR
result    = 1010 1000
```

### 3.3 특정 비트 클리어 (CLEAR)

```c
// 특정 비트를 0으로 클리어
// 방법: value &= ~(1 << bit)

unsigned char reg = 0b11111111;

// 비트 5를 0으로 클리어
reg &= ~(1 << 5);  // 결과: 0b11011111

// 매크로로 정의
#define CLEAR_BIT(value, bit) ((value) &= ~(1 << (bit)))

// 사용 예
CLEAR_BIT(reg, 2);  // 비트 2 클리어
```

**작동 원리:**
```
reg       = 1111 1111
1 << 5    = 0010 0000
~(1 << 5) = 1101 1111
           ----------- AND
result    = 1101 1111
```

### 3.4 특정 비트 토글 (TOGGLE)

```c
// 특정 비트 토글 (0->1, 1->0)
// 방법: value ^= (1 << bit)

unsigned char reg = 0b10101010;

// 비트 4 토글
reg ^= (1 << 4);  // 결과: 0b10111010 (0->1)
reg ^= (1 << 4);  // 결과: 0b10101010 (1->0)

// 매크로로 정의
#define TOGGLE_BIT(value, bit) ((value) ^= (1 << (bit)))
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

// 다중 비트 연산
#define SET_BITS(reg, mask)     ((reg) |= (mask))
#define CLEAR_BITS(reg, mask)   ((reg) &= ~(mask))
#define TOGGLE_BITS(reg, mask)  ((reg) ^= (mask))

// 비트 필드 연산
#define GET_FIELD(reg, mask, shift)     (((reg) & (mask)) >> (shift))
#define SET_FIELD(reg, mask, shift, val) \
    ((reg) = ((reg) & ~(mask)) | (((val) << (shift)) & (mask)))

#endif
```

---

## 4. 플래그 관리

단일 변수에서 여러 상태를 관리합니다.

### 플래그 정의

```c
// flags.c
#include <stdio.h>
#include <stdbool.h>

// 각 비트에 의미 부여
#define FLAG_RUNNING    (1 << 0)  // 비트 0: 실행 중
#define FLAG_ERROR      (1 << 1)  // 비트 1: 오류 발생
#define FLAG_CONNECTED  (1 << 2)  // 비트 2: 연결됨
#define FLAG_READY      (1 << 3)  // 비트 3: 준비됨
#define FLAG_BUSY       (1 << 4)  // 비트 4: 바쁨
#define FLAG_TIMEOUT    (1 << 5)  // 비트 5: 타임아웃

// 전역 상태 플래그
unsigned char system_flags = 0;

// 플래그 설정
void set_flag(unsigned char flag) {
    system_flags |= flag;
}

// 플래그 클리어
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
        printf("System running\n");
    }

    if (is_flag_set(FLAG_ERROR)) {
        printf("Error occurred!\n");
    } else {
        printf("Normal operation\n");
    }

    // 오류 발생
    set_flag(FLAG_ERROR);
    printf("After setting error flag: 0x%02X\n", system_flags);

    // 오류 해결
    clear_flag(FLAG_ERROR);
    printf("After clearing error flag: 0x%02X\n", system_flags);

    return 0;
}
```

---

## 5. 레지스터 개념

### MCU 레지스터란?

레지스터는 MCU 내부에서 하드웨어를 제어하는 특수 메모리 위치입니다.

> **ATmega328P GPIO 레지스터**
>
> - **DDRx** (데이터 방향 레지스터) - 핀 입출력 방향 설정. 0 = 입력(INPUT), 1 = 출력(OUTPUT)
> - **PORTx** (포트 출력 레지스터) - 출력 모드: HIGH/LOW 출력. 입력 모드: 풀업 저항 활성화
> - **PINx** (포트 입력 레지스터) - 현재 핀 상태 읽기
>
> x = B (핀 8~13), C (아날로그 핀), D (핀 0~7)

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

### 레지스터와 volatile

```c
// MCU 레지스터는 항상 volatile
// 하드웨어가 값을 변경할 수 있기 때문

// 실제 Arduino 헤더 파일 정의 (avr/io.h)
#define PORTB (*(volatile uint8_t *)0x25)
#define DDRB  (*(volatile uint8_t *)0x24)
#define PINB  (*(volatile uint8_t *)0x23)

// 설명:
// 0x25 = PORTB 레지스터의 메모리 주소
// (volatile uint8_t *) = 주소를 volatile 포인터로 캐스팅
// * = 해당 주소의 값에 접근
```

---

## 6. volatile 키워드

### volatile이 필요한 이유

```c
// 문제: 컴파일러 최적화

int flag = 0;

// 인터럽트 핸들러 (하드웨어에 의해 호출)
void interrupt_handler() {
    flag = 1;
}

int main() {
    while (flag == 0) {
        // 대기
    }
    // flag가 1이 되면 여기서 실행
}
```

컴파일러가 루프 내에서 `flag`가 변경되지 않는다고 판단하여 최적화할 수 있습니다:

```c
// 컴파일러가 최적화한 코드 (문제!)
if (flag == 0) {
    while (1) { }  // 무한 루프로 변환됨
}
```

### volatile 사용법

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
volatile = "예측 불가능"

컴파일러에게 알림:
1. 이 변수는 외부에서 언제든 변경될 수 있음
2. 최적화하지 말고, 항상 메모리에서 읽기
3. 레지스터에 캐시하지 않기
```

---

## 7. 실습: 비트 조작 유틸리티

### 비트 카운터

```c
// bit_counter.c
#include <stdio.h>

// 1 비트 개수 세기 (popcount)
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

### 비트 반전

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

    printf("Original: ");
    for (int i = 7; i >= 0; i--) printf("%d", (val >> i) & 1);
    printf(" (0x%02X)\n", val);

    unsigned char reversed = reverse_bits(val);

    printf("Reversed: ");
    for (int i = 7; i >= 0; i--) printf("%d", (reversed >> i) & 1);
    printf(" (0x%02X)\n", reversed);

    return 0;
}
```

### 비트 스왑

```c
// bit_swap.c
#include <stdio.h>

// 두 비트 위치 교환
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

    printf("Original: 0x%02X (0b%d%d%d%d%d%d%d%d)\n", val,
           (val>>7)&1, (val>>6)&1, (val>>5)&1, (val>>4)&1,
           (val>>3)&1, (val>>2)&1, (val>>1)&1, val&1);

    // 비트 1과 6 교환
    unsigned char swapped = swap_bits(val, 1, 6);
    printf("Bits 1,6 swapped: 0x%02X\n", swapped);

    // 니블 교환
    unsigned char nibble_swapped = swap_nibbles(val);
    printf("Nibbles swapped: 0x%02X\n", nibble_swapped);

    return 0;
}
```

---

## 연습문제

### 연습문제 1: 비트 필드 추출
8비트 값에서 비트 2~5 (4비트)를 추출하는 함수를 작성하세요.

```c
unsigned char extract_bits(unsigned char value, int start, int length);
// extract_bits(0b11010110, 2, 4) -> 0b0101 (5)
```

### 연습문제 2: 2의 거듭제곱 확인
비트 연산을 사용하여 숫자가 2의 거듭제곱인지 확인하는 함수를 작성하세요.

```c
int is_power_of_two(unsigned int n);
// is_power_of_two(8) -> 1 (참)
// is_power_of_two(6) -> 0 (거짓)
```

### 연습문제 3: 패리티 비트
1 비트의 개수가 홀수이면 1, 짝수이면 0을 반환하는 함수를 작성하세요.

```c
int parity(unsigned char n);
// parity(0b10110001) -> 0 (4개의 1 = 짝수)
// parity(0b10110011) -> 1 (5개의 1 = 홀수)
```

### 연습문제 4: XOR 스왑
임시 변수 없이 XOR 연산만으로 두 정수를 교환하는 스왑 함수를 구현하세요.

### 연습문제 5: 비트 배열
N/8 바이트를 사용하여 N개의 불리언 값을 저장하는 컴팩트 비트 배열을 구현하세요. `bit_set`, `bit_clear`, `bit_get`, `bit_toggle` 함수를 제공하세요.

---

## 핵심 개념 정리

| 연산 | 코드 | 설명 |
|------|------|------|
| 비트 설정 | `val \|= (1 << n)` | n번째 비트를 1로 설정 |
| 비트 클리어 | `val &= ~(1 << n)` | n번째 비트를 0으로 설정 |
| 비트 토글 | `val ^= (1 << n)` | n번째 비트 반전 |
| 비트 확인 | `(val >> n) & 1` | n번째 비트 값 가져오기 |
| 하위 n비트 | `val & ((1 << n) - 1)` | 하위 n비트만 추출 |

| 키워드 | 의미 |
|--------|------|
| volatile | 컴파일러 최적화 방지, 항상 메모리에서 읽기 |
| register | 레지스터에 저장 요청 (힌트) |

---

## 다음 단계

비트 연산을 마스터했다면 다음으로 진행하세요:
- [04. 프로젝트: 동적 배열](./04_Project_Dynamic_Array.md) - 처음부터 가변 배열 만들기
