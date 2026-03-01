// bit_manipulation.c
// 비트 조작 예제

#include <stdio.h>

// 비트 출력 함수
void print_binary(unsigned char n) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
        if (i == 4) printf(" ");
    }
}

// 비트 조작 매크로
// Why: wrapping every parameter in parentheses prevents operator precedence bugs —
// without them, SET_BIT(x, a+b) would expand to (x |= (1 << a+b)) where + binds
// tighter than <<, giving the wrong bit position
#define SET_BIT(reg, bit)    ((reg) |= (1 << (bit)))
#define CLEAR_BIT(reg, bit)  ((reg) &= ~(1 << (bit)))
#define TOGGLE_BIT(reg, bit) ((reg) ^= (1 << (bit)))
#define GET_BIT(reg, bit)    (((reg) >> (bit)) & 1)

int main(void) {
    unsigned char value = 0b10110010;  // 178

    printf("=== 비트 조작 예제 ===\n\n");

    printf("초기값: ");
    print_binary(value);
    printf(" (0x%02X, %d)\n\n", value, value);

    // 비트 설정 (SET)
    printf("비트 3 설정 (SET_BIT):\n");
    SET_BIT(value, 3);
    printf("  결과: ");
    print_binary(value);
    printf(" (0x%02X)\n\n", value);

    // 비트 해제 (CLEAR)
    printf("비트 5 해제 (CLEAR_BIT):\n");
    CLEAR_BIT(value, 5);
    printf("  결과: ");
    print_binary(value);
    printf(" (0x%02X)\n\n", value);

    // 비트 토글 (TOGGLE)
    printf("비트 0 토글 (TOGGLE_BIT):\n");
    TOGGLE_BIT(value, 0);
    printf("  결과: ");
    print_binary(value);
    printf(" (0x%02X)\n\n", value);

    // 비트 읽기 (GET)
    printf("각 비트 값 읽기:\n");
    for (int i = 7; i >= 0; i--) {
        printf("  비트 %d: %d\n", i, GET_BIT(value, i));
    }
    printf("\n");

    // 플래그 예제
    printf("=== 플래그 관리 예제 ===\n\n");

    // Why: using bit shifts (1 << N) for flag values ensures each flag occupies
    // exactly one bit — a single byte can store 8 independent boolean flags,
    // far more memory-efficient than 8 separate bool variables
    #define FLAG_RUNNING   (1 << 0)
    #define FLAG_ERROR     (1 << 1)
    #define FLAG_CONNECTED (1 << 2)
    #define FLAG_READY     (1 << 3)

    unsigned char flags = 0;

    printf("초기 플래그: ");
    print_binary(flags);
    printf("\n\n");

    // 플래그 설정
    flags |= FLAG_RUNNING;
    printf("RUNNING 플래그 설정: ");
    print_binary(flags);
    printf("\n");

    flags |= FLAG_READY;
    printf("READY 플래그 설정:   ");
    print_binary(flags);
    printf("\n\n");

    // Why: bitwise AND with a flag mask tests that specific bit — if the bit is set,
    // the result is non-zero (truthy); if cleared, the result is zero (falsy)
    if (flags & FLAG_RUNNING) {
        printf("시스템이 실행 중입니다.\n");
    }

    if (flags & FLAG_ERROR) {
        printf("에러 발생!\n");
    } else {
        printf("정상 동작 중\n");
    }
    printf("\n");

    // Why: ~FLAG_RUNNING inverts to 0b11111110, and AND clears only bit 0 —
    // this is the only safe way to clear one bit without affecting the others
    flags &= ~FLAG_RUNNING;
    printf("RUNNING 플래그 해제: ");
    print_binary(flags);
    printf("\n");

    return 0;
}
