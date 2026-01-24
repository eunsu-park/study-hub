#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;      // p는 x의 주소를 저장

    printf("x의 값: %d\n", x);        // 42
    printf("x의 주소: %p\n", &x);     // 0x7fff...
    printf("p의 값 (주소): %p\n", p); // 0x7fff... (같은 주소)
    printf("p가 가리키는 값: %d\n", *p);  // 42 (역참조)

    // 포인터로 값 변경
    *p = 100;
    printf("x의 새 값: %d\n", x);     // 100

    return 0;
}