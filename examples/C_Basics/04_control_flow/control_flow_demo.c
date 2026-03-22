/*
 * control_flow_demo.c — if/else, switch, for, while, do-while examples.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o control_flow_demo control_flow_demo.c
 * Run:     ./control_flow_demo
 */

#include <stdio.h>

int main(void)
{
    /* if / else if / else */
    printf("=== if / else ===\n");
    int score = 85;
    if (score >= 90)
        printf("Score %d -> Grade A\n", score);
    else if (score >= 80)
        printf("Score %d -> Grade B\n", score);
    else if (score >= 70)
        printf("Score %d -> Grade C\n", score);
    else
        printf("Score %d -> Grade F\n", score);

    /* switch */
    printf("\n=== switch ===\n");
    int day = 3;
    switch (day) {
        case 1: printf("Day %d: Monday\n", day);    break;
        case 2: printf("Day %d: Tuesday\n", day);   break;
        case 3: printf("Day %d: Wednesday\n", day);  break;
        case 4: printf("Day %d: Thursday\n", day);   break;
        case 5: printf("Day %d: Friday\n", day);     break;
        default: printf("Day %d: Weekend\n", day);   break;
    }

    /* for loop */
    printf("\n=== for loop (sum 1..10) ===\n");
    int sum = 0;
    for (int i = 1; i <= 10; i++)
        sum += i;
    printf("Sum = %d\n", sum);

    /* while loop */
    printf("\n=== while loop (Collatz sequence from 6) ===\n");
    int n = 6;
    printf("%d", n);
    while (n != 1) {
        n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
        printf(" -> %d", n);
    }
    printf("\n");

    /* do-while loop */
    printf("\n=== do-while loop ===\n");
    int count = 0;
    do {
        printf("count = %d\n", count);
        count++;
    } while (count < 3);

    /* break and continue */
    printf("\n=== break & continue (skip multiples of 3, stop at 15) ===\n");
    for (int i = 1; i <= 20; i++) {
        if (i == 15) {
            printf("[break at %d]\n", i);
            break;
        }
        if (i % 3 == 0) continue;
        printf("%d ", i);
    }
    printf("\n");

    return 0;
}
