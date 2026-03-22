/*
 * Exercises for Lesson 04: Control Flow
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex04 04_control_flow.c
 */
#include <stdio.h>

/* === Exercise 1: Pattern Printing === */
/* Problem: Print several patterns using nested loops. */
void exercise_1(void) {
    printf("=== Exercise 1: Pattern Printing ===\n");

    int n = 5;

    /* Right triangle */
    printf("Right triangle (n=%d):\n", n);
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < i; j++) {
            printf("* ");
        }
        printf("\n");
    }

    /* Inverted triangle */
    printf("\nInverted triangle:\n");
    for (int i = n; i >= 1; i--) {
        for (int j = 0; j < i; j++) {
            printf("* ");
        }
        printf("\n");
    }

    /* Diamond */
    printf("\nDiamond (n=%d):\n", n);
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < n - i; j++) printf(" ");
        for (int j = 0; j < 2 * i - 1; j++) printf("*");
        printf("\n");
    }
    for (int i = n - 1; i >= 1; i--) {
        for (int j = 0; j < n - i; j++) printf(" ");
        for (int j = 0; j < 2 * i - 1; j++) printf("*");
        printf("\n");
    }

    /* Number pyramid */
    printf("\nNumber pyramid:\n");
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < n - i; j++) printf(" ");
        for (int j = 1; j <= i; j++) printf("%d ", j);
        printf("\n");
    }
}

/* === Exercise 2: Menu System === */
/* Problem: Implement a menu-driven program using switch and do-while. */
void exercise_2(void) {
    printf("\n=== Exercise 2: Menu System ===\n");

    /*
     * In a real interactive program, this would use scanf in a loop.
     * Here we simulate menu choices to demonstrate the pattern.
     */
    int choices[] = {1, 2, 3, 4, 0};  /* simulated user input */
    int idx = 0;
    int choice;

    do {
        printf("\n--- Menu ---\n");
        printf("1. Celsius to Fahrenheit\n");
        printf("2. Fahrenheit to Celsius\n");
        printf("3. Km to Miles\n");
        printf("4. Miles to Km\n");
        printf("0. Quit\n");

        choice = choices[idx++];
        printf("Choice (simulated): %d\n", choice);

        switch (choice) {
            case 1: {
                double c = 100.0;
                printf("%.1f C = %.1f F\n", c, c * 9.0 / 5.0 + 32.0);
                break;
            }
            case 2: {
                double f = 212.0;
                printf("%.1f F = %.1f C\n", f, (f - 32.0) * 5.0 / 9.0);
                break;
            }
            case 3: {
                double km = 10.0;
                printf("%.1f km = %.2f miles\n", km, km * 0.621371);
                break;
            }
            case 4: {
                double mi = 6.0;
                printf("%.1f miles = %.2f km\n", mi, mi / 0.621371);
                break;
            }
            case 0:
                printf("Goodbye!\n");
                break;
            default:
                printf("Invalid choice.\n");
                break;
        }
    } while (choice != 0);
}

/* === Exercise 3: Number Classification === */
/* Problem: Classify numbers using various control flow constructs. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Number Classification ===\n");

    int numbers[] = {0, 1, 2, 3, 7, 12, 15, 28, 37, 100};
    int count = sizeof(numbers) / sizeof(numbers[0]);

    for (int idx = 0; idx < count; idx++) {
        int n = numbers[idx];
        printf("\n%d: ", n);

        /* Even/odd */
        printf("%s", (n % 2 == 0) ? "even" : "odd");

        /* Prime check */
        if (n > 1) {
            int is_prime = 1;
            for (int i = 2; i * i <= n; i++) {
                if (n % i == 0) {
                    is_prime = 0;
                    break;
                }
            }
            if (is_prime) printf(", prime");
        }

        /* Perfect number check (sum of proper divisors equals number) */
        if (n > 1) {
            int div_sum = 0;
            for (int i = 1; i < n; i++) {
                if (n % i == 0) div_sum += i;
            }
            if (div_sum == n) printf(", perfect");
        }

        /* FizzBuzz classification */
        if (n > 0) {
            if (n % 15 == 0)      printf(", FizzBuzz");
            else if (n % 3 == 0)  printf(", Fizz");
            else if (n % 5 == 0)  printf(", Buzz");
        }
    }
    printf("\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
