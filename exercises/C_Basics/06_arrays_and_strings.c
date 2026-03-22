/*
 * Exercises for Lesson 06: Arrays and Strings
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex06 06_arrays_and_strings.c
 */
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* === Exercise 1: Array Reversal === */
/* Problem: Reverse an array in-place using swap. */

void reverse_array(int *arr, int size) {
    for (int i = 0, j = size - 1; i < j; i++, j--) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

void print_array(const int *arr, int size) {
    printf("{");
    for (int i = 0; i < size; i++) {
        printf("%d%s", arr[i], (i < size - 1) ? ", " : "");
    }
    printf("}\n");
}

void exercise_1(void) {
    printf("=== Exercise 1: Array Reversal ===\n");

    int arr1[] = {1, 2, 3, 4, 5};
    int size1 = sizeof(arr1) / sizeof(arr1[0]);
    printf("Before: "); print_array(arr1, size1);
    reverse_array(arr1, size1);
    printf("After:  "); print_array(arr1, size1);

    int arr2[] = {10, 20};
    int size2 = sizeof(arr2) / sizeof(arr2[0]);
    printf("Before: "); print_array(arr2, size2);
    reverse_array(arr2, size2);
    printf("After:  "); print_array(arr2, size2);

    int arr3[] = {42};
    printf("Before: "); print_array(arr3, 1);
    reverse_array(arr3, 1);
    printf("After:  "); print_array(arr3, 1);
}

/* === Exercise 2: String Reverse === */
/* Problem: Reverse a string in-place without using library functions. */

void reverse_string(char *str) {
    int len = 0;
    while (str[len] != '\0') len++;

    for (int i = 0, j = len - 1; i < j; i++, j--) {
        char tmp = str[i];
        str[i] = str[j];
        str[j] = tmp;
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: String Reverse ===\n");

    char s1[] = "Hello, World!";
    printf("Before: \"%s\"\n", s1);
    reverse_string(s1);
    printf("After:  \"%s\"\n", s1);

    char s2[] = "abcde";
    printf("Before: \"%s\"\n", s2);
    reverse_string(s2);
    printf("After:  \"%s\"\n", s2);

    char s3[] = "";
    printf("Before: \"%s\"\n", s3);
    reverse_string(s3);
    printf("After:  \"%s\"\n", s3);
}

/* === Exercise 3: Count Words === */
/* Problem: Count the number of words in a string (words separated by spaces). */

int count_words(const char *str) {
    int count = 0;
    int in_word = 0;

    while (*str) {
        if (isspace((unsigned char)*str)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            count++;
        }
        str++;
    }
    return count;
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Count Words ===\n");

    const char *tests[] = {
        "Hello World",
        "  leading and trailing spaces  ",
        "single",
        "",
        "   ",
        "multiple   spaces   between   words"
    };
    int n = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n; i++) {
        printf("\"%s\" -> %d words\n", tests[i], count_words(tests[i]));
    }
}

/* === Exercise 4: Capitalize Words === */
/* Problem: Capitalize the first letter of each word in a string. */

void capitalize_words(char *str) {
    int capitalize_next = 1;

    while (*str) {
        if (isspace((unsigned char)*str)) {
            capitalize_next = 1;
        } else if (capitalize_next) {
            *str = (char)toupper((unsigned char)*str);
            capitalize_next = 0;
        }
        str++;
    }
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Capitalize Words ===\n");

    char s1[] = "hello world from c programming";
    printf("Before: \"%s\"\n", s1);
    capitalize_words(s1);
    printf("After:  \"%s\"\n", s1);

    char s2[] = "already Capitalized Some Words";
    printf("Before: \"%s\"\n", s2);
    capitalize_words(s2);
    printf("After:  \"%s\"\n", s2);
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();

    printf("\nAll exercises completed!\n");
    return 0;
}
