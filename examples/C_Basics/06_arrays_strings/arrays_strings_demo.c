/*
 * arrays_strings_demo.c — Array operations and string functions.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o arrays_strings_demo arrays_strings_demo.c
 * Run:     ./arrays_strings_demo
 */

#include <stdio.h>
#include <string.h>

int main(void)
{
    /* Array declaration and initialization */
    printf("=== Arrays ===\n");
    int nums[5] = {10, 20, 30, 40, 50};
    int len = (int)(sizeof(nums) / sizeof(nums[0]));

    printf("Array: ");
    for (int i = 0; i < len; i++)
        printf("%d ", nums[i]);
    printf("\n");

    /* 2D array */
    printf("\n=== 2D Array (3x3 identity matrix) ===\n");
    int matrix[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            printf("%d ", matrix[r][c]);
        printf("\n");
    }

    /* String basics (null-terminated char array) */
    printf("\n=== Strings ===\n");
    char greeting[] = "Hello, C!";
    printf("greeting    = \"%s\"\n", greeting);
    printf("strlen      = %zu\n", strlen(greeting));
    printf("sizeof      = %zu (includes '\\0')\n", sizeof(greeting));

    /* String copy and concatenation */
    char dest[64];
    strcpy(dest, "Hello");
    printf("\nstrcpy  -> \"%s\"\n", dest);
    strcat(dest, ", World!");
    printf("strcat  -> \"%s\"\n", dest);

    /* String comparison */
    printf("\n=== Comparison ===\n");
    const char *a = "apple";
    const char *b = "banana";
    int cmp = strcmp(a, b);
    printf("strcmp(\"%s\", \"%s\") = %d  (%s)\n",
           a, b, cmp, cmp < 0 ? "a < b" : cmp > 0 ? "a > b" : "equal");

    /* String search */
    printf("\n=== Search ===\n");
    const char *haystack = "The quick brown fox";
    const char *found = strstr(haystack, "brown");
    if (found)
        printf("strstr found \"brown\" at index %ld\n", found - haystack);

    char ch = 'q';
    char *pos = strchr(haystack, ch);
    if (pos)
        printf("strchr found '%c' at index %ld\n", ch, pos - haystack);

    /* Character-by-character iteration */
    printf("\n=== Char-by-char (uppercase) ===\n");
    char msg[] = "hello";
    for (int i = 0; msg[i] != '\0'; i++) {
        if (msg[i] >= 'a' && msg[i] <= 'z')
            msg[i] -= 32;  /* ASCII lowercase -> uppercase */
    }
    printf("result = \"%s\"\n", msg);

    return 0;
}
