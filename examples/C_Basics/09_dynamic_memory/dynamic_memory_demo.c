/*
 * dynamic_memory_demo.c — malloc, calloc, realloc, free patterns.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o dynamic_memory_demo dynamic_memory_demo.c
 * Run:     ./dynamic_memory_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    /* malloc: allocate uninitialized memory */
    printf("=== malloc ===\n");
    int *arr = malloc(5 * sizeof(int));
    if (!arr) { perror("malloc"); return 1; }

    for (int i = 0; i < 5; i++)
        arr[i] = (i + 1) * 10;

    printf("arr: ");
    for (int i = 0; i < 5; i++)
        printf("%d ", arr[i]);
    printf("\n");
    free(arr);
    arr = NULL;  /* good practice: avoid dangling pointer */

    /* calloc: allocate zero-initialized memory */
    printf("\n=== calloc ===\n");
    int *zeros = calloc(5, sizeof(int));
    if (!zeros) { perror("calloc"); return 1; }

    printf("zeros: ");
    for (int i = 0; i < 5; i++)
        printf("%d ", zeros[i]);  /* all zeros */
    printf("\n");
    free(zeros);

    /* realloc: resize allocation */
    printf("\n=== realloc (dynamic growth) ===\n");
    int cap = 4, len = 0;
    int *buf = malloc((size_t)cap * sizeof(int));
    if (!buf) { perror("malloc"); return 1; }

    for (int i = 0; i < 10; i++) {
        if (len == cap) {
            cap *= 2;
            int *tmp = realloc(buf, (size_t)cap * sizeof(int));
            if (!tmp) { perror("realloc"); free(buf); return 1; }
            buf = tmp;
            printf("  [resized to capacity %d]\n", cap);
        }
        buf[len++] = i * i;
    }

    printf("buf (%d elements): ", len);
    for (int i = 0; i < len; i++)
        printf("%d ", buf[i]);
    printf("\n");
    free(buf);

    /* Dynamic string */
    printf("\n=== Dynamic String ===\n");
    const char *src = "Hello, dynamic world!";
    size_t slen = strlen(src);
    char *str = malloc(slen + 1);
    if (!str) { perror("malloc"); return 1; }

    strcpy(str, src);
    printf("str = \"%s\" (len=%zu)\n", str, slen);
    free(str);

    /* 2D dynamic array */
    printf("\n=== 2D Dynamic Array (3x4) ===\n");
    int rows = 3, cols = 4;
    int **mat = malloc((size_t)rows * sizeof(int *));
    if (!mat) { perror("malloc"); return 1; }

    for (int r = 0; r < rows; r++) {
        mat[r] = malloc((size_t)cols * sizeof(int));
        if (!mat[r]) { perror("malloc"); return 1; }
        for (int c = 0; c < cols; c++)
            mat[r][c] = r * cols + c;
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            printf("%3d", mat[r][c]);
        printf("\n");
    }

    /* Free 2D array (rows first, then row pointers) */
    for (int r = 0; r < rows; r++)
        free(mat[r]);
    free(mat);

    printf("\nAll memory freed successfully.\n");
    return 0;
}
