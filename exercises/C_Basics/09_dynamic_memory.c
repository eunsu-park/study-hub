/*
 * Exercises for Lesson 09: Dynamic Memory
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex09 09_dynamic_memory.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Exercise 1: Dynamic String Builder === */
/* Problem: Implement a string builder that grows dynamically. */

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} StringBuilder;

StringBuilder *sb_create(size_t initial_cap) {
    StringBuilder *sb = malloc(sizeof(StringBuilder));
    if (!sb) return NULL;
    sb->capacity = (initial_cap > 0) ? initial_cap : 16;
    sb->data = malloc(sb->capacity);
    if (!sb->data) { free(sb); return NULL; }
    sb->data[0] = '\0';
    sb->length = 0;
    return sb;
}

int sb_ensure_capacity(StringBuilder *sb, size_t needed) {
    if (sb->length + needed + 1 <= sb->capacity) return 1;
    size_t new_cap = sb->capacity;
    while (new_cap < sb->length + needed + 1) {
        new_cap *= 2;
    }
    char *new_data = realloc(sb->data, new_cap);
    if (!new_data) return 0;
    sb->data = new_data;
    sb->capacity = new_cap;
    return 1;
}

int sb_append(StringBuilder *sb, const char *str) {
    size_t len = strlen(str);
    if (!sb_ensure_capacity(sb, len)) return 0;
    memcpy(sb->data + sb->length, str, len + 1);
    sb->length += len;
    return 1;
}

int sb_append_char(StringBuilder *sb, char c) {
    if (!sb_ensure_capacity(sb, 1)) return 0;
    sb->data[sb->length++] = c;
    sb->data[sb->length] = '\0';
    return 1;
}

const char *sb_str(const StringBuilder *sb) {
    return sb->data;
}

void sb_free(StringBuilder *sb) {
    if (sb) {
        free(sb->data);
        free(sb);
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Dynamic String Builder ===\n");

    StringBuilder *sb = sb_create(8);
    if (!sb) { printf("Allocation failed!\n"); return; }

    printf("Initial capacity: %zu\n", sb->capacity);

    sb_append(sb, "Hello");
    printf("After 'Hello': len=%zu, cap=%zu\n", sb->length, sb->capacity);

    sb_append(sb, ", ");
    sb_append(sb, "World");
    printf("After ', World': len=%zu, cap=%zu\n", sb->length, sb->capacity);

    sb_append_char(sb, '!');
    printf("Result: \"%s\"\n", sb_str(sb));

    /* Build a longer string to trigger multiple resizes */
    for (int i = 0; i < 10; i++) {
        sb_append(sb, " more");
    }
    printf("After appending more: len=%zu, cap=%zu\n",
           sb->length, sb->capacity);

    sb_free(sb);
    printf("StringBuilder freed.\n");
}

/* === Exercise 2: Resizable Integer Array === */
/* Problem: Implement a dynamic array (vector) for integers. */

typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} IntVector;

IntVector *vec_create(size_t initial_cap) {
    IntVector *v = malloc(sizeof(IntVector));
    if (!v) return NULL;
    v->capacity = (initial_cap > 0) ? initial_cap : 4;
    v->data = malloc(v->capacity * sizeof(int));
    if (!v->data) { free(v); return NULL; }
    v->size = 0;
    return v;
}

int vec_push(IntVector *v, int value) {
    if (v->size >= v->capacity) {
        size_t new_cap = v->capacity * 2;
        int *new_data = realloc(v->data, new_cap * sizeof(int));
        if (!new_data) return 0;
        v->data = new_data;
        v->capacity = new_cap;
    }
    v->data[v->size++] = value;
    return 1;
}

int vec_pop(IntVector *v, int *out) {
    if (v->size == 0) return 0;
    *out = v->data[--v->size];
    return 1;
}

int vec_get(const IntVector *v, size_t index, int *out) {
    if (index >= v->size) return 0;
    *out = v->data[index];
    return 1;
}

void vec_print(const IntVector *v) {
    printf("[");
    for (size_t i = 0; i < v->size; i++) {
        printf("%d%s", v->data[i], (i < v->size - 1) ? ", " : "");
    }
    printf("] (size=%zu, cap=%zu)\n", v->size, v->capacity);
}

void vec_free(IntVector *v) {
    if (v) {
        free(v->data);
        free(v);
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Resizable Integer Array ===\n");

    IntVector *v = vec_create(2);
    if (!v) { printf("Allocation failed!\n"); return; }

    printf("Empty: ");
    vec_print(v);

    /* Push elements, observe capacity growth */
    for (int i = 1; i <= 10; i++) {
        vec_push(v, i * 10);
    }
    printf("After pushing 10 elements: ");
    vec_print(v);

    /* Random access */
    int val;
    vec_get(v, 5, &val);
    printf("Element at index 5: %d\n", val);

    /* Pop elements */
    for (int i = 0; i < 3; i++) {
        vec_pop(v, &val);
        printf("Popped: %d\n", val);
    }
    printf("After 3 pops: ");
    vec_print(v);

    vec_free(v);
    printf("Vector freed.\n");

    /*
     * Key lessons:
     * - Always check malloc/realloc return values
     * - Doubling strategy gives amortized O(1) push
     * - Must free both the data array and the struct itself
     * - realloc(NULL, size) behaves like malloc(size)
     */
}

int main(void) {
    exercise_1();
    exercise_2();

    printf("\nAll exercises completed!\n");
    return 0;
}
