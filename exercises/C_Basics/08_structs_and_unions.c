/*
 * Exercises for Lesson 08: Structs and Unions
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex08 08_structs_and_unions.c
 */
#include <stdio.h>
#include <string.h>

/* === Exercise 1: Student Record Struct === */
/* Problem: Create a student record system with struct operations. */

#define MAX_NAME 50
#define MAX_STUDENTS 5
#define NUM_GRADES 4

typedef struct {
    int id;
    char name[MAX_NAME];
    double grades[NUM_GRADES];
    double gpa;
} Student;

double calculate_gpa(const double *grades, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += grades[i];
    }
    return sum / count;
}

void print_student(const Student *s) {
    printf("  ID: %d, Name: %-15s, Grades: [", s->id, s->name);
    for (int i = 0; i < NUM_GRADES; i++) {
        printf("%.1f%s", s->grades[i], (i < NUM_GRADES - 1) ? ", " : "");
    }
    printf("], GPA: %.2f\n", s->gpa);
}

const Student *find_top_student(const Student *students, int count) {
    const Student *top = &students[0];
    for (int i = 1; i < count; i++) {
        if (students[i].gpa > top->gpa) {
            top = &students[i];
        }
    }
    return top;
}

void exercise_1(void) {
    printf("=== Exercise 1: Student Record Struct ===\n");

    Student students[MAX_STUDENTS] = {
        {101, "Alice Johnson",   {3.8, 3.5, 3.9, 3.7}, 0.0},
        {102, "Bob Smith",       {3.2, 3.0, 3.4, 3.1}, 0.0},
        {103, "Carol Williams",  {4.0, 3.9, 4.0, 3.8}, 0.0},
        {104, "David Brown",     {2.8, 3.1, 2.9, 3.0}, 0.0},
        {105, "Eve Davis",       {3.6, 3.7, 3.5, 3.8}, 0.0},
    };

    /* Calculate GPA for each student */
    for (int i = 0; i < MAX_STUDENTS; i++) {
        students[i].gpa = calculate_gpa(students[i].grades, NUM_GRADES);
    }

    printf("Student Records:\n");
    for (int i = 0; i < MAX_STUDENTS; i++) {
        print_student(&students[i]);
    }

    const Student *top = find_top_student(students, MAX_STUDENTS);
    printf("\nTop student: %s (GPA: %.2f)\n", top->name, top->gpa);

    /* Struct size and padding */
    printf("\nsizeof(Student) = %zu bytes\n", sizeof(Student));
    printf("(may include padding for alignment)\n");
}

/* === Exercise 2: Tagged Union Implementation === */
/* Problem: Implement a tagged union that can hold different value types. */

typedef enum {
    VAL_INT,
    VAL_DOUBLE,
    VAL_STRING,
    VAL_BOOL
} ValueType;

typedef struct {
    ValueType type;
    union {
        int i;
        double d;
        char s[64];
        int b;  /* bool as int for C99 compat */
    } data;
} Value;

Value make_int(int val) {
    Value v = { .type = VAL_INT, .data.i = val };
    return v;
}

Value make_double(double val) {
    Value v = { .type = VAL_DOUBLE, .data.d = val };
    return v;
}

Value make_string(const char *val) {
    Value v = { .type = VAL_STRING };
    strncpy(v.data.s, val, sizeof(v.data.s) - 1);
    v.data.s[sizeof(v.data.s) - 1] = '\0';
    return v;
}

Value make_bool(int val) {
    Value v = { .type = VAL_BOOL, .data.b = val != 0 };
    return v;
}

void print_value(const Value *v) {
    switch (v->type) {
        case VAL_INT:    printf("int(%d)", v->data.i);       break;
        case VAL_DOUBLE: printf("double(%.4f)", v->data.d);  break;
        case VAL_STRING: printf("string(\"%s\")", v->data.s); break;
        case VAL_BOOL:   printf("bool(%s)", v->data.b ? "true" : "false"); break;
    }
}

/* Type-safe addition: only works for numeric types */
int value_add(const Value *a, const Value *b, Value *result) {
    if (a->type == VAL_INT && b->type == VAL_INT) {
        *result = make_int(a->data.i + b->data.i);
        return 1;
    }
    if ((a->type == VAL_INT || a->type == VAL_DOUBLE) &&
        (b->type == VAL_INT || b->type == VAL_DOUBLE)) {
        double va = (a->type == VAL_INT) ? a->data.i : a->data.d;
        double vb = (b->type == VAL_INT) ? b->data.i : b->data.d;
        *result = make_double(va + vb);
        return 1;
    }
    return 0;  /* incompatible types */
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Tagged Union Implementation ===\n");

    Value values[] = {
        make_int(42),
        make_double(3.14159),
        make_string("Hello, C!"),
        make_bool(1)
    };
    int n = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < n; i++) {
        printf("  values[%d] = ", i);
        print_value(&values[i]);
        printf("\n");
    }

    /* Type-safe addition */
    Value result;
    Value a = make_int(10), b = make_double(2.5);
    if (value_add(&a, &b, &result)) {
        printf("\n  ");
        print_value(&a);
        printf(" + ");
        print_value(&b);
        printf(" = ");
        print_value(&result);
        printf("\n");
    }

    /* Show size savings */
    printf("\nsizeof(Value) = %zu bytes\n", sizeof(Value));
    printf("(union shares memory: max member size + tag)\n");
}

int main(void) {
    exercise_1();
    exercise_2();

    printf("\nAll exercises completed!\n");
    return 0;
}
