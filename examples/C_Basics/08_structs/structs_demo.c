/*
 * structs_demo.c — Struct, typedef, union, and enum examples.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o structs_demo structs_demo.c
 * Run:     ./structs_demo
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

/* Basic struct */
struct Point {
    double x;
    double y;
};

/* typedef for convenience */
typedef struct {
    char   name[50];
    int    age;
    double gpa;
} Student;

/* Enum for readable constants */
typedef enum {
    SHAPE_CIRCLE,
    SHAPE_RECTANGLE,
    SHAPE_TRIANGLE
} ShapeType;

/* Union: shared memory for different shape data */
typedef union {
    struct { double radius; }              circle;
    struct { double width, height; }       rect;
    struct { double base, height; }        tri;
} ShapeData;

/* Tagged union pattern */
typedef struct {
    ShapeType type;
    ShapeData data;
} Shape;

/* Function taking struct pointer */
double distance(const struct Point *a, const struct Point *b)
{
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    return sqrt(dx * dx + dy * dy);
}

double shape_area(const Shape *s)
{
    switch (s->type) {
        case SHAPE_CIRCLE:
            return M_PI * s->data.circle.radius * s->data.circle.radius;
        case SHAPE_RECTANGLE:
            return s->data.rect.width * s->data.rect.height;
        case SHAPE_TRIANGLE:
            return 0.5 * s->data.tri.base * s->data.tri.height;
    }
    return 0.0;
}

int main(void)
{
    /* Struct initialization and access */
    printf("=== Struct ===\n");
    struct Point p1 = {3.0, 4.0};
    struct Point p2 = {.x = 0.0, .y = 0.0};  /* designated initializer */
    printf("p1 = (%.1f, %.1f)\n", p1.x, p1.y);
    printf("p2 = (%.1f, %.1f)\n", p2.x, p2.y);
    printf("distance = %.2f\n", distance(&p1, &p2));

    /* typedef struct */
    printf("\n=== typedef Struct ===\n");
    Student s;
    strcpy(s.name, "Alice");
    s.age = 20;
    s.gpa = 3.85;
    printf("Student: %s, age %d, GPA %.2f\n", s.name, s.age, s.gpa);

    /* Enum */
    printf("\n=== Enum ===\n");
    ShapeType t = SHAPE_CIRCLE;
    printf("SHAPE_CIRCLE = %d, SHAPE_RECTANGLE = %d, SHAPE_TRIANGLE = %d\n",
           SHAPE_CIRCLE, SHAPE_RECTANGLE, SHAPE_TRIANGLE);
    printf("t = %d\n", t);

    /* Tagged union */
    printf("\n=== Tagged Union (Shape areas) ===\n");
    Shape shapes[3] = {
        { SHAPE_CIRCLE,    { .circle = {5.0} } },
        { SHAPE_RECTANGLE, { .rect   = {4.0, 6.0} } },
        { SHAPE_TRIANGLE,  { .tri    = {3.0, 8.0} } }
    };
    const char *names[] = {"Circle", "Rectangle", "Triangle"};
    for (int i = 0; i < 3; i++)
        printf("  %s area = %.2f\n", names[i], shape_area(&shapes[i]));

    /* Struct size and padding */
    printf("\n=== Sizes ===\n");
    printf("sizeof(struct Point) = %zu\n", sizeof(struct Point));
    printf("sizeof(Student)      = %zu\n", sizeof(Student));
    printf("sizeof(ShapeData)    = %zu (union: largest member)\n", sizeof(ShapeData));

    return 0;
}
