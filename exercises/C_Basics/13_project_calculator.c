/*
 * Exercises for Lesson 03: Project Calculator
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex03 03_project_calculator.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

/* === Exercise 1: Scientific Calculator Operations === */
/* Problem: Implement trig, logarithmic, and power functions with proper error handling. */
void exercise_1(void) {
    printf("=== Exercise 1: Scientific Calculator Operations ===\n");

    /*
     * Scientific calculators need careful domain checking:
     * - sqrt(x) requires x >= 0
     * - log(x) requires x > 0
     * - asin(x), acos(x) require -1 <= x <= 1
     * - Division by zero must be caught
     */

    /* Trigonometric functions (input in degrees, converted to radians) */
    double angles[] = {0, 30, 45, 60, 90, 180, 270, 360};
    int n_angles = (int)(sizeof(angles) / sizeof(angles[0]));

    printf("Trigonometric functions (degrees -> result):\n");
    printf("%-8s  %-12s  %-12s  %-12s\n", "Angle", "sin", "cos", "tan");
    printf("--------  ------------  ------------  ------------\n");

    for (int i = 0; i < n_angles; i++) {
        double rad = angles[i] * M_PI / 180.0;
        double s = sin(rad);
        double c = cos(rad);

        printf("%-8.0f  %-12.6f  %-12.6f  ", angles[i], s, c);

        /* tan is undefined when cos is zero (90, 270 degrees) */
        if (fabs(c) < 1e-10) {
            printf("UNDEFINED\n");
        } else {
            printf("%-12.6f\n", tan(rad));
        }
    }

    /* Logarithmic functions with domain checking */
    printf("\nLogarithmic functions:\n");
    double log_inputs[] = {0.01, 0.1, 1.0, 2.718281828, 10.0, 100.0, -1.0, 0.0};
    int n_logs = (int)(sizeof(log_inputs) / sizeof(log_inputs[0]));

    for (int i = 0; i < n_logs; i++) {
        double x = log_inputs[i];
        if (x <= 0) {
            printf("  log(%.2f)  = DOMAIN ERROR (x must be > 0)\n", x);
        } else {
            printf("  ln(%.4f) = %-10.6f  log10(%.4f) = %-10.6f\n",
                   x, log(x), x, log10(x));
        }
    }

    /* Power and root operations */
    printf("\nPower and root operations:\n");
    printf("  2^10     = %.0f\n", pow(2, 10));
    printf("  sqrt(2)  = %.10f\n", sqrt(2));
    printf("  cbrt(27) = %.0f\n", cbrt(27));
    printf("  sqrt(-1) = DOMAIN ERROR (negative input)\n");
}

/* === Exercise 2: Input Validation === */
/* Problem: Validate numeric input from a string, handling edge cases. */

typedef enum { INPUT_OK, INPUT_EMPTY, INPUT_OVERFLOW, INPUT_INVALID } InputStatus;

InputStatus validate_number(const char *str, double *result) {
    /*
     * Robust input validation strategy:
     * 1. Skip leading whitespace
     * 2. Check for empty input
     * 3. Use strtod() which sets endptr to the first invalid character
     * 4. Check for partial parses (e.g., "12abc")
     * 5. Check for overflow (HUGE_VAL)
     */
    if (!str || *str == '\0') return INPUT_EMPTY;

    /* Skip whitespace */
    while (isspace((unsigned char)*str)) str++;
    if (*str == '\0') return INPUT_EMPTY;

    char *endptr;
    *result = strtod(str, &endptr);

    /* Skip trailing whitespace */
    while (isspace((unsigned char)*endptr)) endptr++;

    /* Check if entire string was consumed */
    if (*endptr != '\0') return INPUT_INVALID;

    /* Check for overflow */
    if (*result == HUGE_VAL || *result == -HUGE_VAL) return INPUT_OVERFLOW;

    return INPUT_OK;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Input Validation ===\n");

    const char *test_inputs[] = {
        "42",        "3.14",      "-7.5",      "  123  ",
        "",          "   ",       "12abc",     "abc",
        "1e308",     "1e309",     "0.001",     "--5",
        ".",         "1.2.3",     "+42",       "  -0.0  "
    };
    int n_tests = (int)(sizeof(test_inputs) / sizeof(test_inputs[0]));

    const char *status_names[] = {"OK", "EMPTY", "OVERFLOW", "INVALID"};

    printf("%-12s  %-10s  %-12s\n", "Input", "Status", "Value");
    printf("------------  ----------  ------------\n");

    for (int i = 0; i < n_tests; i++) {
        double value = 0;
        InputStatus status = validate_number(test_inputs[i], &value);

        printf("\"%-10s\"  %-10s", test_inputs[i], status_names[status]);
        if (status == INPUT_OK) {
            printf("  %.6g\n", value);
        } else {
            printf("  ---\n");
        }
    }

    /*
     * Pitfall: atof() and scanf("%lf") have poor error detection.
     * - atof("abc") returns 0.0 with no error indication
     * - scanf returns count of matched items but can't detect partial matches
     * - strtod() with endptr is the gold standard for C numeric parsing
     */
}

/* === Exercise 3: Simple Expression Parser === */
/* Problem: Parse and evaluate "num op num" expressions like "3 + 4". */

typedef struct {
    double left;
    char op;
    double right;
    int valid;
} Expression;

Expression parse_expression(const char *input) {
    /*
     * Simple two-operand expression parser:
     * Format: <number> <operator> <number>
     * Supported operators: + - * / % ^
     *
     * This is a "recursive descent" parser in its simplest form --
     * just one production rule. Real expression parsers handle
     * operator precedence and parentheses (see compiler courses).
     */
    Expression expr = {0, 0, 0, 0};
    char *end1;

    expr.left = strtod(input, &end1);
    if (end1 == input) return expr; /* No number found */

    /* Skip whitespace to find operator */
    while (isspace((unsigned char)*end1)) end1++;
    if (*end1 == '\0') return expr;

    expr.op = *end1;
    end1++;

    char *end2;
    expr.right = strtod(end1, &end2);
    if (end2 == end1) return expr; /* No second number */

    /* Check trailing characters */
    while (isspace((unsigned char)*end2)) end2++;
    if (*end2 != '\0') return expr; /* Garbage after expression */

    expr.valid = 1;
    return expr;
}

double evaluate(Expression expr, int *error) {
    *error = 0;
    switch (expr.op) {
        case '+': return expr.left + expr.right;
        case '-': return expr.left - expr.right;
        case '*': return expr.left * expr.right;
        case '/':
            if (fabs(expr.right) < 1e-15) { *error = 1; return 0; }
            return expr.left / expr.right;
        case '%':
            if (fabs(expr.right) < 1e-15) { *error = 1; return 0; }
            return fmod(expr.left, expr.right);
        case '^': return pow(expr.left, expr.right);
        default: *error = 2; return 0;
    }
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Simple Expression Parser ===\n");

    const char *expressions[] = {
        "3 + 4",     "10 - 7",    "6 * 8",     "15 / 4",
        "17 % 5",    "2 ^ 10",    "10 / 0",    "5.5 + 2.3",
        "-3 + 7",    "abc",       "3 +",       "100 ^ 0.5"
    };
    int n_exprs = (int)(sizeof(expressions) / sizeof(expressions[0]));

    printf("%-14s  %-10s\n", "Expression", "Result");
    printf("--------------  ----------\n");

    for (int i = 0; i < n_exprs; i++) {
        Expression expr = parse_expression(expressions[i]);
        if (!expr.valid) {
            printf("%-14s  PARSE ERROR\n", expressions[i]);
            continue;
        }

        int error;
        double result = evaluate(expr, &error);
        if (error == 1) {
            printf("%-14s  DIVISION BY ZERO\n", expressions[i]);
        } else if (error == 2) {
            printf("%-14s  UNKNOWN OPERATOR '%c'\n", expressions[i], expr.op);
        } else {
            printf("%-14s  = %.4g\n", expressions[i], result);
        }
    }
}

/* === Exercise 4: Calculation History === */
/* Problem: Maintain a circular buffer of recent calculations. */

#define HISTORY_SIZE 5

typedef struct {
    char expression[64];
    double result;
} HistoryEntry;

typedef struct {
    HistoryEntry entries[HISTORY_SIZE];
    int head;   /* Next write position */
    int count;  /* Number of valid entries (max HISTORY_SIZE) */
} History;

void history_init(History *h) {
    h->head = 0;
    h->count = 0;
    memset(h->entries, 0, sizeof(h->entries));
}

void history_add(History *h, const char *expr, double result) {
    /*
     * Circular buffer: when buffer is full, oldest entry is overwritten.
     * - head always points to the next write position
     * - After write, head advances: (head + 1) % HISTORY_SIZE
     * - count caps at HISTORY_SIZE
     *
     * Time complexity: O(1) for add, O(n) for display
     * Space complexity: O(HISTORY_SIZE) fixed
     */
    snprintf(h->entries[h->head].expression, 64, "%s", expr);
    h->entries[h->head].result = result;
    h->head = (h->head + 1) % HISTORY_SIZE;
    if (h->count < HISTORY_SIZE) h->count++;
}

void history_display(const History *h) {
    if (h->count == 0) {
        printf("  (empty)\n");
        return;
    }

    /* Start from the oldest entry */
    int start = (h->head - h->count + HISTORY_SIZE) % HISTORY_SIZE;
    for (int i = 0; i < h->count; i++) {
        int idx = (start + i) % HISTORY_SIZE;
        printf("  [%d] %s = %.4g\n", i + 1,
               h->entries[idx].expression, h->entries[idx].result);
    }
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Calculation History ===\n");

    History hist;
    history_init(&hist);

    /* Add 7 entries to a size-5 buffer to show circular overwrite */
    const char *calcs[] = {"1+1", "2*3", "10/4", "5^2", "7-3", "8+9", "4*5"};
    double results[]    = { 2,     6,     2.5,    25,    4,     17,    20  };
    int n_calcs = 7;

    for (int i = 0; i < n_calcs; i++) {
        history_add(&hist, calcs[i], results[i]);
        printf("After adding '%s = %.0f' (count=%d):\n",
               calcs[i], results[i], hist.count);
        history_display(&hist);
        printf("\n");
    }

    printf("Note: Entries '1+1' and '2*3' were overwritten when buffer wrapped.\n");
}

/* === Exercise 5: Unit Converter === */
/* Problem: Convert between related units using conversion factors. */

typedef struct {
    const char *from;
    const char *to;
    double factor;  /* to = from * factor */
} Conversion;

void exercise_5(void) {
    printf("\n=== Exercise 5: Unit Converter ===\n");

    /*
     * Strategy: Store conversion factors in a table.
     * To convert A -> B: result = value * factor
     * To convert B -> A: result = value / factor
     *
     * For a production calculator, you'd use a graph-based approach
     * where conversions chain through intermediate units.
     */
    Conversion conversions[] = {
        {"km",  "miles",  0.621371},
        {"kg",  "lbs",    2.20462},
        {"C",   "F",      0},          /* Special: F = C * 9/5 + 32 */
        {"m",   "ft",     3.28084},
        {"L",   "gal",    0.264172},
        {"cm",  "in",     0.393701},
    };
    int n_conv = (int)(sizeof(conversions) / sizeof(conversions[0]));

    double test_values[] = {1.0, 10.0, 100.0, 0.0, -40.0};
    int n_vals = (int)(sizeof(test_values) / sizeof(test_values[0]));

    for (int c = 0; c < n_conv; c++) {
        printf("\n%s -> %s conversions:\n", conversions[c].from, conversions[c].to);

        for (int v = 0; v < n_vals; v++) {
            double val = test_values[v];
            double result;

            /* Temperature conversion is not a simple multiplication */
            if (strcmp(conversions[c].from, "C") == 0) {
                result = val * 9.0 / 5.0 + 32.0;
            } else {
                result = val * conversions[c].factor;
            }

            printf("  %8.2f %-5s = %8.2f %-5s\n",
                   val, conversions[c].from, result, conversions[c].to);
        }
    }

    /*
     * Edge case: -40 is the same in both Celsius and Fahrenheit.
     * This is a useful sanity check for temperature conversion code.
     */
    printf("\nSanity check: -40C = %.0fF (should be -40)\n",
           -40.0 * 9.0 / 5.0 + 32.0);
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
