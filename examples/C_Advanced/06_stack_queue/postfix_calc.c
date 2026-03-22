// postfix_calc.c
// Postfix notation calculator using a stack
// Infix: (3 + 4) * 5  ->  Postfix: 3 4 + 5 *

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

#define MAX_SIZE 100

// Double stack (results may be floating-point)
typedef struct {
    double data[MAX_SIZE];
    int top;
} Stack;

// Initialize stack
void stack_init(Stack *s) {
    s->top = -1;
}

// Check if empty
bool stack_isEmpty(Stack *s) {
    return s->top == -1;
}

// Push
void stack_push(Stack *s, double v) {
    if (s->top < MAX_SIZE - 1) {
        s->data[++s->top] = v;
    }
}

// Pop
double stack_pop(Stack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top--];
    }
    return 0.0;
}

// Peek
double stack_peek(Stack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top];
    }
    return 0.0;
}

// Check if character is an operator
bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '%';
}

// Perform operation
double applyOperator(double a, double b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) {
                printf("Error: Cannot divide by zero!\n");
                return 0;
            }
            return a / b;
        case '%':
            if (b == 0) {
                printf("Error: Cannot divide by zero!\n");
                return 0;
            }
            return (int)a % (int)b;
        default:
            printf("Unknown operator: %c\n", op);
            return 0;
    }
}

// Evaluate postfix expression
// Space-separated token format: "3 4 + 5 *"
double evaluatePostfix(const char *expr) {
    Stack s;
    stack_init(&s);

    // Copy input string (strtok modifies the original)
    char *str = strdup(expr);
    char *token = strtok(str, " ");

    printf("Evaluation steps:\n");

    while (token) {
        // Check if number (also handles negatives)
        if (isdigit(token[0]) || (token[0] == '-' && strlen(token) > 1)) {
            double num = atof(token);
            stack_push(&s, num);
            printf("  Push number %g onto stack\n", num);
        }
        // Operator
        else if (isOperator(token[0]) && strlen(token) == 1) {
            if (s.top < 1) {
                printf("Error: Not enough operands for operator '%c'!\n", token[0]);
                free(str);
                return 0;
            }

            double b = stack_pop(&s);
            double a = stack_pop(&s);
            double result = applyOperator(a, b, token[0]);

            printf("  Operation: %g %c %g = %g\n", a, token[0], b, result);
            stack_push(&s, result);
        }
        else {
            printf("Error: Invalid token '%s'\n", token);
            free(str);
            return 0;
        }

        token = strtok(NULL, " ");
    }

    free(str);

    // Stack should contain exactly one result
    if (s.top != 0) {
        printf("Error: Invalid notation (%d values remaining on stack)\n", s.top + 1);
        return 0;
    }

    return stack_pop(&s);
}

// Return operator precedence
int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/' || op == '%') return 2;
    return 0;
}

// Convert infix notation to postfix notation (advanced)
// Simple implementation: handles parentheses and operator precedence
void infixToPostfix(const char *infix, char *postfix) {
    Stack s;
    stack_init(&s);
    int j = 0;

    for (int i = 0; infix[i]; i++) {
        char c = infix[i];

        // Skip spaces
        if (c == ' ') continue;

        // Operand (number or variable)
        if (isalnum(c)) {
            postfix[j++] = c;
            postfix[j++] = ' ';
        }
        // Opening parenthesis
        else if (c == '(') {
            stack_push(&s, c);
        }
        // Closing parenthesis
        else if (c == ')') {
            while (!stack_isEmpty(&s) && (char)stack_peek(&s) != '(') {
                postfix[j++] = (char)stack_pop(&s);
                postfix[j++] = ' ';
            }
            stack_pop(&s);  // Remove '('
        }
        // Operator
        else if (isOperator(c)) {
            while (!stack_isEmpty(&s) &&
                   precedence((char)stack_peek(&s)) >= precedence(c)) {
                postfix[j++] = (char)stack_pop(&s);
                postfix[j++] = ' ';
            }
            stack_push(&s, c);
        }
    }

    // Output all remaining operators from the stack
    while (!stack_isEmpty(&s)) {
        postfix[j++] = (char)stack_pop(&s);
        postfix[j++] = ' ';
    }

    postfix[j] = '\0';
}

// Test code
int main(void) {
    printf("=== Postfix Notation Calculator ===\n\n");

    // Postfix evaluation test
    const char *postfixExpressions[] = {
        "3 4 +",                  // 3 + 4 = 7
        "3 4 + 5 *",              // (3 + 4) * 5 = 35
        "10 2 / 3 +",             // 10 / 2 + 3 = 8
        "5 1 2 + 4 * + 3 -",      // 5 + ((1 + 2) * 4) - 3 = 14
        "15 7 1 1 + - / 3 * 2 1 1 + + -",  // Complex expression
        "8 2 /",                  // 8 / 2 = 4
        "9 3 % 2 *"               // (9 % 3) * 2 = 0
    };

    int n = sizeof(postfixExpressions) / sizeof(postfixExpressions[0]);

    printf("[ Postfix Evaluation ]\n");
    for (int i = 0; i < n; i++) {
        printf("\n%d. Expression: %s\n", i + 1, postfixExpressions[i]);
        double result = evaluatePostfix(postfixExpressions[i]);
        printf("   Final result: %.2f\n", result);
    }

    // Infix to postfix conversion test
    printf("\n\n[ Infix to Postfix Conversion ]\n");

    const char *infixExpressions[] = {
        "(3 + 4) * 5",
        "a + b * c",
        "(a + b) * (c - d)",
        "a + b - c",
        "a * b + c / d"
    };

    int m = sizeof(infixExpressions) / sizeof(infixExpressions[0]);

    for (int i = 0; i < m; i++) {
        char postfix[MAX_SIZE];
        infixToPostfix(infixExpressions[i], postfix);
        printf("\nInfix:   %s\n", infixExpressions[i]);
        printf("Postfix: %s\n", postfix);
    }

    // Interactive calculator
    printf("\n\n[ Interactive Postfix Calculator ]\n");
    printf("Enter an expression in postfix notation (e.g., 3 4 +)\n");
    printf("Enter 'q' to quit.\n\n");

    char input[MAX_SIZE];
    while (1) {
        printf("> ");
        if (!fgets(input, MAX_SIZE, stdin)) break;

        // Remove newline character
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "q") == 0) break;

        double result = evaluatePostfix(input);
        printf("Result: %.2f\n\n", result);
    }

    printf("Exiting the program.\n");
    return 0;
}
