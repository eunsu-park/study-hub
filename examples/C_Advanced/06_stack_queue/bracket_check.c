// bracket_check.c
// Bracket validation program using a stack
// Checks whether opening and closing brackets are properly paired

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define MAX_SIZE 100

// Stack that stores characters
typedef struct {
    char data[MAX_SIZE];
    int top;
} CharStack;

// Initialize stack
void stack_init(CharStack *s) {
    s->top = -1;
}

// Check if empty
bool stack_isEmpty(CharStack *s) {
    return s->top == -1;
}

// Check if full
bool stack_isFull(CharStack *s) {
    return s->top == MAX_SIZE - 1;
}

// Push
void stack_push(CharStack *s, char c) {
    if (!stack_isFull(s)) {
        s->data[++s->top] = c;
    }
}

// Pop
char stack_pop(CharStack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top--];
    }
    return '\0';
}

// Peek
char stack_peek(CharStack *s) {
    if (!stack_isEmpty(s)) {
        return s->data[s->top];
    }
    return '\0';
}

// Check if character is an opening bracket
bool isOpeningBracket(char c) {
    return c == '(' || c == '{' || c == '[';
}

// Check if character is a closing bracket
bool isClosingBracket(char c) {
    return c == ')' || c == '}' || c == ']';
}

// Check if brackets form a matching pair
bool isMatchingPair(char open, char close) {
    return (open == '(' && close == ')') ||
           (open == '{' && close == '}') ||
           (open == '[' && close == ']');
}

// Bracket validation function
bool checkBrackets(const char *expr) {
    CharStack s;
    stack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        // Opening bracket: push onto stack
        if (isOpeningBracket(c)) {
            stack_push(&s, c);
            printf("  [Position %d] '%c' opening bracket -> push to stack\n", i, c);
        }
        // Closing bracket: pop from stack and check pair
        else if (isClosingBracket(c)) {
            if (stack_isEmpty(&s)) {
                printf("  [Position %d] '%c' error - unmatched closing bracket\n", i, c);
                return false;
            }

            char open = stack_pop(&s);
            if (!isMatchingPair(open, c)) {
                printf("  [Position %d] error - '%c' and '%c' mismatch\n", i, open, c);
                return false;
            }
            printf("  [Position %d] '%c' closing bracket -> matched with '%c' OK\n", i, c, open);
        }
    }

    // If stack is not empty, there are unmatched opening brackets remaining
    if (!stack_isEmpty(&s)) {
        printf("  Error - %d unclosed opening bracket(s) remaining\n", s.top + 1);
        return false;
    }

    return true;
}

// Simple bracket check (no debug output)
bool checkBracketsQuiet(const char *expr) {
    CharStack s;
    stack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        if (isOpeningBracket(c)) {
            stack_push(&s, c);
        } else if (isClosingBracket(c)) {
            if (stack_isEmpty(&s)) {
                return false;
            }
            char open = stack_pop(&s);
            if (!isMatchingPair(open, c)) {
                return false;
            }
        }
    }

    return stack_isEmpty(&s);
}

// Test code
int main(void) {
    printf("=== Bracket Validation Program ===\n\n");

    const char *tests[] = {
        "(a + b) * (c - d)",     // Valid brackets
        "((a + b) * c",          // Unclosed bracket
        "{[()]}",                // Valid nesting
        "{[(])}",                // Invalid nesting
        "((()))",                // Valid brackets
        ")(",                    // Invalid order
        "{[a + (b * c)] - d}",   // Valid complex expression
        "((a + b)",              // One short
        "a + b)",                // No opening bracket
        "[]{}()"                 // Valid consecutive
    };

    int n = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n; i++) {
        printf("Test %d: \"%s\"\n", i + 1, tests[i]);

        if (checkBrackets(tests[i])) {
            printf("Result: Valid brackets\n");
        } else {
            printf("Result: Invalid brackets\n");
        }
        printf("\n");
    }

    // Additional test: user input
    printf("\n=== Try It Yourself ===\n");
    char input[MAX_SIZE];
    printf("Enter an expression with brackets (quit: q): ");

    while (fgets(input, MAX_SIZE, stdin)) {
        // Remove newline character
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "q") == 0) {
            break;
        }

        if (checkBracketsQuiet(input)) {
            printf("Valid brackets!\n");
        } else {
            printf("Invalid brackets!\n");
        }

        printf("\nEnter another expression (quit: q): ");
    }

    printf("\nExiting the program.\n");
    return 0;
}
