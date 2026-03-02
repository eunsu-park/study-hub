// calculator.c
// Basic calculator program

#include <stdio.h>
#include <stdlib.h>

// Addition
double add(double a, double b) {
    return a + b;
}

// Subtraction
double subtract(double a, double b) {
    return a - b;
}

// Multiplication
double multiply(double a, double b) {
    return a * b;
}

// Division
double divide(double a, double b) {
    if (b == 0) {
        printf("Error: Cannot divide by zero.\n");
        return 0;
    }
    return a / b;
}

// Print menu
void print_menu(void) {
    printf("\n=== Calculator ===\n");
    printf("1. Addition (+)\n");
    printf("2. Subtraction (-)\n");
    printf("3. Multiplication (*)\n");
    printf("4. Division (/)\n");
    printf("5. Quit\n");
    printf("Choice: ");
}

int main(void) {
    int choice;
    double num1, num2, result;

    printf("Simple Calculator Program\n");

    while (1) {
        print_menu();

        if (scanf("%d", &choice) != 1) {
            printf("Invalid input.\n");
            // Clear input buffer
            while (getchar() != '\n');
            continue;
        }

        if (choice == 5) {
            printf("Exiting the program.\n");
            break;
        }

        if (choice < 1 || choice > 5) {
            printf("Please enter a number between 1 and 5.\n");
            continue;
        }

        // Enter two numbers
        printf("First number: ");
        if (scanf("%lf", &num1) != 1) {
            printf("Invalid input.\n");
            while (getchar() != '\n');
            continue;
        }

        printf("Second number: ");
        if (scanf("%lf", &num2) != 1) {
            printf("Invalid input.\n");
            while (getchar() != '\n');
            continue;
        }

        // Perform calculation
        switch (choice) {
            case 1:
                result = add(num1, num2);
                printf("%.2lf + %.2lf = %.2lf\n", num1, num2, result);
                break;
            case 2:
                result = subtract(num1, num2);
                printf("%.2lf - %.2lf = %.2lf\n", num1, num2, result);
                break;
            case 3:
                result = multiply(num1, num2);
                printf("%.2lf * %.2lf = %.2lf\n", num1, num2, result);
                break;
            case 4:
                result = divide(num1, num2);
                if (num2 != 0) {
                    printf("%.2lf / %.2lf = %.2lf\n", num1, num2, result);
                }
                break;
        }
    }

    return 0;
}
