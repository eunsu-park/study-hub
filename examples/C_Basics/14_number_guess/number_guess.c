// number_guess.c
// Number guessing game

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int secret, guess, attempts;
    int min = 1, max = 100;

    // Initialize random seed
    srand(time(NULL));

    printf("=== Number Guessing Game ===\n");
    printf("Guess a number between 1 and 100!\n\n");

    // Generate random number between 1 and 100
    secret = rand() % 100 + 1;
    attempts = 0;

    while (1) {
        printf("Enter a number (%d ~ %d): ", min, max);

        if (scanf("%d", &guess) != 1) {
            printf("Please enter a valid number.\n");
            while (getchar() != '\n');  // Clear input buffer
            continue;
        }

        attempts++;

        if (guess < min || guess > max) {
            printf("Please enter a number within the range!\n");
            continue;
        }

        if (guess == secret) {
            printf("\nCorrect!\n");
            printf("You guessed it in %d attempts.\n", attempts);
            break;
        } else if (guess < secret) {
            printf("The number is higher.\n");
            if (guess > min) min = guess + 1;
        } else {
            printf("The number is lower.\n");
            if (guess < max) max = guess - 1;
        }
    }

    return 0;
}
