# Project 2: Number Guessing Game

**Previous**: [Project 1: Basic Arithmetic Calculator](./03_Project_Calculator.md) | **Next**: [Project 3: Address Book Program](./05_Project_Address_Book.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement pseudo-random number generation using `rand`, `srand`, and `time` with a configurable range
2. Build game loops with `while` and `do-while` that repeat until a win, loss, or quit condition
3. Apply conditional statements to provide directional hints and proximity feedback
4. Design a difficulty system that adjusts number range and attempt limits via `switch-case`
5. Implement input validation that rejects out-of-range or non-numeric entries without crashing
6. Track session statistics using a struct and compute derived metrics like win rate and average attempts

---

Games are an excellent way to internalize loop and conditional logic because the feedback is immediate -- you see the program respond to every decision you make. In this project you will build a number-guessing game that seeds a random target, gives the player hints, tracks scores, and supports multiple difficulty levels, all while reinforcing the C fundamentals from the previous lessons.

## Game Rules

```
1. Computer selects a number between 1-100
2. Player guesses the number
3. "UP!" or "DOWN!" hints provided
4. Repeat until correct answer
5. Display number of attempts
```

---

## Step 1: Understanding Random Number Generation

### Core Syntax: rand() and srand()

```c
#include <stdio.h>
#include <stdlib.h>  // rand, srand
#include <time.h>    // time

int main(void) {
    // Set seed (call only once)
    // time(NULL): use current time (seconds) as seed
    srand(time(NULL));

    // Generate random number
    printf("%d\n", rand());  // Random number between 0 ~ RAND_MAX

    // Specify range: 1 ~ 100
    int num = rand() % 100 + 1;
    printf("Random number between 1~100: %d\n", num);

    // Range formula: rand() % (max - min + 1) + min
    // Example: 50~100 -> rand() % 51 + 50

    return 0;
}
```

### Why Seed Is Needed

```c
// Running without srand generates the same sequence of random numbers every time!
// srand(time(NULL)) uses current time as seed -> different random numbers each run
```

---

## Step 2: Basic Game Implementation

```c
// guess_v1.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    // Initialize random
    srand(time(NULL));

    // Generate answer between 1~100
    int answer = rand() % 100 + 1;
    int guess;
    int attempts = 0;

    printf("=== Number Guessing Game ===\n");
    printf("Guess a number between 1 and 100!\n\n");

    // Game loop
    while (1) {
        printf("Guess: ");
        scanf("%d", &guess);
        attempts++;

        if (guess < answer) {
            printf("UP! (The number is higher)\n\n");
        } else if (guess > answer) {
            printf("DOWN! (The number is lower)\n\n");
        } else {
            printf("\nCorrect!\n");
            printf("You got it in %d attempts!\n", attempts);
            break;
        }
    }

    return 0;
}
```

### Example Output

```
=== Number Guessing Game ===
Guess a number between 1 and 100!

Guess: 50
UP! (The number is higher)

Guess: 75
DOWN! (The number is lower)

Guess: 62
UP! (The number is higher)

Guess: 68
Correct!
You got it in 4 attempts!
```

---

## Step 3: Adding Features

### Additional Features

1. Attempt limit
2. Input validation
3. Restart functionality
4. Difficulty selection

```c
// guess_v2.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function declarations
void print_title(void);
int get_difficulty(void);
int play_game(int max_num, int max_attempts);
void clear_input_buffer(void);

int main(void) {
    char play_again;

    srand(time(NULL));
    print_title();

    do {
        int difficulty = get_difficulty();
        int max_num, max_attempts;

        // Difficulty settings
        switch (difficulty) {
            case 1:  // Easy
                max_num = 50;
                max_attempts = 10;
                break;
            case 2:  // Normal
                max_num = 100;
                max_attempts = 7;
                break;
            case 3:  // Hard
                max_num = 200;
                max_attempts = 8;
                break;
            default:
                max_num = 100;
                max_attempts = 7;
        }

        // Run game
        int result = play_game(max_num, max_attempts);

        if (result) {
            printf("\nCongratulations! You win!\n");
        } else {
            printf("\nToo bad. Try again next time!\n");
        }

        // Restart confirmation
        printf("\nPlay again? (y/n): ");
        scanf(" %c", &play_again);
        clear_input_buffer();
        printf("\n");

    } while (play_again == 'y' || play_again == 'Y');

    printf("Exiting game. Goodbye!\n");
    return 0;
}

void print_title(void) {
    printf("\n");
    printf("================================\n");
    printf("     Number Guessing Game v2    \n");
    printf("================================\n");
    printf("\n");
}

int get_difficulty(void) {
    int choice;

    printf("Select difficulty:\n");
    printf("  1. Easy   (1~50,  10 chances)\n");
    printf("  2. Normal (1~100, 7 chances)\n");
    printf("  3. Hard   (1~200, 8 chances)\n");
    printf("\nChoice: ");
    scanf("%d", &choice);
    clear_input_buffer();

    if (choice < 1 || choice > 3) {
        printf("Invalid choice. Starting with Normal difficulty.\n");
        choice = 2;
    }

    return choice;
}

int play_game(int max_num, int max_attempts) {
    int answer = rand() % max_num + 1;
    int guess;
    int attempts = 0;

    printf("\nGuess a number between 1 and %d!\n", max_num);
    printf("Chances: %d\n\n", max_attempts);

    while (attempts < max_attempts) {
        printf("[%d/%d] Guess: ", attempts + 1, max_attempts);

        if (scanf("%d", &guess) != 1) {
            printf("Please enter a number.\n\n");
            clear_input_buffer();
            continue;
        }

        // Range validation
        if (guess < 1 || guess > max_num) {
            printf("Please enter a number between 1~%d.\n\n", max_num);
            continue;
        }

        attempts++;

        if (guess < answer) {
            printf("UP!\n");
            // Additional hint
            if (answer - guess > max_num / 4) {
                printf("(Big difference)\n");
            }
            printf("\n");
        } else if (guess > answer) {
            printf("DOWN!\n");
            if (guess - answer > max_num / 4) {
                printf("(Big difference)\n");
            }
            printf("\n");
        } else {
            printf("\nCorrect!\n");
            printf("You got it in %d attempts!\n", attempts);

            // Calculate score
            int score = (max_attempts - attempts + 1) * 100;
            printf("Score: %d points\n", score);
            return 1;  // Win
        }
    }

    printf("\nOut of chances.\n");
    printf("The answer was %d.\n", answer);
    return 0;  // Lose
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## Step 4: Final Version (Advanced Features)

### Additional Features

- High score saving (within session)
- Statistics display
- Better UI

```c
// guess_game.c (Final)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Constant definitions
#define MAX_NAME_LEN 50

// Global variables (game statistics)
typedef struct {
    int games_played;
    int games_won;
    int best_score;
    int total_attempts;
    char best_player[MAX_NAME_LEN];
} GameStats;

// Function declarations
void print_title(void);
void print_menu(void);
int get_difficulty(int *max_num, int *max_attempts);
int play_game(int max_num, int max_attempts, GameStats *stats);
void show_stats(GameStats *stats);
void clear_input_buffer(void);

int main(void) {
    int choice;
    GameStats stats = {0, 0, 0, 0, ""};

    srand(time(NULL));

    while (1) {
        print_title();
        print_menu();

        printf("Choice: ");
        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();

        switch (choice) {
            case 1: {
                int max_num, max_attempts;
                get_difficulty(&max_num, &max_attempts);
                play_game(max_num, max_attempts, &stats);
                printf("\nPress Enter to continue...");
                getchar();
                break;
            }
            case 2:
                show_stats(&stats);
                printf("\nPress Enter to continue...");
                getchar();
                break;
            case 3:
                printf("\nExiting game. Goodbye!\n\n");
                return 0;
            default:
                printf("\nInvalid choice.\n");
        }
    }

    return 0;
}

void print_title(void) {
    printf("\n");
    printf("  =====================================\n");
    printf("  |                                   |\n");
    printf("  |      Number Guessing Game         |\n");
    printf("  |                                   |\n");
    printf("  =====================================\n");
    printf("\n");
}

void print_menu(void) {
    printf("  ---------------------------------\n");
    printf("  |  1. Start Game                |\n");
    printf("  |  2. View Statistics           |\n");
    printf("  |  3. Exit                      |\n");
    printf("  ---------------------------------\n");
    printf("\n");
}

int get_difficulty(int *max_num, int *max_attempts) {
    int choice;

    printf("\n  Select difficulty:\n\n");
    printf("    1. Easy     | 1~50   | 10 chances\n");
    printf("    2. Normal   | 1~100  | 7 chances\n");
    printf("    3. Hard     | 1~200  | 8 chances\n");
    printf("    4. Extreme  | 1~1000 | 10 chances\n");
    printf("\n  Choice: ");
    scanf("%d", &choice);
    clear_input_buffer();

    switch (choice) {
        case 1:
            *max_num = 50;
            *max_attempts = 10;
            break;
        case 2:
            *max_num = 100;
            *max_attempts = 7;
            break;
        case 3:
            *max_num = 200;
            *max_attempts = 8;
            break;
        case 4:
            *max_num = 1000;
            *max_attempts = 10;
            break;
        default:
            *max_num = 100;
            *max_attempts = 7;
    }

    return choice;
}

int play_game(int max_num, int max_attempts, GameStats *stats) {
    int answer = rand() % max_num + 1;
    int guess;
    int attempts = 0;
    int low = 1, high = max_num;  // Range for hints

    printf("\n  ----------------------------------\n");
    printf("  Guess a number between 1 and %d!\n", max_num);
    printf("  Chances: %d\n", max_attempts);
    printf("  ----------------------------------\n\n");

    while (attempts < max_attempts) {
        int remaining = max_attempts - attempts;
        printf("  [Remaining: %d] Current range: %d~%d\n", remaining, low, high);
        printf("  Guess: ");

        if (scanf("%d", &guess) != 1) {
            printf("  -> Please enter a number.\n\n");
            clear_input_buffer();
            continue;
        }

        if (guess < 1 || guess > max_num) {
            printf("  -> Please enter a number between 1~%d.\n\n", max_num);
            continue;
        }

        attempts++;
        stats->total_attempts++;

        if (guess < answer) {
            printf("  -> UP! (The number is higher)\n\n");
            if (guess > low) low = guess + 1;
        } else if (guess > answer) {
            printf("  -> DOWN! (The number is lower)\n\n");
            if (guess < high) high = guess - 1;
        } else {
            // Correct!
            int score = (max_attempts - attempts + 1) * 100 + (max_num / 10);

            printf("\n  *** Correct! ***\n\n");
            printf("  Attempts: %d\n", attempts);
            printf("  Score: %d points\n", score);

            stats->games_played++;
            stats->games_won++;

            if (score > stats->best_score) {
                stats->best_score = score;
                printf("\n  New high score!\n");
                printf("  Enter your name: ");
                scanf("%49s", stats->best_player);
                clear_input_buffer();
            }

            return 1;
        }
    }

    // Lose
    printf("\n  X Out of chances.\n");
    printf("  The answer was %d.\n", answer);

    stats->games_played++;
    return 0;
}

void show_stats(GameStats *stats) {
    printf("\n  ========== Game Statistics ==========\n\n");

    if (stats->games_played == 0) {
        printf("  No games played yet.\n");
        return;
    }

    printf("  Total games: %d\n", stats->games_played);
    printf("  Wins: %d\n", stats->games_won);
    printf("  Losses: %d\n", stats->games_played - stats->games_won);

    float win_rate = (float)stats->games_won / stats->games_played * 100;
    printf("  Win rate: %.1f%%\n", win_rate);

    float avg_attempts = (float)stats->total_attempts / stats->games_played;
    printf("  Average attempts: %.1f\n", avg_attempts);

    if (stats->best_score > 0) {
        printf("\n  High Score\n");
        printf("     Score: %d points\n", stats->best_score);
        printf("     Player: %s\n", stats->best_player);
    }

    printf("\n  =====================================\n");
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## Compile and Run

```bash
gcc -Wall -Wextra -std=c11 guess_game.c -o guess_game
./guess_game
```

---

## Summary

| Concept | Description |
|---------|-------------|
| `rand()` | Pseudo-random number generation |
| `srand(time(NULL))` | Seed initialization |
| `while (1)` | Infinite loop |
| `break` | Exit loop |
| `continue` | Skip to next iteration |
| Struct | Grouping related data |

---

## Exercises

1. **Binary Search AI**: Add a mode where the computer guesses the player's number
   - Hint: Always choose the middle value of the range

2. **Multiplayer**: Mode where two players take turns guessing

3. **File Saving**: Save and load high scores to/from a file

---

## Next Step

[Project 3: Address Book Program](./05_Project_Address_Book.md) -> Let's learn about structs and file I/O!
