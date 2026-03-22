/*
 * Exercises for Lesson 04: Project Number Guessing
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex04 04_project_number_guessing.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* === Exercise 1: Binary Search Strategy === */
/* Problem: Demonstrate the optimal guessing strategy using binary search. */
void exercise_1(void) {
    printf("=== Exercise 1: Binary Search Strategy ===\n");

    /*
     * Binary search is the optimal strategy for number guessing:
     * - Always guess the midpoint of the remaining range
     * - Each guess eliminates exactly half the possibilities
     * - For range [1, N], worst case is ceil(log2(N)) guesses
     *
     * Time complexity: O(log N) guesses guaranteed
     * This is provably optimal (information-theoretic lower bound)
     */

    int target = 73;  /* Secret number */
    int low = 1, high = 100;
    int guesses = 0;

    printf("Target: %d, Range: [%d, %d]\n\n", target, low, high);
    printf("%-6s  %-6s  %-6s  %-6s  %-10s\n",
           "Step", "Low", "High", "Guess", "Result");
    printf("------  ------  ------  ------  ----------\n");

    while (low <= high) {
        int guess = low + (high - low) / 2;  /* Avoid overflow vs (low+high)/2 */
        guesses++;

        if (guess == target) {
            printf("%-6d  %-6d  %-6d  %-6d  FOUND!\n",
                   guesses, low, high, guess);
            break;
        } else if (guess < target) {
            printf("%-6d  %-6d  %-6d  %-6d  Too low\n",
                   guesses, low, high, guess);
            low = guess + 1;
        } else {
            printf("%-6d  %-6d  %-6d  %-6d  Too high\n",
                   guesses, low, high, guess);
            high = guess - 1;
        }
    }

    printf("\nFound %d in %d guesses (max possible for [1,100] = 7)\n",
           target, guesses);

    /* Verify: ceil(log2(100)) = 7 */
    int max_guesses = 0;
    int n = 100;
    while (n > 0) { max_guesses++; n /= 2; }
    printf("Theoretical maximum: ceil(log2(100)) = %d\n", max_guesses);
}

/* === Exercise 2: Difficulty Levels === */
/* Problem: Implement configurable difficulty with different ranges and guess limits. */

typedef struct {
    const char *name;
    int min_val;
    int max_val;
    int max_guesses;
} Difficulty;

void exercise_2(void) {
    printf("\n=== Exercise 2: Difficulty Levels ===\n");

    /*
     * Difficulty design: max_guesses should be slightly above the
     * binary search optimum to give non-optimal players a chance,
     * but tight enough to be challenging.
     */
    Difficulty levels[] = {
        {"Easy",   1,   10,  5},
        {"Medium", 1,   100, 10},
        {"Hard",   1,  1000, 12},
        {"Expert", 1, 10000, 15},
    };
    int n_levels = (int)(sizeof(levels) / sizeof(levels[0]));

    printf("%-8s  %-12s  %-12s  %-12s\n",
           "Level", "Range", "Max Guesses", "Optimal (BS)");
    printf("--------  ------------  ------------  ------------\n");

    for (int i = 0; i < n_levels; i++) {
        /* Calculate optimal (binary search) guesses */
        int optimal = 0;
        int range = levels[i].max_val - levels[i].min_val + 1;
        int temp = range;
        while (temp > 0) { optimal++; temp /= 2; }

        printf("%-8s  [%d, %5d]   %-12d  %-12d\n",
               levels[i].name, levels[i].min_val, levels[i].max_val,
               levels[i].max_guesses, optimal);
    }

    /* Simulate a game on Medium difficulty */
    printf("\nSimulating Medium difficulty game:\n");
    srand((unsigned int)time(NULL));
    int target = (rand() % 100) + 1;
    int low = 1, high = 100;

    for (int g = 1; g <= levels[1].max_guesses; g++) {
        int guess = low + (high - low) / 2;
        if (guess == target) {
            printf("  Guess %2d: %3d -> Correct! Won in %d guesses\n", g, guess, g);
            break;
        } else if (guess < target) {
            printf("  Guess %2d: %3d -> Too low  (range now [%d, %d])\n",
                   g, guess, guess + 1, high);
            low = guess + 1;
        } else {
            printf("  Guess %2d: %3d -> Too high (range now [%d, %d])\n",
                   g, guess, low, guess - 1);
            high = guess - 1;
        }
    }
}

/* === Exercise 3: Statistics Tracking === */
/* Problem: Track game statistics across multiple rounds. */

typedef struct {
    int games_played;
    int games_won;
    int total_guesses;
    int best_score;    /* Fewest guesses to win */
    int worst_score;   /* Most guesses to win */
} GameStats;

void stats_init(GameStats *s) {
    s->games_played = 0;
    s->games_won = 0;
    s->total_guesses = 0;
    s->best_score = 9999;
    s->worst_score = 0;
}

void stats_record(GameStats *s, int guesses, int won) {
    s->games_played++;
    s->total_guesses += guesses;
    if (won) {
        s->games_won++;
        if (guesses < s->best_score) s->best_score = guesses;
        if (guesses > s->worst_score) s->worst_score = guesses;
    }
}

void stats_display(const GameStats *s) {
    printf("  Games played:  %d\n", s->games_played);
    printf("  Games won:     %d (%.1f%%)\n",
           s->games_won,
           s->games_played > 0 ? 100.0 * s->games_won / s->games_played : 0.0);
    printf("  Total guesses: %d\n", s->total_guesses);
    if (s->games_won > 0) {
        printf("  Avg guesses:   %.1f\n",
               (double)s->total_guesses / s->games_won);
        printf("  Best score:    %d guesses\n", s->best_score);
        printf("  Worst score:   %d guesses\n", s->worst_score);
    }
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Statistics Tracking ===\n");

    GameStats stats;
    stats_init(&stats);

    /*
     * Simulate 10 games using binary search with random targets.
     * This demonstrates both the stats tracking and the consistency
     * of binary search (always wins, 1-7 guesses for range [1,100]).
     */
    srand(42); /* Fixed seed for reproducible output */

    printf("Simulating 10 games (range [1,100], binary search):\n\n");
    printf("%-6s  %-8s  %-8s\n", "Game", "Target", "Guesses");
    printf("------  --------  --------\n");

    for (int game = 1; game <= 10; game++) {
        int target = (rand() % 100) + 1;
        int low = 1, high = 100, guesses = 0;

        while (low <= high) {
            int guess = low + (high - low) / 2;
            guesses++;
            if (guess == target) break;
            else if (guess < target) low = guess + 1;
            else high = guess - 1;
        }

        stats_record(&stats, guesses, 1);
        printf("%-6d  %-8d  %-8d\n", game, target, guesses);
    }

    printf("\nOverall Statistics:\n");
    stats_display(&stats);
}

/* === Exercise 4: Replay Logic and Game Loop === */
/* Problem: Implement a clean game loop with replay and state management. */

typedef enum { STATE_MENU, STATE_PLAYING, STATE_WON, STATE_LOST, STATE_QUIT } GameState;

void exercise_4(void) {
    printf("\n=== Exercise 4: Replay Logic and Game Loop ===\n");

    /*
     * State machine for game flow:
     *   MENU -> PLAYING -> WON/LOST -> MENU (replay) or QUIT
     *
     * Using an enum for states makes the game loop clean and
     * easy to extend (e.g., add PAUSED, SETTINGS states).
     */
    const char *state_names[] = {"MENU", "PLAYING", "WON", "LOST", "QUIT"};

    /* Simulate a game session with predefined "user actions" */
    GameState transitions[] = {
        STATE_MENU, STATE_PLAYING, STATE_PLAYING, STATE_PLAYING,
        STATE_WON, STATE_MENU, STATE_PLAYING, STATE_PLAYING,
        STATE_LOST, STATE_QUIT
    };
    int n_transitions = (int)(sizeof(transitions) / sizeof(transitions[0]));

    printf("Game state machine trace:\n\n");

    GameState current = STATE_MENU;
    for (int i = 0; i < n_transitions; i++) {
        GameState next = transitions[i];
        printf("  Step %2d: %-8s", i + 1, state_names[next]);

        switch (next) {
            case STATE_MENU:    printf(" -> Show menu, ask to play\n"); break;
            case STATE_PLAYING: printf(" -> Process guess\n"); break;
            case STATE_WON:     printf(" -> Display victory message\n"); break;
            case STATE_LOST:    printf(" -> Display game over\n"); break;
            case STATE_QUIT:    printf(" -> Clean up and exit\n"); break;
        }
        current = next;
    }
    (void)current; /* Suppress unused warning */

    printf("\nKey design insight: A state machine separates 'what to do' from\n");
    printf("'when to do it', making the game loop maintainable and testable.\n");
}

/* === Exercise 5: AI Guesser (Computer Guesses Your Number) === */
/* Problem: Implement computer as the guesser using binary search with feedback. */

typedef enum { CORRECT, TOO_LOW, TOO_HIGH } Feedback;

Feedback get_feedback(int guess, int secret) {
    if (guess == secret) return CORRECT;
    return guess < secret ? TOO_LOW : TOO_HIGH;
}

void exercise_5(void) {
    printf("\n=== Exercise 5: AI Guesser ===\n");

    /*
     * Role reversal: the computer guesses the user's number.
     * The AI uses binary search, which is optimal.
     *
     * A cheating detector can be added: if the user's feedback
     * contradicts previous feedback, they are lying.
     * E.g., user says "too low" for 50 then "too high" for 40.
     */

    int secrets[] = {1, 50, 100, 42, 73, 7, 99};
    int n_secrets = (int)(sizeof(secrets) / sizeof(secrets[0]));

    for (int s = 0; s < n_secrets; s++) {
        int secret = secrets[s];
        int low = 1, high = 100, guesses = 0;

        printf("AI guessing %d in [1,100]:\n", secret);

        while (low <= high) {
            int guess = low + (high - low) / 2;
            guesses++;
            Feedback fb = get_feedback(guess, secret);

            printf("  Guess %d: %d -> %s\n", guesses, guess,
                   fb == CORRECT ? "Correct!" :
                   fb == TOO_LOW ? "Too low" : "Too high");

            if (fb == CORRECT) break;
            else if (fb == TOO_LOW) low = guess + 1;
            else high = guess - 1;
        }
        printf("  Result: found in %d guesses\n\n", guesses);
    }

    /* Cheating detection example */
    printf("Cheat detection example:\n");
    printf("  If user says 50 is 'too low' and 40 is 'too high',\n");
    printf("  the ranges would be [51,100] then [51,39] -> contradiction!\n");
    printf("  Detect: if (low > high) printf(\"You're cheating!\");\n");
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
