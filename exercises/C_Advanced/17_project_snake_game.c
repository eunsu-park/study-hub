/*
 * Exercises for Lesson 11: Project Snake Game
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex11 11_project_snake_game.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* === Exercise 1: ANSI Cursor Control === */
/* Problem: Demonstrate ANSI escape codes for terminal manipulation. */
void exercise_1(void) {
    printf("=== Exercise 1: ANSI Cursor Control ===\n");

    /*
     * ANSI escape sequences start with ESC (0x1B or \033) followed by '['.
     * Common codes for terminal games:
     *
     * Cursor movement:
     *   \033[H       - Move cursor to home (0,0)
     *   \033[y;xH    - Move cursor to row y, column x
     *   \033[nA      - Move cursor up n lines
     *   \033[nB      - Move cursor down n lines
     *   \033[nC      - Move cursor right n columns
     *   \033[nD      - Move cursor left n columns
     *
     * Screen control:
     *   \033[2J      - Clear entire screen
     *   \033[K       - Clear from cursor to end of line
     *
     * Text formatting:
     *   \033[0m      - Reset all attributes
     *   \033[1m      - Bold
     *   \033[31m     - Red text
     *   \033[42m     - Green background
     *   \033[?25l    - Hide cursor
     *   \033[?25h    - Show cursor
     */

    printf("\nANSI Color Palette:\n");
    /* Standard colors (30-37 foreground, 40-47 background) */
    const char *color_names[] = {
        "Black", "Red", "Green", "Yellow", "Blue", "Magenta", "Cyan", "White"
    };
    for (int i = 0; i < 8; i++) {
        printf("  \033[%dm  %-8s \033[0m", 30 + i, color_names[i]);
        printf("  \033[%dm  %-8s \033[0m\n", 40 + i, color_names[i]);
    }

    /* Text attributes */
    printf("\nText Attributes:\n");
    printf("  \033[1mBold\033[0m  ");
    printf("  \033[4mUnderline\033[0m  ");
    printf("  \033[7mReverse\033[0m  ");
    printf("  \033[2mDim\033[0m\n");

    /* Draw a simple box using box-drawing characters */
    printf("\nBox drawing with positioning:\n");
    int w = 20, h = 5;
    for (int y = 0; y < h; y++) {
        printf("  ");
        for (int x = 0; x < w; x++) {
            if (y == 0 && x == 0) printf("+");
            else if (y == 0 && x == w - 1) printf("+");
            else if (y == h - 1 && x == 0) printf("+");
            else if (y == h - 1 && x == w - 1) printf("+");
            else if (y == 0 || y == h - 1) printf("-");
            else if (x == 0 || x == w - 1) printf("|");
            else printf(" ");
        }
        printf("\n");
    }

    printf("\nNote: ANSI escape codes work in most modern terminals.\n");
    printf("Windows cmd.exe requires enabling virtual terminal processing.\n");
}

/* === Exercise 2: Keyboard Input Handling === */
/* Problem: Demonstrate non-blocking input concepts for game loops. */
void exercise_2(void) {
    printf("\n=== Exercise 2: Keyboard Input Handling ===\n");

    /*
     * Terminal input modes (Unix/POSIX):
     *
     * 1. Canonical mode (default): input is line-buffered.
     *    Characters are only available after Enter is pressed.
     *    Not suitable for games.
     *
     * 2. Raw mode: characters available immediately, no echoing.
     *    Set with tcsetattr():
     *      struct termios raw;
     *      tcgetattr(STDIN_FILENO, &raw);
     *      raw.c_lflag &= ~(ECHO | ICANON);  // Disable echo + canonical
     *      raw.c_cc[VMIN] = 0;                // Non-blocking
     *      raw.c_cc[VTIME] = 0;               // No timeout
     *      tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
     *
     * 3. Arrow keys send escape sequences:
     *    Up:    \033[A
     *    Down:  \033[B
     *    Right: \033[C
     *    Left:  \033[D
     */

    /* Simulate key processing */
    typedef enum { KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_QUIT, KEY_UNKNOWN } KeyType;
    const char *key_names[] = {"UP", "DOWN", "LEFT", "RIGHT", "QUIT", "UNKNOWN"};

    struct {
        const char *sequence;
        KeyType type;
    } key_map[] = {
        {"\033[A", KEY_UP},
        {"\033[B", KEY_DOWN},
        {"\033[C", KEY_RIGHT},
        {"\033[D", KEY_LEFT},
        {"q",      KEY_QUIT},
        {"w",      KEY_UP},
        {"s",      KEY_DOWN},
        {"a",      KEY_LEFT},
        {"d",      KEY_RIGHT},
    };
    int n_keys = (int)(sizeof(key_map) / sizeof(key_map[0]));

    printf("Key mapping table:\n");
    printf("%-12s  %-8s\n", "Sequence", "Action");
    printf("------------  --------\n");
    for (int i = 0; i < n_keys; i++) {
        char display[16];
        if (key_map[i].sequence[0] == '\033') {
            snprintf(display, sizeof(display), "ESC%s", key_map[i].sequence + 1);
        } else {
            snprintf(display, sizeof(display), "'%s'", key_map[i].sequence);
        }
        printf("%-12s  %-8s\n", display, key_names[key_map[i].type]);
    }

    printf("\nGame loop pattern (pseudocode):\n");
    printf("  while (running) {\n");
    printf("    key = read_key();  // Non-blocking\n");
    printf("    update_state(key);\n");
    printf("    render();\n");
    printf("    usleep(100000);    // 100ms = ~10 FPS\n");
    printf("  }\n");
}

/* === Exercise 3: Game State Struct === */
/* Problem: Design the game state with proper data structures. */

#define GRID_W 20
#define GRID_H 10
#define MAX_SNAKE 200

typedef enum { DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT } Direction;

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point body[MAX_SNAKE];
    int length;
    Direction dir;
} Snake;

typedef struct {
    Snake snake;
    Point food;
    int score;
    int game_over;
    int grid[GRID_H][GRID_W];
} GameState;

void game_init(GameState *gs) {
    memset(gs, 0, sizeof(GameState));

    /* Place snake in the center */
    gs->snake.length = 3;
    gs->snake.dir = DIR_RIGHT;
    int center_x = GRID_W / 2;
    int center_y = GRID_H / 2;
    for (int i = 0; i < gs->snake.length; i++) {
        gs->snake.body[i].x = center_x - i;
        gs->snake.body[i].y = center_y;
    }

    /* Place food */
    gs->food.x = GRID_W * 3 / 4;
    gs->food.y = GRID_H / 2;
    gs->score = 0;
    gs->game_over = 0;
}

void game_render(const GameState *gs) {
    /* Top border */
    printf("  +");
    for (int x = 0; x < GRID_W; x++) printf("-");
    printf("+\n");

    for (int y = 0; y < GRID_H; y++) {
        printf("  |");
        for (int x = 0; x < GRID_W; x++) {
            int is_snake = 0;
            for (int i = 0; i < gs->snake.length; i++) {
                if (gs->snake.body[i].x == x && gs->snake.body[i].y == y) {
                    printf("%c", i == 0 ? '@' : 'o');
                    is_snake = 1;
                    break;
                }
            }
            if (!is_snake) {
                if (gs->food.x == x && gs->food.y == y) printf("*");
                else printf(" ");
            }
        }
        printf("|\n");
    }

    /* Bottom border */
    printf("  +");
    for (int x = 0; x < GRID_W; x++) printf("-");
    printf("+\n");
    printf("  Score: %d  Length: %d\n", gs->score, gs->snake.length);
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Game State Struct ===\n");

    printf("Game state structure:\n");
    printf("  GameState size: %zu bytes\n", sizeof(GameState));
    printf("  Snake size:     %zu bytes\n", sizeof(Snake));
    printf("  Grid:           %dx%d\n", GRID_W, GRID_H);
    printf("  Max snake len:  %d\n\n", MAX_SNAKE);

    GameState gs;
    game_init(&gs);
    printf("Initial state:\n");
    game_render(&gs);

    /*
     * Design decisions:
     * - Fixed-size array for snake body vs linked list:
     *   Array is cache-friendly and simpler. MAX_SNAKE caps memory usage.
     * - Grid array vs recomputing: trade space for rendering speed.
     * - Direction enum prevents invalid states.
     */
}

/* === Exercise 4: Collision Detection === */
/* Problem: Implement wall and self-collision detection. */

int check_wall_collision(const Point *head) {
    return head->x < 0 || head->x >= GRID_W ||
           head->y < 0 || head->y >= GRID_H;
}

int check_self_collision(const Snake *snake) {
    /*
     * Check if head overlaps with any body segment.
     * Start from index 1 (index 0 is the head itself).
     * Time: O(n) where n = snake length.
     *
     * Optimization for very long snakes: use a hash set of
     * body positions for O(1) lookup.
     */
    const Point *head = &snake->body[0];
    for (int i = 1; i < snake->length; i++) {
        if (head->x == snake->body[i].x && head->y == snake->body[i].y) {
            return 1;
        }
    }
    return 0;
}

int check_food_collision(const Point *head, const Point *food) {
    return head->x == food->x && head->y == food->y;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Collision Detection ===\n");

    /* Wall collision tests */
    printf("Wall collision tests (grid %dx%d):\n", GRID_W, GRID_H);
    Point wall_tests[] = {
        {0, 0}, {5, 5}, {GRID_W - 1, GRID_H - 1},
        {-1, 5}, {5, -1}, {GRID_W, 5}, {5, GRID_H}
    };
    int n_wall = (int)(sizeof(wall_tests) / sizeof(wall_tests[0]));

    for (int i = 0; i < n_wall; i++) {
        printf("  (%2d, %2d): %s\n", wall_tests[i].x, wall_tests[i].y,
               check_wall_collision(&wall_tests[i]) ? "COLLISION" : "safe");
    }

    /* Self collision test */
    printf("\nSelf collision tests:\n");
    Snake s1 = {.length = 5, .dir = DIR_RIGHT};
    Point body1[] = {{5,5}, {4,5}, {3,5}, {2,5}, {1,5}};
    memcpy(s1.body, body1, sizeof(body1));
    printf("  Straight snake: %s\n",
           check_self_collision(&s1) ? "COLLISION" : "safe");

    Snake s2 = {.length = 5, .dir = DIR_UP};
    Point body2[] = {{5,5}, {5,6}, {4,6}, {4,5}, {5,5}};
    memcpy(s2.body, body2, sizeof(body2));
    printf("  Coiled snake (head overlaps tail): %s\n",
           check_self_collision(&s2) ? "COLLISION" : "safe");

    /* Food collision */
    printf("\nFood collision:\n");
    Point head = {10, 5};
    Point food1 = {10, 5}, food2 = {10, 6};
    printf("  Head(%d,%d) vs Food(%d,%d): %s\n",
           head.x, head.y, food1.x, food1.y,
           check_food_collision(&head, &food1) ? "EAT!" : "miss");
    printf("  Head(%d,%d) vs Food(%d,%d): %s\n",
           head.x, head.y, food2.x, food2.y,
           check_food_collision(&head, &food2) ? "EAT!" : "miss");
}

/* === Exercise 5: Score Tracking and High Scores === */
/* Problem: Implement a persistent high score system. */

#define MAX_SCORES 5

typedef struct {
    char name[16];
    int score;
    int length;
} HighScore;

typedef struct {
    HighScore entries[MAX_SCORES];
    int count;
} Scoreboard;

void scoreboard_init(Scoreboard *sb) {
    sb->count = 0;
    memset(sb->entries, 0, sizeof(sb->entries));
}

int scoreboard_qualifies(const Scoreboard *sb, int score) {
    if (sb->count < MAX_SCORES) return 1;
    return score > sb->entries[sb->count - 1].score;
}

void scoreboard_add(Scoreboard *sb, const char *name, int score, int length) {
    /*
     * Insertion sort approach: find the right position and shift
     * lower scores down. Keeps the list sorted at all times.
     * Time: O(MAX_SCORES) = O(1) since MAX_SCORES is a constant.
     */
    int pos = sb->count;
    for (int i = 0; i < sb->count; i++) {
        if (score > sb->entries[i].score) { pos = i; break; }
    }

    if (pos >= MAX_SCORES) return; /* Doesn't qualify */

    /* Shift lower scores down */
    for (int i = MAX_SCORES - 1; i > pos; i--) {
        sb->entries[i] = sb->entries[i - 1];
    }

    strncpy(sb->entries[pos].name, name, 15);
    sb->entries[pos].name[15] = '\0';
    sb->entries[pos].score = score;
    sb->entries[pos].length = length;
    if (sb->count < MAX_SCORES) sb->count++;
}

void scoreboard_display(const Scoreboard *sb) {
    printf("  %-4s  %-12s  %-6s  %-8s\n", "Rank", "Name", "Score", "Length");
    printf("  ----  ------------  ------  --------\n");
    for (int i = 0; i < sb->count; i++) {
        printf("  %-4d  %-12s  %-6d  %-8d\n",
               i + 1, sb->entries[i].name,
               sb->entries[i].score, sb->entries[i].length);
    }
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Score Tracking ===\n");

    Scoreboard sb;
    scoreboard_init(&sb);

    /* Add scores */
    struct { const char *name; int score; int len; } games[] = {
        {"Alice",   150, 18},
        {"Bob",     80,  11},
        {"Charlie", 200, 23},
        {"Diana",   120, 15},
        {"Eve",     300, 33},
        {"Frank",   90,  12},
        {"Grace",   250, 28},
    };
    int n_games = (int)(sizeof(games) / sizeof(games[0]));

    for (int i = 0; i < n_games; i++) {
        int qualifies = scoreboard_qualifies(&sb, games[i].score);
        scoreboard_add(&sb, games[i].name, games[i].score, games[i].len);
        printf("Game %d: %s scored %d -> %s\n",
               i + 1, games[i].name, games[i].score,
               qualifies ? "NEW HIGH SCORE!" : "did not qualify");
    }

    printf("\nFinal Scoreboard:\n");
    scoreboard_display(&sb);

    printf("\nScore 85 qualifies? %s\n",
           scoreboard_qualifies(&sb, 85) ? "yes" : "no");
    printf("Score 500 qualifies? %s\n",
           scoreboard_qualifies(&sb, 500) ? "yes" : "no");
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
