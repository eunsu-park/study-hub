# Project: Snake Game

**Previous**: [Cross-Platform Development](./16_Cross_Platform_Development.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply ANSI escape codes to clear the screen, move the cursor, hide/show the cursor, and render colored text
2. Configure the terminal for raw mode using `termios` to capture keystrokes without line buffering
3. Implement non-blocking keyboard input using `VMIN`/`VTIME` settings and escape-sequence parsing for arrow keys
4. Design a game loop that follows the Input-Update-Render cycle with frame-rate control via `usleep`
5. Build a snake data structure using a linked list where the head grows and the tail shrinks each frame
6. Implement collision detection against walls and the snake's own body
7. Compare raw ANSI-based rendering with the `ncurses` library approach and identify trade-offs

---

Building a game forces you to combine nearly everything you have learned so far -- structs, linked lists, dynamic memory, bit flags, and terminal I/O -- into a single program that must respond to user input in real time. The snake game is a perfect vehicle because its rules are simple enough to focus on the systems-programming challenges: raw terminal control, non-blocking input, and a tight update-render loop that must run at a consistent frame rate.

## Prerequisites
- Structures and pointers
- Dynamic memory management
- Linked lists (for snake body representation)

---

## Step 1: Understanding ANSI Escape Codes

We use ANSI escape codes to display graphics in the terminal.

### Basic ANSI Codes

```c
// ansi_demo.c
#include <stdio.h>
#include <unistd.h>

// ANSI Escape Codes
#define CLEAR_SCREEN "\033[2J"
#define CURSOR_HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"

// Cursor movement: \033[row;colH
#define MOVE_CURSOR(row, col) printf("\033[%d;%dH", row, col)

// Colors
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"

int main(void) {
    printf(CLEAR_SCREEN CURSOR_HOME HIDE_CURSOR);

    MOVE_CURSOR(5, 10);
    printf(COLOR_RED "Red Text" COLOR_RESET);

    MOVE_CURSOR(7, 10);
    printf(COLOR_GREEN "Green Text" COLOR_RESET);

    MOVE_CURSOR(9, 10);
    printf(COLOR_BLUE "Blue Text" COLOR_RESET);

    sleep(3);

    printf(SHOW_CURSOR);
    MOVE_CURSOR(12, 1);

    return 0;
}
```

---

## Step 2: Asynchronous Keyboard Input

In games, execution must continue without waiting for key input.

### Input Handling with termios

```c
// input_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

static struct termios original_termios;

void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &original_termios);

    struct termios raw = original_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_termios);
}

typedef enum {
    KEY_NONE = 0,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_QUIT, KEY_OTHER
} KeyCode;

KeyCode read_key(void) {
    int ch = getchar();
    if (ch == EOF) return KEY_NONE;
    if (ch == 'q' || ch == 'Q') return KEY_QUIT;

    if (ch == '\033') {
        int ch2 = getchar();
        if (ch2 == '[') {
            switch (getchar()) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
    }

    switch (ch) {
        case 'w': case 'W': return KEY_UP;
        case 's': case 'S': return KEY_DOWN;
        case 'a': case 'A': return KEY_LEFT;
        case 'd': case 'D': return KEY_RIGHT;
    }

    return KEY_OTHER;
}
```

---

## Step 3: Game Data Structures

```c
// snake_types.h
#ifndef SNAKE_TYPES_H
#define SNAKE_TYPES_H

#include <stdbool.h>

#define SCREEN_WIDTH 40
#define SCREEN_HEIGHT 20
#define GAME_SPEED 150000

typedef enum { DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT } Direction;

typedef struct { int x, y; } Point;

typedef struct SnakeNode {
    Point pos;
    struct SnakeNode* next;
} SnakeNode;

typedef struct {
    SnakeNode* head;
    SnakeNode* tail;
    Direction dir;
    int length;
} Snake;

typedef struct {
    Snake snake;
    Point food;
    int score;
    bool game_over;
    bool paused;
} GameState;

#endif
```

---

## Step 4: Complete Snake Game

```c
// snake_game.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

// ============ Config ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000

// ============ ANSI Codes ============
#define CLEAR "\033[2J"
#define HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"
#define MOVE(r,c) printf("\033[%d;%dH", r, c)

#define RESET "\033[0m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define CYAN "\033[36m"
#define BOLD "\033[1m"

// ============ Direction ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

typedef struct { int x, y; } Point;

typedef struct Node {
    Point pos;
    struct Node* next;
} Node;

// ============ Game State ============
typedef struct {
    Node* head;
    Node* tail;
    Direction dir;
    Point food;
    int score;
    int length;
    bool game_over;
    int speed;
} Game;

// ============ Terminal Setup ============
static struct termios orig_termios;

void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    printf(SHOW_CURSOR);
}

void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    printf(HIDE_CURSOR);
}

// ============ Input ============
Direction read_direction(Direction current) {
    int ch = getchar();
    if (ch == EOF) return current;

    if (ch == '\033') {
        getchar();
        switch (getchar()) {
            case 'A': return (current != DOWN) ? UP : current;
            case 'B': return (current != UP) ? DOWN : current;
            case 'C': return (current != LEFT) ? RIGHT : current;
            case 'D': return (current != RIGHT) ? LEFT : current;
        }
    }

    switch (ch) {
        case 'w': case 'W': return (current != DOWN) ? UP : current;
        case 's': case 'S': return (current != UP) ? DOWN : current;
        case 'a': case 'A': return (current != RIGHT) ? LEFT : current;
        case 'd': case 'D': return (current != LEFT) ? RIGHT : current;
        case 'q': case 'Q': return -1;
    }

    return current;
}

// ============ Game Functions ============
bool snake_at(Node* head, int x, int y) {
    for (Node* n = head; n; n = n->next) {
        if (n->pos.x == x && n->pos.y == y) return true;
    }
    return false;
}

void spawn_food(Game* g) {
    do {
        g->food.x = 1 + rand() % (WIDTH - 2);
        g->food.y = 1 + rand() % (HEIGHT - 2);
    } while (snake_at(g->head, g->food.x, g->food.y));
}

Game* game_init(void) {
    Game* g = malloc(sizeof(Game));

    g->head = NULL;
    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        n->pos.x = WIDTH / 2 - i;
        n->pos.y = HEIGHT / 2;
        n->next = g->head;
        g->head = n;
        if (i == 0) g->tail = n;
    }

    Node* curr = g->head;
    while (curr->next) curr = curr->next;
    g->tail = curr;

    g->dir = RIGHT;
    g->score = 0;
    g->length = 3;
    g->game_over = false;
    g->speed = INITIAL_SPEED;

    spawn_food(g);
    return g;
}

void game_free(Game* g) {
    Node* n = g->head;
    while (n) {
        Node* next = n->next;
        free(n);
        n = next;
    }
    free(g);
}

bool game_update(Game* g) {
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

    // Wall collision
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }

    // Self collision
    if (snake_at(g->head, next.x, next.y)) {
        g->game_over = true;
        return false;
    }

    // Add new head
    Node* new_head = malloc(sizeof(Node));
    new_head->pos = next;
    new_head->next = g->head;
    g->head = new_head;

    // Check food
    if (next.x == g->food.x && next.y == g->food.y) {
        g->score += 10;
        g->length++;
        spawn_food(g);

        if (g->speed > 50000) {
            g->speed -= 5000;
        }
        return true;
    }

    // Remove tail
    Node* curr = g->head;
    while (curr->next && curr->next->next) {
        curr = curr->next;
    }
    free(curr->next);
    curr->next = NULL;
    g->tail = curr;

    return false;
}

// ============ Drawing ============
void draw_border(void) {
    MOVE(1, 1);
    printf(CYAN "+");
    for (int i = 1; i < WIDTH - 1; i++) printf("-");
    printf("+" RESET);

    for (int i = 2; i < HEIGHT; i++) {
        MOVE(i, 1);
        printf(CYAN "|" RESET);
        MOVE(i, WIDTH);
        printf(CYAN "|" RESET);
    }

    MOVE(HEIGHT, 1);
    printf(CYAN "+");
    for (int i = 1; i < WIDTH - 1; i++) printf("-");
    printf("+" RESET);
}

void draw_game(Game* g) {
    printf(CLEAR HOME);
    draw_border();

    MOVE(g->food.y + 1, g->food.x + 1);
    printf(RED "o" RESET);

    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        MOVE(n->pos.y + 1, n->pos.x + 1);
        if (is_head) {
            printf(BOLD GREEN "@" RESET);
            is_head = false;
        } else {
            printf(GREEN "#" RESET);
        }
    }

    MOVE(HEIGHT + 1, 1);
    printf(YELLOW "Score: %d  Length: %d" RESET, g->score, g->length);

    MOVE(HEIGHT + 2, 1);
    printf("Controls: Arrow keys or WASD, Q: Quit");

    fflush(stdout);
}

void draw_game_over(Game* g) {
    MOVE(HEIGHT / 2, WIDTH / 2 - 5);
    printf(BOLD RED "GAME OVER!" RESET);

    MOVE(HEIGHT / 2 + 1, WIDTH / 2 - 6);
    printf("Final Score: %d", g->score);

    MOVE(HEIGHT / 2 + 2, WIDTH / 2 - 8);
    printf("R: Restart, Q: Quit");

    fflush(stdout);
}

// ============ Main ============
int main(void) {
    srand(time(NULL));
    enable_raw_mode();

    Game* game = game_init();
    draw_game(game);

    while (1) {
        Direction new_dir = read_direction(game->dir);
        if (new_dir == (Direction)-1) break;
        game->dir = new_dir;

        if (!game->game_over) {
            game_update(game);
            draw_game(game);

            if (game->game_over) {
                draw_game_over(game);
            }
        } else {
            int ch = getchar();
            if (ch == 'r' || ch == 'R') {
                game_free(game);
                game = game_init();
                draw_game(game);
            } else if (ch == 'q' || ch == 'Q') {
                break;
            }
        }

        usleep(game->speed);
    }

    game_free(game);

    MOVE(HEIGHT + 4, 1);
    printf("Exiting game.\n");

    return 0;
}
```

### Compile and Run

```bash
gcc -o snake snake_game.c -Wall -Wextra
./snake
```

---

## Step 5: Feature Extensions

### Wall Wrap Mode

```c
// Replace wall collision with wrapping
if (next.x <= 0) next.x = WIDTH - 2;
else if (next.x >= WIDTH - 1) next.x = 1;

if (next.y <= 0) next.y = HEIGHT - 2;
else if (next.y >= HEIGHT - 1) next.y = 1;
```

### Obstacles

```c
#define MAX_OBSTACLES 10

typedef struct {
    Point obstacles[MAX_OBSTACLES];
    int count;
} Obstacles;

void spawn_obstacles(Game* g, Obstacles* obs, int count) {
    obs->count = 0;
    for (int i = 0; i < count && obs->count < MAX_OBSTACLES; i++) {
        Point p;
        do {
            p.x = 2 + rand() % (WIDTH - 4);
            p.y = 2 + rand() % (HEIGHT - 4);
        } while (snake_at(g->head, p.x, p.y) ||
                 (p.x == g->food.x && p.y == g->food.y));
        obs->obstacles[obs->count++] = p;
    }
}
```

### Level System

```c
typedef struct {
    int level;
    int food_to_next;
    int food_eaten;
} LevelSystem;

void level_init(LevelSystem* ls) {
    ls->level = 1;
    ls->food_to_next = 5;
    ls->food_eaten = 0;
}

bool level_eat_food(LevelSystem* ls) {
    ls->food_eaten++;
    if (ls->food_eaten >= ls->food_to_next) {
        ls->level++;
        ls->food_eaten = 0;
        ls->food_to_next += 2;
        return true;  // Level up!
    }
    return false;
}
```

### High Score

```c
#define SCORE_FILE "snake_highscore.dat"

int load_highscore(void) {
    FILE* f = fopen(SCORE_FILE, "r");
    if (!f) return 0;
    int score;
    if (fscanf(f, "%d", &score) != 1) score = 0;
    fclose(f);
    return score;
}

void save_highscore(int score) {
    int current = load_highscore();
    if (score > current) {
        FILE* f = fopen(SCORE_FILE, "w");
        if (f) {
            fprintf(f, "%d", score);
            fclose(f);
        }
    }
}
```

---

## Step 6: ncurses Version (Optional)

Using the ncurses library enables cleaner code with built-in color support, key handling, and window management.

```bash
# Install ncurses
# macOS: brew install ncurses
# Ubuntu: sudo apt install libncurses5-dev

# Compile
gcc -o snake_ncurses snake_ncurses.c -lncurses
```

Key ncurses functions:
- `initscr()` / `endwin()` -- Initialize/cleanup
- `cbreak()` / `noecho()` -- Raw input
- `nodelay(stdscr, TRUE)` -- Non-blocking input
- `keypad(stdscr, TRUE)` -- Arrow key support
- `mvaddch(y, x, ch)` -- Draw character
- `mvprintw(y, x, fmt, ...)` -- Print formatted text
- `attron(COLOR_PAIR(n))` -- Set color

---

## Exercises

### Exercise 1: Pause Feature
Implement pause functionality when P key is pressed. Display "PAUSED" in the center of the screen.

### Exercise 2: Special Items
Add special items that appear occasionally:
- Golden apple: 30 points
- Speed down: Temporarily reduces speed
- Invisibility: Temporarily allows passing through self

### Exercise 3: Two Player Mode
Implement 2-player mode with WASD and arrow keys for each player.

### Exercise 4: AI Snake
Add an AI snake that automatically finds food.
- Hint: Use BFS or simple heuristics

### Exercise 5: Persistent Leaderboard
Save the top 10 scores with player names to a file. Display the leaderboard on game over.

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| ANSI Escape Codes | Terminal screen control (cursor, colors) |
| termios | Terminal I/O configuration |
| Raw mode | Immediate input without buffering |
| Game loop | Input -> Update -> Render cycle |
| Frame rate | Speed control with usleep |
| ncurses | Terminal UI library |

---

## Next Steps

Congratulations on completing C Advanced! You now have deep expertise in systems programming with C. Explore related topics like [Operating Systems](../OS_Theory/00_Overview.md), [Networking](../Networking/00_Overview.md), or [Distributed Systems](../Distributed_Systems/00_Overview.md).
