# Project 10: Terminal Snake Game

**Previous**: [Project 8: Hash Table](./10_Project_Hash_Table.md) | **Next**: [Project 11: Mini Shell](./12_Project_Mini_Shell.md)

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
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"

int main(void) {
    // Clear screen
    printf(CLEAR_SCREEN);
    printf(CURSOR_HOME);

    // Hide cursor
    printf(HIDE_CURSOR);

    // Output at various positions
    MOVE_CURSOR(5, 10);
    printf(COLOR_RED "Red Text" COLOR_RESET);

    MOVE_CURSOR(7, 10);
    printf(COLOR_GREEN "Green Text" COLOR_RESET);

    MOVE_CURSOR(9, 10);
    printf(COLOR_BLUE "Blue Text" COLOR_RESET);

    // Draw box
    MOVE_CURSOR(12, 5);
    printf("┌────────────────────┐");
    for (int i = 13; i < 18; i++) {
        MOVE_CURSOR(i, 5);
        printf("│                    │");
    }
    MOVE_CURSOR(18, 5);
    printf("└────────────────────┘");

    MOVE_CURSOR(15, 10);
    printf(COLOR_YELLOW "Text in Box" COLOR_RESET);

    sleep(3);

    // Show cursor
    printf(SHOW_CURSOR);
    MOVE_CURSOR(20, 1);

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
#include <fcntl.h>

// Store original terminal settings
static struct termios original_termios;

// Set terminal to raw mode
void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &original_termios);

    struct termios raw = original_termios;

    // Input flags: disable echo, disable line buffering
    raw.c_lflag &= ~(ECHO | ICANON);

    // Minimum input characters: 0 (non-blocking possible)
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;  // No timeout

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// Restore terminal settings
void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_termios);
}

// Check for key input (non-blocking)
int kbhit(void) {
    int ch = getchar();
    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

// Read key
int getch(void) {
    return getchar();
}

// Read arrow keys (handle escape sequence)
typedef enum {
    KEY_NONE = 0,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_QUIT,
    KEY_OTHER
} KeyCode;

KeyCode read_key(void) {
    int ch = getchar();

    if (ch == EOF) return KEY_NONE;
    if (ch == 'q' || ch == 'Q') return KEY_QUIT;

    // Escape sequence (arrow keys)
    if (ch == '\033') {
        int ch2 = getchar();
        if (ch2 == '[') {
            int ch3 = getchar();
            switch (ch3) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
    }

    // WASD keys support
    switch (ch) {
        case 'w': case 'W': return KEY_UP;
        case 's': case 'S': return KEY_DOWN;
        case 'a': case 'A': return KEY_LEFT;
        case 'd': case 'D': return KEY_RIGHT;
    }

    return KEY_OTHER;
}

int main(void) {
    enable_raw_mode();
    atexit(disable_raw_mode);  // Auto restore on exit

    printf("\033[2J\033[H");  // Clear screen
    printf("Move with arrow keys or WASD, Q to quit\n\n");

    int x = 10, y = 5;

    while (1) {
        KeyCode key = read_key();

        if (key == KEY_QUIT) break;

        // Erase previous position
        printf("\033[%d;%dH ", y, x);

        switch (key) {
            case KEY_UP:    if (y > 3) y--; break;
            case KEY_DOWN:  if (y < 20) y++; break;
            case KEY_LEFT:  if (x > 1) x--; break;
            case KEY_RIGHT: if (x < 40) x++; break;
            default: break;
        }

        // Display at new position
        printf("\033[%d;%dH@", y, x);
        fflush(stdout);

        usleep(50000);  // 50ms delay
    }

    printf("\033[22;1HExiting.\n");
    return 0;
}
```

---

## Step 3: Basic Game Structure

### Game Data Structure Definition

```c
// snake_types.h
#ifndef SNAKE_TYPES_H
#define SNAKE_TYPES_H

#include <stdbool.h>

// Screen size
#define SCREEN_WIDTH 40
#define SCREEN_HEIGHT 20

// Game speed (microseconds)
#define GAME_SPEED 150000

// Direction
typedef enum {
    DIR_UP,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT
} Direction;

// Coordinates
typedef struct {
    int x;
    int y;
} Point;

// Snake body node
typedef struct SnakeNode {
    Point pos;
    struct SnakeNode* next;
} SnakeNode;

// Snake
typedef struct {
    SnakeNode* head;
    SnakeNode* tail;
    Direction dir;
    int length;
} Snake;

// Game state
typedef struct {
    Snake snake;
    Point food;
    int score;
    bool game_over;
    bool paused;
} GameState;

#endif
```

### Snake Management Functions

```c
// snake.c
#include <stdio.h>
#include <stdlib.h>
#include "snake_types.h"

// Create snake
Snake* snake_create(int start_x, int start_y) {
    Snake* snake = malloc(sizeof(Snake));
    if (!snake) return NULL;

    // Initial body of 3 segments
    snake->head = NULL;
    snake->tail = NULL;
    snake->length = 0;
    snake->dir = DIR_RIGHT;

    // Add from head to tail
    for (int i = 0; i < 3; i++) {
        SnakeNode* node = malloc(sizeof(SnakeNode));
        node->pos.x = start_x - i;
        node->pos.y = start_y;
        node->next = NULL;

        if (snake->head == NULL) {
            snake->head = node;
            snake->tail = node;
        } else {
            snake->tail->next = node;
            snake->tail = node;
        }
        snake->length++;
    }

    return snake;
}

// Free snake
void snake_destroy(Snake* snake) {
    SnakeNode* current = snake->head;
    while (current) {
        SnakeNode* next = current->next;
        free(current);
        current = next;
    }
    free(snake);
}

// Change direction (prevent opposite direction)
void snake_change_direction(Snake* snake, Direction new_dir) {
    // Can't go in opposite direction
    if ((snake->dir == DIR_UP && new_dir == DIR_DOWN) ||
        (snake->dir == DIR_DOWN && new_dir == DIR_UP) ||
        (snake->dir == DIR_LEFT && new_dir == DIR_RIGHT) ||
        (snake->dir == DIR_RIGHT && new_dir == DIR_LEFT)) {
        return;
    }
    snake->dir = new_dir;
}

// Calculate next head position
Point snake_next_head(Snake* snake) {
    Point next = snake->head->pos;

    switch (snake->dir) {
        case DIR_UP:    next.y--; break;
        case DIR_DOWN:  next.y++; break;
        case DIR_LEFT:  next.x--; break;
        case DIR_RIGHT: next.x++; break;
    }

    return next;
}

// Move snake (returns true if food eaten)
bool snake_move(Snake* snake, Point food) {
    Point next = snake_next_head(snake);

    // Create new head node
    SnakeNode* new_head = malloc(sizeof(SnakeNode));
    new_head->pos = next;
    new_head->next = snake->head;
    snake->head = new_head;
    snake->length++;

    // Check if food eaten
    if (next.x == food.x && next.y == food.y) {
        return true;  // Keep tail (grow)
    }

    // Remove tail if no food eaten
    SnakeNode* current = snake->head;
    while (current->next != snake->tail) {
        current = current->next;
    }
    free(snake->tail);
    snake->tail = current;
    snake->tail->next = NULL;
    snake->length--;

    return false;
}

// Collision check: wall
bool snake_hit_wall(Snake* snake, int width, int height) {
    int x = snake->head->pos.x;
    int y = snake->head->pos.y;
    return (x < 1 || x >= width - 1 || y < 1 || y >= height - 1);
}

// Collision check: self
bool snake_hit_self(Snake* snake) {
    SnakeNode* head = snake->head;
    SnakeNode* current = head->next;

    while (current) {
        if (head->pos.x == current->pos.x &&
            head->pos.y == current->pos.y) {
            return true;
        }
        current = current->next;
    }
    return false;
}

// Check if snake occupies position
bool snake_occupies(Snake* snake, int x, int y) {
    SnakeNode* current = snake->head;
    while (current) {
        if (current->pos.x == x && current->pos.y == y) {
            return true;
        }
        current = current->next;
    }
    return false;
}
```

---

## Step 4: Complete Snake Game

### Main Game Code

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
#define INITIAL_SPEED 150000  // microseconds

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

// ============ Coordinates ============
typedef struct {
    int x, y;
} Point;

// ============ Snake Node ============
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

// ============ Input Handler ============
Direction read_direction(Direction current) {
    int ch = getchar();
    if (ch == EOF) return current;

    // ESC sequence (arrow keys)
    if (ch == '\033') {
        getchar();  // '['
        switch (getchar()) {
            case 'A': return (current != DOWN) ? UP : current;
            case 'B': return (current != UP) ? DOWN : current;
            case 'C': return (current != LEFT) ? RIGHT : current;
            case 'D': return (current != RIGHT) ? LEFT : current;
        }
    }

    // WASD
    switch (ch) {
        case 'w': case 'W': return (current != DOWN) ? UP : current;
        case 's': case 'S': return (current != UP) ? DOWN : current;
        case 'a': case 'A': return (current != RIGHT) ? LEFT : current;
        case 'd': case 'D': return (current != LEFT) ? RIGHT : current;
        case 'q': case 'Q': return -1;  // Quit signal
    }

    return current;
}

// ============ Snake Functions ============
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

    // Initialize snake (length 3)
    g->head = NULL;
    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        n->pos.x = WIDTH / 2 - i;
        n->pos.y = HEIGHT / 2;
        n->next = g->head;
        g->head = n;
        if (i == 0) g->tail = n;
    }

    // Find tail
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
    // Calculate next head position
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

        // Increase speed (minimum 50ms)
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

// ============ Draw Functions ============
void draw_border(void) {
    // Top
    MOVE(1, 1);
    printf(CYAN "╔");
    for (int i = 1; i < WIDTH - 1; i++) printf("═");
    printf("╗" RESET);

    // Sides
    for (int i = 2; i < HEIGHT; i++) {
        MOVE(i, 1);
        printf(CYAN "║" RESET);
        MOVE(i, WIDTH);
        printf(CYAN "║" RESET);
    }

    // Bottom
    MOVE(HEIGHT, 1);
    printf(CYAN "╚");
    for (int i = 1; i < WIDTH - 1; i++) printf("═");
    printf("╝" RESET);
}

void draw_game(Game* g) {
    printf(CLEAR HOME);

    draw_border();

    // Food
    MOVE(g->food.y + 1, g->food.x + 1);
    printf(RED "●" RESET);

    // Snake
    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        MOVE(n->pos.y + 1, n->pos.x + 1);
        if (is_head) {
            printf(BOLD GREEN "◆" RESET);
            is_head = false;
        } else {
            printf(GREEN "■" RESET);
        }
    }

    // Score
    MOVE(HEIGHT + 1, 1);
    printf(YELLOW "Score: %d  Length: %d" RESET, g->score, g->length);

    MOVE(HEIGHT + 2, 1);
    printf("Controls: ↑↓←→ or WASD, Q: Quit");

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
        // Handle input
        Direction new_dir = read_direction(game->dir);
        if (new_dir == (Direction)-1) break;  // Q quit
        game->dir = new_dir;

        if (!game->game_over) {
            // Update game
            game_update(game);
            draw_game(game);

            if (game->game_over) {
                draw_game_over(game);
            }
        } else {
            // In game over state, R to restart
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
gcc -o snake snake_game.c
./snake
```

---

## Step 5: Feature Extensions

### Add Wall Wrap Mode

```c
// Add to configuration
#define WALL_WRAP true  // true to wrap to opposite side

// Modify wall collision in game_update function
bool game_update(Game* g) {
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

#if WALL_WRAP
    // Wall wrap: appear on opposite side
    if (next.x <= 0) next.x = WIDTH - 2;
    else if (next.x >= WIDTH - 1) next.x = 1;

    if (next.y <= 0) next.y = HEIGHT - 2;
    else if (next.y >= HEIGHT - 1) next.y = 1;
#else
    // Wall collision: game over
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }
#endif

    // ... rest of code
}
```

### Add Obstacles

```c
// Obstacle structure
#define MAX_OBSTACLES 10

typedef struct {
    Point obstacles[MAX_OBSTACLES];
    int count;
} Obstacles;

// Spawn obstacles
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

// Check obstacle collision
bool hit_obstacle(Obstacles* obs, int x, int y) {
    for (int i = 0; i < obs->count; i++) {
        if (obs->obstacles[i].x == x && obs->obstacles[i].y == y) {
            return true;
        }
    }
    return false;
}

// Draw obstacles
void draw_obstacles(Obstacles* obs) {
    for (int i = 0; i < obs->count; i++) {
        MOVE(obs->obstacles[i].y + 1, obs->obstacles[i].x + 1);
        printf("\033[35m█\033[0m");  // Magenta color
    }
}
```

### Level System

```c
typedef struct {
    int level;
    int food_to_next;  // Food needed for next level
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
        ls->food_to_next += 2;  // More food needed each level
        return true;  // Level up!
    }
    return false;
}

// Calculate speed for level
int get_speed_for_level(int level) {
    int base_speed = 150000;
    int speed = base_speed - (level - 1) * 15000;
    return (speed < 50000) ? 50000 : speed;
}
```

### Score Saving (High Score)

```c
#include <stdio.h>

#define SCORE_FILE "snake_highscore.dat"

int load_highscore(void) {
    FILE* f = fopen(SCORE_FILE, "r");
    if (!f) return 0;

    int score;
    if (fscanf(f, "%d", &score) != 1) {
        score = 0;
    }
    fclose(f);
    return score;
}

void save_highscore(int score) {
    int current_high = load_highscore();
    if (score > current_high) {
        FILE* f = fopen(SCORE_FILE, "w");
        if (f) {
            fprintf(f, "%d", score);
            fclose(f);
        }
    }
}

// Call on game end
void game_end(Game* g) {
    int highscore = load_highscore();

    if (g->score > highscore) {
        MOVE(HEIGHT / 2 + 3, WIDTH / 2 - 8);
        printf("\033[33m★ New Record! ★\033[0m");
        save_highscore(g->score);
    } else {
        MOVE(HEIGHT / 2 + 3, WIDTH / 2 - 8);
        printf("High Score: %d", highscore);
    }
}
```

---

## Step 6: ncurses Version (Optional)

Using the ncurses library enables cleaner code.

### Install ncurses

```bash
# macOS
brew install ncurses

# Ubuntu/Debian
sudo apt install libncurses5-dev

# Fedora
sudo dnf install ncurses-devel
```

### ncurses Version Basic Structure

```c
// snake_ncurses.c
#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define WIDTH 40
#define HEIGHT 20

// Direction
enum { UP, DOWN, LEFT, RIGHT };

// Node
typedef struct Node {
    int x, y;
    struct Node* next;
} Node;

// Global state
Node* head = NULL;
int dir = RIGHT;
int food_x, food_y;
int score = 0;
bool game_over = false;

void spawn_food(void) {
    do {
        food_x = 1 + rand() % (WIDTH - 2);
        food_y = 1 + rand() % (HEIGHT - 2);
    } while (/* check snake position */ 0);
}

void init_game(void) {
    // Initialize ncurses
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);  // non-blocking input
    keypad(stdscr, TRUE);   // enable arrow keys
    curs_set(0);            // hide cursor

    // Initialize colors
    if (has_colors()) {
        start_color();
        init_pair(1, COLOR_GREEN, COLOR_BLACK);  // snake
        init_pair(2, COLOR_RED, COLOR_BLACK);    // food
        init_pair(3, COLOR_CYAN, COLOR_BLACK);   // wall
    }

    // Initialize snake
    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        n->x = WIDTH / 2 - i;
        n->y = HEIGHT / 2;
        n->next = head;
        head = n;
    }

    srand(time(NULL));
    spawn_food();
}

void draw(void) {
    clear();

    // Wall
    attron(COLOR_PAIR(3));
    for (int i = 0; i < WIDTH; i++) {
        mvaddch(0, i, ACS_HLINE);
        mvaddch(HEIGHT - 1, i, ACS_HLINE);
    }
    for (int i = 0; i < HEIGHT; i++) {
        mvaddch(i, 0, ACS_VLINE);
        mvaddch(i, WIDTH - 1, ACS_VLINE);
    }
    mvaddch(0, 0, ACS_ULCORNER);
    mvaddch(0, WIDTH - 1, ACS_URCORNER);
    mvaddch(HEIGHT - 1, 0, ACS_LLCORNER);
    mvaddch(HEIGHT - 1, WIDTH - 1, ACS_LRCORNER);
    attroff(COLOR_PAIR(3));

    // Food
    attron(COLOR_PAIR(2));
    mvaddch(food_y, food_x, 'O');
    attroff(COLOR_PAIR(2));

    // Snake
    attron(COLOR_PAIR(1));
    for (Node* n = head; n; n = n->next) {
        mvaddch(n->y, n->x, n == head ? '@' : '#');
    }
    attroff(COLOR_PAIR(1));

    // Score
    mvprintw(HEIGHT + 1, 0, "Score: %d", score);

    refresh();
}

void input(void) {
    int ch = getch();
    switch (ch) {
        case KEY_UP:    if (dir != DOWN) dir = UP; break;
        case KEY_DOWN:  if (dir != UP) dir = DOWN; break;
        case KEY_LEFT:  if (dir != RIGHT) dir = LEFT; break;
        case KEY_RIGHT: if (dir != LEFT) dir = RIGHT; break;
        case 'q': game_over = true; break;
    }
}

void update(void) {
    // Next position
    int nx = head->x, ny = head->y;
    switch (dir) {
        case UP:    ny--; break;
        case DOWN:  ny++; break;
        case LEFT:  nx--; break;
        case RIGHT: nx++; break;
    }

    // Wall collision
    if (nx <= 0 || nx >= WIDTH - 1 || ny <= 0 || ny >= HEIGHT - 1) {
        game_over = true;
        return;
    }

    // New head
    Node* new_head = malloc(sizeof(Node));
    new_head->x = nx;
    new_head->y = ny;
    new_head->next = head;
    head = new_head;

    // Check food
    if (nx == food_x && ny == food_y) {
        score += 10;
        spawn_food();
    } else {
        // Remove tail
        Node* curr = head;
        while (curr->next && curr->next->next) curr = curr->next;
        free(curr->next);
        curr->next = NULL;
    }
}

void cleanup(void) {
    while (head) {
        Node* next = head->next;
        free(head);
        head = next;
    }
    endwin();
}

int main(void) {
    init_game();

    while (!game_over) {
        input();
        update();
        draw();
        usleep(100000);
    }

    // Game over message
    mvprintw(HEIGHT / 2, WIDTH / 2 - 5, "GAME OVER!");
    mvprintw(HEIGHT / 2 + 1, WIDTH / 2 - 6, "Score: %d", score);
    refresh();
    nodelay(stdscr, FALSE);
    getch();

    cleanup();
    return 0;
}
```

### Compile

```bash
# macOS
gcc -o snake_ncurses snake_ncurses.c -lncurses

# Linux
gcc -o snake_ncurses snake_ncurses.c -lncurses
```

---

## Practice Problems

### Exercise 1: Pause Feature
Implement pause functionality when P key is pressed.

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

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| ANSI Escape Codes | Terminal screen control (cursor, colors) |
| termios | Terminal I/O configuration |
| Raw mode | Immediate input without buffering |
| Game loop | Input → Update → Render cycle |
| Frame rate | Speed control with usleep |
| ncurses | Terminal UI library |

---

## Next Steps

After completing the snake game, move on to the next project:
- [Project 11: Mini Shell](12_Project_Mini_Shell.md) - Simple command shell implementation
