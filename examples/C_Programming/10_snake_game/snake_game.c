// snake_game.c
// Complete snake game implementation
// A terminal-based Snake game using ANSI escape codes.
//
// Compile: gcc -o snake_game snake_game.c
// Run: ./snake_game

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

// ============ Game Settings ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000  // Microseconds (150ms)
#define MIN_SPEED 50000       // Minimum 50ms

// ============ ANSI Control Codes ============
#define CLEAR "\033[2J"
#define HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"
#define MOVE(r,c) printf("\033[%d;%dH", r, c)

// ============ ANSI Color Codes ============
#define RESET "\033[0m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define CYAN "\033[36m"
#define MAGENTA "\033[35m"
#define BOLD "\033[1m"

// ============ Direction Enum ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

// ============ Coordinate Struct ============
typedef struct {
    int x, y;
} Point;

// ============ Snake Node (Linked List) ============
typedef struct Node {
    Point pos;
    struct Node* next;
} Node;

// ============ Game State Struct ============
typedef struct {
    Node* head;        // Snake head
    Node* tail;        // Snake tail
    Direction dir;     // Current direction
    Point food;        // Food position
    int score;         // Score
    int length;        // Snake length
    bool game_over;    // Game over flag
    bool paused;       // Pause flag
    int speed;         // Game speed
    int high_score;    // High score
} Game;

// ============ Terminal Settings ============
static struct termios orig_termios;

// Restore terminal settings
void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    printf(SHOW_CURSOR);
}

// Enable raw mode (non-blocking input)
void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);  // Disable echo, disable line buffering
    raw.c_cc[VMIN] = 0;   // Minimum input character count 0
    raw.c_cc[VTIME] = 0;  // Timeout 0 (return immediately)

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    printf(HIDE_CURSOR);
}

// ============ Input Handling ============

/**
 * Read keyboard input and return direction
 * Handles ESC sequences (arrow keys)
 */
Direction read_direction(Direction current) {
    int ch = getchar();
    if (ch == EOF) return current;

    // ESC sequence (arrow keys)
    if (ch == '\033') {
        if (getchar() == '[') {
            switch (getchar()) {
                case 'A': return (current != DOWN) ? UP : current;
                case 'B': return (current != UP) ? DOWN : current;
                case 'C': return (current != LEFT) ? RIGHT : current;
                case 'D': return (current != RIGHT) ? LEFT : current;
            }
        }
    }

    // WASD keys
    switch (ch) {
        case 'w': case 'W': return (current != DOWN) ? UP : current;
        case 's': case 'S': return (current != UP) ? DOWN : current;
        case 'a': case 'A': return (current != RIGHT) ? LEFT : current;
        case 'd': case 'D': return (current != LEFT) ? RIGHT : current;
        case 'q': case 'Q': return -1;  // Quit signal
    }

    return current;
}

// Check pause key
int check_pause_key(void) {
    int ch = getchar();
    if (ch == 'p' || ch == 'P') return 1;
    if (ch == 'q' || ch == 'Q') return -1;
    return 0;
}

// ============ Snake Functions ============

/**
 * Check if snake occupies a specific position
 */
bool snake_at(Node* head, int x, int y) {
    for (Node* n = head; n; n = n->next) {
        if (n->pos.x == x && n->pos.y == y) return true;
    }
    return false;
}

/**
 * Spawn food (at a position that doesn't overlap with the snake)
 */
void spawn_food(Game* g) {
    do {
        g->food.x = 1 + rand() % (WIDTH - 2);
        g->food.y = 1 + rand() % (HEIGHT - 2);
    } while (snake_at(g->head, g->food.x, g->food.y));
}

// ============ Game Initialization ============

/**
 * Initialize game state
 */
Game* game_init(int high_score) {
    Game* g = malloc(sizeof(Game));
    if (!g) return NULL;

    // Initialize snake (length 3)
    g->head = NULL;
    g->tail = NULL;
    g->length = 0;

    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        if (!n) {
            // Cleanup on allocation failure
            while (g->head) {
                Node* temp = g->head;
                g->head = g->head->next;
                free(temp);
            }
            free(g);
            return NULL;
        }

        n->pos.x = WIDTH / 2 - i;
        n->pos.y = HEIGHT / 2;
        n->next = g->head;
        g->head = n;
        g->length++;
    }

    // Find tail
    Node* curr = g->head;
    while (curr->next) curr = curr->next;
    g->tail = curr;

    // Initialize game state
    g->dir = RIGHT;
    g->score = 0;
    g->game_over = false;
    g->paused = false;
    g->speed = INITIAL_SPEED;
    g->high_score = high_score;

    spawn_food(g);
    return g;
}

/**
 * Free game memory
 */
void game_free(Game* g) {
    if (!g) return;

    Node* n = g->head;
    while (n) {
        Node* next = n->next;
        free(n);
        n = next;
    }
    free(g);
}

// ============ Game Update ============

/**
 * Update game state
 * Returns: true = ate food, false = did not eat food
 */
bool game_update(Game* g) {
    if (g->paused || g->game_over) return false;

    // Calculate next head position
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

    // Wall collision check
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }

    // Self collision check
    if (snake_at(g->head, next.x, next.y)) {
        g->game_over = true;
        return false;
    }

    // Add new head
    Node* new_head = malloc(sizeof(Node));
    if (!new_head) {
        g->game_over = true;
        return false;
    }

    new_head->pos = next;
    new_head->next = g->head;
    g->head = new_head;
    g->length++;

    // Check food
    if (next.x == g->food.x && next.y == g->food.y) {
        g->score += 10;
        spawn_food(g);

        // Increase speed (gets progressively faster)
        if (g->speed > MIN_SPEED) {
            g->speed -= 5000;
            if (g->speed < MIN_SPEED) g->speed = MIN_SPEED;
        }

        return true;
    }

    // If food was not eaten, remove tail
    Node* curr = g->head;
    while (curr->next && curr->next->next) {
        curr = curr->next;
    }
    if (curr->next) {
        free(curr->next);
        curr->next = NULL;
        g->tail = curr;
        g->length--;
    }

    return false;
}

// ============ Screen Drawing ============

/**
 * Draw game border
 */
void draw_border(void) {
    // Top
    MOVE(1, 1);
    printf(CYAN "+");
    for (int i = 1; i < WIDTH - 1; i++) printf("=");
    printf("+" RESET);

    // Left and right sides
    for (int i = 2; i < HEIGHT; i++) {
        MOVE(i, 1);
        printf(CYAN "|" RESET);
        MOVE(i, WIDTH);
        printf(CYAN "|" RESET);
    }

    // Bottom
    MOVE(HEIGHT, 1);
    printf(CYAN "+");
    for (int i = 1; i < WIDTH - 1; i++) printf("=");
    printf("+" RESET);
}

/**
 * Draw game screen
 */
void draw_game(Game* g) {
    printf(CLEAR HOME);

    draw_border();

    // Draw food
    MOVE(g->food.y + 1, g->food.x + 1);
    printf(RED "O" RESET);

    // Draw snake
    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        MOVE(n->pos.y + 1, n->pos.x + 1);
        if (is_head) {
            printf(BOLD GREEN "@" RESET);  // Head
            is_head = false;
        } else {
            printf(GREEN "#" RESET);       // Body
        }
    }

    // Display score and info
    MOVE(HEIGHT + 1, 1);
    printf(YELLOW "Score: %d  |  Length: %d  |  High: %d" RESET,
           g->score, g->length, g->high_score);

    MOVE(HEIGHT + 2, 1);
    printf("Controls: Arrows or WASD  |  P: Pause  |  Q: Quit");

    if (g->paused) {
        MOVE(HEIGHT / 2, WIDTH / 2 - 3);
        printf(BOLD YELLOW "PAUSED" RESET);
    }

    fflush(stdout);
}

/**
 * Game over screen
 */
void draw_game_over(Game* g) {
    MOVE(HEIGHT / 2 - 1, WIDTH / 2 - 5);
    printf(BOLD RED "GAME OVER!" RESET);

    MOVE(HEIGHT / 2, WIDTH / 2 - 7);
    printf("Final Score: " YELLOW "%d" RESET, g->score);

    if (g->score > g->high_score) {
        MOVE(HEIGHT / 2 + 1, WIDTH / 2 - 6);
        printf(BOLD MAGENTA "* New Record! *" RESET);
    }

    MOVE(HEIGHT / 2 + 3, WIDTH / 2 - 8);
    printf("R: Restart  |  Q: Quit");

    fflush(stdout);
}

// ============ High Score Management ============

#define SCORE_FILE ".snake_highscore"

int load_high_score(void) {
    FILE* f = fopen(SCORE_FILE, "r");
    if (!f) return 0;

    int score = 0;
    fscanf(f, "%d", &score);
    fclose(f);
    return score;
}

void save_high_score(int score) {
    FILE* f = fopen(SCORE_FILE, "w");
    if (f) {
        fprintf(f, "%d", score);
        fclose(f);
    }
}

// ============ Main Function ============

int main(void) {
    srand(time(NULL));
    enable_raw_mode();

    int high_score = load_high_score();
    Game* game = game_init(high_score);

    if (!game) {
        fprintf(stderr, "Game initialization failed\n");
        return 1;
    }

    draw_game(game);

    // Main game loop
    while (1) {
        if (!game->game_over) {
            // Handle input
            Direction new_dir = read_direction(game->dir);

            if (new_dir == (Direction)-1) {
                // Quit with Q key
                break;
            }

            game->dir = new_dir;

            // Handle pause
            int pause_key = check_pause_key();
            if (pause_key == 1) {
                game->paused = !game->paused;
            } else if (pause_key == -1) {
                break;
            }

            // Update and draw game
            game_update(game);
            draw_game(game);

            if (game->game_over) {
                // Save high score
                if (game->score > game->high_score) {
                    save_high_score(game->score);
                }
                draw_game_over(game);
            }
        } else {
            // Handle key input in game over state
            int ch = getchar();
            if (ch == 'r' || ch == 'R') {
                // Restart
                int final_high = (game->score > game->high_score) ?
                                 game->score : game->high_score;
                game_free(game);
                game = game_init(final_high);
                if (!game) break;
                draw_game(game);
            } else if (ch == 'q' || ch == 'Q') {
                // Quit
                break;
            }
        }

        usleep(game->speed);
    }

    game_free(game);

    // Screen cleanup
    printf(CLEAR HOME SHOW_CURSOR);
    MOVE(1, 1);
    printf("Exiting the game. Thanks for playing!\n");

    return 0;
}
