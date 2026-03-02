// snake_ncurses.c
// Snake game using the NCurses library
//
// *** This file requires the ncurses library ***
//
// Installation:
//   macOS:   brew install ncurses
//   Ubuntu:  sudo apt install libncurses5-dev
//   Fedora:  sudo dnf install ncurses-devel
//
// Compile:
//   macOS:   gcc -o snake_ncurses snake_ncurses.c -lncurses
//   Linux:   gcc -o snake_ncurses snake_ncurses.c -lncurses
//
// Run:
//   ./snake_ncurses

#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

// ============ Game Settings ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000  // 150ms
#define MIN_SPEED 50000       // 50ms

// ============ Color Definitions ============
enum {
    COLOR_SNAKE = 1,
    COLOR_FOOD,
    COLOR_BORDER,
    COLOR_TEXT
};

// ============ Direction Enum ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

// ============ Coordinate Struct ============
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
    bool paused;
    int speed;
    int high_score;
} Game;

// ============ Global Variables ============
WINDOW* game_win;
WINDOW* info_win;

// ============ Utility Functions ============

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
 * Spawn food
 */
void spawn_food(Game* g) {
    do {
        g->food.x = 1 + rand() % (WIDTH - 2);
        g->food.y = 1 + rand() % (HEIGHT - 2);
    } while (snake_at(g->head, g->food.x, g->food.y));
}

// ============ Game Initialization ============

/**
 * Initialize NCurses
 */
void init_ncurses(void) {
    initscr();              // Start NCurses
    cbreak();               // Disable line buffering
    noecho();               // Don't display typed characters
    nodelay(stdscr, TRUE);  // Non-blocking input
    keypad(stdscr, TRUE);   // Enable arrow keys
    curs_set(0);            // Hide cursor

    // Initialize colors
    if (has_colors()) {
        start_color();
        init_pair(COLOR_SNAKE, COLOR_GREEN, COLOR_BLACK);
        init_pair(COLOR_FOOD, COLOR_RED, COLOR_BLACK);
        init_pair(COLOR_BORDER, COLOR_CYAN, COLOR_BLACK);
        init_pair(COLOR_TEXT, COLOR_YELLOW, COLOR_BLACK);
    }

    // Create game window
    game_win = newwin(HEIGHT, WIDTH, 1, 2);
    info_win = newwin(3, WIDTH, HEIGHT + 2, 2);
}

/**
 * Cleanup NCurses
 */
void cleanup_ncurses(void) {
    delwin(game_win);
    delwin(info_win);
    endwin();
}

/**
 * Initialize game
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

    // Initialize state
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

// ============ Input Handling ============

/**
 * Handle keyboard input
 * Returns: 0 = continue, -1 = quit
 */
int handle_input(Game* g) {
    int ch = getch();

    if (ch == ERR) return 0;  // No input

    if (ch == 'q' || ch == 'Q') return -1;  // Quit

    if (ch == 'p' || ch == 'P') {
        g->paused = !g->paused;
        return 0;
    }

    if (g->paused || g->game_over) return 0;

    // Change direction (opposite direction not allowed)
    switch (ch) {
        case KEY_UP:
        case 'w':
        case 'W':
            if (g->dir != DOWN) g->dir = UP;
            break;
        case KEY_DOWN:
        case 's':
        case 'S':
            if (g->dir != UP) g->dir = DOWN;
            break;
        case KEY_LEFT:
        case 'a':
        case 'A':
            if (g->dir != RIGHT) g->dir = LEFT;
            break;
        case KEY_RIGHT:
        case 'd':
        case 'D':
            if (g->dir != LEFT) g->dir = RIGHT;
            break;
    }

    return 0;
}

// ============ Game Update ============

/**
 * Update game state
 */
bool game_update(Game* g) {
    if (g->paused || g->game_over) return false;

    // Next head position
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

        // Increase speed
        if (g->speed > MIN_SPEED) {
            g->speed -= 5000;
            if (g->speed < MIN_SPEED) g->speed = MIN_SPEED;
        }

        return true;
    }

    // Remove tail
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
 * Draw game screen
 */
void draw_game(Game* g) {
    // Clear game window
    werase(game_win);

    // Draw border
    wattron(game_win, COLOR_PAIR(COLOR_BORDER));
    box(game_win, 0, 0);
    wattroff(game_win, COLOR_PAIR(COLOR_BORDER));

    // Draw food
    wattron(game_win, COLOR_PAIR(COLOR_FOOD) | A_BOLD);
    mvwaddch(game_win, g->food.y, g->food.x, 'O');
    wattroff(game_win, COLOR_PAIR(COLOR_FOOD) | A_BOLD);

    // Draw snake
    wattron(game_win, COLOR_PAIR(COLOR_SNAKE));
    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        if (is_head) {
            wattron(game_win, A_BOLD);
            mvwaddch(game_win, n->pos.y, n->pos.x, '@');
            wattroff(game_win, A_BOLD);
            is_head = false;
        } else {
            mvwaddch(game_win, n->pos.y, n->pos.x, '#');
        }
    }
    wattroff(game_win, COLOR_PAIR(COLOR_SNAKE));

    // Pause message
    if (g->paused) {
        wattron(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);
        mvwprintw(game_win, HEIGHT / 2, WIDTH / 2 - 3, "PAUSED");
        wattroff(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);
    }

    // Refresh game window
    wrefresh(game_win);

    // Draw info window
    werase(info_win);
    wattron(info_win, COLOR_PAIR(COLOR_TEXT));
    mvwprintw(info_win, 0, 1, "Score: %d  |  Length: %d  |  High: %d",
              g->score, g->length, g->high_score);
    mvwprintw(info_win, 1, 1, "Controls: Arrows/WASD  |  P: Pause  |  Q: Quit");
    wattroff(info_win, COLOR_PAIR(COLOR_TEXT));
    wrefresh(info_win);
}

/**
 * Game over screen
 */
void draw_game_over(Game* g) {
    wattron(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);

    mvwprintw(game_win, HEIGHT / 2 - 1, WIDTH / 2 - 5, "GAME OVER!");
    mvwprintw(game_win, HEIGHT / 2, WIDTH / 2 - 7, "Final Score: %d", g->score);

    if (g->score > g->high_score) {
        mvwprintw(game_win, HEIGHT / 2 + 1, WIDTH / 2 - 6, "* New Record! *");
    }

    mvwprintw(game_win, HEIGHT / 2 + 3, WIDTH / 2 - 8, "R: Restart  |  Q: Quit");

    wattroff(game_win, COLOR_PAIR(COLOR_TEXT) | A_BOLD);
    wrefresh(game_win);
}

// ============ High Score Management ============

#define SCORE_FILE ".snake_ncurses_highscore"

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

    init_ncurses();
    atexit(cleanup_ncurses);

    int high_score = load_high_score();
    Game* game = game_init(high_score);

    if (!game) {
        cleanup_ncurses();
        fprintf(stderr, "Game initialization failed\n");
        return 1;
    }

    draw_game(game);

    // Main game loop
    while (1) {
        // Handle input
        if (handle_input(game) == -1) {
            break;  // Quit
        }

        if (!game->game_over) {
            // Update game
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
            // Handle restart in game over state
            int ch = getch();
            if (ch == 'r' || ch == 'R') {
                int final_high = (game->score > game->high_score) ?
                                 game->score : game->high_score;
                game_free(game);
                game = game_init(final_high);
                if (!game) break;
                draw_game(game);
            } else if (ch == 'q' || ch == 'Q') {
                break;
            }
        }

        usleep(game->speed);
    }

    game_free(game);

    // Exit message
    clear();
    mvprintw(0, 0, "Exiting the game. Thanks for playing!");
    refresh();
    nodelay(stdscr, FALSE);
    getch();

    return 0;
}
