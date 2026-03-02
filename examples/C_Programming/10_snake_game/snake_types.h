// snake_types.h
// Snake game data structure definitions
// Defines all types and constants needed for the game.

#ifndef SNAKE_TYPES_H
#define SNAKE_TYPES_H

#include <stdbool.h>

// ============ Game Configuration Constants ============

// Screen size (including border)
#define SCREEN_WIDTH 40
#define SCREEN_HEIGHT 20

// Game area size (excluding border)
#define GAME_WIDTH (SCREEN_WIDTH - 2)
#define GAME_HEIGHT (SCREEN_HEIGHT - 2)

// Game speed (microseconds)
#define INITIAL_GAME_SPEED 150000  // 150ms
#define MIN_GAME_SPEED 50000       // 50ms (max speed)
#define SPEED_INCREMENT 5000       // Gets 5ms faster each time food is eaten

// Score
#define POINTS_PER_FOOD 10

// Snake initial settings
#define INITIAL_SNAKE_LENGTH 3
#define INITIAL_SNAKE_X (SCREEN_WIDTH / 2)
#define INITIAL_SNAKE_Y (SCREEN_HEIGHT / 2)

// ============ Direction Enum ============

// Snake movement direction
typedef enum {
    DIR_UP,     // Up
    DIR_DOWN,   // Down
    DIR_LEFT,   // Left
    DIR_RIGHT   // Right
} Direction;

// ============ Coordinate Struct ============

// Struct representing a 2D coordinate
typedef struct {
    int x;  // X coordinate (horizontal)
    int y;  // Y coordinate (vertical)
} Point;

// ============ Snake Structs ============

// Snake body node (linked list)
// Each node represents one cell of the snake.
typedef struct SnakeNode {
    Point pos;              // Position of this node
    struct SnakeNode* next; // Next node (toward tail)
} SnakeNode;

// Struct representing the entire snake
typedef struct {
    SnakeNode* head;  // Snake head (first node)
    SnakeNode* tail;  // Snake tail (last node)
    Direction dir;    // Current movement direction
    int length;       // Snake length
} Snake;

// ============ Food Struct ============

// Food (only needs coordinates)
typedef Point Food;

// ============ Game State Struct ============

// Struct representing the overall game state
typedef struct {
    Snake snake;        // Snake object
    Food food;          // Food position
    int score;          // Current score
    int speed;          // Current game speed (microseconds)
    bool game_over;     // Game over flag
    bool paused;        // Pause flag
} GameState;

// ============ Utility Macros ============

// Check if two points are equal
#define POINT_EQUALS(p1, p2) ((p1).x == (p2).x && (p1).y == (p2).y)

// Check if a point is within the game area (excluding border)
#define POINT_IN_BOUNDS(p) \
    ((p).x > 0 && (p).x < SCREEN_WIDTH - 1 && \
     (p).y > 0 && (p).y < SCREEN_HEIGHT - 1)

// Check if two directions are opposite
#define IS_OPPOSITE_DIR(d1, d2) \
    ((d1) == DIR_UP && (d2) == DIR_DOWN) || \
    ((d1) == DIR_DOWN && (d2) == DIR_UP) || \
    ((d1) == DIR_LEFT && (d2) == DIR_RIGHT) || \
    ((d1) == DIR_RIGHT && (d2) == DIR_LEFT)

// ============ ANSI Color Codes ============

// Screen control
#define ANSI_CLEAR "\033[2J"
#define ANSI_HOME "\033[H"
#define ANSI_HIDE_CURSOR "\033[?25l"
#define ANSI_SHOW_CURSOR "\033[?25h"

// Colors
#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BLUE "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN "\033[36m"

// ============ Function Declarations (implemented in snake.c) ============

// Snake creation and destruction
Snake* snake_create(int start_x, int start_y, Direction initial_dir);
void snake_destroy(Snake* snake);

// Snake control
void snake_change_direction(Snake* snake, Direction new_dir);
Point snake_next_head_position(const Snake* snake);
bool snake_move(Snake* snake, Point food_pos);

// Collision detection
bool snake_hits_wall(const Snake* snake);
bool snake_hits_self(const Snake* snake);
bool snake_occupies_position(const Snake* snake, int x, int y);

// Utility
int snake_get_length(const Snake* snake);
Point snake_get_head_position(const Snake* snake);

#endif // SNAKE_TYPES_H
