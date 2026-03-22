// snake.c
// Snake game core logic implementation
// Provides snake creation, movement, collision detection, etc.

#include <stdio.h>
#include <stdlib.h>
#include "snake_types.h"

// ============ Snake Creation and Destruction ============

/**
 * Create snake
 *
 * @param start_x Starting X coordinate (head position)
 * @param start_y Starting Y coordinate (head position)
 * @param initial_dir Initial movement direction
 * @return Pointer to created snake (NULL on failure)
 */
Snake* snake_create(int start_x, int start_y, Direction initial_dir) {
    Snake* snake = malloc(sizeof(Snake));
    if (!snake) {
        return NULL;
    }

    // Initialize snake
    snake->head = NULL;
    snake->tail = NULL;
    snake->length = 0;
    snake->dir = initial_dir;

    // Create initial body (default 3 cells)
    // Add from head to tail in order
    for (int i = 0; i < INITIAL_SNAKE_LENGTH; i++) {
        SnakeNode* node = malloc(sizeof(SnakeNode));
        if (!node) {
            // On allocation failure, free already created nodes
            snake_destroy(snake);
            return NULL;
        }

        // Set initial position based on direction
        node->pos.x = start_x;
        node->pos.y = start_y;

        switch (initial_dir) {
            case DIR_RIGHT:
                node->pos.x -= i;
                break;
            case DIR_LEFT:
                node->pos.x += i;
                break;
            case DIR_DOWN:
                node->pos.y -= i;
                break;
            case DIR_UP:
                node->pos.y += i;
                break;
        }

        node->next = NULL;

        // Add to linked list
        if (snake->head == NULL) {
            // First node
            snake->head = node;
            snake->tail = node;
        } else {
            // Append to tail
            snake->tail->next = node;
            snake->tail = node;
        }
        snake->length++;
    }

    return snake;
}

/**
 * Free snake memory
 *
 * @param snake Pointer to snake to free
 */
void snake_destroy(Snake* snake) {
    if (!snake) return;

    SnakeNode* current = snake->head;
    while (current) {
        SnakeNode* next = current->next;
        free(current);
        current = next;
    }
    free(snake);
}

// ============ Snake Control ============

/**
 * Change snake direction
 * Cannot change to the opposite direction (prevents self-collision)
 *
 * @param snake Pointer to snake
 * @param new_dir New direction
 */
void snake_change_direction(Snake* snake, Direction new_dir) {
    if (!snake) return;

    // Cannot go in the opposite direction
    if (IS_OPPOSITE_DIR(snake->dir, new_dir)) {
        return;
    }

    snake->dir = new_dir;
}

/**
 * Calculate next head position (does not actually move)
 *
 * @param snake Pointer to snake
 * @return Coordinate where the next head will be
 */
Point snake_next_head_position(const Snake* snake) {
    Point next = snake->head->pos;

    switch (snake->dir) {
        case DIR_UP:
            next.y--;
            break;
        case DIR_DOWN:
            next.y++;
            break;
        case DIR_LEFT:
            next.x--;
            break;
        case DIR_RIGHT:
            next.x++;
            break;
    }

    return next;
}

/**
 * Move snake
 * Move head one cell forward; remove tail if food was not eaten
 *
 * @param snake Pointer to snake
 * @param food_pos Food position
 * @return true = ate food, false = did not eat food
 */
bool snake_move(Snake* snake, Point food_pos) {
    if (!snake || !snake->head) return false;

    // Calculate next head position
    Point next = snake_next_head_position(snake);

    // Create new head node
    SnakeNode* new_head = malloc(sizeof(SnakeNode));
    if (!new_head) {
        return false;  // Memory allocation failure
    }

    new_head->pos = next;
    new_head->next = snake->head;
    snake->head = new_head;
    snake->length++;

    // Check if food was eaten
    if (POINT_EQUALS(next, food_pos)) {
        // If food was eaten, keep the tail (length increases)
        return true;
    }

    // If food was not eaten, remove the tail
    if (snake->length > 1) {
        // Find the node before the tail
        SnakeNode* current = snake->head;
        while (current->next != snake->tail) {
            current = current->next;
        }

        // Remove tail
        free(snake->tail);
        snake->tail = current;
        snake->tail->next = NULL;
        snake->length--;
    }

    return false;
}

// ============ Collision Detection ============

/**
 * Check if snake hit a wall
 *
 * @param snake Pointer to snake
 * @return true = wall collision, false = no collision
 */
bool snake_hits_wall(const Snake* snake) {
    if (!snake || !snake->head) return false;

    Point head = snake->head->pos;

    // Collision if touching the border
    return (head.x <= 0 || head.x >= SCREEN_WIDTH - 1 ||
            head.y <= 0 || head.y >= SCREEN_HEIGHT - 1);
}

/**
 * Check if snake hit itself
 *
 * @param snake Pointer to snake
 * @return true = self collision, false = no collision
 */
bool snake_hits_self(const Snake* snake) {
    if (!snake || !snake->head) return false;

    Point head = snake->head->pos;
    SnakeNode* current = snake->head->next;  // Check from the node after the head

    while (current) {
        if (POINT_EQUALS(head, current->pos)) {
            return true;
        }
        current = current->next;
    }

    return false;
}

/**
 * Check if snake occupies a specific position
 * Used when spawning food to avoid overlap with the snake
 *
 * @param snake Pointer to snake
 * @param x X coordinate
 * @param y Y coordinate
 * @return true = snake is at that position, false = not there
 */
bool snake_occupies_position(const Snake* snake, int x, int y) {
    if (!snake) return false;

    SnakeNode* current = snake->head;
    while (current) {
        if (current->pos.x == x && current->pos.y == y) {
            return true;
        }
        current = current->next;
    }

    return false;
}

// ============ Utility Functions ============

/**
 * Return snake length
 *
 * @param snake Pointer to snake
 * @return Snake length
 */
int snake_get_length(const Snake* snake) {
    if (!snake) return 0;
    return snake->length;
}

/**
 * Return snake head position
 *
 * @param snake Pointer to snake
 * @return Head position coordinate
 */
Point snake_get_head_position(const Snake* snake) {
    Point invalid = {-1, -1};
    if (!snake || !snake->head) {
        return invalid;
    }
    return snake->head->pos;
}
