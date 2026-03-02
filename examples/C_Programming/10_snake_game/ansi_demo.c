// ansi_demo.c
// ANSI Escape Codes demonstration program
// Shows how to control cursor movement and colors in the terminal.

#include <stdio.h>
#include <unistd.h>

// ANSI Escape Codes definitions
#define CLEAR_SCREEN "\033[2J"
#define CURSOR_HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"

// Cursor movement: \033[row;colH
#define MOVE_CURSOR(row, col) printf("\033[%d;%dH", row, col)

// Color codes
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"

// Background color codes
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE "\033[44m"

// Text styles
#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"

int main(void) {
    // Clear screen
    printf(CLEAR_SCREEN);
    printf(CURSOR_HOME);

    // Hide cursor
    printf(HIDE_CURSOR);

    // Display title
    MOVE_CURSOR(2, 10);
    printf(BOLD COLOR_CYAN "=== ANSI Escape Codes Demo ===" COLOR_RESET);

    // Output colored text at various positions
    MOVE_CURSOR(5, 10);
    printf(COLOR_RED "Red text" COLOR_RESET);

    MOVE_CURSOR(7, 10);
    printf(COLOR_GREEN "Green text" COLOR_RESET);

    MOVE_CURSOR(9, 10);
    printf(COLOR_BLUE "Blue text" COLOR_RESET);

    MOVE_CURSOR(11, 10);
    printf(BOLD COLOR_YELLOW "Bold yellow text" COLOR_RESET);

    MOVE_CURSOR(13, 10);
    printf(UNDERLINE COLOR_MAGENTA "Underlined magenta text" COLOR_RESET);

    // Background color examples
    MOVE_CURSOR(15, 10);
    printf(BG_RED "   Red background   " COLOR_RESET);

    MOVE_CURSOR(16, 10);
    printf(BG_GREEN "  Green background  " COLOR_RESET);

    // Draw box (using UTF-8 box-drawing characters)
    MOVE_CURSOR(19, 5);
    printf(COLOR_CYAN "+-----------------------+" COLOR_RESET);
    for (int i = 20; i < 25; i++) {
        MOVE_CURSOR(i, 5);
        printf(COLOR_CYAN "|                       |" COLOR_RESET);
    }
    MOVE_CURSOR(25, 5);
    printf(COLOR_CYAN "+-----------------------+" COLOR_RESET);

    MOVE_CURSOR(22, 10);
    printf(COLOR_YELLOW "Text inside the box" COLOR_RESET);

    // Draw grid pattern
    MOVE_CURSOR(19, 40);
    printf(BOLD "Grid pattern:" COLOR_RESET);
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 10; col++) {
            MOVE_CURSOR(20 + row, 40 + col * 2);
            if ((row + col) % 2 == 0) {
                printf(COLOR_GREEN "#" COLOR_RESET);
            } else {
                printf(COLOR_RED "." COLOR_RESET);
            }
        }
    }

    // Animation effect (progress bar)
    MOVE_CURSOR(27, 5);
    printf("Loading: [");
    for (int i = 0; i < 20; i++) {
        printf(COLOR_GREEN "=" COLOR_RESET);
        fflush(stdout);
        usleep(100000); // Wait 100ms
    }
    printf("]");

    // Info message
    MOVE_CURSOR(29, 5);
    printf(COLOR_YELLOW "Exiting automatically in 3 seconds..." COLOR_RESET);

    fflush(stdout);
    sleep(3);

    // Clear screen and show cursor
    printf(CLEAR_SCREEN);
    printf(SHOW_CURSOR);
    MOVE_CURSOR(1, 1);

    printf("ANSI Escape Codes demo complete.\n");

    return 0;
}
