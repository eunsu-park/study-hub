// input_demo.c
// Asynchronous keyboard input demonstration program
// Implements non-blocking input using termios.

#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// Store original terminal settings
static struct termios original_termios;

// Set terminal to raw mode
void enable_raw_mode(void) {
    // Save current terminal settings
    tcgetattr(STDIN_FILENO, &original_termios);

    struct termios raw = original_termios;

    // Modify input flags
    // ECHO: Don't display typed characters on screen
    // ICANON: Disable line buffering (read immediately without Enter)
    raw.c_lflag &= ~(ECHO | ICANON);

    // Minimum input character count: 0 (enables non-blocking)
    raw.c_cc[VMIN] = 0;
    // Timeout: 0 (return immediately)
    raw.c_cc[VTIME] = 0;

    // Apply modified settings
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// Restore terminal settings
void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_termios);
}

// Check for key input (non-blocking)
// Returns: 1 = input available, 0 = no input
int kbhit(void) {
    int ch = getchar();
    if (ch != EOF) {
        ungetc(ch, stdin);  // Put the read character back into the buffer
        return 1;
    }
    return 0;
}

// Read key (non-blocking)
int getch(void) {
    return getchar();
}

// Arrow key and special key codes
typedef enum {
    KEY_NONE = 0,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_QUIT,
    KEY_SPACE,
    KEY_ENTER,
    KEY_OTHER
} KeyCode;

// Read and interpret key (handles escape sequences)
KeyCode read_key(void) {
    int ch = getchar();

    // No input
    if (ch == EOF) return KEY_NONE;

    // Quit key
    if (ch == 'q' || ch == 'Q') return KEY_QUIT;

    // Spacebar
    if (ch == ' ') return KEY_SPACE;

    // Enter
    if (ch == '\n' || ch == '\r') return KEY_ENTER;

    // Escape sequence (arrow keys, etc.)
    // Arrow keys are 3 bytes: ESC(27) + '[' + character
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
        // ESC pressed alone
        return KEY_QUIT;
    }

    // WASD key support
    switch (ch) {
        case 'w': case 'W': return KEY_UP;
        case 's': case 'S': return KEY_DOWN;
        case 'a': case 'A': return KEY_LEFT;
        case 'd': case 'D': return KEY_RIGHT;
    }

    return KEY_OTHER;
}

// Convert key code to string
const char* keycode_to_string(KeyCode key) {
    switch (key) {
        case KEY_UP: return "Up";
        case KEY_DOWN: return "Down";
        case KEY_LEFT: return "Left";
        case KEY_RIGHT: return "Right";
        case KEY_SPACE: return "Space";
        case KEY_ENTER: return "Enter";
        case KEY_QUIT: return "Quit";
        default: return "Other";
    }
}

int main(void) {
    // Enable raw mode
    enable_raw_mode();
    // Automatically restore terminal settings on program exit
    atexit(disable_raw_mode);

    // Initialize screen
    printf("\033[2J\033[H");  // Clear screen + cursor home
    printf("\033[?25l");       // Hide cursor

    printf("=== Asynchronous Keyboard Input Demo ===\n\n");
    printf("Controls:\n");
    printf("  - Arrow keys or WASD: Move\n");
    printf("  - Space: Jump\n");
    printf("  - Q or ESC: Quit\n\n");
    printf("Game screen:\n");
    printf("+----------------------------------------+\n");
    for (int i = 0; i < 15; i++) {
        printf("|                                        |\n");
    }
    printf("+----------------------------------------+\n");

    // Player initial position
    int x = 20, y = 12;
    int jump_height = 0;
    int key_count = 0;

    // Status display position
    int status_row = 23;

    // Main loop
    while (1) {
        // Read key input (non-blocking)
        KeyCode key = read_key();

        if (key == KEY_QUIT) break;

        // Clear previous position
        printf("\033[%d;%dH ", y - jump_height, x);

        // Process input
        switch (key) {
            case KEY_UP:
                if (y > 7) y--;
                key_count++;
                break;
            case KEY_DOWN:
                if (y < 20) y++;
                key_count++;
                break;
            case KEY_LEFT:
                if (x > 3) x--;
                key_count++;
                break;
            case KEY_RIGHT:
                if (x < 41) x--;
                key_count++;
                break;
            case KEY_SPACE:
                jump_height = (jump_height == 0) ? 2 : 0;
                key_count++;
                break;
            default:
                break;
        }

        // Draw player at new position
        printf("\033[%d;%dH\033[32m@\033[0m", y - jump_height, x);

        // Update status info
        printf("\033[%d;1H", status_row);
        printf("Position: (%d, %d)  ", x, y);
        printf("Jump: %s  ", jump_height > 0 ? "ON " : "OFF");
        printf("Inputs: %d  ", key_count);
        if (key != KEY_NONE) {
            printf("Last input: %s    ", keycode_to_string(key));
        }

        // Refresh screen
        fflush(stdout);

        // Frame rate control (50ms = 20 FPS)
        usleep(50000);
    }

    // Exit handling
    printf("\033[2J\033[H");  // Clear screen
    printf("\033[?25h");       // Show cursor

    printf("Exiting the program.\n");
    printf("Total inputs: %d\n", key_count);

    return 0;
}
