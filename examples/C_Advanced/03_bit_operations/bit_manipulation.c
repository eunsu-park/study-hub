// bit_manipulation.c
// Bit manipulation examples

#include <stdio.h>

// Print binary representation
void print_binary(unsigned char n) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
        if (i == 4) printf(" ");
    }
}

// Bit manipulation macros
// Why: wrapping every parameter in parentheses prevents operator precedence bugs —
// without them, SET_BIT(x, a+b) would expand to (x |= (1 << a+b)) where + binds
// tighter than <<, giving the wrong bit position
#define SET_BIT(reg, bit)    ((reg) |= (1 << (bit)))
#define CLEAR_BIT(reg, bit)  ((reg) &= ~(1 << (bit)))
#define TOGGLE_BIT(reg, bit) ((reg) ^= (1 << (bit)))
#define GET_BIT(reg, bit)    (((reg) >> (bit)) & 1)

int main(void) {
    unsigned char value = 0b10110010;  // 178

    printf("=== Bit Manipulation Examples ===\n\n");

    printf("Initial value: ");
    print_binary(value);
    printf(" (0x%02X, %d)\n\n", value, value);

    // Set bit (SET)
    printf("Set bit 3 (SET_BIT):\n");
    SET_BIT(value, 3);
    printf("  Result: ");
    print_binary(value);
    printf(" (0x%02X)\n\n", value);

    // Clear bit (CLEAR)
    printf("Clear bit 5 (CLEAR_BIT):\n");
    CLEAR_BIT(value, 5);
    printf("  Result: ");
    print_binary(value);
    printf(" (0x%02X)\n\n", value);

    // Toggle bit (TOGGLE)
    printf("Toggle bit 0 (TOGGLE_BIT):\n");
    TOGGLE_BIT(value, 0);
    printf("  Result: ");
    print_binary(value);
    printf(" (0x%02X)\n\n", value);

    // Read bit (GET)
    printf("Read each bit value:\n");
    for (int i = 7; i >= 0; i--) {
        printf("  Bit %d: %d\n", i, GET_BIT(value, i));
    }
    printf("\n");

    // Flag examples
    printf("=== Flag Management Examples ===\n\n");

    // Why: using bit shifts (1 << N) for flag values ensures each flag occupies
    // exactly one bit — a single byte can store 8 independent boolean flags,
    // far more memory-efficient than 8 separate bool variables
    #define FLAG_RUNNING   (1 << 0)
    #define FLAG_ERROR     (1 << 1)
    #define FLAG_CONNECTED (1 << 2)
    #define FLAG_READY     (1 << 3)

    unsigned char flags = 0;

    printf("Initial flags: ");
    print_binary(flags);
    printf("\n\n");

    // Set flags
    flags |= FLAG_RUNNING;
    printf("RUNNING flag set:  ");
    print_binary(flags);
    printf("\n");

    flags |= FLAG_READY;
    printf("READY flag set:    ");
    print_binary(flags);
    printf("\n\n");

    // Why: bitwise AND with a flag mask tests that specific bit — if the bit is set,
    // the result is non-zero (truthy); if cleared, the result is zero (falsy)
    if (flags & FLAG_RUNNING) {
        printf("System is running.\n");
    }

    if (flags & FLAG_ERROR) {
        printf("Error occurred!\n");
    } else {
        printf("Operating normally\n");
    }
    printf("\n");

    // Why: ~FLAG_RUNNING inverts to 0b11111110, and AND clears only bit 0 —
    // this is the only safe way to clear one bit without affecting the others
    flags &= ~FLAG_RUNNING;
    printf("RUNNING flag cleared: ");
    print_binary(flags);
    printf("\n");

    return 0;
}
