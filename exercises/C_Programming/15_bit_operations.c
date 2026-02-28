/*
 * Exercises for Lesson 15: Bit Operations
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex15 15_bit_operations.c
 */
#include <stdio.h>
#include <stdint.h>

/* Helper: print an 8-bit value in binary */
static void print_bits8(uint8_t val) {
    for (int i = 7; i >= 0; i--) {
        putchar((val >> i) & 1 ? '1' : '0');
    }
}

/* Helper: print a 32-bit value in binary */
static void print_bits32(uint32_t val) {
    for (int i = 31; i >= 0; i--) {
        putchar((val >> i) & 1 ? '1' : '0');
        if (i % 8 == 0 && i > 0) putchar('_');
    }
}

/* === Exercise 1: Bit Field Extraction === */
/* Problem: Extract 'length' bits starting from bit 'start' of an 8-bit value. */

unsigned char extract_bits(unsigned char value, int start, int length) {
    /*
     * Strategy:
     * 1. Right-shift the value by 'start' to move the target bits to position 0
     * 2. Create a mask with 'length' bits set: (1 << length) - 1
     * 3. AND the shifted value with the mask to isolate the bits
     *
     * Example: extract_bits(0b11010110, 2, 4)
     *   Step 1: 0b11010110 >> 2 = 0b00110101
     *   Step 2: mask = (1 << 4) - 1 = 0b00001111
     *   Step 3: 0b00110101 & 0b00001111 = 0b00000101 = 5
     */
    unsigned char mask = (unsigned char)((1 << length) - 1);
    return (value >> start) & mask;
}

void exercise_1(void) {
    printf("=== Exercise 1: Bit Field Extraction ===\n");

    /* Test case from the exercise */
    unsigned char val = 0xD6; /* 0b11010110 */
    unsigned char result = extract_bits(val, 2, 4);

    printf("Value:  0x%02X = ", val);
    print_bits8(val);
    printf("\n");
    printf("extract_bits(0b11010110, start=2, length=4)\n");
    printf("Result: %u = ", result);
    print_bits8(result);
    printf("\n");

    /* Additional test cases */
    printf("\nAdditional tests:\n");

    result = extract_bits(0xFF, 0, 4);
    printf("extract_bits(0xFF, 0, 4) = %u (expected 15)\n", result);

    result = extract_bits(0xFF, 4, 4);
    printf("extract_bits(0xFF, 4, 4) = %u (expected 15)\n", result);

    result = extract_bits(0xA5, 0, 8);
    printf("extract_bits(0xA5, 0, 8) = %u (expected 165)\n", result);

    result = extract_bits(0x80, 7, 1);
    printf("extract_bits(0x80, 7, 1) = %u (expected 1)\n", result);
}

/* === Exercise 2: Power of Two Check === */
/* Problem: Check if a number is a power of 2 using bit operations. */

int is_power_of_two(unsigned int n) {
    /*
     * A power of 2 has exactly one bit set: 1, 2, 4, 8, 16, ...
     *   8  = 0b1000
     *   7  = 0b0111
     *   8 & 7 = 0 -> power of 2
     *
     *   6  = 0b0110
     *   5  = 0b0101
     *   6 & 5 = 0b0100 != 0 -> not a power of 2
     *
     * The trick: n & (n - 1) clears the lowest set bit of n.
     * If n is a power of 2, it has exactly one set bit, so n & (n-1) == 0.
     * Special case: n must be > 0 (0 is not a power of 2).
     */
    return (n > 0) && ((n & (n - 1)) == 0);
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Power of Two Check ===\n");

    unsigned int test_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 63, 64, 100, 128, 256};
    int num_tests = (int)(sizeof(test_values) / sizeof(test_values[0]));

    printf("%-8s  %-10s  %-10s\n", "Value", "Binary", "Power of 2?");
    printf("--------  ----------  ----------\n");

    for (int i = 0; i < num_tests; i++) {
        unsigned int n = test_values[i];
        printf("%-8u  ", n);
        /* Print lower 8 bits for readability */
        for (int b = 7; b >= 0; b--) {
            putchar((n >> b) & 1 ? '1' : '0');
        }
        printf("  %s\n", is_power_of_two(n) ? "YES" : "NO");
    }
}

/* === Exercise 3: Parity Bit === */
/* Problem: Return 1 if number of set bits is odd, 0 if even. */

int parity(unsigned char n) {
    /*
     * Method: XOR-fold the byte down to a single bit.
     * XOR has the property that x ^ x = 0 and x ^ 0 = x.
     * By folding the byte in half repeatedly, we accumulate the parity.
     *
     * Alternative: Count set bits and check if count is odd.
     * The XOR method is more elegant and commonly used in hardware.
     *
     *   n = 0b10110001 (4 ones -> even parity -> 0)
     *   n ^= n >> 4:  0b10110001 ^ 0b00001011 = 0b10111010
     *   n ^= n >> 2:  take lower bits...
     *   n ^= n >> 1:  final bit holds the parity
     *   return n & 1
     */
    n ^= (n >> 4);
    n ^= (n >> 2);
    n ^= (n >> 1);
    return n & 1;
}

/* Alternative: bit-counting approach for clarity */
static int parity_counting(unsigned char n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count % 2; /* 1 if odd number of bits, 0 if even */
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Parity Bit ===\n");

    /* Test cases from the exercise */
    struct {
        unsigned char val;
        int expected;
        const char *binary;
    } tests[] = {
        {0xB1, 0, "10110001 (4 ones -> even)"},
        {0xB3, 1, "10110011 (5 ones -> odd)"},
        {0x00, 0, "00000000 (0 ones -> even)"},
        {0xFF, 0, "11111111 (8 ones -> even)"},
        {0x01, 1, "00000001 (1 one  -> odd)"},
        {0x80, 1, "10000000 (1 one  -> odd)"},
        {0x55, 0, "01010101 (4 ones -> even)"},
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    printf("%-6s  %-28s  %-8s  %-8s  %-5s\n",
           "Hex", "Binary (description)", "XOR", "Count", "Match");
    printf("------  ----------------------------  --------  --------  -----\n");

    for (int i = 0; i < num_tests; i++) {
        int xor_result = parity(tests[i].val);
        int cnt_result = parity_counting(tests[i].val);
        printf("0x%02X    %-28s  %-8d  %-8d  %s\n",
               tests[i].val, tests[i].binary,
               xor_result, cnt_result,
               (xor_result == tests[i].expected && cnt_result == tests[i].expected) ? "OK" : "FAIL");
    }
}

/* === Exercise 4: Binary Counter (4 LEDs) === */
/* Problem: Display 0~15 in binary using 4 LEDs; increment on button press. */
void exercise_4(void) {
    printf("\n=== Exercise 4: Binary Counter (4 LEDs) ===\n");

    /*
     * Arduino version:
     *   const int ledPins[] = {5, 4, 3, 2}; // MSB to LSB
     *   const int buttonPin = 6;
     *   int counter = 0;
     *   bool lastState = HIGH;
     *
     *   void loop() {
     *       bool state = digitalRead(buttonPin);
     *       if (lastState == HIGH && state == LOW) { // Button pressed
     *           counter = (counter + 1) % 16;
     *           for (int i = 0; i < 4; i++) {
     *               digitalWrite(ledPins[i], (counter >> (3-i)) & 1);
     *           }
     *           delay(50); // Debounce
     *       }
     *       lastState = state;
     *   }
     */

    printf("Simulating button presses (0 through 15 and wrap to 0):\n\n");
    printf("Press#  Value  LED4  LED3  LED2  LED1\n");
    printf("------  -----  ----  ----  ----  ----\n");

    /* Simulate 17 button presses (0 through 15, then wrap to 0) */
    for (int press = 0; press <= 16; press++) {
        int counter = press % 16;
        printf("%-6d  %-5d", press, counter);

        /* Display each bit as LED state (MSB to LSB) */
        for (int bit = 3; bit >= 0; bit--) {
            int state = (counter >> bit) & 1;
            printf("  %4s", state ? "ON" : "OFF");
        }
        printf("   (");
        print_bits8((uint8_t)counter);
        printf(")\n");
    }

    printf("\n  Note: At press 16, counter wraps back to 0 (all LEDs OFF)\n");
    printf("  Key insight: (counter >> bit) & 1 extracts each bit for its LED\n");
}

/* === Bonus: Demonstrate common bit operation patterns === */
void bonus_bit_patterns(void) {
    printf("\n=== Bonus: Common Bit Operation Patterns ===\n");

    uint32_t val = 0x00;
    printf("Initial value: 0x%08X = ", val);
    print_bits32(val);
    printf("\n");

    /* Set bit 3 */
    val |= (1U << 3);
    printf("Set bit 3:     0x%08X = ", val);
    print_bits32(val);
    printf("\n");

    /* Set bit 7 */
    val |= (1U << 7);
    printf("Set bit 7:     0x%08X = ", val);
    print_bits32(val);
    printf("\n");

    /* Toggle bit 3 */
    val ^= (1U << 3);
    printf("Toggle bit 3:  0x%08X = ", val);
    print_bits32(val);
    printf("\n");

    /* Check bit 7 */
    printf("Check bit 7:   %s\n", (val >> 7) & 1 ? "SET" : "CLEAR");

    /* Clear bit 7 */
    val &= ~(1U << 7);
    printf("Clear bit 7:   0x%08X = ", val);
    print_bits32(val);
    printf("\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    bonus_bit_patterns();

    printf("\nAll exercises completed!\n");
    return 0;
}
