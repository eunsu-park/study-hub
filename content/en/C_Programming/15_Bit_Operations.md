# Advanced Bit Operations

**Previous**: [Embedded Programming Basics](./14_Embedded_Basics.md) | **Next**: [Project 15: GPIO Control](./16_Project_GPIO_Control.md)

Master bit-level manipulation, which is essential for embedded programming.

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply all six bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`) to solve practical problems
2. Read, set, clear, and toggle individual bits using masking macros
3. Pack multiple Boolean flags into a single byte to conserve memory
4. Explain how MCU hardware registers map to memory addresses and control peripherals
5. Use direct register manipulation for GPIO control that is significantly faster than library calls
6. Describe why the `volatile` keyword is required for hardware registers and interrupt-shared variables
7. Implement utility functions such as population count, bit reversal, and nibble swap

---

At the hardware level, everything is a bit. A single bit in a register can turn on an LED, enable a communication channel, or trigger an interrupt. Mastering bit operations is the bridge between writing high-level C and truly controlling hardware -- it is the language that microcontrollers actually speak.

## Prerequisites
- C language basic syntax
- Binary and hexadecimal representation

---

## 1. Why Are Bit Operations Important?

### Why Bit Operations Are Essential in Embedded Systems

```
1. Register Control
   - All MCU functions are controlled via registers (special memory)
   - Each bit of a register controls a specific function

2. Memory Conservation
   - Embedded systems have limited memory (KB range)
   - 8 flags can be stored in 1 byte

3. Communication Protocols
   - Need bit-level interpretation of data packets

4. Performance Optimization
   - Bit operations are the fastest operations on CPU
```

### Example: LED Control Register

```
Arduino Uno Port B Register (pins 8~13)
┌─────────────────────────────────────────────┐
│  PORTB = 0b00100000                        │
│                                             │
│  Bit:   7    6    5    4    3    2    1    0│
│        [ ]  [ ]  [1]  [ ]  [ ]  [ ]  [ ]  [ ]│
│              │    │    │    │    │    │    │ │
│              │   Pin13Pin12Pin11Pin10Pin9 Pin8│
│              │    ↓                         │
│              │   LED ON (Pin 13)            │
│              │                              │
│  Bit 5 = 1 → HIGH output on pin 13        │
└─────────────────────────────────────────────┘
```

---

## 2. Bit Operators Review

### Basic Bit Operators

```c
// bitwise_operators.c
#include <stdio.h>

void print_binary(unsigned char n) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
        if (i == 4) printf(" ");  // Space for readability
    }
    printf("\n");
}

int main(void) {
    unsigned char a = 0b11001010;  // 202
    unsigned char b = 0b10110011;  // 179

    printf("a        = "); print_binary(a);  // 1100 1010
    printf("b        = "); print_binary(b);  // 1011 0011
    printf("\n");

    // AND (&): 1 only when both are 1
    printf("a & b    = "); print_binary(a & b);   // 1000 0010

    // OR (|): 1 if either is 1
    printf("a | b    = "); print_binary(a | b);   // 1111 1011

    // XOR (^): 1 if different
    printf("a ^ b    = "); print_binary(a ^ b);   // 0111 1001

    // NOT (~): Bit inversion
    printf("~a       = "); print_binary(~a);      // 0011 0101

    // Left Shift (<<): Shift left, fill with 0
    printf("a << 2   = "); print_binary(a << 2);  // 0010 1000

    // Right Shift (>>): Shift right
    printf("a >> 2   = "); print_binary(a >> 2);  // 0011 0010

    return 0;
}
```

### Operator Truth Tables

```
AND (&)          OR (|)           XOR (^)
A  B  A&B        A  B  A|B        A  B  A^B
0  0   0         0  0   0         0  0   0
0  1   0         0  1   1         0  1   1
1  0   0         1  0   1         1  0   1
1  1   1         1  1   1         1  1   0
```

---

## 3. Bit Masking Techniques

### 3.1 Reading Specific Bit (GET)

```c
// Check value of specific bit
// Method: (value >> bit) & 1

unsigned char reg = 0b10110100;

// Read bit 2
int bit2 = (reg >> 2) & 1;  // Result: 1

// Read bit 3
int bit3 = (reg >> 3) & 1;  // Result: 0

// Define as macro
#define GET_BIT(value, bit) (((value) >> (bit)) & 1)

// Usage example
if (GET_BIT(reg, 5)) {
    printf("Bit 5 is set\n");
}
```

### 3.2 Setting Specific Bit (SET)

```c
// Set specific bit to 1
// Method: value |= (1 << bit)

unsigned char reg = 0b10100000;

// Set bit 3 to 1
reg |= (1 << 3);  // Result: 0b10101000

// Set multiple bits simultaneously
reg |= (1 << 1) | (1 << 4);  // Set bits 1, 4

// Define as macro
#define SET_BIT(value, bit) ((value) |= (1 << (bit)))

// Usage example
SET_BIT(reg, 6);  // Set bit 6
```

**How it works:**
```
reg       = 1010 0000
1 << 3    = 0000 1000
           ─────────── OR
result    = 1010 1000
```

### 3.3 Clearing Specific Bit (CLEAR)

```c
// Clear specific bit to 0
// Method: value &= ~(1 << bit)

unsigned char reg = 0b11111111;

// Clear bit 5 to 0
reg &= ~(1 << 5);  // Result: 0b11011111

// Define as macro
#define CLEAR_BIT(value, bit) ((value) &= ~(1 << (bit)))

// Usage example
CLEAR_BIT(reg, 2);  // Clear bit 2
```

**How it works:**
```
reg       = 1111 1111
1 << 5    = 0010 0000
~(1 << 5) = 1101 1111
           ─────────── AND
result    = 1101 1111
```

### 3.4 Toggling Specific Bit (TOGGLE)

```c
// Toggle specific bit (0→1, 1→0)
// Method: value ^= (1 << bit)

unsigned char reg = 0b10101010;

// Toggle bit 4
reg ^= (1 << 4);  // Result: 0b10111010 (0→1)
reg ^= (1 << 4);  // Result: 0b10101010 (1→0)

// Define as macro
#define TOGGLE_BIT(value, bit) ((value) ^= (1 << (bit)))

// Usage example: LED toggle
TOGGLE_BIT(PORTB, 5);  // Toggle pin 13 LED
```

### 3.5 Bit Mask Utility Header

```c
// bit_utils.h
#ifndef BIT_UTILS_H
#define BIT_UTILS_H

// Bit manipulation macros
#define BIT(n)                  (1 << (n))
#define SET_BIT(reg, bit)       ((reg) |= BIT(bit))
#define CLEAR_BIT(reg, bit)     ((reg) &= ~BIT(bit))
#define TOGGLE_BIT(reg, bit)    ((reg) ^= BIT(bit))
#define GET_BIT(reg, bit)       (((reg) >> (bit)) & 1)
#define CHECK_BIT(reg, bit)     ((reg) & BIT(bit))

// Multiple bit operations
#define SET_BITS(reg, mask)     ((reg) |= (mask))
#define CLEAR_BITS(reg, mask)   ((reg) &= ~(mask))
#define TOGGLE_BITS(reg, mask)  ((reg) ^= (mask))

// Bit field operations
#define GET_FIELD(reg, mask, shift)     (((reg) & (mask)) >> (shift))
#define SET_FIELD(reg, mask, shift, val) \
    ((reg) = ((reg) & ~(mask)) | (((val) << (shift)) & (mask)))

#endif
```

---

## 4. Flag Management

Manage multiple states in a single variable.

### Flag Definition

```c
// flags.c
#include <stdio.h>
#include <stdbool.h>

// Assign meaning to each bit
#define FLAG_RUNNING    (1 << 0)  // Bit 0: Running
#define FLAG_ERROR      (1 << 1)  // Bit 1: Error occurred
#define FLAG_CONNECTED  (1 << 2)  // Bit 2: Connected
#define FLAG_READY      (1 << 3)  // Bit 3: Ready
#define FLAG_BUSY       (1 << 4)  // Bit 4: Busy
#define FLAG_TIMEOUT    (1 << 5)  // Bit 5: Timeout

// Global status flags
unsigned char system_flags = 0;

// Set flag
void set_flag(unsigned char flag) {
    system_flags |= flag;
}

// Clear flag
void clear_flag(unsigned char flag) {
    system_flags &= ~flag;
}

// Check flag
bool is_flag_set(unsigned char flag) {
    return (system_flags & flag) != 0;
}

// Toggle flag
void toggle_flag(unsigned char flag) {
    system_flags ^= flag;
}

int main(void) {
    // System start
    set_flag(FLAG_RUNNING);
    set_flag(FLAG_READY);

    printf("FLAGS: 0x%02X\n", system_flags);

    // Check status
    if (is_flag_set(FLAG_RUNNING)) {
        printf("System running\n");
    }

    if (is_flag_set(FLAG_ERROR)) {
        printf("Error occurred!\n");
    } else {
        printf("Normal operation\n");
    }

    // Error occurs
    set_flag(FLAG_ERROR);
    printf("After setting error flag: 0x%02X\n", system_flags);

    // Error resolved
    clear_flag(FLAG_ERROR);
    printf("After clearing error flag: 0x%02X\n", system_flags);

    return 0;
}
```

### Flag Usage Example in Arduino

```cpp
// arduino_flags.ino

// Status flags
#define STATE_IDLE      0x00
#define STATE_RUNNING   0x01
#define STATE_PAUSED    0x02
#define STATE_ERROR     0x04

unsigned char deviceState = STATE_IDLE;

void setup() {
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    // Process serial commands
    if (Serial.available()) {
        char cmd = Serial.read();

        switch (cmd) {
            case 'r':  // Run
                deviceState |= STATE_RUNNING;
                deviceState &= ~STATE_PAUSED;
                Serial.println("Running");
                break;

            case 'p':  // Pause
                deviceState |= STATE_PAUSED;
                Serial.println("Paused");
                break;

            case 's':  // Stop
                deviceState = STATE_IDLE;
                Serial.println("Stopped");
                break;

            case 'e':  // Error toggle
                deviceState ^= STATE_ERROR;
                Serial.println("Error toggled");
                break;
        }

        // Print status
        Serial.print("State: 0x");
        Serial.println(deviceState, HEX);
    }

    // LED control based on status
    if (deviceState & STATE_ERROR) {
        // Error: Fast blinking
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        delay(100);
    } else if (deviceState & STATE_RUNNING) {
        if (!(deviceState & STATE_PAUSED)) {
            // Running: Slow blinking
            digitalWrite(LED_BUILTIN, HIGH);
            delay(500);
            digitalWrite(LED_BUILTIN, LOW);
            delay(500);
        }
    } else {
        // Idle: LED off
        digitalWrite(LED_BUILTIN, LOW);
    }
}
```

---

## 5. Register Concepts

### What are MCU Registers?

Registers are special memory locations inside the MCU that control hardware.

```
Arduino Uno (ATmega328P) GPIO-related registers:

┌─────────────────────────────────────────────────────┐
│ DDRx (Data Direction Register)                      │
│ - Set pin input/output direction                    │
│ - 0 = input (INPUT)                                 │
│ - 1 = output (OUTPUT)                               │
├─────────────────────────────────────────────────────┤
│ PORTx (Port Output Register)                        │
│ - Output mode: HIGH/LOW output                      │
│ - Input mode: Enable pull-up resistor               │
├─────────────────────────────────────────────────────┤
│ PINx (Port Input Register)                          │
│ - Read current pin status                           │
└─────────────────────────────────────────────────────┘

x = B (pins 8~13), C (analog pins), D (pins 0~7)
```

### Arduino Functions vs Direct Register Control

```cpp
// Using Arduino library (convenient but slow)
pinMode(13, OUTPUT);
digitalWrite(13, HIGH);
digitalWrite(13, LOW);

// Direct register control (fast)
DDRB |= (1 << 5);   // Set pin 13 as output
PORTB |= (1 << 5);  // Pin 13 HIGH
PORTB &= ~(1 << 5); // Pin 13 LOW
```

### Speed Comparison

```cpp
// speed_compare.ino
// digitalWrite vs direct register

void setup() {
    Serial.begin(9600);
    DDRB |= (1 << 5);  // Set pin 13 as output
}

void loop() {
    unsigned long start, end;

    // Measure digitalWrite speed
    start = micros();
    for (long i = 0; i < 100000; i++) {
        digitalWrite(13, HIGH);
        digitalWrite(13, LOW);
    }
    end = micros();
    Serial.print("digitalWrite: ");
    Serial.print(end - start);
    Serial.println(" us");

    // Measure direct register speed
    start = micros();
    for (long i = 0; i < 100000; i++) {
        PORTB |= (1 << 5);
        PORTB &= ~(1 << 5);
    }
    end = micros();
    Serial.print("Direct register: ");
    Serial.print(end - start);
    Serial.println(" us");

    delay(3000);
}

// Example results:
// digitalWrite: 625000 us
// Direct register: 12500 us
// → Direct control is about 50x faster!
```

### Direct Register Access Example

```cpp
// register_access.ino
// LED control using registers

void setup() {
    // DDRB: Data Direction Register B
    // Bit 5 = 1 → Pin 13 output mode
    DDRB |= (1 << DDB5);

    Serial.begin(9600);
    Serial.println("Register control demo");
}

void loop() {
    // Toggle bit 5 of PORTB register
    PORTB ^= (1 << PORTB5);

    // Read current status with PINB register
    if (PINB & (1 << PINB5)) {
        Serial.println("LED ON");
    } else {
        Serial.println("LED OFF");
    }

    delay(500);
}
```

---

## 6. volatile Keyword

### Why volatile is Needed

```c
// Problem: Compiler optimization

int flag = 0;

// Interrupt handler (called by hardware)
void interrupt_handler() {
    flag = 1;
}

int main() {
    while (flag == 0) {
        // Wait
    }
    // Execute here when flag becomes 1
}
```

The compiler may determine that `flag` doesn't change inside the loop and optimize:

```c
// Compiler-optimized code (problem!)
if (flag == 0) {
    while (1) { }  // Converted to infinite loop
}
```

### Using volatile

```c
// Solution: Use volatile keyword
volatile int flag = 0;

void interrupt_handler() {
    flag = 1;
}

int main() {
    while (flag == 0) {
        // Read flag value from memory every time
    }
    // Normal operation
}
```

### Meaning of volatile

```
volatile = "unpredictable"

Tells the compiler:
1. This variable can be changed externally at any time
2. Don't optimize, always read from memory
3. Don't cache in register
```

### Using volatile in Arduino

```cpp
// volatile_example.ino
// Interrupts and volatile

volatile bool buttonPressed = false;
const int BUTTON_PIN = 2;
const int LED_PIN = 13;

// Interrupt Service Routine (ISR)
void buttonISR() {
    buttonPressed = true;
}

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);

    // Attach interrupt to pin 2 (FALLING = HIGH→LOW)
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

    Serial.begin(9600);
}

void loop() {
    if (buttonPressed) {
        Serial.println("Button pressed!");
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));  // Toggle LED

        buttonPressed = false;  // Reset flag
        delay(200);  // Debounce
    }
}
```

### Registers and volatile

```c
// MCU registers are always volatile
// Because hardware can change the value

// Actual Arduino header file definition (avr/io.h)
#define PORTB (*(volatile uint8_t *)0x25)
#define DDRB  (*(volatile uint8_t *)0x24)
#define PINB  (*(volatile uint8_t *)0x23)

// Explanation:
// 0x25 = Memory address of PORTB register
// (volatile uint8_t *) = Cast address to volatile pointer
// * = Access value at that address
```

---

## 7. Practice: Bit Manipulation Utilities

### Bit Counter

```c
// bit_counter.c
#include <stdio.h>

// Count number of 1 bits (popcount)
int count_ones(unsigned int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

// Faster method: Brian Kernighan algorithm
int count_ones_fast(unsigned int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);  // Remove rightmost 1 bit
        count++;
    }
    return count;
}

int main(void) {
    unsigned int test[] = {0, 1, 7, 255, 0xABCD};

    for (int i = 0; i < 5; i++) {
        printf("0x%04X (%5u): %d ones\n",
               test[i], test[i], count_ones(test[i]));
    }

    return 0;
}
```

### Bit Reverse

```c
// bit_reverse.c
#include <stdio.h>

// Reverse 8 bits
unsigned char reverse_bits(unsigned char n) {
    unsigned char result = 0;
    for (int i = 0; i < 8; i++) {
        result <<= 1;
        result |= (n & 1);
        n >>= 1;
    }
    return result;
}

int main(void) {
    unsigned char val = 0b10110001;  // 177

    printf("Original: ");
    for (int i = 7; i >= 0; i--) printf("%d", (val >> i) & 1);
    printf(" (0x%02X)\n", val);

    unsigned char reversed = reverse_bits(val);

    printf("Reversed: ");
    for (int i = 7; i >= 0; i--) printf("%d", (reversed >> i) & 1);
    printf(" (0x%02X)\n", reversed);

    return 0;
}
```

### Bit Swap

```c
// bit_swap.c
#include <stdio.h>

// Swap two bit positions
unsigned char swap_bits(unsigned char n, int i, int j) {
    // Only swap if bits at i and j are different
    if (((n >> i) & 1) != ((n >> j) & 1)) {
        n ^= (1 << i) | (1 << j);  // Toggle both
    }
    return n;
}

// Swap upper and lower 4 bits
unsigned char swap_nibbles(unsigned char n) {
    return ((n & 0x0F) << 4) | ((n & 0xF0) >> 4);
}

int main(void) {
    unsigned char val = 0b11001010;

    printf("Original: 0x%02X (0b%d%d%d%d%d%d%d%d)\n", val,
           (val>>7)&1, (val>>6)&1, (val>>5)&1, (val>>4)&1,
           (val>>3)&1, (val>>2)&1, (val>>1)&1, val&1);

    // Swap bits 1 and 6
    unsigned char swapped = swap_bits(val, 1, 6);
    printf("Bits 1,6 swapped: 0x%02X\n", swapped);

    // Swap nibbles
    unsigned char nibble_swapped = swap_nibbles(val);
    printf("Nibbles swapped: 0x%02X\n", nibble_swapped);

    return 0;
}
```

### Arduino Bit Manipulation Example

```cpp
// bit_manipulation.ino
// Control multiple LEDs with bit manipulation

// LEDs connected: pins 8, 9, 10, 11 (PORTB bits 0~3)
#define LED_MASK 0x0F  // Lower 4 bits

void setup() {
    // Set pins 8~11 as output
    DDRB |= LED_MASK;
    PORTB &= ~LED_MASK;  // Turn off all LEDs

    Serial.begin(9600);
    Serial.println("Bit manipulation demo");
    Serial.println("Commands: 0-F (hex), r (rotate), i (invert)");
}

void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();

        if (cmd >= '0' && cmd <= '9') {
            // Digits 0-9: Display pattern
            PORTB = (PORTB & ~LED_MASK) | (cmd - '0');
        }
        else if (cmd >= 'a' && cmd <= 'f') {
            // a-f: Display patterns 10-15
            PORTB = (PORTB & ~LED_MASK) | (cmd - 'a' + 10);
        }
        else if (cmd >= 'A' && cmd <= 'F') {
            // A-F: Display patterns 10-15
            PORTB = (PORTB & ~LED_MASK) | (cmd - 'A' + 10);
        }
        else if (cmd == 'r' || cmd == 'R') {
            // Left circular shift
            unsigned char leds = PORTB & LED_MASK;
            leds = ((leds << 1) | (leds >> 3)) & LED_MASK;
            PORTB = (PORTB & ~LED_MASK) | leds;
        }
        else if (cmd == 'i' || cmd == 'I') {
            // Invert
            PORTB ^= LED_MASK;
        }

        // Print current status
        Serial.print("LED pattern: 0b");
        for (int i = 3; i >= 0; i--) {
            Serial.print((PORTB >> i) & 1);
        }
        Serial.println();
    }
}
```

---

## Exercises

### Exercise 1: Bit Field Extraction
Write a function to extract bits 2~5 (4 bits) from an 8-bit value.

```c
unsigned char extract_bits(unsigned char value, int start, int length);
// extract_bits(0b11010110, 2, 4) → 0b0101 (5)
```

### Exercise 2: Power of Two Check
Write a function using bit operations to check if a number is a power of 2.

```c
int is_power_of_two(unsigned int n);
// is_power_of_two(8) → 1 (true)
// is_power_of_two(6) → 0 (false)
```

### Exercise 3: Parity Bit
Write a function that returns 1 if the number of 1 bits is odd, 0 if even.

```c
int parity(unsigned char n);
// parity(0b10110001) → 0 (4 ones = even)
// parity(0b10110011) → 1 (5 ones = odd)
```

### Exercise 4: Arduino LED Bar
Use 4 LEDs to display values 0~15 in binary, and increment the value by 1 each time a button is pressed.

---

## Key Concepts Summary

| Operation | Code | Description |
|-----------|------|-------------|
| Set bit | `val \|= (1 << n)` | Set nth bit to 1 |
| Clear bit | `val &= ~(1 << n)` | Set nth bit to 0 |
| Toggle bit | `val ^= (1 << n)` | Invert nth bit |
| Check bit | `(val >> n) & 1` | Get nth bit value |
| Lower n bits | `val & ((1 << n) - 1)` | Only lower n bits |

| Keyword | Meaning |
|---------|---------|
| volatile | Prevent compiler optimization, always read from memory |
| register | Request storage in register (hint) |

---

## Next Steps

Once you've mastered bit operations, proceed to the next document:
- [16. GPIO Control](16_Project_GPIO_Control.md) - Practice with LEDs and buttons
