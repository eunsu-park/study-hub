# Written Example Code List

## C Programs (Desktop/Console)

### ✅ 02. Calculator
- **File**: `02_calculator/calculator.c`
- **Compile**: `gcc calculator.c -o calculator`
- **Run**: `./calculator`
- **Description**: Basic arithmetic calculator using functions and switch-case

### ✅ 03. Number Guessing Game
- **File**: `03_number_guess/number_guess.c`
- **Compile**: `gcc number_guess.c -o number_guess`
- **Run**: `./number_guess`
- **Description**: Guess a number between 1-100 using rand()

### ✅ 05. Dynamic Array
- **File**: `05_dynamic_array/dynamic_array.c`
- **Compile**: `gcc dynamic_array.c -o dynamic_array`
- **Run**: `./dynamic_array`
- **Description**: Dynamic array implementation using malloc/realloc

**Sample Output:**
```
=== Dynamic Array Test ===

Adding elements: 10, 20, 30, 40, 50
Capacity expanded: 2 -> 4
Capacity expanded: 4 -> 8
Array: [10, 20, 30, 40, 50]
Size: 5, Capacity: 8
```

### ✅ 06. Linked List
- **File**: `06_linked_list/linked_list.c`
- **Compile**: `gcc linked_list.c -o linked_list`
- **Run**: `./linked_list`
- **Description**: Singly linked list implementation with add/delete/search

### ✅ 12. Multithreading
- **File**: `12_multithread/thread_basic.c`
- **Compile**: `gcc thread_basic.c -o thread_basic -pthread`
- **Run**: `./thread_basic`
- **Description**: Basic multithreading with pthread

### ✅ 14. Bit Operations
- **File**: `14_bit_operations/bit_manipulation.c`
- **Compile**: `gcc bit_manipulation.c -o bit_manipulation`
- **Run**: `./bit_manipulation`
- **Description**: Bit masking, SET/CLEAR/TOGGLE/GET

**Sample Output:**
```
=== Bit Manipulation Examples ===

Initial value: 1011 0010 (0xB2, 178)

Bit 3 SET (SET_BIT):
  Result: 1011 1010 (0xBA)

Bit 5 CLEAR (CLEAR_BIT):
  Result: 1001 1010 (0x9A)
```

---

## Arduino Programs (.ino)

### ✅ 13. LED Blink
- **File**: `13_embedded_basic/blink.ino`
- **Platform**: Arduino Uno
- **Simulator**: https://wokwi.com
- **Description**: The most basic Arduino program, LED blinking at 1-second intervals

**How to Run on Wokwi:**
1. Visit https://wokwi.com
2. New Project -> Arduino Uno
3. Copy/paste the code
4. Start Simulation

### ✅ 15. Button-Controlled LED
- **File**: `15_gpio_control/button_led.ino`
- **Platform**: Arduino Uno
- **Description**: Toggle LED on each button press, with debouncing

**Circuit Setup (Wokwi):**
- Button: Pin 2 <-> GND
- LED: Pin 13 (built-in LED)

### ✅ 16. Serial Calculator
- **File**: `16_serial_comm/serial_calculator.ino`
- **Platform**: Arduino Uno
- **Description**: Calculate expressions entered via Serial Monitor

**Usage Example:**
```
Simple Serial Calculator
Enter expression (e.g., 10 + 5)
---------------------------------
10 + 5
10 + 5 = 15.00
---------------------------------
20 * 3
20 * 3 = 60.00
```

---

## Batch Compilation

### Compile All C Programs
```bash
cd examples
make c-programs
```

### Compile Only Multithreaded Programs
```bash
make thread-programs
```

### Compile Everything
```bash
make
```

### Clean Up
```bash
make clean
```

### Run Individually
```bash
make run-calculator   # Run calculator
make run-guess        # Run number guessing game
make run-array        # Run dynamic array
make run-list         # Run linked list
make run-bit          # Run bit operations
make run-thread       # Run thread
```

---

## Upcoming Examples

The following examples are in the study materials but have not yet been written as code files:

### Planned List
- [ ] 04. Address Book (address_book.c)
- [ ] 07. File Encryption (file_crypto.c)
- [ ] 08. Stack and Queue (stack_queue.c)
- [ ] 09. Hash Table (hash_table.c)
- [ ] 10. Snake Game (snake_game.c)
- [ ] 11. Mini Shell (minishell.c)

These examples can be written by referring to the complete code in the study materials.

---

## Test Status

| Example | Compile | Run | Status |
|---------|---------|-----|--------|
| calculator | ✅ | ✅ | Working correctly |
| number_guess | ✅ | ⏸️ | Interactive |
| dynamic_array | ✅ | ✅ | Working correctly |
| linked_list | ✅ | ✅ | Working correctly |
| thread_basic | ✅ | ⏸️ | Requires pthread |
| bit_manipulation | ✅ | ✅ | Working correctly |
| blink.ino | - | 🌐 | Wokwi simulator |
| button_led.ino | - | 🌐 | Wokwi simulator |
| serial_calculator.ino | - | 🌐 | Wokwi simulator |

✅ = Test completed
⏸️ = Interactive/requires special environment
🌐 = Web simulator
