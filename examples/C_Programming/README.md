# C Programming Example Code

This folder contains all example code from the C programming study materials.

## Directory Structure

```
examples/
├── 02_calculator/          # Calculator
├── 03_number_guess/        # Number Guessing Game
├── 04_address_book/        # Address Book
├── 05_dynamic_array/       # Dynamic Array
├── 06_linked_list/         # Linked List
├── 07_file_crypto/         # File Encryption
├── 08_stack_queue/         # Stack and Queue
├── 09_hash_table/          # Hash Table
├── 10_snake_game/          # Snake Game
├── 11_minishell/           # Mini Shell
├── 12_multithread/         # Multithreading
├── 13_embedded_basic/      # Embedded Basics (Arduino)
├── 14_bit_operations/      # Bit Operations
├── 15_gpio_control/        # GPIO Control (Arduino)
└── 16_serial_comm/         # Serial Communication (Arduino)
```

## How to Compile

### C Programs (Desktop)

```bash
# Basic compilation
gcc program.c -o program

# With warnings
gcc -Wall -Wextra program.c -o program

# With debug info
gcc -g program.c -o program

# Optimized
gcc -O2 program.c -o program
```

### Multithreaded Programs

```bash
# Linux
gcc program.c -o program -pthread

# macOS
gcc program.c -o program -lpthread
```

### Arduino Programs

Arduino programs (.ino) can be run using the following methods:

1. **Arduino IDE**
   - Open the file
   - Select board (Tools -> Board -> Arduino Uno)
   - Click Upload button

2. **Wokwi Simulator** (Recommended)
   - Visit https://wokwi.com
   - New Project -> Arduino Uno
   - Copy/paste the code
   - Start Simulation

3. **PlatformIO (VS Code)**
   ```bash
   pio run
   pio run --target upload
   ```

## Project Descriptions

| Project | Difficulty | Key Concepts |
|---------|------------|--------------|
| 02. Calculator | ⭐ | Functions, switch-case, scanf |
| 03. Number Guessing | ⭐ | Loops, random, conditionals |
| 04. Address Book | ⭐⭐ | Structs, arrays, file I/O |
| 05. Dynamic Array | ⭐⭐ | malloc, realloc, free |
| 06. Linked List | ⭐⭐⭐ | Pointers, dynamic data structures |
| 07. File Encryption | ⭐⭐ | File handling, bit operations |
| 08. Stack/Queue | ⭐⭐ | Data structures, LIFO/FIFO |
| 09. Hash Table | ⭐⭐⭐ | Hashing, collision handling |
| 10. Snake Game | ⭐⭐⭐ | Terminal control, game loop |
| 11. Mini Shell | ⭐⭐⭐⭐ | fork, exec, pipes |
| 12. Multithreading | ⭐⭐⭐⭐ | pthread, synchronization |
| 13. Embedded Basics | ⭐ | Arduino, GPIO |
| 14. Bit Operations | ⭐⭐ | Bit masking, registers |
| 15. GPIO Control | ⭐⭐ | LED, buttons, debouncing |
| 16. Serial Communication | ⭐⭐ | UART, command parsing |

## Learning Order

### Beginner
1. Calculator
2. Number Guessing
3. Address Book

### Intermediate
4. Dynamic Array
5. Linked List
6. File Encryption
7. Stack and Queue
8. Hash Table

### Advanced
9. Snake Game
10. Mini Shell
11. Multithreading

### Embedded (Arduino)
12. Embedded Basics
13. Bit Operations
14. GPIO Control
15. Serial Communication

## Execution Examples

### Calculator
```bash
cd 02_calculator
gcc calculator.c -o calculator
./calculator
```

### Multithreading
```bash
cd 12_multithread
gcc thread_basic.c -o thread_basic -pthread
./thread_basic
```

### Arduino (Wokwi)
1. Copy the code
2. Create a new project at https://wokwi.com
3. Paste and run

## Troubleshooting

### Compilation Errors
- `undefined reference to 'pthread_create'`: Add `-pthread` flag
- `implicit declaration of function`: Add the header file
- `permission denied`: `chmod +x program`

### Runtime Errors
- Segmentation fault: Check pointers, use valgrind
- Bus error: Invalid memory access
- Memory leak: Check with valgrind

## Additional Resources

- [C Reference](https://en.cppreference.com/w/c)
- [Arduino Reference](https://www.arduino.cc/reference/en/)
- [Wokwi Simulator](https://wokwi.com)
