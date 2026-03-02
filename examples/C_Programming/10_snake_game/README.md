# Snake Game Examples

Terminal-based snake game project example files.

## File Structure

### 1. `ansi_demo.c`
ANSI escape codes demonstration program
- Demonstrates cursor movement, colors, box drawing, etc.

**Compile and Run:**
```bash
gcc -o ansi_demo ansi_demo.c
./ansi_demo
```

### 2. `input_demo.c`
Asynchronous keyboard input demonstration program
- Non-blocking input handling using termios
- Arrow keys and WASD keys supported

**Compile and Run:**
```bash
gcc -o input_demo input_demo.c
./input_demo
```

**Controls:**
- Arrow keys or WASD: Move
- Space: Jump
- Q: Quit

### 3. `snake_types.h`
Snake game data structure definitions
- All types and constants needed for the game
- Used by including in other files

### 4. `snake.c`
Snake game core logic implementation
- Snake creation/destruction, movement, collision detection, etc.
- Modular implementation using `snake_types.h`

**Compile (cannot run standalone):**
```bash
# Must be linked with other files
gcc -c snake.c -o snake.o
```

### 5. `snake_game.c`
Complete snake game (using ANSI escape codes)
- Fully functional standalone game
- No external libraries required

**Compile and Run:**
```bash
gcc -o snake_game snake_game.c
./snake_game
```

**Controls:**
- Arrow keys or WASD: Move snake
- P: Pause
- Q: Quit
- R: Restart (after game over)

**Features:**
- Score system
- Speed increase (gets faster as you eat food)
- High score saving (`.snake_highscore` file)
- Pause
- Color display

### 6. `snake_ncurses.c`
Enhanced version using the NCurses library

**⚠️ This file requires the ncurses library!**

**Library Installation:**
```bash
# macOS
brew install ncurses

# Ubuntu/Debian
sudo apt install libncurses5-dev

# Fedora/RHEL
sudo dnf install ncurses-devel
```

**Compile and Run:**
```bash
# macOS
gcc -o snake_ncurses snake_ncurses.c -lncurses

# Linux
gcc -o snake_ncurses snake_ncurses.c -lncurses

# Run
./snake_ncurses
```

**Controls:**
- Arrow keys or WASD: Move snake
- P: Pause
- Q: Quit
- R: Restart (after game over)

**Advantages of the ncurses version:**
- Cleaner screen rendering
- Automatic buffering and flicker prevention
- Standard box drawing characters
- Better color management

## Learning Order

1. **`ansi_demo.c`** - Understand ANSI escape codes
2. **`input_demo.c`** - Understand asynchronous input handling
3. **`snake_types.h`** - Examine game data structures
4. **`snake.c`** - Learn game logic implementation
5. **`snake_game.c`** - Analyze the complete game
6. **`snake_ncurses.c`** - NCurses library usage (optional)

## Development Environment

- C11 standard
- POSIX-compatible systems (Linux, macOS, BSD)
- Terminal: UTF-8 support required
- Compiler: GCC or Clang

## References

This example is based on the following study material:
- `/content/ko/C_Programming/11_Project_Snake_Game.md`

## Troubleshooting

### Problem: Box-drawing characters appear broken
**Solution:** Verify that the terminal supports UTF-8
```bash
echo $LANG
# Example output: en_US.UTF-8 or ko_KR.UTF-8
```

### Problem: Key input is not working
**Solution:** Verify that the terminal supports ANSI escape sequences

### Problem: ncurses link error
**Solution:** Verify that the ncurses development package is installed
```bash
# macOS
brew list ncurses

# Ubuntu
dpkg -l | grep libncurses
```

## License

Free to use for educational purposes.
