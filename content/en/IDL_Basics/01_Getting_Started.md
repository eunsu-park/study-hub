# Getting Started with IDL

**Next**: [Variables and Data Types](./02_Variables_and_Data_Types.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what IDL is and why it is used in scientific computing
2. Install IDL or GDL on your system
3. Navigate the IDLDE (IDL Development Environment) and command-line interface
4. Write and run your first IDL program using PRINT
5. Understand the difference between IDL and GDL
6. Use basic IDL commands such as .RUN, .COMPILE, and RETALL
7. Execute IDL programs in batch mode
8. Understand IDL licensing and alternatives

---

IDL (Interactive Data Language) has been a cornerstone of scientific computing since the 1970s. Developed originally by David Stern at the Laboratory for Atmospheric and Space Physics (LASP) at the University of Colorado, IDL became the standard tool for data analysis in solar physics, astronomy, space science, remote sensing, and medical imaging. Its array-oriented syntax, built-in visualization, and rich library of scientific routines make it uniquely suited for working with multidimensional datasets.

## Why IDL?

### Historical Significance

IDL has been the language of choice for major scientific missions and projects:

- **NASA Solar Dynamics Observatory (SDO)**: Data pipelines built in IDL
- **SOHO (Solar and Heliospheric Observatory)**: Primary analysis language
- **SolarSoft (SSW)**: A massive IDL library for solar physics with thousands of routines
- **Hubble Space Telescope**: Many analysis tools written in IDL
- **GOES X-ray Sensor**: Standard data analysis in IDL

### Key Strengths

- **Array-Oriented**: Operations on entire arrays without explicit loops
- **Built-in Visualization**: Publication-quality plotting with minimal code
- **Scientific Libraries**: Extensive math, statistics, image processing, and signal processing
- **FITS Support**: Native support for the FITS (Flexible Image Transport System) format
- **Interactive**: Test ideas at the command prompt, then save to scripts
- **Mature Ecosystem**: Decades of tested routines for space science

### IDL vs. Modern Alternatives

```
Language     │ License      │ Array Syntax │ Visualization │ FITS Support │ Legacy Code
─────────────┼──────────────┼──────────────┼───────────────┼──────────────┼────────────
IDL          │ Commercial   │ Excellent    │ Built-in      │ Native       │ Vast
GDL          │ Free (GPL)   │ Excellent    │ Built-in      │ Native       │ Compatible
Python       │ Free (BSD)   │ NumPy        │ Matplotlib    │ astropy.io   │ Growing
MATLAB       │ Commercial   │ Excellent    │ Built-in      │ Limited      │ Different
Julia        │ Free (MIT)   │ Excellent    │ Plots.jl      │ FITSIO.jl    │ Minimal
```

---

## Installing IDL

### Commercial IDL

IDL is distributed by NV5 Geospatial Solutions (formerly Harris Geospatial, Exelis VIS, ITT VIS, and Research Systems Inc.).

```
Official website: https://www.nv5geospatialsoftware.com/Products/IDL
```

#### License Types

- **Full License**: Complete IDL with all features
- **IDL Virtual Machine**: Free runtime to execute pre-compiled .sav files (no editing)
- **Student License**: Discounted for academic use
- **Floating License**: Shared across a network (common at universities/labs)

#### Installation on Linux

```bash
# Download the installer from NV5 website
# Run the installer
chmod +x idl_installer.sh
./idl_installer.sh

# Add IDL to your PATH (typical location)
export IDL_DIR=/usr/local/harris/idl
export PATH=$IDL_DIR/bin:$PATH

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export IDL_DIR=/usr/local/harris/idl' >> ~/.bashrc
echo 'export PATH=$IDL_DIR/bin:$PATH' >> ~/.bashrc
```

#### Installation on macOS

```bash
# Download the .dmg installer from NV5 website
# Mount and run the installer
# Default installation: /Applications/harris/idl

export IDL_DIR=/Applications/harris/idl
export PATH=$IDL_DIR/bin:$PATH
```

### GDL: The Free Alternative

GDL (GNU Data Language) is a free, open-source implementation that is compatible with most IDL syntax. For learning purposes, GDL is an excellent choice.

#### Installing GDL

```bash
# macOS (Homebrew)
brew install gnudatalanguage

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install gnudatalanguage

# Fedora / RHEL
sudo dnf install gdl

# Arch Linux
sudo pacman -S gnudatalanguage

# From source (for latest features)
git clone https://github.com/gnudatalanguage/gdl.git
cd gdl
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

#### GDL Compatibility Notes

GDL aims for full IDL compatibility. Most core language features work identically:

```
Feature                  │ GDL Status
─────────────────────────┼───────────────
Core language syntax      │ ✓ Full
Array operations          │ ✓ Full
PLOT, OPLOT, CONTOUR      │ ✓ Full
READFITS/WRITEFITS        │ ✓ Full (with astron lib)
String functions          │ ✓ Full
File I/O                  │ ✓ Full
Structures                │ ✓ Full
SAVE/RESTORE              │ ✓ Partial
WIDGET_* (GUI)            │ ✗ Limited
Some proprietary routines │ ✗ Not available
```

---

## The IDL Environment

### IDLDE (IDL Development Environment)

IDLDE is a graphical development environment that ships with commercial IDL. It provides:

- **Editor**: Syntax-highlighted code editor with multiple tabs
- **Console**: Interactive command prompt at the bottom
- **Variable Watch**: Inspect variable values during execution
- **Project Explorer**: Navigate files and directories
- **Profiler**: Measure execution time of routines

To launch IDLDE:

```bash
idlde    # Launch the graphical environment
```

### Command-Line Interface

For many users, the command-line interface is sufficient and often preferred:

```bash
# Launch IDL interactive session
idl

# Launch GDL interactive session
gdl
```

You will see a prompt like:

```
IDL>
```

or for GDL:

```
GDL>
```

### The IDL Prompt

At the IDL prompt, you can type commands interactively:

```idl
IDL> PRINT, 'Hello, World!'
Hello, World!

IDL> x = 42
IDL> PRINT, x
      42

IDL> PRINT, x * 2 + 10
      94
```

---

## Hello, World!

The simplest IDL program uses the `PRINT` procedure:

```idl
IDL> PRINT, 'Hello, World!'
Hello, World!
```

### PRINT Basics

`PRINT` is a built-in procedure that outputs values to the console:

```idl
; Print a string
PRINT, 'Welcome to IDL!'

; Print a number
PRINT, 3.14159

; Print multiple values (separated by spaces)
PRINT, 'The answer is', 42

; Print with FORMAT keyword for precise control
PRINT, FORMAT='("Pi = ", F8.5)', 3.14159

; Print multiple items with FORMAT
PRINT, FORMAT='("x = ", I4, "  y = ", F6.2)', 10, 3.14
```

### Your First Script

Create a file called `hello.pro`:

```idl
; hello.pro - My first IDL program
;
; This is a comment. Comments start with semicolons.

PRO hello
  PRINT, 'Hello, World!'
  PRINT, 'Welcome to IDL programming!'
  PRINT, 'Today we begin our journey into scientific computing.'

  ; Basic arithmetic
  x = 10
  y = 20
  PRINT, 'x + y =', x + y

  ; Array creation
  arr = [1, 2, 3, 4, 5]
  PRINT, 'Array:', arr
  PRINT, 'Sum:', TOTAL(arr)
  PRINT, 'Mean:', MEAN(arr)
END
```

Run it from the IDL prompt:

```idl
IDL> .RUN hello
% Compiled module: HELLO.
IDL> hello
Hello, World!
Welcome to IDL programming!
Today we begin our journey into scientific computing.
x + y =          30
Array:       1       2       3       4       5
Sum:      15.0000
Mean:      3.00000
```

---

## Essential Commands

### Compilation and Execution Commands

IDL uses dot-commands (commands prefixed with a period) for system-level operations:

```idl
; Compile a file
IDL> .COMPILE filename
; or
IDL> .COMPILE filename.pro

; Compile and run (for programs without a named procedure)
IDL> .RUN filename.pro

; Run a previously compiled procedure
IDL> procedure_name

; Run a procedure with arguments
IDL> procedure_name, arg1, arg2, keyword=value
```

### The Difference Between .RUN and .COMPILE

```idl
; .COMPILE only compiles — does not execute
IDL> .COMPILE hello
% Compiled module: HELLO.
; Now you must call it explicitly:
IDL> hello

; .RUN compiles and executes unnamed code blocks (main-level programs)
; If the file contains a named PRO or FUNCTION, .RUN just compiles it
IDL> .RUN hello
% Compiled module: HELLO.
```

### Session Management

```idl
; Reset the session — return to main level from any breakpoint or error
IDL> RETALL

; Continue execution after a STOP (breakpoint)
IDL> .CONTINUE
; or
IDL> .CON

; Step through code line by line
IDL> .STEP
; or
IDL> .S

; Exit IDL
IDL> EXIT
```

### Getting Help

```idl
; Display information about a variable
IDL> x = FINDGEN(10)
IDL> HELP, x
X               FLOAT     = Array[10]

; Get help on a routine (opens documentation in IDL 8+)
IDL> ? PLOT

; List all variables in the current scope
IDL> HELP

; Display memory usage
IDL> HELP, /MEMORY
```

---

## IDL Program Structure

### Main-Level Programs

A main-level program is a block of code without a PRO or FUNCTION declaration. It is executed directly with `.RUN`:

```idl
; main_example.pro - a main-level program
; No PRO or FUNCTION keyword — runs immediately with .RUN

x = FINDGEN(100)
y = SIN(x / 10.0)
PLOT, x, y, TITLE='Sine Wave'
PRINT, 'Plot complete.'

END
```

```idl
IDL> .RUN main_example
```

### Named Procedures

A named procedure is the most common program unit in IDL:

```idl
; greet.pro
PRO greet, name
  IF N_PARAMS() EQ 0 THEN name = 'World'
  PRINT, 'Hello, ' + name + '!'
END
```

```idl
IDL> .COMPILE greet
IDL> greet, 'Alice'
Hello, Alice!
IDL> greet
Hello, World!
```

### Named Functions

Functions return a value:

```idl
; add_numbers.pro
FUNCTION add_numbers, a, b
  RETURN, a + b
END
```

```idl
IDL> .COMPILE add_numbers
IDL> result = add_numbers(3, 4)
IDL> PRINT, result
       7
```

### File Naming Convention

- **One routine per file**: The filename should match the routine name
- **Lowercase with .pro extension**: `my_routine.pro` contains `PRO my_routine` or `FUNCTION my_routine`
- **IDL searches the path**: IDL automatically finds and compiles routines if they are on the `!PATH`

---

## The IDL Path

IDL uses `!PATH` to locate procedure and function files:

```idl
; View the current path
IDL> PRINT, !PATH

; Add a directory to the path
IDL> !PATH = '/home/user/my_idl_code:' + !PATH

; Or use EXPAND_PATH for recursive directory inclusion
IDL> !PATH = EXPAND_PATH('+/home/user/my_idl_code') + ':' + !PATH
```

You can also set the path in your IDL startup file:

```idl
; ~/.idl/idl/idl_startup.pro  (or set IDL_STARTUP environment variable)
!PATH = EXPAND_PATH('+~/idl_library') + ':' + !PATH
PRINT, 'IDL startup complete.'
```

For GDL, you can set `GDL_PATH` or `GDL_STARTUP`:

```bash
# In ~/.bashrc or ~/.zshrc
export GDL_PATH="+~/idl_library:+/usr/local/share/gnudatalanguage/lib"
export GDL_STARTUP=~/.gdl/gdl_startup.pro
```

---

## Batch Mode

You can run IDL scripts non-interactively using batch mode:

```bash
# Run a script and exit
idl -e "PRINT, 'Hello from batch mode'"

# Run a .pro file in batch mode (main-level program)
idl < my_script.pro

# Run with the @ command
idl -e "@my_script"

# GDL equivalent
gdl -e "PRINT, 'Hello from GDL batch mode'"
gdl < my_script.pro
```

### Using the @ Command

The `@` command executes a file as if its contents were typed at the prompt:

```idl
; At the IDL prompt
IDL> @my_script.pro

; This is different from .RUN — @ does not compile, it executes line by line
```

### Batch Script Example

Create `batch_example.pro`:

```idl
; batch_example.pro - Batch processing example
PRINT, 'Starting batch processing...'
PRINT, 'System time: ' + SYSTIME()

; Generate test data
n = 1000
x = FINDGEN(n)
y = SIN(2.0 * !PI * x / n) + RANDOMN(seed, n) * 0.1

; Calculate statistics
PRINT, 'Mean: ', MEAN(y)
PRINT, 'Std Dev: ', STDDEV(y)
PRINT, 'Min: ', MIN(y)
PRINT, 'Max: ', MAX(y)

PRINT, 'Batch processing complete.'
END
```

Run from the shell:

```bash
idl < batch_example.pro
```

---

## IDL System Variables

IDL has built-in system variables that control behavior. They all start with `!`:

```idl
; Mathematical constants
PRINT, !PI          ; 3.14159...
PRINT, !DTOR        ; Degrees to radians conversion factor
PRINT, !RADEG       ; Radians to degrees conversion factor

; Special values
PRINT, !VALUES.F_NAN      ; Float NaN
PRINT, !VALUES.F_INFINITY ; Float infinity
PRINT, !VALUES.D_NAN      ; Double NaN

; Graphics system variables
PRINT, !P.MULTI     ; Multi-panel plot settings
PRINT, !D.NAME      ; Current graphics device name
PRINT, !D.X_SIZE    ; Current device X size in pixels
PRINT, !D.Y_SIZE    ; Current device Y size in pixels

; Version information
HELP, !VERSION, /STRUCTURE
```

---

## Practical Exercise: First Steps

Try these commands at the IDL prompt to get comfortable:

```idl
; 1. Basic arithmetic
PRINT, 2 + 3
PRINT, 10.0 / 3.0
PRINT, 2^10

; 2. Create an array and compute statistics
data = [4.5, 3.2, 7.8, 1.1, 9.6, 5.3]
PRINT, 'Data:', data
PRINT, 'Sum:', TOTAL(data)
PRINT, 'Mean:', MEAN(data)
PRINT, 'Sorted:', data[SORT(data)]

; 3. Quick plot
x = FINDGEN(360)
y = SIN(x * !DTOR)
PLOT, x, y, TITLE='Sine Wave', XTITLE='Degrees', YTITLE='sin(x)'

; 4. Check system information
PRINT, 'IDL Version:', !VERSION.RELEASE
PRINT, 'OS:', !VERSION.OS
PRINT, 'Architecture:', !VERSION.ARCH
```

---

## Summary

| Concept | Description |
|---------|-------------|
| `PRINT` | Output values to the console |
| `.RUN` | Compile and run a file |
| `.COMPILE` | Compile without running |
| `RETALL` | Return to main level |
| `.CONTINUE` | Continue after STOP |
| `HELP` | Inspect variables and routines |
| `EXIT` | Quit IDL |
| `!PATH` | Search path for .pro files |
| `@filename` | Execute a file line by line |
| System variables | `!PI`, `!DTOR`, `!VALUES`, etc. |

You now have IDL or GDL installed and can write and run basic programs. In the next lesson, we will explore IDL's data types and variables in depth.

---

**Next**: [Variables and Data Types](./02_Variables_and_Data_Types.md)
