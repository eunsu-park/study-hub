# IDL Basics

IDL (Interactive Data Language) is a programming language and environment widely used in solar physics, space science, astronomy, and Earth sciences. Originally developed by Research Systems Inc. (now Harris Geospatial Solutions, part of NV5 Geospatial), IDL excels at array-oriented numerical computation, image processing, and scientific visualization. GDL (GNU Data Language) is a free, open-source drop-in replacement that is compatible with most IDL syntax and routines.

This topic covers IDL fundamentals from installation and basic syntax through file I/O, plotting, and a hands-on solar physics project. Whether you are working with satellite data, telescope observations, or numerical simulations, these lessons will give you the skills to read, process, and visualize scientific data using IDL or GDL.

## What You'll Learn

This topic provides hands-on coverage of:
- **Getting Started**: Installation (IDL and GDL), IDLDE, command-line usage, and batch mode
- **Core Language**: Variables, data types, arrays, operators, and control flow
- **Procedures and Functions**: Writing reusable IDL programs with keywords and positional parameters
- **Strings**: String manipulation functions, formatting, and regular expressions
- **File I/O**: Text files, binary files, SAVE/RESTORE, and CSV reading
- **Structures**: Anonymous and named structures, structure arrays, and dynamic construction
- **Plotting**: PLOT, OPLOT, XYOUTS, PostScript output, and publication-quality figures
- **Image Display**: TV, TVSCL, color tables, byte scaling, and device-independent graphics
- **FITS Files**: Reading and writing FITS files, header manipulation, multi-extension FITS
- **Date and Time**: Julian dates, time parsing, and formatting for plots
- **Debugging**: STOP, HELP, RETALL, memory management, and coding best practices
- **Project**: Building a complete solar light curve from FITS data

## Prerequisites

- [Programming](../Programming/00_Overview.md) — Familiarity with general programming concepts (variables, control flow, functions)

No prior IDL experience is required. If you understand basic programming concepts such as variables, loops, and functions, you are ready to begin.

## Learning Roadmap

```
                          IDL Basics — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 01 Getting    │──▶│ 02 Variables &   │──▶│ 03 Arrays &            │  │
  │  │    Started    │   │    Data Types    │   │    Operations          │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 06 Procedures │◀──│ 05 Control       │◀──│ 04 Operators &         │  │
  │  │  & Functions  │   │    Flow          │   │    Expressions         │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 07 String    │──▶│ 08 File I/O      │──▶│ 09 Structures          │  │
  │  │  Processing  │   │                  │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 12 FITS File │◀──│ 11 Image         │◀──│ 10 Basic               │  │
  │  │  Handling    │   │    Display       │   │    Plotting            │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 13 Date &    │──▶│ 14 Debugging &   │──▶│ 15 Project: Solar      │  │
  │  │    Time      │   │  Best Practices  │   │    Light Curve         │  │
  │  └──────────────┘   └──────────────────┘   └────────────────────────┘  │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Getting Started](01_Getting_Started.md) | ⭐ | Installation, IDLDE, command line, Hello World, GDL, batch mode |
| 02 | [Variables and Data Types](02_Variables_and_Data_Types.md) | ⭐ | BYTE, INT, LONG, FLOAT, DOUBLE, COMPLEX, STRING, type conversion |
| 03 | [Arrays and Operations](03_Arrays_and_Operations.md) | ⭐ | Array creation, indexing, slicing, WHERE, REFORM, array math |
| 04 | [Operators and Expressions](04_Operators_and_Expressions.md) | ⭐ | Arithmetic, relational, logical, bitwise, string concatenation |
| 05 | [Control Flow](05_Control_Flow.md) | ⭐ | IF/THEN/ELSE, FOR, WHILE, REPEAT, CASE, SWITCH, BEGIN/END |
| 06 | [Procedures and Functions](06_Procedures_and_Functions.md) | ⭐ | PRO, FUNCTION, keywords, _EXTRA, scope, COMMON blocks |
| 07 | [String Processing](07_String_Processing.md) | ⭐ | STRMID, STRPOS, STRSPLIT, STRJOIN, FORMAT, STREGEX |
| 08 | [File I/O](08_File_IO.md) | ⭐ | OPENR/OPENW, GET_LUN, READF/PRINTF, binary I/O, SAVE/RESTORE |
| 09 | [Structures](09_Structures.md) | ⭐⭐ | Anonymous/named structures, CREATE_STRUCT, structure arrays |
| 10 | [Basic Plotting](10_Basic_Plotting.md) | ⭐ | PLOT, OPLOT, XYOUTS, axis keywords, PostScript output |
| 11 | [Image Display](11_Image_Display.md) | ⭐⭐ | TV, TVSCL, LOADCT, BYTSCL, CONGRID, REBIN, color tables |
| 12 | [FITS File Handling](12_FITS_File_Handling.md) | ⭐⭐ | READFITS, WRITEFITS, headers, MRDFITS, multi-extension FITS |
| 13 | [Date and Time](13_Date_and_Time.md) | ⭐ | SYSTIME, JULDAY, CALDAT, time parsing, ANYTIM |
| 14 | [Debugging and Best Practices](14_Debugging_and_Best_Practices.md) | ⭐⭐ | STOP, HELP, RETALL, HEAP_GC, coding conventions, vectorization |
| 15 | [Project: Solar Light Curve](15_Project_Solar_Light_Curve.md) | ⭐⭐ | FITS reading, time series, publication-quality plots, PostScript |

## Recommended Learning Order

Follow the lessons sequentially from 01 through 15. Each lesson builds on concepts introduced in the previous one:

1. **Environment Setup (Lesson 1)**: Get IDL or GDL installed and running
2. **Language Fundamentals (Lessons 2-5)**: Variables, arrays, operators, and control flow form the backbone of every IDL program
3. **Modular Code (Lesson 6)**: Organize code into procedures and functions
4. **String Handling (Lesson 7)**: Parse and format text data
5. **Data I/O and Structures (Lessons 8-9)**: Read/write files and organize complex data
6. **Visualization (Lessons 10-11)**: Create plots and display images
7. **Scientific Data (Lessons 12-13)**: Work with FITS files and date/time operations
8. **Professional Skills (Lesson 14)**: Debug code and follow best practices
9. **Capstone Project (Lesson 15)**: Integrate everything into a complete solar physics workflow

## Environment Setup

### Option 1: IDL (Commercial)

IDL is a commercial product from NV5 Geospatial (formerly Harris Geospatial). A license is required.

```
Download: https://www.nv5geospatialsoftware.com/Products/IDL
```

### Option 2: GDL (Free, Open Source)

GDL (GNU Data Language) is a free, open-source drop-in replacement compatible with most IDL syntax.

```bash
# macOS (Homebrew)
brew install gnudatalanguage

# Ubuntu / Debian
sudo apt-get install gnudatalanguage

# Fedora / RHEL
sudo dnf install gdl

# From source
git clone https://github.com/gnudatalanguage/gdl.git
cd gdl && mkdir build && cd build
cmake .. && make -j4 && sudo make install
```

Verify your installation:

```bash
# IDL
idl -e "PRINT, 'Hello from IDL'"

# GDL
gdl -e "PRINT, 'Hello from GDL'"
```

Example code for each lesson is available in `examples/IDL_Basics/`.

## Related Materials

- [Solar Physics](../Solar_Physics/00_Overview.md) — Solar observation data analysis and heliophysics
- [Space Weather](../Space_Weather/00_Overview.md) — Space weather modeling and prediction
- [Programming](../Programming/00_Overview.md) — Language-independent programming concepts

---

**License**: Content licensed under CC BY-NC 4.0
