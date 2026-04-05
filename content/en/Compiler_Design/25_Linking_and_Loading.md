# Linking and Loading

**Previous**: [24. Closure Conversion](./24_Closure_Conversion.md) | **Next**: [26. Debug Information](./26_Debug_Information.md)

---

After compilation produces object files, the linker combines them into an executable or shared library, and the loader maps the result into memory for execution. Linking and loading bridge the gap between compiled object code and running programs. Understanding this process is essential for diagnosing build errors, optimizing startup time, managing shared libraries, and working with complex software systems.

This lesson covers object file formats (ELF), symbol resolution, static and dynamic linking, relocation, position-independent code, and the runtime loader.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [11. Code Generation](./11_Code_Generation.md), [10. Runtime Environments](./10_Runtime_Environments.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the structure of ELF object files and executables
2. Explain symbol resolution and how the linker resolves references
3. Distinguish between static linking and dynamic linking
4. Understand relocation and position-independent code (PIC)
5. Explain how the dynamic linker/loader (ld.so) works at runtime
6. Diagnose common linking errors and library issues

---

## Table of Contents

1. [Object Files](#1-object-files)
2. [ELF Format](#2-elf-format)
3. [Symbols and Symbol Tables](#3-symbols-and-symbol-tables)
4. [Static Linking](#4-static-linking)
5. [Relocation](#5-relocation)
6. [Dynamic Linking](#6-dynamic-linking)
7. [Position-Independent Code](#7-position-independent-code)
8. [The Runtime Loader](#8-the-runtime-loader)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Object Files

### 1.1 Types of Object Files

| Type | Description | Extension |
|------|-------------|-----------|
| **Relocatable** | Compiler output; not yet linked | `.o` |
| **Executable** | Fully linked; ready to run | (none or `.exe`) |
| **Shared object** | Dynamic library | `.so` (Linux), `.dylib` (macOS) |
| **Core dump** | Process memory snapshot | `core` |

### 1.2 From Source to Executable

```
source.c  -->  [Compiler]  -->  source.o  (relocatable object)
                                    |
lib.c     -->  [Compiler]  -->  lib.o     (relocatable object)
                                    |
                               [Linker]
                                    |
                                    v
                              a.out (executable)
```

### 1.3 Sections

Object files are divided into **sections**:

```
┌──────────────────┐
│   ELF Header     │  Magic number, type, architecture
├──────────────────┤
│ .text            │  Executable machine code
├──────────────────┤
│ .data            │  Initialized global variables
├──────────────────┤
│ .bss             │  Uninitialized global variables (zero-filled)
├──────────────────┤
│ .rodata          │  Read-only data (string literals, constants)
├──────────────────┤
│ .symtab          │  Symbol table
├──────────────────┤
│ .strtab          │  String table (symbol names)
├──────────────────┤
│ .rel.text        │  Relocation entries for .text
├──────────────────┤
│ .rel.data        │  Relocation entries for .data
├──────────────────┤
│ Section headers  │  Metadata about all sections
└──────────────────┘
```

---

## 2. ELF Format

### 2.1 ELF Header

```c
// Simplified ELF header structure
typedef struct {
    unsigned char e_ident[16];  // Magic: 0x7f 'E' 'L' 'F'
    uint16_t e_type;            // ET_REL, ET_EXEC, ET_DYN
    uint16_t e_machine;         // EM_X86_64, EM_AARCH64, etc.
    uint32_t e_version;         // EV_CURRENT
    uint64_t e_entry;           // Entry point virtual address
    uint64_t e_phoff;           // Program header table offset
    uint64_t e_shoff;           // Section header table offset
    uint16_t e_phnum;           // Number of program headers
    uint16_t e_shnum;           // Number of section headers
} Elf64_Ehdr;
```

### 2.2 Inspecting ELF Files

```bash
# View ELF header
readelf -h program

# List sections
readelf -S program.o

# View symbol table
readelf -s program.o
# or
nm program.o

# View relocations
readelf -r program.o

# Disassemble .text section
objdump -d program.o

# View program headers (segments)
readelf -l program

# View dynamic section
readelf -d program
```

### 2.3 Sections vs. Segments

```
Object files have SECTIONS (compile-time view):
  .text, .data, .bss, .rodata, ...

Executables have SEGMENTS (runtime view):
  LOAD segments group sections by permissions:

  Segment 1 (READ + EXECUTE):  .text, .rodata
  Segment 2 (READ + WRITE):    .data, .bss
  Segment 3 (READ):            .dynamic, .dynsym
```

---

## 3. Symbols and Symbol Tables

### 3.1 What is a Symbol?

A **symbol** is a named entity: function, global variable, or label:

```c
// main.c
extern int global_var;       // undefined symbol (reference)
int local_var = 42;          // defined symbol (global)
static int static_var = 10;  // defined symbol (local -- not exported)

void my_function(void) {}    // defined symbol (global)
static void helper(void) {}  // defined symbol (local)

int main() {
    my_function();
    printf("hello\n");       // undefined symbol (reference)
}
```

### 3.2 Symbol Table Entries

```c
typedef struct {
    uint32_t st_name;     // Index into string table
    uint8_t  st_info;     // Binding (LOCAL/GLOBAL/WEAK) + Type (FUNC/OBJECT)
    uint8_t  st_other;    // Visibility
    uint16_t st_shndx;    // Section index (or SHN_UNDEF for undefined)
    uint64_t st_value;    // Value (address or offset)
    uint64_t st_size;     // Size of the symbol
} Elf64_Sym;
```

### 3.3 Symbol Binding

| Binding | Meaning |
|---------|---------|
| `LOCAL` | Visible only within the object file (e.g., `static`) |
| `GLOBAL` | Visible to all object files |
| `WEAK` | Like global, but can be overridden by a strong (global) symbol |

### 3.4 Symbol Resolution Example

```bash
$ nm main.o
                 U global_var      # Undefined
0000000000000000 D local_var       # Defined in .data
0000000000000004 d static_var      # Local (lowercase = local)
0000000000000000 T my_function     # Defined in .text
0000000000000010 t helper          # Local in .text
0000000000000020 T main            # Defined in .text
                 U printf          # Undefined
```

---

## 4. Static Linking

### 4.1 How the Linker Works

```
Input:  main.o, lib.o, libc.a
Output: executable

Steps:
1. Read all input object files
2. Resolve symbol references:
   - For each undefined symbol, find a definition in another object
3. Perform relocations:
   - Patch addresses now that final layout is known
4. Write the executable with all sections merged
```

### 4.2 Symbol Resolution Algorithm

```python
def resolve_symbols(object_files):
    """
    Resolve symbols across multiple object files.
    """
    defined = {}     # symbol name -> (object, section, value)
    undefined = {}   # symbol name -> list of (object, relocation)

    for obj in object_files:
        for sym in obj.symbols:
            if sym.is_defined:
                if sym.name in defined and sym.binding == 'GLOBAL':
                    if defined[sym.name].binding == 'GLOBAL':
                        raise LinkError(f"Multiple definition of {sym.name}")
                    # Weak overridden by global
                defined[sym.name] = sym
            else:
                undefined.setdefault(sym.name, []).append((obj, sym))

    # Check for unresolved symbols
    for name, refs in undefined.items():
        if name not in defined:
            raise LinkError(f"Undefined reference to {name}")

    return defined
```

### 4.3 Static Libraries (.a)

A static library is an archive of object files:

```bash
# Create static library
ar rcs libmath.a add.o multiply.o divide.o

# Link with static library
gcc main.o -L. -lmath -o program

# The linker only pulls in .o files that resolve undefined symbols
```

### 4.4 Archive Member Selection

```python
def link_with_archives(objects, archives):
    """
    Link objects with static library archives.
    Only include archive members that resolve undefined symbols.
    """
    undefined = collect_undefined(objects)

    changed = True
    while changed:
        changed = False
        for archive in archives:
            for member in archive.members:
                # Include member if it defines any undefined symbol
                member_defines = get_defined_symbols(member)
                if member_defines & undefined:
                    objects.append(member)
                    undefined = collect_undefined(objects)
                    changed = True
```

---

## 5. Relocation

### 5.1 Why Relocation is Needed

When the compiler generates object code, it doesn't know the final addresses:

```asm
; In main.o:
call printf          ; Address of printf unknown at compile time
mov eax, [global_var] ; Address of global_var unknown
```

The compiler emits **relocation entries** that tell the linker where to patch:

```c
typedef struct {
    uint64_t r_offset;   // Where to apply the relocation
    uint64_t r_info;     // Symbol index + relocation type
    int64_t  r_addend;   // Constant addend
} Elf64_Rela;
```

### 5.2 Common Relocation Types (x86-64)

| Type | Meaning | Calculation |
|------|---------|-------------|
| `R_X86_64_64` | Absolute 64-bit address | S + A |
| `R_X86_64_PC32` | PC-relative 32-bit | S + A - P |
| `R_X86_64_PLT32` | PLT entry (function call) | L + A - P |
| `R_X86_64_GOT64` | GOT entry | G + A |

Where S = symbol value, A = addend, P = place being relocated, L = PLT entry, G = GOT offset.

### 5.3 Relocation Process

```python
def perform_relocations(sections, symbols, relocations):
    """
    Apply relocations to patch addresses in code and data.
    """
    for reloc in relocations:
        sym = symbols[reloc.symbol_index]
        section = sections[reloc.section]
        offset = reloc.offset
        addend = reloc.addend

        if reloc.type == R_X86_64_PC32:
            # PC-relative: target - current_position
            value = sym.final_address + addend - (section.base + offset)
            section.write_i32(offset, value)

        elif reloc.type == R_X86_64_64:
            # Absolute 64-bit address
            value = sym.final_address + addend
            section.write_i64(offset, value)

        elif reloc.type == R_X86_64_PLT32:
            # PLT entry for function call
            plt_entry = get_plt_entry(sym)
            value = plt_entry + addend - (section.base + offset)
            section.write_i32(offset, value)
```

---

## 6. Dynamic Linking

### 6.1 Motivation

Static linking copies all library code into the executable. Dynamic linking shares libraries at runtime:

| Aspect | Static Linking | Dynamic Linking |
|--------|---------------|-----------------|
| Executable size | Large | Small |
| Memory sharing | No | Shared between processes |
| Updates | Recompile to update | Replace .so file |
| Startup time | Faster | Slower (symbol resolution) |
| Distribution | Self-contained | Requires library installed |

### 6.2 Shared Libraries

```bash
# Create shared library
gcc -shared -fPIC -o libmath.so add.c multiply.c

# Link with shared library
gcc main.c -L. -lmath -o program

# Run (library must be findable)
LD_LIBRARY_PATH=. ./program
```

### 6.3 GOT and PLT

Dynamic linking uses two key data structures:

**GOT (Global Offset Table)**: stores resolved addresses of external variables and functions.

**PLT (Procedure Linkage Table)**: stub code that lazily resolves function addresses.

```
First call to printf():           Second call to printf():
  call printf@PLT                   call printf@PLT
       |                                 |
  PLT stub:                         PLT stub:
    jmp *GOT[printf]                  jmp *GOT[printf]
       |                                 |
  GOT[printf] = resolver            GOT[printf] = actual printf
       |                                 |
  Dynamic linker resolves            Direct jump to printf
  printf, patches GOT                (no resolver overhead)
       |
  Jump to actual printf
```

---

## 7. Position-Independent Code

### 7.1 What is PIC?

**Position-Independent Code** works correctly regardless of where it is loaded in memory. Required for shared libraries since different processes map them at different addresses.

### 7.2 PIC for Data Access

```asm
; Non-PIC (absolute address):
mov eax, [0x601020]        ; hardcoded address

; PIC (PC-relative via GOT):
mov rax, [rip + global_var@GOTPCREL]  ; GOT-relative
mov eax, [rax]                         ; load actual value
```

### 7.3 PIC for Function Calls

```asm
; Non-PIC:
call 0x400520              ; absolute address

; PIC (via PLT):
call printf@PLT            ; indirect through PLT/GOT
```

### 7.4 Generating PIC

```bash
# Compile with PIC (required for shared libraries)
gcc -fPIC -c library.c -o library.o

# -fPIC vs -fpic:
# -fPIC: always works (larger GOT allowed)
# -fpic: may use shorter instructions (limited GOT size)
```

---

## 8. The Runtime Loader

### 8.1 How Programs Start

When you run `./program`, the kernel:

1. Reads the ELF header
2. Maps segments into memory
3. Finds the dynamic linker path (from `.interp` section)
4. Transfers control to the dynamic linker (`ld-linux-x86-64.so.2`)

The dynamic linker then:

1. Loads all required shared libraries
2. Resolves symbols (eager or lazy)
3. Performs relocations
4. Calls initialization functions (`.init`, constructors)
5. Transfers control to the program's entry point

### 8.2 Library Search Order

```bash
# The dynamic linker searches for libraries in this order:
# 1. DT_RPATH in the executable (deprecated)
# 2. LD_LIBRARY_PATH environment variable
# 3. DT_RUNPATH in the executable
# 4. /etc/ld.so.cache (ldconfig cache)
# 5. /lib, /usr/lib (default paths)

# View library dependencies
ldd program

# Trace dynamic linker activity
LD_DEBUG=all ./program
```

### 8.3 dlopen: Runtime Loading

```c
#include <dlfcn.h>

// Load library at runtime
void *handle = dlopen("./libplugin.so", RTLD_LAZY);
if (!handle) {
    fprintf(stderr, "dlopen: %s\n", dlerror());
    exit(1);
}

// Look up symbol
typedef int (*plugin_fn)(int);
plugin_fn func = (plugin_fn)dlsym(handle, "process");
if (!func) {
    fprintf(stderr, "dlsym: %s\n", dlerror());
    exit(1);
}

// Use the function
int result = func(42);

// Unload
dlclose(handle);
```

---

## 9. Summary

- **Object files** contain compiled code organized into sections (.text, .data, .bss)
- **ELF** is the standard binary format on Linux, with headers describing sections and segments
- **Symbol tables** map names to addresses; the linker resolves undefined references
- **Static linking** copies library code into the executable at link time
- **Relocation** patches addresses in code once final layout is known
- **Dynamic linking** defers library loading to runtime, using GOT and PLT
- **Position-independent code** (PIC) enables shared libraries to be loaded at any address
- The **runtime loader** (ld.so) maps libraries, resolves symbols, and starts the program

---

## 10. Exercises

1. **Explore ELF**: Compile a small C program and examine its ELF structure using `readelf` and `objdump`.

2. **Symbol resolution**: Create three object files with cross-references and trace how the linker resolves them.

3. **Static vs. dynamic**: Build the same program with static and dynamic linking. Compare executable sizes and `ldd` output.

4. **PLT tracing**: Use `ltrace` or `LD_DEBUG=bindings` to observe lazy symbol resolution via the PLT.

5. **Plugin system**: Write a C program that loads a shared library at runtime using `dlopen`/`dlsym` and calls a function from it.

---

## 11. References

1. Levine, J. R. (1999). *Linkers and Loaders*. Morgan Kaufmann.
2. ELF Specification: https://refspecs.linuxfoundation.org/elf/elf.pdf
3. Drepper, U. (2011). "How To Write Shared Libraries." https://www.akkadia.org/drepper/dsohowto.pdf
4. Bryant, R. E., O'Hallaron, D. R. (2015). *Computer Systems: A Programmer's Perspective*. Chapter 7: Linking.
5. System V ABI: https://refspecs.linuxfoundation.org/SystemV/abi/

---

**Previous**: [24. Closure Conversion](./24_Closure_Conversion.md) | **Next**: [26. Debug Information](./26_Debug_Information.md)
