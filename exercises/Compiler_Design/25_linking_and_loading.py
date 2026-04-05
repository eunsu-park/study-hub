"""
Exercises for Lesson 25: Linking and Loading
Topic: Compiler_Design

Demonstrates ELF concepts, symbol resolution, and dynamic loading.
"""


def exercise_1():
    """Explore ELF structure commands."""
    print("Exercise 1: ELF Exploration Commands")
    print()
    commands = [
        "# Compile a test program",
        "gcc -c main.c -o main.o",
        "gcc main.o -o main",
        "",
        "# View ELF header",
        "readelf -h main.o",
        "",
        "# List sections",
        "readelf -S main.o",
        "",
        "# View symbol table",
        "nm main.o",
        "readelf -s main.o",
        "",
        "# View relocations",
        "readelf -r main.o",
        "",
        "# Disassemble",
        "objdump -d main.o",
        "",
        "# View program headers (executable only)",
        "readelf -l main",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    print()


def exercise_2():
    """Trace symbol resolution across object files."""
    print("Exercise 2: Symbol Resolution")
    print()
    print("  Source files:")
    print("    main.c: extern int x; void foo(); int main() { foo(); return x; }")
    print("    foo.c:  extern int x; void foo() { x = 42; }")
    print("    data.c: int x = 0;")
    print()

    symbols = {
        'main.o': {'defined': {'main'}, 'undefined': {'x', 'foo'}},
        'foo.o':  {'defined': {'foo'},  'undefined': {'x'}},
        'data.o': {'defined': {'x'},    'undefined': set()},
    }

    all_defined = {}
    all_undefined = set()
    for obj, info in symbols.items():
        print(f"  {obj}: defined={info['defined']}, undefined={info['undefined']}")
        for sym in info['defined']:
            all_defined[sym] = obj
        all_undefined |= info['undefined']

    print(f"\n  Resolution:")
    for sym in sorted(all_undefined):
        if sym in all_defined:
            print(f"    {sym} -> {all_defined[sym]}")
        else:
            print(f"    {sym} -> UNRESOLVED!")

    unresolved = all_undefined - set(all_defined.keys())
    if not unresolved:
        print(f"\n  All symbols resolved successfully.")
    print()


def exercise_3():
    """Compare static vs dynamic linking."""
    print("Exercise 3: Static vs Dynamic Linking Comparison")
    print()
    comparison = [
        "# Static linking:",
        "gcc -static main.c -o main_static",
        "ls -lh main_static   # ~1-2 MB (includes libc)",
        "ldd main_static      # 'not a dynamic executable'",
        "file main_static     # 'statically linked'",
        "",
        "# Dynamic linking (default):",
        "gcc main.c -o main_dynamic",
        "ls -lh main_dynamic  # ~16-20 KB",
        "ldd main_dynamic     # shows libc.so, ld-linux.so, etc.",
        "file main_dynamic    # 'dynamically linked'",
    ]
    for line in comparison:
        print(f"  {line}")
    print()


def exercise_4():
    """PLT tracing."""
    print("Exercise 4: PLT/GOT Tracing")
    print()
    commands = [
        "# Trace library calls (PLT):",
        "ltrace ./program",
        "",
        "# Verbose dynamic linker debug:",
        "LD_DEBUG=bindings ./program 2>&1 | head -20",
        "",
        "# Show PLT entries:",
        "objdump -d -j .plt program",
        "",
        "# Show GOT entries:",
        "objdump -R program",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    print()


def exercise_5():
    """Plugin system with dlopen/dlsym (C code)."""
    print("Exercise 5: Plugin System (C code)")
    print()
    plugin_c = '''
// plugin.c - compiled as shared library
// gcc -shared -fPIC -o libplugin.so plugin.c
int process(int x) {
    return x * x;
}
'''.strip()

    main_c = '''
// main.c - loads plugin at runtime
#include <stdio.h>
#include <dlfcn.h>

int main() {
    void *handle = dlopen("./libplugin.so", RTLD_LAZY);
    if (!handle) { fprintf(stderr, "%s\\n", dlerror()); return 1; }

    typedef int (*process_fn)(int);
    process_fn proc = (process_fn)dlsym(handle, "process");
    if (!proc) { fprintf(stderr, "%s\\n", dlerror()); return 1; }

    printf("process(7) = %d\\n", proc(7));  // prints 49
    dlclose(handle);
    return 0;
}
// gcc main.c -ldl -o main
'''.strip()

    print("  plugin.c:")
    for line in plugin_c.split('\n'):
        print(f"    {line}")
    print()
    print("  main.c:")
    for line in main_c.split('\n'):
        print(f"    {line}")
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
