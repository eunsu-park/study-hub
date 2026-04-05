# Lesson 10: Runtime Environments

## Learning Objectives

After completing this lesson, you will be able to:

1. **Describe** the standard memory layout of a running program (code, static, stack, heap)
2. **Explain** activation records (stack frames) and their contents
3. **Compare** calling conventions (cdecl, stdcall, System V AMD64 ABI) and their implications
4. **Implement** parameter passing mechanisms: by value, by reference, and by name
5. **Handle** nested functions using access links and displays
6. **Distinguish** between static and dynamic scoping and their runtime implementations
7. **Describe** heap management strategies (free lists, buddy system, garbage collection)
8. **Simulate** a runtime call stack in Python

---

## 1. Storage Organization

### 1.1 The Memory Model

When a compiled program runs, the operating system allocates memory organized into distinct regions. Each region serves a specific purpose:

```
High addresses
┌─────────────────────┐
│       Stack         │  ← grows downward
│         │           │
│         ▼           │
│                     │
│         ▲           │
│         │           │
│        Heap         │  ← grows upward
├─────────────────────┤
│    Static/Global    │  ← fixed size at load time
│       Data          │
├─────────────────────┤
│    Read-Only Data   │  ← string literals, constants
├─────────────────────┤
│       Code          │  ← text segment (instructions)
│      (Text)         │
└─────────────────────┘
Low addresses
```

### 1.2 Code (Text) Segment

The **code segment** (or text segment) holds the machine instructions of the compiled program.

**Properties**:
- **Read-only**: Prevents accidental or malicious modification of instructions
- **Sharable**: Multiple processes running the same program can share a single copy
- **Fixed size**: Determined at compile/link time
- **Loaded once**: Mapped into memory when the process starts

### 1.3 Static/Global Data

The **static data** area stores global variables and variables declared as `static` within functions. This region has a fixed size determined at compile time.

It is typically subdivided:

| Sub-region | Contents | Example |
|------------|----------|---------|
| `.data` | Initialized global/static variables | `int count = 42;` |
| `.bss` | Uninitialized global/static variables (zero-filled) | `static int buffer[1024];` |
| `.rodata` | Read-only data: string literals, constants | `"Hello, World!"` |

**Addressing**: Variables in the static area have **absolute addresses** known at link time. The compiler generates direct address references.

### 1.4 The Stack

The **stack** is used for function call management. It stores:

- **Activation records** (stack frames) for each active function call
- **Local variables** of functions
- **Parameters** passed to functions
- **Return addresses** for resuming the caller
- **Saved registers** that the callee must preserve

The stack grows **downward** in most architectures (toward lower addresses). The **stack pointer** (SP) marks the current top of the stack, and the **frame pointer** (FP, also called the base pointer BP) marks a fixed reference point within the current frame.

### 1.5 The Heap

The **heap** is used for dynamically allocated memory -- data whose size or lifetime cannot be determined at compile time.

**Examples**:
- `malloc()` / `free()` in C
- `new` / `delete` in C++
- Object creation in Java, Python
- Dynamically sized arrays, linked lists, trees

The heap grows **upward** (toward higher addresses). Between the stack and heap is unused address space, providing room for both to grow.

### 1.6 Address Space Layout Randomization (ASLR)

Modern operating systems randomize the positions of the stack, heap, and shared libraries at each program execution. This makes it harder for attackers to predict memory addresses, mitigating buffer overflow exploits and return-oriented programming (ROP) attacks.

---

## 2. Activation Records (Stack Frames)

### 2.1 What Is an Activation Record?

Every time a function is called, a new **activation record** (or **stack frame**) is pushed onto the runtime stack. This record contains all the information needed to execute the function and return to the caller.

### 2.2 Structure of an Activation Record

A typical activation record looks like this (growing downward from high to low addresses):

```
┌───────────────────────┐  High address
│    Arguments passed    │  ← pushed by caller (if on stack)
│    by the caller       │
├───────────────────────┤
│    Return address      │  ← pushed by CALL instruction
├───────────────────────┤  ◀── Frame pointer (FP / BP)
│    Saved old FP        │  ← dynamic link (previous frame pointer)
├───────────────────────┤
│    Saved registers     │  ← callee-saved registers
├───────────────────────┤
│    Local variables     │  ← function's local storage
├───────────────────────┤
│    Temporaries         │  ← compiler-generated temporaries
├───────────────────────┤
│    Outgoing arguments  │  ← arguments for functions this
│    (if needed)         │     function calls
└───────────────────────┘  ◀── Stack pointer (SP)
                           Low address
```

### 2.3 Components in Detail

#### Return Address

The address of the instruction in the caller to which control should return after the callee finishes. On x86, the `CALL` instruction automatically pushes the return address onto the stack.

#### Dynamic Link (Saved Frame Pointer)

A pointer to the caller's activation record (specifically, the caller's frame pointer). This forms a **chain** of frames that can be traversed for debugging (stack unwinding).

```
       Caller's frame
       ┌──────────┐
FP ──▶ │ saved FP │ ───▶ Previous frame ...
       ├──────────┤
       │  locals  │
       └──────────┘
```

#### Static Link (Access Link)

Used for nested functions (discussed in Section 5). Points to the activation record of the lexically enclosing function.

#### Saved Registers

Registers that the callee is required to preserve (callee-saved registers). The function saves them at entry and restores them before returning.

#### Local Variables

Storage for the function's local variables, allocated at known offsets from the frame pointer.

#### Temporaries

Compiler-generated temporary values that do not fit in registers.

### 2.4 Accessing Local Variables

Local variables are accessed at fixed **offsets** from the frame pointer:

```
Variable x declared at offset -8 from FP:
    x = FP - 8

Parameter p declared at offset +16 from FP (above return address):
    p = FP + 16
```

The frame pointer provides a stable reference even as the stack pointer moves during function execution (e.g., when pushing arguments for a nested call).

### 2.5 Example: Function Call Sequence

Consider the following C code:

```c
int add(int a, int b) {
    int result = a + b;
    return result;
}

int main() {
    int x = 3;
    int y = 4;
    int z = add(x, y);
    return z;
}
```

The sequence of events during the call `add(x, y)`:

**1. Caller (main) -- Before the call**:
```
push y        ; push second argument (or use register)
push x        ; push first argument (or use register)
call add      ; push return address, jump to add
```

**2. Callee (add) -- Function prologue**:
```
push rbp      ; save caller's frame pointer
mov rbp, rsp  ; set new frame pointer
sub rsp, 16   ; allocate space for locals
```

**3. Callee (add) -- Function body**:
```
mov eax, [rbp+16]   ; load parameter a
add eax, [rbp+24]   ; add parameter b
mov [rbp-8], eax     ; store result
```

**4. Callee (add) -- Function epilogue**:
```
mov eax, [rbp-8]     ; load return value into register
mov rsp, rbp         ; deallocate locals
pop rbp              ; restore caller's frame pointer
ret                  ; pop return address and jump back
```

**5. Caller (main) -- After the call**:
```
add rsp, 16          ; clean up arguments (in cdecl)
mov [rbp-24], eax    ; store return value in z
```

---

## 3. Calling Conventions

### 3.1 What Is a Calling Convention?

A **calling convention** is a protocol that defines:

1. **How arguments are passed** (registers? stack? which order?)
2. **Who cleans up the stack** (caller or callee?)
3. **Which registers are preserved** (caller-saved vs callee-saved)
4. **How the return value is delivered**
5. **How the stack frame is structured**

Calling conventions ensure that separately compiled functions can interact correctly.

### 3.2 cdecl (C Declaration)

The default calling convention for C on 32-bit x86.

| Aspect | cdecl |
|--------|-------|
| Arguments | Pushed right-to-left on the stack |
| Stack cleanup | Caller cleans up |
| Return value | In `EAX` (integers), `ST(0)` (floats) |
| Callee-saved | `EBX`, `ESI`, `EDI`, `EBP` |
| Variadic support | Yes (caller knows arg count) |

**Right-to-left pushing** means the first argument ends up at the lowest stack address, closest to the top of the stack. This enables variable-argument functions (`printf`, etc.) because the first argument is always at a known offset.

```
// Call: add(3, 4)
push 4         ; second argument
push 3         ; first argument
call add
add esp, 8    ; caller cleans up (2 args × 4 bytes)
```

### 3.3 stdcall

Used by the Windows API (Win32 API).

| Aspect | stdcall |
|--------|---------|
| Arguments | Pushed right-to-left on the stack |
| Stack cleanup | **Callee** cleans up |
| Return value | In `EAX` |
| Variadic support | No (callee must know exact arg count) |

```
// Callee epilogue includes:
ret 8          ; return and pop 8 bytes (2 args × 4 bytes)
```

**Advantage**: Slightly smaller code size because the cleanup instruction appears once in the callee rather than at every call site.

**Disadvantage**: Cannot support variadic functions.

### 3.4 System V AMD64 ABI (Linux/macOS x86-64)

The modern 64-bit calling convention used on Linux, macOS, FreeBSD, and other Unix-like systems.

| Aspect | System V AMD64 |
|--------|----------------|
| Integer args (first 6) | `RDI`, `RSI`, `RDX`, `RCX`, `R8`, `R9` |
| Float args (first 8) | `XMM0`--`XMM7` |
| Additional args | Pushed right-to-left on the stack |
| Stack cleanup | Caller |
| Return value | `RAX` (integer), `XMM0` (float) |
| Callee-saved | `RBX`, `RBP`, `R12`--`R15` |
| Stack alignment | 16-byte aligned before `CALL` |
| Red zone | 128 bytes below RSP (leaf functions can use without adjusting RSP) |

**Example**: Call `f(1, 2, 3, 4, 5, 6, 7, 8)`:

```asm
; Arguments 1-6 in registers
mov rdi, 1
mov rsi, 2
mov rdx, 3
mov rcx, 4
mov r8, 5
mov r9, 6
; Arguments 7-8 on the stack (right to left)
push 8
push 7
call f
add rsp, 16   ; caller cleans up stack args
```

### 3.5 Comparison Table

| Feature | cdecl (x86) | stdcall (x86) | System V AMD64 |
|---------|-------------|---------------|----------------|
| Arg passing | Stack only | Stack only | 6 regs + stack |
| Arg order | Right-to-left | Right-to-left | Left-to-right (regs) |
| Cleanup | Caller | Callee | Caller |
| Variadic | Yes | No | Yes |
| Performance | Moderate | Moderate | Better (register passing) |
| Platform | Unix/Windows 32-bit | Windows 32-bit | Linux/macOS 64-bit |

### 3.6 Windows x64 Calling Convention

For completeness, Windows 64-bit uses a different convention:

| Aspect | Windows x64 |
|--------|-------------|
| Integer args (first 4) | `RCX`, `RDX`, `R8`, `R9` |
| Float args (first 4) | `XMM0`--`XMM3` |
| Shadow space | 32 bytes reserved by caller for callee's use |
| Return value | `RAX` (integer), `XMM0` (float) |
| Stack alignment | 16-byte aligned |

---

## 4. Parameter Passing Mechanisms

### 4.1 Call by Value

The **caller** evaluates the argument expression and passes a **copy** of the value to the callee. Modifications to the parameter inside the callee do not affect the original variable.

```c
void increment(int x) {
    x = x + 1;   // Modifies the local copy only
}

int main() {
    int a = 5;
    increment(a);
    // a is still 5
}
```

**Implementation**: The value is copied into the callee's parameter slot (register or stack location).

**Languages**: C, Java (primitives), Go (non-pointer types)

### 4.2 Call by Reference

The caller passes the **address** (reference) of the argument. The callee can read and modify the original variable through this address.

```cpp
void increment(int &x) {
    x = x + 1;   // Modifies the original variable
}

int main() {
    int a = 5;
    increment(a);
    // a is now 6
}
```

**Implementation**: The address of the variable is passed. Inside the callee, every access to the parameter is an indirect memory access through the pointer.

```
; Caller:
lea rdi, [rbp-8]    ; pass address of a
call increment

; Callee:
mov eax, [rdi]      ; dereference to read a
add eax, 1
mov [rdi], eax      ; dereference to write a
```

**Languages**: C++ (references), Fortran (default), C# (`ref` parameters)

### 4.3 Call by Value-Result (Copy-In, Copy-Out)

The value is **copied in** when the function is called and **copied out** when the function returns. This differs from call-by-reference when aliasing occurs.

```
procedure swap(x, y):
    // Copies of actual args are made (copy-in)
    temp = x
    x = y
    y = temp
    // Copies are written back to actuals (copy-out)
```

If called as `swap(a, a)` with call-by-reference, the result is undefined because both parameters alias the same variable. With call-by-value-result, the final value depends on which copy-out happens last.

**Languages**: Ada (`in out` parameters)

### 4.4 Call by Name

The argument is not evaluated at the call site. Instead, the **text** (or a closure-like thunk) of the argument expression is passed. Each time the callee refers to the parameter, the expression is re-evaluated in the caller's environment.

This is historically associated with Algol 60. The argument is essentially a **thunk** -- a parameterless function that, when called, evaluates the expression.

**Classic example -- Jensen's device**:

```
// Algol 60 pseudocode
real procedure sum(i, lo, hi, expr);
    name i, expr;       // call by name
    value lo, hi;       // call by value
    integer i, lo, hi;
    real expr;
begin
    real s;
    s := 0;
    for i := lo step 1 until hi do
        s := s + expr;  // expr is re-evaluated each iteration
    sum := s;
end;

// Usage: compute sum of i*i for i from 1 to 10
result = sum(i, 1, 10, i*i);
```

Each time `expr` is referenced, the thunk `i*i` is evaluated with the current value of `i`.

**Implementation**: A thunk is a small closure containing the expression code and the environment in which to evaluate it.

```python
# Python simulation of call-by-name using thunks

def call_by_name_demo():
    """Demonstrate call-by-name with thunks."""
    a = [1, 2, 3, 4, 5]
    i = 0

    def i_thunk():
        """Thunk that returns the current value of i."""
        return i

    def a_i_thunk():
        """Thunk that evaluates a[i] each time it is called."""
        return a[i]

    def set_a_i(val):
        """Thunk to set a[i]."""
        a[i] = val

    # Simulate swap(i, a[i]) with call-by-name
    # Each access to parameters re-evaluates the thunk
    print(f"Before: i={i}, a={a}")

    # swap body: temp = x; x = y; y = temp
    temp = i_thunk()            # temp = i (evaluates to 0)
    i = a_i_thunk()             # i = a[i] = a[0] = 1
    set_a_i(temp)               # a[i] = temp, but now i=1, so a[1] = 0

    print(f"After:  i={i}, a={a}")
    # Result: i=1, a=[1, 0, 3, 4, 5]
    # Note: a[0] is unchanged because by the time we write,
    # i has changed to 1

call_by_name_demo()
```

### 4.5 Comparison of Parameter Passing Mechanisms

| Mechanism | Evaluation time | Aliasing effects | Performance |
|-----------|----------------|------------------|-------------|
| By value | At call site | None | Copy cost |
| By reference | At call site | Yes | Indirection cost |
| By value-result | At call + return | Defined by copy order | Two copies |
| By name | Each use | Complex | Thunk overhead per use |

---

## 5. Nested Functions and Static Scoping

### 5.1 The Problem of Nested Scopes

Languages like Pascal, Ada, Python, and ML allow functions to be nested inside other functions. A nested function can access variables from its enclosing scope:

```python
def outer():
    x = 10

    def middle():
        y = 20

        def inner():
            # inner can access x, y, and its own locals
            return x + y + 30

        return inner()

    return middle()
```

When `inner` executes, it needs to access `x` from `outer`'s frame and `y` from `middle`'s frame. But these frames are on the stack, and `inner`'s frame is the topmost. How does `inner` find the frames of its lexically enclosing functions?

### 5.2 Access Links (Static Links)

An **access link** (or **static link**) is a pointer stored in each activation record that points to the activation record of the **lexically enclosing function**.

```
Stack:
┌────────────────────┐
│  inner's frame     │
│  access link ──────┼───┐
│  local: (none)     │   │
├────────────────────┤   │
│  middle's frame    │ ◀─┘
│  access link ──────┼───┐
│  local: y = 20     │   │
├────────────────────┤   │
│  outer's frame     │ ◀─┘
│  access link = nil │
│  local: x = 10     │
└────────────────────┘
```

To access a variable at nesting depth $d_{\text{var}}$ from a function at nesting depth $d_{\text{func}}$, the runtime follows $d_{\text{func}} - d_{\text{var}}$ access links.

**Time complexity for variable access**: $O(d_{\text{func}} - d_{\text{var}})$, proportional to the nesting depth difference.

### 5.3 How Access Links Are Maintained

When function $f$ at depth $d_f$ calls function $g$ at depth $d_g$:

1. If $d_g = d_f + 1$ (calling a directly nested function):
   - $g$'s access link points to $f$'s frame.

2. If $d_g \leq d_f$ (calling a function at the same or outer level):
   - Follow $d_f - d_g + 1$ access links from $f$'s frame to find the frame of $g$'s lexically enclosing function.
   - $g$'s access link points to that frame.

### 5.4 Displays

A **display** is an array-based optimization for access links. Instead of following a chain of links, the display maintains a global array $D$ where $D[i]$ holds a pointer to the most recent activation record at nesting depth $i$.

```
Display D:
D[0] ──▶ outer's frame
D[1] ──▶ middle's frame
D[2] ──▶ inner's frame
```

**Access a variable at depth $k$**: Simply look up $D[k]$ and add the offset. This is $O(1)$ regardless of nesting depth.

**Maintenance**: When entering a function at depth $k$:
1. Save the old value of $D[k]$
2. Set $D[k]$ to the current frame pointer

When leaving:
1. Restore $D[k]$ from the saved value

### 5.5 Comparison: Access Links vs Displays

| Aspect | Access Links | Displays |
|--------|-------------|----------|
| Storage | One pointer per frame | Global array (size = max depth) |
| Variable access | $O(\text{depth difference})$ | $O(1)$ |
| Maintenance | Simple pointer assignment | Save/restore array entry |
| Closures | Natural (link is part of closure) | More complex (must capture array state) |
| Common in | Modern compilers | Older compilers (Burroughs B5000) |

### 5.6 Closures

A **closure** captures a function together with its lexical environment (the access link or equivalent). When a function is returned as a value or stored in a data structure, the closure ensures it can still access variables from its enclosing scope.

```python
def make_adder(x):
    def add(y):
        return x + y    # x is captured from make_adder's scope
    return add           # Returns a closure

add5 = make_adder(5)
print(add5(3))           # Output: 8
```

**Implementation challenges**:
- If the enclosing function's frame is on the stack, it will be deallocated when the function returns.
- The closure must keep the captured variables alive, typically by allocating them on the **heap** instead of the stack (this is called "variable escape" or "closure conversion").

---

## 6. Dynamic Scoping vs Static Scoping

### 6.1 Static (Lexical) Scoping

In **static scoping**, the binding of a variable is determined by the program's text (the lexical structure). A variable reference is resolved by looking outward through enclosing scopes at compile time.

```python
x = 10

def foo():
    return x     # Always refers to the global x (=10)

def bar():
    x = 20
    return foo() # foo still sees x=10 (static scoping)

print(bar())     # Output: 10
```

**Implementation**: Access links or displays at runtime. The chain of scopes is determined by where functions are **defined**, not where they are **called**.

### 6.2 Dynamic Scoping

In **dynamic scoping**, the binding of a variable is determined by the runtime call chain. A variable reference is resolved by looking through the **call stack** for the most recent binding.

```
x = 10

function foo():
    return x     // In dynamic scoping, depends on who called foo

function bar():
    x = 20
    return foo() // foo sees x=20 because bar's binding is on the stack

bar()            // returns 20 (dynamic scoping)
```

**Languages**: Early Lisp, Bash/shell scripts, Emacs Lisp, Perl (with `local`).

### 6.3 Runtime Implementation

#### Static Scoping Runtime
- Use **access links** that follow lexical nesting
- Variable locations are determined at **compile time** (offset from a specific frame)
- The access link chain is based on **where functions are defined**

#### Dynamic Scoping Runtime

There are two common implementations:

**1. Deep access**: Walk up the **call stack** (dynamic chain) searching for the variable binding. Each frame stores variable names along with their values.

```
function lookup(var_name):
    frame = current_frame
    while frame is not null:
        if var_name in frame.locals:
            return frame.locals[var_name]
        frame = frame.dynamic_link    // follow call chain
    error("unbound variable")
```

**Time complexity**: $O(d)$ where $d$ is the depth of the call stack.

**2. Shallow access**: Maintain a **central table** (one entry per variable name) that always holds the current binding. When a function is entered, save the old binding and install the new one. When the function exits, restore the old binding.

```
central_table = {}
save_stack = {}

function enter_scope(var_name, value):
    save old central_table[var_name]
    central_table[var_name] = value

function exit_scope(var_name):
    central_table[var_name] = saved value
```

**Time complexity**: $O(1)$ for variable access; $O(k)$ for entering/exiting a scope with $k$ local variables.

### 6.4 Python Simulation: Static vs Dynamic Scoping

```python
"""Demonstrate the difference between static and dynamic scoping."""


class Environment:
    """A simple environment for variable lookup."""

    def __init__(self, bindings=None, parent=None):
        self.bindings = bindings or {}
        self.parent = parent

    def lookup(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        raise NameError(f"Unbound variable: {name}")

    def set(self, name, value):
        self.bindings[name] = value


# ---------- Static Scoping ----------

class StaticScopingInterpreter:
    """
    Interpreter with static (lexical) scoping.
    Each function captures its definition environment.
    """

    def __init__(self):
        self.global_env = Environment()

    def define_var(self, name, value):
        self.global_env.set(name, value)

    def define_function(self, name, params, body, def_env=None):
        """
        Store a function as a closure: (params, body, defining_env).
        """
        env = def_env or self.global_env
        closure = (params, body, env)  # Captures definition environment
        self.global_env.set(name, closure)
        return closure

    def call_function(self, name, args):
        """
        Call a function. Variable lookup uses the DEFINITION environment.
        """
        closure = self.global_env.lookup(name)
        params, body, def_env = closure

        # Create new scope with definition environment as parent (static)
        call_env = Environment(
            bindings=dict(zip(params, args)),
            parent=def_env  # <-- STATIC: uses definition environment
        )
        return body(call_env)


class DynamicScopingInterpreter:
    """
    Interpreter with dynamic scoping.
    Variable lookup follows the call chain.
    """

    def __init__(self):
        self.global_env = Environment()

    def define_var(self, name, value):
        self.global_env.set(name, value)

    def define_function(self, name, params, body):
        func = (params, body)
        self.global_env.set(name, func)
        return func

    def call_function(self, name, args, caller_env=None):
        """
        Call a function. Variable lookup uses the CALLER's environment.
        """
        func = self.global_env.lookup(name)
        params, body = func

        parent = caller_env or self.global_env

        # Create new scope with caller's environment as parent (dynamic)
        call_env = Environment(
            bindings=dict(zip(params, args)),
            parent=parent  # <-- DYNAMIC: uses caller's environment
        )
        return body(call_env)


def demo_scoping():
    """Demonstrate the difference."""

    # --- Static Scoping ---
    print("=== Static Scoping ===")
    static = StaticScopingInterpreter()
    static.define_var("x", 10)

    # foo returns x (defined in global scope where x=10)
    static.define_function("foo", [],
        lambda env: env.lookup("x"))

    # bar sets x=20 locally, then calls foo
    def bar_body_static(env):
        return static.call_function("foo", [])

    static.define_function("bar", [],
        lambda env: (
            env.set("x", 20),
            static.call_function("foo", [])
        )[-1])

    result = static.call_function("bar", [])
    print(f"  bar() calls foo(), foo sees x = {result}")
    # Static: foo sees x=10 (from its definition environment)

    # --- Dynamic Scoping ---
    print("\n=== Dynamic Scoping ===")
    dynamic = DynamicScopingInterpreter()
    dynamic.define_var("x", 10)

    dynamic.define_function("foo", [],
        lambda env: env.lookup("x"))

    def bar_body_dynamic(env):
        env.set("x", 20)
        return dynamic.call_function("foo", [], caller_env=env)

    dynamic.define_function("bar", [], bar_body_dynamic)

    result = dynamic.call_function("bar", [])
    print(f"  bar() calls foo(), foo sees x = {result}")
    # Dynamic: foo sees x=20 (from bar's environment on the call chain)


if __name__ == "__main__":
    demo_scoping()
```

**Expected output**:
```
=== Static Scoping ===
  bar() calls foo(), foo sees x = 10

=== Dynamic Scoping ===
  bar() calls foo(), foo sees x = 20
```

---

## 7. Heap Management

### 7.1 Why Heap Allocation?

The stack provides efficient memory management for data with **last-in, first-out** lifetimes. But not all data follows this pattern:

- Objects whose lifetime extends beyond the function that created them
- Data structures that grow or shrink dynamically (lists, trees, hash tables)
- Closures that capture variables from enclosing scopes

Such data must be allocated on the **heap**.

### 7.2 Explicit Allocation and Deallocation

In languages like C and C++, the programmer explicitly manages heap memory:

```c
int *p = malloc(sizeof(int) * 100);  // allocate
// ... use p ...
free(p);                              // deallocate
```

**Problems with explicit management**:
- **Memory leaks**: Forgetting to free memory
- **Dangling pointers**: Using memory after it has been freed
- **Double free**: Freeing the same memory twice
- **Fragmentation**: Free blocks scattered throughout the heap

### 7.3 Free List Management

A **free list** is a linked list of free memory blocks. The allocator searches this list to find a suitable block for each allocation request.

```
Heap:
┌─────┬──────────┬─────┬──────┬─────┬──────────┐
│USED │  FREE    │USED │ FREE │USED │  FREE    │
│100B │  200B    │150B │  50B │80B  │  300B    │
└─────┴──────────┴─────┴──────┴─────┴──────────┘

Free list:
head ──▶ [200B] ──▶ [50B] ──▶ [300B] ──▶ null
```

#### Allocation Strategies

**First Fit**: Scan the free list from the beginning and return the first block that is large enough.
- Fast allocation
- Tends to cause fragmentation at the beginning of the heap

**Best Fit**: Scan the entire free list and return the smallest block that is large enough.
- Minimizes wasted space per allocation
- Slow (must scan entire list); creates many tiny unusable fragments

**Worst Fit**: Return the largest free block.
- Leaves the biggest remaining fragment (might be useful later)
- Also slow; often performs poorly in practice

**Next Fit**: Like first fit, but starts scanning from where the previous search left off.
- Distributes allocations more evenly across the heap
- Avoids always fragmenting the beginning

#### Coalescing

When a block is freed, the allocator checks whether adjacent blocks are also free and **merges** (coalesces) them into a larger block:

```
Before free(B):
┌─────┬──────────┬─────┬──────────┐
│  A  │  B(used) │  C  │  D(free) │
│free │          │free │          │
└─────┴──────────┴─────┴──────────┘

After free(B) with coalescing:
┌─────────────────────────┬──────────┐
│   A + B + C (merged)    │  D(free) │
│        free             │          │
└─────────────────────────┴──────────┘
```

Coalescing reduces external fragmentation. To make coalescing efficient, each block typically stores:
- A **header** with the block size and allocation status
- A **footer** (boundary tag) with the block size, enabling backward coalescing

### 7.4 Buddy System

The **buddy system** organizes heap memory into blocks whose sizes are powers of 2. This simplifies splitting and coalescing.

**Algorithm**:

1. Memory is divided into blocks of sizes $2^0, 2^1, 2^2, \ldots, 2^k$
2. Maintain separate free lists for each block size

**Allocation** of $n$ bytes:
1. Round up $n$ to the next power of 2, say $2^j$
2. Find the smallest free block of size $\geq 2^j$
3. If the found block is larger than needed (say size $2^{j+k}$):
   - Split it repeatedly into **buddies** (two equal halves) until you get a block of size $2^j$
   - Add the unused buddies to their respective free lists

**Deallocation**:
1. Free the block
2. Check if its **buddy** (the other half of the split) is also free
3. If yes, coalesce them into a larger block
4. Repeat coalescing recursively

**Finding the buddy**: For a block of size $2^j$ at address $A$, its buddy is at:

$$\text{buddy}(A, j) = A \oplus 2^j$$

where $\oplus$ is the bitwise XOR operation.

**Example**:

```
Initial: One block of 1024 bytes

Request 100 bytes (rounded to 128):
1024 → split → 512 + 512
           → split → 256 + 256
               → split → 128 + 128
                           ↑ allocated

Free lists after allocation:
512: [block at 512]
256: [block at 256]
128: [block at 128]    (the other buddy)
```

```python
"""Buddy System allocator simulation."""

import math


class BuddyAllocator:
    """
    Simplified buddy system memory allocator.
    All sizes are powers of 2.
    """

    def __init__(self, total_size: int):
        """Initialize with total_size (must be a power of 2)."""
        self.total_size = total_size
        self.min_block = 16  # Minimum block size
        self.max_order = int(math.log2(total_size))
        self.min_order = int(math.log2(self.min_block))

        # Free lists indexed by order (2^order = block size)
        # Each entry is a set of block start addresses
        self.free_lists: dict[int, set] = {
            order: set() for order in range(self.min_order, self.max_order + 1)
        }

        # Initially, one big free block
        self.free_lists[self.max_order].add(0)

        # Track allocated blocks: address -> order
        self.allocated: dict[int, int] = {}

    def _order_for_size(self, size: int) -> int:
        """Find the smallest order whose block size >= requested size."""
        order = max(self.min_order, math.ceil(math.log2(max(size, 1))))
        return order

    def _buddy_address(self, address: int, order: int) -> int:
        """Compute the buddy's address using XOR."""
        return address ^ (1 << order)

    def allocate(self, size: int) -> int:
        """
        Allocate a block of at least 'size' bytes.
        Returns the start address, or -1 if allocation fails.
        """
        needed_order = self._order_for_size(size)

        # Find the smallest available block
        found_order = -1
        for order in range(needed_order, self.max_order + 1):
            if self.free_lists[order]:
                found_order = order
                break

        if found_order == -1:
            print(f"  FAILED: Cannot allocate {size} bytes")
            return -1

        # Remove the block from the free list
        address = min(self.free_lists[found_order])  # Take lowest address
        self.free_lists[found_order].remove(address)

        # Split down to the needed order
        current_order = found_order
        while current_order > needed_order:
            current_order -= 1
            # Create a buddy at the upper half
            buddy_addr = address + (1 << current_order)
            self.free_lists[current_order].add(buddy_addr)

        # Record the allocation
        self.allocated[address] = needed_order
        block_size = 1 << needed_order

        print(f"  Allocated {size}B at address {address} "
              f"(block size {block_size}B, order {needed_order})")
        return address

    def free(self, address: int):
        """Free a previously allocated block and coalesce with buddies."""
        if address not in self.allocated:
            print(f"  ERROR: Address {address} not allocated")
            return

        order = self.allocated.pop(address)
        block_size = 1 << order
        print(f"  Freeing address {address} (block size {block_size}B, order {order})")

        # Coalesce with buddies
        current_addr = address
        current_order = order

        while current_order < self.max_order:
            buddy_addr = self._buddy_address(current_addr, current_order)

            if buddy_addr in self.free_lists[current_order]:
                # Buddy is free -- coalesce!
                self.free_lists[current_order].remove(buddy_addr)
                # The merged block starts at the lower address
                current_addr = min(current_addr, buddy_addr)
                current_order += 1
                print(f"    Coalesced with buddy at {buddy_addr} "
                      f"-> new block at {current_addr} (order {current_order})")
            else:
                break

        # Add the (possibly coalesced) block to the free list
        self.free_lists[current_order].add(current_addr)

    def print_state(self):
        """Print the current state of the allocator."""
        print("\n  Free lists:")
        for order in range(self.min_order, self.max_order + 1):
            if self.free_lists[order]:
                size = 1 << order
                addrs = sorted(self.free_lists[order])
                print(f"    Order {order} ({size:4d}B): {addrs}")

        if self.allocated:
            print("  Allocated blocks:")
            for addr in sorted(self.allocated.keys()):
                order = self.allocated[addr]
                print(f"    Address {addr}: {1 << order}B (order {order})")
        print()


def demo_buddy():
    """Demonstrate buddy system allocation and deallocation."""
    print("=== Buddy System Allocator (1024 bytes) ===\n")
    allocator = BuddyAllocator(1024)

    print("Initial state:")
    allocator.print_state()

    print("--- Allocations ---")
    a1 = allocator.allocate(100)   # Needs 128B
    a2 = allocator.allocate(200)   # Needs 256B
    a3 = allocator.allocate(50)    # Needs 64B
    a4 = allocator.allocate(60)    # Needs 64B

    print("\nAfter allocations:")
    allocator.print_state()

    print("--- Deallocations ---")
    allocator.free(a3)
    allocator.free(a4)

    print("\nAfter freeing a3 and a4:")
    allocator.print_state()

    allocator.free(a1)

    print("After freeing a1:")
    allocator.print_state()

    allocator.free(a2)

    print("After freeing a2 (everything freed):")
    allocator.print_state()


if __name__ == "__main__":
    demo_buddy()
```

### 7.5 Garbage Collection (Overview)

Languages with automatic memory management (Java, Python, Go, OCaml) use **garbage collection** (GC) to reclaim unreachable heap objects.

Major GC strategies:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Reference counting** | Each object tracks how many references point to it | Immediate reclamation; simple | Cannot handle cycles; counter overhead |
| **Mark-and-sweep** | Mark reachable objects from roots, sweep unmarked | Handles cycles | Stop-the-world pauses; fragmentation |
| **Mark-and-compact** | Like mark-sweep, but compacts live objects | No fragmentation | Expensive object movement |
| **Copying (Cheney)** | Copy live objects to a new space | Fast allocation; no fragmentation | Halves available memory |
| **Generational** | Partition objects by age; collect young generation more often | Exploits generational hypothesis | Complex implementation |

The **generational hypothesis** states that most objects die young. Generational collectors exploit this by frequently collecting the **nursery** (young generation) and rarely collecting the **old generation**.

---

## 8. Memory Layout in Practice

### 8.1 Examining a C Program's Layout

```c
#include <stdio.h>
#include <stdlib.h>

// Global/static data (.data and .bss)
int global_initialized = 42;     // .data
int global_uninitialized;        // .bss
static int static_var = 100;     // .data
const char *string_lit = "hello"; // pointer in .data, string in .rodata

void function() {
    // Stack
    int local_var = 10;
    int array[100];

    // Heap
    int *heap_data = malloc(sizeof(int) * 50);

    printf("Code:    %p (function address)\n", (void*)function);
    printf("Global:  %p (global_initialized)\n", (void*)&global_initialized);
    printf("BSS:     %p (global_uninitialized)\n", (void*)&global_uninitialized);
    printf("Static:  %p (static_var)\n", (void*)&static_var);
    printf("Literal: %p (string literal)\n", (void*)string_lit);
    printf("Stack:   %p (local_var)\n", (void*)&local_var);
    printf("Heap:    %p (malloc'd data)\n", (void*)heap_data);

    free(heap_data);
}

int main() {
    function();
    return 0;
}
```

**Typical output on x86-64 Linux** (addresses vary due to ASLR):
```
Code:    0x55a3b7c00169
Global:  0x55a3b7e03010
BSS:     0x55a3b7e03018
Static:  0x55a3b7e03014
Literal: 0x55a3b7c00200
Stack:   0x7ffd9a3b4c0c
Heap:    0x55a3b8a046b0
```

Notice the large gap between stack addresses (`0x7ffd...`) and heap addresses (`0x55a3...`).

### 8.2 Stack Frame Layout on x86-64 (System V ABI)

```
Higher addresses
┌─────────────────────────┐
│ Argument 8 (if any)     │  [RBP + 24]
├─────────────────────────┤
│ Argument 7              │  [RBP + 16]
├─────────────────────────┤
│ Return address          │  [RBP + 8]   (pushed by CALL)
├─────────────────────────┤
│ Saved RBP               │  [RBP + 0]   ◀── RBP points here
├─────────────────────────┤
│ Local variable 1        │  [RBP - 8]
├─────────────────────────┤
│ Local variable 2        │  [RBP - 16]
├─────────────────────────┤
│ Saved callee-saved regs │  [RBP - 24]
├─────────────────────────┤
│ Alignment padding       │
├─────────────────────────┤
│ Outgoing arg 7+         │  ◀── RSP points here
└─────────────────────────┘
Lower addresses
```

**Red zone**: On System V AMD64, leaf functions (functions that do not call other functions) may use up to 128 bytes below RSP without adjusting the stack pointer. This avoids the overhead of `sub rsp` / `add rsp` for small leaf functions.

---

## 9. Stack Unwinding for Exceptions

### 9.1 The Problem

When an exception is thrown, control must transfer to an appropriate exception handler, potentially many frames up the call stack. All intervening frames must be properly cleaned up (destructors called, resources released).

### 9.2 Approaches

#### Table-Based Unwinding

Modern compilers (GCC, Clang) generate **unwind tables** that describe how to restore registers and unwind each frame. These tables are stored alongside the code and consulted only when an exception is thrown.

```
.eh_frame section:
  Function: foo
    At offset 0:  CFA = RSP + 8
    At offset 4:  CFA = RSP + 16, RBP = [CFA - 16]
    At offset 8:  CFA = RBP + 16
```

**Advantage**: Zero cost when no exception is thrown (no extra instructions on the normal path).

**Disadvantage**: Tables add to binary size.

#### Setjmp/Longjmp-Based

Older approach: use `setjmp` at each try block to save the current state, and `longjmp` to jump to the handler.

**Advantage**: Simple implementation.

**Disadvantage**: `setjmp` has a cost even when no exception is thrown.

### 9.3 Stack Unwinding Process

```
Call stack when exception is thrown in baz():

main() → foo() → bar() → baz()
                           ↑ exception thrown here

Unwinding:
1. Search baz() for a handler → none found
2. Unwind baz()'s frame (run cleanups)
3. Search bar() for a handler → none found
4. Unwind bar()'s frame (run cleanups)
5. Search foo() for a handler → FOUND
6. Transfer control to foo()'s catch block
```

---

## 10. Python Simulation: Runtime Call Stack

```python
"""
Simulation of a runtime call stack with activation records,
demonstrating function calls, returns, and nested scopes.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ActivationRecord:
    """
    An activation record (stack frame) for a function call.
    """
    function_name: str
    return_address: int                          # Instruction index to return to
    parameters: dict = field(default_factory=dict)
    local_variables: dict = field(default_factory=dict)
    temporaries: dict = field(default_factory=dict)
    saved_registers: dict = field(default_factory=dict)
    static_link: Optional['ActivationRecord'] = None   # For nested functions
    dynamic_link: Optional['ActivationRecord'] = None   # Caller's frame
    return_value: Any = None

    def get_variable(self, name: str) -> Any:
        """Look up a variable in this frame."""
        if name in self.local_variables:
            return self.local_variables[name]
        if name in self.parameters:
            return self.parameters[name]
        return None

    def set_variable(self, name: str, value: Any):
        """Set a local variable."""
        self.local_variables[name] = value

    def __str__(self):
        parts = [f"  Frame: {self.function_name}()"]
        parts.append(f"    Return addr: {self.return_address}")
        if self.parameters:
            parts.append(f"    Parameters: {self.parameters}")
        if self.local_variables:
            parts.append(f"    Locals: {self.local_variables}")
        if self.temporaries:
            parts.append(f"    Temps: {self.temporaries}")
        if self.return_value is not None:
            parts.append(f"    Return value: {self.return_value}")
        if self.static_link:
            parts.append(f"    Static link -> {self.static_link.function_name}()")
        if self.dynamic_link:
            parts.append(f"    Dynamic link -> {self.dynamic_link.function_name}()")
        return "\n".join(parts)


class RuntimeStack:
    """
    Simulation of the runtime call stack.
    """

    def __init__(self):
        self.frames: list[ActivationRecord] = []
        self.pc: int = 0  # Program counter

    @property
    def current_frame(self) -> Optional[ActivationRecord]:
        return self.frames[-1] if self.frames else None

    @property
    def depth(self) -> int:
        return len(self.frames)

    def push_frame(self, function_name: str, parameters: dict,
                   return_address: int,
                   static_link: Optional[ActivationRecord] = None):
        """Push a new activation record (function call)."""
        frame = ActivationRecord(
            function_name=function_name,
            return_address=return_address,
            parameters=parameters,
            dynamic_link=self.current_frame,
            static_link=static_link,
        )
        self.frames.append(frame)
        print(f"\n>>> CALL {function_name}({parameters})")
        print(f"    Stack depth: {self.depth}")

    def pop_frame(self) -> ActivationRecord:
        """Pop the current activation record (function return)."""
        if not self.frames:
            raise RuntimeError("Stack underflow!")

        frame = self.frames.pop()
        print(f"\n<<< RETURN from {frame.function_name}() "
              f"= {frame.return_value}")
        print(f"    Stack depth: {self.depth}")

        # Restore program counter
        self.pc = frame.return_address

        return frame

    def lookup_variable(self, name: str, use_static_scope: bool = True) -> Any:
        """
        Look up a variable using either static or dynamic scope chain.
        """
        if use_static_scope:
            # Follow static links (lexical scoping)
            frame = self.current_frame
            while frame is not None:
                value = frame.get_variable(name)
                if value is not None:
                    return value
                frame = frame.static_link
        else:
            # Follow dynamic links (dynamic scoping)
            frame = self.current_frame
            while frame is not None:
                value = frame.get_variable(name)
                if value is not None:
                    return value
                frame = frame.dynamic_link

        raise NameError(f"Variable '{name}' not found")

    def print_stack(self):
        """Print the entire call stack."""
        print("\n=== Runtime Stack ===")
        if not self.frames:
            print("  (empty)")
            return
        for i in range(len(self.frames) - 1, -1, -1):
            marker = " ◀── TOP" if i == len(self.frames) - 1 else ""
            print(f"\n  [{i}]{marker}")
            print(self.frames[i])
        print("=" * 40)


def demo_factorial():
    """
    Simulate the execution of a recursive factorial function.

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    result = factorial(4)
    """
    print("=" * 60)
    print("Simulating: result = factorial(4)")
    print("=" * 60)

    stack = RuntimeStack()

    # Push main's frame
    stack.push_frame("main", {}, return_address=0)
    stack.current_frame.set_variable("result", None)

    # Recursive calls: factorial(4) -> factorial(3) -> ... -> factorial(1)
    def simulate_factorial(n, return_addr):
        stack.push_frame("factorial", {"n": n}, return_address=return_addr)

        if n <= 1:
            stack.current_frame.return_value = 1
            stack.print_stack()
            returned = stack.pop_frame()
            return returned.return_value
        else:
            # Recursive call
            sub_result = simulate_factorial(n - 1, return_addr + 1)
            result = n * sub_result
            stack.current_frame.return_value = result
            returned = stack.pop_frame()
            return returned.return_value

    result = simulate_factorial(4, 10)

    stack.current_frame.set_variable("result", result)
    print(f"\nmain: result = {result}")

    stack.print_stack()

    # Pop main
    stack.current_frame.return_value = result
    stack.pop_frame()


def demo_nested_functions():
    """
    Simulate nested function calls with access links.

    def outer():
        x = 10
        def inner():
            return x + 20   # Accesses outer's x via static link
        return inner()

    result = outer()
    """
    print("\n" + "=" * 60)
    print("Simulating: nested functions with access links")
    print("=" * 60)

    stack = RuntimeStack()

    # main frame
    stack.push_frame("main", {}, return_address=0)

    # outer frame
    stack.push_frame("outer", {}, return_address=1)
    stack.current_frame.set_variable("x", 10)
    outer_frame = stack.current_frame

    # inner frame with static link to outer
    stack.push_frame("inner", {}, return_address=2,
                     static_link=outer_frame)

    stack.print_stack()

    # inner accesses x through static link
    x = stack.lookup_variable("x", use_static_scope=True)
    result = x + 20
    print(f"\ninner: x (via static link) = {x}")
    print(f"inner: result = {result}")

    # Return from inner
    stack.current_frame.return_value = result
    stack.pop_frame()

    # Return from outer
    stack.current_frame.return_value = result
    stack.pop_frame()

    # Store in main
    stack.current_frame.set_variable("result", result)
    print(f"\nmain: result = {result}")
    stack.pop_frame()


if __name__ == "__main__":
    demo_factorial()
    demo_nested_functions()
```

---

## 11. Compiler Support for Runtime Environments

### 11.1 What the Compiler Generates

The compiler is responsible for generating code that correctly manages the runtime environment:

1. **Function prologue**: Code at the start of each function that:
   - Saves the old frame pointer
   - Sets up the new frame pointer
   - Allocates space for local variables
   - Saves callee-saved registers

2. **Function epilogue**: Code at the end of each function that:
   - Restores callee-saved registers
   - Deallocates local variables
   - Restores the old frame pointer
   - Returns to the caller

3. **Variable access code**: Instructions that compute the address of a variable based on its scope:
   - Local variables: offset from FP
   - Parameters: positive offset from FP (or register)
   - Non-local variables: follow access links + offset

4. **Call-site code**: Instructions at each function call that:
   - Evaluate and pass arguments
   - Save caller-saved registers
   - Perform the call
   - Handle the return value

### 11.2 Symbol Table Information

The compiler's symbol table must record:

| Information | Purpose |
|-------------|---------|
| Variable type and size | Determines stack offset and access width |
| Scope level | Determines how many access links to follow |
| Offset within frame | Address computation |
| Parameter index | Determines register or stack slot for arguments |
| Is it captured by a closure? | If yes, allocate on heap instead of stack |

### 11.3 Frame Layout Optimization

Modern compilers optimize frame layouts:

- **Reorder local variables** to minimize padding (alignment)
- **Omit the frame pointer** when possible (use SP-relative addressing), freeing up a register
- **Register allocation** to keep frequently used variables in registers rather than on the stack
- **Stack slot sharing**: Variables with non-overlapping lifetimes can share the same stack slot

---

## 12. Summary

In this lesson, we explored how programs are organized in memory at runtime:

1. **Memory layout**: Code, static data, stack (grows down), and heap (grows up) are the four major regions.

2. **Activation records** store everything needed for a function invocation: parameters, locals, return address, saved registers, and scope links.

3. **Calling conventions** (cdecl, stdcall, System V AMD64) define the protocol for argument passing, stack cleanup, and register usage. The System V AMD64 ABI passes the first 6 integer arguments in registers for efficiency.

4. **Parameter passing mechanisms** include call by value (copy), call by reference (address), call by value-result (copy in/out), and call by name (thunks).

5. **Nested functions** require access links or displays to reach variables in enclosing scopes. Closures must keep captured variables alive beyond the enclosing function's lifetime.

6. **Static scoping** resolves variables by lexical nesting (compile-time determinable), while **dynamic scoping** resolves by the runtime call chain.

7. **Heap management** uses free lists (first fit, best fit) or the buddy system. Garbage collection automates reclamation in managed languages.

8. **Stack unwinding** for exceptions uses table-based mechanisms for zero-cost exception handling on the normal path.

---

## Exercises

### Exercise 1: Stack Frame Diagram

Draw the complete stack layout (with actual byte offsets from FP) for the following C function call on System V AMD64:

```c
int compute(int a, int b, int c, int d, int e, int f, int g, int h) {
    int x = a + b;
    int y = c + d;
    int z = e + f + g + h;
    return x + y + z;
}
```

Note: `a`--`f` are passed in registers; `g` and `h` are on the stack.

### Exercise 2: Access Links

Given the following nested function structure, draw the stack frames with access links when `innermost()` is executing:

```
function level0():
    var a = 1
    function level1():
        var b = 2
        function level2():
            var c = 3
            function level3():
                return a + b + c   // How does this find a, b, c?
            return level3()
        return level2()
    return level1()
```

How many access links must be followed to reach `a` from `level3()`?

### Exercise 3: Calling Convention Comparison

Translate the following function call into x86 assembly for both cdecl (32-bit) and System V AMD64 (64-bit):

```c
int result = multiply_add(2, 3, 4, 5, 6, 7, 8);
```

Show the argument passing, call instruction, and stack cleanup for each convention.

### Exercise 4: Dynamic vs Static Scoping

Trace the execution of the following program under both static and dynamic scoping. What value does `baz()` return in each case?

```
x = 1

function foo():
    return x

function bar():
    x = 2
    return foo()

function baz():
    x = 3
    return bar()
```

### Exercise 5: Buddy System

Given a buddy system allocator with 512 bytes of total memory:

1. Show the state after allocating blocks of sizes: 50, 120, 30, 60
2. Show the state after freeing the 50-byte and 120-byte blocks
3. Does any coalescing occur? If so, describe which blocks merge.

### Exercise 6: Implementation Challenge

Extend the `RuntimeStack` simulation to support:
1. **Exception handling**: Implement try/catch blocks with stack unwinding
2. **Closures**: When a function returns a nested function, ensure the captured variables remain accessible (simulate heap allocation of captured variables)

Test with a program that creates a closure and then throws an exception through multiple frames.

---

[Previous: Intermediate Representations](./09_Intermediate_Representations.md) | [Next: Code Generation](./11_Code_Generation.md) | [Overview](./00_Overview.md)
