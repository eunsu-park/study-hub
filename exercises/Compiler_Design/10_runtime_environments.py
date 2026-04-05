"""
Exercises for Lesson 10: Runtime Environments
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math


# === Exercise 1: Stack Frame Diagram ===
# Problem: Draw stack layout for a function with 8 params on System V AMD64.

def exercise_1():
    """Stack frame layout for System V AMD64 ABI."""
    print("Function: int compute(int a,b,c,d,e,f,g,h)")
    print("  int x = a + b; int y = c + d; int z = e + f + g + h; return x+y+z;")
    print()
    print("System V AMD64 ABI: first 6 integer args in registers, rest on stack.")
    print("  a -> RDI, b -> RSI, c -> RDX, d -> RCX, e -> R8, f -> R9")
    print("  g and h are passed on the stack by the caller.")
    print()

    print("Stack layout (from high to low addresses):")
    print()
    print("  Caller's stack:")
    print("  +-------------------+")
    print("  |    h (8th arg)    | [RBP + 24]")
    print("  +-------------------+")
    print("  |    g (7th arg)    | [RBP + 16]")
    print("  +-------------------+")
    print("  |  return address   | [RBP + 8]   (pushed by CALL)")
    print("  +-------------------+")
    print("  |   saved RBP       | [RBP + 0]   (pushed by callee prologue)")
    print("  +-------------------+ <-- RBP points here")
    print("  |   saved RDI (a)   | [RBP - 8]   (callee may spill register args)")
    print("  +-------------------+")
    print("  |   saved RSI (b)   | [RBP - 16]")
    print("  +-------------------+")
    print("  |   saved RDX (c)   | [RBP - 24]")
    print("  +-------------------+")
    print("  |   saved RCX (d)   | [RBP - 32]")
    print("  +-------------------+")
    print("  |   saved R8  (e)   | [RBP - 40]")
    print("  +-------------------+")
    print("  |   saved R9  (f)   | [RBP - 48]")
    print("  +-------------------+")
    print("  |   local x         | [RBP - 56]")
    print("  +-------------------+")
    print("  |   local y         | [RBP - 64]")
    print("  +-------------------+")
    print("  |   local z         | [RBP - 72]")
    print("  +-------------------+ <-- RSP (after alignment)")
    print()
    print("  Note: In optimized code, x, y, z would live in registers, not on stack.")
    print("  The compiler may not spill register arguments either.")
    print("  Total frame size: 72 bytes (+ 16-byte alignment padding if needed)")


# === Exercise 2: Access Links ===
# Problem: Draw stack frames with access links for nested functions.

def exercise_2():
    """Access links for nested function calls."""
    print("Nested functions:")
    print("  function level0():")
    print("      var a = 1")
    print("      function level1():")
    print("          var b = 2")
    print("          function level2():")
    print("              var c = 3")
    print("              function level3():")
    print("                  return a + b + c")
    print("              return level3()")
    print("          return level2()")
    print("      return level1()")
    print()

    print("Stack frames when level3() is executing:")
    print()
    print("  +-------------------+")
    print("  | level0 frame      |")
    print("  |   a = 1           |  <--+--+--+")
    print("  |   access link: -  |     |  |  |")
    print("  +-------------------+     |  |  |")
    print("  | level1 frame      |     |  |  |")
    print("  |   b = 2           |  <--+--+  |")
    print("  |   access link: ---+-----+  |  |")
    print("  +-------------------+        |  |")
    print("  | level2 frame      |        |  |")
    print("  |   c = 3           |  <--+  |  |")
    print("  |   access link: ---+-----+--+  |")
    print("  +-------------------+     |     |")
    print("  | level3 frame      |     |     |")
    print("  |   access link: ---+-----+     |")
    print("  +-------------------+           |")
    print("  (arrows show access links pointing to enclosing scope)")
    print()

    print("To access 'a' from level3():")
    print("  level3's nesting depth = 3, 'a' is at depth 0")
    print("  Number of access links to follow = 3 - 0 = 3")
    print("  1. Follow level3's access link -> level2's frame")
    print("  2. Follow level2's access link -> level1's frame")
    print("  3. Follow level1's access link -> level0's frame")
    print("  4. Read 'a' from level0's frame")
    print()
    print("  Access links needed to reach each variable from level3:")
    print("    a (depth 0): 3 links")
    print("    b (depth 1): 2 links")
    print("    c (depth 2): 1 link")


# === Exercise 3: Calling Convention Comparison ===
# Problem: Translate function call to x86 for cdecl (32-bit) and SysV AMD64.

def exercise_3():
    """Calling convention comparison: cdecl vs System V AMD64."""
    print("Function call: int result = multiply_add(2, 3, 4, 5, 6, 7, 8);")
    print()

    print("1. cdecl (x86 32-bit):")
    print("   All arguments passed on the stack, right to left.")
    print("   Caller cleans up the stack.")
    print()
    asm_cdecl = [
        "push  8          ; 7th arg",
        "push  7          ; 6th arg",
        "push  6          ; 5th arg",
        "push  5          ; 4th arg",
        "push  4          ; 3rd arg",
        "push  3          ; 2nd arg",
        "push  2          ; 1st arg",
        "call  multiply_add",
        "add   esp, 28    ; caller cleans up (7 args * 4 bytes)",
        "mov   [result], eax  ; return value in EAX",
    ]
    for line in asm_cdecl:
        print(f"   {line}")
    print()

    print("2. System V AMD64 (x86-64):")
    print("   First 6 integer args: RDI, RSI, RDX, RCX, R8, R9")
    print("   7th arg on stack. Caller cleans up.")
    print()
    asm_sysv = [
        "mov   rdi, 2     ; 1st arg",
        "mov   rsi, 3     ; 2nd arg",
        "mov   rdx, 4     ; 3rd arg",
        "mov   rcx, 5     ; 4th arg",
        "mov   r8,  6     ; 5th arg",
        "mov   r9,  7     ; 6th arg",
        "push  8          ; 7th arg on stack",
        "call  multiply_add",
        "add   rsp, 8     ; clean up 1 stack arg",
        "mov   [result], eax  ; return value in EAX (lower 32 bits of RAX)",
    ]
    for line in asm_sysv:
        print(f"   {line}")
    print()

    print("Key differences:")
    print("  cdecl: all 7 args on stack (28 bytes), caller cleanup")
    print("  SysV64: 6 args in registers, 1 on stack (8 bytes), caller cleanup")
    print("  SysV64 is faster due to register passing (no memory access for first 6 args)")


# === Exercise 4: Dynamic vs Static Scoping ===
# Problem: Trace execution under static and dynamic scoping.

def exercise_4():
    """Compare static vs dynamic scoping."""
    print("Program:")
    print("  x = 1")
    print()
    print("  function foo():")
    print("      return x")
    print()
    print("  function bar():")
    print("      x = 2")
    print("      return foo()")
    print()
    print("  function baz():")
    print("      x = 3")
    print("      return bar()")
    print()

    # Static scoping simulation
    print("Static Scoping:")
    print("  x = 1 is in the global scope.")
    print("  foo() looks up x in its defining (lexical) scope: global scope -> x = 1")
    print("  bar() sets x = 2 (this modifies the global x)")
    print("  bar() calls foo(), which sees the global x (now 2 due to bar's assignment)")
    print()

    # Simulate static scoping
    class StaticEnv:
        def __init__(self):
            self.globals = {'x': 1}

        def foo(self):
            return self.globals['x']

        def bar(self):
            self.globals['x'] = 2
            return self.foo()

        def baz(self):
            self.globals['x'] = 3
            return self.bar()

    env_s = StaticEnv()
    result_static = env_s.baz()
    print(f"  baz() call trace:")
    print(f"    baz: x = 3 (global x becomes 3)")
    print(f"    baz calls bar:")
    print(f"      bar: x = 2 (global x becomes 2)")
    print(f"      bar calls foo:")
    print(f"        foo: reads global x = 2")
    print(f"  Result (static scoping): {result_static}")
    print()

    # Dynamic scoping simulation
    print("Dynamic Scoping:")
    print("  foo() looks up x in the calling scope chain (runtime stack).")
    print("  Call chain: baz() -> bar() -> foo()")
    print()

    class DynamicEnv:
        def __init__(self):
            self.scope_stack = [{'x': 1}]  # global

        def push_scope(self, bindings):
            self.scope_stack.append(bindings)

        def pop_scope(self):
            self.scope_stack.pop()

        def lookup(self, name):
            for scope in reversed(self.scope_stack):
                if name in scope:
                    return scope[name]
            raise NameError(name)

        def foo(self):
            return self.lookup('x')

        def bar(self):
            self.push_scope({'x': 2})
            result = self.foo()
            self.pop_scope()
            return result

        def baz(self):
            self.push_scope({'x': 3})
            result = self.bar()
            self.pop_scope()
            return result

    env_d = DynamicEnv()
    result_dynamic = env_d.baz()
    print(f"  baz() call trace:")
    print(f"    baz: pushes x = 3")
    print(f"    baz calls bar:")
    print(f"      bar: pushes x = 2")
    print(f"      bar calls foo:")
    print(f"        foo: looks up x in calling chain: bar's x = 2")
    print(f"  Result (dynamic scoping): {result_dynamic}")
    print()

    print(f"Summary:")
    print(f"  Static scoping:  baz() returns {result_static}")
    print(f"  Dynamic scoping: baz() returns {result_dynamic}")
    print(f"  In this case both return 2, but for different reasons:")
    print(f"  - Static: because bar() mutated the global x to 2")
    print(f"  - Dynamic: because foo() sees bar's local x = 2")


# === Exercise 5: Buddy System ===
# Problem: Simulate buddy system allocation and deallocation.

class BuddyAllocator:
    """Buddy system memory allocator."""

    def __init__(self, total_size):
        self.total_size = total_size
        self.max_order = int(math.log2(total_size))
        # free_lists[order] = list of free blocks of size 2^order
        self.free_lists = defaultdict(list)
        self.free_lists[self.max_order] = [0]  # one big free block
        self.allocated = {}  # address -> (size, order)

    def _order_for_size(self, size):
        """Find the smallest order that can fit the requested size."""
        order = 0
        while (1 << order) < size:
            order += 1
        return order

    def allocate(self, size):
        """Allocate a block of the given size."""
        needed_order = self._order_for_size(size)
        # Find the smallest available block >= needed_order
        for order in range(needed_order, self.max_order + 1):
            if self.free_lists[order]:
                # Found a free block
                addr = self.free_lists[order].pop(0)
                # Split down to needed_order
                while order > needed_order:
                    order -= 1
                    buddy_addr = addr + (1 << order)
                    self.free_lists[order].append(buddy_addr)
                self.allocated[addr] = (size, needed_order)
                return addr
        return None  # out of memory

    def free(self, addr):
        """Free a previously allocated block."""
        if addr not in self.allocated:
            return
        size, order = self.allocated.pop(addr)
        # Try to coalesce with buddy
        while order < self.max_order:
            buddy_addr = addr ^ (1 << order)
            if buddy_addr in self.free_lists[order]:
                self.free_lists[order].remove(buddy_addr)
                addr = min(addr, buddy_addr)
                order += 1
            else:
                break
        self.free_lists[order].append(addr)
        self.free_lists[order].sort()

    def display(self):
        """Display current memory state."""
        print("    Allocated:")
        for addr, (size, order) in sorted(self.allocated.items()):
            actual = 1 << order
            print(f"      addr={addr:>3}, requested={size}, actual={actual}")
        print("    Free lists:")
        for order in range(self.max_order + 1):
            if self.free_lists[order]:
                block_size = 1 << order
                print(f"      order {order} (size {block_size}): "
                      f"{self.free_lists[order]}")


def exercise_5():
    """Buddy system allocator simulation."""
    alloc = BuddyAllocator(512)

    print("Initial state: 512 bytes, one free block")
    alloc.display()
    print()

    # Allocate 50, 120, 30, 60 bytes
    print("Step 1: Allocate 50, 120, 30, 60 bytes")
    addresses = {}
    for size in [50, 120, 30, 60]:
        addr = alloc.allocate(size)
        addresses[size] = addr
        print(f"  allocate({size}) -> addr {addr} (actual size {1 << alloc._order_for_size(size)})")

    print()
    alloc.display()
    print()

    # Free the 50-byte and 120-byte blocks
    print("Step 2: Free 50-byte and 120-byte blocks")
    alloc.free(addresses[50])
    print(f"  free(addr={addresses[50]}) -- was 50 bytes")
    alloc.free(addresses[120])
    print(f"  free(addr={addresses[120]}) -- was 120 bytes")

    print()
    alloc.display()
    print()

    print("Step 3: Coalescing analysis")
    print("  After freeing, the buddy system attempts to merge adjacent free blocks.")
    print("  Two buddies can merge if both are free and have the same order.")
    print("  Check the free lists above to see if any coalescing occurred.")


# === Exercise 6: Implementation Challenge ===
# Problem: Extend RuntimeStack with exception handling and closures.

class RuntimeStack:
    """Runtime call stack simulator with exception handling and closures."""

    def __init__(self):
        self.frames = []
        self.heap = {}     # address -> value (for closure-captured variables)
        self.heap_ptr = 0
        self.output = []

    def push_frame(self, func_name, params=None, access_link=None):
        frame = {
            'func': func_name,
            'locals': dict(params) if params else {},
            'access_link': access_link,
            'try_handlers': [],  # stack of (label, handler_func)
        }
        self.frames.append(frame)
        return frame

    def pop_frame(self):
        return self.frames.pop()

    def current_frame(self):
        return self.frames[-1] if self.frames else None

    def set_local(self, name, value):
        self.frames[-1]['locals'][name] = value

    def get_local(self, name):
        # Search current frame first, then follow access links
        frame = self.frames[-1]
        while frame:
            if name in frame['locals']:
                return frame['locals'][name]
            frame = frame.get('access_link')
        # Check heap (for captured variables)
        if name in self.heap:
            return self.heap[name]
        raise NameError(f"Undefined: {name}")

    def heap_alloc(self, name, value):
        """Allocate a variable on the heap (for closures)."""
        self.heap[name] = value
        return name

    def push_try_handler(self, handler):
        """Push exception handler onto current frame."""
        self.frames[-1]['try_handlers'].append(handler)

    def pop_try_handler(self):
        """Pop exception handler from current frame."""
        if self.frames[-1]['try_handlers']:
            return self.frames[-1]['try_handlers'].pop()
        return None

    def throw(self, exception):
        """Throw an exception, unwind stack until handler found."""
        while self.frames:
            frame = self.frames[-1]
            if frame['try_handlers']:
                handler = frame['try_handlers'].pop()
                return handler(exception)
            self.output.append(f"  Unwinding frame: {frame['func']}")
            self.frames.pop()
        self.output.append(f"  Unhandled exception: {exception}")
        return None


def exercise_6():
    """Runtime stack with exception handling and closures."""
    # Test 1: Closure
    print("Test 1: Closure -- function returns inner function that captures variable")
    print()
    print("  function make_counter():")
    print("      count = 0")
    print("      function increment():")
    print("          count = count + 1")
    print("          return count")
    print("      return increment")
    print()

    stack = RuntimeStack()

    # Simulate make_counter()
    stack.push_frame('make_counter')
    # Allocate 'count' on the heap so it survives after make_counter returns
    stack.heap_alloc('count', 0)
    # Create closure: increment captures 'count' from heap
    closure = {
        'func': 'increment',
        'captured': {'count': 'heap:count'},
    }
    stack.pop_frame()

    # Call the closure (increment) multiple times
    for i in range(3):
        stack.push_frame('increment')
        count = stack.heap['count']
        count += 1
        stack.heap['count'] = count
        stack.pop_frame()
        print(f"  increment() call {i+1}: count = {count}")

    print()

    # Test 2: Exception handling with stack unwinding
    print("Test 2: Exception handling with stack unwinding")
    print()
    print("  function outer():")
    print("      try:")
    print("          middle()")
    print("      catch (e):")
    print("          print('Caught:', e)")
    print()
    print("  function middle():")
    print("      inner()")
    print()
    print("  function inner():")
    print("      throw 'something went wrong'")
    print()

    stack2 = RuntimeStack()

    # Simulate outer()
    stack2.push_frame('outer')
    caught_exception = [None]

    def handler(exception):
        caught_exception[0] = exception
        stack2.output.append(f"  Caught exception: {exception}")
        return 'handled'

    stack2.push_try_handler(handler)

    # Simulate middle()
    stack2.push_frame('middle')

    # Simulate inner()
    stack2.push_frame('inner')

    # Throw exception
    stack2.output.append("  inner() throws exception")
    result = stack2.throw('something went wrong')

    for msg in stack2.output:
        print(msg)
    print(f"  Handler returned: {result}")
    print(f"  Exception value: {caught_exception[0]}")
    print(f"  Remaining frames: {[f['func'] for f in stack2.frames]}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Stack Frame Diagram ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Access Links ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Calling Convention Comparison ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Dynamic vs Static Scoping ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Buddy System ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Implementation Challenge ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
