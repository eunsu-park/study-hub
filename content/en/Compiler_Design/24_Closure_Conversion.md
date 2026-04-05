# Closure Conversion

**Previous**: [23. Type Inference](./23_Type_Inference.md) | **Next**: [25. Linking and Loading](./25_Linking_and_Loading.md)

---

Closures -- functions that capture variables from their enclosing scope -- are a fundamental feature of modern programming languages. However, machine code has no notion of nested scopes or captured variables. Closure conversion is the compiler transformation that eliminates free variables from functions, making them suitable for compilation to flat, first-order code.

This lesson covers lambda lifting, flat closure conversion, defunctionalization, and continuation-passing style (CPS) transformations -- all essential techniques for compiling functional and higher-order features.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [08. Semantic Analysis](./08_Semantic_Analysis.md), [10. Runtime Environments](./10_Runtime_Environments.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what closures are and why they require special compilation
2. Implement lambda lifting to eliminate free variables
3. Implement flat closure conversion with explicit environment records
4. Apply defunctionalization to remove higher-order functions
5. Convert programs to continuation-passing style (CPS)
6. Understand how closures are implemented in real compilers (Python, JavaScript, OCaml)

---

## Table of Contents

1. [Closures and Free Variables](#1-closures-and-free-variables)
2. [Lambda Lifting](#2-lambda-lifting)
3. [Flat Closure Conversion](#3-flat-closure-conversion)
4. [Linked (Shared) Closures](#4-linked-shared-closures)
5. [Defunctionalization](#5-defunctionalization)
6. [CPS Transformation](#6-cps-transformation)
7. [Closures in Practice](#7-closures-in-practice)
8. [Summary](#8-summary)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. Closures and Free Variables

### 1.1 What is a Closure?

A **closure** is a function bundled with its lexical environment -- the variables it references from enclosing scopes:

```python
def make_adder(n):
    def adder(x):
        return x + n    # n is a "free variable" -- captured from make_adder
    return adder

add5 = make_adder(5)
print(add5(3))  # 8 -- the closure remembers n=5
```

### 1.2 Free Variables

A **free variable** in a function is one that is used but not defined locally:

```python
def compute_free_variables(func_ast):
    """
    Compute the free variables of a function.
    Free = Used - Defined_locally - Parameters
    """
    used = collect_all_variable_uses(func_ast.body)
    defined = collect_all_definitions(func_ast.body)
    params = set(func_ast.params)
    return used - defined - params

# Example:
# \x -> x + y + z    (y, z are free; x is a parameter)
# \f -> \x -> f x    (f is free in the inner lambda)
```

### 1.3 The Compilation Problem

Machine-level functions take a fixed set of parameters and access only local variables and globals. There is no mechanism for "capturing" variables from an enclosing stack frame that may have already returned.

```
High-level:                     Low-level (no closures):
def make_adder(n):              make_adder(n):
    def adder(x):                   ??? how to access n?
        return x + n                adder is a separate function
    return adder                    n's stack frame is gone!
```

---

## 2. Lambda Lifting

### 2.1 Concept

**Lambda lifting** eliminates free variables by adding them as extra parameters:

```python
# Before lambda lifting:
def make_adder(n):
    def adder(x):       # adder has free variable n
        return x + n
    return adder

# After lambda lifting:
def adder(n, x):         # n is now a parameter
    return x + n

def make_adder(n):
    return partial(adder, n)  # partially apply n
```

### 2.2 Algorithm

```python
def lambda_lift(program):
    """
    Transform nested functions into top-level functions.
    1. Compute free variables for each nested function
    2. Add free variables as extra parameters
    3. Update all call sites to pass the extra arguments
    4. Move functions to top level
    """
    lifted_functions = []

    def lift(func, enclosing_scope):
        free = compute_free_variables(func)

        # Add free variables as parameters
        new_params = list(func.params) + list(free)
        new_body = func.body

        # Recursively lift nested functions inside this one
        for inner_func in find_nested_functions(new_body):
            lifted_name = lift(inner_func, {**enclosing_scope, **local_scope(func)})
            # Replace nested function reference with call to lifted version
            new_body = replace_func_ref(new_body, inner_func, lifted_name, free)

        lifted_name = generate_unique_name(func.name)
        lifted_functions.append(Function(lifted_name, new_params, new_body))
        return lifted_name

    for func in program.top_level_functions:
        lift(func, global_scope(program))

    return Program(lifted_functions)
```

### 2.3 Example: Step by Step

```python
# Original:
def outer(a):
    def middle(b):
        def inner(c):
            return a + b + c
        return inner(10)
    return middle(20)

# Step 1: Lift inner (free vars: a, b)
def inner_lifted(a, b, c):
    return a + b + c

def middle_after(a, b):       # a is still free
    return inner_lifted(a, b, 10)

# Step 2: Lift middle (free var: a)
def middle_lifted(a, b):
    return inner_lifted(a, b, 10)

def outer(a):
    return middle_lifted(a, 20)
```

### 2.4 Limitations

Lambda lifting changes function signatures, which is problematic when functions are passed as values with a fixed expected signature:

```python
# Problem: map expects a function of 1 argument
def make_predicate(threshold):
    def check(x):
        return x > threshold  # free var: threshold
    return check

result = map(make_predicate(10), [1, 5, 15, 20])

# After lifting, check becomes check(threshold, x) -- 2 args!
# map cannot call it directly.
```

This is why closure conversion is often preferred over lambda lifting.

---

## 3. Flat Closure Conversion

### 3.1 Concept

**Closure conversion** represents each function as a pair: (code pointer, environment record). The environment record stores the values of all free variables.

```
Closure = (function_pointer, environment_record)

# A closure for \x -> x + n where n=5:
closure = (code_for_adder, {n: 5})

# Calling the closure:
result = closure.code(closure.env, x)
```

### 3.2 Implementation

```python
def closure_convert(program):
    """
    Convert all functions to closure form.
    Each function becomes: (code_ptr, env)
    Each function body receives env as first parameter.
    """
    converted = []

    def convert_func(func):
        free = compute_free_variables(func)

        # Create environment struct type
        env_type = StructType({v: type_of(v) for v in free})

        # New function: takes env as first argument
        new_params = ["__env__"] + list(func.params)
        new_body = func.body

        # Replace free variable references with env accesses
        for var in free:
            new_body = replace_var(new_body, var,
                                   FieldAccess("__env__", var))

        # Recursively convert nested functions
        new_body = convert_nested(new_body)

        code_name = f"{func.name}_code"
        converted.append(Function(code_name, new_params, new_body))
        return code_name, env_type, free

    def convert_expr(expr):
        """Convert expressions, wrapping function creation as closure allocation."""
        if isinstance(expr, FunctionDef):
            code_name, env_type, free = convert_func(expr)
            # Allocate environment and store free variables
            env = AllocStruct(env_type)
            for var in free:
                env = SetField(env, var, Var(var))
            return MakeClosure(code_name, env)

        elif isinstance(expr, FunctionCall):
            # Convert f(args) to f.code(f.env, args)
            closure = convert_expr(expr.func)
            args = [convert_expr(a) for a in expr.args]
            return Call(GetCode(closure), [GetEnv(closure)] + args)

        # ... handle other expression types
        return expr

    return convert_expr(program)
```

### 3.3 Example

```python
# Before closure conversion:
def make_counter(start):
    count = start
    def increment():
        nonlocal count
        count = count + 1
        return count
    return increment

# After closure conversion:
def increment_code(env):
    env.count = env.count + 1
    return env.count

def make_counter(start):
    env = allocate({count: start})   # heap-allocated environment
    return (increment_code, env)     # closure = (code, env)

# Calling:
counter = make_counter(0)
result = counter[0](counter[1])     # counter.code(counter.env)
```

### 3.4 Flat vs. Linked Closures

**Flat closure**: copies all free variables into the closure's environment record. Each closure is self-contained.

```
make_adder(5) creates:
  env = {n: 5}
  closure = (adder_code, env)
```

---

## 4. Linked (Shared) Closures

### 4.1 Concept

Instead of copying all free variables, linked closures store a pointer to the enclosing environment:

```
Linked closure:
  env = {local_vars..., parent: enclosing_env}

Access to outer variable:
  env.parent.parent. ... .variable
```

### 4.2 Tradeoffs

| Aspect | Flat Closure | Linked Closure |
|--------|-------------|----------------|
| Creation cost | Copy all free vars | Single pointer |
| Access cost | Direct field access | Chain of pointer dereferences |
| Memory | May duplicate values | Shares with parent |
| GC interaction | Independent lifetimes | Keeps parent alive |
| Used by | OCaml, SML | Python, JavaScript |

### 4.3 Display Optimization

For deeply nested closures, a **display** array provides O(1) access to any nesting level:

```python
class DisplayClosure:
    """
    Closure with display for O(1) access to any scope level.
    """
    def __init__(self, code, display):
        self.code = code
        self.display = display  # array: display[depth] = env at that depth

    def access_var(self, depth, offset):
        """Access variable at nesting depth and offset."""
        return self.display[depth][offset]  # O(1)
```

---

## 5. Defunctionalization

### 5.1 Concept

**Defunctionalization** (Reynolds, 1972) eliminates higher-order functions entirely by replacing them with data constructors and a single `apply` function:

```python
# Before defunctionalization:
def map_func(f, lst):
    return [f(x) for x in lst]

double = lambda x: x * 2
add_n = lambda n: lambda x: x + n

map_func(double, [1, 2, 3])
map_func(add_n(5), [1, 2, 3])

# After defunctionalization:
class Double:
    pass

class AddN:
    def __init__(self, n):
        self.n = n

def apply(func_data, x):
    """Single dispatch point for all 'function calls'."""
    if isinstance(func_data, Double):
        return x * 2
    elif isinstance(func_data, AddN):
        return x + func_data.n
    else:
        raise ValueError(f"Unknown function: {func_data}")

def map_func(f_data, lst):
    return [apply(f_data, x) for x in lst]

map_func(Double(), [1, 2, 3])
map_func(AddN(5), [1, 2, 3])
```

### 5.2 Algorithm

```python
def defunctionalize(program):
    """
    1. Collect all lambda expressions / function values
    2. Create a data type variant for each
    3. Replace lambda creation with variant construction
    4. Replace function application with apply dispatch
    """
    variants = []

    for func_expr in find_all_function_values(program):
        free = compute_free_variables(func_expr)
        variant = DataVariant(
            name=generate_name(func_expr),
            fields=list(free),
            params=func_expr.params,
            body=func_expr.body
        )
        variants.append(variant)

    # Build the apply function
    apply_func = build_apply_dispatcher(variants)

    # Transform the program
    transformed = replace_lambdas_with_constructors(program, variants)
    transformed = replace_applications_with_apply(transformed)

    return transformed, apply_func
```

### 5.3 When to Use Defunctionalization

- Compiling to languages without closures (C, Fortran)
- Whole-program optimization (all call sites known)
- Functional language to imperative target compilation

---

## 6. CPS Transformation

### 6.1 What is CPS?

In **Continuation-Passing Style**, every function takes an extra parameter -- the continuation -- that represents "what to do next":

```python
# Direct style:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# CPS:
def factorial_cps(n, k):
    """k is the continuation: what to do with the result."""
    if n == 0:
        return k(1)
    return factorial_cps(n - 1, lambda result: k(n * result))
```

### 6.2 Why CPS in Compilers?

CPS makes control flow explicit:
- Every function call is a tail call (no stack growth with tail-call optimization)
- Continuations are just closures -- uniform representation
- Makes transformations like exception handling and coroutines easy

### 6.3 CPS Transformation Algorithm

```python
def cps_transform(expr, k):
    """
    Transform direct-style expression to CPS.
    k: the current continuation (a function that takes the result)
    """
    if isinstance(expr, Lit):
        return App(k, expr)

    elif isinstance(expr, Var):
        return App(k, expr)

    elif isinstance(expr, Lam):
        # \x -> body  becomes  \x k_inner -> [[body]] k_inner
        k_inner = fresh_var("k")
        body_cps = cps_transform(expr.body, Var(k_inner))
        new_lam = Lam(expr.param, Lam(k_inner, body_cps))
        return App(k, new_lam)

    elif isinstance(expr, App):
        # f(x) becomes: [[f]] (\f_val -> [[x]] (\x_val -> f_val x_val k))
        f_val = fresh_var("f")
        x_val = fresh_var("x")
        inner = App(App(Var(f_val), Var(x_val)), k)
        x_cont = Lam(x_val, inner)
        f_cont = Lam(f_val, cps_transform(expr.arg, x_cont))
        return cps_transform(expr.func, f_cont)

    elif isinstance(expr, IfExpr):
        # if c then t else e becomes:
        # [[c]] (\c_val -> if c_val then [[t]] k else [[e]] k)
        c_val = fresh_var("c")
        then_cps = cps_transform(expr.then_branch, k)
        else_cps = cps_transform(expr.else_branch, k)
        cond_body = IfExpr(Var(c_val), then_cps, else_cps)
        return cps_transform(expr.condition, Lam(c_val, cond_body))
```

### 6.4 CPS Example

```python
# Direct style:
let x = 3 + 4 in x * 2

# CPS:
add(3, 4, lambda x: mul(x, 2, lambda result: halt(result)))

# Every operation receives its continuation explicitly.
# No implicit "return" -- result flows through continuations.
```

---

## 7. Closures in Practice

### 7.1 Python

Python uses linked closures with cell objects:

```python
import dis

def make_adder(n):
    def adder(x):
        return x + n
    return adder

# Inspect the closure
add5 = make_adder(5)
print(add5.__closure__)        # (<cell at 0x...: int object at 0x...>,)
print(add5.__closure__[0].cell_contents)  # 5
dis.dis(add5)
# LOAD_FAST    0 (x)
# LOAD_DEREF   0 (n)    <-- access via closure cell
# BINARY_ADD
# RETURN_VALUE
```

### 7.2 JavaScript (V8)

V8 creates "Context" objects for captured variables:

```javascript
function makeAdder(n) {
    // V8 allocates a Context object: {n: n}
    return function(x) {
        return x + n;  // accesses n via Context pointer
    };
}
```

### 7.3 OCaml

OCaml uses flat closures -- free variables are copied directly into the closure block:

```ocaml
let make_adder n =
  fun x -> x + n
(* Compiles to: alloc [code_ptr; n] *)
(* Closure is a heap block: [header | code_ptr | n_value] *)
```

---

## 8. Summary

- **Closures** bundle functions with their captured environment
- **Lambda lifting** eliminates free variables by adding parameters
- **Flat closure conversion** creates (code, environment) pairs with copied free variables
- **Linked closures** share environment frames via parent pointers
- **Defunctionalization** replaces higher-order functions with data + dispatch
- **CPS transformation** makes continuations explicit, enabling tail-call optimization
- Real languages use various strategies: Python (linked cells), V8 (contexts), OCaml (flat closures)

---

## 9. Exercises

1. **Free variables**: Compute the free variables of several nested lambda expressions.

2. **Lambda lifting**: Transform a program with three levels of nesting using lambda lifting.

3. **Closure conversion**: Implement flat closure conversion for a small functional language.

4. **Defunctionalization**: Defunctionalize a program that uses `map`, `filter`, and `fold` with lambda arguments.

5. **CPS transform**: Convert a recursive factorial function to CPS and trace its execution.

---

## 10. References

1. Appel, A. W. (1992). *Compiling with Continuations*. Cambridge University Press.
2. Reynolds, J. C. (1972). "Definitional Interpreters for Higher-Order Programming Languages." *Higher-Order and Symbolic Computation*, 11(4).
3. Johnsson, T. (1985). "Lambda Lifting: Transforming Programs to Recursive Equations." *FPCA*.
4. Shao, Z., Appel, A. W. (1994). "Space-Efficient Closure Representations." *LISP and Functional Programming*.
5. Danvy, O., Filinski, A. (1992). "Representing Control: A Study of the CPS Transformation." *MSCS*, 2(4).

---

**Previous**: [23. Type Inference](./23_Type_Inference.md) | **Next**: [25. Linking and Loading](./25_Linking_and_Loading.md)
