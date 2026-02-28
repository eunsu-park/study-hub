"""
Exercises for Lesson 08: Semantic Analysis
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto


# === Shared Type System ===

class Type(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    VOID = "void"
    ERROR = "error"
    LIST_INT = "list[int]"
    LIST_FLOAT = "list[float]"
    LIST_STRING = "list[string]"


# === Exercise 1: Symbol Table Extension ===
# Problem: Track unused variables, shadowing warnings, constant propagation.

class SymbolInfo:
    def __init__(self, name, type_, line, is_const=False, const_value=None):
        self.name = name
        self.type = type_
        self.line = line
        self.is_const = is_const
        self.const_value = const_value
        self.references = 0


class SymbolTable:
    """Symbol table with scope management, usage tracking, and constant propagation."""

    def __init__(self):
        self.scopes = [{}]  # stack of scopes
        self.warnings = []
        self.errors = []

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        """Exit scope and report unused variables."""
        scope = self.scopes[-1]
        for name, info in scope.items():
            if info.references == 0:
                self.warnings.append(
                    f"Line {info.line}: Variable '{name}' is defined but never used"
                )
        self.scopes.pop()

    def define(self, name, type_, line, is_const=False, const_value=None):
        """Define a new variable in the current scope."""
        # Check for shadowing
        for scope in reversed(self.scopes[:-1]):
            if name in scope:
                self.warnings.append(
                    f"Line {line}: Variable '{name}' shadows variable "
                    f"defined at line {scope[name].line}"
                )
                break

        # Check for duplicate in current scope
        if name in self.scopes[-1]:
            self.errors.append(
                f"Line {line}: Variable '{name}' already defined in this scope "
                f"(first defined at line {self.scopes[-1][name].line})"
            )
            return

        self.scopes[-1][name] = SymbolInfo(name, type_, line, is_const, const_value)

    def lookup(self, name, line=None):
        """Look up a variable, increment reference count."""
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name].references += 1
                return scope[name]
        if line is not None:
            self.errors.append(f"Line {line}: Undefined variable '{name}'")
        return None

    def get_constant(self, name):
        """Get constant value if variable is a known constant."""
        for scope in reversed(self.scopes):
            if name in scope and scope[name].is_const:
                return scope[name].const_value
        return None


def exercise_1():
    """Symbol table with unused variable detection, shadowing, and constants."""
    st = SymbolTable()

    # Simulate:
    # Line 1: let x = 5;
    # Line 2: let y = 10;
    # Line 3: {
    # Line 4:   let x = 20;   // shadows outer x
    # Line 5:   let z = x;     // uses inner x
    # Line 6: }
    # Line 7: print(x);        // uses outer x
    # Line 8: // y is never used

    print("Simulating program:")
    print("  1: let x = 5;")
    print("  2: let y = 10;")
    print("  3: {")
    print("  4:   let x = 20;")
    print("  5:   let z = x;")
    print("  6: }")
    print("  7: print(x);")
    print()

    st.define("x", Type.INT, 1, is_const=True, const_value=5)
    st.define("y", Type.INT, 2, is_const=True, const_value=10)

    st.enter_scope()
    st.define("x", Type.INT, 4, is_const=True, const_value=20)
    info = st.lookup("x", 5)
    print(f"  Lookup 'x' at line 5: type={info.type.value}, const={info.const_value}")
    st.define("z", Type.INT, 5, is_const=True, const_value=info.const_value)
    st.exit_scope()  # z is unused (defined but lookup count = 0 inside scope)

    st.lookup("x", 7)
    const = st.get_constant("x")
    print(f"  Lookup 'x' at line 7: const value = {const}")

    st.exit_scope()  # y is unused

    print()
    print("Warnings:")
    for w in st.warnings:
        print(f"  {w}")
    print("Errors:")
    for e in st.errors:
        print(f"  {e}")


# === Exercise 2: Full Type Checker ===
# Problem: Type check arrays, strings, compound assignment, ternary operator.

class TypeChecker:
    """Extended type checker supporting arrays, strings, compound assignment, ternary."""

    def __init__(self):
        self.env = {}  # var -> Type
        self.errors = []

    def check_binary(self, op, left_type, right_type, line):
        """Type-check a binary operation."""
        # Arithmetic: int op int -> int, float op float -> float
        if op in ('+', '-', '*', '/'):
            if left_type == Type.INT and right_type == Type.INT:
                return Type.INT if op != '/' else Type.FLOAT
            if left_type in (Type.INT, Type.FLOAT) and right_type in (Type.INT, Type.FLOAT):
                return Type.FLOAT
            # String concatenation
            if op == '+' and left_type == Type.STRING and right_type == Type.STRING:
                return Type.STRING
            self.errors.append(
                f"Line {line}: Cannot apply '{op}' to {left_type.value} and {right_type.value}"
            )
            return Type.ERROR

        # Comparison: same type -> bool
        if op in ('==', '!=', '<', '>', '<=', '>='):
            if left_type == right_type and left_type in (Type.INT, Type.FLOAT, Type.STRING):
                return Type.BOOL
            if {left_type, right_type} <= {Type.INT, Type.FLOAT}:
                return Type.BOOL
            self.errors.append(
                f"Line {line}: Cannot compare {left_type.value} and {right_type.value}"
            )
            return Type.ERROR

        # Logical: bool op bool -> bool
        if op in ('and', 'or'):
            if left_type == Type.BOOL and right_type == Type.BOOL:
                return Type.BOOL
            self.errors.append(
                f"Line {line}: Logical '{op}' requires bool operands, got "
                f"{left_type.value} and {right_type.value}"
            )
            return Type.ERROR

        return Type.ERROR

    def check_compound_assign(self, op, var_type, expr_type, line):
        """Type-check compound assignment: +=, -=, *=, /="""
        base_op = op[0]  # '+=' -> '+'
        result_type = self.check_binary(base_op, var_type, expr_type, line)
        if result_type == Type.ERROR:
            return Type.ERROR
        # Result must be assignable to variable
        if result_type != var_type and not (var_type == Type.FLOAT and result_type == Type.INT):
            self.errors.append(
                f"Line {line}: Cannot assign {result_type.value} to {var_type.value} variable"
            )
            return Type.ERROR
        return var_type

    def check_ternary(self, cond_type, then_type, else_type, line):
        """Type-check ternary: cond ? then_expr : else_expr"""
        if cond_type != Type.BOOL:
            self.errors.append(
                f"Line {line}: Ternary condition must be bool, got {cond_type.value}"
            )
        if then_type != else_type:
            # Allow int/float promotion
            if {then_type, else_type} <= {Type.INT, Type.FLOAT}:
                return Type.FLOAT
            self.errors.append(
                f"Line {line}: Ternary branches have different types: "
                f"{then_type.value} and {else_type.value}"
            )
            return Type.ERROR
        return then_type

    def check_array_op(self, op, arr_type, arg_type, line):
        """Type-check array operations: append, pop, len, indexing."""
        element_type_map = {
            Type.LIST_INT: Type.INT,
            Type.LIST_FLOAT: Type.FLOAT,
            Type.LIST_STRING: Type.STRING,
        }

        if arr_type not in element_type_map:
            self.errors.append(f"Line {line}: '{op}' requires a list type, got {arr_type.value}")
            return Type.ERROR

        elem_type = element_type_map[arr_type]

        if op == 'append':
            if arg_type != elem_type:
                self.errors.append(
                    f"Line {line}: Cannot append {arg_type.value} to {arr_type.value}"
                )
                return Type.ERROR
            return Type.VOID
        elif op == 'pop':
            return elem_type
        elif op == 'len':
            return Type.INT
        elif op == 'index':
            if arg_type != Type.INT:
                self.errors.append(f"Line {line}: Array index must be int, got {arg_type.value}")
                return Type.ERROR
            return elem_type

        return Type.ERROR

    def check_string_op(self, op, str_type, arg_type, line):
        """Type-check string operations."""
        if str_type != Type.STRING:
            self.errors.append(f"Line {line}: '{op}' requires string, got {str_type.value}")
            return Type.ERROR

        if op == 'len':
            return Type.INT
        elif op == '+':
            if arg_type != Type.STRING:
                self.errors.append(f"Line {line}: Cannot concatenate string and {arg_type.value}")
                return Type.ERROR
            return Type.STRING
        elif op == 'index':
            if arg_type != Type.INT:
                self.errors.append(f"Line {line}: String index must be int, got {arg_type.value}")
                return Type.ERROR
            return Type.STRING  # single character as string

        return Type.ERROR


def exercise_2():
    """Extended type checker with arrays, strings, compound assignment, ternary."""
    tc = TypeChecker()

    # Test array operations
    print("Array operations:")
    tests = [
        ("append int to list[int]", 'append', Type.LIST_INT, Type.INT, 1, Type.VOID),
        ("append str to list[int]", 'append', Type.LIST_INT, Type.STRING, 2, Type.ERROR),
        ("pop from list[int]", 'pop', Type.LIST_INT, None, 3, Type.INT),
        ("len of list[float]", 'len', Type.LIST_FLOAT, None, 4, Type.INT),
        ("index list[string]", 'index', Type.LIST_STRING, Type.INT, 5, Type.STRING),
    ]
    for desc, op, arr_t, arg_t, line, expected in tests:
        result = tc.check_array_op(op, arr_t, arg_t, line)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {desc}: {result.value} (expected {expected.value})")
    print()

    # Test string operations
    print("String operations:")
    result = tc.check_string_op('len', Type.STRING, None, 10)
    print(f"  len(str): {result.value}")
    result = tc.check_string_op('+', Type.STRING, Type.STRING, 11)
    print(f"  str + str: {result.value}")
    result = tc.check_string_op('+', Type.STRING, Type.INT, 12)
    print(f"  str + int: {result.value}")
    print()

    # Test compound assignment
    print("Compound assignment:")
    tc2 = TypeChecker()
    result = tc2.check_compound_assign('+=', Type.INT, Type.INT, 20)
    print(f"  int += int: {result.value}")
    result = tc2.check_compound_assign('*=', Type.FLOAT, Type.INT, 21)
    print(f"  float *= int: {result.value}")
    result = tc2.check_compound_assign('+=', Type.STRING, Type.INT, 22)
    print(f"  string += int: {result.value}")
    print()

    # Test ternary
    print("Ternary operator:")
    tc3 = TypeChecker()
    result = tc3.check_ternary(Type.BOOL, Type.INT, Type.INT, 30)
    print(f"  bool ? int : int -> {result.value}")
    result = tc3.check_ternary(Type.BOOL, Type.INT, Type.FLOAT, 31)
    print(f"  bool ? int : float -> {result.value}")
    result = tc3.check_ternary(Type.INT, Type.INT, Type.INT, 32)
    print(f"  int ? int : int -> {result.value} (error: condition not bool)")
    result = tc3.check_ternary(Type.BOOL, Type.STRING, Type.INT, 33)
    print(f"  bool ? string : int -> {result.value}")

    # Print accumulated errors
    all_errors = tc.errors + tc2.errors + tc3.errors
    if all_errors:
        print(f"\nAll errors ({len(all_errors)}):")
        for err in all_errors:
            print(f"  {err}")


# === Exercise 3: Type Inference ===
# Problem: Implement local type inference.

class TypeInferencer:
    """Simple local type inference for let-declarations."""

    def __init__(self):
        self.env = {}
        self.errors = []

    def infer_literal(self, value):
        """Infer type of a literal value."""
        if isinstance(value, bool):
            return Type.BOOL
        elif isinstance(value, int):
            return Type.INT
        elif isinstance(value, float):
            return Type.FLOAT
        elif isinstance(value, str):
            return Type.STRING
        elif isinstance(value, list):
            if not value:
                self.errors.append("Cannot infer element type of empty list")
                return Type.ERROR
            elem_types = {self.infer_literal(e) for e in value}
            if len(elem_types) == 1:
                elem = elem_types.pop()
                return {Type.INT: Type.LIST_INT, Type.FLOAT: Type.LIST_FLOAT,
                        Type.STRING: Type.LIST_STRING}.get(elem, Type.ERROR)
            self.errors.append(f"Heterogeneous list elements: {elem_types}")
            return Type.ERROR
        return Type.ERROR

    def infer_binary(self, op, left_type, right_type):
        """Infer result type of binary operation."""
        if op in ('+', '-', '*'):
            if left_type == Type.INT and right_type == Type.INT:
                return Type.INT
            if {left_type, right_type} <= {Type.INT, Type.FLOAT}:
                return Type.FLOAT
            if op == '+' and left_type == Type.STRING and right_type == Type.STRING:
                return Type.STRING
        if op == '/':
            if {left_type, right_type} <= {Type.INT, Type.FLOAT}:
                return Type.FLOAT
        if op in ('==', '!=', '<', '>', '<=', '>='):
            return Type.BOOL
        if op in ('and', 'or'):
            if left_type == Type.BOOL and right_type == Type.BOOL:
                return Type.BOOL
        return Type.ERROR

    def infer_if(self, cond_type, then_type, else_type):
        """Infer type of if-then-else expression."""
        if cond_type != Type.BOOL:
            self.errors.append(f"If condition must be bool, got {cond_type.value}")
        if then_type == else_type:
            return then_type
        if {then_type, else_type} <= {Type.INT, Type.FLOAT}:
            return Type.FLOAT
        self.errors.append(f"Branches have incompatible types: {then_type.value}, {else_type.value}")
        return Type.ERROR

    def infer_lambda(self, param_types, return_type):
        """Represent function type as a string."""
        params = ', '.join(t.value for t in param_types)
        return f"({params}) -> {return_type.value}"


def exercise_3():
    """Local type inference for declarations."""
    inf = TypeInferencer()

    test_cases = [
        ("let x = 5", 5, "int"),
        ("let y = [1, 2, 3]", [1, 2, 3], "list[int]"),
        ("let z = fn(a) => a + 1", None, "(int) -> int"),  # special case
        ("let w = if true then 1 else 2", None, "int"),      # special case
        ("let s = \"hello\"", "hello", "string"),
        ("let f = 3.14", 3.14, "float"),
        ("let b = true", True, "bool"),
    ]

    for desc, value, expected_str in test_cases:
        if value is not None:
            inferred = inf.infer_literal(value)
            print(f"  {desc:30s} -> inferred: {inferred.value} (expected: {expected_str})")
        elif "fn" in desc:
            result = inf.infer_lambda([Type.INT], Type.INT)
            print(f"  {desc:30s} -> inferred: {result} (expected: {expected_str})")
        elif "if" in desc:
            result = inf.infer_if(Type.BOOL, Type.INT, Type.INT)
            print(f"  {desc:30s} -> inferred: {result.value} (expected: {expected_str})")

    print()
    # Error case
    inf2 = TypeInferencer()
    result = inf2.infer_literal([])
    print(f"  let a = []  -> {result.value}")
    if inf2.errors:
        for err in inf2.errors:
            print(f"    Error: {err}")


# === Exercise 4: Semantic Error Catalog ===
# Problem: Create test suite with 15+ different semantic errors.

def exercise_4():
    """Catalog of semantic errors with minimal programs."""
    errors = [
        {
            "name": "Undefined variable",
            "program": "print(x);",
            "error": "Undefined variable 'x'",
            "fix": "let x = 0; print(x);",
        },
        {
            "name": "Type mismatch in assignment",
            "program": "let x: int = \"hello\";",
            "error": "Cannot assign string to int variable",
            "fix": "let x: int = 42;",
        },
        {
            "name": "Wrong number of arguments",
            "program": "fn add(a, b) { return a + b; }\nadd(1, 2, 3);",
            "error": "Function 'add' expects 2 args, got 3",
            "fix": "add(1, 2);",
        },
        {
            "name": "Return outside function",
            "program": "return 5;",
            "error": "'return' used outside of a function",
            "fix": "fn f() { return 5; }",
        },
        {
            "name": "Break outside loop",
            "program": "break;",
            "error": "'break' used outside of a loop",
            "fix": "while (true) { break; }",
        },
        {
            "name": "Duplicate parameter names",
            "program": "fn f(x, x) { return x; }",
            "error": "Duplicate parameter name 'x'",
            "fix": "fn f(x, y) { return x; }",
        },
        {
            "name": "Duplicate variable declaration",
            "program": "let x = 1; let x = 2;",
            "error": "Variable 'x' already defined in this scope",
            "fix": "let x = 1; x = 2;  // assignment, not redeclaration",
        },
        {
            "name": "Non-boolean condition",
            "program": "if (42) { print(1); }",
            "error": "Condition must be bool, got int",
            "fix": "if (42 > 0) { print(1); }",
        },
        {
            "name": "Void in expression",
            "program": "fn f() {} let x = f();",
            "error": "Cannot use void value in expression",
            "fix": "fn f(): int { return 1; } let x = f();",
        },
        {
            "name": "Division by zero (constant)",
            "program": "let x = 10 / 0;",
            "error": "Division by zero (detected at compile time)",
            "fix": "let x = 10 / 2;",
        },
        {
            "name": "Missing return value",
            "program": "fn f(): int { let x = 1; }",
            "error": "Function 'f' must return int but missing return on some paths",
            "fix": "fn f(): int { let x = 1; return x; }",
        },
        {
            "name": "Type mismatch in return",
            "program": "fn f(): int { return \"hello\"; }",
            "error": "Return type mismatch: expected int, got string",
            "fix": "fn f(): int { return 42; }",
        },
        {
            "name": "Invalid operand types",
            "program": "let x = \"hello\" - 5;",
            "error": "Cannot apply '-' to string and int",
            "fix": "let x = 10 - 5;",
        },
        {
            "name": "Array index out of bounds (constant)",
            "program": "let a = [1, 2, 3]; let x = a[-1];",
            "error": "Negative array index: -1",
            "fix": "let x = a[2];",
        },
        {
            "name": "Use before initialization",
            "program": "let x: int; print(x);",
            "error": "Variable 'x' used before initialization",
            "fix": "let x: int = 0; print(x);",
        },
    ]

    print(f"Semantic Error Catalog ({len(errors)} errors):")
    print()
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err['name']}")
        print(f"     Program:  {err['program']}")
        print(f"     Error:    {err['error']}")
        print(f"     Fix:      {err['fix']}")
        print()


# === Exercise 5: Overloading with Generics ===
# Problem: Generic function instantiation and type checking.

class GenericFunction:
    """Represents a generic function with type parameters."""

    def __init__(self, name, type_params, param_types, return_type):
        self.name = name
        self.type_params = type_params    # e.g., ['T', 'U']
        self.param_types = param_types    # e.g., ['T', 'U']  (references to type params)
        self.return_type = return_type    # e.g., 'T' or '(T, U)'

    def instantiate(self, arg_types):
        """Infer type parameters from argument types."""
        if len(arg_types) != len(self.param_types):
            return None, f"Expected {len(self.param_types)} args, got {len(arg_types)}"

        # Unification: match type parameters to concrete types
        bindings = {}
        for param_t, arg_t in zip(self.param_types, arg_types):
            if param_t in self.type_params:
                if param_t in bindings:
                    if bindings[param_t] != arg_t:
                        return None, (f"Conflicting types for {param_t}: "
                                      f"{bindings[param_t]} vs {arg_t}")
                else:
                    bindings[param_t] = arg_t
            elif param_t != arg_t:
                return None, f"Type mismatch: expected {param_t}, got {arg_t}"

        # Resolve return type
        ret = self.return_type
        if ret in bindings:
            ret = bindings[ret]
        elif ret.startswith('(') and ret.endswith(')'):
            # Tuple type like (T, U)
            parts = ret[1:-1].split(', ')
            resolved = [bindings.get(p, p) for p in parts]
            ret = f"({', '.join(resolved)})"

        return bindings, ret


def exercise_5():
    """Generic function instantiation and type checking."""
    # Define generic functions
    identity = GenericFunction("identity", ['T'], ['T'], 'T')
    pair = GenericFunction("pair", ['T', 'U'], ['T', 'U'], '(T, U)')
    first = GenericFunction("first", ['T', 'U'], ['(T, U)'], 'T')

    # Test identity
    print("Generic function: identity<T>(x: T) -> T")
    for args in [('int',), ('string',), ('bool',)]:
        bindings, ret = identity.instantiate(args)
        print(f"  identity({args[0]}) => T={bindings['T']}, returns {ret}")
    print()

    # Test pair
    print("Generic function: pair<T, U>(a: T, b: U) -> (T, U)")
    test_args = [('int', 'string'), ('bool', 'float'), ('int', 'int')]
    for args in test_args:
        bindings, ret = pair.instantiate(args)
        print(f"  pair({args[0]}, {args[1]}) => bindings={bindings}, returns {ret}")
    print()

    # Test error cases
    print("Error cases:")
    bindings, err = identity.instantiate(('int', 'string'))
    print(f"  identity(int, string) => {err}")

    # Generic with constraint: swap<T>(a: T, b: T) -> (T, T)
    swap = GenericFunction("swap", ['T'], ['T', 'T'], '(T, T)')
    bindings, ret = swap.instantiate(('int', 'int'))
    print(f"  swap(int, int) => bindings={bindings}, returns {ret}")
    bindings, err = swap.instantiate(('int', 'string'))
    print(f"  swap(int, string) => {err}")


# === Exercise 6: Control Flow Analysis ===
# Problem: Check all paths return, break/continue in loops, unreachable code.

class ControlFlowChecker:
    """Semantic analysis pass for control flow properties."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.loop_depth = 0
        self.in_function = False
        self.function_return_type = None

    def check_returns_on_all_paths(self, stmts):
        """Check if a statement list returns on all paths.
        Returns True if every execution path ends with a return."""
        for i, stmt in enumerate(stmts):
            if stmt['type'] == 'return':
                if i < len(stmts) - 1:
                    self.warnings.append(
                        f"Line {stmts[i+1].get('line', '?')}: Unreachable code after return"
                    )
                return True
            elif stmt['type'] == 'if':
                then_returns = self.check_returns_on_all_paths(stmt['then'])
                else_returns = False
                if stmt.get('else'):
                    else_returns = self.check_returns_on_all_paths(stmt['else'])
                if then_returns and else_returns:
                    if i < len(stmts) - 1:
                        self.warnings.append(
                            f"Line {stmts[i+1].get('line', '?')}: Unreachable code "
                            f"after if-else that returns on all paths"
                        )
                    return True
        return False

    def check_break_continue(self, stmts):
        """Check that break/continue only appear inside loops."""
        for stmt in stmts:
            if stmt['type'] in ('break', 'continue'):
                if self.loop_depth == 0:
                    self.errors.append(
                        f"Line {stmt.get('line', '?')}: "
                        f"'{stmt['type']}' used outside of a loop"
                    )
            elif stmt['type'] == 'while' or stmt['type'] == 'for':
                self.loop_depth += 1
                self.check_break_continue(stmt.get('body', []))
                self.loop_depth -= 1
            elif stmt['type'] == 'if':
                self.check_break_continue(stmt.get('then', []))
                self.check_break_continue(stmt.get('else', []))
            elif stmt['type'] == 'function':
                old_loop_depth = self.loop_depth
                self.loop_depth = 0
                self.check_break_continue(stmt.get('body', []))
                self.loop_depth = old_loop_depth

    def check_function(self, func):
        """Full control flow check for a function."""
        self.in_function = True
        self.function_return_type = func.get('return_type', 'void')

        # Check break/continue
        self.check_break_continue(func['body'])

        # Check returns on all paths (if non-void)
        if self.function_return_type != 'void':
            all_return = self.check_returns_on_all_paths(func['body'])
            if not all_return:
                self.errors.append(
                    f"Function '{func['name']}': not all paths return a value"
                )

        self.in_function = False


def exercise_6():
    """Control flow analysis: returns, break/continue, unreachable code."""
    checker = ControlFlowChecker()

    # Test 1: Function that returns on all paths
    print("Test 1: Function with returns on all paths")
    func1 = {
        'name': 'abs_val',
        'return_type': 'int',
        'body': [
            {'type': 'if', 'line': 2,
             'then': [{'type': 'return', 'line': 3}],
             'else': [{'type': 'return', 'line': 5}]},
        ]
    }
    checker.check_function(func1)
    print(f"  Errors: {checker.errors}")
    print(f"  Warnings: {checker.warnings}")
    print()

    # Test 2: Function missing return on else branch
    checker2 = ControlFlowChecker()
    print("Test 2: Function missing return")
    func2 = {
        'name': 'maybe_return',
        'return_type': 'int',
        'body': [
            {'type': 'if', 'line': 2,
             'then': [{'type': 'return', 'line': 3}],
             'else': [{'type': 'expr', 'line': 5}]},
        ]
    }
    checker2.check_function(func2)
    print(f"  Errors: {checker2.errors}")
    print()

    # Test 3: Break outside loop
    checker3 = ControlFlowChecker()
    print("Test 3: Break outside loop")
    func3 = {
        'name': 'bad_break',
        'return_type': 'void',
        'body': [
            {'type': 'break', 'line': 2},
        ]
    }
    checker3.check_function(func3)
    print(f"  Errors: {checker3.errors}")
    print()

    # Test 4: Break inside loop (OK)
    checker4 = ControlFlowChecker()
    print("Test 4: Break inside loop (should be OK)")
    func4 = {
        'name': 'good_break',
        'return_type': 'void',
        'body': [
            {'type': 'while', 'line': 2,
             'body': [{'type': 'break', 'line': 3}]},
        ]
    }
    checker4.check_function(func4)
    print(f"  Errors: {checker4.errors}")
    print()

    # Test 5: Unreachable code after return
    checker5 = ControlFlowChecker()
    print("Test 5: Unreachable code after return")
    func5 = {
        'name': 'unreachable',
        'return_type': 'int',
        'body': [
            {'type': 'return', 'line': 2},
            {'type': 'expr', 'line': 3},
        ]
    }
    checker5.check_function(func5)
    print(f"  Warnings: {checker5.warnings}")
    print(f"  Errors: {checker5.errors}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Symbol Table Extension ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Full Type Checker ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Type Inference ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Semantic Error Catalog ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Overloading with Generics ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Control Flow Analysis ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
