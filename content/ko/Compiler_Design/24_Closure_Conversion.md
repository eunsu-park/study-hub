# 클로저 변환(Closure Conversion)

**이전**: [23. 타입 추론](./23_Type_Inference.md) | **다음**: [25. 링킹과 로딩](./25_Linking_and_Loading.md)

---

클로저(closure) -- 둘러싸는 스코프에서 변수를 포획하는 함수 -- 는 현대 프로그래밍 언어의 기본 기능입니다. 그러나 기계 코드에는 중첩 스코프나 포획된 변수라는 개념이 없습니다. 클로저 변환(closure conversion)은 함수에서 자유 변수(free variable)를 제거하여, 평면적인 일차(first-order) 코드로 컴파일할 수 있게 만드는 컴파일러 변환입니다.

이 레슨은 람다 리프팅(lambda lifting), 평면 클로저 변환(flat closure conversion), 탈함수화(defunctionalization), 연속 전달 스타일(CPS, Continuation-Passing Style) 변환을 다룹니다 -- 모두 함수형 및 고차(higher-order) 기능을 컴파일하기 위한 핵심 기법입니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [08. 의미 분석](./08_Semantic_Analysis.md), [10. 런타임 환경](./10_Runtime_Environments.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 클로저가 무엇이며 왜 특별한 컴파일이 필요한지 설명한다
2. 람다 리프팅을 구현하여 자유 변수를 제거한다
3. 명시적 환경 레코드(environment record)를 사용한 평면 클로저 변환을 구현한다
4. 탈함수화를 적용하여 고차 함수를 제거한다
5. 프로그램을 연속 전달 스타일(CPS)로 변환한다
6. 실제 컴파일러(Python, JavaScript, OCaml)에서 클로저가 어떻게 구현되는지 이해한다

---

## 목차

1. [클로저와 자유 변수](#1-클로저와-자유-변수)
2. [람다 리프팅](#2-람다-리프팅)
3. [평면 클로저 변환](#3-평면-클로저-변환)
4. [연결(공유) 클로저](#4-연결공유-클로저)
5. [탈함수화](#5-탈함수화)
6. [CPS 변환](#6-cps-변환)
7. [실무에서의 클로저](#7-실무에서의-클로저)
8. [요약](#8-요약)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. 클로저와 자유 변수

### 1.1 클로저란?

**클로저(closure)**는 어휘적 환경(lexical environment) -- 둘러싸는 스코프에서 참조하는 변수들 -- 과 함께 묶인 함수입니다:

```python
def make_adder(n):
    def adder(x):
        return x + n    # n is a "free variable" -- captured from make_adder
    return adder

add5 = make_adder(5)
print(add5(3))  # 8 -- the closure remembers n=5
```

### 1.2 자유 변수(Free Variables)

함수에서 **자유 변수(free variable)**란 사용되지만 지역적으로 정의되지 않은 변수입니다:

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

### 1.3 컴파일 문제

기계 수준 함수는 고정된 매개변수 집합을 받고 지역 변수와 전역 변수에만 접근합니다. 이미 반환된 둘러싸는 스택 프레임에서 변수를 "포획"하는 메커니즘이 없습니다.

```
High-level:                     Low-level (no closures):
def make_adder(n):              make_adder(n):
    def adder(x):                   ??? how to access n?
        return x + n                adder is a separate function
    return adder                    n's stack frame is gone!
```

---

## 2. 람다 리프팅

### 2.1 개념

**람다 리프팅(lambda lifting)**은 자유 변수를 추가 매개변수로 만들어 제거합니다:

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

### 2.2 알고리즘

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

### 2.3 예제: 단계별

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

### 2.4 한계

람다 리프팅은 함수 시그니처를 변경하므로, 함수가 고정된 예상 시그니처로 값으로 전달될 때 문제가 됩니다:

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

이것이 클로저 변환이 람다 리프팅보다 선호되는 이유입니다.

---

## 3. 평면 클로저 변환

### 3.1 개념

**클로저 변환(closure conversion)**은 각 함수를 (코드 포인터, 환경 레코드) 쌍으로 표현합니다. 환경 레코드는 모든 자유 변수의 값을 저장합니다.

```
Closure = (function_pointer, environment_record)

# A closure for \x -> x + n where n=5:
closure = (code_for_adder, {n: 5})

# Calling the closure:
result = closure.code(closure.env, x)
```

### 3.2 구현

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

### 3.3 예제

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

### 3.4 평면 클로저 vs. 연결 클로저

**평면 클로저(flat closure)**: 모든 자유 변수를 클로저의 환경 레코드에 복사합니다. 각 클로저가 자체적으로 완결됩니다.

```
make_adder(5) creates:
  env = {n: 5}
  closure = (adder_code, env)
```

---

## 4. 연결(공유) 클로저

### 4.1 개념

모든 자유 변수를 복사하는 대신, 연결 클로저(linked closure)는 둘러싸는 환경에 대한 포인터를 저장합니다:

```
Linked closure:
  env = {local_vars..., parent: enclosing_env}

Access to outer variable:
  env.parent.parent. ... .variable
```

### 4.2 트레이드오프

| 측면 | 평면 클로저 | 연결 클로저 |
|------|-----------|-----------|
| 생성 비용 | 모든 자유 변수 복사 | 포인터 하나 |
| 접근 비용 | 직접 필드 접근 | 포인터 역참조 체인 |
| 메모리 | 값이 중복될 수 있음 | 부모와 공유 |
| GC 상호작용 | 독립적 수명 | 부모를 살려놓음 |
| 사용 언어 | OCaml, SML | Python, JavaScript |

### 4.3 디스플레이 최적화(Display Optimization)

깊이 중첩된 클로저의 경우, **디스플레이(display)** 배열은 어떤 중첩 수준에든 O(1) 접근을 제공합니다:

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

## 5. 탈함수화

### 5.1 개념

**탈함수화(defunctionalization)**(Reynolds, 1972)는 고차 함수를 데이터 구성자(data constructor)와 단일 `apply` 함수로 대체하여 완전히 제거합니다:

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

### 5.2 알고리즘

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

### 5.3 탈함수화를 사용할 때

- 클로저가 없는 언어(C, Fortran)로 컴파일할 때
- 전체 프로그램 최적화(모든 호출 지점이 알려진 경우)
- 함수형 언어에서 명령형 타겟으로의 컴파일

---

## 6. CPS 변환

### 6.1 CPS란?

**연속 전달 스타일(Continuation-Passing Style)**에서 모든 함수는 추가 매개변수 -- 연속(continuation) -- 을 받으며, 이것은 "다음에 할 것"을 나타냅니다:

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

### 6.2 컴파일러에서 CPS를 사용하는 이유

CPS는 제어 흐름을 명시적으로 만듭니다:
- 모든 함수 호출이 꼬리 호출(tail call)이 됩니다 (꼬리 호출 최적화로 스택이 증가하지 않음)
- 연속은 그냥 클로저입니다 -- 균일한 표현
- 예외 처리와 코루틴 같은 변환을 쉽게 만듭니다

### 6.3 CPS 변환 알고리즘

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

### 6.4 CPS 예제

```python
# Direct style:
let x = 3 + 4 in x * 2

# CPS:
add(3, 4, lambda x: mul(x, 2, lambda result: halt(result)))

# Every operation receives its continuation explicitly.
# No implicit "return" -- result flows through continuations.
```

---

## 7. 실무에서의 클로저

### 7.1 Python

Python은 셀 객체(cell object)를 사용한 연결 클로저를 사용합니다:

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

V8은 포획된 변수를 위해 "Context" 객체를 생성합니다:

```javascript
function makeAdder(n) {
    // V8 allocates a Context object: {n: n}
    return function(x) {
        return x + n;  // accesses n via Context pointer
    };
}
```

### 7.3 OCaml

OCaml은 평면 클로저를 사용합니다 -- 자유 변수가 클로저 블록에 직접 복사됩니다:

```ocaml
let make_adder n =
  fun x -> x + n
(* Compiles to: alloc [code_ptr; n] *)
(* Closure is a heap block: [header | code_ptr | n_value] *)
```

---

## 8. 요약

- **클로저**는 포획된 환경과 함께 함수를 묶습니다
- **람다 리프팅**은 매개변수를 추가하여 자유 변수를 제거합니다
- **평면 클로저 변환**은 복사된 자유 변수와 함께 (코드, 환경) 쌍을 생성합니다
- **연결 클로저**는 부모 포인터를 통해 환경 프레임을 공유합니다
- **탈함수화**는 고차 함수를 데이터 + 디스패치로 대체합니다
- **CPS 변환**은 연속을 명시적으로 만들어 꼬리 호출 최적화를 가능하게 합니다
- 실제 언어들은 다양한 전략을 사용합니다: Python(연결 셀), V8(컨텍스트), OCaml(평면 클로저)

---

## 9. 연습 문제

1. **자유 변수**: 여러 중첩 람다 표현식의 자유 변수를 계산하세요.

2. **람다 리프팅**: 3단계 중첩이 있는 프로그램을 람다 리프팅으로 변환하세요.

3. **클로저 변환**: 작은 함수형 언어에 대해 평면 클로저 변환을 구현하세요.

4. **탈함수화**: `map`, `filter`, `fold`를 람다 인수와 함께 사용하는 프로그램을 탈함수화하세요.

5. **CPS 변환**: 재귀적 팩토리얼 함수를 CPS로 변환하고 실행을 추적하세요.

---

## 10. 참고 자료

1. Appel, A. W. (1992). *Compiling with Continuations*. Cambridge University Press.
2. Reynolds, J. C. (1972). "Definitional Interpreters for Higher-Order Programming Languages." *Higher-Order and Symbolic Computation*, 11(4).
3. Johnsson, T. (1985). "Lambda Lifting: Transforming Programs to Recursive Equations." *FPCA*.
4. Shao, Z., Appel, A. W. (1994). "Space-Efficient Closure Representations." *LISP and Functional Programming*.
5. Danvy, O., Filinski, A. (1992). "Representing Control: A Study of the CPS Transformation." *MSCS*, 2(4).

---

**이전**: [23. 타입 추론](./23_Type_Inference.md) | **다음**: [25. 링킹과 로딩](./25_Linking_and_Loading.md)
