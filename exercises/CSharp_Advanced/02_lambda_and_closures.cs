/*
 * Exercises for Lesson 02: Lambda Expressions and Closures
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Linq;

// ---------------------------------------------------------------------------
// Exercise 1: Lambda pipeline — compose transformations on a list of names
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Lambda Pipeline ===");

    List<string> names = new() { "alice", "BOB", "Charlie", "dave", "EVE" };

    Func<string, string> capitalize = s =>
        string.IsNullOrEmpty(s) ? s : char.ToUpper(s[0]) + s[1..].ToLower();

    Func<string, string> addGreeting = s => $"Hello, {s}!";

    Func<string, string> pipeline = s => addGreeting(capitalize(s));

    var results = names.Select(pipeline).ToList();
    results.ForEach(Console.WriteLine);
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Closure capture bug — classic loop variable problem & fix
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Closure Capture Fix ===");

    // Broken version: all closures capture same variable
    var brokenActions = new List<Action>();
    for (int i = 0; i < 5; i++)
    {
        int captured = i; // FIX: capture a copy inside the loop
        brokenActions.Add(() => Console.WriteLine($"  Fixed closure value: {captured}"));
    }
    brokenActions.ForEach(a => a());
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Higher-order function — function factory
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Function Factory ===");

    Func<double, Func<double, double>> multiplierFactory = factor =>
        value => value * factor;

    var doubler = multiplierFactory(2.0);
    var tripler = multiplierFactory(3.0);
    var half = multiplierFactory(0.5);

    Console.WriteLine($"  doubler(5)  = {doubler(5)}");
    Console.WriteLine($"  tripler(5)  = {tripler(5)}");
    Console.WriteLine($"  half(5)     = {half(5)}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Memoization with closures
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Memoization ===");

    Func<int, long> fibonacci = null!;
    fibonacci = Memoize<int, long>(n => n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2));

    for (int i = 0; i <= 15; i++)
        Console.Write($"  F({i})={fibonacci(i)}");
    Console.WriteLine();
    Console.WriteLine();
}

Func<TIn, TOut> Memoize<TIn, TOut>(Func<TIn, TOut> func) where TIn : notnull
{
    var cache = new Dictionary<TIn, TOut>();
    return input =>
    {
        if (!cache.TryGetValue(input, out var result))
        {
            result = func(input);
            cache[input] = result;
        }
        return result;
    };
}

// ---------------------------------------------------------------------------
// Exercise 5: Predicate combinator — AND / OR / NOT for predicates
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Predicate Combinators ===");

    Func<int, bool> isEven = n => n % 2 == 0;
    Func<int, bool> isPositive = n => n > 0;
    Func<int, bool> isSmall = n => Math.Abs(n) < 10;

    Func<int, bool> isEvenAndPositive = And(isEven, isPositive);
    Func<int, bool> isEvenOrSmall = Or(isEven, isSmall);
    Func<int, bool> isOdd = Not(isEven);

    int[] testValues = { -12, -3, 0, 4, 7, 16 };

    Console.WriteLine("  Value  Even&&Pos  Even||Small  Odd");
    foreach (var v in testValues)
    {
        Console.WriteLine($"  {v,5}  {isEvenAndPositive(v),9}  {isEvenOrSmall(v),10}  {isOdd(v)}");
    }
    Console.WriteLine();
}

Func<T, bool> And<T>(Func<T, bool> a, Func<T, bool> b) => x => a(x) && b(x);
Func<T, bool> Or<T>(Func<T, bool> a, Func<T, bool> b) => x => a(x) || b(x);
Func<T, bool> Not<T>(Func<T, bool> a) => x => !a(x);

// ---------------------------------------------------------------------------
// Exercise 6: Currying — transform a multi-parameter function
// ---------------------------------------------------------------------------
void Exercise6()
{
    Console.WriteLine("=== Exercise 6: Currying ===");

    Func<int, int, int, int> add3 = (a, b, c) => a + b + c;

    var curried = Curry(add3);
    var add10 = curried(10);
    var add10And20 = add10(20);

    Console.WriteLine($"  curried(10)(20)(30) = {add10And20(30)}");
    Console.WriteLine($"  curried(1)(2)(3)    = {curried(1)(2)(3)}");
    Console.WriteLine();
}

Func<T1, Func<T2, Func<T3, TResult>>> Curry<T1, T2, T3, TResult>(
    Func<T1, T2, T3, TResult> func)
    => a => b => c => func(a, b, c);

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
Exercise6();
