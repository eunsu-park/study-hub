# Control Flow

**Previous**: [Operators and Expressions](./03_Operators_and_Expressions.md) | **Next**: [Methods](./05_Methods.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `if`, `else if`, and `else` statements for conditional execution
2. Write `switch` statements with constant patterns
3. Use `switch` expressions for concise value mapping
4. Iterate with `for`, `foreach`, `while`, and `do-while` loops
5. Control loop execution with `break`, `continue`, and `return`
6. Handle nested loops and use `goto` for labeled breaks
7. Apply basic pattern matching in `switch` constructs

---

Control flow statements determine the order in which statements are executed. Without them, programs would run every line top to bottom without any branching or repetition. C# provides a rich set of conditional and looping constructs that you will use in every program.

## 1. The if Statement

The `if` statement executes a block of code only when a condition evaluates to `true`:

### 1.1 Simple if

```csharp
int temperature = 35;

if (temperature > 30)
{
    Console.WriteLine("It's hot outside!");
}
```

### 1.2 if-else

```csharp
int age = 16;

if (age >= 18)
{
    Console.WriteLine("You are an adult.");
}
else
{
    Console.WriteLine("You are a minor.");
}
```

### 1.3 if-else if-else Chain

```csharp
int score = 85;

if (score >= 90)
{
    Console.WriteLine("Grade: A");
}
else if (score >= 80)
{
    Console.WriteLine("Grade: B");
}
else if (score >= 70)
{
    Console.WriteLine("Grade: C");
}
else if (score >= 60)
{
    Console.WriteLine("Grade: D");
}
else
{
    Console.WriteLine("Grade: F");
}
```

### 1.4 Nested if Statements

```csharp
int age = 25;
bool hasLicense = true;

if (age >= 18)
{
    if (hasLicense)
    {
        Console.WriteLine("You can drive.");
    }
    else
    {
        Console.WriteLine("You need a license first.");
    }
}
else
{
    Console.WriteLine("You are too young to drive.");
}

// Flattened version using logical AND (preferred when simple)
if (age >= 18 && hasLicense)
{
    Console.WriteLine("You can drive.");
}
```

### 1.5 Single-Line if (Without Braces)

```csharp
int x = 10;

// Legal but discouraged — easy to introduce bugs
if (x > 5)
    Console.WriteLine("x is greater than 5");

// Always use braces — clearer and safer
if (x > 5)
{
    Console.WriteLine("x is greater than 5");
}
```

### 1.6 Conditional Assignment Patterns

```csharp
// Pattern: assign based on condition
int temperature = 28;
string description;

if (temperature > 30)
    description = "hot";
else if (temperature > 20)
    description = "warm";
else if (temperature > 10)
    description = "cool";
else
    description = "cold";

Console.WriteLine($"The weather is {description}.");

// Shorter with ternary (for simple cases)
string simple = temperature > 25 ? "warm" : "not warm";
```

## 2. The switch Statement

The `switch` statement is an alternative to long `if-else if` chains when you are comparing a single value against multiple constants:

### 2.1 Basic switch with Constants

```csharp
int dayNumber = 3;

switch (dayNumber)
{
    case 1:
        Console.WriteLine("Monday");
        break;
    case 2:
        Console.WriteLine("Tuesday");
        break;
    case 3:
        Console.WriteLine("Wednesday");
        break;
    case 4:
        Console.WriteLine("Thursday");
        break;
    case 5:
        Console.WriteLine("Friday");
        break;
    case 6:
        Console.WriteLine("Saturday");
        break;
    case 7:
        Console.WriteLine("Sunday");
        break;
    default:
        Console.WriteLine("Invalid day number");
        break;
}
```

### 2.2 Multiple Case Labels (Fall-Through)

In C#, you cannot fall through from one case to another unless the case body is empty. Multiple labels can share the same body:

```csharp
int month = 4;

switch (month)
{
    case 1: case 3: case 5: case 7: case 8: case 10: case 12:
        Console.WriteLine("31 days");
        break;
    case 4: case 6: case 9: case 11:
        Console.WriteLine("30 days");
        break;
    case 2:
        Console.WriteLine("28 or 29 days");
        break;
    default:
        Console.WriteLine("Invalid month");
        break;
}
```

### 2.3 switch with Strings

```csharp
string command = "start";

switch (command.ToLower())
{
    case "start":
        Console.WriteLine("Starting the engine...");
        break;
    case "stop":
        Console.WriteLine("Stopping the engine...");
        break;
    case "pause":
        Console.WriteLine("Pausing...");
        break;
    case "resume":
        Console.WriteLine("Resuming...");
        break;
    default:
        Console.WriteLine($"Unknown command: {command}");
        break;
}
```

### 2.4 switch with when Guard

```csharp
int number = 42;

switch (number)
{
    case int n when n < 0:
        Console.WriteLine("Negative");
        break;
    case 0:
        Console.WriteLine("Zero");
        break;
    case int n when n > 0 && n <= 10:
        Console.WriteLine("Small positive");
        break;
    case int n when n > 10 && n <= 100:
        Console.WriteLine("Medium positive");
        break;
    default:
        Console.WriteLine("Large positive");
        break;
}
```

## 3. Switch Expressions

C# 8 introduced **switch expressions**, a more concise syntax for mapping a value to a result:

### 3.1 Basic Switch Expression

```csharp
int dayNumber = 5;

string dayName = dayNumber switch
{
    1 => "Monday",
    2 => "Tuesday",
    3 => "Wednesday",
    4 => "Thursday",
    5 => "Friday",
    6 => "Saturday",
    7 => "Sunday",
    _ => "Invalid"  // _ is the discard pattern (like default)
};

Console.WriteLine(dayName);  // Friday
```

### 3.2 Switch Expression with Conditions

```csharp
int score = 85;

string grade = score switch
{
    >= 90 => "A",
    >= 80 => "B",
    >= 70 => "C",
    >= 60 => "D",
    _ => "F"
};

Console.WriteLine($"Score {score} = Grade {grade}");  // Score 85 = Grade B
```

### 3.3 Switch Expression with Multiple Patterns

```csharp
int month = 7;

int daysInMonth = month switch
{
    1 or 3 or 5 or 7 or 8 or 10 or 12 => 31,
    4 or 6 or 9 or 11 => 30,
    2 => 28,
    _ => throw new ArgumentException($"Invalid month: {month}")
};

Console.WriteLine($"Month {month} has {daysInMonth} days.");
```

### 3.4 Switch Expression with Tuples

```csharp
string season = (month: 7, hemisphere: "north") switch
{
    (>= 3 and <= 5, "north") => "Spring",
    (>= 6 and <= 8, "north") => "Summer",
    (>= 9 and <= 11, "north") => "Autumn",
    (12 or 1 or 2, "north") => "Winter",
    (>= 3 and <= 5, "south") => "Autumn",
    (>= 6 and <= 8, "south") => "Winter",
    (>= 9 and <= 11, "south") => "Spring",
    (12 or 1 or 2, "south") => "Summer",
    _ => "Unknown"
};

Console.WriteLine(season);  // Summer
```

## 4. The for Loop

The `for` loop repeats a block of code a specific number of times:

### 4.1 Basic for Loop

```csharp
// Print numbers 1 through 5
for (int i = 1; i <= 5; i++)
{
    Console.WriteLine($"Iteration {i}");
}

// Structure: for (initializer; condition; iterator)
// 1. initializer runs once before the loop
// 2. condition is checked before each iteration
// 3. iterator runs after each iteration
```

### 4.2 Counting Down

```csharp
for (int i = 10; i >= 1; i--)
{
    Console.Write($"{i} ");
}
Console.WriteLine("Liftoff!");
// Output: 10 9 8 7 6 5 4 3 2 1 Liftoff!
```

### 4.3 Step Size

```csharp
// Count by 2s
for (int i = 0; i <= 20; i += 2)
{
    Console.Write($"{i} ");
}
Console.WriteLine();
// Output: 0 2 4 6 8 10 12 14 16 18 20

// Count by 3s
for (int i = 0; i < 30; i += 3)
{
    Console.Write($"{i} ");
}
Console.WriteLine();
```

### 4.4 Multiple Variables

```csharp
// Two variables in a for loop
for (int i = 0, j = 10; i < j; i++, j--)
{
    Console.WriteLine($"i={i}, j={j}");
}
// Output:
// i=0, j=10
// i=1, j=9
// i=2, j=8
// i=3, j=7
// i=4, j=6
```

### 4.5 Infinite Loop

```csharp
// Infinite loop (use break to exit)
// for (;;)
// {
//     Console.Write("Enter 'quit' to exit: ");
//     string? input = Console.ReadLine();
//     if (input == "quit") break;
//     Console.WriteLine($"You entered: {input}");
// }
```

### 4.6 Common Patterns

```csharp
// Sum of numbers 1 to 100
int sum = 0;
for (int i = 1; i <= 100; i++)
{
    sum += i;
}
Console.WriteLine($"Sum 1-100: {sum}");  // 5050

// Factorial
int n = 10;
long factorial = 1;
for (int i = 2; i <= n; i++)
{
    factorial *= i;
}
Console.WriteLine($"{n}! = {factorial}");  // 3628800

// Powers of 2
for (int i = 0; i < 16; i++)
{
    Console.WriteLine($"2^{i} = {1 << i}");
}
```

## 5. The foreach Loop

The `foreach` loop iterates over elements in a collection without managing an index:

```csharp
// Array
int[] numbers = { 10, 20, 30, 40, 50 };
foreach (int num in numbers)
{
    Console.Write($"{num} ");
}
Console.WriteLine();

// String (iterates over characters)
string word = "Hello";
foreach (char ch in word)
{
    Console.Write($"'{ch}' ");
}
Console.WriteLine();
// Output: 'H' 'e' 'l' 'l' 'o'

// List
List<string> fruits = new() { "Apple", "Banana", "Cherry" };
foreach (string fruit in fruits)
{
    Console.WriteLine($"  - {fruit}");
}

// Dictionary
Dictionary<string, int> ages = new()
{
    ["Alice"] = 30,
    ["Bob"] = 25,
    ["Charlie"] = 35
};
foreach (KeyValuePair<string, int> kvp in ages)
{
    Console.WriteLine($"{kvp.Key} is {kvp.Value} years old.");
}

// Tuple deconstruction in foreach (C# 7+)
foreach (var (name, age) in ages)
{
    Console.WriteLine($"{name}: {age}");
}

// Range (C# with LINQ)
foreach (int i in Enumerable.Range(1, 5))
{
    Console.Write($"{i} ");  // 1 2 3 4 5
}
Console.WriteLine();
```

### 5.1 foreach vs for: When to Use Which

```csharp
int[] data = { 10, 20, 30, 40, 50 };

// Use foreach when you just need each element
foreach (int item in data)
{
    Console.WriteLine(item);
}

// Use for when you need the index
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($"data[{i}] = {data[i]}");
}

// Use for when you need to modify elements
for (int i = 0; i < data.Length; i++)
{
    data[i] *= 2;  // Cannot do this with foreach
}
```

## 6. The while Loop

The `while` loop repeats as long as its condition is `true`. The condition is checked **before** each iteration:

```csharp
// Count up
int count = 1;
while (count <= 5)
{
    Console.WriteLine($"Count: {count}");
    count++;
}

// Read input until valid
Console.Write("Enter a positive number: ");
int number = 0;
while (number <= 0)
{
    string? input = Console.ReadLine();
    if (int.TryParse(input, out number) && number > 0)
    {
        Console.WriteLine($"You entered: {number}");
    }
    else
    {
        number = 0;
        Console.Write("Invalid. Try again: ");
    }
}

// Collatz conjecture
int n = 27;
int steps = 0;
Console.Write($"{n}");
while (n != 1)
{
    n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
    Console.Write($" → {n}");
    steps++;
}
Console.WriteLine($"\nReached 1 in {steps} steps.");
```

## 7. The do-while Loop

The `do-while` loop is similar to `while`, but the condition is checked **after** each iteration. This guarantees at least one execution:

```csharp
// Always executes at least once
int count = 10;
do
{
    Console.WriteLine($"Count: {count}");
    count++;
} while (count <= 5);
// Output: Count: 10 (runs once even though 10 > 5)

// Menu system — natural use case for do-while
int choice;
do
{
    Console.WriteLine("\n=== Menu ===");
    Console.WriteLine("1. Say Hello");
    Console.WriteLine("2. Show Date");
    Console.WriteLine("3. Show Random Number");
    Console.WriteLine("0. Exit");
    Console.Write("Choice: ");

    if (!int.TryParse(Console.ReadLine(), out choice))
    {
        choice = -1;
    }

    switch (choice)
    {
        case 1:
            Console.WriteLine("Hello!");
            break;
        case 2:
            Console.WriteLine($"Today: {DateTime.Now:yyyy-MM-dd}");
            break;
        case 3:
            Console.WriteLine($"Random: {Random.Shared.Next(1, 101)}");
            break;
        case 0:
            Console.WriteLine("Goodbye!");
            break;
        default:
            Console.WriteLine("Invalid choice.");
            break;
    }
} while (choice != 0);
```

### 7.1 while vs do-while

```csharp
// while: check first, may never execute
int x = 10;
while (x < 5)
{
    Console.WriteLine("This never prints");
    x++;
}

// do-while: execute first, check after
x = 10;
do
{
    Console.WriteLine("This prints once");  // Prints!
    x++;
} while (x < 5);
```

## 8. break, continue, and return

### 8.1 break — Exit the Loop

```csharp
// Find the first multiple of 7 greater than 50
for (int i = 51; ; i++)
{
    if (i % 7 == 0)
    {
        Console.WriteLine($"Found: {i}");  // 56
        break;
    }
}

// break in a while loop
int sum = 0;
int num = 1;
while (true)
{
    sum += num;
    if (sum > 100)
    {
        Console.WriteLine($"Sum exceeded 100 at num={num}, sum={sum}");
        break;
    }
    num++;
}
```

### 8.2 continue — Skip to Next Iteration

```csharp
// Print only odd numbers
for (int i = 1; i <= 20; i++)
{
    if (i % 2 == 0) continue;  // Skip even numbers
    Console.Write($"{i} ");
}
Console.WriteLine();
// Output: 1 3 5 7 9 11 13 15 17 19

// Skip blank lines when processing input
string[] lines = { "Hello", "", "World", "  ", "C#" };
foreach (string line in lines)
{
    if (string.IsNullOrWhiteSpace(line)) continue;
    Console.WriteLine($"Processing: '{line}'");
}
```

### 8.3 return — Exit the Method

```csharp
// return exits the entire method, not just the loop
int FindFirst(int[] arr, int target)
{
    for (int i = 0; i < arr.Length; i++)
    {
        if (arr[i] == target)
            return i;  // Exit method immediately
    }
    return -1;  // Not found
}

int[] data = { 5, 3, 8, 1, 9 };
int index = FindFirst(data, 8);
Console.WriteLine($"Found at index: {index}");  // 2
```

## 9. Nested Loops and goto

### 9.1 Nested Loops

```csharp
// Multiplication table
for (int i = 1; i <= 9; i++)
{
    for (int j = 1; j <= 9; j++)
    {
        Console.Write($"{i * j,4}");
    }
    Console.WriteLine();
}

// Triangle pattern
for (int i = 1; i <= 5; i++)
{
    for (int j = 0; j < i; j++)
    {
        Console.Write("* ");
    }
    Console.WriteLine();
}
// Output:
// *
// * *
// * * *
// * * * *
// * * * * *
```

### 9.2 Breaking Out of Nested Loops with goto

In C#, `break` only exits the innermost loop. To break out of multiple loops, you can use `goto` with a label:

```csharp
// Find the first pair (i, j) where i * j == 42
for (int i = 1; i <= 10; i++)
{
    for (int j = 1; j <= 10; j++)
    {
        if (i * j == 42)
        {
            Console.WriteLine($"Found: {i} * {j} = 42");
            goto FoundIt;  // Exit both loops
        }
    }
}
Console.WriteLine("Not found.");
FoundIt:
Console.WriteLine("Search complete.");

// Alternative: use a flag variable (no goto)
bool found = false;
for (int i = 1; i <= 10 && !found; i++)
{
    for (int j = 1; j <= 10 && !found; j++)
    {
        if (i * j == 42)
        {
            Console.WriteLine($"Found: {i} * {j} = 42");
            found = true;
        }
    }
}

// Alternative: extract to a method and use return
(int i, int j) FindProduct(int target)
{
    for (int i = 1; i <= 10; i++)
        for (int j = 1; j <= 10; j++)
            if (i * j == target)
                return (i, j);
    return (-1, -1);
}

var result = FindProduct(42);
Console.WriteLine($"Found: {result.i} * {result.j} = 42");
```

## 10. Pattern Matching Basics in switch

C# has powerful pattern matching that extends `switch` beyond simple constant comparisons:

### 10.1 Type Patterns

```csharp
object value = 42;

switch (value)
{
    case int i:
        Console.WriteLine($"Integer: {i}");
        break;
    case string s:
        Console.WriteLine($"String: {s}");
        break;
    case double d:
        Console.WriteLine($"Double: {d}");
        break;
    case null:
        Console.WriteLine("Null value");
        break;
    default:
        Console.WriteLine($"Other type: {value.GetType().Name}");
        break;
}
```

### 10.2 Relational Patterns (C# 9+)

```csharp
int temperature = 25;

string description = temperature switch
{
    < 0 => "Freezing",
    >= 0 and < 10 => "Cold",
    >= 10 and < 20 => "Cool",
    >= 20 and < 30 => "Warm",
    >= 30 and < 40 => "Hot",
    >= 40 => "Extreme heat",
};

Console.WriteLine($"{temperature}°C: {description}");
```

### 10.3 Logical Patterns (and, or, not)

```csharp
char ch = 'A';

string category = ch switch
{
    >= 'a' and <= 'z' => "lowercase letter",
    >= 'A' and <= 'Z' => "uppercase letter",
    >= '0' and <= '9' => "digit",
    ' ' or '\t' or '\n' => "whitespace",
    not (' ' or '\t' or '\n') and (>= ' ' and <= '~') => "symbol",
    _ => "other"
};

Console.WriteLine($"'{ch}' is a {category}");  // 'A' is a uppercase letter
```

### 10.4 Property Patterns

```csharp
// Preview of property patterns (used more with classes)
var point = new { X = 3, Y = 4 };

string quadrant = point switch
{
    { X: 0, Y: 0 } => "Origin",
    { X: > 0, Y: > 0 } => "Quadrant I",
    { X: < 0, Y: > 0 } => "Quadrant II",
    { X: < 0, Y: < 0 } => "Quadrant III",
    { X: > 0, Y: < 0 } => "Quadrant IV",
    { X: 0 } or { Y: 0 } => "On an axis",
    _ => "Unknown"
};

Console.WriteLine($"({point.X}, {point.Y}) is in {quadrant}");
```

## 11. Practice Problems

1. **FizzBuzz**: Write a program that prints numbers from 1 to 100. For multiples of 3, print "Fizz" instead of the number. For multiples of 5, print "Buzz". For multiples of both 3 and 5, print "FizzBuzz". Use a `for` loop and `if`/`else if` statements.

2. **Number Guessing Game**: Generate a random number between 1 and 100 using `Random.Shared.Next(1, 101)`. Use a `do-while` loop to repeatedly ask the user to guess. After each guess, print "Too high", "Too low", or "Correct!". Count and display the number of attempts.

3. **Prime Number Finder**: Write a program that finds and prints all prime numbers between 2 and 200. Use a nested loop: the outer loop iterates through candidates, the inner loop checks for divisors. Use `break` to optimize the inner loop.

4. **Diamond Pattern**: Using nested `for` loops, print a diamond pattern of asterisks. The program should accept an odd number `n` from the user and print a diamond with `n` rows at its widest point. For example, with `n = 5`:
   ```
     *
    ***
   *****
    ***
     *
   ```

5. **Mini Calculator with Switch Expression**: Write a calculator that reads two numbers and an operator (`+`, `-`, `*`, `/`, `%`) from the user. Use a `switch` expression to perform the operation. Handle division by zero, invalid operators, and non-numeric input gracefully.

---

**Previous**: [Operators and Expressions](./03_Operators_and_Expressions.md) | **Next**: [Methods](./05_Methods.md)
