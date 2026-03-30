/*
 * Exercises for Lesson 02: Variables and Types
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Display the size and range of all numeric types
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Numeric Type Sizes and Ranges ===");
    Console.WriteLine($"{"Type",-12} {"Size (bytes)",14} {"Min",22} {"Max",22}");
    Console.WriteLine(new string('-', 72));
    Console.WriteLine($"{"byte",-12} {sizeof(byte),14} {byte.MinValue,22} {byte.MaxValue,22}");
    Console.WriteLine($"{"sbyte",-12} {sizeof(sbyte),14} {sbyte.MinValue,22} {sbyte.MaxValue,22}");
    Console.WriteLine($"{"short",-12} {sizeof(short),14} {short.MinValue,22} {short.MaxValue,22}");
    Console.WriteLine($"{"ushort",-12} {sizeof(ushort),14} {ushort.MinValue,22} {ushort.MaxValue,22}");
    Console.WriteLine($"{"int",-12} {sizeof(int),14} {int.MinValue,22} {int.MaxValue,22}");
    Console.WriteLine($"{"uint",-12} {sizeof(uint),14} {uint.MinValue,22} {uint.MaxValue,22}");
    Console.WriteLine($"{"long",-12} {sizeof(long),14} {long.MinValue,22} {long.MaxValue,22}");
    Console.WriteLine($"{"float",-12} {sizeof(float),14} {float.MinValue,22:E2} {float.MaxValue,22:E2}");
    Console.WriteLine($"{"double",-12} {sizeof(double),14} {double.MinValue,22:E2} {double.MaxValue,22:E2}");
    Console.WriteLine($"{"decimal",-12} {sizeof(decimal),14} {decimal.MinValue,22:E2} {decimal.MaxValue,22:E2}");
    Console.WriteLine();
}

// Exercise 2: Demonstrate implicit and explicit type conversions
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Type Conversions ===");

    // Implicit widening conversions
    int intVal = 42;
    long longVal = intVal;
    float floatVal = intVal;
    double doubleVal = intVal;
    Console.WriteLine($"int -> long:   {intVal} -> {longVal}");
    Console.WriteLine($"int -> float:  {intVal} -> {floatVal}");
    Console.WriteLine($"int -> double: {intVal} -> {doubleVal}");

    // Explicit narrowing conversions
    double pi = 3.14159265;
    int truncated = (int)pi;
    float narrowed = (float)pi;
    Console.WriteLine($"double -> int:   {pi} -> {truncated} (truncated)");
    Console.WriteLine($"double -> float: {pi} -> {narrowed} (precision loss)");

    // Convert class usage
    string numStr = "123";
    int parsed = Convert.ToInt32(numStr);
    double parsedD = Convert.ToDouble(numStr);
    Console.WriteLine($"string -> int:    \"{numStr}\" -> {parsed}");
    Console.WriteLine($"string -> double: \"{numStr}\" -> {parsedD}");

    // TryParse for safe conversion
    string bad = "not_a_number";
    bool success = int.TryParse(bad, out int result);
    Console.WriteLine($"TryParse(\"{bad}\"): success={success}, result={result}");
    Console.WriteLine();
}

// Exercise 3: Explore nullable value types and null-coalescing
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Nullable Types ===");

    int? maybeInt = null;
    Console.WriteLine($"maybeInt is null: {maybeInt == null}");
    Console.WriteLine($"maybeInt.HasValue: {maybeInt.HasValue}");
    Console.WriteLine($"maybeInt ?? -1: {maybeInt ?? -1}");

    maybeInt = 42;
    Console.WriteLine($"After assignment: {maybeInt.Value}");
    Console.WriteLine($"GetValueOrDefault: {maybeInt.GetValueOrDefault(-1)}");

    double? temperature = null;
    double display = temperature ?? double.NaN;
    Console.WriteLine($"Temperature: {display}");

    // Null-conditional chaining
    string? name = null;
    int? length = name?.Length;
    Console.WriteLine($"name?.Length: {length?.ToString() ?? "null"}");

    name = "Hello";
    length = name?.Length;
    Console.WriteLine($"name?.Length: {length}");

    // Null-coalescing assignment
    List<int>? numbers = null;
    numbers ??= new List<int>();
    numbers.Add(1);
    Console.WriteLine($"List count after ??= : {numbers.Count}");
    Console.WriteLine();
}

// Exercise 4: Demonstrate var, const, and readonly behavior
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: var, const, and Literals ===");

    var inferredInt = 42;
    var inferredStr = "hello";
    var inferredDouble = 3.14;
    Console.WriteLine($"var int:    {inferredInt} (type: {inferredInt.GetType().Name})");
    Console.WriteLine($"var string: {inferredStr} (type: {inferredStr.GetType().Name})");
    Console.WriteLine($"var double: {inferredDouble} (type: {inferredDouble.GetType().Name})");

    const double Pi = 3.14159265358979;
    const int MaxRetries = 3;
    Console.WriteLine($"const Pi = {Pi}");
    Console.WriteLine($"const MaxRetries = {MaxRetries}");

    // Numeric literals with digit separators
    int million = 1_000_000;
    long hex = 0xFF_EC_D5;
    int binary = 0b1010_0011;
    Console.WriteLine($"1_000_000 = {million}");
    Console.WriteLine($"0xFF_EC_D5 = {hex}");
    Console.WriteLine($"0b1010_0011 = {binary}");
    Console.WriteLine();
}

// Exercise 5: Build a type-safe unit converter using decimal precision
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Decimal Precision Converter ===");

    decimal usd = 100.00m;
    decimal eurRate = 0.92m;
    decimal gbpRate = 0.79m;
    decimal jpyRate = 149.50m;

    decimal eur = Math.Round(usd * eurRate, 2);
    decimal gbp = Math.Round(usd * gbpRate, 2);
    decimal jpy = Math.Round(usd * jpyRate, 0);

    Console.WriteLine($"${usd} USD = €{eur} EUR");
    Console.WriteLine($"${usd} USD = £{gbp} GBP");
    Console.WriteLine($"${usd} USD = ¥{jpy} JPY");

    // Show why decimal matters for financial math
    double dResult = 0.1 + 0.2;
    decimal mResult = 0.1m + 0.2m;
    Console.WriteLine($"double: 0.1 + 0.2 = {dResult} (exact? {dResult == 0.3})");
    Console.WriteLine($"decimal: 0.1 + 0.2 = {mResult} (exact? {mResult == 0.3m})");
    Console.WriteLine();
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
