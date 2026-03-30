/*
 * Exercises for Lesson 08: Collections
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: List<T> operations — add, remove, sort, search
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: List<T> Operations ===");

    var names = new List<string> { "Charlie", "Alice", "Eve", "Bob", "Diana" };
    Console.WriteLine($"Original: [{string.Join(", ", names)}]");

    names.Sort();
    Console.WriteLine($"Sorted:   [{string.Join(", ", names)}]");

    names.Insert(2, "Frank");
    Console.WriteLine($"Insert(2, Frank): [{string.Join(", ", names)}]");

    names.Remove("Eve");
    Console.WriteLine($"Remove(Eve): [{string.Join(", ", names)}]");

    int idx = names.BinarySearch("Diana");
    Console.WriteLine($"BinarySearch(Diana): index={idx}");

    bool exists = names.Exists(n => n.StartsWith("B"));
    Console.WriteLine($"Any name starts with 'B': {exists}");

    var longNames = names.FindAll(n => n.Length > 4);
    Console.WriteLine($"Names longer than 4: [{string.Join(", ", longNames)}]");
    Console.WriteLine();
}

// Exercise 2: Dictionary<TKey, TValue> — word frequency counter
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Dictionary — Word Frequency ===");

    string text = "the quick brown fox jumps over the lazy dog the fox the dog";
    var frequency = new Dictionary<string, int>();

    foreach (string word in text.Split(' '))
    {
        if (frequency.ContainsKey(word))
            frequency[word]++;
        else
            frequency[word] = 1;
    }

    Console.WriteLine("Word frequencies:");
    foreach (var kvp in frequency.OrderByDescending(kvp => kvp.Value))
        Console.WriteLine($"  {kvp.Key,-10} : {kvp.Value}");

    // TryGetValue pattern
    string search = "fox";
    if (frequency.TryGetValue(search, out int count))
        Console.WriteLine($"\n\"{search}\" appears {count} time(s)");

    // GetValueOrDefault
    int missing = frequency.GetValueOrDefault("cat", 0);
    Console.WriteLine($"\"cat\" appears {missing} time(s)");
    Console.WriteLine();
}

// Exercise 3: HashSet<T> — set operations
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: HashSet — Set Operations ===");

    var setA = new HashSet<int> { 1, 2, 3, 4, 5, 6, 7, 8 };
    var setB = new HashSet<int> { 5, 6, 7, 8, 9, 10, 11, 12 };

    Console.WriteLine($"Set A: {{{string.Join(", ", setA.Order())}}}");
    Console.WriteLine($"Set B: {{{string.Join(", ", setB.Order())}}}");

    // Union
    var union = new HashSet<int>(setA);
    union.UnionWith(setB);
    Console.WriteLine($"A ∪ B: {{{string.Join(", ", union.Order())}}}");

    // Intersection
    var intersection = new HashSet<int>(setA);
    intersection.IntersectWith(setB);
    Console.WriteLine($"A ∩ B: {{{string.Join(", ", intersection.Order())}}}");

    // Difference
    var difference = new HashSet<int>(setA);
    difference.ExceptWith(setB);
    Console.WriteLine($"A - B: {{{string.Join(", ", difference.Order())}}}");

    // Symmetric difference
    var symDiff = new HashSet<int>(setA);
    symDiff.SymmetricExceptWith(setB);
    Console.WriteLine($"A △ B: {{{string.Join(", ", symDiff.Order())}}}");

    // Subset check
    var subset = new HashSet<int> { 2, 4, 6 };
    Console.WriteLine($"\n{{{string.Join(", ", subset)}}} ⊂ A? {subset.IsSubsetOf(setA)}");
    Console.WriteLine($"{{{string.Join(", ", subset)}}} ⊂ B? {subset.IsSubsetOf(setB)}");
    Console.WriteLine();
}

// Exercise 4: Queue<T> and Stack<T> — task processing
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Queue and Stack ===");

    // Queue — task processing
    var taskQueue = new Queue<string>();
    taskQueue.Enqueue("Build project");
    taskQueue.Enqueue("Run tests");
    taskQueue.Enqueue("Deploy to staging");
    taskQueue.Enqueue("Run integration tests");
    taskQueue.Enqueue("Deploy to production");

    Console.WriteLine($"Task queue ({taskQueue.Count} tasks):");
    while (taskQueue.Count > 0)
    {
        string task = taskQueue.Dequeue();
        Console.WriteLine($"  Processing: {task} ({taskQueue.Count} remaining)");
    }

    // Stack — undo system
    Console.WriteLine("\nUndo stack:");
    var undoStack = new Stack<string>();
    string[] actions = { "Type 'Hello'", "Type ' World'", "Bold selection", "Change font", "Delete word" };

    foreach (string action in actions)
    {
        undoStack.Push(action);
        Console.WriteLine($"  Do: {action}");
    }

    Console.WriteLine("\nUndoing:");
    while (undoStack.Count > 0)
        Console.WriteLine($"  Undo: {undoStack.Pop()}");

    // Stack — bracket matching
    Console.WriteLine("\nBracket matching:");
    string[] expressions = { "(a + b) * [c - d]", "((a + b)", "{[()]}", "{[(])}" };
    foreach (string expr in expressions)
        Console.WriteLine($"  \"{expr}\" -> {(IsBalanced(expr) ? "balanced" : "unbalanced")}");
    Console.WriteLine();

    static bool IsBalanced(string s)
    {
        var stack = new Stack<char>();
        foreach (char c in s)
        {
            if ("([{".Contains(c)) stack.Push(c);
            else if (")]}".Contains(c))
            {
                if (stack.Count == 0) return false;
                char open = stack.Pop();
                if ((c == ')' && open != '(') || (c == ']' && open != '[') || (c == '}' && open != '{'))
                    return false;
            }
        }
        return stack.Count == 0;
    }
}

// Exercise 5: LINQ queries on collections
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: LINQ Queries ===");

    var students = new List<(string Name, int Age, double Gpa)>
    {
        ("Alice", 22, 3.8),
        ("Bob", 20, 3.2),
        ("Charlie", 23, 3.9),
        ("Diana", 21, 3.5),
        ("Eve", 22, 3.7),
        ("Frank", 20, 2.9),
        ("Grace", 23, 3.6),
        ("Henry", 21, 3.1)
    };

    // Filter and sort
    var honorRoll = students
        .Where(s => s.Gpa >= 3.5)
        .OrderByDescending(s => s.Gpa)
        .Select(s => $"{s.Name} (GPA: {s.Gpa})");
    Console.WriteLine($"Honor Roll: {string.Join(", ", honorRoll)}");

    // Aggregate
    double avgGpa = students.Average(s => s.Gpa);
    double maxGpa = students.Max(s => s.Gpa);
    string topStudent = students.MaxBy(s => s.Gpa).Name;
    Console.WriteLine($"Average GPA: {avgGpa:F2}");
    Console.WriteLine($"Highest GPA: {maxGpa} ({topStudent})");

    // Group by age
    Console.WriteLine("\nStudents by age:");
    var byAge = students.GroupBy(s => s.Age).OrderBy(g => g.Key);
    foreach (var group in byAge)
        Console.WriteLine($"  Age {group.Key}: {string.Join(", ", group.Select(s => s.Name))}");

    // Chained transformations
    var summary = students
        .GroupBy(s => s.Age)
        .Select(g => new { Age = g.Key, Count = g.Count(), AvgGpa = g.Average(s => s.Gpa) })
        .OrderBy(x => x.Age);
    Console.WriteLine("\nAge group summary:");
    foreach (var s in summary)
        Console.WriteLine($"  Age {s.Age}: {s.Count} students, avg GPA={s.AvgGpa:F2}");
    Console.WriteLine();
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
