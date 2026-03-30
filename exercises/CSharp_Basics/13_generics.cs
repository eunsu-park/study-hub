/*
 * Exercises for Lesson 13: Generics
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Implement a generic Stack<T>
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Generic Stack ===");

    var intStack = new GenericStack<int>(5);
    for (int i = 1; i <= 5; i++)
        intStack.Push(i * 10);

    Console.WriteLine($"Stack size: {intStack.Count}, capacity: {intStack.Capacity}");
    Console.WriteLine($"Peek: {intStack.Peek()}");

    Console.Write("Popping: ");
    while (!intStack.IsEmpty)
        Console.Write($"{intStack.Pop()} ");
    Console.WriteLine();

    // String stack
    var strStack = new GenericStack<string>();
    strStack.Push("Hello");
    strStack.Push("Generic");
    strStack.Push("World");
    Console.WriteLine($"\nString stack ({strStack.Count} items): Peek = \"{strStack.Peek()}\"");

    // Try pop from empty stack
    try
    {
        var empty = new GenericStack<int>();
        empty.Pop();
    }
    catch (InvalidOperationException ex)
    {
        Console.WriteLine($"Empty pop: {ex.Message}");
    }
    Console.WriteLine();
}

// Exercise 2: Generic constraints — a comparable collection
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Generic Constraints ===");

    var numbers = new SortedCollection<int>();
    foreach (int n in new[] { 42, 17, 85, 3, 66, 29 })
        numbers.Add(n);

    Console.WriteLine($"Sorted ints: [{string.Join(", ", numbers)}]");
    Console.WriteLine($"Min: {numbers.Min}, Max: {numbers.Max}");
    Console.WriteLine($"Contains 42: {numbers.Contains(42)}");
    Console.WriteLine($"Contains 99: {numbers.Contains(99)}");

    var words = new SortedCollection<string>();
    foreach (string w in new[] { "cherry", "apple", "banana", "date" })
        words.Add(w);

    Console.WriteLine($"\nSorted strings: [{string.Join(", ", words)}]");
    Console.WriteLine($"Min: {words.Min}, Max: {words.Max}");
    Console.WriteLine();
}

// Exercise 3: Generic methods — utility functions
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Generic Methods ===");

    // Swap
    int x = 10, y = 20;
    Console.Write($"Swap({x}, {y}) -> ");
    Swap(ref x, ref y);
    Console.WriteLine($"({x}, {y})");

    // FindAll with predicate
    int[] nums = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var evens = FindAll(nums, n => n % 2 == 0);
    Console.WriteLine($"Evens: [{string.Join(", ", evens)}]");

    var greaterThan5 = FindAll(nums, n => n > 5);
    Console.WriteLine($">5: [{string.Join(", ", greaterThan5)}]");

    // Transform / Map
    string[] names = { "alice", "bob", "charlie" };
    var upper = Map(names, s => s.ToUpper());
    Console.WriteLine($"Uppercase: [{string.Join(", ", upper)}]");

    var lengths = Map(names, s => s.Length);
    Console.WriteLine($"Lengths: [{string.Join(", ", lengths)}]");

    // Reduce / Fold
    int sum = Reduce(nums, 0, (acc, n) => acc + n);
    int product = Reduce(new[] { 1, 2, 3, 4, 5 }, 1, (acc, n) => acc * n);
    Console.WriteLine($"Sum(1..10): {sum}");
    Console.WriteLine($"Product(1..5): {product}");
    Console.WriteLine();

    static void Swap<T>(ref T a, ref T b) => (a, b) = (b, a);

    static List<T> FindAll<T>(T[] items, Func<T, bool> predicate)
    {
        var result = new List<T>();
        foreach (T item in items)
            if (predicate(item)) result.Add(item);
        return result;
    }

    static TOut[] Map<TIn, TOut>(TIn[] items, Func<TIn, TOut> transform)
    {
        var result = new TOut[items.Length];
        for (int i = 0; i < items.Length; i++)
            result[i] = transform(items[i]);
        return result;
    }

    static TAcc Reduce<T, TAcc>(T[] items, TAcc seed, Func<TAcc, T, TAcc> accumulator)
    {
        TAcc result = seed;
        foreach (T item in items)
            result = accumulator(result, item);
        return result;
    }
}

// Exercise 4: Generic repository pattern
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Generic Repository ===");

    var repo = new Repository<Product>();

    repo.Add(new Product(1, "Laptop", 999.99m));
    repo.Add(new Product(2, "Mouse", 29.99m));
    repo.Add(new Product(3, "Keyboard", 79.99m));
    repo.Add(new Product(4, "Monitor", 399.99m));
    repo.Add(new Product(5, "Headphones", 149.99m));

    Console.WriteLine($"All products ({repo.Count}):");
    foreach (var p in repo.GetAll())
        Console.WriteLine($"  {p}");

    Console.WriteLine($"\nFind by ID 3: {repo.GetById(3)}");

    var expensive = repo.FindWhere(p => p.Price > 100);
    Console.WriteLine($"\nExpensive (>$100): {string.Join(", ", expensive.Select(p => p.Name))}");

    repo.Remove(2);
    Console.WriteLine($"\nAfter removing ID 2, count: {repo.Count}");

    repo.Update(new Product(3, "Mechanical Keyboard", 129.99m));
    Console.WriteLine($"After update ID 3: {repo.GetById(3)}");
    Console.WriteLine();
}

// Exercise 5: Generic event system
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Generic Event Bus ===");

    var bus = new EventBus();

    bus.Subscribe<OrderPlaced>(e => Console.WriteLine($"  Handler A: Order #{e.OrderId} placed for ${e.Amount}"));
    bus.Subscribe<OrderPlaced>(e => Console.WriteLine($"  Handler B: Sending confirmation for order #{e.OrderId}"));
    bus.Subscribe<UserLoggedIn>(e => Console.WriteLine($"  Handler: User '{e.Username}' logged in at {e.Timestamp:HH:mm:ss}"));
    bus.Subscribe<ItemAdded>(e => Console.WriteLine($"  Handler: '{e.ItemName}' added to cart (qty: {e.Quantity})"));

    Console.WriteLine("Publishing OrderPlaced:");
    bus.Publish(new OrderPlaced(1001, 59.99m));

    Console.WriteLine("\nPublishing UserLoggedIn:");
    bus.Publish(new UserLoggedIn("alice", DateTime.Now));

    Console.WriteLine("\nPublishing ItemAdded:");
    bus.Publish(new ItemAdded("Widget", 3));
    Console.WriteLine();
}

// Supporting types

class GenericStack<T>
{
    private T[] _items;
    private int _top;

    public int Count => _top;
    public int Capacity => _items.Length;
    public bool IsEmpty => _top == 0;

    public GenericStack(int capacity = 16) { _items = new T[capacity]; _top = 0; }

    public void Push(T item)
    {
        if (_top == _items.Length)
        {
            var bigger = new T[_items.Length * 2];
            Array.Copy(_items, bigger, _items.Length);
            _items = bigger;
        }
        _items[_top++] = item;
    }

    public T Pop()
    {
        if (_top == 0) throw new InvalidOperationException("Stack is empty");
        return _items[--_top];
    }

    public T Peek()
    {
        if (_top == 0) throw new InvalidOperationException("Stack is empty");
        return _items[_top - 1];
    }
}

class SortedCollection<T> where T : IComparable<T>
{
    private readonly List<T> _items = new();

    public void Add(T item)
    {
        int index = _items.BinarySearch(item);
        if (index < 0) index = ~index;
        _items.Insert(index, item);
    }

    public bool Contains(T item) => _items.BinarySearch(item) >= 0;
    public T Min => _items.Count > 0 ? _items[0] : throw new InvalidOperationException("Empty");
    public T Max => _items.Count > 0 ? _items[^1] : throw new InvalidOperationException("Empty");
    public IEnumerator<T> GetEnumerator() => _items.GetEnumerator();
}

record Product(int Id, string Name, decimal Price) : IIdentifiable
{
    public override string ToString() => $"[{Id}] {Name} (${Price})";
}

interface IIdentifiable { int Id { get; } }

class Repository<T> where T : IIdentifiable
{
    private readonly Dictionary<int, T> _store = new();
    public int Count => _store.Count;
    public void Add(T item) => _store[item.Id] = item;
    public T? GetById(int id) => _store.TryGetValue(id, out var item) ? item : default;
    public IEnumerable<T> GetAll() => _store.Values;
    public void Update(T item) => _store[item.Id] = item;
    public bool Remove(int id) => _store.Remove(id);
    public IEnumerable<T> FindWhere(Func<T, bool> predicate) => _store.Values.Where(predicate);
}

record OrderPlaced(int OrderId, decimal Amount);
record UserLoggedIn(string Username, DateTime Timestamp);
record ItemAdded(string ItemName, int Quantity);

class EventBus
{
    private readonly Dictionary<Type, List<Delegate>> _handlers = new();

    public void Subscribe<TEvent>(Action<TEvent> handler)
    {
        var type = typeof(TEvent);
        if (!_handlers.ContainsKey(type))
            _handlers[type] = new List<Delegate>();
        _handlers[type].Add(handler);
    }

    public void Publish<TEvent>(TEvent evt)
    {
        if (_handlers.TryGetValue(typeof(TEvent), out var handlers))
            foreach (var handler in handlers)
                ((Action<TEvent>)handler)(evt);
    }
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
