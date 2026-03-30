/*
 * Exercises for Lesson 09: Classes and Objects
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Design a BankAccount class with encapsulation
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: BankAccount Class ===");

    var account = new BankAccount("Alice", 1000.00m);
    Console.WriteLine(account);

    account.Deposit(500.00m);
    Console.WriteLine($"After deposit $500: Balance = ${account.Balance}");

    bool success = account.Withdraw(200.00m);
    Console.WriteLine($"Withdraw $200: success={success}, Balance = ${account.Balance}");

    success = account.Withdraw(5000.00m);
    Console.WriteLine($"Withdraw $5000: success={success}, Balance = ${account.Balance}");

    // Transaction history
    Console.WriteLine("\nTransaction history:");
    foreach (string tx in account.GetHistory())
        Console.WriteLine($"  {tx}");
    Console.WriteLine();
}

// Exercise 2: Constructor chaining and static members
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Constructors and Static Members ===");

    var p1 = new Person("Alice", 30);
    var p2 = new Person("Bob");
    var p3 = new Person("Charlie", 25, "charlie@example.com");

    Console.WriteLine(p1);
    Console.WriteLine(p2);
    Console.WriteLine(p3);
    Console.WriteLine($"Total persons created: {Person.Count}");

    // Static factory method
    var p4 = Person.CreateAnonymous();
    Console.WriteLine($"Anonymous: {p4}");
    Console.WriteLine($"Total persons created: {Person.Count}");
    Console.WriteLine();
}

// Exercise 3: Equality and comparison
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Equality ===");

    var m1 = new Money(10.00m, "USD");
    var m2 = new Money(10.00m, "USD");
    var m3 = new Money(20.00m, "USD");
    var m4 = new Money(10.00m, "EUR");

    Console.WriteLine($"m1 = {m1}");
    Console.WriteLine($"m2 = {m2}");
    Console.WriteLine($"m3 = {m3}");
    Console.WriteLine($"m4 = {m4}");

    Console.WriteLine($"\nm1 == m2: {m1 == m2}");
    Console.WriteLine($"m1 == m3: {m1 == m3}");
    Console.WriteLine($"m1 == m4: {m1 == m4}");
    Console.WriteLine($"m1.Equals(m2): {m1.Equals(m2)}");

    // Arithmetic
    var sum = m1 + m3;
    Console.WriteLine($"\n{m1} + {m3} = {sum}");

    var sorted = new List<Money> { m3, m1, new(15.00m, "USD"), m2 };
    sorted.Sort();
    Console.WriteLine($"Sorted: [{string.Join(", ", sorted)}]");
    Console.WriteLine();
}

// Exercise 4: Composition — build a Library system
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Composition — Library ===");

    var library = new Library("City Library");
    library.AddBook(new Book("The Pragmatic Programmer", "Hunt & Thomas", 352));
    library.AddBook(new Book("Clean Code", "Robert C. Martin", 464));
    library.AddBook(new Book("Design Patterns", "Gang of Four", 395));
    library.AddBook(new Book("Refactoring", "Martin Fowler", 448));
    library.AddBook(new Book("Code Complete", "Steve McConnell", 960));

    Console.WriteLine(library);
    Console.WriteLine($"\nBooks with 'Code' in title:");
    foreach (var book in library.Search("Code"))
        Console.WriteLine($"  {book}");

    Console.WriteLine($"\nLongest book: {library.LongestBook()}");
    Console.WriteLine($"Average pages: {library.AveragePages():F0}");
    Console.WriteLine();
}

// Exercise 5: Copy semantics — shallow vs deep copy
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Copy Semantics ===");

    var original = new Department("Engineering");
    original.AddEmployee("Alice");
    original.AddEmployee("Bob");
    Console.WriteLine($"Original: {original}");

    // Shallow copy
    var shallow = original.ShallowCopy();
    shallow.AddEmployee("Charlie");
    Console.WriteLine($"\nAfter shallow copy + add Charlie:");
    Console.WriteLine($"Original: {original}  (affected!)");
    Console.WriteLine($"Shallow:  {shallow}");

    // Deep copy
    var original2 = new Department("Marketing");
    original2.AddEmployee("Diana");
    original2.AddEmployee("Eve");

    var deep = original2.DeepCopy();
    deep.AddEmployee("Frank");
    Console.WriteLine($"\nAfter deep copy + add Frank:");
    Console.WriteLine($"Original: {original2}  (unaffected)");
    Console.WriteLine($"Deep:     {deep}");
    Console.WriteLine();
}

// Supporting classes

class BankAccount
{
    private readonly List<string> _history = new();
    public string Owner { get; }
    public decimal Balance { get; private set; }

    public BankAccount(string owner, decimal initialBalance)
    {
        Owner = owner;
        Balance = initialBalance;
        _history.Add($"Account opened with ${initialBalance}");
    }

    public void Deposit(decimal amount)
    {
        Balance += amount;
        _history.Add($"Deposited ${amount}, balance=${Balance}");
    }

    public bool Withdraw(decimal amount)
    {
        if (amount > Balance)
        {
            _history.Add($"Failed withdrawal of ${amount} (insufficient funds)");
            return false;
        }
        Balance -= amount;
        _history.Add($"Withdrew ${amount}, balance=${Balance}");
        return true;
    }

    public IReadOnlyList<string> GetHistory() => _history.AsReadOnly();
    public override string ToString() => $"BankAccount({Owner}, ${Balance})";
}

class Person
{
    public static int Count { get; private set; }

    public string Name { get; }
    public int Age { get; }
    public string Email { get; }

    public Person(string name, int age, string email) { Name = name; Age = age; Email = email; Count++; }
    public Person(string name, int age) : this(name, age, "N/A") { }
    public Person(string name) : this(name, 0) { }
    public static Person CreateAnonymous() => new("Anonymous");
    public override string ToString() => $"Person({Name}, age={Age}, email={Email})";
}

class Money : IComparable<Money>, IEquatable<Money>
{
    public decimal Amount { get; }
    public string Currency { get; }

    public Money(decimal amount, string currency) { Amount = amount; Currency = currency; }

    public int CompareTo(Money? other) => other is null ? 1 : Amount.CompareTo(other.Amount);
    public bool Equals(Money? other) => other is not null && Amount == other.Amount && Currency == other.Currency;
    public override bool Equals(object? obj) => Equals(obj as Money);
    public override int GetHashCode() => HashCode.Combine(Amount, Currency);
    public static bool operator ==(Money? a, Money? b) => a?.Equals(b) ?? b is null;
    public static bool operator !=(Money? a, Money? b) => !(a == b);
    public static Money operator +(Money a, Money b)
    {
        if (a.Currency != b.Currency) throw new InvalidOperationException("Currency mismatch");
        return new Money(a.Amount + b.Amount, a.Currency);
    }
    public override string ToString() => $"${Amount} {Currency}";
}

class Book
{
    public string Title { get; }
    public string Author { get; }
    public int Pages { get; }
    public Book(string title, string author, int pages) { Title = title; Author = author; Pages = pages; }
    public override string ToString() => $"\"{Title}\" by {Author} ({Pages}p)";
}

class Library
{
    private readonly List<Book> _books = new();
    public string Name { get; }

    public Library(string name) => Name = name;
    public void AddBook(Book book) => _books.Add(book);
    public IEnumerable<Book> Search(string keyword) =>
        _books.Where(b => b.Title.Contains(keyword, StringComparison.OrdinalIgnoreCase));
    public Book? LongestBook() => _books.MaxBy(b => b.Pages);
    public double AveragePages() => _books.Average(b => b.Pages);
    public override string ToString() => $"Library \"{Name}\" ({_books.Count} books)";
}

class Department
{
    public string Name { get; }
    private List<string> _employees;

    public Department(string name) { Name = name; _employees = new List<string>(); }
    private Department(string name, List<string> employees) { Name = name; _employees = employees; }
    public void AddEmployee(string name) => _employees.Add(name);
    public Department ShallowCopy() => new(Name, _employees);
    public Department DeepCopy() => new(Name, new List<string>(_employees));
    public override string ToString() => $"{Name}:[{string.Join(", ", _employees)}]";
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
