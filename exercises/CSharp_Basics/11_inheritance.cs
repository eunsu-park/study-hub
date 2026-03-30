/*
 * Exercises for Lesson 11: Inheritance
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Shape hierarchy with polymorphism
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Shape Hierarchy ===");

    Shape[] shapes =
    {
        new Circle(5),
        new RectangleShape(4, 6),
        new Triangle(3, 4, 5),
        new Circle(10),
        new RectangleShape(8, 8)
    };

    foreach (Shape shape in shapes)
    {
        Console.WriteLine($"  {shape.Name}: Area={shape.Area():F2}, Perimeter={shape.Perimeter():F2}");
    }

    double totalArea = shapes.Sum(s => s.Area());
    Console.WriteLine($"\nTotal area: {totalArea:F2}");

    // Sort by area descending
    var sorted = shapes.OrderByDescending(s => s.Area()).ToArray();
    Console.WriteLine("Sorted by area (desc):");
    foreach (var s in sorted)
        Console.WriteLine($"  {s.Name}: {s.Area():F2}");
    Console.WriteLine();
}

// Exercise 2: Method hiding vs overriding
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Hiding vs Overriding ===");

    Base obj1 = new Derived();
    Derived obj2 = new Derived();

    Console.WriteLine("Base reference to Derived object:");
    Console.WriteLine($"  VirtualMethod:  {obj1.VirtualMethod()}   (overridden — calls Derived)");
    Console.WriteLine($"  HiddenMethod:   {obj1.HiddenMethod()}   (hidden — calls Base)");

    Console.WriteLine("Derived reference to Derived object:");
    Console.WriteLine($"  VirtualMethod:  {obj2.VirtualMethod()}   (overridden — calls Derived)");
    Console.WriteLine($"  HiddenMethod:   {obj2.HiddenMethod()}   (hidden — calls Derived)");

    // Sealed override
    Base obj3 = new MoreDerived();
    Console.WriteLine($"\nMoreDerived.VirtualMethod: {obj3.VirtualMethod()} (sealed in Derived, cannot override)");
    Console.WriteLine();
}

// Exercise 3: Abstract class — employee payroll system
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Employee Payroll ===");

    Employee[] employees =
    {
        new SalariedEmployee("Alice", 75000),
        new HourlyEmployee("Bob", 35.00m, 160),
        new CommissionEmployee("Charlie", 40000, 25000, 0.10m),
        new SalariedEmployee("Diana", 90000),
        new HourlyEmployee("Eve", 28.50m, 180)
    };

    Console.WriteLine($"{"Name",-12} {"Type",-20} {"Monthly Pay",14}");
    Console.WriteLine(new string('-', 48));

    decimal totalPayroll = 0;
    foreach (var emp in employees)
    {
        decimal pay = emp.CalculateMonthlyPay();
        totalPayroll += pay;
        Console.WriteLine($"{emp.Name,-12} {emp.GetType().Name,-20} {pay,14:C}");
    }
    Console.WriteLine(new string('-', 48));
    Console.WriteLine($"{"Total",-32} {totalPayroll,14:C}");
    Console.WriteLine();
}

// Exercise 4: Constructor chaining in inheritance
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Constructor Chaining ===");

    Console.WriteLine("Creating a Manager:");
    var manager = new Manager("Alice", 30, "Engineering", 10);
    Console.WriteLine($"  {manager}\n");

    Console.WriteLine("Creating an Executive:");
    var exec = new Executive("Bob", 45, "Company", 50, 100000m);
    Console.WriteLine($"  {exec}\n");

    // is and as operators
    EmployeeBase[] team = { manager, exec, new EmployeeBase("Charlie", 25) };
    foreach (var member in team)
    {
        Console.Write($"  {member.Name} is Manager? {member is Manager}");
        if (member is Manager m)
            Console.Write($" (reports: {m.DirectReports})");
        Console.WriteLine();
    }
    Console.WriteLine();
}

// Exercise 5: Protected members and template method pattern
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Template Method Pattern ===");

    DataProcessor[] processors =
    {
        new CsvProcessor(),
        new JsonProcessor(),
        new XmlProcessor()
    };

    string[] sampleData = { "Alice,30", "Bob,25", "Charlie,35" };

    foreach (var processor in processors)
    {
        Console.WriteLine($"--- {processor.GetType().Name} ---");
        processor.Process(sampleData);
        Console.WriteLine();
    }
}

// Supporting types

abstract class Shape
{
    public abstract string Name { get; }
    public abstract double Area();
    public abstract double Perimeter();
}

class Circle : Shape
{
    public double Radius { get; }
    public Circle(double radius) => Radius = radius;
    public override string Name => $"Circle(r={Radius})";
    public override double Area() => Math.PI * Radius * Radius;
    public override double Perimeter() => 2 * Math.PI * Radius;
}

class RectangleShape : Shape
{
    public double Width { get; }
    public double Height { get; }
    public RectangleShape(double w, double h) { Width = w; Height = h; }
    public override string Name => $"Rect({Width}x{Height})";
    public override double Area() => Width * Height;
    public override double Perimeter() => 2 * (Width + Height);
}

class Triangle : Shape
{
    public double A { get; }
    public double B { get; }
    public double C { get; }
    public Triangle(double a, double b, double c) { A = a; B = b; C = c; }
    public override string Name => $"Triangle({A},{B},{C})";
    public override double Area()
    {
        double s = (A + B + C) / 2;
        return Math.Sqrt(s * (s - A) * (s - B) * (s - C));
    }
    public override double Perimeter() => A + B + C;
}

class Base
{
    public virtual string VirtualMethod() => "Base";
    public string HiddenMethod() => "Base";
}

class Derived : Base
{
    public sealed override string VirtualMethod() => "Derived";
    public new string HiddenMethod() => "Derived";
}

class MoreDerived : Derived
{
    // Cannot override VirtualMethod — it is sealed in Derived
    // public override string VirtualMethod() => "MoreDerived"; // Compile error
}

abstract class Employee
{
    public string Name { get; }
    protected Employee(string name) => Name = name;
    public abstract decimal CalculateMonthlyPay();
}

class SalariedEmployee : Employee
{
    public decimal AnnualSalary { get; }
    public SalariedEmployee(string name, decimal annual) : base(name) => AnnualSalary = annual;
    public override decimal CalculateMonthlyPay() => AnnualSalary / 12;
}

class HourlyEmployee : Employee
{
    public decimal HourlyRate { get; }
    public int HoursPerMonth { get; }
    public HourlyEmployee(string name, decimal rate, int hours) : base(name)
    {
        HourlyRate = rate; HoursPerMonth = hours;
    }
    public override decimal CalculateMonthlyPay()
    {
        int regular = Math.Min(HoursPerMonth, 160);
        int overtime = Math.Max(0, HoursPerMonth - 160);
        return regular * HourlyRate + overtime * HourlyRate * 1.5m;
    }
}

class CommissionEmployee : Employee
{
    public decimal BaseSalary { get; }
    public decimal Sales { get; }
    public decimal CommissionRate { get; }
    public CommissionEmployee(string name, decimal baseSalary, decimal sales, decimal rate) : base(name)
    {
        BaseSalary = baseSalary; Sales = sales; CommissionRate = rate;
    }
    public override decimal CalculateMonthlyPay() => BaseSalary / 12 + Sales * CommissionRate;
}

class EmployeeBase
{
    public string Name { get; }
    public int Age { get; }
    public EmployeeBase(string name, int age) { Name = name; Age = age; }
    public override string ToString() => $"{Name} (age {Age})";
}

class Manager : EmployeeBase
{
    public string Department { get; }
    public int DirectReports { get; }
    public Manager(string name, int age, string dept, int reports) : base(name, age)
    {
        Department = dept; DirectReports = reports;
    }
    public override string ToString() => $"Manager {base.ToString()}, dept={Department}, reports={DirectReports}";
}

class Executive : Manager
{
    public decimal Bonus { get; }
    public Executive(string name, int age, string dept, int reports, decimal bonus) : base(name, age, dept, reports)
    {
        Bonus = bonus;
    }
    public override string ToString() => $"Executive {Name} (age {Age}), dept={Department}, bonus={Bonus:C}";
}

abstract class DataProcessor
{
    // Template method
    public void Process(string[] rawData)
    {
        var validated = Validate(rawData);
        var transformed = Transform(validated);
        Output(transformed);
    }
    protected virtual string[] Validate(string[] data) { Console.WriteLine("  Validating..."); return data; }
    protected abstract string[] Transform(string[] data);
    protected abstract void Output(string[] data);
}

class CsvProcessor : DataProcessor
{
    protected override string[] Transform(string[] data)
    {
        Console.WriteLine("  Parsing CSV...");
        return data.Select(line => $"CSV: [{string.Join("] [", line.Split(','))}]").ToArray();
    }
    protected override void Output(string[] data) { foreach (var d in data) Console.WriteLine($"  -> {d}"); }
}

class JsonProcessor : DataProcessor
{
    protected override string[] Transform(string[] data)
    {
        Console.WriteLine("  Converting to JSON...");
        return data.Select(line =>
        {
            var parts = line.Split(',');
            return $"{{\"name\":\"{parts[0]}\",\"age\":{(parts.Length > 1 ? parts[1] : "0")}}}";
        }).ToArray();
    }
    protected override void Output(string[] data) { foreach (var d in data) Console.WriteLine($"  -> {d}"); }
}

class XmlProcessor : DataProcessor
{
    protected override string[] Transform(string[] data)
    {
        Console.WriteLine("  Converting to XML...");
        return data.Select(line =>
        {
            var parts = line.Split(',');
            return $"<person name=\"{parts[0]}\" age=\"{(parts.Length > 1 ? parts[1] : "0")}\" />";
        }).ToArray();
    }
    protected override void Output(string[] data) { foreach (var d in data) Console.WriteLine($"  -> {d}"); }
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
