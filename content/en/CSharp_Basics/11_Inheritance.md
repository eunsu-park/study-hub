# Inheritance

**Previous**: [Structs and Enums](./10_Structs_and_Enums.md) | **Next**: [Interfaces](./12_Interfaces.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create class hierarchies using base and derived classes
2. Chain constructors with the `base` keyword
3. Override methods using `virtual` and `override`, and understand method hiding with `new`
4. Design abstract classes and implement abstract methods
5. Restrict inheritance with `sealed` classes and methods
6. Perform type checking and casting with `is` and `as`
7. Understand upcasting, downcasting, and the `protected` access modifier
8. Override key methods from the `Object` class

---

Inheritance is one of the four pillars of object-oriented programming. It allows you to define a new class based on an existing class, inheriting its members and behavior while adding or modifying functionality. In C#, inheritance enables code reuse, establishes natural hierarchies, and provides the foundation for polymorphism. This lesson covers everything from basic class derivation to advanced concepts like abstract and sealed classes.

## 1. Base and Derived Classes

In C#, a class can inherit from another class using the colon (`:`) syntax. The class being inherited from is called the **base class** (or parent class), and the class that inherits is called the **derived class** (or child class).

### 1.1 Basic Inheritance Syntax

```csharp
// Base class
public class Animal
{
    public string Name { get; set; }
    public int Age { get; set; }

    public void Eat()
    {
        Console.WriteLine($"{Name} is eating.");
    }

    public void Sleep()
    {
        Console.WriteLine($"{Name} is sleeping.");
    }
}

// Derived class
public class Dog : Animal
{
    public string Breed { get; set; }

    public void Bark()
    {
        Console.WriteLine($"{Name} says: Woof!");
    }
}
```

```csharp
Dog dog = new Dog();
dog.Name = "Rex";       // Inherited from Animal
dog.Age = 5;            // Inherited from Animal
dog.Breed = "Labrador"; // Defined in Dog

dog.Eat();   // Inherited method: "Rex is eating."
dog.Sleep(); // Inherited method: "Rex is sleeping."
dog.Bark();  // Dog-specific method: "Rex says: Woof!"
```

### 1.2 Single Inheritance Only

C# supports single inheritance for classes. A class can inherit from only one base class, but it can implement multiple interfaces.

```csharp
// Valid: single inheritance
public class GuideDog : Dog
{
    public string HandlerName { get; set; }

    public void Guide()
    {
        Console.WriteLine($"{Name} is guiding {HandlerName}.");
    }
}

// Invalid: multiple inheritance is NOT allowed in C#
// public class Hybrid : Dog, Cat { }  // Compile error
```

### 1.3 What Is and Is Not Inherited

Derived classes inherit all public and protected members. Private members are not accessible directly (though they still exist in memory). Constructors and finalizers are not inherited, but base constructors can be invoked with the `base` keyword.

```csharp
public class Vehicle
{
    public string Make { get; set; }
    protected int year;          // Accessible in derived classes
    private string vin;          // Not accessible in derived classes

    public Vehicle(string make, int year, string vin)
    {
        Make = make;
        this.year = year;
        this.vin = vin;
    }

    public void DisplayInfo()
    {
        Console.WriteLine($"{Make}, Year: {year}");
    }
}

public class Car : Vehicle
{
    public int Doors { get; set; }

    public Car(string make, int year, string vin, int doors)
        : base(make, year, vin)  // Call base constructor
    {
        Doors = doors;
    }

    public void ShowDetails()
    {
        Console.WriteLine($"{Make}, Year: {year}, Doors: {Doors}");
        // Console.WriteLine(vin); // Error: 'vin' is inaccessible due to protection level
    }
}
```

## 2. Constructor Chaining with `base`

Since constructors are not inherited, derived classes must explicitly call a base class constructor when the base class does not have a parameterless constructor.

### 2.1 Calling Base Constructors

```csharp
public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }

    public Person(string name, int age)
    {
        Name = name;
        Age = age;
        Console.WriteLine("Person constructor called.");
    }
}

public class Student : Person
{
    public string School { get; set; }

    // Must call base constructor since Person has no parameterless constructor
    public Student(string name, int age, string school)
        : base(name, age)
    {
        School = school;
        Console.WriteLine("Student constructor called.");
    }
}
```

```csharp
Student s = new Student("Alice", 20, "MIT");
// Output:
// Person constructor called.
// Student constructor called.

Console.WriteLine($"{s.Name}, {s.Age}, {s.School}");
// Output: Alice, 20, MIT
```

### 2.2 Multiple Constructor Overloads

A derived class can have multiple constructors, each chaining to different base constructors or to other constructors in the same class using `this`.

```csharp
public class Employee : Person
{
    public string Company { get; set; }
    public decimal Salary { get; set; }

    // Constructor chaining to base
    public Employee(string name, int age, string company, decimal salary)
        : base(name, age)
    {
        Company = company;
        Salary = salary;
    }

    // Constructor chaining to another constructor in this class
    public Employee(string name, int age, string company)
        : this(name, age, company, 50000m)
    {
    }

    // Minimal constructor
    public Employee(string name)
        : base(name, 0)
    {
        Company = "Unknown";
        Salary = 0m;
    }
}
```

### 2.3 Constructor Execution Order

Constructors execute from the top of the hierarchy downward: the base class constructor runs before the derived class constructor.

```csharp
public class A
{
    public A() { Console.WriteLine("A constructor"); }
}

public class B : A
{
    public B() { Console.WriteLine("B constructor"); }
}

public class C : B
{
    public C() { Console.WriteLine("C constructor"); }
}

// new C() outputs:
// A constructor
// B constructor
// C constructor
```

## 3. Method Overriding: `virtual`, `override`, `new`

Method overriding allows a derived class to provide a specific implementation for a method defined in its base class. C# uses the `virtual` and `override` keywords for true polymorphic overriding.

### 3.1 Virtual and Override

Mark a base class method as `virtual` to allow derived classes to override it. Use `override` in the derived class to provide the new implementation.

```csharp
public class Shape
{
    public virtual double Area()
    {
        return 0;
    }

    public virtual string Describe()
    {
        return "I am a shape.";
    }
}

public class Circle : Shape
{
    public double Radius { get; set; }

    public Circle(double radius)
    {
        Radius = radius;
    }

    public override double Area()
    {
        return Math.PI * Radius * Radius;
    }

    public override string Describe()
    {
        return $"I am a circle with radius {Radius:F2}.";
    }
}
```

```csharp
Shape shape = new Circle(5);
Console.WriteLine(shape.Area());      // 78.54 (Circle's override is called)
Console.WriteLine(shape.Describe());  // "I am a circle with radius 5.00."
```

### 3.2 Method Hiding with `new`

If you define a method in a derived class with the same name as a base class method without using `override`, you are **hiding** the base method. The compiler will issue a warning unless you use the `new` keyword explicitly.

```csharp
public class BaseLogger
{
    public void Log(string message)
    {
        Console.WriteLine($"[Base] {message}");
    }
}

public class DerivedLogger : BaseLogger
{
    // Hides BaseLogger.Log — NOT polymorphic
    public new void Log(string message)
    {
        Console.WriteLine($"[Derived] {message}");
    }
}
```

```csharp
DerivedLogger derived = new DerivedLogger();
derived.Log("Hello");            // [Derived] Hello

BaseLogger baseRef = derived;
baseRef.Log("Hello");            // [Base] Hello — hiding means base version is called
```

### 3.3 Override vs New: The Critical Difference

```csharp
public class Animal
{
    public virtual void Speak() => Console.WriteLine("...");
}

public class Cat : Animal
{
    public override void Speak() => Console.WriteLine("Meow!");
}

public class SilentCat : Animal
{
    public new void Speak() => Console.WriteLine("(silent meow)");
}

Animal cat1 = new Cat();
Animal cat2 = new SilentCat();

cat1.Speak();  // "Meow!"          — override: derived version called
cat2.Speak();  // "..."            — new: base version called through base reference
```

Use `override` when you want polymorphic behavior. Use `new` only when you intentionally want to break the polymorphic chain (rare in practice).

## 4. Abstract Classes and Abstract Methods

An abstract class cannot be instantiated directly. It serves as a base class that defines a contract — derived classes must implement all abstract members.

### 4.1 Declaring Abstract Classes

```csharp
public abstract class Vehicle
{
    public string Make { get; set; }
    public string Model { get; set; }

    // Abstract method: no implementation, must be overridden
    public abstract void StartEngine();

    // Abstract property
    public abstract int MaxSpeed { get; }

    // Concrete method: can be used as-is or overridden if virtual
    public void DisplayInfo()
    {
        Console.WriteLine($"{Make} {Model}, Max Speed: {MaxSpeed} km/h");
    }
}
```

### 4.2 Implementing Abstract Members

```csharp
public class ElectricCar : Vehicle
{
    public int BatteryCapacity { get; set; }

    public override int MaxSpeed => 200;

    public override void StartEngine()
    {
        Console.WriteLine("Electric motor whirring...");
    }
}

public class GasCar : Vehicle
{
    public double EngineSize { get; set; }

    public override int MaxSpeed => 250;

    public override void StartEngine()
    {
        Console.WriteLine("Vroom vroom!");
    }
}
```

```csharp
// Vehicle v = new Vehicle();  // Error: cannot instantiate abstract class

Vehicle car1 = new ElectricCar { Make = "Tesla", Model = "Model 3", BatteryCapacity = 75 };
Vehicle car2 = new GasCar { Make = "BMW", Model = "M3", EngineSize = 3.0 };

car1.StartEngine();   // "Electric motor whirring..."
car2.StartEngine();   // "Vroom vroom!"
car1.DisplayInfo();   // "Tesla Model 3, Max Speed: 200 km/h"
```

### 4.3 Abstract vs Concrete Members in Abstract Classes

An abstract class can contain a mix of abstract and non-abstract members. This is a key difference from interfaces (pre-C# 8).

```csharp
public abstract class DatabaseConnection
{
    // Abstract members — derived class MUST implement
    public abstract string ConnectionString { get; }
    public abstract void Connect();
    public abstract void Disconnect();

    // Concrete members — shared behavior
    public bool IsConnected { get; protected set; }

    public void ExecuteQuery(string query)
    {
        if (!IsConnected)
        {
            Console.WriteLine("Error: Not connected.");
            return;
        }
        Console.WriteLine($"Executing: {query}");
    }
}

public class SqlServerConnection : DatabaseConnection
{
    public override string ConnectionString => "Server=localhost;Database=mydb;";

    public override void Connect()
    {
        IsConnected = true;
        Console.WriteLine("Connected to SQL Server.");
    }

    public override void Disconnect()
    {
        IsConnected = false;
        Console.WriteLine("Disconnected from SQL Server.");
    }
}
```

## 5. Sealed Classes and Sealed Methods

The `sealed` keyword prevents a class from being inherited or a method from being further overridden.

### 5.1 Sealed Classes

```csharp
public sealed class MathHelper
{
    public static double CircleArea(double radius) => Math.PI * radius * radius;
    public static double RectangleArea(double width, double height) => width * height;
}

// Error: cannot derive from sealed type 'MathHelper'
// public class ExtendedMathHelper : MathHelper { }
```

Sealed classes are useful for utility classes, security-sensitive classes, or when you want to guarantee that behavior cannot be changed by subclassing. The `string` type in .NET is sealed.

### 5.2 Sealed Methods

You can seal individual overridden methods to prevent further overriding down the hierarchy.

```csharp
public class Animal
{
    public virtual void MakeSound()
    {
        Console.WriteLine("Some generic sound");
    }
}

public class Dog : Animal
{
    // Override and seal: no further overriding allowed
    public sealed override void MakeSound()
    {
        Console.WriteLine("Bark!");
    }
}

public class GoldenRetriever : Dog
{
    // Error: cannot override sealed method 'Dog.MakeSound()'
    // public override void MakeSound() { }
}
```

### 5.3 When to Use Sealed

Sealing is appropriate when:
- A class is not designed for extension (utility or helper classes)
- Security or correctness depends on specific behavior not being changed
- Performance: the JIT compiler can sometimes optimize calls to sealed methods

```csharp
public abstract class PaymentProcessor
{
    public abstract decimal CalculateFee(decimal amount);

    // Template method pattern: sealed to prevent tampering
    public sealed override string ToString()
    {
        return $"{GetType().Name} processor";
    }
}
```

## 6. The `is` and `as` Operators

C# provides the `is` and `as` operators for safe type checking and casting at runtime.

### 6.1 The `is` Operator

The `is` operator checks whether an object is compatible with a given type and returns `true` or `false`.

```csharp
public class Animal { }
public class Dog : Animal { }
public class Cat : Animal { }

Animal animal = new Dog();

if (animal is Dog)
{
    Console.WriteLine("It's a dog!");     // This prints
}

if (animal is Cat)
{
    Console.WriteLine("It's a cat!");     // This does NOT print
}

if (animal is Animal)
{
    Console.WriteLine("It's an animal!"); // This prints (Dog IS an Animal)
}
```

### 6.2 Pattern Matching with `is` (C# 7+)

The `is` operator can also declare a variable, combining type check and cast in one step.

```csharp
Animal animal = new Dog { Name = "Rex" };

if (animal is Dog dog)
{
    // 'dog' is now a Dog variable, usable in this scope
    Console.WriteLine($"Dog name: {dog.Name}");
}

// Works in switch statements too
void DescribeAnimal(Animal a)
{
    switch (a)
    {
        case Dog d:
            Console.WriteLine($"Dog: {d.Name}");
            break;
        case Cat c:
            Console.WriteLine($"Cat: {c.Name}");
            break;
        default:
            Console.WriteLine("Unknown animal");
            break;
    }
}
```

### 6.3 The `as` Operator

The `as` operator attempts a cast and returns `null` if the cast fails, instead of throwing an exception.

```csharp
Animal animal = new Dog();

Dog dog = animal as Dog;
if (dog != null)
{
    Console.WriteLine("Successfully cast to Dog.");
}

Cat cat = animal as Cat;  // Returns null, no exception
if (cat == null)
{
    Console.WriteLine("Cannot cast to Cat.");
}
```

### 6.4 `as` vs Direct Cast

```csharp
Animal animal = new Dog();

// Direct cast: throws InvalidCastException if it fails
try
{
    Cat cat = (Cat)animal;  // Throws!
}
catch (InvalidCastException ex)
{
    Console.WriteLine($"Cast failed: {ex.Message}");
}

// 'as' operator: returns null if it fails (no exception)
Cat safeCat = animal as Cat;  // null, no exception

// 'is' with pattern matching: cleanest approach
if (animal is Cat patternCat)
{
    // Use patternCat safely
}
```

## 7. Upcasting and Downcasting

### 7.1 Upcasting (Implicit)

Upcasting is converting a derived type to a base type. This is always safe and happens implicitly.

```csharp
Dog dog = new Dog { Name = "Buddy" };
Animal animal = dog;  // Implicit upcast: Dog -> Animal

// The object is still a Dog, but the reference type is Animal
Console.WriteLine(animal.Name);   // Works: Name is defined in Animal
// animal.Bark();                  // Error: Bark is not defined in Animal
```

### 7.2 Downcasting (Explicit)

Downcasting converts a base type reference back to a derived type. This requires an explicit cast and can fail at runtime.

```csharp
Animal animal = new Dog { Name = "Buddy" };

// Safe downcast with 'is' check
if (animal is Dog dog)
{
    dog.Bark();  // Now we can call Dog-specific methods
}

// Alternative with explicit cast (risky without checking)
Dog dog2 = (Dog)animal;  // Works because the object is actually a Dog
dog2.Bark();
```

### 7.3 Polymorphism in Action

Upcasting is the foundation of polymorphism. You can store different derived types in a base type collection and call overridden methods.

```csharp
List<Animal> animals = new List<Animal>
{
    new Dog { Name = "Rex" },
    new Cat { Name = "Whiskers" },
    new Dog { Name = "Buddy" }
};

foreach (Animal a in animals)
{
    a.Speak();  // Calls the correct override for each actual type

    if (a is Dog d)
    {
        d.Bark();  // Dog-specific behavior
    }
}
```

## 8. The `protected` Access Modifier

The `protected` modifier makes a member accessible within the class itself and by any derived class, but not by external code.

### 8.1 Protected Members

```csharp
public class BankAccount
{
    public string Owner { get; set; }
    protected decimal balance;  // Accessible in derived classes

    public BankAccount(string owner, decimal initialBalance)
    {
        Owner = owner;
        balance = initialBalance;
    }

    public decimal GetBalance() => balance;
}

public class SavingsAccount : BankAccount
{
    private decimal interestRate;

    public SavingsAccount(string owner, decimal balance, decimal rate)
        : base(owner, balance)
    {
        interestRate = rate;
    }

    public void ApplyInterest()
    {
        // Can access 'balance' because it is protected
        balance += balance * interestRate;
        Console.WriteLine($"New balance after interest: {balance:C}");
    }
}
```

```csharp
SavingsAccount sa = new SavingsAccount("Alice", 1000m, 0.05m);
sa.ApplyInterest();  // "New balance after interest: $1,050.00"

// sa.balance;  // Error: 'balance' is inaccessible due to protection level
Console.WriteLine(sa.GetBalance());  // Use the public method instead
```

### 8.2 Protected Internal and Private Protected

C# offers additional access modifier combinations:

```csharp
public class MyBase
{
    // protected internal: accessible by derived classes OR any code in the same assembly
    protected internal int value1;

    // private protected: accessible ONLY by derived classes in the same assembly
    private protected int value2;
}
```

| Modifier | Same Class | Derived (Same Assembly) | Derived (Other Assembly) | Same Assembly | External |
|---|---|---|---|---|---|
| `public` | Yes | Yes | Yes | Yes | Yes |
| `protected` | Yes | Yes | Yes | No | No |
| `internal` | Yes | Yes | No | Yes | No |
| `protected internal` | Yes | Yes | Yes | Yes | No |
| `private protected` | Yes | Yes | No | No | No |
| `private` | Yes | No | No | No | No |

## 9. The `Object` Class

Every class in C# implicitly inherits from `System.Object`. This means every object has access to a set of common methods that you can override to customize behavior.

### 9.1 Overriding `ToString()`

```csharp
public class Product
{
    public string Name { get; set; }
    public decimal Price { get; set; }

    public override string ToString()
    {
        return $"{Name} (${Price:F2})";
    }
}

Product p = new Product { Name = "Laptop", Price = 999.99m };
Console.WriteLine(p);              // "Laptop ($999.99)"
Console.WriteLine(p.ToString());   // Same result
```

### 9.2 Overriding `Equals()` and `GetHashCode()`

When you override `Equals`, you should also override `GetHashCode` to maintain consistency (equal objects must produce the same hash code).

```csharp
public class Point
{
    public int X { get; set; }
    public int Y { get; set; }

    public override bool Equals(object obj)
    {
        if (obj is Point other)
        {
            return X == other.X && Y == other.Y;
        }
        return false;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(X, Y);
    }

    public override string ToString()
    {
        return $"({X}, {Y})";
    }
}
```

```csharp
Point a = new Point { X = 3, Y = 4 };
Point b = new Point { X = 3, Y = 4 };
Point c = new Point { X = 1, Y = 2 };

Console.WriteLine(a.Equals(b));    // True (same coordinates)
Console.WriteLine(a.Equals(c));    // False

// Without overriding Equals, reference equality would be used:
// a.Equals(b) would be False because they are different objects

// HashCode consistency
Console.WriteLine(a.GetHashCode() == b.GetHashCode());  // True
```

### 9.3 The Complete Override Pattern

```csharp
public class Student
{
    public string Id { get; set; }
    public string Name { get; set; }
    public double Gpa { get; set; }

    public override string ToString()
    {
        return $"Student {Id}: {Name} (GPA: {Gpa:F2})";
    }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(this, obj)) return true;
        if (obj is null) return false;
        if (GetType() != obj.GetType()) return false;

        Student other = (Student)obj;
        return Id == other.Id;
    }

    public override int GetHashCode()
    {
        return Id?.GetHashCode() ?? 0;
    }

    // Optional: operator overloads for equality
    public static bool operator ==(Student left, Student right)
    {
        if (left is null) return right is null;
        return left.Equals(right);
    }

    public static bool operator !=(Student left, Student right)
    {
        return !(left == right);
    }
}
```

## 10. Practical Example: Shape Hierarchy

Let us bring everything together with a comprehensive shape hierarchy that demonstrates inheritance, abstract classes, method overriding, and polymorphism.

### 10.1 The Abstract Base Class

```csharp
public abstract class Shape
{
    public string Color { get; set; }
    public string Name { get; protected set; }

    protected Shape(string color)
    {
        Color = color;
    }

    // Abstract members: every shape must define these
    public abstract double Area();
    public abstract double Perimeter();

    // Virtual method: can be overridden but has a default implementation
    public virtual void Draw()
    {
        Console.WriteLine($"Drawing a {Color} {Name}");
    }

    public override string ToString()
    {
        return $"{Name} [Color={Color}, Area={Area():F2}, Perimeter={Perimeter():F2}]";
    }
}
```

### 10.2 Concrete Shape Classes

```csharp
public class Circle : Shape
{
    public double Radius { get; set; }

    public Circle(double radius, string color = "red") : base(color)
    {
        Radius = radius;
        Name = "Circle";
    }

    public override double Area() => Math.PI * Radius * Radius;
    public override double Perimeter() => 2 * Math.PI * Radius;
}

public class Rectangle : Shape
{
    public double Width { get; set; }
    public double Height { get; set; }

    public Rectangle(double width, double height, string color = "blue") : base(color)
    {
        Width = width;
        Height = height;
        Name = "Rectangle";
    }

    public override double Area() => Width * Height;
    public override double Perimeter() => 2 * (Width + Height);
}

public class Triangle : Shape
{
    public double SideA { get; set; }
    public double SideB { get; set; }
    public double SideC { get; set; }

    public Triangle(double a, double b, double c, string color = "green") : base(color)
    {
        SideA = a;
        SideB = b;
        SideC = c;
        Name = "Triangle";
    }

    public override double Area()
    {
        double s = Perimeter() / 2; // Semi-perimeter
        return Math.Sqrt(s * (s - SideA) * (s - SideB) * (s - SideC));
    }

    public override double Perimeter() => SideA + SideB + SideC;
}
```

### 10.3 Extending the Hierarchy

```csharp
public class Square : Rectangle
{
    public Square(double side, string color = "purple")
        : base(side, side, color)
    {
        Name = "Square";
    }

    public override void Draw()
    {
        base.Draw();
        Console.WriteLine($"  Side length: {Width}");
    }
}

public sealed class EquilateralTriangle : Triangle
{
    public EquilateralTriangle(double side, string color = "orange")
        : base(side, side, side, color)
    {
        Name = "Equilateral Triangle";
    }
}
```

### 10.4 Using the Hierarchy with Polymorphism

```csharp
class Program
{
    static void Main()
    {
        List<Shape> shapes = new List<Shape>
        {
            new Circle(5, "red"),
            new Rectangle(4, 6, "blue"),
            new Triangle(3, 4, 5, "green"),
            new Square(7, "purple"),
            new EquilateralTriangle(10, "orange")
        };

        Console.WriteLine("=== Shape Report ===");
        double totalArea = 0;

        foreach (Shape shape in shapes)
        {
            Console.WriteLine(shape);   // Calls overridden ToString()
            shape.Draw();               // Calls overridden or default Draw()
            totalArea += shape.Area();  // Polymorphic call
            Console.WriteLine();
        }

        Console.WriteLine($"Total area of all shapes: {totalArea:F2}");

        // Type checking and downcasting
        Console.WriteLine("\n=== Circles Only ===");
        foreach (Shape shape in shapes)
        {
            if (shape is Circle circle)
            {
                Console.WriteLine($"Circle radius: {circle.Radius}");
            }
        }

        // Find the largest shape
        Shape largest = shapes.OrderByDescending(s => s.Area()).First();
        Console.WriteLine($"\nLargest shape: {largest}");
    }
}
```

Output:
```
=== Shape Report ===
Circle [Color=red, Area=78.54, Perimeter=31.42]
Drawing a red Circle

Rectangle [Color=blue, Area=24.00, Perimeter=20.00]
Drawing a blue Rectangle

Triangle [Color=green, Area=6.00, Perimeter=12.00]
Drawing a green Triangle

Square [Color=purple, Area=49.00, Perimeter=28.00]
Drawing a purple Square
  Side length: 7

Equilateral Triangle [Color=orange, Area=43.30, Perimeter=30.00]
Drawing a orange Equilateral Triangle

Total area of all shapes: 200.84

=== Circles Only ===
Circle radius: 5

Largest shape: Circle [Color=red, Area=78.54, Perimeter=31.42]
```

## 11. Practice Problems

1. **Animal Hierarchy**: Create an abstract `Animal` class with abstract `Speak()` and `Move()` methods, a `Name` property, and a concrete `Describe()` method. Derive `Bird`, `Fish`, and `Mammal` classes with appropriate implementations. Then create a `Penguin` class that inherits from `Bird` but overrides `Move()` to say "waddle" instead of "fly". Store several animals in a `List<Animal>` and iterate through them polymorphically.

2. **Employee Payroll**: Create a base class `Employee` with properties `Name` and `Id`, and an abstract method `CalculatePay()`. Derive `SalariedEmployee` (fixed monthly salary), `HourlyEmployee` (hours worked times hourly rate), and `CommissionEmployee` (base salary plus commission percentage of sales). Override `ToString()` and `Equals()` (equality based on `Id`). Write a method that takes a list of employees and prints a payroll report with total pay.

3. **Override All Three**: Create a `Book` class with `Isbn`, `Title`, and `Author` properties. Override `ToString()`, `Equals()` (based on ISBN), and `GetHashCode()`. Demonstrate that two `Book` objects with the same ISBN are considered equal, can be used as dictionary keys correctly, and display meaningful string representations.

4. **Sealed Prevention**: Create a class hierarchy: `Account` (abstract) -> `CheckingAccount` -> `PremiumCheckingAccount`. Seal the `Withdraw()` method in `CheckingAccount` so that `PremiumCheckingAccount` cannot change the withdrawal logic but can add its own methods like `EarnRewards()`. Demonstrate that attempting to override the sealed method causes a compile error (comment it out and explain).

5. **Type Checking Zoo**: Create a hierarchy of zoo animals with at least 5 different concrete types across 2-3 levels of inheritance. Write a method `ZooReport(List<Animal> animals)` that uses `is` pattern matching and `switch` expressions to generate different reports based on the actual type of each animal (e.g., mammals get a "habitat" report, birds get a "wingspan" report, reptiles get a "temperature" report).
