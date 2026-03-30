// Lesson 11: Inheritance and Polymorphism
// Run: dotnet run

using System;
using System.Collections.Generic;

// =============================================================================
// BASIC INHERITANCE — Shape Hierarchy
// =============================================================================
Console.WriteLine("=== Shape Hierarchy (Abstract Base) ===");

// Cannot instantiate abstract class:
// var s = new Shape("test"); // Compile error

Shape[] shapes =
{
    new Circle("Red Circle", 5.0),
    new Rectangle("Blue Rect", 4.0, 6.0),
    new Triangle("Green Tri", 3.0, 4.0, 5.0),
    new Square("Yellow Sq", 7.0)
};

// Polymorphism: each shape calls its own overridden methods
foreach (Shape shape in shapes)
{
    Console.WriteLine($"  {shape.Name}:");
    Console.WriteLine($"    Area:      {shape.Area():F2}");
    Console.WriteLine($"    Perimeter: {shape.Perimeter():F2}");
    Console.WriteLine($"    Describe:  {shape.Describe()}");
}

// =============================================================================
// VIRTUAL / OVERRIDE
// =============================================================================
Console.WriteLine("\n=== Virtual / Override ===");

var baseAnimal = new Animal("Generic Animal");
var dog = new Dog("Rex");
var cat = new Cat("Whiskers");
var kitten = new Kitten("Tiny");

Animal[] animals = { baseAnimal, dog, cat, kitten };
foreach (Animal a in animals)
{
    Console.Write($"  {a.Name}: ");
    a.Speak(); // Polymorphic call
}

// =============================================================================
// BASE KEYWORD — calling parent methods
// =============================================================================
Console.WriteLine("\n=== Base Keyword ===");

var employee = new Manager("Alice", 80000, "Engineering");
Console.WriteLine(employee);
// Manager.ToString() calls base.ToString() and adds department info

// =============================================================================
// SEALED — prevent further inheritance
// =============================================================================
Console.WriteLine("\n=== Sealed Class ===");

var constant = new MathConstant("Pi", Math.PI);
Console.WriteLine($"  {constant.Name} = {constant.Value}");
// class MyConstant : MathConstant { } // Compile error: cannot inherit from sealed

// Sealed override — prevent overriding a specific method
var golden = new GoldenRetriever("Buddy");
golden.Speak(); // Uses sealed Dog.Speak, cannot be overridden further

// =============================================================================
// IS AND AS OPERATORS (Type Checking and Casting)
// =============================================================================
Console.WriteLine("\n=== is / as / Pattern Matching ===");

object[] items = { 42, "hello", 3.14, new Dog("Fido"), new Cat("Luna"), null! };

foreach (object item in items)
{
    // 'is' operator with pattern variable
    if (item is int n)
    {
        Console.WriteLine($"  int: {n}");
    }
    else if (item is string s)
    {
        Console.WriteLine($"  string: \"{s}\"");
    }
    else if (item is Dog d)
    {
        Console.Write($"  Dog({d.Name}): ");
        d.Speak();
    }
    else if (item is Animal a)
    {
        Console.Write($"  Animal({a.Name}): ");
        a.Speak();
    }
    else if (item is null)
    {
        Console.WriteLine("  null");
    }
    else
    {
        Console.WriteLine($"  Other: {item} ({item.GetType().Name})");
    }
}

// 'as' operator — returns null if cast fails (reference types only)
Console.WriteLine("\nUsing 'as':");
object obj = new Dog("Max");
Animal? animal = obj as Animal;  // Succeeds (Dog is Animal)
Cat? maybeCat = obj as Cat;      // Fails, returns null

Console.WriteLine($"  obj as Animal: {(animal != null ? animal.Name : "null")}");
Console.WriteLine($"  obj as Cat: {(maybeCat != null ? maybeCat.Name : "null")}");

// =============================================================================
// PROTECTED ACCESS
// =============================================================================
Console.WriteLine("\n=== Protected Members ===");

var electric = new ElectricVehicle("Tesla Model 3", 75.0);
electric.Drive(30);
electric.Drive(50);
Console.WriteLine(electric);

// electric._fuelLevel — not accessible (protected)

// =============================================================================
// METHOD HIDING WITH 'NEW'
// =============================================================================
Console.WriteLine("\n=== Method Hiding (new) vs Override ===");

BaseClass baseRef = new DerivedClass();
DerivedClass derivedRef = new DerivedClass();

// Override: always calls derived version (polymorphic)
Console.Write("  baseRef.VirtualMethod():    ");
baseRef.VirtualMethod();
Console.Write("  derivedRef.VirtualMethod(): ");
derivedRef.VirtualMethod();

// New (hiding): depends on reference type
Console.Write("  baseRef.HiddenMethod():     ");
baseRef.HiddenMethod();     // Calls BASE version
Console.Write("  derivedRef.HiddenMethod():  ");
derivedRef.HiddenMethod();  // Calls DERIVED version

// =============================================================================
// CLASS DEFINITIONS
// =============================================================================

/// <summary>
/// Abstract base class — cannot be instantiated directly.
/// </summary>
abstract class Shape
{
    public string Name { get; }

    protected Shape(string name) => Name = name;

    // Abstract methods — MUST be overridden by derived classes
    public abstract double Area();
    public abstract double Perimeter();

    // Virtual method — CAN be overridden (has a default implementation)
    public virtual string Describe()
        => $"{GetType().Name} with area {Area():F2}";
}

class Circle : Shape
{
    public double Radius { get; }

    public Circle(string name, double radius) : base(name)
        => Radius = radius;

    public override double Area() => Math.PI * Radius * Radius;
    public override double Perimeter() => 2 * Math.PI * Radius;
}

class Rectangle : Shape
{
    public double Width { get; }
    public double Height { get; }

    public Rectangle(string name, double width, double height) : base(name)
    {
        Width = width;
        Height = height;
    }

    public override double Area() => Width * Height;
    public override double Perimeter() => 2 * (Width + Height);
}

class Triangle : Shape
{
    public double A { get; }
    public double B { get; }
    public double C { get; }

    public Triangle(string name, double a, double b, double c) : base(name)
    {
        A = a; B = b; C = c;
    }

    public override double Area()
    {
        double s = (A + B + C) / 2;
        return Math.Sqrt(s * (s - A) * (s - B) * (s - C));
    }

    public override double Perimeter() => A + B + C;
}

// Square inherits from Rectangle
class Square : Rectangle
{
    public Square(string name, double side) : base(name, side, side) { }

    public override string Describe()
        => $"Square with side {Width} (area {Area():F2})";
}

/// <summary>
/// Virtual / Override demonstration.
/// </summary>
class Animal
{
    public string Name { get; }

    public Animal(string name) => Name = name;

    public virtual void Speak()
        => Console.WriteLine("...");
}

class Dog : Animal
{
    public Dog(string name) : base(name) { }

    // Override: polymorphic behavior
    public sealed override void Speak()
        => Console.WriteLine("Woof!");
}

class Cat : Animal
{
    public Cat(string name) : base(name) { }

    public override void Speak()
        => Console.WriteLine("Meow!");
}

// Kitten overrides Cat's Speak
class Kitten : Cat
{
    public Kitten(string name) : base(name) { }

    public override void Speak()
        => Console.WriteLine("Mew!");
}

// GoldenRetriever cannot override Speak (sealed in Dog)
class GoldenRetriever : Dog
{
    public GoldenRetriever(string name) : base(name) { }
    // public override void Speak() { } // Compile error: sealed
}

/// <summary>
/// Base keyword — calling parent constructor and methods.
/// </summary>
class Employee
{
    public string Name { get; }
    public decimal Salary { get; set; }

    public Employee(string name, decimal salary)
    {
        Name = name;
        Salary = salary;
    }

    public override string ToString()
        => $"Employee({Name}, {Salary:C})";
}

class Manager : Employee
{
    public string Department { get; }

    public Manager(string name, decimal salary, string department)
        : base(name, salary)
    {
        Department = department;
    }

    // Calls base.ToString() and extends it
    public override string ToString()
        => $"{base.ToString()}, Dept: {Department}";
}

/// <summary>
/// Sealed class — cannot be inherited.
/// </summary>
sealed class MathConstant
{
    public string Name { get; }
    public double Value { get; }

    public MathConstant(string name, double value)
    {
        Name = name;
        Value = value;
    }
}

/// <summary>
/// Protected members — accessible to derived classes.
/// </summary>
class Vehicle
{
    public string Model { get; }
    protected double _fuelLevel; // Accessible to derived classes

    public Vehicle(string model, double fuel)
    {
        Model = model;
        _fuelLevel = fuel;
    }

    public virtual void Drive(double distance)
    {
        double consumption = distance * 0.1;
        if (consumption > _fuelLevel)
        {
            Console.WriteLine($"  Not enough fuel to drive {distance}km.");
            return;
        }
        _fuelLevel -= consumption;
        Console.WriteLine($"  Drove {distance}km. Fuel remaining: {_fuelLevel:F1}");
    }

    public override string ToString()
        => $"Vehicle({Model}, fuel={_fuelLevel:F1})";
}

class ElectricVehicle : Vehicle
{
    public ElectricVehicle(string model, double battery)
        : base(model, battery) { }

    public override void Drive(double distance)
    {
        // Uses protected _fuelLevel from base
        double consumption = distance * 0.2; // EVs consume more per km in this model
        if (consumption > _fuelLevel)
        {
            Console.WriteLine($"  Battery too low to drive {distance}km.");
            return;
        }
        _fuelLevel -= consumption;
        Console.WriteLine($"  [EV] Drove {distance}km. Battery: {_fuelLevel:F1}%");
    }
}

/// <summary>
/// Method hiding (new) vs override demonstration.
/// </summary>
class BaseClass
{
    public virtual void VirtualMethod()
        => Console.WriteLine("Base.VirtualMethod");

    public void HiddenMethod()
        => Console.WriteLine("Base.HiddenMethod");
}

class DerivedClass : BaseClass
{
    // Override: polymorphic (called through base reference)
    public override void VirtualMethod()
        => Console.WriteLine("Derived.VirtualMethod");

    // New: hides base method (NOT polymorphic)
    public new void HiddenMethod()
        => Console.WriteLine("Derived.HiddenMethod");
}
