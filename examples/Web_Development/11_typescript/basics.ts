/**
 * TypeScript Basics
 * - Basic types
 * - Type inference
 * - Union/literal types
 * - Type guards
 */

// ============================================
// 1. Basic Types
// ============================================

// Primitive types
const name: string = "John";
const age: number = 25;
const isActive: boolean = true;
const nothing: null = null;
const notDefined: undefined = undefined;

// Arrays
const numbers: number[] = [1, 2, 3, 4, 5];
const names: Array<string> = ["Alice", "Bob", "Charlie"];

// Tuples (fixed length and types)
const person: [string, number] = ["John", 25];
const rgb: [number, number, number] = [255, 128, 0];

// Objects
const user: { name: string; age: number; email?: string } = {
    name: "Jane",
    age: 30,
    // email is optional
};

console.log("=== Basic Types ===");
console.log(`Name: ${name}, Age: ${age}, Active: ${isActive}`);
console.log(`Number array: ${numbers.join(", ")}`);
console.log(`Tuple: ${person[0]} (${person[1]} years old)`);

// ============================================
// 2. Type Inference
// ============================================

// TypeScript automatically infers types
let inferredString = "Hello";  // Inferred as string
let inferredNumber = 42;       // Inferred as number
let inferredArray = [1, 2, 3]; // Inferred as number[]

// Function return type inference
function add(a: number, b: number) {
    return a + b;  // Inferred as number return
}

const sum = add(10, 20);  // sum is inferred as number

console.log("\n=== Type Inference ===");
console.log(`Inferred sum: ${sum}`);

// ============================================
// 3. Union Types
// ============================================

// One of several types
type StringOrNumber = string | number;

function printId(id: StringOrNumber) {
    console.log(`ID: ${id}`);

    // Type-specific handling
    if (typeof id === "string") {
        console.log(`  (string, length: ${id.length})`);
    } else {
        console.log(`  (number, doubled: ${id * 2})`);
    }
}

console.log("\n=== Union Types ===");
printId("user_123");
printId(42);

// ============================================
// 4. Literal Types
// ============================================

// Why: Literal types restrict values to an exact set at compile time, turning invalid
// inputs (e.g., "diagonal") into type errors instead of silent runtime bugs
// Allow only specific values
type Direction = "up" | "down" | "left" | "right";
type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

function move(direction: Direction) {
    console.log(`Moving ${direction}`);
}

function rollDice(): DiceValue {
    return Math.ceil(Math.random() * 6) as DiceValue;
}

console.log("\n=== Literal Types ===");
move("up");
move("left");
console.log(`Dice: ${rollDice()}`);

// ============================================
// 5. any, unknown, never
// ============================================

// any: Allows all types (disables type checking, avoid using)
let anything: any = "hello";
anything = 42;
anything = true;

// Why: unknown is the type-safe alternative to any - it accepts all values but forces you to
// narrow the type before using it, catching bugs that any would silently allow
// unknown: Allows all types but requires type checking before use
let unknownValue: unknown = "hello";

// unknown cannot be used directly, type check required
if (typeof unknownValue === "string") {
    console.log(`unknown value: ${unknownValue.toUpperCase()}`);
}

// never: A type that never occurs
function throwError(message: string): never {
    throw new Error(message);
}

function infiniteLoop(): never {
    while (true) {
        // Infinite loop
    }
}

console.log("\n=== any, unknown ===");
console.log(`any value: ${anything}`);

// ============================================
// 6. Type Alias
// ============================================

// Why: Type aliases give meaningful names to complex types, improving readability and
// enabling reuse across the codebase without duplicating structural definitions
type Point = {
    x: number;
    y: number;
};

type ID = string | number;

type UserRole = "admin" | "editor" | "viewer";

type UserProfile = {
    id: ID;
    name: string;
    role: UserRole;
    location: Point;
};

const admin: UserProfile = {
    id: "admin_001",
    name: "Admin",
    role: "admin",
    location: { x: 0, y: 0 }
};

console.log("\n=== Type Alias ===");
console.log(`User: ${admin.name} (${admin.role})`);

// ============================================
// 7. Type Guards
// ============================================

type Circle = { kind: "circle"; radius: number };
type Rectangle = { kind: "rectangle"; width: number; height: number };
type Shape = Circle | Rectangle;

// Why: Discriminated unions use a literal "kind" field so TypeScript can narrow the type
// in each switch branch, providing exhaustive checking and eliminating type casts
// Discriminated Union
function getArea(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
    }
}

// Type guard with the 'in' operator
function printShape(shape: Shape) {
    if ("radius" in shape) {
        console.log(`Circle: radius ${shape.radius}`);
    } else {
        console.log(`Rectangle: ${shape.width} x ${shape.height}`);
    }
}

console.log("\n=== Type Guards ===");
const circle: Circle = { kind: "circle", radius: 5 };
const rect: Rectangle = { kind: "rectangle", width: 10, height: 20 };

console.log(`Circle area: ${getArea(circle).toFixed(2)}`);
console.log(`Rectangle area: ${getArea(rect)}`);
printShape(circle);
printShape(rect);

// ============================================
// 8. Type Assertion
// ============================================

// Type assertion with the 'as' keyword
const input = document.getElementById("myInput") as HTMLInputElement;
// Or angle-bracket syntax (not usable in JSX)
// const input = <HTMLInputElement>document.getElementById("myInput");

// Note: Type assertions have no runtime effect
const maybeString: unknown = "hello world";
const strLength = (maybeString as string).length;

console.log("\n=== Type Assertion ===");
console.log(`String length: ${strLength}`);

// ============================================
// 9. Function Types
// ============================================

// Function type definition
type MathOperation = (a: number, b: number) => number;

const multiply: MathOperation = (a, b) => a * b;
const divide: MathOperation = (a, b) => a / b;

// Optional parameters and defaults
function greet(name: string, greeting: string = "Hello"): string {
    return `${greeting}, ${name}!`;
}

// Rest parameters
function sumAll(...numbers: number[]): number {
    return numbers.reduce((acc, cur) => acc + cur, 0);
}

// Why: Overload signatures let callers see distinct input-output contracts, giving better
// IntelliSense hints than a single union signature that obscures which output type maps to which input
// Overload signatures
function format(value: string): string;
function format(value: number): string;
function format(value: string | number): string {
    if (typeof value === "string") {
        return value.toUpperCase();
    } else {
        return value.toFixed(2);
    }
}

console.log("\n=== Function Types ===");
console.log(`Multiply: 3 * 4 = ${multiply(3, 4)}`);
console.log(`Divide: 10 / 3 = ${divide(10, 3).toFixed(2)}`);
console.log(greet("John"));
console.log(greet("Jane", "Nice to meet you"));
console.log(`Sum: ${sumAll(1, 2, 3, 4, 5)}`);
console.log(`String format: ${format("hello")}`);
console.log(`Number format: ${format(3.14159)}`);

// ============================================
// 10. Enum
// ============================================

// Why: Enums group related constants under one namespace and provide reverse mapping,
// but prefer string enums or union literals in modern TS for better tree-shaking
// Numeric enum
enum Direction2 {
    Up = 1,
    Down,
    Left,
    Right
}

// String enum
enum HttpStatus {
    OK = "OK",
    NotFound = "NOT_FOUND",
    ServerError = "SERVER_ERROR"
}

// const enum (inlined)
const enum Color {
    Red = "#ff0000",
    Green = "#00ff00",
    Blue = "#0000ff"
}

console.log("\n=== Enum ===");
console.log(`Direction.Up: ${Direction2.Up}`);
console.log(`Direction.Right: ${Direction2.Right}`);
console.log(`HttpStatus.OK: ${HttpStatus.OK}`);
console.log(`Color.Red: ${Color.Red}`);

// ============================================
// 11. Null Checks
// ============================================

// When strictNullChecks is enabled
function processValue(value: string | null | undefined) {
    // Optional chaining
    const length = value?.length;

    // Nullish coalescing
    const defaulted = value ?? "default";

    // Non-null assertion (use only when certain)
    // const definite = value!.length;

    console.log(`Length: ${length}, Value: ${defaulted}`);
}

console.log("\n=== Null Checks ===");
processValue("hello");
processValue(null);
processValue(undefined);

// ============================================
// Execution Example
// ============================================
console.log("\n=== TypeScript Basics Complete ===");
console.log("Compile: npx tsc basics.ts");
console.log("Run: node basics.js");
