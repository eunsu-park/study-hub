/**
 * TypeScript Interfaces and Generics
 * - Interfaces
 * - Classes and types
 * - Generics
 * - Utility types
 */

// ============================================
// 1. Interface Basics
// ============================================

interface User {
    id: number;
    name: string;
    email: string;
    age?: number;  // Optional property
    readonly createdAt: Date;  // Read-only
}

const user1: User = {
    id: 1,
    name: "John",
    email: "john@example.com",
    createdAt: new Date()
};

// user1.createdAt = new Date();  // Error: readonly

console.log("=== Interface Basics ===");
console.log(`User: ${user1.name} (${user1.email})`);

// ============================================
// 2. Interface Extension
// ============================================

interface Person {
    name: string;
    age: number;
}

interface Employee extends Person {
    employeeId: string;
    department: string;
}

interface Manager extends Employee {
    teamSize: number;
    reports: Employee[];
}

const manager: Manager = {
    name: "Kim",
    age: 45,
    employeeId: "E001",
    department: "Engineering",
    teamSize: 5,
    reports: []
};

console.log("\n=== Interface Extension ===");
console.log(`Manager: ${manager.name}, ${manager.department}, ${manager.teamSize} team members`);

// ============================================
// 3. Interface Merging
// ============================================

interface Config {
    apiUrl: string;
}

interface Config {
    timeout: number;
}

// Why: Declaration merging lets libraries expose extensible interfaces that consumers can
// augment without modifying the original source - this is why interfaces are preferred for public APIs
// The two declarations are automatically merged
const config: Config = {
    apiUrl: "https://api.example.com",
    timeout: 5000
};

console.log("\n=== Interface Merging ===");
console.log(`Config: ${config.apiUrl}, timeout: ${config.timeout}ms`);

// ============================================
// 4. Function Interface
// ============================================

interface MathFunc {
    (x: number, y: number): number;
}

interface Calculator {
    add: MathFunc;
    subtract: MathFunc;
    multiply: MathFunc;
    divide: MathFunc;
}

const calculator: Calculator = {
    add: (x, y) => x + y,
    subtract: (x, y) => x - y,
    multiply: (x, y) => x * y,
    divide: (x, y) => x / y
};

console.log("\n=== Function Interface ===");
console.log(`10 + 5 = ${calculator.add(10, 5)}`);
console.log(`10 - 5 = ${calculator.subtract(10, 5)}`);

// ============================================
// 5. Index Signatures
// ============================================

interface StringDictionary {
    [key: string]: string;
}

interface NumberArray {
    [index: number]: string;
}

const translations: StringDictionary = {
    hello: "Hello",
    goodbye: "Goodbye",
    thanks: "Thank you"
};

const colors: NumberArray = ["red", "green", "blue"];

console.log("\n=== Index Signatures ===");
console.log(`hello = ${translations["hello"]}`);
console.log(`colors[0] = ${colors[0]}`);

// ============================================
// 6. Classes and Interfaces
// ============================================

interface Animal {
    name: string;
    makeSound(): void;
}

interface Movable {
    move(distance: number): void;
}

// Why: Implementing multiple interfaces enforces that a class satisfies all required
// contracts at compile time, enabling polymorphism without the fragility of inheritance chains
class Dog implements Animal, Movable {
    constructor(public name: string) {}

    makeSound(): void {
        console.log(`${this.name}: Woof!`);
    }

    move(distance: number): void {
        console.log(`${this.name} moved ${distance}m.`);
    }
}

console.log("\n=== Classes and Interfaces ===");
const dog = new Dog("Buddy");
dog.makeSound();
dog.move(10);

// ============================================
// 7. Generics Basics
// ============================================

// Generic function
function identity<T>(arg: T): T {
    return arg;
}

// Generic array function
function firstElement<T>(arr: T[]): T | undefined {
    return arr[0];
}

// Why: K extends keyof T constrains the key parameter to valid property names of T,
// turning runtime "property not found" errors into compile-time type errors
// Generic object function
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

console.log("\n=== Generics Basics ===");
console.log(`identity("hello"): ${identity("hello")}`);
console.log(`identity(42): ${identity(42)}`);
console.log(`firstElement([1,2,3]): ${firstElement([1, 2, 3])}`);

const person = { name: "John", age: 25 };
console.log(`getProperty: ${getProperty(person, "name")}`);

// ============================================
// 8. Generic Interface
// ============================================

// Why: Making ApiResponse generic avoids duplicating the wrapper structure for every endpoint
// and ensures type safety flows from the API response all the way to component code
interface ApiResponse<T> {
    data: T;
    status: number;
    message: string;
}

interface UserData {
    id: number;
    name: string;
}

interface ProductData {
    id: number;
    title: string;
    price: number;
}

const userResponse: ApiResponse<UserData> = {
    data: { id: 1, name: "John" },
    status: 200,
    message: "Success"
};

const productResponse: ApiResponse<ProductData> = {
    data: { id: 1, title: "Laptop", price: 1000 },
    status: 200,
    message: "Success"
};

console.log("\n=== Generic Interface ===");
console.log(`User: ${userResponse.data.name}`);
console.log(`Product: ${productResponse.data.title}`);

// ============================================
// 9. Generic Class
// ============================================

class Stack<T> {
    private items: T[] = [];

    push(item: T): void {
        this.items.push(item);
    }

    pop(): T | undefined {
        return this.items.pop();
    }

    peek(): T | undefined {
        return this.items[this.items.length - 1];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }
}

console.log("\n=== Generic Class ===");
const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
numberStack.push(3);
console.log(`Stack peek: ${numberStack.peek()}`);
console.log(`Stack pop: ${numberStack.pop()}`);
console.log(`Stack size: ${numberStack.size()}`);

// ============================================
// 10. Generic Constraints
// ============================================

interface HasLength {
    length: number;
}

function logLength<T extends HasLength>(arg: T): number {
    console.log(`Length: ${arg.length}`);
    return arg.length;
}

console.log("\n=== Generic Constraints ===");
logLength("Hello");       // string has length
logLength([1, 2, 3, 4]);  // array has length too
logLength({ length: 10 }); // object works too

// ============================================
// 11. Utility Types
// ============================================

interface Todo {
    title: string;
    description: string;
    completed: boolean;
    createdAt: Date;
}

// Partial<T>: Makes all properties optional
type PartialTodo = Partial<Todo>;

// Required<T>: Makes all properties required
type RequiredTodo = Required<Todo>;

// Readonly<T>: Makes all properties read-only
type ReadonlyTodo = Readonly<Todo>;

// Pick<T, K>: Select specific properties
type TodoPreview = Pick<Todo, "title" | "completed">;

// Omit<T, K>: Exclude specific properties
type TodoWithoutDate = Omit<Todo, "createdAt">;

// Record<K, T>: Define key-value types
type PageInfo = Record<"home" | "about" | "contact", { title: string }>;

const pages: PageInfo = {
    home: { title: "Home" },
    about: { title: "About" },
    contact: { title: "Contact" }
};

console.log("\n=== Utility Types ===");

const partialTodo: PartialTodo = {
    title: "Partial only"
};

const todoPreview: TodoPreview = {
    title: "Preview",
    completed: false
};

console.log(`PartialTodo: ${JSON.stringify(partialTodo)}`);
console.log(`TodoPreview: ${JSON.stringify(todoPreview)}`);
console.log(`Pages: ${Object.keys(pages).join(", ")}`);

// ============================================
// 12. Conditional Types
// ============================================

type IsString<T> = T extends string ? "yes" : "no";

type A = IsString<string>;   // "yes"
type B = IsString<number>;   // "no"

// infer keyword
type GetReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type FuncReturn = GetReturnType<() => number>;  // number
type AsyncReturn = GetReturnType<() => Promise<string>>;  // Promise<string>

// Exclude, Extract
type T1 = Exclude<"a" | "b" | "c", "a">;  // "b" | "c"
type T2 = Extract<"a" | "b" | "c", "a" | "f">;  // "a"

// NonNullable
type T3 = NonNullable<string | null | undefined>;  // string

console.log("\n=== Conditional Types ===");
console.log("IsString<string> = 'yes'");
console.log("IsString<number> = 'no'");
console.log("Exclude<'a'|'b'|'c', 'a'> = 'b' | 'c'");

// ============================================
// 13. Mapped Types
// ============================================

type Nullable<T> = {
    [P in keyof T]: T[P] | null;
};

type Optional<T> = {
    [P in keyof T]?: T[P];
};

// Why: Template literal types combined with mapped types auto-generate getter signatures,
// ensuring type-safe accessor patterns that stay in sync as the source type evolves
type Getters<T> = {
    [P in keyof T as `get${Capitalize<string & P>}`]: () => T[P];
};

interface Point {
    x: number;
    y: number;
}

type NullablePoint = Nullable<Point>;  // { x: number | null; y: number | null }
type PointGetters = Getters<Point>;    // { getX: () => number; getY: () => number }

console.log("\n=== Mapped Types ===");
const nullablePoint: NullablePoint = { x: 10, y: null };
console.log(`NullablePoint: x=${nullablePoint.x}, y=${nullablePoint.y}`);

// ============================================
// 14. Template Literal Types
// ============================================

type EventName = "click" | "scroll" | "mousemove";
type Handler = `on${Capitalize<EventName>}`;  // "onClick" | "onScroll" | "onMousemove"

type Greeting = `Hello, ${string}!`;

const greeting: Greeting = "Hello, World!";

console.log("\n=== Template Literal Types ===");
console.log(`Greeting: ${greeting}`);
console.log("Handler type: 'onClick' | 'onScroll' | 'onMousemove'");

// ============================================
// Execution Example
// ============================================
console.log("\n=== TypeScript Interfaces/Generics Complete ===");
console.log("Compile: npx tsc interfaces.ts");
console.log("Run: node interfaces.js");
