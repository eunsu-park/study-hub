# 10. TypeScript Fundamentals

**Previous**: [Practical Projects](./09_Practical_Projects.md) | **Next**: [Web Accessibility](./11_Web_Accessibility.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the advantages of TypeScript over plain JavaScript for large-scale projects
2. Annotate variables, functions, and return types using TypeScript's basic type system
3. Define object shapes with interfaces and distinguish when to use type aliases
4. Implement reusable, type-safe functions and classes using generics
5. Apply generic constraints with `extends` and `keyof` to enforce type relationships
6. Leverage built-in utility types such as `Partial`, `Pick`, `Omit`, and `Record`
7. Configure a TypeScript project with `tsconfig.json` and compile to JavaScript

---

JavaScript's flexibility is both its greatest strength and its biggest liability at scale. TypeScript adds a static type layer that catches entire categories of bugs at compile time -- before they ever reach your users. Adopting TypeScript does not mean abandoning JavaScript; every valid JavaScript program is already valid TypeScript. This lesson equips you with the core type system features that make TypeScript indispensable for modern web development.

> **Analogy:** TypeScript is JavaScript with guardrails. You can still drive at full speed, but the guardrails keep you from going off the cliff. The type system catches mistakes at compile time that would otherwise crash at runtime, turning "Uncaught TypeError" into a red squiggly line in your editor.

## Table of Contents
1. [Introduction to TypeScript](#1-introduction-to-typescript)
2. [Basic Types](#2-basic-types)
3. [Interfaces and Types](#3-interfaces-and-types)
4. [Function Types](#4-function-types)
5. [Generics](#5-generics)
6. [Utility Types](#6-utility-types)
7. [Practice Problems](#7-practice-problems)

---

## 1. Introduction to TypeScript

### 1.1 What is TypeScript?

```
┌─────────────────────────────────────────────────────────────────┐
│                    TypeScript Overview                           │
│                                                                 │
│   TypeScript = JavaScript + Static Types                       │
│                                                                 │
│   Features:                                                     │
│   - Developed by Microsoft                                      │
│   - Superset of JavaScript                                     │
│   - Compile-time type checking                                 │
│   - All JavaScript code is valid TypeScript                    │
│                                                                 │
│   Advantages:                                                   │
│   - Catch errors before runtime                                │
│   - Enhanced IDE support (autocomplete, refactoring)           │
│   - Code readability and documentation                         │
│   - Easy maintenance for large projects                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Installation and Setup

```bash
# Install TypeScript
npm install -g typescript

# Check version
tsc --version

# Initialize project
npm init -y
npm install typescript --save-dev

# Generate tsconfig.json
npx tsc --init
```

```json
// tsconfig.json basic configuration
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,          // Enables strictNullChecks, noImplicitAny, and 10+ other checks — catches bugs at compile time that would be runtime errors in plain JS
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "moduleResolution": "node"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 1.3 Compilation and Execution

```bash
# Compile single file
tsc hello.ts

# Compile entire project
tsc

# Watch mode (auto-compile on file changes)
tsc --watch

# ts-node for direct execution (development)
npm install -g ts-node
ts-node hello.ts
```

---

## 2. Basic Types

### 2.1 Primitive Types

```typescript
// String
let name: string = "TypeScript";
let greeting: string = `Hello, ${name}!`;

// Number
let age: number = 25;
let price: number = 99.99;
let hex: number = 0xf00d;

// Boolean
let isActive: boolean = true;
let hasError: boolean = false;

// null and undefined
let nothing: null = null;
let notDefined: undefined = undefined;

// BigInt (ES2020+)
let bigNumber: bigint = 9007199254740991n;

// Symbol
let sym: symbol = Symbol("unique");
```

### 2.2 Arrays and Tuples

```typescript
// Array types (two ways)
let numbers: number[] = [1, 2, 3, 4, 5];
let strings: Array<string> = ["a", "b", "c"];

// Multi-dimensional arrays
let matrix: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
];

// Tuple (fixed length, fixed type array)
let tuple: [string, number] = ["Alice", 30];
let rgb: [number, number, number] = [255, 128, 0];

// Tuple element access
const [userName, userAge] = tuple;
console.log(userName); // "Alice"

// Named tuples (improved readability)
type Point = [x: number, y: number];
const point: Point = [10, 20];
```

### 2.3 Object Types

```typescript
// Basic object type
let person: { name: string; age: number } = {
  name: "Bob",
  age: 25,
};

// Optional properties (?)
let config: { host: string; port?: number } = {
  host: "localhost",
  // port is optional
};

// Readonly properties
let user: { readonly id: number; name: string } = {
  id: 1,
  name: "Alice",
};
// user.id = 2;  // Error! readonly

// Index signature
let dictionary: { [key: string]: number } = {
  apple: 1,
  banana: 2,
};
```

### 2.4 Special Types

```typescript
// any - Allows all types (use sparingly)
let anything: any = "hello";
anything = 42;
anything = { foo: "bar" };

// unknown - Safer alternative to any
let unknownValue: unknown = "hello";
// unknownValue.toUpperCase();  // Error!
if (typeof unknownValue === "string") {
  unknownValue.toUpperCase(); // OK - after type guard
}

// void - No return value
function logMessage(msg: string): void {
  console.log(msg);
}

// never - Never returns
function throwError(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {}
}
```

### 2.5 Union and Intersection

```typescript
// Union type (|) - One of several types
let id: string | number;
id = "abc";
id = 123;

type Status = "pending" | "approved" | "rejected";
let orderStatus: Status = "pending";

// Intersection type (&) - Combine all types
type Name = { name: string };
type Age = { age: number };
type Person = Name & Age;

const person: Person = {
  name: "Alice",
  age: 30,
};
```

### 2.6 Type Inference and Type Assertion

```typescript
// Type inference - TypeScript automatically determines type
let message = "Hello"; // inferred as string
let count = 10; // inferred as number

// Type assertion
let someValue: unknown = "this is a string";

// Method 1: as syntax (recommended)
let strLength1: number = (someValue as string).length;

// Method 2: angle-bracket syntax (conflicts with JSX)
let strLength2: number = (<string>someValue).length;

// const assertion
let colors = ["red", "green", "blue"] as const;
// readonly ["red", "green", "blue"] type

// Non-null assertion (!)
function getLength(str: string | null): number {
  return str!.length; // Assert not null
}
```

---

## 3. Interfaces and Types

### 3.1 Interface Basics

```typescript
// Interface definition
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // optional
  readonly createdAt: Date; // readonly
}

// Using interface
const user: User = {
  id: 1,
  name: "Alice",
  email: "alice@example.com",
  createdAt: new Date(),
};

// Function type interface
interface Calculator {
  (a: number, b: number): number;
}

const add: Calculator = (a, b) => a + b;
```

### 3.2 Interface Extension

```typescript
// Interface inheritance
interface Animal {
  name: string;
  age: number;
}

interface Dog extends Animal {
  breed: string;
  bark(): void;
}

const myDog: Dog = {
  name: "Buddy",
  age: 3,
  breed: "Labrador",
  bark() {
    console.log("Woof!");
  },
};

// Multiple inheritance
interface Pet extends Animal {
  owner: string;
}

interface ServiceDog extends Dog, Pet {
  certificationId: string;
}
```

### 3.3 Type Alias

```typescript
// Type alias definition
type ID = string | number;
type Point = { x: number; y: number };
type Callback = (data: string) => void;

// Usage
let userId: ID = "user_123";
let position: Point = { x: 10, y: 20 };

// Useful for union types
type Result<T> = { success: true; data: T } | { success: false; error: string };

function fetchData(): Result<User> {
  return { success: true, data: { id: 1, name: "Alice", email: "a@b.com", createdAt: new Date() } };
}
```

### 3.4 Interface vs Type

```typescript
// Interface - Declaration merging possible
interface Window {
  title: string;
}

interface Window {
  size: number; // Automatically merged
}

// Type - No merging, more flexible
type StringOrNumber = string | number; // Union
type Point = [number, number]; // Tuple

// Recommendations:
// - Object shape definition: use interface
// - Union, tuple, primitive type alias: use type
// - Library API: interface (extensible)
```

---

## 4. Function Types

### 4.1 Function Type Definition

```typescript
// Function declaration
function add(a: number, b: number): number {
  return a + b;
}

// Arrow function
const multiply = (a: number, b: number): number => a * b;

// Function type alias
type MathOperation = (a: number, b: number) => number;

const divide: MathOperation = (a, b) => a / b;

// Function type interface
interface MathFunc {
  (a: number, b: number): number;
  description?: string;
}
```

### 4.2 Parameter Options

```typescript
// Optional parameters (?)
function greet(name: string, greeting?: string): string {
  return `${greeting || "Hello"}, ${name}!`;
}

// Default parameters
function greetWithDefault(name: string, greeting: string = "Hello"): string {
  return `${greeting}, ${name}!`;
}

// Rest parameters
function sum(...numbers: number[]): number {
  return numbers.reduce((acc, n) => acc + n, 0);
}

console.log(sum(1, 2, 3, 4, 5)); // 15
```

### 4.3 Function Overloading

```typescript
// Function overload signatures
function process(x: string): string;
function process(x: number): number;
function process(x: string | number): string | number {
  if (typeof x === "string") {
    return x.toUpperCase();
  }
  return x * 2;
}

console.log(process("hello")); // "HELLO"
console.log(process(5)); // 10
```

### 4.4 this Type

```typescript
interface Button {
  label: string;
  click(this: Button): void;
}

const button: Button = {
  label: "Submit",
  click() {
    console.log(`Clicked: ${this.label}`);
  },
};

button.click(); // OK
// const handler = button.click;
// handler();  // Error! this context lost
```

---

## 5. Generics

### 5.1 Generic Basics

```typescript
// Generic function
// T is inferred from the argument type — TypeScript ensures arg and return value share the same type without manual casting
function identity<T>(arg: T): T {
  return arg;
}

// Usage
let output1 = identity<string>("hello");
let output2 = identity<number>(42);
let output3 = identity("auto"); // Type inference — TypeScript infers T = string from the argument

// Generic array
function firstElement<T>(arr: T[]): T | undefined {
  return arr[0];
}

const first = firstElement([1, 2, 3]); // number | undefined
```

### 5.2 Generic Interfaces and Types

```typescript
// Generic interface
interface Box<T> {
  value: T;
}

const stringBox: Box<string> = { value: "hello" };
const numberBox: Box<number> = { value: 42 };

// Generic type alias
type Result<T> = {
  success: boolean;
  data: T;
};

type Pair<K, V> = {
  key: K;
  value: V;
};

const pair: Pair<string, number> = { key: "age", value: 30 };
```

### 5.3 Generic Constraints

```typescript
// Add constraints with extends
interface Lengthwise {
  length: number;
}

function logLength<T extends Lengthwise>(arg: T): T {
  console.log(arg.length);
  return arg;
}

logLength("hello"); // OK - string has length
logLength([1, 2, 3]); // OK - array has length
// logLength(123);    // Error! number has no length

// keyof constraint
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const person = { name: "Alice", age: 30 };
const name = getProperty(person, "name"); // string
const age = getProperty(person, "age"); // number
// getProperty(person, "email");  // Error!
```

### 5.4 Generic Classes

```typescript
class Queue<T> {
  private items: T[] = [];

  enqueue(item: T): void {
    this.items.push(item);
  }

  dequeue(): T | undefined {
    return this.items.shift();
  }

  peek(): T | undefined {
    return this.items[0];
  }

  get length(): number {
    return this.items.length;
  }
}

const numberQueue = new Queue<number>();
numberQueue.enqueue(1);
numberQueue.enqueue(2);
console.log(numberQueue.dequeue()); // 1
```

---

## 6. Utility Types

### 6.1 Basic Utility Types

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age?: number;
}

// Partial<T> - Makes all User fields optional — useful for update functions where you only change some fields
type PartialUser = Partial<User>;
// { id?: number; name?: string; email?: string; age?: number }

// Required<T> - Make all properties required
type RequiredUser = Required<User>;
// { id: number; name: string; email: string; age: number }

// Readonly<T> - Make all properties readonly
type ReadonlyUser = Readonly<User>;

// Pick<T, K> - Select only the fields you need; avoids exposing sensitive fields (e.g., password, token) in public types
type UserBasic = Pick<User, "id" | "name">;
// { id: number; name: string }

// Omit<T, K> - Exclude fields you don't want; useful for creating "input" types that drop server-generated fields like id or createdAt
type UserWithoutEmail = Omit<User, "email">;
// { id: number; name: string; age?: number }
```

### 6.2 Record and Mapping

```typescript
// Record<K, T> - Key-value mapping
type UserRole = "admin" | "user" | "guest";
type RolePermissions = Record<UserRole, string[]>;

const permissions: RolePermissions = {
  admin: ["read", "write", "delete"],
  user: ["read", "write"],
  guest: ["read"],
};

// Usage example
type PageInfo = {
  title: string;
  url: string;
};

type Pages = Record<"home" | "about" | "contact", PageInfo>;
```

### 6.3 Conditional Types

```typescript
// Exclude<T, U> - Exclude U from T
type Numbers = 1 | 2 | 3 | 4 | 5;
type SmallNumbers = Exclude<Numbers, 4 | 5>; // 1 | 2 | 3

// Extract<T, U> - Common types of T and U
type Common = Extract<"a" | "b" | "c", "a" | "c" | "d">; // "a" | "c"

// NonNullable<T> - Exclude null, undefined
type MaybeString = string | null | undefined;
type DefinitelyString = NonNullable<MaybeString>; // string

// ReturnType<T> - Function return type
function getUser() {
  return { id: 1, name: "Alice" };
}
type UserReturn = ReturnType<typeof getUser>;
// { id: number; name: string }

// Parameters<T> - Function parameter types
type UserParams = Parameters<typeof getUser>; // []
```

### 6.4 Template Literal Types

```typescript
// String literal combination
type Color = "red" | "green" | "blue";
type Size = "small" | "medium" | "large";

type ClassName = `${Size}-${Color}`;
// "small-red" | "small-green" | ... | "large-blue"

// Event name generation
type EventName<T extends string> = `on${Capitalize<T>}`;
type ClickEvent = EventName<"click">; // "onClick"
```

---

## 7. Practice Problems

### Exercise 1: Type Definition
Define types for the following data structure.

```typescript
// Example answer
interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
  inStock: boolean;
  tags?: string[];
}

interface CartItem {
  product: Product;
  quantity: number;
}

interface ShoppingCart {
  items: CartItem[];
  total: number;
  couponCode?: string;
}
```

### Exercise 2: Generic Function
Write a generic function to find the first element in an array matching a condition.

```typescript
// Example answer
function find<T>(arr: T[], predicate: (item: T) => boolean): T | undefined {
  for (const item of arr) {
    if (predicate(item)) {
      return item;
    }
  }
  return undefined;
}

// Usage
const numbers = [1, 2, 3, 4, 5];
const firstEven = find(numbers, (n) => n % 2 === 0); // 2

const users = [{ name: "Alice" }, { name: "Bob" }];
const alice = find(users, (u) => u.name === "Alice");
```

### Exercise 3: Utility Type Usage
Define API response types.

```typescript
// Example answer
interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
  timestamp: number;
}

type User = {
  id: number;
  name: string;
  email: string;
};

type UserResponse = ApiResponse<User>;
type UsersResponse = ApiResponse<User[]>;
type DeleteResponse = ApiResponse<{ deleted: boolean }>;

// Update type using Partial
type UserUpdate = Partial<Omit<User, "id">>;
```

---

## References
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)
- [TypeScript Playground](https://www.typescriptlang.org/play)
- [DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)

---

**Previous**: [Practical Projects](./09_Practical_Projects.md) | **Next**: [Web Accessibility](./11_Web_Accessibility.md)
