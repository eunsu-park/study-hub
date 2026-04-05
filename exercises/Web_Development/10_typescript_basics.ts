/**
 * Exercises for Lesson 10: TypeScript Basics
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: npx ts-node exercises/Web_Development/10_typescript_basics.ts
 *   or: tsc 10_typescript_basics.ts && node 10_typescript_basics.js
 */

// === Exercise 1: Type Annotations ===
// Problem: Create typed variables and a function that formats a user greeting.

function formatGreeting(name: string, age: number, isPremium: boolean): string {
    const prefix = isPremium ? "⭐ " : "";
    return `${prefix}Hello, ${name} (age ${age})!`;
}

function exercise1(): void {
    const testCases: { name: string; age: number; premium: boolean; expected: string }[] = [
        { name: "Alice", age: 30, premium: true, expected: "⭐ Hello, Alice (age 30)!" },
        { name: "Bob", age: 25, premium: false, expected: "Hello, Bob (age 25)!" },
    ];

    console.log("=== Exercise 1: Type Annotations ===");
    for (const tc of testCases) {
        const result = formatGreeting(tc.name, tc.age, tc.premium);
        const ok = result === tc.expected;
        console.log(`  ${ok ? "PASS" : "FAIL"}: ${result}`);
    }
}

// === Exercise 2: Interfaces and Types ===
// Problem: Define a Product interface and functions to work with a shopping cart.

interface Product {
    id: number;
    name: string;
    price: number;
    category: "electronics" | "clothing" | "food";
}

interface CartItem {
    product: Product;
    quantity: number;
}

function cartTotal(items: CartItem[]): number {
    return items.reduce((sum, item) => sum + item.product.price * item.quantity, 0);
}

function filterByCategory(products: Product[], category: Product["category"]): Product[] {
    return products.filter(p => p.category === category);
}

function exercise2(): void {
    console.log("\n=== Exercise 2: Interfaces and Types ===");
    const products: Product[] = [
        { id: 1, name: "Laptop", price: 999, category: "electronics" },
        { id: 2, name: "T-Shirt", price: 25, category: "clothing" },
        { id: 3, name: "Apple", price: 2, category: "food" },
        { id: 4, name: "Headphones", price: 150, category: "electronics" },
    ];

    const cart: CartItem[] = [
        { product: products[0], quantity: 1 },
        { product: products[2], quantity: 5 },
    ];

    console.log(`  Cart total: $${cartTotal(cart)}`);
    console.log(`  Electronics: ${filterByCategory(products, "electronics").map(p => p.name).join(", ")}`);
}

// === Exercise 3: Generics ===
// Problem: Implement a type-safe Stack<T> and a generic find function.

class Stack<T> {
    private items: T[] = [];

    push(item: T): void { this.items.push(item); }
    pop(): T | undefined { return this.items.pop(); }
    peek(): T | undefined { return this.items[this.items.length - 1]; }
    get size(): number { return this.items.length; }
    isEmpty(): boolean { return this.items.length === 0; }
}

function findBy<T>(items: T[], predicate: (item: T) => boolean): T | undefined {
    for (const item of items) {
        if (predicate(item)) return item;
    }
    return undefined;
}

function exercise3(): void {
    console.log("\n=== Exercise 3: Generics ===");

    const numStack = new Stack<number>();
    numStack.push(10);
    numStack.push(20);
    numStack.push(30);
    console.log(`  Peek: ${numStack.peek()}, Size: ${numStack.size}`);
    console.log(`  Pop: ${numStack.pop()}, Size: ${numStack.size}`);

    const strStack = new Stack<string>();
    strStack.push("hello");
    strStack.push("world");
    console.log(`  String peek: ${strStack.peek()}`);

    interface User { id: number; name: string; }
    const users: User[] = [
        { id: 1, name: "Alice" },
        { id: 2, name: "Bob" },
        { id: 3, name: "Charlie" },
    ];
    const found = findBy(users, u => u.name === "Bob");
    console.log(`  Found: ${found ? found.name : "not found"}`);
}

// === Exercise 4: Utility Types ===
// Problem: Use Partial, Pick, Readonly, and Record to transform types.

interface Config {
    host: string;
    port: number;
    debug: boolean;
    logLevel: "info" | "warn" | "error";
}

function mergeConfig(defaults: Config, overrides: Partial<Config>): Config {
    return { ...defaults, ...overrides };
}

type ConfigSummary = Pick<Config, "host" | "port">;

function exercise4(): void {
    console.log("\n=== Exercise 4: Utility Types ===");

    const defaults: Config = { host: "localhost", port: 8080, debug: false, logLevel: "info" };
    const production = mergeConfig(defaults, { host: "0.0.0.0", port: 443, debug: false });
    console.log(`  Production: ${production.host}:${production.port} (debug=${production.debug})`);

    const summary: ConfigSummary = { host: production.host, port: production.port };
    console.log(`  Summary: ${JSON.stringify(summary)}`);

    const frozen: Readonly<Config> = defaults;
    // frozen.port = 3000;  // ERROR: Cannot assign to 'port' because it is a read-only property
    console.log(`  Readonly: ${frozen.host}:${frozen.port}`);

    const statusCodes: Record<number, string> = { 200: "OK", 404: "Not Found", 500: "Server Error" };
    console.log(`  Status 404: ${statusCodes[404]}`);
}

// === Run All ===
exercise1();
exercise2();
exercise3();
exercise4();
