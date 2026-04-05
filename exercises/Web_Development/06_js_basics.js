/**
 * Exercises for Lesson 06: JavaScript Basics
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: node exercises/Web_Development/06_js_basics.js
 */

// === Exercise 1: Variables and Conditionals ===
// Problem: Write a function that categorizes age into groups:
//   0-12: "Child", 13-19: "Teenager", 20-64: "Adult", 65+: "Senior"

function getAgeGroup(age) {
    if (typeof age !== 'number' || age < 0) return 'Invalid age';
    if (age <= 12) return 'Child';
    if (age <= 19) return 'Teenager';
    if (age <= 64) return 'Adult';
    return 'Senior';
}

function exercise1() {
    const testCases = [
        { age: 5, expected: 'Child' },
        { age: 12, expected: 'Child' },
        { age: 13, expected: 'Teenager' },
        { age: 19, expected: 'Teenager' },
        { age: 20, expected: 'Adult' },
        { age: 45, expected: 'Adult' },
        { age: 64, expected: 'Adult' },
        { age: 65, expected: 'Senior' },
        { age: 90, expected: 'Senior' },
        { age: -1, expected: 'Invalid age' },
    ];

    let passed = 0;
    for (const tc of testCases) {
        const result = getAgeGroup(tc.age);
        const ok = result === tc.expected;
        if (ok) passed++;
        console.log(
            `  age=${String(tc.age).padStart(3)} => ${result.padEnd(12)} ${ok ? 'PASS' : `FAIL (expected: ${tc.expected})`}`
        );
    }
    console.log(`  Result: ${passed}/${testCases.length} passed`);
}


// === Exercise 2: Array Methods ===
// Problem: Filter even numbers from an array and square them.
//   Input:  [1, 2, 3, 4, 5, 6]
//   Output: [4, 16, 36]

function filterAndSquareEvens(numbers) {
    return numbers
        .filter(n => n % 2 === 0)
        .map(n => n ** 2);
}

function exercise2() {
    const input = [1, 2, 3, 4, 5, 6];
    const result = filterAndSquareEvens(input);
    const expected = [4, 16, 36];

    console.log(`  Input:    [${input}]`);
    console.log(`  Result:   [${result}]`);
    console.log(`  Expected: [${expected}]`);

    const passed = JSON.stringify(result) === JSON.stringify(expected);
    console.log(`  ${passed ? 'PASS' : 'FAIL'}`);

    // Additional test cases
    const tests = [
        { input: [], expected: [] },
        { input: [1, 3, 5], expected: [] },
        { input: [2, 4], expected: [4, 16] },
        { input: [10], expected: [100] },
    ];

    for (const tc of tests) {
        const res = filterAndSquareEvens(tc.input);
        const ok = JSON.stringify(res) === JSON.stringify(tc.expected);
        console.log(`  [${tc.input}] => [${res}] ${ok ? 'PASS' : 'FAIL'}`);
    }
}


// === Exercise 3: Working with Objects ===
// Problem: Find users aged 28 or older and return their names.
//   Input: [{ id: 1, name: 'Kim', age: 25 }, { id: 2, name: 'Lee', age: 30 }, { id: 3, name: 'Park', age: 28 }]
//   Output: ['Lee', 'Park']

function getNamesOfAdults(users, minAge) {
    return users
        .filter(user => user.age >= minAge)
        .map(user => user.name);
}

function exercise3() {
    const users = [
        { id: 1, name: 'Kim', age: 25 },
        { id: 2, name: 'Lee', age: 30 },
        { id: 3, name: 'Park', age: 28 },
    ];

    const result = getNamesOfAdults(users, 28);
    const expected = ['Lee', 'Park'];

    console.log('  Users:', JSON.stringify(users));
    console.log(`  Result (age >= 28):   [${result}]`);
    console.log(`  Expected:             [${expected}]`);

    const passed = JSON.stringify(result) === JSON.stringify(expected);
    console.log(`  ${passed ? 'PASS' : 'FAIL'}`);

    // Bonus: demonstrate chaining with reduce to get combined info
    const summary = users
        .filter(u => u.age >= 28)
        .reduce((acc, u) => {
            acc.names.push(u.name);
            acc.totalAge += u.age;
            return acc;
        }, { names: [], totalAge: 0 });

    console.log(`  Bonus - Names: [${summary.names}], Average age: ${(summary.totalAge / summary.names.length).toFixed(1)}`);
}


// === Bonus Exercise: Destructuring and Spread ===
// Not in the original exercises, but demonstrates ES6+ features from the lesson.

function bonusExercise() {
    // Destructuring assignment
    const [first, second, ...rest] = [10, 20, 30, 40, 50];
    console.log(`  Array destructuring: first=${first}, second=${second}, rest=[${rest}]`);

    // Object destructuring with defaults and renaming
    const config = { host: 'localhost', port: 3000 };
    const { host, port, protocol = 'http' } = config;
    console.log(`  Object destructuring: ${protocol}://${host}:${port}`);

    // Spread operator for merging
    const defaults = { theme: 'light', lang: 'en', fontSize: 16 };
    const userPrefs = { theme: 'dark', fontSize: 18 };
    const merged = { ...defaults, ...userPrefs };
    console.log('  Spread merge:', JSON.stringify(merged));

    // Optional chaining and nullish coalescing
    const user = { name: 'Alice', address: { city: 'Seoul' } };
    const city = user?.address?.city ?? 'Unknown';
    const zip = user?.address?.zip ?? 'No ZIP';
    console.log(`  Optional chaining: city=${city}, zip=${zip}`);

    // Template literals with expressions
    const items = ['apple', 'banana', 'cherry'];
    const summary = `Found ${items.length} items: ${items.join(', ')}`;
    console.log(`  Template literal: ${summary}`);
}


// ===== Run all exercises =====
console.log('=== Exercise 1: Variables and Conditionals ===');
exercise1();

console.log('\n=== Exercise 2: Array Methods ===');
exercise2();

console.log('\n=== Exercise 3: Working with Objects ===');
exercise3();

console.log('\n=== Bonus: Destructuring and Spread ===');
bonusExercise();

console.log('\nAll exercises completed!');
