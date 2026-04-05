# JavaScript Basics

**Previous**: [CSS Responsive Design](./05_CSS_Responsive.md) | **Next**: [JavaScript Events and DOM](./07_JS_Events_DOM.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how JavaScript fits into the HTML/CSS/JS triad and describe the difference between defer and async script loading
2. Declare variables using let and const, and explain why var is discouraged
3. Distinguish between primitive types (string, number, boolean, null, undefined, symbol, bigint) and reference types (object, array, function)
4. Apply arithmetic, comparison, logical, and nullish coalescing operators
5. Write conditional logic using if/else, switch, and the ternary operator
6. Implement loops with for, for...of, for...in, while, and do...while
7. Define functions using declarations, expressions, and arrow syntax, and use default parameters, rest parameters, and destructuring
8. Manipulate arrays and objects using modern ES6+ methods including map, filter, reduce, spread, and optional chaining

---

> **Analogy:** If HTML is the TV and CSS is the picture settings, JavaScript is the remote control. It lets you change channels (modify content), adjust volume (update styles), and respond to button presses (handle events) -- all without opening up the TV itself.

JavaScript is the programming language of the web browser. While HTML and CSS are declarative -- they describe what something is and how it looks -- JavaScript is imperative: it describes what to do. This makes it the key to interactivity, from validating a form field as the user types to fetching fresh data from a server without reloading the page.

## Table of Contents

1. [Getting Started with JavaScript](#getting-started-with-javascript)
2. [Variables and Constants](#variables-and-constants)
3. [Data Types](#data-types)
4. [Operators](#operators)
5. [Conditionals](#conditionals)
6. [Loops](#loops)
7. [Functions](#functions)
8. [Arrays](#arrays)
9. [Objects](#objects)
10. [ES6+ Syntax](#es6-syntax)

---

## Getting Started with JavaScript

### Using JavaScript in HTML

```html
<!-- Method 1: Internal script -->
<script>
    console.log('Hello, World!');
</script>

<!-- Method 2: External script (recommended) -->
<script src="main.js"></script>

<!-- Method 3: Inline event (not recommended) -->
<button onclick="alert('Clicked!')">Click</button>
```

### Script Location

```html
<!DOCTYPE html>
<html>
<head>
    <!-- In head blocks HTML parsing -->
    <script src="blocking.js"></script>

    <!-- defer: Downloads in parallel while HTML parses, then executes after DOM is ready;
         preferred over placing script at end of <body> because it keeps <script> declarations
         in <head> where they are visible early, while still guaranteeing DOM is fully built -->
    <script src="main.js" defer></script>

    <!-- async: Executes immediately after download (order not guaranteed) -->
    <script src="analytics.js" async></script>
</head>
<body>
    <h1>Hello</h1>

    <!-- At end of body executes after DOM is ready -->
    <script src="main.js"></script>
</body>
</html>
```

| Attribute | Execution Timing | Order Guaranteed | Use Case |
|-----------|------------------|------------------|----------|
| (none) | Immediately | O | - |
| `defer` | After DOM parsing | O | General scripts |
| `async` | After download | X | Analytics, ads |

### Developer Console

Press F12 in browser → Console tab to run JavaScript directly.

```javascript
console.log('Normal output');
console.warn('Warning');
console.error('Error');
console.table([{a: 1}, {a: 2}]);  // Table format
```

---

## Variables and Constants

### let (Variable)

Values can be reassigned.

```javascript
let name = 'John';
name = 'Jane';  // Reassignment allowed

let count;        // Declaration only (undefined)
count = 0;        // Assignment later
```

### const (Constant)

Values cannot be reassigned.

```javascript
const PI = 3.14159;
// PI = 3.14;  // Error! Cannot reassign

const user = { name: 'John' };
user.name = 'Jane';  // Object properties can be changed
// user = {};          // Error! Cannot reassign object itself
```

### var (Legacy, discouraged)

```javascript
var old = 'Old method';  // Function scope, hoisting issues
```

### Variable Naming Rules

```javascript
// Valid
let userName = 'kim';      // camelCase (recommended)
let user_name = 'kim';     // snake_case
let _private = 'secret';   // Start with underscore
let $element = 'dom';      // Start with dollar
let koreanName = 'valid';  // Unicode (not recommended)

// Invalid
// let 1name = 'x';        // Cannot start with number
// let user-name = 'x';    // No hyphens
// let let = 'x';          // Reserved word
```

### Scope

```javascript
// Block scope (let, const)
{
    let blockVar = 'Inside block';
    const blockConst = 'Inside block';
}
// console.log(blockVar);  // Error! Cannot access

// Function scope (var)
function test() {
    var funcVar = 'Inside function';
}
// console.log(funcVar);  // Error!

// Global scope
let globalVar = 'Accessible everywhere';
```

---

## Data Types

### Primitive Types

```javascript
// String
let str1 = 'Single quotes';
let str2 = "Double quotes";
let str3 = `Template literal ${str1}`;

// Number
let int = 42;
let float = 3.14;
let negative = -10;
let infinity = Infinity;
let notANumber = NaN;

// BigInt
let big = 9007199254740991n;

// Boolean
let isTrue = true;
let isFalse = false;

// undefined (no value)
let nothing;
console.log(nothing);  // undefined

// null (intentional empty value)
let empty = null;

// Symbol (unique identifier)
let sym = Symbol('description');
```

### Reference Types

```javascript
// Object
let obj = { name: 'John', age: 30 };

// Array
let arr = [1, 2, 3, 4, 5];

// Function
let func = function() { return 'hello'; };
```

### Type Checking

```javascript
typeof 'hello'      // "string"
typeof 42           // "number"
typeof true         // "boolean"
typeof undefined    // "undefined"
typeof null         // "object" (historical bug)
typeof {}           // "object"
typeof []           // "object"
typeof function(){} // "function"

// Array check
Array.isArray([1, 2, 3])  // true
Array.isArray({})          // false
```

### Type Conversion

```javascript
// To string
String(123)        // "123"
(123).toString()   // "123"
123 + ''           // "123"

// To number
Number('123')      // 123
Number('abc')      // NaN
parseInt('42px')   // 42
parseFloat('3.14') // 3.14
+'123'             // 123

// To boolean
Boolean(1)         // true
Boolean(0)         // false
Boolean('')        // false
Boolean('hello')   // true
!!1                // true

// Falsy values (convert to false)
// false, 0, -0, '', null, undefined, NaN
```

---

## Operators

### Arithmetic Operators

```javascript
10 + 3   // 13 (addition)
10 - 3   // 7  (subtraction)
10 * 3   // 30 (multiplication)
10 / 3   // 3.333... (division)
10 % 3   // 1  (remainder)
10 ** 3  // 1000 (exponentiation)

// Increment/decrement
let a = 5;
a++      // Postfix: use then increment
++a      // Prefix: increment then use
a--
--a
```

### Comparison Operators

```javascript
// Equal (type coercion)
5 == '5'    // true
0 == false  // true
null == undefined  // true

// Strict equal (no type coercion) - recommended!
5 === '5'   // false
0 === false // false

// Not equal
5 != '5'    // false (type coercion)
5 !== '5'   // true  (type comparison)

// Comparison
5 > 3       // true
5 >= 5      // true
5 < 3       // false
5 <= 5      // true
```

### Logical Operators

```javascript
// AND (both true)
true && true    // true
true && false   // false

// OR (at least one true)
true || false   // true
false || false  // false

// NOT (negation)
!true           // false
!false          // true

// Short-circuit evaluation
const name = user && user.name;  // undefined if user doesn't exist
const value = input || 'default';  // 'default' if input is falsy
```

### Nullish Coalescing

```javascript
// ?? returns right side ONLY for null/undefined; use instead of || when 0, "", or false are valid values
null ?? 'default'      // 'default'
undefined ?? 'default' // 'default'
0 ?? 'default'         // 0  ← 0 is a valid score/count, not a missing value
'' ?? 'default'        // '' ← empty string is valid input, not absence of value

// Compare with ||
0 || 'default'         // 'default' (0 is falsy — wrong if 0 is a valid value!)
'' || 'default'        // 'default' ('' is falsy — wrong if empty string is intentional)
```

### Assignment Operators

```javascript
let x = 10;
x += 5;   // x = x + 5  → 15
x -= 3;   // x = x - 3  → 12
x *= 2;   // x = x * 2  → 24
x /= 4;   // x = x / 4  → 6
x %= 4;   // x = x % 4  → 2
x **= 2;  // x = x ** 2 → 4

// Logical assignment (ES2021)
x ||= 10;  // x = x || 10
x &&= 5;   // x = x && 5
x ??= 0;   // x = x ?? 0
```

### Ternary Operator

```javascript
const result = condition ? ifTrue : ifFalse;

const age = 20;
const status = age >= 18 ? 'Adult' : 'Minor';

// Nesting (watch readability)
const grade = score >= 90 ? 'A'
            : score >= 80 ? 'B'
            : score >= 70 ? 'C'
            : 'F';
```

---

## Conditionals

### if...else

```javascript
const age = 20;

if (age >= 18) {
    console.log('Adult');
} else if (age >= 13) {
    console.log('Teenager');
} else {
    console.log('Child');
}
```

### switch

```javascript
const day = 'Monday';

switch (day) {
    case 'Monday':
    case 'Tuesday':
    case 'Wednesday':
    case 'Thursday':
    case 'Friday':
        console.log('Weekday');
        break;
    case 'Saturday':
    case 'Sunday':
        console.log('Weekend');
        break;
    default:
        console.log('Unknown');
}
```

### Conditional Execution

```javascript
// condition && statement
isLoggedIn && showDashboard();

// condition || statement
data || fetchData();
```

---

## Loops

### for

```javascript
for (let i = 0; i < 5; i++) {
    console.log(i);  // 0, 1, 2, 3, 4
}

// Reverse
for (let i = 4; i >= 0; i--) {
    console.log(i);  // 4, 3, 2, 1, 0
}
```

### for...of (Arrays)

```javascript
const fruits = ['apple', 'banana', 'orange'];

for (const fruit of fruits) {
    console.log(fruit);
}
```

### for...in (Objects)

```javascript
const user = { name: 'John', age: 30 };

for (const key in user) {
    console.log(key, user[key]);
    // name John
    // age 30
}
```

### while

```javascript
let count = 0;

while (count < 5) {
    console.log(count);
    count++;
}
```

### do...while

```javascript
let num = 0;

do {
    console.log(num);  // Executes at least once
    num++;
} while (num < 5);
```

### break and continue

```javascript
// break: Exit loop
for (let i = 0; i < 10; i++) {
    if (i === 5) break;
    console.log(i);  // 0, 1, 2, 3, 4
}

// continue: Skip current iteration
for (let i = 0; i < 5; i++) {
    if (i === 2) continue;
    console.log(i);  // 0, 1, 3, 4
}
```

---

## Functions

### Function Declaration

```javascript
function greet(name) {
    return `Hello, ${name}!`;
}

greet('John');  // "Hello, John!"
```

### Function Expression

```javascript
const greet = function(name) {
    return `Hello, ${name}!`;
};

greet('John');
```

### Arrow Function

```javascript
// Arrow functions don't bind their own 'this' — use regular functions for object methods that reference 'this'
// Basic form
const greet = (name) => {
    return `Hello, ${name}!`;
};

// One parameter: parentheses optional
const greet = name => {
    return `Hello, ${name}!`;
};

// One line: braces and return optional
const greet = name => `Hello, ${name}!`;

// No parameters
const sayHello = () => 'Hello!';

// Object return (parentheses required)
const createUser = name => ({ name, created: Date.now() });
```

### Parameters

```javascript
// Default values
function greet(name = 'Guest') {
    return `Hello, ${name}!`;
}
greet();        // "Hello, Guest!"
greet('John');  // "Hello, John!"

// Rest parameters
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}
sum(1, 2, 3, 4);  // 10

// Destructuring — avoids repeated person.name, person.age access; creates local const bindings
function printUser({ name, age }) {
    console.log(`${name} is ${age} years old`);
}
printUser({ name: 'John', age: 30 });
```

### Callback Functions

```javascript
function processData(data, callback) {
    // Process data
    const result = data.toUpperCase();
    callback(result);
}

processData('hello', function(result) {
    console.log(result);  // "HELLO"
});

// With arrow function
processData('hello', result => console.log(result));
```

### IIFE (Immediately Invoked Function Expression)

```javascript
(function() {
    const private = 'Not accessible outside';
    console.log('Immediately executed');
})();

// Arrow function
(() => {
    console.log('Immediately executed');
})();
```

---

## Arrays

### Array Creation

```javascript
// Literal (recommended)
const arr1 = [1, 2, 3];

// Constructor
const arr2 = new Array(3);     // [empty × 3]
const arr3 = Array.of(1, 2, 3); // [1, 2, 3]
const arr4 = Array.from('abc'); // ['a', 'b', 'c']
```

### Array Access

```javascript
const fruits = ['apple', 'banana', 'orange'];

fruits[0]          // 'apple'
fruits[2]          // 'orange'
fruits.length      // 3
fruits.at(-1)      // 'orange' (last element)
fruits.at(-2)      // 'banana'
```

### Array Modification

```javascript
const arr = [1, 2, 3];

// Add
arr.push(4);         // Add to end: [1, 2, 3, 4]
arr.unshift(0);      // Add to beginning: [0, 1, 2, 3, 4]

// Remove
arr.pop();           // Remove from end: [0, 1, 2, 3]
arr.shift();         // Remove from beginning: [1, 2, 3]

// Modify at specific position
arr.splice(1, 1);           // Remove 1 element at index 1: [1, 3]
arr.splice(1, 0, 'a', 'b'); // Insert at index 1: [1, 'a', 'b', 3]
arr.splice(1, 1, 'x');      // Replace at index 1: [1, 'x', 'b', 3]
```

### Array Search

```javascript
const arr = [1, 2, 3, 2, 1];

arr.indexOf(2)        // 1 (first index)
arr.lastIndexOf(2)    // 3 (last index)
arr.includes(2)       // true (contains)

// Search with condition
const users = [
    { id: 1, name: 'Kim' },
    { id: 2, name: 'Lee' }
];

users.find(u => u.id === 1);       // { id: 1, name: 'Kim' }
users.findIndex(u => u.id === 1);  // 0
```

### Array Iteration Methods

```javascript
const numbers = [1, 2, 3, 4, 5];

// forEach: Execute for each element (no return value)
numbers.forEach((num, index) => {
    console.log(index, num);
});

// map: Return new transformed array
const doubled = numbers.map(n => n * 2);
// [2, 4, 6, 8, 10]

// filter: Elements matching condition
const evens = numbers.filter(n => n % 2 === 0);
// [2, 4]

// reduce: acc = running total (starts at initial value 0), curr = current array element
const sum = numbers.reduce((acc, cur) => acc + cur, 0);
// 15

// every: All elements meet condition?
numbers.every(n => n > 0);  // true

// some: At least one element meets condition?
numbers.some(n => n > 4);   // true
```

### Array Sorting

```javascript
const arr = [3, 1, 4, 1, 5];

// Ascending
arr.sort((a, b) => a - b);  // [1, 1, 3, 4, 5]

// Descending
arr.sort((a, b) => b - a);  // [5, 4, 3, 1, 1]

// String sorting
const names = ['John', 'Alice', 'Bob'];
names.sort();               // ['Alice', 'Bob', 'John']

// Reverse
arr.reverse();  // [1, 1, 4, 3, 5]
```

### Array Transformation

```javascript
const arr = [1, 2, 3];

// Copy
const copy = [...arr];        // Spread
const copy2 = arr.slice();    // slice

// Concatenate
const merged = [...arr, ...[4, 5]];  // [1, 2, 3, 4, 5]
const merged2 = arr.concat([4, 5]);

// Flatten
const nested = [1, [2, [3, 4]]];
nested.flat();     // [1, 2, [3, 4]]
nested.flat(2);    // [1, 2, 3, 4]
nested.flat(Infinity);  // Complete flattening

// To string
[1, 2, 3].join('-');  // "1-2-3"
```

---

## Objects

### Object Creation

```javascript
// Literal (recommended)
const user = {
    name: 'John',
    age: 30,
    email: 'john@example.com'
};

// Constructor
const obj = new Object();
obj.name = 'John';
```

### Object Access

```javascript
const user = { name: 'John', age: 30 };

// Dot notation
user.name       // 'John'
user.age        // 30

// Bracket notation (dynamic keys)
user['name']    // 'John'

const key = 'age';
user[key]       // 30
```

### Object Modification

```javascript
const user = { name: 'John' };

// Add/modify
user.age = 30;
user['email'] = 'john@example.com';

// Delete
delete user.email;

// Check property existence
'name' in user           // true
user.hasOwnProperty('name')  // true
```

### Object Iteration

```javascript
const user = { name: 'John', age: 30 };

// for...in
for (const key in user) {
    console.log(key, user[key]);
}

// Object methods
Object.keys(user)    // ['name', 'age']
Object.values(user)  // ['John', 30]
Object.entries(user) // [['name', 'John'], ['age', 30]]

// Iterate with entries
for (const [key, value] of Object.entries(user)) {
    console.log(key, value);
}
```

### Object Copy/Merge

```javascript
const user = { name: 'John', age: 30 };

// Shallow copy
const copy1 = { ...user };           // Spread
const copy2 = Object.assign({}, user);

// Merge
const merged = { ...user, city: 'Seoul' };
const merged2 = Object.assign({}, user, { city: 'Seoul' });

// Deep copy (nested objects)
const deep = JSON.parse(JSON.stringify(user));
const deep2 = structuredClone(user);  // Modern browsers
```

### Property Shorthand

```javascript
const name = 'John';
const age = 30;

// Traditional way
const user1 = { name: name, age: age };

// Shorthand
const user2 = { name, age };
```

### Computed Property Names

```javascript
const key = 'email';
const user = {
    name: 'John',
    [key]: 'john@example.com',
    ['get' + 'Age']() { return 30; }
};

user.email    // 'john@example.com'
user.getAge() // 30
```

### Methods

```javascript
const user = {
    name: 'John',

    // Method
    greet() {
        return `Hello, ${this.name}!`;
    },

    // Arrow function (watch out for this!)
    // this refers to parent scope
    badGreet: () => {
        return `Hello, ${this.name}!`;  // this is not user!
    }
};

user.greet();  // "Hello, John!"
```

---

## ES6+ Syntax

### Template Literals

```javascript
const name = 'John';
const age = 30;

// Traditional way
const msg1 = 'Name: ' + name + ', Age: ' + age;

// Template literal
const msg2 = `Name: ${name}, Age: ${age}`;

// Multi-line
const html = `
    <div>
        <h1>${name}</h1>
        <p>${age} years old</p>
    </div>
`;

// Expressions
const result = `Sum: ${10 + 20}`;
const status = `Status: ${age >= 18 ? 'Adult' : 'Minor'}`;
```

### Destructuring Assignment

```javascript
// Array destructuring
const [a, b, c] = [1, 2, 3];
const [first, , third] = [1, 2, 3];  // Skip
const [x, ...rest] = [1, 2, 3, 4];   // Rest

// Object destructuring
const { name, age } = { name: 'John', age: 30 };

// Default values
const { name, city = 'Seoul' } = { name: 'John' };

// Rename
const { name: userName, age: userAge } = user;

// Nested
const { address: { city } } = {
    address: { city: 'Seoul' }
};

// Function parameters
function greet({ name, age = 0 }) {
    console.log(`${name}, ${age} years old`);
}
```

### Spread Operator

```javascript
// Array
const arr1 = [1, 2, 3];
const arr2 = [...arr1, 4, 5];  // [1, 2, 3, 4, 5]

// Object
const obj1 = { a: 1, b: 2 };
const obj2 = { ...obj1, c: 3 };  // { a: 1, b: 2, c: 3 }

// Function call
const numbers = [1, 2, 3];
Math.max(...numbers);  // 3

// String → Array
[...'hello']  // ['h', 'e', 'l', 'l', 'o']
```

### Optional Chaining (?.)

```javascript
const user = {
    name: 'John',
    address: {
        city: 'Seoul'
    }
};

// Traditional way
const city1 = user && user.address && user.address.city;

// Optional Chaining
const city2 = user?.address?.city;

// Array
const first = arr?.[0];

// Function
const result = obj.method?.();
```

### Nullish Coalescing (??)

```javascript
// ?? returns right side ONLY for null/undefined — use instead of || when 0, "", or false are valid values
const value1 = null ?? 'default';     // 'default'
const value2 = undefined ?? 'default'; // 'default'
const value3 = 0 ?? 'default';        // 0    ← preserves 0 (valid score/count)
const value4 = '' ?? 'default';       // ''   ← preserves empty string (valid input)
const value5 = false ?? 'default';    // false ← preserves false (valid boolean flag)
```

---

## Practice Problems

### Problem 1: Variables and Conditionals

Write a function that categorizes age:
- 0-12: "Child"
- 13-19: "Teenager"
- 20-64: "Adult"
- 65+: "Senior"

<details>
<summary>Show Answer</summary>

```javascript
function getAgeGroup(age) {
    if (age < 0) return 'Invalid age';
    if (age <= 12) return 'Child';
    if (age <= 19) return 'Teenager';
    if (age <= 64) return 'Adult';
    return 'Senior';
}
```

</details>

### Problem 2: Array Methods

Filter even numbers and square them.

```javascript
// Input: [1, 2, 3, 4, 5, 6]
// Output: [4, 16, 36]
```

<details>
<summary>Show Answer</summary>

```javascript
const numbers = [1, 2, 3, 4, 5, 6];

const result = numbers
    .filter(n => n % 2 === 0)
    .map(n => n ** 2);

console.log(result);  // [4, 16, 36]
```

</details>

### Problem 3: Working with Objects

Find users matching specific conditions and output their information.

```javascript
const users = [
    { id: 1, name: 'Kim', age: 25 },
    { id: 2, name: 'Lee', age: 30 },
    { id: 3, name: 'Park', age: 28 }
];
// Return array of names for users aged 28 or older
```

<details>
<summary>Show Answer</summary>

```javascript
const result = users
    .filter(user => user.age >= 28)
    .map(user => user.name);

console.log(result);  // ['Lee', 'Park']
```

</details>

---

---

## References

- [MDN JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
- [JavaScript.info](https://javascript.info/)
- [ECMAScript Specification](https://tc39.es/ecma262/)

---

**Previous**: [CSS Responsive Design](./05_CSS_Responsive.md) | **Next**: [JavaScript Events and DOM](./07_JS_Events_DOM.md)
