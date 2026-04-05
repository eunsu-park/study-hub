/*
 * JavaScript Basics Examples
 * Variables, data types, operators, conditionals, loops, functions, arrays, objects, classes
 */

// Output function
function log(message) {
    const output = document.getElementById('output');
    output.textContent += message + '\n';
    console.log(message);
}

function clearOutput() {
    document.getElementById('output').textContent = '';
}

// ============================================
// 1. Variables
// ============================================
function runVariables() {
    clearOutput();
    log('=== Variables (var, let, const) ===\n');

    // Why: var is function-scoped and hoisted, leading to subtle bugs in loops and closures;
    // prefer let/const which are block-scoped and caught by linters
    // var - function scope (not recommended)
    var oldVar = "I'm var";
    log(`var oldVar = "${oldVar}"`);

    // let - block scope, reassignable
    let count = 0;
    log(`let count = ${count}`);
    count = 1;
    log(`count = ${count} (reassignment allowed)`);

    // const - block scope, not reassignable
    const PI = 3.14159;
    log(`\nconst PI = ${PI}`);
    // PI = 3; // Error!

    const person = { name: "John" };
    log(`const person = { name: "${person.name}" }`);
    person.name = "Jane";
    log(`person.name = "${person.name}" (object properties can be changed)`);

    // Scope difference
    log('\n--- Scope Difference ---');
    if (true) {
        var varInBlock = "var is accessible outside the block";
        let letInBlock = "let is only accessible inside the block";
    }
    log(`var outside block: ${varInBlock}`);
    // log(letInBlock); // Error: not defined
}

// ============================================
// 2. Data Types
// ============================================
function runDataTypes() {
    clearOutput();
    log('=== Data Types ===\n');

    // Why: JS has 8 primitive types but typeof null === "object" is a well-known spec bug;
    // understanding primitives vs references prevents mutation surprises with objects/arrays
    // Primitive types
    let str = "string";
    let num = 42;
    let float = 3.14;
    let bool = true;
    let empty = null;
    let notDefined;
    let sym = Symbol("id");
    let big = 9007199254740991n;

    log('--- Primitive Types ---');
    log(`String: "${str}" (typeof: ${typeof str})`);
    log(`Number (integer): ${num} (typeof: ${typeof num})`);
    log(`Number (float): ${float} (typeof: ${typeof float})`);
    log(`Boolean: ${bool} (typeof: ${typeof bool})`);
    log(`Null: ${empty} (typeof: ${typeof empty})`); // Note: shows as object
    log(`Undefined: ${notDefined} (typeof: ${typeof notDefined})`);
    log(`Symbol: ${String(sym)} (typeof: ${typeof sym})`);
    log(`BigInt: ${big} (typeof: ${typeof big})`);

    // Reference types
    log('\n--- Reference Types ---');
    let obj = { key: "value" };
    let arr = [1, 2, 3];
    let func = function() {};

    log(`Object: ${JSON.stringify(obj)} (typeof: ${typeof obj})`);
    log(`Array: [${arr}] (typeof: ${typeof arr}, Array.isArray: ${Array.isArray(arr)})`);
    log(`Function: ${func.toString()} (typeof: ${typeof func})`);

    // Type conversion
    log('\n--- Type Conversion ---');
    log(`String(123) = "${String(123)}"`);
    log(`Number("42") = ${Number("42")}`);
    log(`Boolean(0) = ${Boolean(0)}`);
    log(`Boolean("") = ${Boolean("")}`);
    log(`Boolean("hello") = ${Boolean("hello")}`);
    log(`parseInt("42px") = ${parseInt("42px")}`);
    log(`parseFloat("3.14abc") = ${parseFloat("3.14abc")}`);
}

// ============================================
// 3. Operators
// ============================================
function runOperators() {
    clearOutput();
    log('=== Operators ===\n');

    let a = 10, b = 3;

    // Arithmetic operators
    log('--- Arithmetic Operators ---');
    log(`${a} + ${b} = ${a + b}`);
    log(`${a} - ${b} = ${a - b}`);
    log(`${a} * ${b} = ${a * b}`);
    log(`${a} / ${b} = ${a / b}`);
    log(`${a} % ${b} = ${a % b} (remainder)`);
    log(`${a} ** ${b} = ${a ** b} (exponentiation)`);

    // Comparison operators
    log('\n--- Comparison Operators ---');
    // Why: Always prefer === over == to avoid implicit coercion surprises (e.g., 0 == "" is true)
    log(`5 == "5" : ${5 == "5"} (comparison after type coercion)`);
    log(`5 === "5" : ${5 === "5"} (strict comparison including type)`);
    log(`5 != "5" : ${5 != "5"}`);
    log(`5 !== "5" : ${5 !== "5"}`);

    // Logical operators
    log('\n--- Logical Operators ---');
    log(`true && false = ${true && false}`);
    log(`true || false = ${true || false}`);
    log(`!true = ${!true}`);

    // Ternary operator
    log('\n--- Ternary Operator ---');
    let age = 20;
    let status = age >= 18 ? "adult" : "minor";
    log(`Age: ${age}, Status: ${status}`);

    // Why: ?? differs from || by only falling back on null/undefined, not on falsy values
    // like 0 or "". This prevents bugs where 0 is a valid value but || would discard it
    // Nullish coalescing (??)
    log('\n--- Nullish Coalescing (??) ---');
    let value1 = null ?? "default";
    let value2 = 0 ?? "default";
    let value3 = "" ?? "default";
    log(`null ?? "default" = "${value1}"`);
    log(`0 ?? "default" = ${value2} (0 is not null/undefined)`);
    log(`"" ?? "default" = "${value3}" (empty string is still a value)`);

    // || vs ?? difference
    log('\n--- || vs ?? ---');
    log(`0 || "default" = "${0 || "default"}" (0 is falsy)`);
    log(`0 ?? "default" = ${0 ?? "default"} (0 is not null)`);

    // Optional chaining (?.)
    log('\n--- Optional Chaining (?.) ---');
    let user = { profile: { name: "John" } };
    let emptyUser = {};
    log(`user?.profile?.name = "${user?.profile?.name}"`);
    log(`emptyUser?.profile?.name = ${emptyUser?.profile?.name}`);
}

// ============================================
// 4. Conditionals
// ============================================
function runConditions() {
    clearOutput();
    log('=== Conditionals ===\n');

    // if-else
    log('--- if-else ---');
    let score = 85;
    let grade;

    if (score >= 90) {
        grade = 'A';
    } else if (score >= 80) {
        grade = 'B';
    } else if (score >= 70) {
        grade = 'C';
    } else {
        grade = 'F';
    }
    log(`Score: ${score}, Grade: ${grade}`);

    // switch
    log('\n--- switch ---');
    let day = new Date().getDay();
    let dayName;

    switch (day) {
        case 0:
            dayName = "Sunday";
            break;
        case 1:
            dayName = "Monday";
            break;
        case 2:
            dayName = "Tuesday";
            break;
        case 3:
            dayName = "Wednesday";
            break;
        case 4:
            dayName = "Thursday";
            break;
        case 5:
            dayName = "Friday";
            break;
        case 6:
            dayName = "Saturday";
            break;
        default:
            dayName = "Unknown";
    }
    log(`Today is ${dayName}.`);

    // Truthy and Falsy
    log('\n--- Truthy and Falsy ---');
    const falsyValues = [false, 0, -0, 0n, "", null, undefined, NaN];
    log('Falsy values:');
    falsyValues.forEach(v => {
        log(`  ${String(v)} (${typeof v}) => ${Boolean(v) ? 'truthy' : 'falsy'}`);
    });

    log('\nAll other values are truthy:');
    log(`  "0" => ${Boolean("0") ? 'truthy' : 'falsy'}`);
    log(`  [] => ${Boolean([]) ? 'truthy' : 'falsy'}`);
    log(`  {} => ${Boolean({}) ? 'truthy' : 'falsy'}`);
}

// ============================================
// 5. Loops
// ============================================
function runLoops() {
    clearOutput();
    log('=== Loops ===\n');

    // for
    log('--- for loop ---');
    let sum = 0;
    for (let i = 1; i <= 5; i++) {
        sum += i;
    }
    log(`Sum from 1 to 5: ${sum}`);

    // while
    log('\n--- while loop ---');
    let count = 0;
    while (count < 3) {
        log(`count: ${count}`);
        count++;
    }

    // for...of (array iteration)
    log('\n--- for...of (array iteration) ---');
    const fruits = ["Apple", "Banana", "Orange"];
    for (const fruit of fruits) {
        log(`Fruit: ${fruit}`);
    }

    // for...in (object key iteration)
    log('\n--- for...in (object key iteration) ---');
    const person = { name: "John", age: 25, city: "Seoul" };
    for (const key in person) {
        log(`${key}: ${person[key]}`);
    }

    // forEach
    log('\n--- forEach ---');
    fruits.forEach((fruit, index) => {
        log(`[${index}] ${fruit}`);
    });

    // break and continue
    log('\n--- break and continue ---');
    log('Skip 3 with continue:');
    for (let i = 1; i <= 5; i++) {
        if (i === 3) continue;
        log(`  i = ${i}`);
    }

    log('Stop at 3 with break:');
    for (let i = 1; i <= 5; i++) {
        if (i === 3) break;
        log(`  i = ${i}`);
    }
}

// ============================================
// 6. Functions
// ============================================
function runFunctions() {
    clearOutput();
    log('=== Functions ===\n');

    // Function declaration
    log('--- Function Declaration ---');
    function greet1(name) {
        return `Hello, ${name}!`;
    }
    log(greet1("John"));

    // Function expression
    log('\n--- Function Expression ---');
    const greet2 = function(name) {
        return `Hi, ${name}!`;
    };
    log(greet2("Jane"));

    // Why: Arrow functions lexically bind `this`, making them safer for callbacks where
    // traditional functions would lose the enclosing context
    // Arrow function
    log('\n--- Arrow Function ---');
    const greet3 = (name) => `Hey, ${name}!`;
    log(greet3("Alice"));

    // Default parameters
    log('\n--- Default Parameters ---');
    const greet4 = (name = "Guest") => `Welcome, ${name}!`;
    log(greet4());
    log(greet4("Bob"));

    // Rest parameters
    log('\n--- Rest Parameters ---');
    const sumAll = (...numbers) => {
        return numbers.reduce((acc, num) => acc + num, 0);
    };
    log(`sumAll(1, 2, 3, 4, 5) = ${sumAll(1, 2, 3, 4, 5)}`);

    // Destructured parameters
    log('\n--- Destructured Parameters ---');
    const printPerson = ({ name, age }) => {
        return `${name} is ${age} years old.`;
    };
    log(printPerson({ name: "John", age: 25 }));

    // Immediately Invoked Function Expression (IIFE)
    log('\n--- IIFE (Immediately Invoked Function Expression) ---');
    const result = (function(x, y) {
        return x + y;
    })(3, 4);
    log(`IIFE result: ${result}`);

    // Why: Closures capture the outer scope's variables, enabling private state that cannot be
    // accessed or tampered with from outside - the foundation of the module pattern
    // Closure
    log('\n--- Closure ---');
    function createCounter() {
        let count = 0;
        return {
            increment: () => ++count,
            decrement: () => --count,
            getCount: () => count
        };
    }
    const counter = createCounter();
    log(`count: ${counter.getCount()}`);
    log(`increment: ${counter.increment()}`);
    log(`increment: ${counter.increment()}`);
    log(`decrement: ${counter.decrement()}`);
}

// ============================================
// 7. Arrays
// ============================================
function runArrays() {
    clearOutput();
    log('=== Arrays ===\n');

    // Array creation
    log('--- Array Creation ---');
    const arr1 = [1, 2, 3];
    const arr2 = new Array(3).fill(0);
    const arr3 = Array.from({ length: 5 }, (_, i) => i + 1);
    log(`Literal: [${arr1}]`);
    log(`fill: [${arr2}]`);
    log(`Array.from: [${arr3}]`);

    // Basic methods
    log('\n--- Basic Methods ---');
    const fruits = ["Apple", "Banana"];
    log(`Original: [${fruits}]`);

    fruits.push("Orange");
    log(`push("Orange"): [${fruits}]`);

    fruits.pop();
    log(`pop(): [${fruits}]`);

    fruits.unshift("Strawberry");
    log(`unshift("Strawberry"): [${fruits}]`);

    fruits.shift();
    log(`shift(): [${fruits}]`);

    // Array manipulation
    log('\n--- Array Manipulation ---');
    const numbers = [1, 2, 3, 4, 5];
    log(`Original: [${numbers}]`);
    log(`slice(1, 3): [${numbers.slice(1, 3)}]`);
    log(`concat([6, 7]): [${numbers.concat([6, 7])}]`);
    log(`indexOf(3): ${numbers.indexOf(3)}`);
    log(`includes(3): ${numbers.includes(3)}`);
    log(`join("-"): "${numbers.join("-")}"`);
    log(`reverse(): [${[...numbers].reverse()}]`);

    // Why: Higher-order functions (map/filter/reduce) express intent declaratively, replacing
    // error-prone manual loops with composable, chainable transformations
    // Higher-order functions
    log('\n--- Higher-Order Functions ---');
    const nums = [1, 2, 3, 4, 5];
    log(`Original: [${nums}]`);
    log(`map(n => n * 2): [${nums.map(n => n * 2)}]`);
    log(`filter(n => n > 2): [${nums.filter(n => n > 2)}]`);
    log(`find(n => n > 2): ${nums.find(n => n > 2)}`);
    log(`findIndex(n => n > 2): ${nums.findIndex(n => n > 2)}`);
    log(`reduce((acc, n) => acc + n, 0): ${nums.reduce((acc, n) => acc + n, 0)}`);
    log(`every(n => n > 0): ${nums.every(n => n > 0)}`);
    log(`some(n => n > 4): ${nums.some(n => n > 4)}`);
    log(`sort((a, b) => b - a): [${[...nums].sort((a, b) => b - a)}]`);

    // Spread operator
    log('\n--- Spread Operator ---');
    const arr = [1, 2, 3];
    const newArr = [...arr, 4, 5];
    log(`[...arr, 4, 5]: [${newArr}]`);
    log(`Math.max(...arr): ${Math.max(...arr)}`);

    // Destructuring
    log('\n--- Destructuring ---');
    const [first, second, ...rest] = [1, 2, 3, 4, 5];
    log(`[first, second, ...rest] = [1, 2, 3, 4, 5]`);
    log(`first: ${first}, second: ${second}, rest: [${rest}]`);
}

// ============================================
// 8. Objects
// ============================================
function runObjects() {
    clearOutput();
    log('=== Objects ===\n');

    // Object literal
    log('--- Object Literal ---');
    const person = {
        name: "John",
        age: 25,
        city: "Seoul",
        greet() {
            return `Hello, I'm ${this.name}.`;
        }
    };
    log(`person: ${JSON.stringify(person, null, 2)}`);
    log(`person.greet(): ${person.greet()}`);

    // Property access
    log('\n--- Property Access ---');
    log(`person.name: "${person.name}"`);
    log(`person["age"]: ${person["age"]}`);

    const key = "city";
    log(`person[key] (key="city"): "${person[key]}"`);

    // Add/modify/delete properties
    log('\n--- Add/Modify/Delete Properties ---');
    person.email = "john@example.com";
    log(`Add: person.email = "${person.email}"`);

    person.age = 26;
    log(`Modify: person.age = ${person.age}`);

    delete person.city;
    log(`After delete, keys: [${Object.keys(person)}]`);

    // Destructuring assignment
    log('\n--- Destructuring Assignment ---');
    const { name, age, email = "none" } = person;
    log(`const { name, age, email = "none" } = person`);
    log(`name: "${name}", age: ${age}, email: "${email}"`);

    // Alias
    const { name: userName } = person;
    log(`const { name: userName } = person => userName: "${userName}"`);

    // Spread operator
    log('\n--- Spread Operator ---');
    const newPerson = { ...person, city: "Busan", country: "Korea" };
    log(`{ ...person, city: "Busan" }:`);
    log(JSON.stringify(newPerson, null, 2));

    // Object methods
    log('\n--- Object Methods ---');
    const obj = { a: 1, b: 2, c: 3 };
    log(`Object.keys(obj): [${Object.keys(obj)}]`);
    log(`Object.values(obj): [${Object.values(obj)}]`);
    log(`Object.entries(obj): ${JSON.stringify(Object.entries(obj))}`);

    // Object.assign
    const merged = Object.assign({}, obj, { d: 4 });
    log(`Object.assign({}, obj, { d: 4 }): ${JSON.stringify(merged)}`);

    // Check property existence
    log('\n--- Check Property Existence ---');
    log(`"a" in obj: ${"a" in obj}`);
    log(`obj.hasOwnProperty("a"): ${obj.hasOwnProperty("a")}`);
}

// ============================================
// 9. Classes
// ============================================
function runClasses() {
    clearOutput();
    log('=== Classes (ES6) ===\n');

    // Basic class
    log('--- Basic Class ---');
    class Animal {
        constructor(name) {
            this.name = name;
        }

        speak() {
            return `${this.name} makes a sound.`;
        }
    }

    const animal = new Animal("Animal");
    log(`new Animal("Animal")`);
    log(`animal.speak(): "${animal.speak()}"`);

    // Why: ES6 class syntax is syntactic sugar over prototype chains, making inheritance
    // readable while preserving JS's prototype-based delegation model under the hood
    // Inheritance
    log('\n--- Inheritance ---');
    class Dog extends Animal {
        constructor(name, breed) {
            super(name);
            this.breed = breed;
        }

        speak() {
            return `${this.name} (${this.breed}) barks!`;
        }

        fetch() {
            return `${this.name} fetches the ball.`;
        }
    }

    const dog = new Dog("Buddy", "Golden Retriever");
    log(`new Dog("Buddy", "Golden Retriever")`);
    log(`dog.speak(): "${dog.speak()}"`);
    log(`dog.fetch(): "${dog.fetch()}"`);

    // Getter and Setter
    log('\n--- Getter and Setter ---');
    class Circle {
        constructor(radius) {
            this._radius = radius;
        }

        get radius() {
            return this._radius;
        }

        set radius(value) {
            if (value > 0) {
                this._radius = value;
            }
        }

        get area() {
            return Math.PI * this._radius ** 2;
        }
    }

    const circle = new Circle(5);
    log(`circle.radius: ${circle.radius}`);
    log(`circle.area: ${circle.area.toFixed(2)}`);
    circle.radius = 10;
    log(`After circle.radius = 10, area: ${circle.area.toFixed(2)}`);

    // Static methods
    log('\n--- Static Methods ---');
    class MathUtil {
        static PI = 3.14159;

        static add(a, b) {
            return a + b;
        }

        static multiply(a, b) {
            return a * b;
        }
    }

    log(`MathUtil.PI: ${MathUtil.PI}`);
    log(`MathUtil.add(3, 4): ${MathUtil.add(3, 4)}`);
    log(`MathUtil.multiply(3, 4): ${MathUtil.multiply(3, 4)}`);

    // Why: The # private field syntax provides true encapsulation enforced by the engine at
    // runtime, unlike the _underscore convention which is merely a naming hint
    // Private fields (# syntax, ES2022)
    log('\n--- Private Fields ---');
    class BankAccount {
        #balance = 0;

        deposit(amount) {
            this.#balance += amount;
            return this.#balance;
        }

        getBalance() {
            return this.#balance;
        }
    }

    const account = new BankAccount();
    log(`account.deposit(1000): ${account.deposit(1000)}`);
    log(`account.deposit(500): ${account.deposit(500)}`);
    log(`account.getBalance(): ${account.getBalance()}`);
    // log(account.#balance); // Error: Private field
}

// Message on page load
log('Click a button to run examples for each section.');
