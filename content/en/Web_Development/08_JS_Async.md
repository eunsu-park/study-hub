# JavaScript Asynchronous Programming

**Previous**: [JavaScript Events and DOM](./07_JS_Events_DOM.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between synchronous and asynchronous execution and describe the JavaScript event loop
2. Identify callback hell and explain why deeply nested callbacks are problematic
3. Create and consume Promises using the resolve/reject constructor, then/catch/finally chains, and static methods (Promise.all, Promise.race, Promise.allSettled, Promise.any)
4. Write asynchronous functions using async/await syntax with proper try/catch error handling
5. Compare sequential and parallel async execution and choose the appropriate pattern
6. Implement HTTP requests using the Fetch API with various methods (GET, POST, PUT, DELETE), headers, and request bodies
7. Apply practical async patterns including retry logic, request cancellation with AbortController, debounce, and throttle

---

Almost everything interesting a web application does involves waiting: waiting for a server response, waiting for a file to upload, waiting for a database query. If JavaScript blocked execution during every wait, the browser would freeze and the user interface would become unresponsive. Asynchronous programming is the technique that allows JavaScript to start a long-running operation, move on to other work, and come back when the result is ready -- keeping the application snappy and the user happy.

## Table of Contents

1. [Synchronous vs Asynchronous](#synchronous-vs-asynchronous)
2. [Callbacks](#callbacks)
3. [Promise](#promise)
4. [async/await](#asyncawait)
5. [Fetch API](#fetch-api)
6. [Error Handling](#error-handling)
7. [Practical Patterns](#practical-patterns)

---

## Synchronous vs Asynchronous

### Synchronous

Code executes in order. Next task starts only after previous task completes.

```javascript
console.log('1');
console.log('2');
console.log('3');
// Output: 1, 2, 3
```

### Asynchronous

Executes next code without waiting for task completion.

```javascript
console.log('1');
setTimeout(() => {
    console.log('2');
}, 1000);
console.log('3');
// Output: 1, 3, 2 (2 after 1 second)
```

### When Asynchronous is Needed

- Server requests (API calls)
- File reading/writing
- Timers (setTimeout, setInterval)
- Event listeners
- Database queries

### Event Loop

```
┌─────────────────────────────────────────┐
│                                         │
│          Call Stack                     │
│                                         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│                                         │
│          Web APIs                       │
│   (setTimeout, fetch, DOM events)       │
│                                         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│                                         │
│     Callback Queue / Task Queue         │
│                                         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
            Event Loop watches
                    │
          When Call Stack is empty
                    │
          Callback to Stack
```

---

## Callbacks

> **Analogy:** A synchronous waiter would take one table's order, stand at the kitchen window waiting for the food, deliver it, and only then visit the next table. An async waiter takes an order, moves on to the next table while the kitchen cooks, and comes back when the food is ready. Promises are the order tickets the kitchen uses to signal completion.

### What is a Callback Function?

A function passed as an argument to another function and executed later.

```javascript
function greet(name, callback) {
    console.log(`Hello, ${name}!`);
    callback();
}

greet('John', function() {
    console.log('Greeting complete');
});
```

### setTimeout / setInterval

```javascript
// Execute after delay
setTimeout(() => {
    console.log('Execute after 3 seconds');
}, 3000);

// Repeat execution
const intervalId = setInterval(() => {
    console.log('Execute every second');
}, 1000);

// Cancel timer
clearTimeout(timeoutId);
clearInterval(intervalId);
```

### Callback Hell

Problem where code becomes complex with nested callbacks

```javascript
// Bad example: Callback hell
getUser(userId, function(user) {
    getOrders(user.id, function(orders) {
        getOrderDetails(orders[0].id, function(details) {
            getProductInfo(details.productId, function(product) {
                console.log(product);
                // More nesting...
            });
        });
    });
});
```

---

## Promise

### What is Promise?

An object representing the eventual completion (or failure) of an asynchronous operation and its resulting value.

```
Promise States:
┌──────────┐     ┌───────────┐
│ pending  │────▶│ fulfilled │  (success)
│ (waiting)│     └───────────┘
│          │     ┌───────────┐
│          │────▶│ rejected  │  (failure)
└──────────┘     └───────────┘
```

### Creating Promise

```javascript
const promise = new Promise((resolve, reject) => {
    // Asynchronous operation
    const success = true;

    if (success) {
        resolve('Success!');  // On success
    } else {
        reject('Failure!');   // On failure
    }
});
```

### then / catch / finally

```javascript
promise
    .then(result => {
        console.log(result);  // Execute on success
        return 'Next value';  // Chainable
    })
    .then(nextResult => {
        console.log(nextResult);
    })
    .catch(error => {
        console.error(error);  // Execute on failure
    })
    .finally(() => {
        console.log('Always executes');  // Success or failure
    });
```

### Promise Chaining

```javascript
fetchUser(userId)
    .then(user => fetchOrders(user.id))
    .then(orders => fetchOrderDetails(orders[0].id))
    .then(details => fetchProductInfo(details.productId))
    .then(product => console.log(product))
    .catch(error => console.error(error));
```

### Promise Static Methods

```javascript
// Immediately resolved Promise
Promise.resolve('value');

// Immediately rejected Promise
Promise.reject('error');

// Promise.all — use when ALL requests must succeed; fails fast if any one rejects (e.g., loading all required dashboard data)
Promise.all([promise1, promise2, promise3])
    .then(results => {
        // results = [result1, result2, result3]
    })
    .catch(error => {
        // First failure — entire operation fails; good when partial results are useless
    });

// Promise.allSettled — use when you need ALL results regardless of success/failure (e.g., batch operations where partial success is acceptable)
Promise.allSettled([promise1, promise2, promise3])
    .then(results => {
        results.forEach(result => {
            if (result.status === 'fulfilled') {
                console.log('Success:', result.value);
            } else {
                console.log('Failure:', result.reason);
            }
        });
    });

// Promise.race — use for timeout patterns: whichever finishes first wins, even if it rejects
Promise.race([promise1, promise2, promise3])
    .then(result => {
        // Fastest result — useful for "fetch from nearest server" or implementing custom timeouts
    });

// Promise.any — use when you need the first success and don't care about failures (e.g., try multiple CDN mirrors)
Promise.any([promise1, promise2, promise3])
    .then(result => {
        // First successful result — unlike race, ignores rejections until ALL fail
    })
    .catch(error => {
        // AggregateError: all promises rejected
    });
```

### Promise Practical Examples

```javascript
// Promisify timer
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

delay(2000).then(() => console.log('2 seconds passed'));

// Convert callback API to Promise
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Image load failed: ${src}`));
        img.src = src;
    });
}

loadImage('image.jpg')
    .then(img => document.body.appendChild(img))
    .catch(error => console.error(error));
```

---

## async/await

### Basic Usage

async function always returns a Promise. await waits for Promise to resolve.

```javascript
// async function always returns a Promise; await unwraps it — syntactic sugar over .then() chains
// Benefit: error handling with try/catch instead of nested .catch(), and code reads top-to-bottom
async function fetchData() {
    const response = await fetch('/api/data');
    const data = await response.json();
    return data;
}

// Call
fetchData().then(data => console.log(data));
```

### Error Handling

```javascript
async function fetchData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error occurred:', error);
        throw error;  // Re-throw error
    }
}
```

### Sequential vs Parallel Execution

```javascript
// Sequential execution (slow)
async function sequential() {
    const user = await fetchUser();      // 1 second
    const orders = await fetchOrders();  // 1 second
    const products = await fetchProducts();  // 1 second
    // Total 3 seconds
}

// Parallel execution (fast)
async function parallel() {
    const [user, orders, products] = await Promise.all([
        fetchUser(),
        fetchOrders(),
        fetchProducts()
    ]);
    // Total 1 second (slowest operation)
}
```

### async/await in Loops

```javascript
// Sequential processing
async function processSequential(items) {
    for (const item of items) {
        await processItem(item);  // One by one in order
    }
}

// Parallel processing
async function processParallel(items) {
    await Promise.all(items.map(item => processItem(item)));
}

// forEach doesn't wait for await (caution!)
items.forEach(async item => {
    await processItem(item);  // All start simultaneously
});
```

### Top-level await (ES2022)

Can use await at top level in modules

```javascript
// module.js
const response = await fetch('/api/config');
export const config = await response.json();
```

---

## Fetch API

### Basic Usage

```javascript
// GET request
fetch('https://api.example.com/data')
    .then(res => res.json())  // res.json() returns another Promise — the body stream must be fully read before parsing; this is the first async step (header arrival), the next .then gets the parsed data
    .then(data => console.log(data))
    .catch(error => console.error(error));

// async/await
async function getData() {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    return data;
}
```

### HTTP Methods

```javascript
// GET (default)
fetch('/api/users');

// POST
fetch('/api/users', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        name: 'John',
        email: 'john@example.com'
    })
});

// PUT
fetch('/api/users/1', {
    method: 'PUT',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        name: 'Jane'
    })
});

// PATCH
fetch('/api/users/1', {
    method: 'PATCH',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        name: 'Alice'
    })
});

// DELETE
fetch('/api/users/1', {
    method: 'DELETE'
});
```

### Response Object

```javascript
const response = await fetch('/api/data');

// Response information
response.ok;         // Success status (200-299)
response.status;     // HTTP status code
response.statusText; // Status message
response.headers;    // Response headers
response.url;        // Request URL

// Read body (can only be read once)
await response.json();   // Parse JSON
await response.text();   // Text
await response.blob();   // Blob (binary)
await response.arrayBuffer();  // ArrayBuffer
await response.formData();     // FormData
```

### Setting Headers

```javascript
fetch('/api/data', {
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
        'X-Custom-Header': 'value'
    }
});

// Headers object
const headers = new Headers();
headers.append('Content-Type', 'application/json');
headers.append('Authorization', 'Bearer ' + token);

fetch('/api/data', { headers });
```

### Query Parameters

```javascript
// Directly in URL
fetch('/api/users?page=1&limit=10');

// Using URLSearchParams
const params = new URLSearchParams({
    page: 1,
    limit: 10,
    search: 'John'
});

fetch(`/api/users?${params}`);

// URL object
const url = new URL('/api/users', 'https://api.example.com');
url.searchParams.set('page', 1);
url.searchParams.set('limit', 10);

fetch(url);
```

### FormData Submission

```javascript
// From HTML form
const form = document.querySelector('form');
const formData = new FormData(form);

fetch('/api/upload', {
    method: 'POST',
    body: formData  // Content-Type set automatically
});

// Create directly
const formData = new FormData();
formData.append('username', 'John');
formData.append('file', fileInput.files[0]);

fetch('/api/upload', {
    method: 'POST',
    body: formData
});
```

### File Upload

```javascript
const fileInput = document.querySelector('input[type="file"]');

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();
        console.log('Upload complete:', result);
    } catch (error) {
        console.error(error);
    }
});
```

### Request Cancellation (AbortController)

```javascript
const controller = new AbortController();
const signal = controller.signal;

// Auto-cancel after 5 seconds
const timeoutId = setTimeout(() => controller.abort(), 5000);

try {
    const response = await fetch('/api/data', { signal });
    clearTimeout(timeoutId);
    const data = await response.json();
} catch (error) {
    if (error.name === 'AbortError') {
        console.log('Request was cancelled');
    } else {
        throw error;
    }
}

// Rejects the fetch Promise with an AbortError — useful for cancelling requests when user navigates away or a timeout expires
controller.abort();
```

### Timeout Utility

```javascript
async function fetchWithTimeout(url, options = {}, timeout = 5000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        return response;
    } finally {
        clearTimeout(timeoutId);
    }
}

// Usage
const response = await fetchWithTimeout('/api/data', {}, 3000);
```

---

## Error Handling

### fetch only rejects on network errors

```javascript
// Caution: 404, 500 etc are not errors!
fetch('/api/not-found')
    .then(response => {
        // Comes here even for 404
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .catch(error => {
        // Network error or error thrown above
        console.error(error);
    });
```

### Custom Error Class

```javascript
class HttpError extends Error {
    constructor(response) {
        super(`HTTP error! status: ${response.status}`);
        this.name = 'HttpError';
        this.response = response;
    }
}

async function fetchJSON(url, options) {
    const response = await fetch(url, options);

    if (!response.ok) {
        throw new HttpError(response);
    }

    return response.json();
}

// Usage
try {
    const data = await fetchJSON('/api/data');
} catch (error) {
    if (error instanceof HttpError) {
        if (error.response.status === 404) {
            console.log('Data not found');
        } else if (error.response.status === 401) {
            console.log('Authentication required');
        }
    } else {
        console.error('Network error:', error);
    }
}
```

### Retry Logic

```javascript
async function fetchWithRetry(url, options = {}, retries = 3, delay = 1000) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response;
        } catch (error) {
            if (i === retries - 1) throw error;

            console.log(`Retry ${i + 1}/${retries}...`);
            await new Promise(r => setTimeout(r, delay * (i + 1)));
        }
    }
}
```

---

## Practical Patterns

### API Client

```javascript
class ApiClient {
    constructor(baseURL) {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || `HTTP ${response.status}`);
        }

        return response.json();
    }

    get(endpoint) {
        return this.request(endpoint);
    }

    post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: data
        });
    }

    put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: data
        });
    }

    delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }
}

// Usage
const api = new ApiClient('https://api.example.com');

const users = await api.get('/users');
const newUser = await api.post('/users', { name: 'John' });
await api.delete('/users/1');
```

### Loading State Management

```javascript
class LoadingState {
    constructor() {
        this.isLoading = false;
        this.error = null;
        this.data = null;
    }

    async execute(asyncFn) {
        this.isLoading = true;
        this.error = null;

        try {
            this.data = await asyncFn();
            return this.data;
        } catch (error) {
            this.error = error;
            throw error;
        } finally {
            this.isLoading = false;
        }
    }
}

// Usage
const state = new LoadingState();

// Update UI
function updateUI() {
    if (state.isLoading) {
        showSpinner();
    } else if (state.error) {
        showError(state.error.message);
    } else {
        renderData(state.data);
    }
}

await state.execute(() => fetch('/api/data').then(r => r.json()));
updateUI();
```

### Debounce and Throttle

```javascript
// Debounce: Execute after delay from last call
function debounce(fn, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

// Search autocomplete
const searchInput = document.querySelector('#search');
searchInput.addEventListener('input', debounce(async (e) => {
    const results = await fetch(`/api/search?q=${e.target.value}`);
    renderResults(await results.json());
}, 300));

// Throttle: Execute once per time period
function throttle(fn, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Scroll event
window.addEventListener('scroll', throttle(() => {
    console.log('Scroll position:', window.scrollY);
}, 100));
```

### Infinite Scroll

```javascript
class InfiniteScroll {
    constructor(container, loadMore) {
        this.container = container;
        this.loadMore = loadMore;
        this.page = 1;
        this.loading = false;
        this.hasMore = true;

        this.setupObserver();
    }

    setupObserver() {
        const sentinel = document.createElement('div');
        sentinel.className = 'sentinel';
        this.container.appendChild(sentinel);

        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && !this.loading && this.hasMore) {
                this.load();
            }
        });

        observer.observe(sentinel);
    }

    async load() {
        this.loading = true;

        try {
            const items = await this.loadMore(this.page);

            if (items.length === 0) {
                this.hasMore = false;
                return;
            }

            items.forEach(item => {
                const el = this.createItem(item);
                this.container.insertBefore(el, this.container.lastChild);
            });

            this.page++;
        } finally {
            this.loading = false;
        }
    }

    createItem(item) {
        const div = document.createElement('div');
        div.className = 'item';
        div.textContent = item.title;
        return div;
    }
}

// Usage
new InfiniteScroll(
    document.querySelector('#list'),
    async (page) => {
        const response = await fetch(`/api/items?page=${page}`);
        return response.json();
    }
);
```

### Caching

```javascript
class ApiCache {
    constructor(ttl = 60000) {  // Default 1 minute
        this.cache = new Map();
        this.ttl = ttl;
    }

    async fetch(url, options) {
        const key = JSON.stringify({ url, options });
        const cached = this.cache.get(key);

        if (cached && Date.now() < cached.expiry) {
            return cached.data;
        }

        const response = await fetch(url, options);
        const data = await response.json();

        this.cache.set(key, {
            data,
            expiry: Date.now() + this.ttl
        });

        return data;
    }

    clear() {
        this.cache.clear();
    }
}

const cachedApi = new ApiCache(30000);  // 30 second cache
const data = await cachedApi.fetch('/api/data');
```

---

## Practice Problems

### Problem 1: Promise Chaining

Call 3 APIs sequentially and combine results.

```javascript
// /api/user/1 → { name: 'John' }
// /api/user/1/posts → [{ id: 1, title: 'Post1' }]
// /api/posts/1/comments → [{ id: 1, text: 'Comment1' }]
```

<details>
<summary>Show Answer</summary>

```javascript
async function getUserData(userId) {
    const user = await fetch(`/api/user/${userId}`).then(r => r.json());
    const posts = await fetch(`/api/user/${userId}/posts`).then(r => r.json());
    const comments = await fetch(`/api/posts/${posts[0].id}/comments`).then(r => r.json());

    return {
        user,
        posts,
        firstPostComments: comments
    };
}
```

</details>

### Problem 2: Parallel Requests

Fetch information for multiple users simultaneously.

```javascript
const userIds = [1, 2, 3, 4, 5];
```

<details>
<summary>Show Answer</summary>

```javascript
async function fetchAllUsers(userIds) {
    const promises = userIds.map(id =>
        fetch(`/api/users/${id}`).then(r => r.json())
    );

    const users = await Promise.all(promises);
    return users;
}

// Or allow failures
async function fetchAllUsersSafe(userIds) {
    const results = await Promise.allSettled(
        userIds.map(id =>
            fetch(`/api/users/${id}`).then(r => r.json())
        )
    );

    return results
        .filter(r => r.status === 'fulfilled')
        .map(r => r.value);
}
```

</details>

### Problem 3: Form Submission

Submit form data to server and handle result.

<details>
<summary>Show Answer</summary>

```javascript
const form = document.querySelector('#userForm');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = form.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';

    try {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        const response = await fetch('/api/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Server error');
        }

        const result = await response.json();
        alert('Registration complete!');
        form.reset();
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';
    }
});
```

</details>

---

---

## References

- [MDN Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)
- [MDN async/await](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Promises)
- [MDN Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [JavaScript.info Async](https://javascript.info/async)

---

**Previous**: [JavaScript Events and DOM](./07_JS_Events_DOM.md)
