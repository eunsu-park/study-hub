# 06. Express Basics

**Previous**: [FastAPI Testing](./05_FastAPI_Testing.md) | **Next**: [Express Advanced](./07_Express_Advanced.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what Express is and how it fits into the Node.js ecosystem
2. Create a basic Express application that listens on a configurable port
3. Define routes for all major HTTP methods with path and query parameters
4. Describe the middleware execution chain and write custom middleware
5. Organize routes into modular Router instances for maintainable codebases

---

Express is the most widely used web framework for Node.js. While Node.js provides low-level HTTP capabilities, Express adds a thin layer of routing, middleware, and convenience methods that makes building web servers practical. It follows a minimalist philosophy -- the core is small, and functionality is added through middleware. This lesson covers the fundamentals you need to build any Express application.

## Table of Contents

1. [Node.js and Express Overview](#1-nodejs-and-express-overview)
2. [Creating an Express Application](#2-creating-an-express-application)
3. [Routing](#3-routing)
4. [Middleware Concept and Chain](#4-middleware-concept-and-chain)
5. [Built-in Middleware](#5-built-in-middleware)
6. [Request and Response Objects](#6-request-and-response-objects)
7. [Router for Modular Routes](#7-router-for-modular-routes)
8. [Practice Problems](#8-practice-problems)

---

## 1. Node.js and Express Overview

### Theory: The Node.js Event Loop

Node.js runs on a single main thread that executes JavaScript in a loop. The loop has six well-defined phases (libuv terminology), and every async callback your code queues runs in one of them. Knowing which phase your code lives in explains every "why did this run in that order?" question.

#### A.1 The six phases of libuv

In each iteration of the event loop, libuv processes phases in order. Each phase has its own callback queue:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────┐                                               │
│  │  timers  │ ← setTimeout / setInterval callbacks          │
│  └────┬─────┘                                               │
│       ↓                                                     │
│  ┌────────────────────┐                                     │
│  │ pending callbacks  │ ← deferred I/O callbacks (TCP errs) │
│  └────┬───────────────┘                                     │
│       ↓                                                     │
│  ┌──────────────────────┐                                   │
│  │ idle, prepare        │ ← internal libuv use              │
│  └────┬─────────────────┘                                   │
│       ↓                                                     │
│  ┌──────┐                                                   │
│  │ poll │ ← I/O callbacks: net, fs, child_process           │
│  └──┬───┘                                                   │
│     ↓                                                       │
│  ┌────────┐                                                 │
│  │ check  │ ← setImmediate callbacks                        │
│  └──┬─────┘                                                 │
│     ↓                                                       │
│  ┌────────────────────┐                                     │
│  │ close callbacks    │ ← socket.on('close', ...)           │
│  └────────────────────┘                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Between every two phase transitions, Node drains two extra queues:

- **`process.nextTick` queue.** Callbacks scheduled with `process.nextTick(...)`. Highest priority.
- **Microtask queue.** Resolved Promises, `await` continuations.

Both run to completion before the next phase. So `Promise.resolve().then(cb)` runs *before* a `setTimeout(cb, 0)` callback, even though both are "0 delay" — the microtask queue drains in between.

#### A.2 Why "non-blocking" matters here

Every I/O operation in Node — `fs.readFile`, `http` requests, `db.query` (when the driver is async) — registers a callback and returns immediately. The thread is freed to handle the next event. When the I/O completes, libuv places the callback in the `poll` phase queue, and the loop runs it on its next iteration.

The cost: a CPU-heavy synchronous function on the main thread blocks *every* in-flight request. `bcrypt.hashSync(password)` on a 50ms input means no other request makes any progress for 50ms. The defenses are `worker_threads`, native async APIs (`bcrypt.hash` instead of `hashSync`), or offloading to a child process.

#### A.3 Practical consequences for Express

Express handlers are just callbacks scheduled into one of these queues:

- A handler that does `res.send("hello")` runs synchronously and finishes inside the same I/O callback.
- A handler that does `await db.query(...)` suspends, lets the loop process other events, and resumes in the `poll` phase when the database response arrives.
- A handler that throws synchronously crashes the process *unless* an error-handling middleware catches it. A handler that throws inside an `async` function rejects a Promise — and unless you catch it, Node logs `UnhandledPromiseRejection` (and in newer Node versions, exits).

Every Express performance issue can be traced back to "what is blocking the loop?" or "what queue is this callback sitting in?"

### What is Node.js?

Node.js is a JavaScript runtime built on Chrome's V8 engine. It uses an **event-driven, non-blocking I/O model** that makes it efficient for network applications.

```
┌─────────────────────────────────────────────────┐
│                  Node.js Runtime                │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ V8 Engine│  │ libuv    │  │ Core Modules │  │
│  │ (JS exec)│  │ (async   │  │ (http, fs,   │  │
│  │          │  │  I/O)    │  │  path, etc.) │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### What is Express?

Express is a **minimal and flexible** web framework that provides:

| Feature | Description |
|---------|-------------|
| **Routing** | Map URL patterns to handler functions |
| **Middleware** | Composable pipeline for request processing |
| **Convenience** | Simplified API over Node's raw `http` module |
| **Ecosystem** | Thousands of middleware packages available |

> **Analogy -- Assembly Line:** Think of Express as a factory assembly line. Each request enters the line and passes through a series of stations (middleware). Each station can inspect the item, modify it, or reject it. The final station produces the response.

### Express vs Raw Node.js

```javascript
// Raw Node.js — you must parse everything yourself
import { createServer } from 'node:http';

const server = createServer((req, res) => {
  // Manual URL parsing, body parsing, header setting...
  if (req.method === 'GET' && req.url === '/api/users') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ users: [] }));
  }
});

server.listen(3000);
```

```javascript
// Express — declarative routing, built-in conveniences
import express from 'express';
const app = express();

// Express handles content-type, serialization, and status codes
app.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

app.listen(3000);
```

---

## 2. Creating an Express Application

### Project Setup

```bash
# Create project directory and initialize
mkdir express-demo && cd express-demo
npm init -y

# Install Express
npm install express

# Enable ES modules — allows import/export syntax instead of require()
# Add "type": "module" to package.json
```

### Minimal Server

```javascript
// app.js
import express from 'express';

const app = express();

// PORT from environment variable with fallback — makes the app configurable
// for different environments (dev, staging, production) without code changes
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.send('Hello, Express!');
});

// The callback confirms the server started — useful for debugging and logging
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

```bash
# Run the server
node app.js

# Or with auto-restart during development (Node.js 18.11+)
# --watch restarts on file changes — eliminates need for nodemon in most cases
node --watch app.js
```

### Application Structure

For anything beyond a trivial script, organize files by responsibility:

```
express-demo/
├── src/
│   ├── app.js           # Express app configuration
│   ├── server.js         # Server startup (listen)
│   ├── routes/           # Route definitions
│   │   └── users.js
│   └── middleware/        # Custom middleware
│       └── logger.js
├── package.json
└── .env
```

---

## 3. Routing

Routes define how the application responds to client requests at specific endpoints.

### Basic Routes

```javascript
// Each route binds an HTTP method + path to a handler function
app.get('/api/items', (req, res) => {
  res.json({ items: ['apple', 'banana'] });
});

app.post('/api/items', (req, res) => {
  // req.body contains the parsed request body (needs express.json() middleware)
  const newItem = req.body;
  res.status(201).json({ created: newItem });
});

app.put('/api/items/:id', (req, res) => {
  res.json({ updated: req.params.id });
});

app.delete('/api/items/:id', (req, res) => {
  // 204 No Content — standard response for successful delete with no body
  res.status(204).send();
});

// PATCH for partial updates — unlike PUT which replaces the entire resource
app.patch('/api/items/:id', (req, res) => {
  res.json({ patched: req.params.id, fields: req.body });
});
```

### Route Parameters

```javascript
// :id is a named parameter — captured in req.params
app.get('/api/users/:id', (req, res) => {
  const { id } = req.params;
  res.json({ userId: id });
});

// Multiple parameters — useful for nested resources
app.get('/api/users/:userId/posts/:postId', (req, res) => {
  const { userId, postId } = req.params;
  res.json({ userId, postId });
});
```

### Query Parameters

```javascript
// Query strings are parsed automatically into req.query
// GET /api/search?q=express&page=2&limit=10
app.get('/api/search', (req, res) => {
  const { q, page = '1', limit = '10' } = req.query;

  // All query values are strings — convert to numbers when needed
  res.json({
    query: q,
    page: parseInt(page, 10),
    limit: parseInt(limit, 10),
  });
});
```

### Route Chaining

```javascript
// app.route() groups handlers for the same path — reduces repetition
app.route('/api/books')
  .get((req, res) => {
    res.json({ books: [] });
  })
  .post((req, res) => {
    res.status(201).json({ created: req.body });
  });

app.route('/api/books/:id')
  .get((req, res) => {
    res.json({ bookId: req.params.id });
  })
  .put((req, res) => {
    res.json({ updated: req.params.id });
  })
  .delete((req, res) => {
    res.status(204).send();
  });
```

---

## 4. Middleware Concept and Chain

Middleware functions have access to the request, response, and the **next** function. They form a pipeline that each request flows through.

```
Request → [Middleware 1] → [Middleware 2] → [Route Handler] → Response
              │                  │                │
              │ next()           │ next()          │ res.send()
              └──────────────────┘────────────────┘
```

### Theory: Middleware as Continuation-Passing Style

The Express middleware signature is:

```javascript
function middleware(req, res, next) {
    // do something
    next();           // hand off to the next middleware
    // or: next(err); // jump to the error-handling middleware
}
```

This is *continuation-passing style* (CPS), a control-flow pattern where each function takes a callback (`next`) representing "what to do next". The chain composes implicitly: the registration order in `app.use(...)` defines a linked list, and `next()` advances the pointer.

#### B.1 The chain is linear, not nested

Unlike FastAPI's onion-shaped middleware (Lesson 03 §C), Express middleware is linear by default. Each middleware runs once on the way in. To run code *after* the response, you wrap `res.end` or use `res.on('finish', ...)` — there is no "after middleware" baked in. (Express 5 / Koa-style frameworks add this with async middleware that can `await next()`.)

```
[mw1] → [mw2] → [mw3 = handler] → res.send()
```

If `mw1` does work after `next()`, that work runs *before* mw2 finishes — synchronously interleaved, not after the response. This is a frequent source of confusion for FastAPI/Koa migrators.

#### B.2 Error-handling middleware: a four-arg variant

Any middleware with arity 4 — `(err, req, res, next)` — is recognized by Express as an *error handler*. When `next(err)` is called, Express skips every regular middleware until it finds the next 4-arg one. This is how a single error handler at the bottom of your stack catches all upstream errors.

```javascript
app.use(routes);
app.use((err, req, res, next) => {
    res.status(500).json({ error: err.message });
});
```

The catch: synchronous `throw` inside an async handler is *not* automatically routed to `next(err)` in Express 4 — you must `try/catch` or wrap with a helper like `express-async-errors`. Express 5 fixes this.

#### B.3 Mounting: middleware as path-prefixed sub-apps

`app.use('/api', router)` mounts a sub-router under a prefix. Inside that router, `req.url` is rewritten to be relative to the mount point (`req.baseUrl` keeps the prefix). This is composition: you can build small focused routers (users, posts, auth) and combine them, with middleware applied per-sub-app.

### Writing Custom Middleware

```javascript
// Middleware signature: (req, res, next) => { ... }
// Calling next() passes control to the next middleware in the chain.
// Forgetting next() causes the request to hang — a common beginner mistake.

const requestLogger = (req, res, next) => {
  const start = Date.now();

  // res.on('finish') fires after the response is sent — lets us measure
  // total request duration without blocking the response
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.originalUrl} ${res.statusCode} ${duration}ms`);
  });

  next();
};

// app.use() registers middleware for ALL routes and methods
app.use(requestLogger);
```

### Middleware Execution Order

```javascript
// Middleware runs in the order it is registered — order matters!
app.use((req, res, next) => {
  console.log('1st middleware');
  next();
});

app.use((req, res, next) => {
  console.log('2nd middleware');
  next();
});

app.get('/test', (req, res) => {
  console.log('Route handler');
  res.send('Done');
});

// Output for GET /test:
// 1st middleware
// 2nd middleware
// Route handler
```

### Path-Specific Middleware

```javascript
// Middleware can be scoped to specific paths — only runs for matching routes
const apiKeyCheck = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  if (!apiKey || apiKey !== process.env.API_KEY) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  next();
};

// Only applies to routes starting with /api/admin
app.use('/api/admin', apiKeyCheck);
```

### Multiple Middleware per Route

```javascript
const authenticate = (req, res, next) => {
  // Attach user info to request for downstream handlers
  req.user = { id: 1, role: 'admin' };
  next();
};

const authorize = (role) => (req, res, next) => {
  if (req.user.role !== role) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  next();
};

// Chain multiple middleware — each must call next() for the request to proceed
app.delete('/api/users/:id', authenticate, authorize('admin'), (req, res) => {
  res.json({ deleted: req.params.id });
});
```

---

## 5. Built-in Middleware

Express 4.x includes several built-in middleware functions:

```javascript
import express from 'express';

// Parse JSON request bodies — required for POST/PUT/PATCH with JSON payloads
// Without this, req.body is undefined for JSON requests
app.use(express.json());

// Parse URL-encoded form data — for traditional HTML form submissions
// extended: true uses the qs library which supports nested objects
app.use(express.urlencoded({ extended: true }));

// Serve static files from a directory — CSS, images, client-side JS
// Files are served relative to the specified directory
app.use(express.static('public'));

// Mount static files at a specific URL prefix
app.use('/assets', express.static('public'));
```

### JSON Body Size Limit

```javascript
// Limit request body size — prevents denial-of-service via oversized payloads
// Default is 100kb; adjust based on your API's needs
app.use(express.json({ limit: '10mb' }));
```

---

## 6. Request and Response Objects

### Theory: Request and Response: Stream Interfaces

`req` is an `IncomingMessage`, a *Readable stream* of the request body. `res` is a `ServerResponse`, a *Writable stream* of the response body. Both inherit from `EventEmitter`. The implications go deep.

#### C.1 The body is a stream, not a buffer

When the handler runs, the request body has *not yet been received*. Only the headers are known. The body arrives as `data` events:

```javascript
req.on('data', (chunk) => { ... });
req.on('end', () => { ... });
```

`express.json()` is exactly this: a middleware that listens for `data` chunks, accumulates them into a `Buffer`, parses the buffer as JSON when `end` fires, and assigns the result to `req.body`. After `express.json()`, `req.body` is the parsed object — but the underlying stream has been consumed and cannot be re-read.

#### C.2 Why this matters: large uploads

A 1 GB file upload that goes through `express.json()` allocates a 1 GB Buffer in memory before the handler ever runs. The right pattern for large payloads is to *not* use the body-parser middleware and consume the stream directly — pipe `req` into a file or an upload SDK chunk by chunk. Memory stays flat regardless of upload size.

#### C.3 Response streaming and back-pressure

`res.write(chunk)` returns `true` if the kernel write buffer accepted the chunk, `false` if it is full. Ignoring that return value and writing as fast as your code can produce data fills the buffer and burns memory. The correct pattern:

```javascript
if (!res.write(chunk)) {
    res.once('drain', writeNext);
} else {
    writeNext();
}
```

Or use `pipe()`, which handles back-pressure automatically: `readableSource.pipe(res)`. This is the idiomatic streaming response in Node — no chunked encoding to think about, no buffer accounting.

#### C.4 The lifecycle events

Both `req` and `res` emit events you can hook:

- `req.on('close', ...)` — the client disconnected mid-request. Cancel any in-flight work.
- `res.on('finish', ...)` — the response was fully flushed to the kernel. Good place for "request done" logging.
- `res.on('close', ...)` — the underlying connection closed (may be before `finish` if the client hung up).

A long-running handler should listen for `req.close` and abort its work — otherwise it ties up event-loop time computing a response no one will read.

### Request Object (req)

```javascript
app.post('/api/users', (req, res) => {
  // Key properties on the request object:
  console.log(req.method);       // 'POST'
  console.log(req.path);         // '/api/users'
  console.log(req.originalUrl);  // '/api/users?sort=name' (includes query string)
  console.log(req.params);       // Route parameters: { id: '42' }
  console.log(req.query);        // Query string: { sort: 'name' }
  console.log(req.body);         // Parsed body (requires express.json())
  console.log(req.headers);      // All headers (lowercase keys)
  console.log(req.ip);           // Client IP address
  console.log(req.hostname);     // 'localhost'
  console.log(req.get('Content-Type')); // Get specific header
});
```

### Response Object (res)

```javascript
app.get('/api/demo', (req, res) => {
  // res.json() — sets Content-Type to application/json and serializes
  res.json({ message: 'hello' });

  // res.status() — sets HTTP status code; chainable
  res.status(201).json({ created: true });

  // res.send() — sends string, Buffer, or object
  res.send('Plain text response');

  // res.redirect() — sends 302 redirect by default
  res.redirect('/new-location');
  res.redirect(301, '/permanent-new-location');

  // res.set() — set response headers
  res.set('X-Request-Id', 'abc-123');

  // res.cookie() — set cookies (needs cookie-parser for reading)
  res.cookie('session', 'token-value', {
    httpOnly: true,   // Not accessible via JavaScript — prevents XSS theft
    secure: true,     // Only sent over HTTPS
    maxAge: 3600000,  // 1 hour in milliseconds
  });

  // res.download() — prompt file download
  res.download('/path/to/file.pdf');
});
```

---

## 7. Router for Modular Routes

As an application grows, putting all routes in a single file becomes unmanageable. Express `Router` creates modular, mountable route handlers.

### Creating a Router Module

```javascript
// src/routes/users.js
import { Router } from 'express';

const router = Router();

// Routes defined here are relative to wherever the router is mounted
router.get('/', (req, res) => {
  res.json({ users: [{ id: 1, name: 'Alice' }] });
});

router.get('/:id', (req, res) => {
  res.json({ userId: req.params.id });
});

router.post('/', (req, res) => {
  const { name, email } = req.body;
  res.status(201).json({ id: 2, name, email });
});

router.put('/:id', (req, res) => {
  res.json({ updated: req.params.id, ...req.body });
});

router.delete('/:id', (req, res) => {
  res.status(204).send();
});

export default router;
```

### Mounting Routers

```javascript
// src/app.js
import express from 'express';
import usersRouter from './routes/users.js';
import postsRouter from './routes/posts.js';

const app = express();
app.use(express.json());

// Mount routers at path prefixes — all routes inside become relative
// e.g., router.get('/') becomes GET /api/users
app.use('/api/users', usersRouter);
app.use('/api/posts', postsRouter);

// 404 handler — must be registered after all routes
// If no route matched, this middleware catches the request
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

export default app;
```

### Separating App from Server

```javascript
// src/server.js — separation allows importing app without starting the server,
// which is essential for testing (Supertest needs the app, not a running server)
import app from './app.js';

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

### Nested Routers

```javascript
// src/routes/posts.js
import { Router } from 'express';

const router = Router();

// Nested resource: /api/posts/:postId/comments
const commentsRouter = Router({ mergeParams: true });
// mergeParams: true — gives child router access to parent's :postId param

commentsRouter.get('/', (req, res) => {
  res.json({ postId: req.params.postId, comments: [] });
});

commentsRouter.post('/', (req, res) => {
  res.status(201).json({
    postId: req.params.postId,
    comment: req.body,
  });
});

router.use('/:postId/comments', commentsRouter);

router.get('/', (req, res) => {
  res.json({ posts: [] });
});

export default router;
```

---

## 8. Practice Problems

### Problem 1: Health Check Endpoint

Create an Express server with a `GET /health` endpoint that returns:
```json
{ "status": "ok", "uptime": 12345, "timestamp": "2025-01-01T00:00:00.000Z" }
```

Where `uptime` is `process.uptime()` rounded to the nearest second.

### Problem 2: Request Timing Middleware

Write a middleware function that:
- Adds an `X-Response-Time` header to every response containing the processing time in milliseconds
- Logs the method, URL, status code, and response time to the console

Hint: Record `Date.now()` at the start and use `res.on('finish', ...)` to compute the duration.

### Problem 3: CRUD API with Router

Build a complete CRUD API for a "tasks" resource using `express.Router()`:
- `GET /api/tasks` -- list all tasks (support `?status=done` filter)
- `GET /api/tasks/:id` -- get a single task (return 404 if not found)
- `POST /api/tasks` -- create a task (validate that `title` is present)
- `PUT /api/tasks/:id` -- update a task
- `DELETE /api/tasks/:id` -- delete a task

Use an in-memory array as the data store.

### Problem 4: Middleware Chain

Create an Express app with three middleware layers:
1. A **logger** that prints the request method and URL
2. An **auth checker** that returns 401 if the `Authorization` header is missing
3. A **role checker** factory function `requireRole(role)` that returns 403 if the user's role does not match

Chain them on a `DELETE /api/users/:id` route.

### Problem 5: Nested Resources

Design a router structure for a blog application with nested resources:
- `GET /api/authors/:authorId/articles` -- list articles by author
- `POST /api/authors/:authorId/articles` -- create an article for an author
- `GET /api/authors/:authorId/articles/:articleId` -- get a specific article

Use `mergeParams: true` so the articles router has access to `:authorId`.

---

## References

- [Express.js Official Documentation](https://expressjs.com/)
- [Express 4.x API Reference](https://expressjs.com/en/4x/api.html)
- [Node.js Documentation](https://nodejs.org/docs/latest/api/)
- [MDN: HTTP Methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)

---

**Previous**: [FastAPI Testing](./05_FastAPI_Testing.md) | **Next**: [Express Advanced](./07_Express_Advanced.md)
