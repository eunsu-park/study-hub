/**
 * Express Basics — Hello Express
 * Demonstrates: routing, middleware chain, request/response, error handling.
 *
 * Run: npm init -y && npm install express
 *      node 01_hello_express.js
 * Test: curl http://localhost:3000/
 */

const express = require('express');
const app = express();
const PORT = 3000;

// --- Built-in Middleware ---
app.use(express.json());                    // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // Parse form data

// --- Custom Middleware: Request Logger ---
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.url} ${res.statusCode} ${duration}ms`);
  });
  next(); // Pass control to next middleware
});

// --- Routes ---

app.get('/', (req, res) => {
  res.json({ message: 'Hello, Express!' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', uptime: process.uptime() });
});

// --- Path Parameters ---
app.get('/users/:id', (req, res) => {
  const { id } = req.params;
  res.json({ userId: parseInt(id, 10) });
});

// --- Query Parameters ---
app.get('/search', (req, res) => {
  const { q, page = 1, limit = 10 } = req.query;
  if (!q) {
    return res.status(400).json({ error: 'Query parameter "q" is required' });
  }
  res.json({ query: q, page: Number(page), limit: Number(limit) });
});

// --- CRUD with In-Memory Store ---
const items = new Map();
let nextId = 1;

app.get('/items', (req, res) => {
  res.json([...items.values()]);
});

app.post('/items', (req, res) => {
  const { name, price } = req.body;
  if (!name || price == null) {
    return res.status(400).json({ error: 'name and price are required' });
  }
  const item = { id: nextId++, name, price: Number(price) };
  items.set(item.id, item);
  res.status(201).json(item);
});

app.get('/items/:id', (req, res) => {
  const item = items.get(Number(req.params.id));
  if (!item) return res.status(404).json({ error: 'Not found' });
  res.json(item);
});

app.delete('/items/:id', (req, res) => {
  const id = Number(req.params.id);
  if (!items.has(id)) return res.status(404).json({ error: 'Not found' });
  items.delete(id);
  res.status(204).end();
});

// --- Router Module Example ---
const userRouter = express.Router();
userRouter.get('/', (req, res) => res.json({ users: ['Alice', 'Bob'] }));
userRouter.get('/:id/posts', (req, res) => {
  res.json({ userId: req.params.id, posts: [] });
});
app.use('/api/users', userRouter);

// --- 404 Handler ---
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// --- Error Handler (4 arguments) ---
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
