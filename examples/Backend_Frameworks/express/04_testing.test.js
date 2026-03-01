/**
 * Express Testing — Supertest + Jest
 * Demonstrates: route testing, middleware testing, async tests.
 *
 * Run: npm install express jest supertest
 *      npx jest 04_testing.test.js --verbose
 */

const express = require('express');
// In a real project: const request = require('supertest');

// --- App Factory (recommended pattern for testing) ---

function createApp() {
  const app = express();
  app.use(express.json());

  const items = new Map();
  let nextId = 1;

  app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/items', (req, res) => {
    res.json([...items.values()]);
  });

  app.post('/items', (req, res) => {
    const { name, price } = req.body;
    if (!name) return res.status(400).json({ error: 'name is required' });
    if (typeof price !== 'number' || price <= 0) {
      return res.status(400).json({ error: 'price must be a positive number' });
    }
    const item = { id: nextId++, name, price };
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

  // Error handler
  app.use((err, req, res, next) => {
    res.status(500).json({ error: 'Internal server error' });
  });

  return app;
}

// --- Tests (using built-in http for demonstration) ---
// In a real project, replace with supertest:
//   const request = require('supertest');
//   const response = await request(app).get('/health');

const http = require('http');

function makeRequest(app, method, path, body = null) {
  return new Promise((resolve, reject) => {
    const server = app.listen(0, () => {
      const port = server.address().port;
      const options = {
        hostname: 'localhost',
        port,
        path,
        method,
        headers: { 'Content-Type': 'application/json' },
      };

      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          server.close();
          resolve({
            statusCode: res.statusCode,
            body: data ? JSON.parse(data) : null,
          });
        });
      });

      req.on('error', (err) => {
        server.close();
        reject(err);
      });

      if (body) req.write(JSON.stringify(body));
      req.end();
    });
  });
}

// --- Test Suite ---

async function runTests() {
  let passed = 0;
  let failed = 0;

  async function test(name, fn) {
    try {
      await fn();
      console.log(`  ✓ ${name}`);
      passed++;
    } catch (err) {
      console.log(`  ✗ ${name}: ${err.message}`);
      failed++;
    }
  }

  function assert(condition, message) {
    if (!condition) throw new Error(message || 'Assertion failed');
  }

  console.log('\nExpress Testing Demo\n');

  // Each test gets a fresh app instance
  await test('GET /health returns ok', async () => {
    const app = createApp();
    const res = await makeRequest(app, 'GET', '/health');
    assert(res.statusCode === 200, `Expected 200, got ${res.statusCode}`);
    assert(res.body.status === 'ok');
  });

  await test('POST /items creates item', async () => {
    const app = createApp();
    const res = await makeRequest(app, 'POST', '/items', {
      name: 'Widget',
      price: 9.99,
    });
    assert(res.statusCode === 201);
    assert(res.body.name === 'Widget');
    assert(res.body.id === 1);
  });

  await test('POST /items validates name', async () => {
    const app = createApp();
    const res = await makeRequest(app, 'POST', '/items', { price: 5.0 });
    assert(res.statusCode === 400);
  });

  await test('POST /items validates price', async () => {
    const app = createApp();
    const res = await makeRequest(app, 'POST', '/items', {
      name: 'X',
      price: -1,
    });
    assert(res.statusCode === 400);
  });

  await test('GET /items/:id returns 404 for missing', async () => {
    const app = createApp();
    const res = await makeRequest(app, 'GET', '/items/999');
    assert(res.statusCode === 404);
  });

  await test('CRUD roundtrip', async () => {
    const app = createApp();
    // Create
    const created = await makeRequest(app, 'POST', '/items', {
      name: 'Gadget',
      price: 19.99,
    });
    assert(created.statusCode === 201);
    const id = created.body.id;

    // Read
    const fetched = await makeRequest(app, 'GET', `/items/${id}`);
    assert(fetched.statusCode === 200);
    assert(fetched.body.name === 'Gadget');

    // Delete
    const deleted = await makeRequest(app, 'DELETE', `/items/${id}`);
    assert(deleted.statusCode === 204);

    // Verify deleted
    const missing = await makeRequest(app, 'GET', `/items/${id}`);
    assert(missing.statusCode === 404);
  });

  console.log(`\n${passed} passed, ${failed} failed\n`);
}

runTests().catch(console.error);
