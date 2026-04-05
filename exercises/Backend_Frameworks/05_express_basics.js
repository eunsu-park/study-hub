/**
 * Exercise: Express Basics
 * Practice with routing, middleware, and request handling.
 *
 * Run: npm install express
 *      node 05_express_basics.js
 */

const express = require('express');

// Exercise 1: Create a REST API for a "notes" resource
// Implement CRUD operations with in-memory storage.
// POST /notes       — Create a note {title, content}
// GET /notes        — List all notes
// GET /notes/:id    — Get single note
// PUT /notes/:id    — Update note
// DELETE /notes/:id — Delete note

function createNotesApp() {
  const app = express();
  app.use(express.json());

  // TODO: Implement CRUD routes

  return app;
}


// Exercise 2: Request Logger Middleware
// Create middleware that logs: timestamp, method, url, status, duration
// Format: "[2024-01-15T10:30:00Z] GET /notes 200 12ms"

function requestLogger(req, res, next) {
  // TODO: Implement
  next();
}


// Exercise 3: Validation Middleware
// Create a middleware factory that validates request body
// against a schema (simple object with required fields).

function validateBody(schema) {
  // schema = { title: "string", price: "number" }
  // Returns middleware that validates req.body
  return (req, res, next) => {
    // TODO: Implement
    next();
  };
}


// Exercise 4: Router Module
// Create a modular router for /api/v1/users
// GET /api/v1/users         — List users
// GET /api/v1/users/:id     — Get user
// GET /api/v1/users/:id/notes — Get user's notes

function createUserRouter() {
  const router = express.Router();

  // TODO: Implement routes

  return router;
}


// Exercise 5: Error Handling
// Create a custom error class and error handling middleware.
// All errors should return JSON: { error: message, statusCode: code }

class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    // TODO: Set properties
  }
}

function errorHandler(err, req, res, next) {
  // TODO: Implement
}


// --- Manual Tests ---
if (require.main === module) {
  const app = createNotesApp();
  app.listen(3000, () => {
    console.log('Notes API on http://localhost:3000');
    console.log('Test with:');
    console.log('  curl -X POST http://localhost:3000/notes -H "Content-Type: application/json" -d \'{"title":"Test","content":"Hello"}\'');
    console.log('  curl http://localhost:3000/notes');
  });
}

module.exports = { createNotesApp, requestLogger, validateBody, createUserRouter };
