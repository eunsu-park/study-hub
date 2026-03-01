/**
 * Exercise: Express Advanced
 * Practice with auth middleware, rate limiting, async error handling.
 *
 * Run: npm install express jsonwebtoken
 *      node 06_express_advanced.js
 */

// Exercise 1: JWT Authentication Middleware
// Implement authenticate() that:
// - Checks for Bearer token in Authorization header
// - Verifies the JWT token
// - Attaches decoded payload to req.user
// - Returns 401 if missing/invalid

const SECRET = 'exercise-secret';

function authenticate(req, res, next) {
  // TODO: Implement
  next();
}


// Exercise 2: Role-Based Authorization
// Create authorize(...roles) middleware factory
// Returns 403 if req.user.role is not in allowed roles.

function authorize(...roles) {
  return (req, res, next) => {
    // TODO: Implement
    next();
  };
}


// Exercise 3: In-Memory Rate Limiter
// Create a rate limiter without external packages.
// Track requests per IP in a Map.
// Config: { windowMs: 60000, maxRequests: 10 }

function createRateLimiter({ windowMs = 60000, maxRequests = 10 } = {}) {
  // TODO: Return middleware
  return (req, res, next) => {
    next();
  };
}


// Exercise 4: Async Handler Wrapper
// Create asyncHandler that catches promise rejections
// and passes errors to Express error middleware.

function asyncHandler(fn) {
  // TODO: Implement
  return fn;
}


// Exercise 5: API Versioning Middleware
// Create middleware that reads API version from:
// 1. URL prefix (/v1/..., /v2/...)
// 2. Header (X-API-Version: 2)
// 3. Query param (?version=2)
// Attaches req.apiVersion (default: 1)

function apiVersion(req, res, next) {
  // TODO: Implement
  next();
}


// --- Test Helpers ---
if (require.main === module) {
  const jwt = require('jsonwebtoken');

  // Generate test tokens
  const adminToken = jwt.sign({ sub: 1, role: 'admin' }, SECRET, { expiresIn: '1h' });
  const userToken = jwt.sign({ sub: 2, role: 'user' }, SECRET, { expiresIn: '1h' });

  console.log('Test tokens:');
  console.log(`  Admin: ${adminToken}`);
  console.log(`  User:  ${userToken}`);
  console.log();
  console.log('Implement the exercises and test with curl.');
}

module.exports = { authenticate, authorize, createRateLimiter, asyncHandler, apiVersion };
