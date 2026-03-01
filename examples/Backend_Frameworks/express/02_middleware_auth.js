/**
 * Express Advanced — Middleware, Authentication, Rate Limiting
 * Demonstrates: custom middleware, JWT auth, error middleware, rate limiting.
 *
 * Run: npm install express jsonwebtoken express-rate-limit helmet cors
 *      node 02_middleware_auth.js
 */

const express = require('express');
const jwt = require('jsonwebtoken');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');

const app = express();
const SECRET = 'demo-secret-key-change-in-production';

// --- Security Middleware ---
app.use(helmet());                          // Security headers
app.use(cors({ origin: 'http://localhost:3000' }));
app.use(express.json());

// --- Rate Limiting ---
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 100,                   // 100 requests per window
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later' },
});
app.use('/api/', limiter);

// --- Auth Middleware ---
function authenticate(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing bearer token' });
  }

  try {
    const token = authHeader.split(' ')[1];
    const payload = jwt.verify(token, SECRET);
    req.user = payload;  // Attach user to request
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
}

function authorize(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }
    next();
  };
}

// --- Routes ---

// Public: get a token
app.post('/api/login', (req, res) => {
  const { username, password } = req.body;

  // Fake user lookup
  if (username === 'admin' && password === 'admin123') {
    const token = jwt.sign(
      { sub: 1, username: 'admin', role: 'admin' },
      SECRET,
      { expiresIn: '1h' }
    );
    return res.json({ token });
  }

  if (username === 'user' && password === 'user123') {
    const token = jwt.sign(
      { sub: 2, username: 'user', role: 'user' },
      SECRET,
      { expiresIn: '1h' }
    );
    return res.json({ token });
  }

  res.status(401).json({ error: 'Invalid credentials' });
});

// Protected: any authenticated user
app.get('/api/profile', authenticate, (req, res) => {
  res.json({ user: req.user });
});

// Protected: admin only
app.get('/api/admin', authenticate, authorize('admin'), (req, res) => {
  res.json({ message: 'Welcome, admin!', user: req.user });
});

// --- Async Error Handling ---
// Express 5 handles async errors automatically.
// For Express 4, wrap async handlers:

const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

app.get('/api/data', authenticate, asyncHandler(async (req, res) => {
  // Simulate async operation
  const data = await new Promise((resolve) =>
    setTimeout(() => resolve({ items: [1, 2, 3] }), 100)
  );
  res.json(data);
}));

// --- Custom Error Classes ---
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true;
  }
}

app.get('/api/error-demo', (req, res) => {
  throw new AppError('This is a custom error', 422);
});

// --- Error Middleware (must be last, 4 args) ---
app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  const message = err.isOperational ? err.message : 'Internal server error';

  console.error(`[ERROR] ${statusCode} - ${err.message}`);

  res.status(statusCode).json({
    error: message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
});

app.listen(3000, () => console.log('Auth demo on http://localhost:3000'));
