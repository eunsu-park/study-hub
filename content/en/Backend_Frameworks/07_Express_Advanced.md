# 07. Express Advanced

**Previous**: [Express Basics](./06_Express_Basics.md) | **Next**: [Express Database](./08_Express_Database.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement centralized error handling using Express's 4-argument middleware pattern
2. Set up authentication with Passport.js using both local and JWT strategies
3. Protect APIs with rate limiting, CORS policies, and security headers
4. Handle file uploads with multer and validate request data with Zod
5. Combine multiple security middleware into a production-ready configuration

---

Building a basic Express server is straightforward, but production applications demand more: authentication, input validation, file handling, rate limiting, and proper error management. This lesson covers the middleware and libraries that transform a prototype into a robust, secure API server. Each section builds on the middleware chain concept from the previous lesson.

## Table of Contents

1. [Error Handling Middleware](#1-error-handling-middleware)
2. [Authentication with Passport.js](#2-authentication-with-passportjs)
3. [Rate Limiting](#3-rate-limiting)
4. [CORS Configuration](#4-cors-configuration)
5. [File Uploads with Multer](#5-file-uploads-with-multer)
6. [Input Validation with Zod](#6-input-validation-with-zod)
7. [Security Headers with Helmet](#7-security-headers-with-helmet)
8. [Putting It All Together](#8-putting-it-all-together)
9. [Practice Problems](#9-practice-problems)

---

## 1. Error Handling Middleware

Express recognizes error-handling middleware by its **four arguments**: `(err, req, res, next)`. This distinguishes it from regular middleware and route handlers.

### The Problem with Unhandled Errors

```javascript
// Without error handling, an unhandled exception crashes the server
app.get('/api/users/:id', (req, res) => {
  const user = getUserById(req.params.id); // Throws if user not found
  res.json(user); // Never reached — server crashes
});
```

### Centralized Error Handler

```javascript
// Define a custom error class — allows attaching HTTP status codes to errors
// so the error handler knows what status to send without guessing
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true; // Distinguishes expected errors from bugs
  }
}

// Route that throws a known error
app.get('/api/users/:id', (req, res, next) => {
  const user = users.find(u => u.id === parseInt(req.params.id));
  if (!user) {
    // Pass error to next() — Express skips to the error handler
    return next(new AppError('User not found', 404));
  }
  res.json(user);
});

// Error-handling middleware — MUST have exactly 4 parameters
// Express uses the argument count to identify this as an error handler
app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;

  // Log full error for server-side debugging
  console.error(`[ERROR] ${err.message}`, {
    statusCode,
    stack: err.stack,
    path: req.originalUrl,
  });

  res.status(statusCode).json({
    error: {
      message: err.isOperational ? err.message : 'Internal server error',
      // Never expose stack traces in production — they reveal implementation details
      ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
    },
  });
});
```

### Async Error Handling

```javascript
// Express 4.x does NOT catch promise rejections automatically.
// Without a wrapper, async errors bypass the error handler entirely.

// Helper that wraps async route handlers — catches rejected promises
// and forwards them to the error middleware via next()
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

app.get('/api/posts', asyncHandler(async (req, res) => {
  const posts = await fetchPostsFromDB(); // If this rejects, error handler catches it
  res.json(posts);
}));

// Note: Express 5.x (currently in beta) handles async errors natively
```

---

## 2. Authentication with Passport.js

Passport.js is an authentication middleware for Node.js with 500+ strategies. We focus on the two most common: **local** (username/password) and **JWT**.

### Installation

```bash
npm install passport passport-local passport-jwt jsonwebtoken bcrypt
```

### Local Strategy (Username + Password)

```javascript
// src/auth/passport.js
import passport from 'passport';
import { Strategy as LocalStrategy } from 'passport-local';
import bcrypt from 'bcrypt';

// The local strategy validates credentials against your database
passport.use(new LocalStrategy(
  {
    usernameField: 'email', // Override default field name 'username'
    passwordField: 'password',
  },
  async (email, password, done) => {
    try {
      const user = await findUserByEmail(email);
      if (!user) {
        // null = no system error, false = authentication failed
        return done(null, false, { message: 'User not found' });
      }

      // bcrypt.compare handles salt extraction automatically —
      // the salt is embedded in the stored hash
      const isValid = await bcrypt.compare(password, user.passwordHash);
      if (!isValid) {
        return done(null, false, { message: 'Incorrect password' });
      }

      return done(null, user);
    } catch (err) {
      return done(err); // System error — gets forwarded to error middleware
    }
  }
));
```

### JWT Strategy

```javascript
// src/auth/passport.js (continued)
import { Strategy as JwtStrategy, ExtractJwt } from 'passport-jwt';

const jwtOptions = {
  // Extract token from the Authorization: Bearer <token> header
  jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
  secretOrKey: process.env.JWT_SECRET,
};

passport.use(new JwtStrategy(jwtOptions, async (payload, done) => {
  try {
    const user = await findUserById(payload.sub);
    if (!user) return done(null, false);
    return done(null, user);
  } catch (err) {
    return done(err);
  }
}));
```

### Login Route (Issue JWT)

```javascript
// src/routes/auth.js
import { Router } from 'express';
import passport from 'passport';
import jwt from 'jsonwebtoken';

const router = Router();

router.post('/login', (req, res, next) => {
  // { session: false } — we use JWT instead of server-side sessions,
  // so there is no need for Passport to create a session
  passport.authenticate('local', { session: false }, (err, user, info) => {
    if (err) return next(err);
    if (!user) return res.status(401).json({ error: info.message });

    const token = jwt.sign(
      { sub: user.id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '1h' } // Short-lived tokens limit damage if compromised
    );

    res.json({ token, expiresIn: 3600 });
  })(req, res, next);
});

export default router;
```

### Protecting Routes

```javascript
// Reusable middleware — attach to any route that requires authentication
const requireAuth = passport.authenticate('jwt', { session: false });

app.get('/api/profile', requireAuth, (req, res) => {
  // req.user is populated by Passport after successful JWT verification
  res.json({ user: req.user });
});
```

---

## 3. Rate Limiting

Rate limiting prevents abuse by restricting how many requests a client can make in a time window.

```bash
npm install express-rate-limit
```

```javascript
import rateLimit from 'express-rate-limit';

// Global rate limit — applies to all routes
const globalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15-minute window
  max: 100,                  // 100 requests per window per IP
  standardHeaders: true,     // Return rate limit info in RateLimit-* headers
  legacyHeaders: false,      // Disable X-RateLimit-* headers
  message: { error: 'Too many requests, please try again later' },
});

app.use(globalLimiter);

// Stricter limit for authentication routes — login endpoints are prime
// targets for brute-force attacks
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: { error: 'Too many login attempts, try again in 15 minutes' },
});

app.use('/api/auth/login', authLimiter);
```

---

## 4. CORS Configuration

Cross-Origin Resource Sharing (CORS) controls which domains can access your API from a browser.

```bash
npm install cors
```

```javascript
import cors from 'cors';

// Allow all origins — suitable for development only
app.use(cors());

// Production configuration — restrict to known frontends
const corsOptions = {
  origin: ['https://myapp.com', 'https://admin.myapp.com'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true, // Allow cookies to be sent cross-origin
  maxAge: 86400,     // Cache preflight response for 24 hours — reduces OPTIONS requests
};

app.use(cors(corsOptions));

// Per-route CORS — useful when only some endpoints need cross-origin access
app.get('/api/public/data', cors(), (req, res) => {
  res.json({ data: 'accessible from any origin' });
});
```

---

## 5. File Uploads with Multer

Multer handles `multipart/form-data`, the encoding type used for file uploads.

```bash
npm install multer
```

### Basic File Upload

```javascript
import multer from 'multer';
import path from 'node:path';

// Configure storage — control where files are saved and how they are named
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    // Prepend timestamp to avoid name collisions when multiple users
    // upload files with the same name
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

// File filter — reject non-image files at the middleware level
// before they consume disk space
const fileFilter = (req, file, cb) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only JPEG, PNG, and WebP images are allowed'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5 MB max
});

// Single file upload — 'avatar' is the form field name
app.post('/api/upload/avatar', upload.single('avatar'), (req, res) => {
  // req.file contains metadata about the uploaded file
  res.json({
    filename: req.file.filename,
    size: req.file.size,
    mimetype: req.file.mimetype,
  });
});

// Multiple file upload — up to 10 files in the 'photos' field
app.post('/api/upload/photos', upload.array('photos', 10), (req, res) => {
  const files = req.files.map(f => ({
    filename: f.filename,
    size: f.size,
  }));
  res.json({ uploaded: files });
});
```

---

## 6. Input Validation with Zod

Zod provides TypeScript-first schema validation. It defines the shape of valid data and produces clear error messages for invalid input.

```bash
npm install zod
```

### Defining Schemas

```javascript
import { z } from 'zod';

// Schema doubles as documentation — readers can see exactly what the endpoint accepts
const createUserSchema = z.object({
  name: z.string().min(2).max(100),
  email: z.string().email(),
  age: z.number().int().min(0).max(150).optional(),
  role: z.enum(['user', 'admin']).default('user'),
});

const updateUserSchema = createUserSchema.partial();
// .partial() makes all fields optional — suitable for PATCH operations
```

### Validation Middleware Factory

```javascript
// Generic middleware that validates req.body against any Zod schema
const validate = (schema) => (req, res, next) => {
  const result = schema.safeParse(req.body);

  if (!result.success) {
    // Zod provides structured error details per field —
    // flatten() groups them by field name for easy frontend consumption
    const errors = result.error.flatten().fieldErrors;
    return res.status(400).json({ error: 'Validation failed', details: errors });
  }

  // Replace req.body with the parsed data — Zod strips unknown fields
  // and applies defaults, so downstream handlers get clean data
  req.body = result.data;
  next();
};

app.post('/api/users', validate(createUserSchema), (req, res) => {
  // req.body is guaranteed to be valid here
  res.status(201).json({ user: req.body });
});

app.patch('/api/users/:id', validate(updateUserSchema), (req, res) => {
  res.json({ updated: req.params.id, changes: req.body });
});
```

### Validating Query Parameters

```javascript
const paginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(['asc', 'desc']).default('desc'),
});

// Validate query instead of body
const validateQuery = (schema) => (req, res, next) => {
  const result = schema.safeParse(req.query);
  if (!result.success) {
    return res.status(400).json({ error: result.error.flatten().fieldErrors });
  }
  req.query = result.data; // Coerced and defaulted values
  next();
};

app.get('/api/posts', validateQuery(paginationSchema), (req, res) => {
  const { page, limit, sort } = req.query;
  res.json({ page, limit, sort });
});
```

---

## 7. Security Headers with Helmet

Helmet sets various HTTP headers to protect against common web vulnerabilities like XSS, clickjacking, and MIME sniffing.

```bash
npm install helmet
```

```javascript
import helmet from 'helmet';

// helmet() enables a sensible set of security headers with one call
app.use(helmet());
```

### What Helmet Sets

| Header | Purpose |
|--------|---------|
| `Content-Security-Policy` | Restricts sources for scripts, styles, images |
| `X-Content-Type-Options: nosniff` | Prevents MIME type sniffing |
| `X-Frame-Options: SAMEORIGIN` | Prevents clickjacking via iframe embedding |
| `Strict-Transport-Security` | Forces HTTPS connections |
| `X-XSS-Protection: 0` | Disables buggy browser XSS filters |
| `X-DNS-Prefetch-Control: off` | Controls DNS prefetching |

### Custom CSP Configuration

```javascript
// Override the default Content-Security-Policy for APIs that serve no HTML
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", 'data:', 'https:'],
      },
    },
    // Disable CSP entirely for pure API servers (no HTML responses)
    // contentSecurityPolicy: false,
  })
);
```

---

## 8. Putting It All Together

A production-ready Express app composes these middleware in a specific order:

```javascript
// src/app.js
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import passport from 'passport';

import './auth/passport.js'; // Initialize strategies (side-effect import)
import authRouter from './routes/auth.js';
import usersRouter from './routes/users.js';

const app = express();

// --- Security middleware first ---
app.use(helmet());                       // Security headers
app.use(cors({ origin: process.env.ALLOWED_ORIGINS?.split(',') }));

// --- Rate limiting before body parsing ---
// Reject abusive requests early, before spending resources parsing bodies
app.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 100 }));

// --- Body parsing ---
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// --- Authentication ---
app.use(passport.initialize());

// --- Routes ---
app.use('/api/auth', authRouter);
app.use('/api/users', usersRouter);

// --- 404 handler ---
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// --- Centralized error handler (must be last) ---
app.use((err, req, res, next) => {
  console.error(err);
  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    error: { message: err.isOperational ? err.message : 'Internal server error' },
  });
});

export default app;
```

---

## 9. Practice Problems

### Problem 1: Custom Error Classes

Create an error handling system with three custom error classes:
- `NotFoundError` (404)
- `ValidationError` (400) that accepts a `details` object with field-level errors
- `UnauthorizedError` (401)

Write a centralized error handler that formats each error type differently in the response.

### Problem 2: JWT Authentication Flow

Implement a complete authentication flow:
- `POST /api/auth/register` -- hash password with bcrypt, store user
- `POST /api/auth/login` -- verify credentials, return JWT
- `GET /api/auth/me` -- return current user profile (protected route)

Use an in-memory array for user storage. Include proper error messages for duplicate emails and wrong passwords.

### Problem 3: Rate Limiter Tiers

Configure three rate limiting tiers:
- **Public endpoints**: 100 requests per 15 minutes
- **Authenticated endpoints**: 1000 requests per 15 minutes (identified by JWT)
- **Admin endpoints**: no limit

Write a middleware factory `rateLimitByRole()` that selects the appropriate tier based on the user's role.

### Problem 4: File Upload with Validation

Build an image upload endpoint that:
- Accepts JPEG and PNG files up to 2MB
- Validates that the image has minimum dimensions of 100x100 pixels (hint: use the `image-size` package)
- Returns the file path, dimensions, and file size in the response
- Returns descriptive error messages for invalid files

### Problem 5: Request Validation Pipeline

Using Zod, create a validation middleware for a "create blog post" endpoint that validates:
- `title`: string, 5-200 characters
- `content`: string, minimum 50 characters
- `tags`: array of strings, 1-5 items, each 2-30 characters
- `publishAt`: optional ISO date string that must be in the future

Handle validation errors with field-level error messages.

---

## References

- [Express Error Handling Guide](https://expressjs.com/en/guide/error-handling.html)
- [Passport.js Documentation](http://www.passportjs.org/docs/)
- [express-rate-limit Documentation](https://github.com/express-rate-limit/express-rate-limit)
- [Helmet.js Documentation](https://helmetjs.github.io/)
- [Multer Documentation](https://github.com/expressjs/multer)
- [Zod Documentation](https://zod.dev/)
- [OWASP Security Headers](https://owasp.org/www-project-secure-headers/)

---

**Previous**: [Express Basics](./06_Express_Basics.md) | **Next**: [Express Database](./08_Express_Database.md)
