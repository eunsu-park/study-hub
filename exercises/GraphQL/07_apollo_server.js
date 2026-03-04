/**
 * Exercise: Apollo Server Configuration
 * Practice configuring Apollo Server 4 with plugins, context, and error handling.
 *
 * Run: node 07_apollo_server.js
 */

// ============================================================
// Exercise 1: Context Factory
// Implement a context function that extracts auth, sets up DataLoaders,
// and provides database connections.
// ============================================================

function createContext(req) {
  // TODO: Extract auth token from Authorization header
  // TODO: Decode token to get userId (simulated)
  // TODO: Return context object with { userId, isAuthenticated, db, loaders }

  const authHeader = req.headers && req.headers.authorization;
  let userId = null;
  let isAuthenticated = false;

  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.slice(7);
    // Simulated token decode
    const tokenMap = { 'token-alice': 'u1', 'token-bob': 'u2', 'token-admin': 'u3' };
    userId = tokenMap[token] || null;
    isAuthenticated = userId !== null;
  }

  return {
    userId,
    isAuthenticated,
    db: mockDB,
    loaders: createMockLoaders(),
  };
}

const mockDB = {
  users: [
    { id: 'u1', name: 'Alice', role: 'user' },
    { id: 'u2', name: 'Bob', role: 'user' },
    { id: 'u3', name: 'Admin', role: 'admin' },
  ],
  posts: [
    { id: 'p1', title: 'Hello', authorId: 'u1' },
    { id: 'p2', title: 'World', authorId: 'u2' },
  ],
};

function createMockLoaders() {
  return {
    userLoader: { load: (id) => mockDB.users.find((u) => u.id === id) },
  };
}


// ============================================================
// Exercise 2: Plugin System
// Implement a plugin that logs request lifecycle events.
// ============================================================

// TODO: Implement a request logging plugin
// Track: requestDidStart, didResolveOperation, executionDidStart, willSendResponse
function createLoggingPlugin() {
  return {
    events: [],

    requestDidStart(requestContext) {
      const start = Date.now();
      this.events.push({ event: 'requestDidStart', query: requestContext.query, time: start });

      return {
        didResolveOperation: (ctx) => {
          this.events.push({ event: 'didResolveOperation', operation: ctx.operationName || 'anonymous' });
        },
        executionDidStart: () => {
          this.events.push({ event: 'executionDidStart' });
          return {
            willResolveField: (fieldCtx) => {
              this.events.push({ event: 'willResolveField', field: fieldCtx.fieldName });
            },
          };
        },
        willSendResponse: () => {
          const duration = Date.now() - start;
          this.events.push({ event: 'willSendResponse', durationMs: duration });
        },
      };
    },
  };
}

// TODO: Implement a query complexity plugin
// Reject queries that exceed a complexity threshold
function createComplexityPlugin(maxComplexity) {
  return {
    rejected: [],
    allowed: [],

    checkQuery(query, complexity) {
      if (complexity > maxComplexity) {
        this.rejected.push({ query, complexity, max: maxComplexity });
        return { allowed: false, reason: `Complexity ${complexity} exceeds max ${maxComplexity}` };
      }
      this.allowed.push({ query, complexity });
      return { allowed: true };
    },
  };
}


// ============================================================
// Exercise 3: Error Formatting
// Implement formatError to sanitize errors for production.
// ============================================================

class AppError extends Error {
  constructor(message, code, statusCode = 400) {
    super(message);
    this.code = code;
    this.statusCode = statusCode;
  }
}

// TODO: Implement formatError function
// - If AppError: include code and message
// - If validation error: include field info
// - Otherwise: mask internal details, return generic message
// - Always include a requestId for tracking
function formatError(error, requestId) {
  const base = { requestId, timestamp: new Date().toISOString() };

  if (error instanceof AppError) {
    return {
      ...base,
      message: error.message,
      code: error.code,
      statusCode: error.statusCode,
    };
  }

  // Check for known error types
  if (error.message && error.message.startsWith('Validation:')) {
    return {
      ...base,
      message: error.message,
      code: 'VALIDATION_ERROR',
      statusCode: 400,
    };
  }

  // Mask internal errors in production
  return {
    ...base,
    message: 'An internal error occurred',
    code: 'INTERNAL_ERROR',
    statusCode: 500,
  };
}


// ============================================================
// Exercise 4: Server Configuration Builder
// Implement a builder pattern for Apollo Server config.
// ============================================================

class ServerConfigBuilder {
  constructor() {
    this.config = {
      typeDefs: '',
      resolvers: {},
      plugins: [],
      introspection: false,
      csrfPrevention: true,
      cors: { origin: [] },
      cache: 'bounded',
    };
  }

  // TODO: Implement builder methods
  setSchema(typeDefs, resolvers) {
    this.config.typeDefs = typeDefs;
    this.config.resolvers = resolvers;
    return this;
  }

  addPlugin(plugin) {
    this.config.plugins.push(plugin);
    return this;
  }

  enableIntrospection(enabled = true) {
    this.config.introspection = enabled;
    return this;
  }

  setCors(origins) {
    this.config.cors.origin = origins;
    return this;
  }

  setEnvironment(env) {
    if (env === 'development') {
      this.config.introspection = true;
      this.config.csrfPrevention = false;
    } else if (env === 'production') {
      this.config.introspection = false;
      this.config.csrfPrevention = true;
    }
    return this;
  }

  build() {
    if (!this.config.typeDefs) throw new Error('Schema typeDefs required');
    return { ...this.config };
  }
}


// ============================================================
// Test all exercises
// ============================================================

console.log('=== Exercise 1: Context Factory ===\n');

const ctx1 = createContext({ headers: { authorization: 'Bearer token-alice' } });
console.log(`Authenticated: ${ctx1.isAuthenticated} (expected true): ${ctx1.isAuthenticated ? 'PASS' : 'FAIL'}`);
console.log(`UserId: ${ctx1.userId} (expected u1): ${ctx1.userId === 'u1' ? 'PASS' : 'FAIL'}`);
console.log(`Has db: ${!!ctx1.db} (expected true): ${ctx1.db ? 'PASS' : 'FAIL'}`);

const ctx2 = createContext({ headers: {} });
console.log(`No auth: ${!ctx2.isAuthenticated} (expected true): ${!ctx2.isAuthenticated ? 'PASS' : 'FAIL'}`);

const ctx3 = createContext({ headers: { authorization: 'Bearer invalid' } });
console.log(`Bad token: ${!ctx3.isAuthenticated} (expected true): ${!ctx3.isAuthenticated ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 2: Plugin System ===\n');

const logger = createLoggingPlugin();
const hooks = logger.requestDidStart({ query: '{ users { name } }' });
hooks.didResolveOperation({ operationName: 'GetUsers' });
hooks.executionDidStart().willResolveField({ fieldName: 'users' });
hooks.willSendResponse();

console.log(`Events tracked: ${logger.events.length} (expected 4): ${logger.events.length === 4 ? 'PASS' : 'FAIL'}`);
logger.events.forEach((e) => console.log(`  ${e.event}${e.field ? `: ${e.field}` : ''}`));

const limiter = createComplexityPlugin(100);
const r1 = limiter.checkQuery('{ users { name } }', 10);
const r2 = limiter.checkQuery('{ users { friends { friends { posts } } } }', 150);
console.log(`\nSimple query: ${r1.allowed ? 'ALLOWED' : 'REJECTED'} (expected ALLOWED): ${r1.allowed ? 'PASS' : 'FAIL'}`);
console.log(`Complex query: ${r2.allowed ? 'ALLOWED' : 'REJECTED'} (expected REJECTED): ${!r2.allowed ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 3: Error Formatting ===\n');

const e1 = formatError(new AppError('User not found', 'NOT_FOUND', 404), 'req-001');
console.log(`AppError code: ${e1.code} (expected NOT_FOUND): ${e1.code === 'NOT_FOUND' ? 'PASS' : 'FAIL'}`);

const e2 = formatError(new Error('Validation: email is required'), 'req-002');
console.log(`Validation code: ${e2.code} (expected VALIDATION_ERROR): ${e2.code === 'VALIDATION_ERROR' ? 'PASS' : 'FAIL'}`);

const e3 = formatError(new Error('ECONNREFUSED db:5432'), 'req-003');
console.log(`Internal masked: ${e3.code} (expected INTERNAL_ERROR): ${e3.code === 'INTERNAL_ERROR' ? 'PASS' : 'FAIL'}`);
console.log(`  Message: "${e3.message}" (should NOT contain db:5432): ${!e3.message.includes('5432') ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 4: Server Config Builder ===\n');

const config = new ServerConfigBuilder()
  .setSchema('type Query { hello: String }', { Query: { hello: () => 'world' } })
  .addPlugin(logger)
  .setCors(['https://example.com', 'https://app.example.com'])
  .setEnvironment('production')
  .build();

console.log(`Has typeDefs: ${!!config.typeDefs}: ${config.typeDefs ? 'PASS' : 'FAIL'}`);
console.log(`Plugins count: ${config.plugins.length} (expected 1): ${config.plugins.length === 1 ? 'PASS' : 'FAIL'}`);
console.log(`Introspection off (prod): ${!config.introspection}: ${!config.introspection ? 'PASS' : 'FAIL'}`);
console.log(`CORS origins: ${config.cors.origin.length} (expected 2): ${config.cors.origin.length === 2 ? 'PASS' : 'FAIL'}`);

const devConfig = new ServerConfigBuilder()
  .setSchema('type Query { hello: String }', {})
  .setEnvironment('development')
  .build();
console.log(`Introspection on (dev): ${devConfig.introspection}: ${devConfig.introspection ? 'PASS' : 'FAIL'}`);

console.log('\nAll exercises completed!');
