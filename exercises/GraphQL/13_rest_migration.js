/**
 * Exercise: REST to GraphQL Migration
 * Practice building REST wrapper resolvers, field mapping, and migration strategies.
 *
 * Run: node 13_rest_migration.js
 */

// ============================================================
// Exercise 1: REST Data Source Wrapper
// Implement a data source that wraps REST endpoints with caching.
// ============================================================

class RESTDataSource {
  constructor(baseURL) {
    this.baseURL = baseURL;
    this.cache = new Map();
    this.callLog = [];
  }

  // Simulated REST responses
  _mockResponses = {
    '/users': [
      { user_id: 1, full_name: 'Alice Smith', email_addr: 'alice@ex.com' },
      { user_id: 2, full_name: 'Bob Jones', email_addr: 'bob@ex.com' },
    ],
    '/users/1': { user_id: 1, full_name: 'Alice Smith', email_addr: 'alice@ex.com' },
    '/users/2': { user_id: 2, full_name: 'Bob Jones', email_addr: 'bob@ex.com' },
    '/products': [
      { product_id: 101, product_name: 'Laptop', unit_price: 999.99, in_stock: true },
      { product_id: 102, product_name: 'Mouse', unit_price: 29.99, in_stock: false },
    ],
    '/products/101': { product_id: 101, product_name: 'Laptop', unit_price: 999.99, in_stock: true },
    '/users/1/orders': [{ order_id: 1001, total: 999.99, status: 'shipped' }],
    '/users/2/orders': [{ order_id: 1002, total: 29.99, status: 'pending' }],
  };

  // TODO: Implement get(path, options) — fetch with optional caching
  async get(path, { ttl = 0 } = {}) {
    const url = `${this.baseURL}${path}`;

    // Check cache
    if (ttl > 0) {
      const cached = this.cache.get(url);
      if (cached && Date.now() < cached.expiry) {
        this.callLog.push({ url, cached: true });
        return cached.data;
      }
    }

    // Simulate fetch
    const data = this._mockResponses[path] || null;
    this.callLog.push({ url, cached: false });

    // Cache if TTL set
    if (ttl > 0 && data) {
      this.cache.set(url, { data, expiry: Date.now() + ttl * 1000 });
    }

    return data;
  }

  // TODO: Implement getCacheStats()
  getCacheStats() {
    const total = this.callLog.length;
    const hits = this.callLog.filter((c) => c.cached).length;
    return { total, hits, misses: total - hits, hitRate: total > 0 ? hits / total : 0 };
  }
}


// ============================================================
// Exercise 2: Field Mapper
// Implement bidirectional field mapping between REST and GraphQL.
// ============================================================

class FieldMapper {
  constructor(mappings) {
    // mappings: [{ rest: 'user_id', graphql: 'id', transform? }]
    this.mappings = mappings;
  }

  // TODO: Implement toGraphQL(restObject) — map REST fields to GraphQL fields
  toGraphQL(restObject) {
    if (!restObject) return null;
    const result = {};
    for (const mapping of this.mappings) {
      const value = restObject[mapping.rest];
      if (value !== undefined) {
        result[mapping.graphql] = mapping.transform ? mapping.transform(value) : value;
      }
    }
    return result;
  }

  // TODO: Implement toREST(graphqlObject) — map GraphQL fields back to REST
  toREST(graphqlObject) {
    if (!graphqlObject) return null;
    const result = {};
    for (const mapping of this.mappings) {
      const value = graphqlObject[mapping.graphql];
      if (value !== undefined) {
        result[mapping.rest] = mapping.reverseTransform
          ? mapping.reverseTransform(value) : value;
      }
    }
    return result;
  }
}

const userMapper = new FieldMapper([
  { rest: 'user_id', graphql: 'id', transform: (v) => String(v) },
  { rest: 'full_name', graphql: 'name' },
  { rest: 'email_addr', graphql: 'email' },
]);

const productMapper = new FieldMapper([
  { rest: 'product_id', graphql: 'id', transform: (v) => String(v) },
  { rest: 'product_name', graphql: 'name' },
  { rest: 'unit_price', graphql: 'price' },
  { rest: 'in_stock', graphql: 'inStock' },
]);


// ============================================================
// Exercise 3: Migration Progress Tracker
// Track which endpoints have been migrated from REST to GraphQL.
// ============================================================

class MigrationTracker {
  constructor() {
    this.endpoints = new Map();
    // status: 'rest-only' | 'wrapped' | 'direct' | 'deprecated' | 'removed'
  }

  // TODO: Implement addEndpoint(path, method, description)
  addEndpoint(path, method = 'GET', description = '') {
    const key = `${method} ${path}`;
    this.endpoints.set(key, { path, method, description, status: 'rest-only', migratedAt: null });
  }

  // TODO: Implement updateStatus(path, method, newStatus)
  updateStatus(path, method, newStatus) {
    const key = `${method} ${path}`;
    const endpoint = this.endpoints.get(key);
    if (!endpoint) return false;
    endpoint.status = newStatus;
    if (newStatus !== 'rest-only') endpoint.migratedAt = new Date().toISOString();
    return true;
  }

  // TODO: Implement getProgress() — return migration progress summary
  getProgress() {
    const total = this.endpoints.size;
    const byStatus = {};
    for (const [, ep] of this.endpoints) {
      byStatus[ep.status] = (byStatus[ep.status] || 0) + 1;
    }
    const migrated = total - (byStatus['rest-only'] || 0);
    return {
      total,
      migrated,
      percentage: total > 0 ? Math.round((migrated / total) * 100) : 0,
      byStatus,
    };
  }

  // TODO: Implement getReport() — formatted migration report
  getReport() {
    const lines = ['Migration Report', '='.repeat(50)];
    const grouped = {};

    for (const [, ep] of this.endpoints) {
      if (!grouped[ep.status]) grouped[ep.status] = [];
      grouped[ep.status].push(`${ep.method} ${ep.path}`);
    }

    const statusOrder = ['rest-only', 'wrapped', 'direct', 'deprecated', 'removed'];
    const icons = { 'rest-only': 'REST', wrapped: 'WRAP', direct: 'GQL', deprecated: 'DEP', removed: 'DEL' };

    for (const status of statusOrder) {
      if (grouped[status]) {
        lines.push(`\n[${icons[status]}] ${status} (${grouped[status].length}):`);
        grouped[status].forEach((ep) => lines.push(`  - ${ep}`));
      }
    }

    const progress = this.getProgress();
    lines.push(`\nProgress: ${progress.migrated}/${progress.total} (${progress.percentage}%)`);
    return lines.join('\n');
  }
}


// ============================================================
// Exercise 4: Error Mapping (REST → GraphQL)
// Convert REST HTTP errors to GraphQL error format.
// ============================================================

function mapRESTError(httpStatus, restBody) {
  // TODO: Map HTTP status codes to GraphQL error extensions
  // 400 → BAD_USER_INPUT
  // 401 → UNAUTHENTICATED
  // 403 → FORBIDDEN
  // 404 → return null (not an error in GraphQL)
  // 429 → RATE_LIMITED
  // 500+ → INTERNAL_SERVER_ERROR

  const statusMap = {
    400: { code: 'BAD_USER_INPUT', message: restBody.message || 'Invalid input' },
    401: { code: 'UNAUTHENTICATED', message: 'Authentication required' },
    403: { code: 'FORBIDDEN', message: 'Access denied' },
    404: null, // GraphQL convention: return null, not error
    429: { code: 'RATE_LIMITED', message: 'Too many requests' },
  };

  if (httpStatus >= 500) {
    return {
      message: 'Internal server error',
      extensions: { code: 'INTERNAL_SERVER_ERROR', httpStatus },
    };
  }

  const mapped = statusMap[httpStatus];
  if (mapped === null) return null; // 404 → null
  if (mapped === undefined) {
    return {
      message: `Unexpected error (HTTP ${httpStatus})`,
      extensions: { code: 'UNKNOWN_ERROR', httpStatus },
    };
  }

  return {
    message: mapped.message,
    extensions: { code: mapped.code, httpStatus },
  };
}


// ============================================================
// Test all exercises
// ============================================================

async function runTests() {
  console.log('=== Exercise 1: REST Data Source ===\n');

  const api = new RESTDataSource('https://api.example.com');

  const users = await api.get('/users', { ttl: 60 });
  console.log(`Users fetched: ${users.length} (expected 2): ${users.length === 2 ? 'PASS' : 'FAIL'}`);

  // Second call should be cached
  await api.get('/users', { ttl: 60 });
  const stats = api.getCacheStats();
  console.log(`Cache stats: hits=${stats.hits}, misses=${stats.misses}: ${stats.hits === 1 ? 'PASS' : 'FAIL'}`);

  // No TTL = no caching
  await api.get('/products');
  await api.get('/products');
  console.log(`No-TTL calls: ${api.callLog.filter((c) => c.url.includes('/products')).length} (expected 2): ${api.callLog.filter((c) => c.url.includes('/products')).length === 2 ? 'PASS' : 'FAIL'}`);

  console.log('\n=== Exercise 2: Field Mapper ===\n');

  const restUser = { user_id: 1, full_name: 'Alice Smith', email_addr: 'alice@ex.com' };
  const gqlUser = userMapper.toGraphQL(restUser);
  console.log(`Mapped user: ${JSON.stringify(gqlUser)}`);
  console.log(`id is string: ${typeof gqlUser.id === 'string' ? 'PASS' : 'FAIL'}`);
  console.log(`name mapped: ${gqlUser.name === 'Alice Smith' ? 'PASS' : 'FAIL'}`);
  console.log(`email mapped: ${gqlUser.email === 'alice@ex.com' ? 'PASS' : 'FAIL'}`);

  const restProduct = { product_id: 101, product_name: 'Laptop', unit_price: 999.99, in_stock: true };
  const gqlProduct = productMapper.toGraphQL(restProduct);
  console.log(`\nMapped product: ${JSON.stringify(gqlProduct)}`);
  console.log(`inStock mapped: ${gqlProduct.inStock === true ? 'PASS' : 'FAIL'}`);

  // Null handling
  const nullResult = userMapper.toGraphQL(null);
  console.log(`Null input handled: ${nullResult === null ? 'PASS' : 'FAIL'}`);

  console.log('\n=== Exercise 3: Migration Tracker ===\n');

  const tracker = new MigrationTracker();

  // Register endpoints
  tracker.addEndpoint('/users', 'GET', 'List users');
  tracker.addEndpoint('/users/:id', 'GET', 'Get user');
  tracker.addEndpoint('/users', 'POST', 'Create user');
  tracker.addEndpoint('/products', 'GET', 'List products');
  tracker.addEndpoint('/products/:id', 'GET', 'Get product');
  tracker.addEndpoint('/orders', 'GET', 'List orders');

  // Migrate some endpoints
  tracker.updateStatus('/users', 'GET', 'wrapped');
  tracker.updateStatus('/users/:id', 'GET', 'wrapped');
  tracker.updateStatus('/products', 'GET', 'direct');

  const progress = tracker.getProgress();
  console.log(`Progress: ${progress.percentage}% (${progress.migrated}/${progress.total})`);
  console.log(`By status:`, progress.byStatus);
  console.log(`50% migrated: ${progress.percentage === 50 ? 'PASS' : 'FAIL'}`);

  console.log(`\n${tracker.getReport()}`);

  console.log('\n=== Exercise 4: Error Mapping ===\n');

  const e400 = mapRESTError(400, { message: 'Invalid email format' });
  console.log(`400 → ${e400.extensions.code}: ${e400.extensions.code === 'BAD_USER_INPUT' ? 'PASS' : 'FAIL'}`);

  const e401 = mapRESTError(401, {});
  console.log(`401 → ${e401.extensions.code}: ${e401.extensions.code === 'UNAUTHENTICATED' ? 'PASS' : 'FAIL'}`);

  const e404 = mapRESTError(404, {});
  console.log(`404 → null: ${e404 === null ? 'PASS' : 'FAIL'}`);

  const e429 = mapRESTError(429, {});
  console.log(`429 → ${e429.extensions.code}: ${e429.extensions.code === 'RATE_LIMITED' ? 'PASS' : 'FAIL'}`);

  const e500 = mapRESTError(500, {});
  console.log(`500 → ${e500.extensions.code}: ${e500.extensions.code === 'INTERNAL_SERVER_ERROR' ? 'PASS' : 'FAIL'}`);

  console.log('\nAll exercises completed!');
}

runTests();
