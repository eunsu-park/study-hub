/**
 * Exercise: Persisted Queries and Caching
 * Practice implementing APQ, cache control, and multi-layer caching strategies.
 *
 * Run: node 10_persisted_queries.js
 */

const crypto = require('crypto');

// ============================================================
// Exercise 1: Automatic Persisted Queries (APQ)
// Implement the APQ protocol with hash-based query storage.
// ============================================================

class APQStore {
  constructor() {
    this.queries = new Map(); // hash -> query string
    this.stats = { hits: 0, misses: 0, stores: 0 };
  }

  // TODO: Implement hash(query) — SHA-256 hash of query string
  hash(query) {
    return crypto.createHash('sha256').update(query).digest('hex');
  }

  // TODO: Implement execute(request)
  // request: { query?: string, extensions?: { persistedQuery: { sha256Hash: string } } }
  // APQ protocol:
  //   1. If hash only (no query): look up stored query → hit or PersistedQueryNotFound
  //   2. If hash + query: verify hash matches, store, execute
  //   3. If query only: execute normally (no APQ)
  execute(request) {
    const { query, extensions } = request;
    const hashInfo = extensions && extensions.persistedQuery;

    if (hashInfo && !query) {
      // Hash only: lookup
      const stored = this.queries.get(hashInfo.sha256Hash);
      if (stored) {
        this.stats.hits++;
        return { status: 'ok', query: stored, source: 'persisted' };
      }
      this.stats.misses++;
      return { status: 'error', code: 'PERSISTED_QUERY_NOT_FOUND' };
    }

    if (hashInfo && query) {
      // Hash + query: verify and store
      const computed = this.hash(query);
      if (computed !== hashInfo.sha256Hash) {
        return { status: 'error', code: 'HASH_MISMATCH' };
      }
      this.queries.set(computed, query);
      this.stats.stores++;
      return { status: 'ok', query, source: 'stored' };
    }

    // Regular query (no APQ)
    return { status: 'ok', query, source: 'direct' };
  }
}


// ============================================================
// Exercise 2: Cache Control Directives
// Implement field-level cache hints and compute overall cache policy.
// ============================================================

class CachePolicy {
  constructor() {
    this.hints = []; // { maxAge, scope }
  }

  // TODO: Implement addHint(maxAge, scope)
  addHint(maxAge, scope = 'PUBLIC') {
    this.hints.push({ maxAge, scope });
  }

  // TODO: Implement computeOverall()
  // Overall policy: minimum maxAge, PRIVATE if any field is PRIVATE
  computeOverall() {
    if (this.hints.length === 0) return { maxAge: 0, scope: 'PUBLIC' };

    let minAge = Infinity;
    let isPrivate = false;

    for (const hint of this.hints) {
      minAge = Math.min(minAge, hint.maxAge);
      if (hint.scope === 'PRIVATE') isPrivate = true;
    }

    return {
      maxAge: minAge === Infinity ? 0 : minAge,
      scope: isPrivate ? 'PRIVATE' : 'PUBLIC',
    };
  }

  // TODO: Implement toCacheControlHeader()
  toCacheControlHeader() {
    const policy = this.computeOverall();
    const scope = policy.scope === 'PRIVATE' ? 'private' : 'public';
    if (policy.maxAge === 0) return 'no-store';
    return `${scope}, max-age=${policy.maxAge}`;
  }
}


// ============================================================
// Exercise 3: Multi-Layer Response Cache
// Implement a response cache with TTL and invalidation.
// ============================================================

class ResponseCache {
  constructor() {
    this.store = new Map(); // key -> { data, expiry, tags }
  }

  // TODO: Implement set(key, data, maxAge, tags)
  set(key, data, maxAge, tags = []) {
    this.store.set(key, {
      data,
      expiry: Date.now() + maxAge * 1000,
      tags,
      createdAt: Date.now(),
    });
  }

  // TODO: Implement get(key) — returns data or null if expired
  get(key) {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiry) {
      this.store.delete(key);
      return null;
    }
    return entry.data;
  }

  // TODO: Implement invalidateByTag(tag) — remove all entries with this tag
  invalidateByTag(tag) {
    let count = 0;
    for (const [key, entry] of this.store) {
      if (entry.tags.includes(tag)) {
        this.store.delete(key);
        count++;
      }
    }
    return count;
  }

  // TODO: Implement invalidateByKey(key)
  invalidateByKey(key) {
    return this.store.delete(key);
  }

  // TODO: Implement getStats() — entries count, expired count
  getStats() {
    let active = 0;
    let expired = 0;
    const now = Date.now();
    for (const [, entry] of this.store) {
      if (now > entry.expiry) expired++;
      else active++;
    }
    return { active, expired, total: this.store.size };
  }
}


// ============================================================
// Exercise 4: Cache Key Generator
// Implement deterministic cache key generation for GraphQL operations.
// ============================================================

function generateCacheKey(operation) {
  // TODO: Generate a deterministic cache key from:
  // - operation name
  // - query hash
  // - sorted variables
  // - optional session ID (for PRIVATE scope)

  const parts = [];

  // Operation name
  parts.push(operation.operationName || 'anonymous');

  // Query hash (short)
  const queryHash = crypto
    .createHash('md5')
    .update(operation.query)
    .digest('hex')
    .slice(0, 8);
  parts.push(queryHash);

  // Sorted variables
  if (operation.variables && Object.keys(operation.variables).length > 0) {
    const sorted = Object.keys(operation.variables).sort();
    const varStr = sorted.map((k) => `${k}:${JSON.stringify(operation.variables[k])}`).join(',');
    parts.push(crypto.createHash('md5').update(varStr).digest('hex').slice(0, 8));
  }

  // Session scope
  if (operation.sessionId) {
    parts.push(`s:${operation.sessionId}`);
  }

  return parts.join(':');
}


// ============================================================
// Test all exercises
// ============================================================

console.log('=== Exercise 1: APQ Store ===\n');

const apq = new APQStore();
const query = '{ products { id name price } }';
const hash = apq.hash(query);

// Step 1: Client sends hash only (first request)
const r1 = apq.execute({ extensions: { persistedQuery: { sha256Hash: hash } } });
console.log(`Hash-only (first): ${r1.code} (expected PERSISTED_QUERY_NOT_FOUND): ${r1.code === 'PERSISTED_QUERY_NOT_FOUND' ? 'PASS' : 'FAIL'}`);

// Step 2: Client retries with hash + query
const r2 = apq.execute({ query, extensions: { persistedQuery: { sha256Hash: hash } } });
console.log(`Hash+query: ${r2.source} (expected stored): ${r2.source === 'stored' ? 'PASS' : 'FAIL'}`);

// Step 3: Subsequent requests with hash only
const r3 = apq.execute({ extensions: { persistedQuery: { sha256Hash: hash } } });
console.log(`Hash-only (cached): ${r3.source} (expected persisted): ${r3.source === 'persisted' ? 'PASS' : 'FAIL'}`);

// Verify hash mismatch detection
const r4 = apq.execute({ query, extensions: { persistedQuery: { sha256Hash: 'wrong' } } });
console.log(`Hash mismatch: ${r4.code} (expected HASH_MISMATCH): ${r4.code === 'HASH_MISMATCH' ? 'PASS' : 'FAIL'}`);

console.log(`Stats: hits=${apq.stats.hits}, misses=${apq.stats.misses}, stores=${apq.stats.stores}`);

console.log('\n=== Exercise 2: Cache Control ===\n');

const policy = new CachePolicy();
policy.addHint(300, 'PUBLIC');   // products: 5 min
policy.addHint(10, 'PRIVATE');   // inventory: 10s, private
policy.addHint(300, 'PUBLIC');   // categories: 5 min

const overall = policy.computeOverall();
console.log(`Overall maxAge: ${overall.maxAge} (expected 10): ${overall.maxAge === 10 ? 'PASS' : 'FAIL'}`);
console.log(`Overall scope: ${overall.scope} (expected PRIVATE): ${overall.scope === 'PRIVATE' ? 'PASS' : 'FAIL'}`);
console.log(`Header: ${policy.toCacheControlHeader()} (expected "private, max-age=10"): ${policy.toCacheControlHeader() === 'private, max-age=10' ? 'PASS' : 'FAIL'}`);

const publicPolicy = new CachePolicy();
publicPolicy.addHint(300, 'PUBLIC');
publicPolicy.addHint(600, 'PUBLIC');
console.log(`Public header: ${publicPolicy.toCacheControlHeader()} (expected "public, max-age=300"): ${publicPolicy.toCacheControlHeader() === 'public, max-age=300' ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 3: Response Cache ===\n');

const cache = new ResponseCache();

cache.set('products:all', [{ id: 1, name: 'Laptop' }], 300, ['products']);
cache.set('product:1', { id: 1, name: 'Laptop' }, 300, ['products', 'product:1']);
cache.set('user:me', { id: 'u1', name: 'Alice' }, 60, ['users']);

const cached = cache.get('products:all');
console.log(`Cache hit: ${cached ? 'YES' : 'NO'} (expected YES): ${cached ? 'PASS' : 'FAIL'}`);

// Invalidate all product caches
const invalidated = cache.invalidateByTag('products');
console.log(`Invalidated by tag: ${invalidated} (expected 2): ${invalidated === 2 ? 'PASS' : 'FAIL'}`);

console.log(`Products after invalidate: ${cache.get('products:all') ? 'HIT' : 'MISS'} (expected MISS): ${!cache.get('products:all') ? 'PASS' : 'FAIL'}`);
console.log(`User cache intact: ${cache.get('user:me') ? 'HIT' : 'MISS'} (expected HIT): ${cache.get('user:me') ? 'PASS' : 'FAIL'}`);

const stats = cache.getStats();
console.log(`Cache stats: active=${stats.active}, total=${stats.total}`);

console.log('\n=== Exercise 4: Cache Key Generation ===\n');

const key1 = generateCacheKey({
  operationName: 'GetProducts',
  query: '{ products { id name } }',
  variables: { category: 'electronics', limit: 10 },
});
console.log(`Key 1: ${key1}`);

const key2 = generateCacheKey({
  operationName: 'GetProducts',
  query: '{ products { id name } }',
  variables: { limit: 10, category: 'electronics' }, // same vars, different order
});
console.log(`Key 2: ${key2}`);
console.log(`Same vars, different order = same key: ${key1 === key2 ? 'PASS' : 'FAIL'}`);

const key3 = generateCacheKey({
  operationName: 'GetProfile',
  query: '{ me { name email } }',
  variables: {},
  sessionId: 'sess-abc',
});
console.log(`Private key: ${key3} (contains session): ${key3.includes('s:sess-abc') ? 'PASS' : 'FAIL'}`);

console.log('\nAll exercises completed!');
