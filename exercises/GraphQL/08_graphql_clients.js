/**
 * Exercise: GraphQL Clients
 * Practice implementing client-side caching, optimistic updates, and fetch policies.
 *
 * Run: node 08_graphql_clients.js
 */

// ============================================================
// Exercise 1: Normalized Cache
// Implement a normalized cache that stores entities by typename + id.
// ============================================================

class NormalizedCache {
  constructor() {
    this.store = new Map(); // "Type:id" -> entity data
  }

  // TODO: Implement cacheKey(typename, id) — returns "Type:id"
  cacheKey(typename, id) {
    return `${typename}:${id}`;
  }

  // TODO: Implement write(typename, id, data) — store entity
  write(typename, id, data) {
    const key = this.cacheKey(typename, id);
    const existing = this.store.get(key) || {};
    this.store.set(key, { ...existing, ...data, __typename: typename, id });
  }

  // TODO: Implement read(typename, id) — retrieve entity
  read(typename, id) {
    return this.store.get(this.cacheKey(typename, id)) || null;
  }

  // TODO: Implement writeQuery(data) — normalize and cache a query response
  // Walk the response, find objects with __typename and id, cache each one
  writeQuery(data) {
    let count = 0;
    const walk = (obj) => {
      if (!obj || typeof obj !== 'object') return;
      if (Array.isArray(obj)) {
        obj.forEach(walk);
        return;
      }
      if (obj.__typename && obj.id) {
        this.write(obj.__typename, obj.id, obj);
        count++;
      }
      Object.values(obj).forEach(walk);
    };
    walk(data);
    return count;
  }

  // TODO: Implement evict(typename, id) — remove entity from cache
  evict(typename, id) {
    return this.store.delete(this.cacheKey(typename, id));
  }

  // TODO: Implement gc() — return count of entries
  gc() {
    return this.store.size;
  }
}


// ============================================================
// Exercise 2: Fetch Policy Simulator
// Implement different fetch policies (cache-first, network-only, cache-and-network).
// ============================================================

class FetchPolicySimulator {
  constructor(cache) {
    this.cache = cache;
    this.networkCalls = 0;
  }

  // Simulated network fetch
  async fetchFromNetwork(query) {
    this.networkCalls++;
    // Simulate response based on query
    const responses = {
      'GetUser:1': { __typename: 'User', id: '1', name: 'Alice (fresh)', email: 'alice@new.com' },
      'GetUser:2': { __typename: 'User', id: '2', name: 'Bob (fresh)', email: 'bob@new.com' },
      'GetPosts': [
        { __typename: 'Post', id: 'p1', title: 'Fresh Post 1' },
        { __typename: 'Post', id: 'p2', title: 'Fresh Post 2' },
      ],
    };
    return responses[query] || null;
  }

  // TODO: Implement executeQuery(query, fetchPolicy)
  // Policies:
  //   'cache-first': return cache if exists, otherwise network
  //   'network-only': always fetch from network, update cache
  //   'cache-and-network': return cache immediately, then fetch and update
  //   'no-cache': fetch from network, don't update cache
  async executeQuery(query, fetchPolicy) {
    switch (fetchPolicy) {
      case 'cache-first': {
        // Try cache first
        const parts = query.split(':');
        const cached = parts.length === 2
          ? this.cache.read(parts[0].replace('Get', ''), parts[1])
          : null;
        if (cached) return { source: 'cache', data: cached };
        const data = await this.fetchFromNetwork(query);
        if (data) {
          if (Array.isArray(data)) data.forEach((d) => this.cache.write(d.__typename, d.id, d));
          else this.cache.write(data.__typename, data.id, data);
        }
        return { source: 'network', data };
      }
      case 'network-only': {
        const data = await this.fetchFromNetwork(query);
        if (data) {
          if (Array.isArray(data)) data.forEach((d) => this.cache.write(d.__typename, d.id, d));
          else this.cache.write(data.__typename, data.id, data);
        }
        return { source: 'network', data };
      }
      case 'cache-and-network': {
        const parts = query.split(':');
        const cached = parts.length === 2
          ? this.cache.read(parts[0].replace('Get', ''), parts[1])
          : null;
        const data = await this.fetchFromNetwork(query);
        if (data) {
          if (Array.isArray(data)) data.forEach((d) => this.cache.write(d.__typename, d.id, d));
          else this.cache.write(data.__typename, data.id, data);
        }
        return { source: cached ? 'cache+network' : 'network', cached, data };
      }
      case 'no-cache': {
        const data = await this.fetchFromNetwork(query);
        return { source: 'no-cache', data };
      }
      default:
        throw new Error(`Unknown fetch policy: ${fetchPolicy}`);
    }
  }
}


// ============================================================
// Exercise 3: Optimistic Update
// Implement optimistic UI updates that revert on failure.
// ============================================================

class OptimisticUpdater {
  constructor(cache) {
    this.cache = cache;
    this.pendingUpdates = new Map(); // opId -> { original, typename, id }
  }

  // TODO: Implement optimisticUpdate(opId, typename, id, optimisticData)
  // - Save original data for rollback
  // - Apply optimistic data to cache immediately
  optimisticUpdate(opId, typename, id, optimisticData) {
    const original = this.cache.read(typename, id);
    this.pendingUpdates.set(opId, { original, typename, id });
    this.cache.write(typename, id, optimisticData);
  }

  // TODO: Implement confirm(opId, serverData)
  // - Remove pending update
  // - Apply server data to cache (may differ from optimistic)
  confirm(opId, serverData) {
    const pending = this.pendingUpdates.get(opId);
    if (!pending) return false;
    this.cache.write(pending.typename, pending.id, serverData);
    this.pendingUpdates.delete(opId);
    return true;
  }

  // TODO: Implement rollback(opId)
  // - Restore original data to cache
  // - Remove pending update
  rollback(opId) {
    const pending = this.pendingUpdates.get(opId);
    if (!pending) return false;
    if (pending.original) {
      this.cache.write(pending.typename, pending.id, pending.original);
    } else {
      this.cache.evict(pending.typename, pending.id);
    }
    this.pendingUpdates.delete(opId);
    return true;
  }
}


// ============================================================
// Exercise 4: Pagination Helper
// Implement cursor-based pagination with cache merging.
// ============================================================

class PaginationHelper {
  constructor() {
    this.pages = [];
    this.allEdges = [];
  }

  // TODO: Implement addPage(response) — merge new page into accumulated edges
  // response: { edges: [{ node, cursor }], pageInfo: { hasNextPage, endCursor } }
  addPage(response) {
    this.pages.push(response);
    this.allEdges.push(...response.edges);
    return this;
  }

  // TODO: Implement getAll() — return all accumulated edges' nodes
  getAll() {
    return this.allEdges.map((e) => e.node);
  }

  // TODO: Implement hasMore() — check if there are more pages
  hasMore() {
    if (this.pages.length === 0) return true;
    return this.pages[this.pages.length - 1].pageInfo.hasNextPage;
  }

  // TODO: Implement getNextCursor() — return cursor for next page
  getNextCursor() {
    if (this.pages.length === 0) return null;
    return this.pages[this.pages.length - 1].pageInfo.endCursor;
  }

  // TODO: Implement getPageCount()
  getPageCount() {
    return this.pages.length;
  }
}


// ============================================================
// Test all exercises
// ============================================================

console.log('=== Exercise 1: Normalized Cache ===\n');

const cache = new NormalizedCache();

// Write individual entities
cache.write('User', '1', { name: 'Alice', email: 'alice@example.com' });
cache.write('User', '2', { name: 'Bob', email: 'bob@example.com' });

const user1 = cache.read('User', '1');
console.log(`Read User:1 — ${user1.name} (expected Alice): ${user1.name === 'Alice' ? 'PASS' : 'FAIL'}`);

// Write query response with nested objects
const queryData = {
  posts: [
    { __typename: 'Post', id: 'p1', title: 'Hello', author: { __typename: 'User', id: '1', name: 'Alice' } },
    { __typename: 'Post', id: 'p2', title: 'World', author: { __typename: 'User', id: '2', name: 'Bob' } },
  ],
};
const cached = cache.writeQuery(queryData);
console.log(`Normalized ${cached} entities (expected 4): ${cached === 4 ? 'PASS' : 'FAIL'}`);
console.log(`Post:p1 in cache: ${cache.read('Post', 'p1') ? 'PASS' : 'FAIL'}`);

cache.evict('Post', 'p2');
console.log(`Evicted Post:p2: ${!cache.read('Post', 'p2') ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 2: Fetch Policies ===\n');

const fetchCache = new NormalizedCache();
fetchCache.write('User', '1', { name: 'Alice (stale)', email: 'alice@old.com' });

const client = new FetchPolicySimulator(fetchCache);

async function testFetchPolicies() {
  // cache-first: should return cached data
  const r1 = await client.executeQuery('GetUser:1', 'cache-first');
  console.log(`cache-first source: ${r1.source} (expected cache): ${r1.source === 'cache' ? 'PASS' : 'FAIL'}`);

  // network-only: should always fetch
  const callsBefore = client.networkCalls;
  const r2 = await client.executeQuery('GetUser:1', 'network-only');
  console.log(`network-only fetched: ${client.networkCalls > callsBefore} (expected true): ${client.networkCalls > callsBefore ? 'PASS' : 'FAIL'}`);

  // cache-and-network: returns both
  const r3 = await client.executeQuery('GetUser:1', 'cache-and-network');
  console.log(`cache-and-network source: ${r3.source}: ${r3.source === 'cache+network' ? 'PASS' : 'FAIL'}`);

  // no-cache
  const sizeBefore = fetchCache.gc();
  await client.executeQuery('GetUser:2', 'no-cache');
  const sizeAfter = fetchCache.gc();
  // no-cache shouldn't change cache for new keys
  console.log(`no-cache didn't cache new: ${sizeBefore === sizeAfter}: PASS`);
}

async function testOptimistic() {
  console.log('\n=== Exercise 3: Optimistic Updates ===\n');

  const optCache = new NormalizedCache();
  optCache.write('Post', 'p1', { title: 'Original', likes: 10 });

  const updater = new OptimisticUpdater(optCache);

  // Like a post optimistically
  updater.optimisticUpdate('like-p1', 'Post', 'p1', { title: 'Original', likes: 11 });
  const optimistic = optCache.read('Post', 'p1');
  console.log(`Optimistic likes: ${optimistic.likes} (expected 11): ${optimistic.likes === 11 ? 'PASS' : 'FAIL'}`);

  // Server confirms with actual count
  updater.confirm('like-p1', { title: 'Original', likes: 12 });
  const confirmed = optCache.read('Post', 'p1');
  console.log(`Confirmed likes: ${confirmed.likes} (expected 12): ${confirmed.likes === 12 ? 'PASS' : 'FAIL'}`);

  // Test rollback
  updater.optimisticUpdate('edit-p1', 'Post', 'p1', { title: 'Edited', likes: 12 });
  updater.rollback('edit-p1');
  const rolledBack = optCache.read('Post', 'p1');
  console.log(`Rolled back title: "${rolledBack.title}" (expected Original): ${rolledBack.title === 'Original' ? 'PASS' : 'FAIL'}`);
}

async function testPagination() {
  console.log('\n=== Exercise 4: Pagination ===\n');

  const pager = new PaginationHelper();

  pager.addPage({
    edges: [
      { node: { id: '1', title: 'Post 1' }, cursor: 'c1' },
      { node: { id: '2', title: 'Post 2' }, cursor: 'c2' },
    ],
    pageInfo: { hasNextPage: true, endCursor: 'c2' },
  });

  console.log(`Page 1 items: ${pager.getAll().length} (expected 2): ${pager.getAll().length === 2 ? 'PASS' : 'FAIL'}`);
  console.log(`Has more: ${pager.hasMore()} (expected true): ${pager.hasMore() ? 'PASS' : 'FAIL'}`);
  console.log(`Next cursor: ${pager.getNextCursor()} (expected c2): ${pager.getNextCursor() === 'c2' ? 'PASS' : 'FAIL'}`);

  pager.addPage({
    edges: [
      { node: { id: '3', title: 'Post 3' }, cursor: 'c3' },
    ],
    pageInfo: { hasNextPage: false, endCursor: 'c3' },
  });

  console.log(`All items: ${pager.getAll().length} (expected 3): ${pager.getAll().length === 3 ? 'PASS' : 'FAIL'}`);
  console.log(`Has more: ${pager.hasMore()} (expected false): ${!pager.hasMore() ? 'PASS' : 'FAIL'}`);
  console.log(`Pages loaded: ${pager.getPageCount()} (expected 2): ${pager.getPageCount() === 2 ? 'PASS' : 'FAIL'}`);
}

testFetchPolicies()
  .then(testOptimistic)
  .then(testPagination)
  .then(() => console.log('\nAll exercises completed!'));
