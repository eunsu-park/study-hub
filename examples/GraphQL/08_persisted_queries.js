/**
 * GraphQL Persisted Queries and Caching
 * Demonstrates: APQ protocol, cache control directives, multi-layer caching.
 *
 * Run: npm install @apollo/server graphql @apollo/cache-control-types
 *      node 08_persisted_queries.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const crypto = require('crypto');

// --- Schema with Cache Control ---

const typeDefs = `#graphql
  enum CacheControlScope {
    PUBLIC
    PRIVATE
  }

  directive @cacheControl(
    maxAge: Int
    scope: CacheControlScope
  ) on FIELD_DEFINITION | OBJECT

  type Product @cacheControl(maxAge: 300) {
    id: ID!
    name: String!
    price: Float!
    description: String!
    inventory: Int! @cacheControl(maxAge: 10, scope: PRIVATE)
  }

  type User @cacheControl(maxAge: 0, scope: PRIVATE) {
    id: ID!
    name: String!
    email: String!
    cart: [CartItem!]!
  }

  type CartItem {
    productId: ID!
    quantity: Int!
  }

  type Query {
    products: [Product!]!           # PUBLIC, cacheable for 5 min
    product(id: ID!): Product       # PUBLIC, cacheable for 5 min
    me: User                        # PRIVATE, no caching
    categories: [String!]!          # PUBLIC, rarely changes
  }
`;

// --- Mock Data ---

const productsDB = [
  { id: '1', name: 'Laptop', price: 999.99, description: 'Fast laptop', inventory: 50 },
  { id: '2', name: 'Mouse', price: 29.99, description: 'Wireless mouse', inventory: 200 },
  { id: '3', name: 'Keyboard', price: 79.99, description: 'Mechanical keyboard', inventory: 150 },
];

const usersDB = {
  '1': { id: '1', name: 'Alice', email: 'alice@example.com', cart: [{ productId: '1', quantity: 1 }] },
};

const categories = ['Electronics', 'Accessories', 'Peripherals'];

// --- Automatic Persisted Queries (APQ) Simulation ---

const queryStore = new Map();

function simulateAPQ(queryString) {
  const hash = crypto.createHash('sha256').update(queryString).digest('hex');

  // First request: client sends hash only
  if (!queryStore.has(hash)) {
    console.log(`APQ MISS: hash=${hash.slice(0, 16)}... — requesting full query`);
    // Client retries with full query + hash
    queryStore.set(hash, queryString);
    console.log(`APQ STORE: saved query (${queryString.length} bytes → ${hash.length} byte hash)`);
    return { hash, cached: false };
  }

  // Subsequent requests: hash resolves to stored query
  console.log(`APQ HIT: hash=${hash.slice(0, 16)}... — using stored query`);
  return { hash, cached: true, query: queryStore.get(hash) };
}

// --- Resolvers ---

const resolvers = {
  Query: {
    products: () => productsDB,
    product: (_, { id }) => productsDB.find((p) => p.id === id),
    me: () => usersDB['1'],
    categories: () => categories,
  },
};

// --- Response Cache Simulation ---

class SimpleResponseCache {
  constructor() {
    this.cache = new Map();
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }
    console.log(`  CACHE HIT: ${key} (expires in ${Math.round((entry.expiry - Date.now()) / 1000)}s)`);
    return entry.data;
  }

  set(key, data, maxAge) {
    this.cache.set(key, { data, expiry: Date.now() + maxAge * 1000 });
    console.log(`  CACHE SET: ${key} (TTL: ${maxAge}s)`);
  }
}

// --- Main ---

async function main() {
  const server = new ApolloServer({ typeDefs, resolvers });

  const { url } = await startStandaloneServer(server, {
    listen: { port: 4000 },
  });

  console.log(`GraphQL server at ${url}`);

  // Demonstrate APQ protocol
  console.log('\n=== Automatic Persisted Queries Demo ===\n');

  const query1 = '{ products { id name price } }';
  const query2 = '{ categories }';

  // First call: miss, stores query
  simulateAPQ(query1);
  simulateAPQ(query2);

  console.log('');

  // Second call: hit, uses stored hash
  simulateAPQ(query1);
  simulateAPQ(query2);

  // Demonstrate response cache
  console.log('\n=== Response Cache Demo ===\n');

  const cache = new SimpleResponseCache();
  cache.set('products', productsDB, 300);           // 5 min (public)
  cache.set('me', usersDB['1'], 0);                 // no cache (private)
  cache.set('product:1', productsDB[0], 300);        // 5 min

  console.log('');
  cache.get('products');   // hit
  cache.get('me');         // miss (TTL 0)
  cache.get('product:1');  // hit
  cache.get('product:99'); // miss (not stored)

  console.log('\n=== Cache Control Hints ===');
  console.log('Products:  maxAge=300, scope=PUBLIC  → CDN cacheable');
  console.log('Inventory: maxAge=10,  scope=PRIVATE → user-specific, short TTL');
  console.log('User:      maxAge=0,   scope=PRIVATE → no caching');
  console.log('Categories: maxAge=300, scope=PUBLIC → CDN cacheable');
}

main();
