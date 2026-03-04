/**
 * GraphQL REST-to-GraphQL Migration
 * Demonstrates: REST wrapper resolvers, data source pattern, field mapping.
 *
 * Run: npm install @apollo/server graphql
 *      node 11_rest_migration.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const http = require('http');

// ============================================================
// Phase 1: Simulated REST API (existing backend)
// ============================================================

const restDB = {
  users: [
    { user_id: 1, full_name: 'Alice Smith', email_address: 'alice@example.com', created_at: '2024-01-15' },
    { user_id: 2, full_name: 'Bob Jones', email_address: 'bob@example.com', created_at: '2024-03-20' },
  ],
  products: [
    { product_id: 101, product_name: 'Laptop', unit_price: 999.99, stock_count: 50, category_id: 1 },
    { product_id: 102, product_name: 'Mouse', unit_price: 29.99, stock_count: 200, category_id: 2 },
    { product_id: 103, product_name: 'Keyboard', unit_price: 79.99, stock_count: 150, category_id: 2 },
  ],
  orders: [
    { order_id: 1001, user_id: 1, items: [{ product_id: 101, qty: 1 }, { product_id: 102, qty: 2 }], total_amount: 1059.97, order_status: 'shipped' },
    { order_id: 1002, user_id: 2, items: [{ product_id: 103, qty: 1 }], total_amount: 79.99, order_status: 'pending' },
  ],
};

// Simulated REST API client (wraps the database)
class RESTClient {
  constructor() {
    this.callCount = 0;
  }

  async get(endpoint) {
    this.callCount++;
    console.log(`  REST GET ${endpoint} (call #${this.callCount})`);

    // Simulate REST endpoints
    if (endpoint === '/api/users') return restDB.users;
    if (endpoint.startsWith('/api/users/')) {
      const id = parseInt(endpoint.split('/').pop());
      return restDB.users.find((u) => u.user_id === id) || null;
    }
    if (endpoint === '/api/products') return restDB.products;
    if (endpoint.startsWith('/api/products/')) {
      const id = parseInt(endpoint.split('/').pop());
      return restDB.products.find((p) => p.product_id === id) || null;
    }
    if (endpoint.startsWith('/api/users/') && endpoint.endsWith('/orders')) {
      const id = parseInt(endpoint.split('/')[3]);
      return restDB.orders.filter((o) => o.user_id === id);
    }
    return null;
  }

  resetCount() { this.callCount = 0; }
}

// ============================================================
// Phase 2: GraphQL Schema (wrapping REST)
// ============================================================

const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!         # mapped from full_name
    email: String!        # mapped from email_address
    createdAt: String!    # mapped from created_at
    orders: [Order!]!     # fetched from /api/users/:id/orders
  }

  type Product {
    id: ID!
    name: String!         # mapped from product_name
    price: Float!         # mapped from unit_price
    inStock: Boolean!     # computed from stock_count > 0
    stockCount: Int!      # mapped from stock_count
  }

  type Order {
    id: ID!
    items: [OrderItem!]!
    total: Float!         # mapped from total_amount
    status: String!       # mapped from order_status
  }

  type OrderItem {
    product: Product!
    quantity: Int!
  }

  type Query {
    users: [User!]!
    user(id: ID!): User
    products: [Product!]!
    product(id: ID!): Product
  }
`;

// ============================================================
// Phase 3: Field Mapping Resolvers
// ============================================================

// Map REST snake_case fields to GraphQL camelCase
function mapUser(restUser) {
  if (!restUser) return null;
  return {
    id: String(restUser.user_id),
    name: restUser.full_name,
    email: restUser.email_address,
    createdAt: restUser.created_at,
    _userId: restUser.user_id,  // internal ref for nested resolvers
  };
}

function mapProduct(restProduct) {
  if (!restProduct) return null;
  return {
    id: String(restProduct.product_id),
    name: restProduct.product_name,
    price: restProduct.unit_price,
    inStock: restProduct.stock_count > 0,
    stockCount: restProduct.stock_count,
  };
}

function mapOrder(restOrder) {
  if (!restOrder) return null;
  return {
    id: String(restOrder.order_id),
    total: restOrder.total_amount,
    status: restOrder.order_status,
    _items: restOrder.items,  // raw items for nested resolution
  };
}

const resolvers = {
  Query: {
    users: async (_, __, { rest }) => {
      const data = await rest.get('/api/users');
      return data.map(mapUser);
    },
    user: async (_, { id }, { rest }) => {
      const data = await rest.get(`/api/users/${id}`);
      return mapUser(data);
    },
    products: async (_, __, { rest }) => {
      const data = await rest.get('/api/products');
      return data.map(mapProduct);
    },
    product: async (_, { id }, { rest }) => {
      const data = await rest.get(`/api/products/${id}`);
      return mapProduct(data);
    },
  },

  User: {
    orders: async (user, _, { rest }) => {
      const data = await rest.get(`/api/users/${user._userId}/orders`);
      return data.map(mapOrder);
    },
  },

  Order: {
    items: async (order, _, { rest }) => {
      // Resolve product references in order items
      const items = [];
      for (const item of order._items) {
        const product = await rest.get(`/api/products/${item.product_id}`);
        items.push({ product: mapProduct(product), quantity: item.qty });
      }
      return items;
    },
  },
};

// ============================================================
// Main
// ============================================================

async function main() {
  const rest = new RESTClient();

  const server = new ApolloServer({ typeDefs, resolvers });

  const { url } = await startStandaloneServer(server, {
    listen: { port: 4000 },
    context: async () => ({ rest }),
  });

  console.log(`GraphQL server (wrapping REST) at ${url}\n`);

  // Demo: show field mapping
  console.log('=== Field Mapping: REST → GraphQL ===\n');
  console.log('REST:    user_id, full_name, email_address, created_at');
  console.log('GraphQL: id,      name,      email,         createdAt\n');
  console.log('REST:    product_id, product_name, unit_price, stock_count');
  console.log('GraphQL: id,         name,         price,      stockCount + inStock\n');

  // Demo: resolve a query
  console.log('=== Query Resolution Trace ===\n');
  rest.resetCount();

  const usersData = await rest.get('/api/users');
  const user = mapUser(usersData[0]);
  console.log(`\nMapped user: ${JSON.stringify(user, null, 2)}`);

  const ordersData = await rest.get(`/api/users/${user._userId}/orders`);
  const order = mapOrder(ordersData[0]);
  console.log(`\nMapped order: ${JSON.stringify(order, null, 2)}`);

  console.log(`\nTotal REST calls: ${rest.callCount}`);
  console.log('\nTip: Use DataLoader to batch product lookups in Order.items');

  console.log('\n=== Migration Strategy ===');
  console.log('Phase 1: Wrap REST endpoints with GraphQL resolvers (current)');
  console.log('Phase 2: Move frequently-accessed data to direct DB queries');
  console.log('Phase 3: Deprecate REST endpoints as clients migrate');
  console.log('Phase 4: Remove REST layer entirely');
}

main();
