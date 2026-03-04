/**
 * GraphQL API Gateway with Federation
 * Demonstrates: Federated supergraph composition, gateway routing, entity resolution.
 *
 * Run: npm install @apollo/server @apollo/subgraph graphql
 *      node 12_api_gateway.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const { buildSubgraphSchema } = require('@apollo/subgraph');
const { gql } = require('graphql-tag');

// ============================================================
// E-Commerce Domain — 4 Federated Subgraphs
// ============================================================

// --- Subgraph: Users ---

const usersSchema = {
  typeDefs: gql`
    extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
      import: ["@key"])

    type User @key(fields: "id") {
      id: ID!
      name: String!
      email: String!
      tier: MembershipTier!
    }

    enum MembershipTier { FREE PREMIUM VIP }

    type Query {
      me: User
      users: [User!]!
    }
  `,
  resolvers: {
    Query: {
      me: () => usersData[0],
      users: () => usersData,
    },
    User: {
      __resolveReference: (ref) => usersData.find((u) => u.id === ref.id),
    },
  },
  data: null,
};

const usersData = [
  { id: 'u1', name: 'Alice', email: 'alice@shop.com', tier: 'VIP' },
  { id: 'u2', name: 'Bob', email: 'bob@shop.com', tier: 'PREMIUM' },
  { id: 'u3', name: 'Charlie', email: 'charlie@shop.com', tier: 'FREE' },
];

// --- Subgraph: Catalog ---

const catalogSchema = {
  typeDefs: gql`
    extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
      import: ["@key"])

    type Product @key(fields: "id") {
      id: ID!
      name: String!
      price: Float!
      category: Category!
      inStock: Boolean!
    }

    type Category @key(fields: "id") {
      id: ID!
      name: String!
      products: [Product!]!
    }

    type Query {
      products(category: String): [Product!]!
      product(id: ID!): Product
      categories: [Category!]!
    }
  `,
  resolvers: {
    Query: {
      products: (_, { category }) => {
        if (category) {
          const cat = categoriesData.find((c) => c.name === category);
          return cat ? productsData.filter((p) => p.categoryId === cat.id) : [];
        }
        return productsData;
      },
      product: (_, { id }) => productsData.find((p) => p.id === id),
      categories: () => categoriesData,
    },
    Product: {
      __resolveReference: (ref) => productsData.find((p) => p.id === ref.id),
      category: (product) => categoriesData.find((c) => c.id === product.categoryId),
    },
    Category: {
      __resolveReference: (ref) => categoriesData.find((c) => c.id === ref.id),
      products: (cat) => productsData.filter((p) => p.categoryId === cat.id),
    },
  },
};

const categoriesData = [
  { id: 'cat1', name: 'Electronics' },
  { id: 'cat2', name: 'Accessories' },
];

const productsData = [
  { id: 'p1', name: 'Laptop', price: 999.99, categoryId: 'cat1', inStock: true },
  { id: 'p2', name: 'Phone', price: 699.99, categoryId: 'cat1', inStock: true },
  { id: 'p3', name: 'USB Cable', price: 9.99, categoryId: 'cat2', inStock: true },
  { id: 'p4', name: 'Keyboard', price: 79.99, categoryId: 'cat2', inStock: false },
];

// --- Subgraph: Orders ---

const ordersSchema = {
  typeDefs: gql`
    extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
      import: ["@key", "@external"])

    type Order @key(fields: "id") {
      id: ID!
      buyer: User!
      items: [OrderItem!]!
      total: Float!
      status: OrderStatus!
      createdAt: String!
    }

    type OrderItem {
      product: Product!
      quantity: Int!
      unitPrice: Float!
    }

    enum OrderStatus { PENDING CONFIRMED SHIPPED DELIVERED CANCELLED }

    # Extend User with orders
    type User @key(fields: "id") {
      id: ID!
      orders: [Order!]!
      totalSpent: Float!
    }

    type Product @key(fields: "id") {
      id: ID!
    }

    type Query {
      order(id: ID!): Order
      recentOrders(limit: Int = 5): [Order!]!
    }

    type Mutation {
      placeOrder(items: [OrderItemInput!]!): Order!
    }

    input OrderItemInput {
      productId: ID!
      quantity: Int!
    }
  `,
  resolvers: {
    Query: {
      order: (_, { id }) => ordersData.find((o) => o.id === id),
      recentOrders: (_, { limit }) => ordersData.slice(0, limit),
    },
    Mutation: {
      placeOrder: (_, { items }) => {
        const order = {
          id: `ord${ordersData.length + 1}`,
          buyerId: 'u1',
          items: items.map((i) => ({ productId: i.productId, quantity: i.quantity, unitPrice: 0 })),
          total: 0,
          status: 'PENDING',
          createdAt: new Date().toISOString(),
        };
        ordersData.push(order);
        return order;
      },
    },
    Order: {
      __resolveReference: (ref) => ordersData.find((o) => o.id === ref.id),
      buyer: (order) => ({ __typename: 'User', id: order.buyerId }),
      items: (order) => order.items.map((i) => ({
        product: { __typename: 'Product', id: i.productId },
        quantity: i.quantity,
        unitPrice: i.unitPrice,
      })),
    },
    User: {
      orders: (user) => ordersData.filter((o) => o.buyerId === user.id),
      totalSpent: (user) => ordersData
        .filter((o) => o.buyerId === user.id)
        .reduce((sum, o) => sum + o.total, 0),
    },
  },
};

const ordersData = [
  { id: 'ord1', buyerId: 'u1', items: [{ productId: 'p1', quantity: 1, unitPrice: 999.99 }], total: 999.99, status: 'DELIVERED', createdAt: '2024-11-01' },
  { id: 'ord2', buyerId: 'u1', items: [{ productId: 'p3', quantity: 3, unitPrice: 9.99 }], total: 29.97, status: 'SHIPPED', createdAt: '2024-12-15' },
  { id: 'ord3', buyerId: 'u2', items: [{ productId: 'p2', quantity: 1, unitPrice: 699.99 }, { productId: 'p4', quantity: 1, unitPrice: 79.99 }], total: 779.98, status: 'PENDING', createdAt: '2025-01-05' },
];

// --- Subgraph: Reviews ---

const reviewsSchema = {
  typeDefs: gql`
    extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
      import: ["@key"])

    type Review @key(fields: "id") {
      id: ID!
      rating: Int!
      title: String!
      body: String
      author: User!
      product: Product!
    }

    type User @key(fields: "id") {
      id: ID!
      reviewCount: Int!
    }

    type Product @key(fields: "id") {
      id: ID!
      reviews: [Review!]!
      averageRating: Float
    }

    type Query {
      review(id: ID!): Review
    }

    type Mutation {
      addReview(productId: ID!, rating: Int!, title: String!, body: String): Review!
    }
  `,
  resolvers: {
    Query: {
      review: (_, { id }) => reviewsData.find((r) => r.id === id),
    },
    Mutation: {
      addReview: (_, { productId, rating, title, body }) => {
        const review = { id: `rev${reviewsData.length + 1}`, rating, title, body, authorId: 'u1', productId };
        reviewsData.push(review);
        return review;
      },
    },
    Review: {
      __resolveReference: (ref) => reviewsData.find((r) => r.id === ref.id),
      author: (review) => ({ __typename: 'User', id: review.authorId }),
      product: (review) => ({ __typename: 'Product', id: review.productId }),
    },
    User: {
      reviewCount: (user) => reviewsData.filter((r) => r.authorId === user.id).length,
    },
    Product: {
      reviews: (product) => reviewsData.filter((r) => r.productId === product.id),
      averageRating: (product) => {
        const pReviews = reviewsData.filter((r) => r.productId === product.id);
        if (!pReviews.length) return null;
        return pReviews.reduce((s, r) => s + r.rating, 0) / pReviews.length;
      },
    },
  },
};

const reviewsData = [
  { id: 'rev1', rating: 5, title: 'Best laptop', body: 'Amazing performance', authorId: 'u1', productId: 'p1' },
  { id: 'rev2', rating: 4, title: 'Good phone', body: 'Great camera', authorId: 'u2', productId: 'p2' },
  { id: 'rev3', rating: 3, title: 'Decent cable', body: null, authorId: 'u3', productId: 'p3' },
];

// ============================================================
// Gateway Simulation
// ============================================================

function simulateQueryPlan(query) {
  console.log('Query Plan:');
  const subgraphs = [];

  if (/\bme\b|\busers\b|\bname\b|\bemail\b|\btier\b/.test(query)) subgraphs.push('Users');
  if (/\bproducts?\b|\bcategor/.test(query)) subgraphs.push('Catalog');
  if (/\borders?\b|\btotal\b|\bstatus\b/.test(query)) subgraphs.push('Orders');
  if (/\breview/.test(query)) subgraphs.push('Reviews');

  subgraphs.forEach((sg, i) => {
    const prefix = i === subgraphs.length - 1 ? '└─' : '├─';
    console.log(`  ${prefix} Fetch from ${sg} subgraph`);
  });

  return subgraphs;
}

// ============================================================
// Main
// ============================================================

async function main() {
  console.log('=== E-Commerce API Gateway (Federation) ===\n');

  // Show supergraph structure
  console.log('Subgraphs:');
  console.log('  Users   → User entity, authentication');
  console.log('  Catalog → Product, Category entities');
  console.log('  Orders  → Order entity, User.orders extension');
  console.log('  Reviews → Review entity, Product.reviews extension');

  // Simulate cross-subgraph queries
  console.log('\n--- Query 1: User dashboard ---\n');
  const q1 = `{
    me {
      name email tier
      orders { id total status }
      reviewCount
    }
  }`;
  console.log(q1);
  simulateQueryPlan(q1);

  console.log('\n--- Query 2: Product detail page ---\n');
  const q2 = `{
    product(id: "p1") {
      name price inStock
      category { name }
      reviews { rating title author { name } }
      averageRating
    }
  }`;
  console.log(q2);
  simulateQueryPlan(q2);

  console.log('\n--- Query 3: Order with full details ---\n');
  const q3 = `{
    order(id: "ord3") {
      id status total createdAt
      buyer { name tier }
      items {
        product { name price }
        quantity
      }
    }
  }`;
  console.log(q3);
  simulateQueryPlan(q3);

  // Start a combined demo server
  console.log('\n--- Starting combined demo server ---\n');
  const server = new ApolloServer({
    schema: buildSubgraphSchema({
      typeDefs: catalogSchema.typeDefs,
      resolvers: catalogSchema.resolvers,
    }),
  });

  const { url } = await startStandaloneServer(server, { listen: { port: 4000 } });
  console.log(`Catalog subgraph demo at ${url}`);
  console.log('\nIn production, use Apollo Router to compose all subgraphs');
  console.log('into a single supergraph endpoint.');
}

main();
