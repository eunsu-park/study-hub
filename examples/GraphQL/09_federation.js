/**
 * GraphQL Federation — Federated Subgraph Example
 * Demonstrates: Apollo Federation 2 with entity references and cross-subgraph resolution.
 *
 * Run: npm install @apollo/server @apollo/subgraph graphql
 *      node 09_federation.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const { buildSubgraphSchema } = require('@apollo/subgraph');
const { gql } = require('graphql-tag');

// ============================================================
// Subgraph 1: Users Service
// ============================================================

const usersTypeDefs = gql`
  extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
    import: ["@key", "@shareable"])

  type User @key(fields: "id") {
    id: ID!
    name: String!
    email: String!
  }

  type Query {
    me: User
    user(id: ID!): User
  }
`;

const usersDB = [
  { id: '1', name: 'Alice', email: 'alice@shop.com' },
  { id: '2', name: 'Bob', email: 'bob@shop.com' },
  { id: '3', name: 'Charlie', email: 'charlie@shop.com' },
];

const usersResolvers = {
  Query: {
    me: () => usersDB[0],
    user: (_, { id }) => usersDB.find((u) => u.id === id),
  },
  User: {
    // Federation: resolve entity by reference
    __resolveReference: (ref) => {
      console.log(`  [Users] Resolving User#${ref.id}`);
      return usersDB.find((u) => u.id === ref.id);
    },
  },
};

// ============================================================
// Subgraph 2: Products Service
// ============================================================

const productsTypeDefs = gql`
  extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
    import: ["@key", "@external", "@requires"])

  type Product @key(fields: "id") {
    id: ID!
    name: String!
    price: Float!
    weight: Float
  }

  # Extend User from Users subgraph
  type User @key(fields: "id") {
    id: ID!
    purchases: [Product!]!
  }

  type Query {
    products: [Product!]!
    product(id: ID!): Product
  }
`;

const productsDB = [
  { id: 'p1', name: 'Laptop', price: 999.99, weight: 2.5 },
  { id: 'p2', name: 'Phone', price: 699.99, weight: 0.2 },
  { id: 'p3', name: 'Tablet', price: 449.99, weight: 0.5 },
];

const purchasesDB = {
  '1': ['p1', 'p2'],
  '2': ['p3'],
  '3': ['p1', 'p3'],
};

const productsResolvers = {
  Query: {
    products: () => productsDB,
    product: (_, { id }) => productsDB.find((p) => p.id === id),
  },
  Product: {
    __resolveReference: (ref) => {
      console.log(`  [Products] Resolving Product#${ref.id}`);
      return productsDB.find((p) => p.id === ref.id);
    },
  },
  User: {
    purchases: (user) => {
      const ids = purchasesDB[user.id] || [];
      return ids.map((id) => productsDB.find((p) => p.id === id)).filter(Boolean);
    },
  },
};

// ============================================================
// Subgraph 3: Reviews Service
// ============================================================

const reviewsTypeDefs = gql`
  extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
    import: ["@key", "@external"])

  type Review @key(fields: "id") {
    id: ID!
    rating: Int!
    comment: String
    author: User!
    product: Product!
  }

  # Extend Product with reviews
  type Product @key(fields: "id") {
    id: ID!
    reviews: [Review!]!
    averageRating: Float
  }

  # Extend User with reviews
  type User @key(fields: "id") {
    id: ID!
    reviews: [Review!]!
  }
`;

const reviewsDB = [
  { id: 'r1', rating: 5, comment: 'Excellent laptop!', authorId: '1', productId: 'p1' },
  { id: 'r2', rating: 4, comment: 'Great phone', authorId: '2', productId: 'p2' },
  { id: 'r3', rating: 3, comment: 'Decent tablet', authorId: '1', productId: 'p3' },
  { id: 'r4', rating: 5, comment: 'Love it', authorId: '3', productId: 'p1' },
];

const reviewsResolvers = {
  Review: {
    author: (review) => ({ __typename: 'User', id: review.authorId }),
    product: (review) => ({ __typename: 'Product', id: review.productId }),
  },
  Product: {
    reviews: (product) => reviewsDB.filter((r) => r.productId === product.id),
    averageRating: (product) => {
      const pReviews = reviewsDB.filter((r) => r.productId === product.id);
      if (pReviews.length === 0) return null;
      return pReviews.reduce((sum, r) => sum + r.rating, 0) / pReviews.length;
    },
  },
  User: {
    reviews: (user) => reviewsDB.filter((r) => r.authorId === user.id),
  },
};

// ============================================================
// Demo: Build and Start Each Subgraph
// ============================================================

async function startSubgraph(name, typeDefs, resolvers, port) {
  const server = new ApolloServer({
    schema: buildSubgraphSchema({ typeDefs, resolvers }),
  });

  const { url } = await startStandaloneServer(server, { listen: { port } });
  console.log(`${name} subgraph at ${url}`);
  return url;
}

async function main() {
  console.log('=== Federation Subgraph Demo ===\n');

  // In production, these run as separate services
  // Apollo Router composes them into a supergraph
  await startSubgraph('Users', usersTypeDefs, usersResolvers, 4001);
  await startSubgraph('Products', productsTypeDefs, productsResolvers, 4002);
  await startSubgraph('Reviews', reviewsTypeDefs, reviewsResolvers, 4003);

  console.log('\n--- Supergraph Query (would span all subgraphs) ---');
  console.log(`
  query {
    me {
      name               # from Users subgraph
      email              # from Users subgraph
      purchases {        # from Products subgraph
        name
        price
        reviews {        # from Reviews subgraph
          rating
          comment
        }
        averageRating    # from Reviews subgraph
      }
      reviews {          # from Reviews subgraph
        rating
        product {
          name           # from Products subgraph
        }
      }
    }
  }

  # Router query plan:
  # 1. Fetch User from Users subgraph
  # 2. Parallel: Fetch purchases from Products + reviews from Reviews
  # 3. Fetch product reviews from Reviews subgraph
  `);
}

main();
