/**
 * GraphQL DataLoader — Solving the N+1 Problem
 * Demonstrates: DataLoader batching and caching per request.
 *
 * Run: npm install @apollo/server graphql dataloader
 *      node 02_dataloader.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const DataLoader = require('dataloader');

// --- Schema ---

const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    author: User!
  }

  type Query {
    posts: [Post!]!
    users: [User!]!
  }
`;

// --- Simulated Database ---

const usersDB = {
  '1': { id: '1', name: 'Alice' },
  '2': { id: '2', name: 'Bob' },
  '3': { id: '3', name: 'Charlie' },
};

const postsDB = [
  { id: '1', title: 'Post A', authorId: '1' },
  { id: '2', title: 'Post B', authorId: '1' },
  { id: '3', title: 'Post C', authorId: '2' },
  { id: '4', title: 'Post D', authorId: '2' },
  { id: '5', title: 'Post E', authorId: '3' },
];

// Track database calls to demonstrate the N+1 problem
let dbCallCount = 0;

function resetDbCallCount() {
  dbCallCount = 0;
}

// Simulates a batch database query
async function fetchUsersByIds(ids) {
  dbCallCount++;
  console.log(`  DB call #${dbCallCount}: SELECT * FROM users WHERE id IN (${ids.join(', ')})`);
  // DataLoader requires results in the same order as the input keys
  return ids.map((id) => usersDB[id] || null);
}

async function fetchPostsByAuthorIds(authorIds) {
  dbCallCount++;
  console.log(`  DB call #${dbCallCount}: SELECT * FROM posts WHERE author_id IN (${authorIds.join(', ')})`);
  // Return an array of arrays (one per key)
  return authorIds.map((authorId) =>
    postsDB.filter((p) => p.authorId === authorId)
  );
}

// --- DataLoader Factory (create per request!) ---

function createLoaders() {
  return {
    userLoader: new DataLoader((ids) => fetchUsersByIds(ids)),
    postsByAuthorLoader: new DataLoader((authorIds) => fetchPostsByAuthorIds(authorIds)),
  };
}

// --- Resolvers ---

const resolvers = {
  Query: {
    posts: () => {
      resetDbCallCount();
      console.log('\n--- Executing posts query ---');
      return postsDB;
    },
    users: () => {
      resetDbCallCount();
      console.log('\n--- Executing users query ---');
      return Object.values(usersDB);
    },
  },

  Post: {
    // WITHOUT DataLoader: N+1 queries
    // author: (post) => fetchUsersByIds([post.authorId]).then(r => r[0]),

    // WITH DataLoader: batched into a single query
    author: (post, _, { loaders }) => {
      return loaders.userLoader.load(post.authorId);
    },
  },

  User: {
    posts: (user, _, { loaders }) => {
      return loaders.postsByAuthorLoader.load(user.id);
    },
  },
};

// --- Start Server ---

async function main() {
  const server = new ApolloServer({ typeDefs, resolvers });

  const { url } = await startStandaloneServer(server, {
    listen: { port: 4000 },
    // Create fresh DataLoaders for each request
    context: async () => ({
      loaders: createLoaders(),
    }),
  });

  console.log(`GraphQL server at ${url}`);
  console.log('\nTry this query to see DataLoader batching in action:');
  console.log(`
  query {
    posts {
      title
      author {
        name
      }
    }
  }

  # Without DataLoader: 1 (posts) + 5 (author per post) = 6 queries
  # With DataLoader:    1 (posts) + 1 (batched authors)  = 2 queries
  `);
}

main();
