/**
 * GraphQL Fundamentals — Schema and Resolvers
 * Demonstrates: SDL, type definitions, resolver functions.
 *
 * Run: npm install @apollo/server graphql
 *      node 01_schema_resolvers.js
 * Open: http://localhost:4000
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');

// --- Schema Definition Language (SDL) ---

const typeDefs = `#graphql
  # Enum type
  enum Status {
    DRAFT
    PUBLISHED
    ARCHIVED
  }

  # Object type
  type Author {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    content: String
    status: Status!
    author: Author!
    createdAt: String!
  }

  # Input type (for mutations)
  input CreatePostInput {
    title: String!
    content: String
    authorId: ID!
  }

  # Query type (read operations)
  type Query {
    posts(status: Status): [Post!]!
    post(id: ID!): Post
    authors: [Author!]!
    author(id: ID!): Author
  }

  # Mutation type (write operations)
  type Mutation {
    createPost(input: CreatePostInput!): Post!
    publishPost(id: ID!): Post
    deletePost(id: ID!): Boolean!
  }
`;

// --- In-Memory Data ---

let nextPostId = 4;

const authors = [
  { id: '1', name: 'Alice', email: 'alice@example.com' },
  { id: '2', name: 'Bob', email: 'bob@example.com' },
];

const posts = [
  { id: '1', title: 'GraphQL Basics', content: 'Learn GraphQL...', status: 'PUBLISHED', authorId: '1', createdAt: '2024-01-15' },
  { id: '2', title: 'Apollo Server', content: 'Setting up Apollo...', status: 'DRAFT', authorId: '1', createdAt: '2024-02-10' },
  { id: '3', title: 'REST vs GraphQL', content: 'Comparing APIs...', status: 'PUBLISHED', authorId: '2', createdAt: '2024-03-01' },
];

// --- Resolvers ---

const resolvers = {
  Query: {
    // Resolver signature: (parent, args, context, info)
    posts: (_, { status }) => {
      if (status) return posts.filter(p => p.status === status);
      return posts;
    },

    post: (_, { id }) => posts.find(p => p.id === id) || null,

    authors: () => authors,

    author: (_, { id }) => authors.find(a => a.id === id) || null,
  },

  Mutation: {
    createPost: (_, { input }) => {
      const post = {
        id: String(nextPostId++),
        title: input.title,
        content: input.content || null,
        status: 'DRAFT',
        authorId: input.authorId,
        createdAt: new Date().toISOString().split('T')[0],
      };
      posts.push(post);
      return post;
    },

    publishPost: (_, { id }) => {
      const post = posts.find(p => p.id === id);
      if (!post) return null;
      post.status = 'PUBLISHED';
      return post;
    },

    deletePost: (_, { id }) => {
      const index = posts.findIndex(p => p.id === id);
      if (index === -1) return false;
      posts.splice(index, 1);
      return true;
    },
  },

  // Field-level resolvers (resolver chain)
  Post: {
    // parent is the Post object returned by the parent resolver
    author: (parent) => authors.find(a => a.id === parent.authorId),
  },

  Author: {
    posts: (parent) => posts.filter(p => p.authorId === parent.id),
  },
};

// --- Start Server ---

async function main() {
  const server = new ApolloServer({ typeDefs, resolvers });
  const { url } = await startStandaloneServer(server, { listen: { port: 4000 } });
  console.log(`GraphQL server ready at ${url}`);
  console.log('\nTry these queries in Apollo Sandbox:');
  console.log(`
  query {
    posts(status: PUBLISHED) {
      title
      author { name }
    }
  }

  mutation {
    createPost(input: { title: "New Post", authorId: "1" }) {
      id
      title
      status
    }
  }
  `);
}

main();
