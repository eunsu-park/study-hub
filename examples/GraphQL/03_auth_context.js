/**
 * GraphQL Authentication and Authorization
 * Demonstrates: context-based auth, field-level permissions, directive approach.
 *
 * Run: npm install @apollo/server graphql jsonwebtoken
 *      node 03_auth_context.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const { GraphQLError } = require('graphql');
const jwt = require('jsonwebtoken');

const SECRET = 'demo-secret-key';

// --- Schema ---

const typeDefs = `#graphql
  enum Role {
    USER
    ADMIN
  }

  type User {
    id: ID!
    username: String!
    email: String!     # Only visible to the user themselves or admins
    role: Role!
  }

  type Post {
    id: ID!
    title: String!
    content: String!
    author: User!
  }

  type AuthPayload {
    token: String!
    user: User!
  }

  type Query {
    me: User
    posts: [Post!]!
    users: [User!]!      # Admin only
  }

  type Mutation {
    login(username: String!, password: String!): AuthPayload!
    createPost(title: String!, content: String!): Post!
    deletePost(id: ID!): Boolean!
  }
`;

// --- Data ---

const users = [
  { id: '1', username: 'alice', password: 'alice123', email: 'alice@example.com', role: 'ADMIN' },
  { id: '2', username: 'bob', password: 'bob123', email: 'bob@example.com', role: 'USER' },
];

const posts = [
  { id: '1', title: 'Hello GraphQL', content: 'Getting started...', authorId: '1' },
  { id: '2', title: 'Auth Patterns', content: 'Context-based auth...', authorId: '2' },
];

// --- Auth Helpers ---

function authenticate(context) {
  if (!context.user) {
    throw new GraphQLError('You must be logged in', {
      extensions: { code: 'UNAUTHENTICATED', http: { status: 401 } },
    });
  }
  return context.user;
}

function authorize(context, ...roles) {
  const user = authenticate(context);
  if (!roles.includes(user.role)) {
    throw new GraphQLError('Not authorized', {
      extensions: { code: 'FORBIDDEN', http: { status: 403 } },
    });
  }
  return user;
}

// --- Resolvers ---

const resolvers = {
  Query: {
    me: (_, __, context) => {
      const user = authenticate(context);
      return users.find((u) => u.id === user.id);
    },

    posts: () => posts,

    users: (_, __, context) => {
      authorize(context, 'ADMIN');
      return users;
    },
  },

  Mutation: {
    login: (_, { username, password }) => {
      const user = users.find((u) => u.username === username && u.password === password);
      if (!user) {
        throw new GraphQLError('Invalid credentials', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      const token = jwt.sign({ id: user.id, role: user.role }, SECRET, { expiresIn: '1h' });
      return { token, user };
    },

    createPost: (_, { title, content }, context) => {
      const user = authenticate(context);
      const post = {
        id: String(posts.length + 1),
        title,
        content,
        authorId: user.id,
      };
      posts.push(post);
      return post;
    },

    deletePost: (_, { id }, context) => {
      const user = authenticate(context);
      const index = posts.findIndex((p) => p.id === id);
      if (index === -1) return false;
      const post = posts[index];
      // Only author or admin can delete
      if (post.authorId !== user.id && user.role !== 'ADMIN') {
        throw new GraphQLError('Not authorized to delete this post', {
          extensions: { code: 'FORBIDDEN' },
        });
      }
      posts.splice(index, 1);
      return true;
    },
  },

  Post: {
    author: (post) => users.find((u) => u.id === post.authorId),
  },

  User: {
    // Field-level authorization: email visible only to self or admin
    email: (user, _, context) => {
      if (!context.user) return null;
      if (context.user.id === user.id || context.user.role === 'ADMIN') {
        return user.email;
      }
      return null;
    },
  },
};

// --- Server with Auth Context ---

async function main() {
  const server = new ApolloServer({ typeDefs, resolvers });

  const { url } = await startStandaloneServer(server, {
    listen: { port: 4000 },
    context: async ({ req }) => {
      // Extract JWT from Authorization header
      const token = req.headers.authorization?.replace('Bearer ', '');
      let user = null;

      if (token) {
        try {
          const decoded = jwt.verify(token, SECRET);
          user = { id: decoded.id, role: decoded.role };
        } catch {
          // Invalid token — user stays null (unauthenticated)
        }
      }

      return { user };
    },
  });

  console.log(`Auth demo at ${url}`);
  console.log('\n1. Login to get a token:');
  console.log(`   mutation { login(username: "alice", password: "alice123") { token } }`);
  console.log('\n2. Set Authorization header: "Bearer <token>"');
  console.log('\n3. Query protected endpoints:');
  console.log(`   query { me { username email role } }`);
}

main();
