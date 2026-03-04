/**
 * GraphQL Performance and Security
 * Demonstrates: Query depth limiting, complexity analysis, rate limiting, input validation.
 *
 * Run: npm install @apollo/server graphql graphql-depth-limit
 *      node 10_performance_security.js
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');

// --- Schema ---

const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!
    email: String!
    friends: [User!]!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    content: String!
    author: User!
    comments: [Comment!]!
  }

  type Comment {
    id: ID!
    text: String!
    author: User!
    replies: [Comment!]!
  }

  input CreatePostInput {
    title: String!
    content: String!
  }

  type Query {
    users: [User!]!
    user(id: ID!): User
    posts(first: Int = 10): [Post!]!
    search(term: String!): [Post!]!
  }

  type Mutation {
    createPost(input: CreatePostInput!): Post!
  }
`;

// --- Mock Data ---

const users = [
  { id: '1', name: 'Alice', email: 'alice@example.com', friendIds: ['2', '3'] },
  { id: '2', name: 'Bob', email: 'bob@example.com', friendIds: ['1'] },
  { id: '3', name: 'Charlie', email: 'charlie@example.com', friendIds: ['1', '2'] },
];

const posts = [
  { id: 'p1', title: 'Hello World', content: 'First post', authorId: '1' },
  { id: 'p2', title: 'GraphQL Tips', content: 'Security matters', authorId: '2' },
];

const comments = [
  { id: 'c1', text: 'Great!', authorId: '2', postId: 'p1', parentId: null },
  { id: 'c2', text: 'Thanks!', authorId: '1', postId: 'p1', parentId: 'c1' },
];

// --- Security: Query Depth Limiter ---

function checkQueryDepth(query, maxDepth) {
  let depth = 0;
  let maxFound = 0;

  for (const char of query) {
    if (char === '{') {
      depth++;
      maxFound = Math.max(maxFound, depth);
    } else if (char === '}') {
      depth--;
    }
  }

  const allowed = maxFound <= maxDepth;
  return { depth: maxFound, maxDepth, allowed };
}

// --- Security: Query Complexity Calculator ---

function calculateComplexity(query, costMap) {
  let complexity = 0;

  for (const [field, cost] of Object.entries(costMap)) {
    const regex = new RegExp(`\\b${field}\\b`, 'g');
    const matches = query.match(regex);
    if (matches) {
      complexity += matches.length * cost;
    }
  }

  return complexity;
}

// --- Security: Rate Limiter ---

class RateLimiter {
  constructor({ windowMs, maxComplexity }) {
    this.windowMs = windowMs;
    this.maxComplexity = maxComplexity;
    this.clients = new Map();
  }

  check(clientId, complexity) {
    const now = Date.now();
    const client = this.clients.get(clientId) || { total: 0, windowStart: now };

    // Reset window if expired
    if (now - client.windowStart > this.windowMs) {
      client.total = 0;
      client.windowStart = now;
    }

    const remaining = this.maxComplexity - client.total;
    const allowed = complexity <= remaining;

    if (allowed) {
      client.total += complexity;
      this.clients.set(clientId, client);
    }

    return {
      allowed,
      used: client.total,
      remaining: Math.max(0, remaining - (allowed ? complexity : 0)),
      resetIn: Math.ceil((client.windowStart + this.windowMs - now) / 1000),
    };
  }
}

// --- Security: Input Sanitizer ---

function sanitizeInput(input) {
  const issues = [];

  if (typeof input.title === 'string') {
    if (input.title.length > 200) issues.push('Title exceeds 200 chars');
    if (/<script/i.test(input.title)) issues.push('Title contains script tag');
  }

  if (typeof input.content === 'string') {
    if (input.content.length > 10000) issues.push('Content exceeds 10000 chars');
  }

  if (typeof input.term === 'string') {
    // Prevent SQL/NoSQL injection patterns
    if (/[${}]/.test(input.term)) issues.push('Search term contains injection pattern');
  }

  return { valid: issues.length === 0, issues };
}

// --- Resolvers ---

const resolvers = {
  Query: {
    users: () => users,
    user: (_, { id }) => users.find((u) => u.id === id),
    posts: (_, { first }) => posts.slice(0, first),
    search: (_, { term }) => {
      const check = sanitizeInput({ term });
      if (!check.valid) throw new Error(`Invalid input: ${check.issues.join(', ')}`);
      const lower = term.toLowerCase();
      return posts.filter((p) => p.title.toLowerCase().includes(lower));
    },
  },
  Mutation: {
    createPost: (_, { input }) => {
      const check = sanitizeInput(input);
      if (!check.valid) throw new Error(`Invalid input: ${check.issues.join(', ')}`);
      const post = { id: `p${posts.length + 1}`, ...input, authorId: '1' };
      posts.push(post);
      return post;
    },
  },
  User: {
    friends: (user) => user.friendIds.map((id) => users.find((u) => u.id === id)).filter(Boolean),
    posts: (user) => posts.filter((p) => p.authorId === user.id),
  },
  Post: {
    author: (post) => users.find((u) => u.id === post.authorId),
    comments: (post) => comments.filter((c) => c.postId === post.id && !c.parentId),
  },
  Comment: {
    author: (comment) => users.find((u) => u.id === comment.authorId),
    replies: (comment) => comments.filter((c) => c.parentId === comment.id),
  },
};

// --- Main: Security Demo ---

async function main() {
  const server = new ApolloServer({ typeDefs, resolvers });

  const { url } = await startStandaloneServer(server, { listen: { port: 4000 } });
  console.log(`GraphQL server at ${url}\n`);

  // 1. Query Depth Limiting
  console.log('=== Query Depth Limiting ===\n');

  const shallowQuery = '{ users { name posts { title } } }';
  const deepQuery = '{ users { friends { friends { friends { friends { name } } } } } }';

  const shallow = checkQueryDepth(shallowQuery, 5);
  console.log(`Shallow (depth ${shallow.depth}): ${shallow.allowed ? 'ALLOWED' : 'BLOCKED'}`);

  const deep = checkQueryDepth(deepQuery, 5);
  console.log(`Deep    (depth ${deep.depth}): ${deep.allowed ? 'ALLOWED' : 'BLOCKED'}`);

  // 2. Complexity Analysis
  console.log('\n=== Complexity Analysis ===\n');

  const costMap = {
    users: 10,
    friends: 20,    // expensive: can expand exponentially
    posts: 5,
    comments: 5,
    replies: 10,
  };
  const maxComplexity = 100;

  const q1 = '{ users { name } }';
  const q2 = '{ users { friends { posts { comments { replies { text } } } } } }';

  const c1 = calculateComplexity(q1, costMap);
  console.log(`Query 1 complexity: ${c1}/${maxComplexity} — ${c1 <= maxComplexity ? 'ALLOWED' : 'BLOCKED'}`);

  const c2 = calculateComplexity(q2, costMap);
  console.log(`Query 2 complexity: ${c2}/${maxComplexity} — ${c2 <= maxComplexity ? 'ALLOWED' : 'BLOCKED'}`);

  // 3. Rate Limiting
  console.log('\n=== Rate Limiting ===\n');

  const limiter = new RateLimiter({ windowMs: 60000, maxComplexity: 200 });

  for (let i = 1; i <= 5; i++) {
    const cost = 50;
    const result = limiter.check('client-1', cost);
    console.log(`Request ${i} (cost=${cost}): ${result.allowed ? 'ALLOWED' : 'RATE LIMITED'} ` +
      `[used=${result.used}, remaining=${result.remaining}]`);
  }

  // 4. Input Validation
  console.log('\n=== Input Validation ===\n');

  const inputs = [
    { title: 'Good Title', content: 'Normal content' },
    { title: '<script>alert("xss")</script>', content: 'Bad' },
    { term: 'normal search' },
    { term: '{ $gt: "" }' },
  ];

  inputs.forEach((input, i) => {
    const result = sanitizeInput(input);
    const label = JSON.stringify(input).slice(0, 50);
    console.log(`Input ${i + 1}: ${result.valid ? 'VALID' : 'REJECTED'} — ${label}...`);
    if (!result.valid) console.log(`  Issues: ${result.issues.join(', ')}`);
  });
}

main();
