/**
 * GraphQL Testing — Resolver Unit Tests and Integration Tests
 * Demonstrates: testing resolvers, executeOperation, mocking context.
 *
 * Run: npm install @apollo/server graphql
 *      node 07_testing.js
 */

const { ApolloServer } = require('@apollo/server');
const assert = require('assert');

// --- Schema ---

const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!
  }

  type Post {
    id: ID!
    title: String!
    author: User!
  }

  type Query {
    post(id: ID!): Post
    posts: [Post!]!
  }

  type Mutation {
    createPost(title: String!, authorId: ID!): Post!
  }
`;

// --- Mock Data ---

const mockUsers = {
  '1': { id: '1', name: 'Alice' },
  '2': { id: '2', name: 'Bob' },
};

const mockPosts = [
  { id: '1', title: 'Test Post 1', authorId: '1' },
  { id: '2', title: 'Test Post 2', authorId: '2' },
];

let testPosts = [...mockPosts];
let nextId = 3;

function resetData() {
  testPosts = [...mockPosts];
  nextId = 3;
}

// --- Resolvers ---

const resolvers = {
  Query: {
    post: (_, { id }) => testPosts.find((p) => p.id === id) || null,
    posts: () => testPosts,
  },
  Mutation: {
    createPost: (_, { title, authorId }, context) => {
      if (!context.user) throw new Error('Not authenticated');
      const post = { id: String(nextId++), title, authorId };
      testPosts.push(post);
      return post;
    },
  },
  Post: {
    author: (post) => mockUsers[post.authorId],
  },
};

// --- Test Suite ---

async function runTests() {
  let passed = 0;
  let failed = 0;

  async function test(name, fn) {
    resetData();
    try {
      await fn();
      console.log(`  ✓ ${name}`);
      passed++;
    } catch (err) {
      console.log(`  ✗ ${name}: ${err.message}`);
      failed++;
    }
  }

  console.log('\n=== GraphQL Testing Demo ===\n');

  // --- 1. Unit Testing Resolvers ---
  console.log('Unit Tests (resolver functions):');

  await test('Query.post returns a post by ID', async () => {
    const result = resolvers.Query.post(null, { id: '1' });
    assert.equal(result.id, '1');
    assert.equal(result.title, 'Test Post 1');
  });

  await test('Query.post returns null for missing ID', async () => {
    const result = resolvers.Query.post(null, { id: '999' });
    assert.equal(result, null);
  });

  await test('Query.posts returns all posts', async () => {
    const result = resolvers.Query.posts();
    assert.equal(result.length, 2);
  });

  await test('Mutation.createPost requires authentication', async () => {
    try {
      resolvers.Mutation.createPost(null, { title: 'New', authorId: '1' }, {});
      assert.fail('Should have thrown');
    } catch (err) {
      assert.equal(err.message, 'Not authenticated');
    }
  });

  await test('Mutation.createPost creates a post when authenticated', async () => {
    const context = { user: { id: '1' } };
    const result = resolvers.Mutation.createPost(
      null,
      { title: 'New Post', authorId: '1' },
      context
    );
    assert.equal(result.title, 'New Post');
    assert.equal(testPosts.length, 3);
  });

  await test('Post.author resolves author from authorId', async () => {
    const post = { id: '1', title: 'Test', authorId: '1' };
    const author = resolvers.Post.author(post);
    assert.equal(author.name, 'Alice');
  });

  // --- 2. Integration Tests (executeOperation) ---
  console.log('\nIntegration Tests (executeOperation):');

  const server = new ApolloServer({ typeDefs, resolvers });

  await test('query posts returns data with author', async () => {
    const response = await server.executeOperation(
      {
        query: `
          query {
            posts {
              id
              title
              author { name }
            }
          }
        `,
      },
      { contextValue: { user: { id: '1' } } }
    );

    assert.equal(response.body.kind, 'single');
    const data = response.body.singleResult.data;
    assert.equal(data.posts.length, 2);
    assert.equal(data.posts[0].author.name, 'Alice');
  });

  await test('query single post by ID', async () => {
    const response = await server.executeOperation(
      {
        query: `query GetPost($id: ID!) { post(id: $id) { title } }`,
        variables: { id: '1' },
      },
      { contextValue: {} }
    );

    const data = response.body.singleResult.data;
    assert.equal(data.post.title, 'Test Post 1');
  });

  await test('query missing post returns null', async () => {
    const response = await server.executeOperation(
      {
        query: `query { post(id: "999") { title } }`,
      },
      { contextValue: {} }
    );

    const data = response.body.singleResult.data;
    assert.equal(data.post, null);
  });

  await test('mutation createPost works when authenticated', async () => {
    const response = await server.executeOperation(
      {
        query: `
          mutation CreatePost($title: String!, $authorId: ID!) {
            createPost(title: $title, authorId: $authorId) {
              id
              title
              author { name }
            }
          }
        `,
        variables: { title: 'Integration Test Post', authorId: '1' },
      },
      { contextValue: { user: { id: '1' } } }
    );

    const data = response.body.singleResult.data;
    assert.equal(data.createPost.title, 'Integration Test Post');
    assert.equal(data.createPost.author.name, 'Alice');
  });

  await test('mutation createPost fails without auth', async () => {
    const response = await server.executeOperation(
      {
        query: `mutation { createPost(title: "Fail", authorId: "1") { id } }`,
      },
      { contextValue: {} }
    );

    const errors = response.body.singleResult.errors;
    assert.ok(errors && errors.length > 0);
    assert.ok(errors[0].message.includes('Not authenticated'));
  });

  console.log(`\n${passed} passed, ${failed} failed\n`);
}

runTests().catch(console.error);
