/**
 * Exercise: GraphQL Testing
 * Practice testing resolvers, schemas, and client components.
 *
 * Run: node 11_testing.js
 */

// ============================================================
// Exercise 1: Resolver Unit Tests
// Write unit tests for resolvers with mocked context.
// ============================================================

// Mini test framework
let testCount = 0;
let passCount = 0;

function test(name, fn) {
  testCount++;
  try {
    fn();
    passCount++;
    console.log(`  PASS: ${name}`);
  } catch (e) {
    console.log(`  FAIL: ${name} — ${e.message}`);
  }
}

function assertEqual(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertThrows(fn, expectedMessage) {
  try {
    fn();
    throw new Error('Expected function to throw');
  } catch (e) {
    if (expectedMessage && !e.message.includes(expectedMessage)) {
      throw new Error(`Expected error containing "${expectedMessage}", got "${e.message}"`);
    }
  }
}

// --- System Under Test ---

const mockDB = {
  users: [
    { id: '1', name: 'Alice', role: 'admin' },
    { id: '2', name: 'Bob', role: 'user' },
  ],
  posts: [
    { id: 'p1', title: 'Hello', content: 'World', authorId: '1', published: true },
    { id: 'p2', title: 'Draft', content: 'WIP', authorId: '1', published: false },
    { id: 'p3', title: 'Public', content: 'Content', authorId: '2', published: true },
  ],
};

const resolvers = {
  Query: {
    user: (_, { id }, { db }) => {
      const user = db.users.find((u) => u.id === id);
      if (!user) throw new Error('User not found');
      return user;
    },
    posts: (_, { published }, { db, currentUser }) => {
      let result = db.posts;
      if (published !== undefined) {
        result = result.filter((p) => p.published === published);
      }
      // Non-admin can only see published posts
      if (!currentUser || currentUser.role !== 'admin') {
        result = result.filter((p) => p.published);
      }
      return result;
    },
  },
  Mutation: {
    createPost: (_, { title, content }, { db, currentUser }) => {
      if (!currentUser) throw new Error('Authentication required');
      const post = {
        id: `p${db.posts.length + 1}`,
        title,
        content,
        authorId: currentUser.id,
        published: false,
      };
      db.posts.push(post);
      return post;
    },
    deletePost: (_, { id }, { db, currentUser }) => {
      if (!currentUser) throw new Error('Authentication required');
      const post = db.posts.find((p) => p.id === id);
      if (!post) throw new Error('Post not found');
      if (post.authorId !== currentUser.id && currentUser.role !== 'admin') {
        throw new Error('Not authorized');
      }
      const idx = db.posts.indexOf(post);
      db.posts.splice(idx, 1);
      return post;
    },
  },
  User: {
    posts: (user, _, { db }) => db.posts.filter((p) => p.authorId === user.id),
  },
};

// --- Helper to create mock context ---
function createMockContext(overrides = {}) {
  return {
    db: JSON.parse(JSON.stringify(mockDB)), // deep clone
    currentUser: null,
    ...overrides,
  };
}

console.log('=== Exercise 1: Resolver Unit Tests ===\n');

// TODO: Write tests for Query.user
test('Query.user returns user by id', () => {
  const ctx = createMockContext();
  const user = resolvers.Query.user(null, { id: '1' }, ctx);
  assertEqual(user.name, 'Alice');
});

test('Query.user throws for unknown id', () => {
  const ctx = createMockContext();
  assertThrows(() => resolvers.Query.user(null, { id: '99' }, ctx), 'not found');
});

// TODO: Write tests for Query.posts with auth
test('Query.posts returns published only for anonymous', () => {
  const ctx = createMockContext();
  const posts = resolvers.Query.posts(null, {}, ctx);
  assertEqual(posts.every((p) => p.published), true);
  assertEqual(posts.length, 2);
});

test('Query.posts returns all for admin', () => {
  const ctx = createMockContext({ currentUser: { id: '1', role: 'admin' } });
  const posts = resolvers.Query.posts(null, {}, ctx);
  assertEqual(posts.length, 3);
});

test('Query.posts filters by published flag', () => {
  const ctx = createMockContext({ currentUser: { id: '1', role: 'admin' } });
  const posts = resolvers.Query.posts(null, { published: false }, ctx);
  assertEqual(posts.length, 1);
  assertEqual(posts[0].title, 'Draft');
});


// ============================================================
// Exercise 2: Mutation Tests with Side Effects
// ============================================================

console.log('\n=== Exercise 2: Mutation Tests ===\n');

test('createPost requires authentication', () => {
  const ctx = createMockContext();
  assertThrows(
    () => resolvers.Mutation.createPost(null, { title: 'Test', content: 'Body' }, ctx),
    'Authentication required'
  );
});

test('createPost creates with correct authorId', () => {
  const ctx = createMockContext({ currentUser: { id: '2', role: 'user' } });
  const post = resolvers.Mutation.createPost(null, { title: 'New', content: 'Body' }, ctx);
  assertEqual(post.authorId, '2');
  assertEqual(post.published, false);
});

test('deletePost checks authorization', () => {
  const ctx = createMockContext({ currentUser: { id: '2', role: 'user' } });
  // Bob trying to delete Alice's post
  assertThrows(
    () => resolvers.Mutation.deletePost(null, { id: 'p1' }, ctx),
    'Not authorized'
  );
});

test('deletePost allows admin to delete any post', () => {
  const ctx = createMockContext({ currentUser: { id: '1', role: 'admin' } });
  const deleted = resolvers.Mutation.deletePost(null, { id: 'p3' }, ctx);
  assertEqual(deleted.id, 'p3');
  assertEqual(ctx.db.posts.find((p) => p.id === 'p3'), undefined);
});

test('deletePost allows owner to delete own post', () => {
  const ctx = createMockContext({ currentUser: { id: '1', role: 'user' } });
  const deleted = resolvers.Mutation.deletePost(null, { id: 'p1' }, ctx);
  assertEqual(deleted.authorId, '1');
});


// ============================================================
// Exercise 3: Schema Validation Tests
// ============================================================

console.log('\n=== Exercise 3: Schema Validation ===\n');

class SchemaValidator {
  constructor(schema) {
    this.schema = schema;
  }

  // TODO: Implement hasType(name) — check if type exists
  hasType(name) {
    return this.schema.types.some((t) => t.name === name);
  }

  // TODO: Implement hasField(typeName, fieldName) — check field exists
  hasField(typeName, fieldName) {
    const type = this.schema.types.find((t) => t.name === typeName);
    if (!type) return false;
    return type.fields.some((f) => f.name === fieldName);
  }

  // TODO: Implement checkBreakingChanges(oldSchema) — detect removed types/fields
  checkBreakingChanges(oldSchema) {
    const breaking = [];

    for (const oldType of oldSchema.types) {
      if (!this.hasType(oldType.name)) {
        breaking.push({ type: 'TYPE_REMOVED', name: oldType.name });
        continue;
      }
      for (const oldField of oldType.fields) {
        if (!this.hasField(oldType.name, oldField.name)) {
          breaking.push({ type: 'FIELD_REMOVED', typeName: oldType.name, fieldName: oldField.name });
        }
      }
    }

    return breaking;
  }
}

const currentSchema = {
  types: [
    { name: 'User', fields: [{ name: 'id' }, { name: 'name' }, { name: 'email' }] },
    { name: 'Post', fields: [{ name: 'id' }, { name: 'title' }, { name: 'content' }] },
    { name: 'Query', fields: [{ name: 'user' }, { name: 'posts' }] },
  ],
};

const newSchema = {
  types: [
    { name: 'User', fields: [{ name: 'id' }, { name: 'name' }] }, // email removed!
    { name: 'Post', fields: [{ name: 'id' }, { name: 'title' }, { name: 'content' }, { name: 'tags' }] },
    { name: 'Query', fields: [{ name: 'user' }, { name: 'posts' }, { name: 'search' }] },
    { name: 'Comment', fields: [{ name: 'id' }, { name: 'text' }] }, // new type
  ],
};

const validator = new SchemaValidator(newSchema);

test('Schema has existing type', () => {
  assertEqual(validator.hasType('User'), true);
});

test('Schema has new type', () => {
  assertEqual(validator.hasType('Comment'), true);
});

test('Schema has new field', () => {
  assertEqual(validator.hasField('Query', 'search'), true);
});

const changes = validator.checkBreakingChanges(currentSchema);
test('Detects breaking changes', () => {
  assertEqual(changes.length, 1); // email removed from User
  assertEqual(changes[0].type, 'FIELD_REMOVED');
  assertEqual(changes[0].fieldName, 'email');
});

console.log(`  Breaking changes found: ${JSON.stringify(changes)}`);


// ============================================================
// Exercise 4: MockedProvider Simulation
// Simulate Apollo MockedProvider for testing React components.
// ============================================================

console.log('\n=== Exercise 4: MockedProvider ===\n');

class MockedProvider {
  constructor(mocks) {
    this.mocks = mocks; // [{ query, variables, result }]
    this.callLog = [];
  }

  // TODO: Implement executeQuery(query, variables)
  // - Find matching mock by query name and variables
  // - Return mock result or error
  executeQuery(query, variables = {}) {
    const mock = this.mocks.find((m) => {
      if (m.query !== query) return false;
      if (m.variables) {
        return JSON.stringify(m.variables) === JSON.stringify(variables);
      }
      return true;
    });

    this.callLog.push({ query, variables, found: !!mock });

    if (!mock) {
      return { error: `No mock found for query: ${query}` };
    }

    if (mock.error) {
      return { error: mock.error };
    }

    return { data: mock.result };
  }

  // TODO: Implement getCallCount(query) — how many times a query was executed
  getCallCount(query) {
    return this.callLog.filter((c) => c.query === query).length;
  }
}

const provider = new MockedProvider([
  {
    query: 'GetUser',
    variables: { id: '1' },
    result: { user: { id: '1', name: 'Alice' } },
  },
  {
    query: 'GetUser',
    variables: { id: '2' },
    result: { user: { id: '2', name: 'Bob' } },
  },
  {
    query: 'GetPosts',
    result: { posts: [{ id: 'p1', title: 'Hello' }] },
  },
  {
    query: 'GetError',
    error: 'Network error',
  },
]);

test('MockedProvider returns matching result', () => {
  const r = provider.executeQuery('GetUser', { id: '1' });
  assertEqual(r.data.user.name, 'Alice');
});

test('MockedProvider matches variables', () => {
  const r = provider.executeQuery('GetUser', { id: '2' });
  assertEqual(r.data.user.name, 'Bob');
});

test('MockedProvider returns error mock', () => {
  const r = provider.executeQuery('GetError');
  assertEqual(r.error, 'Network error');
});

test('MockedProvider returns error for unmatched query', () => {
  const r = provider.executeQuery('Unknown');
  assertEqual(!!r.error, true);
});

test('MockedProvider tracks call count', () => {
  assertEqual(provider.getCallCount('GetUser'), 2);
  assertEqual(provider.getCallCount('GetPosts'), 0);
});

// Execute GetPosts
provider.executeQuery('GetPosts');
test('MockedProvider call count updates', () => {
  assertEqual(provider.getCallCount('GetPosts'), 1);
});


// ============================================================
// Summary
// ============================================================

console.log(`\n=== Results: ${passCount}/${testCount} tests passed ===`);
console.log('\nAll exercises completed!');
