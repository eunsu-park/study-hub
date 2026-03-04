/**
 * Exercise: GraphQL API Gateway with Federation
 * Practice designing federated subgraphs, entity resolution, and query planning.
 *
 * Run: node 14_api_gateway.js
 */

// ============================================================
// Exercise 1: Subgraph Registry
// Implement a registry that tracks subgraphs and their entity ownership.
// ============================================================

class SubgraphRegistry {
  constructor() {
    this.subgraphs = new Map(); // name -> { url, entities, fields }
  }

  // TODO: Implement register(name, config)
  // config: { url, entities: ['User', 'Product'], fields: { User: ['id', 'name', 'email'] } }
  register(name, config) {
    this.subgraphs.set(name, {
      name,
      url: config.url,
      entities: config.entities || [],
      fields: config.fields || {},
    });
  }

  // TODO: Implement findSubgraphForEntity(entityName)
  // Returns all subgraphs that own or extend this entity
  findSubgraphsForEntity(entityName) {
    const result = [];
    for (const [, sg] of this.subgraphs) {
      if (sg.entities.includes(entityName)) {
        result.push(sg.name);
      }
    }
    return result;
  }

  // TODO: Implement findSubgraphForField(entityName, fieldName)
  // Returns the subgraph that owns this specific field
  findSubgraphForField(entityName, fieldName) {
    for (const [, sg] of this.subgraphs) {
      const fields = sg.fields[entityName] || [];
      if (fields.includes(fieldName)) {
        return sg.name;
      }
    }
    return null;
  }

  // TODO: Implement getSupergraphInfo()
  getSupergraphInfo() {
    const entities = new Map(); // entity -> { subgraphs, fields }

    for (const [, sg] of this.subgraphs) {
      for (const entity of sg.entities) {
        if (!entities.has(entity)) {
          entities.set(entity, { subgraphs: [], allFields: [] });
        }
        const info = entities.get(entity);
        info.subgraphs.push(sg.name);
        info.allFields.push(...(sg.fields[entity] || []));
      }
    }

    return {
      subgraphCount: this.subgraphs.size,
      entities: Object.fromEntries(entities),
    };
  }
}


// ============================================================
// Exercise 2: Query Planner
// Implement a basic query planner that determines which subgraphs to query.
// ============================================================

class QueryPlanner {
  constructor(registry) {
    this.registry = registry;
  }

  // TODO: Implement plan(query)
  // Parse the query to determine which subgraphs need to be called
  // Returns a query plan with fetch steps
  plan(query) {
    const steps = [];
    const fields = this._extractFields(query);

    // Group fields by subgraph
    const subgraphFields = new Map();

    for (const { entity, field } of fields) {
      const sg = this.registry.findSubgraphForField(entity, field);
      if (sg) {
        if (!subgraphFields.has(sg)) subgraphFields.set(sg, []);
        subgraphFields.get(sg).push({ entity, field });
      }
    }

    // Create fetch steps
    let sequence = 1;
    for (const [sgName, sgFields] of subgraphFields) {
      const entities = [...new Set(sgFields.map((f) => f.entity))];
      const needsEntityRef = steps.length > 0 && entities.some(
        (e) => steps.some((s) => s.entities.includes(e))
      );

      steps.push({
        sequence,
        subgraph: sgName,
        entities,
        fields: sgFields.map((f) => `${f.entity}.${f.field}`),
        requiresEntityResolution: needsEntityRef,
      });
      sequence++;
    }

    return {
      steps,
      parallel: this._findParallelSteps(steps),
    };
  }

  _extractFields(query) {
    // Simplified field extraction
    const fields = [];
    const entityPattern = /(\w+)\s*\{([^}]+)\}/g;
    let match;

    while ((match = entityPattern.exec(query)) !== null) {
      const entity = match[1].charAt(0).toUpperCase() + match[1].slice(1);
      const fieldNames = match[2].trim().split(/\s+/).filter((f) => f && !f.includes('{'));
      for (const field of fieldNames) {
        fields.push({ entity, field: field.trim() });
      }
    }

    return fields;
  }

  _findParallelSteps(steps) {
    // Steps that don't depend on each other can run in parallel
    const parallel = [];
    const sequential = [];

    for (const step of steps) {
      if (step.requiresEntityResolution) {
        sequential.push(step.sequence);
      } else {
        parallel.push(step.sequence);
      }
    }

    return { parallel, sequential };
  }
}


// ============================================================
// Exercise 3: Entity Reference Resolver
// Implement __resolveReference for cross-subgraph entity resolution.
// ============================================================

class EntityResolver {
  constructor() {
    this.resolvers = new Map(); // entityName -> resolveReference fn
    this.resolveLog = [];
  }

  // TODO: Implement registerResolver(entityName, resolveFn)
  registerResolver(entityName, resolveFn) {
    this.resolvers.set(entityName, resolveFn);
  }

  // TODO: Implement resolveReference(representation)
  // representation: { __typename: 'User', id: '1' }
  async resolveReference(representation) {
    const resolver = this.resolvers.get(representation.__typename);
    if (!resolver) {
      throw new Error(`No resolver for entity: ${representation.__typename}`);
    }

    this.resolveLog.push({
      entity: representation.__typename,
      key: representation.id,
      timestamp: Date.now(),
    });

    return resolver(representation);
  }

  // TODO: Implement batchResolve(representations)
  // Group by typename, resolve each group, return in original order
  async batchResolve(representations) {
    // Group by typename
    const groups = new Map();
    const indexMap = [];

    representations.forEach((rep, idx) => {
      if (!groups.has(rep.__typename)) groups.set(rep.__typename, []);
      groups.get(rep.__typename).push({ rep, idx });
    });

    // Resolve each group
    const results = new Array(representations.length);

    for (const [typename, items] of groups) {
      const resolver = this.resolvers.get(typename);
      if (!resolver) continue;

      // Resolve all items of this type
      for (const { rep, idx } of items) {
        results[idx] = await resolver(rep);
        this.resolveLog.push({ entity: typename, key: rep.id, timestamp: Date.now() });
      }
    }

    return results;
  }
}


// ============================================================
// Exercise 4: Gateway Health Monitor
// Track subgraph health and route requests accordingly.
// ============================================================

class GatewayHealthMonitor {
  constructor(subgraphs) {
    this.subgraphs = new Map();
    for (const sg of subgraphs) {
      this.subgraphs.set(sg, {
        status: 'healthy',
        latency: [],
        errors: 0,
        lastCheck: Date.now(),
      });
    }
  }

  // TODO: Implement recordRequest(subgraph, latencyMs, success)
  recordRequest(subgraph, latencyMs, success) {
    const sg = this.subgraphs.get(subgraph);
    if (!sg) return;

    sg.latency.push(latencyMs);
    if (sg.latency.length > 100) sg.latency.shift(); // keep last 100

    if (!success) sg.errors++;
    sg.lastCheck = Date.now();

    // Update status
    const avgLatency = sg.latency.reduce((a, b) => a + b, 0) / sg.latency.length;
    const errorRate = sg.errors / Math.max(1, sg.latency.length);

    if (errorRate > 0.5) sg.status = 'down';
    else if (errorRate > 0.2 || avgLatency > 2000) sg.status = 'degraded';
    else sg.status = 'healthy';
  }

  // TODO: Implement getHealth() — return health status for all subgraphs
  getHealth() {
    const health = {};
    for (const [name, sg] of this.subgraphs) {
      const avg = sg.latency.length > 0
        ? Math.round(sg.latency.reduce((a, b) => a + b, 0) / sg.latency.length)
        : 0;
      health[name] = {
        status: sg.status,
        avgLatencyMs: avg,
        errorCount: sg.errors,
        requestCount: sg.latency.length,
      };
    }
    return health;
  }

  // TODO: Implement isAvailable(subgraph)
  isAvailable(subgraph) {
    const sg = this.subgraphs.get(subgraph);
    return sg ? sg.status !== 'down' : false;
  }
}


// ============================================================
// Test all exercises
// ============================================================

async function runTests() {
  console.log('=== Exercise 1: Subgraph Registry ===\n');

  const registry = new SubgraphRegistry();

  registry.register('users', {
    url: 'http://users:4001',
    entities: ['User'],
    fields: { User: ['id', 'name', 'email', 'tier'] },
  });

  registry.register('catalog', {
    url: 'http://catalog:4002',
    entities: ['Product', 'Category'],
    fields: {
      Product: ['id', 'name', 'price', 'inStock'],
      Category: ['id', 'name'],
    },
  });

  registry.register('orders', {
    url: 'http://orders:4003',
    entities: ['Order', 'User'],
    fields: {
      Order: ['id', 'total', 'status'],
      User: ['orders', 'totalSpent'],
    },
  });

  registry.register('reviews', {
    url: 'http://reviews:4004',
    entities: ['Review', 'Product', 'User'],
    fields: {
      Review: ['id', 'rating', 'comment'],
      Product: ['reviews', 'averageRating'],
      User: ['reviewCount'],
    },
  });

  const userSgs = registry.findSubgraphsForEntity('User');
  console.log(`User subgraphs: ${userSgs.join(', ')} (expected 3): ${userSgs.length === 3 ? 'PASS' : 'FAIL'}`);

  const nameSg = registry.findSubgraphForField('User', 'name');
  console.log(`User.name owned by: ${nameSg} (expected users): ${nameSg === 'users' ? 'PASS' : 'FAIL'}`);

  const reviewsSg = registry.findSubgraphForField('Product', 'reviews');
  console.log(`Product.reviews owned by: ${reviewsSg} (expected reviews): ${reviewsSg === 'reviews' ? 'PASS' : 'FAIL'}`);

  const info = registry.getSupergraphInfo();
  console.log(`Subgraph count: ${info.subgraphCount} (expected 4): ${info.subgraphCount === 4 ? 'PASS' : 'FAIL'}`);

  console.log('\n=== Exercise 2: Query Planner ===\n');

  const planner = new QueryPlanner(registry);

  const plan = planner.plan(`
    user {
      name email
      orders { total status }
      reviewCount
    }
  `);

  console.log('Query plan steps:');
  plan.steps.forEach((s) => {
    console.log(`  Step ${s.sequence}: ${s.subgraph} — ${s.fields.join(', ')}${s.requiresEntityResolution ? ' (needs entity ref)' : ''}`);
  });
  console.log(`Parallel steps: ${plan.parallel.parallel.join(', ') || 'none'}`);
  console.log(`Sequential steps: ${plan.parallel.sequential.join(', ') || 'none'}`);

  console.log('\n=== Exercise 3: Entity Resolver ===\n');

  const resolver = new EntityResolver();

  const usersDB = [
    { id: '1', name: 'Alice', email: 'alice@example.com' },
    { id: '2', name: 'Bob', email: 'bob@example.com' },
  ];

  const productsDB = [
    { id: 'p1', name: 'Laptop', price: 999.99 },
    { id: 'p2', name: 'Mouse', price: 29.99 },
  ];

  resolver.registerResolver('User', (ref) => usersDB.find((u) => u.id === ref.id));
  resolver.registerResolver('Product', (ref) => productsDB.find((p) => p.id === ref.id));

  const user = await resolver.resolveReference({ __typename: 'User', id: '1' });
  console.log(`Resolved User#1: ${user.name} (expected Alice): ${user.name === 'Alice' ? 'PASS' : 'FAIL'}`);

  // Batch resolve
  const refs = [
    { __typename: 'User', id: '1' },
    { __typename: 'Product', id: 'p1' },
    { __typename: 'User', id: '2' },
    { __typename: 'Product', id: 'p2' },
  ];

  const resolved = await resolver.batchResolve(refs);
  console.log(`Batch resolved: ${resolved.length} (expected 4): ${resolved.length === 4 ? 'PASS' : 'FAIL'}`);
  console.log(`  [0] ${resolved[0].name} (User Alice): ${resolved[0].name === 'Alice' ? 'PASS' : 'FAIL'}`);
  console.log(`  [1] ${resolved[1].name} (Product Laptop): ${resolved[1].name === 'Laptop' ? 'PASS' : 'FAIL'}`);

  console.log(`\nResolve log entries: ${resolver.resolveLog.length}`);

  console.log('\n=== Exercise 4: Gateway Health Monitor ===\n');

  const monitor = new GatewayHealthMonitor(['users', 'catalog', 'orders', 'reviews']);

  // Simulate requests
  for (let i = 0; i < 10; i++) {
    monitor.recordRequest('users', 50 + Math.random() * 30, true);
    monitor.recordRequest('catalog', 30 + Math.random() * 20, true);
    monitor.recordRequest('orders', 100 + Math.random() * 50, true);
  }

  // Reviews has issues
  for (let i = 0; i < 10; i++) {
    monitor.recordRequest('reviews', 3000, i < 3); // 70% errors
  }

  const health = monitor.getHealth();
  console.log('Health status:');
  for (const [name, status] of Object.entries(health)) {
    console.log(`  ${name}: ${status.status} (avg ${status.avgLatencyMs}ms, ${status.errorCount} errors)`);
  }

  console.log(`Users healthy: ${health.users.status === 'healthy' ? 'PASS' : 'FAIL'}`);
  console.log(`Reviews degraded/down: ${health.reviews.status !== 'healthy' ? 'PASS' : 'FAIL'}`);
  console.log(`Reviews available: ${monitor.isAvailable('reviews')}`);

  console.log('\nAll exercises completed!');
}

runTests();
