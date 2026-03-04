/**
 * Exercise: GraphQL Performance and Security
 * Practice implementing query depth limits, complexity analysis, and rate limiting.
 *
 * Run: node 12_performance_security.js
 */

// ============================================================
// Exercise 1: Query Depth Analyzer
// Implement accurate query depth analysis from a parsed query AST.
// ============================================================

function parseQueryToAST(query) {
  // Simplified parser: converts GraphQL query to nested structure
  const stack = [{ fields: [], depth: 0 }];
  let current = '';

  for (const char of query) {
    if (char === '{') {
      const fieldName = current.trim().split(/[\s(]/)[0];
      const node = { name: fieldName, fields: [], depth: stack.length };
      stack[stack.length - 1].fields.push(node);
      stack.push(node);
      current = '';
    } else if (char === '}') {
      if (current.trim()) {
        const names = current.trim().split(/\s+/).filter(Boolean);
        for (const name of names) {
          if (name && !name.startsWith('#')) {
            stack[stack.length - 1].fields.push({ name, fields: [], depth: stack.length });
          }
        }
      }
      stack.pop();
      current = '';
    } else {
      current += char;
    }
  }

  return stack[0].fields[0] || { name: 'root', fields: [], depth: 0 };
}

// TODO: Implement getMaxDepth(ast) — return the maximum nesting depth
function getMaxDepth(ast) {
  if (!ast.fields || ast.fields.length === 0) return 1;
  let max = 0;
  for (const field of ast.fields) {
    const childDepth = getMaxDepth(field);
    max = Math.max(max, childDepth);
  }
  return max + 1;
}

// TODO: Implement checkDepthLimit(query, maxDepth)
// Returns { allowed, depth, maxDepth }
function checkDepthLimit(query, maxDepth) {
  const ast = parseQueryToAST(query);
  const depth = getMaxDepth(ast);
  return { allowed: depth <= maxDepth, depth, maxDepth };
}


// ============================================================
// Exercise 2: Query Complexity Calculator
// Implement field-level complexity with multipliers.
// ============================================================

class ComplexityCalculator {
  constructor(costMap) {
    this.costMap = costMap;
    // costMap: { fieldName: { cost, multiplier? } }
  }

  // TODO: Implement calculate(query) — return total complexity
  // Rules:
  //   - Each field has a base cost (default 1)
  //   - List fields multiply child costs by the "first" or "limit" argument
  //   - Nested fields compound multiplicatively
  calculate(query) {
    const ast = parseQueryToAST(query);
    return this._calcNode(ast, 1);
  }

  _calcNode(node, parentMultiplier) {
    const fieldConfig = this.costMap[node.name] || { cost: 1 };
    const baseCost = (fieldConfig.cost || 1) * parentMultiplier;

    if (node.fields.length === 0) return baseCost;

    const multiplier = fieldConfig.multiplier || 1;
    let total = baseCost;
    for (const child of node.fields) {
      total += this._calcNode(child, parentMultiplier * multiplier);
    }
    return total;
  }
}


// ============================================================
// Exercise 3: Sliding Window Rate Limiter
// Implement a complexity-aware rate limiter with sliding windows.
// ============================================================

class SlidingWindowRateLimiter {
  constructor({ windowMs, maxComplexity }) {
    this.windowMs = windowMs;
    this.maxComplexity = maxComplexity;
    this.windows = new Map(); // clientId -> [{ timestamp, complexity }]
  }

  // TODO: Implement consume(clientId, complexity)
  // - Remove entries outside the current window
  // - Check if adding this complexity exceeds the limit
  // - If allowed, record and return { allowed, remaining, resetMs }
  consume(clientId, complexity) {
    const now = Date.now();
    const entries = this.windows.get(clientId) || [];

    // Remove expired entries
    const active = entries.filter((e) => now - e.timestamp < this.windowMs);

    const used = active.reduce((sum, e) => sum + e.complexity, 0);
    const remaining = this.maxComplexity - used;

    if (complexity > remaining) {
      this.windows.set(clientId, active);
      const oldest = active.length > 0 ? active[0].timestamp : now;
      return {
        allowed: false,
        remaining,
        used,
        resetMs: this.windowMs - (now - oldest),
      };
    }

    active.push({ timestamp: now, complexity });
    this.windows.set(clientId, active);

    return {
      allowed: true,
      remaining: remaining - complexity,
      used: used + complexity,
      resetMs: this.windowMs,
    };
  }

  // TODO: Implement getHeaders(clientId) — return rate limit headers
  getHeaders(clientId) {
    const now = Date.now();
    const entries = (this.windows.get(clientId) || []).filter(
      (e) => now - e.timestamp < this.windowMs
    );
    const used = entries.reduce((sum, e) => sum + e.complexity, 0);

    return {
      'X-RateLimit-Limit': this.maxComplexity,
      'X-RateLimit-Remaining': Math.max(0, this.maxComplexity - used),
      'X-RateLimit-Reset': entries.length > 0
        ? Math.ceil((entries[0].timestamp + this.windowMs) / 1000)
        : Math.ceil(now / 1000),
    };
  }
}


// ============================================================
// Exercise 4: Alias Abuse Detector
// Detect and prevent alias-based attacks.
// ============================================================

function detectAliasAbuse(query, maxAliases) {
  // TODO: Count aliases in a query
  // Aliases look like: aliasName: fieldName
  // Return { allowed, aliasCount, maxAliases, aliases }

  const aliasPattern = /(\w+)\s*:\s*(\w+)/g;
  const aliases = [];
  let match;

  while ((match = aliasPattern.exec(query)) !== null) {
    // Filter out field:value patterns inside arguments
    const beforeMatch = query.slice(0, match.index);
    const openParens = (beforeMatch.match(/\(/g) || []).length;
    const closeParens = (beforeMatch.match(/\)/g) || []).length;

    // Only count aliases that are NOT inside argument lists
    if (openParens <= closeParens) {
      aliases.push({ alias: match[1], field: match[2] });
    }
  }

  return {
    allowed: aliases.length <= maxAliases,
    aliasCount: aliases.length,
    maxAliases,
    aliases,
  };
}


// ============================================================
// Test all exercises
// ============================================================

console.log('=== Exercise 1: Query Depth Analyzer ===\n');

const shallowQ = '{ users { name email } }';
const deepQ = '{ users { friends { friends { friends { name } } } } }';
const mediumQ = '{ users { posts { comments { text } } } }';

const r1 = checkDepthLimit(shallowQ, 5);
console.log(`Shallow depth=${r1.depth}, allowed=${r1.allowed}: ${r1.allowed ? 'PASS' : 'FAIL'}`);

const r2 = checkDepthLimit(deepQ, 3);
console.log(`Deep depth=${r2.depth}, allowed=${r2.allowed} (expected false): ${!r2.allowed ? 'PASS' : 'FAIL'}`);

const r3 = checkDepthLimit(mediumQ, 5);
console.log(`Medium depth=${r3.depth}, allowed=${r3.allowed}: ${r3.allowed ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 2: Complexity Calculator ===\n');

const costMap = {
  query: { cost: 0 },
  users: { cost: 10, multiplier: 10 },   // list field, assumes up to 10 items
  friends: { cost: 5, multiplier: 5 },
  posts: { cost: 3, multiplier: 5 },
  name: { cost: 1 },
  email: { cost: 1 },
  title: { cost: 1 },
};

const calc = new ComplexityCalculator(costMap);

const c1 = calc.calculate('query { users { name } }');
console.log(`Simple query complexity: ${c1}`);

const c2 = calc.calculate('query { users { friends { name } } }');
console.log(`Nested query complexity: ${c2}`);

const maxAllowed = 500;
console.log(`Simple allowed (${c1} <= ${maxAllowed}): ${c1 <= maxAllowed ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 3: Rate Limiter ===\n');

const limiter = new SlidingWindowRateLimiter({ windowMs: 60000, maxComplexity: 1000 });

const l1 = limiter.consume('client-1', 200);
console.log(`Request 1 (200): allowed=${l1.allowed}, remaining=${l1.remaining}: ${l1.allowed ? 'PASS' : 'FAIL'}`);

const l2 = limiter.consume('client-1', 300);
console.log(`Request 2 (300): allowed=${l2.allowed}, remaining=${l2.remaining}: ${l2.allowed ? 'PASS' : 'FAIL'}`);

const l3 = limiter.consume('client-1', 600);
console.log(`Request 3 (600): allowed=${l3.allowed} (expected false, would exceed 1000): ${!l3.allowed ? 'PASS' : 'FAIL'}`);

// Different client has their own limit
const l4 = limiter.consume('client-2', 500);
console.log(`Client-2 (500): allowed=${l4.allowed}, remaining=${l4.remaining}: ${l4.allowed ? 'PASS' : 'FAIL'}`);

const headers = limiter.getHeaders('client-1');
console.log(`Rate limit headers:`, headers);
console.log(`X-RateLimit-Limit: ${headers['X-RateLimit-Limit']} (expected 1000): ${headers['X-RateLimit-Limit'] === 1000 ? 'PASS' : 'FAIL'}`);

console.log('\n=== Exercise 4: Alias Abuse Detector ===\n');

const normalQuery = `{
  user(id: "1") { name email }
}`;
const a1 = detectAliasAbuse(normalQuery, 5);
console.log(`Normal query aliases: ${a1.aliasCount} — allowed: ${a1.allowed}: ${a1.allowed ? 'PASS' : 'FAIL'}`);

const aliasedQuery = `{
  a1: user(id: "1") { name }
  a2: user(id: "2") { name }
  a3: user(id: "3") { name }
}`;
const a2 = detectAliasAbuse(aliasedQuery, 5);
console.log(`3 aliases: count=${a2.aliasCount}, allowed=${a2.allowed}: ${a2.allowed ? 'PASS' : 'FAIL'}`);
console.log(`Aliases found: ${a2.aliases.map((a) => `${a.alias}→${a.field}`).join(', ')}`);

const abuseQuery = `{
  a1: expensiveQuery { data }
  a2: expensiveQuery { data }
  a3: expensiveQuery { data }
  a4: expensiveQuery { data }
  a5: expensiveQuery { data }
  a6: expensiveQuery { data }
}`;
const a3 = detectAliasAbuse(abuseQuery, 5);
console.log(`Abuse (6 aliases, max 5): allowed=${a3.allowed} (expected false): ${!a3.allowed ? 'PASS' : 'FAIL'}`);

console.log('\nAll exercises completed!');
