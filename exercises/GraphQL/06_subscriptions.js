/**
 * Exercise: GraphQL Subscriptions
 * Practice implementing real-time features with PubSub and subscription resolvers.
 *
 * Run: node 06_subscriptions.js
 */

// ============================================================
// Exercise 1: PubSub Event System
// Implement an in-memory PubSub system with subscribe and publish.
// ============================================================

class SimplePubSub {
  constructor() {
    this.channels = new Map(); // channel -> Set of callbacks
  }

  // TODO: Implement subscribe(channel, callback)
  // - Add callback to the channel's subscriber set
  // - Return an unsubscribe function
  subscribe(channel, callback) {
    if (!this.channels.has(channel)) {
      this.channels.set(channel, new Set());
    }
    this.channels.get(channel).add(callback);
    return () => this.channels.get(channel).delete(callback);
  }

  // TODO: Implement publish(channel, payload)
  // - Call all subscribers for the channel with the payload
  // - Return the number of subscribers notified
  publish(channel, payload) {
    const subs = this.channels.get(channel);
    if (!subs || subs.size === 0) return 0;
    subs.forEach((cb) => cb(payload));
    return subs.size;
  }
}


// ============================================================
// Exercise 2: Subscription Resolver with Filtering
// Implement a subscription that filters events based on arguments.
// ============================================================

// Scenario: Chat room with multiple channels
const chatMessages = [];

// TODO: Implement withFilter(asyncIterator, filterFn)
// - Takes an iterator and a filter function
// - Returns a new iterator that only yields events matching the filter
function withFilter(events, filterFn) {
  return events.filter(filterFn);
}

// TODO: Implement subscription resolvers
const subscriptionResolvers = {
  messageSent: {
    // Subscribe to messages, optionally filtered by room
    subscribe: (pubsub, { room }) => {
      const events = [];
      const unsub = pubsub.subscribe('MESSAGE_SENT', (msg) => {
        if (!room || msg.room === room) {
          events.push(msg);
        }
      });
      return { events, unsubscribe: unsub };
    },
  },

  userTyping: {
    // Subscribe to typing indicators for a specific room
    subscribe: (pubsub, { room }) => {
      const events = [];
      const unsub = pubsub.subscribe('USER_TYPING', (event) => {
        if (event.room === room) {
          events.push(event);
        }
      });
      return { events, unsubscribe: unsub };
    },
  },
};


// ============================================================
// Exercise 3: Connection Management
// Implement WebSocket connection lifecycle handling.
// ============================================================

class SubscriptionManager {
  constructor() {
    this.connections = new Map(); // connectionId -> { userId, subscriptions }
    this.nextId = 1;
  }

  // TODO: Implement connect(userId) — returns connectionId
  connect(userId) {
    const id = `conn_${this.nextId++}`;
    this.connections.set(id, { userId, subscriptions: new Set(), connectedAt: Date.now() });
    return id;
  }

  // TODO: Implement addSubscription(connectionId, subscriptionName)
  addSubscription(connectionId, subscriptionName) {
    const conn = this.connections.get(connectionId);
    if (!conn) throw new Error(`Connection ${connectionId} not found`);
    conn.subscriptions.add(subscriptionName);
  }

  // TODO: Implement disconnect(connectionId) — cleanup all subscriptions
  disconnect(connectionId) {
    const conn = this.connections.get(connectionId);
    if (!conn) return null;
    const subs = [...conn.subscriptions];
    this.connections.delete(connectionId);
    return { userId: conn.userId, removedSubscriptions: subs };
  }

  // TODO: Implement getStats() — return connection count, subscription count per channel
  getStats() {
    const stats = { connections: this.connections.size, subscriptions: {} };
    for (const [, conn] of this.connections) {
      for (const sub of conn.subscriptions) {
        stats.subscriptions[sub] = (stats.subscriptions[sub] || 0) + 1;
      }
    }
    return stats;
  }
}


// ============================================================
// Exercise 4: Event Batching for Subscriptions
// Implement event batching to reduce message frequency.
// ============================================================

class EventBatcher {
  constructor(intervalMs) {
    this.intervalMs = intervalMs;
    this.buffers = new Map(); // channel -> events[]
    this.listeners = new Map(); // channel -> callback
  }

  // TODO: Implement addEvent(channel, event) — buffers events
  addEvent(channel, event) {
    if (!this.buffers.has(channel)) {
      this.buffers.set(channel, []);
    }
    this.buffers.get(channel).push(event);
  }

  // TODO: Implement onFlush(channel, callback) — register flush handler
  onFlush(channel, callback) {
    this.listeners.set(channel, callback);
  }

  // TODO: Implement flush() — send batched events to listeners, clear buffers
  flush() {
    const results = {};
    for (const [channel, events] of this.buffers) {
      if (events.length > 0) {
        const listener = this.listeners.get(channel);
        if (listener) listener(events);
        results[channel] = events.length;
        this.buffers.set(channel, []);
      }
    }
    return results;
  }
}


// ============================================================
// Test all exercises
// ============================================================

console.log('=== Exercise 1: PubSub Event System ===\n');

const pubsub = new SimplePubSub();
const received = [];
const unsub = pubsub.subscribe('test', (msg) => received.push(msg));

pubsub.publish('test', { text: 'Hello' });
pubsub.publish('test', { text: 'World' });
unsub();
pubsub.publish('test', { text: 'After unsub' });

console.log(`Received ${received.length} messages (expected 2): ${received.length === 2 ? 'PASS' : 'FAIL'}`);
console.log(`Messages: ${received.map((m) => m.text).join(', ')}`);

console.log('\n=== Exercise 2: Subscription Filtering ===\n');

const chatPubSub = new SimplePubSub();
const sub = subscriptionResolvers.messageSent.subscribe(chatPubSub, { room: 'general' });

chatPubSub.publish('MESSAGE_SENT', { room: 'general', text: 'Hi all', user: 'Alice' });
chatPubSub.publish('MESSAGE_SENT', { room: 'random', text: 'Off-topic', user: 'Bob' });
chatPubSub.publish('MESSAGE_SENT', { room: 'general', text: 'Hello!', user: 'Charlie' });
sub.unsubscribe();

console.log(`Filtered messages: ${sub.events.length} (expected 2): ${sub.events.length === 2 ? 'PASS' : 'FAIL'}`);
sub.events.forEach((e) => console.log(`  [${e.room}] ${e.user}: ${e.text}`));

console.log('\n=== Exercise 3: Connection Management ===\n');

const manager = new SubscriptionManager();
const c1 = manager.connect('alice');
const c2 = manager.connect('bob');

manager.addSubscription(c1, 'messageSent');
manager.addSubscription(c1, 'userTyping');
manager.addSubscription(c2, 'messageSent');

const stats = manager.getStats();
console.log(`Connections: ${stats.connections} (expected 2): ${stats.connections === 2 ? 'PASS' : 'FAIL'}`);
console.log(`Subscriptions:`, stats.subscriptions);

const disconnected = manager.disconnect(c1);
console.log(`Disconnected ${disconnected.userId}, removed: ${disconnected.removedSubscriptions.join(', ')}`);
console.log(`Remaining connections: ${manager.getStats().connections}`);

console.log('\n=== Exercise 4: Event Batching ===\n');

const batcher = new EventBatcher(100);
let flushedEvents = [];
batcher.onFlush('updates', (events) => { flushedEvents = events; });

batcher.addEvent('updates', { type: 'price', product: 'p1', price: 10.99 });
batcher.addEvent('updates', { type: 'price', product: 'p2', price: 20.99 });
batcher.addEvent('updates', { type: 'stock', product: 'p1', stock: 5 });

const flushResult = batcher.flush();
console.log(`Flushed ${flushedEvents.length} events (expected 3): ${flushedEvents.length === 3 ? 'PASS' : 'FAIL'}`);
console.log(`Channels flushed:`, flushResult);

// Verify buffer is empty after flush
batcher.flush();
console.log(`Buffer empty after flush: ${flushedEvents.length === 3 ? 'PASS' : 'FAIL'}`);

console.log('\nAll exercises completed!');
