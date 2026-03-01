# 06. Subscriptions

**Previous**: [DataLoader and N+1](./05_DataLoader_N_plus_1.md) | **Next**: [Authentication and Authorization](./07_Authentication_Authorization.md)

---

GraphQL Subscriptions let the server push data to the client in real-time over a persistent connection. While queries and mutations follow the request-response cycle, subscriptions open a long-lived channel — typically a WebSocket — and deliver updates as they happen. Think of it this way: a query is like asking "What is the current stock price?", while a subscription is like saying "Notify me every time the stock price changes."

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how GraphQL subscriptions differ from queries and mutations
2. Implement server-side subscriptions using Apollo Server with `graphql-ws`
3. Configure PubSub systems (in-memory and Redis) for event distribution
4. Write subscription resolvers with filtering and transformation
5. Consume subscriptions on the client using Apollo Client's `useSubscription` hook

---

## Table of Contents

1. [Real-Time in GraphQL](#1-real-time-in-graphql)
2. [WebSocket Protocols](#2-websocket-protocols)
3. [Setting Up Subscriptions with Apollo Server](#3-setting-up-subscriptions-with-apollo-server)
4. [PubSub Systems](#4-pubsub-systems)
5. [Subscription Resolvers](#5-subscription-resolvers)
6. [Client-Side Subscriptions](#6-client-side-subscriptions)
7. [Filtering Subscriptions](#7-filtering-subscriptions)
8. [Scaling Subscriptions in Production](#8-scaling-subscriptions-in-production)
9. [Practice Problems](#9-practice-problems)
10. [References](#10-references)

---

## 1. Real-Time in GraphQL

GraphQL offers three root operation types. Queries read data, mutations write data, and subscriptions stream data. Subscriptions are defined in the schema just like queries and mutations:

```graphql
type Subscription {
  messageCreated(channelId: ID!): Message!
  orderStatusChanged(orderId: ID!): Order!
  stockPriceUpdated(symbol: String!): StockPrice!
}

type Message {
  id: ID!
  content: String!
  author: User!
  createdAt: String!
}
```

When a client subscribes, the server does not respond once and close the connection. Instead, it holds the connection open and sends a new result each time the subscribed event fires. The client receives these results as a stream.

### Subscriptions vs Polling vs Server-Sent Events

| Approach | Direction | Protocol | Best For |
|----------|-----------|----------|----------|
| Polling | Client → Server (repeated) | HTTP | Low-frequency updates, simple setup |
| Long Polling | Client → Server (held open) | HTTP | Moderate real-time, limited browser support |
| SSE | Server → Client (one-way) | HTTP | One-way streams (feeds, notifications) |
| WebSocket | Bidirectional | WS | Real-time bidirectional (chat, live data) |
| **GraphQL Subscriptions** | Server → Client | WS (typically) | Type-safe real-time with GraphQL schema |

GraphQL subscriptions combine the type safety of GraphQL's schema with the push model of WebSockets. The client specifies exactly which fields it needs — just like a query — and the server sends only those fields when an event occurs.

---

## 2. WebSocket Protocols

Two WebSocket sub-protocols have been used for GraphQL subscriptions. Understanding the difference is important because client and server must agree on the same protocol.

### `subscriptions-transport-ws` (Legacy)

The original protocol, maintained by Apollo. It is now **deprecated** and should not be used for new projects.

- Sub-protocol: `graphql-ws` (confusingly, this is the sub-protocol *name*, not the library)
- No built-in connection acknowledgment timeout
- Limited error handling
- No longer actively maintained

### `graphql-ws` (Current Standard)

The modern replacement, maintained by Denis Badurina. This is the library you should use.

- Sub-protocol: `graphql-transport-ws`
- Proper connection lifecycle (ConnectionInit → ConnectionAck)
- Built-in ping/pong for keep-alive
- Clean error handling and connection termination
- Supports both WebSocket and SSE transports

```
Timeline: Client ↔ Server (graphql-transport-ws protocol)

Client                                Server
  |--- ConnectionInit {payload} ------->|
  |<-- ConnectionAck {payload} ---------|
  |                                     |
  |--- Subscribe {id, payload} -------->|
  |<-- Next {id, payload} -------------|
  |<-- Next {id, payload} -------------|
  |<-- Next {id, payload} -------------|
  |                                     |
  |--- Complete {id} ------------------>|   (client unsubscribes)
  |                                     |
  |<-- Ping ----------------------------|
  |--- Pong --------------------------->|
```

**Key point**: When you see `graphql-ws` in documentation, check whether it refers to the *library* (the current standard) or the deprecated *sub-protocol name* used by the old `subscriptions-transport-ws` library.

---

## 3. Setting Up Subscriptions with Apollo Server

Apollo Server 4 does not handle WebSockets directly. Instead, you run an Express HTTP server for queries/mutations alongside a WebSocket server for subscriptions, both sharing the same schema.

### Installation

```bash
npm install @apollo/server express graphql graphql-ws ws
npm install @graphql-tools/schema
```

### Full Setup

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { createServer } from 'http';
import express from 'express';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import { makeExecutableSchema } from '@graphql-tools/schema';
import cors from 'cors';
import { typeDefs } from './schema';
import { resolvers } from './resolvers';

async function startServer() {
  const app = express();
  const httpServer = createServer(app);

  // Build executable schema (shared by HTTP and WS)
  const schema = makeExecutableSchema({ typeDefs, resolvers });

  // Create WebSocket server for subscriptions
  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/graphql',
  });

  // Set up graphql-ws
  const serverCleanup = useServer(
    {
      schema,
      // Context for subscription resolvers
      context: async (ctx) => {
        // ctx.connectionParams contains data sent by client
        // during ConnectionInit (e.g., auth tokens)
        return {
          token: ctx.connectionParams?.authToken,
        };
      },
      onConnect: async (ctx) => {
        console.log('Client connected for subscriptions');
        // Return false to reject the connection
      },
      onDisconnect: async (ctx, code, reason) => {
        console.log('Client disconnected');
      },
    },
    wsServer
  );

  // Create Apollo Server
  const server = new ApolloServer({
    schema,
    plugins: [
      // Proper shutdown for both HTTP and WS
      {
        async serverWillStart() {
          return {
            async drainServer() {
              await serverCleanup.dispose();
            },
          };
        },
      },
    ],
  });

  await server.start();

  app.use(
    '/graphql',
    cors<cors.CorsRequest>(),
    express.json(),
    expressMiddleware(server, {
      context: async ({ req }) => ({
        token: req.headers.authorization,
      }),
    })
  );

  httpServer.listen(4000, () => {
    console.log('Server running at http://localhost:4000/graphql');
    console.log('Subscriptions at ws://localhost:4000/graphql');
  });
}

startServer();
```

The critical insight is that queries and mutations flow through the Express middleware (HTTP), while subscriptions flow through the WebSocket server. Both use the same schema and resolvers, but their context functions are different — HTTP context reads from `req.headers`, while WebSocket context reads from `connectionParams`.

---

## 4. PubSub Systems

A PubSub (Publish-Subscribe) system acts as the event bus that connects mutation resolvers (publishers) to subscription resolvers (subscribers). When a mutation creates a new message, it publishes an event. The subscription resolver listens for that event and pushes the data to connected clients.

### In-Memory PubSub (Development Only)

Apollo provides a simple in-memory PubSub for development. It stores subscribers in a JavaScript `Map` and does not survive server restarts.

```typescript
// src/pubsub.ts
import { PubSub } from 'graphql-subscriptions';

// WARNING: In-memory PubSub is for development only.
// It does not work with multiple server instances because
// each instance has its own isolated Map of subscribers.
export const pubsub = new PubSub();

// Event name constants to avoid typos
export const EVENTS = {
  MESSAGE_CREATED: 'MESSAGE_CREATED',
  ORDER_STATUS_CHANGED: 'ORDER_STATUS_CHANGED',
  USER_TYPING: 'USER_TYPING',
} as const;
```

### Redis PubSub (Production)

For production, use Redis as the event bus. Redis PubSub broadcasts events to all server instances, so every client receives updates regardless of which server it connects to.

```bash
npm install graphql-redis-subscriptions ioredis
```

```typescript
// src/pubsub.ts
import { RedisPubSub } from 'graphql-redis-subscriptions';
import Redis from 'ioredis';

const options = {
  host: process.env.REDIS_HOST || '127.0.0.1',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  retryStrategy: (times: number) => Math.min(times * 50, 2000),
};

export const pubsub = new RedisPubSub({
  publisher: new Redis(options),
  subscriber: new Redis(options),
});

export const EVENTS = {
  MESSAGE_CREATED: 'MESSAGE_CREATED',
  ORDER_STATUS_CHANGED: 'ORDER_STATUS_CHANGED',
  USER_TYPING: 'USER_TYPING',
} as const;
```

The API is identical — you call `pubsub.publish()` and `pubsub.asyncIterableIterator()` the same way. The only difference is the transport layer.

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  Server #1  │       │    Redis     │       │  Server #2  │
│             │       │             │       │             │
│  publish()──┼──────▶│  channel    │──────▶│──subscribe  │
│             │       │             │       │  ──▶ WS #A  │
│  subscribe  │◀──────│             │       │  ──▶ WS #B  │
│  ──▶ WS #C  │       │             │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
```

---

## 5. Subscription Resolvers

Subscription resolvers differ from query and mutation resolvers. They have two functions: `subscribe` (which returns an async iterable) and an optional resolver function (which transforms the published payload before sending it to the client).

### Schema

```graphql
type Query {
  messages(channelId: ID!): [Message!]!
}

type Mutation {
  sendMessage(channelId: ID!, content: String!): Message!
}

type Subscription {
  messageCreated(channelId: ID!): Message!
}

type Message {
  id: ID!
  content: String!
  author: User!
  channel: Channel!
  createdAt: String!
}
```

### Resolvers

```typescript
// src/resolvers.ts
import { pubsub, EVENTS } from './pubsub';
import { withFilter } from 'graphql-subscriptions';
import { v4 as uuidv4 } from 'uuid';

export const resolvers = {
  Query: {
    messages: async (_, { channelId }, { db }) => {
      return db.messages.findByChannel(channelId);
    },
  },

  Mutation: {
    sendMessage: async (_, { channelId, content }, { currentUser, db }) => {
      const message = {
        id: uuidv4(),
        content,
        authorId: currentUser.id,
        channelId,
        createdAt: new Date().toISOString(),
      };

      await db.messages.create(message);

      // Publish the event — this triggers all active subscriptions
      await pubsub.publish(EVENTS.MESSAGE_CREATED, {
        messageCreated: message,
      });

      return message;
    },
  },

  Subscription: {
    messageCreated: {
      // subscribe() returns an AsyncIterableIterator.
      // The PubSub listens for MESSAGE_CREATED events and yields them.
      subscribe: () =>
        pubsub.asyncIterableIterator([EVENTS.MESSAGE_CREATED]),

      // Optional: transform the payload before sending to client.
      // If omitted, the payload is sent as-is.
      // The first argument is the published payload (not parent).
      resolve: (payload) => {
        return payload.messageCreated;
      },
    },
  },
};
```

### How the Data Flows

1. Client A sends a `sendMessage` mutation
2. The mutation resolver saves the message to the database
3. The mutation resolver calls `pubsub.publish(EVENT, payload)`
4. PubSub notifies all active subscription iterators listening for that event
5. Each subscription iterator yields the payload
6. The subscription's `resolve` function transforms the payload
7. GraphQL executes the normal field resolution (e.g., resolving `author` from `authorId`)
8. The result is pushed to Client B over WebSocket

---

## 6. Client-Side Subscriptions

### Apollo Client Setup

Apollo Client needs a "split link" that routes queries/mutations over HTTP and subscriptions over WebSocket.

```bash
npm install @apollo/client graphql graphql-ws
```

```typescript
// src/apollo-client.ts
import {
  ApolloClient,
  InMemoryCache,
  HttpLink,
  split,
} from '@apollo/client';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { createClient } from 'graphql-ws';
import { getMainDefinition } from '@apollo/client/utilities';

// HTTP link for queries and mutations
const httpLink = new HttpLink({
  uri: 'http://localhost:4000/graphql',
  headers: {
    authorization: `Bearer ${getToken()}`,
  },
});

// WebSocket link for subscriptions
const wsLink = new GraphQLWsLink(
  createClient({
    url: 'ws://localhost:4000/graphql',
    connectionParams: {
      authToken: getToken(),
    },
    // Automatic reconnection
    retryAttempts: 5,
    shouldRetry: () => true,
  })
);

// Split based on operation type:
// - Subscriptions go through WebSocket
// - Everything else goes through HTTP
const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,
  httpLink
);

export const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});
```

### useSubscription Hook

```tsx
// src/components/ChatRoom.tsx
import { useQuery, useMutation, useSubscription, gql } from '@apollo/client';

const GET_MESSAGES = gql`
  query GetMessages($channelId: ID!) {
    messages(channelId: $channelId) {
      id
      content
      author { name avatar }
      createdAt
    }
  }
`;

const SEND_MESSAGE = gql`
  mutation SendMessage($channelId: ID!, $content: String!) {
    sendMessage(channelId: $channelId, content: $content) {
      id
      content
      author { name avatar }
      createdAt
    }
  }
`;

const MESSAGE_SUBSCRIPTION = gql`
  subscription OnMessageCreated($channelId: ID!) {
    messageCreated(channelId: $channelId) {
      id
      content
      author { name avatar }
      createdAt
    }
  }
`;

function ChatRoom({ channelId }: { channelId: string }) {
  const { data, loading } = useQuery(GET_MESSAGES, {
    variables: { channelId },
  });

  const [sendMessage] = useMutation(SEND_MESSAGE);

  // useSubscription automatically manages the WebSocket lifecycle.
  // When the component mounts, it subscribes; on unmount, it unsubscribes.
  useSubscription(MESSAGE_SUBSCRIPTION, {
    variables: { channelId },
    onData: ({ data: subData }) => {
      // This fires each time the server pushes a new message.
      console.log('New message:', subData.data?.messageCreated);
    },
  });

  if (loading) return <p>Loading...</p>;

  return (
    <div>
      {data.messages.map((msg) => (
        <div key={msg.id}>
          <strong>{msg.author.name}:</strong> {msg.content}
        </div>
      ))}
    </div>
  );
}
```

### subscribeToMore: Merging Subscription Data with Query Cache

A common pattern is to use `subscribeToMore` instead of `useSubscription` when you want new data to automatically merge into an existing query's cache.

```tsx
function ChatRoom({ channelId }: { channelId: string }) {
  const { data, loading, subscribeToMore } = useQuery(GET_MESSAGES, {
    variables: { channelId },
  });

  useEffect(() => {
    const unsubscribe = subscribeToMore({
      document: MESSAGE_SUBSCRIPTION,
      variables: { channelId },
      updateQuery: (prev, { subscriptionData }) => {
        if (!subscriptionData.data) return prev;
        const newMessage = subscriptionData.data.messageCreated;

        // Return new data shape — Apollo merges it into cache
        return {
          messages: [...prev.messages, newMessage],
        };
      },
    });

    return () => unsubscribe();
  }, [channelId, subscribeToMore]);

  // ... render messages
}
```

The difference: `useSubscription` gives you the latest subscription result as a standalone value, while `subscribeToMore` appends or merges subscription results into an existing query's cached data.

---

## 7. Filtering Subscriptions

Not every client wants every event. A client subscribed to messages in channel "general" should not receive messages from channel "random". The `withFilter` helper filters events before they reach the client.

```typescript
import { withFilter } from 'graphql-subscriptions';

const resolvers = {
  Subscription: {
    messageCreated: {
      subscribe: withFilter(
        // First argument: the base iterator
        () => pubsub.asyncIterableIterator([EVENTS.MESSAGE_CREATED]),

        // Second argument: filter function.
        // Return true to send this event to this subscriber.
        // payload = published data, variables = subscription variables from client
        (payload, variables) => {
          return payload.messageCreated.channelId === variables.channelId;
        }
      ),
    },

    // More complex example: filter by user permissions
    orderStatusChanged: {
      subscribe: withFilter(
        () => pubsub.asyncIterableIterator([EVENTS.ORDER_STATUS_CHANGED]),
        (payload, variables, context) => {
          const order = payload.orderStatusChanged;
          // Only send to the customer who placed the order
          // or to admin users
          return (
            order.customerId === context.currentUser?.id ||
            context.currentUser?.role === 'ADMIN'
          );
        }
      ),
    },
  },
};
```

**Performance note**: `withFilter` runs for every event against every active subscriber. If you have 10,000 subscribers and 100 events per second, the filter function runs 1,000,000 times per second. For high-throughput systems, consider publishing to channel-specific events (e.g., `MESSAGE_CREATED.${channelId}`) instead of filtering after the fact.

```typescript
// Channel-specific publishing (more efficient for high throughput)
const resolvers = {
  Mutation: {
    sendMessage: async (_, { channelId, content }, ctx) => {
      const message = { /* ... */ };
      // Publish to a channel-specific event
      await pubsub.publish(`MESSAGE_CREATED.${channelId}`, {
        messageCreated: message,
      });
      return message;
    },
  },
  Subscription: {
    messageCreated: {
      subscribe: (_, { channelId }) =>
        // Subscribe to only the relevant channel
        pubsub.asyncIterableIterator([`MESSAGE_CREATED.${channelId}`]),
    },
  },
};
```

---

## 8. Scaling Subscriptions in Production

Subscriptions introduce stateful connections, which complicates horizontal scaling. Here are the key considerations.

### Connection Management

Each WebSocket connection consumes server memory. A server with 2GB RAM can typically handle 10,000-50,000 concurrent WebSocket connections, depending on message frequency and payload size.

```typescript
// Monitor connection count
const wsServer = new WebSocketServer({ server: httpServer, path: '/graphql' });

let connectionCount = 0;

useServer(
  {
    schema,
    onConnect: () => {
      connectionCount++;
      metrics.gauge('ws.connections', connectionCount);
      if (connectionCount > MAX_CONNECTIONS) {
        return false; // Reject new connections
      }
    },
    onDisconnect: () => {
      connectionCount--;
      metrics.gauge('ws.connections', connectionCount);
    },
  },
  wsServer
);
```

### Load Balancing

WebSocket connections are sticky — they must remain on the same server for their lifetime. Use sticky sessions or a layer-4 (TCP) load balancer.

```nginx
# nginx configuration for WebSocket proxy
upstream graphql_servers {
    # ip_hash ensures the same client reconnects to the same server
    ip_hash;
    server app1:4000;
    server app2:4000;
    server app3:4000;
}

server {
    location /graphql {
        proxy_pass http://graphql_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;  # Keep WS alive for 24h
        proxy_send_timeout 86400s;
    }
}
```

### Production Checklist

| Concern | Solution |
|---------|----------|
| Multi-instance events | Redis PubSub or Kafka |
| Connection limits | Max connections per server, backpressure |
| Authentication | Validate tokens in `onConnect`, reject invalid |
| Heartbeat | `graphql-ws` has built-in ping/pong |
| Reconnection | Client-side retry with exponential backoff |
| Monitoring | Track connection count, message throughput, error rate |
| Graceful shutdown | Drain connections before server stop |

---

## 9. Practice Problems

### Exercise 1: Basic Subscription (Beginner)

Implement a subscription for a typing indicator. When a user starts typing in a channel, all other users in that channel should see a "User X is typing..." indicator.

Define the schema types, mutation (to signal typing), subscription (to receive typing events), and resolvers. Use `withFilter` to ensure typing events are channel-specific.

### Exercise 2: Live Dashboard (Intermediate)

Build a real-time dashboard subscription system for a monitoring application. The server should:

- Track CPU usage, memory usage, and request count for multiple services
- Publish metrics every 5 seconds using `setInterval`
- Support subscribing to metrics for a specific service or all services
- Include a `threshold` argument so clients can choose to receive only alerts above a certain CPU percentage

Write the complete schema, resolvers, and a simulated metrics publisher.

### Exercise 3: Subscription with Cache Integration (Intermediate)

Using Apollo Client's `subscribeToMore`, build a notification bell component:

1. Initial query loads the 20 most recent notifications
2. A subscription listens for new notifications in real-time
3. New notifications are prepended to the cache (newest first)
4. An unread count badge updates automatically

Write the GraphQL operations (query + subscription), the React component, and the `updateQuery` function.

### Exercise 4: Redis PubSub with Filtering (Advanced)

Design a stock ticker subscription system:

- Clients subscribe to price updates for specific stock symbols (e.g., `["AAPL", "GOOGL"]`)
- The server publishes price updates for hundreds of symbols
- Use channel-specific events (`PRICE_UPDATE.${symbol}`) for efficiency instead of `withFilter`
- Handle the case where a client subscribes to multiple symbols simultaneously

Write the schema, resolvers, and explain how a single subscription can listen to multiple Redis channels.

### Exercise 5: Scaling Analysis (Advanced)

You have a chat application with 100,000 concurrent users across 10 server instances. Each user is subscribed to an average of 3 channels. Messages arrive at a rate of 500 per second across all channels.

1. Calculate the total number of WebSocket connections per server instance
2. Estimate the number of `withFilter` evaluations per second if using a single `MESSAGE_CREATED` event
3. Explain why channel-specific events (`MESSAGE_CREATED.${channelId}`) reduce this load
4. Design the Redis PubSub topology, including how many Redis connections each server needs
5. What happens if the Redis instance goes down? Propose a fallback strategy.

---

## 10. References

- graphql-ws library — https://github.com/enisdenjo/graphql-ws
- Apollo Server subscriptions documentation — https://www.apollographql.com/docs/apollo-server/data/subscriptions
- graphql-redis-subscriptions — https://github.com/davidyaha/graphql-redis-subscriptions
- GraphQL Specification: Subscriptions — https://spec.graphql.org/October2021/#sec-Subscription
- WebSocket Protocol (RFC 6455) — https://datatracker.ietf.org/doc/html/rfc6455

---

**Previous**: [DataLoader and N+1](./05_DataLoader_N_plus_1.md) | **Next**: [Authentication and Authorization](./07_Authentication_Authorization.md)
