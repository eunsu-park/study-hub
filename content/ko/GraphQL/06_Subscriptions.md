# 06. 구독(Subscriptions)

**이전**: [DataLoader와 N+1 문제](./05_DataLoader_N_plus_1.md) | **다음**: [인증과 인가](./07_Authentication_Authorization.md)

---

GraphQL 구독(Subscription)은 서버가 지속적인 연결을 통해 실시간으로 클라이언트에 데이터를 푸시(push)할 수 있게 합니다. 쿼리(Query)와 뮤테이션(Mutation)이 요청-응답 사이클을 따르는 반면, 구독은 장기(long-lived) 채널 — 일반적으로 WebSocket — 을 열고 이벤트가 발생할 때마다 업데이트를 전달합니다. 이렇게 생각해 보세요: 쿼리는 "현재 주가가 얼마인가요?"라고 묻는 것이고, 구독은 "주가가 변경될 때마다 알려주세요."라고 말하는 것입니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. GraphQL 구독이 쿼리 및 뮤테이션과 어떻게 다른지 설명할 수 있다
2. `graphql-ws`를 사용한 Apollo Server로 서버 측 구독을 구현할 수 있다
3. 이벤트 배포를 위한 PubSub 시스템(인메모리 및 Redis)을 구성할 수 있다
4. 필터링과 변환이 포함된 구독 리졸버를 작성할 수 있다
5. Apollo Client의 `useSubscription` 훅을 사용하여 클라이언트에서 구독을 소비할 수 있다

---

## 목차

1. [GraphQL에서의 실시간 처리](#1-graphql에서의-실시간-처리)
2. [WebSocket 프로토콜](#2-websocket-프로토콜)
3. [Apollo Server로 구독 설정하기](#3-apollo-server로-구독-설정하기)
4. [PubSub 시스템](#4-pubsub-시스템)
5. [구독 리졸버](#5-구독-리졸버)
6. [클라이언트 측 구독](#6-클라이언트-측-구독)
7. [구독 필터링](#7-구독-필터링)
8. [프로덕션에서 구독 확장](#8-프로덕션에서-구독-확장)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. GraphQL에서의 실시간 처리

GraphQL은 세 가지 루트 작업 타입을 제공합니다. 쿼리는 데이터를 읽고, 뮤테이션은 데이터를 쓰고, 구독은 데이터를 스트리밍합니다. 구독은 쿼리 및 뮤테이션과 동일하게 스키마에 정의됩니다:

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

클라이언트가 구독하면 서버는 한 번 응답하고 연결을 닫지 않습니다. 대신 연결을 열어 두고 구독된 이벤트가 발생할 때마다 새로운 결과를 전송합니다. 클라이언트는 이 결과를 스트림으로 수신합니다.

### 구독 vs 폴링(Polling) vs 서버-전송 이벤트(SSE)

| 방식 | 방향 | 프로토콜 | 적합한 사용 사례 |
|------|------|---------|----------------|
| 폴링(Polling) | 클라이언트 → 서버 (반복) | HTTP | 낮은 빈도 업데이트, 단순한 설정 |
| 롱 폴링(Long Polling) | 클라이언트 → 서버 (유지) | HTTP | 중간 수준 실시간, 제한적 브라우저 지원 |
| SSE | 서버 → 클라이언트 (단방향) | HTTP | 단방향 스트림 (피드, 알림) |
| WebSocket | 양방향 | WS | 실시간 양방향 (채팅, 라이브 데이터) |
| **GraphQL 구독** | 서버 → 클라이언트 | WS (일반적) | GraphQL 스키마를 활용한 타입 안전한 실시간 |

GraphQL 구독은 GraphQL 스키마의 타입 안전성과 WebSocket의 푸시 모델을 결합합니다. 클라이언트는 쿼리와 마찬가지로 필요한 필드를 정확히 지정하고, 서버는 이벤트가 발생할 때 해당 필드만 전송합니다.

---

## 2. WebSocket 프로토콜

GraphQL 구독에 두 가지 WebSocket 하위 프로토콜이 사용되어 왔습니다. 클라이언트와 서버가 동일한 프로토콜에 동의해야 하므로 차이점을 이해하는 것이 중요합니다.

### `subscriptions-transport-ws` (레거시)

Apollo가 관리하는 원래 프로토콜입니다. 현재 **더 이상 사용되지 않으며(deprecated)** 새 프로젝트에는 사용하지 않아야 합니다.

- 하위 프로토콜: `graphql-ws` (혼란스럽게도, 이것은 라이브러리가 아닌 하위 프로토콜 *이름*입니다)
- 내장 연결 확인 타임아웃 없음
- 제한적인 오류 처리
- 더 이상 활발히 유지 관리되지 않음

### `graphql-ws` (현재 표준)

Denis Badurina가 관리하는 최신 대체품입니다. 이것이 사용해야 할 라이브러리입니다.

- 하위 프로토콜: `graphql-transport-ws`
- 적절한 연결 생명주기 (ConnectionInit → ConnectionAck)
- keep-alive를 위한 내장 ping/pong
- 깔끔한 오류 처리 및 연결 종료
- WebSocket 및 SSE 전송 모두 지원

```
타임라인: 클라이언트 ↔ 서버 (graphql-transport-ws 프로토콜)

Client                                Server
  |--- ConnectionInit {payload} ------->|
  |<-- ConnectionAck {payload} ---------|
  |                                     |
  |--- Subscribe {id, payload} -------->|
  |<-- Next {id, payload} -------------|
  |<-- Next {id, payload} -------------|
  |<-- Next {id, payload} -------------|
  |                                     |
  |--- Complete {id} ------------------>|   (클라이언트가 구독 취소)
  |                                     |
  |<-- Ping ----------------------------|
  |--- Pong --------------------------->|
```

**핵심 포인트**: 문서에서 `graphql-ws`를 볼 때, 그것이 *라이브러리*(현재 표준)를 가리키는지, 아니면 오래된 `subscriptions-transport-ws` 라이브러리가 사용했던 더 이상 사용되지 않는 *하위 프로토콜 이름*을 가리키는지 확인하세요.

---

## 3. Apollo Server로 구독 설정하기

Apollo Server 4는 WebSocket을 직접 처리하지 않습니다. 대신 쿼리/뮤테이션을 위한 Express HTTP 서버와 구독을 위한 WebSocket 서버를 동일한 스키마를 공유하면서 함께 실행합니다.

### 설치

```bash
npm install @apollo/server express graphql graphql-ws ws
npm install @graphql-tools/schema
```

### 전체 설정

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

핵심은 쿼리와 뮤테이션은 Express 미들웨어(HTTP)를 통해 처리되는 반면, 구독은 WebSocket 서버를 통해 처리된다는 것입니다. 둘 다 동일한 스키마와 리졸버를 사용하지만, 컨텍스트 함수가 다릅니다 — HTTP 컨텍스트는 `req.headers`에서 읽고, WebSocket 컨텍스트는 `connectionParams`에서 읽습니다.

---

## 4. PubSub 시스템

PubSub(게시-구독, Publish-Subscribe) 시스템은 뮤테이션 리졸버(게시자)와 구독 리졸버(구독자)를 연결하는 이벤트 버스 역할을 합니다. 뮤테이션이 새 메시지를 생성하면 이벤트를 게시합니다. 구독 리졸버는 해당 이벤트를 수신하고 연결된 클라이언트에 데이터를 푸시합니다.

### 인메모리 PubSub (개발 전용)

Apollo는 개발용으로 간단한 인메모리 PubSub을 제공합니다. 구독자를 JavaScript `Map`에 저장하며 서버 재시작 후에는 유지되지 않습니다.

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

### Redis PubSub (프로덕션)

프로덕션에서는 Redis를 이벤트 버스로 사용하세요. Redis PubSub은 모든 서버 인스턴스에 이벤트를 브로드캐스트하므로, 어느 서버에 연결하든 모든 클라이언트가 업데이트를 받습니다.

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

API는 동일합니다 — `pubsub.publish()`와 `pubsub.asyncIterableIterator()`를 같은 방식으로 호출합니다. 유일한 차이점은 전송 계층입니다.

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

## 5. 구독 리졸버

구독 리졸버는 쿼리 및 뮤테이션 리졸버와 다릅니다. 두 가지 함수를 가집니다: `subscribe`(비동기 이터러블을 반환)와 선택적 리졸버 함수(클라이언트에 전송하기 전에 게시된 페이로드를 변환).

### 스키마

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

### 리졸버

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

### 데이터 흐름

1. 클라이언트 A가 `sendMessage` 뮤테이션을 전송합니다
2. 뮤테이션 리졸버가 데이터베이스에 메시지를 저장합니다
3. 뮤테이션 리졸버가 `pubsub.publish(EVENT, payload)`를 호출합니다
4. PubSub이 해당 이벤트를 수신 중인 모든 활성 구독 이터레이터에 알립니다
5. 각 구독 이터레이터가 페이로드를 yield합니다
6. 구독의 `resolve` 함수가 페이로드를 변환합니다
7. GraphQL이 일반 필드 해석을 실행합니다 (예: `authorId`에서 `author` 해석)
8. 결과가 WebSocket을 통해 클라이언트 B에 푸시됩니다

---

## 6. 클라이언트 측 구독

### Apollo Client 설정

Apollo Client는 쿼리/뮤테이션을 HTTP로 라우팅하고 구독을 WebSocket으로 라우팅하는 "스플릿 링크(split link)"가 필요합니다.

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

### useSubscription 훅

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

### subscribeToMore: 구독 데이터를 쿼리 캐시에 병합하기

새 데이터가 자동으로 기존 쿼리의 캐시에 병합되기를 원할 때 `useSubscription` 대신 `subscribeToMore`를 사용하는 것이 일반적인 패턴입니다.

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

차이점: `useSubscription`은 최신 구독 결과를 독립적인 값으로 제공하는 반면, `subscribeToMore`는 구독 결과를 기존 쿼리의 캐시된 데이터에 추가하거나 병합합니다.

---

## 7. 구독 필터링

모든 클라이언트가 모든 이벤트를 원하지는 않습니다. "general" 채널의 메시지를 구독한 클라이언트는 "random" 채널의 메시지를 받지 않아야 합니다. `withFilter` 헬퍼는 이벤트가 클라이언트에 도달하기 전에 필터링합니다.

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

**성능 참고**: `withFilter`는 모든 활성 구독자에 대해 모든 이벤트마다 실행됩니다. 구독자가 10,000명이고 초당 100개의 이벤트가 있다면, 필터 함수는 초당 1,000,000번 실행됩니다. 처리량이 높은 시스템의 경우, 사후 필터링 대신 채널별 이벤트(예: `MESSAGE_CREATED.${channelId}`)에 게시하는 것을 고려하세요.

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

## 8. 프로덕션에서 구독 확장

구독은 상태 저장(stateful) 연결을 도입하여 수평 확장을 복잡하게 만듭니다. 주요 고려 사항은 다음과 같습니다.

### 연결 관리

각 WebSocket 연결은 서버 메모리를 소비합니다. RAM이 2GB인 서버는 메시지 빈도와 페이로드 크기에 따라 일반적으로 10,000~50,000개의 동시 WebSocket 연결을 처리할 수 있습니다.

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

### 로드 밸런싱

WebSocket 연결은 스티키(sticky)합니다 — 수명 동안 동일한 서버에 유지되어야 합니다. 스티키 세션 또는 레이어-4(TCP) 로드 밸런서를 사용하세요.

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

### 프로덕션 체크리스트

| 우려 사항 | 해결책 |
|---------|-------|
| 다중 인스턴스 이벤트 | Redis PubSub 또는 Kafka |
| 연결 제한 | 서버당 최대 연결 수, 백프레셔(backpressure) |
| 인증(Authentication) | `onConnect`에서 토큰 검증, 유효하지 않은 경우 거부 |
| 하트비트(Heartbeat) | `graphql-ws`에 내장된 ping/pong |
| 재연결 | 지수 백오프(exponential backoff)를 사용한 클라이언트 측 재시도 |
| 모니터링 | 연결 수, 메시지 처리량, 오류율 추적 |
| 정상 종료(Graceful shutdown) | 서버 중지 전 연결 드레인(drain) |

---

## 9. 연습 문제

### 연습 1: 기본 구독 (입문)

타이핑 인디케이터를 위한 구독을 구현하세요. 사용자가 채널에서 타이핑을 시작하면 해당 채널의 다른 모든 사용자에게 "사용자 X가 타이핑 중..." 인디케이터가 표시되어야 합니다.

스키마 타입, 뮤테이션(타이핑 신호), 구독(타이핑 이벤트 수신), 리졸버를 정의하세요. `withFilter`를 사용하여 타이핑 이벤트가 채널별로 구분되도록 하세요.

### 연습 2: 라이브 대시보드 (중급)

모니터링 애플리케이션을 위한 실시간 대시보드 구독 시스템을 구축하세요. 서버는 다음을 수행해야 합니다:

- 여러 서비스의 CPU 사용량, 메모리 사용량, 요청 수를 추적
- `setInterval`을 사용하여 5초마다 메트릭을 게시
- 특정 서비스 또는 모든 서비스의 메트릭 구독 지원
- 클라이언트가 특정 CPU 비율 이상의 알림만 받을 수 있도록 `threshold` 인수 포함

완전한 스키마, 리졸버, 시뮬레이션된 메트릭 게시자를 작성하세요.

### 연습 3: 캐시 통합이 있는 구독 (중급)

Apollo Client의 `subscribeToMore`를 사용하여 알림 벨 컴포넌트를 구축하세요:

1. 초기 쿼리가 가장 최근 알림 20개를 로드
2. 구독이 실시간으로 새 알림을 수신
3. 새 알림이 캐시의 맨 앞에 추가됨 (최신순)
4. 읽지 않은 수 배지가 자동으로 업데이트됨

GraphQL 작업(쿼리 + 구독), React 컴포넌트, `updateQuery` 함수를 작성하세요.

### 연습 4: 필터링이 있는 Redis PubSub (고급)

주식 티커(stock ticker) 구독 시스템을 설계하세요:

- 클라이언트가 특정 주식 심볼(예: `["AAPL", "GOOGL"]`)의 가격 업데이트를 구독
- 서버가 수백 개 심볼의 가격 업데이트를 게시
- `withFilter` 대신 효율성을 위해 채널별 이벤트(`PRICE_UPDATE.${symbol}`) 사용
- 클라이언트가 여러 심볼을 동시에 구독하는 경우 처리

스키마, 리졸버를 작성하고 단일 구독이 여러 Redis 채널을 수신하는 방법을 설명하세요.

### 연습 5: 확장성 분석 (고급)

10개의 서버 인스턴스에 걸쳐 100,000명의 동시 사용자가 있는 채팅 애플리케이션이 있습니다. 각 사용자는 평균 3개의 채널을 구독하고 있습니다. 메시지는 모든 채널에서 초당 500개의 속도로 도착합니다.

1. 서버 인스턴스당 총 WebSocket 연결 수를 계산하세요
2. 단일 `MESSAGE_CREATED` 이벤트를 사용하는 경우 초당 `withFilter` 평가 수를 추정하세요
3. 채널별 이벤트(`MESSAGE_CREATED.${channelId}`)가 이 부하를 줄이는 이유를 설명하세요
4. 각 서버에 필요한 Redis 연결 수를 포함하여 Redis PubSub 토폴로지를 설계하세요
5. Redis 인스턴스가 다운되면 어떻게 되나요? 폴백(fallback) 전략을 제안하세요.

---

## 10. 참고 자료

- graphql-ws 라이브러리 — https://github.com/enisdenjo/graphql-ws
- Apollo Server 구독 문서 — https://www.apollographql.com/docs/apollo-server/data/subscriptions
- graphql-redis-subscriptions — https://github.com/davidyaha/graphql-redis-subscriptions
- GraphQL 명세: 구독 — https://spec.graphql.org/October2021/#sec-Subscription
- WebSocket 프로토콜 (RFC 6455) — https://datatracker.ietf.org/doc/html/rfc6455

---

**이전**: [DataLoader와 N+1 문제](./05_DataLoader_N_plus_1.md) | **다음**: [인증과 인가](./07_Authentication_Authorization.md)
