/**
 * GraphQL Subscriptions — Real-time with WebSocket
 * Demonstrates: PubSub, subscription resolvers, graphql-ws.
 *
 * Run: npm install @apollo/server graphql graphql-ws ws express cors
 *      node 04_subscriptions.js
 */

const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const { ApolloServerPluginDrainHttpServer } = require('@apollo/server/plugin/drainHttpServer');
const { makeExecutableSchema } = require('@graphql-tools/schema');
const { WebSocketServer } = require('ws');
const { useServer } = require('graphql-ws/lib/use/ws');
const { PubSub } = require('graphql-subscriptions');
const express = require('express');
const cors = require('cors');
const http = require('http');

const pubsub = new PubSub();
const POST_CREATED = 'POST_CREATED';
const POST_UPDATED = 'POST_UPDATED';

// --- Schema ---

const typeDefs = `#graphql
  type Post {
    id: ID!
    title: String!
    content: String!
    author: String!
    createdAt: String!
  }

  type Query {
    posts: [Post!]!
  }

  type Mutation {
    createPost(title: String!, content: String!, author: String!): Post!
    updatePost(id: ID!, title: String, content: String): Post
  }

  type Subscription {
    postCreated: Post!
    postUpdated(id: ID): Post!
  }
`;

// --- Data ---

const posts = [];
let nextId = 1;

// --- Resolvers ---

const resolvers = {
  Query: {
    posts: () => posts,
  },

  Mutation: {
    createPost: (_, { title, content, author }) => {
      const post = {
        id: String(nextId++),
        title,
        content,
        author,
        createdAt: new Date().toISOString(),
      };
      posts.push(post);

      // Publish to subscribers
      pubsub.publish(POST_CREATED, { postCreated: post });
      return post;
    },

    updatePost: (_, { id, title, content }) => {
      const post = posts.find((p) => p.id === id);
      if (!post) return null;
      if (title) post.title = title;
      if (content) post.content = content;

      pubsub.publish(POST_UPDATED, { postUpdated: post });
      return post;
    },
  },

  Subscription: {
    postCreated: {
      subscribe: () => pubsub.asyncIterableIterator([POST_CREATED]),
    },

    postUpdated: {
      subscribe: (_, { id }) => {
        const iterator = pubsub.asyncIterableIterator([POST_UPDATED]);
        // Filter: only send updates for a specific post if ID is provided
        if (!id) return iterator;

        return {
          [Symbol.asyncIterator]() {
            return {
              async next() {
                while (true) {
                  const result = await iterator.next();
                  if (result.done) return result;
                  if (result.value.postUpdated.id === id) return result;
                }
              },
              return: iterator.return?.bind(iterator),
            };
          },
        };
      },
    },
  },
};

// --- Server Setup ---

async function main() {
  const app = express();
  const httpServer = http.createServer(app);

  const schema = makeExecutableSchema({ typeDefs, resolvers });

  // WebSocket server for subscriptions
  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/graphql',
  });

  const serverCleanup = useServer(
    {
      schema,
      onConnect: async (ctx) => {
        console.log('Client connected via WebSocket');
      },
      onDisconnect: (ctx) => {
        console.log('Client disconnected');
      },
    },
    wsServer
  );

  // Apollo Server
  const server = new ApolloServer({
    schema,
    plugins: [
      ApolloServerPluginDrainHttpServer({ httpServer }),
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

  app.use('/graphql', cors(), express.json(), expressMiddleware(server));

  httpServer.listen(4000, () => {
    console.log('Server ready at http://localhost:4000/graphql');
    console.log('WebSocket subscriptions at ws://localhost:4000/graphql');
    console.log('\nSubscription query:');
    console.log('  subscription { postCreated { id title author } }');
    console.log('\nThen create a post in another tab:');
    console.log('  mutation { createPost(title: "Live!", content: "Real-time", author: "Alice") { id } }');
  });
}

main();
