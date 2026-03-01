# 09. GraphQL Clients

**Previous**: [Apollo Server](./08_Apollo_Server.md) | **Next**: [Code-First with Python](./10_Code_First_Python.md)

---

A GraphQL client is responsible for sending queries to the server and managing the response data on the client side. While you could use `fetch` for every request, dedicated GraphQL clients provide caching, optimistic updates, real-time subscriptions, and developer tooling that dramatically improve both performance and developer experience. This lesson covers the three most popular approaches: Apollo Client (feature-rich), urql (lightweight), and TanStack Query with graphql-request (minimal).

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up Apollo Client 3 with InMemoryCache and configure cache policies
2. Explain how Apollo Client's normalized cache stores and updates data
3. Implement optimistic updates for instant UI feedback during mutations
4. Compare Apollo Client, urql, and TanStack Query for different use cases
5. Choose the right GraphQL client based on project requirements

---

## Table of Contents

1. [Why Use a GraphQL Client?](#1-why-use-a-graphql-client)
2. [Apollo Client 3 Setup](#2-apollo-client-3-setup)
3. [Queries with useQuery](#3-queries-with-usequery)
4. [Mutations with useMutation](#4-mutations-with-usemutation)
5. [Cache Normalization](#5-cache-normalization)
6. [Cache Policies](#6-cache-policies)
7. [Optimistic Updates](#7-optimistic-updates)
8. [urql: A Lightweight Alternative](#8-urql-a-lightweight-alternative)
9. [TanStack Query + graphql-request](#9-tanstack-query--graphql-request)
10. [Comparison: Choosing the Right Client](#10-comparison-choosing-the-right-client)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Why Use a GraphQL Client?

You could make GraphQL requests with plain `fetch`:

```typescript
const response = await fetch('/graphql', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: `query { user(id: "1") { name email } }`,
  }),
});
const { data } = await response.json();
```

This works, but you lose out on:

| Feature | fetch | GraphQL Client |
|---------|-------|---------------|
| Caching | Manual | Automatic (normalized) |
| Deduplication | Manual | Automatic (same query = one request) |
| Loading/error states | Manual | Built-in hooks |
| Optimistic UI | Complex | One config option |
| Subscriptions | Manual WebSocket management | Built-in |
| Pagination | Manual | Built-in helpers |
| DevTools | None | Browser extension |
| TypeScript | Manual typing | Generated from schema |

For anything beyond a trivial prototype, a dedicated client pays for itself.

---

## 2. Apollo Client 3 Setup

Apollo Client is the most feature-rich GraphQL client, maintained by Apollo GraphQL.

```bash
npm install @apollo/client graphql
```

```typescript
// src/apollo-client.ts
import {
  ApolloClient,
  InMemoryCache,
  HttpLink,
  ApolloLink,
} from '@apollo/client';

// Create the HTTP link (connection to your GraphQL server)
const httpLink = new HttpLink({
  uri: 'http://localhost:4000/graphql',
  // Cookies are sent with every request (useful for session auth)
  credentials: 'include',
});

// Optional: add an auth link that injects the JWT
const authLink = new ApolloLink((operation, forward) => {
  const token = localStorage.getItem('token');
  operation.setContext({
    headers: {
      authorization: token ? `Bearer ${token}` : '',
    },
  });
  return forward(operation);
});

// Create the client
export const client = new ApolloClient({
  link: ApolloLink.from([authLink, httpLink]),
  cache: new InMemoryCache(),
  defaultOptions: {
    watchQuery: {
      // How to handle errors: 'none' (default), 'all', 'ignore'
      errorPolicy: 'all',
    },
  },
});
```

### Provider Setup (React)

```tsx
// src/main.tsx
import { ApolloProvider } from '@apollo/client';
import { client } from './apollo-client';

function App() {
  return (
    <ApolloProvider client={client}>
      <Router />
    </ApolloProvider>
  );
}
```

---

## 3. Queries with useQuery

The `useQuery` hook executes a GraphQL query and returns reactive loading, error, and data states.

```tsx
import { useQuery, gql } from '@apollo/client';

const GET_POSTS = gql`
  query GetPosts($limit: Int!, $offset: Int!) {
    posts(limit: $limit, offset: $offset) {
      id
      title
      excerpt
      author {
        id
        name
        avatar
      }
      createdAt
    }
  }
`;

function PostList() {
  const { data, loading, error, refetch, fetchMore } = useQuery(GET_POSTS, {
    variables: { limit: 10, offset: 0 },
    // Polling: re-fetch every 30 seconds
    // pollInterval: 30000,
  });

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div>
      {data.posts.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}

      <button onClick={() => refetch()}>Refresh</button>

      <button
        onClick={() =>
          fetchMore({
            variables: { offset: data.posts.length },
            // updateQuery merges the new page into existing results
            updateQuery: (prev, { fetchMoreResult }) => ({
              posts: [...prev.posts, ...fetchMoreResult.posts],
            }),
          })
        }
      >
        Load More
      </button>
    </div>
  );
}
```

### Lazy Queries

Sometimes you do not want a query to run immediately on mount. `useLazyQuery` gives you a function to execute the query on demand.

```tsx
import { useLazyQuery, gql } from '@apollo/client';

const SEARCH_USERS = gql`
  query SearchUsers($term: String!) {
    searchUsers(term: $term) {
      id
      name
      email
    }
  }
`;

function UserSearch() {
  const [search, { data, loading }] = useLazyQuery(SEARCH_USERS);
  const [term, setTerm] = useState('');

  const handleSearch = () => {
    search({ variables: { term } });
  };

  return (
    <div>
      <input value={term} onChange={(e) => setTerm(e.target.value)} />
      <button onClick={handleSearch} disabled={loading}>
        Search
      </button>
      {data?.searchUsers.map((user) => (
        <div key={user.id}>{user.name}</div>
      ))}
    </div>
  );
}
```

---

## 4. Mutations with useMutation

```tsx
import { useMutation, gql } from '@apollo/client';

const CREATE_POST = gql`
  mutation CreatePost($input: CreatePostInput!) {
    createPost(input: $input) {
      id
      title
      excerpt
      author {
        id
        name
      }
    }
  }
`;

function CreatePostForm() {
  const [createPost, { loading, error }] = useMutation(CREATE_POST, {
    // After the mutation completes, refetch the posts list
    // so it includes the new post.
    refetchQueries: ['GetPosts'],

    // Or manually update the cache (more efficient, no extra request)
    update: (cache, { data: { createPost: newPost } }) => {
      cache.modify({
        fields: {
          posts(existingPosts = []) {
            const newPostRef = cache.writeFragment({
              data: newPost,
              fragment: gql`
                fragment NewPost on Post {
                  id
                  title
                  excerpt
                  author {
                    id
                    name
                  }
                }
              `,
            });
            return [newPostRef, ...existingPosts];
          },
        },
      });
    },

    onCompleted: (data) => {
      console.log('Post created:', data.createPost.id);
    },

    onError: (error) => {
      console.error('Failed to create post:', error.message);
    },
  });

  const handleSubmit = async (formData: FormData) => {
    await createPost({
      variables: {
        input: {
          title: formData.get('title'),
          content: formData.get('content'),
        },
      },
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <p className="error">{error.message}</p>}
      <input name="title" placeholder="Title" required />
      <textarea name="content" placeholder="Content" required />
      <button type="submit" disabled={loading}>
        {loading ? 'Creating...' : 'Create Post'}
      </button>
    </form>
  );
}
```

---

## 5. Cache Normalization

Apollo Client's most powerful feature is its normalized cache. Instead of storing query results as-is, it breaks every object into individual cache entries keyed by `__typename:id`.

### How Normalization Works

When this query result arrives:

```json
{
  "data": {
    "post": {
      "__typename": "Post",
      "id": "1",
      "title": "GraphQL Basics",
      "author": {
        "__typename": "User",
        "id": "42",
        "name": "Alice"
      }
    }
  }
}
```

Apollo does **not** store it like this. Instead, it normalizes it:

```
Cache:
  ROOT_QUERY
    post({"id":"1"}) → Reference("Post:1")

  Post:1
    __typename: "Post"
    id: "1"
    title: "GraphQL Basics"
    author: Reference("User:42")

  User:42
    __typename: "User"
    id: "42"
    name: "Alice"
```

Why does this matter? If another query also fetches `User:42` (say, through a `users` list query), they share the same cache entry. When you update Alice's name anywhere, every query that references `User:42` automatically sees the new name.

### Custom Cache Keys

By default, Apollo uses `__typename` + `id` (or `_id`). For types with different key fields, configure `typePolicies`:

```typescript
const cache = new InMemoryCache({
  typePolicies: {
    Product: {
      // This type uses 'sku' instead of 'id'
      keyFields: ['sku'],
    },
    UserSession: {
      // This type has no unique key — do not normalize
      keyFields: false,
    },
    AllPosts: {
      // Composite key
      keyFields: ['category', 'year'],
    },
  },
});
```

---

## 6. Cache Policies

Fetch policies control where Apollo Client reads data from (cache, network, or both).

| Policy | Read From | Write To | Use Case |
|--------|-----------|----------|----------|
| `cache-first` (default) | Cache, then network if miss | Cache | Most queries |
| `network-only` | Network always | Cache | Data that must be fresh |
| `cache-only` | Cache only | — | Offline-first apps |
| `no-cache` | Network always | — | Sensitive data (never store) |
| `cache-and-network` | Cache immediately, then network | Cache | Show stale, update when fresh arrives |

```tsx
// Show cached data immediately, then update when network responds
const { data, loading, networkStatus } = useQuery(GET_DASHBOARD, {
  fetchPolicy: 'cache-and-network',
  notifyOnNetworkStatusChange: true,
});

// loading is true only on initial load
// networkStatus === NetworkStatus.refetch means background update
const isRefreshing = networkStatus === NetworkStatus.refetch;

return (
  <div>
    <Dashboard data={data} />
    {isRefreshing && <small>Updating...</small>}
  </div>
);
```

### Next Fetch Policy

You can set different policies for the initial load vs subsequent re-renders:

```tsx
const { data } = useQuery(GET_NOTIFICATIONS, {
  // First render: always go to network for fresh data
  fetchPolicy: 'network-only',
  // Subsequent renders (e.g., navigating back): use cache
  nextFetchPolicy: 'cache-first',
});
```

---

## 7. Optimistic Updates

Optimistic updates make the UI feel instant by immediately applying the expected mutation result to the cache, then replacing it with the real server response when it arrives.

```tsx
const DELETE_POST = gql`
  mutation DeletePost($id: ID!) {
    deletePost(id: $id) {
      id
    }
  }
`;

function PostActions({ post }) {
  const [deletePost] = useMutation(DELETE_POST);

  const handleDelete = () => {
    deletePost({
      variables: { id: post.id },
      // Immediately update the cache as if the mutation succeeded
      optimisticResponse: {
        deletePost: {
          __typename: 'Post',
          id: post.id,
        },
      },
      // Remove the post from the cache
      update: (cache, { data: { deletePost: deletedPost } }) => {
        cache.modify({
          fields: {
            posts(existingPosts, { readField }) {
              return existingPosts.filter(
                (postRef) => readField('id', postRef) !== deletedPost.id
              );
            },
          },
        });
      },
    });
  };

  return <button onClick={handleDelete}>Delete</button>;
}
```

### How It Works

```
1. User clicks "Delete"
2. optimisticResponse is written to cache immediately
   → UI removes the post (< 1ms)
3. Network request fires in background
4. Server responds with real result
   a. Success → optimistic data is replaced with real data (no visible change)
   b. Error → optimistic data is rolled back (post reappears)
```

### Optimistic Update for Likes

A more complex example — a like button that toggles and updates a count:

```tsx
const TOGGLE_LIKE = gql`
  mutation ToggleLike($postId: ID!) {
    toggleLike(postId: $postId) {
      id
      isLikedByMe
      likeCount
    }
  }
`;

function LikeButton({ post }) {
  const [toggleLike] = useMutation(TOGGLE_LIKE);

  return (
    <button
      onClick={() =>
        toggleLike({
          variables: { postId: post.id },
          optimisticResponse: {
            toggleLike: {
              __typename: 'Post',
              id: post.id,
              // Toggle the current state
              isLikedByMe: !post.isLikedByMe,
              likeCount: post.isLikedByMe
                ? post.likeCount - 1
                : post.likeCount + 1,
            },
          },
          // No `update` needed — Apollo auto-merges by Post:id
        })
      }
    >
      {post.isLikedByMe ? 'Unlike' : 'Like'} ({post.likeCount})
    </button>
  );
}
```

Because the mutation returns fields with `id` and `__typename`, Apollo automatically updates the normalized `Post:{id}` cache entry. No manual `update` function needed.

---

## 8. urql: A Lightweight Alternative

urql (Universal React Query Library) is a smaller, more modular alternative to Apollo Client. It uses an "exchanges" architecture (similar to middleware) for extensibility.

```bash
npm install urql graphql @urql/exchange-graphcache
```

### Setup

```tsx
import { Client, Provider, cacheExchange, fetchExchange } from 'urql';

const client = new Client({
  url: 'http://localhost:4000/graphql',
  exchanges: [cacheExchange, fetchExchange],
  fetchOptions: () => {
    const token = localStorage.getItem('token');
    return {
      headers: { authorization: token ? `Bearer ${token}` : '' },
    };
  },
});

function App() {
  return (
    <Provider value={client}>
      <Router />
    </Provider>
  );
}
```

### Queries and Mutations

```tsx
import { useQuery, useMutation } from 'urql';

const PostsQuery = `
  query GetPosts($limit: Int!) {
    posts(limit: $limit) {
      id
      title
      author { name }
    }
  }
`;

function PostList() {
  const [result, reexecute] = useQuery({
    query: PostsQuery,
    variables: { limit: 10 },
  });

  const { data, fetching, error } = result;

  if (fetching) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div>
      {data.posts.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}
      <button onClick={() => reexecute({ requestPolicy: 'network-only' })}>
        Refresh
      </button>
    </div>
  );
}
```

### Normalized Cache (Graphcache)

urql's default cache is document-based (caches whole query results). For normalized caching (like Apollo), use Graphcache:

```typescript
import { cacheExchange } from '@urql/exchange-graphcache';

const cache = cacheExchange({
  keys: {
    // Custom key functions (like Apollo's keyFields)
    Product: (data) => data.sku as string,
  },
  updates: {
    Mutation: {
      createPost: (result, _args, cache) => {
        // Update the posts list when a post is created
        cache.updateQuery({ query: PostsQuery }, (data) => {
          if (data) {
            data.posts.unshift(result.createPost);
          }
          return data;
        });
      },
    },
  },
});
```

### Exchanges

urql's exchange system is its key differentiator. Exchanges are composable middleware that process operations:

```typescript
import { Client, fetchExchange } from 'urql';
import { cacheExchange } from '@urql/exchange-graphcache';
import { retryExchange } from '@urql/exchange-retry';
import { authExchange } from '@urql/exchange-auth';

const client = new Client({
  url: '/graphql',
  exchanges: [
    cacheExchange({}),
    retryExchange({
      retryIf: (error) => !!error.networkError,
      maxNumberAttempts: 3,
    }),
    authExchange(async (utils) => ({
      addAuthToOperation: (operation) =>
        utils.appendHeaders(operation, {
          Authorization: `Bearer ${getToken()}`,
        }),
      didAuthError: (error) =>
        error.graphQLErrors.some(
          (e) => e.extensions?.code === 'UNAUTHENTICATED'
        ),
      refreshAuth: async () => {
        await refreshToken();
      },
    })),
    fetchExchange,
  ],
});
```

---

## 9. TanStack Query + graphql-request

If you want a minimal setup — no normalized cache, no Apollo overhead — combine TanStack Query (formerly React Query) with `graphql-request`.

```bash
npm install @tanstack/react-query graphql-request graphql
```

### Setup

```tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GraphQLClient } from 'graphql-request';

export const gqlClient = new GraphQLClient('http://localhost:4000/graphql', {
  headers: () => ({
    authorization: `Bearer ${localStorage.getItem('token') || ''}`,
  }),
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
    </QueryClientProvider>
  );
}
```

### Queries

```tsx
import { useQuery } from '@tanstack/react-query';
import { gql } from 'graphql-request';
import { gqlClient } from './graphql-client';

const GET_POSTS = gql`
  query GetPosts($limit: Int!) {
    posts(limit: $limit) {
      id
      title
      author { name }
    }
  }
`;

function PostList() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['posts', { limit: 10 }],
    queryFn: () => gqlClient.request(GET_POSTS, { limit: 10 }),
  });

  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div>
      {data.posts.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}
      <button onClick={() => refetch()}>Refresh</button>
    </div>
  );
}
```

### Mutations

```tsx
import { useMutation, useQueryClient } from '@tanstack/react-query';

const CREATE_POST = gql`
  mutation CreatePost($input: CreatePostInput!) {
    createPost(input: $input) {
      id
      title
    }
  }
`;

function CreatePostForm() {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (input: CreatePostInput) =>
      gqlClient.request(CREATE_POST, { input }),
    onSuccess: () => {
      // Invalidate the posts cache so it refetches
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      mutation.mutate({ title: '...', content: '...' });
    }}>
      {/* form fields */}
      <button disabled={mutation.isPending}>Create</button>
    </form>
  );
}
```

This approach is simpler because TanStack Query uses key-based invalidation rather than normalized caching. When a mutation succeeds, you invalidate the relevant query keys, and those queries refetch automatically.

---

## 10. Comparison: Choosing the Right Client

| Feature | Apollo Client | urql | TanStack Query + graphql-request |
|---------|--------------|------|----------------------------------|
| **Bundle size** | ~40KB min+gz | ~8KB (core) | ~13KB (TQ) + ~5KB (gql-request) |
| **Normalized cache** | Built-in | Plugin (Graphcache) | No |
| **Optimistic updates** | Built-in | Via Graphcache | Manual (queryClient.setQueryData) |
| **Subscriptions** | Built-in | Via exchange | Manual WebSocket |
| **DevTools** | Excellent (browser ext) | Good (browser ext) | Excellent (browser ext) |
| **Learning curve** | Steep | Moderate | Low (if you know TanStack Query) |
| **TypeScript** | Good | Good | Good |
| **SSR support** | Yes | Yes | Yes |
| **Offline support** | Via apollo-link-persist | Via exchanges | Via TanStack Query offline plugin |
| **Community size** | Largest | Growing | Large (TanStack) |

### Decision Matrix

```
Need normalized cache + optimistic UI?
  └── Yes → Apollo Client or urql + Graphcache
  └── No  → Do you already use TanStack Query?
              └── Yes → TanStack Query + graphql-request
              └── No  → How important is bundle size?
                          └── Very → urql
                          └── Not critical → Apollo Client
```

**Summary**:
- **Apollo Client**: Best for complex apps with many interrelated entities (social networks, e-commerce, dashboards)
- **urql**: Best when you want Apollo-like features with a smaller footprint and modular architecture
- **TanStack Query**: Best when you want a simple, familiar data-fetching pattern and do not need normalized caching

---

## 11. Practice Problems

### Exercise 1: Apollo Client Setup with Auth (Beginner)

Set up an Apollo Client instance that:

1. Sends requests to `https://api.example.com/graphql`
2. Includes a JWT from `localStorage` in every request
3. If the server returns an `UNAUTHENTICATED` error, automatically redirects to `/login`
4. Uses `cache-and-network` as the default fetch policy

Write the complete client configuration including error handling link.

### Exercise 2: Optimistic Todo List (Intermediate)

Build a todo list application using Apollo Client with optimistic updates for three operations:

1. **Add todo**: Optimistically add the new todo at the top of the list
2. **Toggle complete**: Optimistically toggle the `completed` boolean
3. **Delete todo**: Optimistically remove the todo from the list

Handle the rollback case: if the server returns an error, the UI should revert to the previous state. Provide the GraphQL operations, React components, and optimistic update configurations.

### Exercise 3: urql Exchange Pipeline (Intermediate)

Build a custom urql exchange that:

1. Logs every operation (query/mutation name, variables) to the console
2. Measures and logs the duration of each operation
3. Retries failed network requests up to 3 times with exponential backoff
4. Does NOT retry mutations (only queries)

Write the exchange function following urql's exchange signature.

### Exercise 4: Pagination Strategy Comparison (Advanced)

Implement the same paginated post list using three different pagination strategies with Apollo Client:

1. **Offset-based**: `posts(limit: 10, offset: 20)` with `fetchMore`
2. **Cursor-based**: `posts(first: 10, after: "cursor123")` with relay-style connections
3. **Keyset**: `posts(limit: 10, afterId: "lastId")` with `fetchMore`

For each strategy, provide:
- The GraphQL query
- The `typePolicies` configuration for `InMemoryCache` to merge pages correctly
- The `fetchMore` call in the React component
- A brief analysis of when each strategy is appropriate

### Exercise 5: Client Migration (Advanced)

You have an existing application using Apollo Client with the following features:

- 15 queries with `useQuery`
- 8 mutations with `useMutation` and optimistic updates
- 2 subscriptions with `useSubscription`
- Normalized cache with custom `typePolicies`
- Response cache using `@cacheControl` headers

The team wants to evaluate migrating to urql. Write a migration plan that:

1. Identifies which Apollo features have direct urql equivalents
2. Identifies features that require additional urql exchanges or packages
3. Estimates which features would be lost or require significant rework
4. Proposes a phased migration strategy (not a big-bang rewrite)
5. Recommends whether to proceed with the migration, with justification

---

## 12. References

- Apollo Client 3 documentation — https://www.apollographql.com/docs/react/
- Apollo Client caching — https://www.apollographql.com/docs/react/caching/overview
- urql documentation — https://formidable.com/open-source/urql/docs/
- urql Graphcache — https://formidable.com/open-source/urql/docs/graphcache/
- TanStack Query documentation — https://tanstack.com/query/latest
- graphql-request — https://github.com/jasonkuhrt/graphql-request

---

**Previous**: [Apollo Server](./08_Apollo_Server.md) | **Next**: [Code-First with Python](./10_Code_First_Python.md)
