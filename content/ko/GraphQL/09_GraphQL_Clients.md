# 09. GraphQL 클라이언트(GraphQL Clients)

**이전**: [Apollo Server](./08_Apollo_Server.md) | **다음**: [Python 코드 우선 방식](./10_Code_First_Python.md)

---

GraphQL 클라이언트는 서버에 쿼리를 전송하고 클라이언트 측에서 응답 데이터를 관리하는 역할을 합니다. 모든 요청에 `fetch`를 사용할 수도 있지만, 전용 GraphQL 클라이언트는 캐싱, 낙관적 업데이트(optimistic updates), 실시간 구독, 개발자 도구를 제공하여 성능과 개발자 경험을 모두 크게 향상시킵니다. 이 레슨에서는 가장 인기 있는 세 가지 접근 방식을 다룹니다: Apollo Client(기능이 풍부함), urql(경량), TanStack Query + graphql-request(최소화).

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. InMemoryCache와 캐시 정책을 구성하여 Apollo Client 3을 설정할 수 있다
2. Apollo Client의 정규화(normalized) 캐시가 데이터를 저장하고 업데이트하는 방법을 설명할 수 있다
3. 뮤테이션 중 즉각적인 UI 피드백을 위한 낙관적 업데이트를 구현할 수 있다
4. 다양한 사용 사례에 따라 Apollo Client, urql, TanStack Query를 비교할 수 있다
5. 프로젝트 요구사항에 따라 적절한 GraphQL 클라이언트를 선택할 수 있다

---

## 목차

1. [GraphQL 클라이언트가 필요한 이유](#1-graphql-클라이언트가-필요한-이유)
2. [Apollo Client 3 설정](#2-apollo-client-3-설정)
3. [useQuery로 쿼리하기](#3-usequery로-쿼리하기)
4. [useMutation으로 뮤테이션하기](#4-usemutation으로-뮤테이션하기)
5. [캐시 정규화](#5-캐시-정규화)
6. [캐시 정책](#6-캐시-정책)
7. [낙관적 업데이트](#7-낙관적-업데이트)
8. [urql: 경량 대안](#8-urql-경량-대안)
9. [TanStack Query + graphql-request](#9-tanstack-query--graphql-request)
10. [비교: 올바른 클라이언트 선택](#10-비교-올바른-클라이언트-선택)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. GraphQL 클라이언트가 필요한 이유

일반 `fetch`로 GraphQL 요청을 만들 수 있습니다:

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

이것은 작동하지만 다음을 잃게 됩니다:

| 기능 | fetch | GraphQL 클라이언트 |
|------|-------|-----------------|
| 캐싱 | 수동 | 자동 (정규화) |
| 중복 제거 | 수동 | 자동 (같은 쿼리 = 한 번의 요청) |
| 로딩/오류 상태 | 수동 | 내장 훅 |
| 낙관적 UI | 복잡 | 설정 옵션 하나 |
| 구독 | 수동 WebSocket 관리 | 내장 |
| 페이지네이션 | 수동 | 내장 헬퍼 |
| DevTools | 없음 | 브라우저 확장 |
| TypeScript | 수동 타이핑 | 스키마에서 생성 |

사소한 프로토타입을 넘어서는 어떤 것에든 전용 클라이언트는 그 가치를 발휘합니다.

---

## 2. Apollo Client 3 설정

Apollo Client는 Apollo GraphQL이 유지 관리하는 가장 기능이 풍부한 GraphQL 클라이언트입니다.

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

### Provider 설정 (React)

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

## 3. useQuery로 쿼리하기

`useQuery` 훅은 GraphQL 쿼리를 실행하고 반응형 로딩, 오류, 데이터 상태를 반환합니다.

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

### 지연 쿼리(Lazy Queries)

때로는 마운트 시 즉시 쿼리를 실행하지 않으려는 경우가 있습니다. `useLazyQuery`는 필요에 따라 쿼리를 실행할 함수를 제공합니다.

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

## 4. useMutation으로 뮤테이션하기

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

## 5. 캐시 정규화

Apollo Client의 가장 강력한 기능은 정규화(normalized) 캐시입니다. 쿼리 결과를 그대로 저장하는 대신, 모든 객체를 `__typename:id`를 키로 하는 개별 캐시 항목으로 분리합니다.

### 정규화 작동 방식

이 쿼리 결과가 도착하면:

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

Apollo는 이것을 **그대로 저장하지 않습니다**. 대신 정규화합니다:

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

이것이 왜 중요할까요? 다른 쿼리도 `User:42`를 가져온다면(예를 들어 `users` 목록 쿼리를 통해), 그들은 동일한 캐시 항목을 공유합니다. Alice의 이름을 어디서든 업데이트하면 `User:42`를 참조하는 모든 쿼리가 자동으로 새 이름을 보게 됩니다.

### 커스텀 캐시 키

기본적으로 Apollo는 `__typename` + `id` (또는 `_id`)를 사용합니다. 다른 키 필드를 가진 타입의 경우 `typePolicies`를 구성하세요:

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

## 6. 캐시 정책

페치 정책(fetch policies)은 Apollo Client가 데이터를 읽는 위치(캐시, 네트워크, 또는 둘 다)를 제어합니다.

| 정책 | 읽기 위치 | 쓰기 위치 | 사용 사례 |
|------|---------|---------|---------|
| `cache-first` (기본) | 캐시, 미스 시 네트워크 | 캐시 | 대부분의 쿼리 |
| `network-only` | 항상 네트워크 | 캐시 | 항상 최신이어야 하는 데이터 |
| `cache-only` | 캐시만 | — | 오프라인 우선 앱 |
| `no-cache` | 항상 네트워크 | — | 민감한 데이터 (절대 저장 안 함) |
| `cache-and-network` | 즉시 캐시, 그 다음 네트워크 | 캐시 | 오래된 것을 보여주고 신선한 것이 도착하면 업데이트 |

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

초기 로드와 이후 다시 렌더링에 대해 다른 정책을 설정할 수 있습니다:

```tsx
const { data } = useQuery(GET_NOTIFICATIONS, {
  // First render: always go to network for fresh data
  fetchPolicy: 'network-only',
  // Subsequent renders (e.g., navigating back): use cache
  nextFetchPolicy: 'cache-first',
});
```

---

## 7. 낙관적 업데이트

낙관적 업데이트(Optimistic updates)는 예상되는 뮤테이션 결과를 즉시 캐시에 적용하고, 서버 응답이 도착하면 실제 데이터로 교체함으로써 UI를 즉각적으로 느끼게 합니다.

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

### 작동 방식

```
1. 사용자가 "Delete"를 클릭
2. optimisticResponse가 즉시 캐시에 기록됨
   → UI가 게시물을 제거 (< 1ms)
3. 백그라운드에서 네트워크 요청 실행
4. 서버가 실제 결과로 응답
   a. 성공 → 낙관적 데이터가 실제 데이터로 교체됨 (변화 없음)
   b. 오류 → 낙관적 데이터가 롤백됨 (게시물이 다시 나타남)
```

### 좋아요 버튼의 낙관적 업데이트

더 복잡한 예시 — 토글하고 카운트를 업데이트하는 좋아요 버튼:

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

뮤테이션이 `id`와 `__typename`이 있는 필드를 반환하기 때문에, Apollo는 자동으로 정규화된 `Post:{id}` 캐시 항목을 업데이트합니다. 수동 `update` 함수가 필요하지 않습니다.

---

## 8. urql: 경량 대안

urql(Universal React Query Library)은 Apollo Client의 더 작고 더 모듈화된 대안입니다. 확장성을 위해 "exchanges" 아키텍처(미들웨어와 유사)를 사용합니다.

```bash
npm install urql graphql @urql/exchange-graphcache
```

### 설정

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

### 쿼리와 뮤테이션

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

### 정규화 캐시 (Graphcache)

urql의 기본 캐시는 문서 기반입니다(전체 쿼리 결과를 캐시). Apollo와 같은 정규화 캐시를 위해서는 Graphcache를 사용하세요:

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

urql의 exchange 시스템이 핵심 차별점입니다. Exchange는 작업을 처리하는 조합 가능한 미들웨어입니다:

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

정규화 캐시가 없고 Apollo 오버헤드도 없는 최소화된 설정을 원한다면, TanStack Query(이전 React Query)와 `graphql-request`를 결합하세요.

```bash
npm install @tanstack/react-query graphql-request graphql
```

### 설정

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

### 쿼리

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

### 뮤테이션

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

이 접근 방식은 TanStack Query가 정규화 캐싱 대신 키 기반 무효화를 사용하기 때문에 더 단순합니다. 뮤테이션이 성공하면 관련 쿼리 키를 무효화하고 해당 쿼리는 자동으로 다시 fetch합니다.

---

## 10. 비교: 올바른 클라이언트 선택

| 기능 | Apollo Client | urql | TanStack Query + graphql-request |
|------|-------------|------|----------------------------------|
| **번들 크기** | ~40KB min+gz | ~8KB (코어) | ~13KB (TQ) + ~5KB (gql-request) |
| **정규화 캐시** | 내장 | 플러그인 (Graphcache) | 없음 |
| **낙관적 업데이트** | 내장 | Graphcache를 통해 | 수동 (queryClient.setQueryData) |
| **구독** | 내장 | exchange를 통해 | 수동 WebSocket |
| **DevTools** | 우수 (브라우저 확장) | 좋음 (브라우저 확장) | 우수 (브라우저 확장) |
| **학습 곡선** | 가파름 | 중간 | 낮음 (TanStack Query를 안다면) |
| **TypeScript** | 좋음 | 좋음 | 좋음 |
| **SSR 지원** | 예 | 예 | 예 |
| **오프라인 지원** | apollo-link-persist를 통해 | exchange를 통해 | TanStack Query 오프라인 플러그인을 통해 |
| **커뮤니티 크기** | 가장 큼 | 성장 중 | 큼 (TanStack) |

### 결정 매트릭스

```
정규화 캐시 + 낙관적 UI가 필요한가?
  └── 예 → Apollo Client 또는 urql + Graphcache
  └── 아니오 → 이미 TanStack Query를 사용하고 있는가?
              └── 예 → TanStack Query + graphql-request
              └── 아니오 → 번들 크기가 얼마나 중요한가?
                          └── 매우 중요 → urql
                          └── 크게 중요하지 않음 → Apollo Client
```

**요약**:
- **Apollo Client**: 상호 연관된 엔티티가 많은 복잡한 앱 (소셜 네트워크, 이커머스, 대시보드)에 최적
- **urql**: 더 작은 설치 공간과 모듈형 아키텍처로 Apollo와 유사한 기능을 원할 때 최적
- **TanStack Query**: 정규화 캐싱이 필요 없고 단순하고 친숙한 데이터 페칭 패턴을 원할 때 최적

---

## 11. 연습 문제

### 연습 1: 인증이 있는 Apollo Client 설정 (입문)

다음을 수행하는 Apollo Client 인스턴스를 설정하세요:

1. `https://api.example.com/graphql`로 요청 전송
2. 모든 요청에 `localStorage`의 JWT 포함
3. 서버가 `UNAUTHENTICATED` 오류를 반환하면 자동으로 `/login`으로 리다이렉트
4. 기본 페치 정책으로 `cache-and-network` 사용

오류 처리 링크를 포함한 완전한 클라이언트 구성을 작성하세요.

### 연습 2: 낙관적 할 일 목록 (중급)

세 가지 작업에 대한 낙관적 업데이트를 사용하여 Apollo Client로 할 일 목록 애플리케이션을 구축하세요:

1. **할 일 추가**: 새 할 일을 목록 맨 위에 낙관적으로 추가
2. **완료 토글**: `completed` 불리언을 낙관적으로 토글
3. **할 일 삭제**: 목록에서 할 일을 낙관적으로 제거

롤백 케이스 처리: 서버가 오류를 반환하면 UI가 이전 상태로 되돌아가야 합니다. GraphQL 작업, React 컴포넌트, 낙관적 업데이트 구성을 제공하세요.

### 연습 3: urql Exchange 파이프라인 (중급)

다음을 수행하는 커스텀 urql exchange를 구축하세요:

1. 모든 작업 (쿼리/뮤테이션 이름, 변수)을 콘솔에 로깅
2. 각 작업의 시간을 측정하고 기록
3. 실패한 네트워크 요청을 지수 백오프(exponential backoff)로 최대 3번 재시도
4. 뮤테이션은 재시도하지 않음 (쿼리만)

urql의 exchange 시그니처를 따르는 exchange 함수를 작성하세요.

### 연습 4: 페이지네이션 전략 비교 (고급)

Apollo Client를 사용하여 세 가지 다른 페이지네이션 전략으로 동일한 페이지화된 게시물 목록을 구현하세요:

1. **오프셋 기반**: `fetchMore`와 함께 `posts(limit: 10, offset: 20)` 사용
2. **커서 기반**: relay 스타일 연결과 함께 `posts(first: 10, after: "cursor123")` 사용
3. **Keyset**: `fetchMore`와 함께 `posts(limit: 10, afterId: "lastId")` 사용

각 전략에 대해:
- GraphQL 쿼리
- 페이지를 올바르게 병합하기 위한 `InMemoryCache`의 `typePolicies` 구성
- React 컴포넌트의 `fetchMore` 호출
- 각 전략이 언제 적합한지에 대한 간략한 분석

### 연습 5: 클라이언트 마이그레이션 (고급)

다음 기능을 갖춘 Apollo Client를 사용하는 기존 애플리케이션이 있습니다:

- `useQuery`가 있는 쿼리 15개
- 낙관적 업데이트가 있는 `useMutation` 8개
- `useSubscription`이 있는 구독 2개
- 커스텀 `typePolicies`가 있는 정규화 캐시
- `@cacheControl` 헤더를 사용하는 응답 캐시

팀이 urql로 마이그레이션하는 것을 평가하려 합니다. 다음을 포함한 마이그레이션 계획을 작성하세요:

1. urql에 직접적인 동등물이 있는 Apollo 기능 식별
2. 추가 urql exchange나 패키지가 필요한 기능 식별
3. 손실될 기능이나 상당한 재작업이 필요한 기능 추정
4. 단계적 마이그레이션 전략 제안 (빅뱅 재작성이 아닌)
5. 근거와 함께 마이그레이션 진행 여부 권장

---

## 12. 참고 자료

- Apollo Client 3 문서 — https://www.apollographql.com/docs/react/
- Apollo Client 캐싱 — https://www.apollographql.com/docs/react/caching/overview
- urql 문서 — https://formidable.com/open-source/urql/docs/
- urql Graphcache — https://formidable.com/open-source/urql/docs/graphcache/
- TanStack Query 문서 — https://tanstack.com/query/latest
- graphql-request — https://github.com/jasonkuhrt/graphql-request

---

**이전**: [Apollo Server](./08_Apollo_Server.md) | **다음**: [Python 코드 우선 방식](./10_Code_First_Python.md)
