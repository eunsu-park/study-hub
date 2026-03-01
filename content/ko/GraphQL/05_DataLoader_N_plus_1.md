# 05. DataLoader와 N+1 문제

**이전**: [리졸버](./04_Resolvers.md) | **다음**: [구독](./06_Subscriptions.md)

---

N+1 문제는 GraphQL에서 가장 흔한 성능 함정입니다. 각 필드 리졸버가 독립적이기 때문에, N개의 항목 목록을 가져온 후 각 항목에 대한 관련 엔티티를 해결하면 데이터베이스 쿼리가 폭발적으로 증가합니다. DataLoader는 단일 요청 틱(Tick) 내에서 개별 로드를 배칭(Batching)하고 캐싱하여 이 문제를 해결합니다. 이 레슨에서는 N+1 문제가 왜 발생하는지, DataLoader가 어떻게 해결하는지, 프로덕션에서 DataLoader를 올바르게 구현하는 방법을 설명합니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GraphQL 리졸버에서 N+1 쿼리 패턴을 식별하고 총 쿼리 수 계산
2. DataLoader의 배칭(Batching) 메커니즘이 개별 로드를 단일 배치 호출로 통합하는 방법 설명
3. 다양한 데이터 관계(일대일, 일대다)를 위한 DataLoader 배치 함수 구현
4. DataLoader의 캐싱 동작 구성 및 요청별 범위 이해
5. 쿼리 로깅과 트레이싱을 사용해 프로덕션에서 N+1 문제 모니터링 및 감지

---

## 목차

1. [N+1 문제 설명](#1-n1-문제-설명)
2. [GraphQL이 N+1에 특히 취약한 이유](#2-graphql이-n1에-특히-취약한-이유)
3. [DataLoader 소개](#3-dataloader-소개)
4. [배칭: 작동 방식](#4-배칭-작동-방식)
5. [DataLoader 구현](#5-dataloader-구현)
6. [일대다 관계를 위한 DataLoader](#6-일대다-관계를-위한-dataloader)
7. [캐싱 동작](#7-캐싱-동작)
8. [요청별 DataLoader 인스턴스](#8-요청별-dataloader-인스턴스)
9. [다양한 데이터 소스와 DataLoader](#9-다양한-데이터-소스와-dataloader)
10. [N+1 모니터링 및 감지](#10-n1-모니터링-및-감지)
11. [흔한 함정](#11-흔한-함정)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. N+1 문제 설명

N+1 문제는 N개의 항목 목록을 가져올 때 목록을 위한 1번의 쿼리와 각 항목의 관련 엔티티를 위한 N번의 추가 쿼리가 필요할 때 발생합니다.

### 1.1 구체적인 예시

다음 스키마와 쿼리를 생각해 보세요:

```graphql
type Query {
  posts(first: Int): [Post!]!
}

type Post {
  id: ID!
  title: String!
  author: User!        # 각 게시글에는 하나의 작성자가 있음
}

type User {
  id: ID!
  name: String!
}
```

```graphql
query {
  posts(first: 10) {
    title
    author {
      name
    }
  }
}
```

그리고 이런 순진한 리졸버:

```javascript
const resolvers = {
  Query: {
    posts: (_, { first }, ctx) => {
      return ctx.db.query('SELECT * FROM posts LIMIT $1', [first]);
      // 쿼리 1: SELECT * FROM posts LIMIT 10
    },
  },
  Post: {
    author: (parent, _, ctx) => {
      return ctx.db.query('SELECT * FROM users WHERE id = $1', [parent.author_id]);
      // 쿼리 2:  SELECT * FROM users WHERE id = 'user_1'
      // 쿼리 3:  SELECT * FROM users WHERE id = 'user_2'
      // 쿼리 4:  SELECT * FROM users WHERE id = 'user_1'  ← 중복!
      // 쿼리 5:  SELECT * FROM users WHERE id = 'user_3'
      // ...
      // 쿼리 11: SELECT * FROM users WHERE id = 'user_5'
    },
  },
};
```

**결과: 단순한 10개 게시글 페이지에 1 + 10 = 11번의 쿼리**. 여러 게시글이 동일한 작성자를 공유하면, 같은 사용자를 여러 번 가져오는 일까지 발생합니다.

### 1.2 문제의 확장성

N+1 문제는 중첩될수록 복합적으로 증가합니다:

```graphql
query {
  posts(first: 10) {           # 1번 쿼리
    title
    author {                    # 10번 쿼리
      name
      followers(first: 5) {     # 10번 쿼리 (작성자당 1번)
        name                    # 50번 쿼리 (팔로워당 1번)
      }
    }
    comments(first: 5) {       # 10번 쿼리
      body
      author {                  # 50번 쿼리
        name
      }
    }
  }
}
```

**총계: 1 + 10 + 10 + 50 + 10 + 50 = 131번의 쿼리** — 단 하나의 GraphQL 요청에서! REST API에서는 서버가 데이터 조회 전략을 제어합니다. GraphQL에서는 클라이언트가 쿼리 형태를 제어하므로, 서버는 임의적인 중첩에 대비해야 합니다.

## 2. GraphQL이 N+1에 특히 취약한 이유

### 2.1 리졸버 독립성

각 리졸버는 다른 리졸버에 대해 알지 못하는 독립적인 함수입니다. `Post.author`는 자신이 10개의 다른 게시글에 대해 10번 호출될 것이라는 사실을 모릅니다. 단지 하나의 parent 객체를 받아 하나의 결과를 반환할 뿐입니다.

```
Query.posts → [post1, post2, ..., post10]
  ├── Post.author(post1)  → db.query("SELECT ... WHERE id = 1")
  ├── Post.author(post2)  → db.query("SELECT ... WHERE id = 2")
  ├── Post.author(post3)  → db.query("SELECT ... WHERE id = 1")  ← 중복
  ├── Post.author(post4)  → db.query("SELECT ... WHERE id = 3")
  └── ... (10개의 별도 쿼리)
```

### 2.2 클라이언트가 제어하는 깊이

REST API는 예측 가능한 쿼리 패턴을 가진 고정된 엔드포인트를 가집니다. GraphQL은 클라이언트가 임의로 깊이 중첩할 수 있습니다:

```graphql
# 클라이언트가 이런 쿼리를 보낼 수 있습니다
query DeeplyNested {
  user(id: "1") {
    posts {
      comments {
        author {
          posts {
            comments {
              author { name }
            }
          }
        }
      }
    }
  }
}
```

보호 장치 없이는 모든 수준에서 N+1 문제의 연쇄가 발생합니다.

### 2.3 왜 그냥 JOIN을 사용하지 않는가?

"그냥 SQL JOIN을 사용하면 되지 않나요?"라고 생각할 수 있습니다. 문제는 리졸버가 실행 시점에 전체 쿼리 형태를 알지 못한다는 것입니다. `Query.posts`는 `Post.author`도 요청될 것이라는 사실을 모릅니다 — 그냥 게시글을 반환할 뿐입니다. 알더라도, "혹시 모르니" 모든 것을 미리 로드하는 것은 클라이언트가 해당 필드를 요청하지 않은 경우 자원을 낭비합니다.

DataLoader는 더 나은 해결책을 제공합니다: **지연된 배칭(Deferred Batching)**.

## 3. DataLoader 소개

DataLoader는 원래 Facebook이 만든 유틸리티 라이브러리로, 두 가지 핵심 기능을 제공합니다:

1. **배칭(Batching)**: 단일 이벤트 루프 틱 내의 개별 로드를 수집하여 배치로 실행
2. **캐싱(Caching)**: 단일 요청 내에서 동일한 키에 대한 요청을 중복 제거

```
DataLoader 없음                   DataLoader 있음
──────────────────               ─────────────────
Post.author(id: 1) → SQL 쿼리   Post.author(id: 1) → loader.load(1)
Post.author(id: 2) → SQL 쿼리   Post.author(id: 2) → loader.load(2)
Post.author(id: 1) → SQL 쿼리   Post.author(id: 1) → loader.load(1) ← 캐시 히트!
Post.author(id: 3) → SQL 쿼리   Post.author(id: 3) → loader.load(3)
                                  ──── 틱 경계 ────
4번의 SQL 쿼리                    1번의 SQL 쿼리: WHERE id IN (1, 2, 3)
                                  + id: 1에 대한 캐시 히트
```

### 3.1 설치

```bash
npm install dataloader
```

### 3.2 기본 사용법

```javascript
import DataLoader from 'dataloader';

// 1. 배치 함수 정의
//    입력: 키 배열 [1, 2, 3]
//    출력: 동일한 순서의 값 배열 Promise
const userLoader = new DataLoader(async (userIds) => {
  // 한 번에 요청된 모든 사용자를 가져오는 단일 쿼리
  const users = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [userIds]
  );

  // 중요: 입력 키와 동일한 순서로 결과 반환
  const userMap = new Map(users.map(u => [u.id, u]));
  return userIds.map(id => userMap.get(id) || null);
});

// 2. 리졸버에서 사용 (직접 DB 쿼리 대신)
const author = await userLoader.load('user_1');   // 큐에 추가
const author2 = await userLoader.load('user_2');  // 큐에 추가
const author3 = await userLoader.load('user_1');  // 캐시 히트!
// 다음 틱에: SELECT * FROM users WHERE id IN ('user_1', 'user_2')
```

## 4. 배칭: 작동 방식

DataLoader의 배칭은 Node.js의 이벤트 루프에 의존합니다. 단계별 메커니즘은 다음과 같습니다:

### 4.1 이벤트 루프 트릭

```
시간  │  리졸버 실행                            DataLoader 내부 상태
──────┼──────────────────────────────────     ─────────────────────────
 t0   │  Post.author(post1) → load("u1")     큐: ["u1"]
 t0   │  Post.author(post2) → load("u2")     큐: ["u1", "u2"]
 t0   │  Post.author(post3) → load("u1")     캐시 히트 → 캐시된 Promise 반환
 t0   │  Post.author(post4) → load("u3")     큐: ["u1", "u2", "u3"]
      │
 t1   │  ── process.nextTick / 마이크로태스크 ──  배치 함수 호출!
      │                                        batchFn(["u1", "u2", "u3"])
      │                                        → SELECT ... WHERE id IN ("u1","u2","u3")
      │
 t2   │  모든 Promise 해결                    호출자에게 결과 배포
```

핵심 통찰: GraphQL은 같은 틱에서 형제 필드를 해결합니다(`Promise.all` 사용). DataLoader는 배치 함수를 `process.nextTick`에 스케줄링하므로, 현재 틱의 모든 로드가 배치 실행 전에 수집됩니다.

### 4.2 배치 스케줄링 시각화

```
┌─────────────────────────────────────────────┐
│            JavaScript 이벤트 루프             │
│                                              │
│  ┌──────── 마이크로태스크 큐 ────────────┐   │
│  │ load("u1") → 배치에 추가             │   │
│  │ load("u2") → 배치에 추가             │   │
│  │ load("u1") → 캐시 히트               │   │
│  │ load("u3") → 배치에 추가             │   │
│  └────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌──────── 다음 틱 ───────────────────┐     │
│  │ batchFn(["u1", "u2", "u3"])       │     │
│  │ → 1번의 데이터베이스 쿼리           │     │
│  │ → 모든 대기 중인 Promise 해결       │     │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## 5. DataLoader 구현

### 5.1 배치 함수 계약

배치 함수는 엄격한 규칙을 따라야 합니다:

```javascript
// 배치 함수는 키 배열을 받고
// 키와 동일한 순서의 값 배열 Promise를 반환해야 합니다
// 여기서 values[i]는 keys[i]에 대응합니다
async function batchUsers(userIds) {
  // ✅ 규칙 1: 키와 동일한 수의 요소 반환
  // ✅ 규칙 2: 키와 동일한 순서로 요소 반환
  // ✅ 규칙 3: 실패한 조회에는 Error 인스턴스 반환 (null 아님, throw 아님)

  const users = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [userIds]
  );

  // O(1) 조회를 위한 Map 생성
  const userMap = new Map(users.map(u => [u.id, u]));

  // 순서를 유지하며 각 입력 키를 결과에 매핑
  return userIds.map(id =>
    userMap.get(id) || new Error(`User ${id} not found`)
  );
}

const userLoader = new DataLoader(batchUsers);
```

### 5.2 일대일 관계: Post → Author

```javascript
// 스키마: Post.author는 단일 User로 해결됩니다
function createUserByIdLoader(db) {
  return new DataLoader(async (userIds) => {
    console.log(`Batch loading users: [${userIds.join(', ')}]`);

    const users = await db.query(
      'SELECT * FROM users WHERE id = ANY($1)',
      [userIds]
    );

    const userMap = new Map(users.map(u => [u.id, u]));
    return userIds.map(id => userMap.get(id) || null);
  });
}

// 리졸버
const resolvers = {
  Post: {
    author: (parent, _, ctx) => ctx.loaders.userById.load(parent.authorId),
  },
};
```

### 5.3 완전한 DataLoader 팩토리

```javascript
// dataloaders.js
import DataLoader from 'dataloader';

export function createDataLoaders(db) {
  return {
    // 일대일: ID로 단일 엔티티 로드
    userById: new DataLoader(async (ids) => {
      const users = await db.user.findMany({ where: { id: { in: [...ids] } } });
      const map = new Map(users.map(u => [u.id, u]));
      return ids.map(id => map.get(id) || null);
    }),

    postById: new DataLoader(async (ids) => {
      const posts = await db.post.findMany({ where: { id: { in: [...ids] } } });
      const map = new Map(posts.map(p => [p.id, p]));
      return ids.map(id => map.get(id) || null);
    }),

    // 일대다: 관련 엔티티의 배열 로드
    postsByAuthorId: new DataLoader(async (authorIds) => {
      const posts = await db.post.findMany({
        where: { authorId: { in: [...authorIds] } },
        orderBy: { createdAt: 'desc' },
      });
      const grouped = new Map();
      for (const post of posts) {
        if (!grouped.has(post.authorId)) grouped.set(post.authorId, []);
        grouped.get(post.authorId).push(post);
      }
      return authorIds.map(id => grouped.get(id) || []);
    }),

    commentsByPostId: new DataLoader(async (postIds) => {
      const comments = await db.comment.findMany({
        where: { postId: { in: [...postIds] } },
        orderBy: { createdAt: 'asc' },
      });
      const grouped = new Map();
      for (const comment of comments) {
        if (!grouped.has(comment.postId)) grouped.set(comment.postId, []);
        grouped.get(comment.postId).push(comment);
      }
      return postIds.map(id => grouped.get(id) || []);
    }),

    // 집계: 카운트 로드
    commentCountByPostId: new DataLoader(async (postIds) => {
      const results = await db.$queryRaw`
        SELECT post_id, COUNT(*)::int as count
        FROM comments
        WHERE post_id = ANY(${postIds})
        GROUP BY post_id
      `;
      const map = new Map(results.map(r => [r.post_id, r.count]));
      return postIds.map(id => map.get(id) || 0);
    }),
  };
}
```

## 6. 일대다 관계를 위한 DataLoader

일대다 관계는 각 키가 결과 배열에 매핑되므로 특별한 처리가 필요합니다.

### 6.1 패턴

```javascript
// User.posts: 한 사용자 → 여러 게시글
const postsByAuthorIdLoader = new DataLoader(async (authorIds) => {
  // 1. 한 번의 쿼리로 모든 요청된 작성자의 게시글을 가져옴
  const posts = await db.query(
    'SELECT * FROM posts WHERE author_id = ANY($1) ORDER BY created_at DESC',
    [authorIds]
  );

  // 2. author_id로 게시글 그룹화
  const postsByAuthor = new Map();
  for (const post of posts) {
    if (!postsByAuthor.has(post.author_id)) {
      postsByAuthor.set(post.author_id, []);
    }
    postsByAuthor.get(post.author_id).push(post);
  }

  // 3. 입력 키와 동일한 순서로 배열 반환
  //    누락된 키는 빈 배열 (null이나 Error 아님)
  return authorIds.map(id => postsByAuthor.get(id) || []);
});
```

### 6.2 제한이 있는 일대다

스키마가 페이지네이션을 허용하면 어떻게 될까요?

```graphql
type User {
  posts(first: Int = 10): [Post!]!
}
```

DataLoader는 키로만 배칭하므로, `posts(first: 5)`와 `posts(first: 10)`을 쉽게 함께 배칭할 수 없습니다. 두 가지 접근 방법:

**접근 방식 A: 전체를 가져와 리졸버에서 슬라이스**

```javascript
// 작성자당 모든 게시글을 가져오고, 리졸버에서 제한 처리
const postsByAuthorIdLoader = new DataLoader(async (authorIds) => {
  const posts = await db.query(
    'SELECT * FROM posts WHERE author_id = ANY($1) ORDER BY created_at DESC',
    [authorIds]
  );
  const grouped = groupBy(posts, 'author_id');
  return authorIds.map(id => grouped.get(id) || []);
});

// 리졸버에서 결과 슬라이스
const resolvers = {
  User: {
    posts: (parent, { first = 10 }, ctx) => {
      const allPosts = ctx.loaders.postsByAuthorId.load(parent.id);
      return allPosts.then(posts => posts.slice(0, first));
    },
  },
};
```

**접근 방식 B: 커스텀 캐시 키로 복합 키 사용**

```javascript
const postsByAuthorIdLoader = new DataLoader(
  async (keys) => {
    // keys = [{ authorId: "1", first: 5 }, { authorId: "2", first: 10 }]
    // 단일 SQL 쿼리로 효율적으로 배칭하기 어려움
    // 종종 접근 방식 A가 더 간단하고 충분합니다
    return Promise.all(keys.map(({ authorId, first }) =>
      db.query(
        'SELECT * FROM posts WHERE author_id = $1 ORDER BY created_at DESC LIMIT $2',
        [authorId, first]
      )
    ));
  },
  {
    cacheKeyFn: (key) => `${key.authorId}:${key.first}`,
  }
);
```

단일 쿼리로 배칭이 이루어지므로 접근 방식 A가 주로 선호됩니다.

## 7. 캐싱 동작

DataLoader는 두 수준의 캐싱을 제공하며, 차이를 이해하는 것이 중요합니다.

### 7.1 요청 수준 캐시 (DataLoader 내장)

DataLoader는 DataLoader 인스턴스의 수명 동안 키로 결과를 캐시합니다. 인스턴스를 요청마다 생성하므로, 이것은 요청별 캐시입니다.

```javascript
// 동일한 요청 내에서:
const user1 = await userLoader.load('user_1');  // DB 쿼리
const user2 = await userLoader.load('user_2');  // DB 쿼리 (위와 배칭)
const user3 = await userLoader.load('user_1');  // 캐시 히트! DB 쿼리 없음.
```

이 캐시는 중복 제거 문제(여러 게시글이 동일한 작성자를 참조)를 해결합니다.

### 7.2 캐싱 비활성화

때로는 캐싱 없이 배칭만 원할 수 있습니다:

```javascript
const freshUserLoader = new DataLoader(batchUsers, {
  cache: false,  // 메모이제이션 캐시 비활성화
});
```

### 7.3 수동 캐시 작업

```javascript
// 캐시 준비 (이미 데이터를 가지고 있을 때 유용)
userLoader.prime('user_1', { id: 'user_1', name: 'Alice' });

// 특정 키 지우기 (예: 뮤테이션 후)
userLoader.clear('user_1');

// 전체 캐시 지우기
userLoader.clearAll();
```

### 7.4 애플리케이션 수준 캐시 (외부)

DataLoader의 캐시는 요청별만 해당됩니다. 요청 간 캐싱에는 외부 캐시를 사용하세요:

```javascript
const userByIdLoader = new DataLoader(async (ids) => {
  // 먼저 외부 캐시 확인 (Redis, Memcached)
  const cached = await redis.mget(ids.map(id => `user:${id}`));
  const missingIds = ids.filter((_, i) => !cached[i]);

  // DB에서 누락된 항목만 가져오기
  let freshUsers = [];
  if (missingIds.length > 0) {
    freshUsers = await db.query(
      'SELECT * FROM users WHERE id = ANY($1)',
      [missingIds]
    );

    // 외부 캐시 채우기
    const pipeline = redis.pipeline();
    for (const user of freshUsers) {
      pipeline.set(`user:${user.id}`, JSON.stringify(user), 'EX', 300);
    }
    await pipeline.exec();
  }

  // 캐시된 결과와 새 결과 병합
  const userMap = new Map(freshUsers.map(u => [u.id, u]));
  return ids.map((id, i) => {
    if (cached[i]) return JSON.parse(cached[i]);
    return userMap.get(id) || null;
  });
});
```

## 8. 요청별 DataLoader 인스턴스

이것이 가장 중요한 DataLoader 규칙입니다: **각 요청마다 새로운 DataLoader 인스턴스를 생성하세요**.

### 8.1 왜 요청별인가?

```javascript
// ❌ 잘못됨: 요청 간에 공유됨
const globalLoader = new DataLoader(batchUsers);

// 요청 1: user_1 로드 → 캐시됨
// 요청 2: user_1 로드 → 요청 1의 캐시에서 오래된 데이터 획득!
// 요청 3: user_1이 요청 사이에 업데이트됨 → 여전히 오래된 데이터!
```

공유 DataLoader 인스턴스는 다음을 야기합니다:
- **오래된 데이터**: 한 요청의 뮤테이션이 다른 요청에 반영되지 않음
- **메모리 누수**: 캐시가 무제한으로 증가
- **인가 누수**: 사용자 A가 사용자 B의 요청에서 캐시된 데이터를 볼 수 있음

### 8.2 올바른 패턴

```javascript
// dataloaders.js
import DataLoader from 'dataloader';

export function createDataLoaders(db) {
  // 요청마다 한 번 호출 — 새로운 인스턴스, 빈 캐시
  return {
    userById: new DataLoader(async (ids) => {
      const users = await db.user.findMany({ where: { id: { in: [...ids] } } });
      const map = new Map(users.map(u => [u.id, u]));
      return ids.map(id => map.get(id) || null);
    }),
    postsByAuthorId: new DataLoader(async (authorIds) => {
      const posts = await db.post.findMany({
        where: { authorId: { in: [...authorIds] } },
      });
      const grouped = new Map();
      for (const p of posts) {
        if (!grouped.has(p.authorId)) grouped.set(p.authorId, []);
        grouped.get(p.authorId).push(p);
      }
      return authorIds.map(id => grouped.get(id) || []);
    }),
  };
}

// server.js
const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    db: prisma,
    user: await authenticate(req),
    loaders: createDataLoaders(prisma),  // 요청마다 새로 생성
  }),
});
```

## 9. 다양한 데이터 소스와 DataLoader

DataLoader는 SQL 데이터베이스에만 국한되지 않습니다. 배치 조회를 지원하는 모든 데이터 소스와 함께 작동합니다.

### 9.1 REST API

```javascript
const userByIdFromAPI = new DataLoader(async (ids) => {
  // 배치: 여러 ID를 포함한 단일 요청
  const response = await fetch(
    `https://api.example.com/users?ids=${ids.join(',')}`
  );
  const users = await response.json();
  const map = new Map(users.map(u => [u.id, u]));
  return ids.map(id => map.get(id) || null);
});
```

### 9.2 Redis

```javascript
const cachedDataLoader = new DataLoader(async (keys) => {
  // Redis MGET: 한 번의 호출로 여러 키를 배치 조회
  const values = await redis.mget(keys.map(k => `cache:${k}`));
  return values.map(v => v ? JSON.parse(v) : null);
});
```

### 9.3 MongoDB

```javascript
const userByIdFromMongo = new DataLoader(async (ids) => {
  const users = await db.collection('users').find({
    _id: { $in: ids.map(id => new ObjectId(id)) },
  }).toArray();
  const map = new Map(users.map(u => [u._id.toString(), u]));
  return ids.map(id => map.get(id) || null);
});
```

### 9.4 Elasticsearch

```javascript
const searchResultsLoader = new DataLoader(async (queries) => {
  // 멀티 검색: 여러 쿼리를 하나의 요청으로 배치
  const body = queries.flatMap(q => [
    { index: 'posts' },
    { query: { match: { title: q } } },
  ]);
  const { responses } = await client.msearch({ body });
  return responses.map(r => r.hits.hits.map(h => h._source));
});
```

## 10. N+1 모니터링 및 감지

### 10.1 쿼리 로깅

가장 간단한 감지 방법은 GraphQL 요청당 쿼리를 세는 것입니다:

```javascript
// 요청당 데이터베이스 쿼리를 세는 미들웨어
function createQueryCounter() {
  let count = 0;
  return {
    increment() { count++; },
    getCount() { return count; },
    reset() { count = 0; },
  };
}

// 컨텍스트 생성에서
context: async ({ req }) => {
  const queryCounter = createQueryCounter();

  // 쿼리를 세도록 DB 클라이언트 래핑
  const countedDb = new Proxy(prisma, {
    get(target, prop) {
      return new Proxy(target[prop], {
        get(model, method) {
          if (typeof model[method] === 'function') {
            return (...args) => {
              queryCounter.increment();
              return model[method](...args);
            };
          }
          return model[method];
        },
      });
    },
  });

  return {
    db: countedDb,
    queryCounter,
    loaders: createDataLoaders(countedDb),
  };
},

// 각 요청 후 쿼리 수를 로깅하는 플러그인
const queryCountPlugin = {
  async requestDidStart() {
    return {
      async willSendResponse({ contextValue }) {
        const count = contextValue.queryCounter.getCount();
        if (count > 20) {
          console.warn(`[N+1 경고] 단일 요청에서 ${count}번의 DB 쿼리`);
        }
      },
    };
  },
};
```

### 10.2 Apollo Studio 트레이싱

Apollo Server는 필드별 타이밍 데이터를 보고할 수 있습니다:

```javascript
const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    ApolloServerPluginUsageReporting({
      sendErrors: { unmodified: true },
    }),
  ],
});
```

Apollo Studio에서 주목할 점:
- 높은 실행 횟수를 가진 필드 (예: `Post.author`가 100번 호출)
- 개별 지연 시간은 낮지만 총 시간이 높은 필드 (많은 순차 쿼리)
- 부모 목록의 항목 수를 초과하는 리졸버 실행 횟수

### 10.3 전후 비교

```
                          DataLoader 없음        DataLoader 있음
────────────────────────  ────────────────────   ──────────────────
posts(first: 50)          1번 쿼리               1번 쿼리
  Post.author (×50)       50번 쿼리              1번 쿼리 (배치)
  Post.comments (×50)     50번 쿼리              1번 쿼리 (배치)
    Comment.author (×200) 200번 쿼리             1번 쿼리 (배치)*
────────────────────────  ────────────────────   ──────────────────
총계                       301번 쿼리             4번 쿼리

* DataLoader 캐시로 중복 제거 (200번 로드 → 약 40명의 고유 작성자)
```

## 11. 흔한 함정

### 11.1 요청별 인스턴스 생성을 잊어버리는 경우

```javascript
// ❌ 시작 시 한 번만 생성된 DataLoader
const userLoader = new DataLoader(batchUsers);
// 오래된 캐시, 메모리 누수, 인가 문제

// ✅ 컨텍스트에서 DataLoader 생성 (요청별)
context: async () => ({
  loaders: { userById: new DataLoader(batchUsers) },
}),
```

### 11.2 반환 순서가 잘못된 경우

```javascript
// ❌ 데이터베이스 순서로 반환 (입력 순서 아님)
const batchUsers = async (ids) => {
  return db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  // DB가 [user_3, user_1, user_2]를 반환하지만 키 순서는 [1, 2, 3]
};

// ✅ 입력 키 순서에 맞게 결과 매핑
const batchUsers = async (ids) => {
  const users = await db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  const map = new Map(users.map(u => [u.id, u]));
  return ids.map(id => map.get(id) || null);  // 입력 순서 유지
};
```

### 11.3 반환 수가 잘못된 경우

```javascript
// ❌ null을 필터링함 (배열 길이 !== 키 길이)
const batchUsers = async (ids) => {
  const users = await db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  return users;  // 5개의 ID 중 2개가 없으면 5 대신 3을 반환
};

// ✅ 키당 하나의 결과 반환 (누락 시 null)
const batchUsers = async (ids) => {
  const users = await db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  const map = new Map(users.map(u => [u.id, u]));
  return ids.map(id => map.get(id) || null);  // 5개의 키에 항상 5개 결과
};
```

### 11.4 로드 후 뮤테이션 처리

뮤테이션 후에 DataLoader 캐시가 오래될 수 있습니다:

```javascript
const resolvers = {
  Mutation: {
    updateUser: async (_, { id, input }, ctx) => {
      const user = await ctx.db.user.update({ where: { id }, data: input });

      // 이후 로드가 새 데이터를 가져오도록 캐시된 값 지우기
      ctx.loaders.userById.clear(id);
      // 선택적으로 업데이트된 값으로 준비
      ctx.loaders.userById.prime(id, user);

      return { user, errors: [] };
    },
  },
};
```

### 11.5 배치를 지원하지 않는 소스에 DataLoader 사용

데이터 소스가 배치 조회를 지원하지 않는 경우(예: ID 하나씩만 받는 REST API), DataLoader는 여전히 중복 제거에 도움이 되지만 배칭의 성능 이점은 없습니다:

```javascript
// 이 "배치" 함수는 여전히 N번의 요청을 만듭니다
const userLoader = new DataLoader(async (ids) => {
  return Promise.all(ids.map(id =>
    fetch(`https://api.example.com/users/${id}`).then(r => r.json())
  ));
});
// 이점: 중복 제거 (같은 ID가 두 번 요청되면 한 번만 fetch)
// 배칭 이점 없음: 여전히 N번의 fetch (다만 중복 제거됨)
```

---

## 12. 연습 문제

### 연습 1: 쿼리 수 계산 (초급)

이 리졸버들이 주어졌을 때 (DataLoader 없음):

```javascript
const resolvers = {
  Query: {
    users: (_, { first }) => db.query('SELECT * FROM users LIMIT $1', [first]),
  },
  User: {
    posts: (parent) => db.query('SELECT * FROM posts WHERE author_id = $1', [parent.id]),
    followerCount: (parent) => db.query('SELECT COUNT(*) FROM follows WHERE followed_id = $1', [parent.id]),
  },
  Post: {
    commentCount: (parent) => db.query('SELECT COUNT(*) FROM comments WHERE post_id = $1', [parent.id]),
  },
};
```

이 요청에서 데이터베이스 쿼리가 몇 번 실행됩니까?

```graphql
query {
  users(first: 20) {
    posts {
      commentCount
    }
    followerCount
  }
}
```

각 사용자가 정확히 3개의 게시글을 가지고 있다고 가정하세요.

### 연습 2: 배치 함수 구현 (중급)

게시글 ID로 댓글을 로드하는 배치 함수를 작성하세요 (일대다). 다음이 주어집니다:

- 데이터베이스 테이블: `comments (id, post_id, body, author_id, created_at)`
- 행을 반환하는 함수 `db.query(sql, params)`
- 입력: 게시글 ID 배열 `["post_1", "post_2", "post_3"]`
- 출력: 게시글 ID당 하나씩 순서에 맞는 댓글 배열들

댓글이 없는 게시글의 경우를 처리하세요 (null이 아닌 빈 배열 반환).

### 연습 3: DataLoader 버그 수정 (중급)

이 코드에 버그가 있습니다. 찾아서 수정하세요.

```javascript
const tagLoader = new DataLoader(async (tagNames) => {
  const tags = await db.query(
    'SELECT * FROM tags WHERE name = ANY($1)',
    [tagNames]
  );
  return tags;
});
```

힌트: `tagNames = ['graphql', 'rest', 'api']`이고 데이터베이스에 `graphql`과 `api` 태그만 있을 때 어떻게 됩니까?

### 연습 4: 완전한 DataLoader 통합 (고급)

이 스키마에 대해 완전한 DataLoader 팩토리와 리졸버 맵을 구현하세요:

```graphql
type Query {
  courses(first: Int = 10): [Course!]!
}

type Course {
  id: ID!
  title: String!
  instructor: User!
  students: [User!]!
  lessonCount: Int!
}

type User {
  id: ID!
  name: String!
  enrolledCourses: [Course!]!
}
```

데이터베이스 테이블:
- `courses (id, title, instructor_id)`
- `users (id, name)`
- `enrollments (user_id, course_id)`
- `lessons (id, course_id, title)`

요구 사항:
1. 모든 관계에 대한 DataLoader 인스턴스 생성
2. `lessonCount`에는 집계 쿼리 사용 (모든 레슨을 로드하지 않음)
3. 다대다 관계(수강 신청)를 올바르게 처리
4. 요청별 DataLoader 인스턴스 생성

### 연습 5: 모니터링 구현 (고급)

다음을 수행하는 간단한 N+1 감지 시스템을 구축하세요:

1. 요청당 쿼리를 세도록 데이터베이스 클라이언트 래핑
2. 단일 GraphQL 요청에서 데이터베이스 쿼리가 10번을 초과하면 경고 로깅
3. 필드 이름별로 가장 많이 호출된 상위 5개 리졸버 보고
4. Apollo Server 플러그인으로 통합

플러그인 코드와 모니터링 미들웨어를 작성하세요.

---

## 13. 참고 자료

- DataLoader GitHub 저장소 - https://github.com/graphql/dataloader
- DataLoader 소스 코드 (레퍼런스 구현) - https://github.com/graphql/dataloader/blob/main/src/index.js
- Lee Byron, "DataLoader — Source Code Walkthrough" (YouTube, 2016)
- Apollo Server DataLoader 가이드 - https://www.apollographql.com/docs/apollo-server/data/fetching-data/#batching-and-caching
- Marc-Andre Giroux, "Production Ready GraphQL" (2020) - Chapter 5: Performance
- Node.js 이벤트 루프 - https://nodejs.org/en/docs/guides/event-loop-timers-and-nexttick

---

**이전**: [리졸버](./04_Resolvers.md) | **다음**: [구독](./06_Subscriptions.md)
