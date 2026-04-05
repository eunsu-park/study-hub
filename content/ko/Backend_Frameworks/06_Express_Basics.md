# 06. Express 기초

**이전**: [FastAPI 테스트](./05_FastAPI_Testing.md) | **다음**: [Express 심화](./07_Express_Advanced.md)

## 학습 목표

이 레슨을 마치면 다음을 수행할 수 있습니다:

1. Express가 무엇인지, Node.js 생태계에서 어떤 역할을 하는지 설명하기
2. 설정 가능한 포트에서 리스닝하는 기본 Express 애플리케이션 생성하기
3. 경로(path)와 쿼리 파라미터(query parameter)를 포함한 주요 HTTP 메서드용 라우트 정의하기
4. 미들웨어(middleware) 실행 체인을 설명하고 커스텀 미들웨어 작성하기
5. 유지보수 가능한 코드베이스를 위해 라우트를 모듈식 Router 인스턴스로 구성하기

---

Express는 Node.js에서 가장 널리 사용되는 웹 프레임워크입니다. Node.js가 저수준 HTTP 기능을 제공하는 반면, Express는 라우팅(routing), 미들웨어(middleware), 편의 메서드의 얇은 레이어를 추가하여 웹 서버 구축을 실용적으로 만듭니다. 미니멀리스트 철학을 따르며 코어는 작고, 기능은 미들웨어를 통해 추가됩니다. 이 레슨은 모든 Express 애플리케이션을 구축하는 데 필요한 핵심 내용을 다룹니다.

## 목차

1. [Node.js와 Express 개요](#1-nodejs와-express-개요)
2. [Express 애플리케이션 생성](#2-express-애플리케이션-생성)
3. [라우팅](#3-라우팅)
4. [미들웨어 개념과 체인](#4-미들웨어-개념과-체인)
5. [내장 미들웨어](#5-내장-미들웨어)
6. [Request와 Response 객체](#6-request와-response-객체)
7. [모듈식 라우트를 위한 Router](#7-모듈식-라우트를-위한-router)
8. [연습 문제](#8-연습-문제)

---

## 1. Node.js와 Express 개요

### Node.js란?

Node.js는 Chrome의 V8 엔진 위에 구축된 JavaScript 런타임입니다. **이벤트 기반(event-driven), 논블로킹(non-blocking) I/O 모델**을 사용하여 네트워크 애플리케이션에 효율적입니다.

```
┌─────────────────────────────────────────────────┐
│                  Node.js Runtime                │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ V8 Engine│  │ libuv    │  │ Core Modules │  │
│  │ (JS exec)│  │ (async   │  │ (http, fs,   │  │
│  │          │  │  I/O)    │  │  path, etc.) │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### Express란?

Express는 다음을 제공하는 **미니멀하고 유연한** 웹 프레임워크입니다:

| 기능 | 설명 |
|---------|-------------|
| **라우팅(Routing)** | URL 패턴을 핸들러 함수에 매핑 |
| **미들웨어(Middleware)** | 요청 처리를 위한 조합 가능한 파이프라인 |
| **편의성(Convenience)** | Node의 raw `http` 모듈 위에 단순화된 API |
| **생태계(Ecosystem)** | 수천 개의 미들웨어 패키지 활용 가능 |

> **비유 -- 조립 라인:** Express를 공장 조립 라인으로 생각하세요. 각 요청이 라인에 들어와 일련의 스테이션(미들웨어)을 통과합니다. 각 스테이션은 항목을 검사하거나 수정하거나 거부할 수 있습니다. 마지막 스테이션에서 응답을 생성합니다.

### Express vs Raw Node.js

```javascript
// Raw Node.js — 모든 것을 직접 파싱해야 합니다
import { createServer } from 'node:http';

const server = createServer((req, res) => {
  // 수동 URL 파싱, 바디 파싱, 헤더 설정...
  if (req.method === 'GET' && req.url === '/api/users') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ users: [] }));
  }
});

server.listen(3000);
```

```javascript
// Express — 선언적 라우팅, 내장 편의 기능
import express from 'express';
const app = express();

// Express가 content-type, 직렬화, 상태 코드를 처리합니다
app.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

app.listen(3000);
```

---

## 2. Express 애플리케이션 생성

### 프로젝트 설정

```bash
# 프로젝트 디렉토리 생성 및 초기화
mkdir express-demo && cd express-demo
npm init -y

# Express 설치
npm install express

# ES 모듈 활성화 — require() 대신 import/export 문법 허용
# package.json에 "type": "module" 추가
```

### 최소한의 서버

```javascript
// app.js
import express from 'express';

const app = express();

// 환경 변수에서 PORT를 읽고 폴백(fallback) 설정 — 코드 변경 없이
// 다른 환경(개발, 스테이징, 프로덕션)에서 앱을 설정 가능하게 합니다
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.send('Hello, Express!');
});

// 콜백이 서버 시작을 확인합니다 — 디버깅과 로깅에 유용합니다
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

```bash
# 서버 실행
node app.js

# 개발 중 자동 재시작 (Node.js 18.11+)
# --watch는 파일 변경 시 재시작합니다 — 대부분의 경우 nodemon이 필요 없습니다
node --watch app.js
```

### 애플리케이션 구조

사소한 스크립트 이상의 경우 책임별로 파일을 구성합니다:

```
express-demo/
├── src/
│   ├── app.js           # Express 앱 설정
│   ├── server.js         # 서버 시작 (listen)
│   ├── routes/           # 라우트 정의
│   │   └── users.js
│   └── middleware/        # 커스텀 미들웨어
│       └── logger.js
├── package.json
└── .env
```

---

## 3. 라우팅

라우트(route)는 특정 엔드포인트에서 클라이언트 요청에 대해 애플리케이션이 어떻게 응답하는지 정의합니다.

### 기본 라우트

```javascript
// 각 라우트는 HTTP 메서드 + 경로를 핸들러 함수에 바인딩합니다
app.get('/api/items', (req, res) => {
  res.json({ items: ['apple', 'banana'] });
});

app.post('/api/items', (req, res) => {
  // req.body에는 파싱된 요청 바디가 포함됩니다 (express.json() 미들웨어 필요)
  const newItem = req.body;
  res.status(201).json({ created: newItem });
});

app.put('/api/items/:id', (req, res) => {
  res.json({ updated: req.params.id });
});

app.delete('/api/items/:id', (req, res) => {
  // 204 No Content — 바디 없는 성공적인 삭제에 대한 표준 응답
  res.status(204).send();
});

// PATCH는 부분 업데이트 용도 — 전체 리소스를 교체하는 PUT과 달리 일부만 수정합니다
app.patch('/api/items/:id', (req, res) => {
  res.json({ patched: req.params.id, fields: req.body });
});
```

### 라우트 파라미터

```javascript
// :id는 이름 있는 파라미터 — req.params에 캡처됩니다
app.get('/api/users/:id', (req, res) => {
  const { id } = req.params;
  res.json({ userId: id });
});

// 여러 파라미터 — 중첩 리소스에 유용합니다
app.get('/api/users/:userId/posts/:postId', (req, res) => {
  const { userId, postId } = req.params;
  res.json({ userId, postId });
});
```

### 쿼리 파라미터

```javascript
// 쿼리 문자열은 req.query로 자동 파싱됩니다
// GET /api/search?q=express&page=2&limit=10
app.get('/api/search', (req, res) => {
  const { q, page = '1', limit = '10' } = req.query;

  // 모든 쿼리 값은 문자열 — 필요 시 숫자로 변환합니다
  res.json({
    query: q,
    page: parseInt(page, 10),
    limit: parseInt(limit, 10),
  });
});
```

### 라우트 체이닝

```javascript
// app.route()는 동일한 경로에 대한 핸들러를 그룹화합니다 — 반복을 줄입니다
app.route('/api/books')
  .get((req, res) => {
    res.json({ books: [] });
  })
  .post((req, res) => {
    res.status(201).json({ created: req.body });
  });

app.route('/api/books/:id')
  .get((req, res) => {
    res.json({ bookId: req.params.id });
  })
  .put((req, res) => {
    res.json({ updated: req.params.id });
  })
  .delete((req, res) => {
    res.status(204).send();
  });
```

---

## 4. 미들웨어 개념과 체인

미들웨어 함수는 요청(request), 응답(response), **next** 함수에 접근할 수 있습니다. 각 요청이 흐르는 파이프라인을 형성합니다.

```
Request → [Middleware 1] → [Middleware 2] → [Route Handler] → Response
              │                  │                │
              │ next()           │ next()          │ res.send()
              └──────────────────┘────────────────┘
```

### 커스텀 미들웨어 작성하기

```javascript
// 미들웨어 시그니처: (req, res, next) => { ... }
// next()를 호출하면 체인의 다음 미들웨어로 제어가 전달됩니다.
// next()를 잊으면 요청이 멈춥니다 — 초보자들이 자주 하는 실수입니다.

const requestLogger = (req, res, next) => {
  const start = Date.now();

  // res.on('finish')는 응답이 전송된 후 실행됩니다 — 응답을 차단하지 않고
  // 전체 요청 시간을 측정할 수 있습니다
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.originalUrl} ${res.statusCode} ${duration}ms`);
  });

  next();
};

// app.use()는 모든 라우트와 메서드에 미들웨어를 등록합니다
app.use(requestLogger);
```

### 미들웨어 실행 순서

```javascript
// 미들웨어는 등록된 순서대로 실행됩니다 — 순서가 중요합니다!
app.use((req, res, next) => {
  console.log('1st middleware');
  next();
});

app.use((req, res, next) => {
  console.log('2nd middleware');
  next();
});

app.get('/test', (req, res) => {
  console.log('Route handler');
  res.send('Done');
});

// GET /test에 대한 출력:
// 1st middleware
// 2nd middleware
// Route handler
```

### 경로별 미들웨어

```javascript
// 미들웨어를 특정 경로에 스코프(scope)할 수 있습니다 — 매칭되는 라우트에서만 실행됩니다
const apiKeyCheck = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  if (!apiKey || apiKey !== process.env.API_KEY) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  next();
};

// /api/admin으로 시작하는 라우트에만 적용됩니다
app.use('/api/admin', apiKeyCheck);
```

### 라우트당 여러 미들웨어

```javascript
const authenticate = (req, res, next) => {
  // 다운스트림 핸들러를 위해 요청에 사용자 정보를 첨부합니다
  req.user = { id: 1, role: 'admin' };
  next();
};

const authorize = (role) => (req, res, next) => {
  if (req.user.role !== role) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  next();
};

// 여러 미들웨어 체인 — 각각이 next()를 호출해야 요청이 진행됩니다
app.delete('/api/users/:id', authenticate, authorize('admin'), (req, res) => {
  res.json({ deleted: req.params.id });
});
```

---

## 5. 내장 미들웨어

Express 4.x에는 여러 내장 미들웨어 함수가 포함되어 있습니다:

```javascript
import express from 'express';

// JSON 요청 바디 파싱 — JSON 페이로드가 있는 POST/PUT/PATCH에 필요합니다
// 없으면 JSON 요청에서 req.body가 undefined입니다
app.use(express.json());

// URL 인코딩된 폼 데이터 파싱 — 전통적인 HTML 폼 제출용
// extended: true는 중첩 객체를 지원하는 qs 라이브러리를 사용합니다
app.use(express.urlencoded({ extended: true }));

// 디렉토리에서 정적 파일 제공 — CSS, 이미지, 클라이언트 사이드 JS
// 지정된 디렉토리를 기준으로 파일이 제공됩니다
app.use(express.static('public'));

// 특정 URL 접두사에 정적 파일 마운트
app.use('/assets', express.static('public'));
```

### JSON 바디 크기 제한

```javascript
// 요청 바디 크기 제한 — 크기가 큰 페이로드로 인한 서비스 거부 공격 방지
// 기본값은 100kb; API 요구사항에 맞게 조정합니다
app.use(express.json({ limit: '10mb' }));
```

---

## 6. Request와 Response 객체

### Request 객체 (req)

```javascript
app.post('/api/users', (req, res) => {
  // 요청 객체의 주요 속성들:
  console.log(req.method);       // 'POST'
  console.log(req.path);         // '/api/users'
  console.log(req.originalUrl);  // '/api/users?sort=name' (쿼리 문자열 포함)
  console.log(req.params);       // 라우트 파라미터: { id: '42' }
  console.log(req.query);        // 쿼리 문자열: { sort: 'name' }
  console.log(req.body);         // 파싱된 바디 (express.json() 필요)
  console.log(req.headers);      // 모든 헤더 (소문자 키)
  console.log(req.ip);           // 클라이언트 IP 주소
  console.log(req.hostname);     // 'localhost'
  console.log(req.get('Content-Type')); // 특정 헤더 가져오기
});
```

### Response 객체 (res)

```javascript
app.get('/api/demo', (req, res) => {
  // res.json() — Content-Type을 application/json으로 설정하고 직렬화합니다
  res.json({ message: 'hello' });

  // res.status() — HTTP 상태 코드 설정; 체이닝 가능합니다
  res.status(201).json({ created: true });

  // res.send() — 문자열, Buffer, 객체를 전송합니다
  res.send('Plain text response');

  // res.redirect() — 기본적으로 302 리다이렉트를 전송합니다
  res.redirect('/new-location');
  res.redirect(301, '/permanent-new-location');

  // res.set() — 응답 헤더를 설정합니다
  res.set('X-Request-Id', 'abc-123');

  // res.cookie() — 쿠키 설정 (읽기에는 cookie-parser 필요)
  res.cookie('session', 'token-value', {
    httpOnly: true,   // JavaScript로 접근 불가 — XSS 도용 방지
    secure: true,     // HTTPS로만 전송
    maxAge: 3600000,  // 밀리초 단위 1시간
  });

  // res.download() — 파일 다운로드 유도
  res.download('/path/to/file.pdf');
});
```

---

## 7. 모듈식 라우트를 위한 Router

애플리케이션이 커지면 단일 파일에 모든 라우트를 넣는 것은 관리가 어려워집니다. Express `Router`는 모듈식으로 마운트 가능한 라우트 핸들러를 생성합니다.

### Router 모듈 생성하기

```javascript
// src/routes/users.js
import { Router } from 'express';

const router = Router();

// 여기에 정의된 라우트는 라우터가 마운트된 위치를 기준으로 합니다
router.get('/', (req, res) => {
  res.json({ users: [{ id: 1, name: 'Alice' }] });
});

router.get('/:id', (req, res) => {
  res.json({ userId: req.params.id });
});

router.post('/', (req, res) => {
  const { name, email } = req.body;
  res.status(201).json({ id: 2, name, email });
});

router.put('/:id', (req, res) => {
  res.json({ updated: req.params.id, ...req.body });
});

router.delete('/:id', (req, res) => {
  res.status(204).send();
});

export default router;
```

### Router 마운트하기

```javascript
// src/app.js
import express from 'express';
import usersRouter from './routes/users.js';
import postsRouter from './routes/posts.js';

const app = express();
app.use(express.json());

// 경로 접두사에 라우터를 마운트합니다 — 내부 라우트는 모두 상대적이 됩니다
// 예: router.get('/')은 GET /api/users가 됩니다
app.use('/api/users', usersRouter);
app.use('/api/posts', postsRouter);

// 404 핸들러 — 모든 라우트 이후에 등록해야 합니다
// 어떤 라우트도 매칭되지 않으면 이 미들웨어가 요청을 처리합니다
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

export default app;
```

### 앱과 서버 분리하기

```javascript
// src/server.js — 분리하면 서버를 시작하지 않고도 앱을 임포트할 수 있습니다.
// 테스트에서 필수적입니다 (Supertest는 실행 중인 서버가 아닌 앱이 필요합니다)
import app from './app.js';

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

### 중첩 Router

```javascript
// src/routes/posts.js
import { Router } from 'express';

const router = Router();

// 중첩 리소스: /api/posts/:postId/comments
const commentsRouter = Router({ mergeParams: true });
// mergeParams: true — 자식 라우터가 부모의 :postId 파라미터에 접근할 수 있습니다

commentsRouter.get('/', (req, res) => {
  res.json({ postId: req.params.postId, comments: [] });
});

commentsRouter.post('/', (req, res) => {
  res.status(201).json({
    postId: req.params.postId,
    comment: req.body,
  });
});

router.use('/:postId/comments', commentsRouter);

router.get('/', (req, res) => {
  res.json({ posts: [] });
});

export default router;
```

---

## 8. 연습 문제

### 문제 1: 헬스 체크 엔드포인트

다음을 반환하는 `GET /health` 엔드포인트가 있는 Express 서버를 만드세요:
```json
{ "status": "ok", "uptime": 12345, "timestamp": "2025-01-01T00:00:00.000Z" }
```

`uptime`은 `process.uptime()`을 반올림한 값입니다.

### 문제 2: 요청 타이밍 미들웨어

다음을 수행하는 미들웨어 함수를 작성하세요:
- 모든 응답에 처리 시간(밀리초)이 담긴 `X-Response-Time` 헤더를 추가합니다
- 메서드, URL, 상태 코드, 응답 시간을 콘솔에 로그합니다

힌트: 시작 시 `Date.now()`를 기록하고 `res.on('finish', ...)`를 사용하여 소요 시간을 계산합니다.

### 문제 3: Router를 사용한 CRUD API

`express.Router()`를 사용하여 "tasks" 리소스에 대한 완전한 CRUD API를 만드세요:
- `GET /api/tasks` -- 모든 태스크 목록 조회 (`?status=done` 필터 지원)
- `GET /api/tasks/:id` -- 단일 태스크 조회 (없으면 404 반환)
- `POST /api/tasks` -- 태스크 생성 (`title`이 있는지 검증)
- `PUT /api/tasks/:id` -- 태스크 업데이트
- `DELETE /api/tasks/:id` -- 태스크 삭제

인메모리 배열을 데이터 저장소로 사용하세요.

### 문제 4: 미들웨어 체인

세 가지 미들웨어 레이어로 구성된 Express 앱을 만드세요:
1. 요청 메서드와 URL을 출력하는 **로거(logger)**
2. `Authorization` 헤더가 없으면 401을 반환하는 **인증 확인기**
3. 사용자의 역할이 일치하지 않으면 403을 반환하는 팩토리 함수 **`requireRole(role)`**

이것들을 `DELETE /api/users/:id` 라우트에 체인으로 연결하세요.

### 문제 5: 중첩 리소스

중첩 리소스를 가진 블로그 애플리케이션을 위한 라우터 구조를 설계하세요:
- `GET /api/authors/:authorId/articles` -- 작성자별 기사 목록 조회
- `POST /api/authors/:authorId/articles` -- 작성자의 기사 생성
- `GET /api/authors/:authorId/articles/:articleId` -- 특정 기사 조회

articles 라우터가 `:authorId`에 접근할 수 있도록 `mergeParams: true`를 사용하세요.

---

## 참고 자료

- [Express.js 공식 문서](https://expressjs.com/)
- [Express 4.x API 레퍼런스](https://expressjs.com/en/4x/api.html)
- [Node.js 공식 문서](https://nodejs.org/docs/latest/api/)
- [MDN: HTTP 메서드](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)

---

**이전**: [FastAPI 테스트](./05_FastAPI_Testing.md) | **다음**: [Express 심화](./07_Express_Advanced.md)
