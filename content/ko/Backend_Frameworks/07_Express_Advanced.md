# 07. Express 심화

**이전**: [Express 기초](./06_Express_Basics.md) | **다음**: [Express 데이터베이스](./08_Express_Database.md)

## 학습 목표

이 레슨을 마치면 다음을 수행할 수 있습니다:

1. Express의 4개 인수 미들웨어 패턴을 사용한 중앙집중식 오류 처리 구현하기
2. 로컬(local)과 JWT 전략을 모두 사용하여 Passport.js로 인증 설정하기
3. 속도 제한(rate limiting), CORS 정책, 보안 헤더로 API 보호하기
4. multer로 파일 업로드를 처리하고 Zod로 요청 데이터를 검증하기
5. 여러 보안 미들웨어를 프로덕션 준비 설정으로 결합하기

---

기본 Express 서버를 구축하는 것은 간단하지만, 프로덕션 애플리케이션에는 더 많은 것이 필요합니다: 인증, 입력 검증, 파일 처리, 속도 제한, 적절한 오류 관리. 이 레슨은 프로토타입을 강력하고 안전한 API 서버로 만드는 미들웨어와 라이브러리를 다룹니다. 각 섹션은 이전 레슨의 미들웨어 체인 개념을 기반으로 합니다.

## 목차

1. [오류 처리 미들웨어](#1-오류-처리-미들웨어)
2. [Passport.js를 사용한 인증](#2-passportjs를-사용한-인증)
3. [속도 제한](#3-속도-제한)
4. [CORS 설정](#4-cors-설정)
5. [Multer를 사용한 파일 업로드](#5-multer를-사용한-파일-업로드)
6. [Zod를 사용한 입력 검증](#6-zod를-사용한-입력-검증)
7. [Helmet을 사용한 보안 헤더](#7-helmet을-사용한-보안-헤더)
8. [모두 합치기](#8-모두-합치기)
9. [연습 문제](#9-연습-문제)

---

## 1. 오류 처리 미들웨어

Express는 **4개 인수** `(err, req, res, next)`를 통해 오류 처리 미들웨어를 인식합니다. 이것이 일반 미들웨어 및 라우트 핸들러와 구별되는 점입니다.

### 처리되지 않은 오류의 문제

```javascript
// 오류 처리 없이 처리되지 않은 예외는 서버를 충돌시킵니다
app.get('/api/users/:id', (req, res) => {
  const user = getUserById(req.params.id); // 사용자를 찾지 못하면 예외 발생
  res.json(user); // 절대 도달하지 않음 — 서버 충돌
});
```

### 중앙집중식 오류 핸들러

```javascript
// 커스텀 오류 클래스 정의 — HTTP 상태 코드를 오류에 첨부할 수 있습니다
// 오류 핸들러가 추측 없이 어떤 상태를 보낼지 알 수 있습니다
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true; // 예상된 오류와 버그를 구별합니다
  }
}

// 알려진 오류를 발생시키는 라우트
app.get('/api/users/:id', (req, res, next) => {
  const user = users.find(u => u.id === parseInt(req.params.id));
  if (!user) {
    // next()에 오류 전달 — Express가 오류 핸들러로 건너뜁니다
    return next(new AppError('User not found', 404));
  }
  res.json(user);
});

// 오류 처리 미들웨어 — 정확히 4개의 파라미터를 가져야 합니다
// Express는 인수 개수로 이것이 오류 핸들러임을 식별합니다
app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;

  // 서버 측 디버깅을 위한 전체 오류 로그
  console.error(`[ERROR] ${err.message}`, {
    statusCode,
    stack: err.stack,
    path: req.originalUrl,
  });

  res.status(statusCode).json({
    error: {
      message: err.isOperational ? err.message : 'Internal server error',
      // 프로덕션에서는 스택 트레이스를 절대 노출하지 마세요 — 구현 세부사항이 드러납니다
      ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
    },
  });
});
```

### 비동기 오류 처리

```javascript
// Express 4.x는 프로미스 거부(rejection)를 자동으로 처리하지 않습니다.
// 래퍼(wrapper) 없이 비동기 오류는 오류 핸들러를 완전히 우회합니다.

// 비동기 라우트 핸들러를 감싸는 헬퍼 — 거부된 프로미스를 잡아서
// next()를 통해 오류 미들웨어로 전달합니다
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

app.get('/api/posts', asyncHandler(async (req, res) => {
  const posts = await fetchPostsFromDB(); // 이것이 거부되면 오류 핸들러가 처리합니다
  res.json(posts);
}));

// 참고: Express 5.x (현재 베타)는 비동기 오류를 네이티브로 처리합니다
```

---

## 2. Passport.js를 사용한 인증

Passport.js는 500개 이상의 전략을 가진 Node.js 인증 미들웨어입니다. 가장 일반적인 두 가지인 **로컬(local)** (사용자명/비밀번호)과 **JWT** 전략에 집중합니다.

### 설치

```bash
npm install passport passport-local passport-jwt jsonwebtoken bcrypt
```

### 로컬 전략 (사용자명 + 비밀번호)

```javascript
// src/auth/passport.js
import passport from 'passport';
import { Strategy as LocalStrategy } from 'passport-local';
import bcrypt from 'bcrypt';

// 로컬 전략은 데이터베이스에 대해 자격 증명을 검증합니다
passport.use(new LocalStrategy(
  {
    usernameField: 'email', // 기본 필드명 'username'을 재정의합니다
    passwordField: 'password',
  },
  async (email, password, done) => {
    try {
      const user = await findUserByEmail(email);
      if (!user) {
        // null = 시스템 오류 없음, false = 인증 실패
        return done(null, false, { message: 'User not found' });
      }

      // bcrypt.compare는 솔트(salt) 추출을 자동으로 처리합니다 —
      // 솔트는 저장된 해시에 내장되어 있습니다
      const isValid = await bcrypt.compare(password, user.passwordHash);
      if (!isValid) {
        return done(null, false, { message: 'Incorrect password' });
      }

      return done(null, user);
    } catch (err) {
      return done(err); // 시스템 오류 — 오류 미들웨어로 전달됩니다
    }
  }
));
```

### JWT 전략

```javascript
// src/auth/passport.js (계속)
import { Strategy as JwtStrategy, ExtractJwt } from 'passport-jwt';

const jwtOptions = {
  // Authorization: Bearer <token> 헤더에서 토큰을 추출합니다
  jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
  secretOrKey: process.env.JWT_SECRET,
};

passport.use(new JwtStrategy(jwtOptions, async (payload, done) => {
  try {
    const user = await findUserById(payload.sub);
    if (!user) return done(null, false);
    return done(null, user);
  } catch (err) {
    return done(err);
  }
}));
```

### 로그인 라우트 (JWT 발급)

```javascript
// src/routes/auth.js
import { Router } from 'express';
import passport from 'passport';
import jwt from 'jsonwebtoken';

const router = Router();

router.post('/login', (req, res, next) => {
  // { session: false } — 서버 사이드 세션 대신 JWT를 사용하므로
  // Passport가 세션을 생성할 필요가 없습니다
  passport.authenticate('local', { session: false }, (err, user, info) => {
    if (err) return next(err);
    if (!user) return res.status(401).json({ error: info.message });

    const token = jwt.sign(
      { sub: user.id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '1h' } // 단기 토큰은 탈취 시 피해를 제한합니다
    );

    res.json({ token, expiresIn: 3600 });
  })(req, res, next);
});

export default router;
```

### 라우트 보호하기

```javascript
// 재사용 가능한 미들웨어 — 인증이 필요한 모든 라우트에 첨부합니다
const requireAuth = passport.authenticate('jwt', { session: false });

app.get('/api/profile', requireAuth, (req, res) => {
  // req.user는 JWT 검증 성공 후 Passport에 의해 채워집니다
  res.json({ user: req.user });
});
```

---

## 3. 속도 제한

속도 제한(rate limiting)은 클라이언트가 시간 창(time window) 내에 만들 수 있는 요청 수를 제한하여 남용을 방지합니다.

```bash
npm install express-rate-limit
```

```javascript
import rateLimit from 'express-rate-limit';

// 전역 속도 제한 — 모든 라우트에 적용됩니다
const globalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15분 창
  max: 100,                  // IP당 창당 100개 요청
  standardHeaders: true,     // RateLimit-* 헤더에 속도 제한 정보 반환
  legacyHeaders: false,      // X-RateLimit-* 헤더 비활성화
  message: { error: 'Too many requests, please try again later' },
});

app.use(globalLimiter);

// 인증 라우트에 대한 더 엄격한 제한 — 로그인 엔드포인트는
// 무차별 대입(brute-force) 공격의 주요 표적입니다
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: { error: 'Too many login attempts, try again in 15 minutes' },
});

app.use('/api/auth/login', authLimiter);
```

---

## 4. CORS 설정

교차 출처 리소스 공유(Cross-Origin Resource Sharing, CORS)는 브라우저에서 어떤 도메인이 API에 접근할 수 있는지를 제어합니다.

```bash
npm install cors
```

```javascript
import cors from 'cors';

// 모든 출처 허용 — 개발 환경에만 적합합니다
app.use(cors());

// 프로덕션 설정 — 알려진 프론트엔드로 제한합니다
const corsOptions = {
  origin: ['https://myapp.com', 'https://admin.myapp.com'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true, // 교차 출처 쿠키 전송 허용
  maxAge: 86400,     // 24시간 동안 프리플라이트(preflight) 응답 캐시 — OPTIONS 요청 감소
};

app.use(cors(corsOptions));

// 라우트별 CORS — 일부 엔드포인트만 교차 출처 접근이 필요한 경우에 유용합니다
app.get('/api/public/data', cors(), (req, res) => {
  res.json({ data: 'accessible from any origin' });
});
```

---

## 5. Multer를 사용한 파일 업로드

Multer는 파일 업로드에 사용되는 인코딩 타입인 `multipart/form-data`를 처리합니다.

```bash
npm install multer
```

### 기본 파일 업로드

```javascript
import multer from 'multer';
import path from 'node:path';

// 스토리지 설정 — 파일이 저장되는 위치와 이름 지정 방법을 제어합니다
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    // 여러 사용자가 같은 이름의 파일을 업로드할 때 이름 충돌을 방지하기 위해
    // 타임스탬프를 앞에 붙입니다
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

// 파일 필터 — 디스크 공간을 소비하기 전에 미들웨어 수준에서
// 이미지가 아닌 파일을 거부합니다
const fileFilter = (req, file, cb) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only JPEG, PNG, and WebP images are allowed'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 }, // 최대 5MB
});

// 단일 파일 업로드 — 'avatar'는 폼 필드 이름입니다
app.post('/api/upload/avatar', upload.single('avatar'), (req, res) => {
  // req.file에는 업로드된 파일에 대한 메타데이터가 포함됩니다
  res.json({
    filename: req.file.filename,
    size: req.file.size,
    mimetype: req.file.mimetype,
  });
});

// 다중 파일 업로드 — 'photos' 필드에서 최대 10개 파일
app.post('/api/upload/photos', upload.array('photos', 10), (req, res) => {
  const files = req.files.map(f => ({
    filename: f.filename,
    size: f.size,
  }));
  res.json({ uploaded: files });
});
```

---

## 6. Zod를 사용한 입력 검증

Zod는 TypeScript 우선 스키마 검증을 제공합니다. 유효한 데이터의 형태를 정의하고 유효하지 않은 입력에 대해 명확한 오류 메시지를 생성합니다.

```bash
npm install zod
```

### 스키마 정의

```javascript
import { z } from 'zod';

// 스키마는 문서 역할도 합니다 — 독자가 엔드포인트가 무엇을 받는지 정확히 알 수 있습니다
const createUserSchema = z.object({
  name: z.string().min(2).max(100),
  email: z.string().email(),
  age: z.number().int().min(0).max(150).optional(),
  role: z.enum(['user', 'admin']).default('user'),
});

const updateUserSchema = createUserSchema.partial();
// .partial()은 모든 필드를 선택적으로 만듭니다 — PATCH 작업에 적합합니다
```

### 검증 미들웨어 팩토리

```javascript
// 모든 Zod 스키마에 대해 req.body를 검증하는 범용 미들웨어
const validate = (schema) => (req, res, next) => {
  const result = schema.safeParse(req.body);

  if (!result.success) {
    // Zod는 필드별로 구조화된 오류 세부사항을 제공합니다 —
    // flatten()은 프론트엔드에서 쉽게 사용할 수 있도록 필드명별로 그룹화합니다
    const errors = result.error.flatten().fieldErrors;
    return res.status(400).json({ error: 'Validation failed', details: errors });
  }

  // req.body를 파싱된 데이터로 교체합니다 — Zod는 알 수 없는 필드를 제거하고
  // 기본값을 적용하므로 다운스트림 핸들러가 깨끗한 데이터를 받습니다
  req.body = result.data;
  next();
};

app.post('/api/users', validate(createUserSchema), (req, res) => {
  // 여기서 req.body는 유효함이 보장됩니다
  res.status(201).json({ user: req.body });
});

app.patch('/api/users/:id', validate(updateUserSchema), (req, res) => {
  res.json({ updated: req.params.id, changes: req.body });
});
```

### 쿼리 파라미터 검증

```javascript
const paginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(['asc', 'desc']).default('desc'),
});

// 바디 대신 쿼리를 검증합니다
const validateQuery = (schema) => (req, res, next) => {
  const result = schema.safeParse(req.query);
  if (!result.success) {
    return res.status(400).json({ error: result.error.flatten().fieldErrors });
  }
  req.query = result.data; // 강제 변환(coerce)되고 기본값이 적용된 값
  next();
};

app.get('/api/posts', validateQuery(paginationSchema), (req, res) => {
  const { page, limit, sort } = req.query;
  res.json({ page, limit, sort });
});
```

---

## 7. Helmet을 사용한 보안 헤더

Helmet은 XSS, 클릭재킹(clickjacking), MIME 스니핑(sniffing) 등 일반적인 웹 취약점을 방지하기 위한 다양한 HTTP 헤더를 설정합니다.

```bash
npm install helmet
```

```javascript
import helmet from 'helmet';

// helmet()은 한 번의 호출로 합리적인 보안 헤더 세트를 활성화합니다
app.use(helmet());
```

### Helmet이 설정하는 것들

| 헤더 | 목적 |
|--------|---------|
| `Content-Security-Policy` | 스크립트, 스타일, 이미지의 출처를 제한합니다 |
| `X-Content-Type-Options: nosniff` | MIME 타입 스니핑을 방지합니다 |
| `X-Frame-Options: SAMEORIGIN` | iframe 임베딩을 통한 클릭재킹을 방지합니다 |
| `Strict-Transport-Security` | HTTPS 연결을 강제합니다 |
| `X-XSS-Protection: 0` | 결함 있는 브라우저 XSS 필터를 비활성화합니다 |
| `X-DNS-Prefetch-Control: off` | DNS 프리페칭(prefetching)을 제어합니다 |

### 커스텀 CSP 설정

```javascript
// HTML을 제공하지 않는 API에 대한 기본 Content-Security-Policy 재정의
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", 'data:', 'https:'],
      },
    },
    // 순수 API 서버(HTML 응답 없음)에 대해 CSP를 완전히 비활성화
    // contentSecurityPolicy: false,
  })
);
```

---

## 8. 모두 합치기

프로덕션 준비가 된 Express 앱은 이 미들웨어들을 특정 순서로 구성합니다:

```javascript
// src/app.js
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import passport from 'passport';

import './auth/passport.js'; // 전략 초기화 (사이드 이펙트 임포트)
import authRouter from './routes/auth.js';
import usersRouter from './routes/users.js';

const app = express();

// --- 보안 미들웨어 먼저 ---
app.use(helmet());                       // 보안 헤더
app.use(cors({ origin: process.env.ALLOWED_ORIGINS?.split(',') }));

// --- 바디 파싱 전에 속도 제한 ---
// 바디를 파싱하는 데 리소스를 소비하기 전에 악의적인 요청을 일찍 거부합니다
app.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 100 }));

// --- 바디 파싱 ---
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// --- 인증 ---
app.use(passport.initialize());

// --- 라우트 ---
app.use('/api/auth', authRouter);
app.use('/api/users', usersRouter);

// --- 404 핸들러 ---
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// --- 중앙집중식 오류 핸들러 (반드시 마지막에) ---
app.use((err, req, res, next) => {
  console.error(err);
  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    error: { message: err.isOperational ? err.message : 'Internal server error' },
  });
});

export default app;
```

---

## 9. 연습 문제

### 문제 1: 커스텀 오류 클래스

세 가지 커스텀 오류 클래스가 있는 오류 처리 시스템을 만드세요:
- `NotFoundError` (404)
- 필드 수준 오류가 있는 `details` 객체를 받는 `ValidationError` (400)
- `UnauthorizedError` (401)

응답에서 각 오류 타입을 다르게 형식화하는 중앙집중식 오류 핸들러를 작성하세요.

### 문제 2: JWT 인증 흐름

완전한 인증 흐름을 구현하세요:
- `POST /api/auth/register` -- bcrypt로 비밀번호 해시, 사용자 저장
- `POST /api/auth/login` -- 자격 증명 검증, JWT 반환
- `GET /api/auth/me` -- 현재 사용자 프로필 반환 (보호된 라우트)

사용자 저장소로 인메모리 배열을 사용하세요. 중복 이메일과 잘못된 비밀번호에 대한 적절한 오류 메시지를 포함하세요.

### 문제 3: 속도 제한 계층

세 가지 속도 제한 계층을 설정하세요:
- **공개 엔드포인트**: 15분당 100개 요청
- **인증된 엔드포인트**: 15분당 1000개 요청 (JWT로 식별)
- **관리자 엔드포인트**: 제한 없음

사용자의 역할에 따라 적절한 계층을 선택하는 미들웨어 팩토리 `rateLimitByRole()`을 작성하세요.

### 문제 4: 검증이 있는 파일 업로드

다음을 수행하는 이미지 업로드 엔드포인트를 만드세요:
- 최대 2MB의 JPEG 및 PNG 파일 허용
- 이미지가 최소 100x100 픽셀 크기인지 검증 (힌트: `image-size` 패키지 사용)
- 응답에 파일 경로, 크기, 파일 크기를 반환
- 유효하지 않은 파일에 대해 설명적인 오류 메시지 반환

### 문제 5: 요청 검증 파이프라인

Zod를 사용하여 "블로그 포스트 생성" 엔드포인트에 대한 검증 미들웨어를 만드세요:
- `title`: 문자열, 5~200자
- `content`: 문자열, 최소 50자
- `tags`: 문자열 배열, 1~5개 항목, 각 2~30자
- `publishAt`: 미래여야 하는 선택적 ISO 날짜 문자열

필드 수준 오류 메시지로 검증 오류를 처리하세요.

---

## 참고 자료

- [Express 오류 처리 가이드](https://expressjs.com/en/guide/error-handling.html)
- [Passport.js 공식 문서](http://www.passportjs.org/docs/)
- [express-rate-limit 공식 문서](https://github.com/express-rate-limit/express-rate-limit)
- [Helmet.js 공식 문서](https://helmetjs.github.io/)
- [Multer 공식 문서](https://github.com/expressjs/multer)
- [Zod 공식 문서](https://zod.dev/)
- [OWASP 보안 헤더](https://owasp.org/www-project-secure-headers/)

---

**이전**: [Express 기초](./06_Express_Basics.md) | **다음**: [Express 데이터베이스](./08_Express_Database.md)
