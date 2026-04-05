# JavaScript 비동기 프로그래밍

**이전**: [JS 이벤트와 DOM](./07_JS_Events_DOM.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 동기(synchronous)와 비동기(asynchronous) 실행의 차이를 설명하고, JavaScript 이벤트 루프(event loop)를 묘사할 수 있다
2. 콜백 지옥(callback hell)을 식별하고, 깊이 중첩된 콜백이 왜 문제인지 설명할 수 있다
3. resolve/reject 생성자, then/catch/finally 체인, 정적 메서드(Promise.all, Promise.race, Promise.allSettled, Promise.any)를 사용하여 Promise를 생성하고 소비할 수 있다
4. 적절한 try/catch 에러 처리와 함께 async/await 구문을 사용하여 비동기 함수를 작성할 수 있다
5. 순차(sequential)와 병렬(parallel) 비동기 실행을 비교하고 적절한 패턴을 선택할 수 있다
6. 다양한 메서드(GET, POST, PUT, DELETE), 헤더, 요청 본문을 포함하여 Fetch API를 사용한 HTTP 요청을 구현할 수 있다
7. 재시도 로직, AbortController를 사용한 요청 취소, 디바운스(debounce), 쓰로틀(throttle) 등 실전 비동기 패턴을 적용할 수 있다

---

웹 애플리케이션에서 흥미로운 동작의 대부분은 기다리는 과정을 포함합니다. 서버 응답을 기다리고, 파일 업로드를 기다리고, 데이터베이스 쿼리를 기다립니다. JavaScript가 매번 기다리는 동안 실행을 차단한다면, 브라우저는 멈추고 사용자 인터페이스는 응답 불가 상태가 될 것입니다. 비동기 프로그래밍은 JavaScript가 오래 걸리는 작업을 시작하고, 다른 작업을 계속 수행하다가, 결과가 준비되면 돌아올 수 있게 해주는 기법입니다. 이를 통해 애플리케이션은 빠릿하게 동작하고 사용자는 쾌적한 경험을 누릴 수 있습니다.

## 목차

1. [동기 vs 비동기](#동기-vs-비동기)
2. [콜백](#콜백)
3. [Promise](#promise)
4. [async/await](#asyncawait)
5. [Fetch API](#fetch-api)
6. [에러 처리](#에러-처리)
7. [실전 패턴](#실전-패턴)

---

## 동기 vs 비동기

### 동기 (Synchronous)

코드가 순서대로 실행됩니다. 한 작업이 끝나야 다음 작업이 시작됩니다.

```javascript
console.log('1');
console.log('2');
console.log('3');
// 출력: 1, 2, 3
```

### 비동기 (Asynchronous)

작업을 기다리지 않고 다음 코드를 실행합니다.

```javascript
console.log('1');
setTimeout(() => {
    console.log('2');
}, 1000);
console.log('3');
// 출력: 1, 3, 2 (1초 후 2)
```

### 비동기가 필요한 상황

- 서버 요청 (API 호출)
- 파일 읽기/쓰기
- 타이머 (setTimeout, setInterval)
- 이벤트 리스너
- 데이터베이스 쿼리

### 이벤트 루프

```
┌─────────────────────────────────────────┐
│                                         │
│          Call Stack (호출 스택)          │
│                                         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│                                         │
│          Web APIs                       │
│   (setTimeout, fetch, DOM events)       │
│                                         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│                                         │
│     Callback Queue / Task Queue         │
│                                         │
└───────────────────┬─────────────────────┘
                    │
                    ▼
            Event Loop가 감시
                    │
          Call Stack이 비면
                    │
          Callback을 Stack으로
```

---

## 콜백

### 콜백 함수란?

다른 함수에 인자로 전달되어 나중에 실행되는 함수입니다.

```javascript
function greet(name, callback) {
    console.log(`안녕, ${name}!`);
    callback();
}

greet('홍길동', function() {
    console.log('인사 완료');
});
```

### setTimeout / setInterval

```javascript
// 일정 시간 후 실행
setTimeout(() => {
    console.log('3초 후 실행');
}, 3000);

// 반복 실행
const intervalId = setInterval(() => {
    console.log('1초마다 실행');
}, 1000);

// 타이머 취소
clearTimeout(timeoutId);
clearInterval(intervalId);
```

### 콜백 지옥 (Callback Hell)

중첩된 콜백으로 코드가 복잡해지는 문제

```javascript
// 나쁜 예: 콜백 지옥
getUser(userId, function(user) {
    getOrders(user.id, function(orders) {
        getOrderDetails(orders[0].id, function(details) {
            getProductInfo(details.productId, function(product) {
                console.log(product);
                // 계속 중첩...
            });
        });
    });
});
```

---

## Promise

### Promise란?

비동기 작업의 최종 완료(또는 실패)와 그 결과값을 나타내는 객체입니다.

```
Promise 상태:
┌──────────┐     ┌───────────┐
│ pending  │────▶│ fulfilled │  (성공)
│ (대기)   │     └───────────┘
│          │     ┌───────────┐
│          │────▶│ rejected  │  (실패)
└──────────┘     └───────────┘
```

### Promise 생성

```javascript
const promise = new Promise((resolve, reject) => {
    // 비동기 작업
    const success = true;

    if (success) {
        resolve('성공!');  // 성공 시
    } else {
        reject('실패!');   // 실패 시
    }
});
```

### then / catch / finally

```javascript
promise
    .then(result => {
        console.log(result);  // 성공 시 실행
        return '다음 값';     // 체이닝 가능
    })
    .then(nextResult => {
        console.log(nextResult);
    })
    .catch(error => {
        console.error(error);  // 실패 시 실행
    })
    .finally(() => {
        console.log('항상 실행');  // 성공/실패 무관
    });
```

### Promise 체이닝

```javascript
fetchUser(userId)
    .then(user => fetchOrders(user.id))
    .then(orders => fetchOrderDetails(orders[0].id))
    .then(details => fetchProductInfo(details.productId))
    .then(product => console.log(product))
    .catch(error => console.error(error));
```

### Promise 정적 메서드

```javascript
// 즉시 이행된 Promise
Promise.resolve('값');

// 즉시 거부된 Promise
Promise.reject('에러');

// Promise.all — 모든 요청이 반드시 성공해야 할 때 사용; 하나라도 실패하면 즉시 실패 (예: 대시보드 필수 데이터 전부 로딩)
Promise.all([promise1, promise2, promise3])
    .then(results => {
        // results = [result1, result2, result3]
    })
    .catch(error => {
        // 첫 번째 실패 — 전체 작업 실패; 부분 결과가 의미 없을 때 적합
    });

// Promise.allSettled — 성공/실패 여부와 무관하게 모든 결과가 필요할 때 사용 (예: 일부 성공도 허용하는 배치 작업)
Promise.allSettled([promise1, promise2, promise3])
    .then(results => {
        results.forEach(result => {
            if (result.status === 'fulfilled') {
                console.log('성공:', result.value);
            } else {
                console.log('실패:', result.reason);
            }
        });
    });

// Promise.race — 타임아웃 패턴에 사용: 가장 먼저 완료되는 것이 이김 (reject여도 포함)
Promise.race([promise1, promise2, promise3])
    .then(result => {
        // 가장 빠른 결과 — "가장 가까운 서버에서 가져오기" 또는 커스텀 타임아웃 구현에 유용
    });

// Promise.any — 첫 번째 성공만 필요하고 실패는 신경 쓰지 않을 때 사용 (예: 여러 CDN 미러 시도)
Promise.any([promise1, promise2, promise3])
    .then(result => {
        // 가장 먼저 성공한 결과 — race와 달리, 모두 실패할 때까지 reject를 무시
    })
    .catch(error => {
        // AggregateError: 모든 Promise가 실패한 경우
    });
```

### Promise 실전 예제

```javascript
// 타이머 Promise화
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

delay(2000).then(() => console.log('2초 경과'));

// 콜백 API를 Promise로 변환
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`이미지 로드 실패: ${src}`));
        img.src = src;
    });
}

loadImage('image.jpg')
    .then(img => document.body.appendChild(img))
    .catch(error => console.error(error));
```

---

## async/await

### 기본 사용법

async 함수는 항상 Promise를 반환합니다. await는 Promise가 처리될 때까지 기다립니다.

```javascript
// async 함수는 항상 Promise를 반환; await는 Promise를 unwrap — .then() 체인의 문법적 설탕(syntactic sugar)
// 장점: 중첩된 .catch() 대신 try/catch로 에러 처리, 코드가 위에서 아래로 읽힘
async function fetchData() {
    const response = await fetch('/api/data');
    const data = await response.json();
    return data;
}

// 호출
fetchData().then(data => console.log(data));
```

### 에러 처리

```javascript
async function fetchData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('에러 발생:', error);
        throw error;  // 에러 재전파
    }
}
```

### 순차 실행 vs 병렬 실행

```javascript
// 순차 실행 (느림)
async function sequential() {
    const user = await fetchUser();      // 1초
    const orders = await fetchOrders();  // 1초
    const products = await fetchProducts();  // 1초
    // 총 3초
}

// 병렬 실행 (빠름)
async function parallel() {
    const [user, orders, products] = await Promise.all([
        fetchUser(),
        fetchOrders(),
        fetchProducts()
    ]);
    // 총 1초 (가장 느린 것 기준)
}
```

### 반복문에서 async/await

```javascript
// 순차 처리
async function processSequential(items) {
    for (const item of items) {
        await processItem(item);  // 하나씩 순서대로
    }
}

// 병렬 처리
async function processParallel(items) {
    await Promise.all(items.map(item => processItem(item)));
}

// forEach는 await를 기다리지 않음 (주의!)
items.forEach(async item => {
    await processItem(item);  // 동시에 모두 시작됨
});
```

### 최상위 await (ES2022)

모듈에서 최상위 레벨에서 await 사용 가능

```javascript
// module.js
const response = await fetch('/api/config');
export const config = await response.json();
```

---

## Fetch API

### 기본 사용법

```javascript
// GET 요청
fetch('https://api.example.com/data')
    .then(res => res.json())  // res.json()은 또 다른 Promise를 반환 — 파싱 전에 응답 바디 스트림 전체를 읽어야 함; 이 단계는 헤더 도착 시점이고, 다음 .then이 파싱된 데이터를 받음
    .then(data => console.log(data))
    .catch(error => console.error(error));

// async/await
async function getData() {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    return data;
}
```

### HTTP 메서드

```javascript
// GET (기본)
fetch('/api/users');

// POST
fetch('/api/users', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        name: '홍길동',
        email: 'hong@example.com'
    })
});

// PUT
fetch('/api/users/1', {
    method: 'PUT',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        name: '김철수'
    })
});

// PATCH
fetch('/api/users/1', {
    method: 'PATCH',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        name: '이영희'
    })
});

// DELETE
fetch('/api/users/1', {
    method: 'DELETE'
});
```

### Response 객체

```javascript
const response = await fetch('/api/data');

// 응답 정보
response.ok;         // 성공 여부 (200-299)
response.status;     // HTTP 상태 코드
response.statusText; // 상태 메시지
response.headers;    // 응답 헤더
response.url;        // 요청 URL

// 본문 읽기 (한 번만 가능)
await response.json();   // JSON 파싱
await response.text();   // 텍스트
await response.blob();   // Blob (바이너리)
await response.arrayBuffer();  // ArrayBuffer
await response.formData();     // FormData
```

### 헤더 설정

```javascript
fetch('/api/data', {
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
        'X-Custom-Header': 'value'
    }
});

// Headers 객체
const headers = new Headers();
headers.append('Content-Type', 'application/json');
headers.append('Authorization', 'Bearer ' + token);

fetch('/api/data', { headers });
```

### 쿼리 파라미터

```javascript
// URL에 직접
fetch('/api/users?page=1&limit=10');

// URLSearchParams 사용
const params = new URLSearchParams({
    page: 1,
    limit: 10,
    search: '홍길동'
});

fetch(`/api/users?${params}`);

// URL 객체
const url = new URL('/api/users', 'https://api.example.com');
url.searchParams.set('page', 1);
url.searchParams.set('limit', 10);

fetch(url);
```

### FormData 전송

```javascript
// HTML 폼에서
const form = document.querySelector('form');
const formData = new FormData(form);

fetch('/api/upload', {
    method: 'POST',
    body: formData  // Content-Type 자동 설정
});

// 직접 생성
const formData = new FormData();
formData.append('username', '홍길동');
formData.append('file', fileInput.files[0]);

fetch('/api/upload', {
    method: 'POST',
    body: formData
});
```

### 파일 업로드

```javascript
const fileInput = document.querySelector('input[type="file"]');

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('업로드 실패');

        const result = await response.json();
        console.log('업로드 완료:', result);
    } catch (error) {
        console.error(error);
    }
});
```

### 요청 취소 (AbortController)

```javascript
const controller = new AbortController();
const signal = controller.signal;

// 5초 후 자동 취소
const timeoutId = setTimeout(() => controller.abort(), 5000);

try {
    const response = await fetch('/api/data', { signal });
    clearTimeout(timeoutId);
    const data = await response.json();
} catch (error) {
    if (error.name === 'AbortError') {
        console.log('요청이 취소되었습니다');
    } else {
        throw error;
    }
}

// fetch Promise를 AbortError로 reject — 사용자가 페이지를 이탈하거나 타임아웃이 만료될 때 요청을 취소하는 데 유용
controller.abort();
```

### 타임아웃 유틸리티

```javascript
async function fetchWithTimeout(url, options = {}, timeout = 5000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        return response;
    } finally {
        clearTimeout(timeoutId);
    }
}

// 사용
const response = await fetchWithTimeout('/api/data', {}, 3000);
```

---

## 에러 처리

### fetch는 네트워크 에러만 reject

```javascript
// 주의: 404, 500 등은 에러가 아님!
fetch('/api/not-found')
    .then(response => {
        // 404여도 여기로 옴
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .catch(error => {
        // 네트워크 에러 또는 위에서 던진 에러
        console.error(error);
    });
```

### 커스텀 에러 클래스

```javascript
class HttpError extends Error {
    constructor(response) {
        super(`HTTP error! status: ${response.status}`);
        this.name = 'HttpError';
        this.response = response;
    }
}

async function fetchJSON(url, options) {
    const response = await fetch(url, options);

    if (!response.ok) {
        throw new HttpError(response);
    }

    return response.json();
}

// 사용
try {
    const data = await fetchJSON('/api/data');
} catch (error) {
    if (error instanceof HttpError) {
        if (error.response.status === 404) {
            console.log('데이터를 찾을 수 없습니다');
        } else if (error.response.status === 401) {
            console.log('인증이 필요합니다');
        }
    } else {
        console.error('네트워크 에러:', error);
    }
}
```

### 재시도 로직

```javascript
async function fetchWithRetry(url, options = {}, retries = 3, delay = 1000) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response;
        } catch (error) {
            if (i === retries - 1) throw error;

            console.log(`재시도 ${i + 1}/${retries}...`);
            await new Promise(r => setTimeout(r, delay * (i + 1)));
        }
    }
}
```

---

## 실전 패턴

### API 클라이언트

```javascript
class ApiClient {
    constructor(baseURL) {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || `HTTP ${response.status}`);
        }

        return response.json();
    }

    get(endpoint) {
        return this.request(endpoint);
    }

    post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: data
        });
    }

    put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: data
        });
    }

    delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }
}

// 사용
const api = new ApiClient('https://api.example.com');

const users = await api.get('/users');
const newUser = await api.post('/users', { name: '홍길동' });
await api.delete('/users/1');
```

### 로딩 상태 관리

```javascript
class LoadingState {
    constructor() {
        this.isLoading = false;
        this.error = null;
        this.data = null;
    }

    async execute(asyncFn) {
        this.isLoading = true;
        this.error = null;

        try {
            this.data = await asyncFn();
            return this.data;
        } catch (error) {
            this.error = error;
            throw error;
        } finally {
            this.isLoading = false;
        }
    }
}

// 사용
const state = new LoadingState();

// UI 업데이트
function updateUI() {
    if (state.isLoading) {
        showSpinner();
    } else if (state.error) {
        showError(state.error.message);
    } else {
        renderData(state.data);
    }
}

await state.execute(() => fetch('/api/data').then(r => r.json()));
updateUI();
```

### 디바운스와 쓰로틀

```javascript
// 디바운스: 마지막 호출 후 일정 시간 후 실행
function debounce(fn, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

// 검색 자동완성
const searchInput = document.querySelector('#search');
searchInput.addEventListener('input', debounce(async (e) => {
    const results = await fetch(`/api/search?q=${e.target.value}`);
    renderResults(await results.json());
}, 300));

// 쓰로틀: 일정 시간에 한 번만 실행
function throttle(fn, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// 스크롤 이벤트
window.addEventListener('scroll', throttle(() => {
    console.log('스크롤 위치:', window.scrollY);
}, 100));
```

### 무한 스크롤

```javascript
class InfiniteScroll {
    constructor(container, loadMore) {
        this.container = container;
        this.loadMore = loadMore;
        this.page = 1;
        this.loading = false;
        this.hasMore = true;

        this.setupObserver();
    }

    setupObserver() {
        const sentinel = document.createElement('div');
        sentinel.className = 'sentinel';
        this.container.appendChild(sentinel);

        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && !this.loading && this.hasMore) {
                this.load();
            }
        });

        observer.observe(sentinel);
    }

    async load() {
        this.loading = true;

        try {
            const items = await this.loadMore(this.page);

            if (items.length === 0) {
                this.hasMore = false;
                return;
            }

            items.forEach(item => {
                const el = this.createItem(item);
                this.container.insertBefore(el, this.container.lastChild);
            });

            this.page++;
        } finally {
            this.loading = false;
        }
    }

    createItem(item) {
        const div = document.createElement('div');
        div.className = 'item';
        div.textContent = item.title;
        return div;
    }
}

// 사용
new InfiniteScroll(
    document.querySelector('#list'),
    async (page) => {
        const response = await fetch(`/api/items?page=${page}`);
        return response.json();
    }
);
```

### 캐싱

```javascript
class ApiCache {
    constructor(ttl = 60000) {  // 기본 1분
        this.cache = new Map();
        this.ttl = ttl;
    }

    async fetch(url, options) {
        const key = JSON.stringify({ url, options });
        const cached = this.cache.get(key);

        if (cached && Date.now() < cached.expiry) {
            return cached.data;
        }

        const response = await fetch(url, options);
        const data = await response.json();

        this.cache.set(key, {
            data,
            expiry: Date.now() + this.ttl
        });

        return data;
    }

    clear() {
        this.cache.clear();
    }
}

const cachedApi = new ApiCache(30000);  // 30초 캐시
const data = await cachedApi.fetch('/api/data');
```

---

## 연습 문제

### 문제 1: Promise 체이닝

3개의 API를 순차적으로 호출하여 결과를 합치세요.

```javascript
// /api/user/1 → { name: '홍길동' }
// /api/user/1/posts → [{ id: 1, title: '글1' }]
// /api/posts/1/comments → [{ id: 1, text: '댓글1' }]
```

<details>
<summary>정답 보기</summary>

```javascript
async function getUserData(userId) {
    const user = await fetch(`/api/user/${userId}`).then(r => r.json());
    const posts = await fetch(`/api/user/${userId}/posts`).then(r => r.json());
    const comments = await fetch(`/api/posts/${posts[0].id}/comments`).then(r => r.json());

    return {
        user,
        posts,
        firstPostComments: comments
    };
}
```

</details>

### 문제 2: 병렬 요청

여러 사용자의 정보를 동시에 가져오세요.

```javascript
const userIds = [1, 2, 3, 4, 5];
```

<details>
<summary>정답 보기</summary>

```javascript
async function fetchAllUsers(userIds) {
    const promises = userIds.map(id =>
        fetch(`/api/users/${id}`).then(r => r.json())
    );

    const users = await Promise.all(promises);
    return users;
}

// 또는 실패 허용
async function fetchAllUsersSafe(userIds) {
    const results = await Promise.allSettled(
        userIds.map(id =>
            fetch(`/api/users/${id}`).then(r => r.json())
        )
    );

    return results
        .filter(r => r.status === 'fulfilled')
        .map(r => r.value);
}
```

</details>

### 문제 3: 폼 제출

폼 데이터를 서버에 전송하고 결과를 처리하세요.

<details>
<summary>정답 보기</summary>

```javascript
const form = document.querySelector('#userForm');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = form.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = '전송 중...';

    try {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        const response = await fetch('/api/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('서버 에러');
        }

        const result = await response.json();
        alert('등록 완료!');
        form.reset();
    } catch (error) {
        alert('에러: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '전송';
    }
});
```

</details>

---

## 다음 단계

- [실전 프로젝트](./09_Practical_Projects.md) - 배운 내용을 종합한 프로젝트

---

## 참고 자료

- [MDN Promise](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Promise)
- [MDN async/await](https://developer.mozilla.org/ko/docs/Learn/JavaScript/Asynchronous/Promises)
- [MDN Fetch API](https://developer.mozilla.org/ko/docs/Web/API/Fetch_API)
- [JavaScript.info 비동기](https://ko.javascript.info/async)
