# 17. 서비스 워커와 PWA

**이전**: [Flask 기초](./16_Flask_Basics.md) | **다음**: [코어 웹 바이탈](./18_Core_Web_Vitals.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프로그레시브 웹 앱(Progressive Web App) 개념을 설명하고 유효한 `manifest.json` 생성
2. 서비스 워커(Service Worker) 생명주기 이해: install, activate, fetch 이벤트
3. 캐싱 전략 구현: Cache First, Network First, Stale-While-Revalidate
4. 웹 애플리케이션의 오프라인 기능 구축
5. Push API를 사용한 푸시 알림(push notification) 전송
6. 백그라운드 싱크(Background Sync)를 사용하여 연결이 복원될 때까지 작업 지연
7. Workbox 라이브러리를 활용한 서비스 워커 개발 간소화
8. PWA 설치 가능성(installability) 기준 충족 및 Lighthouse PWA 감사 통과

---

프로그레시브 웹 앱(PWA)은 웹 페이지와 네이티브 애플리케이션 사이의 간극을 메워줍니다. 일반 웹 페이지처럼 로드되지만, 전통적으로 네이티브 앱에만 가능했던 기능들 -- 오프라인 접근, 푸시 알림, 홈 화면 설치 -- 을 제공합니다. 모든 PWA의 핵심에는 **서비스 워커(Service Worker)**가 있으며, 이는 캐싱과 네트워크 요청에 대한 세밀한 제어를 제공하는 프로그래밍 가능한 네트워크 프록시입니다.

## 1. 프로그레시브 웹 앱 기본 개념

### 1.1 PWA란 무엇인가?

PWA는 특정 기술이 아닙니다. 브라우저에서 앱과 같은 경험을 제공하기 위해 결합되는 모범 사례의 집합입니다.

> **PWA의 세 가지 핵심 요소**
>
> 1. **가능성(Capable)** -- 기기 기능 접근 (카메라, 위치 정보, 알림)
> 2. **신뢰성(Reliable)** -- 즉시 로드되며 오프라인이나 불안정한 네트워크에서도 작동
> 3. **설치 가능성(Installable)** -- 앱 스토어 없이 홈 화면에 설치

### 1.2 PWA vs 네이티브 vs 전통적 웹

| 기능 | 전통적 웹 | PWA | 네이티브 앱 |
|---|---|---|---|
| 검색 가능성 | 검색 엔진 | 검색 엔진 | 앱 스토어 |
| 설치 | 없음 | 홈 화면 추가 | 스토어 다운로드 |
| 오프라인 지원 | 없음 | 서비스 워커 캐시 | 완전 지원 |
| 푸시 알림 | 불가 | 가능 | 가능 |
| 기기 API | 제한적 | 점차 확대 | 완전 지원 |
| 업데이트 방식 | 즉시 (서버) | SW 업데이트 주기 | 스토어 심사 |

### 1.3 웹 앱 매니페스트(Web App Manifest)

`manifest.json` 파일은 브라우저에게 PWA에 대한 정보와 설치 시 동작 방식을 알려줍니다.

```json
{
  "name": "My Study Hub",
  "short_name": "StudyHub",
  "description": "A progressive web app for learning",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#2196F3",
  "orientation": "portrait-primary",
  "icons": [
    {
      "src": "/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ],
  "screenshots": [
    {
      "src": "/screenshots/home.png",
      "sizes": "1280x720",
      "type": "image/png",
      "form_factor": "wide"
    }
  ],
  "categories": ["education"],
  "shortcuts": [
    {
      "name": "Recent Lessons",
      "url": "/recent",
      "icons": [{ "src": "/icons/recent.png", "sizes": "96x96" }]
    }
  ]
}
```

HTML `<head>`에 매니페스트를 연결합니다:

```html
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#2196F3">
<!-- iOS Safari 폴백 -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<link rel="apple-touch-icon" href="/icons/icon-192.png">
```

### 1.4 디스플레이 모드(Display Modes)

`display` 필드는 브라우저 크롬(chrome)의 표시 방식을 제어합니다:

| 모드 | 설명 |
|---|---|
| `fullscreen` | 브라우저 UI 완전히 없음 (게임용) |
| `standalone` | 네이티브 앱처럼 보임 (URL 바 없음) |
| `minimal-ui` | standalone + 최소한의 탐색 컨트롤 |
| `browser` | 표준 브라우저 탭 |

```css
/* CSS에서 standalone 모드 감지 */
@media (display-mode: standalone) {
  .install-banner {
    display: none;
  }
}
```

```javascript
// JavaScript에서 standalone 모드 감지
const isStandalone = window.matchMedia('(display-mode: standalone)').matches
                  || window.navigator.standalone; // iOS Safari
```

---

## 2. 서비스 워커 생명주기(Service Worker Lifecycle)

### 2.1 서비스 워커란?

서비스 워커(SW)는 메인 페이지와 별도의 스레드에서 실행되는 JavaScript 파일입니다. 페이지가 보내는 모든 네트워크 요청을 가로채는 **프로그래밍 가능한 네트워크 프록시** 역할을 합니다.

주요 제약 사항:

- **별도의 스레드**에서 실행 (DOM 접근 불가)
- **HTTPS** 필수 (`localhost` 제외)
- 완전한 **비동기**(asynchronous) (`localStorage`, 동기 XHR 사용 불가)
- 파일 위치로 정의되는 **스코프(scope)** 보유

### 2.2 등록(Registration)

```javascript
// main.js — 서비스 워커 등록
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/'
      });
      console.log('SW registered, scope:', registration.scope);
    } catch (error) {
      console.error('SW registration failed:', error);
    }
  });
}
```

### 2.3 생명주기 이벤트(Lifecycle Events)

```
┌──────────┐     ┌───────────┐     ┌───────────┐
│  Install  │────>│  Waiting   │────>│  Activate  │
└──────────┘     └───────────┘     └───────────┘
                                         │
                                         v
                                   ┌───────────┐
                                   │   Fetch    │  (요청 가로채기)
                                   └───────────┘
```

**Install 이벤트** -- 브라우저가 새로운 또는 업데이트된 SW 파일을 감지했을 때 발생합니다. 일반적으로 필수 자산을 사전 캐싱하는 데 사용됩니다.

```javascript
// sw.js
const CACHE_NAME = 'app-cache-v1';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/app.js',
  '/icons/icon-192.png'
];

self.addEventListener('install', (event) => {
  console.log('[SW] Install event');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Pre-caching assets');
        return cache.addAll(PRECACHE_URLS);
      })
  );
  // 대기 건너뛰고 즉시 활성화
  self.skipWaiting();
});
```

**Activate 이벤트** -- 설치 후 다른 SW가 페이지를 제어하지 않을 때 발생합니다. 오래된 캐시를 정리하는 데 사용됩니다.

```javascript
self.addEventListener('activate', (event) => {
  console.log('[SW] Activate event');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => {
            console.log('[SW] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
    })
  );
  // 열린 모든 클라이언트를 즉시 제어
  self.clients.claim();
});
```

**Fetch 이벤트** -- SW 스코프 내의 모든 네트워크 요청에 대해 발생합니다. 캐싱 전략이 구현되는 곳입니다.

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((cachedResponse) => {
        return cachedResponse || fetch(event.request);
      })
  );
});
```

### 2.4 서비스 워커 업데이트

브라우저는 탐색할 때마다 바이트 단위로 새 SW 파일을 확인합니다. 파일이 변경된 경우:

1. 새 SW가 기존 SW와 함께 **설치**됨
2. 기존 SW를 사용하는 모든 탭이 닫힐 때까지 **대기** 상태 진입
3. 다음 탐색 시 **활성화**

즉시 업데이트를 강제하려면:

```javascript
// install 이벤트에서
self.skipWaiting();

// activate 이벤트에서
self.clients.claim();
```

---

## 3. 캐싱 전략(Caching Strategies)

### 3.1 Cache First (캐시 우선, 네트워크 폴백)

거의 변경되지 않는 **정적 자산** (CSS, JS 번들, 이미지)에 가장 적합합니다.

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((cached) => {
        if (cached) {
          return cached;
        }
        return fetch(event.request).then((response) => {
          // 응답은 한 번만 소비 가능하므로 복제
          const responseClone = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
          return response;
        });
      })
  );
});
```

### 3.2 Network First (네트워크 우선, 캐시 폴백)

API 응답 및 HTML 페이지와 같은 **동적 콘텐츠**에 가장 적합합니다.

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const responseClone = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseClone);
        });
        return response;
      })
      .catch(() => {
        return caches.match(event.request);
      })
  );
});
```

### 3.3 Stale-While-Revalidate

캐시된 콘텐츠를 **즉시** 제공한 후 백그라운드에서 캐시를 업데이트합니다. 최신성이 선호되지만 속도가 중요한 리소스에 가장 적합합니다.

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.match(event.request).then((cachedResponse) => {
        const networkFetch = fetch(event.request).then((networkResponse) => {
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
        return cachedResponse || networkFetch;
      });
    })
  );
});
```

### 3.4 Cache Only와 Network Only

```javascript
// Cache Only — 캐시에 보장된 버전 자산
event.respondWith(caches.match(event.request));

// Network Only — 비 GET 요청이나 실시간 데이터
event.respondWith(fetch(event.request));
```

### 3.5 전략 선택 가이드

| 전략 | 사용 사례 | 오프라인 지원 | 최신성 |
|---|---|---|---|
| Cache First | 정적 자산, 아이콘, 폰트 | 가능 | 낮음 |
| Network First | HTML 페이지, API 데이터 | 가능 (폴백) | 높음 |
| Stale-While-Revalidate | 사용자 아바타, 게시글 목록 | 가능 | 중간 |
| Cache Only | 사전 캐시된 앱 셸 | 가능 | 없음 |
| Network Only | 분석, POST 요청 | 불가 | 실시간 |

---

## 4. 오프라인 기능 구축

### 4.1 앱 셸 모델(App Shell Model)

앱 셸은 UI 스켈레톤을 렌더링하는 데 필요한 최소한의 HTML, CSS, JavaScript입니다. 첫 방문 시 캐시되고 이후 방문 시 즉시 로드됩니다.

```
┌──────────────────────────────────┐
│          App Shell (캐시됨)        │
│  ┌────────┐  ┌─────────────────┐ │
│  │ Header │  │   Navigation    │ │
│  └────────┘  └─────────────────┘ │
│  ┌──────────────────────────────┐│
│  │                              ││
│  │     동적 콘텐츠                ││
│  │     (네트워크에서 가져옴)       ││
│  │                              ││
│  └──────────────────────────────┘│
│  ┌──────────────────────────────┐│
│  │         Footer               ││
│  └──────────────────────────────┘│
└──────────────────────────────────┘
```

### 4.2 오프라인 폴백 페이지

```javascript
// sw.js — 사용자 정의 오프라인 페이지 제공
const OFFLINE_URL = '/offline.html';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll([...PRECACHE_URLS, OFFLINE_URL]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => caches.match(OFFLINE_URL))
    );
  }
});
```

```html
<!-- offline.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Offline</title>
  <style>
    body {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
      font-family: system-ui, sans-serif;
      background: #f5f5f5;
    }
    .offline-message {
      text-align: center;
      padding: 2rem;
    }
    .offline-message h1 { font-size: 3rem; margin-bottom: 0.5rem; }
  </style>
</head>
<body>
  <div class="offline-message">
    <h1>📡</h1>
    <h2>You are offline</h2>
    <p>Please check your internet connection and try again.</p>
    <button onclick="window.location.reload()">Retry</button>
  </div>
</body>
</html>
```

### 4.3 동적 API 응답 캐싱

```javascript
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (url.pathname.startsWith('/api/')) {
    // API 호출에는 Network First
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open('api-cache-v1').then((cache) => {
            cache.put(event.request, clone);
          });
          return response;
        })
        .catch(() => caches.match(event.request))
    );
  } else {
    // 정적 자산에는 Cache First
    event.respondWith(
      caches.match(event.request)
        .then((cached) => cached || fetch(event.request))
    );
  }
});
```

### 4.4 오프라인 데이터를 위한 IndexedDB

쿼리가 필요한 구조화된 데이터에는 Cache API 대신 IndexedDB를 사용합니다.

```javascript
// db.js — 간단한 IndexedDB 래퍼
function openDB(name, version, upgradeCallback) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    request.onupgradeneeded = (event) => upgradeCallback(event.target.result);
    request.onsuccess = (event) => resolve(event.target.result);
    request.onerror = (event) => reject(event.target.error);
  });
}

async function initDB() {
  const db = await openDB('study-hub', 1, (db) => {
    if (!db.objectStoreNames.contains('lessons')) {
      const store = db.createObjectStore('lessons', { keyPath: 'id' });
      store.createIndex('topic', 'topic', { unique: false });
    }
  });
  return db;
}

async function saveLesson(db, lesson) {
  const tx = db.transaction('lessons', 'readwrite');
  tx.objectStore('lessons').put(lesson);
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function getLesson(db, id) {
  const tx = db.transaction('lessons', 'readonly');
  const request = tx.objectStore('lessons').get(id);
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
```

---

## 5. 푸시 알림(Push Notifications)

### 5.1 Push API 개요

푸시 알림은 세 당사자가 관련됩니다:

```
┌────────┐         ┌──────────────┐         ┌──────────┐
│ Server │ ──push──> Push Service  │ ──push──> Service   │
│        │         │ (FCM/APNs)   │         │ Worker   │
└────────┘         └──────────────┘         └──────────┘
                                                 │
                                                 v
                                           ┌──────────┐
                                           │ Browser  │
                                           │ (사용자)  │
                                           └──────────┘
```

### 5.2 권한 요청

```javascript
// main.js
async function requestNotificationPermission() {
  const permission = await Notification.requestPermission();
  if (permission === 'granted') {
    console.log('Notification permission granted');
    await subscribeToPush();
  } else {
    console.log('Notification permission denied');
  }
}
```

### 5.3 푸시 구독(Push Subscription)

```javascript
// main.js
async function subscribeToPush() {
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(
      'BEl62iUYgUiv...'  // VAPID 공개 키
    )
  });

  // 구독 정보를 서버로 전송
  await fetch('/api/push/subscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(subscription)
  });
}

function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  const raw = atob(base64);
  return Uint8Array.from([...raw].map((char) => char.charCodeAt(0)));
}
```

### 5.4 서비스 워커에서 푸시 이벤트 처리

```javascript
// sw.js
self.addEventListener('push', (event) => {
  let data = { title: 'Notification', body: 'New update available' };
  if (event.data) {
    data = event.data.json();
  }

  const options = {
    body: data.body,
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    vibrate: [100, 50, 100],
    data: { url: data.url || '/' },
    actions: [
      { action: 'open', title: 'Open' },
      { action: 'dismiss', title: 'Dismiss' }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  if (event.action === 'dismiss') return;

  event.waitUntil(
    clients.openWindow(event.notification.data.url)
  );
});
```

### 5.5 서버 측 푸시 (Node.js)

```javascript
// server.js
const webpush = require('web-push');

webpush.setVapidDetails(
  'mailto:admin@example.com',
  process.env.VAPID_PUBLIC_KEY,
  process.env.VAPID_PRIVATE_KEY
);

async function sendPush(subscription, payload) {
  try {
    await webpush.sendNotification(subscription, JSON.stringify(payload));
    console.log('Push sent successfully');
  } catch (error) {
    if (error.statusCode === 410) {
      // 구독 만료 — 데이터베이스에서 제거
      console.log('Subscription expired, removing');
    }
  }
}
```

---

## 6. 백그라운드 싱크(Background Sync)

### 6.1 백그라운드 싱크란?

백그라운드 싱크(Background Sync)를 사용하면 사용자에게 안정적인 연결이 생길 때까지 작업을 지연시킬 수 있습니다. 예를 들어, 사용자가 오프라인 상태에서 폼을 제출하면 요청이 큐에 저장되고 연결이 복원될 때 재생됩니다.

### 6.2 싱크 이벤트 등록

```javascript
// main.js
async function saveFormOffline(formData) {
  // IndexedDB에 저장
  const db = await initDB();
  const tx = db.transaction('outbox', 'readwrite');
  tx.objectStore('outbox').add({
    url: '/api/notes',
    method: 'POST',
    body: Object.fromEntries(formData),
    timestamp: Date.now()
  });

  // 싱크 등록
  const registration = await navigator.serviceWorker.ready;
  await registration.sync.register('sync-notes');
}
```

### 6.3 싱크 이벤트 처리

```javascript
// sw.js
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-notes') {
    event.waitUntil(replayOutbox());
  }
});

async function replayOutbox() {
  const db = await openDB('study-hub', 1);
  const tx = db.transaction('outbox', 'readonly');
  const store = tx.objectStore('outbox');
  const request = store.getAll();

  return new Promise((resolve, reject) => {
    request.onsuccess = async () => {
      const items = request.result;
      for (const item of items) {
        try {
          await fetch(item.url, {
            method: item.method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(item.body)
          });
          // 성공 시 outbox에서 제거
          const deleteTx = db.transaction('outbox', 'readwrite');
          deleteTx.objectStore('outbox').delete(item.id);
        } catch (err) {
          // 싱크가 자동으로 재시도
          reject(err);
          return;
        }
      }
      resolve();
    };
    request.onerror = () => reject(request.error);
  });
}
```

---

## 7. Workbox 라이브러리

### 7.1 왜 Workbox인가?

서비스 워커를 수동으로 작성하는 것은 오류가 발생하기 쉽습니다. **Workbox** (Google 제공)는 일반적인 SW 패턴을 위한 검증된 프로덕션용 모듈을 제공합니다.

### 7.2 설치

```bash
npm install workbox-cli --save-dev
```

빠른 프로토타이핑을 위해 CDN을 사용할 수도 있습니다:

```javascript
importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.0.0/workbox-sw.js');
```

### 7.3 Workbox 캐싱 전략

```javascript
// sw.js with Workbox
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';
import { precacheAndRoute } from 'workbox-precaching';

// 앱 셸 사전 캐싱 (빌드 도구에 의해 생성)
precacheAndRoute(self.__WB_MANIFEST);

// 이미지에는 Cache First
registerRoute(
  ({ request }) => request.destination === 'image',
  new CacheFirst({
    cacheName: 'images',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] }),
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 30 * 24 * 60 * 60  // 30일
      })
    ]
  })
);

// HTML 페이지에는 Network First
registerRoute(
  ({ request }) => request.mode === 'navigate',
  new NetworkFirst({
    cacheName: 'pages',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] })
    ]
  })
);

// CSS와 JS에는 Stale-While-Revalidate
registerRoute(
  ({ request }) =>
    request.destination === 'style' || request.destination === 'script',
  new StaleWhileRevalidate({
    cacheName: 'static-resources'
  })
);
```

### 7.4 Workbox 빌드 통합 (Webpack)

```javascript
// webpack.config.js
const { InjectManifest } = require('workbox-webpack-plugin');

module.exports = {
  // ... 기타 설정
  plugins: [
    new InjectManifest({
      swSrc: './src/sw.js',
      swDest: 'sw.js',
      maximumFileSizeToCacheInBytes: 5 * 1024 * 1024
    })
  ]
};
```

### 7.5 Workbox CLI 위자드

```bash
npx workbox wizard
# 몇 가지 질문에 답하면 workbox-config.js 생성

npx workbox generateSW workbox-config.js
# 완전한 서비스 워커 생성
```

---

## 8. PWA 설치 가능성(Installability)

### 8.1 설치 가능성 기준

Chrome에서 설치 프롬프트를 표시하려면 앱이 다음 조건을 충족해야 합니다:

1. **HTTPS**를 통해 제공
2. `name`, `icons` (192px 및 512px), `start_url`, `display`를 포함하는 유효한 **웹 앱 매니페스트**
3. `fetch` 이벤트 핸들러가 있는 **서비스 워커** 등록
4. 사용자가 사이트에 **참여**(최소 2회 방문, 방문 간 30초 간격 -- 브라우저에 따라 다름)

### 8.2 설치 프롬프트 처리

```javascript
// main.js
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (event) => {
  // 자동 프롬프트 방지
  event.preventDefault();
  deferredPrompt = event;
  // 사용자 정의 설치 버튼 표시
  document.getElementById('install-btn').style.display = 'block';
});

document.getElementById('install-btn').addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  console.log('Install prompt outcome:', outcome);
  deferredPrompt = null;
  document.getElementById('install-btn').style.display = 'none';
});

window.addEventListener('appinstalled', () => {
  console.log('PWA installed successfully');
  deferredPrompt = null;
});
```

### 8.3 설치 버튼 UI

```html
<button id="install-btn" style="display: none;" class="install-button">
  Install App
</button>
```

```css
.install-button {
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  padding: 0.75rem 1.5rem;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  z-index: 1000;
}
```

---

## 9. Lighthouse PWA 감사(Audit)

### 9.1 Lighthouse 감사 실행

Lighthouse는 Chrome DevTools에 내장되어 있습니다. PWA 감사를 실행하려면:

1. Chrome DevTools 열기 (F12)
2. **Lighthouse** 탭으로 이동
3. **Progressive Web App** 카테고리 선택
4. **Analyze page load** 클릭

### 9.2 PWA 감사 체크리스트

Lighthouse는 다음 기준을 확인합니다:

| 확인 항목 | 설명 |
|---|---|
| 설치 가능 | 유효한 매니페스트 + 서비스 워커 |
| PWA 최적화 | HTTPS, 혼합 콘텐츠 없음, theme-color 설정 |
| 빠르고 신뢰 가능 | 느린 3G에서 10초 이내 로드 |
| 오프라인 | 오프라인 시 200 반환 |
| HTTP → HTTPS 리디렉션 | 모든 HTTP 트래픽 리디렉션 |

### 9.3 Lighthouse CLI

```bash
# Lighthouse CLI 설치
npm install -g lighthouse

# 감사 실행
lighthouse https://example.com --output=html --output-path=./report.html

# PWA 전용 감사
lighthouse https://example.com --only-categories=pwa --output=json
```

### 9.4 프로그래밍 방식의 Lighthouse

```javascript
// lighthouse-audit.js
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');

async function runAudit(url) {
  const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });
  const options = {
    logLevel: 'info',
    output: 'json',
    port: chrome.port,
    onlyCategories: ['pwa']
  };

  const result = await lighthouse(url, options);
  const score = result.lhr.categories.pwa.score * 100;
  console.log(`PWA Score: ${score}/100`);

  await chrome.kill();
  return result.lhr;
}

runAudit('https://example.com');
```

---

## 10. 종합 예제: 완전한 PWA 예제

### 10.1 프로젝트 구조

```
my-pwa/
├── index.html
├── manifest.json
├── sw.js
├── offline.html
├── css/
│   └── style.css
├── js/
│   ├── app.js
│   └── db.js
└── icons/
    ├── icon-192.png
    └── icon-512.png
```

### 10.2 완전한 서비스 워커

```javascript
// sw.js — 프로덕션용 서비스 워커
const CACHE_VERSION = 'v2';
const STATIC_CACHE = `static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `dynamic-${CACHE_VERSION}`;
const OFFLINE_URL = '/offline.html';

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/offline.html',
  '/css/style.css',
  '/js/app.js',
  '/js/db.js',
  '/manifest.json',
  '/icons/icon-192.png'
];

// Install — 정적 자산 사전 캐싱
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Activate — 오래된 캐시 정리
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
          .map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

// Fetch — 전략 라우터
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // 비 GET 및 교차 출처(cross-origin) 건너뛰기
  if (request.method !== 'GET' || url.origin !== location.origin) return;

  // 탐색(navigation) — Network First
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(DYNAMIC_CACHE).then((c) => c.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request)
          .then((cached) => cached || caches.match(OFFLINE_URL)))
    );
    return;
  }

  // 정적 자산 — Cache First
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(
      caches.match(request).then((cached) => cached || fetch(request))
    );
    return;
  }

  // 기타 모든 것 — Stale-While-Revalidate
  event.respondWith(
    caches.match(request).then((cached) => {
      const networkFetch = fetch(request).then((response) => {
        caches.open(DYNAMIC_CACHE).then((c) => c.put(request, response.clone()));
        return response;
      });
      return cached || networkFetch;
    })
  );
});

// Push
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : { title: 'Update', body: 'New content available' };
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/icons/icon-192.png'
    })
  );
});

// Background Sync
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-data') {
    event.waitUntil(replayOutbox());
  }
});
```

---

## 11. 연습 문제(Practice Exercises)

### 연습 1: 기본 서비스 워커 (난이도: ⭐⭐)

다음을 수행하는 서비스 워커를 생성하세요:
1. 앱 셸(HTML, CSS, JS)을 사전 캐싱
2. 오프라인일 때 캐시된 콘텐츠 제공
3. 캐시되지 않은 탐색 요청에 대해 사용자 정의 오프라인 페이지 표시

### 연습 2: 다중 전략 라우터 (난이도: ⭐⭐⭐)

요청 유형에 따라 다른 캐싱 전략을 사용하는 fetch 핸들러를 구축하세요:
- 이미지와 폰트에는 **Cache First**
- HTML 페이지에는 **Network First**
- CSS와 JavaScript에는 **Stale-While-Revalidate**

### 연습 3: 푸시 알림 시스템 (난이도: ⭐⭐⭐)

완전한 푸시 알림 흐름을 구현하세요:
1. 알림 권한 요청
2. VAPID 키로 푸시 구독
3. 서비스 워커에서 푸시 이벤트 처리
4. 사용자가 알림을 거부할 수 있도록 구현

### 연습 4: 오프라인 폼 제출 (난이도: ⭐⭐⭐)

다음을 수행하는 메모 앱을 만드세요:
1. IndexedDB에 메모 저장
2. 온라인일 때 서버에 메모 동기화
3. Background Sync를 사용하여 실패한 제출 재시도
4. 사용자에게 동기화 상태 표시

### 연습 5: Workbox로 완전한 PWA 구축 (난이도: ⭐⭐⭐)

Workbox를 사용하여 기존 웹 앱을 PWA로 변환하세요:
1. manifest.json 생성
2. 앱 셸에 `workbox-precaching` 사용
3. 적절한 전략으로 런타임 캐싱 구성
4. 설치 프롬프트 추가
5. Lighthouse PWA 만점 달성

---

## 요약(Summary)

이 레슨에서 다룬 내용:

- **PWA 기본 개념**: 세 가지 핵심 요소(가능성, 신뢰성, 설치 가능성)와 웹 앱 매니페스트
- **서비스 워커 생명주기**: 등록, install, activate, fetch 이벤트
- **캐싱 전략**: Cache First, Network First, Stale-While-Revalidate 및 각 전략의 사용 시기
- **오프라인 기능**: 앱 셸 모델, 오프라인 폴백 페이지, 데이터를 위한 IndexedDB
- **푸시 알림**: 권한, 구독, 푸시 이벤트 처리
- **백그라운드 싱크**: 연결이 복원될 때까지 작업 지연
- **Workbox**: 선언적 라우팅을 갖춘 프로덕션용 캐싱
- **설치 가능성**: 기준, 사용자 정의 설치 프롬프트, Lighthouse 감사

서비스 워커와 PWA는 웹 개발의 패러다임 전환을 나타냅니다 -- 웹 앱이 오프라인에서 작동하고, 알림을 보내고, 앱 스토어 없이 홈 화면에 설치될 수 있습니다. 핵심은 각 리소스 유형에 맞는 올바른 캐싱 전략을 선택하고 Lighthouse로 철저히 테스트하는 것입니다.

---

**이전**: [Flask 기초](./16_Flask_Basics.md) | **다음**: [코어 웹 바이탈](./18_Core_Web_Vitals.md)
