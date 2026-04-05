# 17. Service Workers and Progressive Web Apps

**Previous**: [Flask Basics](./16_Flask_Basics.md) | **Next**: [Core Web Vitals](./18_Core_Web_Vitals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Progressive Web App concepts and create a valid `manifest.json`
2. Understand the Service Worker lifecycle: install, activate, and fetch events
3. Implement caching strategies including Cache First, Network First, and Stale-While-Revalidate
4. Build offline functionality for a web application
5. Send push notifications using the Push API
6. Use Background Sync to defer actions until connectivity is restored
7. Simplify Service Worker development with the Workbox library
8. Meet PWA installability criteria and pass a Lighthouse PWA audit

---

Progressive Web Apps bridge the gap between web pages and native applications. They load like regular web pages but offer capabilities traditionally reserved for native apps -- offline access, push notifications, and home screen installation. At the heart of every PWA is the **Service Worker**, a programmable network proxy that gives you fine-grained control over caching and network requests.

## 1. Progressive Web App Fundamentals

### 1.1 What Makes a PWA?

A PWA is not a specific technology; it is a set of best practices that, when combined, deliver an app-like experience in the browser.

> **Three PWA Pillars**
>
> 1. **Capable** -- Access device features (camera, geolocation, notifications)
> 2. **Reliable** -- Load instantly and work offline or on flaky networks
> 3. **Installable** -- Live on the home screen without an app store

### 1.2 PWA vs Native vs Traditional Web

| Feature | Traditional Web | PWA | Native App |
|---|---|---|---|
| Discoverability | Search engines | Search engines | App stores |
| Installation | None | Add to home screen | Store download |
| Offline support | None | Service Worker cache | Full |
| Push notifications | No | Yes | Yes |
| Device APIs | Limited | Growing | Full |
| Update mechanism | Instant (server) | SW update cycle | Store review |

### 1.3 The Web App Manifest

The `manifest.json` file tells the browser about your PWA and how it should behave when installed.

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

Link the manifest in your HTML `<head>`:

```html
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#2196F3">
<!-- iOS Safari fallback -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<link rel="apple-touch-icon" href="/icons/icon-192.png">
```

### 1.4 Display Modes

The `display` field controls how the browser chrome appears:

| Mode | Description |
|---|---|
| `fullscreen` | No browser UI at all (games) |
| `standalone` | Looks like a native app (no URL bar) |
| `minimal-ui` | Standalone + minimal navigation controls |
| `browser` | Standard browser tab |

```css
/* detect standalone mode in CSS */
@media (display-mode: standalone) {
  .install-banner {
    display: none;
  }
}
```

```javascript
// detect standalone mode in JavaScript
const isStandalone = window.matchMedia('(display-mode: standalone)').matches
                  || window.navigator.standalone; // iOS Safari
```

---

## 2. Service Worker Lifecycle

### 2.1 What is a Service Worker?

A Service Worker (SW) is a JavaScript file that runs in a separate thread from the main page. It acts as a **programmable network proxy**, intercepting every network request the page makes.

Key constraints:

- Runs on a **separate thread** (no DOM access)
- Requires **HTTPS** (except `localhost`)
- Fully **asynchronous** (no `localStorage`, no synchronous XHR)
- Has a **scope** defined by its file location

### 2.2 Registration

```javascript
// main.js — register the service worker
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

### 2.3 Lifecycle Events

```
┌──────────┐     ┌───────────┐     ┌───────────┐
│  Install  │────>│  Waiting   │────>│  Activate  │
└──────────┘     └───────────┘     └───────────┘
                                         │
                                         v
                                   ┌───────────┐
                                   │   Fetch    │  (intercepts requests)
                                   └───────────┘
```

**Install event** -- fired when the browser detects a new or updated SW file. Typically used to pre-cache essential assets.

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
  // skip waiting to activate immediately
  self.skipWaiting();
});
```

**Activate event** -- fired after installation when no other SW controls the page. Used to clean up old caches.

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
  // take control of all open clients immediately
  self.clients.claim();
});
```

**Fetch event** -- fired for every network request within the SW scope. This is where caching strategies live.

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

### 2.4 Updating a Service Worker

The browser checks for a new SW file byte-by-byte on every navigation. If the file has changed:

1. The new SW is **installed** alongside the old one
2. It enters a **waiting** state until all tabs using the old SW are closed
3. On next navigation, it **activates**

To force an immediate update:

```javascript
// in the install event
self.skipWaiting();

// in the activate event
self.clients.claim();
```

---

## 3. Caching Strategies

### 3.1 Cache First (Cache Falling Back to Network)

Best for **static assets** that rarely change (CSS, JS bundles, images).

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((cached) => {
        if (cached) {
          return cached;
        }
        return fetch(event.request).then((response) => {
          // clone because response can only be consumed once
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

### 3.2 Network First (Network Falling Back to Cache)

Best for **dynamic content** like API responses and HTML pages.

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

Serves cached content **immediately**, then updates the cache in the background. Best for resources where freshness is preferred but speed is critical.

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

### 3.4 Cache Only and Network Only

```javascript
// Cache Only — for versioned assets guaranteed to be in cache
event.respondWith(caches.match(event.request));

// Network Only — for non-GET requests or real-time data
event.respondWith(fetch(event.request));
```

### 3.5 Strategy Selection Guide

| Strategy | Use Case | Offline? | Freshness |
|---|---|---|---|
| Cache First | Static assets, icons, fonts | Yes | Low |
| Network First | HTML pages, API data | Yes (fallback) | High |
| Stale-While-Revalidate | User avatars, article lists | Yes | Medium |
| Cache Only | Pre-cached app shell | Yes | None |
| Network Only | Analytics, POST requests | No | Real-time |

---

## 4. Building Offline Functionality

### 4.1 The App Shell Model

The app shell is the minimal HTML, CSS, and JavaScript needed to render the UI skeleton. It is cached on first visit and loaded instantly on subsequent visits.

```
┌──────────────────────────────────┐
│          App Shell (cached)       │
│  ┌────────┐  ┌─────────────────┐ │
│  │ Header │  │   Navigation    │ │
│  └────────┘  └─────────────────┘ │
│  ┌──────────────────────────────┐│
│  │                              ││
│  │     Dynamic Content          ││
│  │     (fetched from network)   ││
│  │                              ││
│  └──────────────────────────────┘│
│  ┌──────────────────────────────┐│
│  │         Footer               ││
│  └──────────────────────────────┘│
└──────────────────────────────────┘
```

### 4.2 Offline Fallback Page

```javascript
// sw.js — serve a custom offline page
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

### 4.3 Caching Dynamic API Responses

```javascript
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (url.pathname.startsWith('/api/')) {
    // Network First for API calls
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
    // Cache First for static assets
    event.respondWith(
      caches.match(event.request)
        .then((cached) => cached || fetch(event.request))
    );
  }
});
```

### 4.4 IndexedDB for Offline Data

For structured data that needs querying, use IndexedDB instead of the Cache API.

```javascript
// db.js — simple IndexedDB wrapper
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

## 5. Push Notifications

### 5.1 Push API Overview

Push notifications involve three parties:

```
┌────────┐         ┌──────────────┐         ┌──────────┐
│ Server │ ──push──> Push Service  │ ──push──> Service   │
│        │         │ (FCM/APNs)   │         │ Worker   │
└────────┘         └──────────────┘         └──────────┘
                                                 │
                                                 v
                                           ┌──────────┐
                                           │ Browser  │
                                           │ (user)   │
                                           └──────────┘
```

### 5.2 Requesting Permission

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

### 5.3 Push Subscription

```javascript
// main.js
async function subscribeToPush() {
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(
      'BEl62iUYgUiv...'  // VAPID public key
    )
  });

  // Send subscription to your server
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

### 5.4 Handling Push Events in the Service Worker

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

### 5.5 Server-Side Push (Node.js)

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
      // subscription expired — remove from database
      console.log('Subscription expired, removing');
    }
  }
}
```

---

## 6. Background Sync

### 6.1 What is Background Sync?

Background Sync lets you defer actions until the user has stable connectivity. For example, if a user submits a form while offline, the request is queued and replayed when the connection is restored.

### 6.2 Registering a Sync Event

```javascript
// main.js
async function saveFormOffline(formData) {
  // store in IndexedDB
  const db = await initDB();
  const tx = db.transaction('outbox', 'readwrite');
  tx.objectStore('outbox').add({
    url: '/api/notes',
    method: 'POST',
    body: Object.fromEntries(formData),
    timestamp: Date.now()
  });

  // register sync
  const registration = await navigator.serviceWorker.ready;
  await registration.sync.register('sync-notes');
}
```

### 6.3 Handling the Sync Event

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
          // remove from outbox on success
          const deleteTx = db.transaction('outbox', 'readwrite');
          deleteTx.objectStore('outbox').delete(item.id);
        } catch (err) {
          // sync will retry automatically
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

## 7. Workbox Library

### 7.1 Why Workbox?

Writing Service Workers by hand is error-prone. **Workbox** (by Google) provides tested, production-ready modules for common SW patterns.

### 7.2 Installation

```bash
npm install workbox-cli --save-dev
```

Or use the CDN for quick prototyping:

```javascript
importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.0.0/workbox-sw.js');
```

### 7.3 Workbox Caching Strategies

```javascript
// sw.js with Workbox
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';
import { precacheAndRoute } from 'workbox-precaching';

// Precache app shell (generated by build tool)
precacheAndRoute(self.__WB_MANIFEST);

// Cache First for images
registerRoute(
  ({ request }) => request.destination === 'image',
  new CacheFirst({
    cacheName: 'images',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] }),
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 30 * 24 * 60 * 60  // 30 days
      })
    ]
  })
);

// Network First for HTML pages
registerRoute(
  ({ request }) => request.mode === 'navigate',
  new NetworkFirst({
    cacheName: 'pages',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] })
    ]
  })
);

// Stale-While-Revalidate for CSS and JS
registerRoute(
  ({ request }) =>
    request.destination === 'style' || request.destination === 'script',
  new StaleWhileRevalidate({
    cacheName: 'static-resources'
  })
);
```

### 7.4 Workbox Build Integration (Webpack)

```javascript
// webpack.config.js
const { InjectManifest } = require('workbox-webpack-plugin');

module.exports = {
  // ... other config
  plugins: [
    new InjectManifest({
      swSrc: './src/sw.js',
      swDest: 'sw.js',
      maximumFileSizeToCacheInBytes: 5 * 1024 * 1024
    })
  ]
};
```

### 7.5 Workbox CLI Wizard

```bash
npx workbox wizard
# Answers a few questions and generates workbox-config.js

npx workbox generateSW workbox-config.js
# Generates a complete service worker
```

---

## 8. PWA Installability

### 8.1 Installability Criteria

For Chrome to show the install prompt, your app must:

1. Serve over **HTTPS**
2. Have a valid **Web App Manifest** with `name`, `icons` (192px and 512px), `start_url`, `display`
3. Register a **Service Worker** with a `fetch` event handler
4. The user must have **engaged** with the site (visited at least twice, with 30 seconds between visits -- though this varies by browser)

### 8.2 Handling the Install Prompt

```javascript
// main.js
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (event) => {
  // prevent automatic prompt
  event.preventDefault();
  deferredPrompt = event;
  // show custom install button
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

### 8.3 Install Button UI

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

## 9. Lighthouse PWA Audit

### 9.1 Running a Lighthouse Audit

Lighthouse is built into Chrome DevTools. To run a PWA audit:

1. Open Chrome DevTools (F12)
2. Go to the **Lighthouse** tab
3. Select **Progressive Web App** category
4. Click **Analyze page load**

### 9.2 PWA Audit Checklist

Lighthouse checks these criteria:

| Check | Description |
|---|---|
| Installable | Valid manifest + service worker |
| PWA Optimized | HTTPS, no mixed content, sets theme-color |
| Fast & Reliable | Page loads under 10 seconds on slow 3G |
| Offline | Returns 200 when offline |
| Redirects HTTP → HTTPS | All HTTP traffic redirected |

### 9.3 Lighthouse CLI

```bash
# install Lighthouse CLI
npm install -g lighthouse

# run audit
lighthouse https://example.com --output=html --output-path=./report.html

# PWA-only audit
lighthouse https://example.com --only-categories=pwa --output=json
```

### 9.4 Programmatic Lighthouse

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

## 10. Putting It All Together: Complete PWA Example

### 10.1 Project Structure

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

### 10.2 Complete Service Worker

```javascript
// sw.js — production-ready service worker
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

// Install — precache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Activate — clean old caches
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

// Fetch — strategy router
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // skip non-GET and cross-origin
  if (request.method !== 'GET' || url.origin !== location.origin) return;

  // navigation — Network First
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

  // static assets — Cache First
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(
      caches.match(request).then((cached) => cached || fetch(request))
    );
    return;
  }

  // everything else — Stale-While-Revalidate
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

## 11. Practice Exercises

### Exercise 1: Basic Service Worker (Difficulty: ⭐⭐)

Create a Service Worker that:
1. Pre-caches an app shell (HTML, CSS, JS)
2. Serves cached content when offline
3. Shows a custom offline page for uncached navigation requests

### Exercise 2: Multi-Strategy Router (Difficulty: ⭐⭐⭐)

Build a fetch handler that uses different caching strategies based on request type:
- **Cache First** for images and fonts
- **Network First** for HTML pages
- **Stale-While-Revalidate** for CSS and JavaScript

### Exercise 3: Push Notification System (Difficulty: ⭐⭐⭐)

Implement a complete push notification flow:
1. Request notification permission
2. Subscribe to push with VAPID keys
3. Handle push events in the Service Worker
4. Allow users to opt out of notifications

### Exercise 4: Offline Form Submission (Difficulty: ⭐⭐⭐)

Create a note-taking app that:
1. Stores notes in IndexedDB
2. Syncs notes to a server when online
3. Uses Background Sync to replay failed submissions
4. Shows sync status to the user

### Exercise 5: Full PWA with Workbox (Difficulty: ⭐⭐⭐)

Convert an existing web app into a PWA using Workbox:
1. Generate a manifest.json
2. Use `workbox-precaching` for the app shell
3. Configure runtime caching with appropriate strategies
4. Add an install prompt
5. Achieve a perfect Lighthouse PWA score

---

## Summary

In this lesson, we covered:

- **PWA fundamentals**: The three pillars (capable, reliable, installable) and the web app manifest
- **Service Worker lifecycle**: Registration, install, activate, and fetch events
- **Caching strategies**: Cache First, Network First, Stale-While-Revalidate, and when to use each
- **Offline functionality**: App shell model, offline fallback pages, and IndexedDB for data
- **Push notifications**: Permission, subscription, and handling push events
- **Background Sync**: Deferring actions until connectivity is restored
- **Workbox**: Production-ready caching with declarative routing
- **Installability**: Criteria, custom install prompts, and Lighthouse audits

Service Workers and PWAs represent a paradigm shift in web development -- your web app can now work offline, send notifications, and install on the home screen, all without an app store. The key is choosing the right caching strategy for each resource type and testing thoroughly with Lighthouse.

---

**Previous**: [Flask Basics](./16_Flask_Basics.md) | **Next**: [Core Web Vitals](./18_Core_Web_Vitals.md)
