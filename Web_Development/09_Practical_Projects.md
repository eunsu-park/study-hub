# 실전 프로젝트

## 개요

이 문서에서는 앞서 배운 HTML, CSS, JavaScript를 종합하여 실제 동작하는 웹 애플리케이션을 만들어봅니다.

**선수 지식**: 이전 모든 챕터

---

## 목차

1. [프로젝트 1: Todo 앱](#프로젝트-1-todo-앱)
2. [프로젝트 2: 날씨 앱](#프로젝트-2-날씨-앱)
3. [프로젝트 3: 이미지 갤러리](#프로젝트-3-이미지-갤러리)
4. [다음 단계](#다음-단계)

---

## 프로젝트 1: Todo 앱

로컬 스토리지를 활용한 할 일 관리 애플리케이션입니다.

### 기능

- 할 일 추가/삭제/완료 처리
- 로컬 스토리지에 저장
- 필터링 (전체/진행중/완료)
- 반응형 디자인

### 파일 구조

```
todo-app/
├── index.html
├── css/
│   └── style.css
└── js/
    └── app.js
```

### index.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Todo App</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Todo List</h1>
            <p class="date" id="currentDate"></p>
        </header>

        <form id="todoForm" class="todo-form">
            <input
                type="text"
                id="todoInput"
                class="todo-input"
                placeholder="할 일을 입력하세요"
                required
                autocomplete="off"
            >
            <button type="submit" class="btn btn-primary">추가</button>
        </form>

        <div class="filters">
            <button class="filter-btn active" data-filter="all">전체</button>
            <button class="filter-btn" data-filter="active">진행중</button>
            <button class="filter-btn" data-filter="completed">완료</button>
        </div>

        <ul id="todoList" class="todo-list"></ul>

        <footer class="todo-footer">
            <span id="itemCount">0개의 항목</span>
            <button id="clearCompleted" class="btn btn-text">완료 항목 삭제</button>
        </footer>
    </div>

    <script src="js/app.js"></script>
</body>
</html>
```

### css/style.css

```css
/* 기본 스타일 */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 2rem 1rem;
}

.container {
    max-width: 500px;
    margin: 0 auto;
    background: white;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    overflow: hidden;
}

/* 헤더 */
header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    text-align: center;
}

header h1 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.date {
    opacity: 0.8;
    font-size: 0.9rem;
}

/* 폼 */
.todo-form {
    display: flex;
    padding: 1.5rem;
    gap: 0.5rem;
    border-bottom: 1px solid #eee;
}

.todo-input {
    flex: 1;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    border: 2px solid #eee;
    border-radius: 8px;
    outline: none;
    transition: border-color 0.2s;
}

.todo-input:focus {
    border-color: #667eea;
}

/* 버튼 */
.btn {
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary {
    background: #667eea;
    color: white;
}

.btn-primary:hover {
    background: #5a6fd6;
}

.btn-text {
    background: none;
    color: #999;
    padding: 0.5rem;
}

.btn-text:hover {
    color: #e74c3c;
}

/* 필터 */
.filters {
    display: flex;
    padding: 1rem 1.5rem;
    gap: 0.5rem;
    border-bottom: 1px solid #eee;
}

.filter-btn {
    flex: 1;
    padding: 0.5rem;
    background: none;
    border: 2px solid #eee;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.filter-btn:hover {
    border-color: #667eea;
}

.filter-btn.active {
    background: #667eea;
    border-color: #667eea;
    color: white;
}

/* Todo 리스트 */
.todo-list {
    list-style: none;
    max-height: 400px;
    overflow-y: auto;
}

.todo-item {
    display: flex;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #eee;
    gap: 1rem;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.todo-item.completed .todo-text {
    text-decoration: line-through;
    color: #999;
}

.todo-checkbox {
    width: 22px;
    height: 22px;
    cursor: pointer;
    accent-color: #667eea;
}

.todo-text {
    flex: 1;
    font-size: 1rem;
    word-break: break-word;
}

.todo-delete {
    background: none;
    border: none;
    color: #999;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0.25rem;
    opacity: 0;
    transition: all 0.2s;
}

.todo-item:hover .todo-delete {
    opacity: 1;
}

.todo-delete:hover {
    color: #e74c3c;
}

/* 푸터 */
.todo-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    color: #999;
    font-size: 0.9rem;
}

/* 빈 상태 */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: #999;
}

.empty-state::before {
    content: '📝';
    display: block;
    font-size: 3rem;
    margin-bottom: 1rem;
}

/* 반응형 */
@media (max-width: 480px) {
    body {
        padding: 0;
    }

    .container {
        border-radius: 0;
        min-height: 100vh;
    }

    .todo-form {
        flex-direction: column;
    }

    .btn-primary {
        width: 100%;
    }
}
```

### js/app.js

```javascript
// Todo 앱 클래스
class TodoApp {
    constructor() {
        // DOM 요소
        this.form = document.getElementById('todoForm');
        this.input = document.getElementById('todoInput');
        this.list = document.getElementById('todoList');
        this.itemCount = document.getElementById('itemCount');
        this.clearBtn = document.getElementById('clearCompleted');
        this.filterBtns = document.querySelectorAll('.filter-btn');

        // 상태
        this.todos = this.loadTodos();
        this.filter = 'all';

        // 초기화
        this.init();
    }

    init() {
        // 날짜 표시
        this.displayDate();

        // 이벤트 리스너
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.list.addEventListener('click', (e) => this.handleListClick(e));
        this.clearBtn.addEventListener('click', () => this.clearCompleted());

        this.filterBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.handleFilter(e));
        });

        // 초기 렌더링
        this.render();
    }

    displayDate() {
        const dateEl = document.getElementById('currentDate');
        const options = {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        };
        dateEl.textContent = new Date().toLocaleDateString('ko-KR', options);
    }

    // 로컬 스토리지
    loadTodos() {
        const data = localStorage.getItem('todos');
        return data ? JSON.parse(data) : [];
    }

    saveTodos() {
        localStorage.setItem('todos', JSON.stringify(this.todos));
    }

    // Todo CRUD
    addTodo(text) {
        const todo = {
            id: Date.now(),
            text: text.trim(),
            completed: false,
            createdAt: new Date().toISOString()
        };
        this.todos.unshift(todo);
        this.saveTodos();
        this.render();
    }

    toggleTodo(id) {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
            this.saveTodos();
            this.render();
        }
    }

    deleteTodo(id) {
        this.todos = this.todos.filter(t => t.id !== id);
        this.saveTodos();
        this.render();
    }

    clearCompleted() {
        this.todos = this.todos.filter(t => !t.completed);
        this.saveTodos();
        this.render();
    }

    // 필터링
    getFilteredTodos() {
        switch (this.filter) {
            case 'active':
                return this.todos.filter(t => !t.completed);
            case 'completed':
                return this.todos.filter(t => t.completed);
            default:
                return this.todos;
        }
    }

    // 이벤트 핸들러
    handleSubmit(e) {
        e.preventDefault();
        const text = this.input.value.trim();
        if (text) {
            this.addTodo(text);
            this.input.value = '';
            this.input.focus();
        }
    }

    handleListClick(e) {
        const item = e.target.closest('.todo-item');
        if (!item) return;

        const id = parseInt(item.dataset.id);

        if (e.target.matches('.todo-checkbox')) {
            this.toggleTodo(id);
        } else if (e.target.matches('.todo-delete')) {
            this.deleteTodo(id);
        }
    }

    handleFilter(e) {
        this.filterBtns.forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        this.filter = e.target.dataset.filter;
        this.render();
    }

    // 렌더링
    render() {
        const filteredTodos = this.getFilteredTodos();

        if (filteredTodos.length === 0) {
            this.list.innerHTML = `
                <li class="empty-state">
                    ${this.filter === 'all' ? '할 일을 추가해보세요!' :
                      this.filter === 'active' ? '진행 중인 항목이 없습니다' :
                      '완료된 항목이 없습니다'}
                </li>
            `;
        } else {
            this.list.innerHTML = filteredTodos.map(todo => `
                <li class="todo-item ${todo.completed ? 'completed' : ''}" data-id="${todo.id}">
                    <input
                        type="checkbox"
                        class="todo-checkbox"
                        ${todo.completed ? 'checked' : ''}
                    >
                    <span class="todo-text">${this.escapeHtml(todo.text)}</span>
                    <button class="todo-delete" aria-label="삭제">×</button>
                </li>
            `).join('');
        }

        // 카운트 업데이트
        const activeCount = this.todos.filter(t => !t.completed).length;
        this.itemCount.textContent = `${activeCount}개의 항목`;

        // 완료 삭제 버튼 표시/숨김
        const hasCompleted = this.todos.some(t => t.completed);
        this.clearBtn.style.display = hasCompleted ? 'block' : 'none';
    }

    // XSS 방지
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// 앱 시작
document.addEventListener('DOMContentLoaded', () => {
    new TodoApp();
});
```

---

## 프로젝트 2: 날씨 앱

외부 API를 활용한 날씨 정보 조회 애플리케이션입니다.

### 기능

- 도시명으로 날씨 검색
- 현재 위치 날씨 조회
- 날씨 아이콘 및 상세 정보 표시
- 로딩 상태 및 에러 처리

### 준비사항

[OpenWeatherMap](https://openweathermap.org/api)에서 무료 API 키를 발급받으세요.

### 파일 구조

```
weather-app/
├── index.html
├── css/
│   └── style.css
└── js/
    └── app.js
```

### index.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather App</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="app">
        <div class="search-box">
            <form id="searchForm">
                <input
                    type="text"
                    id="cityInput"
                    placeholder="도시명을 입력하세요"
                    autocomplete="off"
                >
                <button type="submit">검색</button>
            </form>
            <button id="locationBtn" class="location-btn" title="현재 위치">
                📍
            </button>
        </div>

        <div id="loading" class="loading hidden">
            <div class="spinner"></div>
            <p>날씨 정보를 가져오는 중...</p>
        </div>

        <div id="error" class="error hidden">
            <p id="errorMessage"></p>
            <button id="retryBtn">다시 시도</button>
        </div>

        <div id="weather" class="weather-card hidden">
            <div class="weather-main">
                <img id="weatherIcon" src="" alt="날씨 아이콘">
                <div class="temperature">
                    <span id="temp">--</span>
                    <span class="unit">°C</span>
                </div>
            </div>

            <h2 id="cityName">--</h2>
            <p id="description">--</p>

            <div class="weather-details">
                <div class="detail">
                    <span class="label">체감</span>
                    <span id="feelsLike">--°C</span>
                </div>
                <div class="detail">
                    <span class="label">습도</span>
                    <span id="humidity">--%</span>
                </div>
                <div class="detail">
                    <span class="label">풍속</span>
                    <span id="wind">--m/s</span>
                </div>
                <div class="detail">
                    <span class="label">구름</span>
                    <span id="clouds">--%</span>
                </div>
            </div>
        </div>
    </div>

    <script src="js/app.js"></script>
</body>
</html>
```

### css/style.css

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    padding: 1rem;
}

.app {
    width: 100%;
    max-width: 400px;
}

/* 검색 박스 */
.search-box {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}

.search-box form {
    flex: 1;
    display: flex;
    background: white;
    border-radius: 50px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.search-box input {
    flex: 1;
    padding: 1rem 1.5rem;
    border: none;
    outline: none;
    font-size: 1rem;
}

.search-box button[type="submit"] {
    padding: 1rem 1.5rem;
    background: #0984e3;
    color: white;
    border: none;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
}

.search-box button[type="submit"]:hover {
    background: #0874c9;
}

.location-btn {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    border: none;
    background: white;
    font-size: 1.5rem;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
}

.location-btn:hover {
    transform: scale(1.1);
}

/* 날씨 카드 */
.weather-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.weather-main {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.weather-main img {
    width: 100px;
    height: 100px;
}

.temperature {
    display: flex;
    align-items: flex-start;
}

.temperature #temp {
    font-size: 4rem;
    font-weight: 300;
    line-height: 1;
}

.temperature .unit {
    font-size: 1.5rem;
    margin-top: 0.5rem;
}

.weather-card h2 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: #333;
}

.weather-card #description {
    color: #666;
    text-transform: capitalize;
    margin-bottom: 1.5rem;
}

.weather-details {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    padding-top: 1.5rem;
    border-top: 1px solid #eee;
}

.detail {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.detail .label {
    font-size: 0.8rem;
    color: #999;
}

.detail span:last-child {
    font-size: 1.1rem;
    font-weight: 500;
    color: #333;
}

/* 로딩 */
.loading {
    text-align: center;
    padding: 3rem;
    color: white;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 1rem;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* 에러 */
.error {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.error p {
    color: #e74c3c;
    margin-bottom: 1rem;
}

.error button {
    padding: 0.75rem 1.5rem;
    background: #e74c3c;
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-size: 1rem;
}

/* 유틸리티 */
.hidden {
    display: none !important;
}

/* 반응형 */
@media (max-width: 480px) {
    .weather-main img {
        width: 80px;
        height: 80px;
    }

    .temperature #temp {
        font-size: 3rem;
    }
}
```

### js/app.js

```javascript
// API 키 (실제 키로 교체하세요)
const API_KEY = 'YOUR_API_KEY_HERE';
const BASE_URL = 'https://api.openweathermap.org/data/2.5/weather';

class WeatherApp {
    constructor() {
        this.form = document.getElementById('searchForm');
        this.input = document.getElementById('cityInput');
        this.locationBtn = document.getElementById('locationBtn');
        this.loadingEl = document.getElementById('loading');
        this.errorEl = document.getElementById('error');
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');
        this.weatherEl = document.getElementById('weather');

        this.lastSearch = null;

        this.init();
    }

    init() {
        this.form.addEventListener('submit', (e) => this.handleSearch(e));
        this.locationBtn.addEventListener('click', () => this.getCurrentLocation());
        this.retryBtn.addEventListener('click', () => this.retry());

        // 저장된 마지막 검색 복원
        const saved = localStorage.getItem('lastCity');
        if (saved) {
            this.fetchWeather(saved);
        }
    }

    async handleSearch(e) {
        e.preventDefault();
        const city = this.input.value.trim();
        if (city) {
            this.lastSearch = { type: 'city', value: city };
            await this.fetchWeather(city);
        }
    }

    getCurrentLocation() {
        if (!navigator.geolocation) {
            this.showError('위치 정보를 지원하지 않는 브라우저입니다.');
            return;
        }

        this.showLoading();

        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const { latitude, longitude } = position.coords;
                this.lastSearch = { type: 'coords', value: { lat: latitude, lon: longitude } };
                await this.fetchWeatherByCoords(latitude, longitude);
            },
            (error) => {
                let message = '위치를 가져올 수 없습니다.';
                if (error.code === error.PERMISSION_DENIED) {
                    message = '위치 접근 권한이 거부되었습니다.';
                }
                this.showError(message);
            }
        );
    }

    async fetchWeather(city) {
        this.showLoading();

        try {
            const url = `${BASE_URL}?q=${encodeURIComponent(city)}&appid=${API_KEY}&units=metric&lang=kr`;
            const response = await fetch(url);

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error('도시를 찾을 수 없습니다.');
                }
                throw new Error('날씨 정보를 가져올 수 없습니다.');
            }

            const data = await response.json();
            this.displayWeather(data);
            localStorage.setItem('lastCity', city);
        } catch (error) {
            this.showError(error.message);
        }
    }

    async fetchWeatherByCoords(lat, lon) {
        try {
            const url = `${BASE_URL}?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric&lang=kr`;
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error('날씨 정보를 가져올 수 없습니다.');
            }

            const data = await response.json();
            this.displayWeather(data);
        } catch (error) {
            this.showError(error.message);
        }
    }

    displayWeather(data) {
        document.getElementById('cityName').textContent = data.name;
        document.getElementById('temp').textContent = Math.round(data.main.temp);
        document.getElementById('description').textContent = data.weather[0].description;
        document.getElementById('feelsLike').textContent = `${Math.round(data.main.feels_like)}°C`;
        document.getElementById('humidity').textContent = `${data.main.humidity}%`;
        document.getElementById('wind').textContent = `${data.wind.speed}m/s`;
        document.getElementById('clouds').textContent = `${data.clouds.all}%`;

        const iconCode = data.weather[0].icon;
        document.getElementById('weatherIcon').src =
            `https://openweathermap.org/img/wn/${iconCode}@2x.png`;

        this.hideLoading();
        this.hideError();
        this.weatherEl.classList.remove('hidden');
    }

    retry() {
        if (this.lastSearch) {
            if (this.lastSearch.type === 'city') {
                this.fetchWeather(this.lastSearch.value);
            } else {
                const { lat, lon } = this.lastSearch.value;
                this.fetchWeatherByCoords(lat, lon);
            }
        }
    }

    showLoading() {
        this.loadingEl.classList.remove('hidden');
        this.weatherEl.classList.add('hidden');
        this.errorEl.classList.add('hidden');
    }

    hideLoading() {
        this.loadingEl.classList.add('hidden');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorEl.classList.remove('hidden');
        this.loadingEl.classList.add('hidden');
        this.weatherEl.classList.add('hidden');
    }

    hideError() {
        this.errorEl.classList.add('hidden');
    }
}

// 앱 시작
document.addEventListener('DOMContentLoaded', () => {
    new WeatherApp();
});
```

---

## 프로젝트 3: 이미지 갤러리

무한 스크롤과 라이트박스 기능이 있는 이미지 갤러리입니다.

### 기능

- Unsplash API로 이미지 로드
- 무한 스크롤
- 라이트박스 (클릭 시 확대)
- 반응형 그리드

### 준비사항

[Unsplash](https://unsplash.com/developers)에서 API 키를 발급받으세요.

### 파일 구조

```
gallery-app/
├── index.html
├── css/
│   └── style.css
└── js/
    └── app.js
```

### index.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Gallery</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <header class="header">
        <h1>Image Gallery</h1>
        <form id="searchForm" class="search-form">
            <input
                type="text"
                id="searchInput"
                placeholder="이미지 검색..."
                autocomplete="off"
            >
            <button type="submit">검색</button>
        </form>
    </header>

    <main>
        <div id="gallery" class="gallery"></div>
        <div id="loading" class="loading">
            <div class="spinner"></div>
        </div>
        <div id="sentinel" class="sentinel"></div>
    </main>

    <!-- 라이트박스 -->
    <div id="lightbox" class="lightbox hidden">
        <button class="lightbox-close">&times;</button>
        <button class="lightbox-prev">&lt;</button>
        <button class="lightbox-next">&gt;</button>
        <div class="lightbox-content">
            <img id="lightboxImage" src="" alt="">
            <div class="lightbox-info">
                <p id="lightboxAuthor"></p>
                <a id="lightboxLink" href="" target="_blank">Unsplash에서 보기</a>
            </div>
        </div>
    </div>

    <script src="js/app.js"></script>
</body>
</html>
```

### css/style.css

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    min-height: 100vh;
}

/* 헤더 */
.header {
    background: white;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    position: sticky;
    top: 0;
    z-index: 100;
}

.header h1 {
    text-align: center;
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

.search-form {
    display: flex;
    max-width: 500px;
    margin: 0 auto;
}

.search-form input {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 2px solid #ddd;
    border-right: none;
    border-radius: 8px 0 0 8px;
    font-size: 1rem;
    outline: none;
    transition: border-color 0.2s;
}

.search-form input:focus {
    border-color: #333;
}

.search-form button {
    padding: 0.75rem 1.5rem;
    background: #333;
    color: white;
    border: none;
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
}

.search-form button:hover {
    background: #555;
}

/* 갤러리 */
main {
    padding: 1.5rem;
    max-width: 1400px;
    margin: 0 auto;
}

.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.5rem;
}

.gallery-item {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    cursor: pointer;
    background: #ddd;
    aspect-ratio: 4/3;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.gallery-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
}

.gallery-item:hover img {
    transform: scale(1.05);
}

.gallery-item .overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.7));
    color: white;
    opacity: 0;
    transition: opacity 0.3s;
}

.gallery-item:hover .overlay {
    opacity: 1;
}

.overlay .author {
    font-size: 0.9rem;
}

/* 로딩 */
.loading {
    display: flex;
    justify-content: center;
    padding: 2rem;
}

.loading.hidden {
    display: none;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #ddd;
    border-top-color: #333;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.sentinel {
    height: 10px;
}

/* 라이트박스 */
.lightbox {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 1rem;
}

.lightbox.hidden {
    display: none;
}

.lightbox-content {
    max-width: 90vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.lightbox-content img {
    max-width: 100%;
    max-height: 80vh;
    object-fit: contain;
    border-radius: 8px;
}

.lightbox-info {
    margin-top: 1rem;
    text-align: center;
    color: white;
}

.lightbox-info a {
    color: #74b9ff;
    text-decoration: none;
}

.lightbox-close,
.lightbox-prev,
.lightbox-next {
    position: absolute;
    background: none;
    border: none;
    color: white;
    font-size: 2rem;
    cursor: pointer;
    padding: 1rem;
    transition: opacity 0.2s;
}

.lightbox-close:hover,
.lightbox-prev:hover,
.lightbox-next:hover {
    opacity: 0.7;
}

.lightbox-close {
    top: 0;
    right: 0;
}

.lightbox-prev {
    left: 0;
    top: 50%;
    transform: translateY(-50%);
}

.lightbox-next {
    right: 0;
    top: 50%;
    transform: translateY(-50%);
}

/* 반응형 */
@media (max-width: 600px) {
    .gallery {
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.5rem;
    }

    .gallery-item {
        border-radius: 8px;
    }
}
```

### js/app.js

```javascript
// API 키 (실제 키로 교체하세요)
const ACCESS_KEY = 'YOUR_UNSPLASH_ACCESS_KEY';
const BASE_URL = 'https://api.unsplash.com';

class GalleryApp {
    constructor() {
        this.gallery = document.getElementById('gallery');
        this.loading = document.getElementById('loading');
        this.searchForm = document.getElementById('searchForm');
        this.searchInput = document.getElementById('searchInput');
        this.lightbox = document.getElementById('lightbox');
        this.lightboxImage = document.getElementById('lightboxImage');
        this.lightboxAuthor = document.getElementById('lightboxAuthor');
        this.lightboxLink = document.getElementById('lightboxLink');

        this.images = [];
        this.page = 1;
        this.query = '';
        this.isLoading = false;
        this.currentIndex = 0;

        this.init();
    }

    init() {
        // 검색
        this.searchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.search(this.searchInput.value.trim());
        });

        // 무한 스크롤
        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && !this.isLoading) {
                this.loadImages();
            }
        });
        observer.observe(document.getElementById('sentinel'));

        // 갤러리 클릭 (이벤트 위임)
        this.gallery.addEventListener('click', (e) => {
            const item = e.target.closest('.gallery-item');
            if (item) {
                const index = parseInt(item.dataset.index);
                this.openLightbox(index);
            }
        });

        // 라이트박스 컨트롤
        this.lightbox.querySelector('.lightbox-close').addEventListener('click', () => {
            this.closeLightbox();
        });

        this.lightbox.querySelector('.lightbox-prev').addEventListener('click', () => {
            this.prevImage();
        });

        this.lightbox.querySelector('.lightbox-next').addEventListener('click', () => {
            this.nextImage();
        });

        // 키보드 네비게이션
        document.addEventListener('keydown', (e) => {
            if (this.lightbox.classList.contains('hidden')) return;

            switch (e.key) {
                case 'Escape':
                    this.closeLightbox();
                    break;
                case 'ArrowLeft':
                    this.prevImage();
                    break;
                case 'ArrowRight':
                    this.nextImage();
                    break;
            }
        });

        // 라이트박스 배경 클릭으로 닫기
        this.lightbox.addEventListener('click', (e) => {
            if (e.target === this.lightbox) {
                this.closeLightbox();
            }
        });

        // 초기 로드
        this.loadImages();
    }

    async loadImages() {
        if (this.isLoading) return;

        this.isLoading = true;
        this.loading.classList.remove('hidden');

        try {
            let url;
            if (this.query) {
                url = `${BASE_URL}/search/photos?query=${encodeURIComponent(this.query)}&page=${this.page}&per_page=20&client_id=${ACCESS_KEY}`;
            } else {
                url = `${BASE_URL}/photos?page=${this.page}&per_page=20&client_id=${ACCESS_KEY}`;
            }

            const response = await fetch(url);
            if (!response.ok) throw new Error('이미지를 불러올 수 없습니다.');

            const data = await response.json();
            const photos = this.query ? data.results : data;

            if (photos.length === 0) {
                return;
            }

            this.appendImages(photos);
            this.page++;
        } catch (error) {
            console.error(error);
        } finally {
            this.isLoading = false;
            this.loading.classList.add('hidden');
        }
    }

    appendImages(photos) {
        const startIndex = this.images.length;

        photos.forEach((photo, i) => {
            this.images.push(photo);

            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.dataset.index = startIndex + i;

            item.innerHTML = `
                <img
                    src="${photo.urls.small}"
                    alt="${photo.alt_description || '이미지'}"
                    loading="lazy"
                >
                <div class="overlay">
                    <p class="author">📷 ${photo.user.name}</p>
                </div>
            `;

            this.gallery.appendChild(item);
        });
    }

    search(query) {
        this.query = query;
        this.page = 1;
        this.images = [];
        this.gallery.innerHTML = '';
        this.loadImages();
    }

    openLightbox(index) {
        this.currentIndex = index;
        this.updateLightbox();
        this.lightbox.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    closeLightbox() {
        this.lightbox.classList.add('hidden');
        document.body.style.overflow = '';
    }

    prevImage() {
        this.currentIndex = (this.currentIndex - 1 + this.images.length) % this.images.length;
        this.updateLightbox();
    }

    nextImage() {
        this.currentIndex = (this.currentIndex + 1) % this.images.length;
        this.updateLightbox();
    }

    updateLightbox() {
        const image = this.images[this.currentIndex];
        this.lightboxImage.src = image.urls.regular;
        this.lightboxAuthor.textContent = `📷 ${image.user.name}`;
        this.lightboxLink.href = image.links.html;
    }
}

// 앱 시작
document.addEventListener('DOMContentLoaded', () => {
    new GalleryApp();
});
```

---

## 다음 단계

이 프로젝트들을 완성한 후:

### 추가 학습

1. **프레임워크 학습**
   - React, Vue, Svelte 등

2. **빌드 도구**
   - Vite, Webpack, Parcel

3. **CSS 프레임워크**
   - Tailwind CSS, Bootstrap

4. **타입스크립트**
   - 정적 타입 검사

5. **테스팅**
   - Jest, Vitest, Cypress

### 추천 프로젝트 아이디어

- 블로그/포트폴리오 사이트
- 실시간 채팅 앱 (WebSocket)
- 칸반 보드 (드래그 앤 드롭)
- 음악 플레이어
- 마크다운 에디터
- 지출 관리 앱

---

## 참고 자료

- [MDN Web Docs](https://developer.mozilla.org/ko/)
- [JavaScript.info](https://ko.javascript.info/)
- [CSS-Tricks](https://css-tricks.com/)
- [Web.dev](https://web.dev/)
