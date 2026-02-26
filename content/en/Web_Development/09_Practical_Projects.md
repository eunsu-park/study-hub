# Practical Projects

**Previous**: [JavaScript Asynchronous Programming](./08_JS_Async.md) | **Next**: [TypeScript Fundamentals](./10_TypeScript_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Build a complete Todo application with CRUD operations and local storage persistence
2. Integrate external REST APIs to fetch and display dynamic data in a Weather app
3. Implement infinite scroll using the Intersection Observer API
4. Design responsive layouts that adapt from mobile to desktop viewports
5. Apply event delegation patterns for efficient DOM event handling
6. Handle loading states, error conditions, and empty states in user interfaces
7. Structure front-end projects with clean separation of HTML, CSS, and JavaScript
8. Implement a lightbox gallery with keyboard navigation support

---

Building real projects is the bridge between learning individual concepts and becoming a productive web developer. While tutorials teach syntax, projects teach you how to combine HTML structure, CSS styling, and JavaScript logic into cohesive applications that solve actual problems. The three projects in this lesson progressively increase in complexity, each introducing new patterns you will encounter in professional work.

## Overview

In this document, we'll create actual working web applications combining what we've learned about HTML, CSS, and JavaScript.

**Prerequisites**: All previous chapters

---

## Table of Contents

1. [Project 1: Todo App](#project-1-todo-app)
2. [Project 2: Weather App](#project-2-weather-app)
3. [Project 3: Image Gallery](#project-3-image-gallery)

---

## Project 1: Todo App

A task management application using local storage.

### Features

- Add/delete/complete tasks
- Save to local storage
- Filtering (all/active/completed)
- Responsive design

### File Structure

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
<html lang="en">
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
                placeholder="Enter a task"
                required
                autocomplete="off"
            >
            <button type="submit" class="btn btn-primary">Add</button>
        </form>

        <div class="filters">
            <button class="filter-btn active" data-filter="all">All</button>
            <button class="filter-btn" data-filter="active">Active</button>
            <button class="filter-btn" data-filter="completed">Completed</button>
        </div>

        <ul id="todoList" class="todo-list"></ul>

        <footer class="todo-footer">
            <span id="itemCount">0 items</span>
            <button id="clearCompleted" class="btn btn-text">Clear completed</button>
        </footer>
    </div>

    <script src="js/app.js"></script>
</body>
</html>
```

### css/style.css

```css
/* Base styles */
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

/* Header */
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

/* Form */
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

/* Buttons */
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

/* Filters */
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

/* Todo list */
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

/* Footer */
.todo-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    color: #999;
    font-size: 0.9rem;
}

/* Empty state */
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

/* Responsive */
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
// Todo app class
class TodoApp {
    constructor() {
        // DOM elements
        this.form = document.getElementById('todoForm');
        this.input = document.getElementById('todoInput');
        this.list = document.getElementById('todoList');
        this.itemCount = document.getElementById('itemCount');
        this.clearBtn = document.getElementById('clearCompleted');
        this.filterBtns = document.querySelectorAll('.filter-btn');

        // State
        this.todos = this.loadTodos();
        this.filter = 'all';

        // Initialize
        this.init();
    }

    init() {
        // Display date
        this.displayDate();

        // Event listeners
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.list.addEventListener('click', (e) => this.handleListClick(e));
        this.clearBtn.addEventListener('click', () => this.clearCompleted());

        this.filterBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.handleFilter(e));
        });

        // Initial render
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
        dateEl.textContent = new Date().toLocaleDateString('en-US', options);
    }

    // Local storage
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

    // Filtering
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

    // Event handlers
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

    // Rendering
    render() {
        const filteredTodos = this.getFilteredTodos();

        if (filteredTodos.length === 0) {
            this.list.innerHTML = `
                <li class="empty-state">
                    ${this.filter === 'all' ? 'Add your first task!' :
                      this.filter === 'active' ? 'No active items' :
                      'No completed items'}
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
                    <button class="todo-delete" aria-label="Delete">×</button>
                </li>
            `).join('');
        }

        // Update count
        const activeCount = this.todos.filter(t => !t.completed).length;
        this.itemCount.textContent = `${activeCount} item${activeCount !== 1 ? 's' : ''}`;

        // Show/hide clear button
        const hasCompleted = this.todos.some(t => t.completed);
        this.clearBtn.style.display = hasCompleted ? 'block' : 'none';
    }

    // XSS prevention
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Start app
document.addEventListener('DOMContentLoaded', () => {
    new TodoApp();
});
```

---

## Project 2: Weather App

A weather information app using an external API.

### Features

- Search weather by city name
- Current location weather
- Weather icons and detailed info
- Loading state and error handling

### Setup

Get a free API key from [OpenWeatherMap](https://openweathermap.org/api).

### File Structure

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
<html lang="en">
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
                    placeholder="Enter city name"
                    autocomplete="off"
                >
                <button type="submit">Search</button>
            </form>
            <button id="locationBtn" class="location-btn" title="Current location">
                📍
            </button>
        </div>

        <div id="loading" class="loading hidden">
            <div class="spinner"></div>
            <p>Getting weather information...</p>
        </div>

        <div id="error" class="error hidden">
            <p id="errorMessage"></p>
            <button id="retryBtn">Retry</button>
        </div>

        <div id="weather" class="weather-card hidden">
            <div class="weather-main">
                <img id="weatherIcon" src="" alt="Weather icon">
                <div class="temperature">
                    <span id="temp">--</span>
                    <span class="unit">°C</span>
                </div>
            </div>

            <h2 id="cityName">--</h2>
            <p id="description">--</p>

            <div class="weather-details">
                <div class="detail">
                    <span class="label">Feels like</span>
                    <span id="feelsLike">--°C</span>
                </div>
                <div class="detail">
                    <span class="label">Humidity</span>
                    <span id="humidity">--%</span>
                </div>
                <div class="detail">
                    <span class="label">Wind</span>
                    <span id="wind">--m/s</span>
                </div>
                <div class="detail">
                    <span class="label">Clouds</span>
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

/* Search box */
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

/* Weather card */
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

/* Loading */
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

/* Error */
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

/* Utility */
.hidden {
    display: none !important;
}

/* Responsive */
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
// API key (replace with your actual key)
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

        // Restore last search
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
            this.showError('Geolocation is not supported by this browser.');
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
                let message = 'Unable to retrieve your location.';
                if (error.code === error.PERMISSION_DENIED) {
                    message = 'Location access permission denied.';
                }
                this.showError(message);
            }
        );
    }

    async fetchWeather(city) {
        this.showLoading();

        try {
            const url = `${BASE_URL}?q=${encodeURIComponent(city)}&appid=${API_KEY}&units=metric`;
            const response = await fetch(url);

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error('City not found.');
                }
                throw new Error('Unable to fetch weather information.');
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
            const url = `${BASE_URL}?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric`;
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error('Unable to fetch weather information.');
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

// Start app
document.addEventListener('DOMContentLoaded', () => {
    new WeatherApp();
});
```

---

## Project 3: Image Gallery

An image gallery with infinite scroll and lightbox functionality.

### Features

- Load images with Unsplash API
- Infinite scroll
- Lightbox (click to enlarge)
- Responsive grid

### Setup

Get an API key from [Unsplash](https://unsplash.com/developers).

### File Structure

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
<html lang="en">
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
                placeholder="Search images..."
                autocomplete="off"
            >
            <button type="submit">Search</button>
        </form>
    </header>

    <main>
        <div id="gallery" class="gallery"></div>
        <div id="loading" class="loading">
            <div class="spinner"></div>
        </div>
        <div id="sentinel" class="sentinel"></div>
    </main>

    <!-- Lightbox -->
    <div id="lightbox" class="lightbox hidden">
        <button class="lightbox-close">&times;</button>
        <button class="lightbox-prev">&lt;</button>
        <button class="lightbox-next">&gt;</button>
        <div class="lightbox-content">
            <img id="lightboxImage" src="" alt="">
            <div class="lightbox-info">
                <p id="lightboxAuthor"></p>
                <a id="lightboxLink" href="" target="_blank">View on Unsplash</a>
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

/* Header */
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

/* Gallery */
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

/* Loading */
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

/* Lightbox */
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

/* Responsive */
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
// API key (replace with your actual key)
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
        // Search
        this.searchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.search(this.searchInput.value.trim());
        });

        // Infinite scroll
        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && !this.isLoading) {
                this.loadImages();
            }
        });
        observer.observe(document.getElementById('sentinel'));

        // Gallery click (event delegation)
        this.gallery.addEventListener('click', (e) => {
            const item = e.target.closest('.gallery-item');
            if (item) {
                const index = parseInt(item.dataset.index);
                this.openLightbox(index);
            }
        });

        // Lightbox controls
        this.lightbox.querySelector('.lightbox-close').addEventListener('click', () => {
            this.closeLightbox();
        });

        this.lightbox.querySelector('.lightbox-prev').addEventListener('click', () => {
            this.prevImage();
        });

        this.lightbox.querySelector('.lightbox-next').addEventListener('click', () => {
            this.nextImage();
        });

        // Keyboard navigation
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

        // Close lightbox on background click
        this.lightbox.addEventListener('click', (e) => {
            if (e.target === this.lightbox) {
                this.closeLightbox();
            }
        });

        // Initial load
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
            if (!response.ok) throw new Error('Unable to load images.');

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
                    alt="${photo.alt_description || 'Image'}"
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

// Start app
document.addEventListener('DOMContentLoaded', () => {
    new GalleryApp();
});
```

---

---

## Exercises

### Exercise 1: Extend the Todo App

Add the following features to the Todo app from Project 1:

1. **Edit functionality**: Allow users to double-click a todo item to edit its text in-place. Pressing Enter or clicking outside should save the change.
2. **Due dates**: Add an optional due date field when creating a todo. Display overdue items with a red indicator.
3. **Priority levels**: Add a priority selector (low / medium / high) and sort the list so high-priority items appear first.

> **Tip**: For edit mode, toggle between a `<span>` and an `<input>` element using `display: none`. Store the `dueDate` and `priority` fields alongside `id`, `text`, and `completed` in each todo object.

### Exercise 2: Mock API for the Weather App

Rewrite the Weather app from Project 2 to work without a real API key by using a mock data layer:

1. Create a `mockApi.js` file that exports an `async function fetchWeather(city)`. Have it return a hardcoded weather object after a simulated 800 ms delay using `setTimeout` inside a `Promise`.
2. In `app.js`, replace the real `fetch` call with an import from `mockApi.js`.
3. Add a second mock function `fetchWeatherByCoords(lat, lon)` that returns different mock data based on whether latitude is positive (northern hemisphere) or negative.

```javascript
// mockApi.js skeleton
export function fetchWeather(city) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (city.toLowerCase() === 'unknown') {
                reject(new Error('City not found.'));
            } else {
                resolve({ name: city, main: { temp: 22, feels_like: 20, humidity: 60 }, /* ... */ });
            }
        }, 800);
    });
}
```

### Exercise 3: Accessible Gallery with Keyboard Navigation

Improve the Image Gallery from Project 3 to fully support keyboard navigation:

1. Make each gallery item focusable (`tabindex="0"`) and open the lightbox when the user presses **Enter** or **Space** on a focused item.
2. When the lightbox is open, trap focus inside it: pressing **Tab** and **Shift+Tab** should cycle only through the Close, Previous, and Next buttons.
3. When the lightbox closes, return focus to the gallery item that was active before the lightbox opened.
4. Add `aria-label` attributes to the Close (`"Close lightbox"`), Previous (`"Previous image"`), and Next (`"Next image"`) buttons.

### Exercise 4: Responsive Kanban Board (Advanced)

Build a minimal Kanban board as a fourth project:

- Three columns: **To Do**, **In Progress**, **Done**
- Cards can be dragged between columns using the HTML Drag and Drop API (`dragstart`, `dragover`, `drop` events)
- Card state (which column each card belongs to) is persisted in `localStorage`
- Clicking a card opens an inline edit form for the card title

```javascript
// Drag and drop skeleton
card.addEventListener('dragstart', (e) => {
    e.dataTransfer.setData('text/plain', card.dataset.id);
});

column.addEventListener('dragover', (e) => e.preventDefault());

column.addEventListener('drop', (e) => {
    const id = e.dataTransfer.getData('text/plain');
    const card = document.querySelector(`[data-id="${id}"]`);
    column.querySelector('.cards').appendChild(card);
    saveState();
});
```

---

## References

- [MDN Web Docs](https://developer.mozilla.org/en-US/)
- [JavaScript.info](https://javascript.info/)
- [CSS-Tricks](https://css-tricks.com/)
- [Web.dev](https://web.dev/)

---

**Previous**: [JavaScript Asynchronous Programming](./08_JS_Async.md) | **Next**: [TypeScript Fundamentals](./10_TypeScript_Basics.md)
