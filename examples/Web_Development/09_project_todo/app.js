/*
 * Todo App
 * Features: Add, delete, edit, complete, filter, local storage save
 */

// ============================================
// State
// ============================================
let todos = [];
let currentFilter = 'all';

// ============================================
// DOM Elements
// ============================================
const todoInput = document.getElementById('todoInput');
const addBtn = document.getElementById('addBtn');
const todoList = document.getElementById('todoList');
const todoCount = document.getElementById('todoCount');
const clearCompletedBtn = document.getElementById('clearCompleted');
const filterBtns = document.querySelectorAll('.filter-btn');
const currentDateEl = document.getElementById('currentDate');

// ============================================
// Initialize
// ============================================
function init() {
    // Display date
    displayCurrentDate();

    // Load data from local storage
    loadTodos();

    // Register event listeners
    addEventListeners();

    // Initial render
    render();
}

function displayCurrentDate() {
    const now = new Date();
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long'
    };
    currentDateEl.textContent = now.toLocaleDateString('en-US', options);
}

// ============================================
// Event Listeners
// ============================================
function addEventListeners() {
    // Add button click
    addBtn.addEventListener('click', addTodo);

    // Add with Enter key
    todoInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addTodo();
        }
    });

    // Clear completed items
    clearCompletedBtn.addEventListener('click', clearCompleted);

    // Filter buttons
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            setFilter(btn.dataset.filter);
        });
    });

    // Why: Event delegation on the list container avoids re-attaching listeners after every
    // re-render and handles clicks on dynamically created todo items automatically
    // Todo list event delegation
    todoList.addEventListener('click', handleTodoClick);
    todoList.addEventListener('change', handleTodoChange);
}

// ============================================
// Todo CRUD Operations
// ============================================
function addTodo() {
    const text = todoInput.value.trim();

    if (!text) {
        todoInput.focus();
        return;
    }

    // Why: Date.now() provides a simple unique ID without external libraries; sufficient
    // for client-side-only apps since collisions are practically impossible at human input speeds
    const newTodo = {
        id: Date.now(),
        text: text,
        completed: false,
        createdAt: new Date().toISOString()
    };

    todos.unshift(newTodo);
    saveTodos();
    render();

    todoInput.value = '';
    todoInput.focus();
}

function deleteTodo(id) {
    todos = todos.filter(todo => todo.id !== id);
    saveTodos();
    render();
}

// Why: Spreading into a new object ({ ...todo }) produces an immutable update,
// making state changes predictable and easy to debug (no mutation side effects)
function toggleTodo(id) {
    todos = todos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
    );
    saveTodos();
    render();
}

function editTodo(id) {
    const todoItem = document.querySelector(`[data-id="${id}"]`);
    const todo = todos.find(t => t.id === id);

    if (!todoItem || !todo) return;

    // Switch to edit mode
    todoItem.innerHTML = `
        <input type="checkbox" ${todo.completed ? 'checked' : ''} disabled>
        <input type="text" class="edit-input" value="${escapeHtml(todo.text)}">
        <div class="todo-actions" style="opacity: 1;">
            <button class="save-btn" data-action="save">Save</button>
            <button class="cancel-btn" data-action="cancel">Cancel</button>
        </div>
    `;

    const editInput = todoItem.querySelector('.edit-input');
    editInput.focus();
    editInput.select();

    // Save with Enter key
    editInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            saveTodoEdit(id, editInput.value);
        }
    });

    // Cancel with Escape key
    editInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            render();
        }
    });
}

function saveTodoEdit(id, newText) {
    const text = newText.trim();

    if (!text) {
        render();
        return;
    }

    todos = todos.map(todo =>
        todo.id === id ? { ...todo, text: text } : todo
    );
    saveTodos();
    render();
}

function clearCompleted() {
    todos = todos.filter(todo => !todo.completed);
    saveTodos();
    render();
}

// ============================================
// Event Handlers
// ============================================
function handleTodoClick(e) {
    const todoItem = e.target.closest('.todo-item');
    if (!todoItem) return;

    const id = parseInt(todoItem.dataset.id);
    const action = e.target.dataset.action;

    switch (action) {
        case 'delete':
            deleteTodo(id);
            break;
        case 'edit':
            editTodo(id);
            break;
        case 'save':
            const editInput = todoItem.querySelector('.edit-input');
            if (editInput) {
                saveTodoEdit(id, editInput.value);
            }
            break;
        case 'cancel':
            render();
            break;
    }
}

function handleTodoChange(e) {
    if (e.target.type === 'checkbox') {
        const todoItem = e.target.closest('.todo-item');
        if (todoItem) {
            const id = parseInt(todoItem.dataset.id);
            toggleTodo(id);
        }
    }
}

// ============================================
// Filter
// ============================================
function setFilter(filter) {
    currentFilter = filter;

    // Update button active state
    filterBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    render();
}

function getFilteredTodos() {
    switch (currentFilter) {
        case 'active':
            return todos.filter(todo => !todo.completed);
        case 'completed':
            return todos.filter(todo => todo.completed);
        default:
            return todos;
    }
}

// ============================================
// Render
// ============================================
function render() {
    const filteredTodos = getFilteredTodos();

    if (filteredTodos.length === 0) {
        todoList.innerHTML = `
            <li class="empty-state">
                <p>${getEmptyMessage()}</p>
            </li>
        `;
    } else {
        todoList.innerHTML = filteredTodos.map(todo => `
            <li class="todo-item ${todo.completed ? 'completed' : ''}" data-id="${todo.id}">
                <input type="checkbox" ${todo.completed ? 'checked' : ''}>
                <span class="todo-text">${escapeHtml(todo.text)}</span>
                <div class="todo-actions">
                    <button class="edit-btn" data-action="edit">Edit</button>
                    <button class="delete-btn" data-action="delete">Delete</button>
                </div>
            </li>
        `).join('');
    }

    updateCounter();
}

function getEmptyMessage() {
    switch (currentFilter) {
        case 'active':
            return 'No active todos! Well done!';
        case 'completed':
            return 'No completed todos yet.';
        default:
            return 'Add a new todo!';
    }
}

function updateCounter() {
    const activeCount = todos.filter(todo => !todo.completed).length;
    const totalCount = todos.length;
    todoCount.textContent = `${activeCount} items remaining (${totalCount} total)`;
}

// ============================================
// Local Storage
// ============================================
function saveTodos() {
    localStorage.setItem('todos', JSON.stringify(todos));
}

// Why: Wrapping JSON.parse in try-catch guards against corrupted localStorage data
// (e.g., manual edits, quota errors) that would crash the entire app on load
function loadTodos() {
    const stored = localStorage.getItem('todos');
    if (stored) {
        try {
            todos = JSON.parse(stored);
        } catch (e) {
            console.error('Failed to load todos:', e);
            todos = [];
        }
    }
}

// ============================================
// Utility
// ============================================
// Why: Using textContent+innerHTML leverages the browser's own encoding to escape HTML entities,
// preventing XSS attacks when rendering user-supplied text into innerHTML templates
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Start App
// ============================================
init();
