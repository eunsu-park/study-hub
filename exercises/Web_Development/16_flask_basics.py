"""
Exercises for Lesson 16: Flask Web Framework Basics
Topic: Web_Development
Solutions to practice problems from the lesson.
Run: python exercises/Web_Development/16_flask_basics.py

Note: This file provides complete implementations for the three exercises.
To actually run each app, uncomment the app.run() call at the bottom of
each exercise section. Only one Flask app can run at a time.

Requirements: pip install flask flask-sqlalchemy
"""

# =====================================================================
# Exercise 1: Basic CRUD Todo App
# =====================================================================
# Problem: Create a simple Todo list application with:
#   - View list of todos
#   - Add new todo
#   - Mark as completed
#   - Delete

from flask import Flask, request, jsonify, redirect, url_for
import os
import json
from datetime import datetime


def create_todo_app():
    """Exercise 1: Basic CRUD Todo application using in-memory storage."""

    app = Flask(__name__)

    # In-memory storage (use a database in production)
    todos = []
    next_id = 1

    @app.route('/')
    def index():
        """Render the todo list page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Todo App</title>
            <style>
                body { font-family: sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
                h1 { color: #2c3e50; }
                .todo-form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
                .todo-form input { flex: 1; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
                .todo-form button { padding: 0.5rem 1rem; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }
                .todo-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; border-bottom: 1px solid #eee; }
                .todo-item.completed span { text-decoration: line-through; color: #999; }
                .todo-item span { flex: 1; }
                .btn-complete { background: #2ecc71; color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 4px; cursor: pointer; }
                .btn-delete { background: #e74c3c; color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 4px; cursor: pointer; }
                .empty { color: #999; font-style: italic; }
            </style>
        </head>
        <body>
            <h1>Todo List</h1>
            <form class="todo-form" action="/add" method="POST">
                <input type="text" name="title" placeholder="What needs to be done?" required>
                <button type="submit">Add</button>
            </form>
            <div id="todos">
        """
        if not todos:
            html += '<p class="empty">No todos yet. Add one above!</p>'
        else:
            for todo in todos:
                completed_class = ' completed' if todo['completed'] else ''
                html += f'''
                <div class="todo-item{completed_class}">
                    <span>{todo["title"]}</span>
                    <form action="/toggle/{todo["id"]}" method="POST" style="display:inline;">
                        <button class="btn-complete">{"Undo" if todo["completed"] else "Done"}</button>
                    </form>
                    <form action="/delete/{todo["id"]}" method="POST" style="display:inline;">
                        <button class="btn-delete">Delete</button>
                    </form>
                </div>
                '''
        html += '</div></body></html>'
        return html

    @app.route('/add', methods=['POST'])
    def add_todo():
        nonlocal next_id
        title = request.form.get('title', '').strip()
        if title:
            todos.append({
                'id': next_id,
                'title': title,
                'completed': False,
                'created_at': datetime.now().isoformat(),
            })
            next_id += 1
        return redirect(url_for('index'))

    @app.route('/toggle/<int:todo_id>', methods=['POST'])
    def toggle_todo(todo_id):
        for todo in todos:
            if todo['id'] == todo_id:
                todo['completed'] = not todo['completed']
                break
        return redirect(url_for('index'))

    @app.route('/delete/<int:todo_id>', methods=['POST'])
    def delete_todo(todo_id):
        nonlocal todos
        todos = [t for t in todos if t['id'] != todo_id]
        return redirect(url_for('index'))

    return app


# =====================================================================
# Exercise 2: User Authentication
# =====================================================================
# Problem: Add user authentication to the todo app:
#   - Registration/login/logout
#   - Each user can only see their own todos

def create_auth_todo_app():
    """Exercise 2: Todo app with user authentication using sessions."""

    app = Flask(__name__)
    app.secret_key = 'dev-secret-key-change-in-production'

    from flask import session

    # In-memory storage
    users = {}  # {username: {password: ..., todos: [...]}}

    def get_current_user():
        return session.get('username')

    @app.route('/')
    def index():
        username = get_current_user()
        if not username:
            return redirect(url_for('login'))

        user_todos = users.get(username, {}).get('todos', [])

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Todo App - {username}</title>
            <style>
                body {{ font-family: sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }}
                .header h1 {{ color: #2c3e50; }}
                .logout-btn {{ background: #e74c3c; color: white; padding: 0.5rem 1rem; text-decoration: none; border-radius: 4px; }}
                .todo-form {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; }}
                .todo-form input {{ flex: 1; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }}
                .todo-form button {{ padding: 0.5rem 1rem; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                .todo-item {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; border-bottom: 1px solid #eee; }}
                .todo-item.completed span {{ text-decoration: line-through; color: #999; }}
                .todo-item span {{ flex: 1; }}
                .btn {{ border: none; padding: 0.25rem 0.5rem; border-radius: 4px; cursor: pointer; color: white; }}
                .btn-done {{ background: #2ecc71; }}
                .btn-del {{ background: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Todos for {username}</h1>
                <a href="/logout" class="logout-btn">Logout</a>
            </div>
            <form class="todo-form" action="/add" method="POST">
                <input type="text" name="title" placeholder="New todo..." required>
                <button type="submit">Add</button>
            </form>
        """
        for i, todo in enumerate(user_todos):
            cls = ' completed' if todo['completed'] else ''
            label = 'Undo' if todo['completed'] else 'Done'
            html += f'''
            <div class="todo-item{cls}">
                <span>{todo["title"]}</span>
                <form action="/toggle/{i}" method="POST" style="display:inline;">
                    <button class="btn btn-done">{label}</button>
                </form>
                <form action="/delete/{i}" method="POST" style="display:inline;">
                    <button class="btn btn-del">X</button>
                </form>
            </div>
            '''
        html += '</body></html>'
        return html

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            if username and password:
                if username in users:
                    return 'Username already exists. <a href="/register">Try again</a>'
                users[username] = {'password': password, 'todos': []}
                session['username'] = username
                return redirect(url_for('index'))
            return 'Username and password required. <a href="/register">Try again</a>'

        return '''
        <h1>Register</h1>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required><br><br>
            <input type="password" name="password" placeholder="Password" required><br><br>
            <button type="submit">Register</button>
        </form>
        <p>Already have an account? <a href="/login">Login</a></p>
        '''

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            user = users.get(username)
            if user and user['password'] == password:
                session['username'] = username
                return redirect(url_for('index'))
            return 'Invalid credentials. <a href="/login">Try again</a>'

        return '''
        <h1>Login</h1>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required><br><br>
            <input type="password" name="password" placeholder="Password" required><br><br>
            <button type="submit">Login</button>
        </form>
        <p>No account? <a href="/register">Register</a></p>
        '''

    @app.route('/logout')
    def logout():
        session.pop('username', None)
        return redirect(url_for('login'))

    @app.route('/add', methods=['POST'])
    def add_todo():
        username = get_current_user()
        if not username:
            return redirect(url_for('login'))
        title = request.form.get('title', '').strip()
        if title:
            users[username]['todos'].append({
                'title': title,
                'completed': False,
            })
        return redirect(url_for('index'))

    @app.route('/toggle/<int:idx>', methods=['POST'])
    def toggle_todo(idx):
        username = get_current_user()
        if not username:
            return redirect(url_for('login'))
        todos = users[username]['todos']
        if 0 <= idx < len(todos):
            todos[idx]['completed'] = not todos[idx]['completed']
        return redirect(url_for('index'))

    @app.route('/delete/<int:idx>', methods=['POST'])
    def delete_todo(idx):
        username = get_current_user()
        if not username:
            return redirect(url_for('login'))
        todos = users[username]['todos']
        if 0 <= idx < len(todos):
            todos.pop(idx)
        return redirect(url_for('index'))

    return app


# =====================================================================
# Exercise 3: REST API
# =====================================================================
# Problem: Create a Todo list REST API with:
#   - CRUD endpoints
#   - Simple token authentication (simplified JWT)
#   - API documentation

def create_rest_api():
    """Exercise 3: RESTful Todo API with token authentication."""

    app = Flask(__name__)

    # In-memory storage
    users_db = {
        'admin': {'password': 'admin123', 'token': 'token-admin-001'},
    }
    todos_db = {}  # {username: [{id, title, completed, created_at}]}
    next_id = 1

    def authenticate():
        """Extract and validate bearer token from Authorization header."""
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None
        token = auth_header.split(' ', 1)[1]
        for username, data in users_db.items():
            if data['token'] == token:
                return username
        return None

    def error_response(message, status_code):
        return jsonify({'error': message}), status_code

    # --- API Documentation ---
    @app.route('/api')
    def api_docs():
        """Return API documentation."""
        return jsonify({
            'name': 'Todo REST API',
            'version': '1.0',
            'endpoints': {
                'POST /api/auth/register': 'Register new user (body: username, password)',
                'POST /api/auth/login': 'Login and get token (body: username, password)',
                'GET /api/todos': 'List all todos (requires auth)',
                'POST /api/todos': 'Create todo (body: title; requires auth)',
                'GET /api/todos/<id>': 'Get single todo (requires auth)',
                'PUT /api/todos/<id>': 'Update todo (body: title, completed; requires auth)',
                'DELETE /api/todos/<id>': 'Delete todo (requires auth)',
            },
            'authentication': 'Bearer token in Authorization header',
        })

    # --- Auth Endpoints ---
    @app.route('/api/auth/register', methods=['POST'])
    def register():
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return error_response('username and password required', 400)

        username = data['username']
        if username in users_db:
            return error_response('Username already exists', 409)

        import hashlib
        token = f"token-{hashlib.md5(username.encode()).hexdigest()[:8]}"
        users_db[username] = {'password': data['password'], 'token': token}
        todos_db[username] = []

        return jsonify({
            'message': 'Registration successful',
            'token': token,
        }), 201

    @app.route('/api/auth/login', methods=['POST'])
    def login():
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return error_response('username and password required', 400)

        user = users_db.get(data['username'])
        if not user or user['password'] != data['password']:
            return error_response('Invalid credentials', 401)

        return jsonify({
            'message': 'Login successful',
            'token': user['token'],
        })

    # --- Todo CRUD Endpoints ---
    @app.route('/api/todos', methods=['GET'])
    def list_todos():
        username = authenticate()
        if not username:
            return error_response('Authentication required', 401)

        user_todos = todos_db.get(username, [])
        return jsonify({
            'todos': user_todos,
            'count': len(user_todos),
        })

    @app.route('/api/todos', methods=['POST'])
    def create_todo():
        nonlocal next_id
        username = authenticate()
        if not username:
            return error_response('Authentication required', 401)

        data = request.get_json()
        if not data or 'title' not in data:
            return error_response('title is required', 400)

        todo = {
            'id': next_id,
            'title': data['title'],
            'completed': False,
            'created_at': datetime.now().isoformat(),
        }
        next_id += 1

        if username not in todos_db:
            todos_db[username] = []
        todos_db[username].append(todo)

        return jsonify(todo), 201

    @app.route('/api/todos/<int:todo_id>', methods=['GET'])
    def get_todo(todo_id):
        username = authenticate()
        if not username:
            return error_response('Authentication required', 401)

        for todo in todos_db.get(username, []):
            if todo['id'] == todo_id:
                return jsonify(todo)
        return error_response('Todo not found', 404)

    @app.route('/api/todos/<int:todo_id>', methods=['PUT'])
    def update_todo(todo_id):
        username = authenticate()
        if not username:
            return error_response('Authentication required', 401)

        data = request.get_json()
        for todo in todos_db.get(username, []):
            if todo['id'] == todo_id:
                if 'title' in data:
                    todo['title'] = data['title']
                if 'completed' in data:
                    todo['completed'] = data['completed']
                return jsonify(todo)
        return error_response('Todo not found', 404)

    @app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
    def delete_todo(todo_id):
        username = authenticate()
        if not username:
            return error_response('Authentication required', 401)

        user_todos = todos_db.get(username, [])
        for i, todo in enumerate(user_todos):
            if todo['id'] == todo_id:
                user_todos.pop(i)
                return jsonify({'message': 'Todo deleted'})
        return error_response('Todo not found', 404)

    return app


# =====================================================================
# Main: demonstrate all three exercise apps
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Exercises for Lesson 16: Flask Basics")
    print("=" * 60)

    print("\nExercise 1: Basic CRUD Todo App")
    print("  - Routes: /, /add, /toggle/<id>, /delete/<id>")
    print("  - In-memory storage with auto-incrementing IDs")
    print("  - HTML rendered server-side with form submissions")

    print("\nExercise 2: Todo App with Authentication")
    print("  - Routes: /register, /login, /logout + todo CRUD")
    print("  - Session-based authentication")
    print("  - Per-user todo isolation")

    print("\nExercise 3: REST API")
    print("  - Endpoints: /api/auth/*, /api/todos/*")
    print("  - Bearer token authentication")
    print("  - JSON request/response")
    print("  - API documentation at /api")

    print("\n--- Running Exercise 3 (REST API) demo ---\n")

    # Create and test the REST API app
    app = create_rest_api()

    with app.test_client() as client:
        # Register a user
        resp = client.post('/api/auth/register', json={
            'username': 'testuser',
            'password': 'testpass',
        })
        data = resp.get_json()
        token = data['token']
        print(f"Register: {resp.status_code} - token: {token}")

        # Login
        resp = client.post('/api/auth/login', json={
            'username': 'testuser',
            'password': 'testpass',
        })
        print(f"Login: {resp.status_code} - {resp.get_json()['message']}")

        # Create todos
        headers = {'Authorization': f'Bearer {token}'}

        resp = client.post('/api/todos', json={'title': 'Learn Flask'},
                           headers=headers)
        print(f"Create todo: {resp.status_code} - {resp.get_json()['title']}")

        resp = client.post('/api/todos', json={'title': 'Build REST API'},
                           headers=headers)
        print(f"Create todo: {resp.status_code} - {resp.get_json()['title']}")

        resp = client.post('/api/todos', json={'title': 'Add authentication'},
                           headers=headers)
        print(f"Create todo: {resp.status_code} - {resp.get_json()['title']}")

        # List todos
        resp = client.get('/api/todos', headers=headers)
        data = resp.get_json()
        print(f"\nList todos: {data['count']} todos")
        for todo in data['todos']:
            status = 'done' if todo['completed'] else 'pending'
            print(f"  [{status}] {todo['id']}: {todo['title']}")

        # Update todo (mark as completed)
        resp = client.put('/api/todos/1', json={'completed': True},
                          headers=headers)
        print(f"\nUpdate todo 1: {resp.status_code} - completed: {resp.get_json()['completed']}")

        # Delete todo
        resp = client.delete('/api/todos/3', headers=headers)
        print(f"Delete todo 3: {resp.status_code} - {resp.get_json()['message']}")

        # Final list
        resp = client.get('/api/todos', headers=headers)
        data = resp.get_json()
        print(f"\nFinal list: {data['count']} todos")
        for todo in data['todos']:
            status = 'done' if todo['completed'] else 'pending'
            print(f"  [{status}] {todo['id']}: {todo['title']}")

        # Test unauthorized access
        resp = client.get('/api/todos')
        print(f"\nUnauthorized access: {resp.status_code} - {resp.get_json()['error']}")

        # API docs
        resp = client.get('/api')
        print(f"\nAPI docs: {resp.status_code}")
        docs = resp.get_json()
        print(f"  Name: {docs['name']} v{docs['version']}")
        print(f"  Endpoints: {len(docs['endpoints'])}")

    print("\n--- All exercises demonstrated successfully! ---")
    print("\nTo run any app interactively:")
    print("  app = create_todo_app()        # Exercise 1")
    print("  app = create_auth_todo_app()   # Exercise 2")
    print("  app = create_rest_api()        # Exercise 3")
    print("  app.run(port=5050, debug=True)")
