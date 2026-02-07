# 16. Flask 웹 프레임워크 기초

## 학습 목표

- Flask의 핵심 개념과 구조 이해
- 라우팅, 템플릿, 폼 처리 습득
- 데이터베이스 연동 및 RESTful API 구현
- 블루프린트를 활용한 대규모 애플리케이션 구조화

---

## 1. Flask 소개

### 1.1 Flask란?

Flask는 Python으로 작성된 마이크로 웹 프레임워크입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                     Flask 아키텍처                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │ Client  │────▶│  Flask  │────▶│ Response│              │
│   │(Browser)│     │   App   │     │  (HTML) │              │
│   └─────────┘     └────┬────┘     └─────────┘              │
│                        │                                     │
│         ┌──────────────┼──────────────┐                     │
│         │              │              │                     │
│         ▼              ▼              ▼                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│   │ Routing │    │Templates│    │Database │               │
│   │  (URL)  │    │ (Jinja2)│    │(SQLAlchemy)│            │
│   └─────────┘    └─────────┘    └─────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 설치 및 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Flask 설치
pip install flask

# 추가 패키지 (권장)
pip install flask-sqlalchemy flask-migrate flask-wtf python-dotenv
```

### 1.3 최소 Flask 애플리케이션

```python
# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

```bash
# 실행
python app.py
# 또는
flask run --debug

# 브라우저에서 http://127.0.0.1:5000 접속
```

---

## 2. 라우팅 (Routing)

### 2.1 기본 라우트

```python
from flask import Flask

app = Flask(__name__)

# 기본 라우트
@app.route('/')
def index():
    return 'Home Page'

# 정적 경로
@app.route('/about')
def about():
    return 'About Page'

@app.route('/contact')
def contact():
    return 'Contact Page'
```

### 2.2 동적 URL

```python
# 문자열 변수
@app.route('/user/<username>')
def show_user(username):
    return f'User: {username}'

# 정수 변수
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post #{post_id}'

# 경로 변수 (슬래시 포함)
@app.route('/path/<path:subpath>')
def show_path(subpath):
    return f'Path: {subpath}'

# 여러 변수
@app.route('/user/<username>/post/<int:post_id>')
def user_post(username, post_id):
    return f'{username}\'s Post #{post_id}'
```

### 2.3 HTTP 메서드

```python
from flask import request

# GET (기본값)
@app.route('/search')
def search():
    query = request.args.get('q', '')
    return f'Search: {query}'

# POST
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 로그인 처리
        return f'Logging in as {username}'
    return '''
        <form method="post">
            <input name="username" placeholder="Username">
            <input name="password" type="password" placeholder="Password">
            <button type="submit">Login</button>
        </form>
    '''

# RESTful 스타일
@app.route('/api/users', methods=['GET', 'POST'])
def users():
    if request.method == 'GET':
        return {'users': ['Alice', 'Bob']}
    elif request.method == 'POST':
        data = request.json
        return {'created': data['name']}, 201

@app.route('/api/users/<int:user_id>', methods=['GET', 'PUT', 'DELETE'])
def user(user_id):
    if request.method == 'GET':
        return {'id': user_id, 'name': 'Alice'}
    elif request.method == 'PUT':
        data = request.json
        return {'id': user_id, 'name': data['name']}
    elif request.method == 'DELETE':
        return '', 204
```

### 2.4 URL 빌딩

```python
from flask import url_for, redirect

@app.route('/admin')
def admin():
    return 'Admin Page'

@app.route('/user/<username>')
def profile(username):
    return f'Profile: {username}'

@app.route('/redirect-example')
def redirect_example():
    # url_for로 URL 생성
    home_url = url_for('index')  # '/'
    admin_url = url_for('admin')  # '/admin'
    profile_url = url_for('profile', username='alice')  # '/user/alice'

    # 리다이렉트
    return redirect(url_for('profile', username='guest'))
```

---

## 3. 템플릿 (Jinja2)

### 3.1 기본 템플릿

```
프로젝트 구조:
my_app/
├── app.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── user.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

```python
# app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/user/<name>')
def user(name):
    posts = [
        {'title': 'First Post', 'date': '2024-01-01'},
        {'title': 'Second Post', 'date': '2024-01-15'},
    ]
    return render_template('user.html', username=name, posts=posts)
```

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{% endblock %} - My App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
        <a href="{{ url_for('about') }}">About</a>
    </nav>

    <main>
        {% block content %}{% endblock %}
    </main>

    <footer>
        <p>&copy; 2024 My App</p>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

```html
<!-- templates/index.html -->
{% extends 'base.html' %}

{% block title %}{{ title }}{% endblock %}

{% block content %}
<h1>Welcome to {{ title }}</h1>
<p>This is the home page.</p>
{% endblock %}
```

```html
<!-- templates/user.html -->
{% extends 'base.html' %}

{% block title %}{{ username }}'s Profile{% endblock %}

{% block content %}
<h1>{{ username }}'s Profile</h1>

{% if posts %}
<h2>Posts</h2>
<ul>
    {% for post in posts %}
    <li>
        <strong>{{ post.title }}</strong>
        <span>{{ post.date }}</span>
    </li>
    {% endfor %}
</ul>
{% else %}
<p>No posts yet.</p>
{% endif %}
{% endblock %}
```

### 3.2 Jinja2 문법

```html
<!-- 변수 출력 -->
<p>{{ variable }}</p>
<p>{{ user.name }}</p>
<p>{{ items[0] }}</p>

<!-- 필터 -->
<p>{{ name|upper }}</p>
<p>{{ text|truncate(100) }}</p>
<p>{{ date|datetime('%Y-%m-%d') }}</p>
<p>{{ html_content|safe }}</p>

<!-- 조건문 -->
{% if user.is_admin %}
    <p>Admin Panel</p>
{% elif user.is_staff %}
    <p>Staff Panel</p>
{% else %}
    <p>User Panel</p>
{% endif %}

<!-- 반복문 -->
{% for item in items %}
    <p>{{ loop.index }}. {{ item }}</p>
{% endfor %}

<!-- loop 변수 -->
{% for user in users %}
    {{ loop.index }}     {# 1부터 시작하는 인덱스 #}
    {{ loop.index0 }}    {# 0부터 시작하는 인덱스 #}
    {{ loop.first }}     {# 첫 번째 반복이면 True #}
    {{ loop.last }}      {# 마지막 반복이면 True #}
    {{ loop.length }}    {# 전체 항목 수 #}
{% endfor %}

<!-- 매크로 (재사용 가능한 템플릿 함수) -->
{% macro input(name, type='text', value='') %}
    <input type="{{ type }}" name="{{ name }}" value="{{ value }}">
{% endmacro %}

{{ input('username') }}
{{ input('password', type='password') }}

<!-- 매크로 가져오기 -->
{% from 'macros.html' import input %}

<!-- 블록 -->
{% block sidebar %}
    <aside>Default Sidebar</aside>
{% endblock %}

<!-- 부모 블록 내용 포함 -->
{% block content %}
    {{ super() }}
    <p>Additional content</p>
{% endblock %}
```

### 3.3 커스텀 필터

```python
# app.py
from flask import Flask
from datetime import datetime

app = Flask(__name__)

@app.template_filter('datetime')
def format_datetime(value, format='%Y-%m-%d %H:%M'):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

@app.template_filter('money')
def format_money(value):
    return f'{value:,.0f}원'

# 템플릿에서 사용: {{ post.created_at|datetime('%Y년 %m월 %d일') }}
# 템플릿에서 사용: {{ price|money }}
```

---

## 4. 폼 처리 (Forms)

### 4.1 기본 폼 처리

```python
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # flash 메시지용

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # 간단한 유효성 검사
        if not name or not email:
            flash('이름과 이메일은 필수입니다.', 'error')
            return redirect(url_for('contact'))

        # 데이터 처리 (예: 이메일 전송, DB 저장)
        flash('메시지가 전송되었습니다.', 'success')
        return redirect(url_for('index'))

    return render_template('contact.html')
```

```html
<!-- templates/contact.html -->
{% extends 'base.html' %}

{% block content %}
<h1>Contact Us</h1>

{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
            <div class="alert alert-{{ category }}">{{ message }}</div>
        {% endfor %}
    {% endif %}
{% endwith %}

<form method="POST">
    <div>
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required>
    </div>
    <div>
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
    </div>
    <div>
        <label for="message">Message:</label>
        <textarea id="message" name="message" rows="5"></textarea>
    </div>
    <button type="submit">Send</button>
</form>
{% endblock %}
```

### 4.2 Flask-WTF로 폼 처리

```bash
pip install flask-wtf email-validator
```

```python
# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(message='사용자명을 입력하세요.'),
        Length(min=3, max=20, message='3-20자 사이로 입력하세요.')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='비밀번호를 입력하세요.')
    ])
    remember = BooleanField('Remember Me')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=20)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message='유효한 이메일 주소를 입력하세요.')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='비밀번호는 최소 8자 이상이어야 합니다.')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='비밀번호가 일치하지 않습니다.')
    ])

    def validate_username(self, field):
        # 커스텀 유효성 검사
        if field.data.lower() in ['admin', 'root', 'administrator']:
            raise ValidationError('사용할 수 없는 사용자명입니다.')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    category = SelectField('Category', choices=[
        ('general', '일반 문의'),
        ('support', '기술 지원'),
        ('feedback', '피드백')
    ])
    message = TextAreaField('Message', validators=[
        DataRequired(),
        Length(min=10, max=1000)
    ])
```

```python
# app.py
from flask import Flask, render_template, redirect, url_for, flash
from forms import LoginForm, RegistrationForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        # form.username.data, form.password.data 사용
        username = form.username.data
        # 로그인 로직...
        flash(f'{username}님, 환영합니다!', 'success')
        return redirect(url_for('index'))

    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        # 회원가입 로직...
        flash('회원가입이 완료되었습니다.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)
```

```html
<!-- templates/login.html -->
{% extends 'base.html' %}

{% block content %}
<h1>Login</h1>

<form method="POST" novalidate>
    {{ form.hidden_tag() }}

    <div class="form-group">
        {{ form.username.label }}
        {{ form.username(class="form-control") }}
        {% for error in form.username.errors %}
            <span class="error">{{ error }}</span>
        {% endfor %}
    </div>

    <div class="form-group">
        {{ form.password.label }}
        {{ form.password(class="form-control") }}
        {% for error in form.password.errors %}
            <span class="error">{{ error }}</span>
        {% endfor %}
    </div>

    <div class="form-check">
        {{ form.remember(class="form-check-input") }}
        {{ form.remember.label(class="form-check-label") }}
    </div>

    <button type="submit" class="btn btn-primary">Login</button>
</form>
{% endblock %}
```

### 4.3 파일 업로드

```python
import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('파일이 없습니다.')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('선택된 파일이 없습니다.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash(f'{filename} 업로드 완료!')
            return redirect(url_for('index'))
        else:
            flash('허용되지 않는 파일 형식입니다.')
            return redirect(request.url)

    return '''
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''
```

---

## 5. 데이터베이스 (Flask-SQLAlchemy)

### 5.1 설정 및 모델 정의

```python
# app.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 모델 정의
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 관계: 1:N
    posts = db.relationship('Post', backref='author', lazy='dynamic')

    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 외래 키
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # 관계: N:M (태그)
    tags = db.relationship('Tag', secondary='post_tags', backref='posts')

    def __repr__(self):
        return f'<Post {self.title}>'

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

# 다대다 관계를 위한 연결 테이블
post_tags = db.Table('post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('post.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

# 데이터베이스 생성
with app.app_context():
    db.create_all()
```

### 5.2 CRUD 연산

```python
# Create
user = User(username='alice', email='alice@example.com')
db.session.add(user)
db.session.commit()

# 여러 개 추가
users = [
    User(username='bob', email='bob@example.com'),
    User(username='charlie', email='charlie@example.com')
]
db.session.add_all(users)
db.session.commit()

# Read
# 전체 조회
all_users = User.query.all()

# ID로 조회
user = User.query.get(1)  # deprecated in SQLAlchemy 2.0
user = db.session.get(User, 1)  # SQLAlchemy 2.0+

# 조건 조회
user = User.query.filter_by(username='alice').first()
users = User.query.filter(User.email.like('%@example.com')).all()

# 정렬
users = User.query.order_by(User.created_at.desc()).all()

# 페이지네이션
page = User.query.paginate(page=1, per_page=10)
# page.items, page.has_next, page.has_prev, page.pages

# Update
user = User.query.filter_by(username='alice').first()
user.email = 'newemail@example.com'
db.session.commit()

# 또는 bulk update
User.query.filter_by(username='alice').update({'email': 'new@example.com'})
db.session.commit()

# Delete
user = User.query.filter_by(username='alice').first()
db.session.delete(user)
db.session.commit()

# 또는 bulk delete
User.query.filter(User.created_at < some_date).delete()
db.session.commit()
```

### 5.3 라우트에서 사용

```python
from flask import Flask, render_template, request, redirect, url_for, flash, abort

@app.route('/users')
def user_list():
    page = request.args.get('page', 1, type=int)
    users = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template('users/list.html', users=users)

@app.route('/users/<int:user_id>')
def user_detail(user_id):
    user = db.session.get(User, user_id)
    if user is None:
        abort(404)
    return render_template('users/detail.html', user=user)

@app.route('/users/create', methods=['GET', 'POST'])
def user_create():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']

        # 중복 확인
        if User.query.filter_by(username=username).first():
            flash('이미 존재하는 사용자명입니다.', 'error')
            return redirect(url_for('user_create'))

        user = User(username=username, email=email)
        db.session.add(user)
        db.session.commit()

        flash('사용자가 생성되었습니다.', 'success')
        return redirect(url_for('user_detail', user_id=user.id))

    return render_template('users/create.html')

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
def user_edit(user_id):
    user = db.session.get(User, user_id)
    if user is None:
        abort(404)

    if request.method == 'POST':
        user.email = request.form['email']
        db.session.commit()
        flash('사용자 정보가 수정되었습니다.', 'success')
        return redirect(url_for('user_detail', user_id=user.id))

    return render_template('users/edit.html', user=user)

@app.route('/users/<int:user_id>/delete', methods=['POST'])
def user_delete(user_id):
    user = db.session.get(User, user_id)
    if user is None:
        abort(404)

    db.session.delete(user)
    db.session.commit()
    flash('사용자가 삭제되었습니다.', 'success')
    return redirect(url_for('user_list'))
```

### 5.4 마이그레이션 (Flask-Migrate)

```bash
pip install flask-migrate
```

```python
# app.py
from flask_migrate import Migrate

app = Flask(__name__)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
```

```bash
# 마이그레이션 초기화
flask db init

# 마이그레이션 생성
flask db migrate -m "Initial migration"

# 마이그레이션 적용
flask db upgrade

# 롤백
flask db downgrade
```

---

## 6. 세션과 쿠키

### 6.1 세션 사용

```python
from flask import Flask, session, redirect, url_for, request

app = Flask(__name__)
app.secret_key = 'your-secret-key'

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    # 인증 로직...
    session['user_id'] = user.id
    session['username'] = user.username
    session.permanent = True  # 세션 유지 (기본 31일)
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    # 또는 특정 키만 삭제
    # session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return f'Welcome, {session["username"]}!'
```

### 6.2 쿠키 사용

```python
from flask import make_response, request

@app.route('/set-cookie')
def set_cookie():
    response = make_response('Cookie set!')
    response.set_cookie(
        'theme',
        'dark',
        max_age=60*60*24*30,  # 30일
        httponly=True,
        secure=True,  # HTTPS에서만
        samesite='Lax'
    )
    return response

@app.route('/get-cookie')
def get_cookie():
    theme = request.cookies.get('theme', 'light')
    return f'Current theme: {theme}'

@app.route('/delete-cookie')
def delete_cookie():
    response = make_response('Cookie deleted!')
    response.delete_cookie('theme')
    return response
```

---

## 7. 블루프린트 (Blueprints)

### 7.1 블루프린트 구조

```
my_app/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── forms.py
│   │   └── templates/
│   │       └── auth/
│   │           ├── login.html
│   │           └── register.html
│   ├── main/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── templates/
│   │       └── main/
│   │           └── index.html
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── templates/
│   │   └── base.html
│   └── static/
│       ├── css/
│       └── js/
├── config.py
├── requirements.txt
└── run.py
```

### 7.2 블루프린트 정의

```python
# app/auth/__init__.py
from flask import Blueprint

auth_bp = Blueprint('auth', __name__,
                    template_folder='templates',
                    url_prefix='/auth')

from . import routes
```

```python
# app/auth/routes.py
from flask import render_template, redirect, url_for, flash, request
from . import auth_bp
from .forms import LoginForm, RegistrationForm

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # 로그인 로직
        return redirect(url_for('main.index'))
    return render_template('auth/login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # 회원가입 로직
        flash('회원가입이 완료되었습니다.')
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', form=form)

@auth_bp.route('/logout')
def logout():
    # 로그아웃 로직
    return redirect(url_for('main.index'))
```

```python
# app/main/__init__.py
from flask import Blueprint

main_bp = Blueprint('main', __name__,
                    template_folder='templates')

from . import routes
```

```python
# app/main/routes.py
from flask import render_template
from . import main_bp

@main_bp.route('/')
def index():
    return render_template('main/index.html')

@main_bp.route('/about')
def about():
    return render_template('main/about.html')
```

### 7.3 애플리케이션 팩토리

```python
# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config

db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 확장 초기화
    db.init_app(app)
    migrate.init_app(app, db)

    # 블루프린트 등록
    from app.auth import auth_bp
    from app.main import main_bp
    from app.api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    # 에러 핸들러 등록
    from app.errors import register_error_handlers
    register_error_handlers(app)

    return app
```

```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
```

```python
# run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 8. RESTful API

### 8.1 JSON 응답

```python
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# 샘플 데이터
books = [
    {'id': 1, 'title': 'Flask Web Development', 'author': 'Miguel Grinberg'},
    {'id': 2, 'title': 'Python Crash Course', 'author': 'Eric Matthes'},
]

@app.route('/api/books', methods=['GET'])
def get_books():
    return jsonify({'books': books})

@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if book is None:
        abort(404)
    return jsonify(book)

@app.route('/api/books', methods=['POST'])
def create_book():
    if not request.json or 'title' not in request.json:
        abort(400)

    book = {
        'id': books[-1]['id'] + 1 if books else 1,
        'title': request.json['title'],
        'author': request.json.get('author', '')
    }
    books.append(book)
    return jsonify(book), 201

@app.route('/api/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if book is None:
        abort(404)
    if not request.json:
        abort(400)

    book['title'] = request.json.get('title', book['title'])
    book['author'] = request.json.get('author', book['author'])
    return jsonify(book)

@app.route('/api/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if book is None:
        abort(404)
    books.remove(book)
    return '', 204

# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400
```

### 8.2 API 인증 (JWT)

```bash
pip install flask-jwt-extended
```

```python
from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from datetime import timedelta

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-jwt-secret'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

jwt = JWTManager(app)

# 토큰 블랙리스트 (로그아웃된 토큰)
blacklist = set()

@jwt.token_in_blocklist_loader
def check_if_token_in_blocklist(jwt_header, jwt_payload):
    return jwt_payload['jti'] in blacklist

@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    # 사용자 인증 로직
    user = authenticate(username, password)
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    access_token = create_access_token(identity=user.id)
    refresh_token = create_refresh_token(identity=user.id)

    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token
    })

@app.route('/api/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    return jsonify({'access_token': access_token})

@app.route('/api/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()['jti']
    blacklist.add(jti)
    return jsonify({'message': 'Logged out'})

@app.route('/api/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    return jsonify({'user_id': current_user_id})
```

### 8.3 API 문서화 (Flask-RESTX)

```bash
pip install flask-restx
```

```python
from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Book API',
          description='A simple Book API')

ns = api.namespace('books', description='Book operations')

book_model = api.model('Book', {
    'id': fields.Integer(readonly=True, description='Unique identifier'),
    'title': fields.String(required=True, description='Book title'),
    'author': fields.String(required=True, description='Book author'),
})

books = []

@ns.route('/')
class BookList(Resource):
    @ns.doc('list_books')
    @ns.marshal_list_with(book_model)
    def get(self):
        '''List all books'''
        return books

    @ns.doc('create_book')
    @ns.expect(book_model)
    @ns.marshal_with(book_model, code=201)
    def post(self):
        '''Create a new book'''
        book = api.payload
        book['id'] = len(books) + 1
        books.append(book)
        return book, 201

@ns.route('/<int:id>')
@ns.response(404, 'Book not found')
@ns.param('id', 'The book identifier')
class Book(Resource):
    @ns.doc('get_book')
    @ns.marshal_with(book_model)
    def get(self, id):
        '''Get a book by ID'''
        book = next((b for b in books if b['id'] == id), None)
        if book is None:
            api.abort(404, 'Book not found')
        return book

    @ns.doc('delete_book')
    @ns.response(204, 'Book deleted')
    def delete(self, id):
        '''Delete a book'''
        global books
        books = [b for b in books if b['id'] != id]
        return '', 204

if __name__ == '__main__':
    app.run(debug=True)
    # Swagger UI: http://localhost:5000/
```

---

## 9. 에러 처리

### 9.1 에러 핸들러

```python
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()  # 트랜잭션 롤백
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('errors/500.html'), 500

@app.errorhandler(403)
def forbidden(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Forbidden'}), 403
    return render_template('errors/403.html'), 403

# 커스텀 예외
class ValidationError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__()
        self.message = message
        self.status_code = status_code

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    response = jsonify({'error': error.message})
    response.status_code = error.status_code
    return response

# 사용
@app.route('/api/validate', methods=['POST'])
def validate():
    if 'name' not in request.json:
        raise ValidationError('Name is required')
    return jsonify({'valid': True})
```

### 9.2 로깅

```python
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(app):
    if not app.debug:
        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler(
            'logs/app.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')

# 사용
@app.route('/api/action')
def action():
    app.logger.info('Action performed by user')
    try:
        # 로직
        pass
    except Exception as e:
        app.logger.error(f'Error occurred: {e}')
        raise
```

---

## 10. 테스트

### 10.1 pytest로 테스트

```python
# tests/conftest.py
import pytest
from app import create_app, db

@pytest.fixture
def app():
    app = create_app('config.TestingConfig')

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()
```

```python
# tests/test_routes.py
def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data

def test_login_page(client):
    response = client.get('/auth/login')
    assert response.status_code == 200

def test_login(client):
    response = client.post('/auth/login', data={
        'username': 'testuser',
        'password': 'testpass'
    }, follow_redirects=True)
    assert response.status_code == 200

def test_api_get_books(client):
    response = client.get('/api/books')
    assert response.status_code == 200
    assert response.content_type == 'application/json'

def test_api_create_book(client):
    response = client.post('/api/books',
        json={'title': 'Test Book', 'author': 'Test Author'}
    )
    assert response.status_code == 201
    data = response.get_json()
    assert data['title'] == 'Test Book'
```

### 10.2 테스트 실행

```bash
# 테스트 실행
pytest

# 상세 출력
pytest -v

# 커버리지
pip install pytest-cov
pytest --cov=app --cov-report=html
```

---

## 11. 배포

### 11.1 프로덕션 설정

```python
# config.py
import os

class ProductionConfig:
    SECRET_KEY = os.environ['SECRET_KEY']
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 보안 설정
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
```

### 11.2 Gunicorn으로 실행

```bash
pip install gunicorn

# 실행
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"

# 또는 gunicorn.conf.py 사용
gunicorn -c gunicorn.conf.py "app:create_app()"
```

```python
# gunicorn.conf.py
bind = '0.0.0.0:8000'
workers = 4
threads = 2
worker_class = 'gthread'
timeout = 120
keepalive = 5
errorlog = 'logs/gunicorn-error.log'
accesslog = 'logs/gunicorn-access.log'
loglevel = 'info'
```

### 11.3 Docker 배포

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=run.py
ENV FLASK_ENV=production

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:create_app()"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/app
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=app
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## 12. 실전 예제: 블로그 애플리케이션

전체 구조와 주요 파일을 포함한 완전한 블로그 앱 예제는 `examples/Web_Development/flask_blog/` 폴더를 참조하세요.

---

## 연습 문제

### 연습 1: 기본 CRUD 앱
간단한 할 일 목록(Todo) 앱을 만드세요:
- 할 일 목록 보기
- 새 할 일 추가
- 완료 표시
- 삭제

### 연습 2: 사용자 인증
위 앱에 사용자 인증을 추가하세요:
- 회원가입/로그인/로그아웃
- 각 사용자는 자신의 할 일만 볼 수 있음

### 연습 3: REST API
할 일 목록 API를 만드세요:
- CRUD 엔드포인트
- JWT 인증
- Swagger 문서화

---

## 참고 자료

- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [Flask-SQLAlchemy 문서](https://flask-sqlalchemy.palletsprojects.com/)
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Real Python Flask Tutorials](https://realpython.com/tutorials/flask/)
