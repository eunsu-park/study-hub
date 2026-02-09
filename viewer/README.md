# Study Viewer (웹 뷰어)

Flask 기반 Markdown 학습 자료 뷰어입니다.

A Flask-based Markdown study material viewer.

## 기능 / Features

- Markdown 렌더링 (Pygments 코드 하이라이팅) / Markdown rendering with Pygments syntax highlighting
- 전체 텍스트 검색 (SQLite FTS5) / Full-text search (SQLite FTS5)
- 학습 진행률 추적 / Learning progress tracking
- 북마크 / Bookmarks
- 다크/라이트 모드 / Dark/Light mode
- 다국어 지원 (한국어/영어) / Multilingual support (Korean/English)
- 레슨 상/하단 통합 툴바 (네비게이션 + 액션 버튼) / Unified top/bottom lesson toolbar (navigation + action buttons)
- 키보드 단축키 (←/→ 레슨 이동) / Keyboard shortcuts (←/→ lesson navigation)
- 맨 위로 스크롤 플로팅 버튼 / Scroll-to-top floating button

## 설치 및 실행 / Installation & Running

```bash
cd viewer

# 의존성 설치 / Install dependencies
pip install -r requirements.txt

# 데이터베이스 초기화 / Initialize database
flask --app app init-db

# 검색 인덱스 빌드 / Build search index
python build_index.py

# 서버 실행 (기본 포트: 5000) / Run server (default port: 5000)
flask run

# 포트 변경 / Change port
flask run --port 5050

# 디버그 모드 / Debug mode
flask run --debug --port 5050
```

브라우저에서 http://127.0.0.1:5050 접속

Access http://127.0.0.1:5050 in your browser

## 포트 설정 / Port Configuration

### 방법 1: 명령줄 옵션 / Method 1: Command Line Option
```bash
flask run --port 5050
```

### 방법 2: 환경 변수 / Method 2: Environment Variable
```bash
export FLASK_RUN_PORT=5050
flask run
```

### 방법 3: .flaskenv 파일 / Method 3: .flaskenv File
```bash
# viewer/.flaskenv 생성 / Create viewer/.flaskenv
echo "FLASK_RUN_PORT=5050" > .flaskenv
flask run
```

## 프로젝트 구조 / Project Structure

```
viewer/
├── app.py              # Flask 메인 앱 / Flask main app
├── config.py           # 설정 / Configuration
├── models.py           # SQLAlchemy 모델 / SQLAlchemy models
├── build_index.py      # 검색 인덱스 빌드 / Search index builder
├── requirements.txt    # 의존성 / Dependencies
├── data.db             # SQLite DB (자동 생성 / auto-generated)
├── templates/          # Jinja2 템플릿 / Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── topic.html
│   ├── lesson.html
│   ├── search.html
│   ├── dashboard.html
│   └── bookmarks.html
├── static/             # 정적 파일 / Static files
│   ├── css/
│   └── js/
└── utils/              # 유틸리티 / Utilities
    ├── markdown_parser.py
    └── search.py
```

## API 엔드포인트 / API Endpoints

| 메서드 / Method | 경로 / Path | 설명 / Description |
|-----------------|-------------|-------------------|
| GET | `/<lang>/` | 토픽 목록 / Topic list |
| GET | `/<lang>/topic/<name>` | 레슨 목록 / Lesson list |
| GET | `/<lang>/topic/<name>/lesson/<file>` | 레슨 내용 / Lesson content |
| GET | `/<lang>/search?q=<query>` | 검색 / Search |
| GET | `/<lang>/dashboard` | 진행률 대시보드 / Progress dashboard |
| GET | `/<lang>/bookmarks` | 북마크 목록 / Bookmark list |
| POST | `/api/mark-read` | 읽음 표시 / Mark as read |
| POST | `/api/bookmark` | 북마크 토글 / Toggle bookmark |

## 의존성 / Dependencies

- Flask 3.x
- Flask-SQLAlchemy
- Markdown + Pygments
- python-frontmatter
