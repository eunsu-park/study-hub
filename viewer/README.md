# Study Viewer (웹 뷰어)

Flask 기반 Markdown 학습 자료 뷰어입니다.

## 기능

- Markdown 렌더링 (Pygments 코드 하이라이팅)
- 전체 텍스트 검색 (SQLite FTS5)
- 학습 진행률 추적
- 북마크
- 다크/라이트 모드
- 다국어 지원 (한국어/영어)

## 설치 및 실행

```bash
cd viewer

# 의존성 설치
pip install -r requirements.txt

# 데이터베이스 초기화
flask --app app init-db

# 검색 인덱스 빌드
python build_index.py

# 서버 실행 (기본 포트: 5000)
flask run

# 포트 변경
flask run --port 5050

# 디버그 모드
flask run --debug --port 5050
```

브라우저에서 http://127.0.0.1:5050 접속

## 포트 설정

### 방법 1: 명령줄 옵션
```bash
flask run --port 5050
```

### 방법 2: 환경 변수
```bash
export FLASK_RUN_PORT=5050
flask run
```

### 방법 3: .flaskenv 파일
```bash
# viewer/.flaskenv 생성
echo "FLASK_RUN_PORT=5050" > .flaskenv
flask run
```

## 프로젝트 구조

```
viewer/
├── app.py              # Flask 메인 앱
├── config.py           # 설정
├── models.py           # SQLAlchemy 모델
├── build_index.py      # 검색 인덱스 빌드
├── requirements.txt    # 의존성
├── data.db             # SQLite DB (자동 생성)
├── templates/          # Jinja2 템플릿
│   ├── base.html
│   ├── index.html
│   ├── topic.html
│   ├── lesson.html
│   ├── search.html
│   ├── dashboard.html
│   └── bookmarks.html
├── static/             # 정적 파일
│   ├── css/
│   └── js/
└── utils/              # 유틸리티
    ├── markdown_parser.py
    └── search.py
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/<lang>/` | 토픽 목록 |
| GET | `/<lang>/topic/<name>` | 레슨 목록 |
| GET | `/<lang>/topic/<name>/lesson/<file>` | 레슨 내용 |
| GET | `/<lang>/search?q=<query>` | 검색 |
| GET | `/<lang>/dashboard` | 진행률 대시보드 |
| GET | `/<lang>/bookmarks` | 북마크 목록 |
| POST | `/api/mark-read` | 읽음 표시 |
| POST | `/api/bookmark` | 북마크 토글 |

## 의존성

- Flask 3.x
- Flask-SQLAlchemy
- Markdown + Pygments
- python-frontmatter
