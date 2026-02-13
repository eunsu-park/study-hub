# Study Content (학습 자료)

학습 자료 Markdown 파일 모음입니다. 한국어(ko)와 영어(en) 두 언어로 관리됩니다.

A collection of study material Markdown files. Maintained in both Korean (ko) and English (en).

## 디렉토리 구조 / Directory Structure

```
content/
├── ko/                    # 한국어 자료 (번역본) / Korean (translated)
│   ├── Algorithm/         # 29개 레슨 / 29 lessons
│   ├── C_Programming/     # 20개 레슨 / 20 lessons
│   └── ...                # 총 31개 토픽 / 31 topics total
│
└── en/                    # 영어 자료 (원본) / English (original)
    ├── Algorithm/
    ├── C_Programming/
    └── ...                # 총 31개 토픽 / 31 topics total
```

## 토픽 목록 / Topic List

전체 토픽 목록과 레슨 수는 프로젝트 루트의 [README.md](../README.md#목차--table-of-contents)를 참조하세요.

See the project root [README.md](../README.md#목차--table-of-contents) for the full topic list with lesson counts.

각 토픽의 `00_Overview.md`에서 학습 로드맵과 상세 목차를 확인할 수 있습니다.

Check the `00_Overview.md` in each topic for learning roadmaps and detailed table of contents.

## 보기 방법 / How to View

1. **웹 뷰어 / Web Viewer**: `viewer/` 폴더의 Flask 앱 실행 / Run Flask app in `viewer/`
2. **직접 열기 / Direct Open**: VS Code 등에서 Markdown 파일 열기 / Open Markdown files in VS Code
3. **GitHub**: GitHub에서 직접 렌더링 / Render directly on GitHub

## 파일 명명 규칙 / File Naming Convention

- `00_Overview.md` - 토픽 개요 및 목차 / Topic overview and table of contents
- `01_*.md`, `02_*.md`, ... - 순차적 레슨 (숫자 접두사로 순서 지정) / Sequential lessons (numbered prefix)
- 토픽 폴더명 / Topic folders: `Upper_Snake_Case` (예 / e.g., `C_Programming`, `Deep_Learning`)
- 레슨 파일명 / Lesson files: 영문, 공백은 언더스코어로 대체 / English, spaces replaced with underscores (예 / e.g., `01_Getting_Started.md`)

## 예제 코드 / Example Code

예제 코드 파일은 프로젝트 루트의 [`examples/`](../examples/README.md) 폴더에 위치합니다.

Example code files are located in the [`examples/`](../examples/README.md) folder at the project root.

## 다국어 정책 / Multilingual Policy

- **영어(en)**: 원본 자료 (신규 작성 기준). 모든 토픽에 존재
  **English (en)**: Original materials (for new content). Present in all topics
- **한국어(ko)**: 번역본 (신규 작성 기준). 모든 토픽에 존재
  **Korean (ko)**: Translated (for new content). Present in all topics
- 신규 자료는 영문 원본 작성 후 한국어 번역
  New materials are authored in English, then translated to Korean
- 기존 30개 토픽은 한국어 원본 기반으로 작성됨 (Shell_Script부터 영문 원본)
  Existing 30 topics were originally authored in Korean (Shell_Script onwards in English)
