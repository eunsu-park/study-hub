# LaTeX

문서 구조, 수식 조판, 그래픽, 프레젠테이션 및 자동화를 다루는 LaTeX 문서 작성 시스템에 대한 종합 가이드.

## 레슨

| # | 제목 | 설명 |
|---|-------|-------------|
| 01 | [소개 및 설치](01_Introduction_and_Setup.md) | TeX/LaTeX 개요, 설치, 편집기, 첫 문서 |
| 02 | [문서 구조](02_Document_Structure.md) | 문서 클래스, 전문부(preamble), 섹션 구분, 구성 |
| 03 | [텍스트 서식](03_Text_Formatting.md) | 글꼴, 크기, 강조, 색상, 간격 |
| 04 | [수학 조판 기초](04_Math_Basics.md) | 인라인/디스플레이 수식, 그리스 문자, 분수, 구분자, 기본 기호 |
| 05 | [고급 수학](05_Math_Advanced.md) | 다행 방정식, 행렬, 정리 환경, amsmath |
| 06 | [부동체, 그림 & 표](06_Floats_and_Figures.md) | 부동체 시스템, graphicx, 부분그림, 캡션, 위치 지정 |
| 07 | [고급 표](07_Tables_Advanced.md) | Booktabs, multirow/multicolumn, 색상 셀, 긴 표 |
| 08 | [상호 참조 & 인용](08_Cross_References.md) | 레이블, 참조, BibTeX, BibLaTeX, 인용 스타일 |
| 09 | [페이지 레이아웃 & 타이포그래피](09_Page_Layout.md) | Geometry, fancyhdr, 줄 간격, 들여쓰기, 글꼴 |
| 10 | [TikZ 그래픽 기초](10_TikZ_Basics.md) | 좌표계, 기본 도형, 스타일, 노드, 레이블 |
| 11 | [고급 TikZ & PGFPlots](11_TikZ_Advanced.md) | 반복문, 트리, 그래프, PGFPlots, 과학적 삽화 |
| 12 | [Beamer 프레젠테이션](12_Beamer_Presentations.md) | 프레임, 테마, 오버레이, 점진적 콘텐츠, 유인물 |
| 13 | [사용자 정의 명령](13_Custom_Commands.md) | 매크로, 환경, 카운터, 개인 패키지 |
| 14 | [문서 클래스](14_Document_Classes.md) | Article, report, book, KOMA-Script, 논문 템플릿 |
| 15 | [자동화 및 빌드](15_Automation_and_Build.md) | latexmk, arara, Makefile, CI/CD, 버전 관리 |
| 16 | [실전 프로젝트](16_Practical_Projects.md) | 완전한 예제: 논문, 프레젠테이션, 포스터 |

## 사전 요구 사항

- 기본적인 컴퓨터 활용 능력
- 텍스트 편집기 또는 Overleaf 계정
- TeX Live 또는 MiKTeX 설치 (로컬 컴파일용)

## 학습 경로

1. **초보자**: 레슨 01-03으로 문서 구조와 서식 학습
2. **학술 논문 작성**: 레슨 04-08에서 수식, 그림, 표, 참조 집중
3. **페이지 디자인**: 레슨 09에서 레이아웃과 타이포그래피 커스터마이징
4. **그래픽 & 프레젠테이션**: 레슨 10-12에서 TikZ와 Beamer
5. **고급**: 레슨 13-15에서 사용자 정의 명령, 문서 클래스, 빌드 자동화
6. **통합**: 레슨 16은 모든 것을 실제 프로젝트에서 통합

## 예제 코드

실용적인 예제는 `examples/LaTeX/`에서 확인할 수 있습니다.

## 리소스

- **공식 문서**: [LaTeX Project](https://www.latex-project.org/)
- **CTAN**: 패키지를 위한 Comprehensive TeX Archive Network
- **TeX StackExchange**: 문제 해결을 위한 커뮤니티 Q&A
- **Overleaf**: 광범위한 문서 및 템플릿을 제공하는 온라인 편집기
- **서적**: "The LaTeX Companion" (Mittelbach et al.), "Guide to LaTeX" (Kopka & Daly)

## 왜 LaTeX인가?

- **전문적인 조판**: 출판 품질의 문서
- **수식의 우수성**: 복잡한 방정식을 위한 최고의 도구
- **구조화된 콘텐츠**: 서식이 아닌 내용에 집중
- **버전 관리**: 일반 텍스트는 Git과 원활하게 작동
- **재현성**: 동일한 소스는 동일한 출력 생성
- **자유 및 오픈 소스**: 라이선스 비용 없음, 활발한 커뮤니티

## 라이선스

이 콘텐츠는 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)에 따라 라이선스가 부여됩니다.
