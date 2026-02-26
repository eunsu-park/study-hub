# 파이썬 학습 가이드

## 소개

이 폴더는 파이썬의 기초부터 고급 문법까지 체계적으로 학습하기 위한 자료를 담고 있습니다.

**대상 독자**:
- **기초 복습 필요**: 15, 16번 레슨(기초 문법, OOP)부터 시작
- **중급 이상**: 01번 레슨(타입 힌팅)부터 시작

---

## 학습 로드맵

```
[중급]                [중급+]               [고급]
  │                     │                     │
  ▼                     ▼                     ▼
타입 힌팅 ──────▶ 이터레이터 ─────▶ 디스크립터
  │                     │                     │
  ▼                     ▼                     ▼
데코레이터 ─────▶ 클로저 ─────────▶ 비동기
  │                     │                     │
  ▼                     ▼                     ▼
컨텍스트 ───────▶ 메타클래스 ────▶ 함수형
                                              │
                                              ▼
                                         성능 최적화
```

---

## 선수 지식

- 파이썬 기본 문법 (변수, 자료형, 제어문, 함수)
- 객체지향 프로그래밍 기초 (클래스, 상속, 메서드)
- 모듈과 패키지 사용법

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [타입 힌팅 (Type Hints)](./01_Type_Hints.md) | ⭐⭐ | Type Hints, typing 모듈, mypy |
| [데코레이터 (Decorators)](./02_Decorators.md) | ⭐⭐ | 함수/클래스 데코레이터, @wraps |
| [컨텍스트 매니저 (Context Managers)](./03_Context_Managers.md) | ⭐⭐ | with문, contextlib |
| [이터레이터와 제너레이터 (Iterators & Generators)](./04_Iterators_and_Generators.md) | ⭐⭐⭐ | __iter__, yield, itertools |
| [클로저와 스코프 (Closures & Scope)](./05_Closures_and_Scope.md) | ⭐⭐⭐ | LEGB, nonlocal, 클로저 패턴 |
| [메타클래스 (Metaclasses)](./06_Metaclasses.md) | ⭐⭐⭐ | type, __new__, __init_subclass__ |
| [디스크립터 (Descriptors)](./07_Descriptors.md) | ⭐⭐⭐⭐ | __get__, __set__, property 구현 |
| [비동기 프로그래밍 (Async Programming)](./08_Async_Programming.md) | ⭐⭐⭐⭐ | async/await, asyncio |
| [함수형 프로그래밍 (Functional Programming)](./09_Functional_Programming.md) | ⭐⭐⭐⭐ | map, filter, functools |
| [성능 최적화 (Performance Optimization)](./10_Performance_Optimization.md) | ⭐⭐⭐⭐ | 프로파일링, 최적화 기법 |
| [테스트 및 품질 관리 (Testing & Quality Assurance)](./11_Testing_and_Quality.md) | ⭐⭐⭐ | pytest, fixtures, mocking, coverage |
| [패키징 및 배포 (Packaging & Distribution)](./12_Packaging_and_Distribution.md) | ⭐⭐⭐ | pyproject.toml, Poetry, PyPI |
| [데이터클래스 (Dataclasses)](./13_Dataclasses.md) | ⭐⭐ | @dataclass, field(), frozen |
| [패턴 매칭 (Pattern Matching)](./14_Pattern_Matching.md) | ⭐⭐⭐ | match/case, 구조 패턴, 가드 |
| [파이썬 기초 문법](./15_Python_Basics.md) | ⭐ | 변수, 자료형, 제어문, 함수, 자료구조 (선수 지식 복습) |
| [객체지향 프로그래밍 기초](./16_OOP_Basics.md) | ⭐⭐ | 클래스, 상속, 캡슐화, 다형성 (선수 지식 복습) |

---

## 추천 학습 순서

### 기초 복습 (필요 시)
0. 기초 문법 → OOP 기초 (15 → 16)

### 중급 (기본 고급 문법)
1. 타입 힌팅 → 데코레이터 → 컨텍스트 매니저

### 중급+ (심화 문법)
2. 이터레이터/제너레이터 → 클로저 → 메타클래스

### 고급 (전문가 수준)
3. 디스크립터 → 비동기 → 함수형 → 성능 최적화

### 실무 (개발 도구)
4. 테스트 및 품질 관리 → 패키징 및 배포 → 데이터클래스 → 패턴 매칭

---

## 실습 환경

```bash
# Python 버전 확인 (3.10+ 권장)
python --version

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 타입 체커 설치 (선택)
pip install mypy
```

---

## 관련 자료

- [C_Programming/](../C_Programming/00_Overview.md) - 시스템 프로그래밍 기초
- [Linux/](../Linux/00_Overview.md) - 리눅스 환경에서의 개발
- [PostgreSQL/](../PostgreSQL/00_Overview.md) - 데이터베이스 연동
