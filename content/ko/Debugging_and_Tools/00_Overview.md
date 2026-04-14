# 디버깅과 개발 도구

## 소개

이 폴더는 코드 디버깅, 에러 진단, 필수 개발 도구 활용법을 학습하기 위한 자료를 담고 있습니다. 에러 메시지 읽기부터 체계적인 디버깅 전략, 프로파일링, 자동화된 품질 검사까지 단계별로 학습할 수 있습니다.

**대상 독자**: "안 돼요"를 넘어서 체계적으로 버그를 찾고 고치는 방법을 배우고 싶은 입문 개발자

---

## 학습 로드맵

```
[기초]                     [전략]                   [도구]
    |                         |                        |
    v                         v                        v
에러 읽기 ------------> 디버깅 전략 -----------> 린터와 포매터
    |                         |                        |
    v                         v                        v
print 디버깅 ----------> 로깅 -----------------> 타입 체킹
    |                         |                        |
    v                         v                        v
디버거 사용법            테스트 기초 -----------> 프로파일링 기초
    |                                                  |
    v                                                  v
흔한 버그 패턴           VCS 디버깅 -----------> 종합 워크플로우
```

---

## 배울 내용

- Python 트레이스백 읽기와 에러 메시지 해독
- print 문과 logging 모듈의 전략적 활용
- `pdb`, `breakpoint()`, IDE 디버거 실습
- 흔한 버그 패턴 인식 (off-by-one, 가변 기본값, 스코프 문제)
- 체계적 디버깅 전략 (이진 탐색, 최소 재현 예제)
- `assert`, `unittest`, `pytest`로 테스트 작성
- 린터(`pylint`, `flake8`, `ruff`)와 포매터(`black`) 활용
- 타입 힌트 추가 및 `mypy`로 검사
- `cProfile`, `timeit`, 메모리 프로파일러로 코드 프로파일링
- `git bisect`, `git blame`, `git diff`를 활용한 디버깅
- 모든 기법을 결합한 실전 디버깅 워크플로우

---

## 선수 지식

- 기본 Python 프로그래밍 (변수, 함수, 반복문, 클래스)
- 터미널/명령줄 기본 사용법
- 기본 Git 지식 (11과에서 활용)

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [에러 메시지 읽기](./01_Reading_Error_Messages.md) | ⭐ | 트레이스백 구조, 흔한 에러 유형, 스택 트레이스 읽기 |
| [print 디버깅](./02_Print_Debugging.md) | ⭐ | 전략적 print, f-string 활용, 임시 vs 영구 디버그 출력 |
| [디버거 사용법](./03_Using_a_Debugger.md) | ⭐ | pdb, breakpoint(), 스테핑, IDE 디버거 |
| [흔한 버그 패턴](./04_Common_Bug_Patterns.md) | ⭐⭐ | Off-by-one, 가변 기본값, 스코프, None 처리 |
| [디버깅 전략](./05_Debugging_Strategy.md) | ⭐⭐ | 이진 탐색, 재현, 최소 예제, 고무 오리 디버깅 |
| [로깅](./06_Logging.md) | ⭐⭐ | logging 모듈, 레벨, 포매터, 핸들러, 설정 |
| [테스트 기초](./07_Testing_Basics.md) | ⭐⭐ | assert, unittest, pytest, 테스트 주도 디버깅 |
| [린터와 포매터](./08_Linters_and_Formatters.md) | ⭐⭐ | pylint, flake8, black, ruff, pre-commit 훅 |
| [타입 체킹](./09_Type_Checking.md) | ⭐⭐ | 타입 힌트, mypy, 점진적 타이핑, 흔한 타입 에러 |
| [프로파일링 기초](./10_Profiling_Basics.md) | ⭐⭐⭐ | cProfile, timeit, memory_profiler, line_profiler |
| [디버깅을 위한 버전 관리](./11_Version_Control_for_Debugging.md) | ⭐⭐⭐ | git bisect, git blame, git diff, VCS 디버깅 |
| [종합 디버깅 워크플로우](./12_Debugging_Workflow.md) | ⭐⭐⭐ | 실전 케이스 스터디, 모든 기법 결합 |

---

## 추천 학습 순서

### 1단계: 에러 읽기와 기본 디버깅
1. 에러 메시지 읽기 -> print 디버깅 -> 디버거 사용법

### 2단계: 패턴과 전략
2. 흔한 버그 패턴 -> 디버깅 전략

### 3단계: 로깅과 테스트
3. 로깅 -> 테스트 기초

### 4단계: 자동화 도구
4. 린터와 포매터 -> 타입 체킹 -> 프로파일링 기초

### 5단계: 고급 워크플로우
5. 디버깅을 위한 버전 관리 -> 종합 디버깅 워크플로우

---

## 빠른 시작

### 디버거 체험하기

```python
# 이 코드를 buggy.py로 저장하고 디버깅해 보세요
def average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)

# 이 코드는 크래시됩니다 -- 왜일까요?
result = average([])
print(f"Average: {result}")
```

```bash
# 디버거로 실행
python -m pdb buggy.py
```

---

## 관련 자료

- [Programming](../Programming/00_Overview.md) - 프로그래밍 기초
- [Python Basics](../Python_Basics/00_Overview.md) - Python 언어 기초
- [Testing and QA](../Testing_and_QA/00_Overview.md) - 고급 테스트 기법
- [Git](../Git/00_Overview.md) - 디버깅을 위한 버전 관리
