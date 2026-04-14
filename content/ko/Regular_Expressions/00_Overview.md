# 정규 표현식 학습 가이드

## 소개

이 폴더는 정규 표현식(regex)을 학습하기 위한 자료를 담고 있습니다. 정규 표현식은 프로그래머에게 제공되는 가장 강력한 텍스트 처리 도구 중 하나입니다. 단순한 패턴 매칭부터 복잡한 텍스트 변환까지, 데이터 검증, 로그 분석, 검색-치환 작업, 데이터 정제 등에 필수적인 기술입니다.

**대상 독자**: 패턴 매칭과 텍스트 처리를 마스터하고 싶은 개발자

---

## 학습 로드맵

```
[기초]                  [핵심 문법]             [고급]                  [실전]
    |                       |                       |                       |
    v                       v                       v                       v
정규식이란 ──────────> 문자 클래스 ──────────> 전방/후방 ──────────> 자주 쓰는
    |                       |                   탐색                    패턴
    v                       v                       |                       |
리터럴과 ────────────> 수량자                       v                       v
메타문자                    |                   치환과                  성능과
    |                       v                   분할                    함정
    |                   앵커와                      |                       |
    |                   경계                        v                       v
    |                       |                   플래그와                실전
    |                       v                   옵션                    활용
    +────────────────> 그룹과
                        캡처
```

---

## 선수 지식

- 기본 Python 프로그래밍 ([프로그래밍](../Programming/00_Overview.md))
- 문자열과 문자열 메서드에 대한 기본 이해

---

## 배울 내용

- 정규 표현식의 내부 동작 원리 (유한 오토마타 개념)
- 리터럴 텍스트 매칭과 메타문자 사용법
- 문자 클래스, 축약 표기법, 유니코드 지원
- 수량자 (탐욕적, 게으른, 소유적)의 동작 방식
- 앵커, 단어 경계, 멀티라인 매칭
- 캡처 그룹, 역참조, 명명 그룹
- 전방탐색과 후방탐색 단언
- 텍스트 치환, 분할, 콜백 기반 치환
- 정규식 플래그와 인라인 수정자
- 일반적인 검증 작업을 위한 검증된 패턴
- 성능 최적화와 치명적 역추적 방지
- 로그 파싱, 데이터 정제, 리팩토링에서의 실전 활용

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [정규 표현식이란](./01_What_Are_Regular_Expressions.md) | ⭐ | 역사, 용도, Python re 모듈 기초 |
| [리터럴 매칭과 메타문자](./02_Literal_Matching_and_Metacharacters.md) | ⭐ | 리터럴 텍스트, `.`, `^`, `$`, `\|`, 이스케이프 |
| [문자 클래스](./03_Character_Classes.md) | ⭐ | `[abc]`, 범위, `\d`, `\w`, `\s`, 부정 |
| [수량자](./04_Quantifiers.md) | ⭐⭐ | `*`, `+`, `?`, `{n,m}`, 탐욕적 vs 게으른 |
| [앵커와 경계](./05_Anchors_and_Boundaries.md) | ⭐⭐ | `^`, `$`, `\b`, `\B`, 멀티라인 앵커 |
| [그룹과 캡처](./06_Groups_and_Capturing.md) | ⭐⭐ | `()`, 역참조, 명명 그룹, 비캡처 그룹 |
| [전방탐색과 후방탐색](./07_Lookahead_and_Lookbehind.md) | ⭐⭐⭐ | `(?=)`, `(?!)`, `(?<=)`, `(?<!)` |
| [치환과 분할](./08_Substitution_and_Splitting.md) | ⭐⭐ | `re.sub`, `re.split`, 콜백 함수 |
| [플래그와 옵션](./09_Flags_and_Options.md) | ⭐⭐ | `IGNORECASE`, `MULTILINE`, `DOTALL`, `VERBOSE` |
| [자주 쓰는 패턴](./10_Common_Patterns.md) | ⭐⭐ | 이메일, URL, IP, 날짜, 전화번호 검증 |
| [성능과 함정](./11_Performance_and_Pitfalls.md) | ⭐⭐⭐ | 치명적 역추적, 최적화 기법 |
| [실전 활용](./12_Real_World_Applications.md) | ⭐⭐⭐ | 로그 파싱, 데이터 정제, 코드 리팩토링 |

---

## 추천 학습 경로

### 1단계: 기초 (1~3강)
1. 정규 표현식이란 -> 리터럴 매칭 -> 문자 클래스

### 2단계: 핵심 문법 (4~6강)
2. 수량자 -> 앵커와 경계 -> 그룹과 캡처

### 3단계: 고급 패턴 (7~9강)
3. 전방/후방탐색 -> 치환과 분할 -> 플래그와 옵션

### 4단계: 실전 마스터 (10~12강)
4. 자주 쓰는 패턴 -> 성능과 함정 -> 실전 활용

---

## 빠른 시작

### Python에서 정규식 테스트

```python
import re

# 텍스트에서 이메일 주소 찾기
text = "Contact us at support@example.com or sales@example.com"
emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
print(emails)  # ['support@example.com', 'sales@example.com']
```

### 대화형 연습

```python
import re

pattern = r'\d{3}-\d{4}'
text = "Call 555-1234 or 555-5678"

for match in re.finditer(pattern, text):
    print(f"Found: {match.group()} at position {match.start()}-{match.end()}")
```

---

## 관련 자료

- [Python 기초](../Python_Basics/00_Overview.md) - Python 기본 문법
- [Shell Script](../Shell_Script/00_Overview.md) - grep과 sed에서 정규식 활용
- [Data Science](../Data_Science/00_Overview.md) - 정규식을 이용한 텍스트 전처리
