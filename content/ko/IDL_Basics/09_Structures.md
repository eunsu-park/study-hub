# 구조체

**이전**: [파일 I/O](./08_File_IO.md) | **다음**: [기본 플로팅](./10_Basic_Plotting.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 익명 및 명명된 구조체 생성하기
2. 이름과 태그 번호로 구조체 필드에 접근하기
3. CREATE_STRUCT로 동적으로 구조체 구성하기
4. 구조체 배열 작업하기
5. 중첩 구조체 만들기
6. TAG_NAMES, N_TAGS, HELP로 구조체 검사하기
7. 구조체 상속 이해하기

---

IDL의 구조체는 관련 데이터를 하나의 변수로 그룹화하는 복합 데이터 타입입니다. C의 struct이나 Python의 딕셔너리에 해당합니다 (단, 한번 생성하면 필드가 고정됩니다).

## 익명 구조체

```idl
person = {name: 'Alice', age: 30, score: 98.5}
PRINT, person.name       ; Alice
PRINT, person.age        ;       30

; 필드 수정
person.name = 'Bob'

; 기존 구조체에 새 필드를 추가할 수 없습니다
```

## 명명된 구조체

```idl
star = {STAR, name: '', ra: 0.0D0, dec: 0.0D0, magnitude: 0.0}
sirius = {STAR, name: 'Sirius', ra: 101.287D, dec: -16.716D, magnitude: -1.46}
```

## CREATE_STRUCT — 동적 구성

```idl
s = CREATE_STRUCT('x', 1.0)
s = CREATE_STRUCT(s, 'y', 2.0)
s = CREATE_STRUCT(s, 'z', 3.0)

; 두 구조체 병합
s1 = {a: 1, b: 2}
s2 = {c: 3, d: 4}
merged = CREATE_STRUCT(s1, s2)
```

## 구조체 배열

```idl
n = 5
stars = REPLICATE({STAR}, n)
stars[0].name = 'Sirius'   & stars[0].magnitude = -1.46
stars[1].name = 'Canopus'  & stars[1].magnitude = -0.72

; 배열 전체의 필드 접근
PRINT, stars.name
PRINT, stars.magnitude

; WHERE로 필터링
bright = WHERE(stars.magnitude LT 0, count)
```

## 태그 번호로 필드 접근

```idl
s = {name: 'Alice', age: 30, score: 98.5}
PRINT, s.(0)     ; Alice   (s.name과 동일)
PRINT, s.(1)     ;    30   (s.age와 동일)

; 모든 필드 순회
FOR i = 0, N_TAGS(s) - 1 DO $
  PRINT, (TAG_NAMES(s))[i], ' = ', s.(i)
```

## 구조체 검사

```idl
tags = TAG_NAMES(s)
PRINT, tags          ; NAME  AGE  SCORE
PRINT, N_TAGS(s)     ;        3
```

## 구조체 상속

```idl
base = {CELESTIAL_OBJECT, name: '', ra: 0.0D0, dec: 0.0D0}
star_def = {STAR_OBJ, INHERITS CELESTIAL_OBJECT, magnitude: 0.0}
```

---

## 요약

| 개념 | 설명 |
|------|------|
| 익명 구조체 | `s = {field1: val1, field2: val2}` |
| 명명된 구조체 | `s = {TYPE_NAME, field1: val1, ...}` |
| 필드 접근 | `s.field_name` 또는 `s.(tag_number)` |
| CREATE_STRUCT | 동적으로 구조체 구성 |
| REPLICATE | 구조체 배열 생성 |
| TAG_NAMES(s) | 필드 이름 배열 가져오기 |
| N_TAGS(s) | 필드 수 가져오기 |
| INHERITS | 구조체 상속 |

---

**이전**: [파일 I/O](./08_File_IO.md) | **다음**: [기본 플로팅](./10_Basic_Plotting.md)
