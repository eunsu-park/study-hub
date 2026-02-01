# HTML 폼과 테이블

## 1. 폼(Form) 기초

폼은 사용자로부터 데이터를 입력받아 서버로 전송하는 요소입니다.

```html
<form action="/submit" method="POST">
    <label for="name">이름:</label>
    <input type="text" id="name" name="name">
    <button type="submit">전송</button>
</form>
```

### form 속성

| 속성 | 설명 | 예시 |
|------|------|------|
| `action` | 데이터를 보낼 URL | `/api/submit` |
| `method` | 전송 방식 | `GET`, `POST` |
| `enctype` | 인코딩 타입 | `multipart/form-data` (파일 업로드 시) |
| `autocomplete` | 자동완성 | `on`, `off` |
| `novalidate` | 유효성 검사 비활성화 | `novalidate` |

### GET vs POST

```html
<!-- GET: URL에 데이터 포함 (검색 등) -->
<form action="/search" method="GET">
    <input type="text" name="q">
    <!-- 전송 시: /search?q=검색어 -->
</form>

<!-- POST: 본문에 데이터 포함 (로그인, 회원가입 등) -->
<form action="/login" method="POST">
    <input type="text" name="username">
    <input type="password" name="password">
</form>
```

---

## 2. input 태그

### 텍스트 입력

```html
<!-- 한 줄 텍스트 -->
<input type="text" name="username" placeholder="사용자명">

<!-- 비밀번호 -->
<input type="password" name="password" placeholder="비밀번호">

<!-- 이메일 (형식 검증) -->
<input type="email" name="email" placeholder="example@email.com">

<!-- URL -->
<input type="url" name="website" placeholder="https://example.com">

<!-- 전화번호 -->
<input type="tel" name="phone" placeholder="010-1234-5678">

<!-- 검색 (X 버튼 표시) -->
<input type="search" name="search" placeholder="검색어 입력">
```

### 숫자 입력

```html
<!-- 숫자 -->
<input type="number" name="quantity" min="1" max="100" step="1">

<!-- 범위 슬라이더 -->
<input type="range" name="volume" min="0" max="100" value="50">
```

### 날짜/시간 입력

```html
<!-- 날짜 -->
<input type="date" name="birthday">

<!-- 시간 -->
<input type="time" name="meeting-time">

<!-- 날짜 + 시간 -->
<input type="datetime-local" name="appointment">

<!-- 월 -->
<input type="month" name="birth-month">

<!-- 주 -->
<input type="week" name="week">
```

### 선택 입력

```html
<!-- 체크박스 (다중 선택) -->
<input type="checkbox" id="agree" name="agree" value="yes">
<label for="agree">동의합니다</label>

<input type="checkbox" name="hobby" value="reading" id="h1">
<label for="h1">독서</label>
<input type="checkbox" name="hobby" value="music" id="h2">
<label for="h2">음악</label>
<input type="checkbox" name="hobby" value="sports" id="h3">
<label for="h3">운동</label>

<!-- 라디오 버튼 (단일 선택) -->
<input type="radio" name="gender" value="male" id="male">
<label for="male">남성</label>
<input type="radio" name="gender" value="female" id="female">
<label for="female">여성</label>
```

### 기타 입력

```html
<!-- 색상 선택 -->
<input type="color" name="color" value="#ff0000">

<!-- 파일 업로드 -->
<input type="file" name="document">
<input type="file" name="images" multiple accept="image/*">

<!-- 숨김 필드 -->
<input type="hidden" name="user_id" value="12345">
```

---

## 3. input 속성

### 기본 속성

```html
<input type="text"
       name="username"           <!-- 서버로 전송될 이름 -->
       id="username"             <!-- label 연결용 -->
       value="기본값"             <!-- 초기값 -->
       placeholder="힌트 텍스트"  <!-- 입력 전 안내 문구 -->
>
```

### 제한 속성

```html
<input type="text"
       maxlength="20"            <!-- 최대 글자 수 -->
       minlength="5"             <!-- 최소 글자 수 -->
       size="30"                 <!-- 표시 너비 (문자 수) -->
>

<input type="number"
       min="0"                   <!-- 최소값 -->
       max="100"                 <!-- 최대값 -->
       step="5"                  <!-- 증가 단위 -->
>
```

### 상태 속성

```html
<!-- 필수 입력 -->
<input type="text" name="name" required>

<!-- 읽기 전용 (전송됨) -->
<input type="text" name="code" value="ABC123" readonly>

<!-- 비활성화 (전송 안 됨) -->
<input type="text" name="old" value="이전 값" disabled>

<!-- 자동 포커스 -->
<input type="text" name="first" autofocus>

<!-- 자동 완성 -->
<input type="text" name="email" autocomplete="email">
```

### 패턴 검증

```html
<!-- 정규표현식 패턴 -->
<input type="text"
       name="phone"
       pattern="[0-9]{3}-[0-9]{4}-[0-9]{4}"
       title="000-0000-0000 형식으로 입력하세요">

<!-- 이메일 패턴 예시 -->
<input type="text"
       name="email"
       pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"
       title="올바른 이메일 형식으로 입력하세요">
```

---

## 4. label 태그

### label 연결 방법

```html
<!-- 방법 1: for 속성 사용 (권장) -->
<label for="email">이메일:</label>
<input type="email" id="email" name="email">

<!-- 방법 2: input을 label로 감싸기 -->
<label>
    이메일:
    <input type="email" name="email">
</label>
```

### label의 장점

1. **접근성**: 스크린 리더가 읽어줌
2. **사용성**: label 클릭 시 input에 포커스
3. **체크박스/라디오**: 텍스트 클릭으로 선택 가능

---

## 5. 기타 폼 요소

### textarea (여러 줄 텍스트)

```html
<label for="message">메시지:</label>
<textarea id="message"
          name="message"
          rows="5"
          cols="40"
          placeholder="내용을 입력하세요">
기본 텍스트
</textarea>
```

### select (드롭다운)

```html
<label for="country">국가:</label>
<select id="country" name="country">
    <option value="">선택하세요</option>
    <option value="kr">한국</option>
    <option value="us">미국</option>
    <option value="jp" selected>일본</option>
</select>

<!-- 그룹화 -->
<select name="car">
    <optgroup label="한국 자동차">
        <option value="hyundai">현대</option>
        <option value="kia">기아</option>
    </optgroup>
    <optgroup label="외국 자동차">
        <option value="bmw">BMW</option>
        <option value="benz">벤츠</option>
    </optgroup>
</select>

<!-- 다중 선택 -->
<select name="skills" multiple size="4">
    <option value="html">HTML</option>
    <option value="css">CSS</option>
    <option value="js">JavaScript</option>
    <option value="python">Python</option>
</select>
```

### datalist (자동완성 목록)

```html
<label for="browser">브라우저:</label>
<input type="text" id="browser" name="browser" list="browsers">
<datalist id="browsers">
    <option value="Chrome">
    <option value="Firefox">
    <option value="Safari">
    <option value="Edge">
</datalist>
```

### button

```html
<!-- 전송 버튼 -->
<button type="submit">전송</button>

<!-- 리셋 버튼 -->
<button type="reset">초기화</button>

<!-- 일반 버튼 (JavaScript 연동) -->
<button type="button" onclick="alert('클릭!')">클릭</button>

<!-- 이미지나 아이콘 포함 -->
<button type="submit">
    <img src="send.png" alt=""> 전송하기
</button>
```

---

## 6. fieldset과 legend

```html
<form>
    <fieldset>
        <legend>개인 정보</legend>

        <label for="name">이름:</label>
        <input type="text" id="name" name="name"><br>

        <label for="email">이메일:</label>
        <input type="email" id="email" name="email">
    </fieldset>

    <fieldset>
        <legend>배송 정보</legend>

        <label for="address">주소:</label>
        <input type="text" id="address" name="address"><br>

        <label for="phone">전화번호:</label>
        <input type="tel" id="phone" name="phone">
    </fieldset>

    <button type="submit">주문하기</button>
</form>
```

---

## 7. 폼 유효성 검사

### HTML5 기본 검증

```html
<form>
    <!-- 필수 입력 -->
    <input type="text" name="name" required>

    <!-- 이메일 형식 -->
    <input type="email" name="email" required>

    <!-- 최소/최대 길이 -->
    <input type="password" name="password"
           minlength="8" maxlength="20" required>

    <!-- 숫자 범위 -->
    <input type="number" name="age" min="1" max="120">

    <!-- 패턴 -->
    <input type="text" name="zipcode"
           pattern="[0-9]{5}"
           title="5자리 숫자를 입력하세요">

    <button type="submit">제출</button>
</form>
```

### 커스텀 에러 메시지

```html
<input type="email"
       name="email"
       required
       oninvalid="this.setCustomValidity('올바른 이메일을 입력하세요')"
       oninput="this.setCustomValidity('')">
```

---

## 8. 완전한 폼 예제

### 회원가입 폼

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>회원가입</title>
</head>
<body>
    <h1>회원가입</h1>

    <form action="/register" method="POST">
        <fieldset>
            <legend>계정 정보</legend>

            <p>
                <label for="username">아이디:</label>
                <input type="text" id="username" name="username"
                       required minlength="4" maxlength="20"
                       pattern="[a-zA-Z0-9]+"
                       title="영문자와 숫자만 사용 가능합니다">
            </p>

            <p>
                <label for="email">이메일:</label>
                <input type="email" id="email" name="email" required>
            </p>

            <p>
                <label for="password">비밀번호:</label>
                <input type="password" id="password" name="password"
                       required minlength="8">
            </p>

            <p>
                <label for="password2">비밀번호 확인:</label>
                <input type="password" id="password2" name="password2" required>
            </p>
        </fieldset>

        <fieldset>
            <legend>개인 정보</legend>

            <p>
                <label for="name">이름:</label>
                <input type="text" id="name" name="name" required>
            </p>

            <p>
                <label for="birthday">생년월일:</label>
                <input type="date" id="birthday" name="birthday">
            </p>

            <p>
                <label>성별:</label>
                <input type="radio" id="male" name="gender" value="male">
                <label for="male">남성</label>
                <input type="radio" id="female" name="gender" value="female">
                <label for="female">여성</label>
            </p>

            <p>
                <label for="phone">전화번호:</label>
                <input type="tel" id="phone" name="phone"
                       pattern="[0-9]{3}-[0-9]{4}-[0-9]{4}"
                       placeholder="010-1234-5678">
            </p>
        </fieldset>

        <fieldset>
            <legend>추가 정보</legend>

            <p>
                <label for="country">국가:</label>
                <select id="country" name="country">
                    <option value="">선택하세요</option>
                    <option value="kr" selected>한국</option>
                    <option value="us">미국</option>
                    <option value="jp">일본</option>
                </select>
            </p>

            <p>
                <label for="bio">자기소개:</label><br>
                <textarea id="bio" name="bio" rows="4" cols="40"></textarea>
            </p>

            <p>
                <label>관심 분야:</label><br>
                <input type="checkbox" id="web" name="interest" value="web">
                <label for="web">웹 개발</label>
                <input type="checkbox" id="mobile" name="interest" value="mobile">
                <label for="mobile">모바일</label>
                <input type="checkbox" id="ai" name="interest" value="ai">
                <label for="ai">AI</label>
            </p>
        </fieldset>

        <p>
            <input type="checkbox" id="agree" name="agree" required>
            <label for="agree">이용약관에 동의합니다 (필수)</label>
        </p>

        <p>
            <button type="submit">가입하기</button>
            <button type="reset">초기화</button>
        </p>
    </form>
</body>
</html>
```

---

## 9. 테이블 기초

### 기본 테이블

```html
<table>
    <tr>
        <th>이름</th>
        <th>나이</th>
        <th>직업</th>
    </tr>
    <tr>
        <td>홍길동</td>
        <td>30</td>
        <td>개발자</td>
    </tr>
    <tr>
        <td>김철수</td>
        <td>25</td>
        <td>디자이너</td>
    </tr>
</table>
```

### 테이블 태그

| 태그 | 설명 |
|------|------|
| `<table>` | 테이블 전체 |
| `<tr>` | 행 (table row) |
| `<th>` | 헤더 셀 (굵게 표시) |
| `<td>` | 데이터 셀 |
| `<thead>` | 헤더 영역 |
| `<tbody>` | 본문 영역 |
| `<tfoot>` | 푸터 영역 |
| `<caption>` | 테이블 제목 |

---

## 10. 테이블 구조화

```html
<table>
    <caption>2024년 1분기 매출</caption>

    <thead>
        <tr>
            <th>월</th>
            <th>매출</th>
            <th>비용</th>
            <th>이익</th>
        </tr>
    </thead>

    <tbody>
        <tr>
            <td>1월</td>
            <td>1,000만원</td>
            <td>600만원</td>
            <td>400만원</td>
        </tr>
        <tr>
            <td>2월</td>
            <td>1,200만원</td>
            <td>700만원</td>
            <td>500만원</td>
        </tr>
        <tr>
            <td>3월</td>
            <td>1,500만원</td>
            <td>800만원</td>
            <td>700만원</td>
        </tr>
    </tbody>

    <tfoot>
        <tr>
            <th>합계</th>
            <td>3,700만원</td>
            <td>2,100만원</td>
            <td>1,600만원</td>
        </tr>
    </tfoot>
</table>
```

---

## 11. 셀 병합

### colspan (열 병합)

```html
<table border="1">
    <tr>
        <th colspan="3">학생 정보</th>
    </tr>
    <tr>
        <th>이름</th>
        <th>국어</th>
        <th>영어</th>
    </tr>
    <tr>
        <td>홍길동</td>
        <td>90</td>
        <td>85</td>
    </tr>
</table>
```

### rowspan (행 병합)

```html
<table border="1">
    <tr>
        <th>이름</th>
        <td>홍길동</td>
    </tr>
    <tr>
        <th rowspan="2">연락처</th>
        <td>010-1234-5678</td>
    </tr>
    <tr>
        <td>email@example.com</td>
    </tr>
</table>
```

### 복합 병합

```html
<table border="1">
    <tr>
        <th rowspan="2">구분</th>
        <th colspan="2">성적</th>
        <th rowspan="2">평균</th>
    </tr>
    <tr>
        <th>국어</th>
        <th>영어</th>
    </tr>
    <tr>
        <td>홍길동</td>
        <td>90</td>
        <td>85</td>
        <td>87.5</td>
    </tr>
    <tr>
        <td>김철수</td>
        <td>80</td>
        <td>95</td>
        <td>87.5</td>
    </tr>
</table>
```

---

## 12. 테이블 접근성

### scope 속성

```html
<table>
    <tr>
        <th scope="col">이름</th>
        <th scope="col">나이</th>
        <th scope="col">도시</th>
    </tr>
    <tr>
        <th scope="row">홍길동</th>
        <td>30</td>
        <td>서울</td>
    </tr>
    <tr>
        <th scope="row">김철수</th>
        <td>25</td>
        <td>부산</td>
    </tr>
</table>
```

### headers와 id

```html
<table>
    <tr>
        <th id="name">이름</th>
        <th id="score">점수</th>
    </tr>
    <tr>
        <td headers="name">홍길동</td>
        <td headers="score">90</td>
    </tr>
</table>
```

---

## 13. colgroup

```html
<table>
    <colgroup>
        <col style="background-color: #f0f0f0;">
        <col span="2" style="background-color: #e0e0e0;">
        <col style="background-color: #d0d0d0;">
    </colgroup>
    <tr>
        <th>이름</th>
        <th>국어</th>
        <th>영어</th>
        <th>평균</th>
    </tr>
    <tr>
        <td>홍길동</td>
        <td>90</td>
        <td>85</td>
        <td>87.5</td>
    </tr>
</table>
```

---

## 14. 완전한 테이블 예제

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>성적표</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            max-width: 600px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }
        thead {
            background-color: #4CAF50;
            color: white;
        }
        tbody tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tfoot {
            background-color: #333;
            color: white;
            font-weight: bold;
        }
        caption {
            font-size: 1.5em;
            margin-bottom: 10px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <table>
        <caption>2024년 1학기 성적표</caption>

        <colgroup>
            <col style="width: 100px;">
            <col span="3" style="width: 80px;">
            <col style="width: 80px; background-color: #fffacd;">
        </colgroup>

        <thead>
            <tr>
                <th scope="col">이름</th>
                <th scope="col">국어</th>
                <th scope="col">영어</th>
                <th scope="col">수학</th>
                <th scope="col">평균</th>
            </tr>
        </thead>

        <tbody>
            <tr>
                <th scope="row">홍길동</th>
                <td>90</td>
                <td>85</td>
                <td>92</td>
                <td>89</td>
            </tr>
            <tr>
                <th scope="row">김철수</th>
                <td>78</td>
                <td>92</td>
                <td>88</td>
                <td>86</td>
            </tr>
            <tr>
                <th scope="row">이영희</th>
                <td>95</td>
                <td>88</td>
                <td>90</td>
                <td>91</td>
            </tr>
        </tbody>

        <tfoot>
            <tr>
                <th scope="row">평균</th>
                <td>87.7</td>
                <td>88.3</td>
                <td>90</td>
                <td>88.7</td>
            </tr>
        </tfoot>
    </table>
</body>
</html>
```

---

## 15. 요약

### 폼 요약

| 요소 | 용도 |
|------|------|
| `<form>` | 폼 컨테이너 |
| `<input>` | 다양한 입력 필드 |
| `<textarea>` | 여러 줄 텍스트 |
| `<select>` | 드롭다운 |
| `<button>` | 버튼 |
| `<label>` | 입력 필드 레이블 |
| `<fieldset>` | 폼 그룹핑 |

### 테이블 요약

| 요소 | 용도 |
|------|------|
| `<table>` | 테이블 컨테이너 |
| `<tr>` | 행 |
| `<th>` | 헤더 셀 |
| `<td>` | 데이터 셀 |
| `<thead>`, `<tbody>`, `<tfoot>` | 테이블 구역 |
| `colspan`, `rowspan` | 셀 병합 |

---

## 16. 연습 문제

### 연습 1: 로그인 폼

이메일과 비밀번호로 로그인하는 폼을 만드세요.
- "로그인 상태 유지" 체크박스 포함

### 연습 2: 설문조사 폼

간단한 설문조사 폼을 만드세요.
- 이름, 나이, 성별, 만족도(1~5), 의견 포함

### 연습 3: 시간표 테이블

학교 시간표를 테이블로 만드세요.
- 요일(열)과 교시(행), 셀 병합 활용

---

## 다음 단계

[03_CSS_Basics.md](./03_CSS_Basics.md)에서 CSS 스타일링을 배워봅시다!
