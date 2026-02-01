# JavaScript 기초

## 개요

JavaScript는 웹 페이지에 동적인 기능을 추가하는 프로그래밍 언어입니다. HTML이 구조, CSS가 스타일이라면, JavaScript는 **동작**을 담당합니다.

**선수 지식**: HTML, CSS 기초

---

## 목차

1. [JavaScript 시작하기](#javascript-시작하기)
2. [변수와 상수](#변수와-상수)
3. [자료형](#자료형)
4. [연산자](#연산자)
5. [조건문](#조건문)
6. [반복문](#반복문)
7. [함수](#함수)
8. [배열](#배열)
9. [객체](#객체)
10. [ES6+ 문법](#es6-문법)

---

## JavaScript 시작하기

### HTML에서 JavaScript 사용하기

```html
<!-- 방법 1: 내부 스크립트 -->
<script>
    console.log('Hello, World!');
</script>

<!-- 방법 2: 외부 스크립트 (권장) -->
<script src="main.js"></script>

<!-- 방법 3: 인라인 이벤트 (비권장) -->
<button onclick="alert('클릭!')">클릭</button>
```

### 스크립트 위치

```html
<!DOCTYPE html>
<html>
<head>
    <!-- head에 넣으면 HTML 파싱 차단 -->
    <script src="blocking.js"></script>

    <!-- defer: HTML 파싱 완료 후 실행 -->
    <script src="main.js" defer></script>

    <!-- async: 다운로드 완료 즉시 실행 (순서 보장 X) -->
    <script src="analytics.js" async></script>
</head>
<body>
    <h1>Hello</h1>

    <!-- body 끝에 넣으면 DOM 준비 후 실행 -->
    <script src="main.js"></script>
</body>
</html>
```

| 속성 | 실행 시점 | 순서 보장 | 용도 |
|------|-----------|-----------|------|
| (없음) | 즉시 | O | - |
| `defer` | DOM 파싱 후 | O | 일반 스크립트 |
| `async` | 다운로드 후 | X | 분석, 광고 |

### 개발자 도구 콘솔

브라우저에서 F12 → Console 탭에서 JavaScript를 바로 실행할 수 있습니다.

```javascript
console.log('일반 출력');
console.warn('경고');
console.error('에러');
console.table([{a: 1}, {a: 2}]);  // 표 형태
```

---

## 변수와 상수

### let (변수)

값을 변경할 수 있습니다.

```javascript
let name = '홍길동';
name = '김철수';  // 재할당 가능

let count;        // 선언만 (undefined)
count = 0;        // 나중에 할당
```

### const (상수)

값을 변경할 수 없습니다.

```javascript
const PI = 3.14159;
// PI = 3.14;  // 에러! 재할당 불가

const user = { name: '홍길동' };
user.name = '김철수';  // 객체 내부 속성은 변경 가능
// user = {};          // 에러! 객체 자체 재할당 불가
```

### var (레거시, 사용 지양)

```javascript
var old = '오래된 방식';  // 함수 스코프, 호이스팅 문제
```

### 변수 명명 규칙

```javascript
// 사용 가능
let userName = 'kim';      // camelCase (권장)
let user_name = 'kim';     // snake_case
let _private = 'secret';   // 언더스코어 시작
let $element = 'dom';      // 달러 시작
let 한글변수 = '가능';       // 유니코드 (비권장)

// 사용 불가
// let 1name = 'x';        // 숫자로 시작 X
// let user-name = 'x';    // 하이픈 X
// let let = 'x';          // 예약어 X
```

### 스코프

```javascript
// 블록 스코프 (let, const)
{
    let blockVar = '블록 내부';
    const blockConst = '블록 내부';
}
// console.log(blockVar);  // 에러! 접근 불가

// 함수 스코프 (var)
function test() {
    var funcVar = '함수 내부';
}
// console.log(funcVar);  // 에러!

// 전역 스코프
let globalVar = '어디서든 접근 가능';
```

---

## 자료형

### 원시 타입 (Primitive)

```javascript
// String (문자열)
let str1 = '작은따옴표';
let str2 = "큰따옴표";
let str3 = `템플릿 리터럴 ${str1}`;

// Number (숫자)
let int = 42;
let float = 3.14;
let negative = -10;
let infinity = Infinity;
let notANumber = NaN;

// BigInt (큰 정수)
let big = 9007199254740991n;

// Boolean (참/거짓)
let isTrue = true;
let isFalse = false;

// undefined (값 없음)
let nothing;
console.log(nothing);  // undefined

// null (의도적 빈 값)
let empty = null;

// Symbol (고유 식별자)
let sym = Symbol('description');
```

### 참조 타입 (Reference)

```javascript
// Object (객체)
let obj = { name: '홍길동', age: 30 };

// Array (배열)
let arr = [1, 2, 3, 4, 5];

// Function (함수)
let func = function() { return 'hello'; };
```

### 타입 확인

```javascript
typeof 'hello'      // "string"
typeof 42           // "number"
typeof true         // "boolean"
typeof undefined    // "undefined"
typeof null         // "object" (역사적 버그)
typeof {}           // "object"
typeof []           // "object"
typeof function(){} // "function"

// 배열 확인
Array.isArray([1, 2, 3])  // true
Array.isArray({})          // false
```

### 타입 변환

```javascript
// 문자열로 변환
String(123)        // "123"
(123).toString()   // "123"
123 + ''           // "123"

// 숫자로 변환
Number('123')      // 123
Number('abc')      // NaN
parseInt('42px')   // 42
parseFloat('3.14') // 3.14
+'123'             // 123

// 불리언으로 변환
Boolean(1)         // true
Boolean(0)         // false
Boolean('')        // false
Boolean('hello')   // true
!!1                // true

// Falsy 값 (false로 변환되는 값)
// false, 0, -0, '', null, undefined, NaN
```

---

## 연산자

### 산술 연산자

```javascript
10 + 3   // 13 (덧셈)
10 - 3   // 7  (뺄셈)
10 * 3   // 30 (곱셈)
10 / 3   // 3.333... (나눗셈)
10 % 3   // 1  (나머지)
10 ** 3  // 1000 (거듭제곱)

// 증감 연산자
let a = 5;
a++      // 후위: 사용 후 증가
++a      // 전위: 증가 후 사용
a--
--a
```

### 비교 연산자

```javascript
// 동등 비교 (타입 변환 O)
5 == '5'    // true
0 == false  // true
null == undefined  // true

// 일치 비교 (타입 변환 X) - 권장!
5 === '5'   // false
0 === false // false

// 부등 비교
5 != '5'    // false (타입 변환)
5 !== '5'   // true  (타입 비교)

// 크기 비교
5 > 3       // true
5 >= 5      // true
5 < 3       // false
5 <= 5      // true
```

### 논리 연산자

```javascript
// AND (둘 다 true면 true)
true && true    // true
true && false   // false

// OR (하나라도 true면 true)
true || false   // true
false || false  // false

// NOT (반전)
!true           // false
!false          // true

// 단락 평가 (Short-circuit)
const name = user && user.name;  // user가 없으면 undefined
const value = input || '기본값';  // input이 falsy면 '기본값'
```

### Nullish 연산자

```javascript
// ?? (null/undefined일 때만 우측 값)
null ?? '기본값'      // '기본값'
undefined ?? '기본값' // '기본값'
0 ?? '기본값'         // 0
'' ?? '기본값'        // ''

// || 와 비교
0 || '기본값'         // '기본값' (0은 falsy)
'' || '기본값'        // '기본값' (''은 falsy)
```

### 할당 연산자

```javascript
let x = 10;
x += 5;   // x = x + 5  → 15
x -= 3;   // x = x - 3  → 12
x *= 2;   // x = x * 2  → 24
x /= 4;   // x = x / 4  → 6
x %= 4;   // x = x % 4  → 2
x **= 2;  // x = x ** 2 → 4

// 논리 할당 (ES2021)
x ||= 10;  // x = x || 10
x &&= 5;   // x = x && 5
x ??= 0;   // x = x ?? 0
```

### 삼항 연산자

```javascript
const result = 조건 ? 참일때 : 거짓일때;

const age = 20;
const status = age >= 18 ? '성인' : '미성년';

// 중첩 (가독성 주의)
const grade = score >= 90 ? 'A'
            : score >= 80 ? 'B'
            : score >= 70 ? 'C'
            : 'F';
```

---

## 조건문

### if...else

```javascript
const age = 20;

if (age >= 18) {
    console.log('성인');
} else if (age >= 13) {
    console.log('청소년');
} else {
    console.log('어린이');
}
```

### switch

```javascript
const day = '월';

switch (day) {
    case '월':
    case '화':
    case '수':
    case '목':
    case '금':
        console.log('평일');
        break;
    case '토':
    case '일':
        console.log('주말');
        break;
    default:
        console.log('알 수 없음');
}
```

### 조건부 실행

```javascript
// 조건 && 실행문
isLoggedIn && showDashboard();

// 조건 || 실행문
data || fetchData();
```

---

## 반복문

### for

```javascript
for (let i = 0; i < 5; i++) {
    console.log(i);  // 0, 1, 2, 3, 4
}

// 역순
for (let i = 4; i >= 0; i--) {
    console.log(i);  // 4, 3, 2, 1, 0
}
```

### for...of (배열)

```javascript
const fruits = ['사과', '바나나', '오렌지'];

for (const fruit of fruits) {
    console.log(fruit);
}
```

### for...in (객체)

```javascript
const user = { name: '홍길동', age: 30 };

for (const key in user) {
    console.log(key, user[key]);
    // name 홍길동
    // age 30
}
```

### while

```javascript
let count = 0;

while (count < 5) {
    console.log(count);
    count++;
}
```

### do...while

```javascript
let num = 0;

do {
    console.log(num);  // 최소 1번은 실행
    num++;
} while (num < 5);
```

### break와 continue

```javascript
// break: 반복문 종료
for (let i = 0; i < 10; i++) {
    if (i === 5) break;
    console.log(i);  // 0, 1, 2, 3, 4
}

// continue: 현재 반복 건너뛰기
for (let i = 0; i < 5; i++) {
    if (i === 2) continue;
    console.log(i);  // 0, 1, 3, 4
}
```

---

## 함수

### 함수 선언식

```javascript
function greet(name) {
    return `안녕, ${name}!`;
}

greet('홍길동');  // "안녕, 홍길동!"
```

### 함수 표현식

```javascript
const greet = function(name) {
    return `안녕, ${name}!`;
};

greet('홍길동');
```

### 화살표 함수 (Arrow Function)

```javascript
// 기본 형태
const greet = (name) => {
    return `안녕, ${name}!`;
};

// 매개변수 1개: 괄호 생략 가능
const greet = name => {
    return `안녕, ${name}!`;
};

// 한 줄이면 중괄호와 return 생략
const greet = name => `안녕, ${name}!`;

// 매개변수 없음
const sayHello = () => '안녕!';

// 객체 반환 (괄호 필수)
const createUser = name => ({ name, created: Date.now() });
```

### 매개변수

```javascript
// 기본값
function greet(name = '손님') {
    return `안녕, ${name}!`;
}
greet();        // "안녕, 손님!"
greet('철수');  // "안녕, 철수!"

// 나머지 매개변수
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}
sum(1, 2, 3, 4);  // 10

// 구조 분해
function printUser({ name, age }) {
    console.log(`${name}은 ${age}살`);
}
printUser({ name: '홍길동', age: 30 });
```

### 콜백 함수

```javascript
function processData(data, callback) {
    // 데이터 처리
    const result = data.toUpperCase();
    callback(result);
}

processData('hello', function(result) {
    console.log(result);  // "HELLO"
});

// 화살표 함수로
processData('hello', result => console.log(result));
```

### 즉시 실행 함수 (IIFE)

```javascript
(function() {
    const private = '외부에서 접근 불가';
    console.log('즉시 실행');
})();

// 화살표 함수
(() => {
    console.log('즉시 실행');
})();
```

---

## 배열

### 배열 생성

```javascript
// 리터럴 (권장)
const arr1 = [1, 2, 3];

// 생성자
const arr2 = new Array(3);     // [empty × 3]
const arr3 = Array.of(1, 2, 3); // [1, 2, 3]
const arr4 = Array.from('abc'); // ['a', 'b', 'c']
```

### 배열 접근

```javascript
const fruits = ['사과', '바나나', '오렌지'];

fruits[0]          // '사과'
fruits[2]          // '오렌지'
fruits.length      // 3
fruits.at(-1)      // '오렌지' (마지막 요소)
fruits.at(-2)      // '바나나'
```

### 배열 수정

```javascript
const arr = [1, 2, 3];

// 추가
arr.push(4);         // 끝에 추가: [1, 2, 3, 4]
arr.unshift(0);      // 앞에 추가: [0, 1, 2, 3, 4]

// 제거
arr.pop();           // 끝에서 제거: [0, 1, 2, 3]
arr.shift();         // 앞에서 제거: [1, 2, 3]

// 특정 위치 수정
arr.splice(1, 1);           // 인덱스 1에서 1개 제거: [1, 3]
arr.splice(1, 0, 'a', 'b'); // 인덱스 1에 추가: [1, 'a', 'b', 3]
arr.splice(1, 1, 'x');      // 인덱스 1을 교체: [1, 'x', 'b', 3]
```

### 배열 검색

```javascript
const arr = [1, 2, 3, 2, 1];

arr.indexOf(2)        // 1 (첫 번째 인덱스)
arr.lastIndexOf(2)    // 3 (마지막 인덱스)
arr.includes(2)       // true (포함 여부)

// 조건으로 검색
const users = [
    { id: 1, name: 'Kim' },
    { id: 2, name: 'Lee' }
];

users.find(u => u.id === 1);       // { id: 1, name: 'Kim' }
users.findIndex(u => u.id === 1);  // 0
```

### 배열 반복 메서드

```javascript
const numbers = [1, 2, 3, 4, 5];

// forEach: 각 요소 실행 (반환값 없음)
numbers.forEach((num, index) => {
    console.log(index, num);
});

// map: 변환된 새 배열 반환
const doubled = numbers.map(n => n * 2);
// [2, 4, 6, 8, 10]

// filter: 조건에 맞는 요소만
const evens = numbers.filter(n => n % 2 === 0);
// [2, 4]

// reduce: 누적 계산
const sum = numbers.reduce((acc, cur) => acc + cur, 0);
// 15

// every: 모든 요소가 조건 충족?
numbers.every(n => n > 0);  // true

// some: 하나라도 조건 충족?
numbers.some(n => n > 4);   // true
```

### 배열 정렬

```javascript
const arr = [3, 1, 4, 1, 5];

// 오름차순
arr.sort((a, b) => a - b);  // [1, 1, 3, 4, 5]

// 내림차순
arr.sort((a, b) => b - a);  // [5, 4, 3, 1, 1]

// 문자열 정렬
const names = ['홍길동', '김철수', '이영희'];
names.sort();               // ['김철수', '이영희', '홍길동']

// 역순
arr.reverse();  // [1, 1, 4, 3, 5]
```

### 배열 변환

```javascript
const arr = [1, 2, 3];

// 복사
const copy = [...arr];        // 스프레드
const copy2 = arr.slice();    // slice

// 합치기
const merged = [...arr, ...[4, 5]];  // [1, 2, 3, 4, 5]
const merged2 = arr.concat([4, 5]);

// 평탄화
const nested = [1, [2, [3, 4]]];
nested.flat();     // [1, 2, [3, 4]]
nested.flat(2);    // [1, 2, 3, 4]
nested.flat(Infinity);  // 완전 평탄화

// 문자열로
[1, 2, 3].join('-');  // "1-2-3"
```

---

## 객체

### 객체 생성

```javascript
// 리터럴 (권장)
const user = {
    name: '홍길동',
    age: 30,
    email: 'hong@example.com'
};

// 생성자
const obj = new Object();
obj.name = '홍길동';
```

### 객체 접근

```javascript
const user = { name: '홍길동', age: 30 };

// 점 표기법
user.name       // '홍길동'
user.age        // 30

// 대괄호 표기법 (동적 키)
user['name']    // '홍길동'

const key = 'age';
user[key]       // 30
```

### 객체 수정

```javascript
const user = { name: '홍길동' };

// 추가/수정
user.age = 30;
user['email'] = 'hong@example.com';

// 삭제
delete user.email;

// 속성 존재 확인
'name' in user           // true
user.hasOwnProperty('name')  // true
```

### 객체 순회

```javascript
const user = { name: '홍길동', age: 30 };

// for...in
for (const key in user) {
    console.log(key, user[key]);
}

// Object 메서드
Object.keys(user)    // ['name', 'age']
Object.values(user)  // ['홍길동', 30]
Object.entries(user) // [['name', '홍길동'], ['age', 30]]

// entries로 순회
for (const [key, value] of Object.entries(user)) {
    console.log(key, value);
}
```

### 객체 복사/병합

```javascript
const user = { name: '홍길동', age: 30 };

// 얕은 복사
const copy1 = { ...user };           // 스프레드
const copy2 = Object.assign({}, user);

// 병합
const merged = { ...user, city: '서울' };
const merged2 = Object.assign({}, user, { city: '서울' });

// 깊은 복사 (중첩 객체)
const deep = JSON.parse(JSON.stringify(user));
const deep2 = structuredClone(user);  // 최신 브라우저
```

### 단축 속성

```javascript
const name = '홍길동';
const age = 30;

// 기존 방식
const user1 = { name: name, age: age };

// 단축 속성
const user2 = { name, age };
```

### 계산된 속성명

```javascript
const key = 'email';
const user = {
    name: '홍길동',
    [key]: 'hong@example.com',
    ['get' + 'Age']() { return 30; }
};

user.email    // 'hong@example.com'
user.getAge() // 30
```

### 메서드

```javascript
const user = {
    name: '홍길동',

    // 메서드
    greet() {
        return `안녕, ${this.name}!`;
    },

    // 화살표 함수 (this 주의!)
    // this가 상위 스코프를 참조
    badGreet: () => {
        return `안녕, ${this.name}!`;  // this는 user가 아님!
    }
};

user.greet();  // "안녕, 홍길동!"
```

---

## ES6+ 문법

### 템플릿 리터럴

```javascript
const name = '홍길동';
const age = 30;

// 기존 방식
const msg1 = '이름: ' + name + ', 나이: ' + age;

// 템플릿 리터럴
const msg2 = `이름: ${name}, 나이: ${age}`;

// 여러 줄
const html = `
    <div>
        <h1>${name}</h1>
        <p>${age}살</p>
    </div>
`;

// 표현식 사용
const result = `합계: ${10 + 20}`;
const status = `상태: ${age >= 18 ? '성인' : '미성년'}`;
```

### 구조 분해 할당

```javascript
// 배열 구조 분해
const [a, b, c] = [1, 2, 3];
const [first, , third] = [1, 2, 3];  // 건너뛰기
const [x, ...rest] = [1, 2, 3, 4];   // 나머지

// 객체 구조 분해
const { name, age } = { name: '홍길동', age: 30 };

// 기본값
const { name, city = '서울' } = { name: '홍길동' };

// 이름 변경
const { name: userName, age: userAge } = user;

// 중첩
const { address: { city } } = {
    address: { city: '서울' }
};

// 함수 매개변수
function greet({ name, age = 0 }) {
    console.log(`${name}, ${age}살`);
}
```

### 스프레드 연산자

```javascript
// 배열
const arr1 = [1, 2, 3];
const arr2 = [...arr1, 4, 5];  // [1, 2, 3, 4, 5]

// 객체
const obj1 = { a: 1, b: 2 };
const obj2 = { ...obj1, c: 3 };  // { a: 1, b: 2, c: 3 }

// 함수 호출
const numbers = [1, 2, 3];
Math.max(...numbers);  // 3

// 문자열 → 배열
[...'hello']  // ['h', 'e', 'l', 'l', 'o']
```

### Optional Chaining (?.)

```javascript
const user = {
    name: '홍길동',
    address: {
        city: '서울'
    }
};

// 기존 방식
const city1 = user && user.address && user.address.city;

// Optional Chaining
const city2 = user?.address?.city;

// 배열
const first = arr?.[0];

// 함수
const result = obj.method?.();
```

### Nullish Coalescing (??)

```javascript
// || 와 다르게 null/undefined만 체크
const value1 = null ?? '기본값';     // '기본값'
const value2 = undefined ?? '기본값'; // '기본값'
const value3 = 0 ?? '기본값';        // 0
const value4 = '' ?? '기본값';       // ''
const value5 = false ?? '기본값';    // false
```

---

## 연습 문제

### 문제 1: 변수와 조건문

나이를 받아 구분하는 함수를 작성하세요.
- 0-12: "어린이"
- 13-19: "청소년"
- 20-64: "성인"
- 65+: "노인"

<details>
<summary>정답 보기</summary>

```javascript
function getAgeGroup(age) {
    if (age < 0) return '잘못된 나이';
    if (age <= 12) return '어린이';
    if (age <= 19) return '청소년';
    if (age <= 64) return '성인';
    return '노인';
}
```

</details>

### 문제 2: 배열 메서드

숫자 배열에서 짝수만 골라 제곱한 결과를 반환하세요.

```javascript
// 입력: [1, 2, 3, 4, 5, 6]
// 출력: [4, 16, 36]
```

<details>
<summary>정답 보기</summary>

```javascript
const numbers = [1, 2, 3, 4, 5, 6];

const result = numbers
    .filter(n => n % 2 === 0)
    .map(n => n ** 2);

console.log(result);  // [4, 16, 36]
```

</details>

### 문제 3: 객체 다루기

사용자 배열에서 특정 조건의 사용자를 찾고 정보를 출력하세요.

```javascript
const users = [
    { id: 1, name: 'Kim', age: 25 },
    { id: 2, name: 'Lee', age: 30 },
    { id: 3, name: 'Park', age: 28 }
];
// 나이가 28 이상인 사용자의 이름만 배열로 반환
```

<details>
<summary>정답 보기</summary>

```javascript
const result = users
    .filter(user => user.age >= 28)
    .map(user => user.name);

console.log(result);  // ['Lee', 'Park']
```

</details>

---

## 다음 단계

- [07_JS_Events_DOM.md](./07_JS_Events_DOM.md) - DOM 조작과 이벤트 핸들링

---

## 참고 자료

- [MDN JavaScript](https://developer.mozilla.org/ko/docs/Web/JavaScript)
- [JavaScript.info](https://ko.javascript.info/)
- [ECMAScript 사양](https://tc39.es/ecma262/)
