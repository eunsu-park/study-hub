# JavaScript 이벤트와 DOM

**이전**: [JS 기초](./06_JS_Basics.md) | **다음**: [JS 비동기](./08_JS_Async.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. DOM 트리 구조를 설명하고 요소 노드(element node), 텍스트 노드(text node), 문서 노드(document node)를 구분할 수 있다
2. querySelector, querySelectorAll, getElementById를 사용하여 단일 및 다중 요소를 선택할 수 있다
3. textContent, innerHTML, outerHTML을 사용하여 요소 내용을 조작하고, XSS 위험성을 이해할 수 있다
4. HTML 속성(attribute), 데이터 속성(data attribute), CSS 클래스, 인라인 스타일을 프로그래밍으로 읽고 수정할 수 있다
5. DOM 요소를 동적으로 생성, 추가, 삭제, 복제, 교체할 수 있다
6. addEventListener를 사용하여 이벤트 리스너를 등록·제거하고, 이벤트 흐름(캡처(capture), 타겟(target), 버블링(bubble))을 설명할 수 있다
7. 이벤트 위임(event delegation)을 적용하여 동적 자식 요소의 이벤트를 효율적으로 처리할 수 있다
8. 값 추출, FormData, 제약 유효성 검사 API(Constraint Validation API)를 포함한 폼(form) 처리를 구현할 수 있다

---

DOM은 HTML 문서와 JavaScript를 연결하는 다리입니다. HTML이 정적인 텍스트 파일이라면, DOM은 브라우저가 그 파일을 파싱하여 메모리에 구성한 살아있는 트리입니다. JavaScript는 이 트리에 접근하여 언제든지 어떤 요소든 읽고, 수정하고, 추가하거나 제거할 수 있습니다. 이벤트 시스템과 결합하면, DOM은 문서를 클릭, 키 입력, 스크롤 등 모든 사용자 상호작용에 반응하는 인터랙티브 애플리케이션으로 변환합니다.

## 목차

1. [DOM 기초](#dom-기초)
2. [요소 선택](#요소-선택)
3. [요소 내용 조작](#요소-내용-조작)
4. [속성 조작](#속성-조작)
5. [클래스 조작](#클래스-조작)
6. [스타일 조작](#스타일-조작)
7. [요소 생성과 삭제](#요소-생성과-삭제)
8. [이벤트 기초](#이벤트-기초)
9. [이벤트 종류](#이벤트-종류)
10. [이벤트 위임](#이벤트-위임)
11. [폼 처리](#폼-처리)

---

## DOM 기초

### DOM 트리 구조

```
document
└── html
    ├── head
    │   └── title
    └── body
        ├── header
        │   └── h1
        ├── main
        │   ├── p
        │   └── div
        └── footer
```

### 노드 타입

```javascript
// 요소 노드 (Element)
document.body

// 텍스트 노드 (Text)
document.body.firstChild

// 문서 노드 (Document)
document

// 주석 노드 (Comment)
<!-- 주석 -->
```

### DOM 탐색

```javascript
const element = document.querySelector('.box');

// 부모/자식
element.parentNode       // 부모 노드
element.parentElement    // 부모 요소
element.children         // 자식 요소들 (HTMLCollection)
element.childNodes       // 자식 노드들 (텍스트 포함)
element.firstChild       // 첫 번째 자식 노드
element.firstElementChild // 첫 번째 자식 요소
element.lastChild        // 마지막 자식 노드
element.lastElementChild // 마지막 자식 요소

// 형제
element.nextSibling          // 다음 형제 노드
element.nextElementSibling   // 다음 형제 요소
element.previousSibling      // 이전 형제 노드
element.previousElementSibling // 이전 형제 요소
```

```
            parentElement
                 │
    ┌────────────┼────────────┐
    │            │            │
previousElement  element  nextElement
                 │
         ┌───────┴───────┐
         │       │       │
      first   children  last
```

---

## 요소 선택

### 단일 요소 선택

```javascript
// CSS 선택자로 첫 번째 요소 (권장)
document.querySelector('.class');
document.querySelector('#id');
document.querySelector('div.box');
document.querySelector('[data-id="123"]');

// ID로 선택
document.getElementById('myId');
```

### 다중 요소 선택

```javascript
// CSS 선택자로 모든 요소 (NodeList)
document.querySelectorAll('.item');
document.querySelectorAll('ul li');

// 클래스로 선택 (HTMLCollection - 실시간)
document.getElementsByClassName('item');

// 태그로 선택 (HTMLCollection - 실시간)
document.getElementsByTagName('div');

// name 속성으로 선택
document.getElementsByName('username');
```

### NodeList vs HTMLCollection

```javascript
// NodeList (정적)
const nodeList = document.querySelectorAll('.item');
nodeList.forEach(item => console.log(item));  // forEach 사용 가능

// HTMLCollection (동적/실시간)
const htmlCollection = document.getElementsByClassName('item');
// forEach 사용 불가, 배열로 변환 필요
[...htmlCollection].forEach(item => console.log(item));
Array.from(htmlCollection).forEach(item => console.log(item));
```

### 범위 내 선택

```javascript
const container = document.querySelector('.container');

// container 내부에서 선택
const item = container.querySelector('.item');
const items = container.querySelectorAll('.item');
```

### closest()

가장 가까운 조상 요소 찾기

```javascript
const button = document.querySelector('button');

// button의 가장 가까운 .card 조상
const card = button.closest('.card');

// 자기 자신도 포함
const self = button.closest('button');  // 자기 자신 반환
```

### matches()

선택자와 일치하는지 확인

```javascript
const element = document.querySelector('.item');

element.matches('.item');      // true
element.matches('.active');    // false (클래스 없으면)
element.matches('div.item');   // true (div이고 .item이면)
```

---

## 요소 내용 조작

### textContent

텍스트만 다룹니다 (HTML 태그 무시).

```javascript
const el = document.querySelector('.box');

// 읽기
console.log(el.textContent);

// 쓰기 (HTML 태그는 텍스트로 처리)
el.textContent = '<strong>굵게</strong>';  // 태그가 그대로 표시됨
```

### innerHTML

HTML을 포함한 내용을 다룹니다.

```javascript
const el = document.querySelector('.box');

// 읽기
console.log(el.innerHTML);

// 쓰기 (HTML 파싱됨)
el.innerHTML = '<strong>굵게</strong>';  // 실제로 굵게 표시

// 추가
el.innerHTML += '<p>추가 내용</p>';

// ⚠️ 보안 주의: 사용자 입력을 그대로 넣지 말 것!
// el.innerHTML = userInput;  // XSS 취약점!
```

### innerText vs textContent

```javascript
// innerText: 화면에 보이는 텍스트만 (느림)
// textContent: 모든 텍스트 (빠름)

// display: none인 요소의 텍스트
el.innerText;     // 포함 안 됨
el.textContent;   // 포함됨
```

### outerHTML

요소 자체를 포함한 HTML

```javascript
const el = document.querySelector('.box');

// 읽기: 요소 자체 포함
console.log(el.outerHTML);  // <div class="box">내용</div>

// 쓰기: 요소 자체를 교체
el.outerHTML = '<span>새 요소</span>';
```

---

## 속성 조작

### 표준 속성

```javascript
const link = document.querySelector('a');
const img = document.querySelector('img');
const input = document.querySelector('input');

// 직접 접근
link.href = 'https://example.com';
img.src = 'image.jpg';
img.alt = '이미지 설명';
input.value = '입력값';
input.disabled = true;
input.checked = true;
```

### getAttribute / setAttribute

```javascript
const el = document.querySelector('.box');

// 읽기
el.getAttribute('class');
el.getAttribute('data-id');

// 쓰기
el.setAttribute('class', 'box active');
el.setAttribute('data-id', '123');

// 삭제
el.removeAttribute('data-id');

// 존재 확인
el.hasAttribute('data-id');
```

### data 속성

```html
<div id="user" data-user-id="123" data-user-name="홍길동"></div>
```

```javascript
const el = document.querySelector('#user');

// dataset으로 접근 (camelCase 변환)
el.dataset.userId      // "123"
el.dataset.userName    // "홍길동"

// 수정
el.dataset.userId = '456';
el.dataset.newAttr = 'value';  // data-new-attr 생성

// 삭제
delete el.dataset.userName;
```

---

## 클래스 조작

### classList

```javascript
const el = document.querySelector('.box');

// 추가
el.classList.add('active');
el.classList.add('highlight', 'visible');  // 여러 개

// 제거
el.classList.remove('active');
el.classList.remove('highlight', 'visible');

// 토글 (있으면 제거, 없으면 추가)
el.classList.toggle('active');
el.classList.toggle('active', true);   // 강제로 추가
el.classList.toggle('active', false);  // 강제로 제거

// 교체
el.classList.replace('old-class', 'new-class');

// 확인
el.classList.contains('active');  // true/false

// 모든 클래스
el.classList.length;          // 클래스 개수
el.classList.item(0);         // 첫 번째 클래스
[...el.classList];            // 배열로 변환
```

### className

```javascript
const el = document.querySelector('.box');

// 전체 클래스 문자열
el.className;                    // "box highlight"
el.className = 'new-class';      // 전체 교체
el.className += ' another';      // 추가 (공백 주의)
```

---

## 스타일 조작

### style 속성

```javascript
const el = document.querySelector('.box');

// 개별 스타일 (camelCase)
el.style.backgroundColor = 'red';
el.style.fontSize = '20px';
el.style.marginTop = '10px';
el.style.display = 'none';

// CSS 속성명 그대로 (대괄호)
el.style['background-color'] = 'red';

// 여러 스타일 한 번에
el.style.cssText = 'color: red; font-size: 20px;';

// 스타일 제거
el.style.backgroundColor = '';
el.style.removeProperty('background-color');
```

### getComputedStyle

실제 적용된 스타일 읽기

```javascript
const el = document.querySelector('.box');
const styles = getComputedStyle(el);

styles.backgroundColor;  // "rgb(255, 0, 0)"
styles.fontSize;         // "16px"
styles.display;          // "block"

// 의사 요소 스타일
const beforeStyles = getComputedStyle(el, '::before');
```

---

## 요소 생성과 삭제

### 요소 생성

```javascript
// 요소 생성
const div = document.createElement('div');
div.className = 'box';
div.id = 'myBox';
div.textContent = '새 요소';

// 텍스트 노드 생성
const text = document.createTextNode('텍스트');

// 문서 조각 (여러 요소 묶음)
const fragment = document.createDocumentFragment();
for (let i = 0; i < 100; i++) {
    const item = document.createElement('li');
    item.textContent = `항목 ${i}`;
    fragment.appendChild(item);
}
list.appendChild(fragment);  // 한 번만 DOM 업데이트
```

### 요소 추가

```javascript
const parent = document.querySelector('.parent');
const child = document.createElement('div');

// 끝에 추가
parent.appendChild(child);
parent.append(child);            // 텍스트도 가능
parent.append(child, '텍스트');  // 여러 개 가능

// 앞에 추가
parent.prepend(child);

// 특정 위치에 삽입
const reference = document.querySelector('.reference');
parent.insertBefore(child, reference);  // reference 앞에

// insertAdjacentHTML
parent.insertAdjacentHTML('beforebegin', '<div>앞</div>');
parent.insertAdjacentHTML('afterbegin', '<div>첫 자식</div>');
parent.insertAdjacentHTML('beforeend', '<div>마지막 자식</div>');
parent.insertAdjacentHTML('afterend', '<div>뒤</div>');

// insertAdjacentElement
parent.insertAdjacentElement('beforeend', child);
```

```
<!-- beforebegin -->
<parent>
    <!-- afterbegin -->
    기존 내용
    <!-- beforeend -->
</parent>
<!-- afterend -->
```

### 요소 삭제

```javascript
const el = document.querySelector('.box');

// 자기 자신 삭제
el.remove();

// 부모에서 자식 삭제
parent.removeChild(el);

// 모든 자식 삭제
parent.innerHTML = '';
// 또는
while (parent.firstChild) {
    parent.removeChild(parent.firstChild);
}
// 또는
parent.replaceChildren();
```

### 요소 복제

```javascript
const el = document.querySelector('.box');

// 얕은 복제 (요소만)
const shallow = el.cloneNode(false);

// 깊은 복제 (자식 포함)
const deep = el.cloneNode(true);

// 문서에 추가
document.body.appendChild(deep);
```

### 요소 교체

```javascript
const oldEl = document.querySelector('.old');
const newEl = document.createElement('div');

// 교체
oldEl.replaceWith(newEl);

// 부모를 통해 교체
parent.replaceChild(newEl, oldEl);
```

---

## 이벤트 기초

### 이벤트 리스너 등록

```javascript
const button = document.querySelector('button');

// addEventListener (권장)
button.addEventListener('click', function(event) {
    console.log('클릭됨!');
});

// 화살표 함수
button.addEventListener('click', (e) => {
    console.log('클릭됨!');
});

// 핸들러 분리
function handleClick(event) {
    console.log('클릭됨!');
}
button.addEventListener('click', handleClick);
```

### 이벤트 리스너 제거

```javascript
// 같은 함수 참조 필요
function handleClick(event) {
    console.log('클릭됨!');
}

button.addEventListener('click', handleClick);
button.removeEventListener('click', handleClick);

// 익명 함수는 제거 불가
button.addEventListener('click', () => {});  // 제거 불가
```

### 이벤트 객체

```javascript
button.addEventListener('click', function(event) {
    // 이벤트 정보
    event.type;          // "click"
    event.target;        // 실제 클릭된 요소
    event.currentTarget; // 리스너가 등록된 요소
    event.timeStamp;     // 이벤트 발생 시간

    // 마우스 위치
    event.clientX;       // 뷰포트 기준 X
    event.clientY;       // 뷰포트 기준 Y
    event.pageX;         // 문서 기준 X
    event.pageY;         // 문서 기준 Y

    // 키보드 정보
    event.key;           // "Enter", "a", "Escape" 등
    event.code;          // "Enter", "KeyA", "Escape" 등
    event.shiftKey;      // Shift 눌렸는지
    event.ctrlKey;       // Ctrl 눌렸는지
    event.altKey;        // Alt 눌렸는지
    event.metaKey;       // Cmd(Mac)/Win 눌렸는지
});
```

### 기본 동작 방지

```javascript
// 링크 클릭 시 이동 방지
link.addEventListener('click', function(event) {
    event.preventDefault();
    console.log('이동하지 않음');
});

// 폼 제출 방지
form.addEventListener('submit', function(event) {
    event.preventDefault();
    console.log('제출하지 않음');
});
```

### 이벤트 전파 중지

```javascript
// 버블링 중지
inner.addEventListener('click', function(event) {
    event.stopPropagation();
    // 상위 요소로 이벤트 전파되지 않음
});

// 같은 요소의 다른 핸들러도 중지
inner.addEventListener('click', function(event) {
    event.stopImmediatePropagation();
});
```

### 이벤트 옵션

```javascript
element.addEventListener('click', handler, {
    once: true,      // 한 번만 실행 후 제거
    capture: true,   // 캡처 단계에서 실행
    passive: true    // preventDefault 호출 안 함 (스크롤 성능)
});

// 캡처 단계
element.addEventListener('click', handler, true);
```

### 이벤트 흐름

```
       캡처 단계                  버블링 단계
         (1)                        (4)
          ↓                          ↑
    ┌─────────────────────────────────────┐
    │  document                           │
    │   ┌───────────────────────────────┐ │
    │   │ parent                (2) (3) │ │
    │   │   ┌───────────────────────┐   │ │
    │   │   │ target         클릭!  │   │ │
    │   │   └───────────────────────┘   │ │
    │   └───────────────────────────────┘ │
    └─────────────────────────────────────┘
```

---

## 이벤트 종류

### 마우스 이벤트

```javascript
// 클릭
element.addEventListener('click', handler);      // 클릭
element.addEventListener('dblclick', handler);   // 더블클릭
element.addEventListener('contextmenu', handler); // 우클릭

// 마우스 버튼
element.addEventListener('mousedown', handler);  // 버튼 누름
element.addEventListener('mouseup', handler);    // 버튼 뗌

// 마우스 이동
element.addEventListener('mousemove', handler);  // 이동
element.addEventListener('mouseenter', handler); // 요소 진입 (버블링 X)
element.addEventListener('mouseleave', handler); // 요소 이탈 (버블링 X)
element.addEventListener('mouseover', handler);  // 요소 위로 (버블링 O)
element.addEventListener('mouseout', handler);   // 요소 벗어남 (버블링 O)

// 마우스 버튼 확인
element.addEventListener('mousedown', (e) => {
    e.button;  // 0: 좌클릭, 1: 휠, 2: 우클릭
});
```

### 키보드 이벤트

```javascript
// 키 이벤트
document.addEventListener('keydown', handler);  // 키 누름
document.addEventListener('keyup', handler);    // 키 뗌
document.addEventListener('keypress', handler); // 문자 키 (deprecated)

// 키 확인
document.addEventListener('keydown', (e) => {
    console.log(e.key);   // "a", "Enter", "Escape"
    console.log(e.code);  // "KeyA", "Enter", "Escape"

    // 특수 키
    if (e.key === 'Enter') { }
    if (e.key === 'Escape') { }
    if (e.key === 'ArrowUp') { }
    if (e.key === 'ArrowDown') { }

    // 조합 키
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        console.log('저장');
    }
});
```

### 폼 이벤트

```javascript
// 입력
input.addEventListener('input', handler);    // 값 변경될 때마다
input.addEventListener('change', handler);   // 포커스 잃을 때 (값 변경 시)

// 포커스
input.addEventListener('focus', handler);    // 포커스 받음
input.addEventListener('blur', handler);     // 포커스 잃음
input.addEventListener('focusin', handler);  // 포커스 받음 (버블링 O)
input.addEventListener('focusout', handler); // 포커스 잃음 (버블링 O)

// 제출
form.addEventListener('submit', handler);
form.addEventListener('reset', handler);
```

### 스크롤/리사이즈 이벤트

```javascript
// 스크롤
window.addEventListener('scroll', handler);
element.addEventListener('scroll', handler);

// 스크롤 위치
window.addEventListener('scroll', () => {
    console.log(window.scrollY);  // 수직 스크롤 위치
    console.log(window.scrollX);  // 수평 스크롤 위치
});

// 리사이즈
window.addEventListener('resize', handler);
window.addEventListener('resize', () => {
    console.log(window.innerWidth);
    console.log(window.innerHeight);
});

// 성능 최적화: throttle/debounce 필요
```

### 로드 이벤트

```javascript
// 문서 로드
window.addEventListener('load', handler);           // 모든 리소스 로드 후
document.addEventListener('DOMContentLoaded', handler); // DOM 파싱 완료 시

// 권장 패턴
document.addEventListener('DOMContentLoaded', () => {
    // DOM 조작 코드
});

// 또는 defer 스크립트 사용
// <script src="main.js" defer></script>

// 이미지 로드
img.addEventListener('load', handler);
img.addEventListener('error', handler);

// 페이지 이탈
window.addEventListener('beforeunload', (e) => {
    e.preventDefault();
    e.returnValue = '';  // 확인 대화상자 표시
});
```

### 터치 이벤트

```javascript
element.addEventListener('touchstart', handler);  // 터치 시작
element.addEventListener('touchmove', handler);   // 터치 이동
element.addEventListener('touchend', handler);    // 터치 종료
element.addEventListener('touchcancel', handler); // 터치 취소

// 터치 정보
element.addEventListener('touchstart', (e) => {
    const touch = e.touches[0];
    console.log(touch.clientX, touch.clientY);
});
```

---

## 이벤트 위임

### 개념

부모 요소에 이벤트 리스너를 등록하여 자식 요소의 이벤트를 처리합니다.

```html
<ul id="list">
    <li data-id="1">항목 1</li>
    <li data-id="2">항목 2</li>
    <li data-id="3">항목 3</li>
    <!-- 동적으로 추가되는 항목들... -->
</ul>
```

```javascript
// 나쁜 예: 각 요소에 리스너 등록
document.querySelectorAll('#list li').forEach(li => {
    li.addEventListener('click', handleClick);
});

// 좋은 예: 부모에 이벤트 위임
document.querySelector('#list').addEventListener('click', (e) => {
    // 클릭된 요소가 li인지 확인
    if (e.target.tagName === 'LI') {
        console.log('클릭된 항목:', e.target.dataset.id);
    }

    // 또는 closest 사용
    const li = e.target.closest('li');
    if (li) {
        console.log('클릭된 항목:', li.dataset.id);
    }
});
```

### 이벤트 위임의 장점

1. **메모리 효율**: 리스너 수 감소
2. **동적 요소**: 나중에 추가된 요소도 처리
3. **간단한 관리**: 하나의 리스너로 관리

### 실전 예제

```javascript
// Todo 리스트
const todoList = document.querySelector('#todo-list');

todoList.addEventListener('click', (e) => {
    const target = e.target;
    const todoItem = target.closest('.todo-item');

    if (!todoItem) return;

    // 완료 체크
    if (target.matches('.checkbox')) {
        todoItem.classList.toggle('completed');
    }

    // 삭제 버튼
    if (target.matches('.delete-btn')) {
        todoItem.remove();
    }

    // 편집 버튼
    if (target.matches('.edit-btn')) {
        const text = todoItem.querySelector('.text');
        text.contentEditable = 'true';
        text.focus();
    }
});
```

---

## 폼 처리

### 폼 요소 접근

```javascript
const form = document.querySelector('#myForm');

// name으로 접근
form.username;           // name="username"인 요소
form.elements.username;  // 같음
form.elements['user-name']; // 하이픈 포함 시

// 모든 요소
form.elements;           // HTMLFormControlsCollection
form.elements.length;    // 요소 개수
```

### 입력값 가져오기

```javascript
// text, password, email, textarea
const textValue = input.value;

// checkbox
const isChecked = checkbox.checked;

// radio
const radioGroup = document.querySelectorAll('input[name="gender"]');
let selectedValue;
radioGroup.forEach(radio => {
    if (radio.checked) selectedValue = radio.value;
});
// 또는
const selected = document.querySelector('input[name="gender"]:checked');

// select
const selectValue = select.value;
const selectedIndex = select.selectedIndex;
const selectedOption = select.options[select.selectedIndex];

// select multiple
const selectedOptions = [...select.selectedOptions].map(opt => opt.value);

// file
const files = fileInput.files;
const firstFile = files[0];
```

### 폼 이벤트 처리

```javascript
const form = document.querySelector('#myForm');

// 제출
form.addEventListener('submit', (e) => {
    e.preventDefault();

    // FormData로 모든 값 수집
    const formData = new FormData(form);

    // 개별 값
    formData.get('username');

    // 모든 값 객체로
    const data = Object.fromEntries(formData);

    // 또는 순회
    for (const [key, value] of formData) {
        console.log(key, value);
    }
});

// 입력 실시간 검증
input.addEventListener('input', (e) => {
    const value = e.target.value;
    if (value.length < 3) {
        e.target.classList.add('error');
    } else {
        e.target.classList.remove('error');
    }
});

// 변경 감지
input.addEventListener('change', (e) => {
    console.log('값 변경:', e.target.value);
});
```

### 폼 유효성 검사

```javascript
const form = document.querySelector('#myForm');
const email = document.querySelector('#email');

form.addEventListener('submit', (e) => {
    // HTML5 유효성 검사
    if (!form.checkValidity()) {
        e.preventDefault();
        form.reportValidity();  // 에러 메시지 표시
        return;
    }

    // 개별 요소 검사
    if (!email.validity.valid) {
        if (email.validity.valueMissing) {
            console.log('이메일 필수');
        }
        if (email.validity.typeMismatch) {
            console.log('이메일 형식 오류');
        }
    }
});

// 커스텀 에러 메시지
email.addEventListener('invalid', (e) => {
    e.target.setCustomValidity('올바른 이메일을 입력하세요');
});

email.addEventListener('input', (e) => {
    e.target.setCustomValidity('');  // 에러 메시지 초기화
});
```

### validity 속성

```javascript
input.validity.valid          // 전체 유효성
input.validity.valueMissing   // required인데 비어있음
input.validity.typeMismatch   // type 불일치 (email, url 등)
input.validity.patternMismatch // pattern 불일치
input.validity.tooLong        // maxlength 초과
input.validity.tooShort       // minlength 미달
input.validity.rangeOverflow  // max 초과
input.validity.rangeUnderflow // min 미달
input.validity.stepMismatch   // step 불일치
```

---

## 실전 예제

### 탭 메뉴

```html
<div class="tabs">
    <div class="tab-buttons">
        <button class="tab-btn active" data-tab="tab1">탭 1</button>
        <button class="tab-btn" data-tab="tab2">탭 2</button>
        <button class="tab-btn" data-tab="tab3">탭 3</button>
    </div>
    <div class="tab-content">
        <div class="tab-panel active" id="tab1">내용 1</div>
        <div class="tab-panel" id="tab2">내용 2</div>
        <div class="tab-panel" id="tab3">내용 3</div>
    </div>
</div>
```

```javascript
const tabButtons = document.querySelector('.tab-buttons');

tabButtons.addEventListener('click', (e) => {
    const button = e.target.closest('.tab-btn');
    if (!button) return;

    // 버튼 활성화
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    button.classList.add('active');

    // 패널 표시
    const tabId = button.dataset.tab;
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
});
```

### 모달

```html
<button id="openModal">모달 열기</button>

<div class="modal" id="modal">
    <div class="modal-overlay"></div>
    <div class="modal-content">
        <button class="modal-close">&times;</button>
        <h2>모달 제목</h2>
        <p>모달 내용입니다.</p>
    </div>
</div>
```

```javascript
const modal = document.getElementById('modal');
const openBtn = document.getElementById('openModal');

// 열기
openBtn.addEventListener('click', () => {
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
});

// 닫기 (이벤트 위임)
modal.addEventListener('click', (e) => {
    if (e.target.matches('.modal-close') ||
        e.target.matches('.modal-overlay')) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }
});

// ESC 키로 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('open')) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }
});
```

### Todo 리스트

```html
<div class="todo-app">
    <form id="todo-form">
        <input type="text" id="todo-input" placeholder="할 일 입력" required>
        <button type="submit">추가</button>
    </form>
    <ul id="todo-list"></ul>
</div>
```

```javascript
const form = document.getElementById('todo-form');
const input = document.getElementById('todo-input');
const list = document.getElementById('todo-list');

// 추가
form.addEventListener('submit', (e) => {
    e.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    const li = document.createElement('li');
    li.innerHTML = `
        <input type="checkbox" class="todo-check">
        <span class="todo-text">${text}</span>
        <button class="todo-delete">삭제</button>
    `;

    list.appendChild(li);
    input.value = '';
    input.focus();
});

// 완료/삭제 (이벤트 위임)
list.addEventListener('click', (e) => {
    const li = e.target.closest('li');
    if (!li) return;

    if (e.target.matches('.todo-check')) {
        li.classList.toggle('completed', e.target.checked);
    }

    if (e.target.matches('.todo-delete')) {
        li.remove();
    }
});
```

---

## 연습 문제

### 문제 1: 아코디언 메뉴

클릭하면 내용이 열리고 닫히는 아코디언을 구현하세요.

<details>
<summary>정답 보기</summary>

```javascript
const accordion = document.querySelector('.accordion');

accordion.addEventListener('click', (e) => {
    const header = e.target.closest('.accordion-header');
    if (!header) return;

    const item = header.parentElement;
    const content = item.querySelector('.accordion-content');

    // 다른 항목 닫기 (선택사항)
    document.querySelectorAll('.accordion-item').forEach(other => {
        if (other !== item) {
            other.classList.remove('open');
        }
    });

    // 현재 항목 토글
    item.classList.toggle('open');
});
```

</details>

### 문제 2: 글자 수 카운터

textarea에 입력할 때 실시간으로 글자 수를 표시하세요.

<details>
<summary>정답 보기</summary>

```javascript
const textarea = document.querySelector('textarea');
const counter = document.querySelector('.counter');
const maxLength = 200;

textarea.addEventListener('input', (e) => {
    const length = e.target.value.length;
    counter.textContent = `${length} / ${maxLength}`;

    if (length > maxLength) {
        counter.classList.add('error');
    } else {
        counter.classList.remove('error');
    }
});
```

</details>

---

## 다음 단계

- [JS 비동기](./08_JS_Async.md) - Promise, async/await, fetch

---

## 참고 자료

- [MDN DOM](https://developer.mozilla.org/ko/docs/Web/API/Document_Object_Model)
- [MDN 이벤트](https://developer.mozilla.org/ko/docs/Learn/JavaScript/Building_blocks/Events)
- [JavaScript.info DOM](https://ko.javascript.info/document)
