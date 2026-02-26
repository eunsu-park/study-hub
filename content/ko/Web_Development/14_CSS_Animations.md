# 14. CSS 애니메이션 (CSS Animations)

**이전**: [빌드 도구와 환경](./13_Build_Tools_Environment.md) | **다음**: [JavaScript 모듈 시스템](./15_JS_Modules.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트랜지션(transitions)을 사용하여 CSS 속성 값 사이의 부드러운 상태 변화를 구현할 수 있습니다
2. 이동(translate), 크기(scale), 회전(rotate), 기울이기(skew)를 포함한 2D 및 3D 변환(transformations)을 적용할 수 있습니다
3. `@keyframes`로 다단계 애니메이션을 만들고 애니메이션 속성으로 재생을 제어할 수 있습니다
4. Intersection Observer와 최신 CSS 스크롤 타임라인(scroll timelines)을 사용하여 스크롤 기반 애니메이션을 구현할 수 있습니다
5. GPU 가속 속성(`transform`, `opacity`)을 대상으로 하여 애니메이션 성능을 최적화할 수 있습니다
6. `prefers-reduced-motion` 미디어 쿼리를 구현하여 사용자 기본 설정을 존중할 수 있습니다
7. 실용적인 UI 패턴에서 트랜지션, 변환, 키프레임 애니메이션을 조합할 수 있습니다

---

정적인 페이지는 생명이 없어 보입니다. 잘 구성된 애니메이션은 사용자의 주의를 안내하고, 상태 변화를 전달하며, 인터페이스를 반응성 있고 세련되게 만들어 줍니다. 그러나 잘못 구현된 애니메이션은 성능을 저하시키고 모션에 민감한 사용자들을 소외시킬 수 있습니다. 이 레슨에서는 JavaScript 라이브러리 없이 순수 CSS만으로 성능 좋고 접근성 있는 애니메이션을 만드는 방법을 배웁니다.

## 1. CSS Transition

### 1.1 기본 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSS Transition                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Transition: 속성 값이 변할 때 부드럽게 전환                      │
│                                                                 │
│  ┌────────────┐    부드러운 전환    ┌────────────┐              │
│  │ 상태 A     │  ─────────────────▶ │ 상태 B     │              │
│  │ color: red │     (0.3s)          │ color:blue │              │
│  └────────────┘                     └────────────┘              │
│                                                                 │
│  필수 요소:                                                      │
│  1. transition-property: 어떤 속성을                             │
│  2. transition-duration: 얼마나 걸려서                           │
│  3. 트리거: hover, focus, class 변경 등                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Transition 속성

```css
/* 개별 속성 */
.element {
    transition-property: background-color;  /* 전환할 속성 */
    transition-duration: 0.3s;              /* 지속 시간 */
    transition-timing-function: ease;       /* 속도 곡선 */
    transition-delay: 0s;                   /* 지연 시간 */
}

/* 단축 속성 */
.element {
    transition: background-color 0.3s ease 0s;
    /* property | duration | timing-function | delay */
}

/* 여러 속성 전환 */
.element {
    transition:
        background-color 0.3s ease,
        transform 0.5s ease-out,
        opacity 0.2s linear;
}

/* 모든 속성 전환 (성능 주의) */
.element {
    transition: all 0.3s ease;
}
```

### 1.3 Timing Functions

```css
.examples {
    /* 내장 timing functions */
    transition-timing-function: linear;      /* 일정 속도 */
    transition-timing-function: ease;        /* 기본값, 느리게 시작-빠르게-느리게 끝 */
    transition-timing-function: ease-in;     /* 느리게 시작 */
    transition-timing-function: ease-out;    /* 느리게 끝 */
    transition-timing-function: ease-in-out; /* 느리게 시작하고 끝 */

    /* 커스텀 베지어 곡선 */
    transition-timing-function: cubic-bezier(0.68, -0.55, 0.27, 1.55);

    /* 단계별 전환 */
    transition-timing-function: steps(4, end);
}
```

### 1.4 실전 예제

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <style>
        /* 버튼 호버 효과 */
        .btn {
            padding: 12px 24px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition:
                background-color 0.3s ease,
                transform 0.2s ease,
                box-shadow 0.3s ease;
        }

        .btn:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .btn:active {
            transform: translateY(0);
        }

        /* 카드 호버 효과 */
        .card {
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition:
                transform 0.3s ease,
                box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }

        /* 입력 필드 포커스 */
        .input {
            padding: 10px 16px;
            border: 2px solid #ddd;
            border-radius: 4px;
            outline: none;
            transition:
                border-color 0.3s ease,
                box-shadow 0.3s ease;
        }

        .input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }

        /* 메뉴 아이템 */
        .menu-item {
            padding: 10px 20px;
            position: relative;
            transition: color 0.3s ease;
        }

        .menu-item::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 2px;
            background: #3498db;
            transition:
                width 0.3s ease,
                left 0.3s ease;
        }

        .menu-item:hover::after {
            width: 100%;
            left: 0;
        }
    </style>
</head>
<body>
    <button class="btn">버튼</button>
    <div class="card">카드 콘텐츠</div>
    <input class="input" placeholder="입력하세요">
    <nav>
        <a class="menu-item">메뉴 1</a>
        <a class="menu-item">메뉴 2</a>
    </nav>
</body>
</html>
```

---

## 2. CSS Transform

### 2.1 2D Transform

```css
/* 이동 (Translate) */
.translate {
    transform: translateX(50px);     /* X축 이동 */
    transform: translateY(30px);     /* Y축 이동 */
    transform: translate(50px, 30px); /* X, Y 동시 이동 */
}

/* 크기 (Scale) */
.scale {
    transform: scaleX(1.5);          /* X축 확대 */
    transform: scaleY(0.8);          /* Y축 축소 */
    transform: scale(1.5);           /* 균등 확대 */
    transform: scale(1.5, 0.8);      /* X, Y 개별 */
}

/* 회전 (Rotate) */
.rotate {
    transform: rotate(45deg);        /* 시계 방향 45도 */
    transform: rotate(-30deg);       /* 반시계 방향 30도 */
    transform: rotate(0.5turn);      /* 180도 (반 바퀴) */
}

/* 기울이기 (Skew) */
.skew {
    transform: skewX(20deg);         /* X축 기울이기 */
    transform: skewY(10deg);         /* Y축 기울이기 */
    transform: skew(20deg, 10deg);   /* X, Y 동시 */
}

/* 복합 Transform */
.combined {
    transform: translateX(50px) rotate(45deg) scale(1.2);
    /* 순서 중요! 오른쪽부터 적용됨 */
}
```

### 2.2 Transform Origin

```css
/* 변환 기준점 설정 */
.origin {
    transform-origin: center;        /* 기본값 (중앙) */
    transform-origin: top left;      /* 왼쪽 위 */
    transform-origin: 50% 100%;      /* 하단 중앙 */
    transform-origin: 0 0;           /* 왼쪽 위 (px) */
}

/* 회전 예시 - 기준점에 따른 차이 */
.rotate-center {
    transform-origin: center;
    transform: rotate(45deg);
    /* 중앙을 기준으로 회전 */
}

.rotate-corner {
    transform-origin: top left;
    transform: rotate(45deg);
    /* 왼쪽 위를 기준으로 회전 */
}
```

### 2.3 3D Transform

```css
/* 3D 이동 */
.translate3d {
    transform: translateZ(50px);
    transform: translate3d(50px, 30px, 20px);
}

/* 3D 회전 */
.rotate3d {
    transform: rotateX(45deg);       /* X축 기준 회전 */
    transform: rotateY(45deg);       /* Y축 기준 회전 */
    transform: rotateZ(45deg);       /* Z축 기준 회전 (= rotate()) */
    transform: rotate3d(1, 1, 0, 45deg); /* 커스텀 축 */
}

/* 원근감 (Perspective) */
.perspective-parent {
    perspective: 1000px;             /* 부모에 설정 */
}

.perspective-child {
    transform: perspective(1000px) rotateY(45deg);
    /* 또는 개별 요소에 설정 */
}

/* 3D 공간 유지 */
.preserve-3d {
    transform-style: preserve-3d;    /* 자식 요소도 3D 공간 유지 */
}

/* 뒷면 보이기 설정 */
.backface {
    backface-visibility: hidden;     /* 뒷면 숨김 (카드 뒤집기에 유용) */
}
```

### 2.4 3D 카드 뒤집기 예제

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <style>
        .card-container {
            width: 200px;
            height: 300px;
            perspective: 1000px;
        }

        .card {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 0.6s ease;
        }

        .card-container:hover .card {
            transform: rotateY(180deg);
        }

        .card-face {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            font-size: 24px;
            font-weight: bold;
        }

        .card-front {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .card-back {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            transform: rotateY(180deg);
        }
    </style>
</head>
<body>
    <div class="card-container">
        <div class="card">
            <div class="card-face card-front">앞면</div>
            <div class="card-face card-back">뒷면</div>
        </div>
    </div>
</body>
</html>
```

---

## 3. CSS Animation (@keyframes)

### 3.1 기본 구조

```css
/* 애니메이션 정의 */
@keyframes slidein {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* 퍼센트 기반 정의 */
@keyframes bounce {
    0% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-30px);
    }
    100% {
        transform: translateY(0);
    }
}

/* 애니메이션 적용 */
.animated-element {
    animation-name: slidein;
    animation-duration: 1s;
    animation-timing-function: ease-out;
    animation-delay: 0s;
    animation-iteration-count: 1;
    animation-direction: normal;
    animation-fill-mode: forwards;
    animation-play-state: running;
}

/* 단축 속성 */
.animated-element {
    animation: slidein 1s ease-out 0s 1 normal forwards running;
    /* name | duration | timing | delay | count | direction | fill | state */
}

/* 더 간단한 형태 */
.simple {
    animation: bounce 0.5s ease infinite;
}
```

### 3.2 Animation 속성 상세

```css
.animation-props {
    /* 반복 횟수 */
    animation-iteration-count: 3;        /* 3회 */
    animation-iteration-count: infinite; /* 무한 */

    /* 방향 */
    animation-direction: normal;          /* 정방향 */
    animation-direction: reverse;         /* 역방향 */
    animation-direction: alternate;       /* 번갈아 (정→역→정...) */
    animation-direction: alternate-reverse; /* 번갈아 (역→정→역...) */

    /* 채우기 모드 (애니메이션 전후 상태) */
    animation-fill-mode: none;            /* 기본값 */
    animation-fill-mode: forwards;        /* 끝 상태 유지 */
    animation-fill-mode: backwards;       /* 시작 상태 적용 (delay 동안) */
    animation-fill-mode: both;            /* 시작+끝 모두 */

    /* 재생 상태 */
    animation-play-state: running;        /* 재생 */
    animation-play-state: paused;         /* 일시정지 */
}
```

### 3.3 실전 애니메이션 예제

```css
/* 로딩 스피너 */
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

/* 펄스 효과 */
@keyframes pulse {
    0% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7);
    }
    70% {
        transform: scale(1.05);
        box-shadow: 0 0 0 15px rgba(52, 152, 219, 0);
    }
    100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(52, 152, 219, 0);
    }
}

.pulse-btn {
    animation: pulse 2s infinite;
}

/* 타이핑 효과 */
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink {
    50% { border-color: transparent; }
}

.typing-text {
    width: 0;
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid;
    animation:
        typing 3s steps(30) forwards,
        blink 0.75s step-end infinite;
}

/* 흔들림 효과 */
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.shake-error {
    animation: shake 0.5s ease-in-out;
}

/* 페이드 인 업 */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.6s ease-out forwards;
}

/* 시차 애니메이션 (Staggered) */
.item { animation: fadeInUp 0.5s ease-out forwards; opacity: 0; }
.item:nth-child(1) { animation-delay: 0.1s; }
.item:nth-child(2) { animation-delay: 0.2s; }
.item:nth-child(3) { animation-delay: 0.3s; }
.item:nth-child(4) { animation-delay: 0.4s; }
```

---

## 4. 스크롤 기반 애니메이션

### 4.1 Intersection Observer (JavaScript)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <style>
        .animate-on-scroll {
            opacity: 0;
            transform: translateY(50px);
            transition: opacity 0.6s ease, transform 0.6s ease;
        }

        .animate-on-scroll.visible {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="animate-on-scroll">스크롤하면 나타나요</div>

    <script>
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1,  // 10% 보이면 트리거
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    </script>
</body>
</html>
```

### 4.2 CSS Scroll-Driven Animations (최신)

```css
/* Chrome 115+, scroll() 함수 */
@keyframes reveal {
    from { opacity: 0; transform: translateY(100px); }
    to { opacity: 1; transform: translateY(0); }
}

.scroll-reveal {
    animation: reveal linear both;
    animation-timeline: view();
    animation-range: entry 0% cover 40%;
}

/* 스크롤 진행도 표시 */
@keyframes progress {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

.progress-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: #3498db;
    transform-origin: left;
    animation: progress linear;
    animation-timeline: scroll();
}
```

---

## 5. 성능 최적화

### 5.1 GPU 가속 속성

```css
/* GPU로 처리되는 속성 (권장) */
.performant {
    transform: translateX(100px);  /* ✅ 합성 레이어 */
    opacity: 0.5;                  /* ✅ 합성 레이어 */
}

/* CPU로 처리되는 속성 (주의) */
.slow {
    left: 100px;      /* ❌ 레이아웃 재계산 */
    width: 200px;     /* ❌ 레이아웃 재계산 */
    margin-left: 50px; /* ❌ 레이아웃 재계산 */
}

/* will-change로 최적화 힌트 */
.optimized {
    will-change: transform, opacity;
    /* 주의: 과도한 사용은 오히려 성능 저하 */
}

/* 애니메이션 후 will-change 제거 */
.animated {
    transition: transform 0.3s;
}
.animated:hover {
    will-change: transform;
    transform: scale(1.1);
}
```

### 5.2 성능 팁

```css
/* ✅ 좋은 예: transform 사용 */
.good {
    transform: translateY(-10px);
}

/* ❌ 나쁜 예: top 사용 */
.bad {
    position: relative;
    top: -10px;
}

/* ✅ 좋은 예: opacity */
.fade-good {
    opacity: 0;
}

/* ❌ 나쁜 예: visibility + display 변경 */
.fade-bad {
    visibility: hidden;
}

/* 레이어 강제 생성 (디버깅용) */
.debug-layer {
    transform: translateZ(0);
    /* 또는 */
    will-change: transform;
}
```

---

## 6. 접근성 고려

### 6.1 모션 감소 설정 존중

```css
/* 기본 애니메이션 */
.animated {
    animation: bounce 0.5s ease infinite;
    transition: transform 0.3s ease;
}

/* 모션 감소 선호 시 */
@media (prefers-reduced-motion: reduce) {
    .animated {
        animation: none;
        transition: none;
    }

    /* 또는 더 짧고 단순하게 */
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* 필수 애니메이션만 유지 */
@media (prefers-reduced-motion: reduce) {
    .spinner {
        /* 로딩 스피너는 유지 (기능적) */
        animation: spin 2s linear infinite;
    }

    .decorative-animation {
        /* 장식적 애니메이션은 제거 */
        animation: none;
    }
}
```

### 6.2 자동 재생 주의

```css
/* 자동 재생 애니메이션은 일시정지 제공 */
.auto-play {
    animation: slideshow 10s infinite;
    animation-play-state: running;
}

.auto-play:hover,
.auto-play:focus-within {
    animation-play-state: paused;
}

/* 또는 JavaScript로 제어 */
```

```javascript
// 모션 감소 설정 확인
const prefersReducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)'
).matches;

if (prefersReducedMotion) {
    // 애니메이션 비활성화 또는 단순화
    document.documentElement.classList.add('reduced-motion');
}
```

---

## 정리

### 주요 속성 비교

| 기능 | Transition | Animation |
|------|------------|-----------|
| 트리거 | 상태 변화 필요 (hover 등) | 자동/수동 모두 가능 |
| 복잡도 | 단순 (시작→끝) | 복잡 (다단계 가능) |
| 반복 | 불가 | 가능 (infinite) |
| 중간 상태 | 불가 | 가능 (@keyframes) |
| 사용 사례 | 호버 효과, 상태 전환 | 로딩, 배경 애니메이션 |

### Transform 요약

| 함수 | 설명 | 예시 |
|------|------|------|
| translate | 이동 | `translateX(50px)` |
| scale | 크기 | `scale(1.5)` |
| rotate | 회전 | `rotate(45deg)` |
| skew | 기울이기 | `skewX(20deg)` |

### 성능 우선순위

1. `transform`, `opacity` 사용 (GPU 가속)
2. `will-change` 신중하게 사용
3. `left`, `width` 등 레이아웃 속성 피하기

### 다음 단계
- [15. JS 모듈](./15_JS_Modules.md): JavaScript 모듈 시스템

---

## 연습 문제

### 연습 1: 애니메이션 내비게이션 메뉴

CSS 트랜지션(transition)만 사용하여(JavaScript 없이) 다음과 같은 애니메이션 동작을 가진 가로 내비게이션 바를 구축하세요:

1. 각 내비게이션 링크에는 호버(hover) 시 `width: 0`에서 `width: 100%`로 양쪽에서 중앙을 기준으로 펼쳐지는 색상 밑줄이 있습니다.
2. 호버 시 링크 텍스트 색상이 0.25초에 걸쳐 부드럽게 변합니다.
3. 부모 `<li>`에 호버 시 드롭다운 서브메뉴가 `max-height` 트랜지션으로 아래로 슬라이드됩니다.

> **성능 참고**: 가능한 경우 `transform`과 `opacity`를 사용하세요. `max-height` 애니메이션이 GPU 가속을 사용하지 않음에도 여기서 허용 가능한 이유를 주석으로 설명하세요.

### 연습 2: 로딩 스켈레톤 스크린(Loading Skeleton Screen)

CSS 애니메이션을 사용하여 스켈레톤(skeleton) 로딩 화면을 구현하세요:

1. 이미지, 제목 줄, 본문 두 줄의 플레이스홀더 블록이 있는 카드 형태의 스켈레톤을 만드세요.
2. 반투명 그라디언트(gradient)가 각 블록을 왼쪽에서 오른쪽으로 무한 반복하며 이동하는 시머(shimmer) 효과를 `@keyframes` 애니메이션으로 적용하세요.
3. `setTimeout`을 사용해 데이터 로딩을 시뮬레이션하고, `.loaded` 클래스 토글로 스켈레톤을 `opacity`와 `transition`을 사용해 페이드 아웃(fade out)하고 실제 콘텐츠를 페이드 인(fade in)하세요.

```css
/* 시머(Shimmer) 키프레임 스켈레톤 */
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position:  400px 0; }
}

.skeleton-block {
    background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
    background-size: 800px 100%;
    animation: shimmer 1.5s infinite;
}
```

### 연습 3: 스크롤 트리거 섹션 등장 효과

다섯 개의 콘텐츠 섹션이 수직으로 쌓인 단일 페이지 레이아웃을 만드세요. 각 섹션은 뷰포트에 스크롤될 때 애니메이션으로 등장해야 합니다:

1. `IntersectionObserver`를 `threshold` 0.15로 사용하여 가시성을 감지합니다.
2. 등장 방향을 번갈아가며 적용합니다: 홀수 섹션은 왼쪽에서, 짝수 섹션은 오른쪽에서 슬라이드 인(slide in)됩니다.
3. 섹션이 한 번 애니메이션되면 `unobserve`를 호출하여 애니메이션이 한 번만 재생되도록 합니다.
4. 모든 애니메이션 CSS를 `@media (prefers-reduced-motion: no-preference)` 블록으로 감싸서, 모션 감소를 선호하는 사용자는 애니메이션 없이 즉시 콘텐츠를 볼 수 있도록 합니다.

### 연습 4: 부드러운 높이 트랜지션이 있는 CSS 전용 아코디언(Accordion) (심화)

JavaScript 없이 `:checked` 가상 클래스(pseudo-class)만으로 FAQ 아코디언을 구현하세요:

1. 각 FAQ 항목은 `<details>` 요소 또는 숨겨진 체크박스(checkbox) + 레이블(label) 트릭을 사용합니다.
2. 패널이 열릴 때 `max-height: 0`에서 `max-height: 500px`로(`overflow: hidden` 적용) 0.4초에 걸쳐 트랜지션됩니다.
3. 오른쪽의 `+` 아이콘이 패널이 열리면 45도 회전하여 `×`가 되도록 `transform: rotate` 트랜지션을 사용합니다.
4. 한 번에 하나의 패널만 열려야 합니다 — 순수 CSS로 "한 번에 하나만 열림"을 구현하기 어려운 CSS의 한계와 우회 방법을 주석으로 설명하세요.

---

## 참고 자료

- [MDN CSS Transitions](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Transitions)
- [MDN CSS Animations](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Animations)
- [Cubic Bezier Generator](https://cubic-bezier.com/)
- [Animate.css](https://animate.style/) - 애니메이션 라이브러리
