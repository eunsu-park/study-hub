# 12. SEO 기초 (Search Engine Optimization)

**이전**: [웹 접근성](./11_Web_Accessibility.md) | **다음**: [빌드 도구와 환경](./13_Build_Tools_Environment.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 검색 엔진이 웹 페이지를 크롤링(crawling)·인덱싱(indexing)·랭킹(ranking)하는 방식을 설명할 수 있다
2. 온페이지 SEO(on-page SEO)를 위한 효과적인 타이틀 태그(title tag)·메타 디스크립션(meta description)·정식 URL(canonical URL)을 작성할 수 있다
3. 소셜 미디어 공유를 위한 Open Graph와 Twitter Card 메타 태그를 구현할 수 있다
4. 아티클·제품·FAQ·브레드크럼(breadcrumb)에 JSON-LD 구조화 데이터(structured data)를 추가할 수 있다
5. 크롤러 동작을 제어하기 위해 `robots.txt`와 `sitemap.xml`을 설정할 수 있다
6. Core Web Vitals 모범 사례를 적용하여 페이지 속도를 최적화할 수 있다
7. Google Search Console과 Lighthouse를 사용해 사이트의 SEO 상태를 평가할 수 있다

---

훌륭한 웹사이트를 만들더라도 아무도 찾지 못한다면 의미가 없습니다. 검색 엔진은 사용자가 새로운 콘텐츠를 발견하는 주된 경로이며, SEO는 페이지를 검색 엔진에 노출시키는 실천입니다. SEO를 이해하는 것은 단순히 마케팅 역량이 아닙니다 — HTML 구조 설계, 콘텐츠 작성, 사이트의 기술적 기반 설계 방식에 직접적인 영향을 미칩니다.

## 목차
1. [SEO 개요](#1-seo-개요)
2. [메타 태그](#2-메타-태그)
3. [구조화 데이터](#3-구조화-데이터)
4. [기술적 SEO](#4-기술적-seo)
5. [콘텐츠 SEO](#5-콘텐츠-seo)
6. [측정과 도구](#6-측정과-도구)
7. [연습 문제](#7-연습-문제)

---

## 1. SEO 개요

### 1.1 SEO란?

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEO 정의 및 목적                              │
│                                                                 │
│   SEO (Search Engine Optimization):                            │
│   검색 엔진 결과 페이지(SERP)에서 웹사이트의 가시성을             │
│   향상시키는 과정                                                │
│                                                                 │
│   목적:                                                         │
│   - 유기적(Organic) 트래픽 증가                                 │
│   - 브랜드 인지도 향상                                          │
│   - 신뢰도 구축                                                 │
│   - 전환율 개선                                                 │
│                                                                 │
│   검색 엔진 작동 원리:                                           │
│   1. 크롤링 (Crawling): 웹 페이지 발견                          │
│   2. 인덱싱 (Indexing): 콘텐츠 분석 및 저장                     │
│   3. 랭킹 (Ranking): 검색어에 따른 순위 결정                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 SEO 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEO 유형 분류                                 │
│                                                                 │
│   온페이지 SEO (On-page):                                       │
│   - 메타 태그 (title, description)                             │
│   - 콘텐츠 품질                                                 │
│   - 키워드 최적화                                               │
│   - 이미지 최적화                                               │
│   - 내부 링크                                                   │
│                                                                 │
│   오프페이지 SEO (Off-page):                                    │
│   - 백링크 (다른 사이트에서의 링크)                             │
│   - 소셜 시그널                                                 │
│   - 브랜드 멘션                                                 │
│   - 게스트 포스팅                                               │
│                                                                 │
│   기술적 SEO (Technical):                                       │
│   - 사이트 속도                                                 │
│   - 모바일 친화성                                               │
│   - HTTPS                                                       │
│   - 구조화 데이터                                               │
│   - 크롤링/인덱싱 제어                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 메타 태그

### 2.1 필수 메타 태그

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <!-- 문자 인코딩 -->
  <meta charset="UTF-8">

  <!-- 반응형 뷰포트 -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- 페이지 제목 (50-60자 권장) -->
  <title>웹 개발 기초 가이드 - 초보자를 위한 완벽한 튜토리얼 | 사이트명</title>

  <!-- 메타 설명 (150-160자 권장) -->
  <meta name="description" content="HTML, CSS, JavaScript의 기초부터 실전까지.
    웹 개발 입문자를 위한 단계별 가이드. 무료 예제 코드와 함께 배우세요.">

  <!-- 검색 엔진 지시사항 -->
  <meta name="robots" content="index, follow">

  <!-- 정식 URL (중복 콘텐츠 방지) -->
  <link rel="canonical" href="https://example.com/web-development-guide">
</head>
</html>
```

### 2.2 Open Graph (소셜 미디어)

```html
<!-- Facebook / LinkedIn -->
<meta property="og:type" content="article">
<meta property="og:title" content="웹 개발 기초 가이드">
<meta property="og:description" content="초보자를 위한 완벽한 웹 개발 튜토리얼">
<meta property="og:image" content="https://example.com/images/og-image.jpg">
<meta property="og:url" content="https://example.com/web-development-guide">
<meta property="og:site_name" content="사이트명">
<meta property="og:locale" content="ko_KR">

<!-- 이미지 권장 사이즈: 1200x630px -->
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="웹 개발 가이드 썸네일">
```

### 2.3 Twitter Card

```html
<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@username">
<meta name="twitter:creator" content="@author">
<meta name="twitter:title" content="웹 개발 기초 가이드">
<meta name="twitter:description" content="초보자를 위한 완벽한 웹 개발 튜토리얼">
<meta name="twitter:image" content="https://example.com/images/twitter-image.jpg">

<!-- 카드 타입: summary, summary_large_image, app, player -->
```

### 2.4 기타 중요 메타 태그

```html
<!-- 언어/지역 설정 -->
<link rel="alternate" hreflang="ko" href="https://example.com/ko/">
<link rel="alternate" hreflang="en" href="https://example.com/en/">
<link rel="alternate" hreflang="x-default" href="https://example.com/">

<!-- 파비콘 -->
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">

<!-- 테마 색상 (모바일 브라우저) -->
<meta name="theme-color" content="#4285f4">

<!-- 작성자 정보 -->
<meta name="author" content="작성자 이름">

<!-- 발행일/수정일 -->
<meta property="article:published_time" content="2024-01-15T09:00:00+09:00">
<meta property="article:modified_time" content="2024-03-20T14:30:00+09:00">
```

---

## 3. 구조화 데이터

### 3.1 JSON-LD 기본

```html
<!-- JSON-LD 구조화 데이터 (권장 형식) -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "name": "웹 개발 기초 가이드",
  "description": "초보자를 위한 완벽한 웹 개발 튜토리얼",
  "url": "https://example.com/web-development-guide"
}
</script>
```

### 3.2 조직 정보

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "name": "회사명",
  "url": "https://example.com",
  "logo": "https://example.com/logo.png",
  "sameAs": [
    "https://www.facebook.com/company",
    "https://www.twitter.com/company",
    "https://www.linkedin.com/company/company"
  ],
  "contactPoint": {
    "@type": "ContactPoint",
    "telephone": "+82-2-1234-5678",
    "contactType": "customer service",
    "availableLanguage": ["Korean", "English"]
  }
}
</script>
```

### 3.3 기사/블로그 포스트

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "웹 개발 기초 가이드",
  "description": "초보자를 위한 완벽한 웹 개발 튜토리얼",
  "image": [
    "https://example.com/images/1x1.jpg",
    "https://example.com/images/4x3.jpg",
    "https://example.com/images/16x9.jpg"
  ],
  "author": {
    "@type": "Person",
    "name": "작성자 이름",
    "url": "https://example.com/author"
  },
  "publisher": {
    "@type": "Organization",
    "name": "사이트명",
    "logo": {
      "@type": "ImageObject",
      "url": "https://example.com/logo.png"
    }
  },
  "datePublished": "2024-01-15",
  "dateModified": "2024-03-20",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://example.com/web-development-guide"
  }
}
</script>
```

### 3.4 FAQ 페이지

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "HTML이란 무엇인가요?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "HTML(HyperText Markup Language)은 웹 페이지의 구조를 정의하는 마크업 언어입니다."
      }
    },
    {
      "@type": "Question",
      "name": "CSS는 어디에 사용되나요?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "CSS(Cascading Style Sheets)는 웹 페이지의 시각적 스타일을 정의하는 데 사용됩니다."
      }
    }
  ]
}
</script>
```

### 3.5 제품 정보

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "제품명",
  "image": "https://example.com/product.jpg",
  "description": "제품 설명",
  "sku": "SKU12345",
  "brand": {
    "@type": "Brand",
    "name": "브랜드명"
  },
  "offers": {
    "@type": "Offer",
    "url": "https://example.com/product",
    "priceCurrency": "KRW",
    "price": "99000",
    "availability": "https://schema.org/InStock",
    "seller": {
      "@type": "Organization",
      "name": "판매자명"
    }
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.5",
    "reviewCount": "128"
  }
}
</script>
```

### 3.6 브레드크럼

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "홈",
      "item": "https://example.com"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "튜토리얼",
      "item": "https://example.com/tutorials"
    },
    {
      "@type": "ListItem",
      "position": 3,
      "name": "웹 개발",
      "item": "https://example.com/tutorials/web-development"
    }
  ]
}
</script>
```

---

## 4. 기술적 SEO

### 4.1 사이트맵 (sitemap.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/</loc>
    <lastmod>2024-03-20</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://example.com/about</loc>
    <lastmod>2024-02-15</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://example.com/blog/post-1</loc>
    <lastmod>2024-03-18</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.6</priority>
  </url>
</urlset>
```

### 4.2 robots.txt

```
# robots.txt
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /private/
Disallow: /api/
Disallow: /*?*  # URL 파라미터 제외

# 사이트맵 위치
Sitemap: https://example.com/sitemap.xml

# 특정 봇 설정
User-agent: Googlebot
Allow: /
Crawl-delay: 1

User-agent: Bingbot
Crawl-delay: 5
```

### 4.3 페이지 속도 최적화

```html
<!-- 리소스 힌트 -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="dns-prefetch" href="https://cdn.example.com">
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>

<!-- 이미지 최적화 -->
<img
  src="image.webp"
  alt="설명"
  width="800"
  height="600"
  loading="lazy"
  decoding="async"
>

<!-- 반응형 이미지 -->
<picture>
  <source media="(min-width: 800px)" srcset="large.webp">
  <source media="(min-width: 400px)" srcset="medium.webp">
  <img src="small.webp" alt="설명">
</picture>

<!-- CSS 최적화 -->
<link rel="stylesheet" href="critical.css">
<link rel="stylesheet" href="non-critical.css" media="print" onload="this.media='all'">

<!-- JavaScript 최적화 -->
<script src="app.js" defer></script>
<script src="analytics.js" async></script>
```

### 4.4 Core Web Vitals

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core Web Vitals 지표                          │
│                                                                 │
│   LCP (Largest Contentful Paint):                              │
│   - 가장 큰 콘텐츠 요소 로딩 시간                                │
│   - 목표: 2.5초 이내                                            │
│   - 개선: 이미지 최적화, 서버 응답 시간                          │
│                                                                 │
│   FID (First Input Delay) → INP (Interaction to Next Paint):   │
│   - 첫 상호작용 지연 시간 → 전체 상호작용 반응성                 │
│   - 목표: 100ms 이내 → 200ms 이내                              │
│   - 개선: JavaScript 최적화, 메인 스레드 차단 최소화            │
│                                                                 │
│   CLS (Cumulative Layout Shift):                               │
│   - 레이아웃 변경 누적 점수                                      │
│   - 목표: 0.1 이하                                              │
│   - 개선: 이미지/광고 크기 명시, 폰트 로딩 최적화               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 모바일 최적화

```html
<!-- 모바일 친화적 설정 -->
<meta name="viewport" content="width=device-width, initial-scale=1">

<!-- 터치 타겟 크기 (최소 48x48px) -->
<style>
button, a {
  min-height: 48px;
  min-width: 48px;
  padding: 12px 16px;
}
</style>

<!-- 가독성 (최소 16px 폰트) -->
<style>
body {
  font-size: 16px;
  line-height: 1.5;
}
</style>
```

---

## 5. 콘텐츠 SEO

### 5.1 키워드 최적화

```html
<!-- 타이틀에 주요 키워드 포함 -->
<title>웹 개발 기초 | HTML, CSS, JavaScript 튜토리얼</title>

<!-- H1은 페이지당 하나, 키워드 포함 -->
<h1>웹 개발 기초 가이드: HTML, CSS, JavaScript 시작하기</h1>

<!-- 자연스러운 키워드 배치 -->
<p>
  <strong>웹 개발</strong>을 시작하려면 HTML, CSS, JavaScript의
  기초를 이해해야 합니다. 이 <em>웹 개발 튜토리얼</em>에서는...
</p>
```

### 5.2 시맨틱 마크업

```html
<article>
  <header>
    <h1>웹 개발 기초 가이드</h1>
    <p>
      <time datetime="2024-03-20">2024년 3월 20일</time> |
      <a href="/author/john">작성자 이름</a>
    </p>
  </header>

  <section>
    <h2>HTML 기초</h2>
    <p>HTML은 웹 페이지의 구조를 정의합니다...</p>
  </section>

  <section>
    <h2>CSS 스타일링</h2>
    <p>CSS는 시각적 표현을 담당합니다...</p>
  </section>

  <aside>
    <h3>관련 글</h3>
    <ul>
      <li><a href="/css-layout">CSS 레이아웃 마스터</a></li>
      <li><a href="/javascript-basics">JavaScript 입문</a></li>
    </ul>
  </aside>

  <footer>
    <p>태그: <a href="/tag/html">HTML</a>, <a href="/tag/css">CSS</a></p>
  </footer>
</article>
```

### 5.3 링크 최적화

```html
<!-- 내부 링크 (명확한 앵커 텍스트) -->
<a href="/css-tutorial">CSS 튜토리얼 보기</a>
<!-- 피해야 할 것: <a href="/css-tutorial">여기를 클릭</a> -->

<!-- 외부 링크 -->
<a href="https://developer.mozilla.org"
   target="_blank"
   rel="noopener noreferrer">
  MDN Web Docs
</a>

<!-- nofollow (신뢰하지 않는 링크) -->
<a href="https://external-site.com" rel="nofollow">외부 사이트</a>

<!-- 브레드크럼 -->
<nav aria-label="브레드크럼">
  <ol>
    <li><a href="/">홈</a></li>
    <li><a href="/tutorials">튜토리얼</a></li>
    <li aria-current="page">웹 개발 기초</li>
  </ol>
</nav>
```

### 5.4 URL 구조

```
좋은 URL 구조:
✓ https://example.com/tutorials/web-development-basics
✓ https://example.com/products/electronics/smartphones

피해야 할 URL:
✗ https://example.com/page?id=123&cat=5
✗ https://example.com/p/a/b/c/d/e/article
✗ https://example.com/웹개발 (한글 URL은 인코딩됨)

규칙:
- 소문자 사용
- 하이픈(-) 으로 단어 구분
- 간결하고 설명적
- 계층 구조 반영
- 불필요한 파라미터 제거
```

---

## 6. 측정과 도구

### 6.1 Google Search Console

```
주요 기능:
- 검색 성능 분석 (클릭, 노출, CTR, 순위)
- 색인 상태 확인
- 사이트맵 제출
- 크롤링 오류 확인
- Core Web Vitals 리포트
- 모바일 사용성 리포트

설정:
1. Google Search Console 접속
2. 사이트 소유권 확인 (HTML 태그, DNS, 파일 업로드)
3. 사이트맵 제출
```

### 6.2 Google Analytics 4

```html
<!-- GA4 설치 -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>

<!-- 이벤트 추적 -->
<script>
  // 버튼 클릭 추적
  document.querySelector('.cta-button').addEventListener('click', () => {
    gtag('event', 'cta_click', {
      'event_category': 'engagement',
      'event_label': 'signup_button'
    });
  });
</script>
```

### 6.3 SEO 도구들

```
무료 도구:
- Google Search Console: 검색 성능, 색인 상태
- Google PageSpeed Insights: 페이지 속도
- Lighthouse: 종합 웹사이트 품질 측정
- Screaming Frog (무료 버전): 크롤링, 사이트 분석
- Google Rich Results Test: 구조화 데이터 테스트

유료 도구:
- Ahrefs: 백링크 분석, 키워드 리서치
- SEMrush: 경쟁사 분석, 키워드 추적
- Moz Pro: 도메인 권위도, SEO 분석
```

### 6.4 체크리스트

```
┌─────────────────────────────────────────────────────────────────┐
│                   SEO 점검 체크리스트                            │
│                                                                 │
│ 기본 설정:                                                       │
│ □ 고유하고 설명적인 title 태그                                   │
│ □ 매력적인 meta description                                      │
│ □ 적절한 canonical URL                                          │
│ □ robots.txt 설정                                                │
│ □ sitemap.xml 생성 및 제출                                      │
│ □ HTTPS 적용                                                     │
│                                                                 │
│ 콘텐츠:                                                          │
│ □ H1 태그 (페이지당 하나)                                       │
│ □ 논리적인 헤딩 구조 (H1→H2→H3)                                 │
│ □ 이미지 alt 텍스트                                              │
│ □ 내부 링크 구조                                                 │
│ □ 모바일 친화적 콘텐츠                                           │
│                                                                 │
│ 기술적:                                                          │
│ □ Core Web Vitals 통과                                          │
│ □ 모바일 반응형                                                  │
│ □ 구조화 데이터                                                  │
│ □ 404 페이지 처리                                                │
│ □ 리다이렉트 최적화                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 연습 문제

### 연습 1: 메타 태그 작성
블로그 포스트에 필요한 메타 태그를 작성하세요.

```html
<!-- 예시 답안 -->
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <title>React Hooks 완벽 가이드 - useState부터 커스텀 훅까지 | 블로그명</title>
  <meta name="description" content="React Hooks의 모든 것을 알아봅니다.
    useState, useEffect, useContext부터 커스텀 훅 작성법까지 예제와 함께 설명합니다.">

  <link rel="canonical" href="https://blog.example.com/react-hooks-guide">

  <!-- Open Graph -->
  <meta property="og:type" content="article">
  <meta property="og:title" content="React Hooks 완벽 가이드">
  <meta property="og:description" content="React Hooks의 모든 것을 예제와 함께">
  <meta property="og:image" content="https://blog.example.com/images/react-hooks-og.jpg">
  <meta property="og:url" content="https://blog.example.com/react-hooks-guide">

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="React Hooks 완벽 가이드">
  <meta name="twitter:description" content="React Hooks의 모든 것을 예제와 함께">
  <meta name="twitter:image" content="https://blog.example.com/images/react-hooks-twitter.jpg">
</head>
```

### 연습 2: 구조화 데이터 작성
레시피 페이지의 구조화 데이터를 작성하세요.

```html
<!-- 예시 답안 -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "김치찌개",
  "image": "https://example.com/kimchi-jjigae.jpg",
  "author": {
    "@type": "Person",
    "name": "요리사 이름"
  },
  "datePublished": "2024-03-15",
  "description": "진한 맛의 전통 김치찌개 레시피",
  "prepTime": "PT10M",
  "cookTime": "PT20M",
  "totalTime": "PT30M",
  "recipeYield": "2인분",
  "recipeIngredient": [
    "김치 200g",
    "돼지고기 150g",
    "두부 1/2모",
    "양파 1/2개"
  ],
  "recipeInstructions": [
    {
      "@type": "HowToStep",
      "text": "김치를 한입 크기로 썬다"
    },
    {
      "@type": "HowToStep",
      "text": "돼지고기와 함께 볶는다"
    }
  ],
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.8",
    "ratingCount": "156"
  }
}
</script>
```

### 연습 3: robots.txt 작성
다음 요구사항에 맞는 robots.txt를 작성하세요.
- 관리자 페이지 (/admin/) 차단
- API 엔드포인트 (/api/) 차단
- 검색 결과 페이지 (/search?*) 차단
- 나머지는 허용

```
# 예시 답안
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /api/
Disallow: /search

Sitemap: https://example.com/sitemap.xml
```

---

## 다음 단계
- [11. 웹 접근성](./11_Web_Accessibility.md)
- [13. 빌드 도구와 환경](./13_Build_Tools_Environment.md)

## 참고 자료
- [Google Search Central](https://developers.google.com/search)
- [Schema.org](https://schema.org/)
- [Web.dev SEO Guide](https://web.dev/learn/seo/)
- [Moz Beginner's Guide to SEO](https://moz.com/beginners-guide-to-seo)
