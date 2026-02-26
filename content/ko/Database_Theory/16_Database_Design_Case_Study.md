# 데이터베이스 설계 사례 연구

**이전**: [15. NewSQL과 현대 트렌드](./15_NewSQL_and_Modern_Trends.md)

---

이 종합 레슨은 앞선 15개 레슨의 모든 내용을 포괄적이고 종단간 데이터베이스 설계 연습으로 통합합니다. 요구사항 수집부터 최적화된 쿼리 작성까지 전자상거래 플랫폼의 완전한 설계 라이프사이클을 안내하며, ER 모델링, 정규화, 인덱싱, 트랜잭션 및 동시성 제어의 이론을 현실적인 시나리오에 적용합니다. 두 번째 미니 사례 연구(소셜 미디어 플랫폼)는 추가 연습을 제공하며, 레슨은 설계 검토 체크리스트와 일반적인 실수 카탈로그로 마무리됩니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 데이터베이스 시스템에 대한 요구사항 분석 수행
2. 비즈니스 요구사항에서 완전한 ER 다이어그램 생성
3. ER 모델을 관계형 스키마로 매핑하고 BCNF로 정규화
4. 물리적 설계에 대한 정보에 입각한 결정 (인덱싱, 파티셔닝, 비정규화)
5. 쿼리 플랜 분석 및 최적화 기술 적용
6. 적절한 격리 수준으로 트랜잭션 전략 설계
7. 설계 트레이드오프 평가 및 결정 문서화
8. 일반적인 데이터베이스 설계 실수를 인식하고 회피

---

## 목차

1. [설계 라이프사이클 개요](#1-설계-라이프사이클-개요)
2. [단계 1: 요구사항 분석](#2-단계-1-요구사항-분석)
3. [단계 2: 개념적 설계 (ER 다이어그램)](#3-단계-2-개념적-설계-er-다이어그램)
4. [단계 3: 논리적 설계 (관계형 스키마)](#4-단계-3-논리적-설계-관계형-스키마)
5. [단계 4: 물리적 설계](#5-단계-4-물리적-설계)
6. [단계 5: 쿼리 최적화](#6-단계-5-쿼리-최적화)
7. [단계 6: 트랜잭션 설계](#7-단계-6-트랜잭션-설계)
8. [대안 사례 연구: 소셜 미디어 플랫폼](#8-대안-사례-연구-소셜-미디어-플랫폼)
9. [설계 검토 체크리스트](#9-설계-검토-체크리스트)
10. [일반적인 설계 실수 및 회피 방법](#10-일반적인-설계-실수-및-회피-방법)
11. [연습 문제](#11-연습-문제)
12. [참고문헌](#12-참고문헌)

---

## 1. 설계 라이프사이클 개요

데이터베이스 설계는 일회성 프로세스가 아닙니다. 피드백 루프가 있는 구조화된 라이프사이클을 따릅니다:

```
┌──────────────────────────────────────────────────────────────┐
│                    데이터베이스 설계 라이프사이클              │
│                                                              │
│  단계 1            단계 2            단계 3                  │
│  요구사항     ──▶  개념적       ──▶  논리적                  │
│  분석               설계 (ER)        설계 (관계형)           │
│                                                              │
│       │                                    │                 │
│       │              ┌─────────────────────┘                 │
│       │              │                                       │
│       │              ▼                                       │
│       │         단계 4            단계 5                      │
│       │         물리적       ──▶  쿼리                        │
│       └────────▶설계              최적화                      │
│                                                              │
│                      │                 │                      │
│                      ▼                 ▼                      │
│                 단계 6                                        │
│                 트랜잭션 설계                                 │
│                                                              │
│                      │                                       │
│                      ▼                                       │
│              배포, 모니터링, 반복                             │
└──────────────────────────────────────────────────────────────┘
```

각 단계는 다음 단계로 전달되는 산출물을 생성합니다:

| 단계 | 입력 | 출력 |
|-------|-------|--------|
| **1. 요구사항** | 비즈니스 요구, 인터뷰 | 기능 및 데이터 요구사항 |
| **2. 개념적** | 요구사항 | ER 다이어그램 |
| **3. 논리적** | ER 다이어그램 | 정규화된 관계형 스키마 (DDL) |
| **4. 물리적** | 관계형 스키마 + 쿼리 패턴 | 인덱스, 파티션, 비정규화 |
| **5. 쿼리** | 물리적 스키마 + 일반 쿼리 | 실행 계획이 있는 최적화된 쿼리 |
| **6. 트랜잭션** | 비즈니스 규칙 + 동시성 요구 | 격리 수준, 잠금 전략 |

---

## 2. 단계 1: 요구사항 분석

### 2.1 비즈니스 맥락

물리적 제품을 판매하는 온라인 전자상거래 플랫폼인 **ShopWave**의 데이터베이스를 설계하고 있습니다. 플랫폼은 여러 판매자, 제품 리뷰, 프로모션 및 표준 체크아웃 흐름을 지원합니다.

### 2.2 기능 요구사항

이해관계자와의 인터뷰 후, 다음과 같은 주요 비즈니스 기능을 식별합니다:

```
FR-01: 고객 등록 및 인증
FR-02: 검색 및 필터링이 있는 제품 카탈로그 탐색
FR-03: 이미지, 사양 및 리뷰가 있는 제품 상세 페이지
FR-04: 장바구니 관리 (추가, 제거, 수량 업데이트)
FR-05: 여러 결제 방법으로 체크아웃
FR-06: 주문 추적 (주문부터 배송까지 상태 업데이트)
FR-07: 판매자 관리 (판매자가 제품을 나열하고 재고 관리)
FR-08: 제품 리뷰 및 평가 (1-5점, 텍스트 리뷰)
FR-09: 프로모션 및 할인 코드
FR-10: 위시리스트 (고객이 나중에 제품 저장)
FR-11: 주소 관리 (고객당 여러 배송 주소)
FR-12: 계층이 있는 제품 카테고리 (예: 전자제품 > 전화기 > 스마트폰)
```

### 2.3 데이터 요구사항

기능 요구사항에서 데이터 엔티티와 주요 속성을 추출합니다:

```
DR-01: 고객
  - 이름, 이메일, 비밀번호 해시, 전화번호, 등록 날짜
  - 여러 배송 주소
  - 하나 이상의 결제 방법

DR-02: 제품
  - 이름, 설명, SKU, 가격, 무게, 치수
  - 여러 이미지 (URL)
  - 하나 이상의 카테고리에 속함
  - 정확히 한 판매자가 나열
  - 재고 수량 (재고)

DR-03: 카테고리
  - 이름, 설명
  - 계층적 (부모-자식 관계)

DR-04: 주문
  - 고객, 배송 주소, 주문 날짜, 상태
  - 하나 이상의 주문 항목 (제품, 수량, 주문 시점의 단가)
  - 결제 정보
  - 배송 방법 및 추적 번호

DR-05: 리뷰
  - 고객, 제품, 평가 (1-5), 텍스트, 날짜
  - 각 고객은 제품당 최대 한 번 리뷰 가능

DR-06: 판매자
  - 회사명, 연락처 이메일, 수수료율
  - 지급을 위한 은행 계좌

DR-07: 프로모션
  - 코드, 할인 유형 (백분율/고정), 금액, 유효 날짜
  - 특정 제품, 카테고리 또는 사이트 전체에 적용 가능
  - 사용 제한 (총 및 고객당)
```

### 2.4 물량 분석

데이터 볼륨 이해는 물리적 설계 결정에 도움이 됩니다:

| 엔티티 | 예상 볼륨 | 성장률 |
|--------|-----------------|-------------|
| 고객 | 1백만 | 50K/월 |
| 제품 | 500,000 | 10K/월 |
| 주문 | 1천만 | 500K/월 |
| 주문 항목 | 2천5백만 | 1.25M/월 |
| 리뷰 | 2백만 | 100K/월 |
| 카테고리 | 1,000 | 거의 변경 없음 |
| 판매자 | 5,000 | 200/월 |
| 프로모션 | 한 번에 500개 활성 | 계절별 피크 |

### 2.5 쿼리 패턴

가장 빈번하고 중요한 쿼리를 식별합니다:

```
QP-01: [높은 빈도] 키워드 + 카테고리 + 가격 범위로 제품 검색
QP-02: [높은 빈도] 제품 상세 페이지 (제품 + 이미지 + 리뷰 + 판매자)
QP-03: [높은 빈도] 고객의 장바구니 내용
QP-04: [중요]  주문 생성 (재고 감소, 주문 + 항목 생성)
QP-05: [높은 빈도] 고객의 주문 이력 (상태 포함)
QP-06: [중간]    판매자 대시보드 (주문, 수익, 재고)
QP-07: [중간]    카테고리에서 최고 평가 제품
QP-08: [낮은 빈도]  관리자: 카테고리, 날짜 범위별 판매 보고서
QP-09: [중요]  프로모션 코드 적용 (검증, 할인 계산)
QP-10: [높은 빈도] 고객 위시리스트 표시
```

### 2.6 비기능 요구사항

```
NFR-01: 응답 시간: 읽기 쿼리 < 200ms, 쓰기 < 500ms
NFR-02: 가용성: 99.9% 가동 시간
NFR-03: 데이터 내구성: 주문 및 결제에 대한 데이터 손실 없음
NFR-04: 동시성: 10,000명의 동시 사용자 지원
NFR-05: 확장성: 3년간 10배 성장 처리
NFR-06: 보안: 결제 데이터에 대한 PCI-DSS 준수
```

---

## 3. 단계 2: 개념적 설계 (ER 다이어그램)

### 3.1 엔티티 식별

데이터 요구사항에서 다음 엔티티를 식별합니다:

```
강한 엔티티:           약한 엔티티:            연관 엔티티:
├── Customer            ├── OrderItem          ├── Review
├── Product             ├── CartItem           ├── ProductCategory
├── Category            ├── ProductImage       ├── WishlistItem
├── Order               └── Address            └── PromotionUsage
├── Seller
├── Promotion
├── Cart
└── PaymentMethod
```

### 3.2 ER 다이어그램

```
                                    ┌──────────────┐
                              ┌─────│   Category   │─────┐
                              │     │ id           │     │
                              │     │ name         │     │ parent_id
                              │     │ description  │     │ (self-ref)
                              │     └──────────────┘─────┘
                              │           │
                              │     ┌─────┴──────┐
                              │     │ Product     │
                              │     │ Category    │
                              │     │ (M:N)       │
                              │     └─────┬──────┘
                              │           │
┌──────────────┐        ┌─────┴──────────┴──┐        ┌──────────────┐
│   Seller     │ 1   M  │     Product       │  1   M │ ProductImage │
│ id           │────────│ id                │────────│ id           │
│ company_name │        │ name              │        │ product_id   │
│ email        │        │ description       │        │ url          │
│ commission   │        │ sku               │        │ alt_text     │
│ bank_account │        │ price             │        │ position     │
└──────────────┘        │ weight            │        └──────────────┘
                        │ stock_quantity    │
                        │ seller_id (FK)    │
                        └────────┬──────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
        ┌─────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
        │  Review    │   │  WishlistItem│   │  CartItem   │
        │ id         │   │ id           │   │ id          │
        │ customer_id│   │ customer_id  │   │ cart_id     │
        │ product_id │   │ product_id   │   │ product_id  │
        │ rating     │   │ added_at     │   │ quantity    │
        │ text       │   └─────────────┘   │ added_at    │
        │ created_at │                      └──────┬──────┘
        └─────┬──────┘                             │
              │                                    │
              │                              ┌─────┴──────┐
              │                              │    Cart     │
              │                              │ id          │
              │                              │ customer_id │
              │                              │ created_at  │
              │                              └─────┬──────┘
              │                                    │
              │                                    │
        ┌─────┴──────────────────────────────────┴──┐
        │              Customer                      │
        │ id                                         │
        │ email                                      │
        │ password_hash                              │
        │ first_name, last_name                      │
        │ phone                                      │
        │ created_at                                 │
        └────────────┬─────────────┬────────────────┘
                     │             │
              ┌──────┴──────┐ ┌───┴────────────┐
              │   Address   │ │ PaymentMethod  │
              │ id          │ │ id             │
              │ customer_id │ │ customer_id    │
              │ label       │ │ type           │
              │ street      │ │ last_four      │
              │ city        │ │ token          │
              │ state       │ │ is_default     │
              │ zip_code    │ └────────────────┘
              │ country     │
              │ is_default  │
              └──────┬──────┘
                     │
              ┌──────┴──────────────────────────────┐
              │              Order                    │
              │ id                                   │
              │ customer_id                          │
              │ shipping_address_id                  │
              │ payment_method_id                    │
              │ status                               │
              │ subtotal, discount, tax, total       │
              │ promotion_id (nullable)              │
              │ tracking_number                      │
              │ created_at, updated_at               │
              └──────────┬──────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │     OrderItem       │
              │ id                  │
              │ order_id            │
              │ product_id          │
              │ quantity            │
              │ unit_price          │ ← 주문 시점의 가격
              │ subtotal            │   (현재 가격이 아님!)
              └─────────────────────┘

┌──────────────┐
│  Promotion   │
│ id           │
│ code         │
│ discount_type│  (percentage / fixed_amount)
│ amount       │
│ min_purchase │
│ max_uses     │
│ uses_count   │
│ valid_from   │
│ valid_until  │
│ is_active    │
└──────────────┘
```

### 3.3 카디널리티 요약

| 관계 | 카디널리티 | 참여 |
|-------------|-------------|---------------|
| Customer -- Address | 1:M | 전체 (주문을 위해 최소 하나의 주소 필요) |
| Customer -- PaymentMethod | 1:M | 부분 (결제 방법 없이 등록 가능) |
| Customer -- Order | 1:M | 부분 |
| Customer -- Review | 1:M | 부분 |
| Customer -- Cart | 1:1 | 부분 |
| Customer -- WishlistItem | 1:M | 부분 |
| Order -- OrderItem | 1:M | 전체 (주문은 최소 하나의 항목 필요) |
| Product -- ProductImage | 1:M | 부분 (이미지 선택사항) |
| Product -- Review | 1:M | 부분 |
| Product -- Category | M:N | 전체 (제품은 최소 하나의 카테고리에 있어야 함) |
| Seller -- Product | 1:M | 전체 |
| Category -- Category (self) | 1:M | 부분 (루트 카테고리는 부모 없음) |
| Order -- Promotion | M:1 | 부분 (주문에 프로모션이 없을 수 있음) |
| Customer + Product -- Review | 고유 (고객당 제품당 하나의 리뷰) |

---

## 4. 단계 3: 논리적 설계 (관계형 스키마)

### 4.1 ER에서 관계형으로 매핑

[레슨 4](./04_ER_Modeling.md)의 표준 매핑 규칙을 적용합니다:

**규칙 1: 강한 엔티티는 테이블이 됩니다.**
**규칙 2: 약한 엔티티는 소유자의 PK가 PK의 일부 또는 FK로 테이블이 됩니다.**
**규칙 3: 1:M 관계는 "다(many)" 측에 FK가 됩니다.**
**규칙 4: M:N 관계는 접합 테이블이 됩니다.**

### 4.2 완전한 DDL

```sql
-- ============================================================
-- 고객 및 인증
-- ============================================================

CREATE TABLE customers (
    id              BIGSERIAL PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    phone           VARCHAR(20),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE addresses (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    label           VARCHAR(50) NOT NULL DEFAULT 'Home',  -- 'Home', 'Work', etc.
    street_line1    VARCHAR(255) NOT NULL,
    street_line2    VARCHAR(255),
    city            VARCHAR(100) NOT NULL,
    state           VARCHAR(100),
    zip_code        VARCHAR(20) NOT NULL,
    country         VARCHAR(2) NOT NULL,    -- ISO 3166-1 alpha-2
    is_default      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE payment_methods (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    type            VARCHAR(20) NOT NULL CHECK (type IN ('credit_card', 'debit_card', 'paypal')),
    last_four       CHAR(4) NOT NULL,
    token           VARCHAR(255) NOT NULL,  -- 결제 프로세서가 토큰화
    expires_at      DATE,
    is_default      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 판매자
-- ============================================================

CREATE TABLE sellers (
    id              BIGSERIAL PRIMARY KEY,
    company_name    VARCHAR(255) NOT NULL,
    contact_email   VARCHAR(255) NOT NULL UNIQUE,
    commission_rate DECIMAL(5,4) NOT NULL DEFAULT 0.1500,  -- 15.00%
    bank_account    VARCHAR(255),  -- 암호화 또는 토큰화
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 제품 카탈로그
-- ============================================================

CREATE TABLE categories (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    slug            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT,
    parent_id       INT REFERENCES categories(id) ON DELETE SET NULL,
    position        INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE products (
    id              BIGSERIAL PRIMARY KEY,
    seller_id       BIGINT NOT NULL REFERENCES sellers(id),
    name            VARCHAR(255) NOT NULL,
    slug            VARCHAR(255) NOT NULL UNIQUE,
    description     TEXT,
    sku             VARCHAR(50) NOT NULL UNIQUE,
    price           DECIMAL(10,2) NOT NULL CHECK (price >= 0),
    weight_grams    INT,
    stock_quantity  INT NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE product_categories (
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    category_id     INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (product_id, category_id)
);

CREATE TABLE product_images (
    id              BIGSERIAL PRIMARY KEY,
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    url             VARCHAR(500) NOT NULL,
    alt_text        VARCHAR(255),
    position        INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 장바구니
-- ============================================================

CREATE TABLE carts (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT REFERENCES customers(id) ON DELETE SET NULL,
    session_id      VARCHAR(255),  -- 익명 카트용
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT cart_owner CHECK (customer_id IS NOT NULL OR session_id IS NOT NULL)
);

CREATE TABLE cart_items (
    id              BIGSERIAL PRIMARY KEY,
    cart_id         BIGINT NOT NULL REFERENCES carts(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id),
    quantity        INT NOT NULL CHECK (quantity > 0),
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cart_id, product_id)
);

-- ============================================================
-- 프로모션
-- ============================================================

CREATE TABLE promotions (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(50) NOT NULL UNIQUE,
    discount_type   VARCHAR(20) NOT NULL CHECK (discount_type IN ('percentage', 'fixed_amount')),
    amount          DECIMAL(10,2) NOT NULL CHECK (amount > 0),
    min_purchase    DECIMAL(10,2) DEFAULT 0,
    max_uses        INT,                     -- NULL = 무제한
    uses_count      INT NOT NULL DEFAULT 0,
    per_customer    INT NOT NULL DEFAULT 1,  -- 고객당 최대 사용 횟수
    valid_from      TIMESTAMPTZ NOT NULL,
    valid_until     TIMESTAMPTZ NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (valid_from < valid_until)
);

-- ============================================================
-- 주문
-- ============================================================

CREATE TYPE order_status AS ENUM (
    'pending', 'confirmed', 'processing', 'shipped',
    'delivered', 'cancelled', 'refunded'
);

CREATE TABLE orders (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id),
    shipping_address_id BIGINT NOT NULL REFERENCES addresses(id),
    payment_method_id   BIGINT NOT NULL REFERENCES payment_methods(id),
    promotion_id    BIGINT REFERENCES promotions(id),
    status          order_status NOT NULL DEFAULT 'pending',
    subtotal        DECIMAL(10,2) NOT NULL,
    discount_amount DECIMAL(10,2) NOT NULL DEFAULT 0,
    tax_amount      DECIMAL(10,2) NOT NULL DEFAULT 0,
    shipping_cost   DECIMAL(10,2) NOT NULL DEFAULT 0,
    total           DECIMAL(10,2) NOT NULL,
    tracking_number VARCHAR(100),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    id              BIGSERIAL PRIMARY KEY,
    order_id        BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id),
    quantity        INT NOT NULL CHECK (quantity > 0),
    unit_price      DECIMAL(10,2) NOT NULL,  -- 주문 시점의 가격
    subtotal        DECIMAL(10,2) NOT NULL,  -- quantity * unit_price
    UNIQUE (order_id, product_id)
);

-- ============================================================
-- 리뷰
-- ============================================================

CREATE TABLE reviews (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id),
    product_id      BIGINT NOT NULL REFERENCES products(id),
    rating          SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title           VARCHAR(255),
    body            TEXT,
    is_verified     BOOLEAN NOT NULL DEFAULT FALSE,  -- 검증된 구매
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (customer_id, product_id)  -- 고객당 제품당 하나의 리뷰
);

-- ============================================================
-- 위시리스트
-- ============================================================

CREATE TABLE wishlist_items (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (customer_id, product_id)
);
```

### 4.3 정규화 검증

주요 테이블의 함수 종속성을 분석하여 스키마가 BCNF에 있는지 검증합니다.

**customers 테이블**:
```
FDs: id → email, password_hash, first_name, last_name, phone, created_at, updated_at
     email → id  (후보 키)

후보 키: {id}, {email}
두 결정자 모두 슈퍼키 → BCNF ✓
```

**products 테이블**:
```
FDs: id → seller_id, name, slug, description, sku, price, weight_grams,
          stock_quantity, is_active, created_at, updated_at
     sku → id (후보 키)
     slug → id (후보 키)

후보 키: {id}, {sku}, {slug}
모든 결정자가 슈퍼키 → BCNF ✓
```

**order_items 테이블**:
```
FDs: id → order_id, product_id, quantity, unit_price, subtotal
     (order_id, product_id) → id, quantity, unit_price, subtotal

후보 키: {id}, {order_id, product_id}
모든 결정자가 슈퍼키 → BCNF ✓
```

**reviews 테이블**:
```
FDs: id → customer_id, product_id, rating, title, body, is_verified, created_at
     (customer_id, product_id) → id, rating, title, body, is_verified, created_at

후보 키: {id}, {customer_id, product_id}
모든 결정자가 슈퍼키 → BCNF ✓
```

**categories 테이블** (자기 참조 계층):
```
FDs: id → name, slug, description, parent_id, position, created_at
     slug → id

후보 키: {id}, {slug}
모든 결정자가 슈퍼키 → BCNF ✓
```

**스키마는 BCNF에 있습니다** -- 슈퍼키가 아닌 결정자를 가진 비자명 FD가 없습니다.

### 4.4 참조 무결성 요약

```
┌─────────────────┐     ┌─────────────────┐
│    customers     │     │     sellers     │
│    (PK: id)      │     │    (PK: id)     │
└───────┬─────────┘     └───────┬─────────┘
   │    │    │    │              │
   │    │    │    │              │
   │    │    │    │    ┌─────────┴──────────┐
   │    │    │    └───▶│     products       │
   │    │    │         │  (FK: seller_id)   │
   │    │    │         └──┬────────┬────────┘
   │    │    │            │        │
   │    │    │     ┌──────┘        └──────┐
   │    │    │     ▼                      ▼
   │    │    │  ┌────────────┐    ┌──────────────┐
   │    │    │  │product_cats│    │product_images│
   │    │    │  └────────────┘    └──────────────┘
   │    │    │
   │    │    └────▶┌──────────┐
   │    │          │  reviews  │ (FK: customer_id, product_id)
   │    │          └──────────┘
   │    │
   │    └─────────▶┌──────────┐     ┌─────────────┐
   │               │  orders   │────▶│ order_items  │
   │               └──────────┘     └─────────────┘
   │
   ├──────────────▶┌──────────┐
   │               │ addresses│
   ├──────────────▶┌──────────┐
   │               │pay_methods│
   ├──────────────▶┌──────────┐
   │               │   carts   │────▶ cart_items
   └──────────────▶┌──────────┐
                   │ wishlist  │
                   └──────────┘
```

---

## 5. 단계 4: 물리적 설계

### 5.1 인덱싱 전략

단계 1에서 식별한 쿼리 패턴을 기반으로 인덱스를 설계합니다.

**기본 키 인덱스** (자동 생성):

```sql
-- 모든 테이블은 BIGSERIAL PRIMARY KEY에서 PK 인덱스를 이미 가짐
-- 유니크 제약조건도 자동으로 인덱스 생성:
--   customers(email), products(sku), products(slug), promotions(code)
```

**빈번한 쿼리를 위한 보조 인덱스**:

```sql
-- QP-01: 카테고리 + 가격 범위로 제품 검색
CREATE INDEX idx_products_active_price ON products(price)
    WHERE is_active = TRUE;

-- 카테고리 기반 검색용 (접합 테이블을 통해)
CREATE INDEX idx_product_categories_category ON product_categories(category_id);

-- QP-02: 제품 상세 페이지 (제품의 리뷰)
CREATE INDEX idx_reviews_product_rating ON reviews(product_id, rating);

-- QP-03: 고객의 장바구니
CREATE INDEX idx_carts_customer ON carts(customer_id);

-- QP-04: 재고 확인 (제품의 stock_quantity)
-- products(id)의 PK로 충분

-- QP-05: 고객의 주문 이력
CREATE INDEX idx_orders_customer_date ON orders(customer_id, created_at DESC);

-- QP-06: 판매자 대시보드
CREATE INDEX idx_products_seller ON products(seller_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- QP-07: 카테고리에서 최고 평가 제품
-- idx_reviews_product_rating과 idx_product_categories_category가 도움

-- QP-09: 프로모션 검증
CREATE INDEX idx_promotions_code_active ON promotions(code)
    WHERE is_active = TRUE;

-- QP-10: 위시리스트 표시
CREATE INDEX idx_wishlist_customer ON wishlist_items(customer_id, added_at DESC);

-- 제품 이름 및 설명에 대한 전체 텍스트 검색
CREATE INDEX idx_products_search ON products
    USING GIN (to_tsvector('english', name || ' ' || COALESCE(description, '')));
```

### 5.2 파티셔닝 전략

수억 행으로 성장할 것으로 예상되는 테이블에 테이블 파티셔닝을 적용합니다.

```sql
-- 월별로 주문 파티션 (범위 파티셔닝)
-- 다음에 도움:
--   1. 주문 이력 쿼리 (관련 월만 스캔)
--   2. 오래된 주문 아카이빙 (전체 파티션 삭제)
--   3. 유지보수 작업 (더 작은 파티션에서 VACUUM)

CREATE TABLE orders (
    id              BIGSERIAL,
    customer_id     BIGINT NOT NULL,
    -- ... 기타 열 ...
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)  -- 파티션 키는 PK에 있어야 함
) PARTITION BY RANGE (created_at);

-- 파티션 생성
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE orders_2024_02 PARTITION OF orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... (pg_partman 확장으로 자동화)

-- 주문 날짜별로 order_items를 유사하게 파티션
-- 또는 이미 파티션된 주문에 대한 외래 키 사용

-- 리뷰 파티션 (매우 크게 성장하는 경우)
CREATE TABLE reviews (
    -- ...
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);
```

### 5.3 비정규화 결정

순수 정규화는 저장소와 업데이트 일관성을 최적화하지만 읽기 성능을 해칠 수 있습니다. 읽기 성능 이점이 업데이트 복잡성을 능가하는 곳에서 전략적으로 비정규화합니다.

**결정 1: order_items에 unit_price 저장**

```sql
-- 스키마에 이미 있음: order_items.unit_price는 주문 시점의 가격 캡처
-- 이것은 성능을 위한 비정규화가 아님 -- 비즈니스 요구사항!
-- 제품 가격은 주문이 생성된 후 변경될 수 있음.
-- 이것 없이는 과거 주문 합계를 재구성할 수 없음.
```

**결정 2: 구체화된 리뷰 통계**

```sql
-- 모든 제품 페이지 로드에 대해 AVG(rating)과 COUNT(*)를 계산하는 대신,
-- products 테이블에 미리 계산된 값을 저장.

ALTER TABLE products ADD COLUMN avg_rating DECIMAL(3,2) DEFAULT NULL;
ALTER TABLE products ADD COLUMN review_count INT NOT NULL DEFAULT 0;

-- 리뷰가 추가/수정될 때 트리거 또는 애플리케이션 코드를 통해 업데이트:
-- UPDATE products SET
--   avg_rating = (SELECT AVG(rating) FROM reviews WHERE product_id = $1),
--   review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = $1)
-- WHERE id = $1;
```

**결정 3: 주문에 total 저장**

```sql
-- 스키마에 이미 있음: orders.total은 계산되지 않고 저장됨
-- 모든 읽기에서 order_items의 SUM(subtotal)을 계산하는 것은 낭비
-- 주문이 생성되면 total이 고정됨
```

**결정 4: 카테고리 경로 구체화**

```sql
-- 깊은 카테고리 계층의 경우, 재귀 쿼리가 느림
-- 표시를 위한 전체 경로를 구체화:

ALTER TABLE categories ADD COLUMN path TEXT;  -- 예: "Electronics/Phones/Smartphones"
ALTER TABLE categories ADD COLUMN depth INT NOT NULL DEFAULT 0;

-- 카테고리 생성/이동 시 업데이트:
-- path = parent.path || '/' || name
-- depth = parent.depth + 1
```

**비정규화 결정 매트릭스**:

| 비정규화 | 이점 | 비용 | 결정 |
|-----------------|---------|------|----------|
| products의 avg_rating | 모든 제품 페이지에서 집계 회피 | 리뷰 변경 시 업데이트 | 예 (높은 읽기:쓰기 비율) |
| orders의 total | 모든 주문 표시에서 SUM 회피 | order_items와 일치해야 함 | 예 (생성 후 불변) |
| 카테고리 경로 | 재귀 쿼리 회피 | 카테고리 이동 시 업데이트 | 예 (카테고리가 거의 변경되지 않음) |
| orders의 고객 이름 | 주문 목록에 대한 JOIN 회피 | 이름 변경 시 업데이트 | 아니오 (인덱스를 사용한 JOIN이 빠름) |

### 5.4 저장소 고려사항

```sql
-- 저장 효율성을 위한 적절한 데이터 타입 사용

-- 좋음: 돈에 DECIMAL(10,2) (정확한 산술)
-- 나쁨:  돈에 FLOAT (반올림 오류!)

-- 좋음: 평가에 SMALLINT (1-5, 2바이트 사용)
-- 나쁨:  평가에 INT (4바이트 사용, 공간 낭비)

-- 좋음: 국가 코드에 VARCHAR(2)
-- 나쁨:  국가 코드에 TEXT (길이 제약 없음)

-- 좋음: 날짜에 TIMESTAMPTZ (타임존 인식)
-- 나쁨:  날짜에 TIMESTAMP (모호한 타임존)

-- 좋음: 고정된 상태 값에 ENUM
-- 나쁨:  상태에 VARCHAR (오타 허용)

-- 테이블 크기 추정:
-- customers: 1M 행 x ~300 바이트 ≈ 300 MB
-- products:  500K 행 x ~500 바이트 ≈ 250 MB
-- orders:    10M 행 x ~200 바이트 ≈ 2 GB
-- order_items: 25M 행 x ~100 바이트 ≈ 2.5 GB
-- reviews:   2M 행 x ~500 바이트 ≈ 1 GB
-- 총: ~6 GB (캐싱을 위해 RAM에 쉽게 맞음)
```

---

## 6. 단계 5: 쿼리 최적화

### 6.1 제품 검색 (QP-01)

```sql
-- 쿼리: 카테고리에서 키워드로 제품 검색, 가격 필터
-- 플랫폼에서 가장 빈번한 쿼리

SELECT p.id, p.name, p.price, p.avg_rating, p.review_count,
       (SELECT url FROM product_images pi
        WHERE pi.product_id = p.id ORDER BY pi.position LIMIT 1) AS thumbnail
FROM products p
JOIN product_categories pc ON p.id = pc.product_id
WHERE pc.category_id = 42
  AND p.price BETWEEN 10.00 AND 100.00
  AND p.is_active = TRUE
  AND to_tsvector('english', p.name || ' ' || COALESCE(p.description, ''))
      @@ plainto_tsquery('english', 'wireless bluetooth headphones')
ORDER BY p.avg_rating DESC NULLS LAST
LIMIT 20 OFFSET 0;
```

**실행 계획 분석**:

```
EXPLAIN (ANALYZE, BUFFERS)
-- 예상 계획:
-- Limit  (cost=... rows=20)
--   -> Sort  (cost=... key: avg_rating DESC)
--     -> Nested Loop  (cost=...)
--       -> Index Scan on product_categories pc (category_id = 42)
--       -> Index Scan on products p (id = pc.product_id)
--            Filter: is_active AND price BETWEEN 10 AND 100
--            Filter: tsvector @@ tsquery
```

**최적화**: 전체 텍스트 검색과 카테고리를 모두 다루는 복합 GIN 인덱스 생성:

```sql
-- 카테고리 기반 전체 텍스트 검색이 지배적인 패턴인 경우:
-- 제품과 카테고리를 미리 조인하는 구체화된 뷰 고려
CREATE MATERIALIZED VIEW product_search AS
SELECT p.id, p.name, p.description, p.price, p.avg_rating,
       p.review_count, p.is_active, pc.category_id,
       to_tsvector('english', p.name || ' ' || COALESCE(p.description, '')) AS search_vector
FROM products p
JOIN product_categories pc ON p.id = pc.product_id;

CREATE INDEX idx_product_search_category ON product_search(category_id)
    WHERE is_active = TRUE;
CREATE INDEX idx_product_search_fts ON product_search USING GIN(search_vector);
CREATE INDEX idx_product_search_price ON product_search(category_id, price)
    WHERE is_active = TRUE;

-- 주기적으로 또는 제품 변경 시 새로고침
REFRESH MATERIALIZED VIEW CONCURRENTLY product_search;
```

### 6.2 제품 상세 페이지 (QP-02)

```sql
-- 한 쿼리로 이미지, 판매자 및 리뷰 요약과 함께 제품 가져오기
SELECT
  p.id, p.name, p.description, p.price, p.sku,
  p.stock_quantity, p.avg_rating, p.review_count,
  s.company_name AS seller_name,
  json_agg(DISTINCT jsonb_build_object(
    'url', pi.url, 'alt_text', pi.alt_text, 'position', pi.position
  )) AS images
FROM products p
JOIN sellers s ON p.seller_id = s.id
LEFT JOIN product_images pi ON p.id = pi.product_id
WHERE p.id = $1
GROUP BY p.id, s.id;

-- 리뷰를 별도로 가져오기 (페이지네이션)
SELECT r.rating, r.title, r.body, r.created_at, r.is_verified,
       c.first_name || ' ' || LEFT(c.last_name, 1) || '.' AS reviewer
FROM reviews r
JOIN customers c ON r.customer_id = c.id
WHERE r.product_id = $1
ORDER BY r.created_at DESC
LIMIT 10 OFFSET $2;
```

### 6.3 주문 이력 (QP-05)

```sql
-- 항목 요약과 함께 고객의 최근 주문
SELECT o.id, o.status, o.total, o.created_at,
       COUNT(oi.id) AS item_count,
       STRING_AGG(LEFT(p.name, 30), ', ' ORDER BY oi.id LIMIT 3) AS item_preview
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE o.customer_id = $1
GROUP BY o.id
ORDER BY o.created_at DESC
LIMIT 20 OFFSET $2;

-- 인덱스 사용: idx_orders_customer_date (customer_id, created_at DESC)
-- 파티션 가지치기: 최근 주문 쿼리 시, 최근 파티션만 스캔
```

### 6.4 판매자 대시보드 (QP-06)

```sql
-- 판매자의 월별 수익 및 주문 수
SELECT
  DATE_TRUNC('month', o.created_at) AS month,
  COUNT(DISTINCT o.id) AS order_count,
  SUM(oi.subtotal) AS revenue,
  SUM(oi.subtotal * s.commission_rate) AS commission
FROM sellers s
JOIN products p ON s.id = p.seller_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE s.id = $1
  AND o.created_at >= $2  -- 시작 날짜
  AND o.created_at < $3   -- 종료 날짜
  AND o.status NOT IN ('cancelled', 'refunded')
GROUP BY month
ORDER BY month DESC;
```

### 6.5 쿼리 성능 요약

| 쿼리 | 예상 지연시간 | 주요 인덱스 | 비고 |
|-------|-----------------|-----------|-------|
| 제품 검색 | < 50ms | GIN (FTS) + 카테고리 | 구체화된 뷰가 도움 |
| 제품 상세 | < 10ms | PK + 판매자 FK | 단일 행 + 이미지 |
| 장바구니 내용 | < 5ms | carts(customer_id) | 작은 결과 집합 |
| 주문 생성 | < 100ms | 재고 확인 + 삽입 | 트랜잭션 (단계 6 참조) |
| 주문 이력 | < 20ms | orders(customer_id, date) | 파티션 가지치기 |
| 판매자 대시보드 | < 200ms | products(seller_id) + order_items(product_id) | 집계 쿼리 |
| 최고 평가 제품 | < 30ms | product_categories + avg_rating | 비정규화된 평가 사용 |

---

## 7. 단계 6: 트랜잭션 설계

### 7.1 중요 트랜잭션: 주문 생성

이것은 시스템에서 가장 복잡하고 중요한 트랜잭션입니다. 다음을 수행해야 합니다:
1. 장바구니 항목이 재고에 있는지 검증
2. 합계 계산 (잠재적 프로모션 포함)
3. 재고 감소
4. 주문 및 주문 항목 생성
5. 장바구니 비우기
6. 결제 처리 (외부 API 호출)

```sql
-- 격리 수준: SERIALIZABLE 또는 명시적 잠금이 있는 READ COMMITTED
-- 더 나은 성능을 위해 명시적 잠금이 있는 READ COMMITTED 사용

BEGIN;

-- 단계 1: 동시 수정 방지를 위해 장바구니 항목 잠금
-- 과다 판매를 방지하기 위해 제품 행도 잠금
SELECT p.id, p.name, p.price, p.stock_quantity, ci.quantity
FROM cart_items ci
JOIN products p ON ci.product_id = p.id
WHERE ci.cart_id = $cart_id
FOR UPDATE OF p;  -- 제품 행 잠금

-- 단계 2: 재고 검증 (애플리케이션 확인: stock_quantity >= ci.quantity)
-- 항목이 재고 부족이면 ROLLBACK하고 고객에게 알림

-- 단계 3: 프로모션 적용 (있는 경우)
SELECT id, discount_type, amount, min_purchase, max_uses, uses_count
FROM promotions
WHERE code = $promo_code
  AND is_active = TRUE
  AND valid_from <= NOW()
  AND valid_until > NOW()
FOR UPDATE;  -- max_uses를 초과한 동시 사용 방지를 위해 잠금

-- 단계 4: 주문 생성
INSERT INTO orders (customer_id, shipping_address_id, payment_method_id,
                    promotion_id, subtotal, discount_amount, tax_amount,
                    shipping_cost, total, status)
VALUES ($customer_id, $address_id, $payment_id,
        $promo_id, $subtotal, $discount, $tax, $shipping, $total, 'pending')
RETURNING id AS order_id;

-- 단계 5: 주문 항목 생성 및 재고 감소
INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
SELECT $order_id, ci.product_id, ci.quantity, p.price, ci.quantity * p.price
FROM cart_items ci
JOIN products p ON ci.product_id = p.id
WHERE ci.cart_id = $cart_id;

UPDATE products p
SET stock_quantity = stock_quantity - ci.quantity,
    updated_at = NOW()
FROM cart_items ci
WHERE p.id = ci.product_id
  AND ci.cart_id = $cart_id;

-- 단계 6: 프로모션 사용 횟수 업데이트
UPDATE promotions SET uses_count = uses_count + 1
WHERE id = $promo_id;

-- 단계 7: 장바구니 비우기
DELETE FROM cart_items WHERE cart_id = $cart_id;

-- 단계 8: 결제 처리 (외부 API 호출)
-- 참고: 트랜잭션 외부에서 수행하거나 Saga 패턴 사용
-- 결제가 실패하면 전체 트랜잭션이 롤백됨

COMMIT;
```

**중요한 설계 결정**: 결제 API 호출은 데이터베이스 트랜잭션 내부에 있어서는 안 됩니다:
1. 외부 API 호출이 느릴 수 있음 (초), 잠금을 너무 오래 유지
2. DB가 커밋되지만 결제가 실패하면 일관되지 않은 상태

**더 나은 접근: Saga 패턴**

```
1. BEGIN; status='pending_payment'로 주문 생성; 재고 감소; COMMIT;
2. 결제 API 호출
3. 결제 성공 시: UPDATE orders SET status='confirmed' WHERE id=$id;
4. 결제 실패 시: UPDATE orders SET status='payment_failed' WHERE id=$id;
                 재고 복원 (보상 트랜잭션)
```

### 7.2 격리 수준 결정

| 트랜잭션 | 격리 수준 | 이유 |
|------------|-----------------|--------|
| 주문 생성 | READ COMMITTED + FOR UPDATE | 명시적 잠금이 과다 판매를 방지; RC가 Serializable보다 빠름 |
| 제품 탐색 | READ COMMITTED | 카탈로그에 대한 약간의 staleness 허용 가능 |
| 주문 이력 보기 | READ COMMITTED | 과거 데이터, 쓰기 충돌 없음 |
| 리뷰 제출 | READ COMMITTED + UNIQUE 제약조건 | UNIQUE(customer_id, product_id)가 중복 방지 |
| 프로모션 적용 | READ COMMITTED + FOR UPDATE | max_uses 초과 방지를 위해 프로모션 행 잠금 |
| 관리자 보고서 | REPEATABLE READ | 장기 실행 집계 쿼리를 위한 일관된 스냅샷 |

### 7.3 잠금 전략

```sql
-- 과다 판매 방지: 제품 행에 SELECT FOR UPDATE 사용
-- 행 수준 배타적 잠금 획득

-- 교착 상태 방지를 위한 순서대 잠금:
-- 여러 제품을 업데이트할 때 항상 id 순서로 잠금

-- 예: 장바구니에 제품 [42, 17, 88]이 있으면 [17, 42, 88] 순서로 잠금
SELECT id, stock_quantity
FROM products
WHERE id IN (17, 42, 88)
ORDER BY id    -- 일관된 순서가 교착 상태 방지
FOR UPDATE;
```

### 7.4 동시성 시나리오

**시나리오 1: 두 고객이 마지막 항목을 주문**

```
고객 A                              고객 B
BEGIN                               BEGIN
SELECT ... FOR UPDATE (stock=1) ✓   SELECT ... FOR UPDATE → 차단됨
UPDATE stock = 0                    (A의 잠금을 기다림)
INSERT order_item                   ...
COMMIT → 잠금 해제                  → 차단 해제, stock=0 읽음
                                    → stock < quantity → ROLLBACK
                                    → "재고 부족" 메시지
```

**시나리오 2: 고객이 체크아웃하는 동안 장바구니 업데이트**

```
체크아웃 트랜잭션              장바구니 업데이트
BEGIN                          BEGIN
SELECT cart_items FOR UPDATE ✓ UPDATE cart_items → 차단됨
...                            (체크아웃의 잠금을 기다림)
...주문 처리 중...             ...
DELETE cart_items              ...
COMMIT → 잠금 해제             → 행 삭제됨, 업데이트가 0행에 영향
                               → COMMIT (오류 없음, 장바구니는 비어 있음)
```

---

## 8. 대안 사례 연구: 소셜 미디어 플랫폼

### 8.1 요구사항 요약

**SocialBuzz**는 게시물, 팔로우, 좋아요 및 개인화된 피드를 지원하는 소셜 미디어 플랫폼입니다.

### 8.2 주요 엔티티

```sql
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    username    VARCHAR(50) NOT NULL UNIQUE,
    email       VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    bio         TEXT,
    avatar_url  VARCHAR(500),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE follows (
    follower_id BIGINT NOT NULL REFERENCES users(id),
    followee_id BIGINT NOT NULL REFERENCES users(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id),
    CHECK (follower_id != followee_id)
);

CREATE TABLE posts (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id),
    content     TEXT NOT NULL,
    media_urls  TEXT[],  -- 미디어 URL 배열
    like_count  INT NOT NULL DEFAULT 0,      -- 비정규화
    comment_count INT NOT NULL DEFAULT 0,    -- 비정규화
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE likes (
    user_id     BIGINT NOT NULL REFERENCES users(id),
    post_id     BIGINT NOT NULL REFERENCES posts(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, post_id)
);

CREATE TABLE comments (
    id          BIGSERIAL PRIMARY KEY,
    post_id     BIGINT NOT NULL REFERENCES posts(id),
    user_id     BIGINT NOT NULL REFERENCES users(id),
    parent_id   BIGINT REFERENCES comments(id),  -- 스레드 댓글
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 8.3 피드 문제

가장 어려운 쿼리는 개인화된 피드입니다: "내가 팔로우하는 사람들의 최근 게시물을 보여주세요."

**풀 모델** (읽기 시 팬아웃):

```sql
-- 읽기 시점에 팔로우하는 모든 사용자의 게시물 쿼리
SELECT p.*, u.username, u.display_name, u.avatar_url
FROM posts p
JOIN follows f ON p.user_id = f.followee_id
JOIN users u ON p.user_id = u.id
WHERE f.follower_id = $current_user_id
ORDER BY p.created_at DESC
LIMIT 20;

-- 문제: 사용자가 1000명을 팔로우하면 큰 follows 목록을
-- 큰 posts 테이블과 조인. 활성 사용자에게 느림.
```

**푸시 모델** (쓰기 시 팬아웃):

```sql
-- 피드 테이블 생성 (비정규화된 타임라인)
CREATE TABLE feed_items (
    user_id     BIGINT NOT NULL,  -- 피드를 소유한 사용자
    post_id     BIGINT NOT NULL,
    author_id   BIGINT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, created_at DESC, post_id)
);

-- 사용자 X가 게시물을 생성할 때:
-- X의 모든 팔로워에 대해 feed_item 삽입
INSERT INTO feed_items (user_id, post_id, author_id, created_at)
SELECT f.follower_id, $post_id, $author_id, $created_at
FROM follows f
WHERE f.followee_id = $author_id;

-- 피드 읽기는 이제 간단:
SELECT fi.post_id, fi.author_id, fi.created_at,
       p.content, p.media_urls, p.like_count,
       u.username, u.avatar_url
FROM feed_items fi
JOIN posts p ON fi.post_id = p.id
JOIN users u ON fi.author_id = u.id
WHERE fi.user_id = $current_user_id
ORDER BY fi.created_at DESC
LIMIT 20;

-- 문제: 유명인이 1천만 팔로워를 가지고 있으면 게시물 생성에
-- 1천만 INSERT 작업이 필요. 인기 사용자에게 느림.
```

**하이브리드 모델** (Twitter/X가 사용):

```
일반 사용자 (< 10K 팔로워): 푸시 모델 (쓰기 시 팬아웃)
유명인 사용자 (> 10K 팔로워): 풀 모델 (읽기 시 팬아웃)

피드 = Merge(feed_items 테이블, 팔로우한 유명인을 위한 실시간 쿼리)
```

### 8.4 설계 트레이드오프

| 측면 | 전자상거래 | 소셜 미디어 |
|--------|-----------|-------------|
| **중요 트랜잭션** | 주문 생성 (ACID 필요) | 게시물 생성 (최종 일관성 OK) |
| **주요 읽기 패턴** | ID로 제품 조회 | 타임라인 (피드) |
| **쓰기 패턴** | 낮은 빈도, 높은 가치 | 높은 빈도, 낮은 가치 |
| **비정규화** | 보통 (avg_rating, totals) | 많음 (feed_items, 카운터) |
| **일관성** | 강함 (재고, 결제) | 최종 (좋아요 수, 피드) |
| **확장 도전** | 재고 경합 | 팬아웃 (인기 사용자) |

---

## 9. 설계 검토 체크리스트

프로덕션에 데이터베이스 설계를 배포하기 전에 이 체크리스트를 사용하세요:

### 9.1 스키마 설계

```
[ ] 모든 테이블에 명확한 기본 키
[ ] 모든 관계에 대한 외래 키 정의
[ ] ON DELETE / ON UPDATE 작업 지정 (CASCADE, SET NULL, RESTRICT)
[ ] 필수 필드에 NOT NULL 제약조건
[ ] 도메인 검증을 위한 CHECK 제약조건 (예: price >= 0, rating BETWEEN 1 AND 5)
[ ] 자연 키에 대한 UNIQUE 제약조건 (email, username, SKU)
[ ] 적절한 데이터 타입 (돈에 DECIMAL, 날짜에 TIMESTAMPTZ)
[ ] 비정규화가 문서화되고 정당화되지 않는 한 중복 열 없음
[ ] 스키마가 최소 3NF (가급적 BCNF)
[ ] 계층을 위한 자기 참조 FK (카테고리, 댓글)
```

### 9.2 인덱싱

```
[ ] 모든 외래 키에 인덱스 존재 (PostgreSQL은 FK 인덱스를 자동 생성하지 않음!)
[ ] 빈번한 쿼리의 WHERE, JOIN 및 ORDER BY 열을 인덱스가 다룸
[ ] 복합 인덱스가 왼쪽 접두사 규칙을 따름
[ ] 적절한 경우 부분 인덱스 사용 (WHERE is_active = TRUE)
[ ] 전체 텍스트 검색 및 배열 열을 위한 GIN 인덱스
[ ] 중복 인덱스 없음 (인덱스 (a, b)는 인덱스 (a)를 중복으로 만듦)
[ ] 인덱스 비대화 모니터링 계획
```

### 9.3 데이터 무결성

```
[ ] 모든 돈 값이 DECIMAL 사용, FLOAT 안 함
[ ] 타임스탬프가 TIMESTAMPTZ 사용 (타임존 인식)
[ ] 상태 필드에 대한 Enum 타입 또는 CHECK 제약조건
[ ] 불변 필드가 보호됨 (주문 합계, 과거 가격)
[ ] 각 테이블에 대한 소프트 삭제 vs 하드 삭제 결정 문서화
[ ] 데이터 보존 정책 정의
```

### 9.4 성능

```
[ ] 모든 중요 쿼리에 EXPLAIN ANALYZE 실행
[ ] 인덱스된 쿼리에 대한 대형 테이블에서 순차 스캔 없음
[ ] 페이지네이션이 OFFSET이 아닌 키셋 페이지네이션 사용
[ ] N+1 쿼리 패턴 제거 (JOIN 또는 배치 페칭 사용)
[ ] 연결 풀링 구성 (PgBouncer)
[ ] 적절한 work_mem 및 shared_buffers 설정
```

### 9.5 트랜잭션

```
[ ] 각 트랜잭션 유형에 대한 격리 수준 선택
[ ] 잠금 전략 문서화 (어떤 행/테이블이 잠김)
[ ] 교착 상태 방지를 위한 잠금 순서 정의
[ ] 장기 실행 트랜잭션 회피 (트랜잭션 내부에 API 호출 없음)
[ ] 직렬화 실패를 위한 재시도 로직
[ ] 교착 상태 탐지 타임아웃 구성
```

### 9.6 보안

```
[ ] 평문 비밀번호 없음 (bcrypt/argon2 해시 사용)
[ ] 카드 번호가 아닌 결제 토큰 저장 (PCI-DSS)
[ ] 다중 테넌트 데이터를 위한 행 수준 보안 (RLS)
[ ] 애플리케이션 사용자가 최소 권한 (SUPERUSER 없음)
[ ] SQL 인젝션 방지 (매개변수화된 쿼리만)
[ ] 민감한 작업에 대한 감사 로깅
```

### 9.7 운영

```
[ ] 백업 전략 정의 (pg_dump 일정, WAL 아카이빙)
[ ] 시점 복구 (PITR) 테스트됨
[ ] 모니터링: 쿼리 지연시간, 연결 수, 테이블 비대화, 복제 지연
[ ] 스키마 마이그레이션 도구 사용 (Flyway, Alembic, golang-migrate)
[ ] 일반적인 장애 시나리오를 위한 런북 (디스크 가득 참, 복제 지연, 교착 상태)
```

---

## 10. 일반적인 설계 실수 및 회피 방법

### 실수 1: 돈에 FLOAT 사용

```sql
-- 나쁨: 부동소수점 산술이 반올림 오류 발생
CREATE TABLE orders (
    total FLOAT  -- 0.1 + 0.2 = 0.30000000000000004
);

-- 좋음: 정확한 십진수 타입 사용
CREATE TABLE orders (
    total DECIMAL(10,2)  -- 0.1 + 0.2 = 0.30
);
```

### 실수 2: 외래 키 인덱스 누락

```sql
-- PostgreSQL은 PRIMARY KEY와 UNIQUE에 대한 인덱스를 생성하지만 FOREIGN KEY에는 안 함!
-- 이는 FK 열에 대한 JOIN 쿼리가 순차 스캔을 수행한다는 것을 의미.

CREATE TABLE order_items (
    order_id BIGINT REFERENCES orders(id),  -- 자동 인덱스 없음!
    product_id BIGINT REFERENCES products(id)  -- 자동 인덱스 없음!
);

-- 수정: 항상 FK 열에 인덱스 생성
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
```

### 실수 3: 유지보수 전략 없이 파생 데이터 저장

```sql
-- 비정규화는 괜찮지만 일관성을 유지해야 함
ALTER TABLE products ADD COLUMN avg_rating DECIMAL(3,2);

-- 나쁨: 수동 업데이트하고 개발자가 기억하기를 바람
-- 좋음: 트리거 사용
CREATE OR REPLACE FUNCTION update_product_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE products SET
        avg_rating = (SELECT AVG(rating) FROM reviews WHERE product_id = NEW.product_id),
        review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = NEW.product_id)
    WHERE id = NEW.product_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_product_rating
AFTER INSERT OR UPDATE OR DELETE ON reviews
FOR EACH ROW EXECUTE FUNCTION update_product_rating();
```

### 실수 4: OFFSET 기반 페이지네이션

```sql
-- 나쁨: OFFSET이 행을 스캔하고 버림. 페이지 1000이 20,000행 읽음!
SELECT * FROM products ORDER BY created_at DESC LIMIT 20 OFFSET 19980;
-- ↑ 20,000행 읽고, 19,980 버리고, 20 반환.

-- 좋음: 키셋 페이지네이션 (커서 기반)
SELECT * FROM products
WHERE created_at < $last_seen_created_at
ORDER BY created_at DESC
LIMIT 20;
-- ↑ 인덱스 사용, 페이지 번호에 관계없이 정확히 20행 읽음.
```

### 실수 5: 신 테이블 (EAV 안티패턴)

```sql
-- 나쁨: Entity-Attribute-Value 패턴
CREATE TABLE properties (
    entity_type VARCHAR(50),   -- 'product', 'user', etc.
    entity_id   BIGINT,
    key         VARCHAR(100),  -- 'color', 'size', 'price'
    value       TEXT           -- 모든 것이 문자열!
);

-- 문제:
-- 1. 타입 안전성 없음 (가격이 텍스트로 저장됨)
-- 2. 제약조건 없음 ("price >= 0" 강제할 수 없음)
-- 3. 외래 키 없음
-- 4. 끔찍한 쿼리 성능 (피벗 쿼리가 느림)
-- 5. 스키마 문서화 없음

-- 좋음: 타입이 지정된 열이 있는 적절한 테이블 사용
-- 속성이 카테고리별로 다르면 고려:
--   1. 유연한 속성을 위한 PostgreSQL JSONB 열
--   2. 카테고리별 별도 테이블 (product_electronics, product_clothing)
--   3. 하이브리드: products의 공통 열 + 카테고리 특정을 위한 JSONB
```

### 실수 6: 과거 데이터 캡처하지 않음

```sql
-- 나쁨: 현재 상태만 저장
-- 제품 가격이 변경되면 모든 과거 주문이 새 가격을 표시!

-- 좋음: 이벤트 시점의 상태 캡처
CREATE TABLE order_items (
    product_id  BIGINT REFERENCES products(id),
    quantity    INT,
    unit_price  DECIMAL(10,2),  -- 주문 시점의 가격, 현재 가격 아님
    product_name VARCHAR(255)    -- 선택사항: 주문 시점의 제품 이름 스냅샷
);
```

### 실수 7: 조기 비정규화

```
단계: 사용자가 없기 전
개발자: "성능을 위해 모든 것을 비정규화합시다!"

현실 확인:
- 적절하게 인덱스된 PostgreSQL은 최대 1억 행까지 대부분의 워크로드를 처리 가능
- 정규화된 상태로 시작, 측정, 측정이 문제를 보이는 곳에서 비정규화
- 조기 비정규화는 측정 가능한 이점 없이 유지보수 악몽을 만듦

경험칙: 1천만 행 미만이고 쿼리가 인덱스를 사용하면
비정규화가 필요 없을 가능성이 높음.
```

### 실수 8: NULL 의미론 무시

```sql
-- 놀람: NULL 비교가 예상대로 작동하지 않음
SELECT * FROM products WHERE category_id != 5;
-- category_id가 NULL인 행을 반환하지 않음!

-- 수정: NULL을 명시적으로 처리
SELECT * FROM products WHERE category_id != 5 OR category_id IS NULL;
-- 또는: WHERE category_id IS DISTINCT FROM 5;

-- 또한: COUNT(column)은 NULL을 무시, COUNT(*)는 모든 행을 세음
SELECT COUNT(phone) FROM customers;  -- NULL이 아닌 전화번호만 세음
SELECT COUNT(*) FROM customers;      -- 모든 고객 세음
```

### 실수 9: 쉼표로 구분된 값 저장

```sql
-- 나쁨: 단일 열에 여러 값 저장
CREATE TABLE products (
    tags VARCHAR(500)  -- "electronics,wireless,bluetooth"
);

-- 문제:
-- 1. 개별 태그를 효율적으로 인덱스할 수 없음
-- 2. 참조 무결성 강제할 수 없음
-- 3. "태그 'wireless'가 있는 제품 찾기"는 LIKE '%wireless%' 필요 (느리고, 부정확)
-- 4. tags 테이블과 JOIN할 수 없음
-- 5. 1NF 위반

-- 좋음: 접합 테이블 사용
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE product_tags (
    product_id BIGINT REFERENCES products(id),
    tag_id INT REFERENCES tags(id),
    PRIMARY KEY (product_id, tag_id)
);

-- 또는 GIN 인덱스와 함께 PostgreSQL 배열 사용
CREATE TABLE products (
    tags TEXT[]
);
CREATE INDEX idx_products_tags ON products USING GIN(tags);
SELECT * FROM products WHERE tags @> ARRAY['wireless'];
```

### 실수 10: 스키마 진화 계획하지 않음

```sql
-- 나쁨: 마이그레이션 도구 없음, 프로덕션에서 수동 ALTER TABLE

-- 좋음: 마이그레이션 도구 사용 (Flyway, Alembic, golang-migrate)
-- 각 마이그레이션은 버전이 지정된 가역 SQL 파일:

-- V001__create_customers.sql
CREATE TABLE customers (...);

-- V002__add_phone_to_customers.sql
ALTER TABLE customers ADD COLUMN phone VARCHAR(20);

-- V003__create_orders.sql
CREATE TABLE orders (...);

-- 이점:
-- 1. 모든 스키마 변경이 버전 제어됨
-- 2. 환경 간 재현 가능 (개발, 스테이징, 프로덕션)
-- 3. 롤백 기능
-- 4. 팀 협업 (충돌하는 수동 변경 없음)
```

---

## 11. 연습 문제

### 연습 문제 1: 전자상거래 스키마 완성

위의 ShopWave 스키마는 여러 기능이 누락되어 있습니다. 다음을 지원하도록 확장하세요:

1. **제품 변형**: 제품은 변형을 가질 수 있음 (예: "파란색, 대형", "빨간색, 소형") 각각 자체 가격, SKU 및 재고 수량. 테이블 설계.
2. **주문 상태 이력**: 단일 상태 필드 대신, 타임스탬프와 변경한 사용자와 함께 모든 상태 변경을 추적.
3. **판매자 지급**: 판매자를 위한 수수료 계산 및 지급 이력 추적.

각 확장에 대한 CREATE TABLE 문을 작성하고 기존 스키마와 통합하는 방법을 설명하세요.

### 연습 문제 2: 정규화 분석

이 비정규화된 주문 테이블을 고려하세요:

```sql
CREATE TABLE flat_orders (
    order_id        INT,
    customer_name   VARCHAR(100),
    customer_email  VARCHAR(255),
    product_name    VARCHAR(255),
    product_sku     VARCHAR(50),
    product_price   DECIMAL(10,2),
    quantity        INT,
    order_total     DECIMAL(10,2),
    order_date      DATE,
    shipping_street VARCHAR(255),
    shipping_city   VARCHAR(100),
    shipping_zip    VARCHAR(20)
);
```

1. 모든 함수 종속성을 나열하세요.
2. 후보 키를 식별하세요.
3. 테이블이 1NF, 2NF, 3NF 및 BCNF에 있는지 결정하세요.
4. BCNF로 분해하고, 각 분해 단계를 보여주세요.
5. 분해가 무손실 조인인지 확인하세요.

### 연습 문제 3: 인덱스 설계 도전

ShopWave 스키마와 다음 쿼리 패턴을 고려:

```sql
-- Q1: 카테고리에서 판매 중인 제품 (price < original_price)
SELECT * FROM products
WHERE category_id = $1 AND sale_price < price AND is_active = TRUE
ORDER BY (price - sale_price) DESC
LIMIT 20;

-- Q2: 날짜 범위에서 특정 상태의 고객 주문
SELECT * FROM orders
WHERE customer_id = $1 AND status = $2
  AND created_at BETWEEN $3 AND $4
ORDER BY created_at DESC;

-- Q3: 특정 고객이 리뷰한 제품
SELECT p.name, r.rating, r.created_at
FROM reviews r
JOIN products p ON r.product_id = p.id
WHERE r.customer_id = $1
ORDER BY r.created_at DESC;

-- Q4: 이번 달에 가장 많은 주문이 있는 판매자
SELECT s.company_name, COUNT(DISTINCT o.id) AS order_count
FROM sellers s
JOIN products p ON s.id = p.seller_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE o.created_at >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY s.id
ORDER BY order_count DESC
LIMIT 10;
```

각 쿼리에 대해:
1. 최적의 인덱스를 설계하세요.
2. 예상되는 EXPLAIN 계획을 작성하세요.
3. 각 술어의 선택도를 추정하세요.

### 연습 문제 4: 트랜잭션 설계

다음 비즈니스 작업을 위한 트랜잭션을 설계하세요:

**대량 할인 적용**: 관리자가 "전자제품" 카테고리의 모든 제품 가격을 15% 인하하고, 제품당 할인을 $50로 제한하고, 변경을 감사 테이블에 기록하려고 합니다.

요구사항:
- 제품이 중간 상태에서 고객에게 보여서는 안 됨 (부분적으로 할인)
- 작업이 제품 읽기를 1초 이상 차단해서는 안 됨
- 모든 변경은 변경 전/후 값과 함께 기록되어야 함

트랜잭션 SQL을 작성하고 격리 수준을 지정하세요.

### 연습 문제 5: 자신만의 데이터베이스 설계

다음 시나리오 중 하나를 선택하고 전체 데이터베이스 설계를 완성하세요 (단계 1-6):

**옵션 A: 도서관 관리 시스템**
- 이용자, 책 (여러 사본), 대출, 예약, 벌금
- 이용자는 14일간 최대 5권 대출 가능
- 연체 반납은 $0.25/일 벌금 발생
- 모든 사본이 대출 중이면 책을 예약 가능
- 사서는 책을 추가/제거하고 이용자 계정 관리 가능

**옵션 B: 레스토랑 예약 시스템**
- 레스토랑, 테이블, 예약, 고객, 대기 목록
- 레스토랑은 다양한 크기의 테이블 (2, 4, 6, 8명)
- 예약은 특정 날짜/시간 및 인원수
- 테이블이 없으면 대기 목록
- 고객은 레스토랑 리뷰 남길 수 있음

**옵션 C: 온라인 학습 플랫폼**
- 강좌, 레슨, 학생, 등록, 진도 추적, 퀴즈
- 강좌는 정의된 순서로 여러 레슨
- 학생은 진도 추적 (완료된 레슨, 퀴즈 점수)
- 강사는 강좌 생성 및 관리
- 강좌 완료 시 인증서 발급

선택한 시나리오에 대해 다음을 생성하세요:
1. 기능 요구사항 (최소 8개)
2. 물량 추정이 있는 데이터 요구사항
3. 카디널리티가 있는 ER 다이어그램
4. 정규화된 관계형 스키마 (CREATE TABLE 문)
5. 정당화가 있는 인덱싱 전략
6. 격리 수준 및 잠금 전략이 있는 두 개의 중요 트랜잭션
7. 예상 실행 계획이 있는 세 개의 대표 쿼리

### 연습 문제 6: 이 설계 비평

다음 스키마에서 최소 8개의 문제를 찾아 수정하세요:

```sql
CREATE TABLE users (
    id INT,
    name TEXT,
    email TEXT,
    password TEXT,
    created DATE
);

CREATE TABLE products (
    id INT,
    title TEXT,
    price FLOAT,
    category TEXT,
    tags TEXT,  -- 쉼표로 구분: "electronics,sale,new"
    stock INT
);

CREATE TABLE orders (
    id INT,
    user_email TEXT,
    product_ids TEXT,  -- 쉼표로 구분: "1,5,12"
    quantities TEXT,   -- 쉼표로 구분: "2,1,3"
    total FLOAT,
    status TEXT,
    date TEXT
);
```

각 문제에 대해:
1. 문제를 식별하세요.
2. 문제가 있는 이유를 설명하세요.
3. 수정된 SQL을 제공하세요.

### 연습 문제 7: 전자상거래 설계 확장

ShopWave 플랫폼이 다음과 같이 성장했습니다:
- 5천만 고객
- 1천만 제품
- 5억 주문
- 15억 주문 항목
- 1억 리뷰

현재 단일 노드 PostgreSQL은 더 이상 부하를 처리할 수 없습니다. 확장 전략을 제안하세요:

1. 어떤 테이블을 파티션해야 합니까? 어떤 키로?
2. 어떤 테이블에 읽기 복제본이 필요합니까?
3. 어떤 데이터를 Redis로 이동해야 합니까 (캐싱)?
4. 어떤 데이터를 다른 데이터베이스 타입으로 이동해야 합니까 (예: 검색을 위한 Elasticsearch, 주문 이력을 위한 Cassandra)? 정당화하세요.
5. 다운타임 없이 단일 데이터베이스에서 분산 아키텍처로 전환을 어떻게 처리하시겠습니까?

### 연습 문제 8: 소셜 미디어 피드 설계

섹션 8의 SocialBuzz 스키마를 사용:

1. 하이브리드 피드 모델을 구현하세요: 일반 사용자가 게시물을 생성하는 푸시 경로와 읽기 시점에 유명인 게시물을 병합하는 풀 경로 모두에 대한 SQL을 작성하세요.
2. 5백만 팔로워를 가진 유명인이 사용자 이름을 변경합니다. feed_items 테이블을 효율적으로 어떻게 업데이트하시겠습니까?
3. "트렌딩 게시물" 기능을 설계하세요: 지난 24시간 동안 가장 많은 좋아요를 받은 50개 게시물 표시. 어떤 인덱스가 필요합니까? like_count의 쓰기 부하를 어떻게 처리하시겠습니까?
4. 사용자가 다른 사용자를 차단합니다. 피드 시스템에 어떤 변경이 필요합니까?

### 연습 문제 9: 마이그레이션 계획

실시간 ShopWave 데이터베이스에 "제품 변형" 기능을 추가해야 합니다 (연습 문제 1 참조). 다운타임 없는 마이그레이션 계획을 작성하세요:

1. 순서대로 마이그레이션 단계를 나열하세요.
2. 각 단계에 대해, 잠금이 필요한지와 얼마나 오래 걸리는지 지정하세요.
3. 마이그레이션 단계 간에 필요한 역호환 애플리케이션 변경을 설명하세요.
4. 마이그레이션 중간에 문제가 발생하면 롤백 계획은 무엇입니까?

### 연습 문제 10: 설계 검토

설계 검토 체크리스트 (섹션 9)에 대해 완전한 ShopWave 스키마 (단계 3 DDL)를 검토하세요. 아직 충족되지 않은 각 체크리스트 항목에 대해, 간격을 설명하고 SQL 수정을 제공하세요. 최소 5개의 간격을 찾으세요.

---

## 12. 참고문헌

1. Elmasri, R. & Navathe, S. (2016). *Fundamentals of Database Systems*, 7th Edition. Pearson. Chapters 3-9.
2. Garcia-Molina, H., Ullman, J., & Widom, J. (2008). *Database Systems: The Complete Book*, 2nd Edition. Pearson.
3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapters 2-3.
4. Winand, M. (2012). *SQL Performance Explained*. https://use-the-index-luke.com/
5. PostgreSQL Documentation. "Partitioning." https://www.postgresql.org/docs/current/ddl-partitioning.html
6. PostgreSQL Documentation. "Concurrency Control." https://www.postgresql.org/docs/current/mvcc.html
7. Karwin, B. (2010). *SQL Antipatterns*. Pragmatic Bookshelf.
8. Kleppmann, M. (2012). "Designing Data-Intensive Applications: Partitioning." Chapter 6.
9. Sadalage, P. & Fowler, M. (2012). *NoSQL Distilled*. Addison-Wesley. (For polyglot persistence comparisons.)
10. Twitter Engineering Blog. (2013). "How Twitter Stores 250 Million Tweets a Day."

---

**이전**: [15. NewSQL과 현대 트렌드](./15_NewSQL_and_Modern_Trends.md)
