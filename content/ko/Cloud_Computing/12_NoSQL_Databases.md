# NoSQL 데이터베이스

**이전**: [관리형 관계형 데이터베이스](./11_Managed_Relational_DB.md) | **다음**: [IAM](./13_Identity_Access_Management.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CAP 정리(CAP theorem)와 NoSQL 데이터베이스가 일관성과 가용성 사이에서 트레이드오프를 만드는 방식을 설명할 수 있습니다
2. Key-Value, 문서(document), 와이드 컬럼(wide-column), 인메모리(in-memory) NoSQL 데이터베이스 유형을 구분할 수 있습니다
3. 기능과 요금 모델 측면에서 AWS DynamoDB와 GCP Firestore/Bigtable을 비교할 수 있습니다
4. 적절한 파티션 키(partition key)와 정렬 키(sort key)로 DynamoDB 테이블을 설계할 수 있습니다
5. 비용 최적화를 위해 읽기/쓰기 용량 모드(프로비저닝 vs. 온디맨드)를 구성할 수 있습니다
6. 유연한 쿼리 패턴을 위해 보조 인덱스(secondary index)를 구현할 수 있습니다
7. 관계형 데이터베이스보다 NoSQL이 더 적합한 워크로드를 식별할 수 있습니다

---

모든 워크로드가 관계형 모델에 적합한 것은 아닙니다. 대규모 확장, 유연한 스키마, 또는 한 자릿수 밀리초(single-digit-millisecond) 지연 시간이 요구되는 애플리케이션은 NoSQL 데이터베이스로 이점을 얻는 경우가 많습니다. 클라우드 관리형 NoSQL 서비스는 자동 확장, 내장 복제, 서버 관리 없음을 제공하여 현대적인 분산 애플리케이션에 자연스럽게 적합합니다.

## 1. NoSQL 개요

### 1.1 NoSQL vs RDBMS

| 항목 | RDBMS | NoSQL |
|------|-------|-------|
| 스키마 | 엄격한 스키마 | 유연한 스키마 |
| 확장성 | 수직 확장 | 수평 확장 |
| 트랜잭션 | ACID | BASE (일부 ACID) |
| 쿼리 | SQL | 다양한 API |
| 사용 사례 | 트랜잭션, 복잡한 관계 | 대용량, 유연한 데이터 |

### 1.2 서비스 비교

| 유형 | AWS | GCP |
|------|-----|-----|
| Key-Value / Document | DynamoDB | Firestore |
| Wide Column | - | Bigtable |
| In-Memory Cache | ElastiCache | Memorystore |
| Document (MongoDB) | DocumentDB | MongoDB Atlas (마켓플레이스) |

---

## 2. AWS DynamoDB

### 2.1 DynamoDB 개요

**특징:**
- 완전 관리형 Key-Value / Document DB
- 밀리초 지연 시간
- 무한 확장
- 서버리스 (온디맨드 용량)

**핵심 개념:**
- **테이블**: 데이터 컨테이너
- **항목 (Item)**: 레코드
- **속성 (Attribute)**: 필드
- **Primary Key**: 파티션 키 + (선택적) 정렬 키

### 2.2 테이블 생성

```bash
# 테이블 생성 (파티션 키만)
aws dynamodb create-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=userId,AttributeType=S \
    --key-schema \
        AttributeName=userId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# 테이블 생성 (파티션 키 + 정렬 키)
aws dynamodb create-table \
    --table-name Orders \
    --attribute-definitions \
        AttributeName=customerId,AttributeType=S \
        AttributeName=orderId,AttributeType=S \
    --key-schema \
        AttributeName=customerId,KeyType=HASH \
        AttributeName=orderId,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST

# 테이블 목록
aws dynamodb list-tables

# 테이블 정보
aws dynamodb describe-table --table-name Users
```

### 2.3 CRUD 작업

```bash
# 항목 추가 (PutItem)
aws dynamodb put-item \
    --table-name Users \
    --item '{
        "userId": {"S": "user-001"},
        "name": {"S": "John Doe"},
        "email": {"S": "john@example.com"},
        "age": {"N": "30"}
    }'

# 항목 조회 (GetItem)
aws dynamodb get-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# 항목 업데이트 (UpdateItem)
aws dynamodb update-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}' \
    --update-expression "SET age = :newAge" \
    --expression-attribute-values '{":newAge": {"N": "31"}}'

# 항목 삭제 (DeleteItem)
aws dynamodb delete-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# 스캔 (전체 테이블)
aws dynamodb scan --table-name Users

# 쿼리 (파티션 키 기반)
aws dynamodb query \
    --table-name Orders \
    --key-condition-expression "customerId = :cid" \
    --expression-attribute-values '{":cid": {"S": "customer-001"}}'
```

### 2.4 Python SDK (boto3)

```python
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')

# 항목 추가
table.put_item(Item={
    'userId': 'user-002',
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# 항목 조회
response = table.get_item(Key={'userId': 'user-002'})
item = response.get('Item')

# 쿼리 (GSI 사용 시)
response = table.query(
    IndexName='email-index',
    KeyConditionExpression='email = :email',
    ExpressionAttributeValues={':email': 'jane@example.com'}
)

# 배치 쓰기
with table.batch_writer() as batch:
    for i in range(100):
        batch.put_item(Item={'userId': f'user-{i}', 'name': f'User {i}'})
```

### 2.5 글로벌 보조 인덱스 (GSI)

```bash
# GSI 추가
aws dynamodb update-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=email,AttributeType=S \
    --global-secondary-index-updates '[
        {
            "Create": {
                "IndexName": "email-index",
                "KeySchema": [{"AttributeName": "email", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"}
            }
        }
    ]'
```

### 2.6 DynamoDB Streams

변경 데이터 캡처 (CDC)를 위한 스트림입니다.

```bash
# 스트림 활성화
aws dynamodb update-table \
    --table-name Users \
    --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES

# Lambda 트리거 연결
aws lambda create-event-source-mapping \
    --function-name process-dynamodb \
    --event-source-arn arn:aws:dynamodb:...:table/Users/stream/xxx \
    --starting-position LATEST
```

---

## 3. GCP Firestore

### 3.1 Firestore 개요

**특징:**
- 문서 기반 NoSQL DB
- 실시간 동기화
- 오프라인 지원
- 자동 확장

**핵심 개념:**
- **컬렉션**: 문서 그룹
- **문서**: JSON과 유사한 데이터
- **하위 컬렉션**: 계층 구조

### 3.2 Firestore 설정

```bash
# Firestore API 활성화
gcloud services enable firestore.googleapis.com

# 데이터베이스 생성 (Native 모드)
gcloud firestore databases create \
    --location=asia-northeast3 \
    --type=firestore-native
```

### 3.3 Python SDK

```python
from google.cloud import firestore

db = firestore.Client()

# 문서 추가 (자동 ID)
doc_ref = db.collection('users').add({
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
})

# 문서 추가/업데이트 (지정 ID)
db.collection('users').document('user-001').set({
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# 문서 조회
doc = db.collection('users').document('user-001').get()
if doc.exists:
    print(doc.to_dict())

# 부분 업데이트
db.collection('users').document('user-001').update({
    'age': 26
})

# 문서 삭제
db.collection('users').document('user-001').delete()

# 쿼리
users = db.collection('users').where('age', '>=', 25).stream()
for user in users:
    print(f'{user.id} => {user.to_dict()}')

# 복합 쿼리 (인덱스 필요)
users = db.collection('users') \
    .where('age', '>=', 25) \
    .order_by('age') \
    .limit(10) \
    .stream()
```

### 3.4 실시간 리스너

```python
# 문서 변경 감지
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f'Received document snapshot: {doc.id}')

doc_ref = db.collection('users').document('user-001')
doc_watch = doc_ref.on_snapshot(on_snapshot)

# 컬렉션 변경 감지
col_ref = db.collection('users')
col_watch = col_ref.on_snapshot(on_snapshot)
```

### 3.5 보안 규칙

```javascript
// firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // 인증된 사용자만 자신의 문서 접근
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }

    // 공개 읽기
    match /public/{document=**} {
      allow read: if true;
      allow write: if request.auth != null;
    }
  }
}
```

```bash
# 보안 규칙 배포
firebase deploy --only firestore:rules
```

---

## 4. 인메모리 캐시

### 4.1 AWS ElastiCache

**지원 엔진:**
- Redis
- Memcached

```bash
# Redis 클러스터 생성
aws elasticache create-cache-cluster \
    --cache-cluster-id my-redis \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1 \
    --cache-subnet-group-name my-subnet-group \
    --security-group-ids sg-12345678

# 복제 그룹 생성 (고가용성)
aws elasticache create-replication-group \
    --replication-group-id my-redis-cluster \
    --replication-group-description "Redis cluster" \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-node-groups 1 \
    --replicas-per-node-group 1 \
    --automatic-failover-enabled \
    --cache-subnet-group-name my-subnet-group

# 엔드포인트 확인
aws elasticache describe-cache-clusters \
    --cache-cluster-id my-redis \
    --show-cache-node-info
```

**Python 연결:**
```python
import redis

# 단일 노드
r = redis.Redis(
    host='my-redis.xxx.cache.amazonaws.com',
    port=6379,
    decode_responses=True
)

# SET/GET
r.set('key', 'value')
value = r.get('key')

# 해시
r.hset('user:1000', mapping={'name': 'John', 'email': 'john@example.com'})
user = r.hgetall('user:1000')

# TTL
r.setex('session:abc', 3600, 'user_data')
```

### 4.2 GCP Memorystore

**지원 엔진:**
- Redis
- Memcached

```bash
# Redis 인스턴스 생성
gcloud redis instances create my-redis \
    --region=asia-northeast3 \
    --tier=BASIC \
    --size=1 \
    --redis-version=redis_6_x

# 인스턴스 정보 확인
gcloud redis instances describe my-redis \
    --region=asia-northeast3

# 연결 정보 (호스트/포트)
gcloud redis instances describe my-redis \
    --region=asia-northeast3 \
    --format='value(host,port)'
```

**연결:**
```python
import redis

# Memorystore Redis (Private IP)
r = redis.Redis(
    host='10.0.0.3',  # Private IP
    port=6379,
    decode_responses=True
)

r.set('hello', 'world')
print(r.get('hello'))
```

---

## 5. 용량 모드

### 5.1 DynamoDB 용량 모드

| 모드 | 특징 | 적합한 경우 |
|------|------|-----------|
| **온디맨드** | 자동 확장, 요청당 과금 | 트래픽 예측 불가 |
| **프로비저닝** | 용량 사전 지정 | 안정적 트래픽 |

```bash
# 온디맨드 모드
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PAY_PER_REQUEST

# 프로비저닝 모드
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PROVISIONED \
    --provisioned-throughput ReadCapacityUnits=100,WriteCapacityUnits=100

# Auto Scaling 설정
aws application-autoscaling register-scalable-target \
    --service-namespace dynamodb \
    --resource-id "table/Users" \
    --scalable-dimension "dynamodb:table:ReadCapacityUnits" \
    --min-capacity 5 \
    --max-capacity 1000
```

### 5.2 Firestore 용량

Firestore는 완전 서버리스로 자동 확장됩니다.

**과금:**
- 문서 읽기: $0.06 / 100,000
- 문서 쓰기: $0.18 / 100,000
- 문서 삭제: $0.02 / 100,000
- 스토리지: $0.18 / GB / 월

---

## 6. 비용 비교

### 6.1 DynamoDB

| 항목 | 온디맨드 | 프로비저닝 |
|------|---------|-----------|
| 읽기 | $0.25 / 100만 RRU | $0.00013 / RCU / 시간 |
| 쓰기 | $1.25 / 100만 WRU | $0.00065 / WCU / 시간 |
| 스토리지 | $0.25 / GB / 월 | $0.25 / GB / 월 |

### 6.2 Firestore

| 항목 | 비용 |
|------|------|
| 문서 읽기 | $0.06 / 100,000 |
| 문서 쓰기 | $0.18 / 100,000 |
| 스토리지 | $0.18 / GB / 월 |

### 6.3 ElastiCache / Memorystore

| 서비스 | 노드 타입 | 시간당 비용 |
|--------|----------|------------|
| ElastiCache | cache.t3.micro | ~$0.02 |
| ElastiCache | cache.r5.large | ~$0.20 |
| Memorystore | 1GB Basic | ~$0.05 |
| Memorystore | 1GB Standard (HA) | ~$0.10 |

---

## 7. 사용 사례별 선택

| 사용 사례 | 권장 서비스 |
|----------|-----------|
| 세션 관리 | ElastiCache / Memorystore |
| 사용자 프로필 | DynamoDB / Firestore |
| 실시간 채팅 | Firestore (실시간 동기화) |
| 게임 리더보드 | ElastiCache Redis |
| IoT 데이터 | DynamoDB / Bigtable |
| 장바구니 | DynamoDB / Firestore |
| 캐싱 | ElastiCache / Memorystore |

---

## 8. 설계 패턴

### 8.1 DynamoDB 단일 테이블 설계

```
PK              | SK              | 속성
----------------|-----------------|------------------
USER#123        | USER#123        | name, email
USER#123        | ORDER#001       | product, quantity
USER#123        | ORDER#002       | product, quantity
PRODUCT#A       | PRODUCT#A       | name, price
PRODUCT#A       | REVIEW#001      | rating, comment
```

### 8.2 캐시 패턴

**Cache-Aside (Lazy Loading):**
```python
def get_user(user_id):
    # 캐시 확인
    cached = cache.get(f'user:{user_id}')
    if cached:
        return cached

    # DB에서 조회
    user = db.get_user(user_id)

    # 캐시 저장
    cache.setex(f'user:{user_id}', 3600, user)
    return user
```

**Write-Through:**
```python
def update_user(user_id, data):
    # DB 업데이트
    db.update_user(user_id, data)

    # 캐시 업데이트
    cache.set(f'user:{user_id}', data)
```

---

## 9. 다음 단계

- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - RDB

---

## 연습 문제

### 연습 문제 1: NoSQL vs RDBMS 선택

각 시나리오에서 관계형 데이터베이스(RDS/Cloud SQL)와 NoSQL 데이터베이스(DynamoDB/Firestore) 중 어느 것이 더 적합한지 판단하고 이유를 설명하세요.

1. 이커머스 플랫폼이 강력한 참조 무결성(referential integrity)과 보고를 위한 복잡한 조인(join)이 필요한 주문, 고객, 제품을 저장해야 합니다.
2. 일관된 한 자리 밀리초(single-digit millisecond) 읽기와 초당 100,000건의 점수 업데이트를 처리해야 하는 실시간 게임 리더보드
3. 소셜 미디어 애플리케이션이 사용자마다 다른 선택적 필드(일부는 자기 소개, 일부는 회사 정보 등)가 있는 사용자 프로필을 저장합니다.
4. 차변(debit)과 대변(credit) 작업 간의 원자성(atomicity)이 필요한 자금 이체를 처리하는 뱅킹 애플리케이션

<details>
<summary>정답 보기</summary>

1. **관계형(RDBMS)** — 주문/고객/제품 간의 복잡한 조인과 강력한 참조 무결성(외래 키, foreign key)은 RDBMS의 강점입니다. NoSQL 데이터베이스는 기본적으로 다중 테이블 조인을 지원하지 않습니다. 모든 것을 비정규화(denormalize)하거나 여러 번의 쿼리를 수행해야 합니다.

2. **NoSQL(DynamoDB)** — 리더보드는 극도의 쓰기 처리량과 낮은 지연 시간의 읽기가 필요합니다. DynamoDB의 온디맨드(on-demand) 모드는 초당 100,000건의 쓰기를 자동으로 처리하도록 확장됩니다. 파티션 키로 `gameId`, 정렬 키로 `score`를 사용하고 GSI(Global Secondary Index)를 추가하면 밀리초 내에 정렬된 리더보드 쿼리가 가능합니다.

3. **NoSQL(DynamoDB 또는 Firestore)** — 유연하고 스키마가 없는 문서 스토리지는 NoSQL의 강점입니다. DynamoDB 속성(attribute)은 선택적입니다 — 각 사용자에게 존재하는 것만 저장합니다. 새 필드 유형(예: "회사 정보") 추가 시 스키마 마이그레이션이 필요 없습니다.

4. **관계형(RDBMS)** — ACID 트랜잭션은 금융 작업에 중요합니다. 자금 이체는 한 계좌를 차변으로, 다른 계좌를 대변으로 원자적으로 처리해야 합니다. RDS는 다중 행, 다중 테이블 트랜잭션을 지원합니다. DynamoDB도 트랜잭션을 지원하지만 단일 테이블 또는 제한된 아이템 집합 내에서만 가능합니다.

</details>

### 연습 문제 2: DynamoDB 키(Key) 설계

주문 관리 시스템을 위한 DynamoDB 테이블을 설계하고 있습니다. 다음 쿼리 패턴이 있습니다:
- 주문 ID로 특정 주문 조회
- 특정 고객의 모든 주문을 주문 날짜 순으로 나열
- 지난 7일 동안 접수된 모든 주문 조회 (모든 고객 대상)

이 쿼리 패턴을 지원하는 기본 키(파티션 키와 정렬 키)와 필요한 GSI(Global Secondary Index)를 설계하세요.

<details>
<summary>정답 보기</summary>

**기본 키 설계**:

| 키 | 속성 | 이유 |
|----|------|------|
| 파티션 키(HASH) | `customerId` | 한 고객의 모든 주문을 동일 파티션에 그룹화하여 효율적인 검색 가능 |
| 정렬 키(RANGE) | `orderDate#orderId` | 날짜별 범위 쿼리 활성화. `orderDate` 접두사(ISO 형식: `2026-02-24`)로 날짜순 사전식 정렬 보장; `orderId`를 추가하여 유일성 보장 |

이 설계는 다음을 지원합니다:
- **쿼리 2** (고객의 모든 주문, 날짜 순): 정렬 키에 선택적 날짜 범위 필터로 `Query(customerId="C-001")`
- **쿼리 1** (특정 주문): `customerId`와 전체 정렬 키 `orderDate#orderId` 모두 제공

**고객 간 날짜 쿼리를 위한 GSI (쿼리 3)**:

| | 속성 |
|--|------|
| GSI 파티션 키 | `orderDate` (날짜만, 예: `2026-02-24`) |
| GSI 정렬 키 | `orderId` |

지난 7일 동안의 모든 주문: 7개의 날짜 값 각각에 대해 GSI에 쿼리합니다. 이것은 스캐터-개더(scatter-gather) 패턴입니다.

```bash
aws dynamodb create-table \
    --table-name Orders \
    --attribute-definitions \
        AttributeName=customerId,AttributeType=S \
        AttributeName=orderDateOrderId,AttributeType=S \
        AttributeName=orderDate,AttributeType=S \
        AttributeName=orderId,AttributeType=S \
    --key-schema \
        AttributeName=customerId,KeyType=HASH \
        AttributeName=orderDateOrderId,KeyType=RANGE \
    --global-secondary-indexes '[{
        "IndexName": "OrdersByDate",
        "KeySchema": [
            {"AttributeName": "orderDate", "KeyType": "HASH"},
            {"AttributeName": "orderId", "KeyType": "RANGE"}
        ],
        "Projection": {"ProjectionType": "ALL"}
    }]' \
    --billing-mode PAY_PER_REQUEST
```

</details>

### 연습 문제 3: 용량 모드(Capacity Mode) 선택

DynamoDB 테이블이 제품 카탈로그를 제공합니다. 사용 패턴:
- 평일: 일관되게 초당 ~50 읽기, ~5 쓰기
- 블랙 프라이데이(Black Friday): ~6시간 동안 초당 2,000 읽기와 500 쓰기 피크
- 주말: 최소 트래픽 (~10 읽기/초)

프로비저닝(Provisioned) 또는 온디맨드(On-Demand) 용량 중 어느 것을 사용해야 합니까? 이유를 설명하고 비용 차이를 추정하세요.

<details>
<summary>정답 보기</summary>

**권장: 온디맨드(On-Demand) 용량**

**이유**:
- 트래픽 패턴이 블랙 프라이데이 기간에 평균의 40배 피크로 매우 가변적입니다.
- 온디맨드는 사전 프로비저닝 없이 즉각적으로 어떤 수준으로도 확장되어 2,000 RPS 피크를 스로틀링(throttling) 없이 처리합니다.
- 블랙 프라이데이 피크에 맞게 프로비저닝한다면, 주말에 초당 10 읽기만 필요할 때도 2,000 RCU와 500 WCU 비용을 일년 내내 지불해야 합니다.

**비용 비교** (근사치, `ap-northeast-2` 기준):

**온디맨드**:
- 평일 일반: 50 읽기 × 3600 × 16시간 × $0.000000125/RCU = ~일 $0.36
- 블랙 프라이데이(6시간 피크): 2000 × 6 × 3600 × $0.000000125 = ~$5.40 + 쓰기
- 연간 합계: 대략 **연 $200~300**

**블랙 프라이데이 피크로 프로비저닝(2,000 RCU, 500 WCU)**:
- 2,000 RCU × $0.00013/시간 × 8,760시간 = **~연 $2,277** (읽기만)
- 플러스 쓰기 용량

**온디맨드 절감**: 연간 수천 달러.

**프로비저닝이 더 나은 경우**: 부하가 연중 일관되게 초당 50 읽기이고 급증이 없다면, 50 RCU로 프로비저닝 시 비용: 50 × $0.00013 × 8,760 = ~연 $57 — 예측 가능한 워크로드에는 온디맨드보다 훨씬 저렴합니다. 안정적이고 예측 가능한 트래픽에는 프로비저닝을, 가변적이거나 예측 불가능한 트래픽에는 온디맨드를 사용하세요.

</details>

### 연습 문제 4: ElastiCache 사용 사례

이커머스 애플리케이션이 현재 모든 페이지 로드 시 RDS MySQL에서 제품 세부 정보를 쿼리합니다. 제품 카탈로그에는 50,000개 제품이 있으며 하루에 최대 한 번 변경됩니다. 주로 데이터베이스 쿼리로 인해 제품 페이지 응답 시간이 800ms입니다.

ElastiCache(Redis)를 사용하여 응답 시간을 개선하는 방법을 설명하세요. 캐싱 패턴, 캐시 키 설계, TTL 전략을 포함하세요.

<details>
<summary>정답 보기</summary>

**캐싱 패턴: 캐시-어사이드(Cache-Aside, Lazy Loading)**

```
제품 세부 정보에 대한 애플리케이션 요청:
1. 캐시 확인: GET product:{product_id}
2. 캐시 히트(CACHE HIT) → 캐시된 데이터 반환 (1-5ms 응답 시간)
3. 캐시 미스(CACHE MISS) → RDS 쿼리, 캐시에 저장, 데이터 반환
```

**캐시 키 설계**:
- 개별 제품: `product:{product_id}` (예: `product:12345`)
- 카테고리별 제품 목록: `products:category:{category_id}:page:{page}` (예: `products:category:electronics:page:1`)

**TTL 전략**:
- 제품은 하루에 최대 한 번 업데이트됨 → TTL을 **3600초(1시간)** 또는 **86400초(24시간)**으로 설정
- 1시간 TTL의 의미: 제품 업데이트 후 캐시된 이전 값이 1시간 내에 만료됩니다. 대부분의 제품 세부 정보(가격, 설명)에는 허용됩니다.
- 시간에 민감한 데이터(재고 수량, 플래시 세일 가격): 더 짧은 TTL(60초)을 사용하거나 데이터 변경 시 캐시 키를 즉시 무효화합니다.

**업데이트 시 캐시 무효화**:
```python
# 제품이 업데이트될 때
def update_product(product_id, data):
    db.update(f"UPDATE products SET ... WHERE id={product_id}")
    redis.delete(f"product:{product_id}")  # 즉시 무효화
```

**기대 개선 효과**: 캐시된 제품에 대한 데이터베이스 쿼리가 ~95% 감소합니다(캐시 미스만 데이터베이스에 도달). 제품 페이지 응답 시간이 800ms에서 ~50ms로 떨어집니다(Redis 인메모리 조회 + 약간의 처리).

</details>

### 연습 문제 5: DynamoDB 글로벌 테이블(Global Tables)

게임 회사가 한국(아시아-태평양)과 미국에 플레이어가 있습니다. DynamoDB 플레이어 프로필 테이블을 두 리전 모두에서 낮은 지연 시간으로 사용 가능하게 하고 완전한 리전 장애에서 살아남기를 원합니다.

1. 이를 가능하게 하는 DynamoDB 기능은 무엇입니까?
2. 이 기능을 사용하기 전에 이해해야 할 일관성(consistency) 의미는 무엇입니까?
3. 복제본 리전을 추가하는 CLI 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

1. **DynamoDB 글로벌 테이블(Global Tables)** — 완전 관리형 다중 리전, 다중 활성(multi-active) 복제 기능입니다. 어느 리전에서든 테이블에 대한 쓰기가 자동으로 다른 모든 리전에 복제되며, 일반적으로 1초 내에 완료됩니다. 각 리전에는 완전히 쓰기 가능한 로컬 복사본이 있습니다.

2. **일관성 의미**:
   - **리전 간 최종 일관성(eventual consistency)**: 서울에서의 쓰기가 버지니아에 나타나는 데 최대 ~1초가 걸릴 수 있습니다. 한국 플레이어가 프로필을 업데이트하면, 미국 플레이어(또는 미국 리전 읽기)는 잠시 이전 값을 볼 수 있습니다.
   - **충돌 해결(conflict resolution)**: 두 리전이 동일한 아이템을 동시에 업데이트하는 경우(예: 네트워크 파티션으로 두 리전에서 동시에 아이템 구매), DynamoDB는 **"마지막 작성자 승리(last writer wins)"** (타임스탬프 기반)를 사용합니다. 패배한 쓰기는 자동으로 삭제됩니다.
   - **리전 내 읽기**: 강력한 일관성 읽기(strongly consistent reads)는 단일 리전 내에서만 가능합니다. 교차 리전 읽기는 항상 최종 일관성입니다.
   - **설계 권장사항**: 교차 리전 충돌을 최소화하기 위해 특정 사용자의 쓰기가 항상 홈 리전으로 라우팅되도록 애플리케이션을 설계하세요.

3. **복제본 리전 추가 CLI**:
```bash
# 먼저 테이블이 온디맨드 또는 프로비저닝 모드여야 합니다
# us-east-1을 복제본 리전으로 추가
aws dynamodb update-table \
    --table-name PlayerProfiles \
    --replica-updates '[{
        "Create": {
            "RegionName": "us-east-1"
        }
    }]' \
    --region ap-northeast-2

# 복제본 상태 확인
aws dynamodb describe-table \
    --table-name PlayerProfiles \
    --region ap-northeast-2 \
    --query 'Table.Replicas'
```

</details>

---

## 참고 자료

- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [AWS ElastiCache Documentation](https://docs.aws.amazon.com/elasticache/)
- [GCP Firestore Documentation](https://cloud.google.com/firestore/docs)
- [GCP Memorystore Documentation](https://cloud.google.com/memorystore/docs)
