# 서버리스 함수 (Lambda / Cloud Functions)

**이전**: [가상 머신](./04_Virtual_Machines.md) | **다음**: [컨테이너 서비스](./06_Container_Services.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 서버리스(Serverless) 실행 모델과 이벤트 기반(event-driven) 아키텍처를 설명할 수 있다
2. 런타임, 제한, 트리거 측면에서 AWS Lambda와 GCP Cloud Functions를 비교할 수 있다
3. 콜드 스타트(Cold Start)의 원인과 완화 전략을 파악할 수 있다
4. 콘솔과 CLI를 사용해 양 플랫폼에서 서버리스 함수를 배포할 수 있다
5. HTTP, 스토리지, 메시지 큐 등의 이벤트 소스를 함수 트리거로 설정할 수 있다
6. 동시성과 확장을 자동으로 처리하는 서버리스 애플리케이션을 설계할 수 있다
7. 호출 횟수, 실행 시간, 메모리를 기반으로 서버리스 비용을 계산할 수 있다

---

서버리스 컴퓨팅(Serverless Computing)은 서버를 프로비저닝하고 관리할 필요를 완전히 없애줍니다. 함수를 작성하고 트리거를 정의하면, 클라우드 제공자가 나머지 모든 것 — 확장, 패치 적용, 용량 계획 — 을 처리합니다. 이 모델은 이벤트 기반 워크로드에 이상적이며, 예측 불가능한 트래픽 패턴에서 운영 부담과 비용을 크게 줄일 수 있습니다.

> **비유 — 전등 스위치**: 전통적인 서버는 집의 모든 방 전등을 24시간 내내 켜두는 것과 같습니다. 아무도 없을 때도 전기 요금은 나옵니다. 서버리스는 동작 감지 센서 전등과 같습니다 — 누군가 들어올 때만 전력이 흐르고, 떠나면 자동으로 꺼집니다. 배선이나 전구를 직접 관리할 필요가 없습니다. *전등이 켜질 때 무슨 일이 일어날지*만 정의하면 됩니다.

## 1. 서버리스 개요

### 1.1 서버리스란?

서버리스는 서버 관리 없이 코드를 실행하는 컴퓨팅 모델입니다.

**특징:**
- 서버 프로비저닝/관리 불필요
- 자동 확장
- 사용한 만큼만 과금 (실행 시간 + 요청 수)
- 이벤트 기반 실행

### 1.2 서비스 비교

| 항목 | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| 런타임 | Node.js, Python, Java, Go, Ruby, .NET, Custom | Node.js, Python, Go, Java, Ruby, PHP, .NET |
| 메모리 | 128MB ~ 10GB | 128MB ~ 32GB |
| 최대 실행 시간 | 15분 | 9분 (1세대) / 60분 (2세대) |
| 동시 실행 | 1000 (기본, 증가 가능) | 무제한 (기본) |
| 트리거 | API Gateway, S3, DynamoDB, SNS 등 | HTTP, Pub/Sub, Cloud Storage 등 |
| 컨테이너 지원 | 지원 (Container Image) | 2세대만 지원 |

---

## 2. Cold Start

### 2.1 Cold Start란?

함수가 처음 호출되거나 유휴 상태에서 깨어날 때 발생하는 지연입니다.

```
요청 → [Cold Start] → 컨테이너 생성 → 런타임 초기화 → 코드 로드 → 핸들러 실행
        ~100ms-수초                                                    ~수ms-수초

요청 → [Warm Start] → 핸들러 실행
        ~수ms
```

### 2.2 Cold Start 완화 전략

| 전략 | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| **Provisioned Concurrency** | 지원 (유료) | - |
| **최소 인스턴스** | - | 2세대에서 min-instances |
| **경량 런타임** | Python, Node.js 권장 | Python, Node.js 권장 |
| **패키지 최소화** | 불필요한 의존성 제거 | 불필요한 의존성 제거 |
| **지속적 호출** | CloudWatch Events로 warm-up | Cloud Scheduler로 warm-up |

---

## 3. AWS Lambda

### 3.1 함수 생성 (Console)

1. Lambda 콘솔 → "함수 생성"
2. "새로 작성" 선택
3. 함수 이름 입력
4. 런타임 선택 (예: Python 3.12)
5. 아키텍처 선택 (x86_64 또는 arm64)
6. "함수 생성"

### 3.2 함수 코드 (Python)

```python
import json

def lambda_handler(event, context):
    """
    event: 트리거에서 전달된 데이터
    context: 런타임 정보 (함수명, 메모리, 남은 시간 등)
    """
    # 이벤트 로깅
    print(f"Event: {json.dumps(event)}")

    # 비즈니스 로직
    name = event.get('name', 'World')
    message = f"Hello, {name}!"

    # 응답 반환
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'message': message
        })
    }
```

### 3.3 함수 생성 (AWS CLI)

```bash
# 1. 코드 패키징
zip function.zip lambda_function.py

# 2. IAM 역할 생성 (Lambda 실행 역할)
aws iam create-role \
    --role-name lambda-execution-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# 3. 기본 정책 연결
aws iam attach-role-policy \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 4. Lambda 함수 생성
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler lambda_function.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --zip-file fileb://function.zip

# 5. 함수 테스트
aws lambda invoke \
    --function-name my-function \
    --payload '{"name": "Claude"}' \
    --cli-binary-format raw-in-base64-out \
    output.json

cat output.json
```

### 3.4 API Gateway 연동

```bash
# 1. REST API 생성
aws apigateway create-rest-api \
    --name my-api \
    --endpoint-configuration types=REGIONAL

# 2. Lambda와 통합 (Console에서 하는 것이 더 쉬움)
# Lambda 콘솔 → 함수 → 트리거 추가 → API Gateway 선택
```

---

## 4. GCP Cloud Functions

### 4.1 함수 생성 (Console)

1. Cloud Functions 콘솔 → "함수 만들기"
2. 환경 선택 (1세대 또는 2세대)
3. 함수 이름 입력
4. 리전 선택
5. 트리거 유형 선택 (HTTP, Pub/Sub 등)
6. 런타임 선택 (예: Python 3.12)
7. 코드 작성 후 "배포"

### 4.2 함수 코드 (Python)

**HTTP 트리거:**
```python
import functions_framework
from flask import jsonify

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request: Flask request object
    Returns:
        Response object
    """
    request_json = request.get_json(silent=True)
    name = 'World'

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request.args and 'name' in request.args:
        name = request.args.get('name')

    return jsonify({
        'message': f'Hello, {name}!'
    })
```

**Pub/Sub 트리거:**
```python
import base64
import functions_framework

@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    """Pub/Sub Cloud Function.
    Args:
        cloud_event: CloudEvent object
    """
    data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    print(f"Received message: {data}")
```

### 4.3 함수 배포 (gcloud CLI)

```bash
# 1. 프로젝트 구조
# my-function/
# ├── main.py
# └── requirements.txt

# requirements.txt
# functions-framework==3.*

# 2. HTTP 함수 배포
gcloud functions deploy hello-http \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-http \
    --allow-unauthenticated \
    --entry-point=hello_http \
    --source=.

# 3. 함수 URL 확인
gcloud functions describe hello-http \
    --region=asia-northeast3 \
    --format='value(url)'

# 4. 함수 테스트
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name": "Claude"}' \
    https://asia-northeast3-PROJECT_ID.cloudfunctions.net/hello-http
```

### 4.4 Cloud Storage 트리거

```python
import functions_framework

@functions_framework.cloud_event
def hello_gcs(cloud_event):
    """Cloud Storage trigger function.
    Args:
        cloud_event: CloudEvent object
    """
    data = cloud_event.data

    bucket = data["bucket"]
    name = data["name"]

    print(f"File uploaded: gs://{bucket}/{name}")
```

```bash
# Cloud Storage 트리거 배포
gcloud functions deploy process-upload \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=my-bucket" \
    --entry-point=hello_gcs \
    --source=.
```

---

## 5. 트리거 유형 비교

### 5.1 AWS Lambda 트리거

| 트리거 | 설명 | 예시 |
|--------|------|------|
| **API Gateway** | HTTP 요청 | REST API, WebSocket |
| **S3** | 객체 이벤트 | 업로드, 삭제 |
| **DynamoDB Streams** | 테이블 변경 | Insert, Modify, Remove |
| **SNS** | 알림 메시지 | 푸시 알림 |
| **SQS** | 큐 메시지 | 비동기 처리 |
| **CloudWatch Events** | 스케줄, 이벤트 | 크론 작업 |
| **Kinesis** | 스트림 데이터 | 실시간 분석 |
| **Cognito** | 인증 이벤트 | 회원가입 후처리 |

### 5.2 GCP Cloud Functions 트리거

| 트리거 | 설명 | 예시 |
|--------|------|------|
| **HTTP** | HTTP 요청 | REST API |
| **Cloud Storage** | 객체 이벤트 | 업로드, 삭제 |
| **Pub/Sub** | 메시지 | 비동기 처리 |
| **Firestore** | 문서 변경 | Insert, Update, Delete |
| **Cloud Scheduler** | 스케줄 | 크론 작업 |
| **Eventarc** | 다양한 GCP 이벤트 | 2세대 통합 트리거 |

---

## 6. 환경 변수 및 비밀 관리

### 6.1 AWS Lambda 환경 변수

```bash
# 환경 변수 설정
aws lambda update-function-configuration \
    --function-name my-function \
    --environment "Variables={DB_HOST=mydb.example.com,DB_PORT=5432}"
```

**코드에서 사용:**
```python
import os

def lambda_handler(event, context):
    db_host = os.environ.get('DB_HOST')
    db_port = os.environ.get('DB_PORT')
    # ...
```

**Secrets Manager 연동:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

def lambda_handler(event, context):
    secrets = get_secret('my-database-credentials')
    db_password = secrets['password']
    # ...
```

### 6.2 GCP Cloud Functions 환경 변수

```bash
# 환경 변수 설정
gcloud functions deploy my-function \
    --set-env-vars DB_HOST=mydb.example.com,DB_PORT=5432 \
    ...
```

**Secret Manager 연동:**
```bash
# Secret 참조
gcloud functions deploy my-function \
    --set-secrets 'DB_PASSWORD=projects/PROJECT_ID/secrets/db-password:latest' \
    ...
```

**코드에서 사용:**
```python
import os

def hello_http(request):
    db_host = os.environ.get('DB_HOST')
    db_password = os.environ.get('DB_PASSWORD')  # Secret Manager에서 자동 주입
    # ...
```

---

## 7. 과금 비교

### 7.1 AWS Lambda 과금

```
월 비용 = 요청 비용 + 실행 시간 비용

요청 비용: $0.20 / 100만 요청
실행 시간: $0.0000166667 / GB-초 (x86)
         $0.0000133334 / GB-초 (ARM)

무료 티어 (항상 무료):
- 100만 요청/월
- 40만 GB-초/월
```

**예시 계산:**
```
조건: 512MB 메모리, 200ms 실행, 100만 요청/월

요청 비용: (1,000,000 - 1,000,000) × $0.20/1M = $0
실행 시간:
  - 0.5GB × 0.2초 × 1,000,000 = 100,000 GB-초
  - 무료: 400,000 GB-초
  - 비용: $0 (무료 티어 내)

총 비용: $0/월 (무료 티어 활용)
```

### 7.2 GCP Cloud Functions 과금

```
월 비용 = 호출 비용 + 컴퓨팅 시간 + 네트워크 비용

호출 비용: $0.40 / 100만 호출
컴퓨팅 시간:
  - CPU: $0.0000100 / GHz-초
  - 메모리: $0.0000025 / GB-초

무료 티어 (항상 무료):
- 200만 호출/월
- 40만 GB-초, 20만 GHz-초
- 5GB 네트워크 이그레스
```

### 7.3 비용 최적화 팁

1. **적절한 메모리 할당**: 메모리 ↔ 성능 트레이드오프 테스트
2. **ARM 아키텍처 사용** (AWS): 20% 저렴
3. **Provisioned Concurrency 최소화**: 필요한 만큼만
4. **비동기 호출 활용**: API Gateway보다 직접 호출이 저렴
5. **코드 최적화**: 실행 시간 단축

---

## 8. 로컬 개발 및 테스트

### 8.1 AWS SAM (Serverless Application Model)

```bash
# SAM CLI 설치
pip install aws-sam-cli

# 프로젝트 초기화
sam init

# 로컬 테스트
sam local invoke MyFunction --event events/event.json

# 로컬 API 실행
sam local start-api

# 배포
sam build
sam deploy --guided
```

**template.yaml 예시:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  HelloFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.lambda_handler
      Runtime: python3.12
      Events:
        Api:
          Type: Api
          Properties:
            Path: /hello
            Method: get
```

### 8.2 GCP Functions Framework

```bash
# Functions Framework 설치
pip install functions-framework

# 로컬 실행
functions-framework --target=hello_http --debug

# 다른 터미널에서 테스트
curl http://localhost:8080
```

---

## 9. 실습: 이미지 리사이즈 함수

### 9.1 AWS Lambda (S3 트리거)

```python
import boto3
from PIL import Image
import io

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # S3 이벤트에서 버킷과 키 추출
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # 원본 이미지 다운로드
    response = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(io.BytesIO(response['Body'].read()))

    # 리사이즈
    image.thumbnail((200, 200))

    # 썸네일 업로드
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_key = f"thumbnails/{key}"
    s3.put_object(Bucket=bucket, Key=thumb_key, Body=buffer)

    return {'statusCode': 200, 'body': f'Thumbnail created: {thumb_key}'}
```

### 9.2 GCP Cloud Functions (Cloud Storage 트리거)

```python
from google.cloud import storage
from PIL import Image
import io
import functions_framework

@functions_framework.cloud_event
def resize_image(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]

    # 썸네일 폴더의 이미지는 무시
    if file_name.startswith("thumbnails/"):
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # 원본 이미지 다운로드
    blob = bucket.blob(file_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))

    # 리사이즈
    image.thumbnail((200, 200))

    # 썸네일 업로드
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_blob = bucket.blob(f"thumbnails/{file_name}")
    thumb_blob.upload_from_file(buffer, content_type='image/jpeg')

    print(f"Thumbnail created: thumbnails/{file_name}")
```

---

## 10. 다음 단계

- [06_Container_Services.md](./06_Container_Services.md) - 컨테이너 서비스
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - 데이터베이스 연동

---

## 연습 문제

### 연습 문제 1: 서버리스(Serverless) vs VM 트레이드오프 분석

각 사용 사례에 대해 서버리스(Lambda/Cloud Functions)와 전통적인 VM(EC2/Compute Engine) 중 어느 것이 더 적합한지 판단하고 이유를 설명하세요.

1. 하루에 약 500건의 요청을 받으며 각 요청이 200ms 미만이 걸리는 REST API 엔드포인트
2. 4K 동영상 파일을 작업당 최대 30분 처리하는 비디오 트랜스코딩(transcoding) 서비스
3. 새 주문이 들어올 때마다 큐(queue)에서 메시지를 처리하는 이벤트 기반 파이프라인
4. 6시간 동안 지속적으로 실행되는 장시간 머신러닝 학습 작업

<details>
<summary>정답 보기</summary>

1. **서버리스** — 하루 500건은 매우 낮은 트래픽입니다. VM은 99.9%의 시간 동안 유휴 상태가 됩니다. Lambda 무료 티어는 월 100만 건을 지원하므로 이 워크로드는 비용이 사실상 $0입니다. 200ms 미만의 실행 시간은 서버리스 제한 내에 충분히 들어옵니다.

2. **VM** — AWS Lambda의 최대 실행 시간은 15분이고, GCP Cloud Functions(1세대)는 9분입니다. 30분짜리 트랜스코딩 작업은 어느 플랫폼에서도 실행할 수 없습니다. EC2 인스턴스 또는 컨테이너 기반 솔루션(ECS/GKE)을 사용하세요.

3. **서버리스** — 이벤트 기반 큐 처리는 서버리스의 전형적인 사용 사례입니다. Lambda와 Cloud Functions는 SQS/Pub/Sub와의 네이티브 통합을 제공합니다. 서버리스는 메시지 볼륨에 따라 자동으로 스케일링되고, 큐가 비면 0으로 축소됩니다.

4. **VM** — 6시간의 연속 실행은 모든 서버리스 시간 제한을 초과합니다. 장시간 ML 학습은 영구적인 상태와 전용 리소스가 필요합니다. EC2 GPU 인스턴스(p3/g4dn 패밀리) 또는 관리형 ML 플랫폼(SageMaker, Vertex AI)을 사용하세요.

</details>

### 연습 문제 2: 콜드 스타트(Cold Start) 원인과 완화

한 회사의 Lambda 함수가 사용자 인증을 처리합니다. 피크 시간에는 잘 작동하지만(~50ms 응답), 첫 아침 요청이 정기적으로 3~4초가 걸려 사용자 불만이 발생합니다.

1. 3~4초의 지연이 발생하는 원인은 무엇입니까? 메커니즘을 설명하세요.
2. 트레이드오프를 포함하여 두 가지 구체적인 완화 전략을 제안하세요.

<details>
<summary>정답 보기</summary>

1. **콜드 스타트(Cold Start)**입니다. 야간 비활동으로 Lambda 인스턴스가 최근 함수를 실행하지 않은 경우, AWS가 컨테이너를 재활용합니다. 첫 번째 호출에서 새 컨테이너를 시작하고, 언어 런타임(Python 인터프리터)을 초기화하고, 모든 의존성을 임포트(import)하고, 전역 초기화 코드를 실행해야 합니다. 대용량 인증 라이브러리(예: `boto3`, JWT 라이브러리)가 있는 Python 함수의 경우, 이 초기화가 2~4초가 걸릴 수 있습니다. 이후의 "웜(warm)" 호출은 초기화된 컨테이너를 재사용하여 50ms만 걸립니다.

2. **완화 전략**:

   **전략 A — 프로비저닝된 동시성(Provisioned Concurrency)**
   - N개의 사전 초기화된 컨테이너를 항상 웜 상태로 유지하도록 Lambda를 설정합니다.
   - 트레이드오프: 최대 N개의 동시 호출에 대해 콜드 스타트를 완전히 제거하지만, 요청이 없어도 프로비저닝된 동시성 시간에 대해 비용이 발생합니다. N에 비례하여 비용이 증가합니다.
   ```bash
   aws lambda put-provisioned-concurrency-config \
       --function-name my-auth-function \
       --qualifier prod \
       --provisioned-concurrent-executions 5
   ```

   **전략 B — CloudWatch Events를 이용한 스케줄된 웜업(Warm-up)**
   - 컨테이너를 따뜻하게 유지하기 위해 5분마다 함수를 ping하는 CloudWatch Events 규칙을 생성합니다.
   - 트레이드오프: 비용이 낮지만(몇 번의 웜업 ping은 무료 티어 내에서 해결), 소수의 컨테이너만 웜 상태로 유지합니다. 첫 아침 급증이 50개의 동시 실행을 필요로 한다면 대부분이 여전히 콜드 스타트됩니다. 단일 인스턴스 시나리오에서 잘 작동합니다.

</details>

### 연습 문제 3: Lambda 비용 추정

다음 특성을 가진 함수의 월별 AWS Lambda 비용을 추정하세요:
- 메모리: 512 MB
- 평균 실행 시간: 400ms
- 호출 횟수: 월 500만 건
- 아키텍처: x86

가격: 요청 100만 건당 $0.20, GB-초당 $0.0000166667. 무료 티어: 월 100만 건 요청 및 400,000 GB-초.

<details>
<summary>정답 보기</summary>

**1단계: 요청 비용**
- 총 호출 횟수: 5,000,000
- 무료 티어: 1,000,000
- 청구 가능 호출: 4,000,000
- 비용: 4,000,000 / 1,000,000 × $0.20 = **$0.80**

**2단계: 컴퓨팅 비용 (GB-초)**
- 호출당 GB-초: 0.5 GB × 0.4초 = 0.2 GB-초
- 총 GB-초: 5,000,000 × 0.2 = 1,000,000 GB-초
- 무료 티어: 400,000 GB-초
- 청구 가능 GB-초: 600,000 GB-초
- 비용: 600,000 × $0.0000166667 = **$10.00**

**월별 총 비용: $0.80 + $10.00 = $10.80**

**최적화 인사이트**: ARM(Graviton2) 아키텍처로 전환하면 컴퓨팅 비용이 20% 감소하여 컴퓨팅 비용이 ~$8.00, 총 비용이 ~$8.80/월로 줄어들며 — 이 함수 하나만으로 월 ~$2, 연간 ~$24를 절약합니다.

</details>

### 연습 문제 4: 이벤트 소스(Event Source) 설정

다음 시스템을 구축 중입니다: (1) 사용자가 S3 버킷에 이미지를 업로드하면, (2) Lambda 함수가 자동으로 각 업로드 이미지의 썸네일을 생성합니다.

설정해야 할 이벤트 소스 매핑을 설명하고, 무한 루프(infinite loop)를 방지하기 위한 중요한 설정 사항을 찾아보세요.

<details>
<summary>정답 보기</summary>

**이벤트 소스 설정**:
- 소스 버킷에 `s3:ObjectCreated:*` 이벤트에 대해 Lambda를 트리거하는 **S3 이벤트 알림(S3 Event Notification)**을 설정합니다.
- 새 객체가 버킷에 업로드될 때마다 버킷 이름과 객체 키가 포함된 이벤트와 함께 Lambda가 호출됩니다.

**무한 루프 문제**: Lambda 함수가 생성된 썸네일을 동일한 S3 버킷(하위 폴더나 다른 버킷이 아닌)에 다시 쓰면, 썸네일 업로드 자체가 또 다른 Lambda 호출을 트리거하고, 이것이 또 다른 썸네일을 생성하고, 또 다른 호출을 트리거하는 무한 재귀 루프가 발생하여 예상치 못한 요금이 청구됩니다.

**해결 방법**:
1. **별도의 출력 버킷**: 썸네일을 다른 S3 버킷(예: `my-bucket-thumbnails`)에 씁니다. 소스 버킷에만 트리거를 설정합니다.
2. **접두사/접미사 필터링**: 특정 접두사(예: `uploads/`) 하의 객체에 대해서만 S3 트리거가 발동하도록 설정하고 썸네일 접두사에는 적용하지 않습니다. 썸네일은 `thumbnails/`에 씁니다. Lambda 코드에 다음과 같은 가드(guard)를 추가합니다:
   ```python
   if key.startswith("thumbnails/"):
       return  # 무한 루프 방지를 위해 썸네일 파일 건너뜀
   ```
   심층 방어(defense in depth)를 위해 코드 수준 및 S3 알림 수준 필터링을 모두 권장합니다.

</details>

### 연습 문제 5: 서버리스 아키텍처 설계

이커머스(e-commerce) 주문 처리 시스템을 위한 서버리스 아키텍처를 설계하세요. 고객이 주문을 하면:
1. 주문이 데이터베이스에 저장됩니다.
2. 재고 확인이 수행됩니다.
3. 확인 이메일이 발송됩니다.
4. 분석 데이터가 기록됩니다.

사용할 AWS 서비스, 서비스 간 연결 방식, 이 워크플로우에 서버리스가 적합한 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**아키텍처**:

```
고객 → API Gateway → Lambda (주문 핸들러)
                               │
                               ├── DynamoDB (주문 저장)
                               │
                               └── SNS 토픽 "order-created"
                                       │
                         ┌─────────────┼──────────────┐
                         ▼             ▼              ▼
                    Lambda          Lambda         Lambda
                   (재고)          (이메일)        (분석)
                      │               │              │
                  DynamoDB         SES/SNS       Kinesis/S3
               (재고 DB)        (이메일 발송)    (데이터 레이크)
```

**서비스 역할**:
- **API Gateway**: 주문 요청을 수신하고 주문 핸들러 Lambda를 트리거하는 HTTP 엔드포인트
- **Lambda (주문 핸들러)**: 주문을 DynamoDB에 저장한 후 SNS에 `order-created` 이벤트를 발행합니다. 고객에게 응답을 반환합니다.
- **SNS 토픽**: 이벤트를 여러 다운스트림 Lambda에 동시에 전달하는 팬아웃(fan-out) 허브 (병렬 처리)
- **Lambda (재고)**: DynamoDB의 재고를 확인하고 차감합니다.
- **Lambda (이메일)**: SES를 통해 확인 이메일을 발송합니다.
- **Lambda (분석)**: 이벤트 데이터를 Kinesis Data Firehose → S3에 기록하여 나중에 분석합니다.

**서버리스가 적합한 이유**:
- 각 단계는 이벤트 기반이며 단시간 실행됩니다(15분 미만).
- 트래픽이 버스티(bursty)합니다(예측 불가능한 주문량). 서버리스는 자동으로 0으로 줄었다가 다시 확장됩니다.
- SNS를 통한 느슨한 결합(loose coupling)으로 각 함수가 다른 함수에 영향을 주지 않고 독립적으로 실패할 수 있습니다.
- 비용이 실제 주문 량에 비례합니다 — 유휴 용량 비용이 없습니다.

</details>

---

## 참고 자료

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [GCP Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [AWS SAM](https://aws.amazon.com/serverless/sam/)
- [Functions Framework](https://github.com/GoogleCloudPlatform/functions-framework)
