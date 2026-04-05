# 객체 스토리지 (S3 / Cloud Storage)

**이전**: [컨테이너 서비스](./06_Container_Services.md) | **다음**: [블록 및 파일 스토리지](./08_Block_and_File_Storage.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 객체 스토리지 모델과 블록 및 파일 스토리지와의 차이점을 설명할 수 있습니다
2. AWS S3와 GCP Cloud Storage의 기능, 스토리지 클래스, 가격을 비교할 수 있습니다
3. 버킷을 생성하고, 객체를 업로드하며, 접근 정책을 구성할 수 있습니다
4. 수명 주기 규칙(Lifecycle Rules)을 구현하여 스토리지 클래스 간 객체를 자동으로 전환할 수 있습니다
5. 데이터 내구성을 위한 버전 관리(Versioning)와 크로스 리전 복제(Cross-Region Replication)를 구성할 수 있습니다
6. 저장된 객체에 암호화(서버 측 및 클라이언트 측)와 접근 제어를 적용할 수 있습니다
7. 적절한 스토리지 티어를 선택하여 비용 효율적인 스토리지 전략을 설계할 수 있습니다

---

객체 스토리지는 클라우드 데이터 관리의 근간입니다. 애플리케이션 에셋과 로그 아카이브부터 데이터 레이크 파일과 백업 스냅샷까지, 사실상 모든 클라우드 워크로드는 어느 시점에 객체 스토리지에 데이터를 저장합니다. 사실상 무제한에 가까운 용량, 높은 내구성, 사용한 만큼 지불하는 가격 정책 덕분에 객체 스토리지는 전 산업에 걸쳐 가장 널리 사용되는 클라우드 서비스 중 하나입니다.

## 1. 객체 스토리지 개요

### 1.1 객체 스토리지란?

객체 스토리지는 데이터를 객체 단위로 저장하는 스토리지 아키텍처입니다.

**객체 구성요소:**
- **데이터**: 실제 파일 내용
- **메타데이터**: 파일 정보 (생성일, 크기, 커스텀 속성)
- **고유 식별자**: 객체를 찾기 위한 키

### 1.2 서비스 비교

| 항목 | AWS S3 | GCP Cloud Storage |
|------|--------|------------------|
| 서비스명 | Simple Storage Service | Cloud Storage |
| 컨테이너 단위 | Bucket | Bucket |
| 최대 객체 크기 | 5TB | 5TB |
| 멀티파트 업로드 | 지원 (5MB-5GB 파트) | 지원 (복합 업로드) |
| 버전 관리 | Versioning | Object Versioning |
| 수명 주기 | Lifecycle Rules | Lifecycle Management |
| 암호화 | SSE-S3, SSE-KMS, SSE-C | Google-managed, CMEK, CSEK |

---

## 2. 스토리지 클래스

### 2.1 AWS S3 스토리지 클래스

| 클래스 | 용도 | 가용성 | 최소 저장 기간 |
|--------|------|--------|---------------|
| **S3 Standard** | 자주 액세스 | 99.99% | - |
| **S3 Intelligent-Tiering** | 액세스 패턴 불명 | 99.9% | - |
| **S3 Standard-IA** | 드문 액세스 | 99.9% | 30일 |
| **S3 One Zone-IA** | 드문 액세스 (단일 AZ) | 99.5% | 30일 |
| **S3 Glacier Instant** | 아카이브 (즉시 액세스) | 99.9% | 90일 |
| **S3 Glacier Flexible** | 아카이브 (분~시간) | 99.99% | 90일 |
| **S3 Glacier Deep Archive** | 장기 아카이브 | 99.99% | 180일 |

### 2.2 GCP Cloud Storage 클래스

| 클래스 | 용도 | 가용성 SLA | 최소 저장 기간 |
|--------|------|-----------|---------------|
| **Standard** | 자주 액세스 | 99.95% (리전) | - |
| **Nearline** | 월 1회 미만 액세스 | 99.9% | 30일 |
| **Coldline** | 분기 1회 미만 액세스 | 99.9% | 90일 |
| **Archive** | 연 1회 미만 액세스 | 99.9% | 365일 |

### 2.3 비용 비교 (서울 리전 기준)

| 클래스 | S3 ($/GB/월) | GCS ($/GB/월) |
|--------|-------------|---------------|
| 표준 | $0.025 | $0.023 |
| 드문 액세스 | $0.0138 | $0.016 (Nearline) |
| 아카이브 | $0.005 (Glacier) | $0.0025 (Archive) |

*가격은 변동될 수 있음*

---

## 3. 버킷 생성 및 관리

### 3.1 AWS S3 버킷

```bash
# 버킷 생성
aws s3 mb s3://my-unique-bucket-name-2024 --region ap-northeast-2

# 버킷 목록 조회
aws s3 ls

# 버킷 내용 조회
aws s3 ls s3://my-bucket/

# 버킷 삭제 (비어있어야 함)
aws s3 rb s3://my-bucket

# 버킷 삭제 (내용 포함)
aws s3 rb s3://my-bucket --force
```

**버킷 이름 규칙:**
- 전역적으로 고유해야 함
- 3-63자
- 소문자, 숫자, 하이픈만 사용
- 문자 또는 숫자로 시작/끝

### 3.2 GCP Cloud Storage 버킷

```bash
# 버킷 생성
gsutil mb -l asia-northeast3 gs://my-unique-bucket-name-2024

# 또는 gcloud 사용
gcloud storage buckets create gs://my-bucket \
    --location=asia-northeast3

# 버킷 목록 조회
gsutil ls
# 또는
gcloud storage buckets list

# 버킷 내용 조회
gsutil ls gs://my-bucket/

# 버킷 삭제
gsutil rb gs://my-bucket

# 버킷 삭제 (내용 포함)
gsutil rm -r gs://my-bucket
```

---

## 4. 객체 업로드/다운로드

### 4.1 AWS S3 객체 작업

```bash
# 단일 파일 업로드
aws s3 cp myfile.txt s3://my-bucket/

# 폴더 업로드 (재귀)
aws s3 cp ./local-folder s3://my-bucket/remote-folder --recursive

# 파일 다운로드
aws s3 cp s3://my-bucket/myfile.txt ./

# 폴더 다운로드
aws s3 cp s3://my-bucket/folder ./local-folder --recursive

# 동기화 (변경된 파일만)
aws s3 sync ./local-folder s3://my-bucket/folder
aws s3 sync s3://my-bucket/folder ./local-folder

# 파일 삭제
aws s3 rm s3://my-bucket/myfile.txt

# 폴더 삭제
aws s3 rm s3://my-bucket/folder --recursive

# 파일 이동
aws s3 mv s3://my-bucket/file1.txt s3://my-bucket/folder/file1.txt

# 파일 복사
aws s3 cp s3://source-bucket/file.txt s3://dest-bucket/file.txt
```

### 4.2 GCP Cloud Storage 객체 작업

```bash
# 단일 파일 업로드
gsutil cp myfile.txt gs://my-bucket/

# 또는 gcloud 사용
gcloud storage cp myfile.txt gs://my-bucket/

# 폴더 업로드 (재귀)
gsutil cp -r ./local-folder gs://my-bucket/

# 파일 다운로드
gsutil cp gs://my-bucket/myfile.txt ./

# 폴더 다운로드
gsutil cp -r gs://my-bucket/folder ./

# 동기화
gsutil rsync -r ./local-folder gs://my-bucket/folder

# 파일 삭제
gsutil rm gs://my-bucket/myfile.txt

# 폴더 삭제
gsutil rm -r gs://my-bucket/folder

# 파일 이동
gsutil mv gs://my-bucket/file1.txt gs://my-bucket/folder/

# 파일 복사
gsutil cp gs://source-bucket/file.txt gs://dest-bucket/
```

### 4.3 대용량 파일 업로드

**AWS S3 멀티파트 업로드:**
```bash
# AWS CLI는 자동으로 멀티파트 업로드 사용 (8MB 이상)
aws s3 cp large-file.zip s3://my-bucket/ \
    --expected-size 10737418240  # 10GB

# 멀티파트 설정 조정
aws configure set s3.multipart_threshold 64MB
aws configure set s3.multipart_chunksize 16MB
```

**GCP 복합 업로드:**
```bash
# gsutil은 자동으로 복합 업로드 사용 (150MB 이상)
gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
    cp large-file.zip gs://my-bucket/
```

---

## 5. 수명 주기 관리

### 5.1 AWS S3 Lifecycle

```json
{
    "Rules": [
        {
            "ID": "Move to IA after 30 days",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": 365
            }
        },
        {
            "ID": "Delete old versions",
            "Status": "Enabled",
            "Filter": {},
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 30
            }
        }
    ]
}
```

```bash
# 수명 주기 정책 적용
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json

# 수명 주기 정책 조회
aws s3api get-bucket-lifecycle-configuration --bucket my-bucket
```

### 5.2 GCP Lifecycle Management

```json
{
    "lifecycle": {
        "rule": [
            {
                "action": {
                    "type": "SetStorageClass",
                    "storageClass": "NEARLINE"
                },
                "condition": {
                    "age": 30,
                    "matchesPrefix": ["logs/"]
                }
            },
            {
                "action": {
                    "type": "SetStorageClass",
                    "storageClass": "COLDLINE"
                },
                "condition": {
                    "age": 90
                }
            },
            {
                "action": {
                    "type": "Delete"
                },
                "condition": {
                    "age": 365
                }
            }
        ]
    }
}
```

```bash
# 수명 주기 정책 적용
gsutil lifecycle set lifecycle.json gs://my-bucket

# 수명 주기 정책 조회
gsutil lifecycle get gs://my-bucket
```

---

## 6. 접근 제어

### 6.1 AWS S3 접근 제어

**버킷 정책:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicRead",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        }
    ]
}
```

```bash
# 버킷 정책 적용
aws s3api put-bucket-policy \
    --bucket my-bucket \
    --policy file://bucket-policy.json

# 퍼블릭 액세스 차단 (권장)
aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**미리 서명된 URL (Presigned URL):**
```bash
# 다운로드 URL 생성 (1시간 유효)
aws s3 presign s3://my-bucket/private-file.pdf --expires-in 3600

# 업로드 URL 생성
aws s3 presign s3://my-bucket/uploads/file.txt --expires-in 3600
```

### 6.2 GCP Cloud Storage 접근 제어

**IAM 정책:**
```bash
# 사용자에게 버킷 접근 권한 부여
gsutil iam ch user:user@example.com:objectViewer gs://my-bucket

# 모든 사용자에게 읽기 권한 (퍼블릭)
gsutil iam ch allUsers:objectViewer gs://my-bucket
```

**균일 버킷 수준 액세스 (권장):**
```bash
# 균일 액세스 활성화
gsutil uniformbucketlevelaccess set on gs://my-bucket
```

**서명된 URL:**
```bash
# 다운로드 URL 생성 (1시간 유효)
gsutil signurl -d 1h service-account.json gs://my-bucket/private-file.pdf

# gcloud 사용
gcloud storage sign-url gs://my-bucket/file.pdf \
    --private-key-file=key.json \
    --duration=1h
```

---

## 7. 정적 웹사이트 호스팅

### 7.1 AWS S3 정적 호스팅

```bash
# 1. 정적 웹사이트 호스팅 활성화
aws s3 website s3://my-bucket/ \
    --index-document index.html \
    --error-document error.html

# 2. 퍼블릭 액세스 허용 (블록 해제)
aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# 3. 버킷 정책 (퍼블릭 읽기)
aws s3api put-bucket-policy --bucket my-bucket --policy '{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-bucket/*"
    }]
}'

# 4. 파일 업로드
aws s3 sync ./website s3://my-bucket/

# 웹사이트 URL: http://my-bucket.s3-website.ap-northeast-2.amazonaws.com
```

### 7.2 GCP Cloud Storage 정적 호스팅

```bash
# 1. 버킷 생성 (도메인 이름과 일치하면 커스텀 도메인 가능)
gsutil mb -l asia-northeast3 gs://www.example.com

# 2. 웹사이트 설정
gsutil web set -m index.html -e 404.html gs://my-bucket

# 3. 퍼블릭 액세스 허용
gsutil iam ch allUsers:objectViewer gs://my-bucket

# 4. 파일 업로드
gsutil cp -r ./website/* gs://my-bucket/

# 웹사이트 URL: https://storage.googleapis.com/my-bucket/index.html
# 로드 밸런서를 통해 커스텀 도메인 설정 가능
```

---

## 8. 버전 관리

### 8.1 AWS S3 Versioning

```bash
# 버전 관리 활성화
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled

# 버전 관리 상태 확인
aws s3api get-bucket-versioning --bucket my-bucket

# 모든 버전 조회
aws s3api list-object-versions --bucket my-bucket

# 특정 버전 다운로드
aws s3api get-object \
    --bucket my-bucket \
    --key myfile.txt \
    --version-id "abc123" \
    myfile-old.txt

# 특정 버전 삭제
aws s3api delete-object \
    --bucket my-bucket \
    --key myfile.txt \
    --version-id "abc123"
```

### 8.2 GCP Object Versioning

```bash
# 버전 관리 활성화
gsutil versioning set on gs://my-bucket

# 버전 관리 상태 확인
gsutil versioning get gs://my-bucket

# 모든 버전 조회
gsutil ls -a gs://my-bucket/

# 특정 버전 다운로드
gsutil cp gs://my-bucket/myfile.txt#1234567890123456 ./

# 특정 버전 삭제
gsutil rm gs://my-bucket/myfile.txt#1234567890123456
```

---

## 9. 크로스 리전 복제

### 9.1 AWS S3 Cross-Region Replication

```bash
# 1. 소스 버킷 버전 관리 활성화
aws s3api put-bucket-versioning \
    --bucket source-bucket \
    --versioning-configuration Status=Enabled

# 2. 대상 버킷 생성 및 버전 관리 활성화
aws s3 mb s3://dest-bucket --region eu-west-1
aws s3api put-bucket-versioning \
    --bucket dest-bucket \
    --versioning-configuration Status=Enabled

# 3. 복제 규칙 설정
aws s3api put-bucket-replication \
    --bucket source-bucket \
    --replication-configuration '{
        "Role": "arn:aws:iam::123456789012:role/s3-replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Priority": 1,
            "DeleteMarkerReplication": {"Status": "Disabled"},
            "Filter": {},
            "Destination": {
                "Bucket": "arn:aws:s3:::dest-bucket"
            }
        }]
    }'
```

### 9.2 GCP Dual/Multi-Region Buckets

```bash
# 듀얼 리전 버킷 생성
gsutil mb -l asia1 gs://my-dual-region-bucket

# 또는 멀티 리전 버킷
gsutil mb -l asia gs://my-multi-region-bucket

# 리전 간 복사 (수동)
gsutil cp -r gs://source-bucket/* gs://dest-bucket/
```

---

## 10. SDK 사용 예시

### 10.1 Python (boto3 / google-cloud-storage)

**AWS S3 (boto3):**
```python
import boto3

s3 = boto3.client('s3')

# 업로드
s3.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

# 다운로드
s3.download_file('my-bucket', 'remote_file.txt', 'local_file.txt')

# 객체 목록
response = s3.list_objects_v2(Bucket='my-bucket', Prefix='folder/')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Presigned URL 생성
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'file.txt'},
    ExpiresIn=3600
)
```

**GCP Cloud Storage:**
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')

# 업로드
blob = bucket.blob('remote_file.txt')
blob.upload_from_filename('local_file.txt')

# 다운로드
blob = bucket.blob('remote_file.txt')
blob.download_to_filename('local_file.txt')

# 객체 목록
blobs = client.list_blobs('my-bucket', prefix='folder/')
for blob in blobs:
    print(blob.name)

# Signed URL 생성
from datetime import timedelta
url = blob.generate_signed_url(expiration=timedelta(hours=1))
```

---

## 11. 다음 단계

- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - 블록 스토리지
- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - CDN과 함께 사용

---

## 연습 문제

### 연습 문제 1: 스토리지 클래스(Storage Class) 선택

미디어 회사가 다음 카테고리의 파일을 S3에 저장합니다. 각각에 가장 비용 효율적인 스토리지 클래스를 선택하고 이유를 설명하세요:

1. 활성 소셜 미디어 피드의 썸네일 이미지 — 하루에 수천 번 접근
2. 크리에이터가 업로드한 원본 동영상 — 처음 48시간은 자주 접근, 이후 거의 접근 안 함
3. 규정 준수를 위해 7년간 보관해야 하지만 거의 읽히지 않는 분기별 재무 보고서
4. 시스템 로그 파일 — 첫 30일 내에 디버깅 시 가끔 접근, 이후 절대 접근 안 함

<details>
<summary>정답 보기</summary>

1. **S3 Standard** — 고빈도 접근(하루 수천 회)은 Standard의 핵심 사용 사례입니다. 높은 스토리지 비용은 검색 수수료 없음으로 정당화됩니다. Intelligent-Tiering이나 IA를 사용하면 스토리지 절감분을 훨씬 초과하는 검색 수수료가 발생합니다.

2. **S3 Intelligent-Tiering** — 접근 패턴이 48시간 후 극적으로 변화하지만(빈번 → 드문), 각 파일의 정확한 패턴은 다를 수 있습니다. Intelligent-Tiering은 검색 수수료나 최소 보관 기간 페널티 없이 자동으로 객체를 빈번/드문 티어(tier) 간에 이동시켜 패턴이 변하는 경우에 이상적입니다.

3. **S3 Glacier Deep Archive** — 연간 1회 미만으로 접근하고 7년간 보관하는 객체는 Deep Archive의 전형적인 사용 사례입니다. GB당 월 $0.00099로, Standard보다 25배 저렴합니다. 180일 최소 보관 기간은 7년 보관 정책으로 쉽게 충족됩니다. 12시간의 검색 시간은 드물게 접근하는 규정 준수 문서에 허용됩니다.

4. **S3 Standard-IA** — 처음 30일 내 가끔 접근하는 로그는 Standard 대비 낮은 스토리지 비용의 Standard-IA가 유리합니다. 30일 후에는 라이프사이클 규칙으로 Glacier Flexible Retrieval이나 Deep Archive로 전환합니다. 30일 최소 보관 기간이 활성 기간과 일치합니다.

</details>

### 연습 문제 2: 라이프사이클 정책(Lifecycle Policy) 설계

다음 요건에 따라 애플리케이션 로그 파일에 대한 S3 라이프사이클 정책을 설계하세요:
- 처음 7일: 활발한 분석
- 7일~30일: 가끔 디버깅용으로 접근 가능
- 30일~365일: 규정 준수용 보관, 거의 접근 안 함
- 1년 후: 삭제 가능

라이프사이클 규칙 설정을 JSON 또는 일반 텍스트로 작성하세요.

<details>
<summary>정답 보기</summary>

```json
{
  "Rules": [
    {
      "ID": "log-lifecycle",
      "Status": "Enabled",
      "Filter": {"Prefix": "logs/"},
      "Transitions": [
        {
          "Days": 7,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 30,
          "StorageClass": "GLACIER_FLEXIBLE_RETRIEVAL"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    }
  ]
}
```

**전환 설명**:
- **0~7일**: S3 Standard — 활발한 분석에는 빠르고 무료의 검색이 필요
- **7~30일**: S3 Standard-IA — 가끔 디버깅 접근; 낮은 스토리지 비용, 요청별 검색 수수료는 허용 가능
- **30~365일**: S3 Glacier Flexible Retrieval — 매우 낮은 비용으로 규정 준수 보관; 드문 접근에 1~12시간 검색 시간 괜찮음
- **365일+**: 객체가 만료되어 자동 삭제, 스토리지 비용 제거

**AWS CLI 적용**:
```bash
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-log-bucket \
    --lifecycle-configuration file://lifecycle.json
```

</details>

### 연습 문제 3: 버킷 버전 관리(Versioning)와 퍼블릭 접근 차단

개발팀이 프로덕션 설정 파일을 S3 버킷에 저장합니다. 다음을 원합니다:
1. 설정 파일의 실수로 인한 삭제 방지
2. 버킷이 절대 공개적으로 접근 가능하지 않도록 보장

두 요건을 구현하는 AWS CLI 명령어를 제공하세요.

<details>
<summary>정답 보기</summary>

```bash
# 1. 버킷에 버전 관리 활성화
# 버전 관리 활성화 시, 삭제된 객체는 삭제 마커(delete marker)가 생성되고
# (실제로 제거되지 않음), 덮어쓴 파일은 이전 버전을 유지합니다.
aws s3api put-bucket-versioning \
    --bucket my-config-bucket \
    --versioning-configuration Status=Enabled

# 2. 버킷 수준에서 모든 퍼블릭 접근 차단
aws s3api put-public-access-block \
    --bucket my-config-bucket \
    --public-access-block-configuration \
        BlockPublicAcls=true,\
        IgnorePublicAcls=true,\
        BlockPublicPolicy=true,\
        RestrictPublicBuckets=true
```

**추가 보호** — 승인된 사용자도 영구 삭제를 할 수 없도록 MFA Delete 요건 추가:
```bash
aws s3api put-bucket-versioning \
    --bucket my-config-bucket \
    --versioning-configuration Status=Enabled,MFADelete=Enabled \
    --mfa "arn:aws:iam::ACCOUNT_ID:mfa/USER_DEVICE CURRENT_CODE"
```

**버전 관리의 삭제 효과**:
- `aws s3 rm s3://my-config-bucket/prod.yaml` — 삭제 마커 생성; 객체는 숨겨지지만 복구 가능
- 영구 삭제하려면 특정 버전 ID를 명시적으로 삭제해야 합니다

</details>

### 연습 문제 4: 사전 서명된 URL(Pre-Signed URL) 사용 사례

애플리케이션이 버킷을 공개하지 않고 사용자가 프라이빗 S3 객체(`reports/q3-summary.pdf`)를 정확히 1시간 동안 다운로드할 수 있도록 해야 합니다. 사전 서명된 URL을 생성하는 AWS CLI 명령어를 작성하고 작동 원리를 설명하세요.

<details>
<summary>정답 보기</summary>

```bash
aws s3 presign s3://my-reports-bucket/reports/q3-summary.pdf \
    --expires-in 3600
```

이 명령어는 다음과 같은 URL을 생성합니다:
```
https://my-reports-bucket.s3.amazonaws.com/reports/q3-summary.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...&X-Amz-Expires=3600&X-Amz-Signature=...
```

**작동 원리**:
1. AWS는 생성한 IAM 엔터티(사용자)의 자격 증명을 사용하여 URL에 서명합니다.
2. 서명은 버킷, 객체 키, 만료 시간, 자격 증명을 인코딩합니다.
3. URL을 가진 누구든지 HTTP GET 요청으로 객체에 접근할 수 있습니다 — AWS 자격 증명 불필요
4. 3,600초(1시간) 후 서명이 무효화되어 URL이 403 Forbidden 오류를 반환합니다.

**보안 고려 사항**: 사전 서명된 URL은 생성한 IAM 엔터티의 권한을 상속합니다. URL이 만료되기 전에 생성 역할이 삭제되거나 S3 읽기 권한을 잃으면 URL도 작동을 멈춥니다.

</details>

### 연습 문제 5: 교차 리전 복제(Cross-Region Replication) 설정

회사가 `ap-northeast-2`(서울)에 기본 S3 버킷을 가지고 있으며 재해 복구를 위해 모든 새 객체를 `us-east-1`(버지니아)에 복제해야 합니다. S3 교차 리전 복제(CRR, Cross-Region Replication)를 설정하는 데 필요한 전제 조건과 주요 단계를 설명하세요.

<details>
<summary>정답 보기</summary>

**전제 조건**:
1. **소스 버킷과 대상 버킷 모두에 버전 관리가 활성화되어야 합니다** — CRR은 양쪽 모두 버전 관리가 필요합니다.
2. **IAM 역할(role)** — 소스 버킷에서 읽고 대상 버킷에 쓸 수 있는 S3 권한을 부여하는 IAM 역할이 필요합니다.

**단계**:

```bash
# 1단계: 소스 버킷(서울)에 버전 관리 활성화
aws s3api put-bucket-versioning \
    --bucket source-bucket-seoul \
    --region ap-northeast-2 \
    --versioning-configuration Status=Enabled

# 2단계: us-east-1에 대상 버킷 생성 및 버전 관리 활성화
aws s3api create-bucket \
    --bucket destination-bucket-virginia \
    --region us-east-1 \
    --create-bucket-configuration LocationConstraint=us-east-1

aws s3api put-bucket-versioning \
    --bucket destination-bucket-virginia \
    --region us-east-1 \
    --versioning-configuration Status=Enabled

# 3단계: 복제 규칙 설정
aws s3api put-bucket-replication \
    --bucket source-bucket-seoul \
    --region ap-northeast-2 \
    --replication-configuration '{
        "Role": "arn:aws:iam::ACCOUNT_ID:role/s3-replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Destination": {
                "Bucket": "arn:aws:s3:::destination-bucket-virginia"
            }
        }]
    }'
```

**중요 사항**:
- CRR은 규칙 설정 이후에 작성된 새 객체만 복제합니다. 기존 객체는 자동으로 복제되지 않으며, **S3 Batch Operations**를 사용해야 합니다.
- 삭제 마커(delete marker)는 기본적으로 복제되지 않습니다(설정 가능).
- 서울에서 버지니아로의 데이터 전송은 이그레스(egress) 요금이 발생합니다.

</details>

---

## 참고 자료

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [GCP Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [S3 Pricing](https://aws.amazon.com/s3/pricing/)
- [Cloud Storage Pricing](https://cloud.google.com/storage/pricing)
