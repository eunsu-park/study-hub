# CLI & SDK

**이전**: [보안 서비스](./14_Security_Services.md) | **다음**: [Infrastructure as Code](./16_Infrastructure_as_Code.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 주요 운영체제에 AWS CLI와 gcloud CLI를 설치하고 구성한다
2. 서로 다른 환경에 대한 여러 프로필/구성을 인증하고 관리한다
3. 커맨드 라인에서 일반적인 클라우드 작업(리소스 생성, 목록 조회, 상세 조회, 삭제)을 수행한다
4. 출력 형식 지정과 필터링(`--query`, `--filter`)을 사용해 특정 정보를 추출한다
5. 클라우드 SDK(Boto3, Google Cloud Client Libraries)를 Python 스크립트에 통합한다
6. CLI 스크립트와 SDK 프로그램을 사용해 반복적인 클라우드 작업을 자동화한다

---

웹 콘솔은 탐색에는 적합하지만, 실제 클라우드 작업은 커맨드 라인과 코드에서 이루어집니다. CLI 도구와 SDK를 사용하면 반복 가능한 작업을 스크립트로 만들고, 클라우드 관리를 CI/CD 파이프라인에 통합하며, 클라우드 서비스와 프로그래밍 방식으로 상호작용하는 애플리케이션을 구축할 수 있습니다. 이러한 도구에 익숙해지는 것이 콘솔 클릭커와 클라우드 엔지니어를 구분하는 차이입니다.

## 1. CLI 개요

### 1.1 AWS CLI vs gcloud CLI

| 항목 | AWS CLI | gcloud CLI |
|------|---------|------------|
| 설치 패키지 | awscli | google-cloud-sdk |
| 구성 명령 | aws configure | gcloud init |
| 프로필 | --profile | --configuration |
| 출력 형식 | json, text, table, yaml | json, text, yaml, csv |

---

## 2. AWS CLI

### 2.1 설치

```bash
# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# pip
pip install awscli

# 버전 확인
aws --version
```

### 2.2 구성

```bash
# 기본 구성
aws configure
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region name: ap-northeast-2
# Default output format: json

# 프로필 추가
aws configure --profile production
aws configure --profile development

# 프로필 목록
aws configure list-profiles

# 환경 변수로 설정
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI...
export AWS_DEFAULT_REGION=ap-northeast-2
export AWS_PROFILE=production
```

**~/.aws/credentials:**
```ini
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI...

[production]
aws_access_key_id = AKIAI44QH8DHBEXAMPLE
aws_secret_access_key = je7MtGbClwBF/2Zp9Utk...
```

**~/.aws/config:**
```ini
[default]
region = ap-northeast-2
output = json

[profile production]
region = ap-northeast-1
output = table
```

### 2.3 주요 명령어

```bash
# EC2
aws ec2 describe-instances
aws ec2 run-instances --image-id ami-xxx --instance-type t3.micro
aws ec2 start-instances --instance-ids i-xxx
aws ec2 stop-instances --instance-ids i-xxx
aws ec2 terminate-instances --instance-ids i-xxx

# S3
aws s3 ls
aws s3 cp file.txt s3://bucket/
aws s3 sync ./folder s3://bucket/folder
aws s3 rm s3://bucket/file.txt

# IAM
aws iam list-users
aws iam create-user --user-name john
aws iam attach-user-policy --user-name john --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess

# Lambda
aws lambda list-functions
aws lambda invoke --function-name my-func output.json

# RDS
aws rds describe-db-instances
aws rds create-db-snapshot --db-instance-identifier mydb --db-snapshot-identifier mysnap
```

### 2.4 출력 필터링 (--query)

```bash
# JMESPath 쿼리 사용
aws ec2 describe-instances \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
    --output table

# 특정 태그로 필터
aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text

# 정렬
aws ec2 describe-instances \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)[*].[InstanceId,LaunchTime]'

# 조건 필터
aws ec2 describe-instances \
    --query 'Reservations[].Instances[?State.Name==`running`].InstanceId'
```

---

## 3. gcloud CLI

### 3.1 설치

```bash
# macOS
brew install --cask google-cloud-sdk

# 또는 직접 설치
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Linux
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# 버전 확인
gcloud --version
```

### 3.2 구성

```bash
# 초기화 (브라우저 인증)
gcloud init

# 로그인
gcloud auth login

# 서비스 계정 인증
gcloud auth activate-service-account --key-file=key.json

# 프로젝트 설정
gcloud config set project PROJECT_ID

# 리전/존 설정
gcloud config set compute/region asia-northeast3
gcloud config set compute/zone asia-northeast3-a

# 현재 구성 확인
gcloud config list

# 구성 프로필
gcloud config configurations create production
gcloud config configurations activate production
gcloud config configurations list
```

### 3.3 주요 명령어

```bash
# Compute Engine
gcloud compute instances list
gcloud compute instances create my-vm --machine-type=e2-medium
gcloud compute instances start my-vm
gcloud compute instances stop my-vm
gcloud compute instances delete my-vm
gcloud compute ssh my-vm

# Cloud Storage
gsutil ls
gsutil cp file.txt gs://bucket/
gsutil rsync -r ./folder gs://bucket/folder
gsutil rm gs://bucket/file.txt

# IAM
gcloud iam service-accounts list
gcloud iam service-accounts create my-sa
gcloud projects add-iam-policy-binding PROJECT --member=user:john@example.com --role=roles/viewer

# Cloud Functions
gcloud functions list
gcloud functions deploy my-func --runtime=python312 --trigger-http

# Cloud SQL
gcloud sql instances list
gcloud sql instances create mydb --database-version=MYSQL_8_0
```

### 3.4 출력 필터링 (--filter, --format)

```bash
# 필터링
gcloud compute instances list \
    --filter="status=RUNNING AND zone:asia-northeast3"

# 출력 형식
gcloud compute instances list \
    --format="table(name,zone.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)"

# JSON 출력
gcloud compute instances describe my-vm --format=json

# 특정 필드만
gcloud compute instances list \
    --format="value(name,networkInterfaces[0].accessConfigs[0].natIP)"

# CSV
gcloud compute instances list \
    --format="csv(name,zone,status)"
```

---

## 4. Python SDK

### 4.1 AWS SDK (boto3)

**설치:**
```bash
pip install boto3
```

**기본 사용:**
```python
import boto3

# 클라이언트 방식
ec2_client = boto3.client('ec2')
response = ec2_client.describe_instances()

# 리소스 방식 (고수준)
ec2 = boto3.resource('ec2')
instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)
for instance in instances:
    print(instance.id, instance.public_ip_address)
```

**서비스별 예시:**
```python
import boto3

# S3
s3 = boto3.client('s3')
s3.upload_file('file.txt', 'my-bucket', 'file.txt')
s3.download_file('my-bucket', 'file.txt', 'downloaded.txt')

# DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')
table.put_item(Item={'userId': '001', 'name': 'John'})
response = table.get_item(Key={'userId': '001'})

# Lambda
lambda_client = boto3.client('lambda')
response = lambda_client.invoke(
    FunctionName='my-function',
    Payload=json.dumps({'key': 'value'})
)

# SQS
sqs = boto3.client('sqs')
sqs.send_message(
    QueueUrl='https://sqs.../my-queue',
    MessageBody='Hello'
)

# Secrets Manager
secrets = boto3.client('secretsmanager')
secret = secrets.get_secret_value(SecretId='my-secret')
```

### 4.2 GCP SDK (google-cloud)

**설치:**
```bash
pip install google-cloud-storage
pip install google-cloud-compute
pip install google-cloud-firestore
# 필요한 라이브러리별 설치
```

**기본 사용:**
```python
from google.cloud import storage
from google.cloud import compute_v1

# 인증 (서비스 계정)
# export GOOGLE_APPLICATION_CREDENTIALS="key.json"

# Cloud Storage
client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('file.txt')
blob.upload_from_filename('file.txt')
blob.download_to_filename('downloaded.txt')

# Compute Engine
instance_client = compute_v1.InstancesClient()
instances = instance_client.list(project='my-project', zone='asia-northeast3-a')
for instance in instances:
    print(instance.name, instance.status)
```

**서비스별 예시:**
```python
# Firestore
from google.cloud import firestore

db = firestore.Client()
doc_ref = db.collection('users').document('001')
doc_ref.set({'name': 'John', 'age': 30})
doc = doc_ref.get()
print(doc.to_dict())

# Pub/Sub
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('project', 'topic')
publisher.publish(topic_path, b'Hello')

# Secret Manager
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/PROJECT/secrets/my-secret/versions/latest"
response = client.access_secret_version(request={"name": name})
secret = response.payload.data.decode("UTF-8")

# BigQuery
from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT * FROM `project.dataset.table` LIMIT 10"
results = client.query(query)
for row in results:
    print(row)
```

---

## 5. 자동화 스크립트 예시

### 5.1 리소스 정리 스크립트

**AWS - 미사용 EBS 볼륨 삭제:**
```python
import boto3

ec2 = boto3.client('ec2')

# 미사용 볼륨 찾기
volumes = ec2.describe_volumes(
    Filters=[{'Name': 'status', 'Values': ['available']}]
)

for vol in volumes['Volumes']:
    vol_id = vol['VolumeId']
    print(f"Deleting unused volume: {vol_id}")
    # ec2.delete_volume(VolumeId=vol_id)  # 주석 해제하여 실제 삭제
```

**GCP - 오래된 스냅샷 삭제:**
```python
from google.cloud import compute_v1
from datetime import datetime, timedelta

client = compute_v1.SnapshotsClient()
project = 'my-project'

snapshots = client.list(project=project)
cutoff = datetime.now() - timedelta(days=30)

for snapshot in snapshots:
    created = datetime.fromisoformat(snapshot.creation_timestamp.replace('Z', '+00:00'))
    if created < cutoff.replace(tzinfo=created.tzinfo):
        print(f"Deleting old snapshot: {snapshot.name}")
        # client.delete(project=project, snapshot=snapshot.name)
```

### 5.2 배포 스크립트

**AWS Lambda 배포:**
```python
import boto3
import zipfile
import os

def deploy_lambda(function_name, source_dir):
    # 코드 압축
    with zipfile.ZipFile('function.zip', 'w') as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                zf.write(os.path.join(root, file))

    # Lambda 업데이트
    lambda_client = boto3.client('lambda')
    with open('function.zip', 'rb') as f:
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=f.read()
        )

    print(f"Deployed {function_name}")

deploy_lambda('my-function', './src')
```

### 5.3 모니터링 스크립트

**AWS 인스턴스 상태 확인:**
```bash
#!/bin/bash
# check-instances.sh

instances=$(aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
    --output text)

while read -r instance_id state; do
    if [ "$state" != "running" ]; then
        echo "WARNING: $instance_id is $state"
        # 알림 전송
        aws sns publish \
            --topic-arn arn:aws:sns:...:alerts \
            --message "Instance $instance_id is $state"
    fi
done <<< "$instances"
```

---

## 6. 페이지네이션 처리

### 6.1 AWS CLI 페이지네이션

```bash
# 자동 페이지네이션
aws s3api list-objects-v2 --bucket my-bucket

# 수동 페이지네이션
aws s3api list-objects-v2 --bucket my-bucket --max-items 100

# 다음 페이지
aws s3api list-objects-v2 --bucket my-bucket --starting-token TOKEN
```

**boto3 Paginator:**
```python
import boto3

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket='my-bucket'):
    for obj in page.get('Contents', []):
        print(obj['Key'])
```

### 6.2 gcloud 페이지네이션

```bash
# 자동 페이지네이션
gcloud compute instances list

# 수동
gcloud compute instances list --limit=100 --page-token=TOKEN
```

**Python:**
```python
from google.cloud import storage

client = storage.Client()
blobs = client.list_blobs('my-bucket')  # 자동 페이지네이션

for blob in blobs:
    print(blob.name)
```

---

## 7. 에러 처리

### 7.1 boto3 에러 처리

```python
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    s3 = boto3.client('s3')
    s3.head_object(Bucket='my-bucket', Key='file.txt')
except NoCredentialsError:
    print("Credentials not found")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        print("Object not found")
    elif error_code == '403':
        print("Access denied")
    else:
        raise
```

### 7.2 GCP 에러 처리

```python
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden

try:
    client = storage.Client()
    bucket = client.get_bucket('my-bucket')
    blob = bucket.blob('file.txt')
    blob.download_to_filename('file.txt')
except NotFound:
    print("Bucket or object not found")
except Forbidden:
    print("Access denied")
```

---

## 8. 다음 단계

- [16_Infrastructure_as_Code.md](./16_Infrastructure_as_Code.md) - Terraform
- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - 모니터링

---

## 연습 문제

### 연습 문제 1: CLI 프로파일(Profile) 관리

여러분은 세 가지 AWS 환경(`dev`, `staging`, `production`)에서 작업합니다. 각 환경은 서로 다른 AWS 계정과 리전을 사용합니다. 이들 간에 전환할 수 있도록 AWS CLI를 구성하는 방법을 설명하고, 기본 프로파일을 영구적으로 변경하지 않고 `production` 계정의 모든 S3 버킷을 나열하는 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

**구성 단계:**

```bash
# 각 프로파일 설정
aws configure --profile dev
# dev 계정 키, 시크릿, 리전(ap-northeast-2), 형식(json) 입력

aws configure --profile staging
# staging 계정 자격 증명 입력

aws configure --profile production
# production 계정 자격 증명 입력

# 모든 프로파일 확인
aws configure list-profiles
```

**~/.aws/credentials** (결과):
```ini
[default]
aws_access_key_id = ...
aws_secret_access_key = ...

[dev]
aws_access_key_id = AKIA...DEV
aws_secret_access_key = ...

[staging]
aws_access_key_id = AKIA...STG
aws_secret_access_key = ...

[production]
aws_access_key_id = AKIA...PRD
aws_secret_access_key = ...
```

**기본값 변경 없이 production의 S3 버킷 나열:**
```bash
# 옵션 1: --profile 플래그
aws s3 ls --profile production

# 옵션 2: 환경 변수 (현재 쉘 세션에서 임시로)
export AWS_PROFILE=production
aws s3 ls

# 옵션 3: 명령어별 환경 변수
AWS_PROFILE=production aws s3 ls
```

모범 사례: 스크립트에서는 `--profile`을 사용하여 프로파일이 명시적으로 지정되고 쉘 상태에 의존하지 않도록 합니다.

</details>

---

### 연습 문제 2: JMESPath 쿼리 필터링

`Production` 환경 태그가 있는 모든 실행 중인 EC2 인스턴스의 보고서가 필요합니다. 인스턴스 ID, 인스턴스 유형, 퍼블릭 IP만 표시해야 합니다. `--query`와 `--output table`을 사용하는 AWS CLI 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

```bash
aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId, InstanceType, PublicIpAddress]' \
    --output table
```

JMESPath 표현식 설명:
- `Reservations[*]` — 모든 예약 그룹을 순회 (AWS는 인스턴스를 시작 요청별로 그룹화)
- `.Instances[*]` — 각 예약 내의 인스턴스를 순회
- `.[InstanceId, InstanceType, PublicIpAddress]` — 다중 값 배열 선택 (테이블 열이 됨)

**동등한 gcloud 명령어:**
```bash
gcloud compute instances list \
    --filter="status=RUNNING AND labels.environment=production" \
    --format="table(name,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
```

참고: gcloud는 `--filter`(서버측)와 `--format`(클라이언트측 포맷팅)을 사용하는 반면, AWS CLI는 `--filters`(서버측)와 `--query`(클라이언트측 JMESPath)를 사용합니다.

</details>

---

### 연습 문제 3: Boto3 자동화 스크립트

boto3를 사용하여 다음을 수행하는 Python 스크립트를 작성하세요:
1. 계정의 모든 S3 버킷 나열
2. 각 버킷에 대해 이름과 포함된 객체 수 출력
3. 버킷이 다른 리전에 있을 수 있는 경우 처리 (S3 클라이언트의 기본 리전으로 `us-east-1` 사용)

<details>
<summary>정답 보기</summary>

```python
import boto3
from botocore.exceptions import ClientError

def count_bucket_objects(s3_client, bucket_name):
    """대형 버킷 처리를 위해 paginator를 사용하여 객체 수 계산."""
    paginator = s3_client.get_paginator('list_objects_v2')
    count = 0
    try:
        for page in paginator.paginate(Bucket=bucket_name):
            count += page.get('KeyCount', 0)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            return None
        elif error_code in ('AccessDenied', '403'):
            return 'ACCESS_DENIED'
        else:
            raise
    return count

def main():
    # S3는 글로벌 서비스; us-east-1이 표준 엔드포인트
    s3 = boto3.client('s3', region_name='us-east-1')

    response = s3.list_buckets()
    buckets = response.get('Buckets', [])

    print(f"{'버킷 이름':<50} {'객체 수':>15}")
    print("-" * 66)

    for bucket in buckets:
        name = bucket['Name']
        count = count_bucket_objects(s3, name)
        if count == 'ACCESS_DENIED':
            count_str = '접근 거부'
        elif count is None:
            count_str = '없음'
        else:
            count_str = str(count)
        print(f"{name:<50} {count_str:>15}")

if __name__ == '__main__':
    main()
```

핵심 포인트:
- 버킷에는 수백만 개의 객체가 포함될 수 있으므로 `list_objects_v2`에는 항상 paginator를 사용하세요
- 오류 유형을 구분하기 위해 광범위한 `Exception` 대신 `ClientError`를 명시적으로 잡으세요
- `list_buckets`는 글로벌 작업입니다; 리디렉션 오류가 발생하면 개별 버킷 작업에 리전별 클라이언트가 필요할 수 있습니다

</details>

---

### 연습 문제 4: gcloud 출력 포맷팅

`asia-northeast3` 리전에서 `http-server` 태그가 있는 모든 실행 중인 Compute Engine 인스턴스의 외부 IP 주소를 추출하는 쉘 스크립트를 작성하려고 합니다. 출력은 줄당 하나의 순수 IP 주소여야 합니다 (다른 스크립트에서 사용하기 위해).

<details>
<summary>정답 보기</summary>

```bash
gcloud compute instances list \
    --filter="status=RUNNING AND tags.items=http-server AND zone:asia-northeast3" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

`value()` 형식은 헤더나 장식 없이 원시 값을 출력합니다 — 쉘 스크립트 사용에 이상적입니다.

**스크립트에서 출력 사용:**
```bash
#!/bin/bash
# health-check.sh — 모든 http-server 인스턴스의 HTTP 헬스 확인

ips=$(gcloud compute instances list \
    --filter="status=RUNNING AND tags.items=http-server AND zone:asia-northeast3" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

for ip in $ips; do
    if curl -sf --max-time 5 "http://$ip/health" > /dev/null; then
        echo "OK: $ip"
    else
        echo "FAIL: $ip"
    fi
done
```

대안적 형식 옵션:
- `--format="json"` — 프로그래밍 방식 처리를 위한 전체 JSON
- `--format="csv(name,natIP)"` — 열 헤더가 있는 CSV
- `--format="table(name,zone,natIP)"` — 사람이 읽기 좋은 테이블

</details>

---

### 연습 문제 5: SDK 에러 처리

주니어 개발자가 Cloud Storage에서 파일을 다운로드하는 GCP Python 코드를 작성했습니다. 문제점을 파악하고 적절한 에러 처리가 포함된 개선된 버전을 작성하세요.

```python
# 원본 (문제가 있는) 코드
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('data/report.csv')
blob.download_to_filename('/tmp/report.csv')
print("Downloaded!")
```

<details>
<summary>정답 보기</summary>

**원본의 문제점:**
1. 에러 처리 없음 — 버킷이나 블롭이 존재하지 않으면 처리되지 않은 예외로 충돌
2. 인증 확인 없음 — `GOOGLE_APPLICATION_CREDENTIALS`가 설정되지 않은 경우 에러 메시지가 혼란스러움
3. 무엇이 잘못되었는지, 어디서 찾아야 하는지에 대한 피드백 없음

**개선된 버전:**
```python
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden, GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError

def download_blob(bucket_name: str, source_blob_name: str, destination_file: str) -> bool:
    """
    Cloud Storage에서 블롭을 다운로드합니다.
    성공 시 True, 실패 시 False를 반환합니다.
    """
    try:
        client = storage.Client()
    except DefaultCredentialsError:
        print("ERROR: 자격 증명을 찾을 수 없습니다. GOOGLE_APPLICATION_CREDENTIALS를 설정하거나 "
              "'gcloud auth application-default login'을 실행하세요")
        return False

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file)
        print(f"다운로드 완료: gs://{bucket_name}/{source_blob_name} -> {destination_file}")
        return True

    except NotFound:
        print(f"ERROR: gs://{bucket_name}/{source_blob_name}을 찾을 수 없습니다. "
              f"버킷 이름과 객체 경로를 확인하세요.")
        return False
    except Forbidden:
        print(f"ERROR: gs://{bucket_name}/{source_blob_name} 접근 권한이 거부되었습니다. "
              f"서비스 계정 IAM 역할을 확인하세요.")
        return False
    except GoogleAPICallError as e:
        print(f"ERROR: GCP API 호출 실패: {e.message}")
        return False

# 사용법
success = download_blob('my-bucket', 'data/report.csv', '/tmp/report.csv')
if not success:
    exit(1)
```

주요 개선 사항:
- 자격 증명 초기화와 API 호출을 분리하여 오류의 원인을 명확히 파악
- 실행 가능한 오류 메시지를 위해 특정 예외 타입(`NotFound`, `Forbidden`)을 잡음
- 호출자가 실패를 프로그래밍 방식으로 처리할 수 있도록 boolean 반환
- 쉬운 디버깅을 위해 오류 메시지에 전체 GCS 경로 포함

</details>

---

## 참고 자료

- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)
- [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html)
- [gcloud CLI Documentation](https://cloud.google.com/sdk/gcloud/reference)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Google Cloud Python Client](https://googleapis.dev/python/google-api-core/latest/)
