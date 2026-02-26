# 가상 머신 (EC2 / Compute Engine)

**이전**: [리전과 가용 영역](./03_Regions_Availability_Zones.md) | **다음**: [서버리스 함수](./05_Serverless_Functions.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. AWS EC2와 GCP Compute Engine의 서비스 기능 및 용어를 비교할 수 있습니다
2. 워크로드 요구사항(컴퓨팅, 메모리, GPU)에 따라 적합한 인스턴스 유형을 선택할 수 있습니다
3. 콘솔과 CLI를 사용하여 가상 머신을 설정하고 실행할 수 있습니다
4. 과금 모델(온디맨드, 예약, 스팟/선점형)을 설명하고 비용 효율적인 옵션을 선택할 수 있습니다
5. 가변 트래픽 부하를 처리하기 위한 오토 스케일링 그룹(Auto Scaling Group)을 구현할 수 있습니다
6. VM 인스턴스에 스토리지 볼륨을 연결하고 네트워킹을 구성할 수 있습니다
7. SSH 키 페어와 보안 그룹을 적용하여 VM 접근을 제어할 수 있습니다

---

가상 머신은 클라우드 컴퓨팅에서 가장 기본적인 구성 요소입니다. 웹 애플리케이션을 호스팅하든, 배치 작업을 실행하든, 머신러닝 모델을 학습시키든, 어느 시점에서든 반드시 VM을 사용하게 됩니다. VM 프로비저닝(Provisioning), 사이징(Sizing), 과금 방식을 마스터하는 것이 비용 효율적이고 확장 가능한 클라우드 아키텍처를 구축하는 첫 번째 단계입니다.

## 1. 가상 머신 개요

가상 머신(VM)은 클라우드에서 가장 기본적인 컴퓨팅 리소스입니다.

### 1.1 서비스 비교

| 항목 | AWS EC2 | GCP Compute Engine |
|------|---------|-------------------|
| 서비스명 | Elastic Compute Cloud | Compute Engine |
| 인스턴스 단위 | Instance | Instance |
| 이미지 | AMI | Image |
| 인스턴스 유형 | Instance Types | Machine Types |
| 시작 스크립트 | User Data | Startup Script |
| 메타데이터 | Instance Metadata | Metadata Server |

---

## 2. 인스턴스 유형

### 2.1 AWS EC2 인스턴스 유형

**명명 규칙:** `{패밀리}{세대}{추가속성}.{크기}`

예: `t3.medium`, `m5.xlarge`, `c6i.2xlarge`

| 패밀리 | 용도 | 예시 |
|--------|------|------|
| **t** | 범용 (버스터블) | t3.micro, t3.small |
| **m** | 범용 (균형) | m5.large, m6i.xlarge |
| **c** | 컴퓨팅 최적화 | c5.xlarge, c6i.2xlarge |
| **r** | 메모리 최적화 | r5.large, r6i.xlarge |
| **i** | 스토리지 최적화 | i3.large, i3en.xlarge |
| **g/p** | GPU | g4dn.xlarge, p4d.24xlarge |

**주요 인스턴스 스펙:**

| 유형 | vCPU | 메모리 | 네트워크 | 용도 |
|------|------|--------|----------|------|
| t3.micro | 2 | 1 GB | Low | 무료 티어, 개발 |
| t3.medium | 2 | 4 GB | Low-Mod | 소규모 앱 |
| m5.large | 2 | 8 GB | Up to 10 Gbps | 범용 |
| c5.xlarge | 4 | 8 GB | Up to 10 Gbps | CPU 집약 |
| r5.large | 2 | 16 GB | Up to 10 Gbps | 메모리 집약 |

### 2.2 GCP Machine Types

**명명 규칙:** `{시리즈}-{유형}-{vCPU수}` 또는 커스텀

예: `e2-medium`, `n2-standard-4`, `c2-standard-8`

| 시리즈 | 용도 | 예시 |
|--------|------|------|
| **e2** | 비용 효율 범용 | e2-micro, e2-medium |
| **n2/n2d** | 범용 (균형) | n2-standard-2, n2-highmem-4 |
| **c2/c2d** | 컴퓨팅 최적화 | c2-standard-4 |
| **m1/m2** | 메모리 최적화 | m1-megamem-96 |
| **a2** | GPU (A100) | a2-highgpu-1g |

**주요 머신 타입 스펙:**

| 유형 | vCPU | 메모리 | 네트워크 | 용도 |
|------|------|--------|----------|------|
| e2-micro | 0.25-2 | 1 GB | 1 Gbps | 무료 티어 |
| e2-medium | 1-2 | 4 GB | 2 Gbps | 소규모 앱 |
| n2-standard-2 | 2 | 8 GB | 10 Gbps | 범용 |
| c2-standard-4 | 4 | 16 GB | 10 Gbps | CPU 집약 |
| n2-highmem-2 | 2 | 16 GB | 10 Gbps | 메모리 집약 |

### 2.3 커스텀 머신 타입 (GCP)

GCP에서는 vCPU와 메모리를 개별 지정할 수 있습니다.

```bash
# 커스텀 머신 타입 생성
gcloud compute instances create my-instance \
    --custom-cpu=6 \
    --custom-memory=24GB \
    --zone=asia-northeast3-a
```

---

## 3. 이미지 (AMI / Image)

### 3.1 AWS AMI

**AMI (Amazon Machine Image)** 구성요소:
- 루트 볼륨 템플릿 (OS, 애플리케이션)
- 인스턴스 유형, 보안 그룹 기본값
- 블록 디바이스 매핑

```bash
# 사용 가능한 AMI 검색 (Amazon Linux 2023)
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1]'

# 주요 AMI 유형
# Amazon Linux 2023: al2023-ami-*
# Ubuntu 22.04: ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-*
# Windows Server: Windows_Server-2022-*
```

### 3.2 GCP Images

```bash
# 사용 가능한 이미지 목록
gcloud compute images list

# 특정 프로젝트의 이미지
gcloud compute images list \
    --filter="family:ubuntu-2204-lts"

# 주요 이미지 패밀리
# debian-11, debian-12
# ubuntu-2204-lts, ubuntu-2404-lts
# centos-stream-9, rocky-linux-9
# windows-2022
```

---

## 4. 인스턴스 생성

### 4.1 AWS EC2 인스턴스 생성

**Console:**
1. EC2 대시보드 → "Launch instance"
2. 이름 입력
3. AMI 선택 (예: Amazon Linux 2023)
4. 인스턴스 유형 선택 (예: t3.micro)
5. 키 페어 생성/선택
6. 네트워크 설정 (VPC, 서브넷, 보안 그룹)
7. 스토리지 설정
8. "Launch instance"

**AWS CLI:**
```bash
# 키 페어 생성
aws ec2 create-key-pair \
    --key-name my-key \
    --query 'KeyMaterial' \
    --output text > my-key.pem
chmod 400 my-key.pem

# 인스턴스 생성
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MyServer}]'
```

### 4.2 GCP Compute Engine 인스턴스 생성

**Console:**
1. Compute Engine → VM 인스턴스 → "만들기"
2. 이름 입력
3. 리전/Zone 선택
4. 머신 구성 선택 (예: e2-medium)
5. 부팅 디스크 (OS 이미지 선택)
6. 방화벽 설정 (HTTP/HTTPS 허용)
7. "만들기"

**gcloud CLI:**
```bash
# 인스턴스 생성
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --tags=http-server,https-server

# SSH 키는 자동 관리 (OS Login 또는 프로젝트 메타데이터)
```

---

## 5. SSH 접속

### 5.1 AWS EC2 SSH 접속

```bash
# 퍼블릭 IP 확인
aws ec2 describe-instances \
    --instance-ids i-1234567890abcdef0 \
    --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH 접속
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

# Amazon Linux: ec2-user
# Ubuntu: ubuntu
# CentOS: centos
# Debian: admin
```

**EC2 Instance Connect (브라우저):**
1. EC2 Console → 인스턴스 선택
2. "연결" 버튼 클릭
3. "EC2 Instance Connect" 탭
4. "연결" 클릭

### 5.2 GCP SSH 접속

```bash
# gcloud로 SSH (키 자동 관리)
gcloud compute ssh my-instance --zone=asia-northeast3-a

# 외부 IP 확인
gcloud compute instances describe my-instance \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# 직접 SSH (키를 수동 등록한 경우)
ssh -i ~/.ssh/google_compute_engine username@<EXTERNAL_IP>
```

**브라우저 SSH:**
1. Compute Engine → VM 인스턴스
2. 인스턴스 행의 "SSH" 버튼 클릭
3. 새 창에서 브라우저 터미널 열림

---

## 6. User Data / Startup Script

인스턴스 시작 시 자동으로 실행되는 스크립트입니다.

### 6.1 AWS User Data

```bash
#!/bin/bash
# User Data 예시 (Amazon Linux 2023)

# 패키지 업데이트
dnf update -y

# Nginx 설치
dnf install -y nginx
systemctl start nginx
systemctl enable nginx

# 커스텀 페이지
echo "<h1>Hello from $(hostname)</h1>" > /usr/share/nginx/html/index.html
```

**CLI에서 User Data 지정:**
```bash
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t3.micro \
    --user-data file://startup.sh \
    ...
```

**User Data 로그 확인:**
```bash
# 인스턴스 내부에서
cat /var/log/cloud-init-output.log
```

### 6.2 GCP Startup Script

```bash
#!/bin/bash
# Startup Script 예시 (Ubuntu)

# 패키지 업데이트
apt-get update

# Nginx 설치
apt-get install -y nginx
systemctl start nginx
systemctl enable nginx

# 커스텀 페이지
echo "<h1>Hello from $(hostname)</h1>" > /var/www/html/index.html
```

**CLI에서 Startup Script 지정:**
```bash
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --metadata-from-file=startup-script=startup.sh \
    ...

# 또는 인라인으로
gcloud compute instances create my-instance \
    --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install -y nginx'
```

**Startup Script 로그 확인:**
```bash
# 인스턴스 내부에서
sudo journalctl -u google-startup-scripts.service
# 또는
cat /var/log/syslog | grep startup-script
```

---

## 7. 인스턴스 메타데이터

인스턴스 내부에서 자신의 정보를 조회할 수 있습니다.

### 7.1 AWS Instance Metadata Service (IMDS)

```bash
# 인스턴스 ID
curl http://169.254.169.254/latest/meta-data/instance-id

# 퍼블릭 IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# 가용 영역
curl http://169.254.169.254/latest/meta-data/placement/availability-zone

# IAM 역할 자격 증명
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>

# IMDSv2 (권장 - 토큰 필요)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id
```

### 7.2 GCP Metadata Server

```bash
# 인스턴스 이름
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name

# 외부 IP
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip

# Zone
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone

# 서비스 계정 토큰
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token

# 프로젝트 ID
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id
```

---

## 8. 인스턴스 관리

### 8.1 인스턴스 상태 관리

**AWS:**
```bash
# 인스턴스 중지
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# 인스턴스 시작
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# 인스턴스 재부팅
aws ec2 reboot-instances --instance-ids i-1234567890abcdef0

# 인스턴스 종료 (삭제)
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# 인스턴스 상태 확인
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# 인스턴스 중지
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# 인스턴스 시작
gcloud compute instances start my-instance --zone=asia-northeast3-a

# 인스턴스 재시작 (reset)
gcloud compute instances reset my-instance --zone=asia-northeast3-a

# 인스턴스 삭제
gcloud compute instances delete my-instance --zone=asia-northeast3-a

# 인스턴스 상태 확인
gcloud compute instances describe my-instance --zone=asia-northeast3-a
```

### 8.2 인스턴스 유형 변경

**AWS:**
```bash
# 1. 인스턴스 중지
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# 2. 인스턴스 유형 변경
aws ec2 modify-instance-attribute \
    --instance-id i-1234567890abcdef0 \
    --instance-type t3.large

# 3. 인스턴스 시작
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# 1. 인스턴스 중지
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# 2. 머신 타입 변경
gcloud compute instances set-machine-type my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4

# 3. 인스턴스 시작
gcloud compute instances start my-instance --zone=asia-northeast3-a
```

---

## 9. 과금 옵션

### 9.1 온디맨드 vs 예약 vs 스팟

| 옵션 | AWS | GCP | 할인율 | 특징 |
|------|-----|-----|--------|------|
| **온디맨드** | On-Demand | On-demand | 0% | 약정 없음, 유연함 |
| **예약** | Reserved/Savings Plans | Committed Use | 최대 72% | 1-3년 약정 |
| **스팟/선점형** | Spot Instances | Spot/Preemptible | 최대 90% | 중단 가능 |
| **자동 할인** | - | Sustained Use | 최대 30% | 월 사용량 자동 |

### 9.2 AWS Spot Instance

```bash
# 스팟 인스턴스 요청
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.large",
        "KeyName": "my-key"
    }'

# 스팟 가격 확인
aws ec2 describe-spot-price-history \
    --instance-types t3.large \
    --product-descriptions "Linux/UNIX"
```

### 9.3 GCP Preemptible/Spot VM

```bash
# Spot VM 생성 (Preemptible 후속)
gcloud compute instances create spot-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

# Preemptible VM 생성 (레거시)
gcloud compute instances create preemptible-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --preemptible
```

---

## 10. 실습: 웹 서버 배포

### 10.1 AWS EC2 웹 서버

```bash
# 1. 보안 그룹 생성
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group"

# 2. 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

# 3. EC2 인스턴스 생성 (User Data 포함)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-groups web-sg \
    --user-data '#!/bin/bash
dnf update -y
dnf install -y nginx
systemctl start nginx
echo "<h1>AWS EC2 Web Server</h1>" > /usr/share/nginx/html/index.html' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'
```

### 10.2 GCP Compute Engine 웹 서버

```bash
# 1. 방화벽 규칙 생성
gcloud compute firewall-rules create allow-http \
    --allow tcp:80 \
    --target-tags http-server

# 2. Compute Engine 인스턴스 생성
gcloud compute instances create web-server \
    --zone=asia-northeast3-a \
    --machine-type=e2-micro \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y nginx
echo "<h1>GCP Compute Engine Web Server</h1>" > /var/www/html/index.html'

# 3. 외부 IP 확인
gcloud compute instances describe web-server \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

---

## 11. 다음 단계

- [05_Serverless_Functions.md](./05_Serverless_Functions.md) - 서버리스 함수
- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - 블록 스토리지 (EBS/PD)

---

## 연습 문제

### 연습 문제 1: 인스턴스 유형(Instance Type) 선택

각 워크로드(workload)에 가장 적합한 EC2 인스턴스 패밀리(family)를 매핑하고 이유를 설명하세요:

1. 프로모션 중 10배까지 급증할 수 있지만 야간에는 조용한 가변적 트래픽의 웹 애플리케이션
2. 빠른 쿼리를 위해 200 GB 데이터셋을 전체 RAM에 로드하는 인메모리(in-memory) 분석 엔진
3. 몇 시간 동안 집중적인 수학 시뮬레이션을 수행하는 배치 작업(GPU 없이 순수 CPU)
4. GPU 가속이 필요한 딥러닝 모델 학습 작업

<details>
<summary>정답 보기</summary>

1. **t 패밀리 (예: t3.medium 또는 t3.large)** — 버스터블(burstable) 인스턴스는 조용한 기간에 CPU 크레딧을 축적하고 급증 시 소비합니다. 지속적으로 높은 CPU가 필요하지 않은 가변적 워크로드에 비용 효율적입니다. 지속적인 대규모 급증에는 Auto Scaling Group과 m 패밀리 온디맨드 인스턴스가 더 적합합니다.

2. **r 패밀리 (예: r5.2xlarge 또는 r6i.2xlarge)** — 메모리 최적화 인스턴스는 높은 RAM 대 vCPU 비율을 제공합니다. 200 GB 데이터셋에는 256 GB RAM을 제공하는 r5.8xlarge 또는 유사 인스턴스가 필요합니다.

3. **c 패밀리 (예: c5.2xlarge 또는 c6i.4xlarge)** — 컴퓨팅 최적화 인스턴스는 CPU 집약적 워크로드에 대해 달러당 최고의 vCPU 성능을 제공합니다. c5/c6i 패밀리는 과학적 모델링, HPC, 비디오 인코딩에 특화되어 있습니다.

4. **p 또는 g 패밀리 (예: V100 GPU가 있는 p3.2xlarge, 또는 T4 GPU가 있는 g4dn.xlarge)** — GPU 인스턴스. `p` 인스턴스(p3/p4)는 딥러닝 학습용 고성능 NVIDIA GPU를 사용합니다. `g` 인스턴스(g4dn/g5)는 소규모 모델 학습과 추론(inference)에 더 비용 효율적입니다.

</details>

### 연습 문제 2: SSH 키 페어(Key Pair) 생성 및 인스턴스 연결

다음 단계를 위한 완전한 CLI 명령어 순서를 설명하세요:
1. `my-web-key`라는 새 EC2 키 페어를 생성하고 개인 키 파일을 저장합니다.
2. 해당 키 페어로 t3.micro Amazon Linux 2023 인스턴스를 시작합니다(AMI ID로 `ami-0c55b159cbfafe1f0`을 사용하세요).
3. 키 파일을 사용하여 인스턴스에 SSH로 연결합니다.

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 키 페어 생성 및 개인 키 저장
aws ec2 create-key-pair \
    --key-name my-web-key \
    --query 'KeyMaterial' \
    --output text > my-web-key.pem

# 올바른 권한 설정 (SSH에 필요)
chmod 400 my-web-key.pem

# 2단계: 키 페어로 인스턴스 시작
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-web-key \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'

# 새 인스턴스의 공개 IP 주소 조회
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=WebServer" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text

# 3단계: 인스턴스에 SSH 연결
# <PUBLIC_IP>를 이전 명령어에서 얻은 실제 IP로 교체하세요
ssh -i my-web-key.pem ec2-user@<PUBLIC_IP>
```

**참고사항**:
- `chmod 400`은 필수입니다. SSH는 전 세계가 읽을 수 있는 키 파일 사용을 거부합니다.
- Amazon Linux 2023의 기본 사용자는 `ec2-user`입니다. Ubuntu는 `ubuntu`입니다.
- SSH가 작동하려면 보안 그룹이 사용자 IP에서 인바운드 TCP 포트 22를 허용해야 합니다.

</details>

### 연습 문제 3: Auto Scaling Group 시나리오

웹 애플리케이션이 현재 단일 EC2 `m5.large` 인스턴스에서 실행됩니다. 트래픽 분석에 따르면 평일 오전 9시~오후 6시에는 인스턴스 4대, 야간/주말에는 1대만 필요합니다. 이를 자동으로 처리하는 Auto Scaling 설정을 설계하세요.

원하는 용량(desired)/최소(min)/최대(max) 용량 설정과 사용할 스케일링 정책 유형을 설명하세요.

<details>
<summary>정답 보기</summary>

**Auto Scaling Group 설정**:
- **최소 용량(Minimum capacity)**: 1 — 오프피크 시간에도 최소 하나의 인스턴스로 요청을 처리합니다.
- **최대 용량(Maximum capacity)**: 5 — 우발적인 스케일링 루프로 인한 비용 폭증을 방지합니다.
- **원하는 용량(Desired capacity)**: 1 (초기) — 최솟값에서 시작하고 스케일링 정책이 조정합니다.

**스케일링 정책**: 트래픽 패턴이 예측 가능하고 시간 기반이므로 **스케줄 스케일링(Scheduled Scaling)**이 이상적입니다:

```bash
# 평일 오전 9시에 인스턴스 4대로 확장
aws autoscaling put-scheduled-update-group-action \
    --auto-scaling-group-name my-asg \
    --scheduled-action-name scale-up-weekday \
    --recurrence "0 9 * * 1-5" \
    --desired-capacity 4

# 평일 오후 6시에 인스턴스 1대로 축소
aws autoscaling put-scheduled-update-group-action \
    --auto-scaling-group-name my-asg \
    --scheduled-action-name scale-down-weekday \
    --recurrence "0 18 * * 1-5" \
    --desired-capacity 1
```

**권장 추가 사항**: 스케줄된 시간 외의 예상치 못한 트래픽 급증에 대비하여 CPU 사용률에 대한 **타깃 추적(Target Tracking)** 정책(예: 평균 CPU 60% 유지)을 안전망으로 추가합니다.

</details>

### 연습 문제 4: 가격 모델(Pricing Model) 결정

데이터 분석 팀이 매일 밤 배치 처리 클러스터를 실행할 계획입니다:
- 매일 밤 11시~오전 5시(6시간) 실행
- 실행 중 `c5.2xlarge` 인스턴스 20대 필요
- 실행 중단 후 재시작 가능 (작업이 체크포인트됨)
- 18개월 동안 일관되게 실행 중

이 인스턴스에 어떤 가격 모델을 사용해야 하며, 온디맨드(On-Demand) 대비 대략 얼마나 절약할 수 있습니까?

<details>
<summary>정답 보기</summary>

**권장 가격 모델**: **스팟 인스턴스(Spot Instances)**

**이유**:
- 작업이 체크포인트되어 중단을 허용합니다 — 이것이 스팟의 핵심 전제 조건입니다.
- 스팟 인스턴스는 온디맨드 대비 최대 **90% 할인**을 제공합니다.
- `us-east-1`의 `c5.2xlarge` 온디맨드 가격은 약 $0.34/시간입니다. 90% 할인 시 스팟 가격은 약 $0.034/시간입니다.

**비용 비교** (근사치, 실제 가격은 변동될 수 있음):
- 온디맨드: 20대 × $0.34/시간 × 6시간 × 30일 = **월 약 $1,224**
- 스팟 (90% 할인): 20대 × $0.034/시간 × 6시간 × 30일 = **월 약 $122**
- **절감액: 월 약 $1,100 (연간 약 $13,200)**

**예약 인스턴스(Reserved Instances)가 아닌 이유?** 예약 인스턴스는 1~3년 약정이 필요하며 지속적인 사용을 기준으로 가격이 책정됩니다. 이 인스턴스는 하루에 6시간만 실행됩니다(25% 가동률). 25% 가동률 VM의 예약 인스턴스는 스팟 대비 비용 효율적이지 않습니다.

**모범 사례**: 여러 인스턴스 유형과 AZ를 사용하는 스팟 플릿(Spot Fleet)을 사용하여 중단 가능성을 최소화합니다.

</details>

### 연습 문제 5: GCP 커스텀 머신 타입(Custom Machine Type) vs AWS 동등물

GCP 워크로드에 정확히 6 vCPU와 20 GB RAM이 필요합니다. 이 정확한 구성을 가진 표준 GCP 머신 타입은 없습니다.

1. `asia-northeast3-a` 존에서 이 사양으로 커스텀 머신 타입 인스턴스를 생성하는 `gcloud` 명령어를 작성하세요.
2. AWS EC2에서 동등한 접근 방식은 무엇이며, AWS가 커스텀 인스턴스 타입을 제공하지 않는 이유는 무엇입니까?

<details>
<summary>정답 보기</summary>

1. **GCP 커스텀 머신 타입**:
```bash
gcloud compute instances create custom-instance \
    --zone=asia-northeast3-a \
    --custom-cpu=6 \
    --custom-memory=20GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud
```

GCP는 커스텀 메모리가 256 MB의 배수여야 합니다. 20 GB(20480 MB)는 유효한 배수입니다.

2. **AWS 동등 접근 방식**: AWS는 커스텀 인스턴스 타입을 제공하지 않습니다. AWS에서는 사전 정의된 타입 카탈로그에서 선택합니다. 6 vCPU / 20 GB RAM에 가장 가까운 타입을 찾아야 합니다:
   - `c5.xlarge` = 4 vCPU, 8 GB (너무 작음)
   - `m5.2xlarge` = 8 vCPU, 32 GB (과잉 프로비저닝이지만 가장 근접한 균형 잡힌 옵션)
   - `c5.2xlarge` = 8 vCPU, 16 GB (컴퓨팅 중심, CPU 약간 초과)

   **AWS에 커스텀 타입이 없는 이유?** AWS는 특정 인스턴스 패밀리를 중심으로 물리적 하드웨어와 하이퍼바이저(hypervisor) 설정을 최적화하여 예측 가능한 성능 보장과 규모의 경제를 실현합니다. GCP는 존(zone) 내 리소스 한도 내에서 임의의 조합을 허용하는 더 유연한 할당 모델을 사용합니다.

   **실질적 영향**: GCP의 커스텀 머신 타입은 정밀한 사이징(right-sizing)과 미사용 vCPU 또는 RAM에 대한 비용 지불을 방지합니다. AWS에서는 일반적으로 가장 가까운 사용 가능한 크기로 과잉 프로비저닝해야 합니다.

</details>

---

## 참고 자료

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [GCP Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [GCP Machine Types](https://cloud.google.com/compute/docs/machine-types)
