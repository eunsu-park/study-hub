# 클라우드 통합

## 학습 목표

이 문서를 통해 다음을 학습합니다:

- cloud-init을 이용한 인스턴스 초기화
- AWS CLI 설치 및 설정
- EC2 메타데이터 서비스 활용
- 클라우드 환경에서의 Linux 운영

**난이도**: ⭐⭐⭐ (중급-고급)

---

## 목차

1. [cloud-init 개요](#1-cloud-init-개요)
2. [cloud-init 설정](#2-cloud-init-설정)
3. [AWS CLI](#3-aws-cli)
4. [EC2 메타데이터](#4-ec2-메타데이터)
5. [인스턴스 프로파일과 IAM](#5-인스턴스-프로파일과-iam)
6. [기타 클라우드 CLI](#6-기타-클라우드-cli)
7. [클라우드 네이티브 운영](#7-클라우드-네이티브-운영)

---

## 1. cloud-init 개요

### cloud-init이란?

cloud-init은 클라우드 인스턴스의 초기 설정을 자동화하는 도구입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Instance Boot                            │
│                         │                                   │
│                         ▼                                   │
│              ┌─────────────────┐                           │
│              │  cloud-init     │                           │
│              └────────┬────────┘                           │
│                       │                                     │
│         ┌─────────────┼─────────────┐                      │
│         ▼             ▼             ▼                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │user-data │  │meta-data │  │vendor-   │                 │
│  │(사용자)  │  │(클라우드)│  │data      │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│                       │                                     │
│                       ▼                                     │
│        ┌──────────────────────────────┐                    │
│        │ 네트워크 설정                 │                    │
│        │ SSH 키 설정                   │                    │
│        │ 패키지 설치                   │                    │
│        │ 스크립트 실행                 │                    │
│        │ 사용자 생성                   │                    │
│        └──────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### cloud-init 부팅 단계

| 단계 | 설명 |
|------|------|
| **Generator** | systemd에서 cloud-init 실행 결정 |
| **Local** | 로컬 데이터소스에서 네트워크 설정 |
| **Network** | 메타데이터 가져오기 |
| **Config** | cloud-config 모듈 실행 |
| **Final** | 최종 스크립트, 패키지 설치 |

### cloud-init 상태 확인

```bash
# 상태 확인
cloud-init status

# 상세 상태
cloud-init status --long

# 로그 확인
cat /var/log/cloud-init.log
cat /var/log/cloud-init-output.log

# 분석
cloud-init analyze show
cloud-init analyze blame
```

---

## 2. cloud-init 설정

### user-data 형식

```yaml
#cloud-config

# 사용자 생성
users:
  - name: deploy
    groups: sudo
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - ssh-rsa AAAAB3... user@host

# SSH 키
ssh_authorized_keys:
  - ssh-rsa AAAAB3... admin@company

# 패키지 업데이트 및 설치
package_update: true
package_upgrade: true
packages:
  - nginx
  - vim
  - git
  - htop

# 파일 작성
write_files:
  - path: /etc/nginx/sites-available/default
    content: |
      server {
          listen 80;
          server_name _;
          root /var/www/html;
      }
    owner: root:root
    permissions: '0644'

  - path: /opt/scripts/startup.sh
    content: |
      #!/bin/bash
      echo "Instance started at $(date)" >> /var/log/startup.log
    permissions: '0755'

# 명령 실행
runcmd:
  - systemctl enable nginx
  - systemctl start nginx
  - /opt/scripts/startup.sh

# 호스트명 설정
hostname: web-server-01
fqdn: web-server-01.example.com

# 타임존
timezone: Asia/Seoul

# NTP
ntp:
  enabled: true
  servers:
    - 0.pool.ntp.org
    - 1.pool.ntp.org

# 재부팅
power_state:
  mode: reboot
  message: "Rebooting after initial setup"
  timeout: 30
  condition: true
```

### 멀티파트 user-data

```bash
#!/bin/bash
# part-handler나 여러 형식 조합 시 사용

# Content-Type: multipart/mixed; boundary="==BOUNDARY=="
# MIME-Version: 1.0

# --==BOUNDARY==
# Content-Type: text/cloud-config; charset="us-ascii"

#cloud-config
packages:
  - nginx

# --==BOUNDARY==
# Content-Type: text/x-shellscript; charset="us-ascii"

#!/bin/bash
echo "Hello from shell script"

# --==BOUNDARY==--
```

### cloud-config 모듈

```yaml
#cloud-config

# 디스크 설정
disk_setup:
  /dev/xvdf:
    table_type: gpt
    layout: true
    overwrite: false

fs_setup:
  - label: data
    filesystem: ext4
    device: /dev/xvdf1

mounts:
  - ["/dev/xvdf1", "/data", "ext4", "defaults,nofail", "0", "2"]

# chef/puppet/ansible 통합
chef:
  install_type: omnibus
  server_url: https://chef.example.com
  node_name: web01

puppet:
  install_type: packages
  conf:
    agent:
      server: puppet.example.com

ansible:
  install_method: pip
  pull:
    url: https://github.com/user/ansible-repo.git
    playbook_name: site.yml

# 최종 메시지
final_message: "Instance ready after $UPTIME seconds"
```

### cloud-init 로컬 테스트

```bash
# cloud-init 설정 검증
cloud-init schema --config-file user-data.yaml

# 드라이런 (실제 실행 안 함)
cloud-init single --name write_files --frequency once

# 재실행 (테스트 목적)
sudo cloud-init clean --logs
sudo cloud-init init
sudo cloud-init modules --mode config
sudo cloud-init modules --mode final
```

---

## 3. AWS CLI

### AWS CLI 설치

```bash
# Linux x86_64
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Linux ARM
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# 버전 확인
aws --version

# 자동 완성 설정
complete -C '/usr/local/bin/aws_completer' aws
echo "complete -C '/usr/local/bin/aws_completer' aws" >> ~/.bashrc
```

### AWS CLI 설정

```bash
# 초기 설정
aws configure

# 프로파일 설정
aws configure --profile production
aws configure --profile development

# 설정 파일 직접 편집
cat ~/.aws/config
```

```ini
# ~/.aws/config
[default]
region = ap-northeast-2
output = json

[profile production]
region = ap-northeast-2
output = json

[profile development]
region = us-east-1
output = table
```

```ini
# ~/.aws/credentials
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

[production]
aws_access_key_id = AKIAIOSFODNN7PRODKEY
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYPRODKEY
```

### AWS CLI 기본 사용

```bash
# EC2 인스턴스 목록
aws ec2 describe-instances

# 특정 인스턴스 정보
aws ec2 describe-instances --instance-ids i-0123456789abcdef0

# 필터링
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,IP:PrivateIpAddress,Name:Tags[?Key==`Name`].Value|[0]}'

# 프로파일 사용
aws ec2 describe-instances --profile production

# 출력 형식 지정
aws ec2 describe-instances --output table
aws ec2 describe-instances --output yaml
```

### 주요 AWS CLI 명령어

```bash
# S3
aws s3 ls
aws s3 cp file.txt s3://my-bucket/
aws s3 sync ./local-dir s3://my-bucket/dir/
aws s3 rm s3://my-bucket/file.txt

# EC2
aws ec2 start-instances --instance-ids i-0123456789abcdef0
aws ec2 stop-instances --instance-ids i-0123456789abcdef0
aws ec2 create-tags --resources i-0123456789abcdef0 --tags Key=Name,Value=MyServer

# EBS
aws ec2 describe-volumes
aws ec2 create-snapshot --volume-id vol-0123456789abcdef0

# Security Groups
aws ec2 describe-security-groups
aws ec2 authorize-security-group-ingress \
    --group-id sg-0123456789abcdef0 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

# IAM
aws iam list-users
aws iam list-roles

# SSM (Systems Manager)
aws ssm start-session --target i-0123456789abcdef0
aws ssm send-command \
    --instance-ids i-0123456789abcdef0 \
    --document-name "AWS-RunShellScript" \
    --parameters commands=["echo hello"]
```

---

## 4. EC2 메타데이터

### IMDSv1 (레거시)

```bash
# 인스턴스 ID
curl http://169.254.169.254/latest/meta-data/instance-id

# 가용영역
curl http://169.254.169.254/latest/meta-data/placement/availability-zone

# 퍼블릭 IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# 프라이빗 IP
curl http://169.254.169.254/latest/meta-data/local-ipv4

# IAM 역할 자격 증명
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/my-role
```

### IMDSv2 (권장)

```bash
# 토큰 획득
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# 메타데이터 조회
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id

# 함수로 정의
get_metadata() {
    local TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
        "http://169.254.169.254/latest/meta-data/$1"
}

# 사용
get_metadata instance-id
get_metadata local-ipv4
get_metadata placement/availability-zone
```

### 주요 메타데이터 경로

| 경로 | 설명 |
|------|------|
| `instance-id` | 인스턴스 ID |
| `instance-type` | 인스턴스 유형 |
| `ami-id` | AMI ID |
| `local-hostname` | 프라이빗 호스트명 |
| `local-ipv4` | 프라이빗 IP |
| `public-hostname` | 퍼블릭 호스트명 |
| `public-ipv4` | 퍼블릭 IP |
| `placement/availability-zone` | 가용영역 |
| `placement/region` | 리전 |
| `mac` | MAC 주소 |
| `security-groups` | 보안 그룹 |
| `iam/security-credentials/<role>` | IAM 역할 자격 증명 |

### user-data 조회

```bash
# IMDSv2로 user-data 조회
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/user-data
```

---

## 5. 인스턴스 프로파일과 IAM

### IAM 역할로 AWS CLI 사용

```bash
# 인스턴스에 IAM 역할이 연결되어 있으면 자동으로 자격 증명 사용
# 별도의 aws configure 없이 바로 사용 가능

# S3 접근 (역할에 권한이 있다면)
aws s3 ls s3://my-bucket/

# 현재 자격 증명 확인
aws sts get-caller-identity
```

### IAM 역할 자격 증명 직접 사용

```bash
# 메타데이터에서 자격 증명 가져오기
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

ROLE_NAME=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/iam/security-credentials/)

CREDS=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME)

# 환경 변수로 설정
export AWS_ACCESS_KEY_ID=$(echo $CREDS | jq -r '.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo $CREDS | jq -r '.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo $CREDS | jq -r '.Token')

# AWS CLI 사용
aws s3 ls
```

### 스크립트에서 IAM 역할 활용

```bash
#!/bin/bash
# backup-to-s3.sh

BUCKET="my-backup-bucket"
BACKUP_DIR="/var/backup"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# 백업 생성
tar -czf /tmp/backup-$TIMESTAMP.tar.gz $BACKUP_DIR

# S3에 업로드 (인스턴스 역할 사용)
aws s3 cp /tmp/backup-$TIMESTAMP.tar.gz \
    s3://$BUCKET/backups/$INSTANCE_ID/backup-$TIMESTAMP.tar.gz

# 정리
rm /tmp/backup-$TIMESTAMP.tar.gz

echo "Backup completed: s3://$BUCKET/backups/$INSTANCE_ID/backup-$TIMESTAMP.tar.gz"
```

---

## 6. 기타 클라우드 CLI

### Google Cloud CLI (gcloud)

```bash
# 설치
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# 초기화
gcloud init

# 인증
gcloud auth login
gcloud auth application-default login

# 기본 명령
gcloud compute instances list
gcloud compute instances describe my-instance
gcloud compute ssh my-instance
```

### Azure CLI (az)

```bash
# 설치 (Ubuntu)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 로그인
az login

# 구독 설정
az account set --subscription "My Subscription"

# 기본 명령
az vm list
az vm show -g myResourceGroup -n myVM
az vm start -g myResourceGroup -n myVM
```

### 멀티 클라우드 메타데이터

```bash
# AWS
curl http://169.254.169.254/latest/meta-data/

# GCP
curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/

# Azure
curl -H "Metadata:true" "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
```

---

## 7. 클라우드 네이티브 운영

### 인스턴스 부트스트랩 스크립트

```bash
#!/bin/bash
# bootstrap.sh - EC2 user-data 스크립트

set -euo pipefail

# 로그 설정
exec > >(tee /var/log/bootstrap.log) 2>&1
echo "Bootstrap started at $(date)"

# 변수
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# 패키지 업데이트
apt-get update && apt-get upgrade -y

# 필수 패키지 설치
apt-get install -y \
    awscli \
    jq \
    nginx \
    docker.io

# Docker 설정
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# 태그에서 환경 정보 가져오기
ENVIRONMENT=$(aws ec2 describe-tags \
    --region $REGION \
    --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=Environment" \
    --query 'Tags[0].Value' --output text)

# 환경별 설정
case $ENVIRONMENT in
    production)
        echo "Configuring for production"
        # production 설정
        ;;
    staging)
        echo "Configuring for staging"
        # staging 설정
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        ;;
esac

# CloudWatch 에이전트 설치
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Nginx 시작
systemctl enable nginx
systemctl start nginx

echo "Bootstrap completed at $(date)"
```

### Parameter Store 활용

```bash
#!/bin/bash
# SSM Parameter Store에서 설정 가져오기

REGION="ap-northeast-2"

# 파라미터 가져오기
DB_HOST=$(aws ssm get-parameter \
    --region $REGION \
    --name "/myapp/production/db/host" \
    --query 'Parameter.Value' --output text)

DB_PASSWORD=$(aws ssm get-parameter \
    --region $REGION \
    --name "/myapp/production/db/password" \
    --with-decryption \
    --query 'Parameter.Value' --output text)

# 환경 변수로 설정
export DB_HOST
export DB_PASSWORD

# 애플리케이션 시작
exec /opt/myapp/bin/start
```

### 로그 관리 (CloudWatch)

```yaml
# /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/nginx/access.log",
            "log_group_name": "/aws/ec2/nginx/access",
            "log_stream_name": "{instance_id}"
          },
          {
            "file_path": "/var/log/nginx/error.log",
            "log_group_name": "/aws/ec2/nginx/error",
            "log_stream_name": "{instance_id}"
          },
          {
            "file_path": "/var/log/syslog",
            "log_group_name": "/aws/ec2/syslog",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  },
  "metrics": {
    "metrics_collected": {
      "mem": {
        "measurement": ["mem_used_percent"]
      },
      "disk": {
        "measurement": ["disk_used_percent"],
        "resources": ["/"]
      }
    }
  }
}
```

---

## 연습 문제

### 문제 1: cloud-init 설정

다음 요구사항의 cloud-config를 작성하세요:
- 사용자 `webadmin` 생성 (sudo 권한)
- nginx 설치
- 타임존 Asia/Seoul 설정

### 문제 2: AWS CLI 쿼리

실행 중인 EC2 인스턴스의 인스턴스 ID, 이름, 프라이빗 IP를 테이블 형식으로 출력하는 AWS CLI 명령을 작성하세요.

### 문제 3: IMDSv2 스크립트

IMDSv2를 사용하여 인스턴스의 가용영역과 인스턴스 유형을 출력하는 스크립트를 작성하세요.

---

## 정답

### 문제 1 정답

```yaml
#cloud-config

users:
  - name: webadmin
    groups: sudo
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']

package_update: true
packages:
  - nginx

timezone: Asia/Seoul

runcmd:
  - systemctl enable nginx
  - systemctl start nginx
```

### 문제 2 정답

```bash
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,Name:Tags[?Key==`Name`].Value|[0],IP:PrivateIpAddress}' \
    --output table
```

### 문제 3 정답

```bash
#!/bin/bash

# 토큰 획득
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# 메타데이터 조회
AZ=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/availability-zone)

INSTANCE_TYPE=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type)

echo "Availability Zone: $AZ"
echo "Instance Type: $INSTANCE_TYPE"
```

---

## 다음 단계

- [25_High_Availability_Cluster.md](./25_High_Availability_Cluster.md) - Pacemaker, Corosync, DRBD

---

## 참고 자료

- [cloud-init Documentation](https://cloudinit.readthedocs.io/)
- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)
- [EC2 Instance Metadata](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html)
- `man cloud-init`, `aws help`
