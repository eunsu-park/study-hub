# Cloud Integration

**Previous**: [Advanced Networking](./23_Advanced_Networking.md) | **Next**: [High Availability Cluster](./25_High_Availability_Cluster.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write cloud-init user-data configurations to automate instance bootstrapping
2. Install and configure the AWS CLI with profiles and credential management
3. Query EC2 instance metadata using IMDSv2 for secure self-discovery
4. Use IAM instance profiles to grant AWS permissions without static credentials
5. Retrieve secrets from AWS SSM Parameter Store in startup scripts
6. Work with GCP and Azure CLIs for multi-cloud Linux operations
7. Build production-ready bootstrap scripts that configure instances at launch

---

## Table of Contents

1. [cloud-init Overview](#1-cloud-init-overview)
2. [cloud-init Configuration](#2-cloud-init-configuration)
3. [AWS CLI](#3-aws-cli)
4. [EC2 Metadata](#4-ec2-metadata)
5. [Instance Profiles and IAM](#5-instance-profiles-and-iam)
6. [Other Cloud CLIs](#6-other-cloud-clis)
7. [Cloud-Native Operations](#7-cloud-native-operations)

---

Cloud computing has fundamentally changed how Linux servers are provisioned and managed. Instead of carefully installing a single server, you launch dozens of instances from templates and configure them automatically at boot. Understanding cloud-init, metadata services, and cloud CLIs is essential because these are the building blocks of infrastructure-as-code in every major cloud platform.

## 1. cloud-init Overview

### What is cloud-init?

cloud-init is a tool that automates the initial configuration of cloud instances.

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
│  │(user)    │  │(cloud)   │  │data      │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│                       │                                     │
│                       ▼                                     │
│        ┌──────────────────────────────┐                    │
│        │ Network configuration        │                    │
│        │ SSH key setup                │                    │
│        │ Package installation         │                    │
│        │ Script execution             │                    │
│        │ User creation                │                    │
│        └──────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### cloud-init Boot Stages

| Stage | Description |
|-------|-------------|
| **Generator** | systemd decides whether to run cloud-init |
| **Local** | Network configuration from local datasource |
| **Network** | Fetch metadata |
| **Config** | Run cloud-config modules |
| **Final** | Final scripts, package installation |

### Checking cloud-init Status

```bash
# Check status
cloud-init status

# Detailed status
cloud-init status --long

# View logs
cat /var/log/cloud-init.log
cat /var/log/cloud-init-output.log

# Analysis
cloud-init analyze show
cloud-init analyze blame
```

---

## 2. cloud-init Configuration

### user-data Formats

```yaml
#cloud-config

# Create users
users:
  - name: deploy
    groups: sudo
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - ssh-rsa AAAAB3... user@host

# SSH keys
ssh_authorized_keys:
  - ssh-rsa AAAAB3... admin@company

# Package update and installation
package_update: true
package_upgrade: true
packages:
  - nginx
  - vim
  - git
  - htop

# Write files
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

# Run commands
runcmd:
  - systemctl enable nginx
  - systemctl start nginx
  - /opt/scripts/startup.sh

# Hostname configuration
hostname: web-server-01
fqdn: web-server-01.example.com

# Timezone
timezone: Asia/Seoul

# NTP
ntp:
  enabled: true
  servers:
    - 0.pool.ntp.org
    - 1.pool.ntp.org

# Reboot
power_state:
  mode: reboot
  message: "Rebooting after initial setup"
  timeout: 30
  condition: true
```

### Multipart user-data

```bash
#!/bin/bash
# Used for part-handler or combining multiple formats

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

### cloud-config Modules

```yaml
#cloud-config

# Disk configuration
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

# chef/puppet/ansible integration
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

# Final message
final_message: "Instance ready after $UPTIME seconds"
```

### Testing cloud-init Locally

```bash
# Validate cloud-init configuration
cloud-init schema --config-file user-data.yaml

# Dry run (no actual execution)
cloud-init single --name write_files --frequency once

# Re-run (for testing purposes)
sudo cloud-init clean --logs
sudo cloud-init init
sudo cloud-init modules --mode config
sudo cloud-init modules --mode final
```

---

## 3. AWS CLI

### Installing AWS CLI

```bash
# Linux x86_64
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Linux ARM
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Check version
aws --version

# Configure auto-completion
complete -C '/usr/local/bin/aws_completer' aws
echo "complete -C '/usr/local/bin/aws_completer' aws" >> ~/.bashrc
```

### Configuring AWS CLI

```bash
# Initial configuration
aws configure

# Profile configuration
aws configure --profile production
aws configure --profile development

# Directly edit configuration files
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

### Basic AWS CLI Usage

```bash
# List EC2 instances
aws ec2 describe-instances

# Specific instance information
aws ec2 describe-instances --instance-ids i-0123456789abcdef0

# Filtering
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,IP:PrivateIpAddress,Name:Tags[?Key==`Name`].Value|[0]}'

# Use profile
aws ec2 describe-instances --profile production

# Specify output format
aws ec2 describe-instances --output table
aws ec2 describe-instances --output yaml
```

### Common AWS CLI Commands

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

## 4. EC2 Metadata

### IMDSv1 (Legacy)

```bash
# Instance ID
curl http://169.254.169.254/latest/meta-data/instance-id

# Availability Zone
curl http://169.254.169.254/latest/meta-data/placement/availability-zone

# Public IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Private IP
curl http://169.254.169.254/latest/meta-data/local-ipv4

# IAM role credentials
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/my-role
```

### IMDSv2 (Recommended)

```bash
# Get token
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Query metadata
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id

# Define as function
get_metadata() {
    local TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
        "http://169.254.169.254/latest/meta-data/$1"
}

# Usage
get_metadata instance-id
get_metadata local-ipv4
get_metadata placement/availability-zone
```

### Key Metadata Paths

| Path | Description |
|------|-------------|
| `instance-id` | Instance ID |
| `instance-type` | Instance type |
| `ami-id` | AMI ID |
| `local-hostname` | Private hostname |
| `local-ipv4` | Private IP |
| `public-hostname` | Public hostname |
| `public-ipv4` | Public IP |
| `placement/availability-zone` | Availability Zone |
| `placement/region` | Region |
| `mac` | MAC address |
| `security-groups` | Security groups |
| `iam/security-credentials/<role>` | IAM role credentials |

### Querying user-data

```bash
# Query user-data with IMDSv2
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/user-data
```

---

## 5. Instance Profiles and IAM

### Using AWS CLI with IAM Roles

```bash
# If an IAM role is attached to the instance, credentials are used automatically
# No separate aws configure needed

# S3 access (if role has permission)
aws s3 ls s3://my-bucket/

# Check current credentials
aws sts get-caller-identity
```

### Directly Using IAM Role Credentials

```bash
# Get credentials from metadata
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

ROLE_NAME=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/iam/security-credentials/)

CREDS=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME)

# Set as environment variables
export AWS_ACCESS_KEY_ID=$(echo $CREDS | jq -r '.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo $CREDS | jq -r '.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo $CREDS | jq -r '.Token')

# Use AWS CLI
aws s3 ls
```

### Using IAM Roles in Scripts

```bash
#!/bin/bash
# backup-to-s3.sh

BUCKET="my-backup-bucket"
BACKUP_DIR="/var/backup"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Create backup
tar -czf /tmp/backup-$TIMESTAMP.tar.gz $BACKUP_DIR

# Upload to S3 (using instance role)
aws s3 cp /tmp/backup-$TIMESTAMP.tar.gz \
    s3://$BUCKET/backups/$INSTANCE_ID/backup-$TIMESTAMP.tar.gz

# Cleanup
rm /tmp/backup-$TIMESTAMP.tar.gz

echo "Backup completed: s3://$BUCKET/backups/$INSTANCE_ID/backup-$TIMESTAMP.tar.gz"
```

---

## 6. Other Cloud CLIs

### Google Cloud CLI (gcloud)

```bash
# Installation
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Initialize
gcloud init

# Authentication
gcloud auth login
gcloud auth application-default login

# Basic commands
gcloud compute instances list
gcloud compute instances describe my-instance
gcloud compute ssh my-instance
```

### Azure CLI (az)

```bash
# Installation (Ubuntu)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Set subscription
az account set --subscription "My Subscription"

# Basic commands
az vm list
az vm show -g myResourceGroup -n myVM
az vm start -g myResourceGroup -n myVM
```

### Multi-Cloud Metadata

```bash
# AWS
curl http://169.254.169.254/latest/meta-data/

# GCP
curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/

# Azure
curl -H "Metadata:true" "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
```

---

## 7. Cloud-Native Operations

### Instance Bootstrap Script

```bash
#!/bin/bash
# bootstrap.sh - EC2 user-data script

set -euo pipefail

# Log configuration
exec > >(tee /var/log/bootstrap.log) 2>&1
echo "Bootstrap started at $(date)"

# Variables
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Package update
apt-get update && apt-get upgrade -y

# Install required packages
apt-get install -y \
    awscli \
    jq \
    nginx \
    docker.io

# Docker configuration
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# Get environment info from tags
ENVIRONMENT=$(aws ec2 describe-tags \
    --region $REGION \
    --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=Environment" \
    --query 'Tags[0].Value' --output text)

# Environment-specific configuration
case $ENVIRONMENT in
    production)
        echo "Configuring for production"
        # production configuration
        ;;
    staging)
        echo "Configuring for staging"
        # staging configuration
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        ;;
esac

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Start Nginx
systemctl enable nginx
systemctl start nginx

echo "Bootstrap completed at $(date)"
```

### Using Parameter Store

```bash
#!/bin/bash
# Get configuration from SSM Parameter Store

REGION="ap-northeast-2"

# Get parameters
DB_HOST=$(aws ssm get-parameter \
    --region $REGION \
    --name "/myapp/production/db/host" \
    --query 'Parameter.Value' --output text)

DB_PASSWORD=$(aws ssm get-parameter \
    --region $REGION \
    --name "/myapp/production/db/password" \
    --with-decryption \
    --query 'Parameter.Value' --output text)

# Set as environment variables
export DB_HOST
export DB_PASSWORD

# Start application
exec /opt/myapp/bin/start
```

### Log Management (CloudWatch)

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

## Practice Problems

### Problem 1: cloud-init Configuration

Write a cloud-config with the following requirements:
- Create user `webadmin` (with sudo privileges)
- Install nginx
- Set timezone to Asia/Seoul

### Problem 2: AWS CLI Query

Write an AWS CLI command that outputs running EC2 instances' Instance ID, Name, and Private IP in table format.

### Problem 3: IMDSv2 Script

Write a script that outputs the instance's availability zone and instance type using IMDSv2.

---

## Answers

### Problem 1 Answer

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

### Problem 2 Answer

```bash
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,Name:Tags[?Key==`Name`].Value|[0],IP:PrivateIpAddress}' \
    --output table
```

### Problem 3 Answer

```bash
#!/bin/bash

# Get token
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Query metadata
AZ=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/availability-zone)

INSTANCE_TYPE=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type)

echo "Availability Zone: $AZ"
echo "Instance Type: $INSTANCE_TYPE"
```

---

---

## References

- [cloud-init Documentation](https://cloudinit.readthedocs.io/)
- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)
- [EC2 Instance Metadata](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html)
- `man cloud-init`, `aws help`

---

**Previous**: [Advanced Networking](./23_Advanced_Networking.md) | **Next**: [High Availability Cluster](./25_High_Availability_Cluster.md)
