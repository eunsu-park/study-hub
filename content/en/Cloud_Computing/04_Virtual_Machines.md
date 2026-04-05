# Virtual Machines (EC2 / Compute Engine)

**Previous**: [Regions and Availability Zones](./03_Regions_Availability_Zones.md) | **Next**: [Serverless Functions](./05_Serverless_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare AWS EC2 and GCP Compute Engine service features and terminology
2. Select appropriate instance types based on workload requirements (compute, memory, GPU)
3. Configure and launch a virtual machine using the console and CLI
4. Explain pricing models (on-demand, reserved, spot/preemptible) and choose cost-effective options
5. Implement auto-scaling groups to handle variable traffic loads
6. Attach storage volumes and configure networking for VM instances
7. Apply SSH key pairs and security groups to control VM access

---

Virtual machines are the most fundamental building block of cloud computing. Whether you are hosting a web application, running a batch job, or training a machine learning model, you will almost certainly use VMs at some point. Mastering VM provisioning, sizing, and pricing is the first step toward building cost-effective, scalable cloud architectures.

## 1. Virtual Machine Overview

Virtual machines (VMs) are the most fundamental computing resources in the cloud.

### 1.1 Service Comparison

| Item | AWS EC2 | GCP Compute Engine |
|------|---------|-------------------|
| Service Name | Elastic Compute Cloud | Compute Engine |
| Instance Unit | Instance | Instance |
| Image | AMI | Image |
| Instance Type | Instance Types | Machine Types |
| Launch Script | User Data | Startup Script |
| Metadata | Instance Metadata | Metadata Server |

---

## 2. Instance Types

### 2.1 AWS EC2 Instance Types

**Naming Convention:** `{family}{generation}{attributes}.{size}`

Examples: `t3.medium`, `m5.xlarge`, `c6i.2xlarge`

| Family | Purpose | Examples |
|--------|------|------|
| **t** | General Purpose (Burstable) | t3.micro, t3.small |
| **m** | General Purpose (Balanced) | m5.large, m6i.xlarge |
| **c** | Compute Optimized | c5.xlarge, c6i.2xlarge |
| **r** | Memory Optimized | r5.large, r6i.xlarge |
| **i** | Storage Optimized | i3.large, i3en.xlarge |
| **g/p** | GPU | g4dn.xlarge, p4d.24xlarge |

**Key Instance Specifications:**

| Type | vCPU | Memory | Network | Use Case |
|------|------|--------|----------|------|
| t3.micro | 2 | 1 GB | Low | Free tier, development |
| t3.medium | 2 | 4 GB | Low-Mod | Small apps |
| m5.large | 2 | 8 GB | Up to 10 Gbps | General purpose |
| c5.xlarge | 4 | 8 GB | Up to 10 Gbps | CPU intensive |
| r5.large | 2 | 16 GB | Up to 10 Gbps | Memory intensive |

### 2.2 GCP Machine Types

**Naming Convention:** `{series}-{type}-{vCPU-count}` or custom

Examples: `e2-medium`, `n2-standard-4`, `c2-standard-8`

| Series | Purpose | Examples |
|--------|------|------|
| **e2** | Cost-effective General Purpose | e2-micro, e2-medium |
| **n2/n2d** | General Purpose (Balanced) | n2-standard-2, n2-highmem-4 |
| **c2/c2d** | Compute Optimized | c2-standard-4 |
| **m1/m2** | Memory Optimized | m1-megamem-96 |
| **a2** | GPU (A100) | a2-highgpu-1g |

**Key Machine Type Specifications:**

| Type | vCPU | Memory | Network | Use Case |
|------|------|--------|----------|------|
| e2-micro | 0.25-2 | 1 GB | 1 Gbps | Free tier |
| e2-medium | 1-2 | 4 GB | 2 Gbps | Small apps |
| n2-standard-2 | 2 | 8 GB | 10 Gbps | General purpose |
| c2-standard-4 | 4 | 16 GB | 10 Gbps | CPU intensive |
| n2-highmem-2 | 2 | 16 GB | 10 Gbps | Memory intensive |

### 2.3 Custom Machine Types (GCP)

GCP allows you to specify vCPU and memory individually.

```bash
# Create custom machine type
gcloud compute instances create my-instance \
    --custom-cpu=6 \
    --custom-memory=24GB \
    --zone=asia-northeast3-a
```

---

## 3. Images (AMI / Image)

### 3.1 AWS AMI

**AMI (Amazon Machine Image)** Components:
- Root volume template (OS, applications)
- Instance type, security group defaults
- Block device mapping

```bash
# Search available AMI (Amazon Linux 2023)
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1]'

# Major AMI types
# Amazon Linux 2023: al2023-ami-*
# Ubuntu 22.04: ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-*
# Windows Server: Windows_Server-2022-*
```

### 3.2 GCP Images

```bash
# List available images
gcloud compute images list

# Images from specific project
gcloud compute images list \
    --filter="family:ubuntu-2204-lts"

# Major image families
# debian-11, debian-12
# ubuntu-2204-lts, ubuntu-2404-lts
# centos-stream-9, rocky-linux-9
# windows-2022
```

---

## 4. Creating Instances

### 4.1 AWS EC2 Instance Creation

**Console:**
1. EC2 Dashboard → "Launch instance"
2. Enter name
3. Select AMI (e.g., Amazon Linux 2023)
4. Select instance type (e.g., t3.micro)
5. Create/select key pair
6. Network settings (VPC, subnet, security group)
7. Storage configuration
8. "Launch instance"

**AWS CLI:**
```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name my-key \
    --query 'KeyMaterial' \
    --output text > my-key.pem
chmod 400 my-key.pem

# Create instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MyServer}]'
```

### 4.2 GCP Compute Engine Instance Creation

**Console:**
1. Compute Engine → VM instances → "Create"
2. Enter name
3. Select region/zone
4. Select machine configuration (e.g., e2-medium)
5. Boot disk (select OS image)
6. Firewall settings (allow HTTP/HTTPS)
7. "Create"

**gcloud CLI:**
```bash
# Create instance
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --tags=http-server,https-server

# SSH keys are automatically managed (OS Login or project metadata)
```

---

## 5. SSH Connection

### 5.1 AWS EC2 SSH Connection

```bash
# Check public IP
aws ec2 describe-instances \
    --instance-ids i-1234567890abcdef0 \
    --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH connection
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

# Amazon Linux: ec2-user
# Ubuntu: ubuntu
# CentOS: centos
# Debian: admin
```

**EC2 Instance Connect (Browser):**
1. EC2 Console → Select instance
2. Click "Connect" button
3. "EC2 Instance Connect" tab
4. Click "Connect"

### 5.2 GCP SSH Connection

```bash
# SSH with gcloud (automatic key management)
gcloud compute ssh my-instance --zone=asia-northeast3-a

# Check external IP
gcloud compute instances describe my-instance \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Direct SSH (if key manually registered)
ssh -i ~/.ssh/google_compute_engine username@<EXTERNAL_IP>
```

**Browser SSH:**
1. Compute Engine → VM instances
2. Click "SSH" button in instance row
3. Browser terminal opens in new window

---

## 6. User Data / Startup Script

Scripts that run automatically when instances start.

### 6.1 AWS User Data

```bash
#!/bin/bash
# User Data example (Amazon Linux 2023)

# Update packages
dnf update -y

# Install Nginx
dnf install -y nginx
systemctl start nginx
systemctl enable nginx

# Custom page
echo "<h1>Hello from $(hostname)</h1>" > /usr/share/nginx/html/index.html
```

**Specify User Data in CLI:**
```bash
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t3.micro \
    --user-data file://startup.sh \
    ...
```

**Check User Data logs:**
```bash
# Inside instance
cat /var/log/cloud-init-output.log
```

### 6.2 GCP Startup Script

```bash
#!/bin/bash
# Startup Script example (Ubuntu)

# Update packages
apt-get update

# Install Nginx
apt-get install -y nginx
systemctl start nginx
systemctl enable nginx

# Custom page
echo "<h1>Hello from $(hostname)</h1>" > /var/www/html/index.html
```

**Specify Startup Script in CLI:**
```bash
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --metadata-from-file=startup-script=startup.sh \
    ...

# Or inline
gcloud compute instances create my-instance \
    --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install -y nginx'
```

**Check Startup Script logs:**
```bash
# Inside instance
sudo journalctl -u google-startup-scripts.service
# Or
cat /var/log/syslog | grep startup-script
```

---

## 7. Instance Metadata

Query instance information from inside the instance.

### 7.1 AWS Instance Metadata Service (IMDS)

```bash
# Instance ID
curl http://169.254.169.254/latest/meta-data/instance-id

# Public IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Availability Zone
curl http://169.254.169.254/latest/meta-data/placement/availability-zone

# IAM role credentials
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>

# IMDSv2 (recommended - token required)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id
```

### 7.2 GCP Metadata Server

```bash
# Instance name
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name

# External IP
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip

# Zone
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone

# Service account token
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token

# Project ID
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id
```

---

## 8. Instance Management

### 8.1 Instance State Management

**AWS:**
```bash
# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Reboot instance
aws ec2 reboot-instances --instance-ids i-1234567890abcdef0

# Terminate instance (delete)
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# Check instance status
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# Stop instance
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# Start instance
gcloud compute instances start my-instance --zone=asia-northeast3-a

# Restart instance (reset)
gcloud compute instances reset my-instance --zone=asia-northeast3-a

# Delete instance
gcloud compute instances delete my-instance --zone=asia-northeast3-a

# Check instance status
gcloud compute instances describe my-instance --zone=asia-northeast3-a
```

### 8.2 Change Instance Type

**AWS:**
```bash
# 1. Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# 2. Change instance type
aws ec2 modify-instance-attribute \
    --instance-id i-1234567890abcdef0 \
    --instance-type t3.large

# 3. Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# 1. Stop instance
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# 2. Change machine type
gcloud compute instances set-machine-type my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4

# 3. Start instance
gcloud compute instances start my-instance --zone=asia-northeast3-a
```

---

## 9. Pricing Options

### 9.1 On-Demand vs Reserved vs Spot

| Option | AWS | GCP | Discount | Characteristics |
|------|-----|-----|--------|------|
| **On-Demand** | On-Demand | On-demand | 0% | No commitment, flexible |
| **Reserved** | Reserved/Savings Plans | Committed Use | Up to 72% | 1-3 year commitment |
| **Spot/Preemptible** | Spot Instances | Spot/Preemptible | Up to 90% | Can be interrupted |
| **Auto Discount** | - | Sustained Use | Up to 30% | Automatic monthly usage |

### 9.2 AWS Spot Instance

```bash
# Request spot instance
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.large",
        "KeyName": "my-key"
    }'

# Check spot pricing
aws ec2 describe-spot-price-history \
    --instance-types t3.large \
    --product-descriptions "Linux/UNIX"
```

### 9.3 GCP Preemptible/Spot VM

```bash
# Create Spot VM (successor to Preemptible)
gcloud compute instances create spot-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

# Create Preemptible VM (legacy)
gcloud compute instances create preemptible-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --preemptible
```

---

## 10. Practice: Deploy Web Server

### 10.1 AWS EC2 Web Server

```bash
# 1. Create security group
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group"

# 2. Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

# 3. Create EC2 instance (with User Data)
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

### 10.2 GCP Compute Engine Web Server

```bash
# 1. Create firewall rule
gcloud compute firewall-rules create allow-http \
    --allow tcp:80 \
    --target-tags http-server

# 2. Create Compute Engine instance
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

# 3. Check external IP
gcloud compute instances describe web-server \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

---

## 11. Next Steps

- [05_Serverless_Functions.md](./05_Serverless_Functions.md) - Serverless functions
- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - Block storage (EBS/PD)

---

## Exercises

### Exercise 1: Instance Type Selection

Match each workload with the most appropriate EC2 instance family and explain your reasoning:

1. A web application with highly variable traffic that can spike 10x during promotions but is quiet overnight.
2. An in-memory analytics engine that loads a 200 GB dataset entirely into RAM for fast queries.
3. A batch job that performs intensive mathematical simulations (pure CPU, no GPU) for several hours.
4. A deep learning model training job that requires GPU acceleration.

<details>
<summary>Show Answer</summary>

1. **t family (e.g., t3.medium or t3.large)** — Burstable instances accumulate CPU credits during quiet periods and spend them during spikes. They are cost-effective for variable workloads that don't need sustained high-CPU all the time. For very large sustained spikes, an Auto Scaling Group with on-demand m-family would be better.

2. **r family (e.g., r5.2xlarge or r6i.2xlarge)** — Memory-optimized instances provide high RAM-to-vCPU ratios. An r5.2xlarge has 64 GB RAM, and an r5.4xlarge provides 128 GB. For a 200 GB dataset, you'd need r5.8xlarge (256 GB) or similar.

3. **c family (e.g., c5.2xlarge or c6i.4xlarge)** — Compute-optimized instances offer the highest vCPU count and performance per dollar for CPU-bound workloads. The c5/c6i family is specifically tuned for compute-intensive applications like scientific modeling, HPC, and video encoding.

4. **p or g family (e.g., p3.2xlarge with V100 GPU, or g4dn.xlarge with T4 GPU)** — GPU instances. `p` instances (p3/p4) use high-end NVIDIA GPUs designed for deep learning training. `g` instances (g4dn/g5) are more cost-effective for both training smaller models and inference.

</details>

### Exercise 2: SSH Key Pair Creation and Instance Connection

Describe the complete sequence of steps (CLI commands) to:
1. Create a new EC2 key pair named `my-web-key` and save the private key file.
2. Launch a t3.micro Amazon Linux 2023 instance with that key pair (use `ami-0c55b159cbfafe1f0` as a placeholder AMI ID).
3. SSH into the instance using the key file.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Create key pair and save private key
aws ec2 create-key-pair \
    --key-name my-web-key \
    --query 'KeyMaterial' \
    --output text > my-web-key.pem

# Set correct permissions (required for SSH)
chmod 400 my-web-key.pem

# Step 2: Launch instance with key pair
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-web-key \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'

# Get the public IP address of the new instance
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=WebServer" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text

# Step 3: SSH into the instance
# Replace <PUBLIC_IP> with the actual public IP from the previous command
ssh -i my-web-key.pem ec2-user@<PUBLIC_IP>
```

**Notes**:
- `chmod 400` is mandatory; SSH will refuse to use a key file that is world-readable.
- For Amazon Linux 2023, the default user is `ec2-user`. For Ubuntu, it is `ubuntu`.
- The security group must allow inbound TCP port 22 from your IP for SSH to work.

</details>

### Exercise 3: Auto Scaling Group Scenario

Your web application currently runs on a single EC2 `m5.large` instance. Traffic analysis shows that on weekdays between 9 AM–6 PM you need 4 instances, and overnight/weekends you only need 1. Design an Auto Scaling configuration to handle this automatically.

Describe: the desired/min/max capacity settings, and the type of scaling policy you would use.

<details>
<summary>Show Answer</summary>

**Auto Scaling Group Configuration**:
- **Minimum capacity**: 1 — Always have at least one instance to serve requests at off-peak times.
- **Maximum capacity**: 5 — Upper bound to prevent runaway cost from accidental scaling loops.
- **Desired capacity**: 1 (initial) — Start at minimum; scaling policies will adjust it.

**Scaling Policy**: **Scheduled Scaling** is ideal here because the traffic pattern is predictable and time-based:

```bash
# Scale up to 4 instances at 9 AM on weekdays
aws autoscaling put-scheduled-update-group-action \
    --auto-scaling-group-name my-asg \
    --scheduled-action-name scale-up-weekday \
    --recurrence "0 9 * * 1-5" \
    --desired-capacity 4

# Scale back to 1 instance at 6 PM on weekdays
aws autoscaling put-scheduled-update-group-action \
    --auto-scaling-group-name my-asg \
    --scheduled-action-name scale-down-weekday \
    --recurrence "0 18 * * 1-5" \
    --desired-capacity 1
```

**Recommended addition**: Add a **Target Tracking** policy on CPU utilization (e.g., maintain 60% average CPU) as a safety net for unexpected traffic spikes outside scheduled hours.

</details>

### Exercise 4: Pricing Model Decision

A data analytics team is planning to run a nightly batch processing cluster that:
- Runs every night from 11 PM to 5 AM (6 hours)
- Requires 20 `c5.2xlarge` instances during the run
- Can tolerate a run being interrupted and restarted (the job is checkpointed)
- Has been running consistently for 18 months

Which pricing model should they use for these instances, and approximately how much cost savings could they achieve compared to On-Demand? Explain your reasoning.

<details>
<summary>Show Answer</summary>

**Recommended pricing model**: **Spot Instances**

**Reasoning**:
- The job is checkpointed, so interruptions can be tolerated — this is the key prerequisite for Spot.
- Spot instances offer up to **90% discount** compared to On-Demand.
- For `c5.2xlarge` in `us-east-1`, On-Demand is approximately $0.34/hour. At 90% discount, Spot would be approximately $0.034/hour.

**Cost comparison** (approximate, prices vary):
- On-Demand: 20 instances × $0.34/hr × 6 hrs × 30 nights = **$1,224/month**
- Spot (90% off): 20 instances × $0.034/hr × 6 hrs × 30 nights = **$122/month**
- **Savings: ~$1,100/month (~$13,200/year)**

**Why not Reserved Instances?** Reserved Instances require 1–3 year commitment and are priced for continuous usage. These instances only run 6 hours/day. A Reserved Instance for a VM that runs 6/24 hours (25% utilization) would not be cost-effective compared to Spot.

**Best practice**: Use a Spot Fleet with multiple instance types and AZs to minimize interruption probability.

</details>

### Exercise 5: GCP Custom Machine Type vs AWS Equivalents

A GCP workload requires exactly 6 vCPUs and 20 GB of RAM. No standard GCP machine type has this exact configuration.

1. Write the `gcloud` command to create a custom machine type instance with these specifications in zone `asia-northeast3-a`.
2. What is the equivalent approach in AWS EC2, and why does AWS not offer custom instance types?

<details>
<summary>Show Answer</summary>

1. **GCP custom machine type**:
```bash
gcloud compute instances create custom-instance \
    --zone=asia-northeast3-a \
    --custom-cpu=6 \
    --custom-memory=20GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud
```

GCP requires custom memory to be a multiple of 256 MB. 20 GB (20480 MB) is a valid multiple.

2. **AWS equivalent approach**: AWS does not offer custom instance types. In AWS, you choose from a catalog of predefined types. For 6 vCPU / 20 GB RAM, you would look for the closest match:
   - `c5.xlarge` = 4 vCPU, 8 GB (too small)
   - `m5.2xlarge` = 8 vCPU, 32 GB (over-provisioned but closest balanced option)
   - `c5.2xlarge` = 8 vCPU, 16 GB (compute-focused, slightly over on CPU)

   **Why no custom types in AWS?** AWS optimizes its physical hardware and hypervisor configurations around specific instance families, enabling predictable performance guarantees and economies of scale. GCP uses a more flexible allocation model that allows arbitrary combinations within per-zone resource limits.

   **Practical impact**: GCP's custom machine types let you right-size precisely and avoid paying for unused vCPUs or RAM. In AWS, you typically over-provision to the nearest available size.

</details>

---

## References

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [GCP Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [GCP Machine Types](https://cloud.google.com/compute/docs/machine-types)
