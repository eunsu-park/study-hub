# VPC (Virtual Private Cloud)

**Previous**: [Block and File Storage](./08_Block_and_File_Storage.md) | **Next**: [Load Balancing and CDN](./10_Load_Balancing_CDN.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose and architecture of a Virtual Private Cloud
2. Compare AWS VPC (regional scope) with GCP VPC (global scope) and their subnet models
3. Design a VPC with public and private subnets using appropriate CIDR blocks
4. Configure route tables, internet gateways, and NAT gateways for traffic routing
5. Apply security groups and network ACLs to control inbound and outbound traffic
6. Implement VPC peering and VPN connections for cross-network communication
7. Distinguish between public, private, and isolated subnet patterns

---

Networking is the foundation of every cloud deployment. A VPC gives you an isolated, software-defined network where you control IP addressing, subnets, routing, and access rules. Misconfigured networking is one of the top causes of security incidents and outages in the cloud, so understanding VPC design is essential before deploying any production workload.

> **Analogy — Private Rooms in a Shared Hotel**: A public cloud is like a large hotel. Without a VPC, all guests share the same hallways and can knock on any door. A VPC is like booking an entire private floor with its own elevator key — guests on your floor can move freely between rooms (subnets), but outsiders can only enter through the front desk (internet gateway) after passing security (security groups and NACLs).

## 1. VPC Overview

### 1.1 What is VPC?

VPC is a logically isolated virtual network within the cloud.

**Core Concepts:**
- Define your own IP address range
- Divide into subnets
- Control traffic with routing tables
- Access control with security groups/firewalls

### 1.2 AWS vs GCP VPC Differences

| Category | AWS VPC | GCP VPC |
|------|---------|---------|
| **Scope** | Regional | **Global** |
| **Subnet Scope** | Availability Zone (AZ) | Regional |
| **Default VPC** | 1 per region | 1 per project (default) |
| **Peering** | Cross-region possible | Global automatic |
| **IP Range** | Fixed at creation | Subnets can be added |

```
AWS VPC Structure:
┌──────────────────────────────────────────────────────────────┐
│  VPC (Region: ap-northeast-2)                                │
│  CIDR: 10.0.0.0/16                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-a (AZ-a)     │  │ Subnet-b (AZ-b)     │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘

GCP VPC Structure:
┌──────────────────────────────────────────────────────────────┐
│  VPC (Global)                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-asia         │  │ Subnet-us           │            │
│  │ (asia-northeast3)   │  │ (us-central1)       │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Subnets

### 2.1 Public vs Private Subnets

| Type | Internet Access | Use Case |
|------|-----------|------|
| **Public** | Direct access | Web servers, Bastion |
| **Private** | Only through NAT | Applications, Databases |

### 2.2 AWS Subnet Creation

```bash
# 1. Create VPC
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=MyVPC}]'

# 2. Create public subnet
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.1.0/24 \
    --availability-zone ap-northeast-2a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-1}]'

# 3. Create private subnet
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.10.0/24 \
    --availability-zone ap-northeast-2a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-1}]'

# 4. Auto-assign public IP (public subnet)
aws ec2 modify-subnet-attribute \
    --subnet-id subnet-public \
    --map-public-ip-on-launch
```

### 2.3 GCP Subnet Creation

```bash
# 1. Create custom-mode VPC
gcloud compute networks create my-vpc \
    --subnet-mode=custom

# 2. Create subnet (Seoul)
gcloud compute networks subnets create subnet-asia \
    --network=my-vpc \
    --region=asia-northeast3 \
    --range=10.0.1.0/24

# 3. Create subnet (US)
gcloud compute networks subnets create subnet-us \
    --network=my-vpc \
    --region=us-central1 \
    --range=10.0.2.0/24

# 4. Enable Private Google Access
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-private-ip-google-access
```

---

## 3. Internet Gateway

### 3.1 AWS Internet Gateway

```bash
# 1. Create IGW
aws ec2 create-internet-gateway \
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=MyIGW}]'

# 2. Attach to VPC
aws ec2 attach-internet-gateway \
    --internet-gateway-id igw-12345678 \
    --vpc-id vpc-12345678

# 3. Add route to routing table
aws ec2 create-route \
    --route-table-id rtb-12345678 \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id igw-12345678

# 4. Associate routing table with public subnet
aws ec2 associate-route-table \
    --route-table-id rtb-12345678 \
    --subnet-id subnet-public
```

### 3.2 GCP Internet Access

GCP allows internet access without a separate internet gateway if an external IP is present.

```bash
# Assign external IP (at instance creation)
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --network=my-vpc \
    --subnet=subnet-asia \
    --address=EXTERNAL_IP  # or omit to assign an ephemeral IP

# Reserve static IP
gcloud compute addresses create my-static-ip \
    --region=asia-northeast3
```

---

## 4. NAT Gateway

Allows instances in private subnets to access the internet.

### 4.1 AWS NAT Gateway

```bash
# 1. Allocate Elastic IP
aws ec2 allocate-address --domain vpc

# 2. Create NAT Gateway (in public subnet)
aws ec2 create-nat-gateway \
    --subnet-id subnet-public \
    --allocation-id eipalloc-12345678 \
    --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=MyNAT}]'

# 3. Add route to private routing table
aws ec2 create-route \
    --route-table-id rtb-private \
    --destination-cidr-block 0.0.0.0/0 \
    --nat-gateway-id nat-12345678

# 4. Associate routing table with private subnet
aws ec2 associate-route-table \
    --route-table-id rtb-private \
    --subnet-id subnet-private
```

### 4.2 GCP Cloud NAT

```bash
# 1. Create Cloud Router
gcloud compute routers create my-router \
    --network=my-vpc \
    --region=asia-northeast3

# 2. Create Cloud NAT
gcloud compute routers nats create my-nat \
    --router=my-router \
    --region=asia-northeast3 \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
```

---

## 5. Security Groups / Firewalls

### 5.1 AWS Security Groups

Security groups are instance-level **stateful** firewalls.

```bash
# Create security group
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group" \
    --vpc-id vpc-12345678

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 203.0.113.0/24  # Allow only specific IP

# Allow traffic from another security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-db \
    --protocol tcp \
    --port 3306 \
    --source-group sg-web

# List rules
aws ec2 describe-security-groups --group-ids sg-12345678
```

### 5.2 GCP Firewall Rules

GCP firewall rules operate at the VPC level and target resources using **tags** or service accounts.

```bash
# Allow HTTP traffic (tag-based)
gcloud compute firewall-rules create allow-http \
    --network=my-vpc \
    --allow=tcp:80,tcp:443 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0

# Allow SSH (specific IP)
gcloud compute firewall-rules create allow-ssh \
    --network=my-vpc \
    --allow=tcp:22 \
    --target-tags=ssh-server \
    --source-ranges=203.0.113.0/24

# Allow internal communication
gcloud compute firewall-rules create allow-internal \
    --network=my-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8

# List rules
gcloud compute firewall-rules list --filter="network:my-vpc"

# Delete rule
gcloud compute firewall-rules delete allow-http
```

### 5.3 AWS NACL (Network ACL)

NACL is a subnet-level **stateless** firewall.

```bash
# Create NACL
aws ec2 create-network-acl \
    --vpc-id vpc-12345678 \
    --tag-specifications 'ResourceType=network-acl,Tags=[{Key=Name,Value=MyNACL}]'

# Add inbound rule (priority by rule number)
aws ec2 create-network-acl-entry \
    --network-acl-id acl-12345678 \
    --ingress \
    --rule-number 100 \
    --protocol tcp \
    --port-range From=80,To=80 \
    --cidr-block 0.0.0.0/0 \
    --rule-action allow

# Outbound rule also required (stateless)
aws ec2 create-network-acl-entry \
    --network-acl-id acl-12345678 \
    --egress \
    --rule-number 100 \
    --protocol tcp \
    --port-range From=1024,To=65535 \
    --cidr-block 0.0.0.0/0 \
    --rule-action allow
```

---

## 6. VPC Peering

### 6.1 AWS VPC Peering

```bash
# 1. Request peering connection
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-requester \
    --peer-vpc-id vpc-accepter \
    --peer-region ap-northeast-1  # if different region

# 2. Accept peering connection
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-12345678

# 3. Add routes to routing tables on both VPCs
# Requester VPC
aws ec2 create-route \
    --route-table-id rtb-requester \
    --destination-cidr-block 10.1.0.0/16 \
    --vpc-peering-connection-id pcx-12345678

# Accepter VPC
aws ec2 create-route \
    --route-table-id rtb-accepter \
    --destination-cidr-block 10.0.0.0/16 \
    --vpc-peering-connection-id pcx-12345678
```

### 6.2 GCP VPC Peering

```bash
# 1. Create peering from first VPC
gcloud compute networks peerings create peer-vpc1-to-vpc2 \
    --network=vpc1 \
    --peer-network=vpc2

# 2. Create peering from second VPC (both sides required)
gcloud compute networks peerings create peer-vpc2-to-vpc1 \
    --network=vpc2 \
    --peer-network=vpc1

# Routing is added automatically
```

---

## 7. Private Endpoints

Access AWS/GCP services without going through the internet.

### 7.1 AWS VPC Endpoints

**Gateway Endpoint (S3, DynamoDB):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.s3 \
    --route-table-ids rtb-12345678
```

**Interface Endpoint (Other Services):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.secretsmanager \
    --vpc-endpoint-type Interface \
    --subnet-ids subnet-12345678 \
    --security-group-ids sg-12345678
```

### 7.2 GCP Private Service Connect

```bash
# Enable Private Google Access
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-private-ip-google-access

# Private Service Connect endpoint
gcloud compute addresses create psc-endpoint \
    --region=asia-northeast3 \
    --subnet=subnet-asia \
    --purpose=PRIVATE_SERVICE_CONNECT
```

---

## 8. Common VPC Architectures

### 8.1 3-Tier Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  VPC (10.0.0.0/16)                                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Public Subnets (10.0.1.0/24, 10.0.2.0/24)               ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │    ALB      │  │   Bastion   │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Private Subnets - App (10.0.10.0/24, 10.0.11.0/24)      ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │   App-1     │  │   App-2     │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Private Subnets - DB (10.0.20.0/24, 10.0.21.0/24)       ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │  DB Primary │  │  DB Standby │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │     IGW      │  │   NAT GW     │                         │
│  └──────────────┘  └──────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 AWS VPC Example (Terraform)

```hcl
# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "main-vpc" }
}

# Public Subnets
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "public-${count.index + 1}" }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "private-${count.index + 1}" }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

# NAT Gateway
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
}
```

---

## 9. Troubleshooting

### 9.1 Connection Issue Checklist

```
□ Check security group inbound rules
□ Check NACL rules (AWS)
□ Check firewall rules (GCP)
□ Check routing tables
□ Verify internet gateway attachment
□ Check NAT gateway status
□ Verify instance has public IP
□ Check VPC peering routing
```

### 9.2 Debugging Commands

**AWS:**
```bash
# Enable VPC Flow Logs
aws ec2 create-flow-logs \
    --resource-type VPC \
    --resource-ids vpc-12345678 \
    --traffic-type ALL \
    --log-destination-type cloud-watch-logs \
    --log-group-name vpc-flow-logs

# Reachability Analyzer
aws ec2 create-network-insights-path \
    --source i-source \
    --destination i-destination \
    --destination-port 80 \
    --protocol tcp
```

**GCP:**
```bash
# Enable VPC Flow Logs
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-flow-logs

# Connectivity Tests
gcloud network-management connectivity-tests create my-test \
    --source-instance=projects/PROJECT/zones/ZONE/instances/source \
    --destination-instance=projects/PROJECT/zones/ZONE/instances/dest \
    --destination-port=80 \
    --protocol=TCP
```

---

## 10. Next Steps

- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - Load Balancing
- [14_Security_Services.md](./14_Security_Services.md) - Security Details

---

## Exercises

### Exercise 1: CIDR Block Planning

A company is setting up an AWS VPC for a 3-tier web application (web, application, database). They plan to use the CIDR block `10.100.0.0/16` for the VPC in the `ap-northeast-2` region (3 AZs: a, b, c).

Design the subnet layout: create 3 public subnets (one per AZ for web servers) and 3 private subnets (one per AZ for application/database servers). Assign CIDR blocks that divide the space evenly and leave room for growth.

<details>
<summary>Show Answer</summary>

A `/16` VPC provides 65,536 addresses. Using `/24` subnets (256 addresses each) gives plenty of room and is a common standard for simplicity.

| Subnet Name | AZ | CIDR | Purpose |
|-------------|-----|------|---------|
| public-subnet-a | ap-northeast-2a | 10.100.1.0/24 | Web servers (public) |
| public-subnet-b | ap-northeast-2b | 10.100.2.0/24 | Web servers (public) |
| public-subnet-c | ap-northeast-2c | 10.100.3.0/24 | Web servers (public) |
| private-subnet-a | ap-northeast-2a | 10.100.11.0/24 | App/DB servers (private) |
| private-subnet-b | ap-northeast-2b | 10.100.12.0/24 | App/DB servers (private) |
| private-subnet-c | ap-northeast-2c | 10.100.13.0/24 | App/DB servers (private) |

**Design principles**:
- Public subnets use `.1.x`–`.3.x` range for easy identification.
- Private subnets use `.11.x`–`.13.x` range, grouped logically away from public.
- 6 subnets use only a small fraction of the `/16` space — hundreds of additional subnets are available for future services (additional tiers, management, monitoring, etc.).
- AWS reserves 5 IP addresses per subnet (first 4 + last 1), leaving 251 usable IPs per `/24`.

</details>

### Exercise 2: Security Group vs Network ACL

A web application has:
- Web servers in public subnets (port 80/443 inbound)
- Application servers in private subnets (port 8080, only from web tier)
- Database servers in private subnets (port 5432, only from app tier)

For the application server security group, write the inbound rules (describe the source, port, and protocol). Then explain one key difference between security groups and NACLs.

<details>
<summary>Show Answer</summary>

**Application server security group — inbound rules**:

| Rule # | Type | Protocol | Port | Source | Description |
|--------|------|----------|------|--------|-------------|
| 1 | Custom TCP | TCP | 8080 | Web Server SG ID | Allow traffic only from web servers |
| 2 | SSH | TCP | 22 | Bastion SG ID (or admin CIDR) | Allow admin SSH from bastion host |

**AWS CLI example**:
```bash
# Allow port 8080 from the web server security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-app-server \
    --protocol tcp \
    --port 8080 \
    --source-group sg-web-server
```

**Key difference between security groups and NACLs**:

| | Security Group | Network ACL |
|--|---------------|------------|
| **State** | **Stateful** — if inbound traffic is allowed, the return traffic is automatically allowed | **Stateless** — both inbound AND outbound rules must explicitly allow traffic in both directions |
| **Scope** | Applied to individual EC2 instances/ENIs | Applied to entire subnets |
| **Rules** | Allow rules only (no explicit deny) | Allow and deny rules (processed in number order) |

**Practical implication of statelessness**: If you have a NACL allowing inbound HTTP (port 80), you must also add an outbound rule for ephemeral ports (1024–65535) to allow the response packets. Security groups handle this automatically.

</details>

### Exercise 3: NAT Gateway Purpose and Configuration

A private subnet contains EC2 instances running a web application. The instances need to download software packages from the internet during startup, but must NOT be directly accessible from the internet.

1. Which AWS component enables outbound-only internet access for private subnet instances?
2. What route table entry is needed?
3. Write the AWS CLI commands to create the necessary component and configure routing.

<details>
<summary>Show Answer</summary>

1. **NAT Gateway** — Provides outbound-only internet connectivity for private subnet instances. It sits in a public subnet and has a public IP (Elastic IP). Private instances route their outbound internet traffic through the NAT Gateway; the NAT Gateway forwards the traffic and handles the response, but initiating traffic from the internet to private instances is blocked.

2. **Route table entry needed** for the private subnet's route table:
   - Destination: `0.0.0.0/0`
   - Target: `nat-xxxxxxxx` (NAT Gateway ID)

3. **AWS CLI commands**:
```bash
# Step 1: Allocate an Elastic IP for the NAT Gateway
aws ec2 allocate-address --domain vpc

# Step 2: Create NAT Gateway in the PUBLIC subnet
# (Replace eipalloc-xxx with the allocation ID from Step 1)
aws ec2 create-nat-gateway \
    --subnet-id subnet-public-a \
    --allocation-id eipalloc-xxx \
    --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-gw-a}]'

# Step 3: Add default route to private route table pointing to NAT Gateway
# (Replace rtb-private and nat-xxx with actual IDs)
aws ec2 create-route \
    --route-table-id rtb-private \
    --destination-cidr-block 0.0.0.0/0 \
    --nat-gateway-id nat-xxx
```

**Cost note**: NAT Gateways are not free. They charge per hour (~$0.045/hour) plus per GB of data processed. For cost-sensitive development environments, consider NAT Instances (free-tier eligible) as a cheaper alternative.

</details>

### Exercise 4: VPC Peering Scenario

Company A has VPC-A (`10.0.0.0/16`) in `ap-northeast-2` and Company B has VPC-B (`172.16.0.0/16`) in the same region. They need to allow their application servers to communicate directly via private IP.

1. What is VPC Peering, and what are its key limitations?
2. Describe the steps required to establish the peering connection.

<details>
<summary>Show Answer</summary>

1. **VPC Peering** is a networking connection between two VPCs that allows traffic to be routed between them using private IPv4 or IPv6 addresses. Instances in either VPC can communicate with each other as if they are within the same network.

   **Key limitations**:
   - **No transitive routing**: If VPC-A is peered with VPC-B and VPC-B is peered with VPC-C, VPC-A cannot reach VPC-C through VPC-B. Each pair requires its own direct peering.
   - **No overlapping CIDR blocks**: VPC-A (`10.0.0.0/16`) and VPC-B (`172.16.0.0/16`) can peer because they don't overlap. If both used `10.0.0.0/16`, peering would be impossible.
   - **Single-region by default** (cross-region peering is supported but adds latency and data transfer costs).

2. **Steps to establish VPC Peering**:
```bash
# Step 1: Create peering request (from VPC-A to VPC-B)
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-A \
    --peer-vpc-id vpc-B \
    --peer-region ap-northeast-2

# Step 2: Accept the peering request (from VPC-B's owner)
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-xxx

# Step 3: Update route tables on BOTH sides
# In VPC-A's route table: route to VPC-B's CIDR via the peering connection
aws ec2 create-route \
    --route-table-id rtb-A \
    --destination-cidr-block 172.16.0.0/16 \
    --vpc-peering-connection-id pcx-xxx

# In VPC-B's route table: route to VPC-A's CIDR
aws ec2 create-route \
    --route-table-id rtb-B \
    --destination-cidr-block 10.0.0.0/16 \
    --vpc-peering-connection-id pcx-xxx

# Step 4: Update security groups on both sides to allow traffic
# from the peer VPC's CIDR block
```

</details>

### Exercise 5: Three-Tier Architecture Design

Draw or describe the complete VPC architecture for a three-tier web application with high availability across two AZs. Include: subnets, internet gateway, NAT gateways, security groups, and routing.

<details>
<summary>Show Answer</summary>

```
VPC: 10.0.0.0/16  (Region: ap-northeast-2)
│
├── Internet Gateway (attached to VPC)
│
├── AZ-a (ap-northeast-2a)
│   ├── Public Subnet 10.0.1.0/24
│   │   ├── Application Load Balancer (node)
│   │   ├── NAT Gateway (with Elastic IP)
│   │   └── Route table: 0.0.0.0/0 → IGW
│   │
│   ├── Private Subnet (App) 10.0.11.0/24
│   │   ├── EC2 App Servers (Auto Scaling Group)
│   │   └── Route table: 0.0.0.0/0 → NAT-GW-a
│   │
│   └── Private Subnet (DB) 10.0.21.0/24
│       ├── RDS Primary
│       └── Route table: local only (no internet route)
│
└── AZ-b (ap-northeast-2b)
    ├── Public Subnet 10.0.2.0/24
    │   ├── ALB (node)
    │   └── NAT Gateway (with Elastic IP) [redundant]
    │
    ├── Private Subnet (App) 10.0.12.0/24
    │   ├── EC2 App Servers
    │   └── Route table: 0.0.0.0/0 → NAT-GW-b
    │
    └── Private Subnet (DB) 10.0.22.0/24
        ├── RDS Standby (Multi-AZ)
        └── Route table: local only

Security Groups:
- ALB-SG: inbound 80/443 from 0.0.0.0/0
- App-SG: inbound 8080 from ALB-SG only
- DB-SG: inbound 5432 from App-SG only
```

**Design rationale**:
- Each tier is in a separate subnet to enable distinct security group policies.
- Two NAT Gateways (one per AZ) ensure private subnets in both AZs have internet access even if one AZ fails.
- Database subnets have no internet route — DB instances cannot initiate or receive internet traffic under any circumstances.
- The ALB spans both public subnets, routing traffic to healthy app servers in both AZs.

</details>

---

## References

- [AWS VPC Documentation](https://docs.aws.amazon.com/vpc/)
- [GCP VPC Documentation](https://cloud.google.com/vpc/docs)
- [AWS VPC Best Practices](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-best-practices.html)
- [Networking/](../Networking/) - Networking Fundamentals
