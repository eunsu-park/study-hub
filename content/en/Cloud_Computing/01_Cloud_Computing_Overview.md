# Cloud Computing Overview

**Next**: [AWS/GCP Account Setup](./02_AWS_GCP_Account_Setup.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define cloud computing using NIST's five essential characteristics
2. Distinguish between IaaS, PaaS, and SaaS service models with real-world examples
3. Compare the four cloud deployment models (public, private, hybrid, multi-cloud)
4. Explain the shared responsibility model and how it shifts between service models
5. Identify the economic advantages of cloud over traditional on-premises infrastructure
6. Describe the major cloud providers and their market positioning

---

Cloud computing has fundamentally changed how organizations build and operate technology. Instead of purchasing servers and managing data centers, teams can spin up infrastructure in minutes and pay only for what they use. Understanding the core concepts, service models, and deployment models is essential for every engineer, architect, and decision-maker working with modern software systems.

> **Analogy — Renting a Generator vs. Building a Power Plant**: Running your own servers is like building a private power plant: massive upfront cost, you must maintain it yourself, and you pay even when demand is low. Cloud computing is like renting from the electric grid — you plug in, pay per kilowatt-hour, and the provider handles capacity. IaaS gives you raw electricity (VMs), PaaS gives you outlets with voltage regulators (managed platforms), and SaaS gives you a pre-wired appliance (finished software).

## 1. What is Cloud Computing?

Cloud computing is a service model that provides IT resources (servers, storage, networks, databases, etc.) on-demand through the internet.

### 1.1 Traditional Infrastructure vs Cloud

| Category | On-Premises (Traditional) | Cloud |
|------|------------------|----------|
| **Initial Cost** | High (hardware purchase) | Low (usage-based) |
| **Scalability** | Weeks to months | Minutes to hours |
| **Maintenance** | Self-managed | Provider-managed |
| **Risk** | Wasted unused resources | Pay only for what you use |
| **Responsibility** | Manage all layers directly | Shared responsibility model |

### 1.2 NIST's 5 Essential Characteristics of Cloud

Core characteristics of cloud computing defined by the National Institute of Standards and Technology (NIST):

1. **On-demand Self-service**
   - Users provision resources directly
   - Automated deployment without human intervention

2. **Broad Network Access**
   - Access through standard mechanisms over the network
   - Support for various client platforms

3. **Resource Pooling**
   - Resources shared through multi-tenant model
   - Physical location abstraction

4. **Rapid Elasticity**
   - Automatic scaling up/down based on demand
   - Resources appear unlimited to users

5. **Measured Service**
   - Usage monitoring and reporting
   - Transparent billing

---

## 2. Service Models: IaaS, PaaS, SaaS

### 2.1 Concept Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                        SaaS                                 │
│  (Software as a Service)                                    │
│  Examples: Gmail, Salesforce, Slack                         │
├─────────────────────────────────────────────────────────────┤
│                        PaaS                                 │
│  (Platform as a Service)                                    │
│  Examples: Heroku, App Engine, Elastic Beanstalk            │
├─────────────────────────────────────────────────────────────┤
│                        IaaS                                 │
│  (Infrastructure as a Service)                              │
│  Examples: EC2, Compute Engine, Azure VMs                   │
├─────────────────────────────────────────────────────────────┤
│                   Physical Infrastructure                    │
│  Data centers, servers, network equipment                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Responsibility Scope Comparison

| Layer | On-Premises | IaaS | PaaS | SaaS |
|------|-----------|------|------|------|
| Application | Customer | Customer | Customer | **Provider** |
| Data | Customer | Customer | Customer | Customer* |
| Runtime | Customer | Customer | **Provider** | **Provider** |
| Middleware | Customer | Customer | **Provider** | **Provider** |
| OS | Customer | Customer | **Provider** | **Provider** |
| Virtualization | Customer | **Provider** | **Provider** | **Provider** |
| Servers | Customer | **Provider** | **Provider** | **Provider** |
| Storage | Customer | **Provider** | **Provider** | **Provider** |
| Networking | Customer | **Provider** | **Provider** | **Provider** |

*Data management responsibility remains with the customer even in SaaS

### 2.3 Use Cases

**IaaS Best For:**
- When complete infrastructure control is needed
- Legacy application migration
- Development/test environments
- High-performance computing (HPC)

**PaaS Best For:**
- Rapid application development
- Microservices architecture
- API development
- Minimizing infrastructure management burden

**SaaS Best For:**
- Email and collaboration tools
- CRM, ERP systems
- Need for ready-to-use solutions

---

## 3. AWS vs GCP Comparison

### 3.1 Market Positioning

| Item | AWS | GCP |
|------|-----|-----|
| **Launch** | 2006 | 2008 |
| **Market Share** | ~32% (1st) | ~10% (3rd) |
| **Strengths** | Service diversity, ecosystem | Data analytics, ML/AI, pricing |
| **Service Count** | 200+ | 100+ |
| **Global Regions** | 30+ | 35+ |

### 3.2 Core Service Mapping

| Category | AWS | GCP |
|----------|-----|-----|
| **Virtual Machines** | EC2 | Compute Engine |
| **Serverless Functions** | Lambda | Cloud Functions |
| **Container Orchestration** | EKS | GKE |
| **Serverless Containers** | Fargate | Cloud Run |
| **Object Storage** | S3 | Cloud Storage |
| **Block Storage** | EBS | Persistent Disk |
| **Managed RDB** | RDS, Aurora | Cloud SQL, Spanner |
| **NoSQL (Key-Value)** | DynamoDB | Firestore |
| **Cache** | ElastiCache | Memorystore |
| **DNS** | Route 53 | Cloud DNS |
| **CDN** | CloudFront | Cloud CDN |
| **Load Balancer** | ELB (ALB/NLB) | Cloud Load Balancing |
| **VPC** | VPC | VPC |
| **IAM** | IAM | IAM |
| **Key Management** | KMS | Cloud KMS |
| **Secret Management** | Secrets Manager | Secret Manager |
| **Monitoring** | CloudWatch | Cloud Monitoring |
| **Logging** | CloudWatch Logs | Cloud Logging |
| **IaC** | CloudFormation | Deployment Manager |
| **CLI** | AWS CLI | gcloud CLI |

---

## 4. Pricing Model

### 4.1 Pricing Principles

Both platforms follow the **Pay-as-you-go** principle.

```
Total Cost = Computing + Storage + Network + Additional Services
```

### 4.2 Computing Pricing Options

| Option | AWS | GCP | Features |
|------|-----|-----|------|
| **On-Demand** | On-Demand | On-demand | Hourly/per-second, no commitment |
| **Reserved** | Reserved Instances | Committed Use | 1-3 year commitment, up to 72% discount |
| **Spot/Preemptible** | Spot Instances | Preemptible/Spot VMs | Up to 90% discount, can be interrupted |
| **Auto Discount** | - | Sustained Use | Automatic discount based on monthly usage |

### 4.3 Data Transfer Costs

```
┌─────────────────────────────────────────────────────────┐
│                        Cloud                            │
│                                                         │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐  │
│   │ Inbound │   Free   │  Same   │   Free   │Outbound│  │
│   │(Out→In) │ ────→   │ Region  │ ────→   │(In→Out)│  │
│   └─────────┘         └─────────┘         └─────────┘  │
│       Free             Free/Cheap           Charged      │
└─────────────────────────────────────────────────────────┘
```

- **Inbound**: Generally free
- **Within Same Region**: Free or inexpensive
- **Outbound**: Charged per GB (after monthly free quota)

### 4.4 Free Tier

**AWS Free Tier:**
- 12 months free: t2.micro EC2 (750 hours/month), 5GB S3, 750 hours RDS
- Always free: Lambda 1M requests/month, DynamoDB 25GB

**GCP Free Tier:**
- $300 credit for 90 days (new accounts)
- Always Free: e2-micro VM, 5GB Cloud Storage, Cloud Functions 2M invocations/month

---

## 5. Shared Responsibility Model

Cloud security responsibilities are shared between provider and customer.

### 5.1 Responsibility Distribution

```
┌────────────────────────────────────────────────────────────┐
│                Customer Responsibility (IN the cloud)       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Customer data                                      │  │
│  │  • Platform, applications, IAM                        │  │
│  │  • Operating system, network, firewall configuration  │  │
│  │  • Client-side data encryption                        │  │
│  │  • Server-side encryption (file system/data)          │  │
│  │  • Network traffic protection (encryption, integrity, │  │
│  │    authentication)                                    │  │
│  └──────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│                Provider Responsibility (OF the cloud)       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Global infrastructure (regions, AZs, edge          │  │
│  │    locations)                                         │  │
│  │  • Hardware (compute, storage, networking)            │  │
│  │  • Software (host OS, virtualization)                 │  │
│  │  • Physical security (data centers)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Responsibility by Service Type

| Service Type | Customer Responsibility | Provider Responsibility |
|------------|----------|------------|
| **IaaS (EC2)** | OS to application | Hardware, virtualization |
| **PaaS (Lambda)** | Code, data | Runtime, OS, infrastructure |
| **SaaS** | Data, access management | Almost everything |

---

## 6. Cloud Architecture Principles

### 6.1 Well-Architected Framework

Both AWS and GCP present similar design principles:

| Principle | Description |
|------|------|
| **Operational Excellence** | System execution and monitoring, continuous improvement |
| **Security** | Protecting data, systems, and assets |
| **Reliability** | Failure recovery, responding to demand changes |
| **Performance Efficiency** | Efficient resource usage, technology selection |
| **Cost Optimization** | Eliminating unnecessary costs, efficient spending |
| **Sustainability** | Minimizing environmental impact (emphasized by GCP) |

### 6.2 AWS Well-Architected Framework: The 6 Pillars

The AWS Well-Architected Framework provides a structured approach for evaluating architectures against cloud best practices. It is organized around six pillars, each with key design principles:

| Pillar | Description | Key Design Principles |
|--------|-------------|----------------------|
| **Operational Excellence** | Run and monitor systems to deliver business value, and continually improve processes and procedures | 1. Perform operations as code (IaC, runbooks) 2. Make frequent, small, reversible changes |
| **Security** | Protect data, systems, and assets through risk assessments and mitigation strategies | 1. Apply security at all layers (defense in depth) 2. Automate security best practices |
| **Reliability** | Ensure a workload performs its intended function correctly and consistently when expected | 1. Automatically recover from failure 2. Scale horizontally to increase aggregate availability |
| **Performance Efficiency** | Use computing resources efficiently to meet system requirements and maintain efficiency as demand changes | 1. Democratize advanced technologies (use managed services) 2. Go global in minutes |
| **Cost Optimization** | Avoid unnecessary costs and run systems at the lowest price point while meeting requirements | 1. Implement cloud financial management 2. Adopt a consumption model (pay only for what you use) |
| **Sustainability** | Minimize environmental impacts of running cloud workloads through energy-efficient practices | 1. Understand your impact (measure and track) 2. Maximize utilization (right-size resources) |

#### Well-Architected Review Process

A Well-Architected Review (WAR) is a structured evaluation of your workload against the framework:

```
┌────────────────────────────────────────────────────────────┐
│              Well-Architected Review Process                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Scope     → Identify the workload to review            │
│  2. Review    → Answer questions for each pillar           │
│                 (AWS WAR Tool provides guided questions)    │
│  3. Identify  → Find High Risk Issues (HRIs) and          │
│                 Medium Risk Issues (MRIs)                   │
│  4. Prioritize→ Rank improvements by business impact       │
│  5. Remediate → Create action plan and implement fixes     │
│  6. Measure   → Track improvements over time               │
│  7. Repeat    → Re-review periodically (quarterly or       │
│                 after major changes)                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

> **Tip**: AWS provides the **AWS Well-Architected Tool** in the console, which walks you through the review questions interactively and generates a prioritized improvement plan.

#### Cross-Provider Comparison: Architecture Frameworks

Each major cloud provider offers its own architecture framework. While the core principles overlap, emphasis differs:

| Aspect | AWS Well-Architected | GCP Architecture Framework | Azure Well-Architected |
|--------|---------------------|---------------------------|----------------------|
| **Pillars/Areas** | 6 pillars | 6 focus areas (same names) | 5 pillars (no Sustainability pillar separately) |
| **Review Tool** | AWS WAR Tool (console) | Architecture Framework review (docs-based) | Azure Advisor + WAF Assessment |
| **Unique Emphasis** | Extensive lens catalog (SaaS, serverless, ML, etc.) | Data-centric design, BigQuery-first analytics | Hybrid cloud (Azure Arc), enterprise integration |
| **Lenses/Modules** | 10+ specialized lenses | Industry-specific blueprints | Azure landing zones |
| **Automation** | WAR Tool API, custom lenses | Terraform blueprints, Config Connector | Azure Policy, Blueprints |
| **Community** | AWS Partner lens program | Google Cloud Architecture Center | Azure Architecture Center |

### 6.3 Disaster Recovery Strategies

Disaster Recovery (DR) planning ensures business continuity when failures occur. The two critical metrics are:

- **RPO (Recovery Point Objective)**: Maximum acceptable data loss measured in time. "How much data can we afford to lose?"
- **RTO (Recovery Time Objective)**: Maximum acceptable downtime. "How fast must we recover?"

```
         Cost & Complexity →
  Low ─────────────────────────── High

  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Backup & │  │  Pilot   │  │  Warm    │  │Multi-Site│
  │ Restore  │  │  Light   │  │ Standby  │  │ Active-  │
  │          │  │          │  │          │  │ Active   │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
  RTO: hours    RTO: 10s min  RTO: minutes  RTO: ~0
  RPO: hours    RPO: minutes  RPO: seconds  RPO: ~0

  ← Recovery Speed
  Slow ─────────────────────────── Fast
```

#### DR Strategy Comparison

| Strategy | Description | RPO | RTO | Relative Cost | Best For |
|----------|-------------|-----|-----|---------------|----------|
| **Backup & Restore** | Regular backups to another region; restore on disaster | Hours | Hours | $ (lowest) | Non-critical workloads, dev/test |
| **Pilot Light** | Minimal core infrastructure always running (DB replication); scale up on disaster | Minutes | 10s of minutes | $$ | Core systems needing faster recovery |
| **Warm Standby** | Scaled-down but fully functional copy of production; scale up on disaster | Seconds to minutes | Minutes | $$$ | Business-critical applications |
| **Multi-Site Active-Active** | Full production in 2+ regions; traffic distributed across all | Near-zero | Near-zero | $$$$ (highest) | Mission-critical, zero-downtime required |

#### AWS Services for Each DR Strategy

| Strategy | Key AWS Services |
|----------|-----------------|
| **Backup & Restore** | S3 (cross-region replication), AWS Backup, EBS Snapshots, RDS automated backups, Glacier for archives |
| **Pilot Light** | Route 53 (DNS failover), RDS read replicas (cross-region), AMIs copied to DR region, CloudFormation for rapid scale-up |
| **Warm Standby** | Auto Scaling (minimum capacity in DR), Elastic Load Balancing, RDS Multi-AZ + cross-region read replica, Route 53 health checks |
| **Multi-Site Active-Active** | Route 53 latency-based routing, Global Accelerator, DynamoDB Global Tables, Aurora Global Database, S3 cross-region replication |

> **Analogy -- DR Strategies as Home Safety**: **Backup & Restore** is like having homeowner's insurance -- you rebuild after disaster but it takes time. **Pilot Light** is keeping a generator in the garage ready to start. **Warm Standby** is a second home partially furnished and heated. **Multi-Site Active-Active** is living in two fully furnished homes simultaneously, using whichever is more convenient.

#### Choosing the Right Strategy

The right DR strategy depends on your business requirements:

```
┌──────────────────────────────────────────────────────────┐
│           DR Strategy Decision Factors                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Business Impact Analysis                             │
│     └─ How much revenue/reputation lost per hour of     │
│        downtime?                                         │
│                                                          │
│  2. Data Criticality                                     │
│     └─ Can you afford to lose any data? How much?       │
│                                                          │
│  3. Budget Constraints                                   │
│     └─ DR cost should be proportional to business risk  │
│                                                          │
│  4. Regulatory Requirements                              │
│     └─ Some industries mandate specific RPO/RTO          │
│        (e.g., financial services, healthcare)            │
│                                                          │
│  5. Technical Complexity                                 │
│     └─ Team capability to maintain DR infrastructure    │
│                                                          │
│  Rule of Thumb:                                          │
│  Start with Backup & Restore, and move up the           │
│  spectrum only as business requirements demand it.       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.4 Design Best Practices

```
1. Design for Failure
   - Eliminate single points of failure
   - Multi-AZ/region deployment
   - Automatic recovery

2. Loose Coupling
   - Microservices architecture
   - Utilize message queues
   - API-based communication

3. Elasticity
   - Leverage auto-scaling
   - Consider serverless
   - Prepare for unpredictable loads

4. Security by Design
   - Principle of least privilege
   - Apply encryption by default
   - Network isolation
```

---

## 7. Learning Roadmap

### 7.1 Beginner Path (1-2 weeks)

```
[Cloud Concepts] → [Account Creation] → [Console Navigation] → [First VM] → [S3/GCS Practice]
```

### 7.2 Basic Practical Path (1-2 months)

```
[VPC Networking] → [Load Balancer] → [RDS/Cloud SQL] → [IAM Policies] → [Monitoring]
```

### 7.3 Advanced Path (3-6 months)

```
[Container/K8s] → [Serverless] → [Terraform IaC] → [CI/CD] → [Cost Optimization]
```

---

## 8. Next Steps

- [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) - Account creation and initial setup
- [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) - Understanding global infrastructure

---

## Exercises

### Exercise 1: Classify the Service Model

For each of the following scenarios, identify whether it is IaaS, PaaS, or SaaS and explain your reasoning:

1. A startup deploys their Node.js web app to Heroku by pushing code via `git push`.
2. A company rents EC2 instances and installs their own database software on them.
3. A sales team uses Salesforce CRM to manage their customer pipeline.
4. A data engineering team uses Google App Engine to host their Python ETL service.

<details>
<summary>Show Answer</summary>

1. **PaaS** — Heroku manages the runtime, OS, and infrastructure. The developer only provides application code.
2. **IaaS** — EC2 provides raw virtual machines. The company is responsible for the OS, installed software, and configuration.
3. **SaaS** — Salesforce is fully managed software delivered over the internet. Users consume it without managing any underlying infrastructure.
4. **PaaS** — App Engine manages the runtime environment. The team deploys code; Google handles scaling, OS patching, and infrastructure.

</details>

### Exercise 2: Shared Responsibility Mapping

A company hosts a web application on AWS EC2. Match each security task to the correct responsible party (AWS or Customer):

| Task | Responsible Party |
|------|-------------------|
| Patching the host hypervisor | ? |
| Installing OS security updates on the EC2 instance | ? |
| Encrypting application data stored in the database | ? |
| Ensuring physical access controls at the data center | ? |
| Configuring security group firewall rules | ? |

<details>
<summary>Show Answer</summary>

| Task | Responsible Party |
|------|-------------------|
| Patching the host hypervisor | **AWS** — AWS manages the virtualization layer |
| Installing OS security updates on the EC2 instance | **Customer** — The OS on IaaS is the customer's responsibility |
| Encrypting application data stored in the database | **Customer** — Data protection is always the customer's responsibility |
| Ensuring physical access controls at the data center | **AWS** — Physical data center security is AWS's responsibility |
| Configuring security group firewall rules | **Customer** — Network/firewall configuration on IaaS is customer-managed |

**Key insight**: In IaaS, AWS is responsible *of* the cloud (physical infrastructure and virtualization), while the customer is responsible *in* the cloud (OS and everything above it).

</details>

### Exercise 3: Disaster Recovery Strategy Selection

A fintech company processes real-time payment transactions. A 30-minute outage would result in approximately $500,000 in lost revenue, and regulators require that no transaction data be lost. Their budget for DR infrastructure is substantial.

1. Which DR strategy should they choose? Justify your answer using RPO and RTO requirements.
2. Which AWS services would be central to this strategy?

<details>
<summary>Show Answer</summary>

1. **Multi-Site Active-Active** is the appropriate strategy.
   - **RPO requirement**: No data loss → RPO ≈ 0. Only Active-Active meets this with real-time replication.
   - **RTO requirement**: Even a 30-minute outage is extremely costly → RTO ≈ 0. Active-Active routes traffic away from a failed region in seconds.
   - The substantial budget justifies the higher cost of running full production environments in multiple regions.

2. **Key AWS services**:
   - **Route 53** with latency-based routing or health check failover to distribute traffic across regions
   - **DynamoDB Global Tables** for multi-region, multi-active database replication with near-zero RPO
   - **Aurora Global Database** if relational data is needed, with sub-second replication lag
   - **AWS Global Accelerator** for consistent, low-latency global routing
   - **S3 Cross-Region Replication** for any object storage assets

</details>

### Exercise 4: NIST Characteristics in Practice

For each scenario below, identify which of the five NIST essential characteristics of cloud computing is being demonstrated:

1. An e-commerce site automatically adds 50 more servers during a Black Friday sale and scales back down afterward.
2. A developer provisions a new PostgreSQL database in 3 minutes through the AWS Console without contacting anyone.
3. AWS charges a customer based on exact GB-hours of storage used each month, visible in a detailed billing dashboard.
4. Multiple customers' workloads run on the same physical servers, but each customer sees logically isolated resources.

<details>
<summary>Show Answer</summary>

1. **Rapid Elasticity** — The system scales resources up and down automatically to match demand.
2. **On-Demand Self-Service** — The developer provisions resources without human interaction with the provider.
3. **Measured Service** — Usage is monitored, controlled, and reported transparently, enabling pay-per-use billing.
4. **Resource Pooling** — Physical resources are shared across multiple tenants (multi-tenancy) with logical isolation.

</details>

### Exercise 5: Cost Model Analysis

A company is deciding between On-Demand, Reserved Instances, and Spot Instances for three different workloads. Recommend the best pricing option for each workload and explain why:

1. A production web server that must run 24/7 with no interruption for at least the next 2 years.
2. A batch data processing job that runs nightly for 3 hours and can be restarted if interrupted.
3. A new microservice being tested in development that will run unpredictably over the next 2 weeks.

<details>
<summary>Show Answer</summary>

1. **Reserved Instances (1- or 3-year commitment)** — The workload runs continuously for a known duration. Committing upfront saves up to 72% compared to On-Demand. Since it cannot be interrupted, Spot is not viable.

2. **Spot Instances** — Batch jobs that can tolerate interruption and be restarted are the ideal use case for Spot. The up-to-90% discount significantly reduces cost for a workload that does not require guaranteed availability at a specific time.

3. **On-Demand** — Development/test workloads have unpredictable and short-term usage patterns. There is no commitment benefit from Reserved Instances, and the workload may not be interruptible in the middle of a test run. On-Demand provides maximum flexibility.

</details>

---

## References

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework)
- [NIST Cloud Computing Definition](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf)
