# Cloud Computing Study Guide (AWS & GCP)

## Introduction

This folder contains cloud computing learning materials that allow you to study AWS (Amazon Web Services) and GCP (Google Cloud Platform) in parallel comparison. By explaining the core services of both platforms in a corresponding manner, you can easily transition from one platform to another once you learn one.

**Target Audience**: Cloud beginners to intermediate level (practical basics)

---

## File List

| Filename | Topic | Difficulty | AWS Services | GCP Services |
|--------|------|--------|-----------|-----------|
| [01_Cloud_Computing_Overview.md](./01_Cloud_Computing_Overview.md) | Cloud Overview, IaaS/PaaS/SaaS | ⭐ | - | - |
| [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) | Account Creation, Console Navigation, MFA | ⭐ | Console | Console |
| [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) | Regions, Availability Zones, Global Infrastructure | ⭐⭐ | Regions/AZs | Regions/Zones |
| [04_Virtual_Machines.md](./04_Virtual_Machines.md) | Virtual Machine Creation, SSH Connection | ⭐⭐ | EC2 | Compute Engine |
| [05_Serverless_Functions.md](./05_Serverless_Functions.md) | Serverless Functions, Triggers | ⭐⭐⭐ | Lambda | Cloud Functions |
| [06_Container_Services.md](./06_Container_Services.md) | Containers, Kubernetes | ⭐⭐⭐ | ECS, EKS, Fargate | GKE, Cloud Run |
| [07_Object_Storage.md](./07_Object_Storage.md) | Object Storage, Static Hosting | ⭐⭐ | S3 | Cloud Storage |
| [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) | Block/File Storage, Snapshots | ⭐⭐⭐ | EBS, EFS | PD, Filestore |
| [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) | VPC, Subnets, NAT | ⭐⭐⭐ | VPC | VPC |
| [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) | Load Balancers, CDN | ⭐⭐⭐ | ELB, CloudFront | Cloud LB, CDN |
| [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) | Managed RDB, Replication | ⭐⭐⭐ | RDS, Aurora | Cloud SQL |
| [12_NoSQL_Databases.md](./12_NoSQL_Databases.md) | NoSQL, Cache Services | ⭐⭐⭐ | DynamoDB, ElastiCache | Firestore, Memorystore |
| [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) | IAM Users, Roles, Policies | ⭐⭐⭐ | IAM | IAM |
| [14_Security_Services.md](./14_Security_Services.md) | Security Groups, KMS, Secret Management | ⭐⭐⭐⭐ | SG, KMS, Secrets Manager | Firewall, KMS, Secret Manager |
| [15_CLI_and_SDK.md](./15_CLI_and_SDK.md) | CLI Installation, SDK Usage | ⭐⭐ | AWS CLI, boto3 | gcloud, google-cloud |
| [16_Infrastructure_as_Code.md](./16_Infrastructure_as_Code.md) | Terraform Basics, Modules | ⭐⭐⭐⭐ | Terraform | Terraform |
| [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) | Monitoring, Logging, Cost Management | ⭐⭐⭐ | CloudWatch, Cost Explorer | Cloud Monitoring, Billing |

**Total Lessons**: 17

---

## Learning Path

### Phase 1: Cloud Fundamentals (01-03)
1. **Understanding Cloud Concepts**: 01_Cloud_Computing_Overview
2. **Account Setup**: 02_AWS_GCP_Account_Setup
3. **Understanding Infrastructure Architecture**: 03_Regions_Availability_Zones

### Phase 2: Compute Services (04-06)
4. **Virtual Machines**: 04_Virtual_Machines (EC2 / Compute Engine)
5. **Serverless**: 05_Serverless_Functions (Lambda / Cloud Functions)
6. **Containers**: 06_Container_Services (EKS / GKE)

### Phase 3: Storage (07-08)
7. **Object Storage**: 07_Object_Storage (S3 / Cloud Storage)
8. **Block/File Storage**: 08_Block_and_File_Storage (EBS / Persistent Disk)

### Phase 4: Networking (09-10)
9. **VPC Design**: 09_Virtual_Private_Cloud
10. **Load Balancing & CDN**: 10_Load_Balancing_CDN

### Phase 5: Databases (11-12)
11. **Relational Databases**: 11_Managed_Relational_DB (RDS / Cloud SQL)
12. **NoSQL Databases**: 12_NoSQL_Databases (DynamoDB / Firestore)

### Phase 6: Security (13-14)
13. **IAM**: 13_Identity_Access_Management
14. **Security Services**: 14_Security_Services

### Phase 7: DevOps & IaC (15-16)
15. **CLI/SDK**: 15_CLI_and_SDK
16. **Infrastructure as Code**: 16_Infrastructure_as_Code (Terraform)

### Phase 8: Operations (17)
17. **Monitoring & Cost**: 17_Monitoring_Logging_Cost

---

## AWS vs GCP Service Mapping Summary

| Category | AWS | GCP |
|----------|-----|-----|
| **Compute** | EC2, Lambda, ECS, EKS | Compute Engine, Cloud Functions, Cloud Run, GKE |
| **Storage** | S3, EBS, EFS | Cloud Storage, Persistent Disk, Filestore |
| **Networking** | VPC, ELB, CloudFront | VPC, Cloud Load Balancing, Cloud CDN |
| **Databases** | RDS, Aurora, DynamoDB | Cloud SQL, Spanner, Firestore |
| **Security** | IAM, KMS, Secrets Manager | IAM, Cloud KMS, Secret Manager |
| **Monitoring** | CloudWatch, X-Ray | Cloud Monitoring, Cloud Trace |
| **IaC** | CloudFormation | Deployment Manager |
| **CLI** | AWS CLI | gcloud CLI |

---

## Practice Environment Setup

### Required Tools

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"  # macOS
# Or: pip install awscli

# Install GCP gcloud CLI
curl https://sdk.cloud.google.com | bash

# Install Terraform
brew install terraform  # macOS
# Or: https://www.terraform.io/downloads
```

### Free Tier Utilization

| Platform | Free Tier |
|--------|----------|
| **AWS** | 12 months free (t2.micro EC2, 5GB S3, etc.) |
| **GCP** | $300 credit (90 days), Always Free services |

---

## Related Resources

- [AWS Official Documentation](https://docs.aws.amazon.com/)
- [GCP Official Documentation](https://cloud.google.com/docs)
- [Terraform Official Documentation](https://www.terraform.io/docs)
- [AWS vs GCP Comparison Guide](https://cloud.google.com/docs/compare/aws)

### Related Folders

- [Docker/](../Docker/) - Container basics, Kubernetes introduction
- [Networking/](../Networking/) - Network fundamentals
- [Linux/](../Linux/) - Server management basics
