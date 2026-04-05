# Managed Relational Databases (RDS / Cloud SQL)

**Previous**: [Load Balancing and CDN](./10_Load_Balancing_CDN.md) | **Next**: [NoSQL Databases](./12_NoSQL_Databases.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the advantages of managed database services over self-hosted installations
2. Compare AWS RDS and GCP Cloud SQL in terms of supported engines and features
3. Provision a managed relational database instance with appropriate sizing
4. Configure automated backups, point-in-time recovery, and read replicas
5. Implement multi-AZ (high availability) deployments for production workloads
6. Describe AWS Aurora and GCP AlloyDB/Spanner as high-performance alternatives
7. Apply security best practices including encryption, VPC placement, and access control

---

Relational databases remain the backbone of most business applications, but operating them reliably requires expertise in backups, patching, replication, and failover. Managed database services offload these operational burdens to the cloud provider, allowing teams to focus on schema design and query performance rather than infrastructure maintenance.

## 1. Managed DB Overview

### 1.1 Managed vs Self-Managed

| Task | Self-Managed (EC2) | Managed (RDS/Cloud SQL) |
|------|----------------|----------------------|
| Hardware Provisioning | User | Provider |
| OS Patching | User | Provider |
| DB Installation/Setup | User | Provider |
| Backup | User | Automatic |
| High Availability | User | Option provided |
| Scaling | Manual | Button click |
| Monitoring | Setup required | Built-in |

### 1.2 Service Comparison

| Category | AWS | GCP |
|------|-----|-----|
| Managed RDB | RDS | Cloud SQL |
| High-Performance DB | Aurora | Cloud Spanner, AlloyDB |
| Supported Engines | MySQL, PostgreSQL, MariaDB, Oracle, SQL Server | MySQL, PostgreSQL, SQL Server |

---

## 2. AWS RDS

### 2.1 Creating an RDS Instance

```bash
# Create DB subnet group
aws rds create-db-subnet-group \
    --db-subnet-group-name my-subnet-group \
    --db-subnet-group-description "My DB subnets" \
    --subnet-ids subnet-1 subnet-2

# Create parameter group (optional)
aws rds create-db-parameter-group \
    --db-parameter-group-name my-params \
    --db-parameter-group-family mysql8.0 \
    --description "Custom parameters"

# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier my-database \
    --db-instance-class db.t3.micro \
    --engine mysql \
    --engine-version 8.0 \
    --master-username admin \
    --master-user-password MyPassword123! \
    --allocated-storage 20 \
    --storage-type gp3 \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-12345678 \
    --backup-retention-period 7 \
    --multi-az \
    --publicly-accessible false

# Check creation status
aws rds describe-db-instances \
    --db-instance-identifier my-database \
    --query 'DBInstances[0].DBInstanceStatus'
```

### 2.2 Multi-AZ Deployment

```
┌─────────────────────────────────────────────────────────────┐
│  VPC                                                        │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │     AZ-a            │  │     AZ-b            │          │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │          │
│  │  │  Primary DB   │──┼──┼──│  Standby DB   │  │          │
│  │  │  (Read/Write) │  │  │  │  (Sync repl)  │  │          │
│  │  └───────────────┘  │  │  └───────────────┘  │          │
│  └─────────────────────┘  └─────────────────────┘          │
│              ↑ Automatic failover                           │
└─────────────────────────────────────────────────────────────┘
```

```bash
# Convert existing instance to Multi-AZ
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --multi-az \
    --apply-immediately
```

### 2.3 Read Replicas

```bash
# Create read replica (same region)
aws rds create-db-instance-read-replica \
    --db-instance-identifier my-read-replica \
    --source-db-instance-identifier my-database

# Create read replica in another region (cross-region)
aws rds create-db-instance-read-replica \
    --db-instance-identifier my-replica-us \
    --source-db-instance-identifier arn:aws:rds:ap-northeast-2:123456789012:db:my-database \
    --region us-east-1

# Promote replica (convert to master)
aws rds promote-read-replica \
    --db-instance-identifier my-read-replica
```

### 2.4 Backup and Restore

```bash
# Create manual snapshot
aws rds create-db-snapshot \
    --db-instance-identifier my-database \
    --db-snapshot-identifier my-snapshot-2024

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier my-restored-db \
    --db-snapshot-identifier my-snapshot-2024

# Point-in-Time Recovery
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier my-database \
    --target-db-instance-identifier my-pitr-db \
    --restore-time 2024-01-15T10:00:00Z

# Check/modify automatic backup settings
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --backup-retention-period 14 \
    --preferred-backup-window "03:00-04:00"
```

---

## 3. GCP Cloud SQL

### 3.1 Creating a Cloud SQL Instance

```bash
# Enable Cloud SQL API
gcloud services enable sqladmin.googleapis.com

# Create MySQL instance
gcloud sql instances create my-database \
    --database-version=MYSQL_8_0 \
    --tier=db-f1-micro \
    --region=asia-northeast3 \
    --root-password=MyPassword123! \
    --storage-size=10GB \
    --storage-type=SSD \
    --backup-start-time=03:00 \
    --availability-type=REGIONAL

# Create PostgreSQL instance
gcloud sql instances create my-postgres \
    --database-version=POSTGRES_15 \
    --tier=db-g1-small \
    --region=asia-northeast3

# Check instance details
gcloud sql instances describe my-database
```

### 3.2 High Availability (HA)

```bash
# Create high-availability instance
gcloud sql instances create my-ha-db \
    --database-version=MYSQL_8_0 \
    --tier=db-n1-standard-2 \
    --region=asia-northeast3 \
    --availability-type=REGIONAL \
    --root-password=MyPassword123!

# Convert existing instance to HA
gcloud sql instances patch my-database \
    --availability-type=REGIONAL
```

### 3.3 Read Replicas

```bash
# Create read replica
gcloud sql instances create my-read-replica \
    --master-instance-name=my-database \
    --region=asia-northeast3

# Promote replica
gcloud sql instances promote-replica my-read-replica

# List replicas
gcloud sql instances list --filter="masterInstanceName:my-database"
```

### 3.4 Backup and Restore

```bash
# Create on-demand backup
gcloud sql backups create \
    --instance=my-database \
    --description="Manual backup"

# List backups
gcloud sql backups list --instance=my-database

# Restore from backup (new instance)
gcloud sql instances restore-backup my-restored-db \
    --backup-instance=my-database \
    --backup-id=1234567890

# Point-in-Time Recovery
gcloud sql instances clone my-database my-pitr-db \
    --point-in-time="2024-01-15T10:00:00Z"
```

---

## 4. Connection Setup

### 4.1 AWS RDS Connection

**Security Group Setup:**
```bash
# Allow application access to RDS security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-rds \
    --protocol tcp \
    --port 3306 \
    --source-group sg-app

# Check endpoint
aws rds describe-db-instances \
    --db-instance-identifier my-database \
    --query 'DBInstances[0].Endpoint'
```

**Application Connection:**
```python
import pymysql

connection = pymysql.connect(
    host='my-database.xxxx.ap-northeast-2.rds.amazonaws.com',
    user='admin',
    password='MyPassword123!',
    database='mydb',
    port=3306
)
```

### 4.2 GCP Cloud SQL Connection

**Connection Methods:**

1. **Public IP (Not Recommended)**
```bash
# Allow public IP
gcloud sql instances patch my-database \
    --authorized-networks=203.0.113.0/24

# Connect
mysql -h <PUBLIC_IP> -u root -p
```

2. **Cloud SQL Proxy (Recommended)**
```bash
# Download Cloud SQL Proxy
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# Run Proxy
./cloud-sql-proxy PROJECT_ID:asia-northeast3:my-database

# Connect from another terminal
mysql -h 127.0.0.1 -u root -p
```

3. **Private IP (Within VPC)**
```bash
# Enable Private IP
gcloud sql instances patch my-database \
    --network=projects/PROJECT_ID/global/networks/my-vpc

# Connect from instance within VPC
mysql -h <PRIVATE_IP> -u root -p
```

**Python Connection (Cloud SQL Connector):**
```python
from google.cloud.sql.connector import Connector
import pymysql

connector = Connector()

def get_conn():
    return connector.connect(
        "project:region:instance",
        "pymysql",
        user="root",
        password="password",
        db="mydb"
    )

connection = get_conn()
```

---

## 5. Performance Optimization

### 5.1 Instance Resizing

**AWS RDS:**
```bash
# Change instance class
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --db-instance-class db.m5.large \
    --apply-immediately

# Expand storage (cannot shrink)
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --allocated-storage 100
```

**GCP Cloud SQL:**
```bash
# Change machine type
gcloud sql instances patch my-database \
    --tier=db-n1-standard-4

# Expand storage
gcloud sql instances patch my-database \
    --storage-size=100GB
```

### 5.2 Parameter Tuning

**AWS RDS Parameter Group:**
```bash
# Modify parameters
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=max_connections,ParameterValue=500,ApplyMethod=pending-reboot"

aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=innodb_buffer_pool_size,ParameterValue={DBInstanceClassMemory*3/4},ApplyMethod=pending-reboot"
```

**GCP Cloud SQL Flags:**
```bash
# Set flags
gcloud sql instances patch my-database \
    --database-flags=max_connections=500,innodb_buffer_pool_size=1073741824
```

---

## 6. Aurora / AlloyDB / Spanner

### 6.1 AWS Aurora

Aurora is a cloud-native relational database.

**Features:**
- MySQL/PostgreSQL compatible
- Auto-scaling up to 128TB
- 6 replicas (3 AZs)
- Up to 15 read replicas
- Serverless option (Aurora Serverless)

```bash
# Create Aurora cluster
aws rds create-db-cluster \
    --db-cluster-identifier my-aurora \
    --engine aurora-mysql \
    --engine-version 8.0.mysql_aurora.3.04.0 \
    --master-username admin \
    --master-user-password MyPassword123! \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-12345678

# Add Aurora instance
aws rds create-db-instance \
    --db-instance-identifier my-aurora-instance-1 \
    --db-cluster-identifier my-aurora \
    --db-instance-class db.r5.large \
    --engine aurora-mysql
```

### 6.2 GCP Cloud Spanner

Spanner is a globally distributed relational database.

**Features:**
- Global transactions
- Unlimited scaling
- 99.999% SLA
- PostgreSQL-compatible interface

```bash
# Create Spanner instance
gcloud spanner instances create my-spanner \
    --config=regional-asia-northeast3 \
    --nodes=1 \
    --description="My Spanner instance"

# Create database
gcloud spanner databases create mydb \
    --instance=my-spanner
```

### 6.3 GCP AlloyDB

AlloyDB is a PostgreSQL-compatible high-performance database.

```bash
# Create AlloyDB cluster
gcloud alloydb clusters create my-cluster \
    --region=asia-northeast3 \
    --password=MyPassword123!

# Create primary instance
gcloud alloydb instances create primary \
    --cluster=my-cluster \
    --region=asia-northeast3 \
    --instance-type=PRIMARY \
    --cpu-count=2
```

---

## 7. Cost Comparison

### 7.1 AWS RDS Cost (Seoul)

| Instance | vCPU | Memory | Hourly Cost |
|----------|------|--------|------------|
| db.t3.micro | 2 | 1 GB | ~$0.02 |
| db.t3.small | 2 | 2 GB | ~$0.04 |
| db.m5.large | 2 | 8 GB | ~$0.18 |
| db.r5.large | 2 | 16 GB | ~$0.26 |

**Additional Costs:**
- Storage: gp3 $0.114/GB/month
- Backup: retention × $0.095/GB/month
- Multi-AZ: Instance cost × 2

### 7.2 GCP Cloud SQL Cost (Seoul)

| Tier | vCPU | Memory | Hourly Cost |
|------|------|--------|------------|
| db-f1-micro | Shared | 0.6 GB | ~$0.01 |
| db-g1-small | Shared | 1.7 GB | ~$0.03 |
| db-n1-standard-2 | 2 | 7.5 GB | ~$0.13 |
| db-n1-highmem-2 | 2 | 13 GB | ~$0.16 |

**Additional Costs:**
- Storage: SSD $0.180/GB/month
- High Availability: Instance cost × 2
- Backup: $0.08/GB/month

---

## 8. Security

### 8.1 Encryption

**AWS RDS:**
```bash
# Encryption at rest (at creation)
aws rds create-db-instance \
    --storage-encrypted \
    --kms-key-id arn:aws:kms:...:key/xxx \
    ...

# Enforce SSL
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=require_secure_transport,ParameterValue=1"
```

**GCP Cloud SQL:**
```bash
# Create SSL certificate
gcloud sql ssl client-certs create my-client \
    --instance=my-database \
    --common-name=my-client

# Require SSL
gcloud sql instances patch my-database \
    --require-ssl
```

### 8.2 IAM Authentication

**AWS RDS IAM Authentication:**
```bash
# Enable IAM authentication
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --enable-iam-database-authentication

# Generate temporary token
aws rds generate-db-auth-token \
    --hostname my-database.xxxx.rds.amazonaws.com \
    --port 3306 \
    --username iam_user
```

**GCP Cloud SQL IAM:**
```bash
# Enable IAM authentication
gcloud sql instances patch my-database \
    --enable-database-flags \
    --database-flags=cloudsql_iam_authentication=on

# Add IAM user
gcloud sql users create user@example.com \
    --instance=my-database \
    --type=CLOUD_IAM_USER
```

---

## 9. Next Steps

- [12_NoSQL_Databases.md](./12_NoSQL_Databases.md) - NoSQL Databases
- [PostgreSQL/](../PostgreSQL/) - PostgreSQL Details

---

## Exercises

### Exercise 1: Managed DB vs Self-Hosted Decision

A team is considering whether to run PostgreSQL on EC2 instances (self-managed) or use AWS RDS for PostgreSQL (managed). Their priorities are: minimal operational overhead, automated backups, and high availability.

List three concrete operational tasks that RDS handles automatically that would require significant manual effort on self-managed EC2, and identify one scenario where self-managed EC2 would be the better choice.

<details>
<summary>Show Answer</summary>

**Three tasks RDS automates**:

1. **Automated backups and point-in-time recovery (PITR)** — RDS automatically takes daily snapshots and streams transaction logs to S3, enabling PITR to any second within the retention period (up to 35 days). On self-managed EC2, you must write and maintain backup scripts, manage backup storage, test restore procedures, and monitor for failures.

2. **Multi-AZ failover** — With Multi-AZ enabled, RDS automatically maintains a synchronous standby replica in a different AZ. If the primary fails, AWS promotes the standby in ~1–2 minutes with no manual intervention. On EC2, you would need to configure replication manually (e.g., PostgreSQL streaming replication with repmgr) and set up failover scripts or use Pacemaker/Corosync.

3. **Minor version patching** — RDS can be configured to apply minor engine patches during maintenance windows automatically. On EC2, you must monitor for new PostgreSQL releases, test compatibility, schedule downtime, and apply updates manually.

**Scenario where self-managed EC2 is better**:
- When you need **OS-level access** for performance tuning (custom kernel parameters, huge pages, I/O scheduler settings), require **PostgreSQL extensions** not available on RDS (e.g., certain contrib extensions or custom compiled plugins), or need complete control over the exact PostgreSQL version and configuration beyond what RDS parameter groups support.

</details>

### Exercise 2: RDS Multi-AZ vs Read Replica

Explain the difference between RDS Multi-AZ and RDS Read Replicas. For each scenario, state which feature addresses it:

1. The marketing team runs heavy reporting queries that are slowing down the production application.
2. The production database's AZ experienced a hardware failure and the database is unreachable.

<details>
<summary>Show Answer</summary>

**Multi-AZ**:
- Purpose: **High availability and automatic failover**
- The standby is a synchronous replica in a different AZ.
- Standby does NOT serve read traffic; it exists only for failover.
- Failover is automatic (~1–2 minutes), the DNS endpoint is updated.
- The standby is invisible to the application.

**Read Replica**:
- Purpose: **Read scaling and offloading read traffic**
- Asynchronous replication from primary to one or more replicas.
- Replicas can serve read queries, reducing load on the primary.
- Replicas can be in the same region, different region, or promoted to standalone primary.
- No automatic failover to a read replica (promotion is manual).

**Scenario answers**:

1. **Read Replica** — The reporting queries can be directed to one or more read replicas. This offloads the CPU/IO burden from the production primary, restoring primary performance for the application. The reporting tool connects to the read replica endpoint instead of the primary.

2. **Multi-AZ** — RDS Multi-AZ automatically detects the AZ failure and promotes the synchronous standby to primary. The DNS endpoint automatically points to the new primary, typically within 1–2 minutes. The application reconnects and resumes operations with minimal downtime.

</details>

### Exercise 3: Database Security Configuration

You are provisioning an RDS PostgreSQL instance for a production e-commerce application. List the key security configurations you would apply and explain why each is important.

<details>
<summary>Show Answer</summary>

1. **Place the instance in a private subnet with `--publicly-accessible false`**
   - The RDS instance should not have a public IP. Only resources within the VPC (application servers) can connect. This prevents direct internet exposure.

2. **Use a dedicated security group that only allows access from the application tier**
   ```bash
   # Allow PostgreSQL port 5432 only from the application server security group
   aws ec2 authorize-security-group-ingress \
       --group-id sg-rds \
       --protocol tcp \
       --port 5432 \
       --source-group sg-app-server
   ```

3. **Enable encryption at rest with `--storage-encrypted`**
   - All data at rest (storage, backups, snapshots, logs) is encrypted using AES-256. Required for PCI DSS and HIPAA compliance. Minimal performance impact.

4. **Enable automated backups with a retention period of at least 7 days**
   - Enables point-in-time recovery. Essential for recovering from accidental data deletion or corruption.
   ```bash
   --backup-retention-period 7
   ```

5. **Rotate the master password using AWS Secrets Manager**
   - Never hard-code database credentials in application code. Store in Secrets Manager with automatic rotation enabled. Use IAM roles to grant applications access to retrieve the secret.

6. **Enable deletion protection**
   - Prevents accidental deletion of the database instance:
   ```bash
   aws rds modify-db-instance \
       --db-instance-identifier my-db \
       --deletion-protection \
       --apply-immediately
   ```

</details>

### Exercise 4: Aurora vs RDS PostgreSQL

A SaaS company is growing rapidly and expects to need read scaling (currently 2 read replicas) and occasional cross-region replication for disaster recovery. Their current RDS PostgreSQL instance is `db.r5.2xlarge`.

Evaluate whether they should migrate to Aurora PostgreSQL. What are the key advantages Aurora provides for their stated requirements?

<details>
<summary>Show Answer</summary>

**Aurora advantages for their requirements**:

1. **Read scaling**: Aurora supports up to **15 low-latency read replicas** (vs 5 for RDS PostgreSQL). Aurora replicas read from the same shared distributed storage volume — replica lag is typically under 100ms. Adding replicas does not require additional storage copies, reducing cost.

2. **Cross-region disaster recovery**: Aurora Global Database enables **sub-second replication lag** to a secondary region (vs minutes for RDS cross-region read replicas). Failover to the secondary region is managed and completes in under 1 minute.

3. **Storage scalability**: Aurora's shared storage layer automatically grows in 10 GB increments up to 128 TB without any user intervention. No need to estimate storage capacity upfront.

4. **Performance**: Aurora claims up to 3x the throughput of standard PostgreSQL. The shared storage architecture eliminates the replication overhead present in traditional RDS Multi-AZ setups.

**Considerations before migrating**:
- Aurora PostgreSQL is typically 20–40% more expensive per vCPU-hour than equivalent RDS instances.
- Not all PostgreSQL extensions are available on Aurora.
- Migration requires a testing period to validate compatibility.

**Recommendation**: Given the need for multiple read replicas AND cross-region DR, Aurora PostgreSQL is the better fit. The operational simplicity of Aurora Global Database and the superior replica performance outweigh the higher per-instance cost for a growing SaaS workload.

</details>

### Exercise 5: Point-in-Time Recovery Scenario

Your RDS MySQL database had a critical data corruption event at 14:32 UTC today caused by an erroneous `DELETE` statement. The DBA needs to recover the data as of 14:30 UTC (2 minutes before the error).

Write the AWS CLI command to restore the database to a new instance from 14:30 UTC today, and describe what happens after the restore.

<details>
<summary>Show Answer</summary>

```bash
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier my-production-db \
    --target-db-instance-identifier my-production-db-restored \
    --restore-time 2026-02-24T14:30:00Z \
    --db-instance-class db.r5.2xlarge \
    --multi-az \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-rds
```

**What happens after the restore**:

1. AWS creates a **new** RDS instance (`my-production-db-restored`) — it does NOT modify the original database. The original instance remains running.

2. The restore applies all transaction logs from the most recent automated backup up to `14:30:00Z`, reconstructing the database state at that exact moment.

3. The new instance gets a new endpoint (DNS hostname). You must update the application connection string to point to the new instance, or use the new instance to extract and replay only the missing data into the original instance.

4. **Typical approach**: Use the restored instance to export the affected tables/rows, then import them into the production instance. This minimizes downtime and avoids switching the entire application to a new endpoint.

5. After recovery is complete and verified, delete the restored instance to avoid paying for two database instances.

**Prerequisite**: The `--backup-retention-period` must be at least 1 day, and automated backups must be enabled. PITR is available for any point within the retention window.

</details>

---

## References

- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [AWS Aurora Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/)
- [GCP Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [GCP Cloud Spanner](https://cloud.google.com/spanner/docs)
