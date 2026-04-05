# Block and File Storage (EBS/EFS vs Persistent Disk/Filestore)

**Previous**: [Object Storage](./07_Object_Storage.md) | **Next**: [Virtual Private Cloud](./09_Virtual_Private_Cloud.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between block, file, and object storage and their appropriate use cases
2. Compare AWS EBS and GCP Persistent Disk volume types and performance characteristics
3. Select the right volume type based on IOPS, throughput, and cost requirements
4. Configure shared file storage using AWS EFS and GCP Filestore
5. Implement snapshot and backup strategies for block storage volumes
6. Explain the difference between instance-attached storage and network-attached storage

---

While object storage handles unstructured data at scale, many workloads require the low-latency, high-throughput access that only block and file storage can provide. Databases need block volumes for random I/O, and shared application data often requires a POSIX-compliant filesystem. Choosing the right storage type for each workload is critical for both performance and cost optimization.

## 1. Storage Type Comparison

### 1.1 Block vs File vs Object Storage

| Type | Characteristics | Use Cases | AWS | GCP |
|------|------|----------|-----|-----|
| **Block** | Low-level disk access | Databases, OS boot disk | EBS | Persistent Disk |
| **File** | Shared filesystem | Shared storage, CMS | EFS | Filestore |
| **Object** | HTTP-based, unlimited | Backup, media, logs | S3 | Cloud Storage |

### 1.2 Service Mapping

| Feature | AWS | GCP |
|------|-----|-----|
| Block Storage | EBS (Elastic Block Store) | Persistent Disk (PD) |
| Shared File Storage | EFS (Elastic File System) | Filestore |
| Local SSD | Instance Store | Local SSD |

---

## 2. Block Storage

### 2.1 AWS EBS (Elastic Block Store)

**EBS Volume Types:**

| Type | Use Case | IOPS | Throughput | Cost |
|------|------|------|--------|------|
| **gp3** | General purpose SSD | Up to 16,000 | Up to 1,000 MB/s | Low |
| **gp2** | General purpose SSD (legacy) | Up to 16,000 | Up to 250 MB/s | Medium |
| **io2** | Provisioned IOPS | Up to 64,000 | Up to 1,000 MB/s | High |
| **st1** | Throughput-optimized HDD | Up to 500 | Up to 500 MB/s | Low |
| **sc1** | Cold HDD | Up to 250 | Up to 250 MB/s | Very low |

```bash
# Create EBS volume
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --size 100 \
    --volume-type gp3 \
    --iops 3000 \
    --throughput 125 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=MyVolume}]'

# Attach volume to EC2
aws ec2 attach-volume \
    --volume-id vol-1234567890abcdef0 \
    --instance-id i-1234567890abcdef0 \
    --device /dev/sdf

# Mount inside instance
sudo mkfs -t xfs /dev/xvdf
sudo mkdir /data
sudo mount /dev/xvdf /data

# Add to fstab (persistent mount)
echo '/dev/xvdf /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

### 2.2 GCP Persistent Disk

**Persistent Disk Types:**

| Type | Use Case | IOPS (read) | Throughput (read) | Cost |
|------|------|------------|--------------|------|
| **pd-standard** | HDD | Up to 7,500 | Up to 180 MB/s | Low |
| **pd-balanced** | SSD (balanced) | Up to 80,000 | Up to 1,200 MB/s | Medium |
| **pd-ssd** | SSD (high performance) | Up to 100,000 | Up to 1,200 MB/s | High |
| **pd-extreme** | High IOPS SSD | Up to 120,000 | Up to 2,400 MB/s | Very high |

```bash
# Create Persistent Disk
gcloud compute disks create my-disk \
    --zone=asia-northeast3-a \
    --size=100GB \
    --type=pd-ssd

# Attach disk to VM
gcloud compute instances attach-disk my-instance \
    --disk=my-disk \
    --zone=asia-northeast3-a

# Mount inside instance
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir /data
sudo mount -o discard,defaults /dev/sdb /data

# Add to fstab
echo UUID=$(sudo blkid -s UUID -o value /dev/sdb) /data ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
```

---

## 3. Snapshots

### 3.1 AWS EBS Snapshots

```bash
# Create snapshot
aws ec2 create-snapshot \
    --volume-id vol-1234567890abcdef0 \
    --description "My snapshot" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=MySnapshot}]'

# List snapshots
aws ec2 describe-snapshots \
    --owner-ids self \
    --query 'Snapshots[*].[SnapshotId,VolumeId,StartTime,State]'

# Restore volume from snapshot
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --snapshot-id snap-1234567890abcdef0 \
    --volume-type gp3

# Copy snapshot (to another region)
aws ec2 copy-snapshot \
    --source-region ap-northeast-2 \
    --source-snapshot-id snap-1234567890abcdef0 \
    --destination-region us-east-1

# Delete snapshot
aws ec2 delete-snapshot --snapshot-id snap-1234567890abcdef0
```

**Automated Snapshots (Data Lifecycle Manager):**
```bash
# Create DLM policy (daily snapshots, 7-day retention)
aws dlm create-lifecycle-policy \
    --description "Daily snapshots" \
    --state ENABLED \
    --execution-role-arn arn:aws:iam::123456789012:role/AWSDataLifecycleManagerDefaultRole \
    --policy-details '{
        "ResourceTypes": ["VOLUME"],
        "TargetTags": [{"Key": "Backup", "Value": "true"}],
        "Schedules": [{
            "Name": "DailySnapshots",
            "CreateRule": {"Interval": 24, "IntervalUnit": "HOURS", "Times": ["03:00"]},
            "RetainRule": {"Count": 7}
        }]
    }'
```

### 3.2 GCP Snapshots

```bash
# Create snapshot
gcloud compute snapshots create my-snapshot \
    --source-disk=my-disk \
    --source-disk-zone=asia-northeast3-a

# List snapshots
gcloud compute snapshots list

# Restore disk from snapshot
gcloud compute disks create restored-disk \
    --source-snapshot=my-snapshot \
    --zone=asia-northeast3-a

# Delete snapshot
gcloud compute snapshots delete my-snapshot
```

**Snapshot Schedule:**
```bash
# Create schedule policy (daily, 7-day retention)
gcloud compute resource-policies create snapshot-schedule daily-snapshot \
    --region=asia-northeast3 \
    --max-retention-days=7 \
    --start-time=04:00 \
    --daily-schedule

# Attach schedule to disk
gcloud compute disks add-resource-policies my-disk \
    --resource-policies=daily-snapshot \
    --zone=asia-northeast3-a
```

---

## 4. Volume Expansion

### 4.1 AWS EBS Volume Expansion

```bash
# 1. Modify volume size (online possible)
aws ec2 modify-volume \
    --volume-id vol-1234567890abcdef0 \
    --size 200

# 2. Check modification status
aws ec2 describe-volumes-modifications \
    --volume-id vol-1234567890abcdef0

# 3. Expand filesystem inside instance
# XFS
sudo xfs_growfs -d /data

# ext4
sudo resize2fs /dev/xvdf
```

### 4.2 GCP Persistent Disk Expansion

```bash
# 1. Expand disk size (online possible)
gcloud compute disks resize my-disk \
    --size=200GB \
    --zone=asia-northeast3-a

# 2. Expand filesystem inside instance
# ext4
sudo resize2fs /dev/sdb

# XFS
sudo xfs_growfs /data
```

---

## 5. File Storage

### 5.1 AWS EFS (Elastic File System)

**Features:**
- NFS v4.1 protocol
- Auto-scaling (expand/shrink)
- Multi-AZ support
- Supports thousands of concurrent EC2 connections

```bash
# 1. Create EFS filesystem
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode bursting \
    --encrypted \
    --tags Key=Name,Value=my-efs

# 2. Create mount target (per subnet)
aws efs create-mount-target \
    --file-system-id fs-12345678 \
    --subnet-id subnet-12345678 \
    --security-groups sg-12345678

# 3. Mount from EC2
sudo yum install -y amazon-efs-utils
sudo mkdir /efs
sudo mount -t efs fs-12345678:/ /efs

# Or mount via NFS
sudo mount -t nfs4 -o nfsvers=4.1 \
    fs-12345678.efs.ap-northeast-2.amazonaws.com:/ /efs

# Add to fstab
echo 'fs-12345678:/ /efs efs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

**EFS Storage Classes:**
| Class | Use Case | Cost |
|--------|------|------|
| Standard | Frequent access | High |
| Infrequent Access (IA) | Infrequent access | Low |
| Archive | Long-term retention | Very low |

```bash
# Set lifecycle policy (move to IA after 30 days)
aws efs put-lifecycle-configuration \
    --file-system-id fs-12345678 \
    --lifecycle-policies '[{"TransitionToIA":"AFTER_30_DAYS"}]'
```

### 5.2 GCP Filestore

**Features:**
- NFS v3 protocol
- Pre-provisioned capacity
- High-performance options available

**Filestore Tiers:**
| Tier | Capacity | Performance | Use Case |
|------|------|------|------|
| Basic HDD | 1TB-63.9TB | 100 MB/s | File sharing |
| Basic SSD | 2.5TB-63.9TB | 1,200 MB/s | High performance |
| Zonal | 1TB-100TB | Up to 2,560 MB/s | High-performance workloads |
| Enterprise | 1TB-10TB | Up to 1,200 MB/s | Mission-critical |

```bash
# 1. Create Filestore instance
gcloud filestore instances create my-filestore \
    --zone=asia-northeast3-a \
    --tier=BASIC_SSD \
    --file-share=name=vol1,capacity=2.5TB \
    --network=name=default

# 2. Get Filestore info
gcloud filestore instances describe my-filestore \
    --zone=asia-northeast3-a

# 3. Mount from VM
sudo apt-get install -y nfs-common
sudo mkdir /filestore
sudo mount 10.0.0.2:/vol1 /filestore

# Add to fstab
echo '10.0.0.2:/vol1 /filestore nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

---

## 6. Local SSD

### 6.1 AWS Instance Store

Instance Store is temporary storage physically attached to EC2 instances.

**Features:**
- Data loss on instance stop/termination
- Very high IOPS
- No additional cost (included in instance price)

```bash
# Check instance types with Instance Store
aws ec2 describe-instance-types \
    --filters "Name=instance-storage-supported,Values=true" \
    --query 'InstanceTypes[*].[InstanceType,InstanceStorageInfo.TotalSizeInGB]'

# e.g. i3.large, d2.xlarge etc.

# Mount inside instance
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir /local-ssd
sudo mount /dev/nvme1n1 /local-ssd
```

### 6.2 GCP Local SSD

```bash
# Create instance with Local SSD
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4 \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME

# Mount inside instance
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir /local-ssd
sudo mount /dev/nvme0n1 /local-ssd
```

**Local SSD Features:**
- Added in 375GB units
- Up to 24 units (9TB)
- Data loss on instance stop
- Live Migration not available (for some)

---

## 7. Performance Optimization

### 7.1 IOPS vs Throughput

```
IOPS (Input/Output Per Second):
- Number of read/write operations per second
- Important for small random I/O
- Databases, transaction processing

Throughput:
- Amount of data transferred per second (MB/s)
- Important for large sequential I/O
- Video streaming, big data
```

### 7.2 Optimization Tips

**AWS EBS:**
```bash
# Adjust gp3 IOPS/throughput
aws ec2 modify-volume \
    --volume-id vol-xxx \
    --iops 10000 \
    --throughput 500

# Use EBS-optimized instance
aws ec2 run-instances \
    --instance-type m5.large \
    --ebs-optimized \
    ...
```

**GCP Persistent Disk:**
```bash
# Larger disk = higher performance
# pd-ssd 100GB: up to 3,000 IOPS
# pd-ssd 500GB: up to 15,000 IOPS
# pd-ssd 1TB: up to 30,000 IOPS

# Increase disk size for better performance
gcloud compute disks resize my-disk --size=500GB
```

---

## 8. Cost Comparison

### 8.1 Block Storage Cost (Seoul Region)

| Type | AWS EBS | GCP PD |
|------|---------|--------|
| General Purpose SSD | $0.114/GB (gp3) | $0.102/GB (pd-balanced) |
| High Performance SSD | $0.138/GB (io1) | $0.180/GB (pd-ssd) |
| HDD | $0.054/GB (st1) | $0.044/GB (pd-standard) |

### 8.2 File Storage Cost

| Service | Cost |
|--------|------|
| AWS EFS Standard | ~$0.33/GB/month |
| AWS EFS IA | ~$0.025/GB/month |
| GCP Filestore Basic SSD | ~$0.24/GB/month |
| GCP Filestore Basic HDD | ~$0.12/GB/month |

---

## 9. Hands-on: Setting Up Shared Storage

### 9.1 AWS EFS Multi-Instance Mount

```bash
# 1. Create mount targets in two subnets
aws efs create-mount-target --file-system-id fs-xxx --subnet-id subnet-1 --security-groups sg-xxx
aws efs create-mount-target --file-system-id fs-xxx --subnet-id subnet-2 --security-groups sg-xxx

# 2. Add NFS rule to security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxx \
    --protocol tcp \
    --port 2049 \
    --source-group sg-instance

# 3. Mount from each instance
# Instance 1
sudo mkdir /shared && sudo mount -t efs fs-xxx:/ /shared
echo "Hello from Instance 1" | sudo tee /shared/test.txt

# Instance 2
sudo mkdir /shared && sudo mount -t efs fs-xxx:/ /shared
cat /shared/test.txt  # Prints "Hello from Instance 1"
```

### 9.2 GCP Filestore Multi-Instance Mount

```bash
# 1. Add firewall rule
gcloud compute firewall-rules create allow-nfs \
    --allow tcp:2049,tcp:111,udp:2049,udp:111 \
    --source-ranges 10.0.0.0/8

# 2. Mount from each instance
# Instance 1
sudo mkdir /shared && sudo mount 10.0.0.2:/vol1 /shared
echo "Hello from Instance 1" | sudo tee /shared/test.txt

# Instance 2
sudo mkdir /shared && sudo mount 10.0.0.2:/vol1 /shared
cat /shared/test.txt  # Prints "Hello from Instance 1"
```

---

## 10. Next Steps

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Networking
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - Database Storage

---

## Exercises

### Exercise 1: Storage Type Selection

For each workload, select the most appropriate storage type (block/EBS, file/EFS, or object/S3) and explain why:

1. A PostgreSQL database running on EC2 requires low-latency random read/write access.
2. A content management system (CMS) runs across 10 EC2 web servers that all need to read and write the same uploaded media files simultaneously.
3. A log aggregation system archives 500 GB of compressed log files per day, which are rarely read.
4. An application needs a temporary scratch space to store intermediate computation files that are only needed while the EC2 instance is running.

<details>
<summary>Show Answer</summary>

1. **Block storage (EBS)** — Databases require block-level access for random I/O operations (individual row reads/writes). EBS `gp3` or `io2` provides the low latency and high IOPS that PostgreSQL needs. Object storage has high latency per operation; file storage adds NFS overhead.

2. **File storage (EFS)** — Multiple EC2 instances concurrently mounting the same filesystem is the defining use case for NFS-based file storage. EFS allows all 10 web servers to mount the same filesystem and read/write the same media files simultaneously. Block storage (EBS) can only be attached to one instance at a time in read-write mode.

3. **Object storage (S3)** — Log archives are write-once, read-rarely, and potentially massive in total volume. S3's pay-per-GB pricing, unlimited capacity, and lifecycle rules (auto-transition to Glacier) make it the most cost-effective choice. Block and file storage cost significantly more per GB for cold data.

4. **Instance Store (local SSD)** — For temporary scratch space needed only during the instance's lifetime, Instance Store (NVMe SSDs physically attached to the host) provides the highest IOPS and throughput with no additional cost. Data is lost when the instance stops, which is acceptable for temporary scratch data. This avoids paying for an EBS volume that will hold ephemeral data.

</details>

### Exercise 2: EBS Volume Type Selection

A high-frequency trading (HFT) application requires a database with the following I/O characteristics:
- 50,000 IOPS required consistently
- 500 MB/s throughput
- 2 TB capacity
- Sub-millisecond latency is critical

Which EBS volume type should be used? Write the AWS CLI command to create it.

<details>
<summary>Show Answer</summary>

**Volume type: `io2`** (Provisioned IOPS SSD)

Reasoning:
- `gp3` supports up to 16,000 IOPS maximum — insufficient for 50,000 IOPS.
- `io2` supports up to 64,000 IOPS per volume (and up to 256,000 with io2 Block Express on supported instances).
- `io2` provides the consistent, sub-millisecond latency required for HFT workloads.
- `st1`/`sc1` are HDD-based and unsuitable for random I/O with latency requirements.

```bash
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --size 2000 \
    --volume-type io2 \
    --iops 50000 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=HFT-DB-Volume}]'
```

**Note**: To achieve 50,000 IOPS, the EC2 instance must also support it. Use a storage-optimized instance (i3/i4i) or a large compute instance that supports high EBS bandwidth (e.g., `m5.8xlarge` with dedicated EBS bandwidth).

</details>

### Exercise 3: EBS Snapshot and Restore

Your production EC2 instance has an EBS volume (`vol-0abc123`) that needs to be backed up before a major OS update. Write the AWS CLI commands to:
1. Create a snapshot of the volume with a descriptive name.
2. List all snapshots you own for that volume.
3. Create a new EBS volume from the snapshot in AZ `ap-northeast-2b` (in case you need to restore to a different AZ).

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Create snapshot with description
aws ec2 create-snapshot \
    --volume-id vol-0abc123 \
    --description "Pre-OS-update backup - $(date +%Y-%m-%d)" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=pre-os-update-backup},{Key=Purpose,Value=manual-backup}]'

# Step 2: List snapshots for this volume (owned by you)
aws ec2 describe-snapshots \
    --owner-ids self \
    --filters "Name=volume-id,Values=vol-0abc123" \
    --query 'Snapshots[*].[SnapshotId,StartTime,State,Description]' \
    --output table

# Step 3: Create new volume from snapshot in ap-northeast-2b
# (Replace snap-0xyz456 with the actual snapshot ID from Step 2)
aws ec2 create-volume \
    --snapshot-id snap-0xyz456 \
    --availability-zone ap-northeast-2b \
    --volume-type gp3 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=restored-from-pre-os-update}]'
```

**Key insight**: Snapshots are regional resources (not AZ-specific). You can restore a snapshot to any AZ within the same region, which enables both cross-AZ disaster recovery and volume migration.

</details>

### Exercise 4: EFS vs EBS Architecture Decision

A web application team is debating two storage architectures:

**Option A**: Each EC2 web server has its own EBS volume for user-uploaded files.
**Option B**: All EC2 web servers mount a shared EFS filesystem for user-uploaded files.

Analyze the implications of each option for: (1) data consistency, (2) scaling, and (3) cost.

<details>
<summary>Show Answer</summary>

| Aspect | Option A: EBS per instance | Option B: Shared EFS |
|--------|---------------------------|----------------------|
| **Data consistency** | Files uploaded to one server are NOT visible on other servers. Users get different responses depending on which server handles their request — a critical bug for file storage. | All servers see the same filesystem. A file uploaded through Server 1 is immediately readable on Server 2. |
| **Scaling** | When adding new EC2 instances, they start with empty volumes. Files must be manually synchronized across instances (or a CDN/S3 layer must be added). | New instances automatically have access to all existing files by mounting EFS. Scaling out is seamless. |
| **Cost** | EBS: ~$0.10/GB/month (gp3). For 100 GB × 10 servers = 1 TB EBS = **$100/month** | EFS Standard: ~$0.30/GB/month for 100 GB of actual data (shared) = **$30/month**. EFS is more expensive per GB but you only pay for one copy. |

**Conclusion**: Option B (EFS) is the correct architecture for shared web server file storage. Option A requires building a synchronization mechanism (e.g., S3 + sync script) to work correctly, negating any cost advantage.

**Best practice**: For truly high-scale web applications, store user uploads directly to S3 (using pre-signed POST) and serve them via CloudFront. This eliminates the need for EFS entirely and provides global CDN delivery.

</details>

### Exercise 5: Storage Cost Comparison

Calculate the monthly cost for the following storage requirements and determine the cheapest option that meets the requirements:

Requirements: Store 10 TB of data that is accessed twice per month (sequential reads of ~100 GB each time) and must be available within 15 minutes.

Choose between:
- EBS `st1` (Throughput-optimized HDD): $0.045/GB/month
- S3 Standard-IA: $0.0138/GB/month + $0.01 per GB retrieval
- S3 Glacier Flexible Retrieval: $0.005/GB/month + $0.01 per GB retrieval (expedited tier: ~15 min)

<details>
<summary>Show Answer</summary>

**Option 1: EBS st1**
- Storage: 10,000 GB × $0.045 = **$450/month**
- Retrieval: $0 (attached storage, no per-access fee)
- Total: **$450/month**

**Option 2: S3 Standard-IA**
- Storage: 10,000 GB × $0.0138 = **$138/month**
- Retrieval: 2 reads × 100 GB × $0.01/GB = **$2/month**
- Total: **$140/month**

**Option 3: S3 Glacier Flexible Retrieval (Expedited)**
- Storage: 10,000 GB × $0.005 = **$50/month**
- Retrieval: Expedited retrieval (1–5 minutes, meets 15-min SLA) costs ~$0.03/GB
  - 2 reads × 100 GB × $0.03 = **$6/month**
- Expedited requests: $0.01 per request (negligible)
- Total: **~$56/month**

**Winner: S3 Glacier Flexible Retrieval** at ~$56/month — 87% cheaper than EBS and meets the 15-minute availability requirement using the Expedited retrieval tier.

**Important caveat**: Glacier Expedited retrieval is best-effort and may not be available during peak demand. For guaranteed 15-minute availability, consider S3 Glacier Instant Retrieval ($0.004/GB/month + $0.03/GB retrieval) which provides millisecond access.

</details>

---

## References

- [AWS EBS Documentation](https://docs.aws.amazon.com/ebs/)
- [AWS EFS Documentation](https://docs.aws.amazon.com/efs/)
- [GCP Persistent Disk](https://cloud.google.com/compute/docs/disks)
- [GCP Filestore](https://cloud.google.com/filestore/docs)
