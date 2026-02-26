# Object Storage (S3 / Cloud Storage)

**Previous**: [Container Services](./06_Container_Services.md) | **Next**: [Block and File Storage](./08_Block_and_File_Storage.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the object storage model and how it differs from block and file storage
2. Compare AWS S3 and GCP Cloud Storage features, storage classes, and pricing
3. Create buckets, upload objects, and configure access policies
4. Implement lifecycle rules to automatically transition objects between storage classes
5. Configure versioning and cross-region replication for data durability
6. Apply encryption (server-side and client-side) and access controls to stored objects
7. Design a cost-effective storage strategy by selecting appropriate storage tiers

---

Object storage is the backbone of cloud data management. From application assets and log archives to data lake files and backup snapshots, virtually every cloud workload stores data in object storage at some point. Its virtually unlimited capacity, high durability, and pay-per-use pricing make it one of the most widely used cloud services across all industries.

## 1. Object Storage Overview

### 1.1 What is Object Storage?

Object storage is a storage architecture that stores data as discrete objects.

**Object Components:**
- **Data**: Actual file content
- **Metadata**: File information (creation date, size, custom attributes)
- **Unique Identifier**: Key to locate the object

### 1.2 Service Comparison

| Category | AWS S3 | GCP Cloud Storage |
|------|--------|------------------|
| Service Name | Simple Storage Service | Cloud Storage |
| Container Unit | Bucket | Bucket |
| Max Object Size | 5TB | 5TB |
| Multipart Upload | Supported (5MB-5GB parts) | Supported (composite upload) |
| Versioning | Versioning | Object Versioning |
| Lifecycle | Lifecycle Rules | Lifecycle Management |
| Encryption | SSE-S3, SSE-KMS, SSE-C | Google-managed, CMEK, CSEK |

---

## 2. Storage Classes

### 2.1 AWS S3 Storage Classes

| Class | Use Case | Availability | Minimum Storage Duration |
|--------|------|--------|---------------|
| **S3 Standard** | Frequent access | 99.99% | - |
| **S3 Intelligent-Tiering** | Unknown access patterns | 99.9% | - |
| **S3 Standard-IA** | Infrequent access | 99.9% | 30 days |
| **S3 One Zone-IA** | Infrequent access (single AZ) | 99.5% | 30 days |
| **S3 Glacier Instant** | Archive (instant access) | 99.9% | 90 days |
| **S3 Glacier Flexible** | Archive (minutes to hours) | 99.99% | 90 days |
| **S3 Glacier Deep Archive** | Long-term archive | 99.99% | 180 days |

### 2.2 GCP Cloud Storage Classes

| Class | Use Case | Availability SLA | Minimum Storage Duration |
|--------|------|-----------|---------------|
| **Standard** | Frequent access | 99.95% (regional) | - |
| **Nearline** | Less than once per month | 99.9% | 30 days |
| **Coldline** | Less than once per quarter | 99.9% | 90 days |
| **Archive** | Less than once per year | 99.9% | 365 days |

### 2.3 Cost Comparison (Seoul Region)

| Class | S3 ($/GB/month) | GCS ($/GB/month) |
|--------|-------------|---------------|
| Standard | $0.025 | $0.023 |
| Infrequent Access | $0.0138 | $0.016 (Nearline) |
| Archive | $0.005 (Glacier) | $0.0025 (Archive) |

*Prices are subject to change*

---

## 3. Bucket Creation and Management

### 3.1 AWS S3 Buckets

```bash
# Create bucket
aws s3 mb s3://my-unique-bucket-name-2024 --region ap-northeast-2

# List buckets
aws s3 ls

# List bucket contents
aws s3 ls s3://my-bucket/

# Delete bucket (must be empty)
aws s3 rb s3://my-bucket

# Delete bucket (including contents)
aws s3 rb s3://my-bucket --force
```

**Bucket Naming Rules:**
- Globally unique
- 3-63 characters
- Lowercase letters, numbers, hyphens only
- Must start/end with letter or number

### 3.2 GCP Cloud Storage Buckets

```bash
# Create bucket
gsutil mb -l asia-northeast3 gs://my-unique-bucket-name-2024

# Or use gcloud
gcloud storage buckets create gs://my-bucket \
    --location=asia-northeast3

# List buckets
gsutil ls
# Or
gcloud storage buckets list

# List bucket contents
gsutil ls gs://my-bucket/

# Delete bucket
gsutil rb gs://my-bucket

# Delete bucket (including contents)
gsutil rm -r gs://my-bucket
```

---

## 4. Object Upload/Download

### 4.1 AWS S3 Object Operations

```bash
# Upload single file
aws s3 cp myfile.txt s3://my-bucket/

# Upload folder (recursive)
aws s3 cp ./local-folder s3://my-bucket/remote-folder --recursive

# Download file
aws s3 cp s3://my-bucket/myfile.txt ./

# Download folder
aws s3 cp s3://my-bucket/folder ./local-folder --recursive

# Sync (changed files only)
aws s3 sync ./local-folder s3://my-bucket/folder
aws s3 sync s3://my-bucket/folder ./local-folder

# Delete file
aws s3 rm s3://my-bucket/myfile.txt

# Delete folder
aws s3 rm s3://my-bucket/folder --recursive

# Move file
aws s3 mv s3://my-bucket/file1.txt s3://my-bucket/folder/file1.txt

# Copy file
aws s3 cp s3://source-bucket/file.txt s3://dest-bucket/file.txt
```

### 4.2 GCP Cloud Storage Object Operations

```bash
# Upload single file
gsutil cp myfile.txt gs://my-bucket/

# Or use gcloud
gcloud storage cp myfile.txt gs://my-bucket/

# Upload folder (recursive)
gsutil cp -r ./local-folder gs://my-bucket/

# Download file
gsutil cp gs://my-bucket/myfile.txt ./

# Download folder
gsutil cp -r gs://my-bucket/folder ./

# Sync
gsutil rsync -r ./local-folder gs://my-bucket/folder

# Delete file
gsutil rm gs://my-bucket/myfile.txt

# Delete folder
gsutil rm -r gs://my-bucket/folder

# Move file
gsutil mv gs://my-bucket/file1.txt gs://my-bucket/folder/

# Copy file
gsutil cp gs://source-bucket/file.txt gs://dest-bucket/
```

### 4.3 Large File Upload

**AWS S3 Multipart Upload:**
```bash
# AWS CLI automatically uses multipart upload (8MB and above)
aws s3 cp large-file.zip s3://my-bucket/ \
    --expected-size 10737418240  # 10GB

# Adjust multipart settings
aws configure set s3.multipart_threshold 64MB
aws configure set s3.multipart_chunksize 16MB
```

**GCP Composite Upload:**
```bash
# gsutil automatically uses composite upload (150MB and above)
gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
    cp large-file.zip gs://my-bucket/
```

---

## 5. Lifecycle Management

### 5.1 AWS S3 Lifecycle

```json
{
    "Rules": [
        {
            "ID": "Move to IA after 30 days",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": 365
            }
        },
        {
            "ID": "Delete old versions",
            "Status": "Enabled",
            "Filter": {},
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 30
            }
        }
    ]
}
```

```bash
# Apply lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json

# Retrieve lifecycle policy
aws s3api get-bucket-lifecycle-configuration --bucket my-bucket
```

### 5.2 GCP Lifecycle Management

```json
{
    "lifecycle": {
        "rule": [
            {
                "action": {
                    "type": "SetStorageClass",
                    "storageClass": "NEARLINE"
                },
                "condition": {
                    "age": 30,
                    "matchesPrefix": ["logs/"]
                }
            },
            {
                "action": {
                    "type": "SetStorageClass",
                    "storageClass": "COLDLINE"
                },
                "condition": {
                    "age": 90
                }
            },
            {
                "action": {
                    "type": "Delete"
                },
                "condition": {
                    "age": 365
                }
            }
        ]
    }
}
```

```bash
# Apply lifecycle policy
gsutil lifecycle set lifecycle.json gs://my-bucket

# Retrieve lifecycle policy
gsutil lifecycle get gs://my-bucket
```

---

## 6. Access Control

### 6.1 AWS S3 Access Control

**Bucket Policy:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicRead",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        }
    ]
}
```

```bash
# Apply bucket policy
aws s3api put-bucket-policy \
    --bucket my-bucket \
    --policy file://bucket-policy.json

# Block public access (recommended)
aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**Presigned URL:**
```bash
# Generate download URL (valid for 1 hour)
aws s3 presign s3://my-bucket/private-file.pdf --expires-in 3600

# Generate upload URL
aws s3 presign s3://my-bucket/uploads/file.txt --expires-in 3600
```

### 6.2 GCP Cloud Storage Access Control

**IAM Policy:**
```bash
# Grant user access to bucket
gsutil iam ch user:user@example.com:objectViewer gs://my-bucket

# Grant read access to all users (public)
gsutil iam ch allUsers:objectViewer gs://my-bucket
```

**Uniform Bucket-Level Access (Recommended):**
```bash
# Enable uniform access
gsutil uniformbucketlevelaccess set on gs://my-bucket
```

**Signed URL:**
```bash
# Generate download URL (valid for 1 hour)
gsutil signurl -d 1h service-account.json gs://my-bucket/private-file.pdf

# Using gcloud
gcloud storage sign-url gs://my-bucket/file.pdf \
    --private-key-file=key.json \
    --duration=1h
```

---

## 7. Static Website Hosting

### 7.1 AWS S3 Static Hosting

```bash
# 1. Enable static website hosting
aws s3 website s3://my-bucket/ \
    --index-document index.html \
    --error-document error.html

# 2. Allow public access (unblock)
aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# 3. Bucket policy (public read)
aws s3api put-bucket-policy --bucket my-bucket --policy '{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-bucket/*"
    }]
}'

# 4. Upload files
aws s3 sync ./website s3://my-bucket/

# Website URL: http://my-bucket.s3-website.ap-northeast-2.amazonaws.com
```

### 7.2 GCP Cloud Storage Static Hosting

```bash
# 1. Create bucket (custom domain possible if bucket name matches domain)
gsutil mb -l asia-northeast3 gs://www.example.com

# 2. Configure website
gsutil web set -m index.html -e 404.html gs://my-bucket

# 3. Allow public access
gsutil iam ch allUsers:objectViewer gs://my-bucket

# 4. Upload files
gsutil cp -r ./website/* gs://my-bucket/

# Website URL: https://storage.googleapis.com/my-bucket/index.html
# Custom domain can be configured via load balancer
```

---

## 8. Versioning

### 8.1 AWS S3 Versioning

```bash
# Enable versioning
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled

# Check versioning status
aws s3api get-bucket-versioning --bucket my-bucket

# List all versions
aws s3api list-object-versions --bucket my-bucket

# Download specific version
aws s3api get-object \
    --bucket my-bucket \
    --key myfile.txt \
    --version-id "abc123" \
    myfile-old.txt

# Delete specific version
aws s3api delete-object \
    --bucket my-bucket \
    --key myfile.txt \
    --version-id "abc123"
```

### 8.2 GCP Object Versioning

```bash
# Enable versioning
gsutil versioning set on gs://my-bucket

# Check versioning status
gsutil versioning get gs://my-bucket

# List all versions
gsutil ls -a gs://my-bucket/

# Download specific version
gsutil cp gs://my-bucket/myfile.txt#1234567890123456 ./

# Delete specific version
gsutil rm gs://my-bucket/myfile.txt#1234567890123456
```

---

## 9. Cross-Region Replication

### 9.1 AWS S3 Cross-Region Replication

```bash
# 1. Enable versioning on source bucket
aws s3api put-bucket-versioning \
    --bucket source-bucket \
    --versioning-configuration Status=Enabled

# 2. Create destination bucket and enable versioning
aws s3 mb s3://dest-bucket --region eu-west-1
aws s3api put-bucket-versioning \
    --bucket dest-bucket \
    --versioning-configuration Status=Enabled

# 3. Configure replication rule
aws s3api put-bucket-replication \
    --bucket source-bucket \
    --replication-configuration '{
        "Role": "arn:aws:iam::123456789012:role/s3-replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Priority": 1,
            "DeleteMarkerReplication": {"Status": "Disabled"},
            "Filter": {},
            "Destination": {
                "Bucket": "arn:aws:s3:::dest-bucket"
            }
        }]
    }'
```

### 9.2 GCP Dual/Multi-Region Buckets

```bash
# Create dual-region bucket
gsutil mb -l asia1 gs://my-dual-region-bucket

# Or multi-region bucket
gsutil mb -l asia gs://my-multi-region-bucket

# Cross-region copy (manual)
gsutil cp -r gs://source-bucket/* gs://dest-bucket/
```

---

## 10. SDK Usage Examples

### 10.1 Python (boto3 / google-cloud-storage)

**AWS S3 (boto3):**
```python
import boto3

s3 = boto3.client('s3')

# Upload
s3.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

# Download
s3.download_file('my-bucket', 'remote_file.txt', 'local_file.txt')

# List objects
response = s3.list_objects_v2(Bucket='my-bucket', Prefix='folder/')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Generate Presigned URL
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'file.txt'},
    ExpiresIn=3600
)
```

**GCP Cloud Storage:**
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')

# Upload
blob = bucket.blob('remote_file.txt')
blob.upload_from_filename('local_file.txt')

# Download
blob = bucket.blob('remote_file.txt')
blob.download_to_filename('local_file.txt')

# List objects
blobs = client.list_blobs('my-bucket', prefix='folder/')
for blob in blobs:
    print(blob.name)

# Generate Signed URL
from datetime import timedelta
url = blob.generate_signed_url(expiration=timedelta(hours=1))
```

---

## 11. Next Steps

- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - Block Storage
- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - Using with CDN

---

## Exercises

### Exercise 1: Storage Class Selection

A media company stores the following categories of files in S3. Choose the most cost-effective storage class for each and justify your answer:

1. Thumbnail images for an active social media feed — accessed thousands of times per day.
2. Raw video footage uploaded by creators — accessed frequently in the first 48 hours, then rarely.
3. Quarterly financial reports that must be retained for 7 years for compliance but are almost never read.
4. System log files — accessed occasionally for debugging within the first 30 days, never after.

<details>
<summary>Show Answer</summary>

1. **S3 Standard** — High-frequency access (thousands of times/day) is the core use case for Standard. The higher storage cost is justified by eliminating retrieval fees. Using Intelligent-Tiering or IA would generate retrieval fees that would far exceed the storage savings.

2. **S3 Intelligent-Tiering** — The access pattern shifts dramatically (frequent → infrequent) after 48 hours, but the exact pattern for each file may vary. Intelligent-Tiering automatically moves objects between frequent/infrequent tiers without retrieval fees or minimum duration penalties, making it ideal for shifting patterns.

3. **S3 Glacier Deep Archive** — Objects accessed less than once per year and held for 7 years are the classic Deep Archive use case. At $0.00099/GB/month, it is 25x cheaper than Standard. The 180-day minimum storage duration is easily satisfied by a 7-year retention policy. Retrieval times of 12 hours are acceptable for compliance documents rarely accessed.

4. **S3 Standard-IA** — Logs accessed occasionally in the first 30 days benefit from Standard-IA's lower storage cost vs Standard. After 30 days, add a lifecycle rule to transition to Glacier Flexible Retrieval or Deep Archive. The 30-day minimum storage duration aligns with the active period.

</details>

### Exercise 2: Lifecycle Policy Design

Design an S3 lifecycle policy for application log files with the following requirements:
- Logs are actively analyzed for the first 7 days.
- From day 7–30, they may be accessed for occasional debugging.
- From day 30–365, they are kept for compliance but rarely accessed.
- After 1 year, they can be deleted.

Write the lifecycle rule configuration (describe the transitions and expiration in JSON or plain text).

<details>
<summary>Show Answer</summary>

```json
{
  "Rules": [
    {
      "ID": "log-lifecycle",
      "Status": "Enabled",
      "Filter": {"Prefix": "logs/"},
      "Transitions": [
        {
          "Days": 7,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 30,
          "StorageClass": "GLACIER_FLEXIBLE_RETRIEVAL"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    }
  ]
}
```

**Explanation of transitions**:
- **Day 0–7**: S3 Standard — active analysis requires fast, free retrieval.
- **Day 7–30**: S3 Standard-IA — occasional debugging access; lower storage cost, per-request retrieval fee is acceptable.
- **Day 30–365**: S3 Glacier Flexible Retrieval — compliance retention at very low cost; 1–12 hour retrieval time is fine for rare access.
- **Day 365+**: Objects expire and are automatically deleted, eliminating storage cost.

**AWS CLI to apply**:
```bash
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-log-bucket \
    --lifecycle-configuration file://lifecycle.json
```

</details>

### Exercise 3: Bucket Versioning and Public Access

A development team stores production configuration files in an S3 bucket. They want to:
1. Prevent accidental deletion of configuration files.
2. Ensure the bucket is never publicly accessible.

Provide the AWS CLI commands to implement both requirements.

<details>
<summary>Show Answer</summary>

```bash
# 1. Enable versioning on the bucket
# With versioning enabled, deleted objects are marked with a delete marker
# (not actually removed) and overwritten files keep the old version.
aws s3api put-bucket-versioning \
    --bucket my-config-bucket \
    --versioning-configuration Status=Enabled

# 2. Block all public access at the bucket level
aws s3api put-public-access-block \
    --bucket my-config-bucket \
    --public-access-block-configuration \
        BlockPublicAcls=true,\
        IgnorePublicAcls=true,\
        BlockPublicPolicy=true,\
        RestrictPublicBuckets=true
```

**Additional protection** — Add an MFA Delete requirement to prevent permanent deletion even by authorized users:
```bash
aws s3api put-bucket-versioning \
    --bucket my-config-bucket \
    --versioning-configuration Status=Enabled,MFADelete=Enabled \
    --mfa "arn:aws:iam::ACCOUNT_ID:mfa/USER_DEVICE CURRENT_CODE"
```

**Effect of versioning on deletion**:
- `aws s3 rm s3://my-config-bucket/prod.yaml` — Creates a delete marker; the object is hidden but recoverable.
- To permanently delete, you must explicitly delete the specific version ID.

</details>

### Exercise 4: Pre-Signed URL Use Case

Your application needs to let users download a private S3 object (`reports/q3-summary.pdf`) for exactly 1 hour without making the bucket public. Write the AWS CLI command to generate a pre-signed URL and explain how it works.

<details>
<summary>Show Answer</summary>

```bash
aws s3 presign s3://my-reports-bucket/reports/q3-summary.pdf \
    --expires-in 3600
```

This produces a URL like:
```
https://my-reports-bucket.s3.amazonaws.com/reports/q3-summary.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...&X-Amz-Expires=3600&X-Amz-Signature=...
```

**How it works**:
1. AWS signs the URL using the credentials of the IAM entity that generated it (you).
2. The signature encodes: the bucket, object key, expiry time, and your identity.
3. Anyone who has the URL can access the object via an HTTP GET request — no AWS credentials needed.
4. After 3,600 seconds (1 hour), the signature becomes invalid and the URL returns a 403 Forbidden error.

**Security consideration**: Pre-signed URLs inherit the permissions of the generating IAM identity. If the generating role is deleted or loses S3 read permission before the URL expires, the URL will also stop working.

</details>

### Exercise 5: Cross-Region Replication Setup

A company has a primary S3 bucket in `ap-northeast-2` (Seoul) and needs to replicate all new objects to `us-east-1` (Virginia) for disaster recovery. Describe the prerequisites and the key steps required to configure S3 Cross-Region Replication (CRR).

<details>
<summary>Show Answer</summary>

**Prerequisites**:
1. **Versioning must be enabled on both the source and destination buckets** — CRR requires versioning on both ends.
2. **An IAM role** that grants S3 permission to read from the source bucket and write to the destination bucket.

**Steps**:

```bash
# Step 1: Enable versioning on source bucket (Seoul)
aws s3api put-bucket-versioning \
    --bucket source-bucket-seoul \
    --region ap-northeast-2 \
    --versioning-configuration Status=Enabled

# Step 2: Create destination bucket in us-east-1 and enable versioning
aws s3api create-bucket \
    --bucket destination-bucket-virginia \
    --region us-east-1 \
    --create-bucket-configuration LocationConstraint=us-east-1

aws s3api put-bucket-versioning \
    --bucket destination-bucket-virginia \
    --region us-east-1 \
    --versioning-configuration Status=Enabled

# Step 3: Configure replication rule
aws s3api put-bucket-replication \
    --bucket source-bucket-seoul \
    --region ap-northeast-2 \
    --replication-configuration '{
        "Role": "arn:aws:iam::ACCOUNT_ID:role/s3-replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Destination": {
                "Bucket": "arn:aws:s3:::destination-bucket-virginia"
            }
        }]
    }'
```

**Important notes**:
- CRR only replicates new objects written after the rule is configured. Existing objects are NOT replicated automatically; use **S3 Batch Operations** for that.
- Delete markers are not replicated by default (configurable).
- Data transfer from Seoul to Virginia incurs egress charges.

</details>

---

## References

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [GCP Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [S3 Pricing](https://aws.amazon.com/s3/pricing/)
- [Cloud Storage Pricing](https://cloud.google.com/storage/pricing)
