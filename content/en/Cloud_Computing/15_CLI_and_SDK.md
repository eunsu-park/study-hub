# CLI & SDK

**Previous**: [Security Services](./14_Security_Services.md) | **Next**: [Infrastructure as Code](./16_Infrastructure_as_Code.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and configure the AWS CLI and gcloud CLI on major operating systems
2. Authenticate and manage multiple profiles/configurations for different environments
3. Perform common cloud operations (create, list, describe, delete resources) from the command line
4. Use output formatting and filtering (--query, --filter) to extract specific information
5. Integrate cloud SDKs (Boto3, Google Cloud Client Libraries) into Python scripts
6. Automate repetitive cloud tasks using CLI scripts and SDK programs

---

The web console is fine for exploration, but real cloud work happens on the command line and in code. CLI tools and SDKs let you script repeatable operations, integrate cloud management into CI/CD pipelines, and build applications that interact with cloud services programmatically. Proficiency with these tools is what separates a console clicker from a cloud engineer.

## 1. CLI Overview

### 1.1 AWS CLI vs gcloud CLI

| Item | AWS CLI | gcloud CLI |
|------|---------|------------|
| Installation Package | awscli | google-cloud-sdk |
| Configuration Command | aws configure | gcloud init |
| Profiles | --profile | --configuration |
| Output Format | json, text, table, yaml | json, text, yaml, csv |

---

## 2. AWS CLI

### 2.1 Installation

```bash
# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# pip
pip install awscli

# Check version
aws --version
```

### 2.2 Configuration

```bash
# Basic configuration
aws configure
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region name: ap-northeast-2
# Default output format: json

# Add profile
aws configure --profile production
aws configure --profile development

# List profiles
aws configure list-profiles

# Set via environment variables
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI...
export AWS_DEFAULT_REGION=ap-northeast-2
export AWS_PROFILE=production
```

**~/.aws/credentials:**
```ini
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI...

[production]
aws_access_key_id = AKIAI44QH8DHBEXAMPLE
aws_secret_access_key = je7MtGbClwBF/2Zp9Utk...
```

**~/.aws/config:**
```ini
[default]
region = ap-northeast-2
output = json

[profile production]
region = ap-northeast-1
output = table
```

### 2.3 Common Commands

```bash
# EC2
aws ec2 describe-instances
aws ec2 run-instances --image-id ami-xxx --instance-type t3.micro
aws ec2 start-instances --instance-ids i-xxx
aws ec2 stop-instances --instance-ids i-xxx
aws ec2 terminate-instances --instance-ids i-xxx

# S3
aws s3 ls
aws s3 cp file.txt s3://bucket/
aws s3 sync ./folder s3://bucket/folder
aws s3 rm s3://bucket/file.txt

# IAM
aws iam list-users
aws iam create-user --user-name john
aws iam attach-user-policy --user-name john --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess

# Lambda
aws lambda list-functions
aws lambda invoke --function-name my-func output.json

# RDS
aws rds describe-db-instances
aws rds create-db-snapshot --db-instance-identifier mydb --db-snapshot-identifier mysnap
```

### 2.4 Output Filtering (--query)

```bash
# Use JMESPath query
aws ec2 describe-instances \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
    --output table

# Filter by specific tag
aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text

# Sort
aws ec2 describe-instances \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)[*].[InstanceId,LaunchTime]'

# Conditional filter
aws ec2 describe-instances \
    --query 'Reservations[].Instances[?State.Name==`running`].InstanceId'
```

---

## 3. gcloud CLI

### 3.1 Installation

```bash
# macOS
brew install --cask google-cloud-sdk

# Or direct installation
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Linux
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Check version
gcloud --version
```

### 3.2 Configuration

```bash
# Initialize (browser authentication)
gcloud init

# Login
gcloud auth login

# Service account authentication
gcloud auth activate-service-account --key-file=key.json

# Set project
gcloud config set project PROJECT_ID

# Set region/zone
gcloud config set compute/region asia-northeast3
gcloud config set compute/zone asia-northeast3-a

# View current configuration
gcloud config list

# Configuration profiles
gcloud config configurations create production
gcloud config configurations activate production
gcloud config configurations list
```

### 3.3 Common Commands

```bash
# Compute Engine
gcloud compute instances list
gcloud compute instances create my-vm --machine-type=e2-medium
gcloud compute instances start my-vm
gcloud compute instances stop my-vm
gcloud compute instances delete my-vm
gcloud compute ssh my-vm

# Cloud Storage
gsutil ls
gsutil cp file.txt gs://bucket/
gsutil rsync -r ./folder gs://bucket/folder
gsutil rm gs://bucket/file.txt

# IAM
gcloud iam service-accounts list
gcloud iam service-accounts create my-sa
gcloud projects add-iam-policy-binding PROJECT --member=user:john@example.com --role=roles/viewer

# Cloud Functions
gcloud functions list
gcloud functions deploy my-func --runtime=python312 --trigger-http

# Cloud SQL
gcloud sql instances list
gcloud sql instances create mydb --database-version=MYSQL_8_0
```

### 3.4 Output Filtering (--filter, --format)

```bash
# Filtering
gcloud compute instances list \
    --filter="status=RUNNING AND zone:asia-northeast3"

# Output format
gcloud compute instances list \
    --format="table(name,zone.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)"

# JSON output
gcloud compute instances describe my-vm --format=json

# Specific fields only
gcloud compute instances list \
    --format="value(name,networkInterfaces[0].accessConfigs[0].natIP)"

# CSV
gcloud compute instances list \
    --format="csv(name,zone,status)"
```

---

## 4. Python SDK

### 4.1 AWS SDK (boto3)

**Installation:**
```bash
pip install boto3
```

**Basic Usage:**
```python
import boto3

# Client approach
ec2_client = boto3.client('ec2')
response = ec2_client.describe_instances()

# Resource approach (high-level)
ec2 = boto3.resource('ec2')
instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)
for instance in instances:
    print(instance.id, instance.public_ip_address)
```

**Service Examples:**
```python
import boto3

# S3
s3 = boto3.client('s3')
s3.upload_file('file.txt', 'my-bucket', 'file.txt')
s3.download_file('my-bucket', 'file.txt', 'downloaded.txt')

# DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')
table.put_item(Item={'userId': '001', 'name': 'John'})
response = table.get_item(Key={'userId': '001'})

# Lambda
lambda_client = boto3.client('lambda')
response = lambda_client.invoke(
    FunctionName='my-function',
    Payload=json.dumps({'key': 'value'})
)

# SQS
sqs = boto3.client('sqs')
sqs.send_message(
    QueueUrl='https://sqs.../my-queue',
    MessageBody='Hello'
)

# Secrets Manager
secrets = boto3.client('secretsmanager')
secret = secrets.get_secret_value(SecretId='my-secret')
```

### 4.2 GCP SDK (google-cloud)

**Installation:**
```bash
pip install google-cloud-storage
pip install google-cloud-compute
pip install google-cloud-firestore
# Install per library as needed
```

**Basic Usage:**
```python
from google.cloud import storage
from google.cloud import compute_v1

# Authentication (service account)
# export GOOGLE_APPLICATION_CREDENTIALS="key.json"

# Cloud Storage
client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('file.txt')
blob.upload_from_filename('file.txt')
blob.download_to_filename('downloaded.txt')

# Compute Engine
instance_client = compute_v1.InstancesClient()
instances = instance_client.list(project='my-project', zone='asia-northeast3-a')
for instance in instances:
    print(instance.name, instance.status)
```

**Service Examples:**
```python
# Firestore
from google.cloud import firestore

db = firestore.Client()
doc_ref = db.collection('users').document('001')
doc_ref.set({'name': 'John', 'age': 30})
doc = doc_ref.get()
print(doc.to_dict())

# Pub/Sub
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('project', 'topic')
publisher.publish(topic_path, b'Hello')

# Secret Manager
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/PROJECT/secrets/my-secret/versions/latest"
response = client.access_secret_version(request={"name": name})
secret = response.payload.data.decode("UTF-8")

# BigQuery
from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT * FROM `project.dataset.table` LIMIT 10"
results = client.query(query)
for row in results:
    print(row)
```

---

## 5. Automation Script Examples

### 5.1 Resource Cleanup Script

**AWS - Delete Unused EBS Volumes:**
```python
import boto3

ec2 = boto3.client('ec2')

# Find unused volumes
volumes = ec2.describe_volumes(
    Filters=[{'Name': 'status', 'Values': ['available']}]
)

for vol in volumes['Volumes']:
    vol_id = vol['VolumeId']
    print(f"Deleting unused volume: {vol_id}")
    # ec2.delete_volume(VolumeId=vol_id)  # Uncomment to actually delete
```

**GCP - Delete Old Snapshots:**
```python
from google.cloud import compute_v1
from datetime import datetime, timedelta

client = compute_v1.SnapshotsClient()
project = 'my-project'

snapshots = client.list(project=project)
cutoff = datetime.now() - timedelta(days=30)

for snapshot in snapshots:
    created = datetime.fromisoformat(snapshot.creation_timestamp.replace('Z', '+00:00'))
    if created < cutoff.replace(tzinfo=created.tzinfo):
        print(f"Deleting old snapshot: {snapshot.name}")
        # client.delete(project=project, snapshot=snapshot.name)
```

### 5.2 Deployment Script

**AWS Lambda Deployment:**
```python
import boto3
import zipfile
import os

def deploy_lambda(function_name, source_dir):
    # Zip code
    with zipfile.ZipFile('function.zip', 'w') as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                zf.write(os.path.join(root, file))

    # Update Lambda
    lambda_client = boto3.client('lambda')
    with open('function.zip', 'rb') as f:
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=f.read()
        )

    print(f"Deployed {function_name}")

deploy_lambda('my-function', './src')
```

### 5.3 Monitoring Script

**AWS Instance Status Check:**
```bash
#!/bin/bash
# check-instances.sh

instances=$(aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
    --output text)

while read -r instance_id state; do
    if [ "$state" != "running" ]; then
        echo "WARNING: $instance_id is $state"
        # Send notification
        aws sns publish \
            --topic-arn arn:aws:sns:...:alerts \
            --message "Instance $instance_id is $state"
    fi
done <<< "$instances"
```

---

## 6. Pagination Handling

### 6.1 AWS CLI Pagination

```bash
# Automatic pagination
aws s3api list-objects-v2 --bucket my-bucket

# Manual pagination
aws s3api list-objects-v2 --bucket my-bucket --max-items 100

# Next page
aws s3api list-objects-v2 --bucket my-bucket --starting-token TOKEN
```

**boto3 Paginator:**
```python
import boto3

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket='my-bucket'):
    for obj in page.get('Contents', []):
        print(obj['Key'])
```

### 6.2 gcloud Pagination

```bash
# Automatic pagination
gcloud compute instances list

# Manual
gcloud compute instances list --limit=100 --page-token=TOKEN
```

**Python:**
```python
from google.cloud import storage

client = storage.Client()
blobs = client.list_blobs('my-bucket')  # Automatic pagination

for blob in blobs:
    print(blob.name)
```

---

## 7. Error Handling

### 7.1 boto3 Error Handling

```python
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    s3 = boto3.client('s3')
    s3.head_object(Bucket='my-bucket', Key='file.txt')
except NoCredentialsError:
    print("Credentials not found")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        print("Object not found")
    elif error_code == '403':
        print("Access denied")
    else:
        raise
```

### 7.2 GCP Error Handling

```python
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden

try:
    client = storage.Client()
    bucket = client.get_bucket('my-bucket')
    blob = bucket.blob('file.txt')
    blob.download_to_filename('file.txt')
except NotFound:
    print("Bucket or object not found")
except Forbidden:
    print("Access denied")
```

---

## 8. Next Steps

- [16_Infrastructure_as_Code.md](./16_Infrastructure_as_Code.md) - Terraform
- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - Monitoring

---

## Exercises

### Exercise 1: CLI Profile Management

You work with three AWS environments: `dev`, `staging`, and `production`. Each environment uses a different AWS account and region. Describe how to configure your AWS CLI so you can switch between them, and write the commands to list all S3 buckets in the `production` account without changing your default profile permanently.

<details>
<summary>Show Answer</summary>

**Configuration steps:**

```bash
# Configure each profile
aws configure --profile dev
# Enter dev account key, secret, region (ap-northeast-2), format (json)

aws configure --profile staging
# Enter staging account credentials

aws configure --profile production
# Enter production account credentials

# Verify all profiles are set
aws configure list-profiles
```

**~/.aws/credentials** (result):
```ini
[default]
aws_access_key_id = ...
aws_secret_access_key = ...

[dev]
aws_access_key_id = AKIA...DEV
aws_secret_access_key = ...

[staging]
aws_access_key_id = AKIA...STG
aws_secret_access_key = ...

[production]
aws_access_key_id = AKIA...PRD
aws_secret_access_key = ...
```

**List S3 buckets in production without changing the default:**
```bash
# Option 1: --profile flag
aws s3 ls --profile production

# Option 2: environment variable (temporary, for the current shell session)
export AWS_PROFILE=production
aws s3 ls

# Option 3: per-command environment variable
AWS_PROFILE=production aws s3 ls
```

Best practice: Use `--profile` in scripts so the profile is explicit and not dependent on shell state.

</details>

---

### Exercise 2: JMESPath Query Filtering

You need a report of all running EC2 instances in the `Production` environment tag, showing only their Instance ID, instance type, and public IP. Write the AWS CLI command with `--query` and `--output table`.

<details>
<summary>Show Answer</summary>

```bash
aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId, InstanceType, PublicIpAddress]' \
    --output table
```

Explanation of the JMESPath expression:
- `Reservations[*]` — iterate all reservation groups (AWS groups instances by launch request)
- `.Instances[*]` — iterate instances within each reservation
- `.[InstanceId, InstanceType, PublicIpAddress]` — select a multi-value array (becomes table columns)

**Equivalent gcloud command:**
```bash
gcloud compute instances list \
    --filter="status=RUNNING AND labels.environment=production" \
    --format="table(name,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
```

Note: gcloud uses `--filter` (server-side) and `--format` (client-side formatting), while AWS CLI uses `--filters` (server-side) and `--query` (client-side JMESPath).

</details>

---

### Exercise 3: Boto3 Automation Script

Write a Python script using boto3 that:
1. Lists all S3 buckets in the account
2. For each bucket, prints its name and the number of objects it contains
3. Handles the case where a bucket might be in a different region (use `us-east-1` as the default region for the S3 client)

<details>
<summary>Show Answer</summary>

```python
import boto3
from botocore.exceptions import ClientError

def count_bucket_objects(s3_client, bucket_name):
    """Count objects in a bucket using paginator to handle large buckets."""
    paginator = s3_client.get_paginator('list_objects_v2')
    count = 0
    try:
        for page in paginator.paginate(Bucket=bucket_name):
            count += page.get('KeyCount', 0)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            return None
        elif error_code in ('AccessDenied', '403'):
            return 'ACCESS_DENIED'
        else:
            raise
    return count

def main():
    # S3 is global; us-east-1 is the standard endpoint
    s3 = boto3.client('s3', region_name='us-east-1')

    response = s3.list_buckets()
    buckets = response.get('Buckets', [])

    print(f"{'Bucket Name':<50} {'Object Count':>15}")
    print("-" * 66)

    for bucket in buckets:
        name = bucket['Name']
        count = count_bucket_objects(s3, name)
        if count == 'ACCESS_DENIED':
            count_str = 'access denied'
        elif count is None:
            count_str = 'not found'
        else:
            count_str = str(count)
        print(f"{name:<50} {count_str:>15}")

if __name__ == '__main__':
    main()
```

Key points:
- Always use a paginator for `list_objects_v2` — buckets can contain millions of objects
- Catch `ClientError` specifically rather than broad `Exception` to distinguish error types
- `list_buckets` is a global operation; individual bucket operations may need region-specific clients if you encounter redirect errors

</details>

---

### Exercise 4: gcloud Output Formatting

You want to create a shell script that extracts the external IP addresses of all running Compute Engine instances tagged with `http-server` in the `asia-northeast3` region. The output should be plain IP addresses, one per line (for use in another script).

<details>
<summary>Show Answer</summary>

```bash
gcloud compute instances list \
    --filter="status=RUNNING AND tags.items=http-server AND zone:asia-northeast3" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

The `value()` format outputs raw values with no headers or decorators — ideal for shell script consumption.

**Using the output in a script:**
```bash
#!/bin/bash
# health-check.sh — check HTTP health of all http-server instances

ips=$(gcloud compute instances list \
    --filter="status=RUNNING AND tags.items=http-server AND zone:asia-northeast3" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

for ip in $ips; do
    if curl -sf --max-time 5 "http://$ip/health" > /dev/null; then
        echo "OK: $ip"
    else
        echo "FAIL: $ip"
    fi
done
```

Alternative format options:
- `--format="json"` — full JSON for programmatic processing
- `--format="csv(name,natIP)"` — CSV with column headers
- `--format="table(name,zone,natIP)"` — human-readable table

</details>

---

### Exercise 5: SDK Error Handling

A junior developer wrote this GCP Python code to download a file from Cloud Storage. Identify the problems and write an improved version with proper error handling.

```python
# Original (problematic) code
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('data/report.csv')
blob.download_to_filename('/tmp/report.csv')
print("Downloaded!")
```

<details>
<summary>Show Answer</summary>

**Problems with the original:**
1. No error handling — if the bucket or blob doesn't exist, it crashes with an unhandled exception
2. No authentication check — if `GOOGLE_APPLICATION_CREDENTIALS` is not set, the error message is confusing
3. No feedback on what went wrong or where to look

**Improved version:**
```python
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden, GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError

def download_blob(bucket_name: str, source_blob_name: str, destination_file: str) -> bool:
    """
    Download a blob from Cloud Storage.
    Returns True on success, False on failure.
    """
    try:
        client = storage.Client()
    except DefaultCredentialsError:
        print("ERROR: No credentials found. Set GOOGLE_APPLICATION_CREDENTIALS "
              "or run 'gcloud auth application-default login'")
        return False

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file)
        print(f"Downloaded gs://{bucket_name}/{source_blob_name} -> {destination_file}")
        return True

    except NotFound:
        print(f"ERROR: gs://{bucket_name}/{source_blob_name} not found. "
              f"Check bucket name and object path.")
        return False
    except Forbidden:
        print(f"ERROR: Permission denied accessing gs://{bucket_name}/{source_blob_name}. "
              f"Check service account IAM roles.")
        return False
    except GoogleAPICallError as e:
        print(f"ERROR: GCP API call failed: {e.message}")
        return False

# Usage
success = download_blob('my-bucket', 'data/report.csv', '/tmp/report.csv')
if not success:
    exit(1)
```

Key improvements:
- Separate credential initialization from the API call so errors are clearly attributed
- Catch specific exception types (`NotFound`, `Forbidden`) for actionable error messages
- Return a boolean so callers can handle failures programmatically
- Include the full GCS path in error messages for easy debugging

</details>

---

## References

- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)
- [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html)
- [gcloud CLI Documentation](https://cloud.google.com/sdk/gcloud/reference)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Google Cloud Python Client](https://googleapis.dev/python/google-api-core/latest/)
