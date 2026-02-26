# Serverless Functions (Lambda / Cloud Functions)

**Previous**: [Virtual Machines](./04_Virtual_Machines.md) | **Next**: [Container Services](./06_Container_Services.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the serverless execution model and its event-driven architecture
2. Compare AWS Lambda and GCP Cloud Functions in terms of runtimes, limits, and triggers
3. Identify the causes and mitigation strategies for cold starts
4. Deploy a serverless function using the console and CLI on both platforms
5. Configure event sources (HTTP, storage, message queues) to trigger functions
6. Design serverless applications that handle concurrency and scaling automatically
7. Calculate serverless costs based on invocation count, duration, and memory

---

Serverless computing removes the need to provision and manage servers entirely. You write the function, define what triggers it, and the cloud provider handles everything else -- scaling, patching, and capacity planning. This model is ideal for event-driven workloads and can dramatically reduce operational overhead and cost for bursty, unpredictable traffic patterns.

> **Analogy — The Light Switch**: Traditional servers are like leaving the lights on 24/7 in every room. Even when nobody's home, you're paying the electricity bill. Serverless is like motion-activated lights — power flows only when someone walks in, and shuts off automatically when they leave. You don't manage the wiring or the bulbs; you just define *what should happen when the light turns on*.

## 1. Serverless Overview

### 1.1 What is Serverless?

Serverless is a computing model that executes code without server management.

**Characteristics:**
- No server provisioning/management required
- Automatic scaling
- Pay only for what you use (execution time + requests)
- Event-driven execution

### 1.2 Service Comparison

| Item | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| Runtime | Node.js, Python, Java, Go, Ruby, .NET, Custom | Node.js, Python, Go, Java, Ruby, PHP, .NET |
| Memory | 128MB ~ 10GB | 128MB ~ 32GB |
| Max Execution Time | 15 minutes | 9 minutes (1st gen) / 60 minutes (2nd gen) |
| Concurrent Executions | 1000 (default, increasable) | Unlimited (default) |
| Triggers | API Gateway, S3, DynamoDB, SNS, etc. | HTTP, Pub/Sub, Cloud Storage, etc. |
| Container Support | Supported (Container Image) | 2nd gen only |

---

## 2. Cold Start

### 2.1 What is Cold Start?

Latency that occurs when a function is first invoked or wakes from idle state.

```
Request → [Cold Start] → Create Container → Initialize Runtime → Load Code → Execute Handler
          ~100ms-seconds                                                      ~ms-seconds

Request → [Warm Start] → Execute Handler
          ~ms
```

### 2.2 Cold Start Mitigation Strategies

| Strategy | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| **Provisioned Concurrency** | Supported (paid) | - |
| **Minimum Instances** | - | min-instances in 2nd gen |
| **Lightweight Runtime** | Python, Node.js recommended | Python, Node.js recommended |
| **Minimize Packages** | Remove unnecessary dependencies | Remove unnecessary dependencies |
| **Continuous Invocation** | Warm-up with CloudWatch Events | Warm-up with Cloud Scheduler |

---

## 3. AWS Lambda

### 3.1 Function Creation (Console)

1. Lambda console → "Create function"
2. Select "Author from scratch"
3. Enter function name
4. Select runtime (e.g., Python 3.12)
5. Select architecture (x86_64 or arm64)
6. "Create function"

### 3.2 Function Code (Python)

```python
import json

def lambda_handler(event, context):
    """
    event: Data passed from trigger
    context: Runtime information (function name, memory, remaining time, etc.)
    """
    # Log event
    print(f"Event: {json.dumps(event)}")

    # Business logic
    name = event.get('name', 'World')
    message = f"Hello, {name}!"

    # Return response
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'message': message
        })
    }
```

### 3.3 Function Creation (AWS CLI)

```bash
# 1. Package code
zip function.zip lambda_function.py

# 2. Create IAM role (Lambda execution role)
aws iam create-role \
    --role-name lambda-execution-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# 3. Attach basic policy
aws iam attach-role-policy \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 4. Create Lambda function
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler lambda_function.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --zip-file fileb://function.zip

# 5. Test function
aws lambda invoke \
    --function-name my-function \
    --payload '{"name": "Claude"}' \
    --cli-binary-format raw-in-base64-out \
    output.json

cat output.json
```

### 3.4 API Gateway Integration

```bash
# 1. Create REST API
aws apigateway create-rest-api \
    --name my-api \
    --endpoint-configuration types=REGIONAL

# 2. Integrate with Lambda (easier through Console)
# Lambda console → Function → Add trigger → Select API Gateway
```

---

## 4. GCP Cloud Functions

### 4.1 Function Creation (Console)

1. Cloud Functions console → "Create Function"
2. Select environment (1st gen or 2nd gen)
3. Enter function name
4. Select region
5. Select trigger type (HTTP, Pub/Sub, etc.)
6. Select runtime (e.g., Python 3.12)
7. Write code and "Deploy"

### 4.2 Function Code (Python)

**HTTP Trigger:**
```python
import functions_framework
from flask import jsonify

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request: Flask request object
    Returns:
        Response object
    """
    request_json = request.get_json(silent=True)
    name = 'World'

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request.args and 'name' in request.args:
        name = request.args.get('name')

    return jsonify({
        'message': f'Hello, {name}!'
    })
```

**Pub/Sub Trigger:**
```python
import base64
import functions_framework

@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    """Pub/Sub Cloud Function.
    Args:
        cloud_event: CloudEvent object
    """
    data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    print(f"Received message: {data}")
```

### 4.3 Function Deployment (gcloud CLI)

```bash
# 1. Project structure
# my-function/
# ├── main.py
# └── requirements.txt

# requirements.txt
# functions-framework==3.*

# 2. Deploy HTTP function
gcloud functions deploy hello-http \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-http \
    --allow-unauthenticated \
    --entry-point=hello_http \
    --source=.

# 3. Check function URL
gcloud functions describe hello-http \
    --region=asia-northeast3 \
    --format='value(url)'

# 4. Test function
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name": "Claude"}' \
    https://asia-northeast3-PROJECT_ID.cloudfunctions.net/hello-http
```

### 4.4 Cloud Storage Trigger

```python
import functions_framework

@functions_framework.cloud_event
def hello_gcs(cloud_event):
    """Cloud Storage trigger function.
    Args:
        cloud_event: CloudEvent object
    """
    data = cloud_event.data

    bucket = data["bucket"]
    name = data["name"]

    print(f"File uploaded: gs://{bucket}/{name}")
```

```bash
# Deploy Cloud Storage trigger
gcloud functions deploy process-upload \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=my-bucket" \
    --entry-point=hello_gcs \
    --source=.
```

---

## 5. Trigger Type Comparison

### 5.1 AWS Lambda Triggers

| Trigger | Description | Examples |
|--------|------|------|
| **API Gateway** | HTTP requests | REST API, WebSocket |
| **S3** | Object events | Upload, Delete |
| **DynamoDB Streams** | Table changes | Insert, Modify, Remove |
| **SNS** | Notification messages | Push notifications |
| **SQS** | Queue messages | Async processing |
| **CloudWatch Events** | Schedule, Events | Cron jobs |
| **Kinesis** | Stream data | Real-time analytics |
| **Cognito** | Auth events | Post-registration processing |

### 5.2 GCP Cloud Functions Triggers

| Trigger | Description | Examples |
|--------|------|------|
| **HTTP** | HTTP requests | REST API |
| **Cloud Storage** | Object events | Upload, Delete |
| **Pub/Sub** | Messages | Async processing |
| **Firestore** | Document changes | Insert, Update, Delete |
| **Cloud Scheduler** | Schedule | Cron jobs |
| **Eventarc** | Various GCP events | 2nd gen unified trigger |

---

## 6. Environment Variables and Secret Management

### 6.1 AWS Lambda Environment Variables

```bash
# Set environment variables
aws lambda update-function-configuration \
    --function-name my-function \
    --environment "Variables={DB_HOST=mydb.example.com,DB_PORT=5432}"
```

**Use in code:**
```python
import os

def lambda_handler(event, context):
    db_host = os.environ.get('DB_HOST')
    db_port = os.environ.get('DB_PORT')
    # ...
```

**Secrets Manager Integration:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

def lambda_handler(event, context):
    secrets = get_secret('my-database-credentials')
    db_password = secrets['password']
    # ...
```

### 6.2 GCP Cloud Functions Environment Variables

```bash
# Set environment variables
gcloud functions deploy my-function \
    --set-env-vars DB_HOST=mydb.example.com,DB_PORT=5432 \
    ...
```

**Secret Manager Integration:**
```bash
# Reference secret
gcloud functions deploy my-function \
    --set-secrets 'DB_PASSWORD=projects/PROJECT_ID/secrets/db-password:latest' \
    ...
```

**Use in code:**
```python
import os

def hello_http(request):
    db_host = os.environ.get('DB_HOST')
    db_password = os.environ.get('DB_PASSWORD')  # Auto-injected from Secret Manager
    # ...
```

---

## 7. Pricing Comparison

### 7.1 AWS Lambda Pricing

```
Monthly cost = Request cost + Execution time cost

Request cost: $0.20 / 1M requests
Execution time: $0.0000166667 / GB-second (x86)
               $0.0000133334 / GB-second (ARM)

Free tier (always free):
- 1M requests/month
- 400K GB-seconds/month
```

**Example Calculation:**
```
Conditions: 512MB memory, 200ms execution, 1M requests/month

Request cost: (1,000,000 - 1,000,000) × $0.20/1M = $0
Execution time:
  - 0.5GB × 0.2s × 1,000,000 = 100,000 GB-seconds
  - Free: 400,000 GB-seconds
  - Cost: $0 (within free tier)

Total cost: $0/month (using free tier)
```

### 7.2 GCP Cloud Functions Pricing

```
Monthly cost = Invocation cost + Compute time + Network cost

Invocation cost: $0.40 / 1M invocations
Compute time:
  - CPU: $0.0000100 / GHz-second
  - Memory: $0.0000025 / GB-second

Free tier (always free):
- 2M invocations/month
- 400K GB-seconds, 200K GHz-seconds
- 5GB network egress
```

### 7.3 Cost Optimization Tips

1. **Appropriate Memory Allocation**: Test memory ↔ performance trade-off
2. **Use ARM Architecture** (AWS): 20% cheaper
3. **Minimize Provisioned Concurrency**: Only what's needed
4. **Use Async Invocation**: Direct invocation cheaper than API Gateway
5. **Code Optimization**: Reduce execution time

---

## 8. Local Development and Testing

### 8.1 AWS SAM (Serverless Application Model)

```bash
# Install SAM CLI
pip install aws-sam-cli

# Initialize project
sam init

# Local testing
sam local invoke MyFunction --event events/event.json

# Run local API
sam local start-api

# Deploy
sam build
sam deploy --guided
```

**template.yaml example:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  HelloFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.lambda_handler
      Runtime: python3.12
      Events:
        Api:
          Type: Api
          Properties:
            Path: /hello
            Method: get
```

### 8.2 GCP Functions Framework

```bash
# Install Functions Framework
pip install functions-framework

# Run locally
functions-framework --target=hello_http --debug

# Test from another terminal
curl http://localhost:8080
```

---

## 9. Practice: Image Resize Function

### 9.1 AWS Lambda (S3 Trigger)

```python
import boto3
from PIL import Image
import io

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Extract bucket and key from S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download original image
    response = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(io.BytesIO(response['Body'].read()))

    # Resize
    image.thumbnail((200, 200))

    # Upload thumbnail
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_key = f"thumbnails/{key}"
    s3.put_object(Bucket=bucket, Key=thumb_key, Body=buffer)

    return {'statusCode': 200, 'body': f'Thumbnail created: {thumb_key}'}
```

### 9.2 GCP Cloud Functions (Cloud Storage Trigger)

```python
from google.cloud import storage
from PIL import Image
import io
import functions_framework

@functions_framework.cloud_event
def resize_image(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]

    # Ignore thumbnails folder images
    if file_name.startswith("thumbnails/"):
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download original image
    blob = bucket.blob(file_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))

    # Resize
    image.thumbnail((200, 200))

    # Upload thumbnail
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_blob = bucket.blob(f"thumbnails/{file_name}")
    thumb_blob.upload_from_file(buffer, content_type='image/jpeg')

    print(f"Thumbnail created: thumbnails/{file_name}")
```

---

## 10. Next Steps

- [06_Container_Services.md](./06_Container_Services.md) - Container services
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - Database integration

---

## Exercises

### Exercise 1: Serverless vs VM Trade-off Analysis

For each of the following use cases, determine whether serverless (Lambda/Cloud Functions) or a traditional VM (EC2/Compute Engine) is the better fit. Justify your answer.

1. A REST API endpoint that receives ~500 requests per day, each taking < 200ms.
2. A video transcoding service that processes 4K video files for up to 30 minutes per job.
3. An event-driven pipeline that processes messages from a queue whenever new orders arrive.
4. A long-running machine learning training job that runs for 6 hours continuously.

<details>
<summary>Show Answer</summary>

1. **Serverless** — 500 requests/day is extremely low. A VM would sit idle for 99.9% of the time. Lambda's free tier covers 1M requests/month, so this workload would cost essentially $0. The < 200ms execution is well within serverless limits.

2. **VM** — AWS Lambda has a maximum execution time of 15 minutes; GCP Cloud Functions (1st gen) has 9 minutes. A 30-minute transcoding job cannot run on either platform. Use an EC2 instance or a container-based solution (ECS/GKE) instead.

3. **Serverless** — Event-driven queue processing is a textbook serverless use case. Lambda and Cloud Functions have native integrations with SQS/Pub-Sub. Serverless scales automatically with message volume and scales to zero when the queue is empty.

4. **VM** — A 6-hour continuous run exceeds all serverless time limits. Long-running ML training requires persistent state and dedicated resources; use an EC2 GPU instance (p3/g4dn family) or managed ML platforms (SageMaker, Vertex AI).

</details>

### Exercise 2: Cold Start Root Cause and Mitigation

A company's Lambda function handles user authentication. During peak hours it performs well (~50ms response), but first-morning requests regularly take 3–4 seconds, causing user complaints.

1. What is causing the 3–4 second latency? Explain the mechanism.
2. Propose two concrete mitigation strategies with their trade-offs.

<details>
<summary>Show Answer</summary>

1. **Cold start**. When no Lambda instance has executed the function recently (overnight inactivity), AWS recycles the container. The first invocation must: spin up a new container, initialize the language runtime (e.g., Python interpreter), import all dependencies, and run any global initialization code. For a Python function with large authentication libraries (e.g., `boto3`, JWT libraries), this initialization can take 2–4 seconds. Subsequent "warm" invocations reuse the initialized container and take only 50ms.

2. **Mitigation strategies**:

   **Strategy A — Provisioned Concurrency**
   - Configure Lambda to keep N pre-initialized containers always warm.
   - Trade-off: Eliminates cold starts entirely for up to N concurrent invocations, but you pay for the provisioned concurrency hours even when no requests come in. Cost increases proportionally with N.
   ```bash
   aws lambda put-provisioned-concurrency-config \
       --function-name my-auth-function \
       --qualifier prod \
       --provisioned-concurrent-executions 5
   ```

   **Strategy B — Scheduled Warm-up with CloudWatch Events**
   - Create a CloudWatch Events rule that pings the function every 5 minutes to keep containers warm.
   - Trade-off: Low cost (stays within free tier for a few warm-up pings), but only keeps a small number of containers warm. If the first morning spike requires 50 concurrent executions, most will still cold start. Works well for single-instance scenarios.

</details>

### Exercise 3: Lambda Cost Estimation

Estimate the monthly AWS Lambda cost for a function with the following characteristics:
- Memory: 512 MB
- Average execution duration: 400ms
- Invocations: 5 million per month
- Architecture: x86

Use these prices: $0.20 per 1M requests, $0.0000166667 per GB-second. Free tier: 1M requests and 400,000 GB-seconds per month.

<details>
<summary>Show Answer</summary>

**Step 1: Request cost**
- Total invocations: 5,000,000
- Free tier: 1,000,000
- Billable invocations: 4,000,000
- Cost: 4,000,000 / 1,000,000 × $0.20 = **$0.80**

**Step 2: Compute cost (GB-seconds)**
- GB-seconds per invocation: 0.5 GB × 0.4 seconds = 0.2 GB-seconds
- Total GB-seconds: 5,000,000 × 0.2 = 1,000,000 GB-seconds
- Free tier: 400,000 GB-seconds
- Billable GB-seconds: 600,000 GB-seconds
- Cost: 600,000 × $0.0000166667 = **$10.00**

**Total monthly cost: $0.80 + $10.00 = $10.80**

**Optimization insight**: Switching to ARM (Graviton2) architecture reduces compute cost by 20%, bringing the compute cost to ~$8.00 and total to ~$8.80/month — a saving of ~$2/month or ~$24/year for this function alone.

</details>

### Exercise 4: Event Source Configuration

You are building a system where: (1) users upload images to an S3 bucket, and (2) a Lambda function automatically creates a thumbnail for each uploaded image.

Describe the event source mapping you need to configure and identify any important configuration detail to prevent an infinite loop.

<details>
<summary>Show Answer</summary>

**Event source configuration**:
- Configure an **S3 Event Notification** on the source bucket to trigger Lambda on `s3:ObjectCreated:*` events.
- This means every time a new object is uploaded to the bucket, Lambda is invoked with an event containing the bucket name and object key.

**Infinite loop problem**: If the Lambda function writes the generated thumbnail back to the same S3 bucket (not a subfolder or different bucket), the thumbnail upload itself triggers another Lambda invocation, which creates another thumbnail, which triggers another invocation — causing an infinite recursive loop and unexpected charges.

**Solutions**:
1. **Separate output bucket**: Write thumbnails to a different S3 bucket (e.g., `my-bucket-thumbnails`). Only configure the trigger on the source bucket.
2. **Prefix/suffix filtering**: Configure the S3 trigger to only fire for objects under a specific prefix (e.g., `uploads/`) and NOT for the thumbnails prefix. Write thumbnails to `thumbnails/`. In the Lambda code, add a guard:
   ```python
   if key.startswith("thumbnails/"):
       return  # Skip thumbnail files to prevent infinite loop
   ```
   Both code-level and S3-notification-level filtering are recommended as defense in depth.

</details>

### Exercise 5: Serverless Architecture Design

Design a serverless architecture for an e-commerce order processing system. When a customer places an order:
1. The order is saved to a database.
2. An inventory check is performed.
3. A confirmation email is sent.
4. Analytics data is recorded.

Identify which AWS services would be used, how they connect, and why serverless is appropriate for this workflow.

<details>
<summary>Show Answer</summary>

**Architecture**:

```
Customer → API Gateway → Lambda (Order Handler)
                               │
                               ├── DynamoDB (save order)
                               │
                               └── SNS Topic "order-created"
                                       │
                         ┌─────────────┼──────────────┐
                         ▼             ▼              ▼
                    Lambda          Lambda         Lambda
                 (Inventory)       (Email)      (Analytics)
                      │               │              │
                  DynamoDB         SES/SNS       Kinesis/S3
               (inventory DB)   (send email)   (data lake)
```

**Service roles**:
- **API Gateway**: HTTP endpoint that receives order requests and triggers the Order Handler Lambda.
- **Lambda (Order Handler)**: Saves the order to DynamoDB, then publishes an `order-created` event to SNS. Returns a response to the customer.
- **SNS Topic**: Fan-out hub that delivers the event to multiple downstream Lambdas simultaneously (parallel processing).
- **Lambda (Inventory)**: Checks and decrements inventory in DynamoDB.
- **Lambda (Email)**: Sends a confirmation email via SES.
- **Lambda (Analytics)**: Records event data to Kinesis Data Firehose → S3 for later analysis.

**Why serverless is appropriate**:
- Each step is event-driven and short-lived (< 15 minutes).
- Traffic is bursty (unpredictable order volume); serverless scales to zero and back automatically.
- Loose coupling via SNS means each function can fail independently without affecting others.
- Cost is proportional to actual order volume — no idle capacity costs.

</details>

---

## References

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [GCP Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [AWS SAM](https://aws.amazon.com/serverless/sam/)
- [Functions Framework](https://github.com/GoogleCloudPlatform/functions-framework)
