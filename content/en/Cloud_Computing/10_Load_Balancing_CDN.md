# Load Balancing & CDN

**Previous**: [Virtual Private Cloud](./09_Virtual_Private_Cloud.md) | **Next**: [Managed Relational Databases](./11_Managed_Relational_DB.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of load balancers in achieving high availability and scalability
2. Distinguish between Layer 4 (TCP/UDP) and Layer 7 (HTTP/HTTPS) load balancers
3. Compare AWS ELB types (ALB, NLB, CLB) with GCP load balancing options
4. Configure health checks and target groups for automatic failover
5. Describe how CDNs (CloudFront, Cloud CDN) cache and deliver content at the edge
6. Design a multi-tier architecture with load balancers distributing traffic across zones

---

As applications grow beyond a single server, distributing traffic across multiple instances becomes essential for reliability and performance. Load balancers prevent any single server from becoming a bottleneck, and CDNs push content closer to end users to reduce latency. Together, these services form the delivery layer that makes cloud applications fast and resilient at scale.

## 1. Load Balancing Overview

### 1.1 What is a Load Balancer?

A load balancer is a service that distributes incoming traffic across multiple servers.

**Benefits:**
- High availability (automatic exclusion of failed servers)
- Scalability (easy to add/remove servers)
- Performance improvement (load distribution)
- Security (DDoS mitigation, SSL offloading)

### 1.2 Service Comparison

| Category | AWS | GCP |
|------|-----|-----|
| L7 (HTTP/HTTPS) | ALB | HTTP(S) Load Balancing |
| L4 (TCP/UDP) | NLB | TCP/UDP Load Balancing |
| Classic | CLB (legacy) | - |
| Internal | Internal ALB/NLB | Internal Load Balancing |
| Global | Global Accelerator | Global Load Balancing |

---

## 2. AWS Elastic Load Balancing

### 2.1 Load Balancer Types

| Type | Layer | Use Case | Features |
|------|------|----------|------|
| **ALB** | L7 | Web apps, microservices | Path/host routing, WebSocket |
| **NLB** | L4 | High performance, static IP needed | Millions RPS, ultra-low latency |
| **GWLB** | L3 | Firewall, IDS/IPS | Transparent gateway |

### 2.2 ALB (Application Load Balancer)

```bash
# 1. Create target group
aws elbv2 create-target-group \
    --name my-targets \
    --protocol HTTP \
    --port 80 \
    --vpc-id vpc-12345678 \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --target-type instance

# 2. Register instances
aws elbv2 register-targets \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --targets Id=i-12345678 Id=i-87654321

# 3. Create ALB
aws elbv2 create-load-balancer \
    --name my-alb \
    --subnets subnet-1 subnet-2 \
    --security-groups sg-12345678 \
    --scheme internet-facing \
    --type application

# 4. Create listener
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx
```

**Path-Based Routing:**
```bash
# Add rule (/api/* → API target group)
aws elbv2 create-rule \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --priority 10 \
    --conditions Field=path-pattern,Values='/api/*' \
    --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/api-targets/xxx
```

### 2.3 NLB (Network Load Balancer)

```bash
# Create NLB (static IP)
aws elbv2 create-load-balancer \
    --name my-nlb \
    --subnets subnet-1 subnet-2 \
    --type network \
    --scheme internet-facing

# TCP listener
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/net/my-nlb/xxx \
    --protocol TCP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/tcp-targets/xxx
```

### 2.4 SSL/TLS Configuration

```bash
# Request ACM certificate
aws acm request-certificate \
    --domain-name example.com \
    --subject-alternative-names "*.example.com" \
    --validation-method DNS

# Add HTTPS listener
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=arn:aws:acm:...:certificate/xxx \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx

# HTTP → HTTPS redirect
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --default-actions Type=redirect,RedirectConfig='{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}'
```

---

## 3. GCP Cloud Load Balancing

### 3.1 Load Balancer Types

| Type | Scope | Layer | Use Case |
|------|------|------|----------|
| **Global HTTP(S)** | Global | L7 | Web apps, CDN integration |
| **Regional HTTP(S)** | Regional | L7 | Single-region apps |
| **Global TCP/SSL** | Global | L4 | TCP proxy |
| **Regional TCP/UDP** | Regional | L4 | Network LB |
| **Internal HTTP(S)** | Regional | L7 | Internal microservices |
| **Internal TCP/UDP** | Regional | L4 | Internal TCP/UDP |

### 3.2 HTTP(S) Load Balancer

```bash
# 1. Create instance group (unmanaged)
gcloud compute instance-groups unmanaged create my-group \
    --zone=asia-northeast3-a

gcloud compute instance-groups unmanaged add-instances my-group \
    --zone=asia-northeast3-a \
    --instances=instance-1,instance-2

# 2. Create health check
gcloud compute health-checks create http my-health-check \
    --port=80 \
    --request-path=/health

# 3. Create backend service
gcloud compute backend-services create my-backend \
    --protocol=HTTP \
    --health-checks=my-health-check \
    --global

# 4. Add instance group to backend
gcloud compute backend-services add-backend my-backend \
    --instance-group=my-group \
    --instance-group-zone=asia-northeast3-a \
    --global

# 5. Create URL map
gcloud compute url-maps create my-url-map \
    --default-service=my-backend

# 6. Create target HTTP proxy
gcloud compute target-http-proxies create my-proxy \
    --url-map=my-url-map

# 7. Create global forwarding rule
gcloud compute forwarding-rules create my-lb \
    --global \
    --target-http-proxy=my-proxy \
    --ports=80
```

### 3.3 SSL/TLS Configuration

```bash
# 1. Managed SSL certificate
gcloud compute ssl-certificates create my-cert \
    --domains=example.com,www.example.com \
    --global

# 2. HTTPS target proxy
gcloud compute target-https-proxies create my-https-proxy \
    --url-map=my-url-map \
    --ssl-certificates=my-cert

# 3. HTTPS forwarding rule
gcloud compute forwarding-rules create my-https-lb \
    --global \
    --target-https-proxy=my-https-proxy \
    --ports=443

# 4. HTTP → HTTPS redirect
gcloud compute url-maps import my-url-map --source=- <<EOF
name: my-url-map
defaultUrlRedirect:
  httpsRedirect: true
  redirectResponseCode: MOVED_PERMANENTLY_DEFAULT
EOF
```

### 3.4 Path-Based Routing

```bash
# Add path rules to URL map
gcloud compute url-maps add-path-matcher my-url-map \
    --path-matcher-name=api-matcher \
    --default-service=default-backend \
    --path-rules="/api/*=api-backend,/static/*=static-backend"
```

---

## 4. Health Checks

### 4.1 AWS Health Checks

```bash
# Configure target group health check
aws elbv2 modify-target-group \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --health-check-protocol HTTP \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3

# Check target health status
aws elbv2 describe-target-health \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx
```

### 4.2 GCP Health Checks

```bash
# HTTP health check
gcloud compute health-checks create http my-http-check \
    --port=80 \
    --request-path=/health \
    --check-interval=30s \
    --timeout=5s \
    --healthy-threshold=2 \
    --unhealthy-threshold=3

# TCP health check
gcloud compute health-checks create tcp my-tcp-check \
    --port=3306

# Check health check status
gcloud compute backend-services get-health my-backend --global
```

---

## 5. Auto Scaling Integration

### 5.1 AWS Auto Scaling Group + ALB

```bash
# Create launch template
aws ec2 create-launch-template \
    --launch-template-name my-template \
    --launch-template-data '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.micro",
        "SecurityGroupIds": ["sg-12345678"]
    }'

# Create Auto Scaling Group (attach to target group)
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name my-asg \
    --launch-template LaunchTemplateName=my-template,Version='$Latest' \
    --min-size 2 \
    --max-size 10 \
    --desired-capacity 2 \
    --vpc-zone-identifier "subnet-1,subnet-2" \
    --target-group-arns "arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx"

# Scaling policy
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name my-asg \
    --policy-name cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ASGAverageCPUUtilization"
        }
    }'
```

### 5.2 GCP Managed Instance Group + LB

```bash
# Create instance template
gcloud compute instance-templates create my-template \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server

# Create managed instance group
gcloud compute instance-groups managed create my-mig \
    --template=my-template \
    --size=2 \
    --zone=asia-northeast3-a

# Configure autoscaling
gcloud compute instance-groups managed set-autoscaling my-mig \
    --zone=asia-northeast3-a \
    --min-num-replicas=2 \
    --max-num-replicas=10 \
    --target-cpu-utilization=0.7

# Attach to load balancer
gcloud compute backend-services add-backend my-backend \
    --instance-group=my-mig \
    --instance-group-zone=asia-northeast3-a \
    --global
```

---

## 6. CDN (Content Delivery Network)

### 6.1 AWS CloudFront

```bash
# Create CloudFront distribution (S3 origin)
aws cloudfront create-distribution \
    --distribution-config '{
        "CallerReference": "my-distribution-2024",
        "Origins": {
            "Quantity": 1,
            "Items": [{
                "Id": "S3-my-bucket",
                "DomainName": "my-bucket.s3.amazonaws.com",
                "S3OriginConfig": {
                    "OriginAccessIdentity": ""
                }
            }]
        },
        "DefaultCacheBehavior": {
            "TargetOriginId": "S3-my-bucket",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {
                "Quantity": 2,
                "Items": ["GET", "HEAD"]
            },
            "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
            "Compress": true
        },
        "Enabled": true,
        "DefaultRootObject": "index.html"
    }'

# Invalidate cache
aws cloudfront create-invalidation \
    --distribution-id EDFDVBD632BHDS5 \
    --paths "/*"
```

**CloudFront + ALB:**
```bash
# CloudFront with ALB as origin
{
    "Origins": {
        "Items": [{
            "Id": "ALB-origin",
            "DomainName": "my-alb-12345.ap-northeast-2.elb.amazonaws.com",
            "CustomOriginConfig": {
                "HTTPPort": 80,
                "HTTPSPort": 443,
                "OriginProtocolPolicy": "https-only"
            }
        }]
    }
}
```

### 6.2 GCP Cloud CDN

```bash
# 1. Enable CDN on backend service
gcloud compute backend-services update my-backend \
    --enable-cdn \
    --global

# 2. Use Cloud Storage bucket as CDN origin
gcloud compute backend-buckets create my-cdn-bucket \
    --gcs-bucket-name=my-static-bucket \
    --enable-cdn

# 3. Add bucket to URL map
gcloud compute url-maps add-path-matcher my-url-map \
    --path-matcher-name=static-matcher \
    --default-backend-bucket=my-cdn-bucket \
    --path-rules="/static/*=my-cdn-bucket"

# 4. Invalidate cache
gcloud compute url-maps invalidate-cdn-cache my-url-map \
    --path="/*"
```

### 6.3 CDN Cache Policy

**AWS CloudFront Cache Policy:**
```bash
# Create cache policy
aws cloudfront create-cache-policy \
    --cache-policy-config '{
        "Name": "MyPolicy",
        "DefaultTTL": 86400,
        "MaxTTL": 31536000,
        "MinTTL": 0,
        "ParametersInCacheKeyAndForwardedToOrigin": {
            "EnableAcceptEncodingGzip": true,
            "HeadersConfig": {"HeaderBehavior": "none"},
            "CookiesConfig": {"CookieBehavior": "none"},
            "QueryStringsConfig": {"QueryStringBehavior": "none"}
        }
    }'
```

**GCP Cloud CDN Cache Mode:**
```bash
# Set cache mode
gcloud compute backend-services update my-backend \
    --cache-mode=CACHE_ALL_STATIC \
    --default-ttl=3600 \
    --max-ttl=86400 \
    --global
```

---

## 7. Cost Comparison

### 7.1 Load Balancer Cost

| Service | Fixed Cost | Processing Cost |
|--------|----------|----------|
| AWS ALB | ~$18/month | $0.008/LCU-hour |
| AWS NLB | ~$18/month | $0.006/NLCU-hour |
| GCP HTTP(S) LB | ~$18/month | $0.008/GB throughput |
| GCP TCP/UDP LB | $18/month per region | Additional per rule |

### 7.2 CDN Cost

| Service | Data Transfer (first 10TB) |
|--------|---------------------|
| AWS CloudFront | ~$0.085/GB (US/Europe) |
| GCP Cloud CDN | ~$0.08/GB (US/Europe) |

---

## 8. Monitoring

### 8.1 AWS CloudWatch Metrics

```bash
# Query ALB metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/ApplicationELB \
    --metric-name RequestCount \
    --dimensions Name=LoadBalancer,Value=app/my-alb/xxx \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-01T23:59:59Z \
    --period 300 \
    --statistics Sum

# Key metrics:
# - RequestCount
# - HTTPCode_Target_2XX_Count
# - TargetResponseTime
# - HealthyHostCount
# - UnHealthyHostCount
```

### 8.2 GCP Cloud Monitoring

```bash
# Query metrics
gcloud monitoring metrics list \
    --filter="metric.type:loadbalancing"

# Create alert policy
gcloud alpha monitoring policies create \
    --display-name="High Latency Alert" \
    --condition-display-name="Latency > 1s" \
    --condition-filter='metric.type="loadbalancing.googleapis.com/https/backend_latencies"' \
    --condition-threshold-value=1000 \
    --notification-channels=projects/PROJECT/notificationChannels/xxx
```

---

## 9. Next Steps

- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - Databases
- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - Monitoring Details

---

## Exercises

### Exercise 1: ALB vs NLB Selection

For each scenario, choose between ALB (Application Load Balancer) and NLB (Network Load Balancer), and explain your reasoning:

1. A REST API service with multiple microservice routes: `/api/users/*` goes to the user service, `/api/orders/*` goes to the order service.
2. A real-time gaming server that uses UDP for fast packet delivery and requires a static IP address for firewall whitelisting.
3. An HTTPS web application that needs SSL certificate management and sticky sessions for shopping cart state.
4. A financial trading platform that requires sub-millisecond latency and handles millions of TCP connections per second.

<details>
<summary>Show Answer</summary>

1. **ALB** — Path-based routing is an ALB-exclusive feature. ALB inspects HTTP request paths and routes to different target groups based on URL patterns (`/api/users/*` vs `/api/orders/*`). NLB operates at Layer 4 and cannot inspect HTTP paths.

2. **NLB** — NLB supports UDP (ALB is HTTP/HTTPS only) and provides static IP addresses per AZ, which is essential for firewall whitelisting. NLB passes the client source IP directly to the target, which is needed for many game servers.

3. **ALB** — ALB supports SSL termination (offloading HTTPS from your servers), ACM certificate management, and sticky sessions via cookie-based affinity. NLB can also do TLS passthrough, but ALB's L7 features are a better fit for web applications.

4. **NLB** — The NLB is purpose-built for extreme performance: millions of requests per second with ultra-low latency (~100 μs vs ~1 ms for ALB). The NLB operates at Layer 4 with minimal processing overhead, making it ideal for latency-sensitive financial applications.

</details>

### Exercise 2: Health Check Configuration

You have an ALB with a target group of EC2 instances running a web application. The application has a `/health` endpoint that returns HTTP 200 when healthy and HTTP 503 when unavailable.

Describe the health check configuration you would set, and explain what happens when an instance fails the health check.

<details>
<summary>Show Answer</summary>

**Recommended health check configuration**:

```bash
aws elbv2 modify-target-group \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --health-check-protocol HTTP \
    --health-check-port 80 \
    --health-check-path /health \
    --health-check-interval-seconds 15 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --matcher HttpCode=200
```

**Configuration explained**:
- `--health-check-path /health` — Use the dedicated health endpoint, not the root path (which may have redirect logic).
- `--health-check-interval-seconds 15` — Check every 15 seconds (balance between responsiveness and traffic overhead).
- `--healthy-threshold-count 2` — Instance must pass 2 consecutive checks to be marked healthy (prevents flapping).
- `--unhealthy-threshold-count 3` — Instance must fail 3 consecutive checks before being removed (prevents premature removal on transient errors).
- `--matcher HttpCode=200` — Only HTTP 200 is acceptable; 503 signals the unhealthy state.

**What happens when an instance fails**:
1. After 3 consecutive failed checks (3 × 15s = 45 seconds), the instance is marked `unhealthy`.
2. The ALB immediately stops sending new requests to the unhealthy instance.
3. Existing in-flight connections are drained (connection draining period, default 300 seconds).
4. If integrated with an Auto Scaling Group, ASG detects the unhealthy instance and terminates it, launching a replacement.
5. Once the replacement instance passes 2 consecutive health checks, it starts receiving traffic.

</details>

### Exercise 3: CDN Cache Configuration

A media streaming company wants to serve video thumbnails globally with low latency. They have an S3 bucket (`media-bucket`) in `ap-northeast-2` containing thumbnail images.

1. Which AWS service would you use to distribute these images globally?
2. What TTL (Time-to-Live) would you set for the cache, and why?
3. How would you invalidate the cache when a thumbnail is updated?

<details>
<summary>Show Answer</summary>

1. **Amazon CloudFront** — Configure an S3 bucket as the CloudFront origin. CloudFront has 400+ edge locations worldwide. Users in North America, Europe, and Asia all receive thumbnails from the nearest edge location rather than downloading from Seoul every time.

2. **TTL recommendation: Long TTL (86400 seconds = 24 hours or longer)**

   Thumbnails are typically static assets that rarely change. A long TTL means:
   - Edge locations serve from cache for 24 hours without contacting the origin.
   - Dramatically reduced load on S3 (fewer origin requests = lower cost).
   - Faster cache hits for global users.

   For thumbnail images that change (e.g., when a user updates their profile picture), use versioned filenames (e.g., `user-123-v2.jpg`) so old thumbnails can stay cached indefinitely while new ones are fetched by new URLs.

3. **Cache invalidation**:
```bash
# Invalidate a specific thumbnail
aws cloudfront create-invalidation \
    --distribution-id DISTRIBUTION_ID \
    --paths "/thumbnails/user-123.jpg"

# Invalidate all thumbnails
aws cloudfront create-invalidation \
    --distribution-id DISTRIBUTION_ID \
    --paths "/thumbnails/*"
```

**Cost consideration**: CloudFront charges $0.005 per 1,000 invalidation paths (after first 1,000/month free). For frequently updated assets, prefer URL versioning over invalidations to avoid these costs.

</details>

### Exercise 4: Path-Based Routing Rule

You have an ALB serving a monolith being migrated to microservices. Currently, all traffic goes to the `legacy-app` target group. You need to route `/api/v2/*` requests to a new `microservice-api` target group while routing everything else to `legacy-app`.

Write the ALB listener rule configuration (describe it or provide the CLI commands).

<details>
<summary>Show Answer</summary>

```bash
# Add a rule for /api/v2/* → microservice-api target group (priority 10)
# All other traffic falls through to the default rule (legacy-app)
aws elbv2 create-rule \
    --listener-arn arn:aws:elasticloadbalancing:ap-northeast-2:123456789012:listener/app/my-alb/xxx/yyy \
    --priority 10 \
    --conditions '[{"Field":"path-pattern","Values":["/api/v2/*"]}]' \
    --actions '[{"Type":"forward","TargetGroupArn":"arn:aws:elasticloadbalancing:ap-northeast-2:123456789012:targetgroup/microservice-api/zzz"}]'
```

**How ALB rules work**:
1. Rules are evaluated in priority order (lowest number = highest priority).
2. Rule priority 10: if path matches `/api/v2/*` → forward to `microservice-api`.
3. Default rule (priority `default`, always last): all other requests → forward to `legacy-app`.

**Verification**:
```bash
# List all rules for the listener to verify
aws elbv2 describe-rules \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx
```

This pattern (strangler fig architecture) allows incremental migration of a monolith to microservices using the load balancer as a traffic router.

</details>

### Exercise 5: Load Balancer Troubleshooting

A developer reports that their EC2 instance is running and the application is responding correctly when accessed directly via the instance's public IP, but the ALB is returning `502 Bad Gateway` for all requests.

List 4 possible causes and how to diagnose/fix each one.

<details>
<summary>Show Answer</summary>

1. **Instance not registered in the target group or marked unhealthy**
   - Diagnose: `aws elbv2 describe-target-health --target-group-arn <ARN>`
   - Fix: Register the instance if missing. If unhealthy, check the health check path and ensure the application is responding with the expected HTTP status code on the health check endpoint.

2. **Security group blocking traffic from the ALB**
   - Diagnose: The instance's security group must allow inbound traffic from the ALB's security group on the application port (e.g., 80). If it only allows `0.0.0.0/0` or a specific IP instead of the ALB SG, internal ALB-to-instance traffic is blocked.
   - Fix: Add an inbound rule to the instance security group: source = ALB security group ID, port = application port.

3. **Health check path returns non-200 status code**
   - Diagnose: Check the ALB target health status. If `State: unhealthy`, the health check is failing. Test the endpoint manually: `curl http://<INSTANCE_IP>/health`.
   - Fix: Fix the health check endpoint to return 200, or update the health check configuration to match the actual healthy response code.

4. **Application listening on wrong port or not started**
   - Diagnose: Even though the app responds on the public IP, it may be using a different port than what the target group is configured for. Check: `ss -tlnp | grep <port>` on the instance.
   - Fix: Ensure the application is listening on the same port as the target group configuration (e.g., if the target group uses port 8080, the app must bind to 8080).

</details>

---

## References

- [AWS ELB Documentation](https://docs.aws.amazon.com/elasticloadbalancing/)
- [AWS CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [GCP Load Balancing](https://cloud.google.com/load-balancing/docs)
- [GCP Cloud CDN](https://cloud.google.com/cdn/docs)
