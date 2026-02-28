#!/bin/bash
# Exercises for Lesson 24: Cloud Integration
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: cloud-init Configuration ===
# Problem: Write cloud-config to create webadmin user with sudo, install nginx,
#          set timezone to Asia/Seoul.
exercise_1() {
    echo "=== Exercise 1: cloud-init Configuration ==="
    echo ""
    echo "Scenario: Automate initial server setup for a cloud instance using cloud-init."
    echo ""

    echo "Solution: user-data.yaml"
    echo ""
    cat << 'CLOUDINIT'
#cloud-config
# The '#cloud-config' header on line 1 is REQUIRED - it tells cloud-init
# to parse this as a cloud-config YAML file (not a shell script).

# --- User Management ---
users:
  - name: webadmin
    groups: sudo              # Add to sudo group for privilege escalation
    shell: /bin/bash          # Default shell
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    # Passwordless sudo for automation. In production, consider:
    #   sudo: ['ALL=(ALL) ALL']  (requires password)
    # You can also add SSH keys:
    # ssh_authorized_keys:
    #   - ssh-rsa AAAAB3...your-public-key...

# --- Package Management ---
package_update: true          # Run apt-get update / yum update first
# package_upgrade: true       # Uncomment to also upgrade existing packages

packages:
  - nginx                     # Install nginx web server
  # Add more packages as needed:
  # - certbot
  # - python3-certbot-nginx

# --- System Configuration ---
timezone: Asia/Seoul          # Set system timezone
# Equivalent to: timedatectl set-timezone Asia/Seoul

# --- Post-Install Commands ---
runcmd:
  - systemctl enable nginx    # Enable nginx at boot
  - systemctl start nginx     # Start nginx immediately
  # Commands run in order, as root, during first boot only
CLOUDINIT
    echo ""

    echo "How cloud-init works:"
    echo "  1. Cloud provider injects user-data into the instance metadata"
    echo "  2. On first boot, cloud-init reads the metadata (from IMDS or config drive)"
    echo "  3. Executes modules in order: users, packages, timezone, runcmd, etc."
    echo "  4. Only runs once (subsequent reboots skip cloud-init)"
    echo ""

    echo "cloud-init module execution order:"
    echo "  1. disk_setup, mounts       (storage)"
    echo "  2. users, groups, ssh       (users and authentication)"
    echo "  3. package_update, packages (software)"
    echo "  4. timezone, locale         (system settings)"
    echo "  5. write_files              (create config files)"
    echo "  6. runcmd                   (arbitrary commands - runs last)"
    echo ""

    echo "Testing cloud-init locally:"
    echo "  # Validate syntax"
    echo "  cloud-init schema --config-file user-data.yaml"
    echo ""
    echo "  # Check cloud-init logs after boot"
    echo "  cat /var/log/cloud-init.log        # Detailed log"
    echo "  cat /var/log/cloud-init-output.log # Command output"
    echo "  cloud-init status --long           # Current status"
    echo ""

    echo "Using with AWS:"
    echo "  aws ec2 run-instances --user-data file://user-data.yaml ..."
    echo ""
    echo "Using with multipass (local testing):"
    echo "  multipass launch --cloud-init user-data.yaml"
}

# === Exercise 2: AWS CLI Query ===
# Problem: List running EC2 instances with ID, Name, and Private IP in table format.
exercise_2() {
    echo "=== Exercise 2: AWS CLI Query ==="
    echo ""
    echo "Scenario: Query running EC2 instances with formatted output."
    echo ""

    echo "Solution:"
    echo ""
    cat << 'AWSCLI'
aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,Name:Tags[?Key==`Name`].Value|[0],IP:PrivateIpAddress}' \
    --output table
AWSCLI
    echo ""

    echo "Breakdown:"
    echo ""
    echo "  --filters \"Name=instance-state-name,Values=running\""
    echo "    Server-side filter: only return instances in 'running' state."
    echo "    More efficient than filtering client-side (less data transferred)."
    echo "    Other states: pending, shutting-down, terminated, stopping, stopped."
    echo ""
    echo "  --query (JMESPath expression):"
    echo "    Reservations[]                    # Flatten all reservations"
    echo "    .Instances[]                      # Flatten all instances"
    echo "    .{                                # Create custom object per instance"
    echo "      ID: InstanceId,                 # Instance ID (i-0abc123...)"
    echo "      Name: Tags[?Key==\`Name\`]       # Filter tags where Key='Name'"
    echo "            .Value|[0],               # Get first Value (the name)"
    echo "      IP: PrivateIpAddress            # Private IP"
    echo "    }"
    echo ""
    echo "  --output table"
    echo "    Formats output as an ASCII table."
    echo "    Other options: json (default), text (tab-separated), yaml"
    echo ""

    echo "Example output:"
    echo "  ------------------------------------------"
    echo "  |         DescribeInstances              |"
    echo "  +----------+------------+----------------+"
    echo "  |  ID      |  Name      |  IP            |"
    echo "  +----------+------------+----------------+"
    echo "  |  i-0a... |  web-01    |  10.0.1.10     |"
    echo "  |  i-0b... |  web-02    |  10.0.1.11     |"
    echo "  |  i-0c... |  api-01    |  10.0.2.20     |"
    echo "  +----------+------------+----------------+"
    echo ""

    echo "More useful AWS CLI queries:"
    echo "  # List instances by tag"
    echo "  aws ec2 describe-instances --filters \"Name=tag:Environment,Values=production\""
    echo ""
    echo "  # Get public IPs only"
    echo "  aws ec2 describe-instances --query 'Reservations[].Instances[].PublicIpAddress' --output text"
    echo ""
    echo "  # Count running instances per type"
    echo "  aws ec2 describe-instances --filters 'Name=instance-state-name,Values=running' \\"
    echo "    --query 'Reservations[].Instances[].InstanceType' --output text | sort | uniq -c"
}

# === Exercise 3: IMDSv2 Script ===
# Problem: Write a script to query instance metadata using IMDSv2 (token-based).
exercise_3() {
    echo "=== Exercise 3: EC2 Instance Metadata Service v2 (IMDSv2) ==="
    echo ""
    echo "Scenario: Securely query EC2 instance metadata using token-based authentication."
    echo ""

    echo "Solution: get_instance_info.sh"
    echo ""
    cat << 'IMDS'
#!/bin/bash
# Query EC2 instance metadata using IMDSv2 (token-based)

# --- Step 1: Get a session token ---
# IMDSv2 requires a token for all metadata requests
# This prevents SSRF attacks (Server-Side Request Forgery)
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Flags explained:
#   -s     = silent (no progress bar)
#   -X PUT = HTTP PUT method (required for token endpoint)
#   -H     = Custom header: token TTL (21600s = 6 hours max)

# --- Step 2: Query metadata using the token ---
AZ=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/availability-zone)

INSTANCE_TYPE=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type)

# -H passes the token as an HTTP header for authentication

# --- Step 3: Display results ---
echo "Availability Zone: $AZ"
echo "Instance Type: $INSTANCE_TYPE"
IMDS
    echo ""

    echo "IMDSv1 vs IMDSv2:"
    echo "  IMDSv1: Simple GET request (no token needed)"
    echo "    curl http://169.254.169.254/latest/meta-data/instance-id"
    echo "    Risk: Vulnerable to SSRF attacks (attacker tricks app into querying IMDS)"
    echo ""
    echo "  IMDSv2: Requires PUT to get token, then GET with token header"
    echo "    Two-step process prevents SSRF because:"
    echo "    1. PUT requests are blocked by most proxies/WAFs"
    echo "    2. Custom headers (X-aws-ec2-metadata-token) aren't forwarded by default"
    echo "    3. Token has limited TTL and is bound to the session"
    echo ""

    echo "Common metadata endpoints:"
    echo "  /latest/meta-data/instance-id           # i-0abc123..."
    echo "  /latest/meta-data/instance-type          # t3.medium"
    echo "  /latest/meta-data/local-ipv4             # Private IP"
    echo "  /latest/meta-data/public-ipv4            # Public IP"
    echo "  /latest/meta-data/placement/availability-zone  # us-east-1a"
    echo "  /latest/meta-data/iam/security-credentials/<role>  # IAM role credentials"
    echo "  /latest/meta-data/hostname               # Internal DNS name"
    echo "  /latest/dynamic/instance-identity/document  # Full instance identity (JSON)"
    echo ""

    echo "Best practice: Enforce IMDSv2 only (disable v1):"
    echo "  aws ec2 modify-instance-metadata-options \\"
    echo "    --instance-id i-0abc123 \\"
    echo "    --http-tokens required \\"
    echo "    --http-endpoint enabled"
    echo ""
    echo "  'http-tokens required' = IMDSv2 only (v1 requests will fail)"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
