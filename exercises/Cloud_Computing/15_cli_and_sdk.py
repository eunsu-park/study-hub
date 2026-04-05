"""
Exercises for Lesson 15: CLI and SDK
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates CLI profile management, JMESPath queries, SDK automation, and error handling.
"""


# === Exercise 1: CLI Profile Management ===
def exercise_1():
    """Configure AWS CLI profiles for multi-environment access."""

    print("AWS CLI Profile Management:")
    print("=" * 70)
    print()
    print("  Scenario: Three AWS environments (dev, staging, production)")
    print("  Goal: Switch between them without changing the default profile.")
    print()

    # Configuration steps
    print("  Step 1: Configure each profile")
    print("    aws configure --profile dev")
    print("    # Enter dev account key, secret, region (ap-northeast-2), format (json)")
    print()
    print("    aws configure --profile staging")
    print("    # Enter staging account credentials")
    print()
    print("    aws configure --profile production")
    print("    # Enter production account credentials")
    print()

    # Resulting credentials file
    print("  Resulting ~/.aws/credentials:")
    credentials = {
        "default": "aws_access_key_id = ...",
        "dev": "aws_access_key_id = AKIA...DEV",
        "staging": "aws_access_key_id = AKIA...STG",
        "production": "aws_access_key_id = AKIA...PRD",
    }
    for profile, cred in credentials.items():
        print(f"    [{profile}]")
        print(f"    {cred}")
        print(f"    aws_secret_access_key = ...")
        print()

    # Ways to use a specific profile
    # Why --profile is best for scripts: It's explicit and not dependent
    # on shell state, so the same command always targets the same account.
    print("  List S3 buckets in production (without changing default):")
    print()
    print("    # Option 1: --profile flag (best for scripts)")
    print("    aws s3 ls --profile production")
    print()
    print("    # Option 2: Environment variable (for current shell session)")
    print("    export AWS_PROFILE=production")
    print("    aws s3 ls")
    print()
    print("    # Option 3: Per-command environment variable")
    print("    AWS_PROFILE=production aws s3 ls")
    print()
    print("  Best practice: Use --profile in scripts for explicit, "
          "repeatable targeting.")


# === Exercise 2: JMESPath Query Filtering ===
def exercise_2():
    """Write AWS CLI command with JMESPath query for instance reporting."""

    print("JMESPath Query for EC2 Instance Report:")
    print("=" * 70)
    print()
    print("  Goal: List running Production instances showing ID, type, and public IP.")
    print()

    # AWS CLI command
    print("  AWS CLI Command:")
    print("    aws ec2 describe-instances \\")
    print('        --filters "Name=tag:Environment,Values=Production" \\')
    print('                  "Name=instance-state-name,Values=running" \\')
    print("        --query 'Reservations[*].Instances[*]."
          "[InstanceId, InstanceType, PublicIpAddress]' \\")
    print("        --output table")
    print()

    # JMESPath explanation
    # Why JMESPath: AWS CLI returns deeply nested JSON. JMESPath extracts
    # only the fields you need without piping through jq or Python.
    print("  JMESPath Expression Breakdown:")
    parts = [
        ("Reservations[*]", "Iterate all reservation groups (AWS groups by launch request)"),
        (".Instances[*]", "Iterate instances within each reservation"),
        (".[InstanceId, InstanceType, PublicIpAddress]",
         "Select specific fields as array columns"),
    ]
    for expr, explanation in parts:
        print(f"    {expr}")
        print(f"      -> {explanation}")
    print()

    # Simulated output
    print("  Example Output:")
    print("    -----------------------------------------------")
    print("    |  InstanceId      | Type       | PublicIP     |")
    print("    -----------------------------------------------")
    print("    |  i-0abc123def456 | t3.medium  | 54.180.1.10  |")
    print("    |  i-0def789abc012 | t3.large   | 54.180.2.20  |")
    print("    -----------------------------------------------")
    print()

    # Equivalent gcloud command
    print("  Equivalent gcloud command:")
    print("    gcloud compute instances list \\")
    print('        --filter="status=RUNNING AND labels.environment=production" \\')
    print('        --format="table(name,machineType.basename(),'
          'networkInterfaces[0].accessConfigs[0].natIP)"')
    print()
    print("  Note: gcloud uses --filter (server-side) + --format (client-side)")
    print("  while AWS CLI uses --filters (server-side) + --query (JMESPath, client-side).")


# === Exercise 3: Boto3 Automation Script ===
def exercise_3():
    """Demonstrate a boto3 script that lists S3 buckets with object counts."""

    print("Boto3 S3 Bucket Inventory Script:")
    print("=" * 70)
    print()

    # Simulated script output (since we don't have actual AWS credentials)
    print("  Script Design:")
    print("    1. List all S3 buckets in the account")
    print("    2. For each bucket, count objects using paginator")
    print("    3. Handle errors: NoSuchBucket, AccessDenied")
    print()

    # The actual script
    print("  Script:")
    print("  " + "-" * 60)
    script_lines = [
        "import boto3",
        "from botocore.exceptions import ClientError",
        "",
        "def count_bucket_objects(s3_client, bucket_name):",
        '    """Count objects using paginator to handle large buckets."""',
        "    paginator = s3_client.get_paginator('list_objects_v2')",
        "    count = 0",
        "    try:",
        "        for page in paginator.paginate(Bucket=bucket_name):",
        "            count += page.get('KeyCount', 0)",
        "    except ClientError as e:",
        "        error_code = e.response['Error']['Code']",
        "        if error_code == 'NoSuchBucket':",
        "            return None",
        "        elif error_code in ('AccessDenied', '403'):",
        "            return 'ACCESS_DENIED'",
        "        else:",
        "            raise",
        "    return count",
        "",
        "def main():",
        "    # S3 is global; us-east-1 is the standard endpoint",
        "    s3 = boto3.client('s3', region_name='us-east-1')",
        "    response = s3.list_buckets()",
        "    buckets = response.get('Buckets', [])",
        "",
        "    for bucket in buckets:",
        "        name = bucket['Name']",
        "        count = count_bucket_objects(s3, name)",
        "        print(f'{name}: {count} objects')",
    ]
    for line in script_lines:
        print(f"    {line}")
    print()

    # Why paginator: Buckets can contain millions of objects. Without
    # pagination, list_objects_v2 returns at most 1000 keys per call.
    print("  Key Design Decisions:")
    decisions = [
        ("Paginator", "Buckets can contain millions of objects. Without pagination, "
         "list_objects_v2 returns at most 1,000 keys per call."),
        ("ClientError", "Catch specific error types rather than broad Exception "
         "to distinguish between 'bucket missing' and 'permission denied'."),
        ("region_name='us-east-1'", "list_buckets is a global operation; individual "
         "bucket operations may need region-specific clients."),
    ]
    for name, reason in decisions:
        print(f"    - {name}: {reason}")

    # Simulated output
    print()
    print("  Simulated Output:")
    buckets = [
        ("my-app-data", 15_234),
        ("my-logs-bucket", 1_456_789),
        ("my-config-bucket", 42),
        ("restricted-bucket", "access denied"),
    ]
    print(f"    {'Bucket Name':<35} {'Object Count':>15}")
    print("    " + "-" * 51)
    for name, count in buckets:
        print(f"    {name:<35} {str(count):>15}")


# === Exercise 4: gcloud Output Formatting ===
def exercise_4():
    """Extract external IPs from gcloud for use in a shell script."""

    print("gcloud Output Formatting for Scripts:")
    print("=" * 70)
    print()
    print("  Goal: Extract external IPs of all running http-server instances")
    print("  in asia-northeast3, one per line (for shell script consumption).")
    print()

    # The value() format outputs raw values with no headers or decorators.
    # Why value(): It's the cleanest output format for piping into other
    # commands -- no table headers, no decorators, just data.
    print("  Command:")
    print("    gcloud compute instances list \\")
    print('        --filter="status=RUNNING AND tags.items=http-server '
          'AND zone:asia-northeast3" \\')
    print('        --format="value(networkInterfaces[0].accessConfigs[0].natIP)"')
    print()

    print("  Output (one IP per line):")
    ips = ["34.64.100.10", "34.64.100.20", "34.64.100.30"]
    for ip in ips:
        print(f"    {ip}")
    print()

    # Usage in a script
    print("  Usage in a health-check script:")
    print("    #!/bin/bash")
    print("    ips=$(gcloud compute instances list \\")
    print('        --filter="status=RUNNING AND tags.items=http-server" \\')
    print('        --format="value(networkInterfaces[0].accessConfigs[0].natIP)")')
    print()
    print("    for ip in $ips; do")
    print('        if curl -sf --max-time 5 "http://$ip/health" > /dev/null; then')
    print('            echo "OK: $ip"')
    print("        else")
    print('            echo "FAIL: $ip"')
    print("        fi")
    print("    done")
    print()

    # Format alternatives
    print("  Other --format options:")
    alternatives = [
        ("json", "Full JSON for programmatic processing"),
        ("csv(name,natIP)", "CSV with column headers"),
        ("table(name,zone,natIP)", "Human-readable table"),
        ("value(name)", "Raw values, no headers (best for scripts)"),
    ]
    for fmt, desc in alternatives:
        print(f'    --format="{fmt}"  -> {desc}')


# === Exercise 5: SDK Error Handling ===
def exercise_5():
    """Identify problems in GCP code and write an improved version."""

    print("GCP SDK Error Handling:")
    print("=" * 70)
    print()

    # Original problematic code
    print("  Original (problematic) code:")
    print("    from google.cloud import storage")
    print("    client = storage.Client()")
    print("    bucket = client.bucket('my-bucket')")
    print("    blob = bucket.blob('data/report.csv')")
    print("    blob.download_to_filename('/tmp/report.csv')")
    print('    print("Downloaded!")')
    print()

    # Problems identified
    problems = [
        "No error handling -- crashes with unhandled exception if bucket/blob missing.",
        "No authentication check -- confusing error if GOOGLE_APPLICATION_CREDENTIALS unset.",
        "No feedback on what went wrong or where to look.",
    ]
    print("  Problems:")
    for i, p in enumerate(problems, 1):
        print(f"    {i}. {p}")
    print()

    # Improved version
    # Why separate credential check: Distinguishes "no credentials configured"
    # (setup issue) from "bucket not found" (resource issue). Each error type
    # needs different debugging steps.
    print("  Improved version:")
    print("  " + "-" * 60)
    improved_lines = [
        "from google.cloud import storage",
        "from google.api_core.exceptions import NotFound, Forbidden, GoogleAPICallError",
        "from google.auth.exceptions import DefaultCredentialsError",
        "",
        "def download_blob(bucket_name, source_blob_name, destination_file):",
        '    """Download a blob. Returns True on success, False on failure."""',
        "    try:",
        "        client = storage.Client()",
        "    except DefaultCredentialsError:",
        '        print("ERROR: No credentials found. Set GOOGLE_APPLICATION_CREDENTIALS")',
        "        return False",
        "",
        "    try:",
        "        bucket = client.bucket(bucket_name)",
        "        blob = bucket.blob(source_blob_name)",
        "        blob.download_to_filename(destination_file)",
        '        print(f"Downloaded gs://{bucket_name}/{source_blob_name}")',
        "        return True",
        "    except NotFound:",
        '        print(f"ERROR: gs://{bucket_name}/{source_blob_name} not found.")',
        "        return False",
        "    except Forbidden:",
        '        print(f"ERROR: Permission denied. Check IAM roles.")',
        "        return False",
        "    except GoogleAPICallError as e:",
        '        print(f"ERROR: GCP API call failed: {e.message}")',
        "        return False",
    ]
    for line in improved_lines:
        print(f"    {line}")
    print()

    print("  Key Improvements:")
    improvements = [
        "Separate credential init from API call for clear error attribution.",
        "Catch specific exception types (NotFound, Forbidden) for actionable messages.",
        "Return boolean so callers can handle failures programmatically.",
        "Include full GCS path in error messages for easy debugging.",
    ]
    for imp in improvements:
        print(f"    - {imp}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: CLI Profile Management ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: JMESPath Query Filtering ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Boto3 Automation Script ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: gcloud Output Formatting ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: SDK Error Handling ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
