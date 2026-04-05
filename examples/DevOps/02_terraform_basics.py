#!/usr/bin/env python3
"""Example: Terraform Basics — HCL Generation & State File Parsing

Demonstrates programmatic generation of Terraform HCL configuration and
parsing of Terraform state files. Covers resource blocks, variables,
outputs, modules, and state inspection.
Related lesson: 05_Infrastructure_as_Code.md
"""

# =============================================================================
# WHY GENERATE TERRAFORM HCL PROGRAMMATICALLY?
# While Terraform configurations are typically written by hand, generating
# them from Python is useful for:
#   1. Multi-environment scaffolding (dev/staging/prod)
#   2. Enforcing tagging and naming conventions
#   3. Auditing existing state files for compliance
# =============================================================================

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. HCL BLOCK BUILDER
# =============================================================================
# Terraform HCL consists of blocks: resource, variable, output, data, module.
# We model each as a Python object and serialize to HCL-like syntax.

def hcl_value(v: Any, indent: int = 2) -> str:
    """Convert a Python value to HCL representation."""
    pad = " " * indent
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if v.startswith("var.") or v.startswith("local.") or v.startswith("module."):
            return v  # Terraform reference — no quotes
        return f'"{v}"'
    if isinstance(v, list):
        items = ", ".join(hcl_value(i) for i in v)
        return f"[{items}]"
    if isinstance(v, dict):
        lines = ["{"]
        for k2, v2 in v.items():
            lines.append(f"{pad}  {k2} = {hcl_value(v2, indent + 2)}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
    return str(v)


@dataclass
class TerraformBlock:
    """Generic Terraform block (resource, data, variable, output, module)."""
    block_type: str
    labels: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)
    nested_blocks: list["TerraformBlock"] = field(default_factory=list)

    def to_hcl(self, indent: int = 0) -> str:
        pad = " " * indent
        label_str = " ".join(f'"{l}"' for l in self.labels)
        lines = [f"{pad}{self.block_type} {label_str} {{"]

        for k, v in self.attributes.items():
            lines.append(f"{pad}  {k} = {hcl_value(v, indent + 2)}")

        for nested in self.nested_blocks:
            lines.append("")
            lines.append(nested.to_hcl(indent + 2))

        lines.append(f"{pad}}}")
        return "\n".join(lines)


def resource(rtype: str, name: str, **attrs: Any) -> TerraformBlock:
    return TerraformBlock("resource", [rtype, name], attributes=attrs)


def variable(name: str, **attrs: Any) -> TerraformBlock:
    return TerraformBlock("variable", [name], attributes=attrs)


def output(name: str, **attrs: Any) -> TerraformBlock:
    return TerraformBlock("output", [name], attributes=attrs)


def data_source(dtype: str, name: str, **attrs: Any) -> TerraformBlock:
    return TerraformBlock("data", [dtype, name], attributes=attrs)


# =============================================================================
# 2. MULTI-ENVIRONMENT CONFIG GENERATOR
# =============================================================================

ENVIRONMENTS = {
    "dev": {"instance_type": "t3.micro", "min_size": 1, "max_size": 2},
    "staging": {"instance_type": "t3.small", "min_size": 2, "max_size": 4},
    "prod": {"instance_type": "t3.medium", "min_size": 3, "max_size": 10},
}

COMMON_TAGS = {
    "Project": "my-app",
    "ManagedBy": "terraform",
}


def generate_environment(env_name: str, config: dict) -> list[TerraformBlock]:
    """Generate Terraform blocks for a given environment."""
    tags = {**COMMON_TAGS, "Environment": env_name}

    blocks: list[TerraformBlock] = []

    # Provider (simplified — real HCL uses `provider` not `resource`)
    blocks.append(TerraformBlock(
        "provider", ["aws"],
        attributes={"region": "us-east-1", "default_tags": {"tags": tags}},
    ))

    # VPC
    blocks.append(resource(
        "aws_vpc", f"{env_name}_vpc",
        cidr_block="10.0.0.0/16",
        enable_dns_hostnames=True,
        tags={"Name": f"{env_name}-vpc"},
    ))

    # Subnet
    blocks.append(resource(
        "aws_subnet", f"{env_name}_public",
        vpc_id=f"var.{env_name}_vpc_id",
        cidr_block="10.0.1.0/24",
        map_public_ip_on_launch=True,
        availability_zone="us-east-1a",
    ))

    # Security group
    sg = resource(
        "aws_security_group", f"{env_name}_web_sg",
        name=f"{env_name}-web-sg",
        vpc_id=f"var.{env_name}_vpc_id",
        description=f"Web security group for {env_name}",
    )
    sg.nested_blocks.append(TerraformBlock(
        "ingress", [],
        attributes={
            "from_port": 443,
            "to_port": 443,
            "protocol": "tcp",
            "cidr_blocks": ["0.0.0.0/0"],
        },
    ))
    sg.nested_blocks.append(TerraformBlock(
        "egress", [],
        attributes={
            "from_port": 0,
            "to_port": 0,
            "protocol": "-1",
            "cidr_blocks": ["0.0.0.0/0"],
        },
    ))
    blocks.append(sg)

    # Launch template
    blocks.append(resource(
        "aws_launch_template", f"{env_name}_lt",
        name_prefix=f"{env_name}-",
        image_id="ami-0abcdef1234567890",
        instance_type=config["instance_type"],
    ))

    # Auto Scaling Group
    blocks.append(resource(
        "aws_autoscaling_group", f"{env_name}_asg",
        name=f"{env_name}-asg",
        min_size=config["min_size"],
        max_size=config["max_size"],
        desired_capacity=config["min_size"],
        health_check_type="ELB",
        health_check_grace_period=300,
    ))

    # Variables
    blocks.append(variable(
        f"{env_name}_vpc_id",
        type="string",
        description=f"VPC ID for {env_name} environment",
    ))

    # Outputs
    blocks.append(output(
        f"{env_name}_asg_name",
        value=f"module.{env_name}.asg_name",
        description=f"Auto Scaling Group name for {env_name}",
    ))

    return blocks


# =============================================================================
# 3. TERRAFORM STATE FILE PARSER
# =============================================================================
# Terraform state (terraform.tfstate) is JSON. Parsing it lets you:
#   - Audit resources for compliance (tags, encryption, etc.)
#   - Generate inventory of managed infrastructure
#   - Detect drift between desired and actual state

@dataclass
class StateResource:
    """Parsed resource from Terraform state."""
    address: str
    resource_type: str
    name: str
    provider: str
    attributes: dict[str, Any]

    @property
    def tags(self) -> dict[str, str]:
        return self.attributes.get("tags", {}) or {}


def parse_state_file(state_json: dict) -> list[StateResource]:
    """Parse Terraform state JSON (v4 format) into resource objects."""
    resources: list[StateResource] = []

    for res in state_json.get("resources", []):
        rtype = res.get("type", "")
        rname = res.get("name", "")
        provider = res.get("provider", "")
        mode = res.get("mode", "managed")

        if mode != "managed":
            continue

        for instance in res.get("instances", []):
            attrs = instance.get("attributes", {})
            index_key = instance.get("index_key")
            address = f"{rtype}.{rname}"
            if index_key is not None:
                address += f'["{index_key}"]' if isinstance(index_key, str) else f"[{index_key}]"

            resources.append(StateResource(
                address=address,
                resource_type=rtype,
                name=rname,
                provider=provider,
                attributes=attrs,
            ))

    return resources


def audit_state(resources: list[StateResource]) -> list[str]:
    """Audit state resources for common compliance issues."""
    findings: list[str] = []

    for r in resources:
        # Check for missing tags
        if r.resource_type.startswith("aws_") and not r.tags:
            findings.append(f"WARN: {r.address} has no tags")

        # Check for missing Environment tag
        if r.tags and "Environment" not in r.tags:
            findings.append(f"WARN: {r.address} missing 'Environment' tag")

        # Check for unencrypted storage
        if r.resource_type == "aws_s3_bucket":
            encryption = r.attributes.get("server_side_encryption_configuration")
            if not encryption:
                findings.append(f"CRITICAL: {r.address} has no encryption")

        # Check for public access
        if r.resource_type == "aws_security_group":
            for rule in r.attributes.get("ingress", []):
                cidrs = rule.get("cidr_blocks", [])
                if "0.0.0.0/0" in cidrs and rule.get("from_port") == 22:
                    findings.append(
                        f"CRITICAL: {r.address} allows SSH from 0.0.0.0/0"
                    )

    return findings


# =============================================================================
# 4. DEMO
# =============================================================================

# Sample Terraform state for parsing demo
SAMPLE_STATE = {
    "version": 4,
    "terraform_version": "1.7.0",
    "serial": 42,
    "resources": [
        {
            "mode": "managed",
            "type": "aws_s3_bucket",
            "name": "data_lake",
            "provider": "provider[\"registry.terraform.io/hashicorp/aws\"]",
            "instances": [
                {
                    "attributes": {
                        "id": "my-data-lake-bucket",
                        "bucket": "my-data-lake-bucket",
                        "region": "us-east-1",
                        "tags": {"Project": "data-platform"},
                    }
                }
            ],
        },
        {
            "mode": "managed",
            "type": "aws_security_group",
            "name": "bastion_sg",
            "provider": "provider[\"registry.terraform.io/hashicorp/aws\"]",
            "instances": [
                {
                    "attributes": {
                        "id": "sg-12345",
                        "name": "bastion-sg",
                        "tags": {},
                        "ingress": [
                            {
                                "from_port": 22,
                                "to_port": 22,
                                "protocol": "tcp",
                                "cidr_blocks": ["0.0.0.0/0"],
                            }
                        ],
                    }
                }
            ],
        },
        {
            "mode": "managed",
            "type": "aws_instance",
            "name": "web",
            "provider": "provider[\"registry.terraform.io/hashicorp/aws\"]",
            "instances": [
                {
                    "index_key": 0,
                    "attributes": {
                        "id": "i-abc123",
                        "instance_type": "t3.micro",
                        "tags": {"Name": "web-0", "Environment": "dev"},
                    },
                },
                {
                    "index_key": 1,
                    "attributes": {
                        "id": "i-def456",
                        "instance_type": "t3.micro",
                        "tags": {"Name": "web-1", "Environment": "dev"},
                    },
                },
            ],
        },
    ],
}


if __name__ == "__main__":
    # --- Part 1: Generate HCL for all environments ---
    for env_name, config in ENVIRONMENTS.items():
        print("=" * 70)
        print(f"Terraform Configuration: {env_name.upper()}")
        print("=" * 70)
        blocks = generate_environment(env_name, config)
        for block in blocks:
            print(block.to_hcl())
            print()

    # --- Part 2: Parse and audit state file ---
    print("=" * 70)
    print("Terraform State Audit")
    print("=" * 70)
    resources = parse_state_file(SAMPLE_STATE)
    print(f"Managed resources: {len(resources)}\n")

    for r in resources:
        print(f"  {r.address}")
        print(f"    type:     {r.resource_type}")
        provider_name = r.provider.split("/")[-1].rstrip("]").rstrip('"')
        print(f"    provider: {provider_name}")
        print(f"    tags:     {r.tags}")
        print()

    findings = audit_state(resources)
    if findings:
        print("Compliance findings:")
        for f in findings:
            print(f"  {f}")
    else:
        print("No compliance issues found.")
