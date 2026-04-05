#!/usr/bin/env python3
"""Example: Ansible Playbook & Inventory Generator

Demonstrates programmatic generation of Ansible playbooks and inventory
files in YAML format. Covers roles, handlers, variables, templates,
and multi-group inventory.
Related lesson: 06_Configuration_Management.md
"""

# =============================================================================
# WHY GENERATE ANSIBLE CONFIG PROGRAMMATICALLY?
# Large-scale Ansible setups can have hundreds of hosts and dozens of roles.
# Generating inventory and playbooks from a source of truth (CMDB, cloud API)
# ensures consistency and reduces manual errors.
# =============================================================================

import yaml
import json
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


# =============================================================================
# 1. INVENTORY BUILDER
# =============================================================================

@dataclass
class Host:
    """A single managed host."""
    name: str
    ansible_host: str
    ansible_user: str = "ubuntu"
    ansible_port: int = 22
    vars: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ansible_host": self.ansible_host,
            "ansible_user": self.ansible_user,
            "ansible_port": self.ansible_port,
        }
        d.update(self.vars)
        return d


@dataclass
class HostGroup:
    """A group of hosts with optional group vars."""
    name: str
    hosts: list[Host] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)

    def to_inventory_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.hosts:
            d["hosts"] = {h.name: h.to_dict() for h in self.hosts}
        if self.vars:
            d["vars"] = self.vars
        if self.children:
            d["children"] = {c: None for c in self.children}
        return d


@dataclass
class Inventory:
    """Complete Ansible inventory."""
    groups: dict[str, HostGroup] = field(default_factory=dict)

    def add_group(self, group: HostGroup) -> None:
        self.groups[group.name] = group

    def to_dict(self) -> dict:
        return {
            "all": {
                "children": {
                    name: group.to_inventory_dict()
                    for name, group in self.groups.items()
                }
            }
        }

    def to_yaml(self) -> str:
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
        )


def build_sample_inventory() -> Inventory:
    """Build a multi-tier inventory (web, app, db servers)."""
    inv = Inventory()

    # Web servers
    web_group = HostGroup(
        name="webservers",
        hosts=[
            Host("web-01", "10.0.1.10", vars={"nginx_worker_processes": 4}),
            Host("web-02", "10.0.1.11", vars={"nginx_worker_processes": 4}),
            Host("web-03", "10.0.1.12", vars={"nginx_worker_processes": 8}),
        ],
        vars={
            "nginx_version": "1.25",
            "ssl_certificate_path": "/etc/ssl/certs/app.pem",
        },
    )
    inv.add_group(web_group)

    # Application servers
    app_group = HostGroup(
        name="appservers",
        hosts=[
            Host("app-01", "10.0.2.10"),
            Host("app-02", "10.0.2.11"),
        ],
        vars={
            "app_port": 8080,
            "gunicorn_workers": 4,
            "app_env": "production",
        },
    )
    inv.add_group(app_group)

    # Database servers
    db_group = HostGroup(
        name="dbservers",
        hosts=[
            Host("db-primary", "10.0.3.10", vars={"pg_role": "primary"}),
            Host("db-replica", "10.0.3.11", vars={"pg_role": "replica"}),
        ],
        vars={
            "postgresql_version": "16",
            "pg_max_connections": 200,
            "pg_shared_buffers": "4GB",
        },
    )
    inv.add_group(db_group)

    # Meta group: all production hosts
    prod_group = HostGroup(
        name="production",
        children=["webservers", "appservers", "dbservers"],
        vars={
            "ansible_become": True,
            "ntp_server": "time.google.com",
        },
    )
    inv.add_group(prod_group)

    return inv


# =============================================================================
# 2. PLAYBOOK BUILDER
# =============================================================================

@dataclass
class Task:
    """A single Ansible task."""
    name: str
    module: str
    args: dict[str, Any] = field(default_factory=dict)
    become: bool = False
    when: str | None = None
    notify: str | None = None
    register: str | None = None
    loop: list[str] | None = None
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        d[self.module] = self.args if self.args else None
        if self.become:
            d["become"] = True
        if self.when:
            d["when"] = self.when
        if self.notify:
            d["notify"] = self.notify
        if self.register:
            d["register"] = self.register
        if self.loop:
            d["loop"] = self.loop
        if self.tags:
            d["tags"] = self.tags
        return d


@dataclass
class Handler:
    """An Ansible handler (triggered by notify)."""
    name: str
    module: str
    args: dict[str, Any] = field(default_factory=dict)
    become: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        d[self.module] = self.args
        if self.become:
            d["become"] = True
        return d


@dataclass
class Play:
    """A single play within a playbook."""
    name: str
    hosts: str
    become: bool = False
    gather_facts: bool = True
    vars: dict[str, Any] | None = None
    roles: list[str | dict] | None = None
    pre_tasks: list[Task] | None = None
    tasks: list[Task] | None = None
    handlers: list[Handler] | None = None
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "hosts": self.hosts,
            "become": self.become,
            "gather_facts": self.gather_facts,
        }
        if self.vars:
            d["vars"] = self.vars
        if self.roles:
            d["roles"] = self.roles
        if self.pre_tasks:
            d["pre_tasks"] = [t.to_dict() for t in self.pre_tasks]
        if self.tasks:
            d["tasks"] = [t.to_dict() for t in self.tasks]
        if self.handlers:
            d["handlers"] = [h.to_dict() for h in self.handlers]
        if self.tags:
            d["tags"] = self.tags
        return d


@dataclass
class Playbook:
    """A complete Ansible playbook (list of plays)."""
    plays: list[Play] = field(default_factory=list)

    def to_yaml(self) -> str:
        return yaml.dump(
            [p.to_dict() for p in self.plays],
            default_flow_style=False,
            sort_keys=False,
        )


# =============================================================================
# 3. PLAYBOOK GENERATORS
# =============================================================================

def generate_webserver_playbook() -> Playbook:
    """Generate a playbook for configuring Nginx web servers."""
    play = Play(
        name="Configure Nginx Web Servers",
        hosts="webservers",
        become=True,
        vars={
            "nginx_worker_processes": "auto",
            "nginx_worker_connections": 1024,
            "app_upstream_port": 8080,
        },
        tasks=[
            Task(
                name="Install Nginx",
                module="ansible.builtin.apt",
                args={
                    "name": "nginx",
                    "state": "present",
                    "update_cache": True,
                },
                tags=["install"],
            ),
            Task(
                name="Create Nginx config from template",
                module="ansible.builtin.template",
                args={
                    "src": "templates/nginx.conf.j2",
                    "dest": "/etc/nginx/nginx.conf",
                    "mode": "0644",
                    "validate": "nginx -t -c %s",
                },
                notify="Restart Nginx",
                tags=["configure"],
            ),
            Task(
                name="Deploy SSL certificates",
                module="ansible.builtin.copy",
                args={
                    "src": "files/ssl/{{ inventory_hostname }}.pem",
                    "dest": "{{ ssl_certificate_path }}",
                    "mode": "0600",
                },
                tags=["ssl"],
            ),
            Task(
                name="Ensure Nginx is running and enabled",
                module="ansible.builtin.systemd",
                args={
                    "name": "nginx",
                    "state": "started",
                    "enabled": True,
                },
            ),
            Task(
                name="Open firewall ports",
                module="community.general.ufw",
                args={
                    "rule": "allow",
                    "port": "{{ item }}",
                    "proto": "tcp",
                },
                loop=["80", "443"],
                become=True,
                tags=["firewall"],
            ),
        ],
        handlers=[
            Handler(
                name="Restart Nginx",
                module="ansible.builtin.systemd",
                args={"name": "nginx", "state": "restarted"},
            ),
        ],
    )
    return Playbook(plays=[play])


def generate_app_deploy_playbook() -> Playbook:
    """Generate a rolling deployment playbook for the application tier."""
    pre_tasks = [
        Task(
            name="Disable host in load balancer",
            module="ansible.builtin.uri",
            args={
                "url": "http://lb.internal/api/disable/{{ inventory_hostname }}",
                "method": "POST",
            },
        ),
    ]

    tasks = [
        Task(
            name="Pull latest application code",
            module="ansible.builtin.git",
            args={
                "repo": "https://github.com/myorg/myapp.git",
                "dest": "/opt/myapp",
                "version": "{{ app_version | default('main') }}",
                "force": True,
            },
            tags=["deploy"],
        ),
        Task(
            name="Install Python dependencies",
            module="ansible.builtin.pip",
            args={
                "requirements": "/opt/myapp/requirements.txt",
                "virtualenv": "/opt/myapp/venv",
                "virtualenv_python": "python3.12",
            },
            tags=["deps"],
        ),
        Task(
            name="Run database migrations",
            module="ansible.builtin.command",
            args={"cmd": "/opt/myapp/venv/bin/python manage.py migrate"},
            when="inventory_hostname == groups['appservers'][0]",
            tags=["migrate"],
        ),
        Task(
            name="Restart application service",
            module="ansible.builtin.systemd",
            args={"name": "myapp", "state": "restarted", "daemon_reload": True},
            become=True,
        ),
        Task(
            name="Wait for health check",
            module="ansible.builtin.uri",
            args={
                "url": "http://localhost:{{ app_port }}/health",
                "status_code": 200,
            },
            register="health_check",
        ),
        Task(
            name="Re-enable host in load balancer",
            module="ansible.builtin.uri",
            args={
                "url": "http://lb.internal/api/enable/{{ inventory_hostname }}",
                "method": "POST",
            },
        ),
    ]

    play = Play(
        name="Rolling Application Deployment",
        hosts="appservers",
        become=True,
        gather_facts=True,
        vars={"serial": 1},  # One host at a time for rolling deploy
        pre_tasks=pre_tasks,
        tasks=tasks,
        tags=["deploy"],
    )
    return Playbook(plays=[play])


# =============================================================================
# 4. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Inventory ---
    inv = build_sample_inventory()
    print("=" * 70)
    print("Generated Ansible Inventory (inventory.yml)")
    print("=" * 70)
    print(inv.to_yaml())

    # --- Web server playbook ---
    web_pb = generate_webserver_playbook()
    print("=" * 70)
    print("Generated Webserver Playbook (webservers.yml)")
    print("=" * 70)
    print(web_pb.to_yaml())

    # --- App deployment playbook ---
    app_pb = generate_app_deploy_playbook()
    print("=" * 70)
    print("Generated App Deployment Playbook (deploy.yml)")
    print("=" * 70)
    print(app_pb.to_yaml())

    # --- Stats ---
    total_hosts = sum(
        len(g.hosts)
        for g in inv.groups.values()
    )
    print("=" * 70)
    print("Inventory Summary")
    print("=" * 70)
    print(f"  Groups:       {len(inv.groups)}")
    print(f"  Total hosts:  {total_hosts}")
    for name, group in inv.groups.items():
        host_count = len(group.hosts)
        child_count = len(group.children)
        if host_count:
            print(f"    {name}: {host_count} hosts")
        if child_count:
            print(f"    {name}: {child_count} child groups")
