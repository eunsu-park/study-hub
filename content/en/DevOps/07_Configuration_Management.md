# Lesson 7: Configuration Management

**Previous**: [Terraform Advanced](./06_Terraform_Advanced.md) | **Next**: [Container Orchestration Operations](./08_Container_Orchestration_Operations.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain configuration management principles including idempotency, convergence, and desired state
2. Write Ansible playbooks with tasks, handlers, variables, and conditionals
3. Organize Ansible code using roles, inventory files, and group variables
4. Use Jinja2 templates to generate dynamic configuration files
5. Manage sensitive data with Ansible Vault
6. Understand how Ansible complements Terraform in a complete IaC strategy

---

Configuration management is the practice of systematically handling changes to a system's configuration so it maintains its integrity over time. While Terraform provisions infrastructure (creating VMs, networks, databases), configuration management tools like Ansible configure that infrastructure (installing packages, deploying applications, managing services). Together, they form a complete Infrastructure as Code strategy: Terraform creates the servers, Ansible configures them.

> **Analogy -- Furnished Apartment:** Terraform is like a real estate developer who builds the apartment building (VPC, subnets, EC2 instances). Ansible is the interior designer who furnishes each apartment (installs nginx, configures SSL, deploys the application). You need both: a building without furniture is unusable, and furniture without a building has nowhere to go.

## 1. Configuration Management Principles

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Idempotency** | Running the same operation multiple times produces the same result |
| **Convergence** | The system automatically moves toward the desired state |
| **Desired state** | You declare what the system should look like, not how to get there |
| **Agentless** | No software to install on managed nodes (Ansible uses SSH) |
| **Push vs Pull** | Push: controller sends config (Ansible). Pull: nodes fetch config (Puppet, Chef) |

### Idempotency Example

```bash
# NOT idempotent -- running twice creates duplicate entries:
echo "export PATH=/opt/myapp/bin:$PATH" >> ~/.bashrc

# Idempotent -- Ansible checks first, only adds if missing:
# - name: Add myapp to PATH
#   lineinfile:
#     path: ~/.bashrc
#     line: "export PATH=/opt/myapp/bin:$PATH"
#     state: present
```

### Configuration Management vs IaC Provisioning

```
Terraform (Provisioning):           Ansible (Configuration):
─────────────────────────           ────────────────────────
Create EC2 instances                Install packages
Create VPCs and subnets             Configure nginx/Apache
Create RDS databases                Deploy application code
Create S3 buckets                   Manage system users
Create load balancers               Set up cron jobs
Manage DNS records                  Configure firewalls (iptables)
Create IAM roles                    Manage SSL certificates

Together: Terraform creates the VM → Ansible configures it
```

---

## 2. Ansible Architecture

### How Ansible Works

```
┌─────────────────┐
│  Control Node   │     SSH/WinRM
│  (your laptop   │──────────────────────┐
│   or CI server) │                      │
│                 │           ┌──────────┴──────────┐
│  ansible-playbook│          │                      │
│  runs here       │          ▼                      ▼
└─────────────────┘   ┌─────────────┐     ┌─────────────┐
                      │  Managed    │     │  Managed    │
                      │  Node 1    │     │  Node 2    │
                      │  (web-01)  │     │  (web-02)  │
                      │            │     │            │
                      │  No agent  │     │  No agent  │
                      │  needed!   │     │  needed!   │
                      └─────────────┘     └─────────────┘

Ansible:
  1. Connects to nodes via SSH
  2. Copies small Python programs (modules) to the node
  3. Executes the modules
  4. Removes the temporary files
  5. Reports results back to the control node
```

### Installation

```bash
# Install Ansible (Python-based)
pip install ansible

# Or via system package manager
# macOS
brew install ansible

# Ubuntu
sudo apt update
sudo apt install ansible

# Verify
ansible --version
# ansible [core 2.16.x]
```

---

## 3. Inventory

The inventory defines which hosts Ansible manages and how to connect to them.

### INI Format

```ini
# inventory/hosts.ini

# Individual hosts
[webservers]
web-01 ansible_host=10.0.1.10
web-02 ansible_host=10.0.1.11

[dbservers]
db-01 ansible_host=10.0.2.10 ansible_user=dbadmin

[monitoring]
monitor-01 ansible_host=10.0.3.10

# Group of groups
[production:children]
webservers
dbservers
monitoring

# Variables for a group
[webservers:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/production.pem
http_port=80
```

### YAML Format

```yaml
# inventory/hosts.yml
all:
  children:
    webservers:
      hosts:
        web-01:
          ansible_host: 10.0.1.10
        web-02:
          ansible_host: 10.0.1.11
      vars:
        ansible_user: ubuntu
        http_port: 80

    dbservers:
      hosts:
        db-01:
          ansible_host: 10.0.2.10
          ansible_user: dbadmin

    production:
      children:
        webservers:
        dbservers:
```

### Dynamic Inventory

```bash
# AWS EC2 dynamic inventory -- discovers hosts from AWS automatically
# ansible.cfg or command line:
# plugin: amazon.aws.aws_ec2

# inventory/aws_ec2.yml
plugin: amazon.aws.aws_ec2
regions:
  - us-east-1
filters:
  tag:Environment: production
  instance-state-name: running
keyed_groups:
  - key: tags.Role
    prefix: role
  - key: placement.availability_zone
    prefix: az
compose:
  ansible_host: private_ip_address
```

```bash
# List discovered hosts
ansible-inventory -i inventory/aws_ec2.yml --list

# Use in playbook
ansible-playbook -i inventory/aws_ec2.yml playbook.yml
```

---

## 4. Playbooks

Playbooks are YAML files that define the desired state of your systems.

### Basic Playbook

```yaml
# playbooks/setup-webserver.yml
---
- name: Configure web servers
  hosts: webservers                    # Target group from inventory
  become: true                         # Run as root (sudo)

  vars:
    app_name: myapp
    app_port: 8080
    packages:
      - nginx
      - python3
      - python3-pip

  tasks:
    - name: Update apt cache
      apt:
        update_cache: true
        cache_valid_time: 3600         # Only update if cache is older than 1 hour

    - name: Install required packages
      apt:
        name: "{{ packages }}"
        state: present                 # Ensure packages are installed

    - name: Start and enable nginx
      service:
        name: nginx
        state: started
        enabled: true                  # Start on boot

    - name: Copy nginx configuration
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}
        owner: root
        group: root
        mode: '0644'
      notify: Reload nginx              # Trigger handler on change

    - name: Enable site
      file:
        src: /etc/nginx/sites-available/{{ app_name }}
        dest: /etc/nginx/sites-enabled/{{ app_name }}
        state: link
      notify: Reload nginx

    - name: Remove default site
      file:
        path: /etc/nginx/sites-enabled/default
        state: absent
      notify: Reload nginx

    - name: Ensure application directory exists
      file:
        path: /opt/{{ app_name }}
        state: directory
        owner: www-data
        group: www-data
        mode: '0755'

  handlers:
    - name: Reload nginx
      service:
        name: nginx
        state: reloaded
```

### Running Playbooks

```bash
# Run playbook against inventory
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml

# Dry run (check mode -- show what would change without changing)
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml --check

# Show diff of file changes
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml --check --diff

# Limit to specific hosts
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml --limit web-01

# Extra variables
ansible-playbook playbooks/deploy.yml -e "version=1.2.3 environment=production"

# Verbose output for debugging
ansible-playbook playbooks/setup-webserver.yml -vvv
```

---

## 5. Tasks and Modules

### Common Modules

```yaml
# File management
- name: Create a directory
  file:
    path: /opt/myapp
    state: directory
    owner: deploy
    group: deploy
    mode: '0755'

- name: Copy a file
  copy:
    src: files/app.conf
    dest: /etc/myapp/app.conf
    owner: root
    mode: '0644'
    backup: true                       # Keep backup of original

# Package management
- name: Install packages (apt)
  apt:
    name:
      - nginx
      - certbot
    state: present

- name: Install packages (yum)
  yum:
    name: httpd
    state: latest

# Service management
- name: Ensure service is running
  service:
    name: nginx
    state: started
    enabled: true

# User management
- name: Create application user
  user:
    name: deploy
    shell: /bin/bash
    groups: docker
    append: true                       # Add to group without removing from others

# Command execution
- name: Run database migration
  command: python manage.py migrate
  args:
    chdir: /opt/myapp
  register: migration_result          # Capture output
  changed_when: "'No migrations to apply' not in migration_result.stdout"

# Git checkout
- name: Clone application repository
  git:
    repo: https://github.com/org/myapp.git
    dest: /opt/myapp
    version: "{{ app_version }}"
    force: true

# Cron jobs
- name: Schedule log rotation
  cron:
    name: "Rotate app logs"
    minute: "0"
    hour: "2"
    job: "/usr/sbin/logrotate /etc/logrotate.d/myapp"
```

### Conditionals

```yaml
tasks:
  # Run only on Ubuntu
  - name: Install packages (Debian/Ubuntu)
    apt:
      name: nginx
      state: present
    when: ansible_os_family == "Debian"

  # Run only on CentOS/RHEL
  - name: Install packages (RedHat/CentOS)
    yum:
      name: nginx
      state: present
    when: ansible_os_family == "RedHat"

  # Run based on variable
  - name: Configure SSL
    template:
      src: ssl.conf.j2
      dest: /etc/nginx/ssl.conf
    when: enable_ssl | default(false)

  # Multiple conditions
  - name: Deploy to production
    command: deploy.sh
    when:
      - environment == "production"
      - deploy_enabled | bool
```

### Loops

```yaml
tasks:
  # Simple loop
  - name: Create multiple users
    user:
      name: "{{ item }}"
      state: present
      groups: developers
    loop:
      - alice
      - bob
      - carol

  # Loop with dictionaries
  - name: Create users with specific shells
    user:
      name: "{{ item.name }}"
      shell: "{{ item.shell }}"
      groups: "{{ item.groups }}"
    loop:
      - { name: alice, shell: /bin/bash, groups: developers }
      - { name: bob, shell: /bin/zsh, groups: "developers,docker" }

  # Loop with registered results
  - name: Check service status
    command: systemctl is-active {{ item }}
    loop:
      - nginx
      - redis
      - postgresql
    register: service_status
    ignore_errors: true

  - name: Report stopped services
    debug:
      msg: "{{ item.item }} is not running!"
    loop: "{{ service_status.results }}"
    when: item.rc != 0
```

---

## 6. Jinja2 Templates

Jinja2 templates generate dynamic configuration files.

### Template Example

```jinja2
{# templates/nginx.conf.j2 #}
# Managed by Ansible -- do not edit manually
# Last generated: {{ ansible_date_time.iso8601 }}

upstream {{ app_name }}_backend {
{% for server in groups['webservers'] %}
    server {{ hostvars[server]['ansible_host'] }}:{{ app_port }};
{% endfor %}
}

server {
    listen {{ http_port | default(80) }};
    server_name {{ server_name }};

{% if enable_ssl | default(false) %}
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/{{ server_name }}.crt;
    ssl_certificate_key /etc/ssl/private/{{ server_name }}.key;

    # Redirect HTTP to HTTPS
    if ($scheme != "https") {
        return 301 https://$host$request_uri;
    }
{% endif %}

    location / {
        proxy_pass http://{{ app_name }}_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        return 200 'OK';
        add_header Content-Type text/plain;
    }

    # Static files
    location /static/ {
        alias /opt/{{ app_name }}/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    access_log /var/log/nginx/{{ app_name }}_access.log;
    error_log /var/log/nginx/{{ app_name }}_error.log;
}
```

### Jinja2 Filters

```jinja2
{# Common Jinja2 filters used in Ansible templates #}

{# String manipulation #}
{{ hostname | upper }}                    {# MYSERVER #}
{{ hostname | lower }}                    {# myserver #}
{{ hostname | capitalize }}               {# Myserver #}
{{ path | basename }}                     {# file.txt from /path/to/file.txt #}
{{ path | dirname }}                      {# /path/to from /path/to/file.txt #}

{# Default values #}
{{ db_port | default(5432) }}             {# Use 5432 if db_port is undefined #}

{# List operations #}
{{ my_list | join(', ') }}                {# "a, b, c" #}
{{ my_list | length }}                    {# 3 #}
{{ my_list | first }}                     {# "a" #}
{{ my_list | sort }}                      {# sorted list #}
{{ my_list | unique }}                    {# deduplicated list #}

{# Conditional #}
{{ 'enabled' if feature_flag else 'disabled' }}

{# JSON/YAML output #}
{{ my_dict | to_json }}
{{ my_dict | to_nice_yaml }}

{# Hash/crypto #}
{{ password | password_hash('sha512') }}

{# IP address manipulation #}
{{ '10.0.1.0/24' | ipaddr('network') }}   {# 10.0.1.0 #}
{{ '10.0.1.0/24' | ipaddr('netmask') }}   {# 255.255.255.0 #}
```

---

## 7. Roles

Roles are a standardized way to organize Ansible code into reusable components.

### Role Directory Structure

```
roles/
└── webserver/
    ├── defaults/
    │   └── main.yml          # Default variables (lowest priority)
    ├── vars/
    │   └── main.yml          # Role variables (higher priority)
    ├── tasks/
    │   └── main.yml          # Task list
    ├── handlers/
    │   └── main.yml          # Handler definitions
    ├── templates/
    │   └── nginx.conf.j2     # Jinja2 templates
    ├── files/
    │   └── index.html        # Static files to copy
    ├── meta/
    │   └── main.yml          # Role metadata and dependencies
    └── README.md
```

### Role Implementation

```yaml
# roles/webserver/defaults/main.yml
---
webserver_port: 80
webserver_user: www-data
webserver_root: /var/www/html
webserver_packages:
  - nginx
  - certbot
enable_ssl: false
```

```yaml
# roles/webserver/tasks/main.yml
---
- name: Install web server packages
  apt:
    name: "{{ webserver_packages }}"
    state: present
    update_cache: true
  tags: [install]

- name: Create web root directory
  file:
    path: "{{ webserver_root }}"
    state: directory
    owner: "{{ webserver_user }}"
    group: "{{ webserver_user }}"
    mode: '0755'

- name: Deploy nginx configuration
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/sites-available/default
    owner: root
    group: root
    mode: '0644'
  notify: Restart nginx
  tags: [configure]

- name: Ensure nginx is started and enabled
  service:
    name: nginx
    state: started
    enabled: true
  tags: [service]

- name: Configure SSL
  include_tasks: ssl.yml
  when: enable_ssl
  tags: [ssl]
```

```yaml
# roles/webserver/handlers/main.yml
---
- name: Restart nginx
  service:
    name: nginx
    state: restarted

- name: Reload nginx
  service:
    name: nginx
    state: reloaded
```

```yaml
# roles/webserver/meta/main.yml
---
dependencies:
  - role: common                       # This role depends on the 'common' role
    vars:
      firewall_allowed_ports:
        - 80
        - 443

galaxy_info:
  author: Platform Team
  description: Configures nginx web server
  license: MIT
  min_ansible_version: '2.14'
  platforms:
    - name: Ubuntu
      versions: [jammy, focal]
```

### Using Roles in Playbooks

```yaml
# playbooks/site.yml
---
- name: Configure all servers
  hosts: all
  become: true
  roles:
    - common                           # Apply common role to all hosts

- name: Configure web servers
  hosts: webservers
  become: true
  roles:
    - webserver
    - role: monitoring
      vars:
        monitoring_port: 9090

- name: Configure database servers
  hosts: dbservers
  become: true
  roles:
    - role: database
      vars:
        db_name: myapp
        db_user: appuser
```

---

## 8. Ansible Vault

Ansible Vault encrypts sensitive data (passwords, API keys, certificates).

### Vault Commands

```bash
# Create an encrypted file
ansible-vault create secrets.yml

# Encrypt an existing file
ansible-vault encrypt vars/production.yml

# Decrypt a file (view or edit)
ansible-vault decrypt vars/production.yml
ansible-vault view vars/production.yml
ansible-vault edit vars/production.yml

# Encrypt a single string
ansible-vault encrypt_string 'SuperSecretPassword' --name 'db_password'
# Output:
# db_password: !vault |
#   $ANSIBLE_VAULT;1.1;AES256
#   61626364656667...

# Re-key (change encryption password)
ansible-vault rekey secrets.yml
```

### Using Vault in Playbooks

```yaml
# vars/secrets.yml (encrypted)
---
db_password: SuperSecretPassword
api_key: sk-abc123def456
ssl_private_key: |
  -----BEGIN PRIVATE KEY-----
  MIIEvgIBADANBgkqhkiG9w0BAQ...
  -----END PRIVATE KEY-----
```

```yaml
# playbooks/deploy.yml
---
- name: Deploy application
  hosts: webservers
  become: true

  vars_files:
    - vars/common.yml
    - vars/secrets.yml                 # Encrypted file -- Ansible decrypts at runtime

  tasks:
    - name: Configure database connection
      template:
        src: database.conf.j2
        dest: /etc/myapp/database.conf
        mode: '0600'                   # Restrict permissions for sensitive file
```

```bash
# Run playbook with vault password
ansible-playbook playbooks/deploy.yml --ask-vault-pass

# Or use a password file
ansible-playbook playbooks/deploy.yml --vault-password-file ~/.vault_pass

# Or set environment variable
export ANSIBLE_VAULT_PASSWORD_FILE=~/.vault_pass
ansible-playbook playbooks/deploy.yml
```

### Vault with Multiple Passwords

```bash
# Use vault IDs for different passwords (dev vs production)
ansible-vault encrypt --vault-id dev@prompt vars/dev-secrets.yml
ansible-vault encrypt --vault-id prod@prompt vars/prod-secrets.yml

# Run with multiple vault passwords
ansible-playbook playbooks/site.yml \
  --vault-id dev@~/.vault_pass_dev \
  --vault-id prod@~/.vault_pass_prod
```

---

## 9. Group Variables and Host Variables

### Directory Structure

```
inventory/
├── hosts.yml
├── group_vars/
│   ├── all.yml              # Variables for ALL hosts
│   ├── webservers.yml       # Variables for webservers group
│   ├── dbservers.yml        # Variables for dbservers group
│   └── production.yml       # Variables for production group
└── host_vars/
    ├── web-01.yml           # Variables specific to web-01
    └── db-01.yml            # Variables specific to db-01
```

```yaml
# inventory/group_vars/all.yml
---
ntp_servers:
  - 0.pool.ntp.org
  - 1.pool.ntp.org
timezone: UTC
ansible_user: ubuntu
```

```yaml
# inventory/group_vars/webservers.yml
---
http_port: 80
https_port: 443
app_user: www-data
max_connections: 1024
```

```yaml
# inventory/host_vars/web-01.yml
---
# Override for this specific host
max_connections: 2048                  # Higher capacity server
```

### Variable Precedence (Low to High)

```
1.  Role defaults (roles/x/defaults/main.yml)
2.  Inventory file group vars
3.  Inventory group_vars/all
4.  Inventory group_vars/<group>
5.  Inventory host_vars/<host>
6.  Playbook group_vars/all
7.  Playbook group_vars/<group>
8.  Playbook host_vars/<host>
9.  Play vars
10. Play vars_files
11. Play vars_prompt
12. Role vars (roles/x/vars/main.yml)
13. Task vars (set in block or task)
14. Extra vars (-e on command line)       ← HIGHEST PRIORITY
```

---

## 10. AWX / Ansible Tower

AWX (open-source) and Ansible Tower (Red Hat commercial) provide a web-based UI, REST API, and RBAC for Ansible.

### Key Features

```
AWX / Ansible Tower:
────────────────────
✓ Web UI for running playbooks
✓ Role-Based Access Control (RBAC)
✓ Job scheduling and notifications
✓ Credential management (encrypted)
✓ Inventory management (dynamic)
✓ Audit trail and logging
✓ REST API for integration
✓ Workflow templates (chain playbooks)
```

```bash
# Deploy AWX on Kubernetes
# Clone the AWX operator
git clone https://github.com/ansible/awx-operator.git
cd awx-operator

# Deploy the operator
make deploy

# Create an AWX instance
kubectl apply -f awx-demo.yml
```

---

## Exercises

### Exercise 1: Web Server Playbook

Write an Ansible playbook that configures a web server:
1. Install nginx and certbot
2. Create an application user with SSH key access
3. Deploy a custom nginx configuration using a Jinja2 template
4. Copy a static HTML file to the web root
5. Ensure nginx is running and enabled
6. Use handlers to reload nginx only when config changes
7. Run in check mode first and verify the output

### Exercise 2: Create a Reusable Role

Create an Ansible role called `app_deploy` that:
1. Creates an application directory structure (/opt/app/{releases,shared,current})
2. Clones a Git repository to a new release directory
3. Creates a symlink from `current` to the latest release
4. Installs Python dependencies in a virtualenv
5. Copies a systemd service file from a template
6. Restarts the service
7. Keeps only the last 5 releases (cleanup old ones)

### Exercise 3: Multi-Environment Deployment

Set up an Ansible project with:
1. An inventory for dev and production environments (separate files)
2. Group variables that differ between environments (database host, debug mode, replica count)
3. A vault-encrypted secrets file for each environment
4. A playbook that deploys the application to the correct environment based on inventory
5. Demonstrate running against dev vs production

### Exercise 4: Ansible + Terraform Integration

Design a workflow that uses Terraform and Ansible together:
1. Terraform provisions 2 EC2 instances and outputs their IPs
2. Generate an Ansible dynamic inventory from Terraform output
3. Ansible configures the instances (install packages, deploy app)
4. Write a script or CI pipeline that runs both tools in sequence
5. Explain how you would handle the case where Terraform adds a new instance

---

**Previous**: [Terraform Advanced](./06_Terraform_Advanced.md) | [Overview](00_Overview.md) | **Next**: [Container Orchestration Operations](./08_Container_Orchestration_Operations.md)

**License**: CC BY-NC 4.0
