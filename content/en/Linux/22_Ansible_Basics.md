# Ansible Basics

**Previous**: [Virtualization (KVM)](./21_Virtualization_KVM.md) | **Next**: [Advanced Networking](./23_Advanced_Networking.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Ansible's agentless push-based architecture and how it differs from Puppet and Chef
2. Define inventories in INI and YAML formats with groups, variables, and dynamic sources
3. Execute ad-hoc commands across multiple hosts using common Ansible modules
4. Write playbooks with tasks, conditionals, loops, handlers, and blocks
5. Organize automation code into reusable roles with proper directory structure
6. Manage variables across multiple scopes and use Jinja2 templates for configuration files
7. Encrypt sensitive data with Ansible Vault for secure credential management

---

## Table of Contents

1. [Ansible Overview](#1-ansible-overview)
2. [Installation and Configuration](#2-installation-and-configuration)
3. [Inventory](#3-inventory)
4. [Ad-hoc Commands](#4-ad-hoc-commands)
5. [Playbook](#5-playbook)
6. [Variables and Facts](#6-variables-and-facts)
7. [Roles](#7-roles)
8. [Ansible Vault](#8-ansible-vault)

---

Managing one server by hand is feasible; managing ten is tedious; managing hundreds is impossible without automation. Ansible solves this by letting you describe your desired infrastructure state in simple YAML files and applying it across any number of servers over SSH -- no agents to install, no complex setup. It is the most widely adopted configuration management tool in the industry, and learning it is a force multiplier for any system administrator.

## 1. Ansible Overview

### What is Ansible?

Ansible is an IT automation tool that works via SSH without requiring agents.

```
┌─────────────────────────────────────────────────────────────┐
│                     Control Node                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Ansible                                             │   │
│  │  ├── Playbooks (YAML)                                │   │
│  │  ├── Inventory                                       │   │
│  │  ├── Modules                                         │   │
│  │  └── Plugins                                         │   │
│  └───────────────────────┬─────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────┘
                            │ SSH
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Managed Node  │  │ Managed Node  │  │ Managed Node  │
│ (web-server1) │  │ (web-server2) │  │ (db-server)   │
│   No agent    │  │   No agent    │  │   No agent    │
└───────────────┘  └───────────────┘  └───────────────┘
```

### Ansible vs Other Tools

| Feature | Ansible | Puppet | Chef |
|---------|---------|--------|------|
| Agent | Not required | Required | Required |
| Language | YAML | Ruby DSL | Ruby |
| Push/Pull | Push | Pull | Pull |
| Learning Curve | Low | Medium | High |

---

## 2. Installation and Configuration

### Installing Ansible

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ansible

# RHEL/CentOS
sudo yum install epel-release
sudo yum install ansible

# Install via pip (recommended)
pip3 install ansible

# Check version
ansible --version
```

### SSH Key Configuration

```bash
# Generate SSH key
ssh-keygen -t ed25519 -f ~/.ssh/ansible_key

# Copy key to target server
ssh-copy-id -i ~/.ssh/ansible_key.pub user@target-server

# Test SSH connection
ssh -i ~/.ssh/ansible_key user@target-server
```

### ansible.cfg Configuration

```bash
# /etc/ansible/ansible.cfg (global)
# ~/.ansible.cfg (user)
# ./ansible.cfg (project)
```

```ini
# ansible.cfg
[defaults]
inventory = ./inventory
remote_user = ansible
private_key_file = ~/.ssh/ansible_key
host_key_checking = False
retry_files_enabled = False
forks = 20
timeout = 30

[privilege_escalation]
become = True
become_method = sudo
become_user = root
become_ask_pass = False

[ssh_connection]
pipelining = True
ssh_args = -o ControlMaster=auto -o ControlPersist=60s
```

---

## 3. Inventory

### INI Format

```ini
# inventory/hosts
[webservers]
web1.example.com
web2.example.com ansible_host=192.168.1.11

[dbservers]
db1.example.com ansible_host=192.168.1.21 ansible_port=2222
db2.example.com ansible_user=dbadmin

[loadbalancers]
lb1.example.com

# Group of groups
[production:children]
webservers
dbservers
loadbalancers

# Group variables
[webservers:vars]
http_port=80
max_clients=200

[all:vars]
ansible_python_interpreter=/usr/bin/python3
```

### YAML Format

```yaml
# inventory/hosts.yml
all:
  children:
    webservers:
      hosts:
        web1.example.com:
        web2.example.com:
          ansible_host: 192.168.1.11
      vars:
        http_port: 80
        max_clients: 200

    dbservers:
      hosts:
        db1.example.com:
          ansible_host: 192.168.1.21
          ansible_port: 2222
        db2.example.com:
          ansible_user: dbadmin

    production:
      children:
        webservers:
        dbservers:

  vars:
    ansible_python_interpreter: /usr/bin/python3
```

### Dynamic Inventory

```bash
# AWS EC2 dynamic inventory
pip install boto3 botocore

# ansible.cfg
[defaults]
inventory = aws_ec2.yml
```

```yaml
# aws_ec2.yml
plugin: amazon.aws.aws_ec2
regions:
  - ap-northeast-2
keyed_groups:
  - key: tags.Environment
    prefix: env
  - key: instance_type
    prefix: type
compose:
  ansible_host: public_ip_address
```

### Verifying Inventory

```bash
# List hosts
ansible-inventory --list
ansible-inventory --graph

# Specific group
ansible webservers --list-hosts

# Check host variables
ansible-inventory --host web1.example.com
```

---

## 4. Ad-hoc Commands

### Basic Usage

```bash
ansible <pattern> -m <module> -a "<arguments>"

# Examples
ansible all -m ping
ansible webservers -m command -a "uptime"
ansible dbservers -m shell -a "df -h | grep /dev/sda"
```

### Frequently Used Modules

```bash
# ping (connection test)
ansible all -m ping

# command (basic command execution)
ansible all -m command -a "hostname"

# shell (use shell features)
ansible all -m shell -a "cat /etc/os-release | grep VERSION"

# copy (file copy)
ansible all -m copy -a "src=/local/file dest=/remote/path mode=0644"

# file (file/directory management)
ansible all -m file -a "path=/tmp/testdir state=directory mode=0755"

# yum/apt (package management)
ansible webservers -m yum -a "name=httpd state=present"
ansible webservers -m apt -a "name=nginx state=present update_cache=yes"

# service (service management)
ansible webservers -m service -a "name=nginx state=started enabled=yes"

# user (user management)
ansible all -m user -a "name=deploy state=present groups=sudo"

# setup (gather system information)
ansible web1 -m setup
ansible web1 -m setup -a "filter=ansible_distribution*"
```

### Privilege Escalation

```bash
# Use sudo
ansible all -m apt -a "name=vim state=present" --become

# As specific user
ansible all -m command -a "whoami" --become --become-user=postgres
```

### Parallel Execution

```bash
# Adjust number of concurrent hosts
ansible all -m ping -f 50  # 50 concurrent executions

# Sequential execution
ansible all -m ping -f 1
```

---

## 5. Playbook

### Basic Structure

```yaml
# site.yml
---
- name: Configure webservers
  hosts: webservers
  become: yes
  vars:
    http_port: 80

  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
        update_cache: yes

    - name: Start nginx service
      service:
        name: nginx
        state: started
        enabled: yes

- name: Configure databases
  hosts: dbservers
  become: yes

  tasks:
    - name: Install PostgreSQL
      apt:
        name: postgresql
        state: present
```

### Running Playbooks

```bash
# Basic execution
ansible-playbook site.yml

# Specify inventory
ansible-playbook -i inventory/production site.yml

# Dry run (check without changes)
ansible-playbook site.yml --check

# Show differences
ansible-playbook site.yml --diff

# Verbose output
ansible-playbook site.yml -v    # verbose
ansible-playbook site.yml -vvv  # very verbose

# Limit to specific hosts
ansible-playbook site.yml --limit web1.example.com

# Use tags
ansible-playbook site.yml --tags "install,configure"
ansible-playbook site.yml --skip-tags "debug"

# Start at specific task
ansible-playbook site.yml --start-at-task "Start nginx service"
```

### Conditionals (when)

```yaml
tasks:
  - name: Install Apache (RHEL)
    yum:
      name: httpd
      state: present
    when: ansible_os_family == "RedHat"

  - name: Install Apache (Debian)
    apt:
      name: apache2
      state: present
    when: ansible_os_family == "Debian"

  - name: Check if file exists
    stat:
      path: /etc/myapp.conf
    register: config_file

  - name: Create config if not exists
    template:
      src: myapp.conf.j2
      dest: /etc/myapp.conf
    when: not config_file.stat.exists
```

### Loops

```yaml
tasks:
  - name: Install multiple packages
    apt:
      name: "{{ item }}"
      state: present
    loop:
      - nginx
      - vim
      - git
      - curl

  - name: Create users
    user:
      name: "{{ item.name }}"
      groups: "{{ item.groups }}"
      state: present
    loop:
      - { name: 'alice', groups: 'sudo' }
      - { name: 'bob', groups: 'developers' }

  - name: Install packages with apt
    apt:
      name: "{{ packages }}"
      state: present
    vars:
      packages:
        - nginx
        - postgresql
        - redis
```

### Handlers

```yaml
tasks:
  - name: Copy nginx config
    template:
      src: nginx.conf.j2
      dest: /etc/nginx/nginx.conf
    notify: Restart nginx

  - name: Copy site config
    template:
      src: site.conf.j2
      dest: /etc/nginx/sites-available/mysite
    notify:
      - Reload nginx
      - Clear cache

handlers:
  - name: Restart nginx
    service:
      name: nginx
      state: restarted

  - name: Reload nginx
    service:
      name: nginx
      state: reloaded

  - name: Clear cache
    file:
      path: /var/cache/nginx
      state: absent
```

### Blocks

```yaml
tasks:
  - name: Install and configure app
    block:
      - name: Install package
        apt:
          name: myapp
          state: present

      - name: Configure app
        template:
          src: myapp.conf.j2
          dest: /etc/myapp/config.yml

    rescue:
      - name: Rollback on failure
        apt:
          name: myapp
          state: absent

    always:
      - name: Always run cleanup
        file:
          path: /tmp/install_temp
          state: absent

    when: install_myapp | default(true)
    become: yes
```

---

## 6. Variables and Facts

### Variable Definition Locations

```
Priority (low → high):
1. role defaults
2. inventory file vars
3. inventory group_vars
4. inventory host_vars
5. playbook group_vars
6. playbook host_vars
7. host facts
8. play vars
9. play vars_prompt
10. play vars_files
11. role vars
12. block vars
13. task vars
14. include_vars
15. set_facts / registered vars
16. role parameters
17. extra vars (-e)
```

### Variable File Structure

```
project/
├── ansible.cfg
├── inventory/
│   ├── hosts
│   ├── group_vars/
│   │   ├── all.yml
│   │   ├── webservers.yml
│   │   └── dbservers.yml
│   └── host_vars/
│       ├── web1.example.com.yml
│       └── db1.example.com.yml
├── playbooks/
│   └── site.yml
└── roles/
```

### group_vars/host_vars Examples

```yaml
# inventory/group_vars/all.yml
---
ansible_user: deploy
ntp_servers:
  - 0.pool.ntp.org
  - 1.pool.ntp.org

# inventory/group_vars/webservers.yml
---
http_port: 80
nginx_worker_processes: auto

# inventory/host_vars/web1.example.com.yml
---
nginx_worker_processes: 4
custom_config: true
```

### Using Variables in Playbooks

```yaml
---
- name: Configure web servers
  hosts: webservers
  vars:
    app_name: myapp
    app_version: "1.2.3"
    app_env: production

  vars_files:
    - vars/common.yml
    - "vars/{{ ansible_os_family }}.yml"

  tasks:
    - name: Deploy application
      template:
        src: app.conf.j2
        dest: "/etc/{{ app_name }}/config.yml"

    - name: Set runtime variable
      set_fact:
        runtime_var: "{{ ansible_hostname }}-{{ app_version }}"
```

### Facts

```yaml
tasks:
  # Use auto-gathered facts
  - name: Print OS info
    debug:
      msg: "OS: {{ ansible_distribution }} {{ ansible_distribution_version }}"

  # Custom facts (stored on target host)
  # /etc/ansible/facts.d/custom.fact
  - name: Create custom fact
    copy:
      content: |
        [app]
        version=1.2.3
        environment=production
      dest: /etc/ansible/facts.d/custom.fact
      mode: '0644'

  # Refresh facts
  - name: Refresh facts
    setup:
      filter: ansible_local

  - name: Use custom fact
    debug:
      msg: "App version: {{ ansible_local.custom.app.version }}"
```

### Jinja2 Templates

```jinja2
{# templates/nginx.conf.j2 #}
worker_processes {{ nginx_worker_processes | default('auto') }};

events {
    worker_connections {{ nginx_worker_connections | default(1024) }};
}

http {
    server {
        listen {{ http_port }};
        server_name {{ ansible_fqdn }};

        {% for location in nginx_locations | default([]) %}
        location {{ location.path }} {
            proxy_pass {{ location.proxy }};
        }
        {% endfor %}

        {% if enable_ssl | default(false) %}
        ssl_certificate {{ ssl_cert_path }};
        ssl_certificate_key {{ ssl_key_path }};
        {% endif %}
    }
}
```

---

## 7. Roles

### Role Structure

```
roles/
└── webserver/
    ├── defaults/
    │   └── main.yml      # Default variables (low priority)
    ├── files/
    │   └── static.conf   # Static files
    ├── handlers/
    │   └── main.yml      # Handlers
    ├── meta/
    │   └── main.yml      # Dependencies, metadata
    ├── tasks/
    │   └── main.yml      # Tasks
    ├── templates/
    │   └── nginx.conf.j2 # Jinja2 templates
    └── vars/
        └── main.yml      # Variables (high priority)
```

### Creating Roles

```bash
# Create basic structure
ansible-galaxy init roles/webserver
```

### Role Example

```yaml
# roles/webserver/defaults/main.yml
---
nginx_port: 80
nginx_worker_processes: auto
nginx_sites: []
```

```yaml
# roles/webserver/tasks/main.yml
---
- name: Install nginx
  apt:
    name: nginx
    state: present
    update_cache: yes
  tags: install

- name: Configure nginx
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: Restart nginx
  tags: configure

- name: Configure sites
  template:
    src: site.conf.j2
    dest: "/etc/nginx/sites-available/{{ item.name }}"
  loop: "{{ nginx_sites }}"
  notify: Reload nginx
  tags: configure

- name: Enable sites
  file:
    src: "/etc/nginx/sites-available/{{ item.name }}"
    dest: "/etc/nginx/sites-enabled/{{ item.name }}"
    state: link
  loop: "{{ nginx_sites }}"
  notify: Reload nginx
  tags: configure

- name: Start nginx
  service:
    name: nginx
    state: started
    enabled: yes
  tags: service
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
  - role: common
  - role: firewall
    vars:
      firewall_allowed_ports:
        - "{{ nginx_port }}"
```

### Using Roles

```yaml
# site.yml
---
- name: Configure web servers
  hosts: webservers
  become: yes
  roles:
    - common
    - role: webserver
      vars:
        nginx_port: 8080
        nginx_sites:
          - name: mysite
            domain: example.com
            root: /var/www/mysite

- name: Configure databases
  hosts: dbservers
  become: yes
  roles:
    - common
    - postgresql
```

### Ansible Galaxy

```bash
# Search roles
ansible-galaxy search nginx

# Install role
ansible-galaxy install geerlingguy.nginx

# Install collection
ansible-galaxy collection install community.general

# Install from requirements.yml
ansible-galaxy install -r requirements.yml
```

```yaml
# requirements.yml
---
roles:
  - name: geerlingguy.nginx
    version: "3.1.0"
  - name: geerlingguy.postgresql
    version: "3.3.1"

collections:
  - name: community.general
    version: ">=6.0.0"
  - name: amazon.aws
    version: "5.0.0"
```

---

## 8. Ansible Vault

### Basic Vault Usage

```bash
# Create new encrypted file
ansible-vault create secrets.yml

# Encrypt existing file
ansible-vault encrypt vars/secrets.yml

# Decrypt file
ansible-vault decrypt vars/secrets.yml

# Edit file
ansible-vault edit secrets.yml

# View file contents
ansible-vault view secrets.yml

# Change password
ansible-vault rekey secrets.yml
```

### Vault Usage Example

```yaml
# vars/secrets.yml (encrypted)
---
db_password: "SuperSecretPassword123"
api_key: "sk-1234567890abcdef"
```

```yaml
# playbook.yml
---
- name: Deploy application
  hosts: webservers
  vars_files:
    - vars/secrets.yml

  tasks:
    - name: Configure database
      template:
        src: db.conf.j2
        dest: /etc/app/db.conf
      # {{ db_password }} is available
```

### Running Playbooks

```bash
# Password prompt
ansible-playbook site.yml --ask-vault-pass

# Password file
ansible-playbook site.yml --vault-password-file ~/.vault_pass

# Environment variable
export ANSIBLE_VAULT_PASSWORD_FILE=~/.vault_pass
ansible-playbook site.yml
```

### String Encryption

```bash
# Encrypt string only
ansible-vault encrypt_string 'MySecretPassword' --name 'db_password'

# Output (use directly in YAML)
db_password: !vault |
          $ANSIBLE_VAULT;1.1;AES256
          66386439653236...
```

---

## Practice Problems

### Problem 1: Write Inventory

Write the inventory in YAML format for the following servers:
- app1, app2 (webservers group)
- db1 (dbservers group)
- All servers: ansible_user=deploy
- webservers: http_port=8080

### Problem 2: Write Playbook

Write a Playbook that installs and starts nginx:
- Support both RedHat and Debian families
- Use handlers for service restart
- Use tags (install, configure)

### Problem 3: Write Role

Write tasks/main.yml for a Role that installs PostgreSQL:
- Install packages
- Start service
- Create initial database

---

## Answers

### Problem 1 Answer

```yaml
# inventory/hosts.yml
all:
  vars:
    ansible_user: deploy
  children:
    webservers:
      hosts:
        app1:
        app2:
      vars:
        http_port: 8080
    dbservers:
      hosts:
        db1:
```

### Problem 2 Answer

```yaml
---
- name: Install and configure nginx
  hosts: webservers
  become: yes

  tasks:
    - name: Install nginx (RedHat)
      yum:
        name: nginx
        state: present
      when: ansible_os_family == "RedHat"
      tags: install

    - name: Install nginx (Debian)
      apt:
        name: nginx
        state: present
        update_cache: yes
      when: ansible_os_family == "Debian"
      tags: install

    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: Restart nginx
      tags: configure

    - name: Start nginx
      service:
        name: nginx
        state: started
        enabled: yes
      tags: install

  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
```

### Problem 3 Answer

```yaml
# roles/postgresql/tasks/main.yml
---
- name: Install PostgreSQL
  apt:
    name:
      - postgresql
      - postgresql-contrib
      - python3-psycopg2
    state: present
    update_cache: yes

- name: Start PostgreSQL service
  service:
    name: postgresql
    state: started
    enabled: yes

- name: Create application database
  become_user: postgres
  postgresql_db:
    name: "{{ pg_database | default('appdb') }}"
    state: present

- name: Create database user
  become_user: postgres
  postgresql_user:
    name: "{{ pg_user | default('appuser') }}"
    password: "{{ pg_password }}"
    db: "{{ pg_database | default('appdb') }}"
    priv: ALL
    state: present
```

---

---

## References

- [Ansible Documentation](https://docs.ansible.com/)
- [Ansible Galaxy](https://galaxy.ansible.com/)
- [Ansible Best Practices](https://docs.ansible.com/ansible/latest/tips_tricks/ansible_tips_tricks.html)
- `ansible-doc -l`, `ansible-doc <module>`

---

**Previous**: [Virtualization (KVM)](./21_Virtualization_KVM.md) | **Next**: [Advanced Networking](./23_Advanced_Networking.md)
