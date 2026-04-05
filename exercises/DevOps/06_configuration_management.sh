#!/bin/bash
# Exercises for Lesson 06: Configuration Management
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Ansible Playbook Design ===
# Problem: Write an Ansible playbook that provisions a web server with
# Nginx, deploys an application, and configures firewall rules.
exercise_1() {
    echo "=== Exercise 1: Ansible Playbook Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# webserver.yml — Ansible playbook for web server provisioning
---
- name: Provision and configure web servers
  hosts: webservers
  become: true
  vars:
    app_user: "webapp"
    app_dir: "/opt/myapp"
    nginx_port: 80
    app_port: 8080

  pre_tasks:
    - name: Update apt cache
      ansible.builtin.apt:
        update_cache: true
        cache_valid_time: 3600

  tasks:
    - name: Install required packages
      ansible.builtin.apt:
        name:
          - nginx
          - python3
          - python3-pip
          - python3-venv
          - ufw
        state: present

    - name: Create application user
      ansible.builtin.user:
        name: "{{ app_user }}"
        shell: /bin/bash
        system: true
        create_home: false

    - name: Create application directory
      ansible.builtin.file:
        path: "{{ app_dir }}"
        state: directory
        owner: "{{ app_user }}"
        mode: "0755"

    - name: Deploy application code
      ansible.builtin.git:
        repo: "https://github.com/myorg/myapp.git"
        dest: "{{ app_dir }}"
        version: "{{ app_version | default('main') }}"
      notify: Restart application

    - name: Install Python dependencies
      ansible.builtin.pip:
        requirements: "{{ app_dir }}/requirements.txt"
        virtualenv: "{{ app_dir }}/venv"

    - name: Deploy Nginx configuration
      ansible.builtin.template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/myapp.conf
        mode: "0644"
      notify: Reload Nginx

    - name: Enable Nginx site
      ansible.builtin.file:
        src: /etc/nginx/sites-available/myapp.conf
        dest: /etc/nginx/sites-enabled/myapp.conf
        state: link
      notify: Reload Nginx

    - name: Configure UFW firewall
      community.general.ufw:
        rule: allow
        port: "{{ item }}"
        proto: tcp
      loop:
        - "22"
        - "80"
        - "443"

    - name: Enable UFW
      community.general.ufw:
        state: enabled
        policy: deny

  handlers:
    - name: Restart application
      ansible.builtin.systemd:
        name: myapp
        state: restarted
        daemon_reload: true

    - name: Reload Nginx
      ansible.builtin.systemd:
        name: nginx
        state: reloaded
SOLUTION
}

# === Exercise 2: Idempotency ===
# Problem: Explain idempotency and demonstrate how Ansible modules
# achieve it compared to shell scripts.
exercise_2() {
    echo "=== Exercise 2: Idempotency ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Idempotency: Running the same operation multiple times produces the
# same result as running it once. This is THE fundamental principle
# of configuration management.

# BAD (not idempotent) — shell script:
# echo "export PATH=$PATH:/opt/myapp/bin" >> ~/.bashrc
# Problem: Running twice adds the line TWICE. Running 100 times = 100 lines.

# GOOD (idempotent) — Ansible lineinfile:
# - name: Add app to PATH
#   ansible.builtin.lineinfile:
#     path: ~/.bashrc
#     line: 'export PATH=$PATH:/opt/myapp/bin'
#     state: present
# Running 100 times: still exactly ONE line in .bashrc.

# Common idempotency patterns in Ansible:

# 1. Package installation (state: present, not state: latest)
# - ansible.builtin.apt:
#     name: nginx
#     state: present     # Install if missing, skip if present

# 2. File management (desired state declaration)
# - ansible.builtin.file:
#     path: /opt/myapp
#     state: directory   # Create if missing, no-op if exists
#     mode: "0755"       # Correct permissions if wrong

# 3. Service management (ensure running)
# - ansible.builtin.systemd:
#     name: nginx
#     state: started     # Start if stopped, no-op if running
#     enabled: true      # Enable on boot if not enabled

# 4. Template management (content-based change detection)
# - ansible.builtin.template:
#     src: nginx.conf.j2
#     dest: /etc/nginx/nginx.conf
#   # Only overwrites if content differs (checksum comparison)
#   # notify: handler only runs if file actually changed

# Testing idempotency:
# Run the playbook TWICE. On the second run, every task should
# report "ok" (green), not "changed" (yellow).
# If any task shows "changed" on the second run, it's not idempotent.

# ansible-playbook webserver.yml          # First run: many "changed"
# ansible-playbook webserver.yml          # Second run: all "ok"
# ansible-playbook webserver.yml --check  # Dry-run mode (no changes)
SOLUTION
}

# === Exercise 3: Ansible Inventory Management ===
# Problem: Design a dynamic inventory structure for a multi-environment
# setup with host variables and group variables.
exercise_3() {
    echo "=== Exercise 3: Ansible Inventory Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Directory-based inventory layout:
# inventory/
#   production/
#     hosts.yml              # Host definitions
#     group_vars/
#       all.yml              # Variables for ALL hosts
#       webservers.yml       # Variables for webservers group
#       dbservers.yml        # Variables for dbservers group
#     host_vars/
#       web-01.yml           # Variables specific to web-01
#   staging/
#     hosts.yml
#     group_vars/
#       all.yml

# --- inventory/production/hosts.yml ---
all:
  children:
    webservers:
      hosts:
        web-01:
          ansible_host: 10.0.1.10
        web-02:
          ansible_host: 10.0.1.11
    appservers:
      hosts:
        app-01:
          ansible_host: 10.0.2.10
        app-02:
          ansible_host: 10.0.2.11
    dbservers:
      hosts:
        db-primary:
          ansible_host: 10.0.3.10
          pg_role: primary
        db-replica:
          ansible_host: 10.0.3.11
          pg_role: replica

# --- inventory/production/group_vars/all.yml ---
# ansible_user: ubuntu
# ansible_become: true
# ntp_server: time.google.com
# environment: production

# --- inventory/production/group_vars/webservers.yml ---
# nginx_worker_processes: auto
# ssl_cert_path: /etc/ssl/certs/prod.pem

# --- inventory/production/group_vars/dbservers.yml ---
# postgresql_version: 16
# pg_max_connections: 200

# Usage:
# ansible-playbook -i inventory/production site.yml
# ansible-playbook -i inventory/staging site.yml

# Variable precedence (lowest to highest):
# 1. group_vars/all
# 2. group_vars/<group>
# 3. host_vars/<host>
# 4. Playbook vars
# 5. Extra vars (-e "key=value")     <-- highest priority
SOLUTION
}

# === Exercise 4: Ansible Roles ===
# Problem: Create an Ansible role for installing and configuring PostgreSQL
# with proper directory structure.
exercise_4() {
    echo "=== Exercise 4: Ansible Roles ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Role directory structure:
# roles/
#   postgresql/
#     defaults/main.yml    # Default variables (lowest precedence)
#     tasks/main.yml       # Task definitions
#     handlers/main.yml    # Handler definitions
#     templates/           # Jinja2 templates
#       postgresql.conf.j2
#       pg_hba.conf.j2
#     files/               # Static files
#     meta/main.yml        # Role metadata and dependencies

# --- roles/postgresql/defaults/main.yml ---
postgresql_version: 16
postgresql_port: 5432
postgresql_max_connections: 100
postgresql_shared_buffers: "256MB"
postgresql_data_dir: "/var/lib/postgresql/{{ postgresql_version }}/main"
postgresql_hba_entries:
  - { type: local, database: all, user: postgres, method: peer }
  - { type: host, database: all, user: all, address: "127.0.0.1/32", method: scram-sha-256 }

# --- roles/postgresql/tasks/main.yml ---
# - name: Install PostgreSQL
#   ansible.builtin.apt:
#     name:
#       - "postgresql-{{ postgresql_version }}"
#       - "postgresql-client-{{ postgresql_version }}"
#     state: present
#
# - name: Configure postgresql.conf
#   ansible.builtin.template:
#     src: postgresql.conf.j2
#     dest: "{{ postgresql_data_dir }}/postgresql.conf"
#   notify: Restart PostgreSQL
#
# - name: Configure pg_hba.conf
#   ansible.builtin.template:
#     src: pg_hba.conf.j2
#     dest: "{{ postgresql_data_dir }}/pg_hba.conf"
#   notify: Reload PostgreSQL
#
# - name: Ensure PostgreSQL is running
#   ansible.builtin.systemd:
#     name: postgresql
#     state: started
#     enabled: true

# --- roles/postgresql/handlers/main.yml ---
# - name: Restart PostgreSQL
#   ansible.builtin.systemd:
#     name: postgresql
#     state: restarted
#
# - name: Reload PostgreSQL
#   ansible.builtin.systemd:
#     name: postgresql
#     state: reloaded

# Using the role in a playbook:
# - hosts: dbservers
#   roles:
#     - role: postgresql
#       postgresql_max_connections: 300
#       postgresql_shared_buffers: "4GB"

# Create with: ansible-galaxy role init roles/postgresql
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 06: Configuration Management"
echo "==========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
