# 레슨 7: 구성 관리(Configuration Management)

**이전**: [Terraform 심화](./06_Terraform_Advanced.md) | **다음**: [컨테이너 오케스트레이션 운영](./08_Container_Orchestration_Operations.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 멱등성(idempotency), 수렴(convergence), 원하는 상태(desired state)를 포함한 구성 관리 원칙을 설명할 수 있습니다
2. 태스크, 핸들러, 변수, 조건문을 사용하여 Ansible 플레이북을 작성할 수 있습니다
3. 역할(role), 인벤토리 파일, 그룹 변수를 사용하여 Ansible 코드를 체계적으로 구성할 수 있습니다
4. Jinja2 템플릿을 사용하여 동적 구성 파일을 생성할 수 있습니다
5. Ansible Vault를 사용하여 민감한 데이터를 관리할 수 있습니다
6. 완전한 IaC 전략에서 Ansible이 Terraform을 어떻게 보완하는지 이해할 수 있습니다

---

구성 관리(Configuration Management)는 시스템의 구성 변경을 체계적으로 처리하여 시간이 지나도 무결성을 유지하는 실천 방법입니다. Terraform이 인프라를 프로비저닝(VM, 네트워크, 데이터베이스 생성)하는 반면, Ansible과 같은 구성 관리 도구는 해당 인프라를 구성(패키지 설치, 애플리케이션 배포, 서비스 관리)합니다. 이 둘은 함께 완전한 Infrastructure as Code 전략을 구성합니다: Terraform이 서버를 생성하고, Ansible이 서버를 구성합니다.

> **비유 -- 가구가 갖춰진 아파트**: Terraform은 아파트 건물(VPC, 서브넷, EC2 인스턴스)을 짓는 부동산 개발자와 같습니다. Ansible은 각 아파트에 가구를 배치(nginx 설치, SSL 구성, 애플리케이션 배포)하는 인테리어 디자이너입니다. 둘 다 필요합니다: 가구 없는 건물은 사용할 수 없고, 건물 없는 가구는 놓을 곳이 없습니다.

## 1. 구성 관리 원칙

### 핵심 개념

| 개념 | 설명 |
|------|------|
| **멱등성(Idempotency)** | 동일한 작업을 여러 번 실행해도 같은 결과를 생성합니다 |
| **수렴(Convergence)** | 시스템이 원하는 상태로 자동으로 이동합니다 |
| **원하는 상태(Desired state)** | 어떻게 도달할지가 아니라 시스템이 어떤 모습이어야 하는지를 선언합니다 |
| **에이전트리스(Agentless)** | 관리 대상 노드에 소프트웨어를 설치할 필요가 없습니다 (Ansible은 SSH 사용) |
| **Push vs Pull** | Push: 컨트롤러가 구성을 전송 (Ansible). Pull: 노드가 구성을 가져옴 (Puppet, Chef) |

### 멱등성 예시

```bash
# 멱등성이 아닌 경우 -- 두 번 실행하면 중복 항목이 생성됩니다:
echo "export PATH=/opt/myapp/bin:$PATH" >> ~/.bashrc

# 멱등성을 보장하는 경우 -- Ansible이 먼저 확인하고, 없을 때만 추가합니다:
# - name: Add myapp to PATH
#   lineinfile:
#     path: ~/.bashrc
#     line: "export PATH=/opt/myapp/bin:$PATH"
#     state: present
```

### 구성 관리 vs IaC 프로비저닝

```
Terraform (프로비저닝):             Ansible (구성):
─────────────────────────           ────────────────────────
EC2 인스턴스 생성                   패키지 설치
VPC 및 서브넷 생성                  nginx/Apache 구성
RDS 데이터베이스 생성               애플리케이션 코드 배포
S3 버킷 생성                       시스템 사용자 관리
로드밸런서 생성                    cron 작업 설정
DNS 레코드 관리                    방화벽(iptables) 구성
IAM 역할 생성                     SSL 인증서 관리

함께 사용: Terraform이 VM을 생성 → Ansible이 구성
```

---

## 2. Ansible 아키텍처

### Ansible 작동 방식

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
  1. SSH를 통해 노드에 연결합니다
  2. 작은 Python 프로그램(모듈)을 노드에 복사합니다
  3. 모듈을 실행합니다
  4. 임시 파일을 제거합니다
  5. 결과를 컨트롤 노드에 보고합니다
```

### 설치

```bash
# Ansible 설치 (Python 기반)
pip install ansible

# 또는 시스템 패키지 관리자를 통해 설치
# macOS
brew install ansible

# Ubuntu
sudo apt update
sudo apt install ansible

# 확인
ansible --version
# ansible [core 2.16.x]
```

---

## 3. 인벤토리(Inventory)

인벤토리는 Ansible이 관리하는 호스트와 연결 방법을 정의합니다.

### INI 형식

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

### YAML 형식

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

### 동적 인벤토리(Dynamic Inventory)

```bash
# AWS EC2 동적 인벤토리 -- AWS에서 자동으로 호스트를 검색합니다
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
# 검색된 호스트 목록 조회
ansible-inventory -i inventory/aws_ec2.yml --list

# 플레이북에서 사용
ansible-playbook -i inventory/aws_ec2.yml playbook.yml
```

---

## 4. 플레이북(Playbooks)

플레이북은 시스템의 원하는 상태를 정의하는 YAML 파일입니다.

### 기본 플레이북

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

### 플레이북 실행

```bash
# 인벤토리에 대해 플레이북 실행
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml

# 드라이 런 (체크 모드 -- 변경 없이 무엇이 변경될지 표시)
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml --check

# 파일 변경 사항의 diff 표시
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml --check --diff

# 특정 호스트로 제한
ansible-playbook -i inventory/hosts.yml playbooks/setup-webserver.yml --limit web-01

# 추가 변수
ansible-playbook playbooks/deploy.yml -e "version=1.2.3 environment=production"

# 디버깅을 위한 상세 출력
ansible-playbook playbooks/setup-webserver.yml -vvv
```

---

## 5. 태스크와 모듈(Tasks and Modules)

### 주요 모듈

```yaml
# 파일 관리
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

# 패키지 관리
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

# 서비스 관리
- name: Ensure service is running
  service:
    name: nginx
    state: started
    enabled: true

# 사용자 관리
- name: Create application user
  user:
    name: deploy
    shell: /bin/bash
    groups: docker
    append: true                       # Add to group without removing from others

# 명령 실행
- name: Run database migration
  command: python manage.py migrate
  args:
    chdir: /opt/myapp
  register: migration_result          # Capture output
  changed_when: "'No migrations to apply' not in migration_result.stdout"

# Git 체크아웃
- name: Clone application repository
  git:
    repo: https://github.com/org/myapp.git
    dest: /opt/myapp
    version: "{{ app_version }}"
    force: true

# Cron 작업
- name: Schedule log rotation
  cron:
    name: "Rotate app logs"
    minute: "0"
    hour: "2"
    job: "/usr/sbin/logrotate /etc/logrotate.d/myapp"
```

### 조건문(Conditionals)

```yaml
tasks:
  # Ubuntu에서만 실행
  - name: Install packages (Debian/Ubuntu)
    apt:
      name: nginx
      state: present
    when: ansible_os_family == "Debian"

  # CentOS/RHEL에서만 실행
  - name: Install packages (RedHat/CentOS)
    yum:
      name: nginx
      state: present
    when: ansible_os_family == "RedHat"

  # 변수 기반 실행
  - name: Configure SSL
    template:
      src: ssl.conf.j2
      dest: /etc/nginx/ssl.conf
    when: enable_ssl | default(false)

  # 다중 조건
  - name: Deploy to production
    command: deploy.sh
    when:
      - environment == "production"
      - deploy_enabled | bool
```

### 반복문(Loops)

```yaml
tasks:
  # 단순 반복문
  - name: Create multiple users
    user:
      name: "{{ item }}"
      state: present
      groups: developers
    loop:
      - alice
      - bob
      - carol

  # 딕셔너리를 사용한 반복문
  - name: Create users with specific shells
    user:
      name: "{{ item.name }}"
      shell: "{{ item.shell }}"
      groups: "{{ item.groups }}"
    loop:
      - { name: alice, shell: /bin/bash, groups: developers }
      - { name: bob, shell: /bin/zsh, groups: "developers,docker" }

  # 등록된 결과를 사용한 반복문
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

## 6. Jinja2 템플릿

Jinja2 템플릿은 동적 구성 파일을 생성합니다.

### 템플릿 예시

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

### Jinja2 필터

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

## 7. 역할(Roles)

역할은 Ansible 코드를 재사용 가능한 컴포넌트로 구성하는 표준화된 방법입니다.

### 역할 디렉토리 구조

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

### 역할 구현

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

### 플레이북에서 역할 사용

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

Ansible Vault는 민감한 데이터(비밀번호, API 키, 인증서)를 암호화합니다.

### Vault 명령어

```bash
# 암호화된 파일 생성
ansible-vault create secrets.yml

# 기존 파일 암호화
ansible-vault encrypt vars/production.yml

# 파일 복호화 (보기 또는 편집)
ansible-vault decrypt vars/production.yml
ansible-vault view vars/production.yml
ansible-vault edit vars/production.yml

# 단일 문자열 암호화
ansible-vault encrypt_string 'SuperSecretPassword' --name 'db_password'
# Output:
# db_password: !vault |
#   $ANSIBLE_VAULT;1.1;AES256
#   61626364656667...

# 키 재설정 (암호화 비밀번호 변경)
ansible-vault rekey secrets.yml
```

### 플레이북에서 Vault 사용

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
# Vault 비밀번호를 사용하여 플레이북 실행
ansible-playbook playbooks/deploy.yml --ask-vault-pass

# 또는 비밀번호 파일 사용
ansible-playbook playbooks/deploy.yml --vault-password-file ~/.vault_pass

# 또는 환경 변수 설정
export ANSIBLE_VAULT_PASSWORD_FILE=~/.vault_pass
ansible-playbook playbooks/deploy.yml
```

### 다중 비밀번호를 사용한 Vault

```bash
# 서로 다른 비밀번호에 대해 vault ID를 사용 (dev vs production)
ansible-vault encrypt --vault-id dev@prompt vars/dev-secrets.yml
ansible-vault encrypt --vault-id prod@prompt vars/prod-secrets.yml

# 다중 vault 비밀번호로 실행
ansible-playbook playbooks/site.yml \
  --vault-id dev@~/.vault_pass_dev \
  --vault-id prod@~/.vault_pass_prod
```

---

## 9. 그룹 변수와 호스트 변수

### 디렉토리 구조

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

### 변수 우선순위 (낮은 순서에서 높은 순서)

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

AWX(오픈소스)와 Ansible Tower(Red Hat 상용)는 Ansible을 위한 웹 기반 UI, REST API, RBAC를 제공합니다.

### 주요 기능

```
AWX / Ansible Tower:
────────────────────
✓ 플레이북 실행을 위한 웹 UI
✓ 역할 기반 접근 제어 (RBAC)
✓ 작업 스케줄링 및 알림
✓ 자격 증명 관리 (암호화)
✓ 인벤토리 관리 (동적)
✓ 감사 추적 및 로깅
✓ 통합을 위한 REST API
✓ 워크플로 템플릿 (플레이북 체인)
```

```bash
# Kubernetes에 AWX 배포
# AWX 오퍼레이터 클론
git clone https://github.com/ansible/awx-operator.git
cd awx-operator

# 오퍼레이터 배포
make deploy

# AWX 인스턴스 생성
kubectl apply -f awx-demo.yml
```

---

## 연습 문제

### 연습 문제 1: 웹 서버 플레이북

웹 서버를 구성하는 Ansible 플레이북을 작성하십시오:
1. nginx와 certbot을 설치합니다
2. SSH 키 접근이 가능한 애플리케이션 사용자를 생성합니다
3. Jinja2 템플릿을 사용하여 사용자 정의 nginx 구성을 배포합니다
4. 정적 HTML 파일을 웹 루트에 복사합니다
5. nginx가 실행 중이고 활성화되어 있는지 확인합니다
6. 구성이 변경될 때만 nginx를 재로드하는 핸들러를 사용합니다
7. 먼저 체크 모드로 실행하고 출력을 확인합니다

### 연습 문제 2: 재사용 가능한 역할 생성

`app_deploy`라는 Ansible 역할을 생성하십시오:
1. 애플리케이션 디렉토리 구조를 생성합니다 (/opt/app/{releases,shared,current})
2. Git 저장소를 새 릴리스 디렉토리에 클론합니다
3. `current`에서 최신 릴리스로의 심볼릭 링크를 생성합니다
4. virtualenv에 Python 의존성을 설치합니다
5. 템플릿에서 systemd 서비스 파일을 복사합니다
6. 서비스를 재시작합니다
7. 최근 5개의 릴리스만 유지합니다 (이전 릴리스 정리)

### 연습 문제 3: 다중 환경 배포

다음과 같은 Ansible 프로젝트를 설정하십시오:
1. dev와 production 환경에 대한 인벤토리 (별도 파일)
2. 환경 간에 다른 그룹 변수 (데이터베이스 호스트, 디버그 모드, 레플리카 수)
3. 각 환경에 대한 vault 암호화된 시크릿 파일
4. 인벤토리에 따라 올바른 환경에 애플리케이션을 배포하는 플레이북
5. dev vs production에 대한 실행 시연

### 연습 문제 4: Ansible + Terraform 통합

Terraform과 Ansible을 함께 사용하는 워크플로를 설계하십시오:
1. Terraform이 2개의 EC2 인스턴스를 프로비저닝하고 IP를 출력합니다
2. Terraform 출력에서 Ansible 동적 인벤토리를 생성합니다
3. Ansible이 인스턴스를 구성합니다 (패키지 설치, 앱 배포)
4. 두 도구를 순차적으로 실행하는 스크립트 또는 CI 파이프라인을 작성합니다
5. Terraform이 새 인스턴스를 추가하는 경우를 어떻게 처리할지 설명합니다

---

**이전**: [Terraform 심화](./06_Terraform_Advanced.md) | [개요](00_Overview.md) | **다음**: [컨테이너 오케스트레이션 운영](./08_Container_Orchestration_Operations.md)

**License**: CC BY-NC 4.0
