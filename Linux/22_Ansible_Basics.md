# Ansible 기초

## 학습 목표

이 문서를 통해 다음을 학습합니다:

- Ansible의 개념과 아키텍처
- 인벤토리와 Ad-hoc 명령어
- Playbook 작성 및 실행
- Roles와 변수 관리

**난이도**: ⭐⭐⭐ (중급-고급)

---

## 목차

1. [Ansible 개요](#1-ansible-개요)
2. [설치 및 설정](#2-설치-및-설정)
3. [인벤토리](#3-인벤토리)
4. [Ad-hoc 명령어](#4-ad-hoc-명령어)
5. [Playbook](#5-playbook)
6. [변수와 팩트](#6-변수와-팩트)
7. [Roles](#7-roles)
8. [Ansible Vault](#8-ansible-vault)

---

## 1. Ansible 개요

### Ansible이란?

Ansible은 에이전트 없이 SSH로 작동하는 IT 자동화 도구입니다.

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

### Ansible vs 다른 도구

| 특성 | Ansible | Puppet | Chef |
|------|---------|--------|------|
| 에이전트 | 불필요 | 필요 | 필요 |
| 언어 | YAML | Ruby DSL | Ruby |
| Push/Pull | Push | Pull | Pull |
| 학습 곡선 | 낮음 | 중간 | 높음 |

---

## 2. 설치 및 설정

### Ansible 설치

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ansible

# RHEL/CentOS
sudo yum install epel-release
sudo yum install ansible

# pip로 설치 (권장)
pip3 install ansible

# 버전 확인
ansible --version
```

### SSH 키 설정

```bash
# SSH 키 생성
ssh-keygen -t ed25519 -f ~/.ssh/ansible_key

# 대상 서버에 키 복사
ssh-copy-id -i ~/.ssh/ansible_key.pub user@target-server

# SSH 연결 테스트
ssh -i ~/.ssh/ansible_key user@target-server
```

### ansible.cfg 설정

```bash
# /etc/ansible/ansible.cfg (전역)
# ~/.ansible.cfg (사용자)
# ./ansible.cfg (프로젝트)
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

## 3. 인벤토리

### INI 형식

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

# 그룹의 그룹
[production:children]
webservers
dbservers
loadbalancers

# 그룹 변수
[webservers:vars]
http_port=80
max_clients=200

[all:vars]
ansible_python_interpreter=/usr/bin/python3
```

### YAML 형식

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

### 동적 인벤토리

```bash
# AWS EC2 동적 인벤토리
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

### 인벤토리 확인

```bash
# 호스트 목록
ansible-inventory --list
ansible-inventory --graph

# 특정 그룹
ansible webservers --list-hosts

# 호스트 변수 확인
ansible-inventory --host web1.example.com
```

---

## 4. Ad-hoc 명령어

### 기본 사용법

```bash
ansible <pattern> -m <module> -a "<arguments>"

# 예시
ansible all -m ping
ansible webservers -m command -a "uptime"
ansible dbservers -m shell -a "df -h | grep /dev/sda"
```

### 자주 사용하는 모듈

```bash
# ping (연결 테스트)
ansible all -m ping

# command (기본 명령 실행)
ansible all -m command -a "hostname"

# shell (쉘 기능 사용)
ansible all -m shell -a "cat /etc/os-release | grep VERSION"

# copy (파일 복사)
ansible all -m copy -a "src=/local/file dest=/remote/path mode=0644"

# file (파일/디렉토리 관리)
ansible all -m file -a "path=/tmp/testdir state=directory mode=0755"

# yum/apt (패키지 관리)
ansible webservers -m yum -a "name=httpd state=present"
ansible webservers -m apt -a "name=nginx state=present update_cache=yes"

# service (서비스 관리)
ansible webservers -m service -a "name=nginx state=started enabled=yes"

# user (사용자 관리)
ansible all -m user -a "name=deploy state=present groups=sudo"

# setup (시스템 정보 수집)
ansible web1 -m setup
ansible web1 -m setup -a "filter=ansible_distribution*"
```

### 권한 상승

```bash
# sudo 사용
ansible all -m apt -a "name=vim state=present" --become

# 특정 사용자로
ansible all -m command -a "whoami" --become --become-user=postgres
```

### 병렬 실행

```bash
# 동시 실행 호스트 수 조절
ansible all -m ping -f 50  # 50개 동시 실행

# 순차 실행
ansible all -m ping -f 1
```

---

## 5. Playbook

### 기본 구조

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

### Playbook 실행

```bash
# 기본 실행
ansible-playbook site.yml

# 인벤토리 지정
ansible-playbook -i inventory/production site.yml

# 드라이런 (변경 없이 확인)
ansible-playbook site.yml --check

# 차이점 표시
ansible-playbook site.yml --diff

# 상세 출력
ansible-playbook site.yml -v    # 상세
ansible-playbook site.yml -vvv  # 매우 상세

# 특정 호스트만
ansible-playbook site.yml --limit web1.example.com

# 태그 사용
ansible-playbook site.yml --tags "install,configure"
ansible-playbook site.yml --skip-tags "debug"

# 시작 태스크 지정
ansible-playbook site.yml --start-at-task "Start nginx service"
```

### 조건문 (when)

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

### 반복문 (loop)

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

### 핸들러 (handlers)

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

### 블록 (block)

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

## 6. 변수와 팩트

### 변수 정의 위치

```
우선순위 (낮음 → 높음):
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

### 변수 파일 구조

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

### group_vars/host_vars 예시

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

### Playbook에서 변수 사용

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

### 팩트 (Facts)

```yaml
tasks:
  # 자동 수집된 팩트 사용
  - name: Print OS info
    debug:
      msg: "OS: {{ ansible_distribution }} {{ ansible_distribution_version }}"

  # 커스텀 팩트 (대상 호스트에 저장)
  # /etc/ansible/facts.d/custom.fact
  - name: Create custom fact
    copy:
      content: |
        [app]
        version=1.2.3
        environment=production
      dest: /etc/ansible/facts.d/custom.fact
      mode: '0644'

  # 팩트 새로고침
  - name: Refresh facts
    setup:
      filter: ansible_local

  - name: Use custom fact
    debug:
      msg: "App version: {{ ansible_local.custom.app.version }}"
```

### Jinja2 템플릿

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

### Role 구조

```
roles/
└── webserver/
    ├── defaults/
    │   └── main.yml      # 기본 변수 (낮은 우선순위)
    ├── files/
    │   └── static.conf   # 정적 파일
    ├── handlers/
    │   └── main.yml      # 핸들러
    ├── meta/
    │   └── main.yml      # 의존성, 메타데이터
    ├── tasks/
    │   └── main.yml      # 태스크
    ├── templates/
    │   └── nginx.conf.j2 # Jinja2 템플릿
    └── vars/
        └── main.yml      # 변수 (높은 우선순위)
```

### Role 생성

```bash
# 기본 구조 생성
ansible-galaxy init roles/webserver
```

### Role 예시

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

### Role 사용

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
# Role 검색
ansible-galaxy search nginx

# Role 설치
ansible-galaxy install geerlingguy.nginx

# 컬렉션 설치
ansible-galaxy collection install community.general

# requirements.yml로 설치
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

### Vault 기본 사용

```bash
# 새 암호화 파일 생성
ansible-vault create secrets.yml

# 기존 파일 암호화
ansible-vault encrypt vars/secrets.yml

# 파일 복호화
ansible-vault decrypt vars/secrets.yml

# 파일 편집
ansible-vault edit secrets.yml

# 파일 내용 보기
ansible-vault view secrets.yml

# 비밀번호 변경
ansible-vault rekey secrets.yml
```

### Vault 사용 예시

```yaml
# vars/secrets.yml (암호화됨)
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
      # {{ db_password }} 사용 가능
```

### Playbook 실행

```bash
# 비밀번호 프롬프트
ansible-playbook site.yml --ask-vault-pass

# 비밀번호 파일 사용
ansible-playbook site.yml --vault-password-file ~/.vault_pass

# 환경 변수
export ANSIBLE_VAULT_PASSWORD_FILE=~/.vault_pass
ansible-playbook site.yml
```

### 문자열 암호화

```bash
# 문자열만 암호화
ansible-vault encrypt_string 'MySecretPassword' --name 'db_password'

# 출력 (YAML에 직접 사용)
db_password: !vault |
          $ANSIBLE_VAULT;1.1;AES256
          66386439653236...
```

---

## 연습 문제

### 문제 1: 인벤토리 작성

다음 서버들의 인벤토리를 YAML 형식으로 작성하세요:
- app1, app2 (webservers 그룹)
- db1 (dbservers 그룹)
- 모든 서버: ansible_user=deploy
- webservers: http_port=8080

### 문제 2: Playbook 작성

nginx를 설치하고 시작하는 Playbook을 작성하세요:
- RedHat과 Debian 계열 모두 지원
- 핸들러로 서비스 재시작
- 태그 사용 (install, configure)

### 문제 3: Role 작성

PostgreSQL을 설치하는 Role의 tasks/main.yml을 작성하세요:
- 패키지 설치
- 서비스 시작
- 초기 데이터베이스 생성

---

## 정답

### 문제 1 정답

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

### 문제 2 정답

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

### 문제 3 정답

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

## 다음 단계

- [23_Advanced_Networking.md](./23_Advanced_Networking.md) - VLAN, bonding, iptables/nftables

---

## 참고 자료

- [Ansible Documentation](https://docs.ansible.com/)
- [Ansible Galaxy](https://galaxy.ansible.com/)
- [Ansible Best Practices](https://docs.ansible.com/ansible/latest/tips_tricks/ansible_tips_tricks.html)
- `ansible-doc -l`, `ansible-doc <module>`
