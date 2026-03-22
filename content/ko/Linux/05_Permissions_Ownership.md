# 권한과 소유권

**이전**: [텍스트 처리](./04_Text_Processing.md) | **다음**: [사용자/그룹 관리](./06_User_Group_Management.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 기호 표기법(symbolic notation)과 숫자 표기법(numeric notation) 모두로 리눅스 파일 권한(permission)을 읽고 해석한다
2. chmod를 사용해 숫자 모드와 기호 모드로 파일 권한을 수정한다
3. chown과 chgrp으로 파일 소유자(owner)와 그룹(group)을 변경한다
4. SUID, SGID, 스티키 비트(Sticky Bit)의 목적과 동작을 설명한다
5. umask를 사용해 기본 권한(default permission)을 설정한다
6. 실제 시나리오(웹 서버, SSH 키, 공유 디렉토리)에 올바른 권한을 적용한다
7. 흔한 "Permission denied" 오류를 진단하고 해결한다

---

리눅스의 보안은 권한에서 시작됩니다. 모든 파일과 디렉토리에는 소유자, 그룹, 그리고 누가 읽고, 쓰고, 실행할 수 있는지를 결정하는 접근 규칙이 있습니다. 잘못 설정된 권한은 정당한 사용자를 자신의 파일에서 잠가버리거나, 더 심각하게는 민감한 데이터를 시스템 전체에 노출시킬 수 있습니다. 권한을 이해하는 것은 일상 업무와 시스템 강화(hardening) 모두에 필수적입니다.

> **비유: 호텔 키카드** -- 리눅스 권한은 호텔 키카드처럼 작동합니다. 소유자는 방을 예약한 투숙객(전체 접근), 그룹은 청소 직원(공용 공간에 제한적 접근), 기타 사용자는 낯선 사람(명시적으로 허용되지 않으면 접근 불가)입니다. `chmod` 명령어는 다른 접근 수준의 키카드를 새로 발급해주는 프런트 데스크입니다.

## 1. 파일 권한 이해

리눅스의 모든 파일은 권한(permission)을 가집니다.

### 권한 구성

```
-rw-r--r-- 1 ubuntu ubuntu 1234 Jan 23 10:00 file.txt
│└─┬──┘└─┬──┘└─┬──┘
│  │     │     │
│  │     │     └── 기타 사용자 (Others)
│  │     └── 그룹 (Group)
│  └── 소유자 (Owner)
└── 파일 타입
```

### 권한 종류

| 권한 | 문자 | 숫자 | 파일 | 디렉토리 |
|------|------|------|------|----------|
| 읽기 | r | 4 | 내용 읽기 | 목록 보기 (ls) |
| 쓰기 | w | 2 | 내용 수정 | 파일 생성/삭제 |
| 실행 | x | 1 | 실행 | 진입 (cd) |
| 없음 | - | 0 | 권한 없음 | 권한 없음 |

### 권한 읽는 법

```
rwxr-xr--
│││││││││
││││││││└── Others: 읽기만 (r--)
│││││└┴┴── Group: 읽기+실행 (r-x)
└┴┴┴───── Owner: 모든 권한 (rwx)
```

숫자 변환:
```
rwx = 4+2+1 = 7
r-x = 4+0+1 = 5
r-- = 4+0+0 = 4

→ 754
```

---

## 2. chmod - 권한 변경

### 숫자 모드

```bash
# 문법
chmod [권한숫자] 파일

# 예시
chmod 755 script.sh      # rwxr-xr-x
chmod 644 file.txt       # rw-r--r--
chmod 600 secret.key     # rw-------
chmod 777 public/        # rwxrwxrwx (권장하지 않음)
```

### 자주 사용하는 권한

| 권한 | 숫자 | 용도 |
|------|------|------|
| rwxr-xr-x | 755 | 실행 파일, 디렉토리 |
| rw-r--r-- | 644 | 일반 파일 |
| rw------- | 600 | 민감한 파일 (키, 설정) |
| rwx------ | 700 | 개인 디렉토리 |
| rwxrwxr-x | 775 | 그룹 공유 디렉토리 |

### 심볼릭 모드

```bash
# 문법
chmod [대상][연산자][권한] 파일

# 대상: u(소유자), g(그룹), o(기타), a(전체)
# 연산자: +(추가), -(제거), =(설정)
# 권한: r, w, x

# 예시
chmod u+x script.sh      # 소유자에게 실행 권한 추가
chmod g-w file.txt       # 그룹에서 쓰기 권한 제거
chmod o=r file.txt       # 기타 사용자 읽기만 허용
chmod a+r file.txt       # 모두에게 읽기 권한 추가

# 여러 권한
chmod u+rwx,g+rx,o+r file.txt
chmod ug+x script.sh

# 재귀적 적용
chmod -R 755 directory/
```

### 실행 권한 예시

```bash
# 스크립트에 실행 권한 부여
chmod +x script.sh
./script.sh

# 또는
chmod u+x script.sh
```

---

## 3. chown - 소유자 변경

```bash
# 문법
chown [옵션] 소유자[:그룹] 파일

# 소유자만 변경
chown newuser file.txt

# 소유자와 그룹 변경
chown newuser:newgroup file.txt

# 그룹만 변경
chown :newgroup file.txt

# 재귀적 변경
chown -R user:group directory/
```

```bash
# 예시
sudo chown www-data:www-data /var/www/html
sudo chown -R ubuntu:ubuntu ~/projects/
```

---

## 4. chgrp - 그룹 변경

```bash
# 그룹만 변경
chgrp developers file.txt

# 재귀적 변경
chgrp -R www-data /var/www/
```

---

## 5. 특수 권한

### SUID (Set User ID)

실행 시 파일 소유자의 권한으로 실행됩니다.

```
-rwsr-xr-x  → s는 SUID 설정됨
```

```bash
# SUID 설정
chmod u+s program
chmod 4755 program

# 대표적인 SUID 파일
ls -l /usr/bin/passwd
# -rwsr-xr-x 1 root root ... /usr/bin/passwd
```

### SGID (Set Group ID)

실행 시 파일 그룹의 권한으로 실행됩니다.
디렉토리에 설정하면 새 파일이 디렉토리의 그룹을 상속합니다.

```
-rwxr-sr-x  → s는 SGID 설정됨
```

```bash
# SGID 설정
chmod g+s directory/
chmod 2755 directory/

# 공유 디렉토리에 유용
sudo mkdir /shared
sudo chmod 2775 /shared
sudo chgrp developers /shared
# 이제 developers 그룹 멤버가 만든 파일은 모두 developers 그룹 소속
```

### Sticky Bit

디렉토리에 설정하면 파일 소유자만 삭제할 수 있습니다.

```
drwxrwxrwt  → t는 Sticky Bit
```

```bash
# Sticky Bit 설정
chmod +t directory/
chmod 1777 directory/

# /tmp가 대표적인 예
ls -ld /tmp
# drwxrwxrwt 1 root root 4096 Jan 23 10:00 /tmp
```

### 특수 권한 숫자

| 권한 | 숫자 | 위치 |
|------|------|------|
| SUID | 4 | 앞자리 |
| SGID | 2 | 앞자리 |
| Sticky | 1 | 앞자리 |

```bash
# SUID + 755
chmod 4755 file

# SGID + 775
chmod 2775 directory/

# Sticky + 777
chmod 1777 /tmp/
```

---

## 6. umask - 기본 권한

umask는 새 파일/디렉토리의 기본 권한을 결정합니다.

```
파일 기본: 666 - umask
디렉토리 기본: 777 - umask
```

```bash
# 현재 umask 확인
umask
# 0022

# umask 설정
umask 022    # 새 파일 644, 새 디렉토리 755
umask 077    # 새 파일 600, 새 디렉토리 700
umask 002    # 새 파일 664, 새 디렉토리 775
```

### umask 계산 예시

```
umask = 022

파일:      666
         - 022
         ------
           644 (rw-r--r--)

디렉토리:  777
         - 022
         ------
           755 (rwxr-xr-x)
```

### 영구 설정

```bash
# ~/.bashrc 또는 ~/.profile에 추가
echo "umask 022" >> ~/.bashrc
source ~/.bashrc
```

---

## 7. 권한 확인 명령어

### ls -l

```bash
ls -l file.txt
# -rw-r--r-- 1 ubuntu ubuntu 1234 Jan 23 10:00 file.txt
```

### stat

```bash
stat file.txt
```

출력:
```
  File: file.txt
  Size: 1234            Blocks: 8          IO Block: 4096   regular file
Access: (0644/-rw-r--r--)  Uid: ( 1000/  ubuntu)   Gid: ( 1000/  ubuntu)
...
```

### getfacl (ACL 지원 시)

```bash
getfacl file.txt
```

---

## 8. 실무 시나리오

### 웹 서버 디렉토리 설정

```bash
# 웹 루트 디렉토리 설정
sudo mkdir -p /var/www/mysite
sudo chown -R www-data:www-data /var/www/mysite
sudo chmod -R 755 /var/www/mysite

# 업로드 디렉토리 (쓰기 허용)
sudo mkdir /var/www/mysite/uploads
sudo chmod 775 /var/www/mysite/uploads

# 설정 파일 (읽기 전용)
sudo chmod 640 /var/www/mysite/config.php
sudo chown root:www-data /var/www/mysite/config.php
```

### 공유 디렉토리 설정

```bash
# 개발팀 공유 디렉토리
sudo groupadd developers
sudo mkdir /shared/dev
sudo chgrp developers /shared/dev
sudo chmod 2775 /shared/dev

# 사용자를 그룹에 추가
sudo usermod -aG developers username
```

### SSH 키 권한

```bash
# SSH 디렉토리 권한 (필수!)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa          # 개인키
chmod 644 ~/.ssh/id_rsa.pub      # 공개키
chmod 600 ~/.ssh/authorized_keys
chmod 644 ~/.ssh/known_hosts
```

### 스크립트 실행 권한

```bash
# 스크립트에 실행 권한
chmod +x deploy.sh
chmod +x *.sh

# 또는 755
chmod 755 backup.sh
```

---

## 9. 권한 문제 해결

### 권한 거부 오류

```bash
# 오류: Permission denied
# 해결 1: 권한 확인
ls -la file.txt

# 해결 2: 권한 변경
chmod 644 file.txt    # 또는 적절한 권한

# 해결 3: 소유자 변경
sudo chown $USER file.txt

# 해결 4: sudo 사용
sudo cat /etc/shadow
```

### 디렉토리 진입 불가

```bash
# 오류: Permission denied (cd 불가)
# 디렉토리에 x 권한 필요
chmod +x directory/
```

### 파일 수정 불가

```bash
# 오류: 파일 수정 불가
# 해결: 쓰기 권한 추가
chmod u+w file.txt

# 또는 디렉토리에 쓰기 권한 (새 파일 생성 시)
chmod u+w directory/
```

---

## 10. 실습 예제

### 실습 1: 권한 읽기

```bash
# 파일 생성
touch test_file.txt
mkdir test_dir

# 권한 확인
ls -la test_file.txt test_dir
stat test_file.txt
```

### 실습 2: chmod 연습

```bash
# 스크립트 생성
cat > test_script.sh << 'EOF'
#!/bin/bash
echo "Hello from script!"
EOF

# 실행 시도 (권한 없음)
./test_script.sh
# Permission denied

# 실행 권한 부여
chmod +x test_script.sh
./test_script.sh
# Hello from script!

# 다양한 권한 설정
chmod 755 test_script.sh    # rwxr-xr-x
chmod 700 test_script.sh    # rwx------
chmod 644 test_script.sh    # rw-r--r--
```

### 실습 3: 소유권 변경

```bash
# 파일 소유권 확인
ls -l test_file.txt

# 그룹 변경 (sudo 필요할 수 있음)
sudo chgrp users test_file.txt
ls -l test_file.txt
```

### 실습 4: umask 테스트

```bash
# 현재 umask 확인
umask

# umask 변경 후 파일 생성
umask 077
touch secret.txt
mkdir private_dir
ls -la secret.txt private_dir

# 원래대로 복원
umask 022
```

### 실습 5: 공유 디렉토리

```bash
# 공유 디렉토리 생성 (sudo 필요)
sudo mkdir /tmp/shared_test
sudo chmod 1777 /tmp/shared_test

# 테스트 파일 생성
touch /tmp/shared_test/my_file.txt

# 다른 사용자가 삭제 시도하면 실패
# (Sticky bit 때문)
```

---

## 다음 단계

[사용자와 그룹 관리](./06_User_Group_Management.md)에서 사용자와 그룹 관리를 배워봅시다!
