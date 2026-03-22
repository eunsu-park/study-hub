# 파일과 디렉토리 관리

**이전**: [파일시스템 탐색](./02_Filesystem_Navigation.md) | **다음**: [텍스트 처리](./04_Text_Processing.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. touch와 mkdir을 사용해 파일과 디렉토리 구조를 생성한다
2. cp로 파일과 디렉토리를 복사하며 백업에 적합한 옵션을 선택한다
3. mv를 사용해 파일과 디렉토리를 이동하거나 이름을 바꾼다
4. rm과 rmdir로 파일과 디렉토리를 안전하게 삭제하고 치명적인 실수를 피한다
5. cat, less, head, tail을 사용해 파일 내용을 확인한다(실시간 로그 모니터링 포함)
6. ln으로 하드 링크(hard link)와 심볼릭 링크(symbolic link)를 생성하고 관리한다
7. tar, gzip, zip으로 아카이브를 압축하고 압축을 푼다
8. du와 df로 디스크 사용량을 확인한다

---

파일과 디렉토리 관리는 리눅스 관리자의 일상적인 업무입니다. 코드 배포, 백업 정리, 임시 파일 정리 등 이 명령어들을 하루에도 수백 번 사용하게 됩니다. 안전 플래그와 함께 이 명령어들을 완전히 익히면 프로젝트 디렉토리 전체를 실수로 삭제하는 등 치명적인 실수를 예방할 수 있습니다.

## 1. 파일/디렉토리 생성

### touch - 빈 파일 생성

```bash
# 빈 파일 생성
touch newfile.txt

# 여러 파일 생성
touch file1.txt file2.txt file3.txt

# 파일이 존재하면 타임스탬프만 갱신
touch existing_file.txt
```

### mkdir - 디렉토리 생성

```bash
# 단일 디렉토리 생성
mkdir projects

# 여러 디렉토리 생성
mkdir dir1 dir2 dir3

# 중첩 디렉토리 생성 (-p: parents)
mkdir -p projects/web/frontend/src

# 권한과 함께 생성
mkdir -m 755 public_dir
```

```bash
# 디렉토리 구조 한번에 생성
mkdir -p myproject/{src,tests,docs,config}
```

생성 결과:
```
myproject/
├── src/
├── tests/
├── docs/
└── config/
```

---

## 2. 파일/디렉토리 복사

### cp - 복사

```bash
# 파일 복사
cp source.txt destination.txt

# 다른 디렉토리로 복사
cp file.txt /home/user/backup/

# 여러 파일 복사
cp file1.txt file2.txt /backup/
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-r`, `-R` | 디렉토리 재귀 복사 |
| `-i` | 덮어쓰기 전 확인 |
| `-v` | 진행 상황 표시 |
| `-p` | 권한, 소유자, 타임스탬프 유지 |
| `-a` | 아카이브 모드 (-rpP와 동일) |
| `-u` | 더 새로운 파일만 복사 |
| `-n` | 덮어쓰지 않음 |

```bash
# 디렉토리 복사 (재귀)
cp -r projects/ projects_backup/

# 대화형 복사 (덮어쓰기 확인)
cp -i important.txt backup/

# 진행 상황 표시
cp -v largefile.zip /backup/

# 속성 유지하며 복사
cp -p config.txt /backup/

# 아카이브 모드 (백업에 권장)
cp -a /var/www/ /backup/www/

# 새로운 파일만 복사
cp -u *.txt /backup/
```

---

## 3. 파일/디렉토리 이동 및 이름 변경

### mv - 이동/이름 변경

```bash
# 파일 이름 변경
mv oldname.txt newname.txt

# 파일 이동
mv file.txt /home/user/documents/

# 디렉토리 이동
mv projects/ /home/user/

# 여러 파일 이동
mv file1.txt file2.txt /backup/

# 이동하면서 이름 변경
mv old_project/ /home/user/new_project/
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-i` | 덮어쓰기 전 확인 |
| `-v` | 진행 상황 표시 |
| `-n` | 덮어쓰지 않음 |
| `-u` | 더 새로운 경우만 이동 |

```bash
# 대화형 이동
mv -i file.txt /backup/

# 진행 상황 표시
mv -v *.log /archive/

# 기존 파일 덮어쓰지 않음
mv -n newfile.txt /shared/
```

---

## 4. 파일/디렉토리 삭제

### rm - 파일 삭제

```bash
# 파일 삭제
rm file.txt

# 여러 파일 삭제
rm file1.txt file2.txt file3.txt

# 와일드카드로 삭제
rm *.tmp
rm log_2023*
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-r`, `-R` | 디렉토리 재귀 삭제 |
| `-f` | 강제 삭제 (확인 없음) |
| `-i` | 삭제 전 확인 |
| `-v` | 삭제 파일 표시 |

```bash
# 디렉토리 삭제
rm -r directory/

# 강제 삭제 (주의!)
rm -f file.txt

# 디렉토리 강제 삭제 (매우 주의!)
rm -rf old_project/

# 대화형 삭제
rm -i important_file.txt

# 삭제 과정 표시
rm -rv logs/
```

### rmdir - 빈 디렉토리 삭제

```bash
# 빈 디렉토리만 삭제 가능
rmdir empty_dir/

# 상위 빈 디렉토리까지 삭제
rmdir -p a/b/c/  # c, b, a 순서로 삭제 (모두 비어있어야 함)
```

### 위험한 명령어 경고

```bash
# 절대 실행하지 마세요!
# rm -rf /           # 시스템 전체 삭제
# rm -rf /*          # 루트 아래 모든 것 삭제
# rm -rf ~/*         # 홈 디렉토리 전체 삭제
# rm -rf .           # 현재 디렉토리 삭제

# 안전한 습관
rm -ri directory/   # 대화형으로 확인
ls directory/       # 삭제 전 내용 확인
```

---

## 5. 파일 내용 확인

### cat - 전체 내용 출력

```bash
# 파일 내용 출력
cat file.txt

# 여러 파일 연결 출력
cat file1.txt file2.txt

# 줄 번호 표시
cat -n file.txt

# 빈 줄 압축
cat -s file.txt
```

### less - 페이지 단위 보기

큰 파일을 편하게 볼 수 있습니다.

```bash
less largefile.txt
```

| 키 | 동작 |
|----|------|
| `Space` / `f` | 다음 페이지 |
| `b` | 이전 페이지 |
| `g` | 파일 처음으로 |
| `G` | 파일 끝으로 |
| `/검색어` | 앞으로 검색 |
| `?검색어` | 뒤로 검색 |
| `n` | 다음 검색 결과 |
| `N` | 이전 검색 결과 |
| `q` | 종료 |

### more - 간단한 페이지 보기

```bash
more file.txt
```

### head - 파일 앞부분

```bash
# 처음 10줄 (기본)
head file.txt

# 처음 20줄
head -n 20 file.txt
head -20 file.txt

# 처음 100바이트
head -c 100 file.txt
```

### tail - 파일 뒷부분

```bash
# 마지막 10줄 (기본)
tail file.txt

# 마지막 20줄
tail -n 20 file.txt

# 실시간 모니터링 (로그 확인에 유용)
tail -f /var/log/syslog

# 여러 파일 실시간 모니터링
tail -f file1.log file2.log
```

---

## 6. 링크

### 하드링크 vs 심볼릭링크

```
┌──────────────────────────────────────────────────────────┐
│                    하드링크                               │
│                                                          │
│   file.txt ─────┬───▶ [inode 123] ───▶ [데이터 블록]     │
│                 │                                        │
│   hardlink.txt ─┘                                        │
│                                                          │
│   • 같은 inode를 가리킴                                   │
│   • 원본 삭제해도 데이터 유지                              │
│   • 같은 파일시스템 내에서만 가능                          │
│   • 디렉토리는 불가능                                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   심볼릭링크 (소프트링크)                   │
│                                                          │
│   file.txt ─────────▶ [inode 123] ───▶ [데이터 블록]     │
│                 ▲                                        │
│   symlink.txt ──┘  (경로를 가리킴)                        │
│                                                          │
│   • 파일 경로를 가리킴                                    │
│   • 원본 삭제 시 깨진 링크                                │
│   • 다른 파일시스템 가능                                  │
│   • 디렉토리도 가능                                       │
└──────────────────────────────────────────────────────────┘
```

### ln - 링크 생성

```bash
# 하드링크 생성
ln original.txt hardlink.txt

# 심볼릭링크 생성
ln -s original.txt symlink.txt

# 디렉토리 심볼릭링크
ln -s /var/log/ ~/logs

# 강제 덮어쓰기
ln -sf new_target.txt symlink.txt
```

### 링크 확인

```bash
# 심볼릭링크 확인 (ls -l)
ls -l symlink.txt
```

출력:
```
lrwxrwxrwx 1 user user 12 Jan 23 10:00 symlink.txt -> original.txt
```

```bash
# 링크 개수 확인 (ls -l 두 번째 컬럼)
ls -l hardlink.txt original.txt
```

출력:
```
-rw-r--r-- 2 user user 100 Jan 23 10:00 hardlink.txt
-rw-r--r-- 2 user user 100 Jan 23 10:00 original.txt
```

---

## 7. 압축과 아카이브

### tar - 아카이브

tar는 여러 파일을 하나로 묶습니다.

| 옵션 | 설명 |
|------|------|
| `-c` | 아카이브 생성 (create) |
| `-x` | 아카이브 추출 (extract) |
| `-t` | 내용 확인 (list) |
| `-v` | 상세 출력 (verbose) |
| `-f` | 파일명 지정 (file) |
| `-z` | gzip 압축 (.tar.gz) |
| `-j` | bzip2 압축 (.tar.bz2) |
| `-J` | xz 압축 (.tar.xz) |
| `-C` | 추출 디렉토리 지정 |

```bash
# 아카이브 생성
tar -cvf archive.tar directory/

# gzip 압축 아카이브
tar -czvf archive.tar.gz directory/

# bzip2 압축 (더 높은 압축률)
tar -cjvf archive.tar.bz2 directory/

# xz 압축 (가장 높은 압축률)
tar -cJvf archive.tar.xz directory/

# 아카이브 내용 확인
tar -tvf archive.tar.gz

# 아카이브 추출
tar -xvf archive.tar
tar -xzvf archive.tar.gz

# 특정 디렉토리에 추출
tar -xzvf archive.tar.gz -C /tmp/

# 특정 파일만 추출
tar -xzvf archive.tar.gz file1.txt file2.txt
```

### gzip / gunzip - 압축

```bash
# 압축 (원본 삭제)
gzip file.txt          # → file.txt.gz

# 압축 해제
gunzip file.txt.gz     # → file.txt

# 원본 유지하며 압축
gzip -k file.txt

# 압축 레벨 (1-9, 9가 최고)
gzip -9 file.txt
```

### zip / unzip - ZIP 압축

```bash
# 압축
zip archive.zip file1.txt file2.txt

# 디렉토리 포함 압축
zip -r archive.zip directory/

# 압축 해제
unzip archive.zip

# 특정 디렉토리에 해제
unzip archive.zip -d /tmp/

# 내용 확인
unzip -l archive.zip
```

### 압축 형식 비교

| 형식 | 명령어 | 압축률 | 속도 | 호환성 |
|------|--------|--------|------|--------|
| .gz | gzip | 중간 | 빠름 | 높음 |
| .bz2 | bzip2 | 높음 | 중간 | 높음 |
| .xz | xz | 매우 높음 | 느림 | 보통 |
| .zip | zip | 중간 | 빠름 | 최고 |

---

## 8. 파일 타입 확인

### file 명령어

```bash
file document.pdf
file script.sh
file /bin/ls
file archive.tar.gz
```

출력:
```
document.pdf: PDF document, version 1.4
script.sh: Bourne-Again shell script, ASCII text executable
/bin/ls: ELF 64-bit LSB pie executable, x86-64
archive.tar.gz: gzip compressed data
```

---

## 9. 디스크 사용량

### du - 디렉토리 사용량

```bash
# 디렉토리 크기
du -h directory/

# 요약만 출력
du -sh directory/

# 현재 디렉토리 하위 폴더별 크기
du -h --max-depth=1

# 큰 폴더 찾기
du -h --max-depth=1 | sort -hr | head -10
```

### df - 디스크 여유 공간

```bash
# 파일시스템별 사용량
df -h

# 특정 경로의 파일시스템
df -h /home
```

---

## 10. 실습 예제

### 실습 1: 프로젝트 구조 만들기

```bash
# 프로젝트 디렉토리 생성
mkdir -p myapp/{src,tests,docs,config}

# 구조 확인
ls -la myapp/

# 빈 파일 생성
touch myapp/src/main.py
touch myapp/tests/test_main.py
touch myapp/config/settings.conf
touch myapp/README.md

# 결과 확인
find myapp -type f
```

### 실습 2: 파일 백업

```bash
# 백업 디렉토리 생성
mkdir -p backup/$(date +%Y%m%d)

# 파일 복사
cp -v important.txt backup/$(date +%Y%m%d)/

# 디렉토리 백업
cp -a myapp/ backup/$(date +%Y%m%d)/myapp_backup/

# 압축 백업
tar -czvf backup/myapp_$(date +%Y%m%d).tar.gz myapp/
```

### 실습 3: 로그 파일 관리

```bash
# 로그 디렉토리로 이동
cd /var/log

# 큰 로그 파일 찾기
ls -lhS *.log 2>/dev/null | head -5

# 최근 로그 확인
tail -20 syslog

# 실시간 모니터링
tail -f syslog
# (Ctrl+C로 종료)
```

### 실습 4: 임시 파일 정리

```bash
# /tmp 내용 확인
ls -la /tmp/

# 7일 이상 된 임시 파일 찾기
find /tmp -mtime +7 -type f 2>/dev/null

# 특정 패턴 파일 삭제 (주의)
# find /tmp -name "*.tmp" -mtime +7 -delete
```

### 실습 5: 심볼릭 링크 활용

```bash
# 설정 파일 링크
mkdir -p ~/dotfiles
ln -s ~/.bashrc ~/dotfiles/bashrc
ln -s ~/.vimrc ~/dotfiles/vimrc

# 링크 확인
ls -la ~/dotfiles/

# 로그 디렉토리 바로가기
ln -s /var/log ~/logs
ls ~/logs/
```

---

## 다음 단계

[텍스트 처리](./04_Text_Processing.md)에서 grep, sed, awk를 사용한 텍스트 처리를 배워봅시다!
