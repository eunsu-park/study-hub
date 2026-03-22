# 11. Git 내부 구조

**이전**: [모노레포 관리](./10_Monorepo_Management.md) | **다음**: [Git Bisect와 디버깅](./12_Git_Bisect_and_Debugging.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Git의 내용 주소 지정(content-addressable) 객체 모델과 네 가지 객체 유형(블롭(blob), 트리(tree), 커밋(commit), 태그(tag))을 설명할 수 있습니다
2. `.git` 디렉토리 구조를 탐색하고 이해할 수 있습니다
3. 플러밍 명령어(plumbing commands)인 `hash-object`, `cat-file`, `ls-tree`, `rev-parse`, `update-ref`를 사용하여 Git과 저수준으로 상호작용할 수 있습니다
4. 커밋(commit)이 방향성 비순환 그래프(DAG, Directed Acyclic Graph)를 형성하는 방식과 브랜치가 단순한 포인터임을 설명할 수 있습니다
5. 팩 파일(pack files), 가비지 컬렉션(garbage collection), Git의 저장소 최적화 방식을 이해할 수 있습니다
6. 플러밍(plumbing)과 포슬린(porcelain) 명령어를 구분하고 각각의 사용 시점을 알 수 있습니다

---

대부분의 Git 사용자는 `commit`, `push`, `merge`와 같은 고수준 명령어만 사용합니다. 하지만 이러한 친숙한 명령어 아래에는 놀랍도록 우아한 내용 주소 지정 파일시스템(content-addressable filesystem)이 존재합니다. Git의 내부를 이해하면 단순한 Git 사용자에서 Git이 실제로 무엇을 하는지 진정으로 이해하는 사람으로 변모하게 됩니다 -- 그리고 더 중요한 것은, 문제가 왜 발생하고 어떻게 고칠 수 있는지를 알게 됩니다.

## 목차
1. [.git 디렉토리](#1-git-디렉토리)
2. [Git의 객체 모델](#2-git의-객체-모델)
3. [내용 주소 지정 저장소](#3-내용-주소-지정-저장소)
4. [플러밍 vs 포슬린](#4-플러밍-vs-포슬린)
5. [DAG: 커밋이 히스토리를 구성하는 방식](#5-dag-커밋이-히스토리를-구성하는-방식)
6. [팩 파일과 가비지 컬렉션](#6-팩-파일과-가비지-컬렉션)
7. [저수준 명령어 참조](#7-저수준-명령어-참조)
8. [연습 문제](#8-연습-문제)

---

## 1. .git 디렉토리

`git init`을 실행하면 Git은 Git이 필요로 하는 모든 것을 포함하는 `.git` 디렉토리를 생성합니다. 작업 디렉토리(working directory)는 한 버전의 체크아웃(checkout)에 불과하며, 실제 저장소(repository)는 `.git` 내부에 존재합니다.

### 1.1 디렉토리 구조

```
.git/
├── HEAD                 # 현재 브랜치 참조(ref)를 가리킴
├── config               # 저장소별 설정
├── description          # GitWeb에서 사용 (거의 수정하지 않음)
├── index                # 스테이징 영역 (바이너리 파일)
├── packed-refs          # 효율성을 위한 패킹된 참조
├── objects/             # 모든 콘텐츠 (블롭, 트리, 커밋, 태그)
│   ├── info/
│   └── pack/            # 압축을 위한 팩 파일
├── refs/                # 커밋 객체를 가리키는 포인터
│   ├── heads/           # 브랜치 끝(tip)
│   ├── tags/            # 태그 참조
│   └── remotes/         # 원격 추적 브랜치
├── hooks/               # 클라이언트/서버 측 훅 스크립트
├── info/                # 전역 제외 패턴 등
│   └── exclude          # .gitignore와 유사하지만 커밋되지 않음
└── logs/                # Reflog 항목
    ├── HEAD
    └── refs/
```

### 1.2 HEAD 파일

`HEAD`는 가장 단순하면서도 가장 중요한 파일입니다. Git에게 현재 어떤 브랜치에 있는지 알려줍니다.

```bash
# HEAD 내용 보기
cat .git/HEAD
# ref: refs/heads/main

# 분리된 HEAD 상태(detached HEAD state)일 때
cat .git/HEAD
# a1b2c3d4e5f6... (직접 SHA-1 해시)
```

### 1.3 인덱스(스테이징 영역)

인덱스(index)는 스테이징 영역(staging area)을 저장하는 바이너리 파일(`.git/index`)입니다. 작업 디렉토리와 객체 데이터베이스 사이에 위치합니다.

```bash
# 인덱스 내용 보기
git ls-files --stage
# 100644 ce013625030ba8dba906f756967f9e9ca394464a 0	README.md
# 100644 8baef1b4abc478178b004d62031cf7fe6db6f903 0	src/main.py

# 세 열은 각각: 모드, SHA-1, 스테이지 번호, 파일명
# 스테이지 0 = 정상, 1/2/3 = 병합 충돌(merge conflict) 스테이지
```

### 1.4 refs 디렉토리

브랜치와 태그는 단순히 SHA-1 해시를 담고 있는 파일입니다.

```bash
# 브랜치는 커밋 해시가 들어있는 파일에 불과
cat .git/refs/heads/main
# e83c5163316f89bfbde7d9ab23ca2e25604af290

# 경량 태그(lightweight tag)도 동일
cat .git/refs/tags/v1.0
# e83c5163316f89bfbde7d9ab23ca2e25604af290

# 원격 추적 브랜치(remote tracking branches)
cat .git/refs/remotes/origin/main
# e83c5163316f89bfbde7d9ab23ca2e25604af290
```

---

## 2. Git의 객체 모델

Git에는 정확히 네 가지 객체 유형이 있습니다. Git의 모든 것 -- 모든 파일, 모든 디렉토리 스냅샷, 모든 커밋 -- 은 이 객체 중 하나로 저장됩니다.

### 2.1 블롭 객체(Blob Objects) — 파일 내용

블롭(blob)은 단일 파일의 내용을 저장합니다. 파일명, 권한, 기타 메타데이터는 저장하지 않고 -- 오직 원시 내용만 저장합니다.

```bash
# 수동으로 블롭 생성
echo "Hello, Git internals!" | git hash-object -w --stdin
# 반환: 8d0e41234f24b6da002d962a26c2495ea16a425f

# 객체 저장 위치:
# .git/objects/8d/0e41234f24b6da002d962a26c2495ea16a425f
#              ^^ 처음 2글자 = 디렉토리
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 나머지 글자 = 파일명
```

블롭의 핵심 특성:
- 내용만 저장: 동일한 내용의 두 파일은 같은 블롭을 공유합니다
- 파일명 미저장: 트리(tree) 객체가 이름과 블롭을 매핑합니다
- zlib으로 압축됩니다

### 2.2 트리 객체(Tree Objects) — 디렉토리 스냅샷

트리(tree) 객체는 디렉토리를 나타냅니다. 파일명을 블롭(파일)이나 다른 트리(하위 디렉토리)에 매핑합니다.

```bash
# 트리 객체 보기
git ls-tree HEAD
# 100644 blob ce013625030ba8dba906f756967f9e9ca394464a    README.md
# 040000 tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579    src
# 100644 blob 8baef1b4abc478178b004d62031cf7fe6db6f903    Makefile

# 모드 값:
# 100644 = 일반 파일
# 100755 = 실행 파일
# 040000 = 하위 디렉토리 (트리)
# 120000 = 심볼릭 링크
# 160000 = gitlink (서브모듈)
```

트리 구조 시각화:

```
tree (루트)
├── blob "README.md"  → ce0136...
├── blob "Makefile"   → 8baef1...
└── tree "src/"       → d8329f...
    ├── blob "main.py"    → a1b2c3...
    └── blob "utils.py"   → d4e5f6...
```

### 2.3 커밋 객체(Commit Objects)

커밋(commit) 객체는 모든 것을 하나로 연결합니다. 다음을 포함합니다:
- 트리 객체를 가리키는 포인터 (프로젝트 스냅샷)
- 0개 이상의 부모 커밋 (초기 커밋은 0개, 일반은 1개, 병합은 2개 이상)
- 저자 정보(author) (이름, 이메일, 타임스탬프)
- 커미터 정보(committer) (저자와 다를 수 있음)
- 커밋 메시지

```bash
# 커밋 객체 보기
git cat-file -p HEAD
# tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579
# parent a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# author John Doe <john@example.com> 1709200000 +0900
# committer John Doe <john@example.com> 1709200000 +0900
#
# Add user authentication module
```

### 2.4 태그 객체(Tag Objects) — 주석 태그(Annotated Tags)

주석 태그(annotated tags)는 완전한 객체입니다 (단순 참조인 경량 태그(lightweight tags)와 달리).

```bash
# 주석 태그 생성
git tag -a v1.0 -m "Release version 1.0"

# 태그 객체 보기
git cat-file -p v1.0
# object e83c5163316f89bfbde7d9ab23ca2e25604af290
# type commit
# tag v1.0
# tagger John Doe <john@example.com> 1709200000 +0900
#
# Release version 1.0
```

### 2.5 객체 관계(Object Relationships)

```
tag v1.0
  │
  ▼
commit c3 ─────► tree (루트 스냅샷)
  │                ├── blob README.md
  │                └── tree src/
  │                     └── blob main.py
  ▼
commit c2 ─────► tree (이전 스냅샷)
  │                ├── blob README.md
  │                └── tree src/
  │                     └── blob main.py
  ▼
commit c1 ─────► tree (초기 스냅샷)
                   └── blob README.md
```

---

## 3. 내용 주소 지정 저장소

Git은 근본적으로 내용 주소 지정 파일시스템(content-addressable filesystem)입니다. 모든 객체의 주소(키)는 그 내용의 SHA-1 해시입니다.

### 3.1 해싱 작동 방식

```bash
# Git은 헤더와 함께 내용을 해시: "<type> <size>\0<content>"
# 블롭의 경우:
echo -n "Hello, Git internals!" | git hash-object --stdin
# 8d0e41234f24b6da002d962a26c2495ea16a425f

# 동등한 수동 계산:
echo -en "blob 21\0Hello, Git internals!" | shasum
# 8d0e41234f24b6da002d962a26c2495ea16a425f  -

# 헤더 형식: "blob <내용-길이>\0"
```

### 3.2 내용 주소 지정 저장소의 함의

**무결성(Integrity)**: 어떤 손상이든 해시를 변경하여 체인을 깨뜨립니다. Git은 자동으로 손상을 감지합니다.

```bash
# 저장소 무결성 검증
git fsck
# Checking object directories: 100% (256/256), done.
# Checking objects: 100% (1234/1234), done.
```

**중복 제거(Deduplication)**: 동일한 내용은 몇 개의 파일이나 커밋이 참조하든 한 번만 저장됩니다.

```bash
# 동일한 내용의 두 파일은 하나의 블롭을 공유
echo "shared content" > file_a.txt
echo "shared content" > file_b.txt
git add file_a.txt file_b.txt

git ls-files --stage
# 100644 abc123... 0  file_a.txt
# 100644 abc123... 0  file_b.txt   # 동일한 해시!
```

**불변성(Immutability)**: 객체는 절대 수정할 수 없습니다. "변경"은 새로운 해시를 가진 새로운 객체를 생성합니다.

### 3.3 디스크 상의 객체 저장

```bash
# 객체는 zlib으로 압축된 파일로 저장
# 경로: .git/objects/<처음-2글자>/<나머지-38글자>

# 원시 객체 보기 (Python)
python3 -c "
import zlib, sys
with open('.git/objects/8d/0e41234f24b6da002d962a26c2495ea16a425f', 'rb') as f:
    print(zlib.decompress(f.read()))
"
# b'blob 21\x00Hello, Git internals!'
```

---

## 4. 플러밍 vs 포슬린

Git 명령어는 욕실 설비에서 이름을 딴 두 범주로 나뉩니다.

### 4.1 포슬린 명령어(Porcelain Commands) — 사용자 대면

매일 사용하는 명령어입니다:

```bash
git add          git commit       git push
git pull         git merge        git rebase
git log          git status       git diff
git branch       git checkout     git switch
git stash        git tag          git fetch
git clone        git remote       git reset
```

### 4.2 플러밍 명령어(Plumbing Commands) — 저수준

포슬린 명령어가 내부적으로 사용하는 구성 요소입니다:

```bash
# 객체 조작
git hash-object     # 객체 해시 계산 / 데이터베이스에 기록
git cat-file        # 객체 내용, 유형 또는 크기 표시
git write-tree      # 인덱스를 트리 객체로 기록
git commit-tree     # 트리 객체로부터 커밋 생성
git mktag           # 태그 객체 생성

# 인덱스 조작
git update-index    # 인덱스에 파일 내용 등록
git read-tree       # 트리를 인덱스로 읽기
git ls-files        # 인덱스의 파일 정보 표시

# 참조 조작
git update-ref      # 참조를 안전하게 업데이트
git symbolic-ref    # 심볼릭 참조(HEAD 등) 읽기/업데이트

# 검사
git ls-tree         # 트리 객체의 내용 나열
git rev-parse       # 리비전 식별자 파싱
git rev-list        # 역시간순으로 커밋 객체 나열
git for-each-ref    # 참조 반복 처리
git diff-tree       # 두 트리 객체 비교
```

### 4.3 플러밍 명령어로 커밋 만들기

`git add` + `git commit`이 내부에서 작동하는 방식입니다:

```bash
# 1단계: 파일 내용을 블롭으로 저장
BLOB_HASH=$(echo "Hello World" | git hash-object -w --stdin)
echo "Blob: $BLOB_HASH"

# 2단계: 블롭을 인덱스에 추가
git update-index --add --cacheinfo 100644 $BLOB_HASH hello.txt

# 3단계: 인덱스를 트리 객체로 기록
TREE_HASH=$(git write-tree)
echo "Tree: $TREE_HASH"

# 4단계: 트리를 가리키는 커밋 생성
COMMIT_HASH=$(echo "Initial commit via plumbing" | \
  git commit-tree $TREE_HASH)
echo "Commit: $COMMIT_HASH"

# 5단계: 브랜치가 새 커밋을 가리키도록 업데이트
git update-ref refs/heads/main $COMMIT_HASH

# 6단계: HEAD가 브랜치를 가리키도록 설정
git symbolic-ref HEAD refs/heads/main

# 이제 'git log'에서 우리의 커밋을 볼 수 있습니다!
git log --oneline
```

---

## 5. DAG: 커밋이 히스토리를 구성하는 방식

### 5.1 DAG 이해하기

Git의 히스토리는 방향성 비순환 그래프(DAG, Directed Acyclic Graph)입니다. 각 커밋은 부모를 가리키며, 한 방향(시간 역순)으로 흐르고 절대 순환하지 않는 그래프를 형성합니다.

```
# 선형 히스토리
A ← B ← C ← D  (main)

# 브랜치와 병합
A ← B ← C ← F  (main)
     ↖         ↗
      D ← E     (feature)

# 다중 병합 부모
A ← B ← C ← G  (main)
     ↖       ↗
      D ← E
     ↖       ↗
      F ─────   (hotfix)
```

### 5.2 DAG 순회

```bash
# HEAD에서 도달 가능한 모든 커밋 나열
git rev-list HEAD
# 최신 것부터 모든 커밋의 SHA-1 표시

# 전체 커밋 수 세기
git rev-list --count HEAD
# 142

# main에서 도달 가능하지만 feature에서는 아닌 커밋
git rev-list main --not feature
# (분기점 이후 main의 커밋)

# 두 브랜치의 공통 조상
git merge-base main feature
# a1b2c3d4...

# 그래프 시각화
git log --all --graph --oneline --decorate
# * e83c516 (HEAD -> main) Merge feature
# |\
# | * a1b2c3d (feature) Add feature
# | * d4e5f6a Work on feature
# |/
# * 1234567 Base commit
```

### 5.3 도달 가능성(Reachability)

참조를 따라가서 찾을 수 있는 객체는 **도달 가능(reachable)**합니다. 도달 불가능한 객체는 가비지 컬렉션(garbage collection)의 대상이 됩니다.

```
refs/heads/main → commit C → commit B → commit A
                      │           │           │
                      ▼           ▼           ▼
                   tree T3     tree T2     tree T1
                      │           │           │
                      ▼           ▼           ▼
                   blobs...    blobs...    blobs...

위의 모든 객체는 refs/heads/main에서 도달 가능합니다.
```

```bash
# 도달 불가능한 객체 찾기
git fsck --unreachable
# unreachable blob 8d0e412...
# unreachable commit a1b2c3d...

# 댕글링(dangling) 객체 찾기 (도달 불가능하면서 다른 도달 불가능 객체에서도 참조되지 않는 것)
git fsck --dangling
```

### 5.4 조상 참조(Ancestry References)

```bash
# 부모 참조
HEAD^       # HEAD의 첫 번째 부모
HEAD^2      # 두 번째 부모 (병합 커밋에서만 의미 있음)
HEAD^^      # 조부모 (첫 번째 부모의 첫 번째 부모)

# 조상 참조
HEAD~1      # HEAD^와 동일
HEAD~2      # HEAD^^와 동일
HEAD~3      # 증조부모

# 조합
HEAD~2^2    # 조부모의 두 번째 부모

# 실용적 예시
git log --oneline HEAD~5..HEAD   # 최근 5개 커밋
git diff HEAD~3 HEAD             # 최근 3개 커밋의 변경 사항
```

---

## 6. 팩 파일과 가비지 컬렉션

### 6.1 느슨한 객체 vs 팩된 객체

처음에는 모든 객체가 별도 파일(느슨한 객체, loose object)로 저장됩니다. 이는 대규모 저장소에서 비효율적입니다.

```bash
# 느슨한 객체 수 세기
git count-objects
# 1234 objects, 5678 kilobytes

# 상세 통계
git count-objects -v
# count: 1234           # 느슨한 객체
# size: 5678            # 느슨한 객체 디스크 크기 (KB)
# in-pack: 45678        # 팩된 객체
# packs: 3              # 팩 파일 수
# size-pack: 12345      # 팩 파일 디스크 크기 (KB)
# prune-packable: 0     # 팩에도 있는 느슨한 객체
# garbage: 0            # objects 디렉토리에서 객체가 아닌 파일
# size-garbage: 0
```

### 6.2 팩 파일 작동 방식

팩 파일(pack files)은 델타 압축(delta compression)을 사용하여 객체를 저장합니다. 파일의 모든 버전을 저장하는 대신, 하나의 전체 버전과 다른 버전과의 델타(차이)를 저장합니다.

```bash
# 수동으로 팩 파일 생성
git repack -a -d
# -a: 모든 객체 팩
# -d: 중복 느슨한 객체 제거

# 팩 파일 내용 나열
git verify-pack -v .git/objects/pack/pack-*.idx
# SHA-1  type  size  size-in-pack  offset  depth  base-SHA-1
# a1b2c3 commit 234  180           12
# d4e5f6 tree   120  95            192
# 789abc blob   5678 1234          287     2      fedcba...
#                                          ^ 델타 깊이
#                                                 ^ 기반 객체
```

### 6.3 가비지 컬렉션(Garbage Collection)

```bash
# 가비지 컬렉션 실행
git gc
# Enumerating objects: 1234, done.
# Counting objects: 100% (1234/1234), done.
# Delta compression using up to 8 threads
# Compressing objects: 100% (567/567), done.
# Writing objects: 100% (1234/1234), done.

# 적극적 GC (더 느리지만 더 좋은 압축)
git gc --aggressive

# 자동 GC (임계값 도달 시 실행)
git gc --auto
# 느슨한 객체 > gc.auto (기본 6700)일 때
# 또는 팩 > gc.autoPackLimit (기본 50)일 때 트리거
```

### 6.4 프루닝(Pruning)

```bash
# 2주(기본값)보다 오래된 도달 불가능 객체 제거
git prune

# 제거될 항목 미리 보기
git prune --dry-run

# 즉시 제거 (위험 -- 유예 기간 없음)
git prune --expire=now

# Reflog 만료 (프루닝의 전제 조건)
git reflog expire --expire=90.days --all
git gc --prune=now
```

### 6.5 히스토리에서 대용량 파일 제거

대용량 파일이 실수로 커밋되면, 삭제해도 블롭이 히스토리에 남아있어 공간이 회수되지 않습니다.

```bash
# 대용량 객체 찾기
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  grep ^blob | sort -k3 -n -r | head -10
# blob a1b2c3d4 104857600 data/huge_file.bin

# git-filter-repo로 제거 (filter-branch보다 권장)
pip install git-filter-repo
git filter-repo --invert-paths --path data/huge_file.bin

# 히스토리 재작성 후 강제 가비지 컬렉션
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## 7. 저수준 명령어 참조

### 7.1 git hash-object

```bash
# 저장하지 않고 SHA-1 계산
echo "test content" | git hash-object --stdin
# d670460b4b4aece5915caf5c68d12f560a9fe3e4

# 계산하고 객체 데이터베이스에 저장
echo "test content" | git hash-object -w --stdin

# 파일 해시
git hash-object myfile.txt

# 파일을 해시하고 저장
git hash-object -w myfile.txt
```

### 7.2 git cat-file

```bash
# 객체 내용 표시
git cat-file -p HEAD          # 모든 객체를 보기 좋게 출력
git cat-file -p HEAD:file.txt # HEAD의 파일 내용

# 객체 유형 표시
git cat-file -t HEAD
# commit

# 객체 크기 표시
git cat-file -s HEAD
# 234

# 배치 모드 (많은 객체에 효율적)
echo "HEAD" | git cat-file --batch
echo "HEAD" | git cat-file --batch-check  # 유형과 크기만
```

### 7.3 git ls-tree

```bash
# 트리 내용 나열 (비재귀적)
git ls-tree HEAD
# 100644 blob ce01362... README.md
# 040000 tree d8329fc... src

# 재귀적 나열 (모든 파일)
git ls-tree -r HEAD
# 100644 blob ce01362... README.md
# 100644 blob a1b2c3d... src/main.py
# 100644 blob d4e5f6a... src/utils.py

# 이름만 표시
git ls-tree --name-only HEAD

# 특정 하위 디렉토리 나열
git ls-tree HEAD src/

# 객체 크기 포함
git ls-tree -l HEAD
# 100644 blob ce01362...    1234    README.md
```

### 7.4 git rev-parse

```bash
# 심볼릭 이름을 SHA-1로 변환
git rev-parse HEAD
# e83c5163316f89bfbde7d9ab23ca2e25604af290

git rev-parse main
git rev-parse HEAD~3
git rev-parse v1.0

# .git 디렉토리 보기
git rev-parse --git-dir
# .git

# 저장소 루트 보기
git rev-parse --show-toplevel
# /home/user/project

# Git 저장소 내부인지 확인
git rev-parse --is-inside-work-tree
# true

# 리비전 범위 파싱
git rev-parse main...feature   # 대칭 차집합
git rev-parse main..feature    # feature에 있지만 main에는 없는 것
```

### 7.5 git update-ref

```bash
# 참조를 특정 커밋으로 설정
git update-ref refs/heads/new-branch $COMMIT_HASH

# 참조 삭제
git update-ref -d refs/heads/old-branch

# 안전한 업데이트 (이전 값 먼저 확인)
git update-ref refs/heads/main $NEW_HASH $OLD_HASH
# main이 $OLD_HASH를 가리키지 않으면 실패 (경쟁 조건 방지)

# 경량 태그 생성
git update-ref refs/tags/v2.0 $COMMIT_HASH
```

### 7.6 git for-each-ref

```bash
# 서식을 사용하여 모든 참조 나열
git for-each-ref --format='%(refname:short) %(objecttype) %(objectname:short)' refs/heads/
# main commit e83c516
# feature commit a1b2c3d
# dev commit d4e5f6a

# 커미터 날짜순 정렬
git for-each-ref --sort=-committerdate --format='%(refname:short) %(committerdate:relative)' refs/heads/
# feature 2 hours ago
# main 1 day ago
# dev 3 days ago

# 병합된 브랜치 찾기
git for-each-ref --merged=main refs/heads/ --format='%(refname:short)'
```

---

## 8. 연습 문제

### 연습 1: 객체 데이터베이스 탐색

```bash
# 1. 새 저장소를 생성하고 커밋 만들기
git init internals-lab && cd internals-lab
echo "first file" > hello.txt
git add hello.txt
git commit -m "Initial commit"

# 2. 과제:
# a) git ls-files --stage를 사용하여 hello.txt의 블롭 해시 찾기
# b) git cat-file -p를 사용하여 블롭 내용 보기
# c) git cat-file -t를 사용하여 객체 유형 확인
# d) git ls-tree HEAD로 루트 트리 보기
# e) 커밋 객체에서 git cat-file -p를 사용하여 트리 포인터 확인
# f) 해시 수동 검증: echo -en "blob 10\0first file" | shasum
```

### 연습 2: 플러밍 명령어만으로 커밋 만들기

```bash
# 포슬린 명령어를 사용하지 않고 완전한 커밋을 만드세요.
# 사용 가능: hash-object, update-index, write-tree, commit-tree, update-ref

# 1. 블롭 생성:
#    echo "plumbing test" | git hash-object -w --stdin
#
# 2. 인덱스에 추가:
#    git update-index --add --cacheinfo 100644 <blob-hash> test.txt
#
# 3. 트리 기록:
#    git write-tree
#
# 4. 커밋 생성:
#    echo "Plumbing commit" | git commit-tree <tree-hash> -p HEAD
#
# 5. 브랜치 업데이트:
#    git update-ref refs/heads/main <commit-hash>
#
# 6. git log로 검증
```

### 연습 3: 객체 관계 조사

```bash
# 1. 각각 다른 파일을 수정하는 3개의 커밋이 있는 저장소 생성
# 2. git rev-list와 git cat-file을 사용하여 손으로 DAG 그리기
# 3. 각 커밋에 대해:
#    a) 트리 해시 찾기
#    b) 해당 트리의 모든 블롭 나열
#    c) 커밋 간 공유되는 블롭 식별
# 4. git gc 전후로 git count-objects 실행
# 5. 동일한 내용이 같은 블롭 해시를 공유하는지 확인
```

### 연습 4: 팩 파일 분석

```bash
# 1. 대용량 텍스트 파일이 있는 저장소 생성
dd if=/dev/urandom bs=1024 count=100 | base64 > large_file.txt
git add large_file.txt && git commit -m "Add large file"

# 2. large_file.txt에 5번의 작은 수정을 하고 각각 커밋
# 3. git count-objects -v로 느슨한 객체 통계 확인
# 4. git gc를 실행하고 다시 git count-objects -v 확인
# 5. git verify-pack -v로 델타 체인 검사
# 6. 느슨한 객체와 팩된 객체의 전체 크기 비교
# 7. 팩된 크기가 더 작은 이유 설명 (델타 압축)
```

### 연습 5: 플러밍 명령어를 이용한 복구

```bash
# 1. 커밋을 만들고 해시 기록
# 2. 이전 커밋으로 reset --hard (사고 시뮬레이션)
# 3. "잃어버린" 커밋은 여전히 댕글링 객체로 존재
# 4. 다음으로 찾기: git fsck --dangling
# 5. 다음으로 복구: git update-ref refs/heads/recovered <hash>
# 6. git log recovered로 복구 확인
```

---

## 다음 단계

- [Git Bisect와 디버깅](./12_Git_Bisect_and_Debugging.md) - Git을 활용한 디버깅
- [Git 공식 문서 - Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain) - 공식 참조
- [Pro Git 책 - 10장](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain) - 내부 구조 심층 탐구

## 참고 자료

- [Git Internals - Pro Git Book](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)
- [Git Object Model](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
- [Git Packfiles](https://git-scm.com/book/en/v2/Git-Internals-Packfiles)
- [git-filter-repo](https://github.com/newren/git-filter-repo)
- [SHA-1 and Git](https://git-scm.com/docs/hash-function-transition)

---

[← 이전: 모노레포 관리](10_Monorepo_Management.md) | [다음: Git Bisect와 디버깅 →](12_Git_Bisect_and_Debugging.md) | [목차](00_Overview.md)
