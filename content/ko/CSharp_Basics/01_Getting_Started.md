# 시작하기

**이전**: [개요](./00_Overview.md) | **다음**: [변수와 타입](./02_Variables_and_Types.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. macOS, Windows, Linux에 .NET SDK를 설치한다
2. `dotnet` CLI를 사용하여 프로젝트를 생성, 빌드, 실행한다
3. 최상위 문(Top-Level Statement)과 전통적인 `Main` 메서드 두 가지 방식으로 Hello World 프로그램을 작성한다
4. C# 콘솔 애플리케이션의 프로젝트 구조를 이해한다
5. 솔루션 파일과 다중 프로젝트 설정을 다룬다
6. C# Dev Kit 확장과 함께 VS Code를 구성한다
7. 소스 코드에서 실행까지의 C# 컴파일 과정을 설명한다

---

C#은 Microsoft가 .NET 플랫폼의 일부로 개발한 현대적인 객체지향, 강타입(Strongly-Typed) 프로그래밍 언어입니다. 2000년에 처음 출시된 C#은 여러 버전을 거치며 크게 발전했으며, 이 글 작성 시점에서 C# 12(.NET 8과 함께 출시)가 최신 안정 릴리스입니다. 이 언어는 데스크톱 소프트웨어(WPF, WinForms), 웹 서비스(ASP.NET Core), 모바일 앱(.NET MAUI), 게임 개발(Unity), 클라우드 서비스(Azure Functions) 등 매우 다양한 애플리케이션에 사용됩니다. 이 첫 번째 레슨에서는 개발 환경을 설정하고 첫 번째 C# 프로그램을 작성합니다.

## 1. .NET SDK 설치

.NET SDK(소프트웨어 개발 키트)에는 C# 애플리케이션을 빌드하고 실행하는 데 필요한 모든 것이 포함되어 있습니다: 컴파일러, 런타임, 그리고 `dotnet` CLI 도구입니다.

### 1.1 버전 선택

.NET은 예측 가능한 릴리스 주기를 따릅니다. 짝수 번호 릴리스(6, 8, 10)는 3년간 지원되는 장기 지원(LTS) 버전입니다. 홀수 번호 릴리스(7, 9)는 18개월 지원되는 표준 지원(STS) 버전입니다. 학습용으로는 항상 최신 LTS 릴리스를 선택하세요.

```
릴리스 타임라인:
  .NET 6  (LTS)  — 2021년 11월 ~ 2024년 11월
  .NET 7  (STS)  — 2022년 11월 ~ 2024년 5월
  .NET 8  (LTS)  — 2023년 11월 ~ 2026년 11월
  .NET 9  (STS)  — 2024년 11월 ~ 2026년 5월
  .NET 10 (LTS)  — 2025년 11월 ~ 2028년 11월
```

### 1.2 macOS에서 설치

macOS에서 권장되는 방법은 공식 설치 프로그램이나 Homebrew를 사용하는 것입니다:

```bash
# 방법 1: Homebrew (권장)
brew install --cask dotnet-sdk

# 방법 2: 공식 웹사이트에서 설치 프로그램 다운로드
# https://dotnet.microsoft.com/download

# 설치 확인
dotnet --version
# 8.0.401 (또는 유사)

dotnet --list-sdks
# 8.0.401 [/usr/local/share/dotnet/sdk]
```

Homebrew로 설치한 경우, dotnet 경로가 셸 프로파일에 포함되어 있는지 확인하세요:

```bash
# 필요 시 ~/.zshrc 또는 ~/.bash_profile에 추가
export DOTNET_ROOT="/usr/local/share/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
```

### 1.3 Windows에서 설치

Windows에서는 공식 웹사이트에서 설치 프로그램을 다운로드하거나 `winget`을 사용합니다:

```powershell
# 방법 1: winget (Windows 패키지 관리자)
winget install Microsoft.DotNet.SDK.8

# 방법 2: https://dotnet.microsoft.com/download 에서 다운로드

# 설치 확인
dotnet --version
```

Windows 설치 프로그램은 자동으로 `dotnet`을 시스템 PATH에 추가합니다.

### 1.4 Linux에서 설치

Ubuntu/Debian 기반 배포판에서:

```bash
# Microsoft 패키지 저장소 추가
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

# SDK 설치
sudo apt-get update
sudo apt-get install -y dotnet-sdk-8.0

# 설치 확인
dotnet --version
```

Fedora에서:

```bash
sudo dnf install dotnet-sdk-8.0
dotnet --version
```

### 1.5 설치 확인

운영체제에 관계없이 다음 명령어를 실행하여 모든 것이 정상적으로 동작하는지 확인합니다:

```bash
# SDK 버전 표시
dotnet --version

# 설치된 모든 SDK 표시
dotnet --list-sdks

# 설치된 모든 런타임 표시
dotnet --list-runtimes

# 종합 정보 표시
dotnet --info
```

## 2. dotnet CLI

`dotnet` 명령줄 인터페이스는 .NET 애플리케이션을 생성, 빌드, 실행, 게시하기 위한 기본 도구입니다. 이 과정 전반에 걸쳐 지속적으로 사용하게 됩니다.

### 2.1 새 프로젝트 생성

`dotnet new` 명령은 내장 템플릿으로 프로젝트를 생성합니다:

```bash
# 새 콘솔 애플리케이션 생성
dotnet new console -n MyFirstApp

# 특정 프레임워크 버전으로 생성
dotnet new console -n MyFirstApp --framework net8.0

# 사용 가능한 모든 템플릿 나열
dotnet new list
```

`-n`(또는 `--name`) 플래그는 프로젝트 디렉토리 이름과 기본 네임스페이스를 모두 지정합니다. 지정하지 않으면 현재 디렉토리 이름이 사용됩니다.

일반적인 템플릿은 다음과 같습니다:

```
템플릿 이름             축약 이름      언어
---------------------  -----------   --------
Console App            console       C#
Class Library          classlib      C#
ASP.NET Core Web App   webapp        C#
ASP.NET Core Web API   webapi        C#
xUnit Test Project     xunit         C#
NUnit Test Project     nunit         C#
```

### 2.2 빌드와 실행

```bash
# 프로젝트 디렉토리로 이동
cd MyFirstApp

# 프로젝트 빌드 (실행 없이 컴파일)
dotnet build

# 프로젝트 실행 (필요하면 자동으로 빌드)
dotnet run

# 빌드 아티팩트 정리
dotnet clean

# Release 모드로 빌드 (최적화)
dotnet build -c Release

# Release 모드로 실행
dotnet run -c Release
```

### 2.3 패키지 추가

NuGet은 .NET의 패키지 관리자입니다. 다음과 같이 서드파티 라이브러리를 추가할 수 있습니다:

```bash
# NuGet 패키지 추가
dotnet add package Newtonsoft.Json

# 특정 버전 추가
dotnet add package Newtonsoft.Json --version 13.0.3

# 패키지 제거
dotnet remove package Newtonsoft.Json

# 설치된 패키지 나열
dotnet list package
```

### 2.4 기타 유용한 명령어

```bash
# 자체 포함 애플리케이션 게시
dotnet publish -c Release --self-contained

# 테스트 실행
dotnet test

# .editorconfig 규칙에 따라 코드 포맷
dotnet format

# 파일 변경 감지 및 자동 재빌드
dotnet watch run
```

## 3. Hello World: 두 가지 스타일

C#은 콘솔 애플리케이션에서 두 가지 진입점 스타일을 지원합니다. 문서와 실제 코드에서 두 가지 모두를 접하게 되므로 둘 다 이해하는 것이 중요합니다.

### 3.1 최상위 문(Top-Level Statement) (현대적 스타일)

C# 9와 .NET 5부터 클래스와 `Main` 메서드의 보일러플레이트(Boilerplate) 없이 콘솔 애플리케이션을 작성할 수 있습니다. 이것이 .NET 6 이후 `dotnet new console`의 기본 스타일입니다:

```csharp
// Program.cs — 최상위 문
// 클래스나 Main 메서드 불필요

Console.WriteLine("Hello, World!");
```

이것이 전체 파일입니다. 컴파일러가 뒤에서 `Main` 메서드를 자동으로 생성합니다. `args`를 사용하여 명령줄 인수에 접근할 수도 있습니다:

```csharp
// Program.cs — args를 사용하는 최상위 문
if (args.Length > 0)
{
    Console.WriteLine($"Hello, {args[0]}!");
}
else
{
    Console.WriteLine("Hello, World!");
}
```

인수와 함께 실행:

```bash
dotnet run -- Alice
# 출력: Hello, Alice!
```

최상위 문 파일에서 `await`를 사용하고, 메서드를 정의하고, 클래스를 선언할 수도 있습니다:

```csharp
// Program.cs — 로컬 함수를 사용하는 최상위 문
string greeting = CreateGreeting("C#");
Console.WriteLine(greeting);

string CreateGreeting(string name)
{
    return $"Hello, {name}! Welcome to the world of programming.";
}
```

### 3.2 전통적인 Main 메서드

C# 9 이전에는 모든 프로그램에 클래스 내부에 명시적인 `Main` 메서드가 필요했습니다. 이 스타일은 여전히 완전히 지원되며 규모가 큰 애플리케이션에서 선호됩니다:

```csharp
// Program.cs — 전통적인 진입점
namespace MyFirstApp;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");

        if (args.Length > 0)
        {
            Console.WriteLine($"Arguments: {string.Join(", ", args)}");
        }
    }
}
```

`Main` 메서드는 네 가지 유효한 시그니처를 가질 수 있습니다:

```csharp
// 1. 매개변수 없음, 반환 값 없음
static void Main() { }

// 2. 명령줄 인수 포함
static void Main(string[] args) { }

// 3. 종료 코드 반환
static int Main() { return 0; }

// 4. 인수와 종료 코드 모두 포함
static int Main(string[] args) { return 0; }

// 5. 비동기 변형 (C# 7.1+)
static async Task Main() { }
static async Task<int> Main(string[] args) { return 0; }
```

### 3.3 어떤 스타일을 사용해야 할까?

**최상위 문**은 다음 경우에 사용하세요:
- 작은 프로그램, 스크립트, 학습 연습
- 빠른 프로토타입과 실험
- 단일 파일 프로그램

**전통적인 Main 메서드**는 다음 경우에 사용하세요:
- 프로덕션 애플리케이션
- 여러 진입점이 있는 프로젝트
- 진입점 클래스에 대한 명시적 제어가 필요할 때

이 과정에서는 간결함을 위해 주로 최상위 문을 사용하지만, 전통적인 접근 방식이 더 적절한 경우에는 별도로 언급하겠습니다.

## 4. 프로젝트 구조

`dotnet new console -n MyFirstApp`으로 새 콘솔 애플리케이션을 생성하면 다음과 같은 구조가 생성됩니다:

```
MyFirstApp/
├── MyFirstApp.csproj    # 프로젝트 파일 (빌드 구성)
├── Program.cs           # 소스 코드 (진입점)
├── obj/                 # 중간 빌드 파일 (자동 생성)
│   ├── project.assets.json
│   ├── project.nuget.cache
│   └── ...
└── bin/                 # 컴파일된 출력 (빌드 후)
    └── Debug/
        └── net8.0/
            ├── MyFirstApp.dll    # 컴파일된 어셈블리
            ├── MyFirstApp.exe    # 실행 파일 (Windows) / 네이티브 호스트
            ├── MyFirstApp.deps.json
            ├── MyFirstApp.runtimeconfig.json
            └── ...
```

### 4.1 .csproj 파일

`.csproj`(C# 프로젝트) 파일은 프로젝트의 빌드 구성을 정의하는 XML 파일입니다:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

</Project>
```

주요 요소 설명:

- **`OutputType`**: 실행 파일은 `Exe`, 클래스 라이브러리는 `Library`
- **`TargetFramework`**: 대상 .NET 버전 (`net8.0`, `net9.0` 등)
- **`ImplicitUsings`**: `enable`이면 일반적인 `using` 지시문이 자동으로 포함됩니다 (`System`, `System.Collections.Generic`, `System.IO`, `System.Linq`, `System.Threading.Tasks` 등)
- **`Nullable`**: `enable`이면 컴파일러가 잠재적인 null 참조 문제에 대해 경고합니다

NuGet 패키지를 추가하면 `<PackageReference>` 요소로 나타납니다:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
  </ItemGroup>

</Project>
```

### 4.2 obj/ 디렉토리

`obj/` 디렉토리에는 중간 빌드 파일이 포함됩니다. 이 디렉토리의 파일을 편집해서는 안 됩니다. 삭제해도 안전하며 다음 빌드 시 다시 생성됩니다. `.gitignore`에 추가하세요.

### 4.3 bin/ 디렉토리

`bin/` 디렉토리에는 컴파일된 출력이 포함됩니다. `dotnet build`를 실행하면 여기에서 애플리케이션의 DLL과 지원 파일을 찾을 수 있습니다. 구조는 `bin/<Configuration>/<TargetFramework>/` 패턴을 따릅니다. `.gitignore`에 추가하세요.

### 4.4 .gitignore 파일

C# 프로젝트에 적합한 `.gitignore`:

```gitignore
# 빌드 출력
bin/
obj/

# IDE 파일
.vs/
.vscode/
*.user
*.suo

# OS 파일
.DS_Store
Thumbs.db
```

자동으로 생성할 수도 있습니다:

```bash
dotnet new gitignore
```

## 5. 솔루션 파일과 다중 프로젝트 설정

애플리케이션이 커지면 여러 프로젝트(메인 애플리케이션, 클래스 라이브러리, 테스트 프로젝트)를 갖게 됩니다. **솔루션 파일**(`.sln`)은 관련 프로젝트를 함께 그룹화합니다.

### 5.1 솔루션 생성

```bash
# 새 솔루션 생성
dotnet new sln -n MySolution

# 프로젝트 생성
dotnet new console -n MyApp
dotnet new classlib -n MyLibrary
dotnet new xunit -n MyTests

# 프로젝트를 솔루션에 추가
dotnet sln MySolution.sln add MyApp/MyApp.csproj
dotnet sln MySolution.sln add MyLibrary/MyLibrary.csproj
dotnet sln MySolution.sln add MyTests/MyTests.csproj
```

결과 구조:

```
MySolution/
├── MySolution.sln
├── MyApp/
│   ├── MyApp.csproj
│   └── Program.cs
├── MyLibrary/
│   ├── MyLibrary.csproj
│   └── Class1.cs
└── MyTests/
    ├── MyTests.csproj
    └── UnitTest1.cs
```

### 5.2 프로젝트 참조 추가

`MyApp`에서 `MyLibrary`를 사용하려면:

```bash
dotnet add MyApp/MyApp.csproj reference MyLibrary/MyLibrary.csproj
```

이것은 `.csproj` 파일에 `<ProjectReference>`를 추가합니다:

```xml
<ItemGroup>
  <ProjectReference Include="..\MyLibrary\MyLibrary.csproj" />
</ItemGroup>
```

이제 `MyApp`에서 `MyLibrary`의 클래스를 사용할 수 있습니다:

```csharp
// MyLibrary/MathHelper.cs
namespace MyLibrary;

public static class MathHelper
{
    public static int Add(int a, int b) => a + b;
    public static int Multiply(int a, int b) => a * b;
}
```

```csharp
// MyApp/Program.cs
using MyLibrary;

int result = MathHelper.Add(3, 5);
Console.WriteLine($"3 + 5 = {result}");
```

### 5.3 솔루션으로 빌드와 실행

```bash
# 솔루션의 모든 프로젝트 빌드
dotnet build MySolution.sln

# 특정 프로젝트 실행
dotnet run --project MyApp

# 모든 테스트 실행
dotnet test MySolution.sln
```

## 6. C# Dev Kit으로 VS Code 설정

어떤 텍스트 편집기로도 C#을 작성할 수 있지만, C# Dev Kit 확장이 포함된 Visual Studio Code는 훌륭한 경량 IDE 경험을 제공합니다.

### 6.1 확장 설치

1. VS Code 열기
2. 확장 뷰로 이동 (`Ctrl+Shift+X` / `Cmd+Shift+X`)
3. "C# Dev Kit" 검색
4. **C# Dev Kit** 확장 설치 (Microsoft 제공)

이것은 자동으로 세 가지 확장을 설치합니다:
- **C# Dev Kit** — 솔루션 탐색기, 프로젝트 관리, 테스트 탐색기
- **C#** (OmniSharp 기반) — IntelliSense, 구문 강조, 디버깅
- **IntelliCode for C# Dev Kit** — AI 지원 코드 완성

### 6.2 VS Code 구성

프로젝트나 워크스페이스에 `.vscode/settings.json`을 생성합니다:

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-dotnettools.csharp",
    "omnisharp.enableEditorConfigSupport": true,
    "dotnet.defaultSolution": "MySolution.sln"
}
```

### 6.3 VS Code에서 디버깅

C# 확장은 디버깅을 위한 `.vscode/launch.json`을 생성합니다. `F5`를 누르고 ".NET 5+ and .NET Core"를 선택하여 생성할 수 있습니다:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": ".NET Core Launch (console)",
            "type": "coreclr",
            "request": "launch",
            "preLaunchTask": "build",
            "program": "${workspaceFolder}/MyApp/bin/Debug/net8.0/MyApp.dll",
            "args": [],
            "cwd": "${workspaceFolder}/MyApp",
            "console": "integratedTerminal",
            "stopAtEntry": false
        }
    ]
}
```

줄 번호 옆의 거터를 클릭하여 중단점을 설정한 후, `F5`를 눌러 디버깅을 시작합니다. 변수를 검사하고, 코드를 단계별로 실행하며, 디버그 콘솔을 사용할 수 있습니다.

## 7. 컴파일 과정 이해

C# 코드가 소스에서 실행까지 어떻게 변환되는지 이해하면 문제를 디버깅하고 성능을 최적화하는 데 도움이 됩니다.

### 7.1 컴파일 파이프라인

```
  소스 코드 (.cs)
       │
       ▼
  ┌─────────────┐
  │  C# 컴파일러 │  (Roslyn)
  │   (csc)      │
  └──────┬──────┘
         │
         ▼
  중간 언어 (.dll)
  (IL / MSIL / CIL)
         │
         ▼
  ┌─────────────┐
  │  CLR         │  공용 언어 런타임
  │  (런타임)    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  JIT         │  적시 컴파일러
  │  컴파일러    │
  └──────┬──────┘
         │
         ▼
  네이티브 머신 코드
  (CPU에 의해 실행)
```

### 7.2 Roslyn: C# 컴파일러

Roslyn은 오픈 소스 C# 및 Visual Basic 컴파일러입니다. `dotnet build`를 실행하면 Roslyn이 `.cs` 파일을 **중간 언어(IL)**로 컴파일하며, 이는 `.dll` 어셈블리 파일에 저장됩니다.

`ildasm` 도구나 ILSpy 디컴파일러를 사용하여 IL을 검사할 수 있습니다:

```bash
# 프로젝트 빌드
dotnet build

# 출력 DLL에 IL 코드가 포함됨
# 위치: bin/Debug/net8.0/MyFirstApp.dll
```

### 7.3 공용 언어 런타임(CLR)

CLR은 .NET 프로그램의 실행을 관리하는 가상 머신입니다. 다음을 제공합니다:

- **메모리 관리**: 자동 가비지 컬렉션(Garbage Collection)
- **타입 안전성**: 런타임 타입 검사
- **예외 처리**: 구조화된 예외 지원
- **스레드 관리**: 스레드 풀과 동기화
- **보안**: 코드 접근 보안 및 검증

### 7.4 JIT 컴파일

.NET 애플리케이션을 실행하면 **적시(JIT) 컴파일러**가 IL을 즉석에서 네이티브 머신 코드로 변환합니다. 이는 각 메서드가 처음 호출될 때 메서드별로 수행됩니다:

```csharp
// 이 C# 코드가...
int Sum(int a, int b)
{
    return a + b;
}

// ...IL로 컴파일되고 (단순화)...
// IL_0000: ldarg.1
// IL_0001: ldarg.2
// IL_0002: add
// IL_0003: ret

// ...JIT가 네이티브 x64 코드로 컴파일:
// mov eax, ecx
// add eax, edx
// ret
```

### 7.5 사전 컴파일(AOT)

.NET 8은 런타임에 JIT 없이 직접 네이티브 코드로 컴파일하는 네이티브 AOT(Ahead-of-Time) 컴파일을 도입했습니다:

```bash
# 네이티브 AOT로 게시
dotnet publish -c Release -r linux-x64 /p:PublishAot=true
```

AOT는 더 작고 빠르게 시작하는 실행 파일을 생성하지만 일부 제한이 있습니다(런타임 코드 생성 불가, 제한된 리플렉션).

### 7.6 완전한 예제

좀 더 실질적인 프로그램으로 모든 것을 종합해 봅시다:

```csharp
// Program.cs — 시작하기 완전한 예제
Console.WriteLine("=== .NET 환경 정보 ===");
Console.WriteLine($"OS: {Environment.OSVersion}");
Console.WriteLine($"런타임: {Environment.Version}");
Console.WriteLine($"머신: {Environment.MachineName}");
Console.WriteLine($"64비트 OS: {Environment.Is64BitOperatingSystem}");
Console.WriteLine($"64비트 프로세스: {Environment.Is64BitProcess}");
Console.WriteLine();

Console.Write("이름이 무엇인가요? ");
string? name = Console.ReadLine();

if (!string.IsNullOrWhiteSpace(name))
{
    Console.WriteLine($"C#에 오신 것을 환영합니다, {name}!");
    Console.WriteLine($"오늘은 {DateTime.Now:dddd, MMMM dd, yyyy}입니다");
    Console.WriteLine($"현재 시각: {DateTime.Now:HH:mm:ss}");
}
else
{
    Console.WriteLine("C#에 오신 것을 환영합니다, 익명 사용자!");
}

Console.WriteLine("\n아무 키나 눌러 종료하세요...");
Console.ReadKey();
```

빌드와 실행:

```bash
dotnet build
dotnet run
```

## 8. 연습 문제

1. **SDK 탐색**: 자신의 머신에서 `dotnet --info`를 실행하고 다음을 확인하세요: (a) SDK 버전, (b) 런타임 버전, (c) 호스트 운영체제, (d) 기본 경로. 출력의 각 섹션이 의미하는 바를 요약하는 텍스트 파일을 작성하세요.

2. **템플릿 실험**: `dotnet new list`를 사용하여 사용 가능한 템플릿을 탐색하세요. 클래스 라이브러리 프로젝트(`dotnet new classlib -n MathLib`)와 콘솔 프로젝트(`dotnet new console -n MathApp`)를 생성하세요. 라이브러리에 정적 메서드 `Factorial(int n)`을 추가하고 콘솔 앱에서 호출하세요. `dotnet add reference`를 사용하여 연결하세요.

3. **두 가지 진입점 스타일**: 같은 프로그램을 두 개의 파일로 작성하세요. 프로그램은 명령줄 인수로 숫자를 받아 그 제곱을 출력해야 합니다. 최상위 문을 사용하는 버전과 전통적인 `Main` 메서드를 사용하는 버전을 만드세요. 코드를 나란히 비교하세요.

4. **빌드 구성**: `.csproj` 파일을 수정하여 `net8.0`과 `net9.0`을 동시에 대상으로 설정하세요 (힌트: 복수형 `<TargetFrameworks>` 사용). 두 대상 모두에 대해 빌드하고 `bin/` 디렉토리 구조를 살펴보세요.

5. **솔루션 설정**: 세 개의 프로젝트로 완전한 솔루션을 만드세요: 콘솔 앱(`Calculator`), 클래스 라이브러리(`CalculatorLib`), 테스트 프로젝트(`CalculatorTests`). 라이브러리에 `Add`, `Subtract`, `Multiply`, `Divide` 메서드를 추가하세요. 콘솔 앱에서 호출하고 메서드당 최소 하나의 테스트를 작성하세요.

---

**이전**: [개요](./00_Overview.md) | **다음**: [변수와 타입](./02_Variables_and_Types.md)
