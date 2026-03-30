# NuGet과 프로젝트 시스템

**이전**: [소스 생성기](./12_Source_Generators.md) | **다음**: [리플렉션과 어트리뷰트](./14_Reflection_and_Attributes.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. SDK 스타일 `.csproj` 파일을 자신 있게 이해하고 수정할 수 있다
2. 크로스 플랫폼 라이브러리를 위한 대상 프레임워크와 멀티 타겟팅을 구성할 수 있다
3. 패키지 참조와 프로젝트 참조를 구분할 수 있다
4. 패키지 관리 워크플로우에 `dotnet` CLI를 사용할 수 있다
5. `Directory.Packages.props`로 중앙 패키지 관리를 설정할 수 있다
6. `Directory.Build.props`와 `Directory.Build.targets`로 빌드 구성을 공유할 수 있다
7. 자신만의 NuGet 패키지를 만들고 게시할 수 있다
8. 여러 프로젝트로 구성된 솔루션을 효과적으로 관리할 수 있다

---

.NET 프로젝트 시스템과 NuGet 패키지 관리자는 모든 C# 애플리케이션의 근간을 이룹니다. 작은 콘솔 도구를 만들든 대규모 마이크로서비스 솔루션을 만들든, 프로젝트가 어떻게 구조화되는지, 종속성이 어떻게 해결되는지, 빌드가 어떻게 구성되는지를 이해하는 것이 생산성과 코드베이스의 유지보수성을 결정합니다. 이 레슨에서는 최신 SDK 스타일 프로젝트 형식, NuGet 생태계, 그리고 .NET이 제공하는 강력한 빌드 커스터마이징 기능을 살펴봅니다.

## 1. SDK 스타일 .csproj 파일 구조

### 1.1 최신 .csproj의 구조

.NET Core와 함께 도입된 SDK 스타일 프로젝트 파일은 레거시 형식보다 훨씬 간단합니다. 암시적 파일 포함과 합리적인 기본값을 사용합니다.

```csharp
// 콘솔 애플리케이션을 위한 최소 .csproj
// 파일: MyApp/MyApp.csproj
```

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

</Project>
```

레거시 `.csproj`와의 주요 차이점:
- 모든 `.cs` 파일을 개별적으로 나열할 필요 없음 (파일은 glob 패턴으로 자동 포함)
- `Sdk` 어트리뷰트가 모든 기본 빌드 로직을 처리
- 훨씬 적은 XML 요소가 필요

### 1.2 일반적인 속성 그룹

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <!-- 빌드 구성 -->
    <OutputType>Exe</OutputType>          <!-- Exe, Library, WinExe -->
    <TargetFramework>net9.0</TargetFramework>
    <RootNamespace>MyCompany.MyApp</RootNamespace>
    <AssemblyName>MyApp</AssemblyName>

    <!-- 언어 기능 -->
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>

    <!-- 빌드 동작 -->
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <WarningLevel>7</WarningLevel>
    <EnforceCodeStyleInBuild>true</EnforceCodeStyleInBuild>

    <!-- 출력 -->
    <PublishSingleFile>true</PublishSingleFile>
    <SelfContained>true</SelfContained>
    <RuntimeIdentifier>linux-x64</RuntimeIdentifier>
  </PropertyGroup>

</Project>
```

### 1.3 조건부 속성 그룹

빌드 구성이나 대상 프레임워크에 따라 속성을 설정할 수 있습니다:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>

  <!-- 디버그 전용 설정 -->
  <PropertyGroup Condition="'$(Configuration)' == 'Debug'">
    <DefineConstants>DEBUG;TRACE</DefineConstants>
    <DebugType>full</DebugType>
    <Optimize>false</Optimize>
  </PropertyGroup>

  <!-- 릴리스 전용 설정 -->
  <PropertyGroup Condition="'$(Configuration)' == 'Release'">
    <DefineConstants>TRACE</DefineConstants>
    <DebugType>pdbonly</DebugType>
    <Optimize>true</Optimize>
  </PropertyGroup>

</Project>
```

### 1.4 항목 그룹과 파일 포함

SDK 스타일 프로젝트는 파일을 자동으로 포함하지만, 명시적 제어가 필요할 때도 있습니다:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>

  <ItemGroup>
    <!-- 파일을 리소스로 포함 -->
    <EmbeddedResource Include="Resources\schema.json" />

    <!-- 파일을 출력 디렉토리에 복사 -->
    <Content Include="appsettings.json">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>

    <!-- 컴파일에서 파일 제외 -->
    <Compile Remove="Legacy\OldCode.cs" />

    <!-- 프로젝트 디렉토리 외부에서 추가 파일 포함 -->
    <Compile Include="..\Shared\Utilities.cs" Link="Shared\Utilities.cs" />
  </ItemGroup>

</Project>
```

## 2. 대상 프레임워크와 멀티 타겟팅

### 2.1 대상 프레임워크 모니커 (TFM)

각 .NET 버전에는 프로젝트에서 지정하는 대상 프레임워크 모니커(Target Framework Moniker)가 있습니다:

```xml
<!-- 단일 대상 -->
<TargetFramework>net8.0</TargetFramework>

<!-- 일반적인 TFM:
     net8.0        - .NET 8
     net9.0        - .NET 9
     netstandard2.0 - .NET Standard 2.0 (넓은 호환성)
     netstandard2.1 - .NET Standard 2.1
     net48         - .NET Framework 4.8
-->
```

### 2.2 멀티 타겟팅

여러 .NET 버전을 지원해야 하는 라이브러리는 멀티 타겟팅을 사용합니다. 복수형 `TargetFrameworks`에 주목하세요:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <!-- 참고: 세미콜론으로 구분된 TargetFrameworks (복수형) -->
    <TargetFrameworks>net8.0;net9.0;netstandard2.0</TargetFrameworks>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <!-- 이전 프레임워크를 위한 조건부 패키지 참조 -->
  <ItemGroup Condition="'$(TargetFramework)' == 'netstandard2.0'">
    <PackageReference Include="System.Text.Json" Version="8.0.0" />
  </ItemGroup>

</Project>
```

### 2.3 멀티 타겟팅에서의 전처리기 지시문

멀티 타겟팅 시 전처리기 지시문(Preprocessor Directive)을 사용하여 API 차이를 처리할 수 있습니다:

```csharp
public static class StringHelper
{
    public static bool ContainsIgnoreCase(string source, string value)
    {
#if NET8_0_OR_GREATER
        // .NET 8+에는 이 오버로드가 내장되어 있음
        return source.Contains(value, StringComparison.OrdinalIgnoreCase);
#elif NETSTANDARD2_0
        // .NET Standard 2.0에는 해당 오버로드가 없음
        return source.IndexOf(value, StringComparison.OrdinalIgnoreCase) >= 0;
#endif
    }

    public static string TruncateAt(string input, int maxLength)
    {
#if NET9_0_OR_GREATER
        // .NET 9의 가상 신규 API
        return string.Truncate(input, maxLength);
#else
        if (input.Length <= maxLength) return input;
        return input[..maxLength];
#endif
    }
}
```

## 3. 패키지 참조 vs 프로젝트 참조

### 3.1 패키지 참조

패키지 참조(Package Reference)는 NuGet 피드에서 미리 빌드된 라이브러리를 가져옵니다:

```xml
<ItemGroup>
  <!-- 명시적 버전이 있는 기본 패키지 참조 -->
  <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />

  <!-- 버전 범위 -->
  <PackageReference Include="Serilog" Version="[3.0.0, 4.0.0)" />

  <!-- 유동 버전 (최신 패치) -->
  <PackageReference Include="Dapper" Version="2.1.*" />

  <!-- 비공개 자산: 빌드 시에만 사용, 소비자에게 노출되지 않음 -->
  <PackageReference Include="Microsoft.SourceLink.GitHub" Version="8.0.0" PrivateAssets="All" />
</ItemGroup>
```

### 3.2 프로젝트 참조

프로젝트 참조(Project Reference)는 솔루션의 다른 프로젝트에 연결하여 소스 수준 종속성을 가능하게 합니다:

```xml
<ItemGroup>
  <!-- 형제 프로젝트 참조 -->
  <ProjectReference Include="..\MyApp.Core\MyApp.Core.csproj" />

  <!-- 출력 항목 유형이 있는 참조 -->
  <ProjectReference Include="..\MyApp.Analyzers\MyApp.Analyzers.csproj"
                    OutputItemType="Analyzer"
                    ReferenceOutputAssembly="false" />
</ItemGroup>
```

### 3.3 어떤 것을 사용할지

```csharp
// 시나리오 1: 제어할 수 없는 서드파티 라이브러리
// => PackageReference 사용
// 예시: Newtonsoft.Json, Serilog, Dapper

// 시나리오 2: 같은 솔루션에 있는 자체 코드
// => ProjectReference 사용
// 예시: MyApp.Core, MyApp.Data, MyApp.Tests

// 시나리오 3: 비공개 피드에 게시된 내부 회사 라이브러리
// => 비공개 NuGet 소스와 함께 PackageReference 사용
// 예시: CompanyName.SharedKernel

// 시나리오 4: 분석기 또는 소스 생성기 프로젝트
// => OutputItemType="Analyzer"와 함께 ProjectReference 사용
```

## 4. NuGet 기초

### 4.1 NuGet이란?

NuGet은 .NET의 패키지 관리자입니다. 패키지는 컴파일된 DLL, 메타데이터, 콘텐츠 파일을 포함하는 `.nupkg` 파일(ZIP 아카이브)로 배포됩니다.

```csharp
// NuGet 복원 프로세스:
// 1. .csproj에서 PackageReference 읽기
// 2. 버전 제약 조건과 전이적 종속성 해결
// 3. 글로벌 캐시(~/.nuget/packages)에 패키지 다운로드
// 4. obj/project.assets.json (종속성 그래프) 생성
// 5. 컴파일을 위해 어셈블리 사용 가능하게 함
```

### 4.2 NuGet 구성 (nuget.config)

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <!-- 패키지 소스 -->
  <packageSources>
    <clear />  <!-- 상속된 소스 초기화 -->
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
    <add key="MyCompanyFeed" value="https://pkgs.dev.azure.com/mycompany/_packaging/myfeed/nuget/v3/index.json" />
    <add key="LocalPackages" value="./local-packages" />
  </packageSources>

  <!-- 인증이 필요한 피드의 자격 증명 -->
  <packageSourceCredentials>
    <MyCompanyFeed>
      <add key="Username" value="user@company.com" />
      <add key="ClearTextPassword" value="%MY_NUGET_TOKEN%" />
    </MyCompanyFeed>
  </packageSourceCredentials>

  <!-- 글로벌 패키지 관리 설정 -->
  <packageManagement>
    <add key="format" value="1" />
    <add key="disabled" value="false" />
  </packageManagement>
</configuration>
```

### 4.3 패키지 복원

복원은 빌드 중 자동으로 수행되지만, 명시적으로 트리거할 수도 있습니다:

```bash
# 솔루션의 모든 패키지 복원
dotnet restore

# 특정 프로젝트 복원
dotnet restore src/MyApp/MyApp.csproj

# 특정 소스로 복원
dotnet restore --source https://api.nuget.org/v3/index.json

# 종속성 재평가 강제
dotnet restore --force

# 로컬 캐시 초기화 후 복원
dotnet nuget locals all --clear
dotnet restore
```

## 5. 패키지 관리를 위한 dotnet CLI

### 5.1 패키지 추가

```bash
# 패키지 추가 (최신 안정 버전)
dotnet add package Newtonsoft.Json

# 특정 버전 추가
dotnet add package Serilog --version 3.1.1

# 프리릴리스 패키지 추가
dotnet add package Microsoft.Extensions.Logging --prerelease

# 특정 프로젝트에 패키지 추가
dotnet add src/MyApp/MyApp.csproj package Dapper

# 특정 소스에서 패키지 추가
dotnet add package MyCompany.Shared --source https://pkgs.dev.azure.com/myco/feed/nuget/v3/index.json
```

### 5.2 패키지 나열

```bash
# 프로젝트의 모든 패키지 나열
dotnet list package

# 전체 솔루션의 패키지 나열
dotnet list MySolution.sln package

# 전이적 (간접) 종속성 표시
dotnet list package --include-transitive

# 오래된 패키지 확인
dotnet list package --outdated

# 알려진 취약점이 있는 패키지 확인
dotnet list package --vulnerable

# 사용 가능한 업데이트가 있는 최상위 패키지만 표시
dotnet list package --outdated --highest-minor
```

### 5.3 패키지 제거 및 업데이트

```bash
# 패키지 제거
dotnet remove package Newtonsoft.Json

# 패키지 업데이트 (제거 후 새 버전으로 다시 추가)
dotnet add package Serilog --version 4.0.0

# "dotnet update package" 명령은 없습니다.
# 모든 패키지를 업데이트하려면 dotnet-outdated 도구를 사용하세요:
dotnet tool install --global dotnet-outdated-tool
dotnet outdated --upgrade
```

### 5.4 글로벌 도구 관리

```bash
# 글로벌 도구 설치
dotnet tool install --global dotnet-ef

# 설치된 글로벌 도구 나열
dotnet tool list --global

# 글로벌 도구 업데이트
dotnet tool update --global dotnet-ef

# 글로벌 도구 제거
dotnet tool uninstall --global dotnet-ef

# 로컬 도구 (리포지토리별)
dotnet new tool-manifest   # .config/dotnet-tools.json 생성
dotnet tool install dotnet-ef
dotnet tool restore        # 매니페스트에 나열된 도구 복원
```

## 6. 중앙 패키지 관리

### 6.1 문제점

많은 프로젝트가 있는 대규모 솔루션에서 각 프로젝트가 자체 패키지 버전을 지정하면 버전 불일치가 발생합니다:

```xml
<!-- 프로젝트 A: MyApp.Core.csproj -->
<PackageReference Include="Newtonsoft.Json" Version="13.0.1" />

<!-- 프로젝트 B: MyApp.Api.csproj -->
<PackageReference Include="Newtonsoft.Json" Version="13.0.3" />

<!-- 프로젝트 C: MyApp.Tests.csproj -->
<PackageReference Include="Newtonsoft.Json" Version="12.0.3" />
<!-- 이런! 같은 패키지의 세 가지 다른 버전 -->
```

### 6.2 Directory.Packages.props

중앙 패키지 관리(CPM, Central Package Management)는 솔루션 루트에 단일 `Directory.Packages.props` 파일로 이 문제를 해결합니다:

```xml
<!-- Directory.Packages.props (솔루션 루트) -->
<Project>
  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
  </PropertyGroup>

  <ItemGroup>
    <!-- 버전을 중앙에서 정의 -->
    <PackageVersion Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageVersion Include="Serilog" Version="3.1.1" />
    <PackageVersion Include="Serilog.Sinks.Console" Version="5.0.1" />
    <PackageVersion Include="Dapper" Version="2.1.35" />
    <PackageVersion Include="Microsoft.Extensions.DependencyInjection" Version="8.0.0" />

    <!-- 테스트 패키지 -->
    <PackageVersion Include="xunit" Version="2.7.0" />
    <PackageVersion Include="xunit.runner.visualstudio" Version="2.5.7" />
    <PackageVersion Include="Moq" Version="4.20.70" />
    <PackageVersion Include="FluentAssertions" Version="6.12.0" />
    <PackageVersion Include="Microsoft.NET.Test.Sdk" Version="17.9.0" />
  </ItemGroup>
</Project>
```

이제 개별 프로젝트는 버전을 지정하지 않고 패키지를 참조합니다:

```xml
<!-- MyApp.Core.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <!-- Version 어트리뷰트가 필요 없음! Directory.Packages.props에서 가져옴 -->
    <PackageReference Include="Newtonsoft.Json" />
    <PackageReference Include="Serilog" />
  </ItemGroup>
</Project>
```

### 6.3 버전 재정의

드문 경우에 특정 프로젝트에서 다른 버전이 필요할 수 있습니다:

```xml
<!-- MyApp.Legacy.csproj -->
<ItemGroup>
  <!-- 이 프로젝트에서만 중앙 관리 버전을 재정의 -->
  <PackageReference Include="Newtonsoft.Json" VersionOverride="12.0.3" />
</ItemGroup>
```

## 7. Directory.Build.props와 Directory.Build.targets

### 7.1 Directory.Build.props

이 파일은 해당 디렉토리 트리의 모든 프로젝트 시작 부분에 자동으로 가져옵니다. 공통 속성을 공유하는 데 사용합니다:

```xml
<!-- Directory.Build.props (솔루션 루트) -->
<Project>
  <PropertyGroup>
    <!-- 모든 프로젝트의 공유 설정 -->
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>

    <!-- NuGet 게시를 위한 패키지 메타데이터 -->
    <Authors>My Company</Authors>
    <Company>My Company Inc.</Company>
    <Copyright>Copyright (c) 2025 My Company Inc.</Copyright>
    <RepositoryUrl>https://github.com/mycompany/myapp</RepositoryUrl>
  </PropertyGroup>
</Project>
```

### 7.2 Directory.Build.targets

이 파일은 모든 프로젝트의 끝 부분에 가져옵니다. 평가된 속성에 의존하는 사용자 지정 빌드 로직에 사용합니다:

```xml
<!-- Directory.Build.targets (솔루션 루트) -->
<Project>
  <!-- 모든 프로젝트에서 분석기 실행 -->
  <ItemGroup>
    <PackageReference Include="Microsoft.CodeAnalysis.NetAnalyzers" Version="8.0.0">
      <PrivateAssets>all</PrivateAssets>
      <IncludeAssets>runtime; build; native; contentfiles; analyzers</IncludeAssets>
    </PackageReference>
  </ItemGroup>

  <!-- 사용자 지정 타겟: 빌드 시간을 어셈블리에 기록 -->
  <Target Name="StampBuildTime" BeforeTargets="CoreCompile">
    <PropertyGroup>
      <BuildTimestamp>$([System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"))</BuildTimestamp>
    </PropertyGroup>
    <ItemGroup>
      <AssemblyAttribute Include="System.Reflection.AssemblyMetadataAttribute">
        <_Parameter1>BuildTimestamp</_Parameter1>
        <_Parameter2>$(BuildTimestamp)</_Parameter2>
      </AssemblyAttribute>
    </ItemGroup>
  </Target>
</Project>
```

### 7.3 계층적 가져오기

`Directory.Build.props`와 `Directory.Build.targets` 파일은 중첩될 수 있습니다. 내부 파일은 외부 파일을 자동으로 가져오지 않으므로 명시적으로 수행해야 합니다:

```xml
<!-- src/Directory.Build.props -->
<Project>
  <!-- 먼저 상위 Directory.Build.props를 가져옴 -->
  <Import Project="$([MSBuild]::GetPathOfFileAbove('Directory.Build.props', '$(MSBuildThisFileDirectory)../'))" />

  <PropertyGroup>
    <!-- src/ 프로젝트에만 적용되는 추가 설정 -->
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
  </PropertyGroup>
</Project>
```

```xml
<!-- tests/Directory.Build.props -->
<Project>
  <Import Project="$([MSBuild]::GetPathOfFileAbove('Directory.Build.props', '$(MSBuildThisFileDirectory)../'))" />

  <PropertyGroup>
    <!-- 테스트 프로젝트는 경고를 오류로 처리하지 않아야 함 -->
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
    <IsPackable>false</IsPackable>
  </PropertyGroup>
</Project>
```

## 8. NuGet 패키지 만들기와 게시하기

### 8.1 .csproj의 패키지 메타데이터

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFrameworks>net8.0;netstandard2.0</TargetFrameworks>
    <Nullable>enable</Nullable>

    <!-- NuGet 패키지 메타데이터 -->
    <PackageId>MyCompany.Utilities</PackageId>
    <Version>1.2.0</Version>
    <Authors>Jane Developer</Authors>
    <Description>A collection of utility classes for common operations.</Description>
    <PackageTags>utilities;helpers;extensions</PackageTags>
    <PackageLicenseExpression>MIT</PackageLicenseExpression>
    <PackageReadmeFile>README.md</PackageReadmeFile>
    <PackageProjectUrl>https://github.com/mycompany/utilities</PackageProjectUrl>
    <RepositoryUrl>https://github.com/mycompany/utilities</RepositoryUrl>
    <RepositoryType>git</RepositoryType>

    <!-- 빌드 시 패키지 생성 -->
    <GeneratePackageOnBuild>false</GeneratePackageOnBuild>

    <!-- 디버깅을 위한 심볼 포함 -->
    <IncludeSymbols>true</IncludeSymbols>
    <SymbolPackageFormat>snupkg</SymbolPackageFormat>

    <!-- XML 문서 생성 -->
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
  </PropertyGroup>

  <!-- 패키지에 README 포함 -->
  <ItemGroup>
    <None Include="README.md" Pack="true" PackagePath="\" />
  </ItemGroup>

</Project>
```

### 8.2 빌드와 패킹

```bash
# NuGet 패키지 생성
dotnet pack --configuration Release

# 특정 버전으로 패킹
dotnet pack --configuration Release /p:Version=1.2.0

# 출력은 bin/Release/*.nupkg로 이동

# 패키지 내용 검사
dotnet nuget verify bin/Release/MyCompany.Utilities.1.2.0.nupkg
```

### 8.3 NuGet.org에 게시

```bash
# nuget.org에 푸시
dotnet nuget push bin/Release/MyCompany.Utilities.1.2.0.nupkg \
  --api-key YOUR_API_KEY \
  --source https://api.nuget.org/v3/index.json

# 비공개 피드에 푸시
dotnet nuget push bin/Release/MyCompany.Utilities.1.2.0.nupkg \
  --source MyCompanyFeed

# 심볼 패키지와 함께 푸시
dotnet nuget push bin/Release/MyCompany.Utilities.1.2.0.snupkg \
  --api-key YOUR_API_KEY \
  --source https://api.nuget.org/v3/index.json
```

### 8.4 완전한 라이브러리 예제

```csharp
// 파일: src/MyCompany.Utilities/StringExtensions.cs
namespace MyCompany.Utilities;

/// <summary>
/// 문자열 조작을 위한 확장 메서드.
/// </summary>
public static class StringExtensions
{
    /// <summary>
    /// 문자열을 지정된 최대 길이로 잘라냅니다.
    /// </summary>
    /// <param name="value">잘라낼 문자열.</param>
    /// <param name="maxLength">최대 문자 수.</param>
    /// <param name="suffix">잘린 경우 추가할 접미사 (기본값: "...").</param>
    /// <returns>잘린 문자열.</returns>
    public static string Truncate(this string value, int maxLength, string suffix = "...")
    {
        ArgumentNullException.ThrowIfNull(value);
        ArgumentOutOfRangeException.ThrowIfNegative(maxLength);

        if (value.Length <= maxLength) return value;
        if (maxLength <= suffix.Length) return suffix[..maxLength];

        return string.Concat(value.AsSpan(0, maxLength - suffix.Length), suffix);
    }

    /// <summary>
    /// 문자열을 슬러그 형식(URL 친화적)으로 변환합니다.
    /// </summary>
    public static string ToSlug(this string value)
    {
        ArgumentNullException.ThrowIfNull(value);

        return value
            .ToLowerInvariant()
            .Replace(' ', '-')
            .Where(c => char.IsLetterOrDigit(c) || c == '-')
            .Aggregate(new System.Text.StringBuilder(), (sb, c) => sb.Append(c))
            .ToString()
            .Trim('-');
    }
}
```

## 9. 전역 Using과 암시적 Using

### 9.1 암시적 Using

`ImplicitUsings`가 활성화되면 SDK가 프로젝트 유형에 따라 일반적인 `using` 지시문을 자동으로 추가합니다:

```xml
<!-- .csproj에서 활성화 -->
<ImplicitUsings>enable</ImplicitUsings>
```

```csharp
// Microsoft.NET.Sdk의 경우 다음이 암시적으로 포함됩니다:
// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// using System.Net.Http;
// using System.Threading;
// using System.Threading.Tasks;

// 따라서 using 문 없이 이렇게 작성할 수 있습니다:
List<int> numbers = [1, 2, 3, 4, 5];
var doubled = numbers.Select(n => n * 2).ToList();
Console.WriteLine(string.Join(", ", doubled));
```

### 9.2 .csproj에서 사용자 지정 전역 Using

```xml
<ItemGroup>
  <!-- 프로젝트 파일을 통해 전역 using 추가 -->
  <Using Include="System.Text.Json" />
  <Using Include="Microsoft.Extensions.Logging" />

  <!-- 별칭이 있는 전역 using -->
  <Using Include="System.Text.Json.JsonSerializer" Alias="Json" />

  <!-- 암시적 using 제거 -->
  <Using Remove="System.Net.Http" />
</ItemGroup>
```

### 9.3 GlobalUsings.cs 파일

또는 전용 파일에서 전역 using을 선언할 수 있습니다:

```csharp
// 파일: GlobalUsings.cs
global using System.Text.Json;
global using System.Text.Json.Serialization;
global using Microsoft.Extensions.Logging;
global using MyApp.Core.Models;
global using MyApp.Core.Interfaces;

// 별칭이 있는 전역 using
global using JsonOptions = System.Text.Json.JsonSerializerOptions;
```

## 10. 솔루션 파일 (.sln) 관리

### 10.1 솔루션 만들기와 관리하기

```bash
# 새 솔루션 생성
dotnet new sln --name MyApp

# 프로젝트 구조 생성
dotnet new classlib -o src/MyApp.Core
dotnet new classlib -o src/MyApp.Data
dotnet new webapi -o src/MyApp.Api
dotnet new xunit -o tests/MyApp.Core.Tests
dotnet new xunit -o tests/MyApp.Api.Tests

# 솔루션에 프로젝트 추가
dotnet sln add src/MyApp.Core/MyApp.Core.csproj
dotnet sln add src/MyApp.Data/MyApp.Data.csproj
dotnet sln add src/MyApp.Api/MyApp.Api.csproj
dotnet sln add tests/MyApp.Core.Tests/MyApp.Core.Tests.csproj
dotnet sln add tests/MyApp.Api.Tests/MyApp.Api.Tests.csproj

# 솔루션 폴더에 프로젝트 추가
dotnet sln add src/MyApp.Core/MyApp.Core.csproj --solution-folder src
dotnet sln add tests/MyApp.Core.Tests/MyApp.Core.Tests.csproj --solution-folder tests

# 솔루션의 프로젝트 나열
dotnet sln list

# 솔루션에서 프로젝트 제거
dotnet sln remove tests/MyApp.Api.Tests/MyApp.Api.Tests.csproj
```

### 10.2 프로젝트 참조 추가

```bash
# 프로젝트 참조 추가
dotnet add src/MyApp.Data/MyApp.Data.csproj reference src/MyApp.Core/MyApp.Core.csproj
dotnet add src/MyApp.Api/MyApp.Api.csproj reference src/MyApp.Core/MyApp.Core.csproj
dotnet add src/MyApp.Api/MyApp.Api.csproj reference src/MyApp.Data/MyApp.Data.csproj

# 테스트 프로젝트 참조 추가
dotnet add tests/MyApp.Core.Tests/MyApp.Core.Tests.csproj reference src/MyApp.Core/MyApp.Core.csproj

# 참조 나열
dotnet list src/MyApp.Api/MyApp.Api.csproj reference
```

### 10.3 솔루션 빌드

```bash
# 전체 솔루션 빌드
dotnet build

# 릴리스 모드로 빌드
dotnet build --configuration Release

# 특정 프로젝트 빌드
dotnet build src/MyApp.Api/MyApp.Api.csproj

# 솔루션의 모든 테스트 실행
dotnet test

# API 프로젝트 게시
dotnet publish src/MyApp.Api/MyApp.Api.csproj -c Release -o ./publish
```

## 11. 실전 예제: 멀티 프로젝트 솔루션 설정

이 레슨에서 다룬 모든 패턴을 사용하여 완전한 멀티 프로젝트 솔루션을 처음부터 구축해 보겠습니다.

### 11.1 솔루션 구조

```
MyShop/
├── Directory.Build.props
├── Directory.Build.targets
├── Directory.Packages.props
├── nuget.config
├── MyShop.sln
├── src/
│   ├── MyShop.Core/          # 도메인 모델과 인터페이스
│   ├── MyShop.Data/          # 데이터 접근 (EF Core)
│   └── MyShop.Api/           # Web API
├── tests/
│   └── MyShop.Core.Tests/    # 단위 테스트
└── global.json
```

### 11.2 global.json

```json
{
  "sdk": {
    "version": "8.0.100",
    "rollForward": "latestMinor"
  }
}
```

### 11.3 공유 구성 파일

```xml
<!-- Directory.Build.props -->
<Project>
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>

    <Company>MyShop Inc.</Company>
    <Authors>MyShop Team</Authors>
  </PropertyGroup>
</Project>
```

```xml
<!-- Directory.Packages.props -->
<Project>
  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
  </PropertyGroup>
  <ItemGroup>
    <PackageVersion Include="Microsoft.EntityFrameworkCore" Version="8.0.4" />
    <PackageVersion Include="Microsoft.EntityFrameworkCore.Sqlite" Version="8.0.4" />
    <PackageVersion Include="Microsoft.EntityFrameworkCore.Design" Version="8.0.4" />
    <PackageVersion Include="FluentValidation" Version="11.9.0" />
    <PackageVersion Include="xunit" Version="2.7.0" />
    <PackageVersion Include="xunit.runner.visualstudio" Version="2.5.7" />
    <PackageVersion Include="Microsoft.NET.Test.Sdk" Version="17.9.0" />
    <PackageVersion Include="FluentAssertions" Version="6.12.0" />
  </ItemGroup>
</Project>
```

### 11.4 코어 라이브러리

```xml
<!-- src/MyShop.Core/MyShop.Core.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="FluentValidation" />
  </ItemGroup>
</Project>
```

```csharp
// src/MyShop.Core/Models/Product.cs
namespace MyShop.Core.Models;

public class Product
{
    public int Id { get; set; }
    public required string Name { get; set; }
    public string? Description { get; set; }
    public decimal Price { get; set; }
    public int StockQuantity { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }
}
```

```csharp
// src/MyShop.Core/Interfaces/IProductRepository.cs
namespace MyShop.Core.Interfaces;

using MyShop.Core.Models;

public interface IProductRepository
{
    Task<Product?> GetByIdAsync(int id);
    Task<IReadOnlyList<Product>> GetAllAsync();
    Task<Product> CreateAsync(Product product);
    Task<Product> UpdateAsync(Product product);
    Task<bool> DeleteAsync(int id);
}
```

```csharp
// src/MyShop.Core/Validators/ProductValidator.cs
namespace MyShop.Core.Validators;

using FluentValidation;
using MyShop.Core.Models;

public class ProductValidator : AbstractValidator<Product>
{
    public ProductValidator()
    {
        RuleFor(p => p.Name)
            .NotEmpty().WithMessage("Product name is required")
            .MaximumLength(200).WithMessage("Product name cannot exceed 200 characters");

        RuleFor(p => p.Price)
            .GreaterThan(0).WithMessage("Price must be greater than zero");

        RuleFor(p => p.StockQuantity)
            .GreaterThanOrEqualTo(0).WithMessage("Stock cannot be negative");
    }
}
```

### 11.5 데이터 라이브러리

```xml
<!-- src/MyShop.Data/MyShop.Data.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <ProjectReference Include="..\MyShop.Core\MyShop.Core.csproj" />
    <PackageReference Include="Microsoft.EntityFrameworkCore" />
    <PackageReference Include="Microsoft.EntityFrameworkCore.Sqlite" />
  </ItemGroup>
</Project>
```

```csharp
// src/MyShop.Data/ShopDbContext.cs
namespace MyShop.Data;

using Microsoft.EntityFrameworkCore;
using MyShop.Core.Models;

public class ShopDbContext : DbContext
{
    public ShopDbContext(DbContextOptions<ShopDbContext> options) : base(options) { }

    public DbSet<Product> Products => Set<Product>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Product>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Name).IsRequired().HasMaxLength(200);
            entity.Property(e => e.Price).HasColumnType("decimal(18,2)");
        });
    }
}
```

```csharp
// src/MyShop.Data/Repositories/ProductRepository.cs
namespace MyShop.Data.Repositories;

using Microsoft.EntityFrameworkCore;
using MyShop.Core.Interfaces;
using MyShop.Core.Models;

public class ProductRepository : IProductRepository
{
    private readonly ShopDbContext _context;

    public ProductRepository(ShopDbContext context) => _context = context;

    public async Task<Product?> GetByIdAsync(int id)
        => await _context.Products.FindAsync(id);

    public async Task<IReadOnlyList<Product>> GetAllAsync()
        => await _context.Products.OrderBy(p => p.Name).ToListAsync();

    public async Task<Product> CreateAsync(Product product)
    {
        _context.Products.Add(product);
        await _context.SaveChangesAsync();
        return product;
    }

    public async Task<Product> UpdateAsync(Product product)
    {
        product.UpdatedAt = DateTime.UtcNow;
        _context.Products.Update(product);
        await _context.SaveChangesAsync();
        return product;
    }

    public async Task<bool> DeleteAsync(int id)
    {
        var product = await _context.Products.FindAsync(id);
        if (product is null) return false;
        _context.Products.Remove(product);
        await _context.SaveChangesAsync();
        return true;
    }
}
```

### 11.6 테스트 프로젝트

```xml
<!-- tests/MyShop.Core.Tests/MyShop.Core.Tests.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <IsPackable>false</IsPackable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\..\src\MyShop.Core\MyShop.Core.csproj" />
    <PackageReference Include="xunit" />
    <PackageReference Include="xunit.runner.visualstudio" />
    <PackageReference Include="Microsoft.NET.Test.Sdk" />
    <PackageReference Include="FluentAssertions" />
  </ItemGroup>
</Project>
```

```csharp
// tests/MyShop.Core.Tests/Validators/ProductValidatorTests.cs
namespace MyShop.Core.Tests.Validators;

using FluentAssertions;
using FluentValidation.TestHelper;
using MyShop.Core.Models;
using MyShop.Core.Validators;

public class ProductValidatorTests
{
    private readonly ProductValidator _validator = new();

    [Fact]
    public void Valid_product_passes_validation()
    {
        var product = new Product { Name = "Widget", Price = 9.99m, StockQuantity = 10 };
        var result = _validator.TestValidate(product);
        result.ShouldNotHaveAnyValidationErrors();
    }

    [Fact]
    public void Empty_name_fails_validation()
    {
        var product = new Product { Name = "", Price = 9.99m };
        var result = _validator.TestValidate(product);
        result.ShouldHaveValidationErrorFor(p => p.Name);
    }

    [Fact]
    public void Negative_price_fails_validation()
    {
        var product = new Product { Name = "Widget", Price = -1m };
        var result = _validator.TestValidate(product);
        result.ShouldHaveValidationErrorFor(p => p.Price);
    }

    [Fact]
    public void Negative_stock_fails_validation()
    {
        var product = new Product { Name = "Widget", Price = 5m, StockQuantity = -1 };
        var result = _validator.TestValidate(product);
        result.ShouldHaveValidationErrorFor(p => p.StockQuantity);
    }
}
```

## 12. 연습 문제

1. **프로젝트 파일 수정**: `net8.0`을 대상으로 하는 `.csproj` 파일이 주어졌을 때, `net8.0`과 `netstandard2.0`을 멀티 타겟팅하도록 수정하세요. 조건부 컴파일을 추가하여 메서드가 `net8.0`에서는 `ReadOnlySpan<char>`를, `netstandard2.0`에서는 `string.Substring`을 사용하도록 만드세요.

2. **중앙 패키지 관리**: 5개의 프로젝트가 있는 솔루션에서 각각 다양한 버전의 `Serilog`, `Dapper`, `FluentValidation`을 참조하고 있습니다. 모든 버전을 중앙 집중화하는 `Directory.Packages.props` 파일을 만들고, 프로젝트 파일 중 하나가 어떻게 변경되는지 보여주세요.

3. **NuGet 패키지 생성**: 날짜/시간 유틸리티 메서드(`IsWeekend`, `GetNextBusinessDay`, `FormatRelative`)를 제공하는 클래스 라이브러리를 만드세요. 모든 NuGet 메타데이터가 포함된 완전한 `.csproj`를 작성한 다음, 패킹하고 게시하는 명령을 보여주세요.

4. **Directory.Build.props 계층 구조**: 모든 프로젝트가 nullable 참조와 최신 언어 버전을 공유하고, `src/` 프로젝트는 XML 문서를 생성하며, `tests/` 프로젝트는 `TreatWarningsAsErrors`를 비활성화하는 솔루션을 위한 `Directory.Build.props` 구조를 설계하세요. 세 파일 모두를 보여주세요.

5. **처음부터 솔루션 만들기**: `dotnet` CLI 명령만 사용하여 `BlogEngine.Core` (클래스 라이브러리), `BlogEngine.Api` (웹 API), `BlogEngine.Tests` (xunit) 프로젝트를 포함하는 `BlogEngine` 솔루션을 만드세요. 필요한 프로젝트 참조(Api는 Core에 의존, Tests는 Core에 의존)를 추가하세요. 모든 명령을 순서대로 나열하세요.
