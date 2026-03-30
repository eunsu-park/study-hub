# NuGet and Project System

**Previous**: [Source Generators](./12_Source_Generators.md) | **Next**: [Reflection and Attributes](./14_Reflection_and_Attributes.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand and modify SDK-style `.csproj` files confidently
2. Configure target frameworks and multi-targeting for cross-platform libraries
3. Distinguish between package references and project references
4. Use the `dotnet` CLI for package management workflows
5. Set up Central Package Management with `Directory.Packages.props`
6. Share build configuration with `Directory.Build.props` and `Directory.Build.targets`
7. Create and publish your own NuGet packages
8. Manage solutions with multiple projects effectively

---

The .NET project system and NuGet package manager form the backbone of every C# application. Whether you are building a small console tool or a large microservices solution, understanding how projects are structured, how dependencies are resolved, and how builds are configured determines your productivity and the maintainability of your codebase. This lesson explores the modern SDK-style project format, the NuGet ecosystem, and the powerful build customization features that .NET provides.

## 1. SDK-Style .csproj File Structure

### 1.1 Anatomy of a Modern .csproj

The SDK-style project file, introduced with .NET Core, is dramatically simpler than the legacy format. It uses implicit file inclusion and sensible defaults.

```csharp
// A minimal .csproj for a console application
// File: MyApp/MyApp.csproj
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

Key differences from legacy `.csproj`:
- No need to list every `.cs` file individually (files are included by glob patterns automatically)
- The `Sdk` attribute handles all default build logic
- Far fewer XML elements required

### 1.2 Common Property Groups

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <!-- Build configuration -->
    <OutputType>Exe</OutputType>          <!-- Exe, Library, WinExe -->
    <TargetFramework>net9.0</TargetFramework>
    <RootNamespace>MyCompany.MyApp</RootNamespace>
    <AssemblyName>MyApp</AssemblyName>

    <!-- Language features -->
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>

    <!-- Build behavior -->
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <WarningLevel>7</WarningLevel>
    <EnforceCodeStyleInBuild>true</EnforceCodeStyleInBuild>

    <!-- Output -->
    <PublishSingleFile>true</PublishSingleFile>
    <SelfContained>true</SelfContained>
    <RuntimeIdentifier>linux-x64</RuntimeIdentifier>
  </PropertyGroup>

</Project>
```

### 1.3 Conditional Property Groups

You can set properties based on the build configuration or target framework:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>

  <!-- Debug-only settings -->
  <PropertyGroup Condition="'$(Configuration)' == 'Debug'">
    <DefineConstants>DEBUG;TRACE</DefineConstants>
    <DebugType>full</DebugType>
    <Optimize>false</Optimize>
  </PropertyGroup>

  <!-- Release-only settings -->
  <PropertyGroup Condition="'$(Configuration)' == 'Release'">
    <DefineConstants>TRACE</DefineConstants>
    <DebugType>pdbonly</DebugType>
    <Optimize>true</Optimize>
  </PropertyGroup>

</Project>
```

### 1.4 Item Groups and File Inclusion

While SDK-style projects include files automatically, you sometimes need explicit control:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>

  <ItemGroup>
    <!-- Embed a file as a resource -->
    <EmbeddedResource Include="Resources\schema.json" />

    <!-- Copy a file to the output directory -->
    <Content Include="appsettings.json">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>

    <!-- Exclude a file from compilation -->
    <Compile Remove="Legacy\OldCode.cs" />

    <!-- Include additional files from outside the project directory -->
    <Compile Include="..\Shared\Utilities.cs" Link="Shared\Utilities.cs" />
  </ItemGroup>

</Project>
```

## 2. Target Frameworks and Multi-Targeting

### 2.1 Target Framework Monikers (TFMs)

Each .NET version has a Target Framework Moniker that you specify in your project:

```xml
<!-- Single target -->
<TargetFramework>net8.0</TargetFramework>

<!-- Common TFMs:
     net8.0        - .NET 8
     net9.0        - .NET 9
     netstandard2.0 - .NET Standard 2.0 (broad compatibility)
     netstandard2.1 - .NET Standard 2.1
     net48         - .NET Framework 4.8
-->
```

### 2.2 Multi-Targeting

Libraries that need to support multiple .NET versions use multi-targeting. Note the plural `TargetFrameworks`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <!-- Note: TargetFrameworks (plural) with semicolon separator -->
    <TargetFrameworks>net8.0;net9.0;netstandard2.0</TargetFrameworks>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <!-- Conditional package reference for older frameworks -->
  <ItemGroup Condition="'$(TargetFramework)' == 'netstandard2.0'">
    <PackageReference Include="System.Text.Json" Version="8.0.0" />
  </ItemGroup>

</Project>
```

### 2.3 Preprocessor Directives with Multi-Targeting

When multi-targeting, you can use preprocessor directives to handle API differences:

```csharp
public static class StringHelper
{
    public static bool ContainsIgnoreCase(string source, string value)
    {
#if NET8_0_OR_GREATER
        // .NET 8+ has this overload built-in
        return source.Contains(value, StringComparison.OrdinalIgnoreCase);
#elif NETSTANDARD2_0
        // .NET Standard 2.0 lacks the overload
        return source.IndexOf(value, StringComparison.OrdinalIgnoreCase) >= 0;
#endif
    }

    public static string TruncateAt(string input, int maxLength)
    {
#if NET9_0_OR_GREATER
        // Hypothetical new API in .NET 9
        return string.Truncate(input, maxLength);
#else
        if (input.Length <= maxLength) return input;
        return input[..maxLength];
#endif
    }
}
```

## 3. Package References vs Project References

### 3.1 Package References

Package references pull pre-built libraries from NuGet feeds:

```xml
<ItemGroup>
  <!-- Basic package reference with explicit version -->
  <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />

  <!-- Version ranges -->
  <PackageReference Include="Serilog" Version="[3.0.0, 4.0.0)" />

  <!-- Floating version (latest patch) -->
  <PackageReference Include="Dapper" Version="2.1.*" />

  <!-- Private assets: used only at build time, not exposed to consumers -->
  <PackageReference Include="Microsoft.SourceLink.GitHub" Version="8.0.0" PrivateAssets="All" />
</ItemGroup>
```

### 3.2 Project References

Project references link to other projects in your solution, enabling source-level dependency:

```xml
<ItemGroup>
  <!-- Reference a sibling project -->
  <ProjectReference Include="..\MyApp.Core\MyApp.Core.csproj" />

  <!-- Reference with output item type -->
  <ProjectReference Include="..\MyApp.Analyzers\MyApp.Analyzers.csproj"
                    OutputItemType="Analyzer"
                    ReferenceOutputAssembly="false" />
</ItemGroup>
```

### 3.3 When to Use Which

```csharp
// Scenario 1: Third-party library you don't control
// => Use PackageReference
// Example: Newtonsoft.Json, Serilog, Dapper

// Scenario 2: Code you own in the same solution
// => Use ProjectReference
// Example: MyApp.Core, MyApp.Data, MyApp.Tests

// Scenario 3: Internal company library published to private feed
// => Use PackageReference with private NuGet source
// Example: CompanyName.SharedKernel

// Scenario 4: Analyzer or source generator project
// => Use ProjectReference with OutputItemType="Analyzer"
```

## 4. NuGet Basics

### 4.1 What Is NuGet?

NuGet is the package manager for .NET. Packages are distributed as `.nupkg` files (ZIP archives containing compiled DLLs, metadata, and content files).

```csharp
// The NuGet restore process:
// 1. Read PackageReferences from .csproj
// 2. Resolve version constraints and transitive dependencies
// 3. Download packages to global cache (~/.nuget/packages)
// 4. Generate obj/project.assets.json (dependency graph)
// 5. Make assemblies available for compilation
```

### 4.2 NuGet Configuration (nuget.config)

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <!-- Package sources -->
  <packageSources>
    <clear />  <!-- Clear inherited sources -->
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
    <add key="MyCompanyFeed" value="https://pkgs.dev.azure.com/mycompany/_packaging/myfeed/nuget/v3/index.json" />
    <add key="LocalPackages" value="./local-packages" />
  </packageSources>

  <!-- Credentials for authenticated feeds -->
  <packageSourceCredentials>
    <MyCompanyFeed>
      <add key="Username" value="user@company.com" />
      <add key="ClearTextPassword" value="%MY_NUGET_TOKEN%" />
    </MyCompanyFeed>
  </packageSourceCredentials>

  <!-- Global package management settings -->
  <packageManagement>
    <add key="format" value="1" />
    <add key="disabled" value="false" />
  </packageManagement>
</configuration>
```

### 4.3 Package Restore

Restore happens automatically during build, but you can trigger it explicitly:

```bash
# Restore all packages in the solution
dotnet restore

# Restore a specific project
dotnet restore src/MyApp/MyApp.csproj

# Restore with a specific source
dotnet restore --source https://api.nuget.org/v3/index.json

# Force re-evaluation of dependencies
dotnet restore --force

# Clear local caches and restore
dotnet nuget locals all --clear
dotnet restore
```

## 5. dotnet CLI for Package Management

### 5.1 Adding Packages

```bash
# Add a package (latest stable version)
dotnet add package Newtonsoft.Json

# Add a specific version
dotnet add package Serilog --version 3.1.1

# Add a prerelease package
dotnet add package Microsoft.Extensions.Logging --prerelease

# Add a package to a specific project
dotnet add src/MyApp/MyApp.csproj package Dapper

# Add a package from a specific source
dotnet add package MyCompany.Shared --source https://pkgs.dev.azure.com/myco/feed/nuget/v3/index.json
```

### 5.2 Listing Packages

```bash
# List all packages in a project
dotnet list package

# List packages for the entire solution
dotnet list MySolution.sln package

# Show transitive (indirect) dependencies
dotnet list package --include-transitive

# Check for outdated packages
dotnet list package --outdated

# Check for packages with known vulnerabilities
dotnet list package --vulnerable

# Show only top-level packages with available updates
dotnet list package --outdated --highest-minor
```

### 5.3 Removing and Updating Packages

```bash
# Remove a package
dotnet remove package Newtonsoft.Json

# Update a package (remove and re-add with new version)
dotnet add package Serilog --version 4.0.0

# There is no "dotnet update package" command.
# To update all packages, use dotnet-outdated tool:
dotnet tool install --global dotnet-outdated-tool
dotnet outdated --upgrade
```

### 5.4 Managing Global Tools

```bash
# Install a global tool
dotnet tool install --global dotnet-ef

# List installed global tools
dotnet tool list --global

# Update a global tool
dotnet tool update --global dotnet-ef

# Uninstall a global tool
dotnet tool uninstall --global dotnet-ef

# Local tools (per-repository)
dotnet new tool-manifest   # Creates .config/dotnet-tools.json
dotnet tool install dotnet-ef
dotnet tool restore        # Restore tools listed in manifest
```

## 6. Central Package Management

### 6.1 The Problem

In a large solution with many projects, each project specifies its own package versions, leading to version inconsistencies:

```xml
<!-- Project A: MyApp.Core.csproj -->
<PackageReference Include="Newtonsoft.Json" Version="13.0.1" />

<!-- Project B: MyApp.Api.csproj -->
<PackageReference Include="Newtonsoft.Json" Version="13.0.3" />

<!-- Project C: MyApp.Tests.csproj -->
<PackageReference Include="Newtonsoft.Json" Version="12.0.3" />
<!-- Oops! Three different versions of the same package -->
```

### 6.2 Directory.Packages.props

Central Package Management (CPM) solves this with a single `Directory.Packages.props` file at the solution root:

```xml
<!-- Directory.Packages.props (at solution root) -->
<Project>
  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
  </PropertyGroup>

  <ItemGroup>
    <!-- Define versions centrally -->
    <PackageVersion Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageVersion Include="Serilog" Version="3.1.1" />
    <PackageVersion Include="Serilog.Sinks.Console" Version="5.0.1" />
    <PackageVersion Include="Dapper" Version="2.1.35" />
    <PackageVersion Include="Microsoft.Extensions.DependencyInjection" Version="8.0.0" />

    <!-- Test packages -->
    <PackageVersion Include="xunit" Version="2.7.0" />
    <PackageVersion Include="xunit.runner.visualstudio" Version="2.5.7" />
    <PackageVersion Include="Moq" Version="4.20.70" />
    <PackageVersion Include="FluentAssertions" Version="6.12.0" />
    <PackageVersion Include="Microsoft.NET.Test.Sdk" Version="17.9.0" />
  </ItemGroup>
</Project>
```

Now individual projects reference packages without specifying versions:

```xml
<!-- MyApp.Core.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <!-- No Version attribute needed! Pulled from Directory.Packages.props -->
    <PackageReference Include="Newtonsoft.Json" />
    <PackageReference Include="Serilog" />
  </ItemGroup>
</Project>
```

### 6.3 Version Overrides

In rare cases, a specific project may need a different version:

```xml
<!-- MyApp.Legacy.csproj -->
<ItemGroup>
  <!-- Override the centrally-managed version for this project only -->
  <PackageReference Include="Newtonsoft.Json" VersionOverride="12.0.3" />
</ItemGroup>
```

## 7. Directory.Build.props and Directory.Build.targets

### 7.1 Directory.Build.props

This file is automatically imported at the start of every project in its directory tree. Use it to share common properties:

```xml
<!-- Directory.Build.props (at solution root) -->
<Project>
  <PropertyGroup>
    <!-- Shared settings for all projects -->
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>latest</LangVersion>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>

    <!-- Package metadata for NuGet publishing -->
    <Authors>My Company</Authors>
    <Company>My Company Inc.</Company>
    <Copyright>Copyright (c) 2025 My Company Inc.</Copyright>
    <RepositoryUrl>https://github.com/mycompany/myapp</RepositoryUrl>
  </PropertyGroup>
</Project>
```

### 7.2 Directory.Build.targets

This file is imported at the end of every project. Use it for custom build logic that depends on evaluated properties:

```xml
<!-- Directory.Build.targets (at solution root) -->
<Project>
  <!-- Run analyzers on all projects -->
  <ItemGroup>
    <PackageReference Include="Microsoft.CodeAnalysis.NetAnalyzers" Version="8.0.0">
      <PrivateAssets>all</PrivateAssets>
      <IncludeAssets>runtime; build; native; contentfiles; analyzers</IncludeAssets>
    </PackageReference>
  </ItemGroup>

  <!-- Custom target: stamp build time into assembly -->
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

### 7.3 Hierarchical Imports

`Directory.Build.props` and `Directory.Build.targets` files can be nested. Inner files do NOT automatically import outer files; you must do so explicitly:

```xml
<!-- src/Directory.Build.props -->
<Project>
  <!-- Import the parent Directory.Build.props first -->
  <Import Project="$([MSBuild]::GetPathOfFileAbove('Directory.Build.props', '$(MSBuildThisFileDirectory)../'))" />

  <PropertyGroup>
    <!-- Additional settings for src/ projects only -->
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
  </PropertyGroup>
</Project>
```

```xml
<!-- tests/Directory.Build.props -->
<Project>
  <Import Project="$([MSBuild]::GetPathOfFileAbove('Directory.Build.props', '$(MSBuildThisFileDirectory)../'))" />

  <PropertyGroup>
    <!-- Test projects should not treat warnings as errors -->
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
    <IsPackable>false</IsPackable>
  </PropertyGroup>
</Project>
```

## 8. Creating and Publishing NuGet Packages

### 8.1 Package Metadata in .csproj

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFrameworks>net8.0;netstandard2.0</TargetFrameworks>
    <Nullable>enable</Nullable>

    <!-- NuGet package metadata -->
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

    <!-- Generate package on build -->
    <GeneratePackageOnBuild>false</GeneratePackageOnBuild>

    <!-- Include symbols for debugging -->
    <IncludeSymbols>true</IncludeSymbols>
    <SymbolPackageFormat>snupkg</SymbolPackageFormat>

    <!-- Generate XML documentation -->
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
  </PropertyGroup>

  <!-- Include README in the package -->
  <ItemGroup>
    <None Include="README.md" Pack="true" PackagePath="\" />
  </ItemGroup>

</Project>
```

### 8.2 Building and Packing

```bash
# Create a NuGet package
dotnet pack --configuration Release

# Pack with a specific version
dotnet pack --configuration Release /p:Version=1.2.0

# Output goes to bin/Release/*.nupkg

# Inspect the package contents
dotnet nuget verify bin/Release/MyCompany.Utilities.1.2.0.nupkg
```

### 8.3 Publishing to NuGet.org

```bash
# Push to nuget.org
dotnet nuget push bin/Release/MyCompany.Utilities.1.2.0.nupkg \
  --api-key YOUR_API_KEY \
  --source https://api.nuget.org/v3/index.json

# Push to a private feed
dotnet nuget push bin/Release/MyCompany.Utilities.1.2.0.nupkg \
  --source MyCompanyFeed

# Push with symbol package
dotnet nuget push bin/Release/MyCompany.Utilities.1.2.0.snupkg \
  --api-key YOUR_API_KEY \
  --source https://api.nuget.org/v3/index.json
```

### 8.4 A Complete Library Example

```csharp
// File: src/MyCompany.Utilities/StringExtensions.cs
namespace MyCompany.Utilities;

/// <summary>
/// Extension methods for string manipulation.
/// </summary>
public static class StringExtensions
{
    /// <summary>
    /// Truncates a string to the specified maximum length.
    /// </summary>
    /// <param name="value">The string to truncate.</param>
    /// <param name="maxLength">Maximum number of characters.</param>
    /// <param name="suffix">Suffix to append if truncated (default: "...").</param>
    /// <returns>The truncated string.</returns>
    public static string Truncate(this string value, int maxLength, string suffix = "...")
    {
        ArgumentNullException.ThrowIfNull(value);
        ArgumentOutOfRangeException.ThrowIfNegative(maxLength);

        if (value.Length <= maxLength) return value;
        if (maxLength <= suffix.Length) return suffix[..maxLength];

        return string.Concat(value.AsSpan(0, maxLength - suffix.Length), suffix);
    }

    /// <summary>
    /// Converts a string to slug format (URL-friendly).
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

## 9. Global Usings and Implicit Usings

### 9.1 Implicit Usings

When `ImplicitUsings` is enabled, the SDK automatically adds common `using` directives based on the project type:

```xml
<!-- Enabled in .csproj -->
<ImplicitUsings>enable</ImplicitUsings>
```

```csharp
// For Microsoft.NET.Sdk, these are implicitly included:
// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// using System.Net.Http;
// using System.Threading;
// using System.Threading.Tasks;

// So you can write this without any using statements:
List<int> numbers = [1, 2, 3, 4, 5];
var doubled = numbers.Select(n => n * 2).ToList();
Console.WriteLine(string.Join(", ", doubled));
```

### 9.2 Custom Global Usings in .csproj

```xml
<ItemGroup>
  <!-- Add global usings via the project file -->
  <Using Include="System.Text.Json" />
  <Using Include="Microsoft.Extensions.Logging" />

  <!-- Global using with alias -->
  <Using Include="System.Text.Json.JsonSerializer" Alias="Json" />

  <!-- Remove an implicit using -->
  <Using Remove="System.Net.Http" />
</ItemGroup>
```

### 9.3 GlobalUsings.cs File

Alternatively, declare global usings in a dedicated file:

```csharp
// File: GlobalUsings.cs
global using System.Text.Json;
global using System.Text.Json.Serialization;
global using Microsoft.Extensions.Logging;
global using MyApp.Core.Models;
global using MyApp.Core.Interfaces;

// Global using with alias
global using JsonOptions = System.Text.Json.JsonSerializerOptions;
```

## 10. Solution Files (.sln) Management

### 10.1 Creating and Managing Solutions

```bash
# Create a new solution
dotnet new sln --name MyApp

# Create project structure
dotnet new classlib -o src/MyApp.Core
dotnet new classlib -o src/MyApp.Data
dotnet new webapi -o src/MyApp.Api
dotnet new xunit -o tests/MyApp.Core.Tests
dotnet new xunit -o tests/MyApp.Api.Tests

# Add projects to the solution
dotnet sln add src/MyApp.Core/MyApp.Core.csproj
dotnet sln add src/MyApp.Data/MyApp.Data.csproj
dotnet sln add src/MyApp.Api/MyApp.Api.csproj
dotnet sln add tests/MyApp.Core.Tests/MyApp.Core.Tests.csproj
dotnet sln add tests/MyApp.Api.Tests/MyApp.Api.Tests.csproj

# Add projects to solution folders
dotnet sln add src/MyApp.Core/MyApp.Core.csproj --solution-folder src
dotnet sln add tests/MyApp.Core.Tests/MyApp.Core.Tests.csproj --solution-folder tests

# List projects in the solution
dotnet sln list

# Remove a project from the solution
dotnet sln remove tests/MyApp.Api.Tests/MyApp.Api.Tests.csproj
```

### 10.2 Adding Project References

```bash
# Add a project reference
dotnet add src/MyApp.Data/MyApp.Data.csproj reference src/MyApp.Core/MyApp.Core.csproj
dotnet add src/MyApp.Api/MyApp.Api.csproj reference src/MyApp.Core/MyApp.Core.csproj
dotnet add src/MyApp.Api/MyApp.Api.csproj reference src/MyApp.Data/MyApp.Data.csproj

# Add test project references
dotnet add tests/MyApp.Core.Tests/MyApp.Core.Tests.csproj reference src/MyApp.Core/MyApp.Core.csproj

# List references
dotnet list src/MyApp.Api/MyApp.Api.csproj reference
```

### 10.3 Building the Solution

```bash
# Build the entire solution
dotnet build

# Build in Release mode
dotnet build --configuration Release

# Build a specific project
dotnet build src/MyApp.Api/MyApp.Api.csproj

# Run all tests in the solution
dotnet test

# Publish the API project
dotnet publish src/MyApp.Api/MyApp.Api.csproj -c Release -o ./publish
```

## 11. Practical Example: Multi-Project Solution Setup

Let us build a complete multi-project solution from scratch with all the patterns covered in this lesson.

### 11.1 Solution Structure

```
MyShop/
├── Directory.Build.props
├── Directory.Build.targets
├── Directory.Packages.props
├── nuget.config
├── MyShop.sln
├── src/
│   ├── MyShop.Core/          # Domain models and interfaces
│   ├── MyShop.Data/          # Data access (EF Core)
│   └── MyShop.Api/           # Web API
├── tests/
│   └── MyShop.Core.Tests/    # Unit tests
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

### 11.3 Shared Configuration Files

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

### 11.4 Core Library

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

### 11.5 Data Library

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

### 11.6 Test Project

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

## 12. Practice Problems

1. **Project File Modification**: Given a `.csproj` file targeting `net8.0`, modify it to multi-target `net8.0` and `netstandard2.0`. Add conditional compilation so that a method uses `ReadOnlySpan<char>` on `net8.0` and falls back to `string.Substring` on `netstandard2.0`.

2. **Central Package Management**: You have a solution with 5 projects, each referencing various versions of `Serilog`, `Dapper`, and `FluentValidation`. Create a `Directory.Packages.props` file that centralizes all versions, and show how one of the project files would change.

3. **NuGet Package Creation**: Create a class library that provides date/time utility methods (`IsWeekend`, `GetNextBusinessDay`, `FormatRelative`). Write the complete `.csproj` with all NuGet metadata, then show the commands to pack and publish it.

4. **Directory.Build.props Hierarchy**: Design a `Directory.Build.props` structure for a solution where all projects share nullable references and the latest language version, `src/` projects generate XML documentation, and `tests/` projects disable `TreatWarningsAsErrors`. Show all three files.

5. **Solution from Scratch**: Using only `dotnet` CLI commands, create a solution called `BlogEngine` with projects `BlogEngine.Core` (class library), `BlogEngine.Api` (web API), and `BlogEngine.Tests` (xunit). Add the necessary project references (Api depends on Core, Tests depends on Core). List every command in order.
