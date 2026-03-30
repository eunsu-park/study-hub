// Lesson 13: NuGet and Project Configuration
// Run: dotnet run
// Note: This file demonstrates .csproj concepts with executable examples.
//   The actual project structure is shown in comments.

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Xml.Linq;

// ============================================================
// 1. .csproj Structure Overview (in comments)
// ============================================================

Console.WriteLine("=== .csproj Structure Examples ===\n");

// Minimal console app .csproj:
string minimalCsproj = """
    <Project Sdk="Microsoft.NET.Sdk">
      <PropertyGroup>
        <OutputType>Exe</OutputType>
        <TargetFramework>net8.0</TargetFramework>
        <ImplicitUsings>enable</ImplicitUsings>
        <Nullable>enable</Nullable>
      </PropertyGroup>
    </Project>
    """;
Console.WriteLine("Minimal .csproj:");
Console.WriteLine(minimalCsproj);

// .csproj with NuGet packages:
string withPackages = """
    <Project Sdk="Microsoft.NET.Sdk">
      <PropertyGroup>
        <OutputType>Exe</OutputType>
        <TargetFramework>net8.0</TargetFramework>
      </PropertyGroup>

      <!-- NuGet package references -->
      <ItemGroup>
        <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
        <PackageReference Include="Serilog" Version="3.1.1" />
        <PackageReference Include="Microsoft.Extensions.DependencyInjection" Version="8.0.0" />
      </ItemGroup>

      <!-- Project references (other projects in solution) -->
      <ItemGroup>
        <ProjectReference Include="../MyLibrary/MyLibrary.csproj" />
      </ItemGroup>
    </Project>
    """;
Console.WriteLine("\n.csproj with packages:");
Console.WriteLine(withPackages);

// ============================================================
// 2. Multi-Targeting
// ============================================================

Console.WriteLine("\n=== Multi-Targeting ===");

string multiTarget = """
    <Project Sdk="Microsoft.NET.Sdk">
      <PropertyGroup>
        <!-- Target multiple frameworks -->
        <TargetFrameworks>net8.0;net6.0;netstandard2.0</TargetFrameworks>
      </PropertyGroup>

      <!-- Conditional package references per framework -->
      <ItemGroup Condition="'$(TargetFramework)' == 'netstandard2.0'">
        <PackageReference Include="System.Text.Json" Version="8.0.0" />
      </ItemGroup>
    </Project>
    """;
Console.WriteLine("Multi-target .csproj:");
Console.WriteLine(multiTarget);

// Runtime information
Console.WriteLine($"\nCurrent runtime: {RuntimeInformation.FrameworkDescription}");
Console.WriteLine($"OS: {RuntimeInformation.OSDescription}");
Console.WriteLine($"Architecture: {RuntimeInformation.ProcessArchitecture}");

// ============================================================
// 3. Common NuGet Commands
// ============================================================

Console.WriteLine("\n=== Common NuGet Commands ===");

var commands = new Dictionary<string, string>
{
    ["dotnet add package <name>"]          = "Add a NuGet package",
    ["dotnet add package <name> -v 1.0"]   = "Add specific version",
    ["dotnet remove package <name>"]       = "Remove a package",
    ["dotnet list package"]                = "List installed packages",
    ["dotnet list package --outdated"]     = "Show outdated packages",
    ["dotnet restore"]                     = "Restore all packages",
    ["dotnet nuget locals all --clear"]    = "Clear NuGet cache",
    ["dotnet new nugetconfig"]             = "Create nuget.config file",
};

foreach (var (cmd, desc) in commands)
    Console.WriteLine($"  {cmd,-45} # {desc}");

// ============================================================
// 4. Assembly Information
// ============================================================

Console.WriteLine("\n=== Assembly Information ===");

// Properties configurable in .csproj
string assemblyProps = """
    <PropertyGroup>
      <AssemblyName>MyApp</AssemblyName>
      <RootNamespace>MyApp</RootNamespace>
      <Version>1.2.3</Version>
      <Authors>Jane Doe</Authors>
      <Company>Acme Corp</Company>
      <Description>A sample application</Description>
    </PropertyGroup>
    """;
Console.WriteLine("Assembly properties in .csproj:");
Console.WriteLine(assemblyProps);

// Read current assembly info at runtime
var assembly = Assembly.GetExecutingAssembly();
Console.WriteLine($"\nCurrent assembly: {assembly.GetName().Name}");
Console.WriteLine($"Version: {assembly.GetName().Version}");
Console.WriteLine($"Location: {assembly.Location}");

// ============================================================
// 5. Build Configurations
// ============================================================

Console.WriteLine("\n=== Build Configurations ===");

string buildConfig = """
    <PropertyGroup Condition="'$(Configuration)' == 'Debug'">
      <DefineConstants>DEBUG;TRACE</DefineConstants>
      <DebugType>full</DebugType>
      <Optimize>false</Optimize>
    </PropertyGroup>

    <PropertyGroup Condition="'$(Configuration)' == 'Release'">
      <DefineConstants>TRACE</DefineConstants>
      <DebugType>pdbonly</DebugType>
      <Optimize>true</Optimize>
    </PropertyGroup>
    """;
Console.WriteLine("Conditional build properties:");
Console.WriteLine(buildConfig);

// Conditional compilation in code
#if DEBUG
Console.WriteLine("\n  Currently in DEBUG mode");
#else
Console.WriteLine("\n  Currently in RELEASE mode");
#endif

// ============================================================
// 6. Project Types
// ============================================================

Console.WriteLine("\n=== Common Project Types ===");

var projectTypes = new (string Sdk, string Type, string Command)[]
{
    ("Microsoft.NET.Sdk",          "Console App / Library", "dotnet new console / classlib"),
    ("Microsoft.NET.Sdk.Web",      "ASP.NET Web App",      "dotnet new web / webapi / mvc"),
    ("Microsoft.NET.Sdk.Worker",   "Background Service",   "dotnet new worker"),
    ("Microsoft.NET.Sdk.BlazorWebAssembly", "Blazor WASM", "dotnet new blazorwasm"),
};

Console.WriteLine($"  {"SDK",-42} {"Type",-25} {"Command"}");
Console.WriteLine($"  {new string('-', 42)} {new string('-', 25)} {new string('-', 30)}");
foreach (var (sdk, type, cmd) in projectTypes)
    Console.WriteLine($"  {sdk,-42} {type,-25} {cmd}");

// ============================================================
// 7. Programmatic .csproj Parsing
// ============================================================

Console.WriteLine("\n=== Parsing .csproj with XElement ===");

string sampleCsproj = """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Serilog" Version="3.1.1" />
    <PackageReference Include="Dapper" Version="2.1.28" />
    <PackageReference Include="Polly" Version="8.2.0" />
  </ItemGroup>
</Project>
""";

var doc = XDocument.Parse(sampleCsproj);
var ns = doc.Root?.Name.Namespace ?? XNamespace.None;

// Extract target framework
string? tfm = doc.Root?
    .Element(ns + "PropertyGroup")?
    .Element(ns + "TargetFramework")?.Value;
Console.WriteLine($"Target framework: {tfm}");

// List all package references
var packages = doc.Descendants(ns + "PackageReference")
    .Select(e => new
    {
        Name = e.Attribute("Include")?.Value,
        Version = e.Attribute("Version")?.Value
    });

Console.WriteLine("Packages:");
foreach (var pkg in packages)
    Console.WriteLine($"  {pkg.Name} v{pkg.Version}");

// ============================================================
// 8. Solution Structure
// ============================================================

Console.WriteLine("\n=== Solution Commands ===");

var slnCommands = new Dictionary<string, string>
{
    ["dotnet new sln -n MySolution"]                  = "Create solution",
    ["dotnet sln add src/MyApp/MyApp.csproj"]         = "Add project to solution",
    ["dotnet sln add tests/MyTests/MyTests.csproj"]   = "Add test project",
    ["dotnet sln list"]                               = "List projects in solution",
    ["dotnet build MySolution.sln"]                   = "Build entire solution",
    ["dotnet test MySolution.sln"]                    = "Run all tests",
};

foreach (var (cmd, desc) in slnCommands)
    Console.WriteLine($"  {cmd,-50} # {desc}");

// ============================================================
// 9. Global.json and Directory.Build.props
// ============================================================

Console.WriteLine("\n=== Global Configuration Files ===");

string globalJson = """
    global.json — Pin SDK version:
    {
      "sdk": {
        "version": "8.0.100",
        "rollForward": "latestPatch"
      }
    }
    """;
Console.WriteLine(globalJson);

string dirBuildProps = """

    Directory.Build.props — Shared properties for all projects:
    <Project>
      <PropertyGroup>
        <Nullable>enable</Nullable>
        <ImplicitUsings>enable</ImplicitUsings>
        <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
      </PropertyGroup>
    </Project>
    """;
Console.WriteLine(dirBuildProps);
