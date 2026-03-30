/*
 * Exercises for Lesson 13: NuGet and Project Configuration
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Linq;

// ---------------------------------------------------------------------------
// Exercise 1: Parse and display .csproj file contents
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Parse .csproj Structure ===");

    string csprojContent = """
        <Project Sdk="Microsoft.NET.Sdk">
          <PropertyGroup>
            <OutputType>Exe</OutputType>
            <TargetFramework>net8.0</TargetFramework>
            <Nullable>enable</Nullable>
            <ImplicitUsings>enable</ImplicitUsings>
            <Version>1.2.3</Version>
          </PropertyGroup>
          <ItemGroup>
            <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
            <PackageReference Include="Serilog" Version="3.1.1" />
            <PackageReference Include="Dapper" Version="2.1.24" />
          </ItemGroup>
        </Project>
        """;

    var doc = XDocument.Parse(csprojContent);
    var props = doc.Descendants("PropertyGroup").First();

    Console.WriteLine("  Project Properties:");
    foreach (var el in props.Elements())
        Console.WriteLine($"    {el.Name}: {el.Value}");

    var packages = doc.Descendants("PackageReference");
    Console.WriteLine("  Package References:");
    foreach (var pkg in packages)
        Console.WriteLine($"    {pkg.Attribute("Include")?.Value} v{pkg.Attribute("Version")?.Value}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Semantic versioning — parse and compare versions
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Semantic Version Parsing ===");

    var versions = new[] { "1.0.0", "2.1.0", "1.9.3", "2.0.0-beta", "1.0.1", "2.1.0-rc.1" };

    var parsed = versions
        .Select(v => new SemVer(v))
        .OrderBy(v => v)
        .ToList();

    Console.WriteLine("  Sorted versions:");
    foreach (var v in parsed)
        Console.WriteLine($"    {v} (stable={v.IsStable})");

    var latest = parsed.Where(v => v.IsStable).Last();
    Console.WriteLine($"  Latest stable: {latest}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Build configuration — conditional compilation
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Conditional Compilation ===");

    // Simulate checking build configuration
    var config = new BuildConfig("Release", "net8.0", "win-x64");

    Console.WriteLine($"  Configuration: {config.Configuration}");
    Console.WriteLine($"  Framework    : {config.TargetFramework}");
    Console.WriteLine($"  Runtime      : {config.RuntimeIdentifier}");
    Console.WriteLine($"  Is Debug     : {config.IsDebug}");
    Console.WriteLine($"  Output path  : {config.GetOutputPath()}");

    var debugConfig = new BuildConfig("Debug", "net8.0", "linux-x64");
    Console.WriteLine($"  Debug output : {debugConfig.GetOutputPath()}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Directory.Build.props generator
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Directory.Build.props Generator ===");

    var settings = new DirectoryBuildProps
    {
        Company = "Contoso",
        Authors = "Dev Team",
        LangVersion = "12.0",
        Nullable = true,
        ImplicitUsings = true,
        TreatWarningsAsErrors = true,
        CommonPackages = new()
        {
            ("Microsoft.Extensions.Logging", "8.0.0"),
            ("FluentValidation", "11.9.0"),
        }
    };

    string xml = settings.Generate();
    Console.WriteLine(xml);
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Package dependency resolver — detect version conflicts
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Dependency Conflict Detection ===");

    var projectA = new ProjectDeps("WebAPI", new()
    {
        { "Newtonsoft.Json", "13.0.3" },
        { "Serilog", "3.1.1" },
        { "AutoMapper", "12.0.1" },
    });

    var projectB = new ProjectDeps("DataLayer", new()
    {
        { "Newtonsoft.Json", "13.0.1" },
        { "Dapper", "2.1.24" },
        { "AutoMapper", "12.0.1" },
    });

    var conflicts = DetectConflicts(projectA, projectB);

    if (conflicts.Count == 0)
        Console.WriteLine("  No conflicts detected.");
    else
    {
        Console.WriteLine("  Version conflicts detected:");
        foreach (var c in conflicts)
            Console.WriteLine($"    {c.Package}: {projectA.Name}={c.VersionA}, {projectB.Name}={c.VersionB}");
    }
    Console.WriteLine();
}

List<Conflict> DetectConflicts(ProjectDeps a, ProjectDeps b)
{
    var conflicts = new List<Conflict>();
    foreach (var (pkg, verA) in a.Packages)
    {
        if (b.Packages.TryGetValue(pkg, out var verB) && verA != verB)
            conflicts.Add(new Conflict(pkg, verA, verB));
    }
    return conflicts;
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();

// ===========================================================================
// Supporting types
// ===========================================================================

class SemVer : IComparable<SemVer>
{
    public int Major { get; }
    public int Minor { get; }
    public int Patch { get; }
    public string? PreRelease { get; }
    public bool IsStable => PreRelease == null;

    public SemVer(string version)
    {
        var parts = version.Split('-', 2);
        var nums = parts[0].Split('.');
        Major = int.Parse(nums[0]);
        Minor = int.Parse(nums[1]);
        Patch = int.Parse(nums[2]);
        PreRelease = parts.Length > 1 ? parts[1] : null;
    }

    public int CompareTo(SemVer? other)
    {
        if (other == null) return 1;
        int c = Major.CompareTo(other.Major);
        if (c != 0) return c;
        c = Minor.CompareTo(other.Minor);
        if (c != 0) return c;
        c = Patch.CompareTo(other.Patch);
        if (c != 0) return c;
        if (IsStable && !other.IsStable) return 1;
        if (!IsStable && other.IsStable) return -1;
        return string.Compare(PreRelease, other.PreRelease, StringComparison.Ordinal);
    }

    public override string ToString() => PreRelease != null ? $"{Major}.{Minor}.{Patch}-{PreRelease}" : $"{Major}.{Minor}.{Patch}";
}

record BuildConfig(string Configuration, string TargetFramework, string RuntimeIdentifier)
{
    public bool IsDebug => Configuration == "Debug";
    public string GetOutputPath() => $"bin/{Configuration}/{TargetFramework}/{RuntimeIdentifier}/";
}

class DirectoryBuildProps
{
    public string Company { get; set; } = "";
    public string Authors { get; set; } = "";
    public string LangVersion { get; set; } = "latest";
    public bool Nullable { get; set; }
    public bool ImplicitUsings { get; set; }
    public bool TreatWarningsAsErrors { get; set; }
    public List<(string Name, string Version)> CommonPackages { get; set; } = new();

    public string Generate()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<Project>");
        sb.AppendLine("  <PropertyGroup>");
        sb.AppendLine($"    <Company>{Company}</Company>");
        sb.AppendLine($"    <Authors>{Authors}</Authors>");
        sb.AppendLine($"    <LangVersion>{LangVersion}</LangVersion>");
        if (Nullable) sb.AppendLine("    <Nullable>enable</Nullable>");
        if (ImplicitUsings) sb.AppendLine("    <ImplicitUsings>enable</ImplicitUsings>");
        if (TreatWarningsAsErrors) sb.AppendLine("    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>");
        sb.AppendLine("  </PropertyGroup>");
        if (CommonPackages.Count > 0)
        {
            sb.AppendLine("  <ItemGroup>");
            foreach (var (name, ver) in CommonPackages)
                sb.AppendLine($"    <PackageReference Include=\"{name}\" Version=\"{ver}\" />");
            sb.AppendLine("  </ItemGroup>");
        }
        sb.AppendLine("</Project>");
        return sb.ToString();
    }
}

record ProjectDeps(string Name, Dictionary<string, string> Packages);
record Conflict(string Package, string VersionA, string VersionB);
