# Getting Started

**Previous**: [Overview](./00_Overview.md) | **Next**: [Variables and Types](./02_Variables_and_Types.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install the .NET SDK on macOS, Windows, and Linux
2. Use the `dotnet` CLI to create, build, and run projects
3. Write a Hello World program using both top-level statements and the traditional `Main` method
4. Understand the project structure of a C# console application
5. Work with solution files and multi-project setups
6. Configure VS Code with the C# Dev Kit extension
7. Explain the C# compilation process from source code to execution

---

C# is a modern, object-oriented, strongly-typed programming language developed by Microsoft as part of the .NET platform. Originally released in 2000, C# has evolved significantly through its many versions, with C# 12 (released with .NET 8) being the latest stable release at the time of writing. The language is used for a wide variety of applications: desktop software (WPF, WinForms), web services (ASP.NET Core), mobile apps (.NET MAUI), game development (Unity), cloud services (Azure Functions), and much more. In this first lesson, you will set up your development environment and write your first C# program.

## 1. Installing the .NET SDK

The .NET SDK (Software Development Kit) includes everything you need to build and run C# applications: the compiler, the runtime, and the `dotnet` CLI tool.

### 1.1 Choosing a Version

.NET follows a predictable release cadence. Even-numbered releases (6, 8, 10) are Long-Term Support (LTS) with three years of support. Odd-numbered releases (7, 9) are Standard-Term Support (STS) with 18 months of support. For learning, always choose the latest LTS release.

```
Release Timeline:
  .NET 6  (LTS)  — Nov 2021 to Nov 2024
  .NET 7  (STS)  — Nov 2022 to May 2024
  .NET 8  (LTS)  — Nov 2023 to Nov 2026
  .NET 9  (STS)  — Nov 2024 to May 2026
  .NET 10 (LTS)  — Nov 2025 to Nov 2028
```

### 1.2 Installation on macOS

The recommended approach on macOS is to use the official installer or Homebrew:

```bash
# Option 1: Homebrew (recommended)
brew install --cask dotnet-sdk

# Option 2: Download the installer from
# https://dotnet.microsoft.com/download

# Verify installation
dotnet --version
# 8.0.401 (or similar)

dotnet --list-sdks
# 8.0.401 [/usr/local/share/dotnet/sdk]
```

If you use the Homebrew install, make sure the dotnet path is in your shell profile:

```bash
# Add to ~/.zshrc or ~/.bash_profile if needed
export DOTNET_ROOT="/usr/local/share/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
```

### 1.3 Installation on Windows

On Windows, download the installer from the official website or use `winget`:

```powershell
# Option 1: winget (Windows Package Manager)
winget install Microsoft.DotNet.SDK.8

# Option 2: Download from https://dotnet.microsoft.com/download

# Verify installation
dotnet --version
```

The Windows installer automatically adds `dotnet` to your system PATH.

### 1.4 Installation on Linux

On Ubuntu/Debian-based distributions:

```bash
# Add the Microsoft package repository
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

# Install the SDK
sudo apt-get update
sudo apt-get install -y dotnet-sdk-8.0

# Verify installation
dotnet --version
```

On Fedora:

```bash
sudo dnf install dotnet-sdk-8.0
dotnet --version
```

### 1.5 Verifying Your Installation

Regardless of your operating system, run these commands to confirm everything is working:

```bash
# Show SDK version
dotnet --version

# Show all installed SDKs
dotnet --list-sdks

# Show all installed runtimes
dotnet --list-runtimes

# Display comprehensive information
dotnet --info
```

## 2. The dotnet CLI

The `dotnet` command-line interface is the primary tool for creating, building, running, and publishing .NET applications. You will use it constantly throughout this course.

### 2.1 Creating a New Project

The `dotnet new` command creates projects from built-in templates:

```bash
# Create a new console application
dotnet new console -n MyFirstApp

# Create with a specific framework version
dotnet new console -n MyFirstApp --framework net8.0

# List all available templates
dotnet new list
```

The `-n` (or `--name`) flag specifies both the project directory name and the default namespace. Without it, the current directory name is used.

Common templates include:

```
Template Name          Short Name    Language
---------------------  -----------   --------
Console App            console       C#
Class Library          classlib      C#
ASP.NET Core Web App   webapp        C#
ASP.NET Core Web API   webapi        C#
xUnit Test Project     xunit         C#
NUnit Test Project     nunit         C#
```

### 2.2 Building and Running

```bash
# Navigate into the project directory
cd MyFirstApp

# Build the project (compile without running)
dotnet build

# Run the project (builds automatically if needed)
dotnet run

# Clean build artifacts
dotnet clean

# Build in Release mode (optimized)
dotnet build -c Release

# Run in Release mode
dotnet run -c Release
```

### 2.3 Adding Packages

NuGet is the package manager for .NET. You can add third-party libraries with:

```bash
# Add a NuGet package
dotnet add package Newtonsoft.Json

# Add a specific version
dotnet add package Newtonsoft.Json --version 13.0.3

# Remove a package
dotnet remove package Newtonsoft.Json

# List installed packages
dotnet list package
```

### 2.4 Other Useful Commands

```bash
# Publish a self-contained application
dotnet publish -c Release --self-contained

# Run tests
dotnet test

# Format code according to .editorconfig rules
dotnet format

# Watch for file changes and auto-rebuild
dotnet watch run
```

## 3. Hello World: Two Styles

C# supports two styles of entry point for console applications. Understanding both is important because you will encounter both in documentation and real-world code.

### 3.1 Top-Level Statements (Modern Style)

Starting with C# 9 and .NET 5, you can write a console application without the boilerplate of a class and `Main` method. This is the default for `dotnet new console` in .NET 6 and later:

```csharp
// Program.cs — Top-level statements
// No class or Main method required

Console.WriteLine("Hello, World!");
```

That is the entire file. The compiler generates the `Main` method for you behind the scenes. You can still use `args` to access command-line arguments:

```csharp
// Program.cs — Top-level statements with args
if (args.Length > 0)
{
    Console.WriteLine($"Hello, {args[0]}!");
}
else
{
    Console.WriteLine("Hello, World!");
}
```

Run with arguments:

```bash
dotnet run -- Alice
# Output: Hello, Alice!
```

You can also use `await`, define methods, and declare classes in a top-level statements file:

```csharp
// Program.cs — Top-level statements with local functions
string greeting = CreateGreeting("C#");
Console.WriteLine(greeting);

string CreateGreeting(string name)
{
    return $"Hello, {name}! Welcome to the world of programming.";
}
```

### 3.2 Traditional Main Method

Before C# 9, every program required an explicit `Main` method inside a class. This style is still fully supported and preferred for larger applications:

```csharp
// Program.cs — Traditional entry point
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

The `Main` method can have four valid signatures:

```csharp
// 1. No parameters, no return value
static void Main() { }

// 2. With command-line arguments
static void Main(string[] args) { }

// 3. Return an exit code
static int Main() { return 0; }

// 4. Both arguments and exit code
static int Main(string[] args) { return 0; }

// 5. Async variants (C# 7.1+)
static async Task Main() { }
static async Task<int> Main(string[] args) { return 0; }
```

### 3.3 Which Style Should You Use?

Use **top-level statements** for:
- Small programs, scripts, and learning exercises
- Quick prototypes and experiments
- Single-file programs

Use the **traditional Main method** for:
- Production applications
- Projects with multiple entry points
- When you need explicit control over the entry point class

Throughout this course, we will primarily use top-level statements for brevity, but we will note when the traditional approach is more appropriate.

## 4. Project Structure

When you create a new console application with `dotnet new console -n MyFirstApp`, the following structure is generated:

```
MyFirstApp/
├── MyFirstApp.csproj    # Project file (build configuration)
├── Program.cs           # Source code (entry point)
├── obj/                 # Intermediate build files (auto-generated)
│   ├── project.assets.json
│   ├── project.nuget.cache
│   └── ...
└── bin/                 # Compiled output (after build)
    └── Debug/
        └── net8.0/
            ├── MyFirstApp.dll    # Compiled assembly
            ├── MyFirstApp.exe    # Executable (Windows) / native host
            ├── MyFirstApp.deps.json
            ├── MyFirstApp.runtimeconfig.json
            └── ...
```

### 4.1 The .csproj File

The `.csproj` (C# project) file is an XML file that defines your project's build configuration:

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

Key elements explained:

- **`OutputType`**: `Exe` for executable, `Library` for class libraries
- **`TargetFramework`**: The .NET version to target (`net8.0`, `net9.0`, etc.)
- **`ImplicitUsings`**: When `enable`, common `using` directives are automatically included (`System`, `System.Collections.Generic`, `System.IO`, `System.Linq`, `System.Threading.Tasks`, etc.)
- **`Nullable`**: When `enable`, the compiler warns about potential null reference issues

When you add NuGet packages, they appear as `<PackageReference>` elements:

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

### 4.2 The obj/ Directory

The `obj/` directory contains intermediate build files. You should never edit files here. It is safe to delete this directory; it will be regenerated on the next build. Add it to `.gitignore`.

### 4.3 The bin/ Directory

The `bin/` directory contains the compiled output. After running `dotnet build`, you will find your application's DLL and supporting files here. The structure follows the pattern `bin/<Configuration>/<TargetFramework>/`. Add it to `.gitignore`.

### 4.4 The .gitignore File

A proper `.gitignore` for C# projects:

```gitignore
# Build output
bin/
obj/

# IDE files
.vs/
.vscode/
*.user
*.suo

# OS files
.DS_Store
Thumbs.db
```

You can generate one automatically:

```bash
dotnet new gitignore
```

## 5. Solution Files and Multi-Project Setup

As applications grow, you will often have multiple projects (a main application, a class library, and test projects). A **solution file** (`.sln`) groups related projects together.

### 5.1 Creating a Solution

```bash
# Create a new solution
dotnet new sln -n MySolution

# Create projects
dotnet new console -n MyApp
dotnet new classlib -n MyLibrary
dotnet new xunit -n MyTests

# Add projects to the solution
dotnet sln MySolution.sln add MyApp/MyApp.csproj
dotnet sln MySolution.sln add MyLibrary/MyLibrary.csproj
dotnet sln MySolution.sln add MyTests/MyTests.csproj
```

The resulting structure:

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

### 5.2 Adding Project References

To use `MyLibrary` from `MyApp`:

```bash
dotnet add MyApp/MyApp.csproj reference MyLibrary/MyLibrary.csproj
```

This adds a `<ProjectReference>` to the `.csproj` file:

```xml
<ItemGroup>
  <ProjectReference Include="..\MyLibrary\MyLibrary.csproj" />
</ItemGroup>
```

Now you can use classes from `MyLibrary` in `MyApp`:

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

### 5.3 Building and Running with Solutions

```bash
# Build all projects in the solution
dotnet build MySolution.sln

# Run a specific project
dotnet run --project MyApp

# Run all tests
dotnet test MySolution.sln
```

## 6. VS Code Setup with C# Dev Kit

While you can write C# in any text editor, Visual Studio Code with the C# Dev Kit extension provides an excellent lightweight IDE experience.

### 6.1 Installing the Extension

1. Open VS Code
2. Go to the Extensions view (`Ctrl+Shift+X` / `Cmd+Shift+X`)
3. Search for "C# Dev Kit"
4. Install the **C# Dev Kit** extension (by Microsoft)

This automatically installs three extensions:
- **C# Dev Kit** — Solution explorer, project management, test explorer
- **C#** (powered by OmniSharp) — IntelliSense, syntax highlighting, debugging
- **IntelliCode for C# Dev Kit** — AI-assisted code completions

### 6.2 Configuring VS Code

Create a `.vscode/settings.json` in your project or workspace:

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-dotnettools.csharp",
    "omnisharp.enableEditorConfigSupport": true,
    "dotnet.defaultSolution": "MySolution.sln"
}
```

### 6.3 Debugging in VS Code

The C# extension creates a `.vscode/launch.json` for debugging. You can generate one by pressing `F5` and selecting ".NET 5+ and .NET Core":

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

Set breakpoints by clicking in the gutter next to line numbers, then press `F5` to start debugging. You can inspect variables, step through code, and use the debug console.

## 7. Understanding the Compilation Process

Understanding how C# code goes from source to execution helps you debug problems and optimize performance.

### 7.1 The Compilation Pipeline

```
  Source Code (.cs)
       │
       ▼
  ┌─────────────┐
  │  C# Compiler │  (Roslyn)
  │   (csc)      │
  └──────┬──────┘
         │
         ▼
  Intermediate Language (.dll)
  (IL / MSIL / CIL)
         │
         ▼
  ┌─────────────┐
  │  CLR         │  Common Language Runtime
  │  (Runtime)   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  JIT         │  Just-In-Time Compiler
  │  Compiler    │
  └──────┬──────┘
         │
         ▼
  Native Machine Code
  (executed by CPU)
```

### 7.2 Roslyn: The C# Compiler

Roslyn is the open-source C# and Visual Basic compiler. When you run `dotnet build`, Roslyn compiles your `.cs` files into **Intermediate Language (IL)**, which is stored in a `.dll` assembly file.

You can inspect IL using the `ildasm` tool or the ILSpy decompiler:

```bash
# Build the project
dotnet build

# The output DLL contains IL code
# Located at bin/Debug/net8.0/MyFirstApp.dll
```

### 7.3 The Common Language Runtime (CLR)

The CLR is the virtual machine that manages the execution of .NET programs. It provides:

- **Memory management**: Automatic garbage collection
- **Type safety**: Runtime type checking
- **Exception handling**: Structured exception support
- **Thread management**: Thread pool and synchronization
- **Security**: Code access security and verification

### 7.4 JIT Compilation

When you run a .NET application, the **Just-In-Time (JIT) compiler** converts IL to native machine code on the fly. This happens method-by-method as each method is called for the first time:

```csharp
// This C# code...
int Sum(int a, int b)
{
    return a + b;
}

// ...is compiled to IL (simplified)...
// IL_0000: ldarg.1
// IL_0001: ldarg.2
// IL_0002: add
// IL_0003: ret

// ...which is JIT-compiled to native x64 code:
// mov eax, ecx
// add eax, edx
// ret
```

### 7.5 Ahead-of-Time (AOT) Compilation

.NET 8 introduced Native AOT compilation, which compiles directly to native code without needing the JIT at runtime:

```bash
# Publish with Native AOT
dotnet publish -c Release -r linux-x64 /p:PublishAot=true
```

AOT produces smaller, faster-starting executables but has some limitations (no runtime code generation, limited reflection).

### 7.6 A Complete Example

Let us put it all together with a slightly more substantial program:

```csharp
// Program.cs — A complete getting started example
Console.WriteLine("=== .NET Environment Info ===");
Console.WriteLine($"OS: {Environment.OSVersion}");
Console.WriteLine($"Runtime: {Environment.Version}");
Console.WriteLine($"Machine: {Environment.MachineName}");
Console.WriteLine($"64-bit OS: {Environment.Is64BitOperatingSystem}");
Console.WriteLine($"64-bit Process: {Environment.Is64BitProcess}");
Console.WriteLine();

Console.Write("What is your name? ");
string? name = Console.ReadLine();

if (!string.IsNullOrWhiteSpace(name))
{
    Console.WriteLine($"Welcome to C#, {name}!");
    Console.WriteLine($"Today is {DateTime.Now:dddd, MMMM dd, yyyy}");
    Console.WriteLine($"Current time: {DateTime.Now:HH:mm:ss}");
}
else
{
    Console.WriteLine("Welcome to C#, anonymous user!");
}

Console.WriteLine("\nPress any key to exit...");
Console.ReadKey();
```

Build and run:

```bash
dotnet build
dotnet run
```

## 8. Practice Problems

1. **SDK Exploration**: Run `dotnet --info` on your machine and identify: (a) the SDK version, (b) the runtime version, (c) the host operating system, and (d) the base path. Create a text file summarizing what each section of the output means.

2. **Template Experiment**: Use `dotnet new list` to explore available templates. Create a class library project (`dotnet new classlib -n MathLib`) and a console project (`dotnet new console -n MathApp`). Add a static method `Factorial(int n)` to the library and call it from the console app. Use `dotnet add reference` to connect them.

3. **Both Entry Point Styles**: Write the same program in two files. The program should accept a number as a command-line argument and print its square. Create one version using top-level statements and another using the traditional `Main` method. Compare the code side by side.

4. **Build Configuration**: Modify the `.csproj` file to target `net8.0` and `net9.0` simultaneously (hint: use `<TargetFrameworks>` with plural). Build for both targets and examine the `bin/` directory structure.

5. **Solution Setup**: Create a complete solution with three projects: a console app (`Calculator`), a class library (`CalculatorLib`), and a test project (`CalculatorTests`). Add methods for `Add`, `Subtract`, `Multiply`, and `Divide` in the library. Call them from the console app and write at least one test per method.

---

**Previous**: [Overview](./00_Overview.md) | **Next**: [Variables and Types](./02_Variables_and_Types.md)
