// Exercise: Build System Deep Dive
// Practice with Cargo workspaces, feature flags, profiles, and build scripts.
//
// Since build system concepts are primarily about Cargo.toml configuration,
// this file tests your understanding through code that interacts with
// cfg attributes, conditional compilation, and build-time patterns.
//
// Run: rustc 19_build_system.rs && ./19_build_system

// ============================================================
// Exercise 1: Conditional Compilation with cfg
// ============================================================
// Implement functions that behave differently based on cfg flags.
// When compiled normally, use the default implementations.
// When compiled with --cfg custom_alloc, use the custom path.

#[cfg(not(custom_alloc))]
fn allocate_buffer(size: usize) -> Vec<u8> {
    // TODO: Return a zero-filled Vec<u8> of the given size
    todo!()
}

#[cfg(custom_alloc)]
fn allocate_buffer(size: usize) -> Vec<u8> {
    // Custom allocator path: fill with 0xFF instead
    vec![0xFF; size]
}

// ============================================================
// Exercise 2: Platform-Specific Code
// ============================================================
// Write a function that returns the platform name using cfg attributes.

fn platform_name() -> &'static str {
    // TODO: Return "linux", "macos", "windows", or "unknown"
    // Hint: Use #[cfg(target_os = "...")] blocks or cfg! macro
    todo!()
}

// ============================================================
// Exercise 3: Simulating Feature Flags
// ============================================================
// In real projects, features come from Cargo.toml. Here we simulate
// them with cfg flags to practice the pattern.
//
// Compile with: rustc --cfg 'feature="json"' --cfg 'feature="xml"' 19_build_system.rs

/// A configuration value that can be serialized in different formats.
struct Config {
    key: String,
    value: String,
}

impl Config {
    fn new(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
        }
    }
}

// TODO: Implement to_json() that is only available when feature="json" is set.
// It should return a String like: {"key":"name","value":"rust"}
#[cfg(feature = "json")]
fn to_json(config: &Config) -> String {
    todo!()
}

// TODO: Implement to_xml() that is only available when feature="xml" is set.
// It should return a String like: <config><key>name</key><value>rust</value></config>
#[cfg(feature = "xml")]
fn to_xml(config: &Config) -> String {
    todo!()
}

// This function always exists regardless of features
fn to_plain(config: &Config) -> String {
    format!("{}={}", config.key, config.value)
}

// ============================================================
// Exercise 4: Simulating Build Script Output
// ============================================================
// Build scripts generate code or set environment variables.
// Here, simulate what a build script would generate.

/// Simulates what `include!(concat!(env!("OUT_DIR"), "/generated.rs"))` would provide.
/// In a real project, build.rs would generate this.
mod generated {
    // TODO: Define a constant array of supported formats.
    // pub const FORMATS: &[&str] = &["plain", "json", "xml", "yaml"];

    // TODO: Define a build version constant.
    // pub const BUILD_VERSION: &str = "0.1.0-dev";
}

// ============================================================
// Exercise 5: Dependency Version Parsing
// ============================================================
// Implement a simple SemVer version type and compatibility checker,
// similar to how Cargo resolves dependency versions.

#[derive(Debug, Clone, PartialEq)]
struct SemVer {
    major: u32,
    minor: u32,
    patch: u32,
}

impl SemVer {
    // TODO: Parse a version string like "1.2.3" into a SemVer.
    // Return None if the format is invalid.
    fn parse(version: &str) -> Option<Self> {
        todo!()
    }

    // TODO: Check if `self` is compatible with `other` under caret semantics.
    // For major >= 1: same major version (e.g., 1.2.3 compatible with 1.9.0)
    // For major == 0: same major AND minor (e.g., 0.2.3 compatible with 0.2.9)
    fn is_compatible(&self, other: &SemVer) -> bool {
        todo!()
    }

    fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ============================================================
// Exercise 6: Workspace Dependency Graph
// ============================================================
// Simulate a workspace with multiple crates and determine build order.

#[derive(Debug, Clone)]
struct Crate {
    name: String,
    dependencies: Vec<String>,
}

// TODO: Given a list of crates with their dependencies, return a valid
// build order (topological sort). Return None if there is a cycle.
fn build_order(crates: &[Crate]) -> Option<Vec<String>> {
    todo!()
}

// ============================================================
// Exercise 7: Profile Configuration
// ============================================================
// Simulate Cargo profile settings and their effects.

#[derive(Debug, Clone)]
struct Profile {
    name: String,
    opt_level: u8,       // 0, 1, 2, 3
    debug: bool,
    lto: LtoSetting,
    codegen_units: u32,
    strip: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum LtoSetting {
    Off,
    Thin,
    Fat,
}

impl Profile {
    // TODO: Return the default "dev" profile settings
    fn dev() -> Self {
        todo!()
    }

    // TODO: Return the default "release" profile settings
    fn release() -> Self {
        todo!()
    }

    // TODO: Create a new profile that inherits from a base profile
    // and applies overrides. Only override fields that are Some.
    fn inherit(
        base: &Profile,
        name: &str,
        opt_level: Option<u8>,
        debug: Option<bool>,
        lto: Option<LtoSetting>,
        codegen_units: Option<u32>,
        strip: Option<bool>,
    ) -> Self {
        todo!()
    }

    // TODO: Estimate relative compile time (1-10 scale) based on settings.
    // Higher opt_level = slower, LTO adds time, fewer codegen_units = slower
    fn estimated_compile_time(&self) -> u8 {
        todo!()
    }

    // TODO: Estimate relative binary size (1-10 scale) based on settings.
    // Higher opt_level = smaller, LTO = smaller, strip = smaller, debug = larger
    fn estimated_binary_size(&self) -> u8 {
        todo!()
    }
}

// ============================================================
// Main — run all exercises
// ============================================================

fn main() {
    println!("=== Exercise 1: Conditional Compilation ===");
    let buf = allocate_buffer(4);
    #[cfg(not(custom_alloc))]
    assert_eq!(buf, vec![0, 0, 0, 0]);
    #[cfg(custom_alloc)]
    assert_eq!(buf, vec![0xFF, 0xFF, 0xFF, 0xFF]);
    println!("Buffer: {:?}", buf);

    println!("\n=== Exercise 2: Platform Detection ===");
    let name = platform_name();
    println!("Platform: {name}");
    assert!(["linux", "macos", "windows", "unknown"].contains(&name));

    println!("\n=== Exercise 3: Feature Flags ===");
    let config = Config::new("lang", "rust");
    println!("Plain: {}", to_plain(&config));

    #[cfg(feature = "json")]
    {
        let json = to_json(&config);
        println!("JSON: {json}");
        assert!(json.contains("lang"));
        assert!(json.contains("rust"));
    }

    #[cfg(feature = "xml")]
    {
        let xml = to_xml(&config);
        println!("XML: {xml}");
        assert!(xml.contains("<key>lang</key>"));
        assert!(xml.contains("<value>rust</value>"));
    }

    println!("\n=== Exercise 4: Generated Constants ===");
    // Uncomment after implementing:
    // println!("Formats: {:?}", generated::FORMATS);
    // println!("Build version: {}", generated::BUILD_VERSION);
    // assert_eq!(generated::FORMATS.len(), 4);

    println!("\n=== Exercise 5: SemVer Parsing ===");
    let v1 = SemVer::parse("1.2.3").unwrap();
    assert_eq!(v1.major, 1);
    assert_eq!(v1.minor, 2);
    assert_eq!(v1.patch, 3);

    let v2 = SemVer::parse("1.9.0").unwrap();
    assert!(v1.is_compatible(&v2), "1.2.3 should be compatible with 1.9.0");

    let v3 = SemVer::parse("2.0.0").unwrap();
    assert!(!v1.is_compatible(&v3), "1.2.3 should NOT be compatible with 2.0.0");

    let v4 = SemVer::parse("0.2.3").unwrap();
    let v5 = SemVer::parse("0.2.9").unwrap();
    let v6 = SemVer::parse("0.3.0").unwrap();
    assert!(v4.is_compatible(&v5), "0.2.3 should be compatible with 0.2.9");
    assert!(!v4.is_compatible(&v6), "0.2.3 should NOT be compatible with 0.3.0");

    assert!(SemVer::parse("invalid").is_none());
    assert!(SemVer::parse("1.2").is_none());
    println!("SemVer parsing and compatibility: OK");

    println!("\n=== Exercise 6: Build Order ===");
    let workspace = vec![
        Crate { name: "my-cli".into(), dependencies: vec!["my-core".into(), "shared-utils".into()] },
        Crate { name: "my-core".into(), dependencies: vec!["shared-utils".into()] },
        Crate { name: "shared-utils".into(), dependencies: vec![] },
        Crate { name: "my-server".into(), dependencies: vec!["my-core".into()] },
    ];
    let order = build_order(&workspace).expect("Should produce a valid build order");
    println!("Build order: {:?}", order);

    // Verify: each crate appears after all its dependencies
    for (i, name) in order.iter().enumerate() {
        let crate_def = workspace.iter().find(|c| &c.name == name).unwrap();
        for dep in &crate_def.dependencies {
            let dep_pos = order.iter().position(|n| n == dep)
                .expect(&format!("{dep} should be in the build order"));
            assert!(dep_pos < i, "{dep} should come before {name}");
        }
    }

    // Test cycle detection
    let cyclic = vec![
        Crate { name: "a".into(), dependencies: vec!["b".into()] },
        Crate { name: "b".into(), dependencies: vec!["a".into()] },
    ];
    assert!(build_order(&cyclic).is_none(), "Cyclic deps should return None");

    println!("\n=== Exercise 7: Profiles ===");
    let dev = Profile::dev();
    assert_eq!(dev.opt_level, 0);
    assert!(dev.debug);
    assert_eq!(dev.lto, LtoSetting::Off);

    let release = Profile::release();
    assert_eq!(release.opt_level, 3);
    assert!(!release.debug);

    // Create a profiling profile: release + debug info
    let profiling = Profile::inherit(
        &release,
        "profiling",
        None,              // keep opt_level from release
        Some(true),        // enable debug
        None,              // keep lto from release
        None,              // keep codegen_units
        Some(false),       // don't strip
    );
    assert_eq!(profiling.name, "profiling");
    assert_eq!(profiling.opt_level, 3);
    assert!(profiling.debug);

    println!("Dev compile time estimate: {}", dev.estimated_compile_time());
    println!("Release compile time estimate: {}", release.estimated_compile_time());
    assert!(dev.estimated_compile_time() < release.estimated_compile_time());

    println!("Dev binary size estimate: {}", dev.estimated_binary_size());
    println!("Release binary size estimate: {}", release.estimated_binary_size());
    assert!(release.estimated_binary_size() < dev.estimated_binary_size());

    println!("\n=== All exercises passed! ===");
}
