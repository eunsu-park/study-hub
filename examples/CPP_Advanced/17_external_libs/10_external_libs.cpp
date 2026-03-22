/**
 * Example: External C++ Libraries
 * Topic: CPP – Lesson 22
 *
 * Demonstrates usage patterns for popular C++ libraries:
 *   1. nlohmann/json — JSON parsing and serialization
 *   2. fmt — Type-safe string formatting
 *   3. spdlog — Structured logging
 *   4. Boost.Asio — Async I/O (TCP server)
 *   5. CMake integration examples
 *
 * This file shows the API patterns. To compile, install the libraries
 * via vcpkg or Conan and use the CMakeLists.txt below.
 *
 * CMakeLists.txt:
 *   cmake_minimum_required(VERSION 3.20)
 *   project(ext_libs_demo LANGUAGES CXX)
 *   find_package(nlohmann_json REQUIRED)
 *   find_package(fmt REQUIRED)
 *   find_package(spdlog REQUIRED)
 *   add_executable(demo 10_external_libs.cpp)
 *   target_link_libraries(demo PRIVATE
 *       nlohmann_json::nlohmann_json fmt::fmt spdlog::spdlog)
 *   target_compile_features(demo PRIVATE cxx_std_20)
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>


// ============================================================
// Demo 1: nlohmann/json patterns
// ============================================================
/*
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Auto-serialization for custom structs
struct ServerConfig {
    std::string host;
    int port;
    bool tls;
    std::vector<std::string> allowed_origins;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        ServerConfig, host, port, tls, allowed_origins)
};

void demo_json() {
    std::cout << "=== Demo 1: nlohmann/json ===\n\n";

    // Create JSON from initializer list
    json config = {
        {"server", {
            {"host", "0.0.0.0"},
            {"port", 8080},
            {"tls", true},
            {"allowed_origins", {"localhost", "example.com"}}
        }},
        {"database", {
            {"url", "postgres://localhost:5432/mydb"},
            {"pool_size", 10}
        }},
        {"features", {
            {"caching", true},
            {"rate_limit", 100}
        }}
    };

    // Pretty print
    std::cout << "  Config:\n" << config.dump(4) << "\n\n";

    // Access nested values
    std::string host = config["server"]["host"];
    int port = config["server"]["port"];
    std::cout << "  Server: " << host << ":" << port << "\n";

    // Safe access with default
    int timeout = config.value("/server/timeout"_json_pointer, 30);
    std::cout << "  Timeout (default): " << timeout << "\n";

    // Iterate
    std::cout << "\n  Top-level keys: ";
    for (auto& [key, val] : config.items()) {
        std::cout << key << " ";
    }
    std::cout << "\n";

    // Struct serialization
    ServerConfig sc{"0.0.0.0", 443, true, {"*.example.com"}};
    json j = sc;
    std::cout << "\n  Serialized struct: " << j.dump() << "\n";

    // Struct deserialization
    auto sc2 = j.get<ServerConfig>();
    std::cout << "  Deserialized: " << sc2.host << ":" << sc2.port << "\n";

    // JSON merge patch (RFC 7396)
    json defaults = {{"host", "localhost"}, {"port", 80}, {"debug", false}};
    json overrides = {{"port", 8080}, {"debug", true}};
    defaults.merge_patch(overrides);
    std::cout << "\n  Merged: " << defaults.dump() << "\n\n";
}
*/


// ============================================================
// Demo 2: fmt formatting patterns
// ============================================================
/*
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

// Custom formatter for user-defined types
struct Point { double x, y; };

template<>
struct fmt::formatter<Point> {
    bool use_parens = true;

    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && *it == 'b') {  // bracket format
            use_parens = false;
            ++it;
        }
        return it;
    }

    auto format(const Point& p, format_context& ctx) const {
        if (use_parens)
            return fmt::format_to(ctx.out(), "({:.2f}, {:.2f})", p.x, p.y);
        else
            return fmt::format_to(ctx.out(), "[{:.2f}, {:.2f}]", p.x, p.y);
    }
};

void demo_fmt() {
    std::cout << "=== Demo 2: fmt library ===\n\n";

    // Basic formatting
    fmt::println("  Hello, {}!", "fmt");
    fmt::println("  int={}, float={:.3f}, bool={}", 42, 3.14159, true);

    // Alignment and fill
    fmt::println("  {:>15}", "right-aligned");
    fmt::println("  {:*^15}", "centered");
    fmt::println("  {:<15}", "left-aligned");

    // Number formatting
    fmt::println("  Hex: {:#x}, Oct: {:#o}", 255, 255);
    fmt::println("  Thousands: {:,}", 1234567890);
    fmt::println("  Scientific: {:.3e}", 0.000123456);

    // Containers (fmt::ranges.h)
    std::vector<int> v = {1, 2, 3, 4, 5};
    fmt::println("  Vector: {}", v);
    fmt::println("  Joined: {}", fmt::join(v, " | "));

    // Maps
    std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
    fmt::println("  Map: {}", m);

    // Custom type
    Point p{1.5, 2.7};
    fmt::println("  Point: {}", p);      // (1.50, 2.70)
    fmt::println("  Point: {:b}", p);    // [1.50, 2.70]

    // Colored output (terminal)
    fmt::print(fg(fmt::color::green), "  Success!\n");
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "  Error!\n");

    // Chrono
    auto now = std::chrono::system_clock::now();
    fmt::println("  Time: {:%Y-%m-%d %H:%M:%S}", now);
    fmt::println();
}
*/


// ============================================================
// Demo 3: spdlog structured logging
// ============================================================
/*
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

void demo_spdlog() {
    std::cout << "=== Demo 3: spdlog ===\n\n";

    // Console logger with colors
    auto console = spdlog::stdout_color_mt("console");
    console->set_level(spdlog::level::debug);

    console->debug("Debug message: x={}", 42);
    console->info("Starting server on port {}", 8080);
    console->warn("Memory usage: {:.1f}%", 85.3);
    console->error("Failed to connect: {}", "timeout");

    // File logger with rotation
    auto file_logger = spdlog::rotating_logger_mt(
        "file", "logs/app.log",
        1024 * 1024 * 5,  // 5 MB
        3                   // 3 backup files
    );
    file_logger->info("Application started");

    // Custom format
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

    // Structured logging with key-value pairs
    console->info("request: method={} path={} status={} latency_ms={}",
                  "GET", "/api/users", 200, 42);

    spdlog::shutdown();
    std::cout << "  (check terminal for colored log output)\n\n";
}
*/


// ============================================================
// Demo 4: Boost.Asio TCP echo server pattern
// ============================================================
/*
#include <boost/asio.hpp>
#include <boost/asio/co_spawn.hpp>

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

// Coroutine-based async echo handler
asio::awaitable<void> echo_session(tcp::socket socket) {
    try {
        char data[1024];
        while (true) {
            std::size_t n = co_await socket.async_read_some(
                asio::buffer(data), asio::use_awaitable);
            co_await asio::async_write(
                socket, asio::buffer(data, n), asio::use_awaitable);
        }
    } catch (std::exception& e) {
        // Client disconnected
    }
}

asio::awaitable<void> listener() {
    auto executor = co_await asio::this_coro::executor;
    tcp::acceptor acceptor(executor, {tcp::v4(), 8080});

    while (true) {
        tcp::socket socket = co_await acceptor.async_accept(
            asio::use_awaitable);
        asio::co_spawn(executor,
            echo_session(std::move(socket)),
            asio::detached);
    }
}

void demo_asio() {
    std::cout << "=== Demo 4: Boost.Asio ===\n\n";
    std::cout << "  TCP echo server on port 8080\n";
    std::cout << "  (Uncomment and link Boost.Asio to run)\n\n";

    // asio::io_context io;
    // asio::co_spawn(io, listener(), asio::detached);
    // io.run();
}
*/


// ============================================================
// Standalone Demo (no external dependencies)
// ============================================================
// This section runs without any external libraries installed,
// demonstrating the patterns and API shapes.

void demo_standalone() {
    std::cout << "=== External Libraries Demo (API Patterns) ===\n\n";

    // --- nlohmann/json-like pattern ---
    std::cout << "1. JSON Pattern (nlohmann/json):\n";
    std::cout << R"(
   json config = {
       {"host", "localhost"},
       {"port", 8080},
       {"features", {"cache", "auth"}}
   };
   std::string host = config["host"];
   auto j = my_struct;  // auto-serialize
)" << "\n";

    // --- fmt-like pattern ---
    std::cout << "2. Format Pattern (fmt):\n";
    std::cout << R"(
   fmt::println("Hello, {}!", name);
   fmt::println("{:>10} {:*^10}", "right", "center");
   fmt::println("Vector: {}", std::vector{1,2,3});
)" << "\n";

    // --- spdlog pattern ---
    std::cout << "3. Logging Pattern (spdlog):\n";
    std::cout << R"(
   spdlog::info("Request: method={} path={} status={}", "GET", "/", 200);
   spdlog::error("Connection failed: {}", ec.message());
)" << "\n";

    // --- Boost.Asio pattern ---
    std::cout << "4. Async I/O Pattern (Boost.Asio):\n";
    std::cout << R"(
   asio::awaitable<void> handler(tcp::socket sock) {
       auto n = co_await sock.async_read_some(buf, use_awaitable);
       co_await asio::async_write(sock, buf, use_awaitable);
   }
)" << "\n";

    // --- CMake integration ---
    std::cout << "5. CMake Integration:\n";
    std::cout << R"(
   # vcpkg manifest (vcpkg.json)
   {"dependencies": ["nlohmann-json", "fmt", "spdlog", "boost-asio"]}

   # CMakeLists.txt
   find_package(nlohmann_json REQUIRED)
   find_package(fmt REQUIRED)
   target_link_libraries(app PRIVATE nlohmann_json::nlohmann_json fmt::fmt)
)" << "\n";
}


int main() {
    demo_standalone();

    // Uncomment individual demos after installing dependencies:
    // demo_json();
    // demo_fmt();
    // demo_spdlog();
    // demo_asio();

    return 0;
}
