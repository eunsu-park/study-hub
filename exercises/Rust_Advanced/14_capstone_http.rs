// Exercise: Capstone HTTP Server (Conceptual)
// This capstone requires multiple crates. Create a Cargo project.
//
// [dependencies]
// axum = { version = "0.8", features = ["macros"] }
// tokio = { version = "1", features = ["full"] }
// sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite", "migrate"] }
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"
// jsonwebtoken = "9"
// argon2 = "0.5"
// thiserror = "2"
// anyhow = "1"
// tracing = "0.1"
// tracing-subscriber = "0.3"

// Build a complete blog API with:
// 1. User registration and login (JWT + Argon2)
// 2. CRUD for posts (with ownership checks)
// 3. Comments on posts
// 4. Pagination for list endpoints
// 5. Proper error handling (typed errors → HTTP status codes)
// 6. Middleware (timing, CORS, compression)
// 7. Integration tests with axum-test
// 8. Docker deployment

// See Lesson 30 for the complete architecture and code.

fn main() {
    println!("Capstone: HTTP Server");
    println!("This is a project-based exercise.");
    println!();
    println!("Steps:");
    println!("  1. cargo new blog-api");
    println!("  2. Add dependencies (see above)");
    println!("  3. Create migrations/ directory with SQL schema");
    println!("  4. Implement modules: config, error, auth, models, routes");
    println!("  5. Add integration tests");
    println!("  6. Dockerize");
    println!();
    println!("See Lesson 30 for detailed guidance.");
}
