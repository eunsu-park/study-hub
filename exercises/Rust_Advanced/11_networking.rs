// Exercise: Network Programming (Conceptual)
// These exercises require tokio and axum. Create a Cargo project.
//
// [dependencies]
// axum = "0.8"
// tokio = { version = "1", features = ["full"] }
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"

// Exercise 1: Multi-room chat server (TCP)
// Clients send "/join room_name" to join a room.
// Messages are broadcast only to members of the same room.

// Exercise 2: REST API for a blog
// GET /posts — list all (paginated)
// POST /posts — create
// GET /posts/:id — get one
// PUT /posts/:id — update
// DELETE /posts/:id — delete
// Include proper error responses (404, 400, etc.)

// Exercise 3: WebSocket dashboard
// Stream system metrics (simulated CPU, memory) every second
// to all connected WebSocket clients.

// Exercise 4: HTTP proxy
// Forward requests to the target server.
// Log method, URL, status, and response time.
// Support a configurable domain blocklist.

// Exercise 5: File upload server
// Accept multipart file uploads.
// Store with hash-based names for deduplication.
// Serve files back with correct MIME types.

fn main() {
    println!("Network programming exercises require tokio + axum.");
    println!("Create a Cargo project for each exercise.");
}
