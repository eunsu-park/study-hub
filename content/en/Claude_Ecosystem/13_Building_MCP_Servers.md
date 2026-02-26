# Building Custom MCP Servers

**Previous**: [12. Model Context Protocol (MCP)](./12_Model_Context_Protocol.md) | **Next**: [14. Claude Projects and Artifacts](./14_Claude_Projects_and_Artifacts.md)

---

In the previous lesson, you learned what MCP is and how to connect pre-built servers. Now it is time to build your own. This lesson walks through the complete process of creating MCP servers in both TypeScript and Python -- from project scaffolding and defining resources, tools, and prompts, through testing with the MCP Inspector, to deployment and publishing considerations.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Understanding of MCP architecture from Lesson 12
- Familiarity with TypeScript or Python
- Basic knowledge of JSON Schema
- Node.js 18+ (for TypeScript) or Python 3.10+ (for Python)

## Learning Objectives

After completing this lesson, you will be able to:

1. Scaffold and configure an MCP server project in TypeScript and Python
2. Define resources with URI templates and dynamic content handlers
3. Define tools with JSON Schema input validation and handler functions
4. Define prompt templates with typed arguments
5. Configure transports for local (stdio) and remote (HTTP/SSE) deployment
6. Test MCP servers using the MCP Inspector and unit tests
7. Deploy, secure, and publish MCP servers

---

## Table of Contents

1. [MCP Server Architecture Recap](#1-mcp-server-architecture-recap)
2. [TypeScript Implementation](#2-typescript-implementation)
3. [Python Implementation](#3-python-implementation)
4. [Testing MCP Servers](#4-testing-mcp-servers)
5. [Deployment Considerations](#5-deployment-considerations)
6. [Publishing and Sharing](#6-publishing-and-sharing)
7. [Exercises](#7-exercises)
8. [References](#8-references)

---

## 1. MCP Server Architecture Recap

Before diving into code, recall the three primitives that an MCP server can expose:

```
┌─────────────────────────────────────────────────────────────────┐
│                   MCP Server Primitives                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Resources  ──  Data that the model can read                     │
│  ├── Identified by URI (e.g., weather://city/london)             │
│  ├── Return text or binary content                               │
│  └── Can be static or dynamic (URI templates)                    │
│                                                                  │
│  Tools      ──  Functions the model can invoke                   │
│  ├── Defined with name, description, input schema                │
│  ├── Handler receives validated input, returns result             │
│  └── Model decides when to call them (with user approval)        │
│                                                                  │
│  Prompts    ──  Reusable prompt templates                         │
│  ├── Parameterized message templates                              │
│  ├── Can include multi-turn conversations                         │
│  └── Users select them explicitly (not auto-invoked)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

A server communicates with clients (Claude Code, Claude Desktop) over a **transport**:

- **stdio**: Server runs as a child process; communication over stdin/stdout. Best for local tools.
- **Streamable HTTP**: Server runs as an HTTP endpoint; communication via HTTP requests with optional SSE streaming. Best for remote/shared servers.

---

## 2. TypeScript Implementation

TypeScript is the primary language for MCP server development. The official `@modelcontextprotocol/sdk` package provides a high-level API.

### 2.1 Project Setup

```bash
# Create a new MCP server project
mkdir weather-mcp-server && cd weather-mcp-server

# Initialize the project
npm init -y
npm install @modelcontextprotocol/sdk zod
npm install -D typescript @types/node tsx

# Initialize TypeScript
npx tsc --init --target ES2022 --module Node16 --moduleResolution Node16 \
  --outDir dist --rootDir src --strict
```

Create the directory structure:

```
weather-mcp-server/
├── src/
│   └── index.ts        # Main server entry point
├── package.json
└── tsconfig.json
```

Update `package.json`:

```json
{
  "name": "weather-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "bin": {
    "weather-mcp-server": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsx src/index.ts",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.12.0",
    "zod": "^3.24.0"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "tsx": "^4.19.0",
    "typescript": "^5.7.0"
  }
}
```

### 2.2 Defining Resources

Resources represent data that Claude can read. Each resource has a URI, a name, a MIME type, and a handler that returns the content.

```typescript
// src/index.ts
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "weather-server",
  version: "1.0.0",
});

// --- Static Resource ---
// A fixed resource that always returns the same data
server.resource(
  "supported-cities",               // Resource name
  "weather://cities",               // URI
  async (uri) => ({
    contents: [{
      uri: uri.href,
      mimeType: "application/json",
      text: JSON.stringify({
        cities: ["london", "tokyo", "new-york", "seoul", "berlin"],
        lastUpdated: new Date().toISOString(),
      }),
    }],
  })
);

// --- Dynamic Resource with URI Template ---
// The {city} parameter is extracted from the URI
server.resource(
  "city-weather",                       // Resource name
  new ResourceTemplate("weather://city/{city}", { list: undefined }),  // URI template
  async (uri, { city }) => {
    // In production, call a real weather API here
    const weatherData = await fetchWeatherData(city as string);
    return {
      contents: [{
        uri: uri.href,
        mimeType: "application/json",
        text: JSON.stringify(weatherData),
      }],
    };
  }
);
```

**Key points about resources:**
- Static resources have fixed URIs; dynamic resources use URI templates with `{param}` placeholders.
- The `list` property on `ResourceTemplate` controls whether the server can enumerate available values. Setting it to `undefined` means no enumeration; providing a callback allows the server to list available resources.
- Return `mimeType: "application/json"` for structured data or `"text/plain"` for text.

### 2.3 Defining Tools

Tools are functions that Claude can call. Each tool has a name, description, an input schema (using Zod for validation), and a handler.

```typescript
// --- Tool: Get Current Weather ---
server.tool(
  "get-current-weather",                           // Tool name
  "Get the current weather for a city",            // Description (crucial for Claude)
  {                                                 // Input schema (Zod shape)
    city: z.string().describe("City name, e.g., 'london' or 'tokyo'"),
    units: z.enum(["celsius", "fahrenheit"])
      .default("celsius")
      .describe("Temperature unit"),
  },
  async ({ city, units }) => {                      // Handler
    try {
      const weather = await fetchWeatherData(city);
      const temp = units === "fahrenheit"
        ? (weather.temperature * 9/5) + 32
        : weather.temperature;

      return {
        content: [{
          type: "text" as const,
          text: `Weather in ${city}: ${temp}°${units === "celsius" ? "C" : "F"}, ` +
                `${weather.condition}, humidity ${weather.humidity}%`,
        }],
      };
    } catch (error) {
      return {
        content: [{
          type: "text" as const,
          text: `Error fetching weather for "${city}": ${(error as Error).message}`,
        }],
        isError: true,
      };
    }
  }
);

// --- Tool: Get Forecast ---
server.tool(
  "get-forecast",
  "Get a multi-day weather forecast for a city",
  {
    city: z.string().describe("City name"),
    days: z.number().min(1).max(7).default(3)
      .describe("Number of forecast days (1-7)"),
  },
  async ({ city, days }) => {
    const forecast = await fetchForecast(city, days);
    const formatted = forecast.map(
      (day) => `${day.date}: ${day.high}°C / ${day.low}°C, ${day.condition}`
    ).join("\n");

    return {
      content: [{
        type: "text" as const,
        text: `${days}-day forecast for ${city}:\n${formatted}`,
      }],
    };
  }
);
```

**Tool design best practices:**
- Write clear, specific descriptions. Claude uses these to decide when to call the tool.
- Use Zod's `.describe()` on each parameter to help Claude understand what to pass.
- Always handle errors gracefully and return `isError: true` on failure.
- Keep tool outputs concise -- Claude has a context window limit.

### 2.4 Defining Prompts

Prompts are reusable templates that users can select from. They differ from tools: prompts are user-initiated, while tools are model-initiated.

```typescript
// --- Prompt: Weather Briefing ---
server.prompt(
  "weather-briefing",                              // Prompt name
  "Generate a weather briefing for a city",        // Description
  {                                                 // Arguments
    city: z.string().describe("City for the briefing"),
    audience: z.enum(["general", "aviation", "marine"])
      .default("general")
      .describe("Target audience for the briefing"),
  },
  async ({ city, audience }) => {
    const weather = await fetchWeatherData(city);
    const forecast = await fetchForecast(city, 3);

    let systemContext = "You are a meteorologist providing weather briefings.";
    if (audience === "aviation") {
      systemContext += " Use aviation terminology (METAR format where appropriate).";
    } else if (audience === "marine") {
      systemContext += " Include wind speed, wave height, and sea conditions.";
    }

    return {
      messages: [
        {
          role: "user" as const,
          content: {
            type: "text" as const,
            text: `Please provide a ${audience} weather briefing for ${city}.\n\n` +
                  `Current conditions: ${JSON.stringify(weather)}\n` +
                  `3-day forecast: ${JSON.stringify(forecast)}`,
          },
        },
      ],
    };
  }
);
```

### 2.5 Helper Functions and Transport Setup

```typescript
// --- Mock Weather Data (replace with real API calls in production) ---

interface WeatherData {
  city: string;
  temperature: number;
  condition: string;
  humidity: number;
  windSpeed: number;
  timestamp: string;
}

interface ForecastDay {
  date: string;
  high: number;
  low: number;
  condition: string;
}

async function fetchWeatherData(city: string): Promise<WeatherData> {
  // In production, call OpenWeatherMap, WeatherAPI, etc.
  const mockData: Record<string, WeatherData> = {
    london: { city: "London", temperature: 12, condition: "Cloudy",
              humidity: 78, windSpeed: 15, timestamp: new Date().toISOString() },
    tokyo: { city: "Tokyo", temperature: 22, condition: "Sunny",
             humidity: 55, windSpeed: 8, timestamp: new Date().toISOString() },
    seoul: { city: "Seoul", temperature: 18, condition: "Partly Cloudy",
             humidity: 62, windSpeed: 10, timestamp: new Date().toISOString() },
  };

  const data = mockData[city.toLowerCase()];
  if (!data) {
    throw new Error(`City "${city}" not found. Available: ${Object.keys(mockData).join(", ")}`);
  }
  return data;
}

async function fetchForecast(city: string, days: number): Promise<ForecastDay[]> {
  const base = await fetchWeatherData(city);
  return Array.from({ length: days }, (_, i) => ({
    date: new Date(Date.now() + (i + 1) * 86400000).toISOString().split("T")[0],
    high: base.temperature + Math.floor(Math.random() * 5),
    low: base.temperature - Math.floor(Math.random() * 5),
    condition: ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"][Math.floor(Math.random() * 4)],
  }));
}

// --- Start the Server ---

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Weather MCP server running on stdio");
}

main().catch(console.error);
```

### 2.6 Connecting to Claude Code

Register your server in Claude Code's settings:

```json
// ~/.claude/settings.json (global) or .claude/settings.json (project)
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/absolute/path/to/weather-mcp-server/dist/index.js"],
      "env": {
        "WEATHER_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

For development, use `tsx` for hot reloading:

```json
{
  "mcpServers": {
    "weather-dev": {
      "command": "npx",
      "args": ["tsx", "/absolute/path/to/weather-mcp-server/src/index.ts"]
    }
  }
}
```

### 2.7 Remote Transport with Streamable HTTP

For servers that need to be shared across machines or users, use Streamable HTTP transport:

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";

const app = express();
app.use(express.json());

const server = new McpServer({
  name: "weather-server-remote",
  version: "1.0.0",
});

// ... (same resource/tool/prompt definitions as above) ...

// Set up the Streamable HTTP transport
const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });

app.post("/mcp", async (req, res) => {
  await transport.handleRequest(req, res, req.body);
});

// Stateless: no need for GET or DELETE endpoints
// For stateful sessions, add GET (SSE) and DELETE (session cleanup) handlers

await server.connect(transport);

app.listen(3000, () => {
  console.log("MCP server listening on http://localhost:3000/mcp");
});
```

Connect Claude Code to a remote server:

```json
{
  "mcpServers": {
    "weather-remote": {
      "type": "url",
      "url": "http://localhost:3000/mcp"
    }
  }
}
```

---

## 3. Python Implementation

Python is the second officially supported language for MCP server development. The `mcp` package provides a decorator-based API that feels natural in Python.

### 3.1 Project Setup

```bash
# Create project with uv (recommended)
uv init db-query-mcp-server
cd db-query-mcp-server

# Add dependencies
uv add "mcp[cli]"

# Or with pip
pip install "mcp[cli]"
```

### 3.2 Decorator-Based API

The Python SDK uses decorators to define resources, tools, and prompts. Here is a complete database query MCP server:

```python
# server.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP(
    name="db-query-server",
    version="1.0.0",
)

# Database path (configure via environment variable in production)
DB_PATH = Path("./sample.db")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ─── Resources ───────────────────────────────────────────────────────

@mcp.resource("db://schema")
def get_schema() -> str:
    """Return the database schema as text."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    schemas = [row["sql"] for row in cursor.fetchall() if row["sql"]]
    conn.close()
    return "\n\n".join(schemas)


@mcp.resource("db://tables")
def list_tables() -> str:
    """List all tables with row counts."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = []
    for row in cursor.fetchall():
        name = row["name"]
        count = conn.execute(f"SELECT COUNT(*) as cnt FROM [{name}]").fetchone()["cnt"]
        tables.append({"name": name, "rowCount": count})
    conn.close()
    return json.dumps(tables, indent=2)


@mcp.resource("db://table/{table_name}/sample")
def get_table_sample(table_name: str) -> str:
    """Return the first 10 rows of a table as JSON."""
    conn = get_connection()
    # Validate table name to prevent SQL injection
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    if table_name not in tables:
        conn.close()
        return json.dumps({"error": f"Table '{table_name}' not found"})

    cursor = conn.execute(f"SELECT * FROM [{table_name}] LIMIT 10")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return json.dumps(rows, indent=2)


# ─── Tools ───────────────────────────────────────────────────────────

@mcp.tool()
def execute_query(query: str, params: list[str] | None = None) -> str:
    """
    Execute a read-only SQL query against the database.

    Args:
        query: SQL SELECT query to execute. Only SELECT statements are allowed.
        params: Optional list of query parameters for parameterized queries.

    Returns:
        Query results as a JSON array of objects, or an error message.
    """
    # Security: only allow SELECT queries
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        return json.dumps({
            "error": "Only SELECT queries are allowed. "
                     "Use INSERT/UPDATE/DELETE tools for write operations."
        })

    # Block dangerous patterns
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
    for keyword in dangerous:
        # Check if dangerous keywords appear outside of string literals
        if keyword in normalized.split("'")[0::2]:
            return json.dumps({"error": f"Query contains disallowed keyword: {keyword}"})

    conn = get_connection()
    try:
        cursor = conn.execute(query, params or [])
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps({
            "rowCount": len(rows),
            "rows": rows,
            "executedAt": datetime.now().isoformat(),
        }, indent=2, default=str)
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


@mcp.tool()
def describe_table(table_name: str) -> str:
    """
    Get detailed information about a table's structure.

    Args:
        table_name: Name of the table to describe.

    Returns:
        Table structure including columns, types, and constraints.
    """
    conn = get_connection()
    try:
        # Get column info
        cursor = conn.execute(f"PRAGMA table_info([{table_name}])")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row["name"],
                "type": row["type"],
                "nullable": not row["notnull"],
                "defaultValue": row["dflt_value"],
                "primaryKey": bool(row["pk"]),
            })

        # Get index info
        idx_cursor = conn.execute(f"PRAGMA index_list([{table_name}])")
        indexes = []
        for idx in idx_cursor.fetchall():
            idx_info = conn.execute(
                f"PRAGMA index_info([{idx['name']}])"
            ).fetchall()
            indexes.append({
                "name": idx["name"],
                "unique": bool(idx["unique"]),
                "columns": [col["name"] for col in idx_info],
            })

        # Get row count
        count = conn.execute(
            f"SELECT COUNT(*) as cnt FROM [{table_name}]"
        ).fetchone()["cnt"]

        return json.dumps({
            "table": table_name,
            "columns": columns,
            "indexes": indexes,
            "rowCount": count,
        }, indent=2)
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


@mcp.tool()
def explain_query(query: str) -> str:
    """
    Show the query execution plan (EXPLAIN QUERY PLAN).

    Args:
        query: SQL query to analyze.

    Returns:
        Query execution plan as text.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = [dict(row) for row in cursor.fetchall()]
        return json.dumps(plan, indent=2)
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


# ─── Prompts ─────────────────────────────────────────────────────────

@mcp.prompt()
def analyze_table(table_name: str) -> str:
    """Generate a data analysis prompt for a specific table."""
    schema = get_schema()
    sample = get_table_sample(table_name)
    return (
        f"Analyze the '{table_name}' table from this database.\n\n"
        f"Database schema:\n```sql\n{schema}\n```\n\n"
        f"Sample data (first 10 rows):\n```json\n{sample}\n```\n\n"
        "Please provide:\n"
        "1. A description of what this table stores\n"
        "2. Key observations about the data\n"
        "3. Suggested queries for common analysis tasks\n"
        "4. Any data quality issues you notice"
    )


@mcp.prompt()
def optimize_query(query: str) -> str:
    """Generate a query optimization prompt."""
    schema = get_schema()
    return (
        f"Optimize this SQL query for the following database:\n\n"
        f"Query:\n```sql\n{query}\n```\n\n"
        f"Database schema:\n```sql\n{schema}\n```\n\n"
        "Please:\n"
        "1. Analyze the query for performance issues\n"
        "2. Suggest index additions if helpful\n"
        "3. Rewrite the query if a more efficient form exists\n"
        "4. Explain the expected improvement"
    )


# ─── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3.3 Running the Python Server

```bash
# Run directly
python server.py

# Or use the MCP CLI for development
mcp dev server.py

# Register in Claude Code settings
# ~/.claude/settings.json
```

```json
{
  "mcpServers": {
    "db-query": {
      "command": "python",
      "args": ["/absolute/path/to/db-query-mcp-server/server.py"],
      "env": {
        "DB_PATH": "/path/to/your/database.db"
      }
    }
  }
}
```

### 3.4 Python vs TypeScript Comparison

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│ Feature         │ TypeScript           │ Python               │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Package         │ @modelcontextproto-  │ mcp                  │
│                 │ col/sdk              │                      │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Schema          │ Zod objects          │ Type hints +         │
│ Validation      │                      │ docstrings           │
├─────────────────┼──────────────────────┼──────────────────────┤
│ API Style       │ server.tool(name,    │ @mcp.tool()          │
│                 │ desc, schema, fn)    │ decorator            │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Resource URI    │ ResourceTemplate     │ String with {param}  │
│ Templates       │ class                │ in decorator          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Transport       │ StdioServerTransport │ mcp.run(transport=)  │
│ Setup           │ + server.connect()   │                      │
├─────────────────┼──────────────────────┼──────────────────────┤
│ HTTP Transport  │ StreamableHTTP-      │ mcp.run(transport=   │
│                 │ ServerTransport      │ "streamable-http")   │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Error Returns   │ { isError: true }    │ Raise or return      │
│                 │                      │ error string         │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Ecosystem       │ Largest MCP server   │ Growing; great for   │
│                 │ ecosystem            │ data/ML tools        │
└─────────────────┴──────────────────────┴──────────────────────┘
```

---

## 4. Testing MCP Servers

### 4.1 MCP Inspector

The MCP Inspector is an interactive debugging tool that lets you test your server without connecting it to Claude.

```bash
# For TypeScript servers
npx @modelcontextprotocol/inspector node dist/index.js

# For Python servers
npx @modelcontextprotocol/inspector python server.py

# Or use the mcp CLI for Python
mcp dev server.py
```

The Inspector opens a web UI where you can:
- Browse and read resources
- Call tools with custom inputs
- Select and render prompts
- View raw JSON-RPC messages
- Test error handling

### 4.2 Unit Testing Handlers (TypeScript)

Test your tool handlers independently of the MCP framework:

```typescript
// src/__tests__/tools.test.ts
import { describe, it, expect, beforeAll, afterAll } from "vitest";

// Import your handler functions directly
import { fetchWeatherData, fetchForecast } from "../weather.js";

describe("Weather Tools", () => {
  describe("fetchWeatherData", () => {
    it("returns weather data for a known city", async () => {
      const result = await fetchWeatherData("london");
      expect(result).toHaveProperty("city", "London");
      expect(result).toHaveProperty("temperature");
      expect(result).toHaveProperty("condition");
      expect(typeof result.temperature).toBe("number");
    });

    it("throws for an unknown city", async () => {
      await expect(fetchWeatherData("atlantis"))
        .rejects.toThrow(/not found/);
    });
  });

  describe("fetchForecast", () => {
    it("returns the requested number of days", async () => {
      const forecast = await fetchForecast("tokyo", 5);
      expect(forecast).toHaveLength(5);
      forecast.forEach((day) => {
        expect(day).toHaveProperty("date");
        expect(day).toHaveProperty("high");
        expect(day).toHaveProperty("low");
        expect(day).toHaveProperty("condition");
      });
    });
  });
});
```

### 4.3 Unit Testing Handlers (Python)

```python
# tests/test_tools.py
import json
import pytest
import sqlite3
from pathlib import Path

# Set up a test database before importing server
@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@example.com')")
    conn.commit()
    conn.close()

    # Patch the DB_PATH in the server module
    import server
    monkeypatch.setattr(server, "DB_PATH", db_path)


def test_get_schema():
    from server import get_schema
    schema = get_schema()
    assert "CREATE TABLE users" in schema
    assert "name TEXT" in schema


def test_list_tables():
    from server import list_tables
    result = json.loads(list_tables())
    table_names = [t["name"] for t in result]
    assert "users" in table_names


def test_execute_query_select():
    from server import execute_query
    result = json.loads(execute_query("SELECT * FROM users WHERE name = ?", ["Alice"]))
    assert result["rowCount"] == 1
    assert result["rows"][0]["name"] == "Alice"


def test_execute_query_blocks_delete():
    from server import execute_query
    result = json.loads(execute_query("DELETE FROM users WHERE id = 1"))
    assert "error" in result
    assert "Only SELECT" in result["error"]


def test_describe_table():
    from server import describe_table
    result = json.loads(describe_table("users"))
    assert result["table"] == "users"
    column_names = [c["name"] for c in result["columns"]]
    assert "id" in column_names
    assert "name" in column_names
    assert "email" in column_names
```

### 4.4 Integration Testing with Claude Code

Test the end-to-end flow by registering the server and using Claude Code:

```bash
# 1. Build and register the server
cd weather-mcp-server
npm run build

# 2. Add to Claude Code settings (see section 2.6)

# 3. Restart Claude Code to pick up the new server
# Claude Code will show: "Connected to MCP server: weather"

# 4. Test in conversation
# You: "What's the weather in Tokyo?"
# Claude will call the get-current-weather tool and respond with the data
```

You can also use the Claude Code SDK to test programmatically (covered in Lesson 17).

---

## 5. Deployment Considerations

### 5.1 Local vs Remote Hosting

```
┌──────────────────────────────────────────────────────────────────┐
│                  Deployment Options                                │
├──────────────────┬──────────────────┬────────────────────────────┤
│                  │ Local (stdio)    │ Remote (HTTP)              │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Transport        │ stdin/stdout     │ Streamable HTTP (+ SSE)    │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Use case         │ Personal tools,  │ Team-shared tools,         │
│                  │ local files      │ cloud-hosted services      │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Authentication   │ OS-level (local) │ API keys, OAuth, mTLS      │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Performance      │ No network       │ Network latency            │
│                  │ overhead         │                            │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Scaling          │ Single user      │ Multiple concurrent users  │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Examples         │ File system,     │ Database, SaaS APIs,       │
│                  │ local Git        │ monitoring dashboards      │
└──────────────────┴──────────────────┴────────────────────────────┘
```

### 5.2 Authentication and Security

For remote MCP servers, always implement authentication:

```typescript
// Example: API key authentication middleware
import { Request, Response, NextFunction } from "express";

function authenticate(req: Request, res: Response, next: NextFunction) {
  const apiKey = req.headers["x-api-key"] as string;

  if (!apiKey) {
    res.status(401).json({ error: "Missing API key" });
    return;
  }

  // In production, validate against a database or secrets manager
  const validKeys = new Set(process.env.VALID_API_KEYS?.split(",") ?? []);
  if (!validKeys.has(apiKey)) {
    res.status(403).json({ error: "Invalid API key" });
    return;
  }

  next();
}

// Apply to MCP endpoint
app.post("/mcp", authenticate, async (req, res) => {
  await transport.handleRequest(req, res, req.body);
});
```

**Security best practices:**
- **Validate all inputs**: Never trust data from the client. Sanitize tool inputs.
- **Principle of least privilege**: Only expose the minimum necessary capabilities.
- **Rate limiting**: Prevent abuse of expensive operations.
- **Audit logging**: Log all tool invocations with timestamps and inputs.
- **Read-only by default**: Make write operations opt-in and require explicit confirmation.

### 5.3 Error Handling and Logging

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

const server = new McpServer({
  name: "production-server",
  version: "1.0.0",
});

// Global error handling
server.server.onerror = (error) => {
  console.error("[MCP Server Error]", {
    message: error.message,
    timestamp: new Date().toISOString(),
  });
};

// Tool-level error handling pattern
server.tool(
  "risky-operation",
  "An operation that might fail",
  { input: z.string() },
  async ({ input }) => {
    const startTime = Date.now();
    try {
      const result = await performOperation(input);

      // Log successful operations
      console.error(JSON.stringify({
        level: "info",
        tool: "risky-operation",
        input: input.substring(0, 100),  // Truncate for logging
        duration: Date.now() - startTime,
        success: true,
      }));

      return {
        content: [{ type: "text" as const, text: result }],
      };
    } catch (error) {
      // Log failures
      console.error(JSON.stringify({
        level: "error",
        tool: "risky-operation",
        input: input.substring(0, 100),
        duration: Date.now() - startTime,
        error: (error as Error).message,
      }));

      return {
        content: [{
          type: "text" as const,
          text: `Operation failed: ${(error as Error).message}`,
        }],
        isError: true,
      };
    }
  }
);
```

Note: MCP servers using stdio transport must write logs to **stderr** (not stdout), because stdout is reserved for JSON-RPC messages. In Node.js, `console.error()` writes to stderr.

---

## 6. Publishing and Sharing

### 6.1 Publishing to npm (TypeScript)

```bash
# Prepare for publishing
# 1. Update package.json with correct metadata
# 2. Build the project
npm run build

# 3. Test the built output
node dist/index.js

# 4. Publish
npm publish

# Users install with:
# npm install -g your-mcp-server
```

Ensure your `package.json` has the `bin` field so the server can be invoked as a command:

```json
{
  "name": "@yourorg/weather-mcp-server",
  "bin": {
    "weather-mcp-server": "./dist/index.js"
  }
}
```

### 6.2 Publishing to PyPI (Python)

```bash
# Create a pyproject.toml
# Build and publish
uv build
uv publish

# Users install with:
# pip install your-mcp-server
# or: uvx your-mcp-server
```

### 6.3 Listing on the MCP Server Registry

Anthropic maintains a registry of community MCP servers. To list yours:

1. Create a public GitHub repository for your server.
2. Include a clear README with installation and configuration instructions.
3. Add a `LICENSE` file (MIT or Apache-2.0 recommended).
4. Submit to the official MCP servers list (github.com/modelcontextprotocol/servers).

### 6.4 Distribution via Docker

For complex servers with system dependencies:

```dockerfile
FROM node:22-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY dist/ ./dist/
ENTRYPOINT ["node", "dist/index.js"]
```

Register in Claude Code:

```json
{
  "mcpServers": {
    "weather": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "weather-mcp-server:latest"]
    }
  }
}
```

---

## 7. Exercises

### Exercise 1: Basic MCP Server (Beginner)

Create an MCP server that exposes a single tool called `random-number` that generates a random number between a `min` and `max` value provided as inputs. Test it with the MCP Inspector.

### Exercise 2: File System Resources (Intermediate)

Build an MCP server that exposes a project directory as resources. Include:
- A static resource listing all files in the directory.
- A dynamic resource `file://path/{filepath}` that returns file contents.
- A tool `search-files` that searches for a pattern across all files.

### Exercise 3: REST API Wrapper (Intermediate)

Create an MCP server that wraps a public REST API (e.g., the GitHub API). Include:
- Resources for fetching repository info.
- Tools for searching repositories and listing issues.
- A prompt template for generating a repository summary.
Handle rate limiting and authentication.

### Exercise 4: Database Query Server (Advanced)

Extend the Python database server from section 3.2 to support:
- Write operations (INSERT/UPDATE) with a confirmation prompt.
- Query history tracking.
- Automatic query timeout after 30 seconds.
- Schema migration suggestions based on query patterns.

### Exercise 5: Multi-Service MCP Server (Advanced)

Build an MCP server that aggregates data from multiple sources (e.g., database + REST API + local files). Implement:
- Cross-source queries (e.g., "find all users who have open GitHub issues").
- Caching with TTL for expensive operations.
- Health check endpoints for each data source.

---

## 8. References

- MCP Specification - https://spec.modelcontextprotocol.io
- MCP TypeScript SDK - https://github.com/modelcontextprotocol/typescript-sdk
- MCP Python SDK - https://github.com/modelcontextprotocol/python-sdk
- MCP Server Registry - https://github.com/modelcontextprotocol/servers
- MCP Inspector - https://github.com/modelcontextprotocol/inspector
- Building MCP Servers (Anthropic Docs) - https://docs.anthropic.com/en/docs/agents-and-tools/mcp
- JSON Schema Reference - https://json-schema.org/understanding-json-schema/

---

## Next Lesson

[14. Claude Projects and Artifacts](./14_Claude_Projects_and_Artifacts.md) explores how to organize knowledge and context in Claude Projects, leverage artifacts for prototyping, and understand when to use Projects versus CLAUDE.md for project configuration.
