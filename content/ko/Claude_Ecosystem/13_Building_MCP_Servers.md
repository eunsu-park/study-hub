# 커스텀 MCP 서버 구축

**이전**: [12. 모델 컨텍스트 프로토콜(MCP)](./12_Model_Context_Protocol.md) | **다음**: [14. Claude Projects와 Artifacts](./14_Claude_Projects_and_Artifacts.md)

---

이전 레슨에서 MCP가 무엇인지, 사전 구축된 서버를 어떻게 연결하는지 배웠습니다. 이제 직접 구축할 차례입니다. 이 레슨에서는 TypeScript와 Python 양쪽에서 MCP 서버를 만드는 전체 과정을 안내합니다. 프로젝트 스캐폴딩과 리소스, 도구, 프롬프트 정의에서 시작하여 MCP 인스펙터로 테스트하고 배포 및 게시 고려사항까지 다룹니다.

**난이도**: ⭐⭐⭐

**사전 요구 사항**:
- 레슨 12에서 MCP 아키텍처 이해
- TypeScript 또는 Python에 대한 친숙함
- JSON Schema에 대한 기본 지식
- Node.js 18+ (TypeScript용) 또는 Python 3.10+ (Python용)

**학습 목표**:
- TypeScript와 Python에서 MCP 서버 프로젝트를 스캐폴딩하고 구성할 수 있다
- URI 템플릿과 동적 콘텐츠 핸들러로 리소스를 정의할 수 있다
- JSON Schema 입력 유효성 검사와 핸들러 함수로 도구를 정의할 수 있다
- 타입이 지정된 인수로 프롬프트 템플릿을 정의할 수 있다
- 로컬(stdio)과 원격(HTTP/SSE) 배포를 위한 전송을 설정할 수 있다
- MCP 인스펙터와 단위 테스트를 사용하여 MCP 서버를 테스트할 수 있다
- MCP 서버를 배포, 보안 설정 및 게시할 수 있다

---

## 목차

1. [MCP 서버 아키텍처 복습](#1-mcp-서버-아키텍처-복습)
2. [TypeScript 구현](#2-typescript-구현)
3. [Python 구현](#3-python-구현)
4. [MCP 서버 테스트](#4-mcp-서버-테스트)
5. [배포 고려사항](#5-배포-고려사항)
6. [게시 및 공유](#6-게시-및-공유)
7. [연습 문제](#7-연습-문제)
8. [참고 자료](#8-참고-자료)

---

## 1. MCP 서버 아키텍처 복습

코드에 들어가기 전에, MCP 서버가 노출할 수 있는 세 가지 기본 요소를 다시 살펴봅니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                   MCP 서버 기본 요소                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  리소스(Resources)  ──  모델이 읽을 수 있는 데이터               │
│  ├── URI로 식별 (예: weather://city/london)                      │
│  ├── 텍스트 또는 바이너리 콘텐츠 반환                            │
│  └── 정적 또는 동적 가능 (URI 템플릿)                            │
│                                                                  │
│  도구(Tools)        ──  모델이 호출할 수 있는 함수               │
│  ├── 이름, 설명, 입력 스키마로 정의                              │
│  ├── 핸들러가 검증된 입력을 받아 결과 반환                       │
│  └── 모델이 호출 시점 결정 (사용자 승인 후)                      │
│                                                                  │
│  프롬프트(Prompts)  ──  재사용 가능한 프롬프트 템플릿            │
│  ├── 파라미터화된 메시지 템플릿                                   │
│  ├── 다중 턴 대화 포함 가능                                      │
│  └── 사용자가 명시적으로 선택 (자동 호출 아님)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

서버는 **전송(transport)** 을 통해 클라이언트(Claude Code, Claude Desktop)와 통신합니다:

- **stdio**: 서버가 자식 프로세스로 실행되며 stdin/stdout을 통해 통신. 로컬 도구에 적합.
- **스트리밍 가능한 HTTP(Streamable HTTP)**: 서버가 HTTP 엔드포인트로 실행되며 선택적 SSE 스트리밍이 있는 HTTP 요청으로 통신. 원격/공유 서버에 적합.

---

## 2. TypeScript 구현

TypeScript는 MCP 서버 개발을 위한 주요 언어입니다. 공식 `@modelcontextprotocol/sdk` 패키지가 고수준 API를 제공합니다.

### 2.1 프로젝트 설정

```bash
# 새 MCP 서버 프로젝트 생성
mkdir weather-mcp-server && cd weather-mcp-server

# 프로젝트 초기화
npm init -y
npm install @modelcontextprotocol/sdk zod
npm install -D typescript @types/node tsx

# TypeScript 초기화
npx tsc --init --target ES2022 --module Node16 --moduleResolution Node16 \
  --outDir dist --rootDir src --strict
```

디렉토리 구조 생성:

```
weather-mcp-server/
├── src/
│   └── index.ts        # 메인 서버 진입점
├── package.json
└── tsconfig.json
```

`package.json` 업데이트:

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

### 2.2 리소스 정의

리소스는 Claude가 읽을 수 있는 데이터를 나타냅니다. 각 리소스는 URI, 이름, MIME 타입, 그리고 콘텐츠를 반환하는 핸들러를 가집니다.

```typescript
// src/index.ts
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "weather-server",
  version: "1.0.0",
});

// --- 정적 리소스 ---
// 항상 동일한 데이터를 반환하는 고정 리소스
server.resource(
  "supported-cities",               // 리소스 이름
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

// --- URI 템플릿을 사용한 동적 리소스 ---
// {city} 파라미터가 URI에서 추출됨
server.resource(
  "city-weather",                       // 리소스 이름
  new ResourceTemplate("weather://city/{city}", { list: undefined }),  // URI 템플릿
  async (uri, { city }) => {
    // 프로덕션에서는 실제 날씨 API 호출
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

**리소스에 대한 주요 사항:**
- 정적 리소스는 고정 URI를 가지며, 동적 리소스는 `{param}` 플레이스홀더를 사용하는 URI 템플릿을 사용합니다.
- `ResourceTemplate`의 `list` 속성은 서버가 사용 가능한 값을 열거할 수 있는지를 제어합니다. `undefined`로 설정하면 열거가 없으며, 콜백을 제공하면 서버가 사용 가능한 리소스를 나열할 수 있습니다.
- 구조화된 데이터에는 `mimeType: "application/json"`, 텍스트에는 `"text/plain"`을 반환합니다.

### 2.3 도구 정의

도구는 Claude가 호출할 수 있는 함수입니다. 각 도구는 이름, 설명, 입력 스키마(유효성 검사를 위해 Zod 사용), 그리고 핸들러를 가집니다.

```typescript
// --- 도구: 현재 날씨 가져오기 ---
server.tool(
  "get-current-weather",                           // 도구 이름
  "Get the current weather for a city",            // 설명 (Claude에게 중요)
  {                                                 // 입력 스키마 (Zod 형태)
    city: z.string().describe("City name, e.g., 'london' or 'tokyo'"),
    units: z.enum(["celsius", "fahrenheit"])
      .default("celsius")
      .describe("Temperature unit"),
  },
  async ({ city, units }) => {                      // 핸들러
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

// --- 도구: 예보 가져오기 ---
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

**도구 설계 모범 사례:**
- 명확하고 구체적인 설명을 작성합니다. Claude는 이것을 사용하여 도구를 언제 호출할지 결정합니다.
- 각 파라미터에 Zod의 `.describe()`를 사용하여 Claude가 무엇을 전달할지 이해하도록 돕습니다.
- 항상 오류를 우아하게 처리하고 실패 시 `isError: true`를 반환합니다.
- 도구 출력을 간결하게 유지합니다 — Claude는 컨텍스트 창 제한이 있습니다.

### 2.4 프롬프트 정의

프롬프트는 사용자가 선택할 수 있는 재사용 가능한 템플릿입니다. 도구와 다릅니다: 프롬프트는 사용자가 시작하고, 도구는 모델이 시작합니다.

```typescript
// --- 프롬프트: 날씨 브리핑 ---
server.prompt(
  "weather-briefing",                              // 프롬프트 이름
  "Generate a weather briefing for a city",        // 설명
  {                                                 // 인수
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

### 2.5 헬퍼 함수와 전송 설정

```typescript
// --- 모의 날씨 데이터 (프로덕션에서는 실제 API 호출로 교체) ---

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
  // 프로덕션에서는 OpenWeatherMap, WeatherAPI 등을 호출
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

// --- 서버 시작 ---

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Weather MCP server running on stdio");
}

main().catch(console.error);
```

### 2.6 Claude Code에 연결

Claude Code의 설정에 서버를 등록합니다:

```json
// ~/.claude/settings.json (전역) 또는 .claude/settings.json (프로젝트)
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

개발 시 핫 리로딩을 위해 `tsx`를 사용합니다:

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

### 2.7 스트리밍 가능한 HTTP를 사용한 원격 전송

여러 머신이나 사용자 간에 공유해야 하는 서버의 경우 스트리밍 가능한 HTTP 전송을 사용합니다:

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

// ... (위와 동일한 리소스/도구/프롬프트 정의) ...

// 스트리밍 가능한 HTTP 전송 설정
const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });

app.post("/mcp", async (req, res) => {
  await transport.handleRequest(req, res, req.body);
});

// 무상태: GET 또는 DELETE 엔드포인트 불필요
// 상태 유지 세션의 경우 GET (SSE) 및 DELETE (세션 정리) 핸들러 추가

await server.connect(transport);

app.listen(3000, () => {
  console.log("MCP server listening on http://localhost:3000/mcp");
});
```

원격 서버에 Claude Code 연결:

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

## 3. Python 구현

Python은 MCP 서버 개발을 위해 공식적으로 지원되는 두 번째 언어입니다. `mcp` 패키지는 Python에서 자연스러운 데코레이터 기반 API를 제공합니다.

### 3.1 프로젝트 설정

```bash
# uv로 프로젝트 생성 (권장)
uv init db-query-mcp-server
cd db-query-mcp-server

# 의존성 추가
uv add "mcp[cli]"

# 또는 pip 사용
pip install "mcp[cli]"
```

### 3.2 데코레이터 기반 API

Python SDK는 데코레이터를 사용하여 리소스, 도구, 프롬프트를 정의합니다. 다음은 완전한 데이터베이스 쿼리 MCP 서버입니다:

```python
# server.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# MCP 서버 초기화
mcp = FastMCP(
    name="db-query-server",
    version="1.0.0",
)

# 데이터베이스 경로 (프로덕션에서는 환경 변수로 설정)
DB_PATH = Path("./sample.db")


def get_connection() -> sqlite3.Connection:
    """행 팩토리를 사용한 데이터베이스 연결 가져오기."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ─── 리소스 ───────────────────────────────────────────────────────

@mcp.resource("db://schema")
def get_schema() -> str:
    """데이터베이스 스키마를 텍스트로 반환."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    schemas = [row["sql"] for row in cursor.fetchall() if row["sql"]]
    conn.close()
    return "\n\n".join(schemas)


@mcp.resource("db://tables")
def list_tables() -> str:
    """행 수와 함께 모든 테이블 나열."""
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
    """테이블의 처음 10개 행을 JSON으로 반환."""
    conn = get_connection()
    # SQL 인젝션 방지를 위해 테이블 이름 유효성 검사
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


# ─── 도구 ───────────────────────────────────────────────────────────

@mcp.tool()
def execute_query(query: str, params: list[str] | None = None) -> str:
    """
    데이터베이스에 읽기 전용 SQL 쿼리를 실행합니다.

    Args:
        query: 실행할 SQL SELECT 쿼리. SELECT 문만 허용됩니다.
        params: 파라미터화된 쿼리를 위한 선택적 쿼리 파라미터 목록.

    Returns:
        객체의 JSON 배열로서의 쿼리 결과 또는 오류 메시지.
    """
    # 보안: SELECT 쿼리만 허용
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        return json.dumps({
            "error": "Only SELECT queries are allowed. "
                     "Use INSERT/UPDATE/DELETE tools for write operations."
        })

    # 위험한 패턴 차단
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
    for keyword in dangerous:
        # 문자열 리터럴 외부에서 위험한 키워드가 나타나는지 확인
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
    테이블 구조에 대한 자세한 정보를 가져옵니다.

    Args:
        table_name: 설명할 테이블의 이름.

    Returns:
        열, 타입, 제약 조건을 포함한 테이블 구조.
    """
    conn = get_connection()
    try:
        # 열 정보 가져오기
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

        # 인덱스 정보 가져오기
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

        # 행 수 가져오기
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
    쿼리 실행 계획을 보여줍니다 (EXPLAIN QUERY PLAN).

    Args:
        query: 분석할 SQL 쿼리.

    Returns:
        텍스트로서의 쿼리 실행 계획.
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


# ─── 프롬프트 ─────────────────────────────────────────────────────────

@mcp.prompt()
def analyze_table(table_name: str) -> str:
    """특정 테이블에 대한 데이터 분석 프롬프트 생성."""
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
    """쿼리 최적화 프롬프트 생성."""
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


# ─── 진입점 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3.3 Python 서버 실행

```bash
# 직접 실행
python server.py

# 또는 개발용으로 MCP CLI 사용
mcp dev server.py

# Claude Code 설정에 등록
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

### 3.4 Python과 TypeScript 비교

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│ 기능            │ TypeScript           │ Python               │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 패키지          │ @modelcontextproto-  │ mcp                  │
│                 │ col/sdk              │                      │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 스키마          │ Zod 객체             │ 타입 힌트 +          │
│ 유효성 검사     │                      │ 독스트링             │
├─────────────────┼──────────────────────┼──────────────────────┤
│ API 스타일      │ server.tool(name,    │ @mcp.tool()          │
│                 │ desc, schema, fn)    │ 데코레이터           │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 리소스 URI      │ ResourceTemplate     │ 데코레이터의 {param} │
│ 템플릿          │ 클래스               │ 문자열               │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 전송            │ StdioServerTransport │ mcp.run(transport=)  │
│ 설정            │ + server.connect()   │                      │
├─────────────────┼──────────────────────┼──────────────────────┤
│ HTTP 전송       │ StreamableHTTP-      │ mcp.run(transport=   │
│                 │ ServerTransport      │ "streamable-http")   │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 오류 반환       │ { isError: true }    │ 발생 또는 오류       │
│                 │                      │ 문자열 반환          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 생태계          │ 가장 큰 MCP 서버     │ 성장 중; 데이터/ML   │
│                 │ 생태계               │ 도구에 적합          │
└─────────────────┴──────────────────────┴──────────────────────┘
```

---

## 4. MCP 서버 테스트

### 4.1 MCP 인스펙터

MCP 인스펙터는 Claude에 연결하지 않고도 서버를 테스트할 수 있는 대화형 디버깅 도구입니다.

```bash
# TypeScript 서버의 경우
npx @modelcontextprotocol/inspector node dist/index.js

# Python 서버의 경우
npx @modelcontextprotocol/inspector python server.py

# 또는 Python용 mcp CLI 사용
mcp dev server.py
```

인스펙터는 다음을 할 수 있는 웹 UI를 엽니다:
- 리소스 탐색 및 읽기
- 사용자 정의 입력으로 도구 호출
- 프롬프트 선택 및 렌더링
- 원시 JSON-RPC 메시지 확인
- 오류 처리 테스트

### 4.2 핸들러 단위 테스트 (TypeScript)

MCP 프레임워크와 독립적으로 도구 핸들러를 테스트합니다:

```typescript
// src/__tests__/tools.test.ts
import { describe, it, expect, beforeAll, afterAll } from "vitest";

// 핸들러 함수를 직접 임포트
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

### 4.3 핸들러 단위 테스트 (Python)

```python
# tests/test_tools.py
import json
import pytest
import sqlite3
from pathlib import Path

# 서버를 임포트하기 전에 테스트 데이터베이스 설정
@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """임시 테스트 데이터베이스 생성."""
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

    # 서버 모듈의 DB_PATH 패치
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

### 4.4 Claude Code를 사용한 통합 테스트

서버를 등록하고 Claude Code를 사용하여 엔드투엔드 흐름을 테스트합니다:

```bash
# 1. 서버 빌드 및 등록
cd weather-mcp-server
npm run build

# 2. Claude Code 설정에 추가 (2.6절 참조)

# 3. 새 서버를 가져오도록 Claude Code 재시작
# Claude Code가 표시: "Connected to MCP server: weather"

# 4. 대화에서 테스트
# 사용자: "도쿄의 날씨는 어떻습니까?"
# Claude가 get-current-weather 도구를 호출하고 데이터로 응답
```

레슨 17에서 다루는 Claude Code SDK를 사용하여 프로그래밍 방식으로 테스트할 수도 있습니다.

---

## 5. 배포 고려사항

### 5.1 로컬 대 원격 호스팅

```
┌──────────────────────────────────────────────────────────────────┐
│                    배포 옵션                                      │
├──────────────────┬──────────────────┬────────────────────────────┤
│                  │ 로컬 (stdio)     │ 원격 (HTTP)                │
├──────────────────┼──────────────────┼────────────────────────────┤
│ 전송             │ stdin/stdout     │ 스트리밍 가능한 HTTP (+ SSE)│
├──────────────────┼──────────────────┼────────────────────────────┤
│ 사용 사례        │ 개인 도구,       │ 팀 공유 도구,             │
│                  │ 로컬 파일        │ 클라우드 호스팅 서비스     │
├──────────────────┼──────────────────┼────────────────────────────┤
│ 인증             │ OS 수준 (로컬)   │ API 키, OAuth, mTLS        │
├──────────────────┼──────────────────┼────────────────────────────┤
│ 성능             │ 네트워크 없음    │ 네트워크 지연              │
│                  │ 오버헤드 없음    │                            │
├──────────────────┼──────────────────┼────────────────────────────┤
│ 스케일링         │ 단일 사용자      │ 여러 동시 사용자           │
├──────────────────┼──────────────────┼────────────────────────────┤
│ 예시             │ 파일 시스템,     │ 데이터베이스, SaaS API,    │
│                  │ 로컬 Git         │ 모니터링 대시보드          │
└──────────────────┴──────────────────┴────────────────────────────┘
```

### 5.2 인증 및 보안

원격 MCP 서버의 경우 항상 인증을 구현합니다:

```typescript
// 예시: API 키 인증 미들웨어
import { Request, Response, NextFunction } from "express";

function authenticate(req: Request, res: Response, next: NextFunction) {
  const apiKey = req.headers["x-api-key"] as string;

  if (!apiKey) {
    res.status(401).json({ error: "Missing API key" });
    return;
  }

  // 프로덕션에서는 데이터베이스 또는 시크릿 매니저에 대해 유효성 검사
  const validKeys = new Set(process.env.VALID_API_KEYS?.split(",") ?? []);
  if (!validKeys.has(apiKey)) {
    res.status(403).json({ error: "Invalid API key" });
    return;
  }

  next();
}

// MCP 엔드포인트에 적용
app.post("/mcp", authenticate, async (req, res) => {
  await transport.handleRequest(req, res, req.body);
});
```

**보안 모범 사례:**
- **모든 입력 유효성 검사**: 클라이언트의 데이터를 절대 신뢰하지 않습니다. 도구 입력을 살균합니다.
- **최소 권한 원칙**: 필요한 최소한의 기능만 노출합니다.
- **속도 제한(Rate limiting)**: 비용이 많이 드는 작업의 남용을 방지합니다.
- **감사 로깅(Audit logging)**: 타임스탬프와 입력을 포함한 모든 도구 호출을 기록합니다.
- **기본적으로 읽기 전용**: 쓰기 작업을 옵트인으로 만들고 명시적 확인을 요구합니다.

### 5.3 오류 처리 및 로깅

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

const server = new McpServer({
  name: "production-server",
  version: "1.0.0",
});

// 전역 오류 처리
server.server.onerror = (error) => {
  console.error("[MCP Server Error]", {
    message: error.message,
    timestamp: new Date().toISOString(),
  });
};

// 도구 수준 오류 처리 패턴
server.tool(
  "risky-operation",
  "An operation that might fail",
  { input: z.string() },
  async ({ input }) => {
    const startTime = Date.now();
    try {
      const result = await performOperation(input);

      // 성공한 작업 기록
      console.error(JSON.stringify({
        level: "info",
        tool: "risky-operation",
        input: input.substring(0, 100),  // 로깅을 위해 잘라내기
        duration: Date.now() - startTime,
        success: true,
      }));

      return {
        content: [{ type: "text" as const, text: result }],
      };
    } catch (error) {
      // 실패 기록
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

참고: stdio 전송을 사용하는 MCP 서버는 로그를 **stderr**(stdout이 아님)에 기록해야 합니다. stdout은 JSON-RPC 메시지를 위해 예약되어 있기 때문입니다. Node.js에서는 `console.error()`가 stderr에 기록됩니다.

---

## 6. 게시 및 공유

### 6.1 npm에 게시 (TypeScript)

```bash
# 게시 준비
# 1. package.json에 올바른 메타데이터 업데이트
# 2. 프로젝트 빌드
npm run build

# 3. 빌드 출력 테스트
node dist/index.js

# 4. 게시
npm publish

# 사용자가 다음과 같이 설치:
# npm install -g your-mcp-server
```

서버를 명령어로 호출할 수 있도록 `package.json`에 `bin` 필드가 있는지 확인합니다:

```json
{
  "name": "@yourorg/weather-mcp-server",
  "bin": {
    "weather-mcp-server": "./dist/index.js"
  }
}
```

### 6.2 PyPI에 게시 (Python)

```bash
# pyproject.toml 생성
# 빌드 및 게시
uv build
uv publish

# 사용자가 다음과 같이 설치:
# pip install your-mcp-server
# 또는: uvx your-mcp-server
```

### 6.3 MCP 서버 레지스트리에 등록

Anthropic은 커뮤니티 MCP 서버의 레지스트리를 유지 관리합니다. 서버를 등록하려면:

1. 서버를 위한 공개 GitHub 저장소를 생성합니다.
2. 설치 및 설정 지침이 있는 명확한 README를 포함합니다.
3. `LICENSE` 파일을 추가합니다 (MIT 또는 Apache-2.0 권장).
4. 공식 MCP 서버 목록에 제출합니다 (github.com/modelcontextprotocol/servers).

### 6.4 Docker를 통한 배포

시스템 의존성이 있는 복잡한 서버의 경우:

```dockerfile
FROM node:22-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY dist/ ./dist/
ENTRYPOINT ["node", "dist/index.js"]
```

Claude Code에서 등록:

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

## 7. 연습 문제

### 연습 문제 1: 기본 MCP 서버 (초급)

`min`과 `max` 값을 입력으로 받아 그 사이의 난수를 생성하는 `random-number`라는 단일 도구를 노출하는 MCP 서버를 만듭니다. MCP 인스펙터로 테스트합니다.

### 연습 문제 2: 파일 시스템 리소스 (중급)

프로젝트 디렉토리를 리소스로 노출하는 MCP 서버를 구축합니다. 포함 사항:
- 디렉토리의 모든 파일을 나열하는 정적 리소스.
- 파일 내용을 반환하는 동적 리소스 `file://path/{filepath}`.
- 모든 파일에서 패턴을 검색하는 도구 `search-files`.

### 연습 문제 3: REST API 래퍼 (중급)

공개 REST API(예: GitHub API)를 래핑하는 MCP 서버를 만듭니다. 포함 사항:
- 저장소 정보를 가져오는 리소스.
- 저장소 검색 및 이슈 나열 도구.
- 저장소 요약 생성을 위한 프롬프트 템플릿.
속도 제한 및 인증을 처리합니다.

### 연습 문제 4: 데이터베이스 쿼리 서버 (고급)

3.2절의 Python 데이터베이스 서버를 확장하여 다음을 지원합니다:
- 확인 프롬프트가 있는 쓰기 작업 (INSERT/UPDATE).
- 쿼리 기록 추적.
- 30초 후 자동 쿼리 타임아웃.
- 쿼리 패턴에 기반한 스키마 마이그레이션 제안.

### 연습 문제 5: 멀티 서비스 MCP 서버 (고급)

여러 소스(예: 데이터베이스 + REST API + 로컬 파일)에서 데이터를 집계하는 MCP 서버를 구축합니다. 구현 사항:
- 크로스 소스 쿼리 (예: "열린 GitHub 이슈가 있는 모든 사용자 찾기").
- 비용이 많이 드는 작업을 위한 TTL이 있는 캐싱.
- 각 데이터 소스에 대한 상태 확인 엔드포인트.

---

## 8. 참고 자료

- MCP 명세 - https://spec.modelcontextprotocol.io
- MCP TypeScript SDK - https://github.com/modelcontextprotocol/typescript-sdk
- MCP Python SDK - https://github.com/modelcontextprotocol/python-sdk
- MCP 서버 레지스트리 - https://github.com/modelcontextprotocol/servers
- MCP 인스펙터 - https://github.com/modelcontextprotocol/inspector
- MCP 서버 구축 (Anthropic 문서) - https://docs.anthropic.com/en/docs/agents-and-tools/mcp
- JSON Schema 참고 - https://json-schema.org/understanding-json-schema/

---

## 다음 레슨

[14. Claude Projects와 Artifacts](./14_Claude_Projects_and_Artifacts.md)에서는 Claude Projects에서 지식과 컨텍스트를 구성하고, 프로토타이핑을 위해 아티팩트를 활용하고, 프로젝트 구성에 Projects와 CLAUDE.md를 언제 사용할지 이해하는 방법을 탐구합니다.
