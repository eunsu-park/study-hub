"""
Simple MCP Server Example (Python)

A Model Context Protocol server that provides weather data tools.
Demonstrates: Resources, Tools, and Prompts in MCP.

Requirements:
    pip install mcp

Usage:
    # Add to Claude Code settings:
    # claude mcp add weather-server python simple_mcp_server.py
"""

from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
from mcp.server.stdio import stdio_server

import json

# Create the MCP server
server = Server("weather-server")

# --- Simulated Data ---
WEATHER_DATA = {
    "seoul": {"temp": 5, "condition": "Cloudy", "humidity": 65},
    "tokyo": {"temp": 12, "condition": "Sunny", "humidity": 45},
    "new_york": {"temp": -2, "condition": "Snow", "humidity": 80},
    "london": {"temp": 8, "condition": "Rainy", "humidity": 90},
    "san_francisco": {"temp": 15, "condition": "Foggy", "humidity": 70},
}


# --- Resources ---
# Resources expose data that the model can read

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available weather data resources."""
    return [
        Resource(
            uri=f"weather://{city}",
            name=f"Weather in {city.replace('_', ' ').title()}",
            description=f"Current weather data for {city.replace('_', ' ').title()}",
            mimeType="application/json",
        )
        for city in WEATHER_DATA
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read weather data for a specific city."""
    # Parse city from URI: weather://seoul -> seoul
    city = uri.replace("weather://", "")
    if city not in WEATHER_DATA:
        raise ValueError(f"Unknown city: {city}")
    return json.dumps(WEATHER_DATA[city], indent=2)


# --- Tools ---
# Tools let the model take actions

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'seoul', 'tokyo', 'new_york')",
                    }
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="compare_weather",
            description="Compare weather between two cities",
            inputSchema={
                "type": "object",
                "properties": {
                    "city1": {"type": "string", "description": "First city"},
                    "city2": {"type": "string", "description": "Second city"},
                },
                "required": ["city1", "city2"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_weather":
        city = arguments["city"].lower().replace(" ", "_")
        if city not in WEATHER_DATA:
            return [TextContent(
                type="text",
                text=f"No weather data available for '{city}'. "
                     f"Available cities: {', '.join(WEATHER_DATA.keys())}",
            )]
        data = WEATHER_DATA[city]
        return [TextContent(
            type="text",
            text=(
                f"Weather in {city.replace('_', ' ').title()}:\n"
                f"  Temperature: {data['temp']}°C\n"
                f"  Condition: {data['condition']}\n"
                f"  Humidity: {data['humidity']}%"
            ),
        )]

    elif name == "compare_weather":
        city1 = arguments["city1"].lower().replace(" ", "_")
        city2 = arguments["city2"].lower().replace(" ", "_")

        if city1 not in WEATHER_DATA or city2 not in WEATHER_DATA:
            return [TextContent(type="text", text="One or both cities not found.")]

        d1, d2 = WEATHER_DATA[city1], WEATHER_DATA[city2]
        diff = d1["temp"] - d2["temp"]
        warmer = city1 if diff > 0 else city2

        return [TextContent(
            type="text",
            text=(
                f"Weather Comparison:\n"
                f"  {city1.replace('_', ' ').title()}: {d1['temp']}°C, {d1['condition']}\n"
                f"  {city2.replace('_', ' ').title()}: {d2['temp']}°C, {d2['condition']}\n"
                f"  Difference: {abs(diff)}°C "
                f"({warmer.replace('_', ' ').title()} is warmer)"
            ),
        )]

    raise ValueError(f"Unknown tool: {name}")


# --- Main ---
async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
