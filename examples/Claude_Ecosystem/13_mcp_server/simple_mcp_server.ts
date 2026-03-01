/**
 * Simple MCP Server Example (TypeScript)
 *
 * A Model Context Protocol server providing a note-taking system.
 * Demonstrates: Resources, Tools, and Prompts in MCP.
 *
 * Requirements:
 *   npm install @modelcontextprotocol/sdk
 *
 * Usage:
 *   claude mcp add notes-server npx ts-node simple_mcp_server.ts
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// --- In-Memory Store ---
interface Note {
  title: string;
  content: string;
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

const notes: Map<string, Note> = new Map();

// Seed with sample data
notes.set("welcome", {
  title: "Welcome",
  content: "Welcome to the MCP Notes Server! Create, read, and manage notes.",
  tags: ["intro", "help"],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
});

// --- Create Server ---
const server = new Server(
  { name: "notes-server", version: "1.0.0" },
  { capabilities: { resources: {}, tools: {}, prompts: {} } }
);

// --- Resources ---

server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: Array.from(notes.entries()).map(([id, note]) => ({
    uri: `note:///${id}`,
    name: note.title,
    description: `Note: ${note.title} (${note.tags.join(", ")})`,
    mimeType: "text/plain",
  })),
}));

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const id = request.params.uri.replace("note:///", "");
  const note = notes.get(id);

  if (!note) {
    throw new Error(`Note not found: ${id}`);
  }

  return {
    contents: [
      {
        uri: request.params.uri,
        mimeType: "text/plain",
        text: `# ${note.title}\n\nTags: ${note.tags.join(", ")}\nCreated: ${note.createdAt}\nUpdated: ${note.updatedAt}\n\n${note.content}`,
      },
    ],
  };
});

// --- Tools ---

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "create_note",
      description: "Create a new note",
      inputSchema: {
        type: "object" as const,
        properties: {
          id: { type: "string", description: "Unique note identifier" },
          title: { type: "string", description: "Note title" },
          content: { type: "string", description: "Note content (markdown)" },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Tags for categorization",
          },
        },
        required: ["id", "title", "content"],
      },
    },
    {
      name: "search_notes",
      description: "Search notes by keyword or tag",
      inputSchema: {
        type: "object" as const,
        properties: {
          query: { type: "string", description: "Search keyword" },
          tag: { type: "string", description: "Filter by tag" },
        },
      },
    },
    {
      name: "delete_note",
      description: "Delete a note by ID",
      inputSchema: {
        type: "object" as const,
        properties: {
          id: { type: "string", description: "Note ID to delete" },
        },
        required: ["id"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "create_note": {
      const { id, title, content, tags = [] } = args as {
        id: string;
        title: string;
        content: string;
        tags?: string[];
      };

      const now = new Date().toISOString();
      notes.set(id, { title, content, tags, createdAt: now, updatedAt: now });

      return {
        content: [
          { type: "text" as const, text: `Note '${title}' created with ID '${id}'.` },
        ],
      };
    }

    case "search_notes": {
      const { query, tag } = args as { query?: string; tag?: string };
      const results: string[] = [];

      for (const [id, note] of notes) {
        const matchesQuery =
          !query ||
          note.title.toLowerCase().includes(query.toLowerCase()) ||
          note.content.toLowerCase().includes(query.toLowerCase());
        const matchesTag = !tag || note.tags.includes(tag);

        if (matchesQuery && matchesTag) {
          results.push(`- [${id}] ${note.title} (tags: ${note.tags.join(", ")})`);
        }
      }

      return {
        content: [
          {
            type: "text" as const,
            text:
              results.length > 0
                ? `Found ${results.length} note(s):\n${results.join("\n")}`
                : "No notes found matching your search.",
          },
        ],
      };
    }

    case "delete_note": {
      const { id } = args as { id: string };
      if (!notes.has(id)) {
        return {
          content: [{ type: "text" as const, text: `Note '${id}' not found.` }],
        };
      }
      notes.delete(id);
      return {
        content: [{ type: "text" as const, text: `Note '${id}' deleted.` }],
      };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// --- Prompts ---

server.setRequestHandler(ListPromptsRequestSchema, async () => ({
  prompts: [
    {
      name: "summarize_notes",
      description: "Generate a summary of all notes",
    },
    {
      name: "daily_review",
      description: "Create a daily review from recent notes",
      arguments: [
        {
          name: "focus_tag",
          description: "Optional tag to focus the review on",
          required: false,
        },
      ],
    },
  ],
}));

server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "summarize_notes") {
    const allNotes = Array.from(notes.values())
      .map((n) => `## ${n.title}\n${n.content}`)
      .join("\n\n");

    return {
      messages: [
        {
          role: "user" as const,
          content: {
            type: "text" as const,
            text: `Please summarize the following notes concisely:\n\n${allNotes}`,
          },
        },
      ],
    };
  }

  if (name === "daily_review") {
    const focusTag = args?.focus_tag as string | undefined;
    const filtered = Array.from(notes.values())
      .filter((n) => !focusTag || n.tags.includes(focusTag))
      .map((n) => `- ${n.title}: ${n.content.slice(0, 100)}...`)
      .join("\n");

    return {
      messages: [
        {
          role: "user" as const,
          content: {
            type: "text" as const,
            text: `Create a daily review based on these notes${focusTag ? ` (focus: ${focusTag})` : ""}:\n\n${filtered}`,
          },
        },
      ],
    };
  }

  throw new Error(`Unknown prompt: ${name}`);
});

// --- Start Server ---
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Notes MCP Server running on stdio");
}

main().catch(console.error);
