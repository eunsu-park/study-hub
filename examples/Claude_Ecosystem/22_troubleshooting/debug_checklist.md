# Claude Code Troubleshooting Checklist

## Quick Diagnostic Steps

### 1. Setup Issues

- [ ] Node.js 18+ installed: `node --version`
- [ ] Claude Code installed: `claude --version`
- [ ] API key set: `echo $ANTHROPIC_API_KEY | head -c 8`
- [ ] Network connectivity: `curl -s https://api.anthropic.com/v1/messages -w "%{http_code}" -o /dev/null`

### 2. Permission Errors

- [ ] Check current mode: review settings in `.claude/settings.json`
- [ ] Check allow rules: are the needed commands explicitly allowed?
- [ ] Check deny rules: is the command pattern accidentally blocked?
- [ ] Try running the command manually in terminal first

### 3. Context Limit Issues

- [ ] Run `/cost` to see token usage
- [ ] Run `/compact` to compress conversation history
- [ ] Consider starting a new session with `/clear`
- [ ] Reduce CLAUDE.md size if very large (>500 lines)

### 4. Hook Failures

- [ ] Test hook command manually: run the command in your terminal
- [ ] Check JSON syntax: `python -m json.tool .claude/settings.json`
- [ ] Verify file patterns in matchers match your file paths
- [ ] Check hook timeout (default: 60s)

### 5. MCP Server Issues

- [ ] Server binary exists: check the `command` path
- [ ] Run server manually: execute the command from settings
- [ ] Check environment variables (API keys, URLs)
- [ ] Check server logs for startup errors
- [ ] Verify transport type (stdio vs SSE)

### 6. API Errors

| Code | Meaning | Action |
|------|---------|--------|
| 401 | Invalid API key | Check/regenerate key |
| 403 | Forbidden | Check account permissions |
| 429 | Rate limited | Wait and retry |
| 500 | Server error | Retry after brief wait |
| 529 | Overloaded | Wait 30-60 seconds |

### 7. Performance Issues

- [ ] Large files: avoid reading files >10K lines
- [ ] Many tool calls: simplify the request
- [ ] Slow hooks: add timeout to hook config
- [ ] MCP latency: check server response time
