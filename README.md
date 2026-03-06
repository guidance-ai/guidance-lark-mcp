# MCP Grammar Tools

<!-- mcp-name: io.github.guidance-ai/guidance-lark-mcp -->

MCP server for validating and testing [llguidance](https://github.com/guidance-ai/llguidance) grammars (Lark format). Provides grammar validation, batch test execution, and syntax documentation — ideal for iteratively building grammars with AI coding assistants.

## Installation

### With uvx (recommended)
```bash
uvx guidance-lark-mcp
```

### With pip
```bash
pip install guidance-lark-mcp
```

### From source
```bash
cd mcp-grammar-tools
pip install -e .
```

## MCP Client Configuration

### GitHub Copilot CLI

You can add the server using the interactive `/mcp add` command or by editing the config file directly. See the [Copilot CLI MCP documentation](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers) for full details.

**Option 1: Interactive setup**

In the Copilot CLI, run `/mcp add`, select **Local/STDIO**, and enter `uvx guidance-lark-mcp` as the command.

**Option 2: Edit config file**

Add the following to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "grammar-tools": {
      "type": "local",
      "command": "uvx",
      "args": ["guidance-lark-mcp"],
      "tools": ["*"]
    }
  }
}
```

This gives you grammar validation and batch testing out of the box. To also enable LLM-powered generation (`generate_with_grammar`), add `ENABLE_GENERATION` and your credentials to `env`:

```json
"env": {
  "ENABLE_GENERATION": "true",
  "OPENAI_API_KEY": "your-key-here"
}
```

For Azure OpenAI (with Entra ID via `az login`), use `guidance-lark-mcp[azure]` and set the endpoint instead:

```json
"args": ["guidance-lark-mcp[azure]"],
"env": {
  "ENABLE_GENERATION": "true",
  "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/",
  "OPENAI_MODEL": "your-deployment-name"
}
```

See [Backend Configuration](#backend-configuration) for all supported backends.

After saving, use `/mcp show` to verify the server is connected.

### VS Code

```json
{
  "mcpServers": {
    "grammar-tools": {
      "type": "local",
      "command": "uvx",
      "args": ["guidance-lark-mcp"],
      "env": {
        "ENABLE_GENERATION": "true",
        "OPENAI_API_KEY": "your-key-here"
      },
      "tools": ["*"]
    }
  }
}
```

### Claude Desktop
```json
{
  "mcpServers": {
    "grammar-tools": {
      "command": "uvx",
      "args": ["guidance-lark-mcp"],
      "env": {
        "ENABLE_GENERATION": "true",
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Usage

### Available Tools

1. **`validate_grammar`** — Validate grammar completeness and consistency using llguidance's built-in validator.
   ```json
   {"grammar": "start: \"hello\" \"world\""}
   ```

2. **`run_batch_validation_tests`** — Run batch validation tests from a JSON file against a grammar. Returns pass/fail statistics and detailed failure info.
   ```json
   {
     "grammar": "start: /[0-9]+/",
     "test_file": "tests.json"
   }
   ```

   Test file format:
   ```json
   [
     {"input": "123", "should_parse": true, "description": "Valid number"},
     {"input": "abc", "should_parse": false, "description": "Not a number"}
   ]
   ```

3. **`get_llguidance_documentation`** — Fetch the llguidance grammar syntax documentation from the official repo.

4. **`generate_with_grammar`** *(optional, requires `ENABLE_GENERATION=true`)* — Generate text using an OpenAI model constrained by a grammar. Uses the [Responses API with custom tool grammar format](https://developers.openai.com/api/docs/guides/function-calling/#context-free-grammars), so output is guaranteed to conform to the grammar. Requires `OPENAI_API_KEY` environment variable. See [Backend Configuration](#backend-configuration) for Azure and other endpoints.

## Backend Configuration

The `generate_with_grammar` tool uses the OpenAI Python SDK, which natively supports multiple backends via environment variables:

| Backend | Required env vars | Optional env vars |
|---------|-------------------|-------------------|
| **OpenAI** (default) | `OPENAI_API_KEY` | `OPENAI_MODEL` |
| **Azure OpenAI (API key)** | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_API_VERSION`, `OPENAI_MODEL` |
| **Azure OpenAI (Entra ID)** | `AZURE_OPENAI_ENDPOINT` + `az login` | `AZURE_OPENAI_API_VERSION`, `OPENAI_MODEL` |
| **Custom endpoint** | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | `OPENAI_MODEL` |

The server auto-detects which backend to use:
- If `AZURE_OPENAI_ENDPOINT` is set → uses `AzureOpenAI` client (with Entra ID or API key)
- Otherwise → uses `OpenAI` client (reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` automatically)

The server logs which backend it detects on startup.

### Example: Azure OpenAI (API key)
```json
{
  "mcpServers": {
    "grammar-tools": {
      "type": "local",
      "command": "uvx",
      "args": ["guidance-lark-mcp"],
      "env": {
        "ENABLE_GENERATION": "true",
        "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "your-azure-key",
        "OPENAI_MODEL": "gpt-4.1"
      },
      "tools": ["*"]
    }
  }
}
```

### Example: Azure OpenAI (Entra ID / keyless)

Requires `az login` and the `azure` extra: `pip install guidance-lark-mcp[azure]`

```json
{
  "mcpServers": {
    "grammar-tools": {
      "type": "local",
      "command": "uvx",
      "args": ["guidance-lark-mcp[azure]"],
      "env": {
        "ENABLE_GENERATION": "true",
        "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com",
        "OPENAI_MODEL": "gpt-4.1"
      },
      "tools": ["*"]
    }
  }
}
```

## Example Workflow

Build a grammar iteratively with an AI assistant:

1. **Start with the spec** — paste EBNF rules from a language specification
2. **Write a basic grammar** — translate a few rules to Lark format
3. **Validate** — use `validate_grammar` to check for missing rules
4. **Write tests** — create a JSON test file with sample inputs
5. **Batch test** — use `run_batch_validation_tests` to find failures
6. **Fix & repeat** — refine the grammar until all tests pass

## Example Grammars

The `examples/` directory includes sample grammars built using these tools, with Lark grammar files, test suites, and documentation:

- **[GraphQL](examples/graphql/)** — executable subset of the GraphQL spec (queries, mutations, fragments, variables)

## Troubleshooting

**Server fails to connect in Copilot CLI / VS Code?**

MCP clients like Copilot CLI only show "Connection closed" when a server crashes on startup. To see the actual error, run the server directly in your terminal:

```bash
uvx guidance-lark-mcp
```

Or with generation enabled:

```bash
ENABLE_GENERATION=true OPENAI_API_KEY=your-key uvx guidance-lark-mcp
```

Common issues:
- **Missing credentials** — `ENABLE_GENERATION=true` without a valid `OPENAI_API_KEY` or `AZURE_OPENAI_ENDPOINT`. The server will still start and serve validation tools; `generate_with_grammar` will return a descriptive error.
- **Azure Entra ID** — make sure you've run `az login` and are using `guidance-lark-mcp[azure]` (not the base package).
- **Slow first start** — `uvx` needs to resolve and install dependencies on first run, which may exceed the MCP client's connection timeout. Run `uvx guidance-lark-mcp` once manually to warm the cache.
- **Updating to a new version** — `uvx` caches packages, so after a new release you may need to clear the cache and restart your MCP client:
  ```bash
  uv cache clean guidance-lark-mcp
  ```

## Development

```bash
git clone https://github.com/guidance-ai/guidance-lark-mcp
cd guidance-lark-mcp
uv sync
uv run pytest tests/ -q
```

