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

### VS Code / Copilot CLI (`~/.copilot/mcp-config.json`)
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

4. **`generate_with_grammar`** *(optional, requires `ENABLE_GENERATION=true`)* — Generate text using an OpenAI model constrained by a grammar. Uses the [Responses API with custom tool grammar format](https://developers.openai.com/api/docs/guides/function-calling/#context-free-grammars), so output is guaranteed to conform to the grammar. Requires `OPENAI_API_KEY` environment variable.

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

## Development

```bash
git clone https://github.com/guidance-ai/guidance-lark-mcp
cd guidance-lark-mcp
uv venv && uv pip install -e . && uv pip install pytest pytest-asyncio
pytest tests/test_llg_tools.py -q
```

## Publishing

Releases are automated via GitHub Actions. To publish a new version:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This will build, publish to PyPI, publish to the MCP Registry, and create a GitHub Release.

Requires `PYPI_API_TOKEN` secret in the GitHub repository settings.