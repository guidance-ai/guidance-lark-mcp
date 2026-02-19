#!/usr/bin/env python3
"""MCP server for llguidance grammar validation tools."""

import asyncio
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
import mcp.types as types

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_grammar_tools.llg_tools import LLGuidanceToolContext

# Global variables for server configuration
ENABLE_GENERATION = False

# Initialize server
server = Server("llguidance-grammar-tools")
tool_context = LLGuidanceToolContext()  # Default context without LLM; main() may reinitialize

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available llguidance grammar validation tools."""
    tools = [
        types.Tool(
            name="validate_grammar",
            description="Validate llguidance grammar completeness and consistency using llguidance's built-in validator. Returns validation result with any errors or warnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "grammar": {
                        "type": "string",
                        "description": "llguidance grammar string or path to grammar file (.lark, .grammar)"
                    }
                },
                "required": ["grammar"]
            }
        ),
        types.Tool(
            name="run_batch_validation_tests",
            description="Run batch validation tests from a JSON file against an llguidance grammar. Test file should contain array of {input, should_parse, description?} objects. Returns high-level statistics (total, passed, failed, success_rate) and detailed information for failed tests.",
            inputSchema={
                "type": "object",
                "properties": {
                    "grammar": {
                        "type": "string",
                        "description": "llguidance grammar string or path to grammar file (.lark, .grammar)"
                    },
                    "test_file": {
                        "type": "string",
                        "description": "Path to JSON test file with format: [{\"input\": \"test\", \"should_parse\": true}] or {\"tests\": [...]}"
                    }
                },
                "required": ["grammar", "test_file"]
            }
        ),
        types.Tool(
            name="get_llguidance_documentation",
            description="Fetch the llguidance grammar syntax documentation.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]
    
    # Add generation tool if enabled
    if ENABLE_GENERATION:
        tools.append(
            types.Tool(
                name="generate_with_grammar",
                description="Generate text using an OpenAI model constrained by an llguidance grammar. Uses the OpenAI Responses API with a custom tool grammar format, so output is guaranteed to conform to the grammar. Use this to test how well a grammar guides actual LLM generation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "User prompt describing what text to generate"
                        },
                        "grammar": {
                            "type": "string",
                            "description": "llguidance grammar string or path to grammar file (.lark, .grammar)"
                        },
                        "model": {
                            "type": "string",
                            "description": "OpenAI model name (default: gpt-4.1)",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate (default: 300)",
                            "default": 300
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (default: 0.7)",
                            "default": 0.7
                        }
                    },
                    "required": ["prompt", "grammar"]
                }
            )
        )
    
    return tools

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """List available resources."""
    return [
        types.Resource(
            uri="llguidance://docs/syntax",
            name="llguidance Grammar Syntax Documentation",
            description="Complete llguidance grammar syntax documentation from the official GitHub repository",
            mimeType="text/markdown"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    if uri == "llguidance://docs/syntax":
        docs = tool_context.get_llguidance_documentation()
        return docs
    else:
        raise ValueError(f"Unknown resource URI: {uri}")

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool and return results."""

    try:
        if name == "validate_grammar":
            validation_result = tool_context.validate_grammar(arguments["grammar"])
            result = validation_result.model_dump()

        elif name == "run_batch_validation_tests":
            batch_result = tool_context.run_batch_validation_tests(
                arguments["grammar"],
                arguments["test_file"]
            )
            result = batch_result.model_dump(exclude={"results"})

        elif name == "get_llguidance_documentation":
            result = { "documentation": tool_context.get_llguidance_documentation() }
        
        elif name == "generate_with_grammar":
            if not ENABLE_GENERATION:
                result = {"error": "Generation not enabled. Set ENABLE_GENERATION=true environment variable."}
            else:
                generation_result = tool_context.generate_with_grammar(
                    prompt=arguments["prompt"],
                    grammar=arguments["grammar"],
                    model=arguments.get("model"),
                    max_tokens=arguments.get("max_tokens", 300),
                    temperature=arguments.get("temperature", 0.7)
                )
                result = generation_result.model_dump()

        else:
            result = {"error": f"Unknown tool: {name}"}

    except FileNotFoundError as e:
        result = {"error": f"File not found: {str(e)}"}
    except ValueError as e:
        result = {"error": f"Validation error: {str(e)}"}
    except Exception as e:
        result = {"error": f"Tool execution failed: {str(e)}", "type": type(e).__name__}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

async def async_main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="llguidance-grammar-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

def main():
    """Synchronous entry point to start the MCP server."""
    global ENABLE_GENERATION, tool_context
    
    ENABLE_GENERATION = os.environ.get("ENABLE_GENERATION", "").lower() in ("1", "true", "yes")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    
    # Initialize tool context
    tool_context = LLGuidanceToolContext(enable_generation=ENABLE_GENERATION, model=model)
    
    # Log to stderr
    if ENABLE_GENERATION:
        print(f"Starting MCP LLGuidance Grammar Tools Server with generation enabled...", file=sys.stderr, flush=True)
        print(f"Model: {model}", file=sys.stderr, flush=True)
    else:
        print("Starting MCP LLGuidance Grammar Tools Server (generation disabled)...", file=sys.stderr, flush=True)
    
    asyncio.run(async_main())

if __name__ == "__main__":
    main()