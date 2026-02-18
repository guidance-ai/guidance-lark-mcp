"""Tests for MCP server functionality."""

import pytest
import json
from mcp_grammar_tools.server import server, tool_context, call_tool, list_tools, list_resources, read_resource


@pytest.mark.asyncio
async def test_list_tools():
    """Test that server lists the correct tools."""
    tools = await list_tools()

    # Should have exactly 3 tools (without LLM)
    assert len(tools) == 3

    # Get tool names
    tool_names = [tool.name for tool in tools]
    assert "validate_grammar" in tool_names
    assert "run_batch_validation_tests" in tool_names
    assert "get_llguidance_documentation" in tool_names

    # Check that each tool has required fields
    for tool in tools:
        assert tool.name
        assert tool.description
        assert tool.inputSchema
        assert "properties" in tool.inputSchema


@pytest.mark.asyncio
async def test_validate_grammar_tool_valid():
    """Test validate_grammar tool with valid grammar."""
    grammar = """
    start: "hello"
    """

    result = await call_tool("validate_grammar", {"grammar": grammar})

    # Should return TextContent
    assert len(result) == 1
    assert result[0].type == "text"

    # Parse JSON response
    data = json.loads(result[0].text)
    assert data["is_valid"] is True
    assert data["errors_and_warnings"] == []


@pytest.mark.asyncio
async def test_validate_grammar_tool_invalid():
    """Test validate_grammar tool with invalid grammar."""
    invalid_grammar = """
    start: undefined_rule
    """

    result = await call_tool("validate_grammar", {"grammar": invalid_grammar})

    # Should return TextContent with error
    assert len(result) == 1
    data = json.loads(result[0].text)

    assert data["is_valid"] is False
    assert len(data["errors_and_warnings"]) > 0
    assert "undefined_rule" in data["errors_and_warnings"][0]


@pytest.mark.asyncio
async def test_validate_grammar_from_file(tmp_path):
    """Test validate_grammar tool loading grammar from file."""
    # Create temporary grammar file
    grammar_file = tmp_path / "test.grammar"
    grammar_file.write_text("""
start: "hello" "world"
""")

    result = await call_tool("validate_grammar", {"grammar": str(grammar_file)})

    data = json.loads(result[0].text)
    assert data["is_valid"] is True


@pytest.mark.asyncio
async def test_batch_validation_tool(tmp_path):
    """Test run_batch_validation_tests tool."""
    # Create test file
    test_file = tmp_path / "tests.json"
    test_file.write_text(json.dumps({
        "tests": [
            {"input": "hello", "should_parse": True, "description": "Valid"},
            {"input": "goodbye", "should_parse": False, "description": "Invalid"}
        ]
    }))

    grammar = """
start: "hello"
"""

    result = await call_tool("run_batch_validation_tests", {
        "grammar": grammar,
        "test_file": str(test_file)
    })

    # Parse result
    data = json.loads(result[0].text)

    assert data["total"] == 2
    assert data["passed"] == 2
    assert data["failed"] == 0
    assert data["success_rate"] == 1.0

    # Check that no failures are reported (failed_tests should be empty)
    assert data["failed_tests"] == []


@pytest.mark.asyncio
async def test_batch_validation_with_failures(tmp_path):
    """Test batch validation with some failing tests."""
    test_file = tmp_path / "tests.json"
    test_file.write_text(json.dumps([
        {"input": "hello", "should_parse": True},
        {"input": "world", "should_parse": True}  # This should fail
    ]))

    grammar = """
start: "hello"
"""

    result = await call_tool("run_batch_validation_tests", {
        "grammar": grammar,
        "test_file": str(test_file)
    })

    data = json.loads(result[0].text)

    assert data["total"] == 2
    assert data["passed"] == 1
    assert data["failed"] == 1

    # Check that failed_tests is populated
    assert len(data["failed_tests"]) == 1
    assert data["failed_tests"][0]["test_id"] == 1


@pytest.mark.asyncio
async def test_batch_validation_invalid_grammar(tmp_path):
    """Test that batch validation fails with invalid grammar."""
    test_file = tmp_path / "tests.json"
    test_file.write_text(json.dumps([
        {"input": "test", "should_parse": True}
    ]))

    invalid_grammar = """
    start: undefined_rule
    """

    result = await call_tool("run_batch_validation_tests", {
        "grammar": invalid_grammar,
        "test_file": str(test_file)
    })

    data = json.loads(result[0].text)

    # Should have error
    assert "error" in data
    assert "Invalid grammar" in data["error"] or "Validation error" in data["error"]


@pytest.mark.asyncio
async def test_file_not_found_error(tmp_path):
    """Test error handling for missing test file."""
    result = await call_tool("run_batch_validation_tests", {
        "grammar": 'start: "test"',
        "test_file": str(tmp_path / "nonexistent.json")
    })

    data = json.loads(result[0].text)
    assert "error" in data
    # Could be either file not found or JSON parsing error depending on order of operations
    assert "File not found" in data["error"] or "error" in data["error"].lower()


@pytest.mark.asyncio
async def test_unknown_tool():
    """Test error handling for unknown tool."""
    result = await call_tool("unknown_tool", {})

    data = json.loads(result[0].text)
    assert "error" in data
    assert "Unknown tool" in data["error"]


@pytest.mark.asyncio
async def test_validate_grammar_tool_schema():
    """Test that validate_grammar tool has correct schema."""
    tools = await list_tools()
    validate_tool = next(t for t in tools if t.name == "validate_grammar")

    schema = validate_tool.inputSchema
    assert "grammar" in schema["properties"]
    assert "grammar" in schema["required"]
    assert schema["properties"]["grammar"]["type"] == "string"


@pytest.mark.asyncio
async def test_batch_validation_tool_schema():
    """Test that run_batch_validation_tests tool has correct schema."""
    tools = await list_tools()
    batch_tool = next(t for t in tools if t.name == "run_batch_validation_tests")

    schema = batch_tool.inputSchema
    assert "grammar" in schema["properties"]
    assert "test_file" in schema["properties"]
    assert "grammar" in schema["required"]
    assert "test_file" in schema["required"]


@pytest.mark.asyncio
async def test_server_instance():
    """Test that server instance is properly initialized."""
    assert server is not None
    assert server.name == "llguidance-grammar-tools"


@pytest.mark.asyncio
async def test_tool_context_instance():
    """Test that tool_context is properly initialized."""
    assert tool_context is not None
    from mcp_grammar_tools.llg_tools import LLGuidanceToolContext
    assert isinstance(tool_context, LLGuidanceToolContext)


@pytest.mark.asyncio
async def test_list_resources():
    """Test that server lists available resources."""
    resources = await list_resources()

    # Should have exactly 1 resource
    assert len(resources) == 1

    # Check the documentation resource
    doc_resource = resources[0]
    assert str(doc_resource.uri) == "llguidance://docs/syntax"
    assert "llguidance" in doc_resource.name.lower()
    assert "documentation" in doc_resource.name.lower()
    assert doc_resource.mimeType == "text/markdown"


@pytest.mark.asyncio
async def test_read_documentation_resource():
    """Test reading the llguidance documentation resource."""
    docs = await read_resource("llguidance://docs/syntax")

    # Should return markdown documentation
    assert isinstance(docs, str)
    assert len(docs) > 100  # Should be substantial documentation
    assert "recursive" in docs.lower() or "grammar" in docs.lower()


@pytest.mark.asyncio
async def test_read_unknown_resource():
    """Test that reading an unknown resource raises an error."""
    with pytest.raises(ValueError, match="Unknown resource URI"):
        await read_resource("unknown://resource")
