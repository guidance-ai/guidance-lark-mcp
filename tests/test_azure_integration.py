"""Integration tests that call Azure OpenAI with grammar-constrained generation."""

import os

import pytest

from mcp_grammar_tools.llg_tools import LLGuidanceToolContext


requires_azure = pytest.mark.skipif(
    not os.environ.get("AZURE_OPENAI_ENDPOINT"),
    reason="AZURE_OPENAI_ENDPOINT not set",
)


@pytest.fixture
def azure_context() -> LLGuidanceToolContext:
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    ctx = LLGuidanceToolContext(enable_generation=True, model=model)
    return ctx


def assert_parses(ctx: LLGuidanceToolContext, grammar: str, text: str):
    """Assert that text parses against the grammar."""
    error = ctx._check_parse_error(grammar, text)
    assert error is None, f"Text '{text}' did not parse: {error}"


@requires_azure
def test_generate_simple_grammar(azure_context: LLGuidanceToolContext):
    """Generate text constrained to a simple grammar and verify it parses."""
    grammar = r"""
start: greeting " " name
greeting: "hello" | "hi" | "hey"
name: /[A-Z][a-z]+/
"""
    result = azure_context.generate_with_grammar(
        prompt="Greet someone by name.",
        grammar=grammar,
        max_tokens=500,
        reasoning_effort="low",
    )
    assert result.is_valid, f"Generation failed: {result.error}"
    assert result.generated_text
    assert_parses(azure_context, grammar, result.generated_text)


@requires_azure
def test_generate_json_grammar(azure_context: LLGuidanceToolContext):
    """Generate JSON conforming to a grammar."""
    grammar = r"""
start: "{" ws "\"name\":" ws STRING "," ws "\"age\":" ws NUMBER ws "}"
STRING: "\"" /[a-zA-Z ]+/ "\""
NUMBER: /[1-9][0-9]*/
ws: /\s*/
"""
    result = azure_context.generate_with_grammar(
        prompt="Generate a JSON object with a person's name and age.",
        grammar=grammar,
        max_tokens=500,
        reasoning_effort="low",
    )
    assert result.is_valid, f"Generation failed: {result.error}"
    assert result.generated_text
    assert_parses(azure_context, grammar, result.generated_text)


@requires_azure
def test_generate_arithmetic_grammar(azure_context: LLGuidanceToolContext):
    """Generate a simple arithmetic expression."""
    grammar = r"""
start: expr
expr: NUMBER OP NUMBER
NUMBER: /[1-9][0-9]*/
OP: " + " | " - " | " * "
"""
    result = azure_context.generate_with_grammar(
        prompt="Write a simple arithmetic expression.",
        grammar=grammar,
        max_tokens=500,
        reasoning_effort="low",
    )
    assert result.is_valid, f"Generation failed: {result.error}"
    assert result.generated_text
    assert_parses(azure_context, grammar, result.generated_text)


@requires_azure
def test_generate_with_prefix_constraint(azure_context: LLGuidanceToolContext):
    """Generate text that must start with a specific prefix, like the beam search pattern."""
    grammar = r"""
start: PREFIX CONTINUATION
PREFIX: "Once upon a time"
CONTINUATION: /[^"]*/
"""
    result = azure_context.generate_with_grammar(
        prompt="Tell me a story.",
        grammar=grammar,
        max_tokens=500,
        reasoning_effort="low",
    )
    assert result.is_valid, f"Generation failed: {result.error}"
    assert result.generated_text.startswith("Once upon a time"), (
        f"Expected prefix 'Once upon a time', got: '{result.generated_text[:30]}...'"
    )


@requires_azure
def test_generate_enum_values(azure_context: LLGuidanceToolContext):
    """Generate text constrained to a set of enum values."""
    grammar = r"""
start: color
color: "red" | "green" | "blue" | "yellow" | "purple"
"""
    result = azure_context.generate_with_grammar(
        prompt="Pick a color.",
        grammar=grammar,
        max_tokens=500,
        reasoning_effort="low",
    )
    assert result.is_valid, f"Generation failed: {result.error}"
    assert result.generated_text in {"red", "green", "blue", "yellow", "purple"}, (
        f"Unexpected color: '{result.generated_text}'"
    )
