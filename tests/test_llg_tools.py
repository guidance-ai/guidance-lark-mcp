import pytest
from unittest.mock import patch, MagicMock
from mcp_grammar_tools.llg_tools import (
    LLGuidanceToolContext,
    UnexpectedToken,
    UnexpectedEOF,
    ParseResult,
    GrammarValidationResult,
    GenerationResult,
)


@pytest.fixture
def tool_context() -> LLGuidanceToolContext:
    return LLGuidanceToolContext()


def test_valid_grammar(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "hello"
    """
    result = tool_context.validate_grammar(grammar)
    assert result.is_valid
    assert result.errors_and_warnings == []


def test_no_start(tool_context: LLGuidanceToolContext):
    grammar = """
    blah: "hello"
    """
    result = tool_context.validate_grammar(grammar)
    assert not result.is_valid
    assert "no start rule found" in result.errors_and_warnings


def test_good_string_match(tool_context: LLGuidanceToolContext):
    grammar = """
start: "hello, world!"
"""
    assert tool_context._check_parse_error(grammar, "hello, world!") is None


def test_unexpected_token_error(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "hello"
    """
    error = tool_context._check_parse_error(grammar, "goodbye")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.error_type == "unexpected_token"
    assert error.position == 0


def test_unexpected_eof_error(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "hello" "world"
    """
    error = tool_context._check_parse_error(grammar, "hello")
    assert error is not None
    assert isinstance(error, UnexpectedEOF)
    assert error.error_type == "unexpected_eof"


def test_complex_grammar_with_rules(tool_context: LLGuidanceToolContext):
    # Using %ignore to handle whitespace automatically
    grammar = """
start: greeting name
greeting: "Hello" | "Hi"
name: /[A-Z][a-z]+/
%ignore /[ \t\n\r]+/
"""
    error = tool_context._check_parse_error(grammar, "Hello Alice")
    assert error is None


def test_complex_grammar_failure(tool_context: LLGuidanceToolContext):
    grammar = """
    start: greeting name
    greeting: "Hello" | "Hi"
    name: /[A-Z][a-z]+/
    """
    error = tool_context._check_parse_error(grammar, "Hello alice")
    assert error is not None
    assert isinstance(error, UnexpectedToken)


def test_grammar_with_optional(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "hello" name?
    name: /[A-Z][a-z]+/
    """
    error = tool_context._check_parse_error(grammar, "hello")
    assert error is None


def test_grammar_with_repetition(tool_context: LLGuidanceToolContext):
    grammar = """
start: word+
word: /[a-z]+/
%ignore /[ \t]+/
"""
    error = tool_context._check_parse_error(grammar, "hello world foo bar")
    assert error is None


def test_grammar_with_alternatives(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "yes" | "no" | "maybe"
    """
    error = tool_context._check_parse_error(grammar, "maybe")
    assert error is None


def test_json_like_grammar(tool_context: LLGuidanceToolContext):
    grammar = r"""
start: "{" string ":" number "}"
string: "\"" /[a-z]+/ "\""
number: /[0-9]+/
"""
    error = tool_context._check_parse_error(grammar, '{"age":25}')
    assert error is None


def test_nested_structures(tool_context: LLGuidanceToolContext):
    grammar = """
    start: list
    list: "[" (item ("," item)*)? "]"
    item: /[0-9]+/
    """
    error = tool_context._check_parse_error(grammar, "[1,2,3,4,5]")
    assert error is None


def test_whitespace_handling(tool_context: LLGuidanceToolContext):
    grammar = """
start: word word
word: /[a-z]+/
%ignore /[ \t]+/
"""
    error = tool_context._check_parse_error(grammar, "hello world")
    assert error is None


def test_empty_input_expected(tool_context: LLGuidanceToolContext):
    grammar = """
    start: ""
    """
    error = tool_context._check_parse_error(grammar, "")
    assert error is None


def test_empty_input_unexpected(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "hello"
    """
    error = tool_context._check_parse_error(grammar, "")
    assert error is not None
    assert isinstance(error, UnexpectedEOF)


def test_partial_match_fails(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "hello world"
    """
    error = tool_context._check_parse_error(grammar, "hello world!")
    assert error is not None


def test_case_sensitive_matching(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "Hello"
    """
    error = tool_context._check_parse_error(grammar, "hello")
    assert error is not None


def test_unicode_support(tool_context: LLGuidanceToolContext):
    grammar = r"""
    start: "Hello" emoji
    emoji: /[\u{1F600}-\u{1F64F}]/
    """
    error = tool_context._check_parse_error(grammar, "Hello😀")
    assert error is None


def test_validate_grammar_missing_rule(tool_context: LLGuidanceToolContext):
    grammar = """
    start: undefined_rule
    """
    result = tool_context.validate_grammar(grammar)
    assert not result.is_valid
    assert len(result.errors_and_warnings) > 0


def test_error_context_reporting(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "The quick brown fox jumps"
    """
    error = tool_context._check_parse_error(grammar, "The quick brown cat jumps")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.position > 0
    assert len(error.context) > 0
    assert error.token != ""


def test_multiline_grammar(tool_context: LLGuidanceToolContext):
    grammar = r"""
start: line+
line: /[^\n]+/ "\n"
"""
    error = tool_context._check_parse_error(grammar, "line1\nline2\nline3\n")
    assert error is None


def test_get_documentation_url(tool_context: LLGuidanceToolContext):
    docs = tool_context.get_llguidance_documentation()
    assert "recursive" in docs.lower()

def test_grammar_with_quantifiers(tool_context: LLGuidanceToolContext):
    grammar = """
    start: /a{3}/ /b{2,4}/ /c{1,}/
    """
    error = tool_context._check_parse_error(grammar, "aaabbbcccc")
    assert error is None


def test_invalid_regex_in_grammar(tool_context: LLGuidanceToolContext):
    grammar = """
    start: /[unclosed/
    """
    result = tool_context.validate_grammar(grammar)
    assert not result.is_valid


def test_ignore_directive_for_whitespace(tool_context: LLGuidanceToolContext):
    """Test that %ignore directive properly handles whitespace between tokens."""
    grammar = """
start: "hello" "world"
%ignore /[ \t\n\r]+/
"""
    error = tool_context._check_parse_error(grammar, "hello world")
    assert error is None

    # Also test with multiple spaces and newlines
    error2 = tool_context._check_parse_error(grammar, "hello   \n  world")
    assert error2 is None


def test_without_ignore_directive_fails(tool_context: LLGuidanceToolContext):
    """Test that without %ignore, whitespace causes failures."""
    grammar = """
start: "hello" "world"
"""
    error = tool_context._check_parse_error(grammar, "hello world")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.token == " "


def test_line_column_reporting_single_line(tool_context: LLGuidanceToolContext):
    """Test line and column numbers are correctly reported for single-line input."""
    grammar = """
start: "The quick brown fox jumps"
"""
    error = tool_context._check_parse_error(grammar, "The quick brown cat jumps")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.line == 1
    assert error.column == 17  # Position of 'c' in 'cat'
    assert error.position == 16


def test_line_column_reporting_multiline(tool_context: LLGuidanceToolContext):
    """Test line and column numbers for multiline input."""
    grammar = r"""
start: line line line
line: /[^\n]+/ "\n"
"""
    # This input actually matches the grammar, so let's create a real error
    test_input = "hello\nworld\nthis is line3\nextra line"
    error = tool_context._check_parse_error(grammar, test_input)
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    # Error should be at line 4 (extra line that shouldn't be there)
    assert error.line == 4
    assert error.column == 1


def test_line_column_reporting_first_char(tool_context: LLGuidanceToolContext):
    """Test line and column numbers when error is at first character."""
    grammar = """
start: "hello"
"""
    error = tool_context._check_parse_error(grammar, "goodbye")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.line == 1
    assert error.column == 1
    assert error.position == 0


def test_line_column_reporting_newline_positions(tool_context: LLGuidanceToolContext):
    """Test line and column calculation across different newline positions."""
    grammar = r"""
start: "line1" "\n" "line2" "\n" "line3"
"""
    # Error at start of line 2
    error = tool_context._check_parse_error(grammar, "line1\nwrong\nline3")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.line == 2
    assert error.column == 1


def test_line_column_with_tabs(tool_context: LLGuidanceToolContext):
    """Test that tabs are counted correctly in column calculation."""
    # Tab needs to be escaped with raw string
    grammar = r"""
start: "hello" "\t" "world"
"""
    error = tool_context._check_parse_error(grammar, "hello\tWRONG")
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    assert error.line == 1
    assert error.column == 7  # After "hello\t"


def test_error_context_includes_line_column(tool_context: LLGuidanceToolContext):
    """Test that error reporting includes useful line/column information."""
    grammar = r"""
start: "first" "\n" "second" "\n" "third"
"""
    test_input = "first\nsecond\nwrong"
    error = tool_context._check_parse_error(grammar, test_input)
    assert error is not None
    assert isinstance(error, UnexpectedToken)
    # Error should be on line 3
    assert error.line == 3
    assert error.column == 1
    # Context should show some preceding text
    assert len(error.context) > 0

def test_bad_quotes(tool_context: LLGuidanceToolContext):
    grammar = """
    start: "<START>" /(.\n)*/ " "<END>"
    """
    result = tool_context.validate_grammar(grammar)
    assert not result.is_valid


def test_batch_validation_all_passing(tool_context: LLGuidanceToolContext, tmp_path):
    """Test batch validation with all tests passing."""
    grammar = """
start: greeting name
greeting: "Hello" | "Hi"
name: /[A-Z][a-z]+/
%ignore /[ \t\n\r]+/
"""

    # Create test file
    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""{
        "tests": [
            {"input": "Hello Alice", "should_parse": true, "description": "Standard greeting"},
            {"input": "Hi Bob", "should_parse": true, "description": "Alternative greeting"},
            {"input": "Hello Charlie", "should_parse": true}
        ]
    }""")

    result = tool_context.run_batch_validation_tests(grammar, str(test_file))

    assert result.total == 3
    assert result.passed == 3
    assert result.failed == 0
    assert result.success_rate == 1.0
    assert result.failed_tests == []

    # Check individual results
    for test_result in result.results:
        assert test_result.correct
        assert test_result.actual_pass
        assert test_result.expected_pass


def test_batch_validation_with_failures(tool_context: LLGuidanceToolContext, tmp_path):
    """Test batch validation with some failures."""
    grammar = """
start: "hello"
"""

    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""{
        "tests": [
            {"input": "hello", "should_parse": true, "description": "Valid input"},
            {"input": "goodbye", "should_parse": false, "description": "Invalid input (expected to fail)"},
            {"input": "world", "should_parse": true, "description": "Wrong input"}
        ]
    }""")

    result = tool_context.run_batch_validation_tests(grammar, str(test_file))

    assert result.total == 3
    assert result.passed == 2
    assert result.failed == 1
    assert result.success_rate == 2/3

    # Check failed_tests
    assert len(result.failed_tests) == 1
    assert result.failed_tests[0].test_id == 2
    assert result.failed_tests[0].description == "Wrong input"
    assert not result.failed_tests[0].correct


def test_batch_validation_list_format(tool_context: LLGuidanceToolContext, tmp_path):
    """Test batch validation with simple list format (no 'tests' wrapper)."""
    grammar = """
start: /[0-9]+/
"""

    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""[
        {"input": "123", "should_parse": true},
        {"input": "456", "should_parse": true},
        {"input": "abc", "should_parse": false}
    ]""")

    result = tool_context.run_batch_validation_tests(grammar, str(test_file))

    assert result.total == 3
    assert result.passed == 3
    assert result.failed == 0


def test_batch_validation_input_preview(tool_context: LLGuidanceToolContext, tmp_path):
    """Test that long inputs are truncated in preview."""
    grammar = """
start: /[a-z]+/
"""

    long_input = "a" * 100
    test_file = tmp_path / "test_cases.json"
    test_file.write_text(f"""[
        {{"input": "{long_input}", "should_parse": true}}
    ]""")

    result = tool_context.run_batch_validation_tests(grammar, str(test_file))

    assert result.total == 1
    assert len(result.results[0].input_preview) == 53  # 50 chars + "..."
    assert result.results[0].input_preview.endswith("...")


def test_batch_validation_parse_errors(tool_context: LLGuidanceToolContext, tmp_path):
    """Test that parse errors are captured in results."""
    grammar = """
start: "hello" "world"
"""

    test_file = tmp_path / "test_cases.json"
    # yes, these tests aren't self consistent, but they test the mechanics
    test_file.write_text("""[
        {"input": "hello world", "should_parse": false},
        {"input": "helloworld", "should_parse": false},
        {"input": "hello world", "should_parse": true},
        {"input": "helloworld", "should_parse": true}
    ]""")

    result = tool_context.run_batch_validation_tests(grammar, str(test_file))

    # First test expects failure and gets failure (space is unexpected token)
    assert result.results[0].parse_error is not None
    assert result.results[0].correct

    # Second test expects failure and gets success
    assert result.results[1].parse_error is None
    assert not result.results[1].correct

    # Third test expects success and gets failure (space is unexpected token)
    assert result.results[2].parse_error is not None
    assert not result.results[2].correct

    # Fourth test expects success and gets success
    assert result.results[3].parse_error is None
    assert result.results[3].correct


def test_batch_validation_empty_tests(tool_context: LLGuidanceToolContext, tmp_path):
    """Test batch validation with empty test list."""
    grammar = """
start: "hello"
"""

    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""{"tests": []}""")

    result = tool_context.run_batch_validation_tests(grammar, str(test_file))

    assert result.total == 0
    assert result.passed == 0
    assert result.failed == 0
    assert result.success_rate == 0.0


def test_batch_validation_invalid_json_format(tool_context: LLGuidanceToolContext, tmp_path):
    """Test that invalid JSON format raises appropriate error."""
    grammar = """
start: "hello"
"""

    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""{"invalid": "format"}""")

    with pytest.raises(ValueError, match="Invalid test file format"):
        tool_context.run_batch_validation_tests(grammar, str(test_file))


def test_batch_validation_invalid_grammar(tool_context: LLGuidanceToolContext, tmp_path):
    """Test that invalid grammar raises appropriate error."""
    invalid_grammar = """
    start: undefined_rule
    """

    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""[
        {"input": "test", "should_parse": true}
    ]""")

    with pytest.raises(ValueError, match="Invalid grammar"):
        tool_context.run_batch_validation_tests(invalid_grammar, str(test_file))


def test_batch_validation_from_grammar_file(tool_context: LLGuidanceToolContext, tmp_path):
    """Test batch validation can load grammar from file."""
    grammar_file = tmp_path / "test.grammar"
    grammar_file.write_text("""
start: "hello"
""")

    test_file = tmp_path / "test_cases.json"
    test_file.write_text("""[
        {"input": "hello", "should_parse": true}
    ]""")

    result = tool_context.run_batch_validation_tests(str(grammar_file), str(test_file))

    assert result.total == 1
    assert result.passed == 1


# --- generate_with_grammar tests ---

def test_generate_disabled_by_default(tool_context: LLGuidanceToolContext):
    """Test that generation returns error when not enabled."""
    result = tool_context.generate_with_grammar(
        prompt="Generate hello",
        grammar='start: "hello"',
    )
    assert not result.is_valid
    assert "not enabled" in result.error


def test_generate_with_mock_openai():
    """Test generate_with_grammar calls OpenAI API correctly."""
    mock_tool_call = MagicMock()
    mock_tool_call.type = "custom_tool_call"
    mock_tool_call.input = "4 + 4"

    mock_response = MagicMock()
    mock_response.output = [mock_tool_call]

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    ctx = LLGuidanceToolContext.__new__(LLGuidanceToolContext)
    ctx.tokenizer = MagicMock()
    ctx.enable_generation = True
    ctx.model = "gpt-4.1"
    ctx._openai_client = mock_client

    grammar = """
start: expr
expr: term ("+" term)*
term: /[0-9]+/
"""

    result = ctx.generate_with_grammar(
        prompt="Add four plus four",
        grammar=grammar,
    )

    assert result.is_valid
    assert result.generated_text == "4 + 4"
    assert result.model == "gpt-4.1"

    # Verify API was called with correct structure
    call_args = mock_client.responses.create.call_args
    assert call_args.kwargs["model"] == "gpt-4.1"
    tools = call_args.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "custom"
    assert tools[0]["format"]["type"] == "grammar"
    assert tools[0]["format"]["syntax"] == "lark"


def test_generate_with_model_override():
    """Test that model parameter overrides default model."""
    mock_tool_call = MagicMock()
    mock_tool_call.type = "custom_tool_call"
    mock_tool_call.input = "hello"

    mock_response = MagicMock()
    mock_response.output = [mock_tool_call]

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    ctx = LLGuidanceToolContext.__new__(LLGuidanceToolContext)
    ctx.tokenizer = MagicMock()
    ctx.enable_generation = True
    ctx.model = "gpt-4.1"
    ctx._openai_client = mock_client

    result = ctx.generate_with_grammar(
        prompt="Say hello",
        grammar='start: "hello"',
        model="gpt-4.1-mini",
    )

    assert result.model == "gpt-4.1-mini"
    call_args = mock_client.responses.create.call_args
    assert call_args.kwargs["model"] == "gpt-4.1-mini"


def test_generate_handles_api_error():
    """Test that API errors are caught and returned in result."""
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = Exception("API rate limit exceeded")

    ctx = LLGuidanceToolContext.__new__(LLGuidanceToolContext)
    ctx.tokenizer = MagicMock()
    ctx.enable_generation = True
    ctx.model = "gpt-4.1"
    ctx._openai_client = mock_client

    result = ctx.generate_with_grammar(
        prompt="Generate something",
        grammar='start: "hello"',
    )

    assert not result.is_valid
    assert "rate limit" in result.error


def test_generate_no_tool_call_in_response():
    """Test handling when model returns no custom_tool_call."""
    mock_item = MagicMock()
    mock_item.type = "reasoning"

    mock_response = MagicMock()
    mock_response.output = [mock_item]

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    ctx = LLGuidanceToolContext.__new__(LLGuidanceToolContext)
    ctx.tokenizer = MagicMock()
    ctx.enable_generation = True
    ctx.model = "gpt-4.1"
    ctx._openai_client = mock_client

    result = ctx.generate_with_grammar(
        prompt="Generate something",
        grammar='start: "hello"',
    )

    assert not result.is_valid
    assert "No grammar-constrained output" in result.error


def test_generate_no_client():
    """Test error when OpenAI client is not initialized."""
    ctx = LLGuidanceToolContext.__new__(LLGuidanceToolContext)
    ctx.tokenizer = MagicMock()
    ctx.enable_generation = True
    ctx.model = "gpt-4.1"
    ctx._openai_client = None

    result = ctx.generate_with_grammar(
        prompt="Generate something",
        grammar='start: "hello"',
    )

    assert not result.is_valid
    assert "not initialized" in result.error


@patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com", "AZURE_OPENAI_API_KEY": "test-key"})
@patch("mcp_grammar_tools.llg_tools.LLGuidanceToolContext._init_azure_client")
def test_init_detects_azure_with_api_key(mock_azure_client):
    """Test that AZURE_OPENAI_ENDPOINT triggers Azure client init."""
    mock_azure_client.return_value = MagicMock()
    ctx = LLGuidanceToolContext(enable_generation=True)
    mock_azure_client.assert_called_once_with("https://test.openai.azure.com")
    assert ctx._openai_client is not None


@patch.dict("os.environ", {}, clear=False)
@patch("openai.OpenAI")
def test_init_defaults_to_openai(mock_openai_cls):
    """Test that without AZURE_OPENAI_ENDPOINT, plain OpenAI client is used."""
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("AZURE_OPENAI_ENDPOINT")}
    with patch.dict("os.environ", env, clear=True):
        mock_openai_cls.return_value = MagicMock()
        ctx = LLGuidanceToolContext(enable_generation=True)
        mock_openai_cls.assert_called_once()
        assert ctx._openai_client is not None


@patch.dict("os.environ", {}, clear=False)
def test_init_missing_credentials_does_not_crash():
    """Test that missing credentials don't crash the server on startup."""
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")}
    with patch.dict("os.environ", env, clear=True):
        ctx = LLGuidanceToolContext(enable_generation=True)
        # Server should start — client is None but no crash
        assert ctx._openai_client is None
        # Validation tools still work
        result = ctx.validate_grammar('start: "hello"')
        assert result.is_valid
        # Generation returns a clear error
        gen_result = ctx.generate_with_grammar(prompt="test", grammar='start: "hello"')
        assert not gen_result.is_valid
        assert "not initialized" in gen_result.error
        assert "Initialization error" in gen_result.error
