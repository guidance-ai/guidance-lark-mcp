from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, computed_field
from llguidance import LLTokenizer, LLMatcher
from pathlib import Path
import requests
import json
import os
import logging

logger = logging.getLogger(__name__)

class UnexpectedToken(BaseModel):
    error_type: Literal["unexpected_token"] = "unexpected_token"
    message: str = "Unexpected token"
    llguidance_error: str
    position: int
    line: int
    column: int
    token: str
    context: str


class UnexpectedEOF(BaseModel):
    error_type: Literal["unexpected_eof"] = "unexpected_eof"
    message: str = "Unexpected EOF: grammar expects more input"


ParseError = Annotated[Union[
    UnexpectedToken,
    UnexpectedEOF,
], Field(discriminator="error_type")]


class ParseResult(BaseModel):
    expected_valid: bool
    parse_valid: bool
    parse_error: ParseError | None

    @computed_field
    def success(self) -> bool:
        return self.expected_valid == self.parse_valid


class BatchTestCase(BaseModel):
    """A single test case for batch validation."""
    input: str
    should_parse: bool
    description: str | None = None


class BatchTestCaseResult(BaseModel):
    """Result of running a single test case."""
    test_id: int
    input_preview: str  # First 50 chars of input
    description: str | None
    expected_pass: bool
    actual_pass: bool
    correct: bool
    parse_error: ParseError | None


class BatchTestResult(BaseModel):
    """Results of running a batch of test cases."""
    total: int
    passed: int
    failed: int
    success_rate: float
    results: list[BatchTestCaseResult]

    @computed_field
    def failed_tests(self) -> list[BatchTestCaseResult]:
        """Return only the failing test cases."""
        return [result for result in self.results if not result.correct]


class GrammarValidationResult(BaseModel):
    is_valid: bool
    errors_and_warnings: list[str]


class GenerationResult(BaseModel):
    """Result of LLM generation with grammar."""
    generated_text: str
    is_valid: bool = True
    error: str | None = None
    model: str | None = None


class LLGuidanceToolContext:
    def __init__(self, enable_generation: bool = False, model: str = "gpt-4.1"):
        self.tokenizer = LLTokenizer("byte")
        self.enable_generation = enable_generation
        self.model = model
        self._openai_client = None

        if enable_generation:
            self._init_openai_client()

    def _init_openai_client(self):
        """Initialize OpenAI client with auto-detection of Azure Entra ID vs API key auth."""
        try:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

            if azure_endpoint:
                self._openai_client = self._init_azure_client(azure_endpoint)
            else:
                from openai import OpenAI
                self._openai_client = OpenAI()

                base_url = os.environ.get("OPENAI_BASE_URL", "")
                if base_url:
                    logger.info("OpenAI backend: custom endpoint (%s)", base_url)
                else:
                    logger.info("OpenAI backend: OpenAI (default)")

            logger.info("Default model: %s", self.model)
        except Exception as e:
            self._openai_client = None
            self._generation_init_error = str(e)
            logger.warning("Generation disabled: failed to initialize OpenAI client: %s", e)

    def _init_azure_client(self, azure_endpoint: str):
        """Initialize AzureOpenAI client with Entra ID (DefaultAzureCredential) or API key."""
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if azure_api_key:
            from openai import AzureOpenAI
            logger.info("OpenAI backend: Azure OpenAI with API key (%s)", azure_endpoint)
            return AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=api_version,
            )
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                from openai import AzureOpenAI
            except ImportError:
                raise RuntimeError(
                    "AZURE_OPENAI_ENDPOINT is set but azure-identity is not installed. "
                    "Install with: pip install guidance-lark-mcp[azure]"
                )
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            logger.info("OpenAI backend: Azure OpenAI with Entra ID (%s)", azure_endpoint)
            return AzureOpenAI(
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )

    def generate_with_grammar(
        self,
        prompt: str,
        grammar: str,
        model: str | None = None,
        max_tokens: int = 300,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        """
        Generate text using OpenAI API constrained by a Lark grammar.

        Uses the Responses API with a custom tool that has a grammar format
        constraint, so the model output is guaranteed to conform to the grammar.

        Args:
            prompt: The user prompt describing what to generate
            grammar: Lark grammar string or path to grammar file
            model: OpenAI model name (defaults to instance model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (omitted if None, as some models don't support it)
            reasoning_effort: Reasoning effort ("low", "medium", "high")

        Returns:
            GenerationResult with generated text or error
        """
        if not self.enable_generation:
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error="Generation not enabled. Set ENABLE_GENERATION=true environment variable.",
            )

        if self._openai_client is None:
            init_error = getattr(self, "_generation_init_error", None)
            detail = f" Initialization error: {init_error}" if init_error else ""
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error=f"OpenAI client not initialized. Check OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT.{detail}",
            )

        try:
            grammar_content = self._resolve_grammar_input(grammar)
            use_model = model or self.model

            kwargs = dict(
                model=use_model,
                input=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "type": "custom",
                        "name": "generate",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": grammar_content,
                        },
                    }
                ],
                tool_choice="required",
                max_output_tokens=max_tokens,
                stream=False,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}

            response = self._openai_client.responses.create(**kwargs)

            # Extract generated text from custom_tool_call output
            for item in response.output:
                logger.debug("Response output item: type=%s, %s", getattr(item, "type", None), item)
                if getattr(item, "type", None) == "custom_tool_call":
                    return GenerationResult(
                        generated_text=item.input,
                        model=use_model,
                    )

            output_types = [getattr(item, "type", None) for item in response.output]
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error=f"No grammar-constrained output returned by model. Output types: {output_types}",
                model=use_model,
            )

        except Exception as e:
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error=f"Generation error: {str(e)}",
            )

    def _calculate_line_column(self, text: str, position: int) -> tuple[int, int]:
        """Calculate line and column numbers from character position (1-indexed)."""
        if position == 0:
            return 1, 1

        lines = text[:position].split("\n")
        line = len(lines)
        column = len(lines[-1]) + 1
        return line, column

    def _check_parse_error(self, grammar: str, test_input: str) -> ParseError | None:
        # Assume grammar is valid
        matcher = LLMatcher(self.tokenizer, grammar)
        tokens = self.tokenizer.tokenize_str(test_input)

        for ix, token in enumerate(tokens):
            if not matcher.consume_token(token):
                position = len(
                    self.tokenizer.decode_bytes(tokens[:ix]).decode(errors="ignore")
                )
                line, column = self._calculate_line_column(test_input, position)
                return UnexpectedToken(
                    llguidance_error=matcher.get_error(),
                    position=position,
                    line=line,
                    column=column,
                    token=self.tokenizer.decode_bytes([token]).decode(errors="ignore"),
                    context=self.tokenizer.decode_bytes(
                        tokens[max(0, ix - 5) : ix]
                    ).decode(errors="ignore"),
                )
        if not matcher.is_accepting():
            return UnexpectedEOF()


    def validate_grammar(self, grammar: str) -> GrammarValidationResult:
        """Validate grammar completeness and consistency."""
        grammar = self._resolve_grammar_input(grammar)
        is_err, warnings = LLMatcher.validate_grammar_with_warnings(grammar)
        return GrammarValidationResult(
            is_valid=not is_err,
            errors_and_warnings=warnings,
        )

    def get_llguidance_documentation(self) -> str:
        """Get the URL to the llguidance grammar syntax documentation."""
        return requests.get("https://raw.githubusercontent.com/guidance-ai/llguidance/refs/heads/main/docs/syntax.md").text

    def run_batch_validation_tests(self, grammar: str, test_file: str) -> BatchTestResult:
        """
        Run batch validation tests from a JSON file.

        Args:
            grammar: Grammar string or path to grammar file
            test_file: Path to JSON file containing test cases

        Returns:
            BatchTestResult with detailed results for each test case

        Raises:
            ValueError: If grammar is invalid or test file format is incorrect

        The test file should be JSON with format:
        {
            "tests": [
                {"input": "test string", "should_parse": true, "description": "optional"},
                ...
            ]
        }

        Or a simple list:
        [
            {"input": "test string", "should_parse": true},
            ...
        ]
        """
        # Resolve grammar input
        grammar_content = self._resolve_grammar_input(grammar)

        # Validate grammar before running tests
        validation_result = self.validate_grammar(grammar_content)
        if not validation_result.is_valid:
            error_summary = "; ".join(validation_result.errors_and_warnings[:3])
            raise ValueError(f"Invalid grammar: {error_summary}")

        # Load and parse test file
        test_file_path = Path(test_file).expanduser()
        test_data = json.loads(test_file_path.read_text())

        # Handle different JSON formats
        if isinstance(test_data, dict) and "tests" in test_data:
            tests_raw = test_data["tests"]
        elif isinstance(test_data, list):
            tests_raw = test_data
        else:
            raise ValueError("Invalid test file format. Expected list or {'tests': [...]}")

        # Parse test cases
        test_cases = [BatchTestCase(**test) for test in tests_raw]

        # Run tests
        results: list[BatchTestCaseResult] = []
        for i, test_case in enumerate(test_cases):
            parse_error = self._check_parse_error(grammar_content, test_case.input)
            actual_pass = parse_error is None

            # Create input preview (first 50 chars)
            input_preview = test_case.input[:50]
            if len(test_case.input) > 50:
                input_preview += "..."

            result = BatchTestCaseResult(
                test_id=i,
                input_preview=input_preview,
                description=test_case.description,
                expected_pass=test_case.should_parse,
                actual_pass=actual_pass,
                correct=(actual_pass == test_case.should_parse),
                parse_error=parse_error,
            )
            results.append(result)

        # Calculate statistics
        passed = sum(1 for r in results if r.correct)
        failed = len(results) - passed
        success_rate = (passed / len(results)) if results else 0.0

        return BatchTestResult(
            total=len(results),
            passed=passed,
            failed=failed,
            success_rate=success_rate,
            results=results,
        )

    def _resolve_grammar_input(self, grammar: str) -> str:
        """Handle grammar parameter - could be file path or actual grammar content."""
        # Check if it looks like a file path
        if (
            grammar.endswith(".lark")
            or grammar.endswith(".grammar")
            or grammar.startswith("/")
            or grammar.startswith("~/")
            or grammar.startswith("./")
            or grammar.startswith("../")
        ):
            try:
                # Expand ~ and read file
                file_path = Path(grammar).expanduser()
                return file_path.read_text()
            except FileNotFoundError:
                # If file doesn't exist, maybe it's actually grammar content that looks like a path
                if "\n" in grammar or ":" in grammar:
                    return grammar  # Treat as grammar content
                else:
                    raise FileNotFoundError(f"Grammar file not found: {grammar}")
            except Exception as e:
                # If reading fails, maybe it's grammar content
                if "\n" in grammar or ":" in grammar:
                    return grammar
                else:
                    raise e
        else:
            # Treat as direct grammar content
            return grammar
