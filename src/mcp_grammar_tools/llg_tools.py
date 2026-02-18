from typing import Literal, Union, Annotated, Optional
from pydantic import BaseModel, Field, computed_field
from llguidance import LLTokenizer, LLMatcher
from pathlib import Path
import requests
import json

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
    error: Optional[str] = None


class LLGuidanceToolContext:
    def __init__(self, enable_llm: bool = False, model_path: Optional[str] = None):
        self.tokenizer = LLTokenizer("byte")
        self.enable_llm = enable_llm
        self.model_path = model_path
        self._phi4_model = None
        self._phi4_tokenizer = None
        
        if enable_llm and model_path:
            self._load_phi4_model()

    def _load_phi4_model(self):
        """Load Phi-4 model into memory."""
        if self._phi4_model is not None:
            return  # Already loaded
            
        try:
            import onnxruntime_genai as og
            from pathlib import Path
            import json
            
            # Apply tokenizer fix
            model_dir = Path(self.model_path)
            config_path = model_dir / "tokenizer_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                
                tokens_to_fix = ["200019", "200020", "200021", "200022"]
                for token_id in tokens_to_fix:
                    if token_id in config.get("added_tokens_decoder", {}):
                        config["added_tokens_decoder"][token_id]["special"] = False
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            tokenizer_path = model_dir / "tokenizer.json"
            if tokenizer_path.exists():
                with open(tokenizer_path) as f:
                    tokenizer_data = json.load(f)
                
                for token in tokenizer_data.get("added_tokens", []):
                    if token.get("id") in [200019, 200020, 200021, 200022]:
                        token["special"] = False
                
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
            
            # Load model
            self._phi4_model = og.Model(self.model_path)
            self._phi4_tokenizer = og.Tokenizer(self._phi4_model)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Phi-4 model: {e}")

    def generate_with_grammar(
        self,
        messages: list[dict[str, str]],
        grammar: str,
        max_tokens: int = 300,
        temperature: float = 0.7
    ) -> GenerationResult:
        """
        Generate text using Phi-4 model with grammar constraint.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            grammar: Lark grammar string or path to grammar file
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            GenerationResult with generated text or error
        """
        if not self.enable_llm:
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error="LLM generation not enabled. Start server with --enable-llm flag."
            )
        
        if self._phi4_model is None:
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error="Phi-4 model not loaded"
            )
        
        try:
            import onnxruntime_genai as og
            
            # Resolve grammar
            grammar_content = self._resolve_grammar_input(grammar)
            
            # Format messages into prompt
            prompt = self._phi4_tokenizer.apply_chat_template(
                json.dumps(messages),
                add_generation_prompt=True
            )
            
            # Tokenize
            tokens = self._phi4_tokenizer.encode(prompt)
            
            # Setup generation parameters
            params = og.GeneratorParams(self._phi4_model)
            params.set_search_options(
                max_length=len(tokens) + max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0
            )
            
            # Apply grammar constraint
            params.set_guidance("lark_grammar", grammar_content)
            
            # Generate
            generator = og.Generator(self._phi4_model, params)
            generator.append_tokens(tokens)
            
            token_count = 0
            while not generator.is_done() and token_count < max_tokens:
                generator.generate_next_token()
                token_count += 1
            
            # Decode output
            sequence = generator.get_sequence(0)
            full_output = self._phi4_tokenizer.decode(sequence)
            
            # Extract assistant's response
            if "<|assistant|>" in full_output:
                response = full_output.split("<|assistant|>")[-1].strip()
            else:
                response = full_output[len(prompt):].strip()
            
            response = response.replace("<|end|>", "").strip()
            
            return GenerationResult(generated_text=response)
            
        except Exception as e:
            return GenerationResult(
                generated_text="",
                is_valid=False,
                error=f"Generation error: {str(e)}"
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
