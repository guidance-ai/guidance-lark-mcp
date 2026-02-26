"""MCP LLGuidance Grammar Tools - Validate and test llguidance grammars."""

from .llg_tools import (
    LLGuidanceToolContext,
    UnexpectedToken,
    UnexpectedEOF,
    ParseError,
    ParseResult,
    BatchTestCase,
    BatchTestCaseResult,
    BatchTestResult,
    GrammarValidationResult,
)

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("guidance-lark-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "LLGuidanceToolContext",
    "UnexpectedToken",
    "UnexpectedEOF",
    "ParseError",
    "ParseResult",
    "BatchTestCase",
    "BatchTestCaseResult",
    "BatchTestResult",
    "GrammarValidationResult",
]