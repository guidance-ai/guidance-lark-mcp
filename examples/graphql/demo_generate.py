#!/usr/bin/env python3
"""
Demo: Generate valid GraphQL queries using guidance + a Lark grammar.

This script loads a language model and uses the GraphQL grammar from this
example directory to constrain generation so the output is always syntactically
valid GraphQL.

Usage:
    # With a local Transformers model (default: Phi-4-mini-instruct):
    python demo_generate.py

    # With a specific model:
    python demo_generate.py --model microsoft/Phi-3.5-mini-instruct

    # With a GGUF model via llama.cpp:
    python demo_generate.py --backend llamacpp --model path/to/model.gguf

Requirements:
    pip install guidance torch transformers
"""

import argparse
from pathlib import Path

import guidance
from guidance import lark, models


GRAMMAR_FILE = Path(__file__).parent / "graphql.lark"

PROMPTS = [
    "Write a GraphQL query to fetch a user's id, name, and email.",
    "Write a GraphQL mutation to create a new blog post with a title and content, returning the post id.",
    "Write a GraphQL query to search for users and posts, using inline fragments to get name from users and title from posts.",
]


def load_model(backend: str, model_name: str, **kwargs):
    """Load a guidance model."""
    if backend == "transformers":
        return models.Transformers(model_name, **kwargs)
    elif backend == "llamacpp":
        return models.LlamaCpp(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def generate_graphql(lm, grammar: str, prompt: str, max_tokens: int = 256) -> str:
    """Generate a GraphQL query constrained by the grammar."""
    with guidance.user():
        lm += prompt + " Output ONLY the raw GraphQL, no explanation."

    with guidance.assistant():
        lm += lark(lark_grammar=grammar, name="graphql", max_tokens=max_tokens)

    return lm["graphql"]


def main():
    parser = argparse.ArgumentParser(description="Generate valid GraphQL with guidance + Lark grammar")
    parser.add_argument("--model", default="microsoft/Phi-4-mini-instruct", help="Model name or path")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "llamacpp"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt (otherwise runs built-in examples)")
    args = parser.parse_args()

    # Load grammar
    grammar = GRAMMAR_FILE.read_text()
    print(f"Loaded grammar from {GRAMMAR_FILE} ({len(grammar)} chars)\n")

    # Load model
    print(f"Loading model: {args.model} (backend: {args.backend})")
    lm = load_model(args.backend, args.model)
    print("Model loaded.\n")

    # Generate
    prompts = [args.prompt] if args.prompt else PROMPTS
    for i, prompt in enumerate(prompts, 1):
        print(f"{'='*72}")
        print(f"Prompt {i}: {prompt}")
        print(f"{'='*72}")

        result = generate_graphql(lm, grammar, prompt, args.max_tokens)

        print(result)
        print()


if __name__ == "__main__":
    main()
