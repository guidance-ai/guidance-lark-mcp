# GraphQL Query Grammar

An [llguidance](https://github.com/guidance-ai/llguidance) grammar for the executable subset of the [GraphQL specification](https://spec.graphql.org/June2018/) (June 2018).

## Scope

**Included:** queries, mutations, subscriptions, selection sets, fields, aliases, arguments, variables with types and defaults, fragment definitions/spreads/inline fragments, and all value types (int, float, string, boolean, null, enum, list, object).

**Not included:** schema definition language (SDL), directives (`@skip`, `@include`), or type system definitions.

## Files

| File | Description |
|------|-------------|
| `graphql.lark` | The grammar (Lark format for llguidance) |
| `tests.json` | 34 test cases — 25 valid inputs, 9 invalid inputs |
| `demo_generate.py` | Demo script — generate valid GraphQL with [guidance](https://github.com/guidance-ai/guidance) |

## Validation

```bash
# Validate grammar
# → uses the validate_grammar MCP tool

# Run test suite
# → uses the run_batch_validation_tests MCP tool against tests.json
# → 34/34 passing (25 should_parse:true, 9 should_parse:false)
```

## Demo: Constrained Generation

The demo script uses the [guidance](https://github.com/guidance-ai/guidance) library to generate GraphQL queries that are **guaranteed** to be syntactically valid:

```bash
pip install -r examples/graphql/requirements.txt

# Run with default model (Qwen2.5-0.5B-Instruct)
python demo_generate.py

# Or specify a model
python demo_generate.py --model microsoft/Phi-3.5-mini-instruct
```

The default model ([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) is ~950MB and runs on CPU. It's small enough for a quick demo — the grammar constraint guarantees valid syntax regardless of model size, though larger models will produce more semantically meaningful queries.

## Example Inputs

**Valid:**
```graphql
{ user { id name email } }
```

```graphql
query GetUser($id: ID!) {
  user(id: $id) {
    ...UserFields
  }
}
fragment UserFields on User { id name email }
```

**Invalid:**
```
query { }           # empty selection set
{ user(id: ) }      # argument missing value
fragment on User {}  # fragment missing name
```
