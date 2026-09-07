*This project has been created as part of the 42 curriculum by wkerdad.*

# Call Me Maybe

## Description

Call Me Maybe is a function calling tool for small language models. It translates
natural language requests into structured function calls with typed arguments. For
example, given the question *"What is the sum of 40 and 2?"*, the system does not
answer `42` directly. Instead, it produces a machine-executable function call:

```json
{
  "function": "fn_add_numbers",
  "arguments": {"a": 40, "b": 2}
}
```

The goal is to achieve near-perfect reliability (90%+ accuracy) even with a very
small 0.6B parameter model (Qwen/Qwen3-0.6B). This is accomplished through
**constrained decoding**, a technique that guides the model's output token-by-token
to guarantee 100% syntactically valid and schema-compliant JSON, rather than relying
on the model spontaneously generating correct structured output from a prompt.

## Instructions

### Prerequisites

- Python 3.10 or later (the project is developed and tested against 3.14)
- [uv](https://docs.astral.sh/uv/) as the package and environment manager

### Installation

```bash
uv sync
```

This installs the project dependencies (`numpy`, `pydantic`) and the `llm_sdk`
package. The first run also downloads the Qwen/Qwen3-0.6B model weights.

### Running the program

```bash
uv run python -m src
```

By default, the program reads input files from `data/input/` and writes its output
to `data/output/function_calling_results.json`.

Custom paths can be passed with command-line arguments:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

A different model can be selected with `--model` (defaults to `Qwen/Qwen3-0.6B`).

### Makefile

- `make install` — synch dependencies with `uv sync`
- `make run` — run the main script
- `make debug` — run the main script under `pdb`
- `make clean` — remove caches and generated output
- `make lint` — run `flake8` and `mypy` over the project
- `make lint-strict` — run the same linters with the `--strict` flag

## Algorithm Explanation

The implementation uses a constrained decoding loop grounded in the LLM SDK's
`get_logits_from_input_ids` method. At each generation step the model returns the raw
logits (unnormalized probabilities) for every possible next token. The decoder masks
out every token that would violate the expected JSON structure or the target schema,
sets those logits to `-inf`, and then greedily selects the remaining token with the
highest logit.

The output is built piece by piece in a rigid grammar:

1. **Function name**: decoded under the constraint that the partially generated text
   is always a prefix of at least one known function name. Once a full name matches,
   generation stops.
2. **String parameters**: decoded until the closing quote is reached. The generated
   text is accumulated and stripped, guaranteeing a well-formed string value.
3. **Number/integer parameters**: decoded token-by-token while enforcing that the
   accumulated text always remains a valid float representation.
4. **Boolean parameters**: decoded under the constraint that the partial text remains
   a prefix of either `true` or `false`.

The full context (system prompt + already emitted JSON) is passed back into the model
at each step so the model's conditional distribution guides which of the structurally
valid tokens is actually selected. This is what separates constrained decoding from
mere prompting: the structure is *guaranteed* by the mask, while the *meaning* is
chosen by the model.

## Design Decisions

- **Greedy (argmax) selection** keeps the pipeline deterministic and fast, important
  for running several thousand logits computations within the time budget.
- **Structured incremental context**: the emitted JSON prefix is re-encoded at every
  step so that the masked distribution has full visibility of everything generated so
  far.
- **Pydantic everywhere**: input files and function definitions are validated with
  Pydantic models to fail fast on malformed data and to keep type expectations explicit.
- **Separation of concerns**: loaders, validators, the CLI parser, and the constrained
  decoder live in separate modules, making the constrained decoding logic easily
  testable in isolation.
- **Configurable model path** via `--model` so the default `Qwen/Qwen3-0.6B` can be
  swapped for another model without code changes.

## Performance Analysis

Using a 0.6B model, the constrained decoder reliably produces 100% parseable JSON
because invalid tokens are forbidden by construction. Accuracy of function selection
and argument extraction comfortably exceeds 90% on the provided test sets, and every
output object is schema-compliant by design (correct keys, types, and allowed values).
All test prompts are processed well within the 5 minute budget; the dominant cost is
the serial forward passes through the small model for each token.

## Challenges Faced

- **Small-model reliability**: a 0.6B model is notoriously bad at emitting valid JSON
  spontaneously. Solving this required moving the guarantee out of the prompt and into
  the decoding loop.
- **Token reconstruction**: mapping raw token IDs back to their string representations
  for prefix/validity checks required careful handling, especially because tokenizers
  keep leading spaces and punctuation as part of a token.
- **Schema-aware masking**: enforcing not just *valid* JSON but the *specific* schema
  (e.g. a field restricted to `number`, an enumerable boolean) meant the mask logic had
  to track the current parse state across steps.
- **Escape handling in strings**: prompts referencing quoted content (e.g. escaped
  quotes inside a string argument) required care so that decoded strings remain valid
  JSON without breaking the surrounding structure.

## Testing Strategy

The project validates functionality through a mix of manual runs and unit-style checks:

- Running the full pipeline against the provided `data/input` set and confirming the
  output JSON parses and matches the expected schema exactly.
- Edge cases such as empty prompts, large/decimal numbers, booleans, and multiple
  parameters per function.
- Verifying graceful handling of malformed or missing input files (clear error
  messages, no crashes).
- Confirming the output file is created in `data/output/` and contains the exact keys
  `prompt`, `name`, and `parameters`.

## Resources

- [Qwen3 models](https://huggingface.co/Qwen/Qwen3-0.6B) — the small LLM used for
  inference.
- [Transformers documentation](https://huggingface.co/docs/transformers) — for
  tokenization and model inference details.
- [Pydantic documentation](https://docs.pydantic.dev/) — for data validation.
- [uv documentation](https://docs.astral.sh/uv/) — for dependency and environment
  management.
- *How AI was used*: AI assisted with structuring the constrained decoding loop,
  debugging token/reconstruction edge cases, and writing the test scenarios. All
  generated code was reviewed, understood, and validated by running the pipeline
  against real data.
