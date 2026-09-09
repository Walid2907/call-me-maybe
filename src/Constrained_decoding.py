import json
from typing import Any, Dict, List


def load_vocab(model: Any) -> Dict[int, str]:
    """Load vocabulary mapping from token ID to string.

    Args:
        model: Small_LLM_Model instance.

    Returns:
        Dictionary mapping token_id -> token_string.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab: Dict[int, str] = json.load(f)

    return vocab


def system_prompt_builder(functions: List[Any]) -> str:
    """Build a system prompt listing available functions.

    Args:
        functions: List of Functions pydantic models.

    Returns:
        Formatted system prompt string.
    """
    lines = [
        "You are a natural language to function call system.",
        "Given this function registry:",
    ]
    for f in functions:
        param_parts = []
        for p, info in f.parameters.items():
            param_parts.append(
                f'"{p}": {{"type": "{info.type}"}}'
            )
        params_str = ", ".join(param_parts)
        lines.append(
            f'{{"name": "{f.name}", '
            f'"description": "{f.description}", '
            f'"parameters": {{{params_str}}}}}'
        )
    lines.append(
        "Chose the appropriate function "
        "and its parameters based on the user input."
    )
    lines.append(
        "Rules: generate only valid JSON. "
        "Use exact types: numbers without quotes, "
        "strings with quotes. "
        "strings in the value of a key should be inside double quotes. "
        "and for nested strings use single quotes for the inside string. "
        "For regex use standard syntax without extra "
        "parentheses."
    )
    return "\n".join(lines)


def generate_function_name(
    model: Any,
    context: str,
    function_names: List[str]
) -> str:
    """Generate a function name using constrained decoding.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending with '"name": "'.
        function_names: Valid function name strings.
        vocab: Pre-loaded token_id -> string mapping.
        max_tokens: Safety limit.

    Returns:
        Generated function name.
    """
    generated: str = ""
    max_tokens = len(sorted(function_names,
                            key=lambda x: len(x), reverse=True)[0])

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + generated)[0].tolist()
        )

        for token_id in range(len(logits)):
            token_str = model.decode([token_id])
            combined = generated + token_str
            if not any(
                fn.startswith(combined)
                for fn in function_names
            ):
                logits[token_id] = float("-inf")
        best_id = _argmax(logits)
        best_str: str = model.decode([best_id])

        generated += best_str

        if generated in function_names:
            break

    return generated


def generate_string_value(
    model: Any,
    context: str,
    max_tokens: int
) -> str:
    """Generate a string parameter value token by token.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending with an opening '"'.
        vocab: Pre-loaded token_id -> string mapping.
        max_tokens: Safety limit.

    Returns:
        Generated string value.
    """
    value: str = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + value)[0].tolist()
        )
        best_id = _argmax(logits)
        best_str: str = model.decode([best_id])
        if '"' in best_str:
            value += best_str.split('"')[0]
            break
        value += best_str
    return value.strip()


def _is_float(s: str) -> bool:
    """Check if string is a valid float representation."""
    try:
        float(s)
        return True
    except (ValueError, OverflowError):
        return False


def generate_number_value(
    model: Any,
    context: str,
    max_tokens: int,
) -> str:
    """Generate a numeric value with constrained decoding.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending before the number.
        vocab: Pre-loaded token_id -> string mapping.
        max_tokens: Safety limit.

    Returns:
        Generated number as string.
    """
    number: str = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + number)[0].tolist()
        )
        for token_id in range(len(logits)):
            token_str: str = model.decode([token_id])
            if (token_str in (",", "}") and not _is_float(number)):
                logits[token_id] = float("-inf")
            elif (token_str not in (",", "}")
                  and not _is_float(number + token_str)):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str: str = model.decode([best_id])
        if best_str == "," or best_str == "}":
            break
        number += best_str
    return number


def _argmax(logits: List[float]) -> int:
    """Return the index of the maximum value."""
    best_idx = 0
    best_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_val:
            best_val = logits[i]
            best_idx = i
    return best_idx


def generate_boolean_value(
    model: Any,
    context: str,
    max_tokens: int
) -> str:
    """Generate a boolean value (true/false) with constrained decoding.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending before the boolean value.
        max_tokens: Safety limit.

    Returns:
        Generated boolean value as string "true" or "false".
    """
    valid_values = ["true", "false"]
    generated: str = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + generated)[0].tolist()
        )

        for token_id in range(len(logits)):
            token_str = model.decode([token_id])
            combined = generated + token_str
            if not any(
                val.startswith(combined)
                for val in valid_values
            ):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str = model.decode([best_id])

        generated += best_str

        if generated in valid_values:
            break

    return generated
