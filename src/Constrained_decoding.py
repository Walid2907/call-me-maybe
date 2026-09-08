import json
import re
from typing import Any, Dict, List

_FLOAT_PREFIX_RE = re.compile(
    r"(?:"
    r"[+-]?\.\d+(?:[eE][+-]?\d*)?|"
    r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d*)?|"
    r"[+-]?\.|"
    r"[+-]?"
    r")"
)


def build_token_map(model: Any) -> Dict[int, str]:
    """Build a token id -> decoded string mapping from the vocabulary file.

    Decoding each vocabulary entry with the model's public ``decode`` method
    guarantees the strings used for prefix/validity checks exactly match what
    the tokenizer would produce, including byte-level encodings (space =
    ``\\u0120``, newline = ``\\u010a``) and special tokens.

    Args:
        model: Small_LLM_Model instance.

    Returns:
        Dictionary mapping token id to its decoded string representation.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return {
        int(token_id): model.decode([int(token_id)])
        for token, token_id in vocab.items()
    }


def _token_str(
    model: Any, token_id: int, token_map: Dict[int, str]
) -> str:
    """Return the string form of a token id.

    Uses the pre-built vocabulary mapping when possible and falls
    back to the model decoder for ids not present in the vocabulary.

    Args:
        model: Small_LLM_Model instance.
        token_id: The id of the token.
        token_map: Pre-built id -> string vocabulary mapping.

    Returns:
        The decoded token string.
    """
    token = token_map.get(token_id)
    if token is not None:
        return token
    return str(model.decode([token_id]))


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
                f'"{p}": {{"type": "{info.type}"}}')
        params_str = ", ".join(param_parts)
        lines.append(
            f'{{"name": "{f.name}", '
            f'"description": "{f.description}", '
            f'"parameters": {{{params_str}}}}}')
    lines.append(
        "Choose the appropriate function "
        "and its parameters based on the user input.")
    lines.append(
        "Rules: generate only valid JSON. "
        "Use exact types: numbers without quotes, "
        "strings with quotes. "
        "For regex use standard syntax without extra "
        "parentheses.")
    return "\n".join(lines)


def generate_function_name(
    model: Any,
    context: str,
    function_names: List[str],
    token_map: Dict[int, str],
) -> str:
    """Generate a function name using constrained decoding.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending with '"name": "'.
        function_names: Valid function name strings.
        token_map: Pre-loaded id -> string vocabulary mapping.

    Returns:
        Generated function name.
    """
    if not function_names:
        return ""
    generated: str = ""
    longest = len(
        sorted(function_names, key=lambda x: len(x), reverse=True)[0])
    max_tokens = max(longest * 2, 16)

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + generated)[0].tolist())

        for token_id in range(len(logits)):
            token_str = _token_str(model, token_id, token_map)
            combined = generated + token_str
            if not any(
                fn.startswith(combined)
                for fn in function_names):
                logits[token_id] = float("-inf")
        best_id = _argmax(logits)
        best_str: str = _token_str(model, best_id, token_map)

        generated += best_str

        if generated in function_names:
            break

    return generated


def _find_all(text: str, sub: str) -> List[int]:
    """Return every position where ``sub`` occurs inside ``text``."""
    positions: List[int] = []
    start = 0
    while True:
        pos = text.find(sub, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


_BY_FIRST_CHAR_CACHE: Dict[int, Dict[str, List[int]]] = {}


def _by_first_char(token_map: Dict[int, str]) -> Dict[str, List[int]]:
    """Group token ids by the first character of their decoded string.

    Used to speed up substring-constrained token selection.

    Args:
        token_map: Pre-loaded id -> string vocabulary mapping.

    Returns:
        Mapping of first character to list of token ids.
    """
    cache_key = id(token_map)
    cached = _BY_FIRST_CHAR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    grouped: Dict[str, List[int]] = {}
    for token_id, token_str in token_map.items():
        if token_str:
            grouped.setdefault(token_str[0], []).append(token_id)
    _BY_FIRST_CHAR_CACHE[cache_key] = grouped
    return grouped


def _allowed_token_ids(
    value: str,
    matches: List[int],
    source: str,
    token_map: Dict[int, str],
    by_first_char: Dict[str, List[int]],
) -> set:
    """Return token ids that keep ``value`` a contiguous span of ``source``.

    Function calling arguments are normally quoted verbatim from the user
    request. Enforcing that the partially generated value stays a substring
    of the request prevents the model from inventing or corrupting argument
    text (e.g. doubled backslashes, dropped quotes).

    Args:
        value: Partially generated string value.
        matches: Positions in ``source`` where ``value`` currently occurs.
        source: The user request text.
        token_map: Pre-loaded id -> string vocabulary mapping.
        by_first_char: Token ids grouped by first decoded character.

    Returns:
        Set of admissible next token ids (empty when the value can no
        longer be extended, i.e. it should terminate).
    """
    allowed: set = set()
    for pos in matches:
        suffix = source[pos + len(value):]
        if not suffix:
            continue
        first = suffix[0]
        for token_id in by_first_char.get(first, ()):
            if suffix.startswith(token_map[token_id]):
                allowed.add(token_id)
    return allowed


def _finish_string(value: str, source: str) -> str:
    """Normalize a generated string value before returning it.

    Trims surrounding whitespace and, when the value matches the user
    request exactly once and directly follows a path separator, restores
    that separator (a request such as "...at /home/user/data.json" is the
    same path as the fragment "home/user/data.json").

    Args:
        value: Raw generated string value.
        source: User request text used as the substring anchor.

    Returns:
        Final string value.
    """
    result = value.strip()
    if not result or not source:
        return result
    pos = source.find(result)
    if pos <= 0:
        return result
    if source.find(result, pos + 1) != -1:
        return result
    if source[pos - 1] in "/\\":
        return source[pos - 1] + result
    return result


def generate_string_value(
    model: Any,
    context: str,
    max_tokens: int,
    token_map: Dict[int, str],
    source: str = "",
) -> str:
    """Generate a string parameter value token by token.

    Two guarantees are combined:

    - Valid JSON content: interior quotes are probed and kept as content,
      while the string closes the moment the most likely continuation after
      a quote is a JSON separator (``,`` or ``}``) or whitespace leading to
      one.
    - Faithful arguments: when ``source`` is provided, the partial value is
      additionally constrained to remain a contiguous span of the user
      request, so the model cannot silently drop quotes, duplicate
      backslashes or hallucinate argument text.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending with an opening '"'.
        max_tokens: Safety limit on the number of generated tokens.
        token_map: Pre-loaded id -> string vocabulary mapping.
        source: User request text used as the substring anchor.

    Returns:
        Generated string value.
    """
    value: str = ""
    constrained = False
    matches: List[int] = []
    if source:
        by_first_char = _by_first_char(token_map)

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + value)[0].tolist()
        )
        best_id = _argmax(logits)
        best_str = _token_str(model, best_id, token_map)

        if constrained:
            allowed = _allowed_token_ids(
                value, matches, source, token_map, by_first_char
            )
            if not allowed:
                return _finish_string(value, source)
            if best_id not in allowed:
                for token_id in range(len(logits)):
                    if token_id not in allowed:
                        logits[token_id] = float("-inf")
                best_id = _argmax(logits)
                best_str = _token_str(model, best_id, token_map)
            value += best_str

        elif '"' in best_str:
            remaining = best_str
            while '"' in remaining:
                idx = remaining.index('"')
                value += remaining[:idx]
                remaining = remaining[idx + 1:]
                probe_logits = model.get_logits_from_input_ids(
                    model.encode(context + value + '"')[0].tolist()
                )
                probe_id = _argmax(probe_logits)
                probe_str = _token_str(model, probe_id, token_map)
                trimmed = probe_str.strip()
                if (trimmed == "" or trimmed in (",", "}")
                        or trimmed.startswith(",") or trimmed.startswith("}")):
                    return _finish_string(value, source)
                value += '"'
            value += remaining

        else:
            value += best_str

        if constrained:
            matches = [pos for pos in matches
                    if source[pos:pos + len(value)] == value]
            if not matches:
                constrained = False
        elif source and value:
            found = _find_all(source, value)
            if found:
                constrained = True
                matches = found

    return _finish_string(value, source)


def _is_float(s: str) -> bool:
    """Check if a string is a complete valid float representation."""
    try:
        float(s)
        return True
    except (ValueError, OverflowError):
        return False


def _is_float_prefix(s: str) -> bool:
    """Check if a string can become a valid float with more digits."""
    if s == "":
        return True
    return _FLOAT_PREFIX_RE.fullmatch(s) is not None


def generate_number_value(
    model: Any,
    context: str,
    max_tokens: int,
    token_map: Dict[int, str],
) -> str:
    """Generate a numeric value with constrained decoding.

    Tokens are only accepted when they keep the accumulated text on a
    valid path toward a valid float. The number can only terminate on a
    JSON separator once it forms a complete float.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending before the number.
        max_tokens: Safety limit on the number of generated tokens.
        token_map: Pre-loaded id -> string vocabulary mapping.

    Returns:
        Generated number as a string.
    """
    number: str = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + number)[0].tolist()
        )
        for token_id in range(len(logits)):
            token_str = _token_str(
                model, token_id, token_map
            ).strip()
            if token_str == "":
                logits[token_id] = float("-inf")
            elif token_str == "," or token_str == "}":
                if not _is_float(number):
                    logits[token_id] = float("-inf")
            elif not _is_float_prefix(number + token_str):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str: str = _token_str(model, best_id, token_map).strip()
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
    max_tokens: int,
    token_map: Dict[int, str],
) -> str:
    """Generate a boolean value (true/false) with constrained decoding.

    Args:
        model: Small_LLM_Model instance.
        context: Prompt context ending before the boolean value.
        max_tokens: Safety limit on the number of generated tokens.
        token_map: Pre-loaded id -> string vocabulary mapping.

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
            token_str = _token_str(model, token_id, token_map)
            combined = generated + token_str
            if not any(
                val.startswith(combined)
                for val in valid_values
            ):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str = _token_str(model, best_id, token_map)

        generated += best_str

        if generated in valid_values:
            break

    return generated
