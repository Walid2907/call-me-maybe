import json
import sys
from time import time
from pathlib import Path
from typing import Dict, List

from llm_sdk.llm_sdk import Small_LLM_Model

from .Constrained_decoding import (
    build_token_map,
    system_prompt_builder,
    generate_function_name,
    generate_string_value,
    generate_number_value,
    generate_boolean_value,
)
from .json_validator import Functions
from .loader import prompts_loader, function_loader
from .parser import argparser


def _get_function_by_name(
    functions: List[Functions], name: str
) -> Functions:
    """Retrieve a function definition by name.

    Args:
        functions: List of Functions pydantic models.
        name: Function name to look up.

    Returns:
        The matching Functions model.

    Raises:
        ValueError: If function name is not found.
    """
    for f in functions:
        if f.name == name:
            return f
    raise ValueError(
        f"Function '{name}' not found in definitions."
    )


def _parse_number(raw: str, is_integer: bool) -> object:
    """Safely convert a generated numeric string to a value.

    Args:
        raw: The numeric string produced by constrained decoding.
        is_integer: True to coerce the value to an integer.

    Returns:
        The numeric value, or a safe fallback if parsing fails.
    """
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return 0 if is_integer else 0.0
    if is_integer:
        return int(value)
    return value


def main() -> None:
    """Load data, run constrained decoding, write output."""
    parser = argparser()

    try:
        functions = function_loader(
            parser.functions_definition
        )
    except FileNotFoundError:
        print(f"Error: function definitions file not found: "
              f"{parser.functions_definition}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in function definitions "
              f"({parser.functions_definition}): {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: invalid function definitions: {e}")
        sys.exit(1)
    if not functions:
        print("Error: no function definitions found.")
        sys.exit(1)

    try:
        prompts = prompts_loader(parser.input)
    except FileNotFoundError:
        print(f"Error: input file not found: {parser.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in input file "
              f"({parser.input}): {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: invalid input prompts: {e}")
        sys.exit(1)
    if not prompts:
        print("Error: no input prompts found.")
        sys.exit(1)

    prompt_system = system_prompt_builder(functions)
    function_names = [f.name for f in functions]

    try:
        model = Small_LLM_Model(parser.model)
    except OSError as e:
        print(f"Error: failed to load model "
              f"'{parser.model}': {e}")
        sys.exit(1)

    token_map: Dict[int, str] = build_token_map(model)

    start_time = time()
    results = []
    for p in prompts:
        prompt = p.prompt
        if not prompt:
            print("Skipping empty prompt.")
            results.append({
                "prompt": prompt,
                "name": "",
                "parameters": {},
            })
            continue
        prompt_len = max(len(prompt), 16)
        context = (
            f"{prompt_system}\n"
            f'{{"prompt": "{prompt}",'
            f'"name": "'
        )
        print(f"Processing: {prompt}")

        try:
            func_name = generate_function_name(
                model, context, function_names, token_map
            )
            func_def = _get_function_by_name(
                functions, func_name
            )

            context += (
                f'{func_name}", "parameters": {{'
            )

            parameters: Dict[str, object] = {}
            param_items = list(
                func_def.parameters.items()
            )
            for i, (param_name, param_info) in enumerate(
                param_items
            ):
                context += f'"{param_name}": '
                param_type = param_info.type

                if param_type == "string":
                    context += '"'
                    val = generate_string_value(
                        model, context, prompt_len, token_map, prompt
                    )
                    context += val + '"'
                    parameters[param_name] = val
                elif param_type == "number":
                    val = generate_number_value(
                        model, context, prompt_len, token_map
                    )
                    context += val
                    parameters[param_name] = _parse_number(
                        val, is_integer=False
                    )
                elif param_type == "integer":
                    val = generate_number_value(
                        model, context, prompt_len, token_map
                    )
                    context += val
                    parameters[param_name] = _parse_number(
                        val, is_integer=True
                    )
                elif param_type == "boolean":
                    val = generate_boolean_value(
                        model, context, prompt_len, token_map
                    )
                    context += val
                    parameters[param_name] = (
                        val == "true"
                    )
                else:
                    context += '"'
                    val = generate_string_value(
                        model, context, prompt_len, token_map, prompt
                    )
                    context += val + '"'
                    parameters[param_name] = val

                if i < len(param_items) - 1:
                    context += ", "

            context += "}}"

            results.append({
                "prompt": prompt,
                "name": func_name,
                "parameters": parameters,
            })
            print(
                f"  -> {func_name}({parameters})"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "prompt": prompt,
                "name": "",
                "parameters": {},
            })
    end_time = time()

    output_path = Path(parser.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    time_cost = end_time - start_time
    print(f"Time: {int(time_cost / 60)}:{int(time_cost % 60)}m")


if __name__ == "__main__":
    main()
