import json
from time import time
from pathlib import Path
from typing import List

from llm_sdk.llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .Constrained_decoding import (
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
        f"Function '{name}' not found in definitions.")


def main() -> None:
    """Load data, run constrained decoding, write output."""
    parser = argparser()

    try:
        functions = function_loader(parser.functions_definition)
    except (FileNotFoundError, json.JSONDecodeError,
            ValueError) as e:
        raise RuntimeError(
            f"Failed to load function definitions: {e}")
    if not functions:
        raise RuntimeError("No function definitions found.")

    try:
        prompts = prompts_loader(parser.input)
    except (FileNotFoundError, json.JSONDecodeError,
            ValueError) as e:
        raise RuntimeError(f"Failed to load prompts: {e}")
    if not prompts:
        raise RuntimeError("No input prompts found.")
# build the prompte to pass to the llm
    prompt_system = system_prompt_builder(functions)
# function names list
    function_names = [f.name for f in functions]
    try:
        model = Small_LLM_Model(parser.model)
    except OSError:
        raise RuntimeError(f"Model {parser.model} not found "
                           f"or failed to download.")
    start_time = time()
    results = []
    for p in prompts:
        prompt = p.prompt
        if not prompt:
            print("Invalid prompt!")
            continue
        prompt_len = len(prompt) * 50
        context = (f"{prompt_system}\n"
                   f'{{"prompt": "{prompt}",'
                   f'"name": "')
        print(f"Processing: {prompt}")

        try:
            # generate function name and then check if
            # it's a valid name
            func_name = generate_function_name(
                model, context, function_names)
            func_def = _get_function_by_name(
                functions, func_name)
            # we append it to the context string
            # before starting to get the params
            context += (f'{func_name}", "parameters": {{')
            parameters: dict[str, object] = {}
            param_items = list(func_def.parameters.items())
            for i, (param_name, param_info) in enumerate(param_items):
                context += f'"{param_name}": '
                param_type = param_info.type

                if param_type == "string":
                    context += '"'
                    val = generate_string_value(model, context, prompt_len)
                    context += val + '"'
                    parameters[param_name] = val
                elif param_type == "number":
                    val = generate_number_value(
                        model, context, prompt_len
                    )
                    context += val
                    parameters[param_name] = float(val)
                elif param_type == "integer":
                    val = generate_number_value(
                        model, context, prompt_len
                    )
                    context += val
                    parameters[param_name] = int(
                        float(val)
                    )
                elif param_type == "boolean":
                    val = generate_boolean_value(
                        model, context, prompt_len
                    )
                    context += val
                    parameters[param_name] = (
                        val == "true"
                    )
                else:
                    val = generate_string_value(
                        model, context, prompt_len
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
    print(f"Time: {int(time_cost / 60)}minute {int(time_cost % 60)}second")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)
