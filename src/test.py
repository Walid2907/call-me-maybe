# load function from the function def
from llm_sdk.llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel


class Params(BaseModel):
    type: str

class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Params]
    returns: Params

class Prompt(BaseModel):
    prompt: str

def load_functions(path: str) -> list[Function]:
    with open(path, "r") as functions:
        data = json.load(functions)
    results = []
    for function in data:
        results.append(Function(**function))
    return results

def load_prompts(path: str) -> list[Prompt]:
    with open(path, "r") as prompts:
        data = json.load(prompts)
    results = []
    for prompt in data:
        results.append(Prompt(**prompt))
    return results


def get_funcions_name(fucntions: list[Function]) -> list[str]:
    results = []
    for f in functions:
        results.append(f.name)
    return results

def prompt_builder(functions: list[Function]) -> str:
    lines = [
        "you are an text to function call system Given this functions"
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
    return lines
functions = load_functions("/home/wkerdad/Desktop/call-me-maybe/data/input/functions_definition.json")
functions_name = get_funcions_name(functions)
promtpts = load_prompts("/home/wkerdad/Desktop/call-me-maybe/data/input/function_calling_tests.json")
sys_prompt = prompt_builder(functions)
llm = Small_LLM_Model()
prompt = (
    f"{sys_prompt} \n"
    f'{{"prompt": "What is the sum of 2 and 3?", '
    f'"name": "'
)
max_tokens = 1
generated = ""
for _ in range(max_tokens):
    logits = llm.get_logits_from_input_ids(llm.encode(prompt + generated)[0].tolist())
    print(logits)
# for l in prompt:
    # print(l)

