from llm_sdk.llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel


class Parametres_type(BaseModel):
    type: str


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parametres_type]
    returns: Parametres_type



def load_functions(path: str) -> list(Function):
    data = []
    with open(path, "r") as f:
        functions = json.load(f)
    for func in functions:
        data.append(Function(**func))
    return data        
    

def get_functions_names(functions: list[Function]) -> list[str]:
    names = []
    for f in functions:
        names.append(f.name)
    return names

def build_prompt(functions: list[Function]) -> str:
    prompt = [
        "you are text to function call system, given this functions"
    ]
    for f in functions:
        params = []
        for p, typ in f.parameters.items():
            params.append(f'"{p}": {{"type": "{typ.type}"}}')
        params_str = ", ".join(params)
        prompt.append(
            f'{{"name": "{f.name}", '
            f'"description": "{f.description}", '
            f'"parameters": {{{params_str}}}}}'
        )
    prompt.append(
        "choose the right function and it's parametres based on the user input and following those rules:"
        "generate only valide json"

    )
    return "\n".join(prompt)

def _argmax(logits: List[float]) -> int:
    """Return the index of the maximum value."""
    best_idx = 0
    best_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_val:
            best_val = logits[i]
            best_idx = i
    return best_idx

def generate_function_name(llm: Small_LLM_Model, prompt: str, system_promt: str, function_names: list[str]) -> str:
    generated = ""
    max_tocken = 1000
    for _ in range(max_tocken):
        logits = llm.get_logits_from_input_ids(llm.encode(system_promt + "\nUser request: " + prompt + "\nFunction: " + generated)[0].tolist())
        for token_id in range(len(logits)):
            token_str = llm.decode([token_id])
            combined = generated + token_str
            if not any(fn.startswith(combined) for fn in function_names):
                logits[token_id] = float("-inf")
        best_id = _argmax(logits)
        best_str: str = llm.decode([best_id])

        generated += best_str
        print(generated)

        if generated in function_names:
            break
    return generated




llm = Small_LLM_Model()
functions = load_functions("/home/zues/Projects/call-me-maybe/data/input/functions_definition.json")
# for f in functions:
    # print(f)
functions_name = get_functions_names(functions)
sys_prompt = build_prompt(functions)
print(generate_function_name(llm, "What is the sum of 2 and 3?", sys_prompt, functions_name))