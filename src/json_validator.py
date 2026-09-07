from pydantic import BaseModel


class Prompts(BaseModel):
    prompt: str


class ParametreType(BaseModel):
    type: str


class Functions(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametreType]
    returns: ParametreType
