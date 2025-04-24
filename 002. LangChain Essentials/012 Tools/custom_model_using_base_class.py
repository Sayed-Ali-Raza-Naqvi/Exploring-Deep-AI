from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class AddToolInput(BaseModel):
    a: int = Field(required=True, description="The first number to add.")
    b: int = Field(required=True, description="The second number to add.")


class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers together."
    
    args_schema: Type[BaseModel] = AddToolInput

    def _run(self, a: int, b: int) -> int:
        return a + b

    async def _arun(self, a: int, b: int) -> int:
        return a + b
    

add_tool = AddTool()

result = add_tool.invoke({"a": 1, "b": 2})
print(result)