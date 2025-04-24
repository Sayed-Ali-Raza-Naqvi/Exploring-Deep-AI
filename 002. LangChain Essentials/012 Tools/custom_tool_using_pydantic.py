from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class AdditionInput(BaseModel):
    """Input for addition tool."""
    a: int = Field(required=True, description="First number to add")
    b: int = Field(required=True, description="Second number to add")


def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


addition_tool = StructuredTool.from_function(
    func=add_numbers,
    name="addition_tool",
    description="A tool to add two numbers.",
    args_schema=AdditionInput,
    return_type=int,
)


result = addition_tool.invoke({"a": 5, "b": 3})
print(result)