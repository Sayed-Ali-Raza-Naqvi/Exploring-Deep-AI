from langchain_community.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b


@tool
def divide(a: int, b: int) -> float:
    """Divides two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


user_input = input("Enter the operation you want to perform (multiply, add, subtract, divide): ")

operation = user_input.strip().lower()

if operation == "multiply":
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    result = multiply.invoke({"a": a, "b": b})
    print(f"Name: {multiply.name}")
    print(f"Description: {multiply.description}")
    print(f"Parameters: {multiply.args}")
elif operation == "add":
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    result = add.invoke({"a": a, "b": b})
    print(f"Name: {add.name}")
    print(f"Description: {add.description}")
    print(f"Parameters: {add.args}")
elif operation == "subtract":
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    result = subtract.invoke({"a": a, "b": b})
    print(f"Name: {subtract.name}")
    print(f"Description: {subtract.description}")
    print(f"Parameters: {subtract.args}")
elif operation == "divide":
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    result = divide.invoke({"a": a, "b": b})
    print(f"Name: {divide.name}")
    print(f"Description: {divide.description}")
    print(f"Parameters: {divide.args}")
else:
    print("Invalid operation.")
    exit()


class MathTookit:
    def get_tools(self):
        return [multiply, add, subtract, divide]
    

toolkit = MathTookit()

tools = toolkit.get_tools()

for tool in tools:
    print(f"Tool Name: {tool.name}")
    print(f"Tool Description: {tool.description}")
    print(f"Tool Parameters: {tool.args}")
    print("\n")