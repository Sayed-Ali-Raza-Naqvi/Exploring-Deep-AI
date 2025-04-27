from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([multiply])

user_query = input("Enter your query: ")
query = HumanMessage(content=user_query)

messages = [query]

result = llm_with_tools.invoke(messages)

messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])

messages.append(tool_result)

llm_with_tools.invoke(messages).content