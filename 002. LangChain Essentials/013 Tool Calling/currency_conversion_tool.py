import json
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Get the conversion factor from base_currency to target_currency.
    """
    API_KEY = ""

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()


@tool
def convert(base_currency: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Convert amount from base_currency to target_currency.
    """
    return base_currency * conversion_rate


llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

user_query = input("Enter your query: ")
query = HumanMessage(content=user_query)

messages = [query]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        tool_message_conversion_factor = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message_conversion_factor.content["conversion_rate"])
        messages.append(tool_message_conversion_factor)
    
    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message_convert = convert.invoke(tool_call)
        messages.append(tool_message_convert)

result = llm_with_tools.invoke(messages).content

print(result)