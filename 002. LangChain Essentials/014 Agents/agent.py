import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Get the current weather data for a given city using OpenWeatherMap API."""

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('')}"
    response = requests.get(url)

    return response.json()


llm = ChatOpenAI()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

response = agent_executor.invoke({"input": "What are the best travel destinations in Pakistan? Also tell me the current weather in Islamabad."})
print(response['output'])