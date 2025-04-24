from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool


## -------------------------------------------------------------

search_tool = DuckDuckGoSearchRun()

search_input = input("Enter the query you want to search: ")

search_result = search_tool.invoke(search_input)
print(f"Search Result\n{search_result}")

## -------------------------------------------------------------

shell_tool = ShellTool()

shell_input = input("Enter the shell command you want to run: ")

results = shell_tool.invoke(shell_input)
print(f"Shell Command Result\n{results}")

## -------------------------------------------------------------


## --------------------------------------------------------------

