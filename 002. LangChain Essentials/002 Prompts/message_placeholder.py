from langchain_core.prompts import MessagePlaceholder, ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful AI biologist."),
    MessagePlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({"chat_history": chat_history, "query": "What is the meaning of life?"})
print(prompt)