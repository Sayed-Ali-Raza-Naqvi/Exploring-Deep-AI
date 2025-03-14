from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

prompt = template = PromptTemplate(
    template="Explain {topic} in a few sentences.",
    input_variables=["topic"]
)

chain = prompt | model | parser

result = chain.invoke({"topic": "CRISPR"})

print(result)

chain.get_graph().print_ascii()