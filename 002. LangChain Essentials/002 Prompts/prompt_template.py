from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI()

template2 = PromptTemplate(
    template="Tell the description in 5 lines about {topic}.",,
    input_variables=["topic"]
)

prompt = template2.invoke({"topic": "Bioinformatics"})

result = model.invoke(prompt)

print(result.content)