from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

report_prompt = PromptTemplate(
    template="Genertae a report on {topic}.",
    input_variables=["topic"]
)

summary_prompt  = PromptTemplate(
    template="Generate a 5 pointer summary from the following text: \n {text}.",
    input_variables=["text"]
)

chain = report_prompt | model | parser | summary_prompt | model | parser

result = chain.invoke({"topic": "AI in CRISPR technology"})

print(result)

chain.get_graph().print_ascii()