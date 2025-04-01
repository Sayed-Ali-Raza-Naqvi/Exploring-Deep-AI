from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

loader = PyPDFLoader(r"D:\Projects\LangChain Projects\007 Document Loaders\AI in Bioinformatics.pdf")

docs = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the following text in 5 bullet points:\n\n{input}",
    input_variables=["input"],
)

chain = prompt | model | parser

result = chain.invoke({"input": docs[0].page_content})

print(result)