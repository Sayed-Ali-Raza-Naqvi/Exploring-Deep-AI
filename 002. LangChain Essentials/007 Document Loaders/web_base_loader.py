from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the following question:\n{question} from the following text:\n{text}",
    input_variables=["question", "text"],
)

url = "https://www.bioinformatics.org/"

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question": "What is bioinformatics?", "text": docs[0].page_content})

print(result)