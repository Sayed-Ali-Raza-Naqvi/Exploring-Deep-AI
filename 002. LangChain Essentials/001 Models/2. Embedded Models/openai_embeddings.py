from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embeddings.embed_query("What is the capital city of Pakistan?")

print(str(result))

docs = [
    "Islamabad is the capital city of Pakistan.",
    "Karachi is the largest city in Pakistan.",
    "Pakistan is a country in South Asia.",
    "Pakistan has a population of over 220 million people.",
    "Pakistan has a diverse culture."
]

result = embeddings.embed_documents(docs)

print(str(result))