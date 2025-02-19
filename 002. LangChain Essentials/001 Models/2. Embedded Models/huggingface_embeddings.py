from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital city of Pakistan?"

result = embeddings.embed_query(text)

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