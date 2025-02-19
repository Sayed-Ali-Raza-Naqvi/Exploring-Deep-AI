from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

docs = [
    "Islamabad is the capital city of Pakistan.",
    "Karachi is the largest city in Pakistan.",
    "Pakistan is a country in South Asia.",
    "Pakistan has a population of over 220 million people.",
    "Pakistan has a diverse culture.",
]

query = "Tell me what is the largest city of Pakistan."

doc_embeddings = embeddings.embed_documents(docs)
query_embedding = embeddings.embed_query(query)

similarity = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(similarity)), key=lambda x: x[1])[-1]

print(docs[index])
print(f"Similairty score: {score}")