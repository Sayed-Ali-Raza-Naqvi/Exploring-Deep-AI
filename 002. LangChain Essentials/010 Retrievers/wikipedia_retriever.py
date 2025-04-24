from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "What is the origin of Bioinformatics?"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"----- Result {i + 1} -----")
    print(f"Title: {doc.metadata['title']}")
    print(f"Page Content: {doc.page_content}")