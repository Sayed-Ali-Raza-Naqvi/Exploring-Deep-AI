from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

documents = [
    Document(page_content="Gene expression profiling has become a pivotal tool in cancer biomarker discovery and precision oncology."),
    Document(page_content="Next-generation sequencing technologies enable comprehensive analysis of somatic mutations in cancer genomes."),
    Document(page_content="Bioinformatics approaches facilitate pathway enrichment analysis to identify dysregulated biological processes in tumors."),
    Document(page_content="Integrating multi-omics data enhances the understanding of cancer heterogeneity and therapeutic resistance mechanisms.")
]

embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    documents=documents, 
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "lambda_mult": 0.5},
    search_type="mmr",
)

query = "How is gene expression and multi-omics data used to discover cancer biomarkers and understand tumor heterogeneity?"

results = retriever.invoke(query)

for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"Page Content: {result.page_content}")
    