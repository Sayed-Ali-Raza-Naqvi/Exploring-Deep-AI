from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

documents = [
    Document(page_content="Gene expression profiling has become a pivotal tool in cancer biomarker discovery and precision oncology."),
    Document(page_content="Next-generation sequencing technologies enable comprehensive analysis of somatic mutations in cancer genomes."),
    Document(page_content="Bioinformatics approaches facilitate pathway enrichment analysis to identify dysregulated biological processes in tumors."),
    Document(page_content="Integrating multi-omics data enhances the understanding of cancer heterogeneity and therapeutic resistance mechanisms.")
]

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents, 
    embedding=embedding_model,
    collection_name="cancer_research",
)

retreiver = vector_store.as_retriever(search_kwargs={"k": 2})

query = "How is bioinformatics used to identify biomarkers and analyze gene expression in cancer research?"

results = retreiver.invoke(query)

for i, doc in enumerate(results):
    print(f"Document {i + 1}:")
    print(doc.page_content)
    print("\n")