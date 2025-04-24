from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever

documents = [
    Document(page_content="Gene expression profiling has become a pivotal tool in cancer biomarker discovery and precision oncology.",
            metadata={"source": "Cancer Genomics Review"}),
    Document(page_content="Next-generation sequencing technologies enable comprehensive analysis of somatic mutations in cancer genomes.",
            metadata={"source": "NGS in Oncology"}),
    Document(page_content="Bioinformatics approaches facilitate pathway enrichment analysis to identify dysregulated biological processes in tumors.",
            metadata={"source": "Bioinformatics for Cancer"}),
    Document(page_content="Integrating multi-omics data enhances the understanding of cancer heterogeneity and therapeutic resistance mechanisms.",
            metadata={"source": "Multi-Omics Integration Study"}),
    Document(page_content="Machine learning algorithms are increasingly applied to classify tumor subtypes based on genomic and transcriptomic profiles.",
            metadata={"source": "ML in Cancer Research"}),
    Document(page_content="Single-cell RNA sequencing reveals the cellular diversity within tumors and uncovers rare malignant cell populations.",
            metadata={"source": "scRNA-seq Cancer Study"}),
    Document(page_content="Public repositories like GEO and TCGA provide extensive cancer genomics datasets for research and clinical validation.",
            metadata={"source": "Cancer Data Resources"}),
    Document(page_content="Copy number variation analysis helps detect genomic amplifications and deletions associated with cancer progression.",
            metadata={"source": "CNV in Oncology"}),
    Document(page_content="Gene set enrichment analysis identifies significantly overrepresented pathways in differentially expressed genes.",
            metadata={"source": "GSEA Methods Paper"}),
    Document(page_content="Integrative bioinformatics pipelines combine sequencing, clinical, and imaging data to support personalized cancer therapy decisions.",
            metadata={"source": "Bioinformatics Pipelines in Precision Medicine"})
]

embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    documents=documents, 
    embedding=embedding_model
)

retreiver = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
)

query = "How are gene expression profiling and bioinformatics approaches used to analyze cancer genomics data and identify therapeutic targets?"

results = retreiver.invoke(query)

for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"Page Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print()