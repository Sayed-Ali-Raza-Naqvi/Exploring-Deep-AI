from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.schema import Document

documents = [
    Document(
        page_content=(
            "Gene expression analysis reveals patterns of upregulated and downregulated genes in cancer tissues.\n"
            "Pathway enrichment tools identify biological processes affected by these gene expression changes.\n"
            "Machine learning models can then use this information to classify tumor subtypes and predict patient outcomes."
        ),
        metadata={"source": "Integrated Cancer Genomics Study"}
    ),
    Document(
        page_content=(
            "Next-generation sequencing technologies enable rapid detection of somatic mutations in cancer genomes.\n"
            "Copy number variation and structural variant analysis provide insights into genomic instability in tumors.\n"
            "These data support the discovery of novel biomarkers and therapeutic targets in oncology research."
        ),
        metadata={"source": "NGS Applications in Cancer"}
    ),
    Document(
        page_content=(
            "Bioinformatics workflows integrate multi-omics datasets including genomics, transcriptomics, and proteomics.\n"
            "Network analysis uncovers gene and protein interactions that drive disease progression.\n"
            "Integrative pipelines aid in identifying mechanisms of drug resistance and new therapeutic strategies."
        ),
        metadata={"source": "Systems Biology and Bioinformatics Review"}
    )
]

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-3.5-turbo")

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

query = "How can gene expression analysis and multi-omics integration be used to identify cancer biomarkers and predict patient outcomes?"

results = compression_retriever.invoke(query)

for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"Page Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print()