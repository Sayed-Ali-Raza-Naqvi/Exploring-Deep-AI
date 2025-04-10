from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="TP53 is a tumor suppressor gene that plays a critical role in preventing cancer formation. It regulates the cell cycle and can trigger apoptosis in response to DNA damage.",
    metadata={"pathway": "Cell Cycle Regulation"}
)

doc2 = Document(
    page_content="BRCA1 is involved in DNA repair and maintains genomic stability. Mutations in BRCA1 are linked to a higher risk of breast and ovarian cancers.",
    metadata={"pathway": "DNA Repair"}
)

doc3 = Document(
    page_content="EGFR is a receptor tyrosine kinase involved in cell proliferation and survival. Overexpression or mutation of EGFR is commonly observed in non-small cell lung cancers.",
    metadata={"pathway": "EGFR Signaling"}
)

doc4 = Document(
    page_content="AKT1 is a serine/threonine kinase that plays a key role in the PI3K/AKT signaling pathway. It promotes cell growth and survival, and is often hyperactivated in cancer.",
    metadata={"pathway": "PI3K-AKT Pathway"}
)

doc5 = Document(
    page_content="VEGFA is a signal protein that stimulates the formation of blood vessels. It is a key regulator in angiogenesis, especially in tumor development.",
    metadata={"pathway": "Angiogenesis"}
)


docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'
)

vector_store.add_documents(docs)

vector_store.get(include=['embeddings','documents', 'metadatas'])

vector_store.similarity_search(
    query='Which gene is involved in blood vessel formation?',
    k=2
)

vector_store.similarity_search_with_score(
    query='Which gene is involved in blood vessel formation?',
    k=2
)

vector_store.similarity_search_with_score(
    query='',
    filter={'pathway': 'Cell Cycle Regulation'}
)

updated_doc1 = Document(
    page_content="TP53, often referred to as the 'guardian of the genome', is a crucial tumor suppressor gene that monitors cellular stress and genomic integrity. It encodes the p53 protein, which becomes activated in response to DNA damage, oxidative stress, or oncogene activation. Once triggered, p53 can halt the cell cycle, initiate DNA repair mechanisms, or induce apoptosis to prevent the propagation of damaged cells. Loss or mutation of TP53 is one of the most common alterations in human cancers, contributing to uncontrolled cell division and tumor progression.",
    metadata={"pathway": "Cell Cycle Regulation"}
)

vector_store.update_document(document_id='', document=updated_doc1) # Add document_id to update the document

vector_store.delete(ids=['']) # Add document_id to delete the document