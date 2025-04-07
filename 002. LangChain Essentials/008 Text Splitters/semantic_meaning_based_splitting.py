from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

text = """Artificial Intelligence (AI) is revolutionizing industries by 
enabling machines to perform tasks that typically require human 
intelligence. It plays a major role in data analysis, decision-making, 
and automation. Biotechnology, on the other hand, involves using living 
organisms or biological systems for technological advancements. It 
has led to breakthroughs in medicine, agriculture, and environmental 
solutions.

Bioinformatics is an interdisciplinary field that combines biology, 
computer science, and statistics to analyze and interpret biological 
data. It is especially vital in genomics and proteomics, helping 
researchers understand genetic information and discover new therapeutic 
targets.
"""

docs = text_splitter.create_documents([text])

print("Number of chunks:", len(docs))