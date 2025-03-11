from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Gene(TypedDict):
    summary: Annotated[str, "A brief summary of the gene."]
    function: Annotated[str, "The function of the gene."]
    location: Annotated[str, "The location of the gene on chromosome."]

structured_model = model.with_structured_output(Gene)

result = structured_model.invoke("""TP53 (Tumor Protein P53) is a tumor suppressor gene that regulates the cell cycle, apoptosis, and DNA repair. It prevents uncontrolled cell growth by activating repair mechanisms or inducing cell death when damage is detected. Mutations in TP53 are common in cancers, including HCC, leading to dysfunctional tumor suppression. It plays a key role in the p53 signaling pathway and the DNA damage response. Located on chromosome 17p13.1, it is often called the guardian of the genome.""")
print(result)