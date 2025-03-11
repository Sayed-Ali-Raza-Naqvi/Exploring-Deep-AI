from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Gene(BaseModel):
    summary: str = Field(description="A brief summary of the gene.")
    function: str = Field(description="The function of the gene.")
    location: str = Field(description="The location of the gene on the chromosome.")
    associated_diseases: List[str] = Field(description="A list of diseases associated with this gene.")
    expression_level: Optional[float] = Field(None, description="The expression level of the gene in a given condition. Optional.")
    mutation_types: Optional[List[str]] = Field(None, description="Types of mutations commonly found in this gene. Optional.")
    gene_type: Literal["oncogene", "tumor suppressor", "housekeeping"] = Field(description="The functional category of the gene.")

structured_model = model.with_structured_output(Gene)

result = structured_model.invoke("""TP53 (Tumor Protein P53) is a tumor suppressor gene that regulates the cell cycle, apoptosis, and DNA repair. It prevents uncontrolled cell growth by activating repair mechanisms or inducing cell death when damage is detected. Mutations in TP53 are common in cancers, including HCC, leading to dysfunctional tumor suppression. It plays a key role in the p53 signaling pathway and the DNA damage response. Located on chromosome 17p13.1, it is often called the guardian of the genome.""")
print(result)