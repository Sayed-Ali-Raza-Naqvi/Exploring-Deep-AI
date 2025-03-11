from pydantic import BaseModel
from typing import Optional

class Gene(BaseModel):
    name: str
    function: str
    location: Optional[str] = None

gene = {
    "name": "TP53",
    "function": "tumor suppressor gene",
    "location": "chromosome 17p13.1"
}

genes = Gene(**gene)

genes_dict = dict(genes)
genes_json = genes.model_dump_json()

print(genes)