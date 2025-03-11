from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

json_schema = {
    "title": "Gene",
    "description": "A gene in the human genome with key attributes related to its function and genomic location.",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A brief summary of the gene."
        },
        "function": {
            "type": "string",
            "description": "The function of the gene."
        },
        "location": {
            "type": "string",
            "description": "The location of the gene on the chromosome."
        },
        "associated_diseases": {
            "type": "array",
            "items": { "type": "string" },
            "description": "A list of diseases associated with this gene."
        },
        "expression_level": {
            "type": "number",
            "description": "The expression level of the gene in a given condition. Optional."
        },
        "mutation_types": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Types of mutations commonly found in this gene. Optional."
        },
        "gene_type": {
            "type": "string",
            "enum": ["oncogene", "tumor suppressor", "housekeeping"],
            "description": "The functional category of the gene."
        }
    },
    "required": ["summary", "function", "location", "associated_diseases", "gene_type"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""TP53 (Tumor Protein P53) is a tumor suppressor gene that regulates the cell cycle, apoptosis, and DNA repair. It prevents uncontrolled cell growth by activating repair mechanisms or inducing cell death when damage is detected. Mutations in TP53 are common in cancers, including HCC, leading to dysfunctional tumor suppression. It plays a key role in the p53 signaling pathway and the DNA damage response. Located on chromosome 17p13.1, it is often called the guardian of the genome.""")
print(result)