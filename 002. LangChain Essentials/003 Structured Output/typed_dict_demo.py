from typing import TypedDict, List

class GeneExpression(TypedDict):
    gene_id: str
    gene_name: str
    expression_values: List[float]
    condition: str

sample_gene: GeneExpression = {
    'gene_id': 'ENSG00000141510',
    'gene_name': 'TP53',
    'expression_values': [2.5, 3.1, 4.0, 2.8],  
    'condition': 'disease'
}

print(sample_gene)