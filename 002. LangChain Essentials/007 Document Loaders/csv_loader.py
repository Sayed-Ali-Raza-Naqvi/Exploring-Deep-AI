from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="gene_expression.csv")

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)