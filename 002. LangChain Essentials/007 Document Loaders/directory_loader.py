from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader,
)

docs = loader.load()

print(docs[0].page_content)
print(docs[1].metadata)