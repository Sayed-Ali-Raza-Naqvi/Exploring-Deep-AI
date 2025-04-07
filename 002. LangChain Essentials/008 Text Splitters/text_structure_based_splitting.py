from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Cancer is a group of diseases characterized by uncontrolled 
cell growth and the ability to invade or spread to other parts of the 
body. These abnormal cells can form tumors, disrupt normal bodily 
functions, and, if untreated, become life-threatening.

There are many types of cancer, including breast, lung, 
liver, and colon cancer, each with distinct causes and symptoms.
Risk factors include genetics, lifestyle choices, environmental 
exposures, and certain infections.

Early detection and treatment significantly improve survival rates 
and quality of life for patients. Advances in medical research continue 
to offer new therapies, from targeted treatments to immunotherapy, 
bringing hope to millions worldwide."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))