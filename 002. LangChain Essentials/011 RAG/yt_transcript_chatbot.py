from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

input_id = input("Enter the YouTube video ID: ")

video_id = input_id.strip()

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    transcript = " ".join([text["text"] for text in transcript_list])
    print("Transcript retrieved successfully.")

except TranscriptsDisabled:
    print("Transcripts are not available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.create_documents([transcript])

print(f"Number of chunks: {len(chunks)}")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(chunks, embedding_model)

print("Vector store created successfully.")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

prompt_template = PromptTemplate(
    template="""
        You are a helpful assistant. Answer the question only from the provided transcrpt context.
        If the answer is not in the context, say "I don't know".
        
        Context: {context}
        Question: {question}
    """,
    input_variables=["context", "question"],
)

def format_docs(retreived_docs):
    context_text = "\n\n".join([doc.page_content for doc in retreived_docs])
    return context_text


parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
})

parser = StrOutputParser()

chain = parallel_chain | prompt_template | llm | parser

print("Chain created successfully.")

while True:
    question = input("Enter your question: ")

    if question == "exit":
        break
    
    result = chain.invoke({"question": question})
    
    print(result)

print("Goodbye!")