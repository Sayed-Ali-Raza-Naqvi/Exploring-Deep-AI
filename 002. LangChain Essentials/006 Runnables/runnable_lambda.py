from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a short description (2-3 lines) about {topic}",
    input_variables=["topic"]
)

desc_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "description": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split()))
})

chain = RunnableSequence(desc_chain, parallel_chain)

result = chain.invoke({"topic": "AI in cancer research"})

print(result)