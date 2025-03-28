from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

prompt_desc = PromptTemplate(
    template="Write a short description about {topic}",
    input_variables=["topic"]
)

prompt_exp = PromptTemplate(
    template="Explain the following topic: \n{text}",
    input_variables=["text"]
)

desc_chain = RunnableSequence(prompt_desc, model, parser)

parallel_chain = RunnableParallel({
    "description": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt_exp, model, parser)
})

chain = RunnableSequence(desc_chain, parallel_chain)

result = chain.invoke({"topic": "AI in Cancer Research"})

print(result)