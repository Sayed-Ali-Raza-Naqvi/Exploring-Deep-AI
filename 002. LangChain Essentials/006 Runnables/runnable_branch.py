from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

prompt_report = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt_word_limit = PromptTemplate(
    template="Summarize the following text: \n {text}",
    input_variables=["text"]
)

chain_report = RunnableSequence(prompt_report, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt_word_limit, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(chain_report, branch_chain)

output = chain.invoke({"topic": "The history Bioinformatics"})

print(output)