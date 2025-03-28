from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

prompt_tweet = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"],
)

prompt_post = PromptTemplate(
    template="Write a LinkedIn post about {topic}",
    input_variables=["topic"],
)

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt_tweet, model, parser),
    "post": RunnableSequence(prompt_post, model, parser)
})

result = parallel_chain.invoke({"topic": "AI in cancer research"})

print(result)