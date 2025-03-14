from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Feedback sentiment")


pydantic_parser = PydanticOutputParser(Feedback)

prompt_for_classification = PromptTemplate(
    template="Classify the sentiment of the following feedback text into poisitve or negative: \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt_for_classification | model | parser

pos_prompt = PromptTemplate(
    template="Write an appropriate response for this positove feedback: \n {feedback}",
    input_variables=["feedback"]
)

neg_prompt = PromptTemplate(
    template="Write an appropriate response for this negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", pos_prompt | model | parser),
    (lambda x: x["sentiment"] == "negative", neg_prompt | model | parser),
    RunnableLambda(lambda x: "Unable to find sentiment.")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "Powerful, interdisciplinary, data-driven, evolving, impactful."})

print(result)