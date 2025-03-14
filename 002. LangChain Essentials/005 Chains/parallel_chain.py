from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

text = """Artificial Intelligence (AI) has revolutionized molecular 
docking, a critical technique in drug discovery that predicts how small 
molecules interact with target proteins. Traditional docking methods 
rely on physics-based scoring functions and exhaustive conformational 
searches, which can be computationally expensive and time-consuming. 
AI-powered approaches, particularly deep learning models, have enhanced docking 
accuracy by predicting binding affinities and optimizing molecular poses with 
greater efficiency. Machine learning algorithms trained on vast chemical datasets 
can rapidly screen billions of compounds, identifying potential drug candidates in a 
fraction of the time required by conventional methods.

Recent advancements in AI-driven docking have incorporated generative models and 
reinforcement learning to design novel molecules with improved binding properties. 
Neural networks can learn from existing molecular interactions to suggest optimized 
ligand structures, accelerating lead optimization. Additionally, AI-powered docking 
integrates seamlessly with molecular dynamics simulations, refining binding predictions 
by accounting for protein flexibility. These innovations are transforming drug discovery 
pipelines, enabling researchers to explore chemical space more effectively and develop 
targeted therapies with higher success rates."""

model_for_notes = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
model_for_quiz = ChatAnthropic(model="claude-3.5-sonnet-20241022")
final_model = ChatOpenAI(model="gpt-4")

parser = StrOutputParser()

prompt_for_notes = PromptTemplate(
    template="Generate short and simple notes from the following text: \n {text}",
    input_variables=["text"]
)

prompt_for_quiz = PromptTemplate(
    template="Genrate 5 short question answer quiz from the follwoing text: \n {text}",
    input_variables=["text"]
)

final_prompt = PromptTemplate(
    template="Merge the provided notes and quiz questions and answers to create a single document: \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parallel_chain = RunnableParallel({
    "notes": prompt_for_notes | model_for_notes | parser,
    "quiz": prompt_for_quiz | model_for_quiz | parser
})

merge_chain = final_prompt | final_model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"text": text})

print(result)