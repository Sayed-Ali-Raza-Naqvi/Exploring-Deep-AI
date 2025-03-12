from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, origin, and features of a {protein database}. \n {format_instruction}",
    input_variables=["protein database"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# parsed_result = parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({"protein database": "PDB"})

print(result)