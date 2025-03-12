from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

schema = [
    ResponseSchema(name="feature_1", description="Feature 1 of the database."),
    ResponseSchema(name="feature_2", description="Feature 2 of the database."),
    ResponseSchema(name="feature_3", description="Feature 3 of the database."),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Provide 3 features of the {database}. \n {format_instructions}",
    input_variables=["database"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# prompt = template.invoke({"database": "PDB"})

# result = model.invoke(prompt)

# parsed_result = parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({"database": "Swiss-prot"})

print(result)