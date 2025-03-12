from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

class DatabaseFeatures(BaseModel):
    name: str = Field(description="Name of the database.")
    origin: str = Field(description="Origin of the database.")
    data_size: Optional[float] = Field(gt=0, description="Size of the database in GB (must be > 0).")
    applications: str = Field(description="Uses and applications of the database.")

parser = PydanticOutputParser(pydantic_object=DatabaseFeatures)

template = PromptTemplate(
    template="Provide name, origin, data size (in GB), and applications of the {database}. \n {format_instructions}",
    input_variables=["database"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# prompt = template.invoke({"database": "PDB"})

# result = model.invoke(prompt)

# parsed_result = parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({"database": "Swiss-prot"})

print(result)