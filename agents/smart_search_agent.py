from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import GOOGLE_API_KEY

SYSTEM_PROMPT = """
You are a query extraction engine.

Your task is to convert a Persian food-related user query into a JSON object
used to search a recipe database.

You MUST return valid JSON only.
DO NOT include explanations or text outside JSON.

JSON schema:
{{
  "query": string,
  "include_ingredients": string[],
  "limit": number
}}

Rules:
- If a food name is mentioned, put it in "query".
- If ingredients are mentioned, extract them as individual strings.
- If no ingredients are mentioned, return an empty list.
- If the user asks for many results, set limit accordingly; otherwise use 1.
- Do not hallucinate ingredients.
- Do not invent foods.
- Use Persian language exactly as provided by the user.

Return JSON only.
"""

class ChromaQuery(BaseModel):
    query: str
    include_ingredients: list[str] = Field(default_factory=list)
    limit: int = 1


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{item}")
    ]
)

parser = JsonOutputParser()
chain = prompt | model | parser


def run_smart_search_agent(text: str) -> ChromaQuery:
    extracted = chain.invoke({"item": text})
    return ChromaQuery(**extracted)
