import os
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# LangChain Imports

load_dotenv(override=True)

# --- Configuration ---
CHROMA_PATH = "./chroma_db"
CHROMA_SEARCH_URL = "http://0.0.0.0:8324/search"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_ACCESS_TOKEN = os.getenv("CHROMA_ACCESS_TOKEN")
AGENT_ACCESS_TOKEN = os.getenv("AGENT_ACCESS_TOKEN")
SYSTEM_PROMPT = """
You are a query extraction engine.

Your task is to convert a Persian food-related user query into a JSON object
used to search a recipe database.

You MUST return valid JSON only.
DO NOT include explanations or text outside JSON.

JSON schema:
{{
  "query": string,                 // name of the food or main dish
  "include_ingredients": string[], // ingredient keywords (optional, may be empty)
  "limit": number                  // integer, default 1
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


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.environ["GOOGLE_API_KEY"]
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{item}")
    ]
)

parser = JsonOutputParser()

chain = prompt | model | parser


app = FastAPI(title="Smart Recipe Search Agent")


class SmartSearchRequest(BaseModel):
    text: str


class ChromaQuery(BaseModel):
    query: str
    include_ingredients: list[str] = Field(default_factory=list)
    limit: int = 1


auth_scheme = HTTPBearer()
def authenticate(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != AGENT_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


@app.post("/smart-search")
def smart_search(request: SmartSearchRequest):
    try:
        # 1. Run LLM extraction
        extracted = chain.invoke({"item": request.text})

        # 2. Validate extracted JSON
        chroma_query = ChromaQuery(**extracted)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"LLM extraction failed: {e}"
        )

    # 3. Call Chroma DB search tool
    headers = {
        "Authorization": f"Bearer {CHROMA_ACCESS_TOKEN}",
        "Accept": "*/*"
    }

    try:
        response = requests.post(
            CHROMA_SEARCH_URL,
            headers=headers,
            json=chroma_query.model_dump(),
            timeout=10
        )
        response.raise_for_status()
        response.encoding = "utf-8"

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Chroma search failed: {e}"
        )

    # 4. Return combined response
    return {
        "parsed_query": chroma_query.model_dump(),
        "results": response.json()
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)