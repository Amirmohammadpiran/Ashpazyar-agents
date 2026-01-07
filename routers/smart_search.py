import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import authenticate
from config import CHROMA_ACCESS_TOKEN, CHROMA_SEARCH_URL
from agents.smart_search_agent import run_smart_search_agent

router = APIRouter(prefix="/smart-search", tags=["Smart Search"])


class SmartSearchRequest(BaseModel):
    text: str


@router.post("")
def smart_search(
    request: SmartSearchRequest,
    _=Depends(authenticate)
):
    try:
        chroma_query = run_smart_search_agent(request.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM failed: {e}")

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
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "parsed_query": chroma_query.model_dump(),
        "results": response.json()
    }
