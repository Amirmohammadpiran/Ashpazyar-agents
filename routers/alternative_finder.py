import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import authenticate
from config import CHROMA_ACCESS_TOKEN, CHROMA_SEARCH_URL
from agents.alternative_finder_agent import run_alternative_finder_agent

router = APIRouter(prefix="/replace-ingredient", tags=["Finds alternatives for ingredients"])


class findAlternativeRequest(BaseModel):
    text: str


@router.post("")
def smart_search(
    request: findAlternativeRequest,
    _=Depends(authenticate)
):
    try:
        response = run_alternative_finder_agent(request.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM failed: {e}")

    return response
