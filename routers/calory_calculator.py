import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import authenticate
from config import CHROMA_ACCESS_TOKEN, CHROMA_SEARCH_URL
from agents.calory_calculator_agent import run_calory_calculator_agent

router = APIRouter(prefix="/calculate-calory", tags=["Estimates calory based on ingredients"])


class findAlternativeRequest(BaseModel):
    text: str


@router.post("")
def smart_search(
    request: findAlternativeRequest,
    _=Depends(authenticate)
):
    try:
        response = run_calory_calculator_agent(request.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM failed: {e}")

    return response
