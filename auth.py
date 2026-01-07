from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config import AGENT_ACCESS_TOKEN

auth_scheme = HTTPBearer()

def authenticate(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != AGENT_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials