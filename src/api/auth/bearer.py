from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

from src.config.env import SECRET_KEY

bearer_scheme = HTTPBearer()


async def auth_secret_key(bearer_auth: Optional[str] = Depends(bearer_scheme)) -> None:
    if not bearer_auth or bearer_auth.credentials != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
