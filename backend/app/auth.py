import os
import logging

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_AUDIENCE = "authenticated"

# auto_error=False so we can return a clean 401 instead of FastAPI's default 403.
bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing authentication token",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Validate the Supabase access token and return the authenticated user id.

    The user id is taken from the verified JWT's `sub` claim, never from the
    request body, so callers cannot act on behalf of another user.
    """
    if credentials is None or not credentials.credentials:
        raise _UNAUTHORIZED

    if not SUPABASE_JWT_SECRET:
        logger.error("SUPABASE_JWT_SECRET is not configured; cannot verify tokens")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured.",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            SUPABASE_JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
        )
    except jwt.PyJWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise _UNAUTHORIZED

    user_id = payload.get("sub")
    if not user_id:
        logger.warning("JWT is valid but missing 'sub' claim")
        raise _UNAUTHORIZED

    return user_id
