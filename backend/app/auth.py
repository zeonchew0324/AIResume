import os
import logging

import jwt
from jwt import PyJWKClient
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.concurrency import run_in_threadpool

load_dotenv()

logger = logging.getLogger(__name__)

# Base URL of the Supabase project, e.g. https://xxxx.supabase.co
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
JWT_AUDIENCE = "authenticated"
# Supabase signs access tokens with asymmetric keys served from the project's
# JWKS endpoint (ES256 by default; RS256 for some projects).
JWT_ALGORITHMS = ["ES256", "RS256"]

# auto_error=False so we can return a clean 401 instead of FastAPI's default 403.
bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing authentication token",
    headers={"WWW-Authenticate": "Bearer"},
)

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient | None:
    """Lazily build a cached JWKS client for the configured Supabase project."""
    global _jwks_client
    if _jwks_client is None and SUPABASE_URL:
        _jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    return _jwks_client


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Validate the Supabase access token and return the authenticated user id.

    The token is verified against the project's published JWKS public key, and
    the user id is taken from the verified JWT's `sub` claim — never from the
    request body — so callers cannot act on behalf of another user.
    """
    if credentials is None or not credentials.credentials:
        raise _UNAUTHORIZED

    client = _get_jwks_client()
    if client is None:
        logger.error("SUPABASE_URL is not configured; cannot verify tokens")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured.",
        )

    token = credentials.credentials
    try:
        # JWKS fetch/verification is blocking I/O — keep it off the event loop.
        signing_key = await run_in_threadpool(client.get_signing_key_from_jwt, token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=JWT_ALGORITHMS,
            audience=JWT_AUDIENCE,
        )
    except (jwt.PyJWTError, jwt.exceptions.PyJWKClientError) as e:
        logger.warning(f"JWT validation failed: {e}")
        raise _UNAUTHORIZED

    user_id = payload.get("sub")
    if not user_id:
        logger.warning("JWT is valid but missing 'sub' claim")
        raise _UNAUTHORIZED

    return user_id
