import jwt
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import get_current_user_id


def bearer(token="some-token"):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

def fake_request():
    return SimpleNamespace(state=SimpleNamespace())

def fake_jwks_client():
    client = Mock()
    client.get_signing_key_from_jwt.return_value = Mock(key="public-key")
    return client


async def test_missing_credentials_rejected():
    with pytest.raises(HTTPException) as exc:
        await get_current_user_id(request=fake_request(), credentials=None)
    assert exc.value.status_code == 401

async def test_empty_token_rejected():
    with pytest.raises(HTTPException) as exc:
        await get_current_user_id(request=fake_request(), credentials=bearer(""))
    assert exc.value.status_code == 401

@patch("app.auth._get_jwks_client", return_value=None)
async def test_unconfigured_supabase_url_returns_500(mock_client):
    with pytest.raises(HTTPException) as exc:
        await get_current_user_id(request=fake_request(), credentials=bearer())
    assert exc.value.status_code == 500

@patch("app.auth.jwt.decode", side_effect=jwt.InvalidSignatureError("bad signature"))
@patch("app.auth._get_jwks_client")
async def test_invalid_token_rejected(mock_client, mock_decode):
    mock_client.return_value = fake_jwks_client()
    with pytest.raises(HTTPException) as exc:
        await get_current_user_id(request=fake_request(), credentials=bearer())
    assert exc.value.status_code == 401

@patch("app.auth._get_jwks_client")
async def test_unknown_signing_key_rejected(mock_client):
    client = fake_jwks_client()
    client.get_signing_key_from_jwt.side_effect = jwt.exceptions.PyJWKClientError("kid not found")
    mock_client.return_value = client
    with pytest.raises(HTTPException) as exc:
        await get_current_user_id(request=fake_request(), credentials=bearer())
    assert exc.value.status_code == 401

@patch("app.auth.jwt.decode", return_value={"role": "authenticated"})
@patch("app.auth._get_jwks_client")
async def test_token_without_sub_claim_rejected(mock_client, mock_decode):
    mock_client.return_value = fake_jwks_client()
    with pytest.raises(HTTPException) as exc:
        await get_current_user_id(request=fake_request(), credentials=bearer())
    assert exc.value.status_code == 401

@patch("app.auth.jwt.decode", return_value={"sub": "user-123"})
@patch("app.auth._get_jwks_client")
async def test_valid_token_returns_user_id(mock_client, mock_decode):
    mock_client.return_value = fake_jwks_client()
    request = fake_request()
    user_id = await get_current_user_id(request=request, credentials=bearer())
    assert user_id == "user-123"
    # The rate limiter keys on this (see app.limiter.user_or_ip)
    assert request.state.user_id == "user-123"
    # Verification must pin the expected audience and asymmetric algorithms
    kwargs = mock_decode.call_args.kwargs
    assert kwargs["audience"] == "authenticated"
    assert set(kwargs["algorithms"]) == {"ES256", "RS256"}
