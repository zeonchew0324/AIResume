from slowapi import Limiter
from slowapi.util import get_remote_address


def user_or_ip(request) -> str:
    """Key rate limits by the authenticated user, falling back to client IP.

    get_current_user_id stores the verified user id on request.state before
    the handler (and therefore the rate-limit check) runs, so authenticated
    traffic is limited per user — per-IP limiting is unfair behind shared
    NATs and collapses to a single bucket behind a reverse proxy.
    """
    user_id = getattr(request.state, "user_id", None)
    return user_id or get_remote_address(request)


limiter = Limiter(key_func=user_or_ip)
