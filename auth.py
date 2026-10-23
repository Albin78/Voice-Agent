"""
Self-healing API token manager for the kiosk backend.

Token resolution order (first one that yields a usable token wins):
  1. In-memory cached access token (if JWT `exp` is not within the expiry skew)
  2. Disk cache file (survives service restarts — avoids re-login on every boot)
  3. Refresh endpoint using the cached/env refresh token
  4. Login endpoint using API_LOGIN_EMAIL / API_LOGIN_PASSWORD from env
  5. Static API_TOKEN from env (legacy fallback)

On any HTTP 401 the token is invalidated and the request is retried once with a
freshly obtained token, so an expired token never surfaces as a service failure.
"""
import base64
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

from config import Config

logger = logging.getLogger("Auth")

# Renew this many seconds before the JWT actually expires (avoids edge-of-expiry 401s)
EXPIRY_SKEW_SECONDS = 60


def _decode_jwt_exp(token: str) -> Optional[float]:
    """Return the `exp` claim (epoch seconds) of a JWT without verifying its signature."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _extract_tokens(body: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Pull (access_token, refresh_token) out of an auth response.
    Tolerates common shapes: {data: {accessToken, refreshToken}}, {accessToken}, {token},
    {data: {tokens: {...}}}, etc.
    """
    if not isinstance(body, dict):
        return None, None

    candidates = [body]
    data = body.get("data")
    if isinstance(data, dict):
        candidates.append(data)
        for nested_key in ("tokens", "token", "auth"):
            nested = data.get(nested_key)
            if isinstance(nested, dict):
                candidates.append(nested)

    access, refresh = None, None
    for c in candidates:
        access = access or c.get("accessToken") or c.get("access_token") or (
            c.get("token") if isinstance(c.get("token"), str) else None
        )
        refresh = refresh or c.get("refreshToken") or c.get("refresh_token")
    return access, refresh


class TokenManager:
    """Thread-safe singleton that always hands out a valid bearer token."""

    _instance: Optional["TokenManager"] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
            return cls._instance

    def _init(self):
        self._lock = threading.RLock()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = (Config.REFRESH_TOKEN or "").strip() or None
        self._load_cache()

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _load_cache(self):
        path = Config.TOKEN_CACHE_PATH
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            self._access_token = cached.get("access_token") or None
            self._refresh_token = cached.get("refresh_token") or self._refresh_token
            logger.info("Loaded API tokens from cache file.")
        except Exception as e:
            logger.warning(f"Could not read token cache ({e}); ignoring.")

    def _save_cache(self):
        path = Config.TOKEN_CACHE_PATH
        if not path:
            return
        try:
            cache_dir = os.path.dirname(os.path.abspath(path))
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "access_token": self._access_token,
                        "refresh_token": self._refresh_token,
                        "saved_at": int(time.time()),
                    },
                    f,
                )
            os.chmod(tmp, 0o600)  # tokens are credentials — owner read/write only
            os.replace(tmp, path)  # atomic swap so a crash never leaves a half-written file
        except Exception as e:
            logger.warning(f"Could not write token cache ({e}).")

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid(token: Optional[str]) -> bool:
        if not token:
            return False
        exp = _decode_jwt_exp(token)
        if exp is None:
            return True  # opaque/non-JWT token: trust it until the server returns 401
        return time.time() < exp - EXPIRY_SKEW_SECONDS

    # ------------------------------------------------------------------
    # Acquisition strategies
    # ------------------------------------------------------------------

    def _try_refresh(self) -> bool:
        if not self._refresh_token or not Config.REFRESH_API_URL:
            return False
        try:
            resp = requests.post(
                Config.REFRESH_API_URL,
                json={"refreshToken": self._refresh_token},
                timeout=8,
            )
            if resp.status_code == 200:
                access, refresh = _extract_tokens(resp.json())
                if access:
                    self._access_token = access
                    if refresh:
                        self._refresh_token = refresh
                    logger.info("API token renewed via refresh endpoint.")
                    return True
            logger.info(f"Refresh endpoint returned {resp.status_code}; falling back to login.")
            self._refresh_token = None  # stale refresh token — don't keep retrying it
        except Exception as e:
            logger.warning(f"Refresh request failed: {e}")
        return False

    def _try_login(self) -> bool:
        email = (Config.API_LOGIN_EMAIL or os.getenv("API_LOGIN_EMAIL") or "").strip()
        password = Config.API_LOGIN_PASSWORD or os.getenv("API_LOGIN_PASSWORD") or ""
        login_url = Config.LOGIN_API_URL or os.getenv("LOGIN_API_URL") or "http://51.222.143.153:5001/api/v1/auth/login"
        field = Config.LOGIN_USERNAME_FIELD or os.getenv("LOGIN_USERNAME_FIELD") or "email"
        if not (login_url and email and password):
            logger.warning("LOGIN_API_URL / API_LOGIN_EMAIL / API_LOGIN_PASSWORD not set; cannot auto-login.")
            return False
        try:
            resp = requests.post(
                login_url,
                json={field: email, "password": password},
                timeout=10,
            )
            if resp.status_code == 200:
                access, refresh = _extract_tokens(resp.json())
                if access:
                    self._access_token = access
                    if refresh:
                        self._refresh_token = refresh
                    logger.info("API token obtained via login.")
                    return True
                logger.error(f"Login succeeded but no access token found in response: {resp.text[:300]}")
            else:
                # Never log the password; the server message is enough to diagnose
                logger.error(f"Login failed ({resp.status_code}): {resp.text[:300]}")
        except Exception as e:
            logger.error(f"Login request failed: {e}")
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_token(self, force_renew: bool = False) -> str:
        """Return a valid access token, renewing it transparently if needed."""
        with self._lock:
            if not force_renew and self._is_valid(self._access_token):
                return self._access_token  # type: ignore[return-value]

            if self._try_refresh() or self._try_login():
                self._save_cache()
                return self._access_token  # type: ignore[return-value]

            # Last resort: static token from env (may itself be expired)
            static = (Config.API_TOKEN or "").strip()
            if static:
                logger.warning("Falling back to static API_TOKEN from env.")
            return static

    def invalidate(self, token: Optional[str] = None):
        """
        Drop the current access token so the next get_token() renews it.
        If `token` is passed, only invalidate if the current token matches it.
        This prevents multiple concurrent threads from invalidating a newly renewed token.
        """
        with self._lock:
            if token is None or self._access_token == token:
                self._access_token = None

    def auth_headers(self, force_renew: bool = False) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.get_token(force_renew=force_renew)
        if token:
            headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        return headers

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Perform an authenticated request. On 401, renew the token and retry exactly once.
        """
        extra_headers = kwargs.pop("headers", None) or {}
        kwargs.setdefault("timeout", 5)

        token_used = self.get_token()
        resp = requests.request(method, url, headers={**self.auth_headers(), **extra_headers}, **kwargs)
        if resp.status_code == 401:
            logger.info(f"401 from {url}; renewing token and retrying once.")
            self.invalidate(token_used)
            resp = requests.request(
                method, url, headers={**self.auth_headers(), **extra_headers}, **kwargs
            )
        return resp


def get_token_manager() -> TokenManager:
    return TokenManager()
