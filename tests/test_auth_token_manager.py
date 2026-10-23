import base64
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

import auth
from auth import TokenManager, _decode_jwt_exp, _extract_tokens
from config import Config


def make_jwt(exp_offset: int) -> str:
    """Build an unsigned JWT whose exp is now + exp_offset seconds."""
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{b64({'alg': 'HS256'})}.{b64({'exp': int(time.time()) + exp_offset})}.sig"


def resp(status, body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body or {}
    r.text = json.dumps(body or {})
    return r


class TestTokenManager(unittest.TestCase):
    def setUp(self):
        TokenManager._instance = None  # fresh singleton per test
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch.object(Config, "TOKEN_CACHE_PATH", os.path.join(self.tmpdir, "cache.json")),
            patch.object(Config, "LOGIN_API_URL", "http://auth/login"),
            patch.object(Config, "REFRESH_API_URL", "http://auth/refresh"),
            patch.object(Config, "API_LOGIN_EMAIL", "kiosk@example.com"),
            patch.object(Config, "API_LOGIN_PASSWORD", "secret"),
            patch.object(Config, "REFRESH_TOKEN", ""),
            patch.object(Config, "API_TOKEN", ""),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        TokenManager._instance = None

    def test_decode_and_extract_helpers(self):
        tok = make_jwt(900)
        self.assertAlmostEqual(_decode_jwt_exp(tok), time.time() + 900, delta=5)
        self.assertEqual(_extract_tokens({"data": {"accessToken": "a", "refreshToken": "r"}}), ("a", "r"))
        self.assertEqual(_extract_tokens({"token": "a"}), ("a", None))
        self.assertEqual(_extract_tokens({"data": {"tokens": {"access_token": "a"}}}), ("a", None))

    @patch("auth.requests.post")
    def test_login_when_no_token(self, mock_post):
        good = make_jwt(900)
        mock_post.return_value = resp(200, {"data": {"accessToken": good, "refreshToken": "R1"}})
        self.assertEqual(TokenManager().get_token(), good)
        url, kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
        self.assertEqual(url, "http://auth/login")
        self.assertEqual(kwargs["json"], {"email": "kiosk@example.com", "password": "secret"})

    @patch("auth.requests.post")
    def test_cached_valid_token_not_renewed(self, mock_post):
        good = make_jwt(900)
        mock_post.return_value = resp(200, {"data": {"accessToken": good}})
        tm = TokenManager()
        tm.get_token()
        tm.get_token()
        tm.get_token()
        self.assertEqual(mock_post.call_count, 1)

    @patch("auth.requests.post")
    def test_expired_token_uses_refresh_first(self, mock_post):
        new_tok = make_jwt(900)
        tm = TokenManager()
        tm._access_token = make_jwt(-10)  # expired
        tm._refresh_token = "R1"
        mock_post.return_value = resp(200, {"data": {"accessToken": new_tok, "refreshToken": "R2"}})
        self.assertEqual(tm.get_token(), new_tok)
        self.assertEqual(mock_post.call_args[0][0], "http://auth/refresh")
        self.assertEqual(tm._refresh_token, "R2")

    @patch("auth.requests.post")
    def test_near_expiry_token_is_renewed_proactively(self, mock_post):
        new_tok = make_jwt(900)
        tm = TokenManager()
        tm._access_token = make_jwt(30)  # within 60s skew
        mock_post.return_value = resp(200, {"data": {"accessToken": new_tok}})
        self.assertEqual(tm.get_token(), new_tok)

    @patch("auth.requests.post")
    def test_refresh_fails_then_login(self, mock_post):
        new_tok = make_jwt(900)
        tm = TokenManager()
        tm._access_token = make_jwt(-10)
        tm._refresh_token = "STALE"
        mock_post.side_effect = [
            resp(401, {"message": "Invalid refresh token"}),
            resp(200, {"data": {"accessToken": new_tok}}),
        ]
        self.assertEqual(tm.get_token(), new_tok)
        self.assertEqual([c[0][0] for c in mock_post.call_args_list], ["http://auth/refresh", "http://auth/login"])

    @patch("auth.requests.request")
    @patch("auth.requests.post")
    def test_401_triggers_renew_and_single_retry(self, mock_post, mock_request):
        tok1, tok2 = make_jwt(900), make_jwt(1800)
        mock_post.side_effect = [
            resp(200, {"data": {"accessToken": tok1}}),
            resp(200, {"data": {"accessToken": tok2}}),
        ]
        mock_request.side_effect = [resp(401), resp(200, {"data": []})]
        r = TokenManager().request("GET", "http://api/products")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(mock_request.call_args_list[1][1]["headers"]["Authorization"], f"Bearer {tok2}")

    @patch("auth.requests.post")
    def test_tokens_persist_across_restart(self, mock_post):
        good = make_jwt(900)
        mock_post.return_value = resp(200, {"data": {"accessToken": good, "refreshToken": "R1"}})
        TokenManager().get_token()
        self.assertEqual(oct(os.stat(Config.TOKEN_CACHE_PATH).st_mode & 0o777), "0o600")

        TokenManager._instance = None  # simulate service restart
        mock_post.reset_mock()
        self.assertEqual(TokenManager().get_token(), good)
        mock_post.assert_not_called()

    @patch("auth.requests.post")
    def test_auth_server_down_falls_back_to_static_token(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        with patch.object(Config, "API_TOKEN", "STATIC"):
            self.assertEqual(TokenManager().get_token(), "STATIC")


if __name__ == "__main__":
    unittest.main()
