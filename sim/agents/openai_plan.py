"""Provider that runs OpenAI models on the user's ChatGPT plan through Sign in with ChatGPT.

Uses OpenAI's open-source / locally hosted app flow (developers.openai.com/siwc/
token-sharing-open-source): an OAuth sign-in grants `chatgpt.tokens.use.direct`, and the
access token then authorizes requests to the public Responses API, drawing on the plan's
usage instead of API credits. No API key is involved.

Nothing is added to the model context beyond what the engine sends: the rules go in
`instructions`, the seat's own user/assistant turns go in `input`, and no tools are
declared. The route requires store=false and stream=true and has no server-side
conversation state, so the full history is sent on every call.

Credentials live in ~/.gt6263/chatgpt_plan/ (outside the project) and are never written
to logs or metadata. Run tools/chatgpt_plan_login.py once to sign in.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import random
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path

AUTH_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
RESPONSES_URL = RESOURCE + "/responses"
MODELS_URL = RESOURCE + "/models"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
REDIRECT_PORT = 1455
REDIRECT_URI = f"http://127.0.0.1:{REDIRECT_PORT}/auth/callback"
AGENT_NAME = "gt6263-survival-sim"

STORE = Path.home() / ".gt6263" / "chatgpt_plan"
HOST_FILE = STORE / "host.json"
CRED_FILE = STORE / "credentials.json"
PENDING_FILE = STORE / "pending.json"

RETRY_STATUS = {500, 502, 503, 504}
_refresh_lock = threading.Lock()


def _ssl_context() -> ssl.SSLContext:
    """certifi's CA bundle when installed (this machine's Python store has an expired root)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


class PlanUsageLimit(RuntimeError):
    """The plan (or this app's cap) has no usage left; pause instead of retrying."""


# ---------------------------------------------------------------- credentials

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jwt_payload(token: str) -> dict:
    part = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def host_id() -> str:
    """Stable opaque per-machine id, generated once (required by the flow)."""
    if HOST_FILE.is_file():
        return json.loads(HOST_FILE.read_text(encoding="utf-8"))["ext_agent_host_id"]
    STORE.mkdir(parents=True, exist_ok=True)
    hid = f"urn:uuid:{uuid.uuid4()}"
    HOST_FILE.write_text(json.dumps({"ext_agent_host_id": hid}), encoding="utf-8")
    return hid


def _load_creds() -> dict:
    if not CRED_FILE.is_file():
        raise RuntimeError(f"no ChatGPT plan credentials at {CRED_FILE}; run tools/chatgpt_plan_login.py")
    return json.loads(CRED_FILE.read_text(encoding="utf-8"))


def _save_creds(c: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = CRED_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(c), encoding="utf-8")
    tmp.replace(CRED_FILE)


def _post_form(url: str, fields: dict) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), method="POST",
                                 headers={"content-type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
        return json.loads(r.read().decode("utf-8"))


def _store_token_response(tok: dict, client_id: str, old: dict | None = None) -> dict:
    scope = tok.get("scope", (old or {}).get("scope", ""))
    if PLAN_SCOPE not in scope.split():
        raise RuntimeError(f"plan usage was not granted (scopes: {scope})")
    c = {"client_id": client_id,
         "access_token": tok["access_token"],
         "refresh_token": tok.get("refresh_token") or (old or {}).get("refresh_token"),
         "id_token": tok.get("id_token") or (old or {}).get("id_token"),
         "scope": scope,
         "expires_at": time.time() + int(tok.get("expires_in", 3600))}
    _save_creds(c)
    return c


def start_login() -> str:
    """Create a sign-in attempt and return its authorize URL. The PKCE verifier, state and
    nonce stay in STORE/pending.json on this machine."""
    old = _load_creds() if CRED_FILE.is_file() else {}
    client_id = old.get("client_id")
    verifier = _b64url(secrets.token_bytes(48))
    state, nonce = _b64url(secrets.token_bytes(24)), _b64url(secrets.token_bytes(24))
    params = {"client_id": client_id or "dynamic_agent_client",
              "ext_agent_host_id": host_id(),
              "response_type": "code", "redirect_uri": REDIRECT_URI, "scope": SCOPE,
              "resource": RESOURCE, "state": state, "nonce": nonce,
              "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
              "code_challenge_method": "S256"}
    if not client_id:
        params["agent_name_hint"] = AGENT_NAME
    PENDING_FILE.write_text(json.dumps({"verifier": verifier, "state": state, "nonce": nonce,
                                        "client_id": client_id, "created": time.time()}),
                            encoding="utf-8")
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def finish_login(callback: str | dict) -> dict:
    """Complete a pending sign-in from the callback URL (or its parsed query). The callback
    may come from another device: after allowing access there, the browser lands on an
    unreachable 127.0.0.1 address whose URL carries the code."""
    if isinstance(callback, str):
        callback = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(callback.strip()).query).items()}
    if not PENDING_FILE.is_file():
        raise RuntimeError("no pending sign-in; run start first")
    pend = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    old = _load_creds() if CRED_FILE.is_file() else {}
    if callback.get("state") != pend["state"]:
        raise RuntimeError("state mismatch; discarding callback")
    if "error" in callback:
        raise RuntimeError(f"authorization failed: {callback.get('error')} {callback.get('error_description', '')}")
    client_id, issued = pend["client_id"], callback.get("client_id")
    if client_id and issued and issued != client_id:
        raise RuntimeError("callback returned a different client_id; rejecting")
    client_id = client_id or issued
    if not client_id:
        raise RuntimeError("no client_id returned for the new registration")
    tok = _post_form(TOKEN_URL, {"grant_type": "authorization_code", "client_id": client_id,
                                 "code": callback["code"], "code_verifier": pend["verifier"],
                                 "redirect_uri": REDIRECT_URI, "resource": RESOURCE})
    if tok.get("id_token") and _jwt_payload(tok["id_token"]).get("nonce") != pend["nonce"]:
        raise RuntimeError("ID token nonce mismatch")
    c = _store_token_response(tok, client_id, old)
    PENDING_FILE.unlink()
    return grant_facts(c)


def login(open_browser: bool = True, timeout_s: float = 600.0) -> dict:
    """Browser sign-in on this machine via the loopback callback."""
    url = start_login()
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path != "/auth/callback":
                self.send_response(404); self.end_headers(); return
            got.update({k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()})
            self.send_response(200)
            self.send_header("content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("Sign-in received. You can close this tab.".encode())

        def log_message(self, *a):  # keep the code out of the console
            pass

    srv = http.server.HTTPServer(("127.0.0.1", REDIRECT_PORT), Handler)
    srv.timeout = 1.0
    print("Open this URL to sign in (it opens automatically):\n" + url, flush=True)
    if open_browser:
        webbrowser.open(url)
    deadline = time.time() + timeout_s
    while not got and time.time() < deadline:
        srv.handle_request()
    srv.server_close()
    if not got:
        raise RuntimeError("timed out waiting for the sign-in callback")
    return finish_login(got)


def grant_facts(c: dict | None = None) -> dict:
    """Non-secret description of the stored grant (no tokens, no e-mail)."""
    c = c or _load_creds()
    claims = _jwt_payload(c["access_token"])
    auth = claims.get("https://api.openai.com/auth", {})
    return {"scope": c["scope"], "plan_scope_granted": PLAN_SCOPE in c["scope"].split(),
            "audience": claims.get("aud"),
            "plan_type": auth.get("chatgpt_plan_type"),
            "auth_claim_keys": sorted(auth),
            "expires_in_s": int(c["expires_at"] - time.time())}


def access_token() -> str:
    """Current access token, refreshed near expiry. Refreshes are serialized."""
    with _refresh_lock:
        c = _load_creds()
        if c["expires_at"] - time.time() > 300:
            return c["access_token"]
        tok = _post_form(TOKEN_URL, {"grant_type": "refresh_token", "client_id": c["client_id"],
                                     "refresh_token": c["refresh_token"], "resource": RESOURCE})
        return _store_token_response(tok, c["client_id"], c)["access_token"]


def list_models() -> list[dict]:
    req = urllib.request.Request(MODELS_URL, headers={"authorization": f"Bearer {access_token()}"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
        data = json.loads(r.read().decode("utf-8"))
    rows = data.get("models", data.get("data", []))
    return [{"slug": m.get("slug", m.get("id")), "display_name": m.get("display_name")}
            for m in rows if m.get("visibility", "list") == "list"]


# ---------------------------------------------------------------- provider

class OpenAIPlanProvider:
    def __init__(self, model: str, reasoning_effort: str | None = None,
                 timeout_s: float = 300.0, max_http_retries: int = 4,
                 reasoning_summary: str | None = None):
        """reasoning_summary: e.g. "auto" asks for the provider's reasoning summary, kept in
        meta["thinking"] for the logs. Raw reasoning is not available on this route."""
        self.model = model
        self.model_id = f"openai-plan:{model}"
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_http_retries = max_http_retries
        self.reasoning_summary = reasoning_summary

    def _bearer(self) -> str:
        return access_token()

    def _http_error(self, code: int, detail: str) -> bool:
        """Raise for errors that must stop the run; True to retry, False to report the call as failed."""
        if code == 429 or "usage_limit_exceeded" in detail:
            raise PlanUsageLimit(f"HTTP {code}: {detail}")
        return code in RETRY_STATUS

    def _stream_error(self, code: str, err: str) -> bool:
        """Same contract as _http_error, for an error reported inside the response stream."""
        if code == "subscription_sharing_usage_limit_exceeded":
            raise PlanUsageLimit(err)
        return code in ("subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable")

    def _add_cost(self, meta: dict) -> None:
        """Plan calls report no price."""

    def _body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "instructions": messages[0]["content"],
                "input": [{"role": m["role"], "content": m["content"]} for m in messages[1:]],
                "store": False, "stream": True}
        if self.reasoning_effort or self.reasoning_summary:
            body["reasoning"] = {}
        if self.reasoning_effort:
            body["reasoning"]["effort"] = self.reasoning_effort
        if self.reasoning_summary:
            body["reasoning"]["summary"] = self.reasoning_summary
        return body

    def _post(self, body: dict) -> dict:
        """POST and read the SSE stream; returns the final response object."""
        req = urllib.request.Request(
            RESPONSES_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"authorization": f"Bearer {self._bearer()}", "content-type": "application/json",
                     "accept": "text/event-stream"})
        final, text, summary = None, [], {}
        event_counts = {}

        def remember_summary(item):
            if item.get("type") == "reasoning":
                for index, part in enumerate(item.get("summary") or []):
                    if part.get("type") == "summary_text" and part.get("text"):
                        summary[(item.get("id"), index)] = part["text"]

        with urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL) as r:
            request_id = r.headers.get("x-request-id")
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                t = ev.get("type", "")
                event_counts[t] = event_counts.get(t, 0) + 1
                if t == "response.output_text.delta":
                    text.append(ev.get("delta", ""))
                elif t == "response.reasoning_summary_text.delta":
                    k = (ev.get("item_id"), ev.get("summary_index", 0))
                    summary[k] = summary.get(k, "") + ev.get("delta", "")
                elif t in ("response.reasoning_summary_text.done",
                           "response.reasoning_summary_part.added",
                           "response.reasoning_summary_part.done"):
                    part_text = ev.get("text") if t.endswith("text.done") else (ev.get("part") or {}).get("text")
                    if part_text:
                        # Completed text replaces deltas for this part; do not append twice.
                        summary[(ev.get("item_id"), ev.get("summary_index", 0))] = part_text
                elif t == "response.output_item.done":
                    remember_summary(ev.get("item") or {})
                elif t in ("response.completed", "response.failed", "response.incomplete"):
                    final = {**ev["response"], "_event": t}
                    for item in final.get("output") or []:
                        remember_summary(item)
                    break
                elif t == "error":
                    final = {"_event": "error", "error": ev.get("error", ev)}
                    break
        if final is None:
            final = {"_event": "stream_ended"}
        final["_text"] = "".join(text)
        final["_summary"] = "\n\n".join(summary.values())
        final["_request_id"] = request_id
        final["_event_counts"] = event_counts
        return final

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        body = self._body(messages)
        last_err = None
        for attempt in range(self.max_http_retries + 1):
            try:
                data = await asyncio.to_thread(self._post, body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if not self._http_error(e.code, detail):
                    return "", {"error": last_err}
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = f"network: {e}"
            else:
                ev = data["_event"]
                err = data.get("error") or {}
                if ev != "response.completed":
                    code = err.get("code", "") if isinstance(err, dict) else ""
                    last_err = f"{ev}: {json.dumps(err)[:500]}"
                    if not self._stream_error(code, last_err) and ev != "stream_ended":
                        return data["_text"], {"error": last_err, "request_id": data["_request_id"]}
                else:
                    break
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
        else:
            return "", {"error": f"gave up after retries; last: {last_err}"}

        text = data["_text"] or "".join(
            c.get("text", "") for o in data.get("output", []) if o.get("type") == "message"
            for c in o.get("content", []) if c.get("type") == "output_text")
        thinking = data.get("_summary") or "\n\n".join(
            s.get("text", "") for o in data.get("output", []) if o.get("type") == "reasoning"
            for s in o.get("summary") or [] if s.get("type") == "summary_text")
        u = data.get("usage") or {}
        meta = {
            "input_tokens": u.get("input_tokens", 0),
            "cached_input_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
            "stop_reason": data.get("status"),
            "models": [data.get("model")],
            "request_id": data["_request_id"] or data.get("id"),
            # Event names/counts only: no credentials, request bodies or encrypted payloads.
            "response_event_counts": data.get("_event_counts", {}),
        }
        if thinking:
            meta["thinking"] = thinking
        self._add_cost(meta)
        return text, meta
