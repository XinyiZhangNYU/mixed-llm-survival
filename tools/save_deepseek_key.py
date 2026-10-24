"""Save a DeepSeek API key to ~/.gt6263/deepseek_api_key after checking that it works.

The key is read from stdin (hidden prompt when run in a terminal, or piped), checked against
the DeepSeek API (balance and model list), and only then written. It is never printed.

Usage:
    python tools/save_deepseek_key.py            # prompt for the key, verify, save
    echo KEY | python tools/save_deepseek_key.py # same, key piped in
    python tools/save_deepseek_key.py --check    # verify the saved key (or DEEPSEEK_API_KEY)
"""
import getpass, json, os, sys, urllib.error, urllib.request
from pathlib import Path

KEY_FILE = Path.home() / ".gt6263" / "deepseek_api_key"
SHOWN = "~/.gt6263/deepseek_api_key"   # printed instead of the home-folder path
BASE = "https://api.deepseek.com"


def get(path, key):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]


def verify(key):
    """Returns (ok, lines describing the account). Never includes the key."""
    status, body = get("/user/balance", key)
    if status != 200:
        return False, [f"Key rejected: HTTP {status} {body}"]
    lines = [f"Account available: {body.get('is_available')}"]
    for b in body.get("balance_infos", []):
        lines.append(f"  balance {b.get('total_balance')} {b.get('currency')} "
                     f"(granted {b.get('granted_balance')}, topped up {b.get('topped_up_balance')})")
    status, body = get("/models", key)
    if status == 200:
        lines.append("Models: " + ", ".join(m["id"] for m in body.get("data", [])))
    return True, lines


def save(key):
    """Check format, verify against the API, then write. Returns (ok, lines)."""
    key = key.strip()
    if not key.startswith("sk-") or any(c.isspace() for c in key) or len(key) < 20:
        return False, [f"That does not look like a DeepSeek key (expected 'sk-...'; got {len(key)} characters). Not saved."]
    ok, lines = verify(key)
    if not ok:
        return False, lines + ["Not saved."]
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key + "\n")
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass
    return True, lines + [f"Saved to {SHOWN} ({len(key)} characters)."]


def main():
    if "--check" in sys.argv:
        key = os.environ.get("DEEPSEEK_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
        if not key:
            sys.exit(f"No key in DEEPSEEK_API_KEY or {SHOWN}")
        ok, lines = verify(key)
        print("\n".join(lines))
        sys.exit(0 if ok else 1)
    key = getpass.getpass("DeepSeek API key (hidden): ") if sys.stdin.isatty() else sys.stdin.readline()
    ok, lines = save(key)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
