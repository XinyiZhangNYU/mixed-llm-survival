"""Save a Zhipu (GLM) or Gemini API key under ~/.gt6263/ after checking that it works.

The key is read from stdin (hidden prompt when run in a terminal, or piped), checked against
the vendor's free model-list endpoint, and only then written. It is never printed.
DeepSeek keys keep their own script, tools/save_deepseek_key.py.

Usage:
    python tools/save_api_key.py glm|gemini            # prompt for the key, verify, save
    python tools/save_api_key.py glm|gemini --check    # verify the saved key
"""
import getpass, json, os, sys, urllib.error, urllib.request
from pathlib import Path

DIR = Path.home() / ".gt6263"


def shown(f):
    """The key file as ~/.gt6263/NAME, so no home-folder path is printed."""
    return f"~/.gt6263/{f.name}"
VENDORS = {
    "glm": {"name": "Zhipu GLM", "file": DIR / "zhipu_api_key", "model": "glm-5.3-flash",
            "models_url": "https://open.bigmodel.cn/api/paas/v4/models"},
    "gemini": {"name": "Gemini", "file": DIR / "gemini_api_key", "model": "gemini-3.8-flash",
               "models_url": "https://generativelanguage.googleapis.com/v1beta/openai/models"},
}


def verify(vendor, key):
    """Returns (ok, lines describing the account). Never includes the key."""
    v = VENDORS[vendor]
    req = urllib.request.Request(v["models_url"], headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        if e.code == 404:   # no model list on this endpoint; the key is checked by its first call
            return True, ["Model list not available here; run tools/check_seat.py to check the key."]
        return False, [f"Key rejected: HTTP {e.code} {detail}"]
    ids = [m.get("id", "") for m in body.get("data", [])]
    found = any(i.split("/")[-1] == v["model"] for i in ids)
    return True, [f"Key accepted; {len(ids)} models listed; {v['model']} "
                  + ("listed." if found else "NOT listed (check the model name before a game).")]


def save(vendor, key):
    """Check format, verify against the API, then write. Returns (ok, lines)."""
    key = "".join(key.split())   # keys never contain whitespace; drop line breaks from pasting
    if not 10 <= len(key) <= 300:
        return False, [f"Expected a {VENDORS[vendor]['name']} key of 10-300 characters; got {len(key)}. Not saved."]
    ok, lines = verify(vendor, key)
    if not ok:
        return False, lines + ["Not saved."]
    f = VENDORS[vendor]["file"]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(key + "\n")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass
    return True, lines + [f"Saved to {shown(f)} ({len(key)} characters)."]


def main():
    vendor = sys.argv[1] if len(sys.argv) > 1 else ""
    if vendor not in VENDORS:
        sys.exit(__doc__)
    f = VENDORS[vendor]["file"]
    if "--check" in sys.argv:
        if not f.exists():
            sys.exit(f"No key in {shown(f)}")
        ok, lines = verify(vendor, f.read_text().strip())
        print("\n".join(lines))
        sys.exit(0 if ok else 1)
    prompt = f"{VENDORS[vendor]['name']} API key (hidden): "
    key = getpass.getpass(prompt) if sys.stdin.isatty() else sys.stdin.readline()
    ok, lines = save(vendor, key)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
