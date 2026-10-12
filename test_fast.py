import re
FAST_REPLIES = {
    "hello": "Hello, sir. How can I help?",
    "hi": "Hi there. What can I do for you?",
    "hey": "Hey. Ready when you are.",
}
def _resolve(val): return val() if callable(val) else val
def fast_reply(norm):
    if not norm: return None
    if norm in FAST_REPLIES: return _resolve(FAST_REPLIES[norm])
    stripped = re.sub(r"[.!?]+$", "", norm).strip()
    if stripped in FAST_REPLIES: return _resolve(FAST_REPLIES[stripped])
    for key, val in FAST_REPLIES.items():
        if norm == key or norm.startswith(key + " ") or norm.endswith(" " + key):
            return _resolve(val)
    return None

print(fast_reply("hey jarvis"))
