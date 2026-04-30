import httpx

GROK_API_KEY = "xai-VOTRE_CLE_ICI"  # Remplacez par votre clé

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {GROK_API_KEY}",
    "Content-Type": "application/json",
}

# Tous les modèles Grok connus à tester
models = [
    "grok-2-vision-1212",
    "grok-2-vision",
    "grok-vision-beta",
    "grok-2-1212",
    "grok-2",
    "grok-2-latest",
    "grok-beta",
    "grok-3",
    "grok-3-mini",
    "grok-3-fast",
]

for model in models:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5,
    }
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            print(f"✅ {model} — FONCTIONNE!")
        elif r.status_code == 400:
            err = r.json().get("error", "")
            if "not found" in err.lower() or "invalid" in err.lower():
                print(f"❌ {model} — Modèle inexistant")
            else:
                print(f"⚠️  {model} — 400: {err[:80]}")
        elif r.status_code == 401:
            print(f"🔑 {model} — Clé invalide (401)")
            break
        elif r.status_code == 403:
            print(f"🚫 {model} — Accès refusé (403)")
        elif r.status_code == 429:
            print(f"⏳ {model} — Quota dépassé (429)")
        else:
            print(f"?  {model} — Status {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"💥 {model} — Erreur: {e}")