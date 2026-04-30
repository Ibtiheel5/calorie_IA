import httpx

OPENROUTER_API_KEY = "sk-or-VOTRE_CLE_ICI"  # Remplacez par votre clé

print("🔍 Récupération des modèles gratuits avec vision sur OpenRouter...\n")

r = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
    timeout=30
)

print(f"Status: {r.status_code}\n")

if r.status_code == 200:
    models = r.json().get("data", [])
    
    free_vision = []
    for m in models:
        model_id = m.get("id", "")
        pricing   = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "1") or "1")
        
        # Check if free
        is_free = prompt_price == 0 or ":free" in model_id
        
        # Check if supports vision
        architecture = m.get("architecture", {})
        modalities   = architecture.get("input_modalities", [])
        has_vision   = "image" in modalities
        
        if is_free and has_vision:
            free_vision.append(model_id)
    
    if free_vision:
        print(f"✅ {len(free_vision)} modèles gratuits avec vision trouvés:\n")
        for m in free_vision:
            print(f"  - {m}")
    else:
        print("⚠️  Aucun modèle gratuit avec vision trouvé.")
        print("\nTous les modèles gratuits disponibles:")
        for m in models:
            pricing = m.get("pricing", {})
            if float(pricing.get("prompt", "1") or "1") == 0:
                print(f"  - {m['id']}")
else:
    print(f"Erreur: {r.text[:300]}")