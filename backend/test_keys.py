import google.generativeai as genai

# Testez CHAQUE clé individuellement
API_KEYS = [
    "AIzaSyBVtZohYOi1wZvOhQP0DUqqqv04c4E6DMM",  # Clé 1
    "AIzaSyCUTxGx5CtZTOaNqcvReL98ubktAIymdIM",  # Remplacez par votre nouvelle clé
    "AIzaSyDeN5H-P3JZnULmxjVqb4y1mG4gISLyhRs",  # Remplacez par votre nouvelle clé
]

for i, key in enumerate(API_KEYS, 1):
    print(f"\n{'='*50}")
    print(f"Test de la clé #{i}")
    print('='*50)
    
    try:
        genai.configure(api_key=key)
        
        # Lister les modèles disponibles
        models = genai.list_models()
        print(f"✅ Clé #{i} valide!")
        
        # Tester une requête simple
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Say 'Hello' in one word")
        print(f"✅ Requête test réussie: {response.text}")
        
    except Exception as e:
        print(f"❌ Clé #{i} erreur: {str(e)[:200]}")