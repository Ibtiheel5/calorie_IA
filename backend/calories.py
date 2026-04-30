# Base de données des calories par ingrédient (pour 100g)
foods_calories = {
    # Viandes
    "chicken": 165,
    "beef": 250,
    "fish": 206,
    "salmon": 208,
    "ham": 145,
    "bacon": 541,
    
    # Céréales
    "rice": 130,
    "pasta": 131,
    "bread": 265,
    "bun": 265,
    "pizza dough": 266,
    "tortilla": 300,
    "noodles": 138,
    
    # Légumes
    "tomato": 18,
    "lettuce": 15,
    "cucumber": 15,
    "onion": 40,
    "garlic": 149,
    "potatoes": 77,
    "carrot": 41,
    "broccoli": 34,
    "spinach": 23,
    "pepper": 20,
    "mushroom": 22,
    "avocado": 160,
    "olive": 115,
    
    # Sauces et condiments
    "tomato sauce": 24,
    "soy sauce": 53,
    "olive oil": 884,
    "butter": 717,
    "mayonnaise": 680,
    "ketchup": 101,
    "pesto": 339,
    "vinegar": 18,
    
    # Produits laitiers
    "cheese": 402,
    "mozzarella": 280,
    "parmesan": 431,
    "eggs": 155,
    "milk": 42,
    "cream": 340,
    
    # Herbes et épices
    "basil": 23,
    "oregano": 25,
    "herbs": 20,
    "salt": 0,
    "pepper": 251,
    
    # Fruits
    "lemon": 29,
    "lime": 30,
    "banana": 89,
    "apple": 52,
    
    # Autres
    "seaweed": 45,
    "chocolate": 546,
    "sugar": 387,
    "flour": 364,
    "honey": 304,
}

def get_calories(ingredients):
    """
    Calcule les calories totales pour une liste d'ingrédients
    """
    total_calories = 0
    ingredient_details = []
    
    for ingredient in ingredients:
        ingredient_lower = ingredient.lower()
        
        # Chercher une correspondance exacte
        calories = foods_calories.get(ingredient_lower)
        
        # Si pas trouvé, chercher une correspondance partielle
        if calories is None:
            for key, value in foods_calories.items():
                if key in ingredient_lower or ingredient_lower in key:
                    calories = value
                    break
        
        # Valeur par défaut
        if calories is None:
            calories = 150
        
        total_calories += calories
        ingredient_details.append({
            "name": ingredient,
            "calories": calories
        })
    
    return {
        "total_calories": total_calories,
        "ingredients": ingredient_details
    }