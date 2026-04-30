from PIL import Image
import io
import numpy as np
import json
import os
from ml_model.train_model_light import detect_food, detect_food_with_portions

# Les fonctions sont maintenant importées du module ML
# Base de données de plats avec leurs ingrédients et quantités types
FOOD_DATABASE = {
    "couscous": {
        "ingredients": [
            {"name": "couscous semolina", "calories_per_100g": 112, "typical_portion_g": 200},
            {"name": "chicken", "calories_per_100g": 165, "typical_portion_g": 150},
            {"name": "carrots", "calories_per_100g": 41, "typical_portion_g": 80},
            {"name": "zucchini", "calories_per_100g": 17, "typical_portion_g": 80},
            {"name": "chickpeas", "calories_per_100g": 139, "typical_portion_g": 100},
            {"name": "olives", "calories_per_100g": 115, "typical_portion_g": 30},
            {"name": "olive oil", "calories_per_100g": 884, "typical_portion_g": 15},
            {"name": "onions", "calories_per_100g": 40, "typical_portion_g": 50},
            {"name": "tomatoes", "calories_per_100g": 18, "typical_portion_g": 100},
            {"name": "spices (cumin, paprika, turmeric)", "calories_per_100g": 250, "typical_portion_g": 5}
        ],
        "colors": [(255, 200, 100), (200, 150, 50), (255, 150, 50)],
        "texture": "granular"
    },
    "pizza": {
        "ingredients": [
            {"name": "pizza dough", "calories_per_100g": 266, "typical_portion_g": 200},
            {"name": "tomato sauce", "calories_per_100g": 24, "typical_portion_g": 80},
            {"name": "mozzarella", "calories_per_100g": 280, "typical_portion_g": 100},
            {"name": "olive oil", "calories_per_100g": 884, "typical_portion_g": 10}
        ],
        "colors": [(255, 100, 80), (255, 200, 150), (255, 255, 200)],
        "texture": "flat"
    },
    "tajine": {
        "ingredients": [
            {"name": "lamb", "calories_per_100g": 294, "typical_portion_g": 150},
            {"name": "prunes", "calories_per_100g": 240, "typical_portion_g": 50},
            {"name": "almonds", "calories_per_100g": 579, "typical_portion_g": 30},
            {"name": "onions", "calories_per_100g": 40, "typical_portion_g": 100},
            {"name": "olive oil", "calories_per_100g": 884, "typical_portion_g": 15},
            {"name": "spices", "calories_per_100g": 250, "typical_portion_g": 5}
        ],
        "colors": [(150, 100, 50), (200, 150, 100), (100, 50, 50)],
        "texture": "stew"
    },
    "salad": {
        "ingredients": [
            {"name": "lettuce", "calories_per_100g": 15, "typical_portion_g": 100},
            {"name": "tomatoes", "calories_per_100g": 18, "typical_portion_g": 100},
            {"name": "cucumber", "calories_per_100g": 15, "typical_portion_g": 80},
            {"name": "olive oil", "calories_per_100g": 884, "typical_portion_g": 10},
            {"name": "lemon juice", "calories_per_100g": 29, "typical_portion_g": 10}
        ],
        "colors": [(0, 200, 0), (255, 100, 100), (100, 200, 100)],
        "texture": "mixed"
    },
    "burger": {
        "ingredients": [
            {"name": "burger bun", "calories_per_100g": 265, "typical_portion_g": 100},
            {"name": "beef patty", "calories_per_100g": 250, "typical_portion_g": 150},
            {"name": "cheese", "calories_per_100g": 402, "typical_portion_g": 30},
            {"name": "lettuce", "calories_per_100g": 15, "typical_portion_g": 20},
            {"name": "tomato", "calories_per_100g": 18, "typical_portion_g": 50}
        ],
        "colors": [(200, 150, 100), (255, 200, 100), (0, 150, 0)],
        "texture": "layered"
    }
}

def analyze_image_advanced(image):
    """
    Analyse avancée de l'image pour identifier le plat
    """
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner pour l'analyse
    img_small = image.resize((100, 100))
    pixels = list(img_small.getdata())
    
    # 1. ANALYSE DES COULEURS
    # Diviser l'image en régions (haut, milieu, bas)
    height = 100
    top_region = pixels[:33*100]
    middle_region = pixels[33*100:66*100]
    bottom_region = pixels[66*100:]
    
    # Couleurs moyennes par région
    regions = {
        'top': np.mean(top_region, axis=0),
        'middle': np.mean(middle_region, axis=0),
        'bottom': np.mean(bottom_region, axis=0),
        'overall': np.mean(pixels, axis=0)
    }
    
    # 2. ANALYSE DE LA TEXTURE
    img_gray = image.convert('L')
    img_array = np.array(img_gray)
    
    # Variance locale (texture)
    texture_variance = np.var(img_array)
    
    # Détection de bords (contours)
    edges = np.abs(np.diff(img_array, axis=0)).mean() + np.abs(np.diff(img_array, axis=1)).mean()
    
    # 3. ANALYSE DE LA COMPOSITION
    # Nombre de couleurs distinctes
    color_diversity = len(set([(p[0]//30, p[1]//30, p[2]//30) for p in pixels]))
    
    # Ratio vert/rouge (indicateur de légumes)
    green_ratio = regions['overall'][1] / (regions['overall'][0] + 1)
    red_ratio = regions['overall'][0] / (regions['overall'][1] + 1)
    
    return {
        'regions': regions,
        'texture_variance': texture_variance,
        'edges': edges,
        'color_diversity': color_diversity,
        'green_ratio': green_ratio,
        'red_ratio': red_ratio
    }

def identify_dish(image_features):
    """
    Identifie le plat en fonction des caractéristiques de l'image
    """
    scores = {}
    
    for dish_name, dish_info in FOOD_DATABASE.items():
        score = 0
        r = image_features['regions']['overall']
        
        # Score pour le couscous (plats granulaires, couleur or/jaune)
        if dish_name == "couscous":
            if 100 < r[0] < 220 and 100 < r[1] < 200 and 50 < r[2] < 150:
                score += 30  # Couleur or/jaune caractéristique
            if image_features['texture_variance'] > 2000:
                score += 25  # Texture granulaire
            if image_features['color_diversity'] > 15:
                score += 20  # Plusieurs ingrédients visibles
            if 70 < image_features['edges'] < 150:
                score += 15  # Texture caractéristique de la semoule
        
        # Score pour la pizza (plat, rouge, circulaire)
        elif dish_name == "pizza":
            if r[0] > 150 and r[1] < 150:
                score += 30  # Dominance rouge
            if image_features['texture_variance'] < 1500:
                score += 20  # Surface relativement uniforme
            if color_similarity(r, (200, 150, 100)) < 50:
                score += 15
        
        # Score pour le tajine (marron, stew)
        elif dish_name == "tajine":
            if r[0] < 150 and r[1] < 120 and r[2] < 100:
                score += 30  # Couleurs brunes
            if image_features['texture_variance'] > 1500:
                score += 20  # Texture de ragoût
        
        # Score pour la salade (vert, mixte)
        elif dish_name == "salad":
            if image_features['green_ratio'] > 1.2:
                score += 30  # Dominance verte
            if image_features['color_diversity'] > 20:
                score += 25  # Beaucoup de couleurs différentes
        
        # Score pour le burger (couches, brun)
        elif dish_name == "burger":
            if abs(r[0] - r[1]) < 50 and r[2] < 100:
                score += 20  # Couleurs brunes
            if image_features['edges'] > 100:
                score += 20  # Structure en couches
        
        scores[dish_name] = score
    
    # Retourner le meilleur match
    if scores:
        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]
        print(f"🎯 Meilleur match: {best_match} (score: {confidence})")
        return best_match, confidence
    
    return "couscous", 0  # Par défaut

def color_similarity(c1, c2):
    """Calcule la similarité entre deux couleurs"""
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def estimate_portion_size(image):
    """
    Estime la taille des portions basée sur la taille de l'image
    """
    width, height = image.size
    image_area = width * height
    
    # Logique simple : plus l'image est grande, plus la portion est grande
    if image_area > 1000000:  # > 1MP
        portion_multiplier = 1.5  # Grande portion
    elif image_area > 500000:  # > 0.5MP
        portion_multiplier = 1.0  # Portion normale
    else:
        portion_multiplier = 0.7  # Petite portion
    
    return portion_multiplier

def detect_food(image_bytes):
    """
    Détecte les aliments avec analyse d'image avancée
    """
    try:
        # Ouvrir l'image
        image = Image.open(io.BytesIO(image_bytes))
        print(f"📸 Image: {image.size}, Mode: {image.mode}")
        
        # Analyse avancée
        features = analyze_image_advanced(image)
        print(f"📊 Caractéristiques extraites:")
        print(f"   Couleur: RGB({features['regions']['overall'][0]:.0f}, {features['regions']['overall'][1]:.0f}, {features['regions']['overall'][2]:.0f})")
        print(f"   Texture: {features['texture_variance']:.0f}")
        print(f"   Diversité couleurs: {features['color_diversity']}")
        
        # Identifier le plat
        dish_name, confidence = identify_dish(features)
        
        # Estimer la taille de la portion
        portion_multiplier = estimate_portion_size(image)
        
        # Obtenir les ingrédients avec quantités ajustées
        dish_info = FOOD_DATABASE[dish_name]
        ingredients = []
        
        for ing in dish_info['ingredients']:
            # Ajuster la portion selon la taille de l'image
            adjusted_portion = ing['typical_portion_g'] * portion_multiplier
            adjusted_calories = (ing['calories_per_100g'] * adjusted_portion) / 100
            
            ingredients.append({
                "name": ing['name'],
                "calories": round(adjusted_calories),
                "portion_g": round(adjusted_portion)
            })
        
        print(f"🍽️ Plat identifié: {dish_name} (confiance: {confidence})")
        print(f"📏 Multiplicateur de portion: {portion_multiplier:.2f}")
        
        # Retourner juste les noms pour la compatibilité
        ingredient_names = [ing['name'] for ing in ingredients]
        return ingredient_names
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        # Fallback: ingrédients du couscous par défaut
        return ["couscous semolina", "chicken", "carrots", "zucchini", "chickpeas", "olives"]

# Nouvelle fonction pour obtenir les détails avec portions
def detect_food_with_portions(image_bytes):
    """
    Version détaillée avec portions
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        features = analyze_image_advanced(image)
        dish_name, confidence = identify_dish(features)
        portion_multiplier = estimate_portion_size(image)
        
        dish_info = FOOD_DATABASE[dish_name]
        ingredients_details = []
        
        for ing in dish_info['ingredients']:
            adjusted_portion = ing['typical_portion_g'] * portion_multiplier
            adjusted_calories = (ing['calories_per_100g'] * adjusted_portion) / 100
            
            ingredients_details.append({
                "name": ing['name'],
                "calories_per_100g": ing['calories_per_100g'],
                "portion_g": round(adjusted_portion),
                "calories": round(adjusted_calories)
            })
        
        return {
            "dish": dish_name,
            "confidence": confidence,
            "ingredients": ingredients_details,
            "total_calories": sum(ing['calories'] for ing in ingredients_details)
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None