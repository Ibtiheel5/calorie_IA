
# 🍽️ Calorie IA

> Analysez vos repas en photo — calories, ingrédients et macros en quelques secondes.  
> Spécialisé cuisine **tunisienne & nord-africaine**, étendu aux plats internationaux.

---

## ✨ Fonctionnalités

| Feature | Description |
|---|---|
| 📸 **Analyse par photo** | Prenez une photo de votre repas, l'IA identifie le plat |
| 🤖 **Modèle local** | MobileNetV2 entraîné sur 70+ classes dont 38 plats tunisiens/maghrébins |
| 🔢 **Calcul calorique** | Calories, protéines, glucides, lipides par portion |
| 👤 **Profil nutritionnel** | BMR · TDEE · objectif personnalisé (Mifflin-St Jeor) |
| 📊 **Historique des repas** | Sauvegarde dans PostgreSQL, accessible partout |
| 🏆 **Gamification** | XP · badges · défis quotidiens · classement |
| 🤝 **Mode partenaire** | Suivez votre progression avec un ami |
| 🔐 **Authentification** | JWT · PBKDF2 · tokens sécurisés |

---

## 🍲 Plats reconnus

### Tunisiens & Maghrébins (38 classes)
`couscous` · `tajine` · `brik` · `lablabi` · `mechouia` · `ojja` · `chorba` · `kafteji` · `macarouna` · `fricassée` · `shakshuka` · `slata` · `merguez` · `mlawi` · `bambalouni` · `makloub` · `asida` · `mloukhia` · `marqa` · `camounia` · `pkaila` · `osban` · `hraimi` · `kefta` · `dolma` · `sandwich tunisien` · `poisson frit` · `tagine marocain` · `msemen` · `harira` · `pastilla` · `mansaf` · `rechta` · `chakhchoukha` · et plus...

### Internationaux (32 classes)
`pizza` · `burger` · `pasta` · `sushi` · `tacos` · `curry` · `steak` · `caesar salad` · `fried rice` · `falafel` · `hummus` · `baklava` · `waffles` · `donuts` · et plus...

---

## 🏗️ Architecture

```
calorie_IA/
├── backend/                  # FastAPI
│   ├── main.py               # Routes API (auth, meals, analyze)
│   ├── predictor.py          # Moteur ML local
│   ├── ml_model/
│   │   ├── food_model.h5     # Modèle entraîné (MobileNetV2)
│   │   └── class_names.json  # 70+ classes
│   ├── models.py             # SQLAlchemy (User, Meal)
│   ├── schemas.py            # Pydantic schemas
│   ├── security.py           # JWT + PBKDF2
│   └── database.py           # PostgreSQL
│
├── frontend/                 # React
│   ├── App.js
│   ├── components/
│   │   ├── AnalysisPage.js   # Upload + résultats
│   │   ├── HomePage.js       # Dashboard
│   │   ├── MealLibrary.js    # Historique
│   │   ├── GamificationPage.js
│   │   ├── PartnerPage.js
│   │   ├── UserProfile.js    # Onboarding
│   │   └── AuthPage.js
│   └── services/api.js
│
└── training/                 # Entraînement Kaggle
    ├── calorie_ai_training.ipynb
    └── 1_prepare_dataset.py
```

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- Node.js 18+
- PostgreSQL

### Backend

```bash
cd backend
pip install fastapi uvicorn sqlalchemy psycopg2-binary \
            passlib python-jose pillow numpy tensorflow

# Configurer la base de données
export DATABASE_URL="postgresql://user:password@localhost/calorieIA"

# Lancer
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

L'app est accessible sur `http://localhost:3000`

---

## 🤖 Modèle ML

Le modèle est entraîné sur **Kaggle** avec GPU P100 :

| Paramètre | Valeur |
|---|---|
| Architecture | MobileNetV2 + Transfer Learning |
| Dataset | Food-101 (32 classes) + scraping tunisien (38 classes) |
| Images totales | ~17 500 |
| Taille d'entrée | 224 × 224 px |
| Précision | ~70% top-1 · ~90% top-3 |
| Durée entraînement | ~45 min sur GPU P100 |

### Entraîner votre propre modèle

1. Ouvrez `calorie_ai_training.ipynb` sur [Kaggle](https://kaggle.com)
2. Ajoutez le dataset `dansbecker/food-101`
3. Activez **GPU P100** dans les settings
4. **Run All** (~45 min)
5. Téléchargez `model_export.zip` → décompressez dans `backend/ml_model/`

---

## 📡 API Endpoints

```
POST  /auth/register        Créer un compte
POST  /auth/login           Connexion
GET   /auth/me              Profil courant

PUT   /profile              Mettre à jour le profil nutritionnel

POST  /analyze              Analyser une image (multipart/form-data)
                            Body: file (image), quantity_g (optionnel)

GET   /meals                Historique des repas
POST  /meals                Enregistrer un repas
```

### Exemple `/analyze`

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@couscous.jpg" \
  -F "quantity_g=450"
```

```json
{
  "dish": "Couscous Tunisien",
  "confidence": 87,
  "total_calories": 612,
  "portion_multiplier": 1.1,
  "source": "local_model",
  "ingredients": [
    { "name": "semoule de couscous", "portion_g": 138, "calories": 155 },
    { "name": "poulet",              "portion_g": 99,  "calories": 163 },
    { "name": "carottes",            "portion_g": 50,  "calories": 21  },
    { "name": "pois chiches",        "portion_g": 50,  "calories": 70  }
  ]
}
```

---

## 🗄️ Base de données

```sql
-- Users
id · email · password_hash · name · gender · age
weight · height · activity_level · goal
target_calories · bmr · tdee · profile_completed

-- Meals
id · user_id · dish · total_calories · source
quantity_g · portion_multiplier · ingredients (JSON)
note · consumed_at · created_at
```

---

## 🛠️ Stack technique

**Backend** — FastAPI · SQLAlchemy · PostgreSQL · TensorFlow · Pillow · python-jose · Passlib

**Frontend** — React · Axios · react-dropzone · CSS custom (no UI framework)

**ML** — MobileNetV2 · ImageDataGenerator · Transfer Learning · TFLite export

**Infra entraînement** — Kaggle Notebooks · GPU P100 · icrawler (Bing + Google)

---

## 📸 Screenshots

> *À venir*

---

## 📄 Licence

MIT — libre d'utilisation, modification et distribution.

---

<p align="center">
  Fait avec ❤️ en Tunisie 🇹🇳
</p>

