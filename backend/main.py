"""
main.py — Calorie AI
====================
Backend FastAPI utilisant le modèle MobileNetV2 entraîné localement.
Plus de Gemini / OpenRouter — 100 % local, 0 latence réseau.

Structure attendue :
  backend/
  ├── main.py          ← ce fichier
  ├── predictor.py     ← moteur ML
  ├── database.py
  ├── models.py
  ├── schemas.py
  ├── security.py
  └── ml_model/
      ├── food_model.h5
      └── class_names.json

Lancement :
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from datetime import datetime

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import traceback
import os
from sqlalchemy.orm import Session

# ── Imports internes (relatif ou absolu selon votre setup) ────────
try:
    from .database import Base, SessionLocal, engine
    from .models   import Meal, User
    from .schemas  import LoginRequest, MealCreateRequest, ProfileUpdateRequest, RegisterRequest
    from .security import create_access_token, decode_access_token, hash_password, verify_password
    from .predictor import predict as ml_predict
except ImportError:
    from database  import Base, SessionLocal, engine
    from models    import Meal, User
    from schemas   import LoginRequest, MealCreateRequest, ProfileUpdateRequest, RegisterRequest
    from security  import create_access_token, decode_access_token, hash_password, verify_password
    from predictor import predict as ml_predict

# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="Calorie AI", version="3.0.0 — Local ML")
security_scheme = HTTPBearer(auto_error=False)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════
#  DB DEPENDENCY
# ══════════════════════════════════════════════════════════════════

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ══════════════════════════════════════════════════════════════════
#  SERIALIZERS
# ══════════════════════════════════════════════════════════════════

def serialize_user(user: User) -> dict:
    return {
        "id":               user.id,
        "email":            user.email,
        "name":             user.name,
        "gender":           user.gender,
        "age":              user.age,
        "weight":           user.weight,
        "height":           user.height,
        "activityLevel":    user.activity_level,
        "goal":             user.goal,
        "targetCalories":   user.target_calories,
        "bmr":              user.bmr,
        "tdee":             user.tdee,
        "profileCompleted": user.profile_completed,
        "createdAt":        user.created_at.isoformat(),
    }


def serialize_meal(meal: Meal) -> dict:
    return {
        "id":                meal.id,
        "dish":              meal.dish,
        "total_calories":    meal.total_calories,
        "source":            meal.source,
        "quantity_g":        meal.quantity_g,
        "portion_multiplier":meal.portion_multiplier,
        "ingredients":       meal.ingredients or [],
        "note":              meal.note,
        "consumed_at":       meal.consumed_at.isoformat(),
        "created_at":        meal.created_at.isoformat(),
    }

# ══════════════════════════════════════════════════════════════════
#  AUTH DEPENDENCY
# ══════════════════════════════════════════════════════════════════

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authentification requise.")
    email = decode_access_token(credentials.credentials)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token invalide ou expiré.")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Utilisateur introuvable.")
    return user

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def apply_quantity_override(result: dict, quantity_g: float | None) -> dict:
    """Recalcule les portions si l'utilisateur a saisi un poids réel."""
    if not quantity_g or quantity_g <= 0:
        return result

    ingredients   = result.get("ingredients") or []
    estimated_g   = sum(max(float(i.get("portion_g", 0)), 0) for i in ingredients)
    if estimated_g <= 0:
        return result

    scale = quantity_g / estimated_g
    adjusted = []
    for item in ingredients:
        adjusted.append({
            **item,
            "portion_g": round(float(item.get("portion_g", 0)) * scale),
            "calories":  round(float(item.get("calories",  0)) * scale),
        })

    result["ingredients"]         = adjusted
    result["total_calories"]      = sum(i["calories"] for i in adjusted)
    result["requested_portion_g"] = round(quantity_g)
    result["estimated_portion_g"] = round(estimated_g)
    result["portion_multiplier"]  = round(
        float(result.get("portion_multiplier", 1.0)) * scale, 2
    )
    return result

# ══════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=400, detail="Cet email existe déjà.")

    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        name=payload.name.strip() if payload.name else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"token": create_access_token(user.email), "user": serialize_user(user)}


@app.post("/auth/login")
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")
    return {"token": create_access_token(user.email), "user": serialize_user(user)}


@app.get("/auth/me")
async def auth_me(current_user: User = Depends(get_current_user)):
    return {"user": serialize_user(current_user)}

# ══════════════════════════════════════════════════════════════════
#  PROFILE ROUTE
# ══════════════════════════════════════════════════════════════════

@app.put("/profile")
async def update_profile(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.name           = payload.name.strip() if payload.name else current_user.name
    current_user.gender         = payload.gender
    current_user.age            = payload.age
    current_user.weight         = payload.weight
    current_user.height         = payload.height
    current_user.activity_level = payload.activityLevel
    current_user.goal           = payload.goal
    current_user.target_calories= payload.targetCalories
    current_user.bmr            = payload.bmr
    current_user.tdee           = payload.tdee
    current_user.profile_completed = True

    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return {"user": serialize_user(current_user)}

# ══════════════════════════════════════════════════════════════════
#  MEALS ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/meals")
async def list_meals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meals = (
        db.query(Meal)
        .filter(Meal.user_id == current_user.id)
        .order_by(Meal.consumed_at.desc())
        .all()
    )
    return {"meals": [serialize_meal(m) for m in meals]}


@app.post("/meals")
async def create_meal(
    payload: MealCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meal = Meal(
        user_id=current_user.id,
        dish=payload.dish,
        total_calories=payload.total_calories,
        source=payload.source,
        quantity_g=payload.quantity_g,
        portion_multiplier=payload.portion_multiplier,
        ingredients=payload.ingredients,
        note=payload.note,
        consumed_at=payload.consumed_at or datetime.utcnow(),
    )
    db.add(meal)
    db.commit()
    db.refresh(meal)
    return {"meal": serialize_meal(meal)}

# ══════════════════════════════════════════════════════════════════
#  ANALYSE ROUTE  ← cœur de l'application
# ══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "Calorie AI v3 — Modèle local MobileNetV2",
        "status":  "running",
        "engine":  "local_ml",
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile,
    quantity_g: float | None = Form(default=None),
):
    try:
        print(f"\n📸 {file.filename}")
        image_bytes = await file.read()
        print(f"✅ {len(image_bytes):,} bytes")

        # ── Prédiction locale (MobileNetV2) ───────────────────────
        print("🤖 Prédiction locale en cours…")
        result = ml_predict(image_bytes, quantity_g=quantity_g)

        # ── Ajustement par quantité saisie (si pas déjà fait) ─────
        if quantity_g and "requested_portion_g" not in result:
            result = apply_quantity_override(result, quantity_g)

        print(f"🍽️  {result['dish']} | {result['total_calories']} kcal "
              f"| confiance: {result['confidence']}% "
              f"| source: {result.get('source', 'local_model')}")

        return result

    except Exception as e:
        print(f"❌ Erreur fatale:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})