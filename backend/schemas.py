from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdateRequest(BaseModel):
    name:           str | None = None
    gender:         str
    age:            int
    weight:         float
    height:         float
    activityLevel:  str
    goal:           str
    targetCalories: int
    bmr:            int
    tdee:           int


class MealCreateRequest(BaseModel):
    dish:               str                     # ← ajouté (obligatoire)
    total_calories:     int
    source:             str | None = None
    quantity_g:         int | None = None
    portion_multiplier: float | None = None
    ingredients:        list[dict[str, Any]] = Field(default_factory=list)
    consumed_at:        datetime | None = None
    note:               str | None = None