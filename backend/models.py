from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

try:
    from .database import Base
except ImportError:
    from database import Base


class User(Base):
    __tablename__ = "users"

    id                 = Column(Integer, primary_key=True, index=True)
    email              = Column(String(255), unique=True, index=True, nullable=False)
    password_hash      = Column(String(255), nullable=False)
    name               = Column(String(120), nullable=True)
    gender             = Column(String(20), nullable=True)
    age                = Column(Integer, nullable=True)
    weight             = Column(Float, nullable=True)
    height             = Column(Float, nullable=True)
    activity_level     = Column(String(50), nullable=True)
    goal               = Column(String(50), nullable=True)
    target_calories    = Column(Integer, nullable=True)
    bmr                = Column(Integer, nullable=True)
    tdee               = Column(Integer, nullable=True)
    profile_completed  = Column(Boolean, nullable=False, default=False)
    created_at         = Column(DateTime, default=datetime.utcnow, nullable=False)

    meals = relationship("Meal", back_populates="user", cascade="all, delete-orphan")


class Meal(Base):
    __tablename__ = "meals"

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    dish               = Column(String(255), nullable=False)
    total_calories     = Column(Integer, nullable=False)
    source             = Column(String(50), nullable=True)
    quantity_g         = Column(Integer, nullable=True)       # ← ajouté
    portion_multiplier = Column(Float, nullable=True)
    ingredients        = Column(JSON, nullable=True)
    note               = Column(Text, nullable=True)
    consumed_at        = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at         = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="meals")