
import json
import io
import numpy as np
from pathlib import Path
from PIL import Image

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
MODEL_DIR  = Path(__file__).parent / "ml_model"
MODEL_PATH = MODEL_DIR / "food_model.h5"
NAMES_PATH = MODEL_DIR / "class_names.json"
IMG_SIZE   = 224
TOP_K      = 3
MIN_CONF   = 0.25

# ══════════════════════════════════════════════
#  BASE DE DONNÉES NUTRITIONNELLE
# ══════════════════════════════════════════════
FOOD_DATABASE: dict = {
    # ── Plats tunisiens ───────────────────────
    "couscous_tunisien": {
        "display": "Couscous Tunisien",
        "base_portion": 550,
        "ingredients": [
            {"name": "semoule de couscous",     "cal_per_g": 1.12, "pct": 0.28},
            {"name": "poulet",                  "cal_per_g": 1.65, "pct": 0.20},
            {"name": "carottes",                "cal_per_g": 0.41, "pct": 0.10},
            {"name": "courgettes",              "cal_per_g": 0.17, "pct": 0.10},
            {"name": "pois chiches",            "cal_per_g": 1.39, "pct": 0.10},
            {"name": "tomates",                 "cal_per_g": 0.18, "pct": 0.07},
            {"name": "oignons",                 "cal_per_g": 0.40, "pct": 0.05},
            {"name": "huile d'olive",           "cal_per_g": 8.84, "pct": 0.03},
            {"name": "épices (harissa, cumin)", "cal_per_g": 2.50, "pct": 0.02},
            {"name": "bouillon",                "cal_per_g": 0.05, "pct": 0.05},
        ],
    },
    "macarouna_tunisien": {
        "display": "Macarouna Tunisien",
        "base_portion": 450,
        "ingredients": [
            {"name": "pâtes",               "cal_per_g": 1.31, "pct": 0.40},
            {"name": "sauce tomate épicée", "cal_per_g": 0.60, "pct": 0.25},
            {"name": "viande hachée",       "cal_per_g": 2.50, "pct": 0.20},
            {"name": "harissa",             "cal_per_g": 0.50, "pct": 0.05},
            {"name": "huile d'olive",       "cal_per_g": 8.84, "pct": 0.04},
            {"name": "ail + épices",        "cal_per_g": 1.50, "pct": 0.03},
            {"name": "fromage",             "cal_per_g": 4.02, "pct": 0.03},
        ],
    },
    "tajine_tunisien": {
        "display": "Tajine Tunisien (au four)",
        "base_portion": 400,
        "ingredients": [
            {"name": "œufs",          "cal_per_g": 1.55, "pct": 0.30},
            {"name": "viande hachée", "cal_per_g": 2.50, "pct": 0.25},
            {"name": "fromage",       "cal_per_g": 4.02, "pct": 0.15},
            {"name": "persil",        "cal_per_g": 0.36, "pct": 0.05},
            {"name": "oignons",       "cal_per_g": 0.40, "pct": 0.10},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.05},
            {"name": "épices",        "cal_per_g": 2.50, "pct": 0.03},
            {"name": "chapelure",     "cal_per_g": 3.95, "pct": 0.07},
        ],
    },
    "tagine_merguez": {
        "display": "Tagine Merguez",
        "base_portion": 400,
        "ingredients": [
            {"name": "merguez",       "cal_per_g": 3.00, "pct": 0.35},
            {"name": "œufs",          "cal_per_g": 1.55, "pct": 0.20},
            {"name": "poivrons",      "cal_per_g": 0.20, "pct": 0.15},
            {"name": "tomates",       "cal_per_g": 0.18, "pct": 0.15},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.05},
            {"name": "épices",        "cal_per_g": 2.50, "pct": 0.05},
            {"name": "harissa",       "cal_per_g": 0.50, "pct": 0.05},
        ],
    },
    "tagine_marocain": {
        "display": "Tagine Marocain",
        "base_portion": 500,
        "ingredients": [
            {"name": "agneau",        "cal_per_g": 2.94, "pct": 0.30},
            {"name": "pruneaux",      "cal_per_g": 2.40, "pct": 0.10},
            {"name": "amandes",       "cal_per_g": 5.79, "pct": 0.05},
            {"name": "oignons",       "cal_per_g": 0.40, "pct": 0.15},
            {"name": "carottes",      "cal_per_g": 0.41, "pct": 0.15},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.05},
            {"name": "épices (ras el hanout)", "cal_per_g": 2.50, "pct": 0.05},
            {"name": "bouillon",      "cal_per_g": 0.05, "pct": 0.15},
        ],
    },
    "brik": {
        "display": "Brik Tunisien",
        "base_portion": 200,
        "ingredients": [
            {"name": "feuille de brick", "cal_per_g": 3.30, "pct": 0.25},
            {"name": "œuf",              "cal_per_g": 1.55, "pct": 0.30},
            {"name": "thon",             "cal_per_g": 1.16, "pct": 0.20},
            {"name": "câpres",           "cal_per_g": 0.23, "pct": 0.05},
            {"name": "persil",           "cal_per_g": 0.36, "pct": 0.05},
            {"name": "huile de friture", "cal_per_g": 8.84, "pct": 0.10},
            {"name": "harissa",          "cal_per_g": 0.50, "pct": 0.05},
        ],
    },
    "lablabi": {
        "display": "Lablabi",
        "base_portion": 500,
        "ingredients": [
            {"name": "pois chiches",  "cal_per_g": 1.39, "pct": 0.35},
            {"name": "pain rassis",   "cal_per_g": 2.65, "pct": 0.20},
            {"name": "œuf",           "cal_per_g": 1.55, "pct": 0.10},
            {"name": "thon",          "cal_per_g": 1.16, "pct": 0.10},
            {"name": "harissa",       "cal_per_g": 0.50, "pct": 0.05},
            {"name": "citron",        "cal_per_g": 0.29, "pct": 0.03},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.05},
            {"name": "cumin",         "cal_per_g": 3.75, "pct": 0.02},
            {"name": "bouillon",      "cal_per_g": 0.05, "pct": 0.10},
        ],
    },
    "leblebi": {
        "display": "Leblebi",
        "base_portion": 500,
        "ingredients": [
            {"name": "pois chiches",  "cal_per_g": 1.39, "pct": 0.35},
            {"name": "pain rassis",   "cal_per_g": 2.65, "pct": 0.20},
            {"name": "œuf",           "cal_per_g": 1.55, "pct": 0.10},
            {"name": "thon",          "cal_per_g": 1.16, "pct": 0.10},
            {"name": "harissa",       "cal_per_g": 0.50, "pct": 0.05},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.05},
            {"name": "cumin",         "cal_per_g": 3.75, "pct": 0.02},
            {"name": "bouillon",      "cal_per_g": 0.05, "pct": 0.13},
        ],
    },
    "mechouia": {
        "display": "Salade Mechouia",
        "base_portion": 300,
        "ingredients": [
            {"name": "poivrons grillés", "cal_per_g": 0.31, "pct": 0.35},
            {"name": "tomates grillées", "cal_per_g": 0.18, "pct": 0.30},
            {"name": "oignons grillés",  "cal_per_g": 0.40, "pct": 0.10},
            {"name": "piment",           "cal_per_g": 0.40, "pct": 0.05},
            {"name": "huile d'olive",    "cal_per_g": 8.84, "pct": 0.08},
            {"name": "ail",              "cal_per_g": 1.49, "pct": 0.03},
            {"name": "thon",             "cal_per_g": 1.16, "pct": 0.05},
            {"name": "câpres + olives",  "cal_per_g": 1.15, "pct": 0.04},
        ],
    },
    "slata_tunisienne": {
        "display": "Slata Tunisienne",
        "base_portion": 250,
        "ingredients": [
            {"name": "tomates",       "cal_per_g": 0.18, "pct": 0.35},
            {"name": "concombre",     "cal_per_g": 0.15, "pct": 0.25},
            {"name": "oignons",       "cal_per_g": 0.40, "pct": 0.10},
            {"name": "poivrons",      "cal_per_g": 0.20, "pct": 0.10},
            {"name": "olives",        "cal_per_g": 1.15, "pct": 0.05},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.08},
            {"name": "citron",        "cal_per_g": 0.29, "pct": 0.04},
            {"name": "thon",          "cal_per_g": 1.16, "pct": 0.03},
        ],
    },
    "ojja": {
        "display": "Ojja Merguez",
        "base_portion": 400,
        "ingredients": [
            {"name": "merguez",       "cal_per_g": 3.00, "pct": 0.30},
            {"name": "œufs",          "cal_per_g": 1.55, "pct": 0.20},
            {"name": "tomates",       "cal_per_g": 0.18, "pct": 0.25},
            {"name": "poivrons",      "cal_per_g": 0.20, "pct": 0.10},
            {"name": "harissa",       "cal_per_g": 0.50, "pct": 0.05},
            {"name": "huile d'olive", "cal_per_g": 8.84, "pct": 0.05},
            {"name": "ail + épices",  "cal_per_g": 1.50, "pct": 0.05},
        ],
    },
    "chorba": {
        "display": "Chorba Frik",
        "base_portion": 500,
        "ingredients": [
            {"name": "frik (blé vert)",  "cal_per_g": 3.38, "pct": 0.20},
            {"name": "agneau",           "cal_per_g": 2.94, "pct": 0.20},
            {"name": "tomates",          "cal_per_g": 0.18, "pct": 0.15},
            {"name": "pois chiches",     "cal_per_g": 1.39, "pct": 0.10},
            {"name": "céleri + oignons", "cal_per_g": 0.30, "pct": 0.10},
            {"name": "concentré tomate", "cal_per_g": 0.82, "pct": 0.05},
            {"name": "huile d'olive",    "cal_per_g": 8.84, "pct": 0.04},
            {"name": "épices + persil",  "cal_per_g": 1.50, "pct": 0.05},
            {"name": "bouillon",         "cal_per_g": 0.05, "pct": 0.11},
        ],
    },
    "harira": {
        "display": "Harira",
        "base_portion": 500,
        "ingredients": [
            {"name": "lentilles",        "cal_per_g": 1.16, "pct": 0.20},
            {"name": "pois chiches",     "cal_per_g": 1.39, "pct": 0.15},
            {"name": "agneau",           "cal_per_g": 2.94, "pct": 0.15},
            {"name": "tomates",          "cal_per_g": 0.18, "pct": 0.20},
            {"name": "céleri + oignons", "cal_per_g": 0.30, "pct": 0.10},
            {"name": "farine + vermicelles", "cal_per_g": 3.00, "pct": 0.08},
            {"name": "huile d'olive",    "cal_per_g": 8.84, "pct": 0.04},
            {"name": "épices (cumin, gingembre)", "cal_per_g": 2.50, "pct": 0.03},
            {"name": "coriandre + persil", "cal_per_g": 0.23, "pct": 0.05},
        ],
    },
    "kafteji": {
        "display": "Kafteji",
        "base_portion": 400,
        "ingredients": [
            {"name": "pommes de terre frites", "cal_per_g": 3.12, "pct": 0.30},
            {"name": "poivrons frits",          "cal_per_g": 0.60, "pct": 0.15},
            {"name": "courgettes frites",       "cal_per_g": 0.50, "pct": 0.10},
            {"name": "œufs",                    "cal_per_g": 1.55, "pct": 0.15},
            {"name": "tomates",                 "cal_per_g": 0.18, "pct": 0.10},
            {"name": "huile de friture",        "cal_per_g": 8.84, "pct": 0.12},
            {"name": "harissa + épices",        "cal_per_g": 0.80, "pct": 0.05},
            {"name": "citron",                  "cal_per_g": 0.29, "pct": 0.03},
        ],
    },
    "fricasse_tunisien": {
        "display": "Fricassée Tunisienne",
        "base_portion": 250,
        "ingredients": [
            {"name": "pain brioché frit", "cal_per_g": 3.50, "pct": 0.35},
            {"name": "thon",              "cal_per_g": 1.16, "pct": 0.20},
            {"name": "œuf dur",           "cal_per_g": 1.55, "pct": 0.10},
            {"name": "olives",            "cal_per_g": 1.15, "pct": 0.05},
            {"name": "câpres",            "cal_per_g": 0.23, "pct": 0.03},
            {"name": "harissa",           "cal_per_g": 0.50, "pct": 0.05},
            {"name": "pommes de terre",   "cal_per_g": 0.77, "pct": 0.10},
            {"name": "huile d'olive",     "cal_per_g": 8.84, "pct": 0.07},
            {"name": "citron",            "cal_per_g": 0.29, "pct": 0.05},
        ],
    },
    "shakshuka": {
        "display": "Shakshuka",
        "base_portion": 400,
        "ingredients": [
            {"name": "œufs",            "cal_per_g": 1.55, "pct": 0.25},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.35},
            {"name": "poivrons",        "cal_per_g": 0.20, "pct": 0.15},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.08},
            {"name": "ail",             "cal_per_g": 1.49, "pct": 0.03},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.06},
            {"name": "épices",          "cal_per_g": 2.50, "pct": 0.03},
            {"name": "harissa",         "cal_per_g": 0.50, "pct": 0.05},
        ],
    },
    "kefta_tunisienne": {
        "display": "Kefta Tunisienne",
        "base_portion": 350,
        "ingredients": [
            {"name": "viande hachée",   "cal_per_g": 2.50, "pct": 0.50},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.10},
            {"name": "persil",          "cal_per_g": 0.36, "pct": 0.05},
            {"name": "épices",          "cal_per_g": 2.50, "pct": 0.05},
            {"name": "pain",            "cal_per_g": 2.65, "pct": 0.15},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.05},
            {"name": "harissa",         "cal_per_g": 0.50, "pct": 0.05},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.05},
        ],
    },
    "merguez_grillee": {
        "display": "Merguez Grillée",
        "base_portion": 300,
        "ingredients": [
            {"name": "merguez",         "cal_per_g": 3.00, "pct": 0.60},
            {"name": "pain",            "cal_per_g": 2.65, "pct": 0.20},
            {"name": "harissa",         "cal_per_g": 0.50, "pct": 0.05},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.08},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.05},
            {"name": "moutarde",        "cal_per_g": 0.66, "pct": 0.02},
        ],
    },
    "mloukhia": {
        "display": "Mloukhia",
        "base_portion": 450,
        "ingredients": [
            {"name": "feuilles mloukhia séchées", "cal_per_g": 2.80, "pct": 0.15},
            {"name": "viande (agneau/bœuf)",       "cal_per_g": 2.50, "pct": 0.30},
            {"name": "huile d'olive",              "cal_per_g": 8.84, "pct": 0.10},
            {"name": "ail",                        "cal_per_g": 1.49, "pct": 0.05},
            {"name": "harissa + épices",           "cal_per_g": 1.00, "pct": 0.05},
            {"name": "bouillon",                   "cal_per_g": 0.05, "pct": 0.20},
            {"name": "pain ou couscous",           "cal_per_g": 2.00, "pct": 0.15},
        ],
    },
    "osban": {
        "display": "Osban (Saucisse Tunisienne)",
        "base_portion": 350,
        "ingredients": [
            {"name": "tripes/abats",    "cal_per_g": 1.50, "pct": 0.30},
            {"name": "riz",             "cal_per_g": 1.30, "pct": 0.25},
            {"name": "herbes",          "cal_per_g": 0.30, "pct": 0.10},
            {"name": "épices",          "cal_per_g": 2.50, "pct": 0.05},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.05},
            {"name": "sauce tomate",    "cal_per_g": 0.60, "pct": 0.15},
            {"name": "bouillon",        "cal_per_g": 0.05, "pct": 0.10},
        ],
    },
    "camounia": {
        "display": "Kamounia",
        "base_portion": 400,
        "ingredients": [
            {"name": "foie/viande",     "cal_per_g": 1.75, "pct": 0.40},
            {"name": "cumin",           "cal_per_g": 3.75, "pct": 0.05},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.20},
            {"name": "ail",             "cal_per_g": 1.49, "pct": 0.05},
            {"name": "harissa",         "cal_per_g": 0.50, "pct": 0.05},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.08},
            {"name": "épices",          "cal_per_g": 2.50, "pct": 0.05},
            {"name": "bouillon",        "cal_per_g": 0.05, "pct": 0.12},
        ],
    },
    "hraimi": {
        "display": "Hraimi (Poisson Épicé)",
        "base_portion": 400,
        "ingredients": [
            {"name": "poisson",         "cal_per_g": 1.50, "pct": 0.45},
            {"name": "sauce harissa",   "cal_per_g": 0.50, "pct": 0.10},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.20},
            {"name": "ail",             "cal_per_g": 1.49, "pct": 0.05},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.08},
            {"name": "carvi + cumin",   "cal_per_g": 3.00, "pct": 0.04},
            {"name": "citron",          "cal_per_g": 0.29, "pct": 0.05},
            {"name": "bouillon",        "cal_per_g": 0.05, "pct": 0.03},
        ],
    },
    "poisson_frit_tunisien": {
        "display": "Poisson Frit Tunisien",
        "base_portion": 350,
        "ingredients": [
            {"name": "poisson (friture)", "cal_per_g": 1.80, "pct": 0.55},
            {"name": "farine/chapelure",  "cal_per_g": 3.64, "pct": 0.10},
            {"name": "huile de friture",  "cal_per_g": 8.84, "pct": 0.15},
            {"name": "citron",            "cal_per_g": 0.29, "pct": 0.05},
            {"name": "harissa",           "cal_per_g": 0.50, "pct": 0.05},
            {"name": "salade",            "cal_per_g": 0.15, "pct": 0.10},
        ],
    },
    "dolma_tunisien": {
        "display": "Dolma Tunisien",
        "base_portion": 400,
        "ingredients": [
            {"name": "feuilles de vigne/légumes farcis", "cal_per_g": 0.50, "pct": 0.30},
            {"name": "riz",             "cal_per_g": 1.30, "pct": 0.20},
            {"name": "viande hachée",   "cal_per_g": 2.50, "pct": 0.25},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.10},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.05},
            {"name": "épices + herbes", "cal_per_g": 1.50, "pct": 0.05},
            {"name": "bouillon",        "cal_per_g": 0.05, "pct": 0.05},
        ],
    },
    "makloub_tunisien": {
        "display": "Makloub Tunisien",
        "base_portion": 450,
        "ingredients": [
            {"name": "pain (malsouka)", "cal_per_g": 2.65, "pct": 0.30},
            {"name": "thon",            "cal_per_g": 1.16, "pct": 0.20},
            {"name": "œufs",            "cal_per_g": 1.55, "pct": 0.15},
            {"name": "fromage",         "cal_per_g": 4.02, "pct": 0.10},
            {"name": "harissa",         "cal_per_g": 0.50, "pct": 0.05},
            {"name": "olives",          "cal_per_g": 1.15, "pct": 0.05},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.05},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.10},
        ],
    },
    "marqa": {
        "display": "Marqa (Ragoût Tunisien)",
        "base_portion": 450,
        "ingredients": [
            {"name": "viande (agneau/bœuf)", "cal_per_g": 2.50, "pct": 0.30},
            {"name": "légumes variés",        "cal_per_g": 0.40, "pct": 0.25},
            {"name": "sauce tomate",          "cal_per_g": 0.60, "pct": 0.20},
            {"name": "pois chiches",          "cal_per_g": 1.39, "pct": 0.10},
            {"name": "huile d'olive",         "cal_per_g": 8.84, "pct": 0.05},
            {"name": "épices + harissa",      "cal_per_g": 1.50, "pct": 0.05},
            {"name": "bouillon",              "cal_per_g": 0.05, "pct": 0.05},
        ],
    },
    "mosli": {
        "display": "Mosli (Agneau Rôti)",
        "base_portion": 400,
        "ingredients": [
            {"name": "agneau rôti",     "cal_per_g": 2.94, "pct": 0.60},
            {"name": "pommes de terre", "cal_per_g": 0.77, "pct": 0.20},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.08},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.05},
            {"name": "épices",          "cal_per_g": 2.50, "pct": 0.04},
            {"name": "ail",             "cal_per_g": 1.49, "pct": 0.03},
        ],
    },
    "pkaila": {
        "display": "Pkaila",
        "base_portion": 400,
        "ingredients": [
            {"name": "épinards/oseille", "cal_per_g": 0.23, "pct": 0.30},
            {"name": "viande (agneau)",  "cal_per_g": 2.94, "pct": 0.30},
            {"name": "haricots blancs",  "cal_per_g": 1.27, "pct": 0.15},
            {"name": "huile d'olive",    "cal_per_g": 8.84, "pct": 0.10},
            {"name": "ail + épices",     "cal_per_g": 1.50, "pct": 0.05},
            {"name": "bouillon",         "cal_per_g": 0.05, "pct": 0.10},
        ],
    },
    "rechta": {
        "display": "Rechta",
        "base_portion": 450,
        "ingredients": [
            {"name": "pâtes rechta",    "cal_per_g": 1.31, "pct": 0.35},
            {"name": "poulet",          "cal_per_g": 1.65, "pct": 0.25},
            {"name": "pois chiches",    "cal_per_g": 1.39, "pct": 0.10},
            {"name": "navet",           "cal_per_g": 0.28, "pct": 0.10},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.05},
            {"name": "cannelle",        "cal_per_g": 2.47, "pct": 0.02},
            {"name": "bouillon",        "cal_per_g": 0.05, "pct": 0.13},
        ],
    },
    "mansaf": {
        "display": "Mansaf",
        "base_portion": 550,
        "ingredients": [
            {"name": "agneau",          "cal_per_g": 2.94, "pct": 0.30},
            {"name": "riz",             "cal_per_g": 1.30, "pct": 0.30},
            {"name": "jameed (yaourt fermenté)", "cal_per_g": 1.00, "pct": 0.20},
            {"name": "amandes + pignons", "cal_per_g": 5.79, "pct": 0.05},
            {"name": "pain (shrak)",    "cal_per_g": 2.65, "pct": 0.10},
            {"name": "beurre clarifié", "cal_per_g": 7.17, "pct": 0.05},
        ],
    },
    "chakhchoukha": {
        "display": "Chakhchoukha",
        "base_portion": 500,
        "ingredients": [
            {"name": "galettes rompues (rougag)", "cal_per_g": 3.00, "pct": 0.35},
            {"name": "agneau",          "cal_per_g": 2.94, "pct": 0.25},
            {"name": "pois chiches",    "cal_per_g": 1.39, "pct": 0.10},
            {"name": "sauce tomate",    "cal_per_g": 0.60, "pct": 0.15},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.05},
            {"name": "épices (ras el hanout)", "cal_per_g": 2.50, "pct": 0.03},
            {"name": "bouillon",        "cal_per_g": 0.05, "pct": 0.07},
        ],
    },
    "harissa": {
        "display": "Harissa",
        "base_portion": 50,
        "ingredients": [
            {"name": "piments rouges",  "cal_per_g": 0.40, "pct": 0.60},
            {"name": "ail",             "cal_per_g": 1.49, "pct": 0.10},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.20},
            {"name": "sel + épices",    "cal_per_g": 0.10, "pct": 0.10},
        ],
    },
    "rouz_jben": {
        "display": "Rouz Jben (Riz au Fromage)",
        "base_portion": 400,
        "ingredients": [
            {"name": "riz",             "cal_per_g": 1.30, "pct": 0.50},
            {"name": "jben (fromage frais tunisien)", "cal_per_g": 1.00, "pct": 0.25},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.08},
            {"name": "lait",            "cal_per_g": 0.42, "pct": 0.10},
            {"name": "sucre",           "cal_per_g": 3.87, "pct": 0.05},
            {"name": "eau de fleur d'oranger", "cal_per_g": 0.01, "pct": 0.02},
        ],
    },
    "asida": {
        "display": "Assida",
        "base_portion": 350,
        "ingredients": [
            {"name": "semoule/farine",  "cal_per_g": 3.64, "pct": 0.50},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.15},
            {"name": "miel",            "cal_per_g": 3.04, "pct": 0.15},
            {"name": "noix/amandes",    "cal_per_g": 6.54, "pct": 0.10},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.10},
        ],
    },
    "assida_zgougou": {
        "display": "Assida Zgougou",
        "base_portion": 300,
        "ingredients": [
            {"name": "graines de pin (zgougou)", "cal_per_g": 6.73, "pct": 0.30},
            {"name": "semoule",         "cal_per_g": 3.64, "pct": 0.25},
            {"name": "sucre",           "cal_per_g": 3.87, "pct": 0.15},
            {"name": "crème/lait",      "cal_per_g": 1.50, "pct": 0.20},
            {"name": "pignons de pin",  "cal_per_g": 6.73, "pct": 0.05},
            {"name": "eau de fleur d'oranger", "cal_per_g": 0.01, "pct": 0.05},
        ],
    },
    "bambalouni": {
        "display": "Bambalouni (Beignet Tunisien)",
        "base_portion": 200,
        "ingredients": [
            {"name": "pâte frite (farine, levure)", "cal_per_g": 3.50, "pct": 0.55},
            {"name": "huile de friture", "cal_per_g": 8.84, "pct": 0.20},
            {"name": "sucre glace",      "cal_per_g": 3.87, "pct": 0.15},
            {"name": "sel",              "cal_per_g": 0.00, "pct": 0.02},
            {"name": "levure",           "cal_per_g": 1.05, "pct": 0.08},
        ],
    },
    "ftair": {
        "display": "Ftair (Galette Tunisienne)",
        "base_portion": 250,
        "ingredients": [
            {"name": "farine",          "cal_per_g": 3.64, "pct": 0.45},
            {"name": "beurre/huile",    "cal_per_g": 7.17, "pct": 0.20},
            {"name": "œuf",             "cal_per_g": 1.55, "pct": 0.10},
            {"name": "fromage/viande",  "cal_per_g": 2.50, "pct": 0.15},
            {"name": "épices",          "cal_per_g": 2.50, "pct": 0.05},
            {"name": "sel",             "cal_per_g": 0.00, "pct": 0.05},
        ],
    },
    "mlawi": {
        "display": "Mlawi (Crêpe Tunisienne Feuilletée)",
        "base_portion": 200,
        "ingredients": [
            {"name": "farine",          "cal_per_g": 3.64, "pct": 0.50},
            {"name": "beurre/huile",    "cal_per_g": 7.17, "pct": 0.20},
            {"name": "semoule fine",    "cal_per_g": 3.64, "pct": 0.15},
            {"name": "sel",             "cal_per_g": 0.00, "pct": 0.05},
            {"name": "miel (optionnel)","cal_per_g": 3.04, "pct": 0.10},
        ],
    },
    "msemen": {
        "display": "Msemen (Crêpe Carrée)",
        "base_portion": 200,
        "ingredients": [
            {"name": "farine",          "cal_per_g": 3.64, "pct": 0.48},
            {"name": "semoule fine",    "cal_per_g": 3.64, "pct": 0.15},
            {"name": "huile/beurre",    "cal_per_g": 7.17, "pct": 0.20},
            {"name": "sel + levure",    "cal_per_g": 0.30, "pct": 0.07},
            {"name": "miel",            "cal_per_g": 3.04, "pct": 0.10},
        ],
    },
    "chapati_tunisien": {
        "display": "Chapati Tunisien",
        "base_portion": 200,
        "ingredients": [
            {"name": "farine complète", "cal_per_g": 3.40, "pct": 0.55},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.10},
            {"name": "eau",             "cal_per_g": 0.00, "pct": 0.30},
            {"name": "sel",             "cal_per_g": 0.00, "pct": 0.05},
        ],
    },
    "hsou": {
        "display": "Hsou (Bouillie Tunisienne)",
        "base_portion": 300,
        "ingredients": [
            {"name": "farine d'orge",   "cal_per_g": 3.54, "pct": 0.40},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.15},
            {"name": "eau",             "cal_per_g": 0.00, "pct": 0.30},
            {"name": "sel",             "cal_per_g": 0.00, "pct": 0.05},
            {"name": "miel (optionnel)","cal_per_g": 3.04, "pct": 0.10},
        ],
    },
    "pastilla": {
        "display": "Pastilla",
        "base_portion": 350,
        "ingredients": [
            {"name": "pâte filo/brick", "cal_per_g": 4.00, "pct": 0.30},
            {"name": "poulet/pigeon",   "cal_per_g": 1.65, "pct": 0.30},
            {"name": "amandes",         "cal_per_g": 5.79, "pct": 0.15},
            {"name": "œufs",            "cal_per_g": 1.55, "pct": 0.10},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.08},
            {"name": "sucre glace + cannelle", "cal_per_g": 3.87, "pct": 0.05},
            {"name": "épices (safran, gingembre)", "cal_per_g": 2.50, "pct": 0.02},
        ],
    },
    "grombalia_raisin": {
        "display": "Raisin de Grombalia",
        "base_portion": 200,
        "ingredients": [
            {"name": "raisins frais",   "cal_per_g": 0.69, "pct": 1.00},
        ],
    },
    "sandwich_tunisien": {
        "display": "Sandwich Tunisien",
        "base_portion": 300,
        "ingredients": [
            {"name": "pain (baguette/tabouna)", "cal_per_g": 2.65, "pct": 0.30},
            {"name": "thon",            "cal_per_g": 1.16, "pct": 0.20},
            {"name": "œuf dur",         "cal_per_g": 1.55, "pct": 0.10},
            {"name": "harissa",         "cal_per_g": 0.50, "pct": 0.05},
            {"name": "olives",          "cal_per_g": 1.15, "pct": 0.05},
            {"name": "câpres",          "cal_per_g": 0.23, "pct": 0.03},
            {"name": "pommes de terre", "cal_per_g": 0.77, "pct": 0.10},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.07},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.10},
        ],
    },
    # ── Plats internationaux ──────────────────
    "pizza": {
        "display": "Pizza",
        "base_portion": 400,
        "ingredients": [
            {"name": "pâte à pizza",    "cal_per_g": 2.66, "pct": 0.35},
            {"name": "sauce tomate",    "cal_per_g": 0.24, "pct": 0.15},
            {"name": "mozzarella",      "cal_per_g": 2.80, "pct": 0.30},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.05},
            {"name": "garnitures",      "cal_per_g": 2.00, "pct": 0.15},
        ],
    },
    "hamburger": {
        "display": "Hamburger",
        "base_portion": 450,
        "ingredients": [
            {"name": "pain burger",    "cal_per_g": 2.65, "pct": 0.20},
            {"name": "steak haché",    "cal_per_g": 2.50, "pct": 0.30},
            {"name": "fromage",        "cal_per_g": 4.02, "pct": 0.10},
            {"name": "laitue",         "cal_per_g": 0.15, "pct": 0.05},
            {"name": "tomate",         "cal_per_g": 0.18, "pct": 0.08},
            {"name": "frites",         "cal_per_g": 3.12, "pct": 0.17},
            {"name": "sauce",          "cal_per_g": 1.50, "pct": 0.10},
        ],
    },
    "spaghetti_bolognese": {
        "display": "Spaghetti Bolognese",
        "base_portion": 450,
        "ingredients": [
            {"name": "pâtes",           "cal_per_g": 1.31, "pct": 0.40},
            {"name": "bœuf haché",      "cal_per_g": 2.50, "pct": 0.20},
            {"name": "sauce tomate",    "cal_per_g": 0.24, "pct": 0.20},
            {"name": "parmesan",        "cal_per_g": 4.31, "pct": 0.08},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.04},
            {"name": "oignon + ail",    "cal_per_g": 0.60, "pct": 0.05},
            {"name": "herbes",          "cal_per_g": 0.25, "pct": 0.03},
        ],
    },
    "caesar_salad": {
        "display": "Salade César",
        "base_portion": 350,
        "ingredients": [
            {"name": "laitue romaine",  "cal_per_g": 0.17, "pct": 0.40},
            {"name": "poulet grillé",   "cal_per_g": 1.65, "pct": 0.25},
            {"name": "parmesan",        "cal_per_g": 4.31, "pct": 0.08},
            {"name": "croûtons",        "cal_per_g": 4.00, "pct": 0.10},
            {"name": "sauce César",     "cal_per_g": 3.50, "pct": 0.12},
            {"name": "citron",          "cal_per_g": 0.29, "pct": 0.05},
        ],
    },
    "chicken_curry": {
        "display": "Chicken Curry",
        "base_portion": 500,
        "ingredients": [
            {"name": "poulet",          "cal_per_g": 1.65, "pct": 0.30},
            {"name": "riz basmati",     "cal_per_g": 1.30, "pct": 0.30},
            {"name": "lait de coco",    "cal_per_g": 2.30, "pct": 0.15},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.08},
            {"name": "tomates",         "cal_per_g": 0.18, "pct": 0.07},
            {"name": "épices curry",    "cal_per_g": 3.25, "pct": 0.04},
            {"name": "huile",           "cal_per_g": 8.84, "pct": 0.04},
            {"name": "ail + gingembre", "cal_per_g": 0.80, "pct": 0.02},
        ],
    },
    "fried_rice": {
        "display": "Riz Frit",
        "base_portion": 450,
        "ingredients": [
            {"name": "riz cuit",        "cal_per_g": 1.30, "pct": 0.50},
            {"name": "œufs",            "cal_per_g": 1.55, "pct": 0.10},
            {"name": "légumes variés",  "cal_per_g": 0.50, "pct": 0.15},
            {"name": "sauce soja",      "cal_per_g": 0.53, "pct": 0.05},
            {"name": "huile sésame",    "cal_per_g": 8.84, "pct": 0.05},
            {"name": "poulet/crevettes","cal_per_g": 1.50, "pct": 0.10},
            {"name": "oignon vert",     "cal_per_g": 0.32, "pct": 0.05},
        ],
    },
    "steak": {
        "display": "Steak",
        "base_portion": 400,
        "ingredients": [
            {"name": "steak bœuf",      "cal_per_g": 2.50, "pct": 0.55},
            {"name": "frites",          "cal_per_g": 3.12, "pct": 0.25},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.05},
            {"name": "salade verte",    "cal_per_g": 0.15, "pct": 0.10},
            {"name": "sauce",           "cal_per_g": 1.50, "pct": 0.05},
        ],
    },
    "tacos": {
        "display": "Tacos",
        "base_portion": 350,
        "ingredients": [
            {"name": "tortillas",       "cal_per_g": 3.00, "pct": 0.25},
            {"name": "viande épicée",   "cal_per_g": 2.20, "pct": 0.30},
            {"name": "fromage",         "cal_per_g": 4.02, "pct": 0.10},
            {"name": "laitue",          "cal_per_g": 0.15, "pct": 0.08},
            {"name": "tomate",          "cal_per_g": 0.18, "pct": 0.08},
            {"name": "crème fraîche",   "cal_per_g": 1.98, "pct": 0.10},
            {"name": "guacamole",       "cal_per_g": 1.60, "pct": 0.09},
        ],
    },
    "falafel": {
        "display": "Falafel",
        "base_portion": 300,
        "ingredients": [
            {"name": "falafel",         "cal_per_g": 3.33, "pct": 0.45},
            {"name": "pain pita",       "cal_per_g": 2.75, "pct": 0.20},
            {"name": "tahini",          "cal_per_g": 5.95, "pct": 0.10},
            {"name": "salade + tomates","cal_per_g": 0.20, "pct": 0.15},
            {"name": "citron",          "cal_per_g": 0.29, "pct": 0.05},
            {"name": "oignons",         "cal_per_g": 0.40, "pct": 0.05},
        ],
    },
    "hummus": {
        "display": "Hummus",
        "base_portion": 200,
        "ingredients": [
            {"name": "pois chiches",    "cal_per_g": 1.39, "pct": 0.45},
            {"name": "tahini",          "cal_per_g": 5.95, "pct": 0.20},
            {"name": "huile d'olive",   "cal_per_g": 8.84, "pct": 0.10},
            {"name": "citron",          "cal_per_g": 0.29, "pct": 0.08},
            {"name": "ail",             "cal_per_g": 1.49, "pct": 0.05},
            {"name": "pain pita",       "cal_per_g": 2.75, "pct": 0.12},
        ],
    },
    "baklava": {
        "display": "Baklava",
        "base_portion": 150,
        "ingredients": [
            {"name": "pâte filo",       "cal_per_g": 4.00, "pct": 0.35},
            {"name": "noix/pistaches",  "cal_per_g": 6.54, "pct": 0.30},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.15},
            {"name": "sirop miel",      "cal_per_g": 3.04, "pct": 0.20},
        ],
    },
    "waffles": {
        "display": "Gaufres",
        "base_portion": 250,
        "ingredients": [
            {"name": "pâte à gaufres",  "cal_per_g": 2.20, "pct": 0.55},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.08},
            {"name": "sirop érable",    "cal_per_g": 2.60, "pct": 0.15},
            {"name": "fruits frais",    "cal_per_g": 0.50, "pct": 0.12},
            {"name": "crème chantilly", "cal_per_g": 2.57, "pct": 0.10},
        ],
    },
    "pancakes": {
        "display": "Pancakes",
        "base_portion": 250,
        "ingredients": [
            {"name": "pâte à pancakes", "cal_per_g": 2.27, "pct": 0.55},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.08},
            {"name": "sirop érable",    "cal_per_g": 2.60, "pct": 0.20},
            {"name": "fruits",          "cal_per_g": 0.50, "pct": 0.12},
            {"name": "crème",           "cal_per_g": 2.57, "pct": 0.05},
        ],
    },
    "french_fries": {
        "display": "Frites",
        "base_portion": 300,
        "ingredients": [
            {"name": "pommes de terre frites", "cal_per_g": 3.12, "pct": 0.80},
            {"name": "huile de friture",        "cal_per_g": 8.84, "pct": 0.10},
            {"name": "ketchup",                 "cal_per_g": 1.01, "pct": 0.10},
        ],
    },
    "chocolate_cake": {
        "display": "Gâteau au Chocolat",
        "base_portion": 200,
        "ingredients": [
            {"name": "génoise chocolat", "cal_per_g": 3.80, "pct": 0.50},
            {"name": "ganache",          "cal_per_g": 4.50, "pct": 0.30},
            {"name": "beurre",           "cal_per_g": 7.17, "pct": 0.10},
            {"name": "sucre",            "cal_per_g": 3.87, "pct": 0.10},
        ],
    },
    "donuts": {
        "display": "Donuts",
        "base_portion": 180,
        "ingredients": [
            {"name": "pâte frite",      "cal_per_g": 3.90, "pct": 0.60},
            {"name": "glaçage sucre",   "cal_per_g": 3.80, "pct": 0.25},
            {"name": "huile",           "cal_per_g": 8.84, "pct": 0.10},
            {"name": "sprinkles",       "cal_per_g": 3.90, "pct": 0.05},
        ],
    },
    "omelette": {
        "display": "Omelette",
        "base_portion": 250,
        "ingredients": [
            {"name": "œufs",            "cal_per_g": 1.55, "pct": 0.55},
            {"name": "beurre/huile",    "cal_per_g": 7.17, "pct": 0.10},
            {"name": "fromage",         "cal_per_g": 4.02, "pct": 0.15},
            {"name": "légumes variés",  "cal_per_g": 0.30, "pct": 0.15},
            {"name": "sel + poivre",    "cal_per_g": 0.00, "pct": 0.05},
        ],
    },
    "eggs_benedict": {
        "display": "Œufs Bénédicte",
        "base_portion": 300,
        "ingredients": [
            {"name": "muffin anglais",  "cal_per_g": 2.27, "pct": 0.25},
            {"name": "œufs pochés",     "cal_per_g": 1.55, "pct": 0.20},
            {"name": "jambon/saumon",   "cal_per_g": 1.45, "pct": 0.20},
            {"name": "sauce hollandaise","cal_per_g": 4.00, "pct": 0.25},
            {"name": "beurre",          "cal_per_g": 7.17, "pct": 0.05},
            {"name": "herbes",          "cal_per_g": 0.25, "pct": 0.05},
        ],
    },
    "club_sandwich": {
        "display": "Club Sandwich",
        "base_portion": 350,
        "ingredients": [
            {"name": "pain de mie",     "cal_per_g": 2.65, "pct": 0.30},
            {"name": "poulet/dinde",    "cal_per_g": 1.65, "pct": 0.20},
            {"name": "bacon",           "cal_per_g": 5.41, "pct": 0.10},
            {"name": "laitue + tomate", "cal_per_g": 0.18, "pct": 0.10},
            {"name": "mayonnaise",      "cal_per_g": 6.80, "pct": 0.10},
            {"name": "fromage",         "cal_per_g": 4.02, "pct": 0.10},
            {"name": "frites",          "cal_per_g": 3.12, "pct": 0.10},
        ],
    },
    # ── Fallback générique ────────────────────
    "other": {
        "display": "Plat Mixte",
        "base_portion": 400,
        "ingredients": [
            {"name": "féculents",           "cal_per_g": 1.50, "pct": 0.35},
            {"name": "protéines",           "cal_per_g": 2.00, "pct": 0.25},
            {"name": "légumes",             "cal_per_g": 0.35, "pct": 0.25},
            {"name": "matières grasses",    "cal_per_g": 5.00, "pct": 0.10},
            {"name": "sauce",               "cal_per_g": 1.00, "pct": 0.05},
        ],
    },
}

# ── Aliases : noms Kaggle → clés FOOD_DATABASE ──
CLASS_ALIASES: dict[str, str] = {
    # Food-101 internationaux
    "hot_dog":             "hamburger",
    "spaghetti_carbonara": "spaghetti_bolognese",
    "greek_salad":         "caesar_salad",
    "chicken_wings":       "chicken_curry",
    "risotto":             "spaghetti_bolognese",
    "grilled_salmon":      "steak",
    "fish_and_chips":      "steak",
    "nachos":              "tacos",
    "spring_rolls":        "falafel",
    "samosa":              "falafel",
    "onion_rings":         "french_fries",
    "cheesecake":          "chocolate_cake",
    "beef_carpaccio":      "steak",
    # Tunisiens déjà dans la DB (mapping direct)
    "leblebi":             "lablabi",
    "harira":              "chorba",
}

# ══════════════════════════════════════════════
#  CHARGEMENT DU MODÈLE (singleton)
# ══════════════════════════════════════════════
_model       = None
_class_names: dict[str, str] = {}


def _load_model():
    global _model, _class_names
    if _model is not None:
        return True
    if not MODEL_PATH.exists():
        print(f"⚠️  Modèle introuvable : {MODEL_PATH}")
        return False
    if not NAMES_PATH.exists():
        print(f"⚠️  class_names.json introuvable : {NAMES_PATH}")
        return False
    try:
        import tensorflow as tf
        print(f"🔄 Chargement du modèle…")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(NAMES_PATH, encoding="utf-8") as f:
            _class_names = json.load(f)
        print(f"✅ Modèle chargé — {len(_class_names)} classes")
        return True
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        return False


# ══════════════════════════════════════════════
#  PRÉDICTION
# ══════════════════════════════════════════════
def _preprocess(image_bytes: bytes) -> "np.ndarray":
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _resolve_class(raw_name: str) -> str:
    name = raw_name.lower().strip()
    return CLASS_ALIASES.get(name, name)


def _build_result(dish_key: str, confidence: float, portion_multiplier: float) -> dict:
    if dish_key not in FOOD_DATABASE:
        dish_key = "other"
    db     = FOOD_DATABASE[dish_key]
    base_g = db["base_portion"] * portion_multiplier
    ingredients, total_cal = [], 0.0
    for ing in db["ingredients"]:
        grams = base_g * ing["pct"]
        cal   = ing["cal_per_g"] * grams
        ingredients.append({"name": ing["name"], "portion_g": round(grams), "calories": round(cal)})
        total_cal += cal
    return {
        "dish":               db["display"],
        "confidence":         round(confidence * 100),
        "ingredients":        ingredients,
        "total_calories":     round(total_cal),
        "portion_multiplier": round(portion_multiplier, 2),
        "source":             "local_model",
    }


def predict(image_bytes: bytes, quantity_g: float | None = None) -> dict:
    if not _load_model():
        return _color_fallback(image_bytes, quantity_g)
    try:
        arr         = _preprocess(image_bytes)
        predictions = _model.predict(arr, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:TOP_K]

        print(f"🔍 Top-{TOP_K} prédictions :")
        for i in top_indices:
            cls = _class_names.get(str(i), "?")
            print(f"   {cls:<35} {predictions[i]*100:.1f}%")

        # Sélection intelligente Top-3
        chosen_key  = None
        chosen_conf = 0.0

        for i in top_indices:
            raw_name = _class_names.get(str(i), "other")
            conf     = float(predictions[i])
            key      = _resolve_class(raw_name)

            # Prise directe si confiance >= 70% ou si dans la DB
            if conf >= 0.70 or key in FOOD_DATABASE:
                chosen_key  = key
                chosen_conf = conf
                print(f"   ✅ Choix : {raw_name} → {key} ({conf*100:.1f}%)")
                break

        if chosen_key is None:
            raw_name    = _class_names.get(str(top_indices[0]), "other")
            chosen_key  = _resolve_class(raw_name)
            chosen_conf = float(predictions[top_indices[0]])
            print(f"   ✅ Choix défaut : {chosen_key} ({chosen_conf*100:.1f}%)")

        if chosen_conf < MIN_CONF:
            return _color_fallback(image_bytes, quantity_g)

        if quantity_g and quantity_g > 0:
            db_base = FOOD_DATABASE.get(chosen_key, FOOD_DATABASE["other"])["base_portion"]
            mult    = quantity_g / db_base
        else:
            mult = _estimate_portion(image_bytes)

        result = _build_result(chosen_key, chosen_conf, mult)
        if quantity_g and quantity_g > 0:
            result = _apply_quantity(result, quantity_g)
        return result

    except Exception as e:
        print(f"❌ Erreur prédiction : {e}")
        return _color_fallback(image_bytes, quantity_g)


def _estimate_portion(image_bytes: bytes) -> float:
    try:
        img  = Image.open(io.BytesIO(image_bytes))
        area = img.width * img.height
        if area > 1_000_000: return 1.5
        if area > 500_000:   return 1.2
        if area > 200_000:   return 1.0
        return 0.8
    except Exception:
        return 1.0


def _color_fallback(image_bytes: bytes, quantity_g: float | None = None) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img.resize((32, 32)), dtype=np.float32)
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        if g > r and g > b:       dish = "caesar_salad"
        elif r > 160 and g < 100: dish = "pizza"
        elif r > 140 and b < 80:  dish = "couscous_tunisien"
        else:                      dish = "couscous_tunisien"
    except Exception:
        dish = "couscous_tunisien"

    mult   = _estimate_portion(image_bytes)
    result = _build_result(dish, 0.50, mult)
    result["source"] = "local_fallback"
    if quantity_g and quantity_g > 0:
        result = _apply_quantity(result, quantity_g)
    return result


def _apply_quantity(result: dict, quantity_g: float) -> dict:
    ingredients = result.get("ingredients", [])
    estimated_g = sum(i.get("portion_g", 0) for i in ingredients)
    if estimated_g <= 0:
        return result
    scale = quantity_g / estimated_g
    adjusted = [
        {**item, "portion_g": round(item["portion_g"] * scale),
                 "calories":  round(item["calories"]  * scale)}
        for item in ingredients
    ]
    result["ingredients"]         = adjusted
    result["total_calories"]      = sum(i["calories"] for i in adjusted)
    result["requested_portion_g"] = round(quantity_g)
    result["estimated_portion_g"] = round(estimated_g)
    result["portion_multiplier"]  = round(result.get("portion_multiplier", 1.0) * scale, 2)
    return result