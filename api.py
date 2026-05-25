from __future__ import annotations

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from threading import Lock, Thread
from typing import Any, Literal
from unicodedata import normalize

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from listing_parser import parse_listing_url
from traitement import FEATURE_LABELS, construire_dataset, entrainer_modele


MetricName = Literal["prix_m2_median", "apl_score", "opportunity"]


class ListingAnalyzeRequest(BaseModel):
    code_insee: str | None = None
    commune: str | None = None
    price: float = Field(gt=0)
    surface: float = Field(gt=0)


class ListingParseRequest(BaseModel):
    url: str = Field(min_length=3)


class ModelPredictRequest(BaseModel):
    code_insee: str | None = None
    commune: str | None = None
    surface: float = Field(gt=0)
    price: float | None = Field(default=None, gt=0)
    type_bien: str | None = None


class SmartSearchRequest(BaseModel):
    budget: float = Field(gt=0)
    surface: float = Field(gt=0)
    pivot_code_insee: str | None = None
    radius_km: float | None = Field(default=40, gt=0)
    min_apl: float | None = Field(default=2.5, ge=0)
    zone_type: str = "Tous"
    require_budget_fit: bool = True
    strict_apl: bool = False
    limit: int = Field(default=20, ge=1, le=100)


@dataclass
class ApiState:
    dataset: pd.DataFrame = None  # type: ignore[assignment]
    model_summary: dict[str, Any] | None = None
    model_result: dict[str, Any] | None = None
    loading: bool = False
    model_loading: bool = False
    error: str | None = None
    model_error: str | None = None


state = ApiState()
state_lock = Lock()


def _clean_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if pd.isna(value):
        return None
    return value


def _row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row:
        return default
    value = _clean_number(row[key])
    return default if value is None else value


def _slug(value: str | None) -> str:
    if not value:
        return ""
    text = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _format_bool(value: Any) -> bool | None:
    value = _clean_number(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value.lower() in {"true", "1", "oui", "yes"}
    return bool(value)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def _clamp_score(value: Any) -> float:
    value = float(value or 0)
    return min(max(value, 0), 100)


def _score_budget(ratio: float | None) -> float:
    if ratio is None:
        return 0
    if ratio >= 1.15:
        return 100
    if ratio >= 1:
        return 80 + (ratio - 1) / 0.15 * 20
    if ratio >= 0.85:
        return (ratio - 0.85) / 0.15 * 80
    return 0


def _smart_search_weights(priority: str) -> dict[str, float]:
    weights = {
        "balanced": {"budget": 0.35, "sante": 0.25, "dynamique": 0.15, "marche": 0.15, "distance": 0.10},
        "price": {"budget": 0.55, "sante": 0.15, "dynamique": 0.10, "marche": 0.10, "distance": 0.10},
        "health": {"budget": 0.25, "sante": 0.45, "dynamique": 0.10, "marche": 0.10, "distance": 0.10},
        "growth": {"budget": 0.25, "sante": 0.15, "dynamique": 0.35, "marche": 0.15, "distance": 0.10},
        "market": {"budget": 0.25, "sante": 0.15, "dynamique": 0.10, "marche": 0.40, "distance": 0.10},
    }
    return weights.get(priority, weights["balanced"])


SMART_SEARCH_WEIGHTS = _smart_search_weights("balanced")


def _smart_search_badges(row: pd.Series, budget_delta: float | None, min_apl: float | None) -> list[str]:
    apl = _row_value(row, "apl_score")
    pop_change = _row_value(row, "variation_population_2022_2024_pct", 0) or 0
    sales = _row_value(row, "nb_ventes", 0) or 0
    badges = ["Budget OK" if budget_delta is not None and budget_delta >= 0 else "Budget dépassé"]

    if apl is not None and float(apl) >= 3.5:
        badges.append("Bon accès soins")
    if apl is not None and min_apl is not None and float(apl) < float(min_apl):
        badges.append("Sous seuil APL")
    if float(pop_change) > 2:
        badges.append("Population en hausse")
    if float(sales) >= 30:
        badges.append("Marché actif")
    if float(sales) < 8:
        badges.append("Peu de ventes")
    if _format_bool(_row_value(row, "is_zrr")):
        badges.append("ZRR")
    return badges


def _smart_search_reason(badges: list[str]) -> str:
    if "Budget OK" in badges and "Bon accès soins" in badges and "Marché actif" in badges:
        return "Budget compatible, bon accès aux soins et marché actif."
    if "Budget OK" in badges and "Sous seuil APL" in badges:
        return "Très abordable, mais accès aux soins sous le seuil APL."
    if "Population en hausse" in badges and "Budget dépassé" in badges:
        return "Commune dynamique, mais budget limite pour la surface cible."
    if "Budget dépassé" in badges:
        return "Prix supérieur au budget cible, à considérer avec prudence."
    if "Peu de ventes" in badges:
        return "Budget compatible, mais le marché local présente peu de ventes."
    return "Commune bien positionnée selon les critères de recherche."


def _opportunity_score(row: pd.Series, df: pd.DataFrame) -> int:
    price = _row_value(row, "prix_m2_median")
    apl = _row_value(row, "apl_score")
    sales = _row_value(row, "nb_ventes", 0)
    pop_change = _row_value(row, "variation_population_2022_2024_pct", 0)

    price_score = 50
    if price is not None and "prix_m2_median" in df.columns:
        price_rank = df["prix_m2_median"].rank(pct=True, ascending=True).get(row.name, np.nan)
        if pd.notna(price_rank):
            price_score = int((1 - min(max(float(price_rank), 0), 1)) * 100)

    apl_score = int(min(max((float(apl or 0) / 5) * 100, 0), 100))
    liquidity_score = int(min(max(float(sales or 0) / 50 * 100, 0), 100))
    momentum_score = int(min(max((float(pop_change or 0) + 5) / 10 * 100, 0), 100))

    return round(0.35 * price_score + 0.25 * apl_score + 0.25 * liquidity_score + 0.15 * momentum_score)


def _attractiveness_score(row: pd.Series, df: pd.DataFrame) -> int:
    apl = _row_value(row, "apl_score", 0)
    density = _row_value(row, "densite", 0)
    sales = _row_value(row, "nb_ventes", 0)
    pop_change = _row_value(row, "variation_population_2022_2024_pct", 0)

    density_score = 50
    if "densite" in df.columns and density:
        rank = df["densite"].rank(pct=True, ascending=True).get(row.name, np.nan)
        if pd.notna(rank):
            density_score = int(min(max(float(rank) * 100, 0), 100))

    return round(
        0.35 * min(max((float(apl or 0) / 5) * 100, 0), 100)
        + 0.25 * density_score
        + 0.25 * min(max(float(sales or 0) / 50 * 100, 0), 100)
        + 0.15 * min(max((float(pop_change or 0) + 5) / 10 * 100, 0), 100)
    )


def _commune_payload(row: pd.Series, df: pd.DataFrame) -> dict[str, Any]:
    apl = _row_value(row, "apl_score")
    price = _row_value(row, "prix_m2_median")
    zrr = _format_bool(_row_value(row, "is_zrr"))
    population = _row_value(row, "population_2024", _row_value(row, "population"))

    return {
        "code_insee": str(_row_value(row, "code_insee", "")),
        "commune": _row_value(row, "commune", str(_row_value(row, "code_insee", ""))),
        "departement": _row_value(row, "departement"),
        "prix_m2_median": price,
        "prix_m2_moyen": _row_value(row, "prix_m2_moyen"),
        "apl_score": apl,
        "desert_medical": bool(apl is not None and float(apl) < 2.5),
        "score_attractivite": _attractiveness_score(row, df),
        "opportunity_score": _opportunity_score(row, df),
        "population": population,
        "population_2012": _row_value(row, "population_2012"),
        "population_2022": _row_value(row, "population_2022"),
        "population_2024": _row_value(row, "population_2024"),
        "variation_population_2012_2022_pct": _row_value(row, "variation_population_2012_2022_pct"),
        "variation_population_2022_2024_pct": _row_value(row, "variation_population_2022_2024_pct"),
        "densite": _row_value(row, "densite"),
        "nb_ventes": _row_value(row, "nb_ventes"),
        "surface_mediane": _row_value(row, "surface_mediane"),
        "pct_maisons": _row_value(row, "pct_maisons"),
        "is_zrr": zrr,
        "revenu_median": _row_value(row, "revenu_median"),
        "age_median": _row_value(row, "age_median"),
        "taux_chomage": _row_value(row, "taux_chomage"),
        "type_zone": _row_value(row, "type_zone"),
        "latitude": _row_value(row, "latitude"),
        "longitude": _row_value(row, "longitude"),
    }


def _summarize_model(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None

    importance = result.get("importance")
    comparison = result.get("comparaison")

    return {
        "nom_modele": result.get("nom_modele"),
        "features": result.get("features", []),
        "r2": _clean_number(result.get("r2")),
        "mae": _clean_number(result.get("mae")),
        "rmse": _clean_number(result.get("rmse")),
        "n": _clean_number(result.get("n")),
        "cv_r2_mean": _clean_number(result.get("cv_r2_mean")),
        "cv_r2_std": _clean_number(result.get("cv_r2_std")),
        "cv_mae_mean": _clean_number(result.get("cv_mae_mean")),
        "cv_mae_std": _clean_number(result.get("cv_mae_std")),
        "cv_folds": _clean_number(result.get("cv_folds")),
        "importance": importance.to_dict("records") if isinstance(importance, pd.DataFrame) else [],
        "comparaison": comparison.to_dict("records") if isinstance(comparison, pd.DataFrame) else [],
    }


def _confidence_label(row: pd.Series, missing_features: int, mae: float | None, prediction_m2: float) -> str:
    sales = _row_value(row, "nb_ventes", 0) or 0
    relative_error = (mae / prediction_m2) if mae and prediction_m2 else 1

    if sales >= 20 and missing_features <= 1 and relative_error <= 0.18:
        return "Élevée"
    if sales >= 8 and missing_features <= 3 and relative_error <= 0.28:
        return "Modérée"
    return "Prudente"


def _verdict_from_delta(delta_pct: float | None) -> str:
    if delta_pct is None:
        return "Estimation ML disponible"
    if delta_pct < -8:
        return "Annonce sous le marché"
    if delta_pct > 8:
        return "Annonce chère"
    return "Prix cohérent"


def _build_prediction_payload(payload: ModelPredictRequest, row: pd.Series, df: pd.DataFrame) -> dict[str, Any]:
    with state_lock:
        model_result = state.model_result
        model_summary = state.model_summary

    if not model_result or not model_result.get("model"):
        raise HTTPException(status_code=503, detail="Modèle ML indisponible ou en cours d'entraînement.")

    features: list[str] = model_result.get("features", [])
    if not features:
        raise HTTPException(status_code=503, detail="Variables du modèle indisponibles.")

    values: dict[str, Any] = {}
    missing: list[str] = []
    for feature in features:
        value = _row_value(row, feature)
        if value is None:
            value = _clean_number(df[feature].median()) if feature in df.columns else 0
            missing.append(feature)
        values[feature] = value

    X = pd.DataFrame([values], columns=features)
    try:
        prediction_m2 = float(model_result["model"].predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prédiction impossible : {exc}") from exc

    prediction_m2 = max(prediction_m2, 0)
    predicted_price = prediction_m2 * payload.surface
    announced_price = payload.price
    announced_price_m2 = announced_price / payload.surface if announced_price else None
    delta_pct = (
        ((announced_price_m2 - prediction_m2) / prediction_m2) * 100
        if announced_price_m2 is not None and prediction_m2 > 0
        else None
    )

    mae = _clean_number((model_summary or {}).get("mae"))
    interval_m2 = float(mae) if mae else max(prediction_m2 * 0.12, 250)
    low_total = max((prediction_m2 - interval_m2) * payload.surface, 0)
    high_total = (prediction_m2 + interval_m2) * payload.surface

    importance = (model_summary or {}).get("importance") or []
    factors = []
    for item in importance[:5]:
        label = item.get("variable")
        raw_feature = next((key for key, nice in FEATURE_LABELS.items() if nice == label), None)
        feature_key = raw_feature or label
        factors.append(
            {
                "label": label,
                "importance": round(float(item.get("importance") or 0), 3),
                "value": _clean_number(values.get(feature_key)),
            }
        )

    confidence = _confidence_label(row, len(missing), float(mae) if mae else None, prediction_m2)
    commune_name = _row_value(row, "commune")
    code_insee = str(_row_value(row, "code_insee", ""))

    return {
        "commune": commune_name,
        "code_insee": code_insee,
        "surface": payload.surface,
        "type_bien": payload.type_bien,
        "announced_price": announced_price,
        "announced_price_m2": None if announced_price_m2 is None else round(announced_price_m2),
        "predicted_price_m2": round(prediction_m2),
        "predicted_price": round(predicted_price),
        "range_low": round(low_total),
        "range_high": round(high_total),
        "delta_pct": None if delta_pct is None else round(delta_pct, 1),
        "verdict": _verdict_from_delta(delta_pct),
        "confidence": confidence,
        "confidence_note": (
            f"{int(_row_value(row, 'nb_ventes', 0) or 0)} ventes DVF locales, "
            f"{len(missing)} variable(s) complétée(s) par la médiane."
        ),
        "factors": factors,
        "model": {
            "name": (model_summary or {}).get("nom_modele"),
            "mae": mae,
            "r2": (model_summary or {}).get("r2"),
            "n": (model_summary or {}).get("n"),
        },
        "coverage": "Prototype entraîné sur les données DVF chargées pour le département 01.",
    }


def _get_dataset() -> pd.DataFrame:
    with state_lock:
        df = state.dataset
        loading = state.loading
        error = state.error

    if df is None or df.empty:
        detail = "Dataset en cours de chargement." if loading else error or "Dataset indisponible."
        raise HTTPException(status_code=503, detail=detail)
    return df


def _find_commune(df: pd.DataFrame, code_insee: str | None = None, commune: str | None = None) -> pd.Series:
    if code_insee:
        match = df[df["code_insee"].astype(str).str.zfill(5) == str(code_insee).zfill(5)]
        if not match.empty:
            return match.iloc[0]

    if commune:
        query = _slug(commune)
        matches = df[df["commune"].astype(str).map(_slug).str.contains(query, na=False)]
        if not matches.empty:
            return matches.sort_values("nb_ventes", ascending=False).iloc[0]

    raise HTTPException(status_code=404, detail="Commune introuvable.")


def _refresh_dataset_and_model(train_model: bool = True) -> None:
    with state_lock:
        if state.loading or state.model_loading:
            return
        state.loading = True
        state.error = None

    try:
        df = construire_dataset()
        with state_lock:
            state.dataset = df
            state.loading = False

        if train_model and not df.empty:
            with state_lock:
                state.model_loading = True
                state.model_error = None
            try:
                model = entrainer_modele(df)
                with state_lock:
                    state.model_result = model
                    state.model_summary = _summarize_model(model)
                    state.model_loading = False
            except Exception as exc:  # Keep the API usable even if the heavy ML run fails.
                with state_lock:
                    state.model_error = str(exc)
                    state.model_loading = False
    except Exception as exc:
        with state_lock:
            state.error = str(exc)
            state.loading = False
            state.model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=_refresh_dataset_and_model, kwargs={"train_model": True}, daemon=True).start()
    yield


app = FastAPI(title="Territoire Immo API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    with state_lock:
        df = state.dataset
        return {
            "status": "ok",
            "dataset_ready": df is not None and not df.empty,
            "dataset_loading": state.loading,
            "dataset_error": state.error,
            "model_ready": state.model_summary is not None,
            "model_loading": state.model_loading,
            "model_error": state.model_error,
            "rows": 0 if df is None else len(df),
        }


@app.get("/api/communes")
def search_communes(query: str = Query("", min_length=0), limit: int = Query(8, ge=1, le=30)) -> dict[str, Any]:
    df = _get_dataset()
    if query:
        slug = _slug(query)
        mask = df["code_insee"].astype(str).str.contains(query, na=False) | df["commune"].astype(str).map(_slug).str.contains(slug, na=False)
        filtered = df[mask]
    else:
        filtered = df

    if "nb_ventes" in filtered.columns:
        filtered = filtered.sort_values("nb_ventes", ascending=False)

    items = [
        {
            "code_insee": str(_row_value(row, "code_insee", "")),
            "commune": _row_value(row, "commune"),
            "departement": _row_value(row, "departement"),
            "prix_m2_median": _row_value(row, "prix_m2_median"),
            "apl_score": _row_value(row, "apl_score"),
        }
        for _, row in filtered.head(limit).iterrows()
    ]
    return {"items": items}


@app.post("/api/communes/search-smart")
def smart_search(payload: SmartSearchRequest) -> dict[str, Any]:
    return _smart_search_impl(payload)


@app.get("/api/communes/{code_insee}")
def commune_detail(code_insee: str) -> dict[str, Any]:
    df = _get_dataset()
    row = _find_commune(df, code_insee=code_insee)
    return _commune_payload(row, df)


@app.get("/api/communes/{code_insee}/nearby")
def nearby_communes(code_insee: str, limit: int = Query(4, ge=1, le=12)) -> dict[str, Any]:
    df = _get_dataset()
    row = _find_commune(df, code_insee=code_insee)
    current_code = str(_row_value(row, "code_insee", ""))
    candidates = df[df["code_insee"].astype(str) != current_code].copy()

    lat = _row_value(row, "latitude")
    lon = _row_value(row, "longitude")
    if lat is not None and lon is not None and {"latitude", "longitude"}.issubset(candidates.columns):
        candidates = candidates.dropna(subset=["latitude", "longitude"]).copy()
        candidates["distance_km"] = candidates.apply(
            lambda item: _distance_km(float(lat), float(lon), float(item["latitude"]), float(item["longitude"])),
            axis=1,
        )
        candidates = candidates.sort_values("distance_km")
    else:
        price = _row_value(row, "prix_m2_median", 0)
        candidates["distance_km"] = None
        candidates["price_delta"] = (candidates["prix_m2_median"] - price).abs()
        candidates = candidates.sort_values("price_delta")

    items = []
    for _, item in candidates.head(limit).iterrows():
        payload = _commune_payload(item, df)
        payload["distance_km"] = _row_value(item, "distance_km")
        items.append(payload)
    return {"items": items}


def _smart_search_impl(payload: SmartSearchRequest) -> dict[str, Any]:
    df = _get_dataset()
    candidates = df.copy()
    max_price_m2 = payload.budget / payload.surface
    radius_ignored = False
    pivot_name: str | None = None

    if payload.zone_type != "Tous":
        if "type_zone" not in candidates.columns:
            candidates = candidates.iloc[0:0].copy()
        else:
            candidates = candidates[candidates["type_zone"].astype(str) == payload.zone_type]

    min_apl = payload.min_apl
    if payload.strict_apl and min_apl is not None and "apl_score" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["apl_score"], errors="coerce") >= float(min_apl)]

    pivot_row: pd.Series | None = None
    if payload.pivot_code_insee:
        pivot_row = _find_commune(df, code_insee=payload.pivot_code_insee)
        pivot_name = _row_value(pivot_row, "commune")
        pivot_lat = _row_value(pivot_row, "latitude")
        pivot_lon = _row_value(pivot_row, "longitude")
        if pivot_lat is not None and pivot_lon is not None and {"latitude", "longitude"}.issubset(candidates.columns):
            candidates = candidates.dropna(subset=["latitude", "longitude"]).copy()
            candidates["distance_km"] = candidates.apply(
                lambda row: _distance_km(float(pivot_lat), float(pivot_lon), float(row["latitude"]), float(row["longitude"])),
                axis=1,
            )
            if payload.radius_km is not None:
                candidates = candidates[candidates["distance_km"] <= float(payload.radius_km)]
        else:
            candidates["distance_km"] = None
            radius_ignored = True
    else:
        candidates["distance_km"] = None

    if "prix_m2_median" not in candidates.columns:
        return {
            "summary": {
                "total_matches": 0,
                "returned": 0,
                "budget": payload.budget,
                "surface": payload.surface,
                "max_price_m2": round(max_price_m2),
                "pivot_commune": pivot_name,
                "radius_km": payload.radius_km,
                "require_budget_fit": payload.require_budget_fit,
                "radius_ignored": radius_ignored,
            },
            "items": [],
        }

    candidates = candidates.dropna(subset=["prix_m2_median"]).copy()
    candidates["prix_m2_median"] = pd.to_numeric(candidates["prix_m2_median"], errors="coerce")
    candidates = candidates.dropna(subset=["prix_m2_median"]).copy()
    if payload.require_budget_fit:
        candidates = candidates[candidates["prix_m2_median"] <= max_price_m2].copy()

    weights = SMART_SEARCH_WEIGHTS
    items: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        price_m2 = _row_value(row, "prix_m2_median")
        estimated_price = float(price_m2) * payload.surface if price_m2 is not None else None
        budget_delta = payload.budget - estimated_price if estimated_price else None
        budget_fit_ratio = payload.budget / estimated_price if estimated_price and estimated_price > 0 else None
        apl = _row_value(row, "apl_score")
        pop_change = _row_value(row, "variation_population_2022_2024_pct", 0) or 0
        sales = _row_value(row, "nb_ventes", 0) or 0
        distance = _row_value(row, "distance_km")

        score_budget = _clamp_score(_score_budget(budget_fit_ratio))
        score_sante = _clamp_score((float(apl or 0) / 5) * 100)
        if min_apl is not None and apl is not None and float(apl) < float(min_apl):
            score_sante *= 0.65
        score_dynamique = _clamp_score((float(pop_change) + 5) / 10 * 100)
        score_marche = _clamp_score(float(sales) / 50 * 100)
        if payload.pivot_code_insee and distance is not None and payload.radius_km:
            score_distance = _clamp_score(100 * (1 - float(distance) / float(payload.radius_km)))
        else:
            score_distance = 100

        score = (
            weights["budget"] * score_budget
            + weights["sante"] * score_sante
            + weights["dynamique"] * score_dynamique
            + weights["marche"] * score_marche
            + weights["distance"] * score_distance
        )
        badges = _smart_search_badges(row, budget_delta, min_apl)

        items.append(
            {
                "rank": 0,
                "code_insee": str(_row_value(row, "code_insee", "")),
                "commune": _row_value(row, "commune"),
                "departement": _row_value(row, "departement"),
                "score": round(score),
                "score_budget": round(score_budget),
                "score_sante": round(score_sante),
                "score_dynamique": round(score_dynamique),
                "score_marche": round(score_marche),
                "score_distance": round(score_distance),
                "prix_m2_median": price_m2,
                "estimated_price": None if estimated_price is None else round(estimated_price),
                "budget_delta": None if budget_delta is None else round(budget_delta),
                "budget_fit_ratio": None if budget_fit_ratio is None else round(budget_fit_ratio, 2),
                "apl_score": apl,
                "desert_medical": bool(apl is not None and float(apl) < 2.5),
                "population_2024": _row_value(row, "population_2024"),
                "variation_population_2022_2024_pct": _row_value(row, "variation_population_2022_2024_pct"),
                "nb_ventes": _row_value(row, "nb_ventes"),
                "type_zone": _row_value(row, "type_zone"),
                "is_zrr": _format_bool(_row_value(row, "is_zrr")),
                "distance_km": None if distance is None else round(float(distance), 1),
                "latitude": _row_value(row, "latitude"),
                "longitude": _row_value(row, "longitude"),
                "badges": badges,
                "reason": _smart_search_reason(badges),
            }
        )

    items.sort(key=lambda item: item["score"], reverse=True)
    total_matches = len(items)
    limited = items[: payload.limit]
    for index, item in enumerate(limited, start=1):
        item["rank"] = index

    return {
        "summary": {
            "total_matches": total_matches,
            "returned": len(limited),
            "budget": payload.budget,
            "surface": payload.surface,
            "max_price_m2": round(max_price_m2),
            "pivot_commune": pivot_name,
            "radius_km": payload.radius_km,
            "require_budget_fit": payload.require_budget_fit,
            "radius_ignored": radius_ignored,
        },
        "items": limited,
    }


@app.post("/api/listing/analyze")
def analyze_listing(payload: ListingAnalyzeRequest) -> dict[str, Any]:
    df = _get_dataset()
    row = _find_commune(df, code_insee=payload.code_insee, commune=payload.commune)

    price_m2 = payload.price / payload.surface
    median = _row_value(row, "prix_m2_median")
    delta_pct = None if not median else (price_m2 - float(median)) / float(median) * 100

    if delta_pct is None:
        verdict = "Référence insuffisante"
    elif delta_pct < -8:
        verdict = "Prix attractif"
    elif delta_pct > 8:
        verdict = "Prix élevé"
    else:
        verdict = "Prix cohérent"

    return {
        "commune": _row_value(row, "commune"),
        "code_insee": str(_row_value(row, "code_insee", "")),
        "price": payload.price,
        "surface": payload.surface,
        "price_m2": round(price_m2),
        "median_price_m2": median,
        "delta_pct": None if delta_pct is None else round(delta_pct, 1),
        "verdict": verdict,
    }


@app.post("/api/listing/parse")
def parse_listing(payload: ListingParseRequest) -> dict[str, Any]:
    parsed = parse_listing_url(payload.url)
    commune_name = parsed.get("commune")
    code_insee = None

    if commune_name:
        try:
            df = _get_dataset()
            row = _find_commune(df, commune=commune_name)
            code_insee = str(_row_value(row, "code_insee", ""))
            parsed["commune"] = _row_value(row, "commune")
        except HTTPException:
            parsed["warnings"].append("Commune détectée mais absente du dataset local.")

    parsed["code_insee"] = code_insee
    return parsed


@app.get("/api/model/summary")
def model_summary() -> dict[str, Any]:
    with state_lock:
        return {
            "ready": state.model_summary is not None,
            "loading": state.model_loading,
            "error": state.model_error,
            "summary": state.model_summary,
        }


@app.post("/api/model/predict")
def model_predict(payload: ModelPredictRequest) -> dict[str, Any]:
    df = _get_dataset()
    row = _find_commune(df, code_insee=payload.code_insee, commune=payload.commune)
    return _build_prediction_payload(payload, row, df)


@app.post("/api/model/refresh")
def refresh_model(background_tasks: BackgroundTasks) -> dict[str, Any]:
    background_tasks.add_task(_refresh_dataset_and_model, True)
    return {"accepted": True}


@app.get("/api/map")
def map_points(metric: MetricName = "prix_m2_median", limit: int = Query(2500, ge=1, le=10000)) -> dict[str, Any]:
    df = _get_dataset()
    required = ["code_insee", "commune", "latitude", "longitude", "prix_m2_median"]
    available = _colonnes_presentes_for_api(df, required)
    if len(available) < len(required):
        raise HTTPException(status_code=503, detail="Coordonnées cartographiques indisponibles.")

    df_map = df.dropna(subset=["latitude", "longitude"]).copy()
    if df_map.empty:
        return {"items": []}

    if metric == "opportunity":
        df_map["metric_value"] = df_map.apply(lambda row: _opportunity_score(row, df), axis=1)
    else:
        df_map["metric_value"] = df_map[metric]

    df_map = df_map.dropna(subset=["metric_value"]).head(limit)
    items = [
        {
            "code_insee": str(_row_value(row, "code_insee", "")),
            "commune": _row_value(row, "commune"),
            "latitude": _row_value(row, "latitude"),
            "longitude": _row_value(row, "longitude"),
            "metric": metric,
            "metric_value": _row_value(row, "metric_value"),
            "prix_m2_median": _row_value(row, "prix_m2_median"),
            "apl_score": _row_value(row, "apl_score"),
            "opportunity_score": _opportunity_score(row, df),
        }
        for _, row in df_map.iterrows()
    ]
    return {"items": items}


def _colonnes_presentes_for_api(df: pd.DataFrame, colonnes: list[str]) -> list[str]:
    return [col for col in colonnes if col in df.columns]
