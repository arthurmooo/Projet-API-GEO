#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, VotingRegressor,
    StackingRegressor, BaggingRegressor,
)


DATA_DIR = Path("/Users/julietterey/Downloads/Projet-API-GEO-Juliette/data")

#distance en km entre chaque commune et la grande ville la plus proche
#coordonées gps
#utiles pour l'instant Lyon Geneve et Grenoble 
VILLES_REF = {
    "Lyon":     (45.7640,  4.8357),
    "Grenoble": (45.1885,  5.7245),
    "Geneve":   (46.2044,  6.1432),
    "Paris":    (48.8566,  2.3522),
    "Marseille":(43.2965,  5.3698),
}

FEATURES = [
    "apl_score", "densite", "surface_mediane", "nb_ventes",
    "pct_maisons", "population", "is_zrr",
    "revenu_median", "age_median", "taux_chomage",
    "dist_ville_min",
]


TARGET = "prix_m2_median"


FEATURE_LABELS = {
    "apl_score":       "Score APL (accès médecins)",
    "densite":         "Densité population",
    "surface_mediane": "Surface médiane",
    "nb_ventes":       "Nombre de ventes",
    "pct_maisons":     "% maisons",
    "population":      "Population",
    "is_zrr":          "Zone de revitalisation rurale",
    "revenu_median":   "Revenu médian (€/an)",
    "age_median":      "Âge médian",
    "taux_chomage":    "Taux de chômage (%)",
    "dist_ville_min":  "Distance ville la plus proche (km)",
}


def _dist_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * 6371 * asin(sqrt(a))


def charger_dvf() -> pd.DataFrame:
    fichiers = []
    for f in glob.glob(str(DATA_DIR / "dvf*.csv")):
        fichiers.append(f)
    if not fichiers:
        return pd.DataFrame()

    dfs = []
    for f in fichiers:
        try:
            tmp = pd.read_csv(f, low_memory=False,
                              dtype={"code_commune": str, "code_departement": str})
            dfs.append(tmp)
            print(f"[DVF] {Path(f).name} — {len(tmp):,} lignes")
        except Exception as e:
            print(f"[DVF] Erreur {Path(f).name} : {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    colonnes = ["code_commune", "nom_commune", "date_mutation",
                "valeur_fonciere", "type_local", "surface_reelle_bati",
                "latitude", "longitude"]
    colonnes = [c for c in colonnes if c in df.columns]
    df = df[colonnes].copy().reset_index(drop=True)

    df = df.rename(columns={
        "valeur_fonciere":     "prix",
        "type_local":          "type_bien",
        "surface_reelle_bati": "surface",
    })

    if "code_commune" in df.columns:
        df["code_commune"] = df["code_commune"].astype(str).str.zfill(5)
    
    df["prix"]    = pd.to_numeric(df["prix"],    errors="coerce")
    df["surface"] = pd.to_numeric(df["surface"], errors="coerce")
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["type_bien"].isin(["Maison", "Appartement"])].copy()

    ok = (df["surface"] > 9) & (df["surface"] < 1000) & df["prix"].notna()
    df.loc[ok, "prix_m2"] = (df.loc[ok, "prix"] / df.loc[ok, "surface"]).round(0)
    df = df[(df["prix_m2"] > 200) & (df["prix_m2"] < 20000)].reset_index(drop=True)

    print(f"[DVF] Total : {len(df):,} lignes — {df['code_commune'].nunique()} communes")
    return df


def charger_apl() -> pd.DataFrame:
    chemin_xlsx = DATA_DIR / "apl.xlsx"
    chemin_csv  = DATA_DIR / "apl.csv"

    try:
        if chemin_xlsx.exists():
            try:
                df = pd.read_excel(chemin_xlsx, sheet_name="APL 2023",
                                   header=8, dtype=str)
                df = df.iloc[1:].reset_index(drop=True)
                cols = df.columns.tolist()
                df = df.rename(columns={
                    cols[0]: "code_insee",
                    cols[1]: "commune_apl",
                    cols[2]: "apl_score",
                })
            except Exception:
                xl = pd.ExcelFile(chemin_xlsx)
                best_df = pd.DataFrame()
                for sheet in xl.sheet_names:
                    tmp = xl.parse(sheet, dtype=str)
                    if len(tmp) > len(best_df):
                        best_df = tmp
                df = best_df
                renommage = {}
                for col in df.columns:
                    c = col.lower().strip()
                    if any(x in c for x in ["insee", "depcom", "codgeo"]):
                        renommage[col] = "code_insee"
                    elif "apl" in c and any(x in c for x in ["mg", "score", "valeur"]):
                        renommage[col] = "apl_score"
                df = df.rename(columns=renommage)

        elif chemin_csv.exists():
            for sep in [";", ","]:
                try:
                    tmp = pd.read_csv(chemin_csv, sep=sep, dtype=str)
                    if len(tmp.columns) > 2:
                        df = tmp
                        break
                except Exception:
                    continue
            renommage = {}
            for col in df.columns:
                c = col.lower().strip()
                if any(x in c for x in ["insee", "depcom", "codgeo"]):
                    renommage[col] = "code_insee"
                elif "apl" in c and any(x in c for x in ["mg", "score", "valeur"]):
                    renommage[col] = "apl_score"
            df = df.rename(columns=renommage)
        else:
            return pd.DataFrame()

        if "code_insee" not in df.columns:
            df = df.rename(columns={df.columns[0]: "code_insee"})
        if "apl_score" not in df.columns:
            for c in df.columns:
                if c != "code_insee":
                    test = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
                    if test.notna().sum() > len(df) * 0.5:
                        df = df.rename(columns={c: "apl_score"})
                        break

        df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)
        df["apl_score"]  = pd.to_numeric(
            df["apl_score"].astype(str).str.replace(",", "."), errors="coerce"
        )
        result = df[["code_insee", "apl_score"]].dropna().reset_index(drop=True)
        print(f"[APL] {len(result):,} communes")
        return result

    except Exception as e:
        print(f"[APL] Erreur : {e}")
        return pd.DataFrame()


def charger_geo() -> pd.DataFrame:
    cache = DATA_DIR / "geo_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype={"code_insee": str})
        print(f"[API Géo] Cache : {len(df):,} communes")
        return df

    print("[API Géo] Appel API...")
    try:
        r = requests.get(
            "https://geo.api.gouv.fr/communes",
            params={"fields": "code,nom,population,surface,departement", "format": "json"},
            timeout=60,
        )
        records = []
        for c in r.json():
            pop  = c.get("population") or 0
            surf = (c.get("surface") or 0) / 100
            records.append({
                "code_insee":  str(c.get("code", "")).zfill(5),
                "commune":     c.get("nom"),
                "population":  pop,
                "densite":     round(pop / surf, 1) if surf > 0 else 0,
                "departement": (c.get("departement") or {}).get("code", ""),
            })
        df = pd.DataFrame(records)
        df.to_csv(cache, index=False)
        print(f"[API Géo] {len(df):,} communes")
        return df
    except Exception as e:
        print(f"[API Géo] Indisponible : {e}")
        return pd.DataFrame()


def charger_zrr() -> pd.DataFrame:
    from io import BytesIO, StringIO
    cache = DATA_DIR / "zrr_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype={"code_insee": str})
        print(f"[ZRR] Cache : {len(df):,} communes")
        return df

    print("[ZRR] Téléchargement...")
    try:
        meta = requests.get(
            "https://www.data.gouv.fr/api/1/datasets/zones-de-revitalisation-rurale-zrr/",
            timeout=30,
        ).json()
        ressource = next(
            (r for r in meta.get("resources", [])
             if r.get("format", "").lower() in {"csv", "xls", "xlsx"}), None,
        )
        if not ressource:
            return pd.DataFrame()

        fmt = ressource["format"].lower()
        url = ressource.get("latest") or ressource["url"]
        raw = requests.get(url, timeout=90)
        raw.raise_for_status()

        if fmt == "csv":
            df_raw = pd.read_csv(StringIO(raw.text), dtype=str, sep=None, engine="python")
            col_code = next(
                (c for c in df_raw.columns
                 if c.upper() in {"CODGEO","CODE_COMMUNE","CODE_INSEE","COG","CODE INSEE"}),
                df_raw.columns[0],
            )
            col_zrr = next(
                (c for c in df_raw.columns
                 if "ZRR" in c.upper() or "ZONAGE" in c.upper() or "CLASSEMENT" in c.upper()),
                None,
            )
            df = df_raw
        else:
            engine = "xlrd" if fmt == "xls" else "openpyxl"
            df_raw = pd.read_excel(BytesIO(raw.content), sheet_name=0,
                                   header=4, dtype=str, engine=engine)
            df = df_raw.iloc[1:].reset_index(drop=True)
            col_code = df.columns[0]
            col_zrr  = df.columns[2]

        df = df.rename(columns={col_code: "code_insee"})
        df["code_insee"] = df["code_insee"].astype(str).str.strip().str.zfill(5)
        if col_zrr:
            df["is_zrr"] = ~df[col_zrr].str.upper().str.startswith("NC")
        else:
            df["is_zrr"] = True

        df = df[["code_insee","is_zrr"]].drop_duplicates("code_insee").reset_index(drop=True)
        df.to_csv(cache, index=False)
        print(f"[ZRR] {len(df):,} communes")
        return df

    except Exception as e:
        print(f"[ZRR] Erreur : {e}")
        return pd.DataFrame()


def charger_insee(codes: list = None) -> pd.DataFrame:
    cache = DATA_DIR / "insee_cache.csv"

    if cache.exists():
        df_cache = pd.read_csv(cache, dtype={"code_insee": str})
        codes_manquants = (
            [c for c in (codes or []) if c not in df_cache["code_insee"].values]
            if codes is not None else []
        )
        if not codes_manquants:
            print(f"[INSEE] Cache : {len(df_cache):,} communes")
            return df_cache
    else:
        df_cache = pd.DataFrame()
        codes_manquants = codes or []

    if not codes_manquants:
        return df_cache

    print(f"[INSEE] Appel API pour {len(codes_manquants)} communes...")

    BASE  = "https://api.insee.fr/melodi/data"
    DELAI = 60.0 / 30

    AGE_BOUNDS = {
        "Y_LT15": (0,  15), "Y15T24": (15, 25), "Y25T39": (25, 40),
        "Y40T54": (40, 55), "Y55T64": (55, 65), "Y65T79": (65, 80),
        "Y_GE80": (80, 100),
    }

    def _obs_value(r):
        try:
            obs = r.json().get("observations", [])
            return obs[0]["measures"]["OBS_VALUE_NIVEAU"]["value"] if obs else np.nan
        except Exception:
            return np.nan

    def _get(url, params):
        for _ in range(3):
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 429:
                return r
            time.sleep(65)
        return r

    def _age_median(pops):
        groupes = sorted(
            (inf, sup, pops[k])
            for k, (inf, sup) in AGE_BOUNDS.items()
            if k in pops and pops[k] is not None
        )
        total = sum(g[2] for g in groupes)
        if total == 0:
            return np.nan
        target, cumul = total / 2, 0
        for inf, sup, pop in groupes:
            if cumul + pop >= target:
                return round(inf + (target - cumul) / pop * (sup - inf), 1)
            cumul += pop
        return np.nan

    records = []
    for code in codes_manquants:
        geo = f"COM-{code}"
        row = {"code_insee": code}

        try:
            r = _get(f"{BASE}/DS_FILOSOFI_CC",
                     {"GEO": geo, "FILOSOFI_MEASURE": "MED_SL",
                      "UNIT_MEASURE": "EUR_YR", "maxResult": 1})
            row["revenu_median"] = _obs_value(r)
        except Exception:
            row["revenu_median"] = np.nan
        time.sleep(DELAI)

        try:
            employes = chomeurs = None
            for sta, key in (("1", "employes"), ("2", "chomeurs")):
                r = _get(f"{BASE}/DS_RP_EMPLOI_LR_PRINC",
                         {"GEO": geo, "EMPSTA_ENQ": sta, "SEX": "_T",
                          "AGE": "Y15T64", "EDUC": "_T", "TIME_PERIOD": "2022",
                          "maxResult": 1})
                val = _obs_value(r)
                if key == "employes":
                    employes = val
                else:
                    chomeurs = val
                time.sleep(DELAI)
            if employes and chomeurs and not np.isnan(employes) and not np.isnan(chomeurs):
                row["taux_chomage"] = round(chomeurs / (employes + chomeurs) * 100, 1)
            else:
                row["taux_chomage"] = np.nan
        except Exception:
            row["taux_chomage"] = np.nan

        try:
            r = _get(f"{BASE}/DS_RP_POPULATION_PRINC",
                     {"GEO": geo, "SEX": "_T", "RP_MEASURE": "POP",
                      "TIME_PERIOD": "2022", "maxResult": 50})
            pops = {}
            for o in r.json().get("observations", []):
                age = o["dimensions"].get("AGE")
                val = o["measures"].get("OBS_VALUE_NIVEAU", {}).get("value")
                if age in AGE_BOUNDS:
                    pops[age] = val
            row["age_median"] = _age_median(pops)
        except Exception:
            row["age_median"] = np.nan
        time.sleep(DELAI)

        records.append(row)

    if records:
        df_new = pd.DataFrame(records)
        df_cache = (
            pd.concat([df_cache, df_new], ignore_index=True)
            if not df_cache.empty else df_new
        )
        df_cache.to_csv(cache, index=False)

    return df_cache


def construire_dataset() -> pd.DataFrame:
    df_dvf = charger_dvf()
    df_apl = charger_apl()
    df_geo = charger_geo()

    if df_dvf.empty:
        return pd.DataFrame()

    agg_dict = dict(
        prix_m2_median  = ("prix_m2", "median"),
        prix_m2_moyen   = ("prix_m2", "mean"),
        nb_ventes       = ("prix_m2", "count"),
        surface_mediane = ("surface", "median"),
    )
    if "latitude" in df_dvf.columns:
        agg_dict["latitude"]  = ("latitude",  "median")
    if "longitude" in df_dvf.columns:
        agg_dict["longitude"] = ("longitude", "median")

    df = (
        df_dvf.groupby("code_commune", as_index=False)
        .agg(**agg_dict)
        .rename(columns={"code_commune": "code_insee"})
    )
    df["prix_m2_median"] = df["prix_m2_median"].round(0)

    pct = (
        df_dvf.groupby("code_commune", as_index=False)["type_bien"]
        .apply(lambda x: round((x == "Maison").mean() * 100, 1))
        .rename(columns={"code_commune": "code_insee", "type_bien": "pct_maisons"})
    )
    df = df.merge(pct, on="code_insee", how="left")

    if not df_apl.empty:
        df = df.merge(df_apl, on="code_insee", how="left")
        df["desert_medical"] = (df["apl_score"] < 2.5).astype(int)
        print(f"[Fusion] DVF × APL : {df['apl_score'].notna().mean()*100:.1f}%")

    if not df_geo.empty:
        cols = [c for c in ["code_insee","commune","population","densite","departement"]
                if c in df_geo.columns]
        df = df.merge(df_geo[cols], on="code_insee", how="left")

    if "densite" in df.columns:
        df["type_zone"] = pd.cut(
            df["densite"], bins=[-1, 50, 500, 999999],
            labels=["Rural", "Périurbain", "Urbain"],
        )

    df_zrr = charger_zrr()
    if not df_zrr.empty:
        df = df.merge(df_zrr, on="code_insee", how="left")
        df["is_zrr"] = df["is_zrr"].fillna(False).astype(int)

    df_insee = charger_insee(codes=df["code_insee"].tolist())
    if not df_insee.empty:
        cols_insee = [c for c in ["code_insee","revenu_median","age_median","taux_chomage"]
                      if c in df_insee.columns]
        df = df.merge(df_insee[cols_insee], on="code_insee", how="left")

    if "latitude" in df.columns and "longitude" in df.columns:
        for ville, (vlat, vlon) in VILLES_REF.items():
            df[f"dist_{ville.lower()}"] = df.apply(
                lambda r: _dist_km(r["latitude"], r["longitude"], vlat, vlon)
                if pd.notna(r.get("latitude")) else np.nan, axis=1
            )
        dist_cols = [f"dist_{v.lower()}" for v in VILLES_REF]
        df["dist_ville_min"] = df[dist_cols].min(axis=1)
        df = df.drop(columns=dist_cols)

    df = df[df["nb_ventes"] >= 3].reset_index(drop=True)
    print(f"[Dataset] {len(df):,} communes — prix médian {df['prix_m2_median'].median():.0f} €/m²")
    
    # Sauvegarde du dataset final pour le professeur
    df.to_csv(DATA_DIR / "dataset_final.csv", index=False)
    print(f"[Export] dataset_final.csv sauvegardé — {len(df)} communes, {len(df.columns)} colonnes")
    return df


def _preparer_X_y(df: pd.DataFrame):
    features_dispo = [f for f in FEATURES if f in df.columns]
    df_ml = df[features_dispo + [TARGET]].dropna()
    if len(df_ml) < 30:
        return None, None, []
    return df_ml[features_dispo], df_ml[TARGET], features_dispo


def _metriques(y_test, y_pred) -> dict:
    return {
        "r2":   round(float(r2_score(y_test, y_pred)), 3),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 0),
    }


def _extraire_importance(estimateur, features) -> pd.DataFrame:
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    if hasattr(estimateur, "feature_importances_"):
        imp = estimateur.feature_importances_
    elif hasattr(estimateur, "coef_"):
        coef = np.abs(estimateur.coef_)
        imp = coef / coef.sum() if coef.sum() > 0 else np.ones(len(features)) / len(features)
    elif hasattr(estimateur, "estimators_"):
        imps = [s.feature_importances_ for s in estimateur.estimators_
                if hasattr(s, "feature_importances_")]
        imp = np.mean(imps, axis=0) if imps else np.ones(len(features)) / len(features)
    else:
        imp = np.ones(len(features)) / len(features)
    return pd.DataFrame({"variable": labels, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


_CATALOGUE_HP = {

    "Régression linéaire": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        "params": {},
    },
    "Ridge (L2)": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", Ridge())]),
        "params": {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    },
    "LASSO (L1)": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", Lasso(max_iter=10000))]),
        "params": {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    },
    "ElasticNet": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", ElasticNet(max_iter=10000))]),
        "params": {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                   "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    },
    "Bayesian Ridge": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", BayesianRidge())]),
        "params": {"m__alpha_1": [1e-7, 1e-6, 1e-5, 1e-4],
                   "m__lambda_1": [1e-7, 1e-6, 1e-5, 1e-4]},
    },
    "Huber Regressor": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", HuberRegressor(max_iter=500))]),
        "params": {"m__epsilon": [1.1, 1.35, 1.5, 2.0, 3.0],
                   "m__alpha": [0.00001, 0.0001, 0.001, 0.01]},
    },
    "K-Nearest Neighbors": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor())]),
        "params": {"m__n_neighbors": [8, 10, 12, 15, 20],
                   "m__weights":     ["distance"],
                   "m__p":           [1],
                   "m__metric":      ["minkowski", "euclidean"]},
    },
    "SVR": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", SVR())]),
        "params": {"m__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
                   "m__kernel": ["rbf", "poly", "linear"],
                   "m__epsilon": [0.01, 0.1, 0.5, 1.0],
                   "m__gamma": ["scale", "auto"]},
    },
    "Kernel Ridge": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", KernelRidge())]),
        "params": {"m__alpha": [0.01, 0.1, 1.0, 10.0],
                   "m__kernel": ["rbf", "polynomial"],
                   "m__gamma": [0.001, 0.01, 0.1, 1.0]},
    },
    "Decision Tree": {
        "model":  DecisionTreeRegressor(random_state=42),
        "params": {"max_depth": [4, 6, 8, 12, None],
                   "min_samples_split": [2, 5, 10, 20],
                   "min_samples_leaf": [1, 2, 4, 8]},
    },
    "Random Forest": {
        "model":  RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {"n_estimators":      [200, 300, 500],
                   "max_depth":         [15, 20, 25, 30],
                   "max_features":      ["log2", 0.3, 0.5],
                   "min_samples_split": [2, 3, 5],
                   "min_samples_leaf":  [1, 2]},
    },
    "Extra Trees": {
        "model":  ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "params": {"n_estimators":      [200, 300, 500],
                   "max_depth":         [20, 30, None],
                   "max_features":      [None, "sqrt", 0.5],
                   "min_samples_split": [5, 10, 20],
                   "min_samples_leaf":  [1, 2]},
    },
    "Bagging": {
        "model":  BaggingRegressor(random_state=42, n_jobs=-1),
        "params": {"n_estimators": [200, 300, 500],
                   "max_samples":  [0.6, 0.7, 0.75, 0.8],
                   "max_features": [0.7, 0.8, 0.9]},
    },
    "AdaBoost": {
        "model":  AdaBoostRegressor(random_state=42),
        "params": {"n_estimators": [50, 100, 200, 300],
                   "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0]},
    },
    "Gradient Boosting": {
        "model":  GradientBoostingRegressor(random_state=42),
        "params": {"n_estimators":     [300, 500, 700],
                   "max_depth":        [3, 4],
                   "learning_rate":    [0.005, 0.01, 0.02],
                   "subsample":        [0.7, 0.8, 0.85],
                   "min_samples_leaf": [1, 2, 4]},
    },
    "HistGradientBoosting": {
        "model":  HistGradientBoostingRegressor(random_state=42),
        "params": {"max_iter": [100, 200, 300],
                   "max_depth": [3, 4, 6, None],
                   "learning_rate": [0.01, 0.05, 0.1, 0.2],
                   "l2_regularization": [0.0, 0.1, 1.0]},
    },
    "MLP Regressor": {
        "model":  Pipeline([("scaler", StandardScaler()),
                            ("m", MLPRegressor(max_iter=1000, random_state=42,
                                               early_stopping=True))]),
        "params": {"m__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                   "m__alpha": [0.0001, 0.001, 0.01],
                   "m__learning_rate_init": [0.001, 0.01],
                   "m__activation": ["relu", "tanh"]},
    },
    "Voting Regressor": {
        "model": VotingRegressor(estimators=[
            ("rf",  RandomForestRegressor(n_estimators=300, max_depth=20,
                                          max_features="log2", random_state=42, n_jobs=-1)),
            ("et",  ExtraTreesRegressor(n_estimators=300, max_depth=None,
                                         max_features=None, random_state=42, n_jobs=-1)),
            ("knn", Pipeline([("scaler", StandardScaler()),
                               ("m", KNeighborsRegressor(n_neighbors=10,
                                                          weights="distance", p=1))])),
        ]),
        "params": {},
    },
    "Stacking": {
        "model": StackingRegressor(
            estimators=[
                ("rf",  RandomForestRegressor(n_estimators=300, max_depth=20,
                                              max_features="log2", random_state=42, n_jobs=-1)),
                ("et",  ExtraTreesRegressor(n_estimators=300, max_depth=None,
                                             random_state=42, n_jobs=-1)),
                ("knn", Pipeline([("scaler", StandardScaler()),
                                   ("m", KNeighborsRegressor(n_neighbors=10,
                                                              weights="distance", p=1))])),
                ("ada", AdaBoostRegressor(n_estimators=200, learning_rate=0.1,
                                          random_state=42)),
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=10, n_jobs=-1,
        ),
        "params": {},
    },
}

#plus de folds : 5-> 10
def comparer_tous_modeles(df: pd.DataFrame, cv: int = 10) -> pd.DataFrame:
    X, y, features = _preparer_X_y(df)
    if X is None:
        print("[ML] Données insuffisantes")
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    resultats = []
    for nom, cfg in _CATALOGUE_HP.items():
        print(f"[ML] {nom}...", end=" ", flush=True)
        try:
            if cfg["params"]:
                gs = GridSearchCV(cfg["model"], cfg["params"],
                                  scoring="r2", cv=cv, n_jobs=-1, refit=True)
                gs.fit(X_train, y_train)
                best_model  = gs.best_estimator_
                best_params = gs.best_params_
            else:
                best_model = cfg["model"]
                best_model.fit(X_train, y_train)
                best_params = {}

            y_pred = best_model.predict(X_test)
            m = _metriques(y_test, y_pred)
            print(f"R²={m['r2']:.3f}  MAE={m['mae']:.0f}€")
            resultats.append({
                "Modèle":           nom,
                "R2":               m["r2"],
                "MAE (€/m2)":       m["mae"],
                "RMSE (€/m2)":      m["rmse"],
                "Meilleurs params": str(best_params) if best_params else "—",
                "_model":           best_model,
                "_features":        features,
            })
        except Exception as e:
            print(f"ERREUR : {e}")
            resultats.append({
                "Modèle": nom, "R2": None, "MAE (€/m2)": None,
                "RMSE (€/m2)": None, "Meilleurs params": "Erreur",
                "_model": None, "_features": features,
            })

    return (pd.DataFrame(resultats)
            .dropna(subset=["R2"])
            .sort_values("R2", ascending=False)
            .reset_index(drop=True))


def entrainer_modele(df: pd.DataFrame) -> dict:
    X, y, features = _preparer_X_y(df)
    if X is None:
        return {}

    print("\n[ML] Comparaison des modèles...")
    df_comparaison = comparer_tous_modeles(df)

    if df_comparaison.empty:
        return {}

    meilleur   = df_comparaison.iloc[0]
    nom_modele = meilleur["Modèle"]
    best_model = meilleur["_model"]
    best_feats = meilleur["_features"]

    print(f"\n[ML] Meilleur : {nom_modele} (R²={meilleur['R2']:.3f})")

    # K-Fold sur le meilleur modèle — évaluation plus robuste qu'un seul split
    print(f"[ML] K-Fold {cv} sur le meilleur modèle...")
    cv_r2  = cross_val_score(best_model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    cv_mae = cross_val_score(best_model, X, y, cv=cv,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
    print(f"[ML] K-Fold R² : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"[ML] K-Fold MAE : {(-cv_mae).mean():.0f} ± {(-cv_mae).std():.0f} €/m²")

    X_all, y_all, _ = _preparer_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42)
    y_pred = best_model.predict(X_test)

    estimateur = (best_model.named_steps.get("m", list(best_model.named_steps.values())[-1])
                  if hasattr(best_model, "named_steps") else best_model)

    importance = _extraire_importance(estimateur, best_feats)
    colonnes   = ["Modèle", "R2", "MAE (€/m2)", "RMSE (€/m2)", "Meilleurs params"]

    return {
        "model":       best_model,
        "nom_modele":  nom_modele,
        "features":    best_feats,
        "importance":  importance,
        "r2":          meilleur["R2"],
        "mae":         meilleur["MAE (€/m2)"],
        "rmse":        meilleur["RMSE (€/m2)"],
        "n":           len(X_all),
        "X_test":      X_test,
        "y_test":      y_test,
        "y_pred":      y_pred,
        "comparaison": df_comparaison[colonnes].copy(),
        "cv_r2_mean":  round(float(cv_r2.mean()), 3),
        "cv_r2_std":   round(float(cv_r2.std()), 3),
        "cv_mae_mean": round(float((-cv_mae).mean()), 0),
        "cv_mae_std":  round(float((-cv_mae).std()), 0),
        "cv_folds":    cv,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\nCHARGEMENT DES DONNÉES:")
    df = construire_dataset()

    if df.empty:
        print("Aucune donnée.")
    else:
        print(f"\nDataset : {len(df)} communes")
        cols_desc = [c for c in FEATURES + [TARGET] if c in df.columns]
        print(df[cols_desc].describe().round(2))

        cols_corr = [c for c in FEATURES + [TARGET] if c in df.columns]
        corr = df[cols_corr].corr().round(2)
        plt.figure(figsize=(11, 9))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.show()

        print("\nENTRAÎNEMENT DES MODÈLES:")
        res = entrainer_modele(df)

        if res:
            print("\nCOMPARAISON:")
            print(res["comparaison"].to_string(index=False))
            print(f"  MEILLEUR MODÈLE : {res['nom_modele']}")
            print(f"  R²   (test)      : {res['r2']}")
            print(f"  MAE  (test)      : {res['mae']:.0f} €/m²")
            print(f"  RMSE (test)      : {res['rmse']:.0f} €/m²")
            print(f"  R²   (K-Fold {res['cv_folds']}) : {res['cv_r2_mean']} ± {res['cv_r2_std']}")
            print(f"  MAE  (K-Fold {res['cv_folds']}) : {res['cv_mae_mean']:.0f} ± {res['cv_mae_std']:.0f} €/m²")
            print(f"  Communes : {res['n']}")


            print("\nIMPORTANCE DES VARIABLES:")
            print(res["importance"].to_string(index=False))

            plt.figure(figsize=(8, 5))
            plt.barh(res["importance"]["variable"], res["importance"]["importance"],
                     color="#4e8df5")
            plt.xlabel("Importance")
            plt.title(f"Importance — {res['nom_modele']}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

            comp   = res["comparaison"].sort_values("R2")
            colors = ["#27ae60" if i == len(comp)-1 else "#4e8df5" for i in range(len(comp))]
            plt.figure(figsize=(10, 9))
            plt.barh(comp["Modèle"], comp["R2"], color=colors)
            plt.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
            plt.xlabel("R²")
            plt.title("Comparaison des modèles (vert = meilleur)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6, 6))
            plt.scatter(res["y_test"], res["y_pred"], alpha=0.5, color="#4e8df5")
            mn = min(float(res["y_test"].min()), float(res["y_pred"].min()))
            mx = max(float(res["y_test"].max()), float(res["y_pred"].max()))
            plt.plot([mn, mx], [mn, mx], "r--", label="Parfait")
            plt.xlabel("Prix réel (€/m²)")
            plt.ylabel("Prix prédit (€/m²)")
            plt.title(f"Réel vs Prédit — {res['nom_modele']}")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
