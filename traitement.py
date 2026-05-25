#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import glob
import os
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
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    BaggingRegressor,
)


DATA_DIR = Path(__file__).parent / "data"
APL_DESERT_THRESHOLD = 2.5
OFGL_YEARS = [2012, 2022, 2024]
ZONE_BINS = [-1, 50, 500, 999999]
ZONE_LABELS = ["Rural", "Périurbain", "Urbain"]

VILLES_REF = {
    "Lyon": (45.7640, 4.8357),
    "Grenoble": (45.1885, 5.7245),
    "Geneve": (46.2044, 6.1432),
    "Paris": (48.8566, 2.3522),
    "Marseille": (43.2965, 5.3698),
}

CORE_FEATURES = [
    "apl_score",
    "densite",
    "population_2024",
    "variation_population_2022_2024_pct",
    "variation_population_2012_2022_pct",
    "surface_mediane",
    "nb_ventes",
    "pct_maisons",
    "is_zrr",
    "dist_ville_min",
]

OPTIONAL_INSEE_FEATURES = [
    "revenu_median",
    "age_median",
    "taux_chomage",
]

FEATURES = CORE_FEATURES + (
    OPTIONAL_INSEE_FEATURES if os.getenv("TERRITOIRE_IMMO_USE_INSEE_FEATURES", "0") == "1" else []
)

TARGET = "prix_m2_median"

FEATURE_LABELS = {
    "apl_score": "Score APL (accès médecins)",
    "densite": "Densité population",
    "population_2024": "Population 2024",
    "variation_population_2022_2024_pct": "Variation population 2022-2024",
    "variation_population_2012_2022_pct": "Variation population 2012-2022",
    "surface_mediane": "Surface médiane",
    "nb_ventes": "Nombre de ventes",
    "pct_maisons": "% maisons",
    "is_zrr": "Zone de revitalisation rurale",
    "revenu_median": "Revenu médian (€/an)",
    "age_median": "Âge médian",
    "taux_chomage": "Taux de chômage (%)",
    "dist_ville_min": "Distance ville la plus proche (km)",
}


def _normaliser_code_insee(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.zfill(5)


def _to_numeric_fr(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype(str).str.replace(",", "."), errors="coerce")


def _colonnes_presentes(df: pd.DataFrame, colonnes: list[str]) -> list[str]:
    return [col for col in colonnes if col in df.columns]


def _dist_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def charger_dvf() -> pd.DataFrame:
    fichiers = glob.glob(str(DATA_DIR / "dvf*.csv"))
    if not fichiers:
        return pd.DataFrame()

    dfs = []
    for fichier in fichiers:
        try:
            tmp = pd.read_csv(
                fichier,
                low_memory=False,
                dtype={"code_commune": str, "code_departement": str},
            )
            dfs.append(tmp)
            print(f"[DVF] {Path(fichier).name} — {len(tmp):,} lignes")
        except Exception as exc:
            print(f"[DVF] Erreur {Path(fichier).name} : {exc}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    renommage = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ("codecommune", "code_commune", "l_codinsee"):
            renommage[col] = "code_insee"
        elif c in ("nom_commune", "nom commune", "libcom"):
            renommage[col] = "commune"
        elif c in ("valeur_fonciere", "valeur foncière", "prix"):
            renommage[col] = "prix"
        elif c in ("surface_reelle_bati", "surface réelle bâti", "surface"):
            renommage[col] = "surface"
        elif c in ("type_local", "type local"):
            renommage[col] = "type_bien"
        elif c in ("date_mutation", "date mutation"):
            renommage[col] = "date"
        elif c in ("latitude", "lat"):
            renommage[col] = "latitude"
        elif c in ("longitude", "lon"):
            renommage[col] = "longitude"

    df = df.rename(columns=renommage)
    df = df.loc[:, ~df.columns.duplicated()]

    colonnes = [
        "code_insee",
        "commune",
        "date",
        "prix",
        "type_bien",
        "surface",
        "latitude",
        "longitude",
    ]
    df = df[_colonnes_presentes(df, colonnes)].copy().reset_index(drop=True)

    if "code_insee" in df.columns:
        df["code_insee"] = _normaliser_code_insee(df["code_insee"])
        df["code_commune"] = df["code_insee"]

    for col in ["prix", "surface", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = _to_numeric_fr(df[col])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["annee"] = df["date"].dt.year

    if "type_bien" in df.columns:
        df = df[df["type_bien"].isin(["Maison", "Appartement"])].copy()

    ok = (df["surface"] > 9) & (df["surface"] < 1000) & df["prix"].notna()
    df.loc[ok, "prix_m2"] = (df.loc[ok, "prix"] / df.loc[ok, "surface"]).round(0)
    df = df[(df["prix_m2"] > 200) & (df["prix_m2"] < 20000)].reset_index(drop=True)

    print(f"[DVF] Total : {len(df):,} lignes — {df['code_insee'].nunique()} communes")
    return df


def charger_apl() -> pd.DataFrame:
    chemin_xlsx = DATA_DIR / "apl.xlsx"
    chemin_csv = DATA_DIR / "apl.csv"

    try:
        if chemin_xlsx.exists():
            try:
                df = pd.read_excel(chemin_xlsx, sheet_name="APL 2023", header=8, dtype=str)
                df = df.iloc[1:].reset_index(drop=True)
                cols = df.columns.tolist()
                df = df.rename(columns={cols[0]: "code_insee", cols[1]: "commune_apl", cols[2]: "apl_score"})
            except Exception:
                xl = pd.ExcelFile(chemin_xlsx)
                best_df = pd.DataFrame()
                for sheet in xl.sheet_names:
                    tmp = xl.parse(sheet, dtype=str)
                    if len(tmp) > len(best_df):
                        best_df = tmp
                df = best_df
        elif chemin_csv.exists():
            df = pd.DataFrame()
            for sep in [";", ","]:
                try:
                    tmp = pd.read_csv(chemin_csv, sep=sep, dtype=str)
                    if len(tmp.columns) > 2:
                        df = tmp
                        break
                except Exception:
                    continue
        else:
            return pd.DataFrame()

        renommage = {}
        for col in df.columns:
            c = col.lower().strip()
            if any(x in c for x in ["insee", "depcom", "codgeo"]):
                renommage[col] = "code_insee"
            elif "apl" in c and any(x in c for x in ["mg", "score", "valeur", "indic"]):
                renommage[col] = "apl_score"
            elif any(x in c for x in ["an", "annee", "millesime", "année"]):
                renommage[col] = "annee_apl"
        df = df.rename(columns=renommage)
        df = df.loc[:, ~df.columns.duplicated()]

        if "code_insee" not in df.columns:
            df = df.rename(columns={df.columns[0]: "code_insee"})
        if "apl_score" not in df.columns:
            for col in df.columns:
                if col != "code_insee":
                    test = _to_numeric_fr(df[col])
                    if test.notna().sum() > len(df) * 0.5:
                        df = df.rename(columns={col: "apl_score"})
                        break

        if "apl_score" not in df.columns:
            return pd.DataFrame()

        df["code_insee"] = _normaliser_code_insee(df["code_insee"])
        df["apl_score"] = _to_numeric_fr(df["apl_score"])

        if "annee_apl" in df.columns:
            df["annee_apl"] = pd.to_numeric(df["annee_apl"], errors="coerce")
            df = df.sort_values("annee_apl").drop_duplicates("code_insee", keep="last")
        else:
            df = df.drop_duplicates("code_insee")

        result = df[["code_insee", "apl_score"]].dropna().reset_index(drop=True)
        print(f"[APL] {len(result):,} communes")
        return result

    except Exception as exc:
        print(f"[APL] Erreur : {exc}")
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
        for commune in r.json():
            pop = commune.get("population") or 0
            surf = (commune.get("surface") or 0) / 100
            records.append(
                {
                    "code_insee": str(commune.get("code", "")).zfill(5),
                    "commune": commune.get("nom"),
                    "population": pop,
                    "surface_km2": round(surf, 2),
                    "densite": round(pop / surf, 1) if surf > 0 else 0,
                    "departement": (commune.get("departement") or {}).get("code", ""),
                }
            )
        df = pd.DataFrame(records)
        df.to_csv(cache, index=False)
        print(f"[API Géo] {len(df):,} communes")
        return df
    except Exception as exc:
        print(f"[API Géo] Indisponible : {exc}")
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
            (
                r
                for r in meta.get("resources", [])
                if r.get("format", "").lower() in {"csv", "xls", "xlsx"}
            ),
            None,
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
                (
                    c
                    for c in df_raw.columns
                    if c.upper() in {"CODGEO", "CODE_COMMUNE", "CODE_INSEE", "COG", "CODE INSEE"}
                ),
                df_raw.columns[0],
            )
            col_zrr = next(
                (
                    c
                    for c in df_raw.columns
                    if "ZRR" in c.upper() or "ZONAGE" in c.upper() or "CLASSEMENT" in c.upper()
                ),
                None,
            )
            df = df_raw
        else:
            engine = "xlrd" if fmt == "xls" else "openpyxl"
            df_raw = pd.read_excel(BytesIO(raw.content), sheet_name=0, header=4, dtype=str, engine=engine)
            df = df_raw.iloc[1:].reset_index(drop=True)
            col_code = df.columns[0]
            col_zrr = df.columns[2]

        df = df.rename(columns={col_code: "code_insee"})
        df["code_insee"] = _normaliser_code_insee(df["code_insee"].str.strip())
        df["is_zrr"] = ~df[col_zrr].str.upper().str.startswith("NC") if col_zrr else True

        df = df[["code_insee", "is_zrr"]].drop_duplicates("code_insee").reset_index(drop=True)
        df.to_csv(cache, index=False)
        print(f"[ZRR] {len(df):,} communes")
        return df

    except Exception as exc:
        print(f"[ZRR] Erreur : {exc}")
        return pd.DataFrame()


def charger_insee(codes: list | None = None) -> pd.DataFrame:
    cache = DATA_DIR / "insee_cache.csv"

    if cache.exists():
        df_cache = pd.read_csv(cache, dtype={"code_insee": str})
        codes_manquants = (
            [c for c in (codes or []) if c not in df_cache["code_insee"].values]
            if codes is not None
            else []
        )
        if not codes_manquants:
            print(f"[INSEE] Cache : {len(df_cache):,} communes")
            return df_cache
    else:
        df_cache = pd.DataFrame()
        codes_manquants = codes or []

    if not codes_manquants:
        return df_cache

    if os.getenv("TERRITOIRE_IMMO_FETCH_INSEE", "0") != "1":
        print(
            f"[INSEE] Cache partiel : {len(df_cache):,} communes — "
            f"{len(codes_manquants):,} manquantes non téléchargées "
            "(définir TERRITOIRE_IMMO_FETCH_INSEE=1 pour compléter)."
        )
        return df_cache

    print(f"[INSEE] Appel API pour {len(codes_manquants)} communes...")

    base = "https://api.insee.fr/melodi/data"
    delai = 60.0 / 30
    age_bounds = {
        "Y_LT15": (0, 15),
        "Y15T24": (15, 25),
        "Y25T39": (25, 40),
        "Y40T54": (40, 55),
        "Y55T64": (55, 65),
        "Y65T79": (65, 80),
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
            for k, (inf, sup) in age_bounds.items()
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
            r = _get(
                f"{base}/DS_FILOSOFI_CC",
                {"GEO": geo, "FILOSOFI_MEASURE": "MED_SL", "UNIT_MEASURE": "EUR_YR", "maxResult": 1},
            )
            row["revenu_median"] = _obs_value(r)
        except Exception:
            row["revenu_median"] = np.nan
        time.sleep(delai)

        try:
            employes = chomeurs = None
            for sta, key in (("1", "employes"), ("2", "chomeurs")):
                r = _get(
                    f"{base}/DS_RP_EMPLOI_LR_PRINC",
                    {
                        "GEO": geo,
                        "EMPSTA_ENQ": sta,
                        "SEX": "_T",
                        "AGE": "Y15T64",
                        "EDUC": "_T",
                        "TIME_PERIOD": "2022",
                        "maxResult": 1,
                    },
                )
                val = _obs_value(r)
                if key == "employes":
                    employes = val
                else:
                    chomeurs = val
                time.sleep(delai)
            if employes and chomeurs and not np.isnan(employes) and not np.isnan(chomeurs):
                row["taux_chomage"] = round(chomeurs / (employes + chomeurs) * 100, 1)
            else:
                row["taux_chomage"] = np.nan
        except Exception:
            row["taux_chomage"] = np.nan

        try:
            r = _get(
                f"{base}/DS_RP_POPULATION_PRINC",
                {"GEO": geo, "SEX": "_T", "RP_MEASURE": "POP", "TIME_PERIOD": "2022", "maxResult": 50},
            )
            pops = {}
            for obs in r.json().get("observations", []):
                age = obs["dimensions"].get("AGE")
                val = obs["measures"].get("OBS_VALUE_NIVEAU", {}).get("value")
                if age in age_bounds:
                    pops[age] = val
            row["age_median"] = _age_median(pops)
        except Exception:
            row["age_median"] = np.nan
        time.sleep(delai)

        records.append(row)

    if records:
        df_new = pd.DataFrame(records)
        df_cache = pd.concat([df_cache, df_new], ignore_index=True) if not df_cache.empty else df_new
        df_cache.to_csv(cache, index=False)

    return df_cache


def charger_population_ofgl() -> pd.DataFrame:
    cache = DATA_DIR / "population_ofgl_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype={"code_insee": str})
        print(f"[OFGL] Cache : {len(df):,} communes")
        return df

    print("[OFGL] Récupération des populations communales...")
    try:
        frames = [_telecharger_population_ofgl(annee) for annee in OFGL_YEARS]
        df = pd.concat(frames, ignore_index=True)

        df["code_insee"] = _normaliser_code_insee(df["com_code"])
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
        for col in ["pmun", "ptot"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["code_insee", "annee"])
        df["annee"] = df["annee"].astype(int)

        pop = (
            df.pivot_table(index="code_insee", columns="annee", values="pmun", aggfunc="first")
            .rename(columns=lambda annee: f"population_{annee}")
            .reset_index()
        )

        for debut, fin in [(2012, 2022), (2022, 2024)]:
            c_debut = f"population_{debut}"
            c_fin = f"population_{fin}"
            if c_debut in pop.columns and c_fin in pop.columns:
                pop[f"variation_population_{debut}_{fin}_pct"] = (
                    (pop[c_fin] - pop[c_debut]) / pop[c_debut].replace(0, np.nan) * 100
                ).round(2)

        derniere_annee = int(df["annee"].max())
        df_last = df[df["annee"] == derniere_annee].copy()
        extra_cols = _colonnes_presentes(df_last, ["code_insee", "rural", "touristique", "montagne"])
        if len(extra_cols) > 1:
            pop = pop.merge(df_last[extra_cols].drop_duplicates("code_insee"), on="code_insee", how="left")

        pop.to_csv(cache, index=False)
        print(f"[OFGL] {len(pop):,} communes")
        return pop

    except Exception as exc:
        print(f"[OFGL] Indisponible : {exc}")
        return pd.DataFrame()


def _telecharger_population_ofgl(annee: int) -> pd.DataFrame:
    from io import StringIO

    url = "https://data.ofgl.fr/api/explore/v2.1/catalog/datasets/populations-ofgl-communes/exports/csv"
    r = requests.get(
        url,
        params={
            "select": "com_code,annee,pmun,ptot,rural,touristique,montagne",
            "where": f"annee = date'{annee}'",
            "lang": "fr",
            "timezone": "Europe/Paris",
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"statut {r.status_code}")

    return pd.read_csv(StringIO(r.text), sep=";", dtype=str, low_memory=False)


def construire_dataset() -> pd.DataFrame:
    df_dvf = charger_dvf()
    df_apl = charger_apl()
    df_geo = charger_geo()
    df_pop = charger_population_ofgl()

    if df_dvf.empty:
        return pd.DataFrame()

    agg_dict = dict(
        prix_m2_median=("prix_m2", "median"),
        prix_m2_moyen=("prix_m2", "mean"),
        nb_ventes=("prix_m2", "count"),
        surface_mediane=("surface", "median"),
    )
    if "latitude" in df_dvf.columns:
        agg_dict["latitude"] = ("latitude", "median")
    if "longitude" in df_dvf.columns:
        agg_dict["longitude"] = ("longitude", "median")

    df = df_dvf.groupby("code_insee", as_index=False).agg(**agg_dict)
    df["prix_m2_median"] = df["prix_m2_median"].round(0)

    if "type_bien" in df_dvf.columns:
        pct = (
            df_dvf.groupby("code_insee", as_index=False)["type_bien"]
            .apply(lambda x: round((x == "Maison").mean() * 100, 1))
            .rename(columns={"type_bien": "pct_maisons"})
        )
        df = df.merge(pct, on="code_insee", how="left")

    if not df_apl.empty:
        df = df.merge(df_apl, on="code_insee", how="left")
        df["desert_medical"] = (df["apl_score"] < APL_DESERT_THRESHOLD).astype(int)
        print(f"[Fusion] DVF × APL : {df['apl_score'].notna().mean() * 100:.1f}%")

    if not df_geo.empty:
        cols_geo = _colonnes_presentes(df_geo, ["code_insee", "commune", "population", "densite", "departement"])
        df = df.merge(df_geo[cols_geo], on="code_insee", how="left")

    if not df_pop.empty:
        cols_pop = _colonnes_presentes(
            df_pop,
            [
                "code_insee",
                "population_2012",
                "population_2022",
                "population_2024",
                "variation_population_2012_2022_pct",
                "variation_population_2022_2024_pct",
                "rural",
                "touristique",
                "montagne",
            ],
        )
        df = df.merge(df_pop[cols_pop], on="code_insee", how="left")
        if "population_2024" in df.columns:
            df["population"] = df["population_2024"].combine_first(df.get("population"))

    if "densite" in df.columns:
        df["type_zone"] = pd.cut(df["densite"], bins=ZONE_BINS, labels=ZONE_LABELS)

    df_zrr = charger_zrr()
    if not df_zrr.empty:
        df = df.merge(df_zrr, on="code_insee", how="left")
        df["is_zrr"] = df["is_zrr"].fillna(False).astype(int)

    df_insee = charger_insee(codes=df["code_insee"].tolist())
    if not df_insee.empty:
        cols_insee = _colonnes_presentes(df_insee, ["code_insee", "revenu_median", "age_median", "taux_chomage"])
        df = df.merge(df_insee[cols_insee], on="code_insee", how="left")

    if "latitude" in df.columns and "longitude" in df.columns:
        for ville, (vlat, vlon) in VILLES_REF.items():
            df[f"dist_{ville.lower()}"] = df.apply(
                lambda row: _dist_km(row["latitude"], row["longitude"], vlat, vlon)
                if pd.notna(row.get("latitude"))
                else np.nan,
                axis=1,
            )
        dist_cols = [f"dist_{ville.lower()}" for ville in VILLES_REF]
        df["dist_ville_min"] = df[dist_cols].min(axis=1)
        df = df.drop(columns=dist_cols)

    df = df[df["nb_ventes"] >= 3].reset_index(drop=True)
    if not df.empty:
        print(f"[Dataset] {len(df):,} communes — prix médian {df['prix_m2_median'].median():.0f} €/m²")
        df.to_csv(DATA_DIR / "dataset_final.csv", index=False)
        print(f"[Export] dataset_final.csv sauvegardé — {len(df)} communes, {len(df.columns)} colonnes")
    return df


def _preparer_X_y(df: pd.DataFrame):
    features_dispo = [
        feature
        for feature in FEATURES
        if feature in df.columns and df[feature].notna().sum() >= 30
    ]
    df_ml = df[features_dispo + [TARGET]].dropna()
    if len(df_ml) < 30:
        return None, None, []
    return df_ml[features_dispo], df_ml[TARGET], features_dispo


def _metriques(y_test, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_test, y_pred)), 3),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 0),
    }


def _extraire_importance(estimateur, features) -> pd.DataFrame:
    labels = [FEATURE_LABELS.get(feature, feature) for feature in features]
    if hasattr(estimateur, "feature_importances_"):
        imp = estimateur.feature_importances_
    elif hasattr(estimateur, "coef_"):
        coef = np.abs(estimateur.coef_)
        imp = coef / coef.sum() if coef.sum() > 0 else np.ones(len(features)) / len(features)
    elif hasattr(estimateur, "estimators_"):
        imps = [
            estimator.feature_importances_
            for estimator in estimateur.estimators_
            if hasattr(estimator, "feature_importances_")
        ]
        imp = np.mean(imps, axis=0) if imps else np.ones(len(features)) / len(features)
    else:
        imp = np.ones(len(features)) / len(features)

    return (
        pd.DataFrame({"variable": labels, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


_CATALOGUE_HP = {
    "Régression linéaire": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        "params": {},
    },
    "Ridge (L2)": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", Ridge())]),
        "params": {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    },
    "LASSO (L1)": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", Lasso(max_iter=10000))]),
        "params": {"m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    },
    "ElasticNet": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", ElasticNet(max_iter=10000))]),
        "params": {
            "m__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
    },
    "Bayesian Ridge": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", BayesianRidge())]),
        "params": {
            "m__alpha_1": [1e-7, 1e-6, 1e-5, 1e-4],
            "m__lambda_1": [1e-7, 1e-6, 1e-5, 1e-4],
        },
    },
    "Huber Regressor": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", HuberRegressor(max_iter=500))]),
        "params": {
            "m__epsilon": [1.1, 1.35, 1.5, 2.0, 3.0],
            "m__alpha": [0.00001, 0.0001, 0.001, 0.01],
        },
    },
    "K-Nearest Neighbors": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor())]),
        "params": {
            "m__n_neighbors": [8, 10, 12, 15, 20],
            "m__weights": ["distance"],
            "m__p": [1],
            "m__metric": ["minkowski", "euclidean"],
        },
    },
    "SVR": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", SVR())]),
        "params": {
            "m__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "m__kernel": ["rbf", "poly", "linear"],
            "m__epsilon": [0.01, 0.1, 0.5, 1.0],
            "m__gamma": ["scale", "auto"],
        },
    },
    "Kernel Ridge": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", KernelRidge())]),
        "params": {
            "m__alpha": [0.01, 0.1, 1.0, 10.0],
            "m__kernel": ["rbf", "polynomial"],
            "m__gamma": [0.001, 0.01, 0.1, 1.0],
        },
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
        },
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [200, 300, 500],
            "max_depth": [15, 20, 25, 30],
            "max_features": ["log2", 0.3, 0.5],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [200, 300, 500],
            "max_depth": [20, 30, None],
            "max_features": [None, "sqrt", 0.5],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [1, 2],
        },
    },
    "Bagging": {
        "model": BaggingRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [200, 300, 500],
            "max_samples": [0.6, 0.7, 0.75, 0.8],
            "max_features": [0.7, 0.8, 0.9],
        },
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [300, 500, 700],
            "max_depth": [3, 4],
            "learning_rate": [0.005, 0.01, 0.02],
            "subsample": [0.7, 0.8, 0.85],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "HistGradientBoosting": {
        "model": HistGradientBoostingRegressor(random_state=42),
        "params": {
            "max_iter": [100, 200, 300],
            "max_depth": [3, 4, 6, None],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
    },
    "MLP Regressor": {
        "model": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("m", MLPRegressor(max_iter=1000, random_state=42, early_stopping=True)),
            ]
        ),
        "params": {
            "m__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "m__alpha": [0.0001, 0.001, 0.01],
            "m__learning_rate_init": [0.001, 0.01],
            "m__activation": ["relu", "tanh"],
        },
    },
    "Voting Regressor": {
        "model": VotingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=20,
                        max_features="log2",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=None,
                        max_features=None,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "knn",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("m", KNeighborsRegressor(n_neighbors=10, weights="distance", p=1)),
                        ]
                    ),
                ),
            ]
        ),
        "params": {},
    },
    "Stacking": {
        "model": StackingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=20,
                        max_features="log2",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                ("et", ExtraTreesRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
                (
                    "knn",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("m", KNeighborsRegressor(n_neighbors=10, weights="distance", p=1)),
                        ]
                    ),
                ),
                ("ada", AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=10,
            n_jobs=-1,
        ),
        "params": {},
    },
}


def comparer_tous_modeles(df: pd.DataFrame, cv: int = 10) -> pd.DataFrame:
    X, y, features = _preparer_X_y(df)
    if X is None:
        print("[ML] Données insuffisantes")
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    resultats = []
    for nom, cfg in _CATALOGUE_HP.items():
        print(f"[ML] {nom}...", end=" ", flush=True)
        try:
            if cfg["params"]:
                gs = GridSearchCV(cfg["model"], cfg["params"], scoring="r2", cv=cv, n_jobs=-1, refit=True)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                best_model = cfg["model"]
                best_model.fit(X_train, y_train)
                best_params = {}

            y_pred = best_model.predict(X_test)
            metriques = _metriques(y_test, y_pred)
            print(f"R²={metriques['r2']:.3f}  MAE={metriques['mae']:.0f}€")
            resultats.append(
                {
                    "Modèle": nom,
                    "R2": metriques["r2"],
                    "MAE (€/m2)": metriques["mae"],
                    "RMSE (€/m2)": metriques["rmse"],
                    "Meilleurs params": str(best_params) if best_params else "—",
                    "_model": best_model,
                    "_features": features,
                }
            )
        except Exception as exc:
            print(f"ERREUR : {exc}")
            resultats.append(
                {
                    "Modèle": nom,
                    "R2": None,
                    "MAE (€/m2)": None,
                    "RMSE (€/m2)": None,
                    "Meilleurs params": "Erreur",
                    "_model": None,
                    "_features": features,
                }
            )

    return (
        pd.DataFrame(resultats)
        .dropna(subset=["R2"])
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )


def entrainer_modele(df: pd.DataFrame, cv: int = 10) -> dict:
    X, y, features = _preparer_X_y(df)
    if X is None:
        return {}

    print("\n[ML] Comparaison des modèles...")
    df_comparaison = comparer_tous_modeles(df, cv=cv)
    if df_comparaison.empty:
        return {}

    meilleur = df_comparaison.iloc[0]
    nom_modele = meilleur["Modèle"]
    best_model = meilleur["_model"]
    best_feats = meilleur["_features"]

    print(f"\n[ML] Meilleur : {nom_modele} (R²={meilleur['R2']:.3f})")
    print(f"[ML] K-Fold {cv} sur le meilleur modèle...")
    cv_r2 = cross_val_score(best_model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    cv_mae = cross_val_score(best_model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    print(f"[ML] K-Fold R² : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"[ML] K-Fold MAE : {(-cv_mae).mean():.0f} ± {(-cv_mae).std():.0f} €/m²")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    y_pred = best_model.predict(X_test)

    estimateur = (
        best_model.named_steps.get("m", list(best_model.named_steps.values())[-1])
        if hasattr(best_model, "named_steps")
        else best_model
    )
    importance = _extraire_importance(estimateur, best_feats)
    colonnes = ["Modèle", "R2", "MAE (€/m2)", "RMSE (€/m2)", "Meilleurs params"]

    return {
        "model": best_model,
        "nom_modele": nom_modele,
        "features": best_feats,
        "importance": importance,
        "r2": meilleur["R2"],
        "mae": meilleur["MAE (€/m2)"],
        "rmse": meilleur["RMSE (€/m2)"],
        "n": len(X),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "comparaison": df_comparaison[colonnes].copy(),
        "cv_r2_mean": round(float(cv_r2.mean()), 3),
        "cv_r2_std": round(float(cv_r2.std()), 3),
        "cv_mae_mean": round(float((-cv_mae).mean()), 0),
        "cv_mae_std": round(float((-cv_mae).std()), 0),
        "cv_folds": cv,
    }


def entrainer_modele_stacking_api(df: pd.DataFrame, cv: int = 5) -> dict:
    X, y, features = _preparer_X_y(df)
    if X is None:
        return {}

    model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=1)),
            ("hgb", HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42)),
            ("knn", Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=7))])),
            ("en", Pipeline([("scaler", StandardScaler()), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5))])),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metriques = _metriques(y_test, y_pred)

    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)
    cv_mae = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)

    estimateur = model.estimators_[0] if getattr(model, "estimators_", None) else model
    importance = _extraire_importance(estimateur, features)
    comparaison = pd.DataFrame(
        [
            {
                "Modèle": "Stacking",
                "R2": metriques["r2"],
                "MAE (€/m2)": metriques["mae"],
                "RMSE (€/m2)": metriques["rmse"],
                "Meilleurs params": "Profil API fixe",
            }
        ]
    )

    return {
        "model": model,
        "nom_modele": "Stacking",
        "features": features,
        "importance": importance,
        "r2": metriques["r2"],
        "mae": metriques["mae"],
        "rmse": metriques["rmse"],
        "n": len(X),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "comparaison": comparaison,
        "cv_r2_mean": round(float(cv_r2.mean()), 3),
        "cv_r2_std": round(float(cv_r2.std()), 3),
        "cv_mae_mean": round(float((-cv_mae).mean()), 0),
        "cv_mae_std": round(float((-cv_mae).std()), 0),
        "cv_folds": cv,
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
        cols_desc = [col for col in FEATURES + [TARGET] if col in df.columns]
        print(df[cols_desc].describe().round(2))

        cols_corr = [col for col in FEATURES + [TARGET] if col in df.columns]
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
            print(f"\n  MEILLEUR MODÈLE : {res['nom_modele']}")
            print(f"  R²   (test)    : {res['r2']}")
            print(f"  MAE  (test)    : {res['mae']:.0f} €/m²")
            print(f"  RMSE (test)    : {res['rmse']:.0f} €/m²")
            print(f"  R²   (K-Fold {res['cv_folds']}) : {res['cv_r2_mean']} ± {res['cv_r2_std']}")
            print(f"  MAE  (K-Fold {res['cv_folds']}) : {res['cv_mae_mean']:.0f} ± {res['cv_mae_std']:.0f} €/m²")
            print(f"  Communes : {res['n']}")

            print("\nIMPORTANCE DES VARIABLES:")
            print(res["importance"].to_string(index=False))

            plt.figure(figsize=(8, 5))
            plt.barh(res["importance"]["variable"], res["importance"]["importance"], color="#4e8df5")
            plt.xlabel("Importance")
            plt.title(f"Importance — {res['nom_modele']}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

            comp = res["comparaison"].sort_values("R2")
            colors = ["#27ae60" if i == len(comp) - 1 else "#4e8df5" for i in range(len(comp))]
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
