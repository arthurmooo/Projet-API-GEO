# =============================================================================
# traitement.py — Chargement et préparation des données
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
APL_DESERT_THRESHOLD = 2.5
OFGL_YEARS = [2012, 2022, 2024]
ZONE_BINS = [-1, 50, 500, 999999]
ZONE_LABELS = ["Rural", "Périurbain", "Urbain"]

MODEL_FEATURES = [
    "apl_score",
    "densite",
    "population_2024",
    "variation_population_2022_2024_pct",
    "variation_population_2012_2022_pct",
    "surface_mediane",
    "nb_ventes",
    "pct_maisons",
    "is_zrr",
    "revenu_median",
    "age_median",
    "taux_chomage",
]

FEATURE_LABELS = {
    "apl_score": "Score APL (accès médecins)",
    "densite": "Densité population",
    "population_2024": "Population 2024",
    "variation_population_2022_2024_pct": "Variation population 2022-2024",
    "variation_population_2012_2022_pct": "Variation population 2012-2022",
    "surface_mediane": "Surface médiane",
    "nb_ventes": "Nombre de ventes",
    "pct_maisons": "% de maisons",
    "is_zrr": "Zone de revitalisation rurale",
    "revenu_median": "Revenu médian (€/an)",
    "age_median": "Âge médian",
    "taux_chomage": "Taux de chômage (%)",
}


# =============================================================================
# Chargement des données
# =============================================================================

def _normaliser_code_insee(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.zfill(5)


def _to_numeric_fr(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype(str).str.replace(",", "."), errors="coerce")


def _colonnes_presentes(df: pd.DataFrame, colonnes: list[str]) -> list[str]:
    return [col for col in colonnes if col in df.columns]


def charger_dvf() -> pd.DataFrame:
    """
    Charge le fichier DVF téléchargé manuellement.
    Accepte les formats data.gouv.fr (colonnes françaises).
    """
    chemin = DATA_DIR / "dvf.csv"
    if not chemin.exists():
        return pd.DataFrame()

    df = pd.read_csv(chemin, sep=",", low_memory=False, dtype=str)

    # Renommage flexible selon la version du fichier
    renommage = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ("codecommune", "code_commune", "l_codinsee"):
            renommage[col] = "code_insee"
        elif c in ("valeur_fonciere", "valeur foncière", "prix"):
            renommage[col] = "prix"
        elif c in ("surface_reelle_bati", "surface réelle bâti", "surface"):
            renommage[col] = "surface"
        elif c in ("type_local", "type local"):
            renommage[col] = "type_bien"
        elif c in ("date_mutation", "date mutation"):
            renommage[col] = "date"
        elif c in ("nom_commune", "nom commune", "libcom"):
            renommage[col] = "commune"
        elif c in ("latitude", "lat"):
            renommage[col] = "latitude"
        elif c in ("longitude", "lon"):
            renommage[col] = "longitude"

    df = df.rename(columns=renommage)

    # Supprimer les colonnes dupliquées issues du renommage (garder la première)
    df = df.loc[:, ~df.columns.duplicated()]

    # Conversions numériques
    for col in ["prix", "surface", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = _to_numeric_fr(df[col])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["annee"] = df["date"].dt.year

    if "code_insee" in df.columns:
        df["code_insee"] = _normaliser_code_insee(df["code_insee"])

    print(f"DVF chargé : {len(df)} lignes brutes")

    # Garder seulement maisons et appartements
    if "type_bien" in df.columns:
        df = df[df["type_bien"].isin(["Maison", "Appartement"])]

    # Calcul prix au m²
    mask = (df["surface"] > 0) & df["prix"].notna() & (df["surface"] < 1000)
    df.loc[mask, "prix_m2"] = df.loc[mask, "prix"] / df.loc[mask, "surface"]

    # Filtrage outliers
    df = df[(df["prix_m2"] > 200) & (df["prix_m2"] < 20000)]

    df = df.reset_index(drop=True)
    print(f"DVF après filtres : {len(df)} lignes — {df['code_insee'].nunique()} communes uniques")
    return df


def charger_apl() -> pd.DataFrame:
    """
    Charge le fichier APL (Accessibilité Potentielle Localisée).
    Accepte apl.xlsx ou apl.csv dans le dossier data/.
    """
    # Chercher xlsx ou csv
    chemin_xlsx = DATA_DIR / "apl.xlsx"
    chemin_csv  = DATA_DIR / "apl.csv"

    if chemin_xlsx.exists():
        # Lire le xlsx — prendre la feuille avec le plus de données
        try:
            xl = pd.ExcelFile(chemin_xlsx)
            best_df = pd.DataFrame()
            for sheet in xl.sheet_names:
                tmp = xl.parse(sheet, dtype=str)
                if len(tmp) > len(best_df):
                    best_df = tmp
            df = best_df
        except Exception as e:
            print(f"Erreur lecture xlsx : {e}")
            return pd.DataFrame()

    elif chemin_csv.exists():
        # Essai avec différents séparateurs
        df = pd.DataFrame()
        for sep in [";", ","]:
            try:
                tmp = pd.read_csv(chemin_csv, sep=sep, low_memory=False, dtype=str)
                if len(tmp.columns) > 2:
                    df = tmp
                    break
            except Exception:
                continue
    else:
        return pd.DataFrame()

    # Renommage flexible
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

    if "code_insee" not in df.columns:
        # Prendre la première colonne comme code INSEE
        df = df.rename(columns={df.columns[0]: "code_insee"})

    if "apl_score" not in df.columns:
        # Chercher une colonne numérique
        num_cols = [c for c in df.columns if c != "code_insee"]
        for c in num_cols:
            test = _to_numeric_fr(df[c])
            if test.notna().sum() > len(df) * 0.5:
                df = df.rename(columns={c: "apl_score"})
                break

    if "apl_score" not in df.columns:
        return pd.DataFrame()

    df["code_insee"] = _normaliser_code_insee(df["code_insee"])
    df["apl_score"] = _to_numeric_fr(df["apl_score"])

    # Garder le dernier millésime si plusieurs années
    if "annee_apl" in df.columns:
        df["annee_apl"] = pd.to_numeric(df["annee_apl"], errors="coerce")
        df = df.sort_values("annee_apl").drop_duplicates("code_insee", keep="last")
    else:
        df = df.drop_duplicates("code_insee")

    return df[["code_insee", "apl_score"]].dropna()


def charger_geo() -> pd.DataFrame:
    """
    Récupère nom commune + densité via l'API Géo (légère, rapide).
    Fallback sur CSV local si pas de connexion.
    """
    import requests

    chemin_cache = DATA_DIR / "geo_cache.csv"
    if chemin_cache.exists():
        return pd.read_csv(chemin_cache, dtype={"code_insee": str})

    print("Récupération des métadonnées communes (API Géo)...")
    try:
        r = requests.get(
            "https://geo.api.gouv.fr/communes",
            params={"fields": "code,nom,population,surface,departement", "format": "json"},
            timeout=30,
        )
        if r.status_code != 200:
            return pd.DataFrame()

        communes = r.json()
        records = []
        for c in communes:
            pop  = c.get("population", 0) or 0
            surf = (c.get("surface") or 0) / 100  # hectares → km²
            records.append({
                "code_insee":  c.get("code"),
                "commune":     c.get("nom"),
                "population":  pop,
                "surface_km2": round(surf, 2),
                "densite":     round(pop / surf, 1) if surf > 0 else 0,
                "departement": (c.get("departement") or {}).get("code", ""),
            })

        df = pd.DataFrame(records)
        df["code_insee"] = _normaliser_code_insee(df["code_insee"])
        df.to_csv(chemin_cache, index=False)
        return df

    except Exception as e:
        print(f"API Géo indisponible : {e}")
        return pd.DataFrame()


def charger_zrr() -> pd.DataFrame:
    import requests
    from io import BytesIO, StringIO

    chemin_cache = DATA_DIR / "zrr_cache.csv"
    if chemin_cache.exists():
        return pd.read_csv(chemin_cache, dtype={"code_insee": str})

    try:
        meta = requests.get(
            "https://www.data.gouv.fr/api/1/datasets/zones-de-revitalisation-rurale-zrr/",
            timeout=30,
        ).json()

        ressource = next(
            (
                r for r in meta.get("resources", [])
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
                    c for c in df_raw.columns
                    if c.upper() in {"CODGEO", "CODE_COMMUNE", "CODE_INSEE", "COG", "CODE INSEE"}
                ),
                df_raw.columns[0],
            )
            col_zrr = next(
                (
                    c for c in df_raw.columns
                    if "ZRR" in c.upper() or "ZONAGE" in c.upper() or "CLASSEMENT" in c.upper()
                ),
                None,
            )
            df = df_raw
        else:
            engine = "xlrd" if fmt == "xls" else "openpyxl"
            df_raw = pd.read_excel(
                BytesIO(raw.content), sheet_name=0, header=4, dtype=str, engine=engine
            )
            df = df_raw.iloc[1:].reset_index(drop=True)
            col_code = df.columns[0]
            col_zrr = df.columns[2]

        df = df.rename(columns={col_code: "code_insee"})
        df["code_insee"] = _normaliser_code_insee(df["code_insee"].str.strip())

        if col_zrr:
            df["is_zrr"] = ~df[col_zrr].str.upper().str.startswith("NC")
        else:
            df["is_zrr"] = True

        df = df[["code_insee", "is_zrr"]].drop_duplicates("code_insee").reset_index(drop=True)
        df.to_csv(chemin_cache, index=False)
        return df

    except Exception as e:
        print(f"Erreur ZRR : {e}")
        return pd.DataFrame()


def charger_insee(codes: list | None = None) -> pd.DataFrame:
    """Récupère revenu_median, age_median et taux_chomage via l'API Melodi."""
    import requests
    import time

    chemin_cache = DATA_DIR / "insee_cache.csv"

    if chemin_cache.exists():
        df_cache = pd.read_csv(chemin_cache, dtype={"code_insee": str})
        codes_manquants = (
            [c for c in (codes or []) if c not in df_cache["code_insee"].values]
            if codes is not None else []
        )
        if not codes_manquants:
            return df_cache
    else:
        df_cache = pd.DataFrame()
        codes_manquants = codes or []

    if not codes_manquants:
        return df_cache

    base_url = "https://api.insee.fr/melodi/data"
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

    def _obs_value(r: requests.Response) -> float:
        try:
            obs = r.json().get("observations", [])
            return obs[0]["measures"]["OBS_VALUE_NIVEAU"]["value"] if obs else np.nan
        except Exception:
            return np.nan

    def _get(url: str, params: dict) -> requests.Response:
        for _ in range(3):
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 429:
                return r
            time.sleep(65)
        return r

    def _age_median(pops: dict) -> float:
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
                f"{base_url}/DS_FILOSOFI_CC",
                {
                    "GEO": geo,
                    "FILOSOFI_MEASURE": "MED_SL",
                    "UNIT_MEASURE": "EUR_YR",
                    "maxResult": 1,
                },
            )
            row["revenu_median"] = _obs_value(r)
        except Exception:
            row["revenu_median"] = np.nan
        time.sleep(delai)

        try:
            employes = chomeurs = None
            for sta, key in (("1", "employes"), ("2", "chomeurs")):
                r = _get(
                    f"{base_url}/DS_RP_EMPLOI_LR_PRINC",
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
                f"{base_url}/DS_RP_POPULATION_PRINC",
                {
                    "GEO": geo,
                    "SEX": "_T",
                    "RP_MEASURE": "POP",
                    "TIME_PERIOD": "2022",
                    "maxResult": 50,
                },
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
        df_cache = (
            pd.concat([df_cache, df_new], ignore_index=True)
            if not df_cache.empty else df_new
        )
        df_cache.to_csv(chemin_cache, index=False)

    return df_cache


def charger_population_ofgl() -> pd.DataFrame:
    """
    Récupère les populations communales 2012-2024 via l'API OFGL.
    Données OFGL retraitées depuis l'INSEE, sous Licence Ouverte Etalab 2.0.
    """
    chemin_cache = DATA_DIR / "population_ofgl_cache.csv"
    if chemin_cache.exists():
        return pd.read_csv(chemin_cache, dtype={"code_insee": str})

    print("Récupération des populations communales (API OFGL)...")
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
            extras = df_last[extra_cols].drop_duplicates("code_insee")
            pop = pop.merge(extras, on="code_insee", how="left")

        pop.to_csv(chemin_cache, index=False)
        return pop

    except Exception as e:
        print(f"API OFGL indisponible : {e}")
        return pd.DataFrame()


def _telecharger_population_ofgl(annee: int) -> pd.DataFrame:
    """
    Télécharge une année OFGL. L'export CSV évite de paginer l'API record par record.
    """
    import requests
    from io import StringIO

    url = (
        "https://data.ofgl.fr/api/explore/v2.1/catalog/datasets/"
        "populations-ofgl-communes/exports/csv"
    )
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


# =============================================================================
# Construction du dataset final
# =============================================================================

def construire_dataset() -> pd.DataFrame:
    """
    Fusionne DVF + APL + GEO + OFGL en un seul dataset agrégé par commune.
    """
    df_dvf = charger_dvf()
    df_apl = charger_apl()
    df_geo = charger_geo()
    df_pop = charger_population_ofgl()

    if df_dvf.empty:
        return pd.DataFrame()

    # --- Agrégation DVF par commune ---
    agg_dict = dict(
        prix_m2_median  = ("prix_m2", "median"),
        prix_m2_moyen   = ("prix_m2", "mean"),
        nb_ventes        = ("prix_m2", "count"),
        surface_mediane  = ("surface", "median"),
    )
    if "latitude" in df_dvf.columns:
        agg_dict["latitude"] = ("latitude", "median")
    if "longitude" in df_dvf.columns:
        agg_dict["longitude"] = ("longitude", "median")

    df = (
        df_dvf.groupby("code_insee")
        .agg(**agg_dict)
        .reset_index()
    )
    df["prix_m2_median"] = df["prix_m2_median"].round(0)

    # Pourcentage maisons
    if "type_bien" in df_dvf.columns:
        pct = (
            df_dvf.groupby("code_insee")["type_bien"]
            .apply(lambda x: (x == "Maison").mean() * 100)
            .reset_index()
            .rename(columns={"type_bien": "pct_maisons"})
        )
        df = df.merge(pct, on="code_insee", how="left")

    # --- Jointure APL ---
    if not df_apl.empty:
        df = df.merge(df_apl, on="code_insee", how="left")
        df["desert_medical"] = (df["apl_score"] < APL_DESERT_THRESHOLD).astype(int)

    # --- Jointure GEO ---
    if not df_geo.empty:
        geo_cols = _colonnes_presentes(
            df_geo, ["code_insee", "commune", "population", "densite", "departement"]
        )
        df = df.merge(df_geo[geo_cols], on="code_insee", how="left")

    # --- Jointure population OFGL ---
    if not df_pop.empty:
        pop_cols = _colonnes_presentes(
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
        df = df.merge(df_pop[pop_cols], on="code_insee", how="left")

        if "population_2024" in df.columns:
            df["population"] = df["population_2024"].combine_first(df.get("population"))

    # --- Jointure ZRR ---
    df_zrr = charger_zrr()
    if not df_zrr.empty:
        df = df.merge(df_zrr, on="code_insee", how="left")
        df["is_zrr"] = df["is_zrr"].fillna(False).astype(int)

    # --- Jointure INSEE ---
    df_insee = charger_insee(codes=df["code_insee"].tolist())
    if not df_insee.empty:
        cols_insee = _colonnes_presentes(
            df_insee, ["code_insee", "revenu_median", "age_median", "taux_chomage"]
        )
        df = df.merge(df_insee[cols_insee], on="code_insee", how="left")

    # --- Classification zone ---
    if "densite" in df.columns:
        df["type_zone"] = pd.cut(
            df["densite"],
            bins=ZONE_BINS,
            labels=ZONE_LABELS,
        )
    elif "commune" not in df.columns:
        df["commune"] = df["code_insee"]

    # Filtrer communes avec trop peu de ventes
    df = df[df["nb_ventes"] >= 1]

    return df.reset_index(drop=True)


# =============================================================================
# Modèle ML
# =============================================================================

def entrainer_modele(df: pd.DataFrame) -> dict:
    """
    Entraîne un Random Forest pour prédire le prix au m².
    Retourne le modèle, les métriques et l'importance des variables.
    """
    features = [feature for feature in MODEL_FEATURES if feature in df.columns]

    df_ml = df[features + ["prix_m2_median"]].dropna()

    if len(df_ml) < 30:
        return {}

    X = df_ml[features]
    y = df_ml["prix_m2_median"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    df_imp = pd.DataFrame({
        "variable":   [FEATURE_LABELS.get(f, f) for f in features],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model":      model,
        "features":   features,
        "r2":         round(r2, 3),
        "mae":        round(mae, 0),
        "n":          len(df_ml),
        "importance": df_imp,
        "X_test":     X_test,
        "y_test":     y_test,
        "y_pred":     y_pred,
    }
