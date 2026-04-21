# =============================================================================
# traitement.py — Chargement et préparation des données
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Chargement des données
# =============================================================================

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
            df[col] = df[col].str.replace(",", ".").pipe(pd.to_numeric, errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["annee"] = df["date"].dt.year

    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)

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
            test = pd.to_numeric(df[c].str.replace(",", "."), errors="coerce")
            if test.notna().sum() > len(df) * 0.5:
                df = df.rename(columns={c: "apl_score"})
                break

    df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)
    df["apl_score"]  = pd.to_numeric(
        df["apl_score"].astype(str).str.replace(",", "."), errors="coerce"
    )

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
    import requests, time

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
        df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)
        df.to_csv(chemin_cache, index=False)
        return df

    except Exception as e:
        print(f"API Géo indisponible : {e}")
        return pd.DataFrame()


# =============================================================================
# Construction du dataset final
# =============================================================================

def construire_dataset() -> pd.DataFrame:
    """
    Fusionne DVF + APL + GEO en un seul dataset agrégé par commune.
    """
    df_dvf = charger_dvf()
    df_apl = charger_apl()
    df_geo = charger_geo()

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
        df["desert_medical"] = (df["apl_score"] < 2.5).astype(int)

    # --- Jointure GEO ---
    if not df_geo.empty:
        geo_cols = ["code_insee", "commune", "population", "densite", "departement"]
        geo_cols = [c for c in geo_cols if c in df_geo.columns]
        df = df.merge(df_geo[geo_cols], on="code_insee", how="left")

    # --- Classification zone ---
    if "densite" in df.columns:
        df["type_zone"] = pd.cut(
            df["densite"],
            bins=[-1, 50, 500, 999999],
            labels=["Rural", "Périurbain", "Urbain"],
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
    features_possibles = ["apl_score", "densite", "surface_mediane", "nb_ventes"]
    features = [f for f in features_possibles if f in df.columns]

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

    labels = {
        "apl_score":       "Score APL (accès médecins)",
        "densite":         "Densité population",
        "surface_mediane": "Surface médiane",
        "nb_ventes":       "Nombre de ventes",
    }

    df_imp = pd.DataFrame({
        "variable":   [labels.get(f, f) for f in features],
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