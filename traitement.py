import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR = Path(__file__).parent / "data"


def charger_dvf() -> pd.DataFrame:
    chemin = DATA_DIR / "dvf.csv"
    if not chemin.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        chemin,
        low_memory=False,
        dtype={"code_commune": str, "code_departement": str},
    )

    # Ici on remet les numéro de lignes a zero pour eviter des erreurs plus tard dans le traitemen
    df = df.reset_index(drop=True)

    # On selectionne uniquement les colonnes utiles
    colonnes = ["code_commune", "nom_commune", "date_mutation",
                "valeur_fonciere", "type_local", "surface_reelle_bati",
                "latitude", "longitude"]
    colonnes = [c for c in colonnes if c in df.columns]
    df = df[colonnes].copy()
    df = df.reset_index(drop=True)

    # On rename pour rendre ca plus comprensible 
    df = df.rename(columns={
        "valeur_fonciere":     "prix",
        "type_local":          "type_bien",
        "surface_reelle_bati": "surface",
    })

    # Nettoyage des types
    df["code_commune"] = df["code_commune"].astype(str).str.zfill(5)
    df["prix"]         = pd.to_numeric(df["prix"],    errors="coerce")
    df["surface"]      = pd.to_numeric(df["surface"], errors="coerce")
    df["type_bien"]    = df["type_bien"].astype(str)

    # On filtre maisons et appartements 
    mask = df["type_bien"].isin(["Maison", "Appartement"])
    df = df.loc[mask].copy().reset_index(drop=True)

    # Prix au m²
    ok = (df["surface"] > 9) & (df["surface"] < 1000) & df["prix"].notna()
    df.loc[ok, "prix_m2"] = (df.loc[ok, "prix"] / df.loc[ok, "surface"]).round(0)

    # Supprimer aberrations
    df = df[(df["prix_m2"] > 200) & (df["prix_m2"] < 20000)].copy()
    df = df.reset_index(drop=True)

    return df


def charger_apl() -> pd.DataFrame:
    chemin_xlsx = DATA_DIR / "apl.xlsx" # chemin possible pour le fichier apl ( je l ai trouver en xlsx mais au cas ou on a un csv j ai mis l autre)
    chemin_csv  = DATA_DIR / "apl.csv"

    try:
        if chemin_xlsx.exists():
            df = pd.read_excel(
                chemin_xlsx, sheet_name="APL 2023", header=8, dtype=str
            )
            df = df.iloc[1:].reset_index(drop=True) 
            cols = df.columns.tolist()
            df = df.rename(columns={     # on rename pour rendre les choses plus claire
                cols[0]: "code_insee",
                cols[1]: "commune_apl",
                cols[2]: "apl_score",
            })
        elif chemin_csv.exists():    
            df = pd.read_csv(chemin_csv, sep=";", dtype=str)
            df.columns = ["code_insee", "commune_apl", "apl_score"] + list(df.columns[3:])
        else:
            return pd.DataFrame()
        # on converti le score apl en nombre decimal car le code insee doit toujours faire 5 chiffres 
        df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)
        df["apl_score"]  = pd.to_numeric(df["apl_score"], errors="coerce")
        df = df[["code_insee", "apl_score"]].dropna().reset_index(drop=True) # ici on garde juste les 2 colones utiles
        
        return df

    except Exception as e:
        print(f"Erreur APL : {e}")
        return pd.DataFrame()


def charger_geo() -> pd.DataFrame:
    import requests
    cache = DATA_DIR / "geo_cache.csv"  # chemin du cache local histoire de pas tout retelecharger a chaque fois
    if cache.exists():
        return pd.read_csv(cache, dtype={"code_insee": str})
    # appel a l api, on demande uniquement ce qu on a besoin
    try:
        r = requests.get(
            "https://geo.api.gouv.fr/communes",
            params={"fields": "code,nom,population,surface,departement", "format": "json"},
            timeout=30, # j ai mis ca pour ne pas blouquer indefiniment si l api est lente
        )
        records = []
        for c in r.json():
            pop  = c.get("population") or 0 # ca c est pour eviter les erreures si le champ est vide
            surf = (c.get("surface") or 0) / 100 # et ca,ca  renvoie la surface en hectares, on divise par 100 pour avoir des km²
            records.append({
                "code_insee":  str(c.get("code", "")).zfill(5), # ici on garantit encore que le code INSEE fait toujours 5 chiffres
                "commune":     c.get("nom"),
                "population":  pop,
                "densite":     round(pop / surf, 1) if surf > 0 else 0,
                "departement": (c.get("departement") or {}).get("code", ""),
            })
        df = pd.DataFrame(records)
        df.to_csv(cache, index=False) # on sauvegarde le resultat en local pour les prochains lancements
        return df
    except Exception as e:
        
        return pd.DataFrame()


def charger_zrr() -> pd.DataFrame:
    import requests
    from io import BytesIO, StringIO

    cache = DATA_DIR / "zrr_cache.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"code_insee": str})

    try:
        meta = requests.get(
            "https://www.data.gouv.fr/api/1/datasets/zones-de-revitalisation-rurale-zrr/",
            timeout=30,
        ).json()

        ressource = next(
            (r for r in meta.get("resources", [])
             if r.get("format", "").lower() in {"csv", "xls", "xlsx"}),
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
                (c for c in df_raw.columns
                 if c.upper() in {"CODGEO", "CODE_COMMUNE", "CODE_INSEE", "COG", "CODE INSEE"}),
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
            # Les vraies en-têtes sont en ligne 4 ; la ligne 5 est un sous-header technique (CODGEO, ZRR_SIMP…)
            df_raw = pd.read_excel(
                BytesIO(raw.content), sheet_name=0, header=4, dtype=str, engine=engine
            )
            df = df_raw.iloc[1:].reset_index(drop=True)
            col_code = df.columns[0]  # 'Code Insee'
            col_zrr  = df.columns[2]  # 'Classement en ZRR'

        df = df.rename(columns={col_code: "code_insee"})
        df["code_insee"] = df["code_insee"].astype(str).str.strip().str.zfill(5)

        if col_zrr:
            # "NC - Commune non classée" est le seul cas négatif ; C et P sont ZRR
            df["is_zrr"] = ~df[col_zrr].str.upper().str.startswith("NC")
        else:
            df["is_zrr"] = True

        df = df[["code_insee", "is_zrr"]].drop_duplicates("code_insee").reset_index(drop=True)
        df.to_csv(cache, index=False)
        return df

    except Exception as e:
        print(f"Erreur ZRR : {e}")
        return pd.DataFrame()


def charger_insee(codes: list = None) -> pd.DataFrame:
    """Récupère revenu_median, age_median et taux_chomage via l'API Melodi (sans clé API)."""
    import requests, time

    cache = DATA_DIR / "insee_cache.csv"

    if cache.exists():
        df_cache = pd.read_csv(cache, dtype={"code_insee": str})
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

    BASE  = "https://api.insee.fr/melodi/data"
    DELAI = 60.0 / 30  # 30 req/min → 2 s entre chaque requête

    # Tranches d'âge RP non chevauchantes → (borne_inf, borne_sup)
    AGE_BOUNDS = {
        "Y_LT15": (0,  15),
        "Y15T24": (15, 25),
        "Y25T39": (25, 40),
        "Y40T54": (40, 55),
        "Y55T64": (55, 65),
        "Y65T79": (65, 80),
        "Y_GE80": (80, 100),
    }

    def _obs_value(r: requests.Response) -> float:
        """Extrait la première valeur numérique d'une réponse Melodi."""
        try:
            obs = r.json().get("observations", [])
            return obs[0]["measures"]["OBS_VALUE_NIVEAU"]["value"] if obs else np.nan
        except Exception:
            return np.nan

    def _get(url: str, params: dict) -> requests.Response:
        """GET avec retry automatique sur 429."""
        for tentative in range(3):
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 429:
                return r
            time.sleep(65)  # attend la fin de la fenêtre de rate limit
        return r

    def _age_median(pops: dict) -> float:
        """Interpole linéairement la médiane d'âge depuis les tranches RP."""
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

        # revenu_median — DS_FILOSOFI_CC, niveau de vie médian (€/an)
        try:
            r = _get(f"{BASE}/DS_FILOSOFI_CC",
                     {"GEO": geo, "FILOSOFI_MEASURE": "MED_SL",
                      "UNIT_MEASURE": "EUR_YR", "maxResult": 1})
            row["revenu_median"] = _obs_value(r)
        except Exception:
            row["revenu_median"] = np.nan
        time.sleep(DELAI)

        # taux_chomage — DS_RP_EMPLOI_LR_PRINC
        # EMPSTA_ENQ=1 : actifs occupés, EMPSTA_ENQ=2 : chômeurs (15-64 ans)
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

        # age_median — DS_RP_POPULATION_PRINC, interpolation sur les tranches d'âge
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

    if df_dvf.empty: # si les donnees DVF sont absentes, on ne peut pas continuer
        return pd.DataFrame()

    # agregation par commune
    df = (
        df_dvf.groupby("code_commune", as_index=False)
        .agg(
            prix_m2_median  = ("prix_m2", "median"),
            prix_m2_moyen   = ("prix_m2", "mean"),
            nb_ventes       = ("prix_m2", "count"),
            surface_mediane = ("surface", "median"),
        )
    )
    df = df.rename(columns={"code_commune": "code_insee"}) # on renomme la colonne pour avoir un nom coherent avec les autres datasets
    df["prix_m2_median"] = df["prix_m2_median"].round(0)

    # pourcentage des maisons
    pct = (
        df_dvf.groupby("code_commune", as_index=False)["type_bien"]
        .apply(lambda x: round((x == "Maison").mean() * 100, 1))
        .rename(columns={"code_commune": "code_insee", "type_bien": "pct_maisons"})
    )
    df = df.merge(pct, on="code_insee", how="left")

    # APL
    if not df_apl.empty:
        df = df.merge(df_apl, on="code_insee", how="left")
        df["desert_medical"] = (df["apl_score"] < 2.5).astype(int)

    # GEO
    if not df_geo.empty:
        cols = [c for c in ["code_insee","commune","population","densite","departement"]
                if c in df_geo.columns]
        df = df.merge(df_geo[cols], on="code_insee", how="left")
    # Basée sur la densité de population (hab/km²
    if "densite" in df.columns:
        df["type_zone"] = pd.cut(
            df["densite"], bins=[-1, 50, 500, 999999],
            labels=["Rural", "Périurbain", "Urbain"],
        )

    # ZRR
    df_zrr = charger_zrr()
    if not df_zrr.empty:
        df = df.merge(df_zrr, on="code_insee", how="left")
        df["is_zrr"] = df["is_zrr"].fillna(False).astype(int)

    # INSEE
    df_insee = charger_insee(codes=df["code_insee"].tolist())
    if not df_insee.empty:
        cols_insee = [c for c in ["code_insee", "revenu_median", "age_median", "taux_chomage"]
                      if c in df_insee.columns]
        df = df.merge(df_insee[cols_insee], on="code_insee", how="left")

    # On garde uniquement les communes avec au moins 3 ventes
    df = df[df["nb_ventes"] >= 3].reset_index(drop=True)
    return df


def entrainer_modele(df: pd.DataFrame) -> dict:
    # liste des variables explicatives qu on veut utiliser pour la prediction
    features_dispo = [
        "apl_score", "densite", "surface_mediane", "nb_ventes", "pct_maisons",
        "is_zrr", "revenu_median", "age_median", "taux_chomage",
    ]
    features = [f for f in features_dispo if f in df.columns]
    df_ml = df[features + ["prix_m2_median"]].dropna()
    # On verifie qu on a assez de donnees pour entrainer un modèle en dessous de 30 communes les resultats ne seraient pas fiables
    if len(df_ml) < 30:
        return {}

    X, y = df_ml[features], df_ml["prix_m2_median"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Noms lisibles pour afficher les variables dans le dashboar
    labels = {
        "apl_score":       "Score APL (accès médecins)",
        "densite":         "Densité population",
        "surface_mediane": "Surface médiane",
        "nb_ventes":       "Nombre de ventes",
        "pct_maisons":     "% de maisons",
        "is_zrr":          "Zone de revitalisation rurale",
        "revenu_median":   "Revenu médian (€/an)",
        "age_median":      "Âge médian",
        "taux_chomage":    "Taux de chômage (%)",
    }
    # Tableau de l importance de chaque variable dans la prediction trie du plus important au moins important
    df_imp = pd.DataFrame({
        "variable":   [labels.get(f, f) for f in features],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model, "features": features,
        "r2":    round(r2_score(y_test, y_pred), 3),
        "mae":   round(mean_absolute_error(y_test, y_pred), 0),
        "n":     len(df_ml), "importance": df_imp,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
    }