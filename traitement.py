from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)


DATA_DIR = Path("/Users/julietterey/Downloads/Projet-API-GEO-Juliette/data")


#features ml
FEATURES = ["apl_score", "densite", "surface_mediane", "nb_ventes"]
TARGET   = "prix_m2_median"
 
FEATURE_LABELS = {
    "apl_score":       "Score APL (accès médecins)",
    "densite":         "Densité population",
    "surface_mediane": "Surface médiane",
    "nb_ventes":       "Nombre de ventes",
}
 
 
#implementation des données 

#file DVF : valeur fonciere 
def charger_dvf() -> pd.DataFrame:
    chemin = DATA_DIR / "dvf.csv"
    if not chemin.exists():
        return pd.DataFrame()

    df = pd.read_csv(chemin, sep=",", low_memory=False, dtype=str)

    #Renommage flexible selon la version du fichier
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

    #Supp collonnes doubles  
    df = df.loc[:, ~df.columns.duplicated()]

    #conversions numériques
    for col in ["prix", "surface", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].str.replace(",", ".").pipe(pd.to_numeric, errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["annee"] = df["date"].dt.year

    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str).str.zfill(5)

    print(f"DVF chargé : {len(df)} lignes brutes")

    #garder seulement maisons et appartements
    if "type_bien" in df.columns:
        df = df[df["type_bien"].isin(["Maison", "Appartement"])]

    #calcul prix au m2
    mask = (df["surface"] > 0) & df["prix"].notna() & (df["surface"] < 1000)
    df.loc[mask, "prix_m2"] = df.loc[mask, "prix"] / df.loc[mask, "surface"]

    #filtrage outliers 
    df = df[(df["prix_m2"] > 200) & (df["prix_m2"] < 20000)]

    df = df.reset_index(drop=True)
    print(f"DVF après filtres : {len(df)} lignes — {df['code_insee'].nunique()} communes uniques")
    return df

#File APL : accessibilité potentielle localsiée 
def charger_apl() -> pd.DataFrame:
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



#API

#implementation API GEO : lat/lon → code INSEE + métadonnées communes
def charger_geo() -> pd.DataFrame:
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

#DATASET FINAL (fusion files)

# DVF  + APL + API GEO
def construire_dataset() -> pd.DataFrame:
    df_dvf = charger_dvf()
    df_apl = charger_apl()
    df_geo = charger_geo()

    if df_dvf.empty:
        return pd.DataFrame()

    #agrégation dvf x commune 
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

    #pourcentage de maisons
    if "type_bien" in df_dvf.columns:
        pct = (
            df_dvf.groupby("code_insee")["type_bien"]
            .apply(lambda x: (x == "Maison").mean() * 100)
            .reset_index()
            .rename(columns={"type_bien": "pct_maisons"})
        )
        df = df.merge(pct, on="code_insee", how="left")

    #jointure APL
    if not df_apl.empty:
        df = df.merge(df_apl, on="code_insee", how="left")
        df["desert_medical"] = (df["apl_score"] < 2.5).astype(int)

    #jointure API GEO 
    if not df_geo.empty:
        geo_cols = ["code_insee", "commune", "population", "densite", "departement"]
        geo_cols = [c for c in geo_cols if c in df_geo.columns]
        df = df.merge(df_geo[geo_cols], on="code_insee", how="left")

    #classificiation des zones 
    if "densite" in df.columns:
        df["type_zone"] = pd.cut(
            df["densite"],
            bins=[-1, 50, 500, 999999],
            labels=["Rural", "Périurbain", "Urbain"],
        )
    elif "commune" not in df.columns:
        df["commune"] = df["code_insee"]

    #filtrage commune ss ventes
    #trop petits
    df = df[df["nb_ventes"] >= 1]

    return df.reset_index(drop=True)



#MODELES MACHINE LEARNING 

#selection features dispo
def _preparer_X_y(df: pd.DataFrame):
    features_dispo = [f for f in FEATURES if f in df.columns]
    df_ml = df[features_dispo + [TARGET]].dropna()
    if len(df_ml) < 30:
        return None, None, []
    X = df_ml[features_dispo]
    y = df_ml[TARGET]
    return X, y, features_dispo
 
#metrqiues : R2, MAE, RSME
def _metriques(y_test, y_pred) -> dict:
    return {
        "r2":   round(float(r2_score(y_test, y_pred)), 3),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 0),
    }
 

#importance de svariables selon estimator
def _extraire_importance(estimateur, features: list) -> pd.DataFrame:
    labels = [FEATURE_LABELS.get(f, f) for f in features]
 
    if hasattr(estimateur, "feature_importances_"):
        imp = estimateur.feature_importances_
 
    elif hasattr(estimateur, "coef_"):
        coef = np.abs(estimateur.coef_)
        imp  = coef / coef.sum() if coef.sum() > 0 else np.ones(len(features)) / len(features)
 
    elif hasattr(estimateur, "estimators_"):
        imps = [sub.feature_importances_
                for sub in estimateur.estimators_
                if hasattr(sub, "feature_importances_")]
        imp = np.mean(imps, axis=0) if imps else np.ones(len(features)) / len(features)
 
    else:
        imp = np.ones(len(features)) / len(features)
 
    return pd.DataFrame({
        "variable":   labels,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
 
 
# Catalogue complet : modèle de base + grille d'hyperparamètres

_CATALOGUE_HP = {
 
    #LINEAR REGRESSION
    "Régression linéaire": {
        "model": Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        "params": {},  # pas d'hyperparamètre à tuner
    },
 
    #RIDGE
    "Ridge (L2)": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", Ridge())]),
        "params": {"m__alpha": [0.1, 1.0, 10.0, 100.0]},
    },
 
    #LASSO
    "LASSO (L1)": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", Lasso(max_iter=5000))]),
        "params": {"m__alpha": [0.1, 1.0, 10.0, 100.0]},
    },
 
    #KNN
    "K-Nearest Neighbors": {
        "model":  Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor())]),
        "params": {"m__n_neighbors": [3, 5, 7, 10],
                   "m__weights":     ["uniform", "distance"]},
    },
 
    #DECISION TREE
    "Decision Tree": {
        "model":  DecisionTreeRegressor(random_state=42),
        "params": {"max_depth":   [4, 6, 8, None],
                   "min_samples_split": [2, 5, 10]},
    },
 
    #RANDOM FOREST
    "Random Forest": {
        "model":  RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {"n_estimators": [100, 200],
                   "max_depth":    [6, 8, None],
                   "min_samples_split": [2, 5]},
    },
 
    #ADABOOST
    "AdaBoost": {
        "model":  AdaBoostRegressor(random_state=42),
        "params": {"n_estimators":  [50, 100, 200],
                   "learning_rate": [0.01, 0.1, 1.0]},
    },
 
    #GRADIENT BOOSTING
    "Gradient Boosting": {
        "model":  GradientBoostingRegressor(random_state=42),
        "params": {"n_estimators":  [100, 200],
                   "max_depth":     [3, 4, 5],
                   "learning_rate": [0.05, 0.1, 0.2]},
    },
 
    #VOTING REGRESSOR
    "Voting Regressor": {
        "model": VotingRegressor(estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=6,
                                         random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, random_state=42)),
            ("dt", DecisionTreeRegressor(max_depth=6, random_state=42)),
        ]),
        "params": {},  # pas de tuning sur l'ensemble lui-même
    },
}
 
#entriane les 8 models (gridsearchcv = hyperparametres)
def comparer_tous_modeles(df: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    X, y, features = _preparer_X_y(df)
    if X is None:
        print("[ML] Données insuffisantes pour entraîner les modèles (< 30 communes)")
        return pd.DataFrame()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    resultats = []
 
    for nom, cfg in _CATALOGUE_HP.items():
        print(f"[ML] Entraînement : {nom}...", end=" ")
        try:
            if cfg["params"]:
                # GridSearchCV sur les hyperparamètres définis
                gs = GridSearchCV(
                    cfg["model"],
                    cfg["params"],
                    scoring="r2",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(X_train, y_train)
                best_model  = gs.best_estimator_
                best_params = gs.best_params_
            else:
                #Pas d'hyperparamètre → entraînement direct
                best_model  = cfg["model"]
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
                "_model":           best_model,   # objet modèle (usage interne)
                "_features":        features,
            })
 
        except Exception as e:
            print(f"ERREUR : {e}")
            resultats.append({
                "Modèle":           nom,
                "R2":               None,
                "MAE (€/m2)":       None,
                "RMSE (€/m2)":      None,
                "Meilleurs params": "Erreur",
                "_model":           None,
                "_features":        features,
            })
 
    df_res = pd.DataFrame(resultats)
    df_res = df_res.dropna(subset=["R2"]).sort_values("R2", ascending=False).reset_index(drop=True)
    return df_res


#SELECTION DU MEILLEUR MODELE 

#entriane tout les modeles et selection du best (metriques)

def entrainer_modele(df: pd.DataFrame) -> dict:
    X, y, features = _preparer_X_y(df)
    if X is None:
        return {}
 
    #COmparaison de tout les modeles 
    print("\n[ML] Comparaison de tous les modèles ")
    df_comparaison = comparer_tous_modeles(df)
 
    if df_comparaison.empty:
        return {}
 
    #BEST MODEL
    meilleur = df_comparaison.iloc[0]
    nom_modele  = meilleur["Modèle"]
    best_model  = meilleur["_model"]
    best_feats  = meilleur["_features"]
 
    print(f"\n[ML] Meilleur modèle : {nom_modele} "
          f"(R²={meilleur['R2']:.3f}) ")
 
    #re calcul des predictions 
    X_all, y_all, _ = _preparer_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    y_pred = best_model.predict(X_test)
 
    #Importance des variables 
    #Récupere l'estimateur final (Pipeline)
    if hasattr(best_model, "named_steps"):
        estimateur = best_model.named_steps.get(
            "m", list(best_model.named_steps.values())[-1]
        )
    else:
        estimateur = best_model
 
    importance = _extraire_importance(estimateur, best_feats)
 
    #tableau comparatiuf 
    colonnes_affichage = ["Modèle", "R2", "MAE (€/m2)", "RMSE (€/m2)", "Meilleurs params"]
    df_comparaison_propre = df_comparaison[colonnes_affichage].copy()
 
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
        "comparaison": df_comparaison_propre,
    }

# =============================================================================
# EXECUTION PRINCIPALE
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Chargement et construction du dataset
    print("\nCHARGEMENT DES DONNÉES:")
    df = construire_dataset()

    if df.empty:
        print("Aucune donnée disponible.")
    else:
        print(f"\nDataset final : {len(df)} communes")
        print(df[["prix_m2_median", "apl_score", "densite", "nb_ventes"]].describe().round(2))

        # Matrice de corrélation
        print("\nMATRICE DE CORRÉLATION:")
        cols_corr = [c for c in FEATURES + [TARGET] if c in df.columns]
        corr = df[cols_corr].corr().round(2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5)
        plt.title("Matrice de corrélation — features ML")
        plt.tight_layout()
        plt.show()

        # Entraînement de tous les modèles + sélection du meilleur
        print("\nENTRAÎNEMENT DES MODÈLES:")
        res = entrainer_modele(df)

        if res:
            # Tableau comparatif
            print("\nCOMPARAISON DES MODÈLES:")
            print(res["comparaison"].to_string(index=False))

            # Résumé du meilleur modèle
            print(f"  MEILLEUR MODÈLE : {res['nom_modele']}")
            print(f"  R2   : {res['r2']}")
            print(f"  MAE  : {res['mae']:.0f} €/m²")
            print(f"  RMSE : {res['rmse']:.0f} €/m²")
            print(f"  Communes utilisées : {res['n']}")

            # Importance des variables
            print("\nIMPORTANCE DES VARIABLES :")
            print(res["importance"].to_string(index=False))

            plt.figure(figsize=(7, 4))
            plt.barh(res["importance"]["variable"], res["importance"]["importance"],
                     color="#4e8df5")
            plt.xlabel("Importance")
            plt.title(f"Importance des variables — {res['nom_modele']}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

            # 7. Graphique Réel vs Prédit
            plt.figure(figsize=(6, 6))
            plt.scatter(res["y_test"], res["y_pred"], alpha=0.5, color="#4e8df5")
            mn = min(res["y_test"].min(), res["y_pred"].min())
            mx = max(res["y_test"].max(), res["y_pred"].max())
            plt.plot([mn, mx], [mn, mx], "r--", label="Prédiction parfaite")
            plt.xlabel("Prix réel (€/m2)")
            plt.ylabel("Prix prédit (€/m2)")
            plt.title(f"Réel vs Prédit — {res['nom_modele']}")
            plt.legend()
            plt.tight_layout()
            plt.show()
 
