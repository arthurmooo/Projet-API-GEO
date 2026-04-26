# 🏥 Densité médicale × Prix immobiliers — Version simplifiée

## Structure (3 fichiers seulement)

```
projet_immobilier/
├── app.py            ← Dashboard Streamlit
├── traitement.py     ← Chargement + ML
├── requirements.txt
└── data/
    ├── dvf.csv       ← À télécharger (voir ci-dessous)
    └── apl.csv       ← À télécharger (optionnel)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Télécharger les données

### DVF (obligatoire)
1. https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/
2. Télécharge le fichier de ton département (ex: `76.csv`)
3. Renomme-le **`dvf.csv`** → place dans `data/`

### APL (recommandé)
1. https://data.drees.solidarites-sante.gouv.fr
2. Cherche "Accessibilité Potentielle Localisée APL"
3. Télécharge le CSV → renomme **`apl.csv`** → place dans `data/`

### Population communale OFGL / INSEE (automatique)
Le projet interroge automatiquement l'API OFGL :

https://data.ofgl.fr/explore/dataset/populations-ofgl-communes/api/

Source : traitement OFGL à partir de données INSEE, Licence Ouverte Etalab 2.0.

Variables ajoutées :
- population municipale 2012, 2022 et 2024
- variation de population 2012-2022 et 2022-2024
- typologies commune rurale, touristique et montagne

Un cache local est créé au premier lancement :

```
data/population_ofgl_cache.csv
```

---

## Lancer

```bash
streamlit run app.py
```

Le dashboard s'ouvre dans le navigateur.  
Sans données → mode démo automatique.
