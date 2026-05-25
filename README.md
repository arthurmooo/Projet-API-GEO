# Territoire Immo - API GEO

Prototype d'aide à la décision territoriale pour l'immobilier dans le département 01.

Le projet combine les ventes DVF, les données INSEE/OFGL, l'accès aux soins APL et les zones ZRR pour proposer deux usages :

- une API FastAPI avec frontend Next.js ;
- un dashboard Streamlit historique.

## Feature principale

La recherche intelligente classe les communes selon la faisabilité d'un projet immobilier :

- budget total ;
- surface cible ;
- rayon autour d'une commune pivot ;
- accès aux soins via APL ;
- dynamique démographique ;
- activité du marché local ;
- type de zone rural, périurbain ou urbain.

Le score est indicatif et explicable. Il ne prétend pas estimer précisément un bien à l'adresse.

## Installation

```bash
pip install -r requirements.txt
cd frontend
npm install
```

## Données

### DVF

1. Télécharger un fichier depuis https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/
2. Le placer dans `data/dvf.csv`

### APL

Le projet attend les données d'accessibilité potentielle localisée dans `data/apl.xlsx`.

### Caches locaux

Le chargement peut créer ou réutiliser :

- `data/population_ofgl_cache.csv`
- `data/insee_cache.csv`
- `data/zrr_cache.csv`

## Lancer l'API + frontend

Terminal 1 :

```bash
uvicorn api:app --reload
```

Terminal 2 :

```bash
cd frontend
npm run dev
```

Frontend : http://localhost:3000

API : http://localhost:8000

## Lancer Streamlit

```bash
streamlit run app.py
```

## Validation

```bash
python -m py_compile api.py app.py traitement.py listing_parser.py
cd frontend
npm run lint
npm run build
```
