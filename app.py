# =============================================================================
# app.py — Dashboard Streamlit (version simplifiée)
# Densité médicale × Prix immobiliers
# =============================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traitement import construire_dataset, entrainer_modele, charger_dvf, charger_apl

# =============================================================================
# Config
# =============================================================================

st.set_page_config(
    page_title="Densité médicale × Prix immobiliers",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; }
.titre-section {
    font-size: 1.1rem; font-weight: 700;
    color: #4e8df5; border-bottom: 2px solid #4e8df5;
    padding-bottom: 4px; margin: 1rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Chargement (avec cache)
# =============================================================================

@st.cache_data
def charger_tout():
    return construire_dataset()


# =============================================================================
# Sidebar
# =============================================================================

def sidebar(df):
    st.sidebar.title("🔧 Filtres")
    df_f = df.copy()

    # Département
    if "departement" in df.columns:
        depts = sorted(df["departement"].dropna().unique())
        sel = st.sidebar.multiselect("Département", depts, default=list(depts))
        if sel:
            df_f = df_f[df_f["departement"].isin(sel)]

    # Type de zone
    if "type_zone" in df.columns:
        zones = df["type_zone"].dropna().unique().tolist()
        sel_z = st.sidebar.multiselect("Type de zone", zones, default=zones)
        if sel_z:
            df_f = df_f[df_f["type_zone"].isin(sel_z)]

    # Score APL
    if "apl_score" in df.columns and df["apl_score"].notna().any():
        vmin = float(df["apl_score"].min())
        vmax = float(df["apl_score"].max())
        rng = st.sidebar.slider("Score APL", vmin, vmax, (vmin, vmax), 0.1)
        df_f = df_f[df_f["apl_score"].between(*rng)]

    st.sidebar.markdown("---")
    st.sidebar.metric("Communes sélectionnées", f"{len(df_f):,}")
    return df_f


# =============================================================================
# Onglet 1 — Vue d'ensemble
# =============================================================================

def onglet_overview(df):
    has_apl = "apl_score" in df.columns and df["apl_score"].notna().any()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Communes", f"{len(df):,}")
    c2.metric("Prix médian au m²", f"{df['prix_m2_median'].median():,.0f} €")
    c3.metric("Total ventes", f"{df['nb_ventes'].sum():,}")
    if has_apl:
        nb_desert = (df["apl_score"] < 2.5).sum()
        c4.metric("Déserts médicaux", f"{nb_desert} ({nb_desert/len(df)*100:.0f}%)")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Distribution prix
    with col1:
        st.subheader("Distribution des prix au m²")
        color = "type_zone" if "type_zone" in df.columns else None
        fig = px.histogram(
            df.dropna(subset=["prix_m2_median"]),
            x="prix_m2_median", color=color, nbins=40,
            labels={"prix_m2_median": "Prix médian (€/m²)"},
            color_discrete_map={"Rural":"#27ae60","Périurbain":"#f39c12","Urbain":"#2980b9"},
        )
        fig.update_layout(template="plotly_dark", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Distribution APL
    with col2:
        if has_apl:
            st.subheader("Distribution du score APL")
            fig = px.histogram(
                df.dropna(subset=["apl_score"]), x="apl_score", nbins=40,
                color_discrete_sequence=["#4e8df5"],
                labels={"apl_score": "Score APL"},
            )
            fig.add_vline(x=2.5, line_dash="dash", line_color="red",
                          annotation_text="Seuil désert (2.5)")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Répartition par type de zone")
            if "type_zone" in df.columns:
                counts = df["type_zone"].value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                             color_discrete_map={"Rural":"#27ae60","Périurbain":"#f39c12","Urbain":"#2980b9"})
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    # Top / Flop communes
    st.markdown("---")
    nom_col = "commune" if "commune" in df.columns else "code_insee"
    afficher = [nom_col, "prix_m2_median", "nb_ventes"]
    if has_apl:
        afficher += ["apl_score"]
    afficher = [c for c in afficher if c in df.columns]

    c_top, c_bot = st.columns(2)
    with c_top:
        st.markdown("**🔴 Communes les plus chères**")
        st.dataframe(df.nlargest(8, "prix_m2_median")[afficher].reset_index(drop=True))
    with c_bot:
        st.markdown("**🔵 Communes les moins chères**")
        st.dataframe(df.nsmallest(8, "prix_m2_median")[afficher].reset_index(drop=True))


# =============================================================================
# Onglet 2 — APL × Prix
# =============================================================================

def onglet_apl(df):
    has_apl = "apl_score" in df.columns and df["apl_score"].notna().any()

    if not has_apl:
        st.warning("⚠️ Fichier APL non chargé. Ajoute `data/apl.csv` pour voir cette analyse.")
        return

    df_p = df.dropna(subset=["apl_score", "prix_m2_median"])
    corr = df_p["apl_score"].corr(df_p["prix_m2_median"])

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Corrélation APL × Prix", f"{corr:.3f}",
              help="Positif = plus d'accès aux soins → prix plus élevés")

    desert_mask = df_p["apl_score"] < 2.5
    p_desert  = df_p[desert_mask]["prix_m2_median"].median()
    p_couvert = df_p[~desert_mask]["prix_m2_median"].median()

    if pd.notna(p_desert) and pd.notna(p_couvert):
        decote = (p_couvert - p_desert) / p_couvert * 100
        c2.metric("Prix médian désert médical", f"{p_desert:,.0f} €/m²")
        c3.metric("Décote vs zones couvertes", f"-{decote:.1f}%",
                  delta=f"vs {p_couvert:,.0f} €/m²")

    st.markdown("---")

    # Scatter
    st.subheader("Score APL vs Prix au m²")
    color = "type_zone" if "type_zone" in df_p.columns else None
    nom   = "commune"   if "commune"   in df_p.columns else None

    fig = px.scatter(
        df_p, x="apl_score", y="prix_m2_median",
        color=color,
        size="nb_ventes", size_max=18,
        hover_name=nom,
        trendline="ols",
        labels={"apl_score":"Score APL","prix_m2_median":"Prix médian (€/m²)"},
        color_discrete_map={"Rural":"#27ae60","Périurbain":"#f39c12","Urbain":"#2980b9"},
    )
    fig.add_vline(x=2.5, line_dash="dot", line_color="red",
                  annotation_text="Seuil désert")
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Box par quintile
    st.subheader("Prix par niveau d'accès aux soins")
    df_q = df_p.copy()
    df_q["Niveau APL"] = pd.qcut(
        df_q["apl_score"], 4,
        labels=["Très faible\n(désert)", "Faible", "Moyen", "Bon accès"],
        duplicates="drop",
    )
    fig2 = px.box(df_q, x="Niveau APL", y="prix_m2_median", color="Niveau APL",
                  labels={"prix_m2_median": "Prix médian (€/m²)"},
                  color_discrete_sequence=["#e74c3c","#e67e22","#f1c40f","#27ae60"])
    fig2.update_layout(template="plotly_dark", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# Onglet 3 — Carte
# =============================================================================

def onglet_carte(df):
    lat_col = next((c for c in ["latitude","lat"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude","lon"] if c in df.columns), None)

    if lat_col is None or lon_col is None:
        st.info("""
        📍 **Coordonnées GPS non disponibles.**

        Le fichier DVF de data.gouv.fr contient les colonnes latitude/longitude.
        Assure-toi d'avoir téléchargé le fichier **complet** (pas la version allégée).
        """)
        return

    df_map = df.dropna(subset=[lat_col, lon_col, "prix_m2_median"])
    df_map = df_map[df_map[lat_col].between(41, 51)]

    choix = st.selectbox(
        "Colorier par :",
        ["prix_m2_median", "apl_score"] if "apl_score" in df.columns else ["prix_m2_median"],
        format_func=lambda x: {"prix_m2_median": "Prix au m²", "apl_score": "Score APL"}.get(x, x),
    )

    nom = "commune" if "commune" in df_map.columns else None
    fig = px.scatter_mapbox(
        df_map, lat=lat_col, lon=lon_col,
        color=choix, size="prix_m2_median", size_max=12,
        hover_name=nom,
        color_continuous_scale="RdYlGn" if choix == "apl_score" else "Viridis",
        zoom=7, height=600,
    )
    fig.update_layout(mapbox_style="open-street-map",
                      margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Onglet 4 — Modèle ML
# =============================================================================

def onglet_ml(df):
    st.markdown("""
    Le modèle **Random Forest** prédit le prix au m² à partir des caractéristiques
    de chaque commune. L'importance des variables montre le **poids de l'APL**
    dans la formation des prix.
    """)

    if st.button("🚀 Entraîner le modèle", type="primary"):
        with st.spinner("Entraînement..."):
            res = entrainer_modele(df)
            st.session_state["ml"] = res

    res = st.session_state.get("ml")

    if not res:
        st.info("Clique sur 'Entraîner' pour lancer le modèle.")
        return

    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{res['r2']:.3f}", help="1.0 = prédiction parfaite")
    c2.metric("MAE", f"{res['mae']:,.0f} €/m²", help="Erreur absolue moyenne")
    c3.metric("Communes utilisées", f"{res['n']:,}")

    col1, col2 = st.columns(2)

    # Importance
    with col1:
        st.subheader("Importance des variables")
        fig = px.bar(
            res["importance"], x="importance", y="variable",
            orientation="h", color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(template="plotly_dark", showlegend=False,
                          coloraxis_showscale=False,
                          yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        # Mise en avant APL
        apl_row = res["importance"][res["importance"]["variable"].str.contains("APL")]
        if not apl_row.empty:
            rank = int(apl_row.index[0]) + 1
            imp  = float(apl_row["importance"].iloc[0])
            st.success(
                f"📌 L'**accès aux médecins (APL)** est la **{rank}ème variable** "
                f"la plus importante ({imp*100:.1f}%)"
            )

    # Réel vs Prédit
    with col2:
        st.subheader("Réel vs Prédit")
        fig = px.scatter(
            x=res["y_test"], y=res["y_pred"],
            labels={"x":"Prix réel (€/m²)", "y":"Prix prédit (€/m²)"},
            opacity=0.5, color_discrete_sequence=["#4e8df5"],
        )
        mn = min(res["y_test"].min(), res["y_pred"].min())
        mx = max(res["y_test"].max(), res["y_pred"].max())
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(color="red", dash="dash"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Onglet 5 — Données
# =============================================================================

def onglet_donnees(df):
    st.metric("Communes", f"{len(df):,}")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    recherche = st.text_input("🔍 Rechercher une commune")
    nom_col   = "commune" if "commune" in df.columns else "code_insee"
    df_show   = df[df[nom_col].str.contains(recherche, case=False, na=False)] if recherche else df
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=400)

    st.download_button(
        "⬇️ Télécharger le dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "dataset_final.csv", "text/csv",
    )


# =============================================================================
# Application principale
# =============================================================================

def main():
    st.title("🏥 Densité médicale × Prix immobiliers")
    st.caption("Impact de l'accessibilité aux soins (score APL) sur les prix immobiliers en France")

    df = charger_tout()

    # --- Mode démo si pas de données ---
    if df.empty:
        st.warning("**Aucune donnée trouvée.** Place tes fichiers CSV dans le dossier `data/`.")

        with st.expander("📥 Comment télécharger les données ?", expanded=True):
            st.markdown("""
**Fichier 1 — DVF (transactions immobilières)**
1. Va sur [data.gouv.fr/dvf](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)
2. Télécharge le fichier du département voulu (ex: `76.csv` pour Seine-Maritime)
3. Renomme-le `dvf.csv` et place-le dans le dossier `data/`

**Fichier 2 — APL (optionnel mais recommandé)**
1. Va sur [data.drees.solidarites-sante.gouv.fr](https://data.drees.solidarites-sante.gouv.fr)
2. Cherche "Accessibilité Potentielle Localisée"
3. Télécharge le CSV et renomme-le `apl.csv` dans `data/`

Ensuite **relance le dashboard** : `streamlit run app.py`
            """)

        st.info("📊 Démo avec données simulées...")
        df = _demo()

    df_filtré = sidebar(df)

    tabs = st.tabs(["📊 Vue d'ensemble", "🏥 APL × Prix", "🗺️ Carte", "🤖 Modèle ML", "📋 Données"])
    with tabs[0]: onglet_overview(df_filtré)
    with tabs[1]: onglet_apl(df_filtré)
    with tabs[2]: onglet_carte(df_filtré)
    with tabs[3]: onglet_ml(df_filtré)
    with tabs[4]: onglet_donnees(df_filtré)


def _demo(n=300):
    rng   = np.random.default_rng(42)
    types = rng.choice(["Rural","Périurbain","Urbain"], n, p=[0.45,0.35,0.20])
    apl   = np.where(types=="Rural", rng.normal(1.8,.8,n),
            np.where(types=="Périurbain", rng.normal(3.5,1,n),
                     rng.normal(6,1.5,n))).clip(.1,15)
    prix  = (1200 + apl*180 + (types=="Urbain")*400 + rng.normal(0,200,n)).clip(500,8000)
    return pd.DataFrame({
        "code_insee":     [f"7{i:04d}" for i in range(n)],
        "commune":        [f"Commune_{i}" for i in range(n)],
        "prix_m2_median": prix.round(0),
        "nb_ventes":      rng.integers(5,150,n),
        "apl_score":      apl.round(2),
        "type_zone":      types,
        "desert_medical": (apl<2.5).astype(int),
        "densite":        np.where(types=="Rural",rng.uniform(5,50,n),
                          np.where(types=="Périurbain",rng.uniform(50,500,n),
                                   rng.uniform(500,5000,n))).round(0),
        "departement":    rng.choice(["76","27","14"],n),
        "latitude":       rng.uniform(48.5,50.0,n).round(4),
        "longitude":      rng.uniform(-1.5,2.5,n).round(4),
    })


if __name__ == "__main__":
    main()
