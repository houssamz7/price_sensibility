import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ajoute la racine du projet (qui contient "src") au PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]   # = dossier PROJET_SENSIBILITE_PRIX
sys.path.insert(0, str(ROOT))



from src.analytics import (
    add_event_weight,
    apply_filters,
    aggregate_curve,
    compute_reference_and_best
)

st.title("Dashboard - Données réelles")

# Cache : Streamlit ne relit pas le fichier à chaque interaction

@st.cache_data
def load_clean(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-16")

df = load_clean("data/clean/Projet_Sensibilite_Prix_clean.csv")

# Sidebar : filtres utilisateur
st.sidebar.header("Filtres")

# On ajoute les éléments à filtrer
property_ = st.sidebar.selectbox(
    "Propoerty",
    ["All"] + sorted(df["Property"].astype(str).unique().tolist())

)

mc = st.sidebar.selectbox(
    "Market Code",
    ["All"] + sorted(df["Market Code"].astype(str).unique().tolist())
)

rt = st.sidebar.selectbox(
    "Room Type",
    ["All"] + sorted(df["Room Type"].astype(str).unique().tolist())
)

sc = st.sidebar.selectbox(
    "Source Code",
    ["All"] + sorted(df["Source Code"].astype(str).unique().tolist())
)

season = st.sidebar.selectbox("Season", ["All", "LOW", "HIGH"])

month_num = st.sidebar.selectbox(
    "Month Number",
    ["All"] + sorted(df["Month Number"].dropna().astype(int).unique().tolist())
)

weekday = st.sidebar.selectbox(
    "Weekday",
    ["All"] + sorted(df["Weekday"].astype(str).unique().tolist())

)

event_mode = st.sidebar.selectbox(
    "Gestion Event",
    ["downweight", "exclude", "include_full"]
)

y_mode = st.sidebar.selectbox(
    "Axe Y",
    ["reservations", "nights", "revenue"]
)

bin_size = st.sidebar.slider("Taille bins (Transches de prix) ADR (€)", 5, 50, 15, 5)
# Choisir le prix de référence
reference_mode = st.sidebar.selectbox("Prix de référence", ["median", "mean"])

# Pipeline : pondération + filtre
df2 = add_event_weight(df, mode=event_mode)

# Construction du dictionnaire filters, 
# L’utilisateur choisit dans la sidebar des valeurs : 
# soit une valeur précise (ex: PARHDC), soit “All” (ça veut dire : pas de filtre)
# Donc ici on construit un dictionnaire où : si l’utilisateur a choisi All -> on met None (ça veut dire : “ne filtre pas”), sinon on met la valeur choisie

filters = {
    "Property": None if property_ == "All" else property_,
    "Market Code": None if mc == "All" else mc,
    "Room Type": None if rt == "All" else rt,
    "Source Code": None if sc == "All" else sc,
    "Season": None if season == "All" else season,
    "Month Number": None if month_num == "All" else int(month_num),
    "Weekday": None if weekday == "All" else weekday,
}

# On utilise la fonction apply filters déjà définie sur analytics.py
df2 = apply_filters(df2, filters)
# Afficher le nombre avec des virgules (séparateur de miliers) pour le rendre plus lisible, ** c'est pour rendre gras streamlit
st.write(f"Lignes après filtres : **{len(df2):,}**")
# Warning si peu de données
if len(df2) < 200:
    st.warning("Peu de données après filtres : la courbe est moins stable!")

# Construire la courbe sulement si on a assez de lignes (20 lignes ou plus)
if len(df2) >= 20:
    agg = aggregate_curve(df2, bin_size=bin_size, y_mode=y_mode)

    st.subheader("Courbe Prix -> " + y_mode)
    # Affichage de la courbe
    fig = plt.figure(figsize=(9, 5))
    plt.plot(agg["ADR_mean"], agg["Y"], marker="o")
    plt.title(f"Sensibilité prix - {y_mode}")
    plt.xlabel("Prix d'une nuit (€)")
    # Récupère le libellé de l’axe Y depuis la colonne "Y_label"
    # - .iloc[0] : on prend la première valeur (le libellé est identique pour toutes les lignes)
    # - if len(agg) else "Y" : sécurité si le DataFrame est vide (évite une erreur)
    plt.ylabel(agg["Y_label"].iloc[0] if len(agg) else "Y")
    plt.grid(True)
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)
    # Montrer le tableau agrégé, utile pour vérifier
    st.dataframe(agg)

    # Calcul “prix optimal & gain vs référence”, res est un dictionnaire
    res = compute_reference_and_best(agg, reference_mode)

    # Si le dictionnaire n'est pas vide, on a des résultats à afficher, sinon rien n'est affiché
    if res:
        # On crée 3 colonnes dans l'interface Streamlit, Juste pour afficher 3 chiffres côte à côte
        c1, c2, c3 = st.columns(3)
        # Metric dans streamlit affiche un badge KPI avec un titre, valeur, et une variation
        # res['ref_price'] : récupère la valeur du dictionnaire
        # :.0f : format float sans décimales, .1f = une décimale
        c1.metric("Prix de référence", f"{res['ref_price']:.0f} €")
        c2.metric("Meilleur prix (hist.)", f"{res['best_price']:.0f} €")
        c3.metric("Gain potentiel (bin)", f"{res['gain']:,.0f} €", f"{res['gain_pct']:.1f}%")
    else : 
        st.info("Pas assez de données pour calculer l’optimum.")
else:
    st.info("Pas assez de lignes pour construire une courbe (min 20 lignes).")
