# app/pages/1_Donnees_reelles.py
# ============================================================
# Page Streamlit : Analyse sur données réelles (V1)
# - Courbe de sensibilité prix (visuel principal)
# - Top Nationalities
# - Saisonnalité (heatmap Mois x Jour de semaine)
#
# Cette page suppose que la page Home (streamlit_app.py) a déjà :
# - chargé le fichier brut R&A
# - généré df_clean_final
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analytics import (
    apply_filters,
    aggregate_curve,
    compute_reference_and_best,
    replace_event_period_with_other_years
)

st.title("Données réelles")

# ============================================================
# 0) Mapping Segment -> Market Codes
# ============================================================
# Objectif RM :
# - Un filtre "Segment" (ex: Indiv. Public)
# - Un filtre "Market Code" dépendant du Segment choisi
#
# Exemple :
# - Segment = Indiv. Public
# - Market Code = All  => on regroupe tous les MC du segment
# - Market Code = 3CNR => on filtre uniquement ce MC
SEGMENT_TO_MARKET_CODES = {
    "Indiv. Public": [
        "3CNR", "3CRE", "3FLA", "3NNR", "3NRE",
        "PBAR", "PDIS", "PPKG", "PPRO", "PRES"
    ],
    "Indiv. Negocié": ["NEMB", "NGLO", "NLOC", "NCON"],
    "Indiv Tour Operator": ["ITOU"],
    "Groupe Affaires": ["ETNMT", "GBAH", "GBCE", "GBDI", "GBIN", "GBSE", "GBSP", "GCAS"],
    "Groupe Loisirs": ["GLOI", "GLTS"],
    "Offert": ["OCOM"],
    "Interne": ["OHOU"],
    "Equipages": ["GCME", "GCRE"],
}

# Ordre demandé pour le dropdown Segment (affichage "propre")
SEGMENT_ORDER = [
    "Indiv. Public",
    "Indiv. Negocié",
    "Indiv Tour Operator",
    "Groupe Affaires",
    "Groupe Loisirs",
    "Offert",
    "Interne",
    "Equipages",
]

# ============================================================
# 1) Charger df_clean_final (préparé sur la page Home)
# ============================================================
if "df_clean_final" not in st.session_state:
    st.warning("Va d'abord sur la page Home pour uploader le fichier et générer le CLEAN.")
    st.stop()

df = st.session_state["df_clean_final"].copy()

# Assurer que les colonnes dates sont bien en datetime (sécurité)
for c in ["Arrival Date", "Reservation Date", "Departure Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# ============================================================
# 2) Sidebar - filtres
# ============================================================
st.sidebar.header("Filtres")

# --- Property (Hôtel)
property_ = st.sidebar.selectbox(
    "Property",
    ["All"] + sorted(df["Property"].astype(str).unique().tolist())
)

# --- Segment (ordre imposé)
segment_options = ["All"] + [s for s in SEGMENT_ORDER if s in SEGMENT_TO_MARKET_CODES]
segment = st.sidebar.selectbox("Segment", segment_options)

# --- Market Code dépendant du Segment
# Si Segment = All -> propose tous les MC
# Si Segment = X -> propose uniquement les MC du segment X
if segment == "All":
    mc_options = ["All"] + sorted(df["Market Code"].astype(str).unique().tolist())
else:
    mc_options = ["All"] + SEGMENT_TO_MARKET_CODES.get(segment, [])
mc = st.sidebar.selectbox("Market Code", mc_options)

# --- Room Type
room_type = st.sidebar.selectbox(
    "Room Type",
    ["All"] + sorted(df["Room Type"].astype(str).unique().tolist())
)

# --- Source Code
sc = st.sidebar.selectbox(
    "Source Code",
    ["All"] + sorted(df["Source Code"].astype(str).unique().tolist())
)

# --- Season
season = st.sidebar.selectbox("Season", ["All", "LOW", "HIGH"])

# --- Nationality (multi choix)
nationalities = st.sidebar.multiselect(
    "Nationality",
    sorted(df["Nationality"].astype(str).unique().tolist())
)

# --- Lead Time (Days) (slider)
lt_col = "Lead Time(Days)"
lead_range = None
if lt_col in df.columns:
    lt_min = int(np.nanmin(df[lt_col]))
    lt_max = int(np.nanmax(df[lt_col]))
    lead_range = st.sidebar.slider(
        "Lead Time(Days)",
        lt_min, lt_max,
        (lt_min, lt_max)
    )
else:
    st.sidebar.info("Colonne 'Lead Time(Days)' non trouvée.")

# --- Dates : on choisit quelle date filtrer (Arrival / Reservation / Departure)
date_field = st.sidebar.selectbox(
    "Date utilisée",
    ["Arrival Date", "Reservation Date", "Departure Date"]
)

# Plage de dates
dmin = pd.to_datetime(df[date_field]).min().date()
dmax = pd.to_datetime(df[date_field]).max().date()
date_range = st.sidebar.date_input("Plage de dates", (dmin, dmax))

# --- Filtre "Jour du mois" : ex 1->13 ou 25->30
day_min, day_max = st.sidebar.slider("Jour du mois", 1, 31, (1, 31))

# --- Filtre "Mois" (optionnel) : utile pour "25->30 janvier"
months = st.sidebar.multiselect(
    "Mois (optionnel)",
    options=list(range(1, 13)),
    format_func=lambda x: pd.Timestamp(year=2024, month=x, day=1).strftime("%B"),
    default=[]
)

# --- Options de courbe
y_mode = st.sidebar.selectbox("Axe Y", ["reservations", "nights", "revenue"])
bin_size = st.sidebar.slider("Bins ADR (€)", 5, 50, 15, 5)

# --- Prix de référence
reference_mode = st.sidebar.selectbox("Référence", ["median", "mean"])
ref_weight = st.sidebar.selectbox("Référence pondérée par", ["Nights", "Reservations"])

# ============================================================
# 3) Gestion Event (JO + autres)
# ============================================================
st.sidebar.header("Gestion Event")

event_mode = st.sidebar.selectbox(
    "Traitement des events",
    ["Inclure", "Exclure", "Remplacer (moyenne autres années)"]
)

events_available = sorted([e for e in df["Event"].astype(str).unique().tolist() if e != "None"])
event_to_replace = None
if event_mode == "Remplacer (moyenne autres années)":
    event_to_replace = st.sidebar.selectbox(
        "Quel event remplacer ?",
        options=events_available if events_available else ["(aucun event)"]
    )

# ============================================================
# 4) Application des filtres
# ============================================================

# 4.1 Filtres "simples" via apply_filters (égalité)
filters = {
    "Property": None if property_ == "All" else property_,
    "Room Type": None if room_type == "All" else room_type,
    "Source Code": None if sc == "All" else sc,
    "Season": None if season == "All" else season,
}
df2 = apply_filters(df, filters)

# 4.2 Filtre Segment / Market Code (lié)
# Règle :
# - Segment choisi + MC=All => on garde tous les MC du segment
# - MC choisi => on filtre uniquement ce MC
if segment != "All" and mc == "All":
    allowed = SEGMENT_TO_MARKET_CODES.get(segment, [])
    df2 = df2[df2["Market Code"].astype(str).isin(allowed)]
elif mc != "All":
    df2 = df2[df2["Market Code"].astype(str) == str(mc)]

# 4.3 Nationalities
if nationalities:
    df2 = df2[df2["Nationality"].astype(str).isin(nationalities)]

# 4.4 Lead time
if lead_range is not None:
    df2 = df2[(df2[lt_col] >= lead_range[0]) & (df2[lt_col] <= lead_range[1])]

# 4.5 Date range
start_date, end_date = date_range
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df2 = df2[(df2[date_field] >= start_date) & (df2[date_field] <= end_date)]

# 4.6 Jour du mois
df2 = df2[(df2[date_field].dt.day >= day_min) & (df2[date_field].dt.day <= day_max)]

# 4.7 Mois
if months:
    df2 = df2[df2[date_field].dt.month.isin(months)]

# 4.8 weight (utilisé si on remplace un event)
df2["weight"] = 1.0

# 4.9 Traitement Event
info_msg = None
if event_mode == "Exclure":
    # On retire toutes les lignes qui appartiennent à un event
    df2 = df2[df2["Event"] == "None"].copy()

elif event_mode == "Remplacer (moyenne autres années)":
    # On remplace temporairement l'event par un baseline des autres années
    if events_available and event_to_replace and "(aucun" not in str(event_to_replace):
        df2, info_msg = replace_event_period_with_other_years(
            df2,
            event_to_replace,
            date_col="Arrival Date"
        )

# Infos utilisateur
st.write(f"Lignes après filtres : **{len(df2):,}**")
if info_msg:
    st.info(info_msg)

# Si trop peu de données, pas la peine de tracer
if len(df2) < 20:
    st.warning("Pas assez de données après filtres pour tracer une courbe.")
    st.dataframe(df2.head(200))
    st.stop()

# ============================================================
# 5) Agrégation pour la courbe (bins ADR)
# ============================================================
agg = aggregate_curve(df2, bin_size=bin_size, y_mode=y_mode)

# ============================================================
# 6) Affichage : 3 tabs (comme ta version V1)
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "Courbe sensibilité",
    "Top Nationalities",
    "Saisonnalité (heatmap)"
])

# ------------------------------------------------------------
# TAB 1 : Courbe sensibilité + Prix référence / optimal / gain
# ------------------------------------------------------------
with tab1:
    st.subheader("Courbe de sensibilité prix")

    # --- Zoom intelligent (robuste)
    # On calcule le zoom sur les ADR "réelles" des réservations filtrées (df2),
    # car c'est la meilleure façon d'ignorer les outliers.
    adr_raw = df2["ADR"].dropna().to_numpy(dtype=float)

    # S'il y a assez de points, on garde 99% des valeurs (1% à 99%)
    if len(adr_raw) >= 20:
        xmin = float(np.nanpercentile(adr_raw, 1))
        xmax = float(np.nanpercentile(adr_raw, 99))
    else:
        # Pas assez de données : on ne zoome pas (fallback)
        xmin = float(np.nanmin(agg["ADR_mean"]))
        xmax = float(np.nanmax(agg["ADR_mean"]))

    # On peut aussi filtrer l'affichage des points agrégés,
    # pour ne pas avoir une ligne "plate" à droite
    agg_plot = agg[(agg["ADR_mean"] >= xmin) & (agg["ADR_mean"] <= xmax)].copy()
    if len(agg_plot) < 5:
        # si trop peu de points après filtre, on affiche tout
        agg_plot = agg.copy()

    # --- Plot
    fig = plt.figure(figsize=(9, 5))
    plt.plot(agg_plot["ADR_mean"], agg_plot["Y"], marker="o")
    plt.xlabel("ADR (euros)")
    plt.ylabel(agg_plot["Y_label"].iloc[0] if len(agg_plot) else "Y")
    plt.grid(True)
    plt.xlim(xmin, xmax)
    st.pyplot(fig)

    # --- KPI : Prix de référence / meilleur prix / gain
    st.subheader("Prix optimal et comparaison au prix de référence")

    res = compute_reference_and_best(
        agg,
        reference=reference_mode,
        ref_weight=ref_weight
    )

    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix de référence", f"{res['ref_price']:.0f} euros")
        c2.metric("Meilleur prix (historique)", f"{res['best_price']:.0f} euros")
        c3.metric("Gain (bin)", f"{res['gain']:,.0f} euros", f"{res['gain_pct']:.1f}%")

        st.caption(
            "Note : il est possible que le prix de référence soit égal au prix optimal "
            "si le bin 'typique' (pondéré par volume/nuitées) est aussi celui qui maximise le revenue, "
            "ou si les filtres laissent peu de données."
        )
    else:
        st.info("Impossible de calculer la référence et l'optimum (pas assez de bins).")

# ------------------------------------------------------------
# TAB 2 : Top Nationalities
# ------------------------------------------------------------
with tab2:
    st.subheader("Top Nationalities")

    # On compte les nationalités sur les données filtrées (df2)
    nat = df2["Nationality"].astype(str).value_counts().head(15)

    fig = plt.figure(figsize=(8, 6))
    plt.barh(nat.index[::-1], nat.values[::-1])
    plt.xlabel("Nombre de réservations")
    plt.ylabel("Nationality")
    plt.grid(True, axis="x")
    st.pyplot(fig)

# ------------------------------------------------------------
# TAB 3 : Saisonnalité (heatmap)
# ------------------------------------------------------------
with tab3:
    st.subheader("Saisonnalité : Mois x Jour de semaine (volume)")

    tmp = df2.copy()
    tmp["Weekday"] = tmp["Arrival Date"].dt.day_name()
    tmp["Month Number"] = tmp["Arrival Date"].dt.month

    # Ordre logique pour les jours (sinon c'est alphabétique)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp["Weekday"] = pd.Categorical(tmp["Weekday"], categories=weekday_order, ordered=True)

    # Pivot : lignes = weekday, colonnes = mois, valeurs = volume (count)
    pivot = pd.pivot_table(
        tmp,
        index="Weekday",
        columns="Month Number",
        values="Confirmation Number",
        aggfunc="count",
        fill_value=0
    )

    fig = plt.figure(figsize=(10, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.xlabel("Mois (1-12)")
    plt.ylabel("Jour de semaine")
    plt.title("Volume de réservations (count)")
    plt.colorbar()
    st.pyplot(fig)
