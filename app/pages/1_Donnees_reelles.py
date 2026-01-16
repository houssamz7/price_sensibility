# app/pages/1_Donnees_reelles.py
# ============================================================
# PAGE STREAMLIT — DONNEES REELLES (V1)
# ============================================================
# Cette page sert à analyser la sensibilité au prix à partir des
# données historiques (export R&A -> nettoyage -> df_clean_final).
#
# L'utilisateur (RM) :
# - applique des filtres (Property, Segment, dates, etc.)
# - obtient une courbe "ADR vs Volume/Nuitées/Revenue"
# - obtient un prix de référence + un "meilleur prix historique"
# - peut afficher une table d'agrégation (debug) si besoin
#
# IMPORTANT :
# df_clean_final est créé sur la page Home (streamlit_app.py).
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
# 0) SEGMENTS : mapping Segment -> Market Codes
# ============================================================
# Pourquoi ?
# - Un RM pense en "Segment" (ex: Indiv. Public)
# - Mais il peut aussi vouloir un "Market Code" précis
#
# On propose donc 2 filtres liés :
# 1) Segment
# 2) Market Code (limité aux codes du segment choisi)
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

# Ordre d'affichage demandé pour le filtre Segment
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
# 1) CHARGER LES DONNEES CLEAN
# ============================================================
# Si la page Home n'a pas été utilisée, df_clean_final n'existe pas
if "df_clean_final" not in st.session_state:
    st.warning("Va d'abord sur la page Home pour uploader le fichier et générer le CLEAN.")
    st.stop()

df = st.session_state["df_clean_final"].copy()

# Assurer que les colonnes date sont bien en datetime
for c in ["Arrival Date", "Reservation Date", "Departure Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# ============================================================
# 2) SIDEBAR : FILTRES
# ============================================================
st.sidebar.header("Filtres")

# -----------------------------
# 2.1 Property
# -----------------------------
property_ = st.sidebar.selectbox(
    "Property",
    ["All"] + sorted(df["Property"].astype(str).unique().tolist())
)

# -----------------------------
# 2.2 Segment (ordre imposé)
# -----------------------------
segment_options = ["All"] + [s for s in SEGMENT_ORDER if s in SEGMENT_TO_MARKET_CODES]
segment = st.sidebar.selectbox("Segment", segment_options)

# -----------------------------
# 2.3 Market Code (dépend du Segment)
# -----------------------------
if segment == "All":
    mc_options = ["All"] + sorted(df["Market Code"].astype(str).unique().tolist())
else:
    mc_options = ["All"] + SEGMENT_TO_MARKET_CODES.get(segment, [])
mc = st.sidebar.selectbox("Market Code", mc_options)

# -----------------------------
# 2.4 Room Type
# -----------------------------
room_type = st.sidebar.selectbox(
    "Room Type",
    ["All"] + sorted(df["Room Type"].astype(str).unique().tolist())
)

# -----------------------------
# 2.5 Source Code
# -----------------------------
sc = st.sidebar.selectbox(
    "Source Code",
    ["All"] + sorted(df["Source Code"].astype(str).unique().tolist())
)

# -----------------------------
# 2.6 Season
# -----------------------------
season = st.sidebar.selectbox("Season", ["All", "LOW", "HIGH"])

# -----------------------------
# 2.7 Nationality (multi)
# -----------------------------
nationalities = st.sidebar.multiselect(
    "Nationality",
    sorted(df["Nationality"].astype(str).unique().tolist())
)

# -----------------------------
# 2.8 Lead Time (Days) (nom exact colonne)
# -----------------------------
lt_col = "Lead Time(Days)"
lead_range = None
if lt_col in df.columns:
    lt_min = int(np.nanmin(df[lt_col]))
    lt_max = int(np.nanmax(df[lt_col]))
    lead_range = st.sidebar.slider(
        "Lead Time (Days)",
        lt_min, lt_max,
        (lt_min, lt_max)
    )
else:
    st.sidebar.info(f"Colonne '{lt_col}' non trouvée dans le fichier.")

# -----------------------------
# 2.9 Date utilisée (Arrival/Reservation/Departure)
# -----------------------------
date_field = st.sidebar.selectbox(
    "Date utilisée",
    ["Arrival Date", "Reservation Date", "Departure Date"]
)

# ============================================================
# 2.10 Choose Date Range (avec All par défaut)
# ============================================================
# Objectif :
# - ajouter une option "All" qui prend :
#   start = date min du dataset, end = date max du dataset
# - All doit être le choix par défaut
#
# Ensuite, l'utilisateur peut ajuster à la main via date_input.
dmin = pd.to_datetime(df[date_field]).min().date()
dmax = pd.to_datetime(df[date_field]).max().date()

quick_range = st.sidebar.selectbox(
    "Choose Date Range",
    options=["All", "Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "Past 2 Years"],
    index=0  # All par défaut
)

# On calcule la plage de dates selon le preset choisi
# (on se base sur la date max du dataset pour rester cohérent)
end_default = dmax

if quick_range == "All":
    start_default = dmin
elif quick_range == "Past Week":
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=7)).date())
elif quick_range == "Past Month":
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=30)).date())
elif quick_range == "Past 3 Months":
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=90)).date())
elif quick_range == "Past 6 Months":
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=180)).date())
elif quick_range == "Past Year":
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=365)).date())
else:  # Past 2 Years
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=730)).date())

# Date input final : l'utilisateur peut ajuster manuellement
date_range = st.sidebar.date_input(
    "Plage de dates",
    value=(start_default, end_default),
    min_value=dmin,
    max_value=dmax
)

# Sécurité : date_input peut renvoyer un tuple (start,end) ou une seule date
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# -----------------------------
# 2.11 Jour du mois
# -----------------------------
day_min, day_max = st.sidebar.slider("Jour du mois", 1, 31, (1, 31))

# -----------------------------
# 2.12 Mois (optionnel)
# -----------------------------
months = st.sidebar.multiselect(
    "Mois (optionnel)",
    options=list(range(1, 13)),
    format_func=lambda x: pd.Timestamp(year=2024, month=x, day=1).strftime("%B"),
    default=[]
)

# -----------------------------
# 2.13 Options courbe
# -----------------------------
y_mode = st.sidebar.selectbox("Axe Y", ["reservations", "nights", "revenue"])
bin_size = st.sidebar.slider("Bins ADR (€)", 5, 50, 15, 5)

# -----------------------------
# 2.14 Référence (median/mean)
# -----------------------------
reference_mode = st.sidebar.selectbox("Référence", ["median", "mean"])

# IMPORTANT : on fixe toujours la référence pondérée par nuitées (Nights).
# Donc pas de selectbox ici.
REF_WEIGHT_FIXED = "Nights"

# -----------------------------
# 2.15 Debug : table agg (masquée par défaut)
# -----------------------------
show_agg_table = st.sidebar.checkbox(
    "Afficher table d'agrégation (debug)",
    value=False,
    help="Affiche la table agg (bins ADR) sous la courbe. Utile pour debug."
)

# ============================================================
# 3) GESTION EVENT
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
# 4) APPLICATION DES FILTRES
# ============================================================

# 4.1 Filtres simples
filters = {
    "Property": None if property_ == "All" else property_,
    "Room Type": None if room_type == "All" else room_type,
    "Source Code": None if sc == "All" else sc,
    "Season": None if season == "All" else season,
}
df2 = apply_filters(df, filters)

# 4.2 Segment / Market Code
if segment != "All" and mc == "All":
    allowed = SEGMENT_TO_MARKET_CODES.get(segment, [])
    df2 = df2[df2["Market Code"].astype(str).isin(allowed)]
elif mc != "All":
    df2 = df2[df2["Market Code"].astype(str) == str(mc)]

# 4.3 Nationality
if nationalities:
    df2 = df2[df2["Nationality"].astype(str).isin(nationalities)]

# 4.4 Lead time
if lead_range is not None and lt_col in df2.columns:
    df2 = df2[(df2[lt_col] >= lead_range[0]) & (df2[lt_col] <= lead_range[1])]

# 4.5 Plage de dates (sur la date choisie)
df2 = df2[(df2[date_field] >= start_date) & (df2[date_field] <= end_date)]

# 4.6 Jour du mois
df2 = df2[(df2[date_field].dt.day >= day_min) & (df2[date_field].dt.day <= day_max)]

# 4.7 Mois
if months:
    df2 = df2[df2[date_field].dt.month.isin(months)]

# 4.8 weight (utile pour event replacement)
df2["weight"] = 1.0

# 4.9 Events
info_msg = None
if event_mode == "Exclure":
    df2 = df2[df2["Event"] == "None"].copy()
elif event_mode == "Remplacer (moyenne autres années)":
    if events_available and event_to_replace and "(aucun" not in str(event_to_replace):
        df2, info_msg = replace_event_period_with_other_years(
            df2,
            event_to_replace,
            date_col="Arrival Date"
        )

# Infos
st.write(f"Lignes après filtres : **{len(df2):,}**")
if info_msg:
    st.info(info_msg)

# Pas assez de data = pas de courbe fiable
if len(df2) < 20:
    st.warning("Pas assez de données après filtres pour tracer une courbe.")
    st.dataframe(df2.head(200))
    st.stop()

# ============================================================
# 5) AGREGER (bins ADR) => agg
# ============================================================
agg = aggregate_curve(df2, bin_size=bin_size, y_mode=y_mode)

# ============================================================
# 6) AFFICHAGE : 3 tabs
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "Courbe sensibilité",
    "Top Nationalities",
    "Saisonnalité (heatmap)"
])

# ------------------------------------------------------------
# TAB 1 : Courbe + KPI + table agg debug optionnelle
# ------------------------------------------------------------
with tab1:
    st.subheader("Courbe de sensibilité prix")

    # -----------------------------
    # Zoom intelligent (anti-outliers)
    # -----------------------------
    # Problème :
    # - Si quelques ADR extrêmes existent (6000-10000€),
    #   matplotlib dézoome l'axe X et on ne voit plus la zone 200-700€.
    #
    # Solution :
    # - calculer xmin/xmax sur la distribution réelle des ADR filtrés (df2["ADR"])
    # - prendre les quantiles 1% et 99% (garde 99% des data)
    adr_raw = df2["ADR"].dropna().to_numpy(dtype=float)

    if len(adr_raw) >= 20:
        xmin = float(np.nanpercentile(adr_raw, 1))
        xmax = float(np.nanpercentile(adr_raw, 99))
    else:
        xmin = float(np.nanmin(agg["ADR_mean"]))
        xmax = float(np.nanmax(agg["ADR_mean"]))

    # On limite aussi les points affichés pour éviter la "ligne plate" à droite
    agg_plot = agg[(agg["ADR_mean"] >= xmin) & (agg["ADR_mean"] <= xmax)].copy()
    if len(agg_plot) < 5:
        agg_plot = agg.copy()

    # Plot
    fig = plt.figure(figsize=(9, 5))
    plt.plot(agg_plot["ADR_mean"], agg_plot["Y"], marker="o")
    plt.xlabel("ADR (euros)")
    plt.ylabel(agg_plot["Y_label"].iloc[0] if len(agg_plot) else "Y")
    plt.grid(True)
    plt.xlim(xmin, xmax)
    st.pyplot(fig)

    # -----------------------------
    # KPI : Prix référence / optimal / gain
    # -----------------------------
    st.subheader("Prix optimal et comparaison au prix de référence")

    # Ici ref_weight est fixe = "Nights" (pondéré par nuitées)
    res = compute_reference_and_best(
        agg,
        reference=reference_mode,
        ref_weight=REF_WEIGHT_FIXED
    )

    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix de référence", f"{res['ref_price']:.0f} euros")
        c2.metric("Meilleur prix (historique)", f"{res['best_price']:.0f} euros")
        c3.metric("Gain (bin)", f"{res['gain']:,.0f} euros", f"{res['gain_pct']:.1f}%")

        st.caption(
            "Note : si le prix de référence = prix optimal, c'est souvent normal "
            "(bin typique = bin revenu max) ou bien il reste peu de données après filtres."
        )
    else:
        st.info("Impossible de calculer la référence et l'optimum (pas assez de bins).")

    # -----------------------------
    # Table d'agrégation (debug) — masquée par défaut
    # -----------------------------
    if show_agg_table:
        st.markdown("### Table d’agrégation (debug)")
        st.caption(
            "Chaque ligne correspond à un bin d'ADR. "
            "Tu peux vérifier Volume/Nights/Revenue/ADR_mean pour comprendre la courbe."
        )
        st.dataframe(agg)

# ------------------------------------------------------------
# TAB 2 : Top Nationalities
# ------------------------------------------------------------
with tab2:
    st.subheader("Top Nationalities")
    nat = df2["Nationality"].astype(str).value_counts().head(15)

    fig = plt.figure(figsize=(8, 6))
    plt.barh(nat.index[::-1], nat.values[::-1])
    plt.xlabel("Nombre de réservations")
    plt.ylabel("Nationality")
    plt.grid(True, axis="x")
    st.pyplot(fig)

# ------------------------------------------------------------
# TAB 3 : Heatmap Saisonnalité
# ------------------------------------------------------------
with tab3:
    st.subheader("Saisonnalité : Mois x Jour de semaine (volume)")

    tmp = df2.copy()
    tmp["Weekday"] = tmp["Arrival Date"].dt.day_name()
    tmp["Month Number"] = tmp["Arrival Date"].dt.month

    # Ordre logique des weekdays
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp["Weekday"] = pd.Categorical(tmp["Weekday"], categories=weekday_order, ordered=True)

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
