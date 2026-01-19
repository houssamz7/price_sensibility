# app/pages/1_Donnees_reelles.py
# DONNEES REELLES (V1) - Page Streamlit
# 
# Objectifs :
# 1) Filtrer les données (Property, Segment, dates, etc.)
# 2) Tracer la courbe de sensibilité prix (ADR vs Y)
# 3) Calculer un prix de référence (pondéré par nuitées) + meilleur prix historique
# 4) Afficher 2 visuels simples (Top Nationalities, Saisonnalité heatmap)
# 5) Avoir des outils debug (table agg et top bins) masqués par défaut
# 

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


# 0) Segments : mapping Segment -> Market Codes
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


# 1) Charger le dataset clean depuis session_state
if "df_clean_final" not in st.session_state:
    st.warning("Va d'abord sur la page Home pour uploader le fichier et générer le CLEAN.")
    st.stop()

df = st.session_state["df_clean_final"].copy()

# Sécurité : dates en datetime
for c in ["Arrival Date", "Reservation Date", "Departure Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")


# 2) Sidebar : filtres
st.sidebar.header("Filtres")

# Property (1 choix)
property_ = st.sidebar.selectbox(
    "Property",
    ["All"] + sorted(df["Property"].astype(str).unique().tolist())
)

# Segment (1 choix) : logique RM => un segment à la fois
segment_options = ["All"] + [s for s in SEGMENT_ORDER if s in SEGMENT_TO_MARKET_CODES]
segment = st.sidebar.selectbox("Segment", segment_options)

# Market Code (multi)
# Si Segment = All -> on propose tous les codes
# Sinon -> seulement les codes du segment choisi
if segment == "All":
    mc_pool = sorted(df["Market Code"].astype(str).unique().tolist())
else:
    mc_pool = SEGMENT_TO_MARKET_CODES.get(segment, [])

market_codes = st.sidebar.multiselect(
    "Market Code (multi)",
    options=mc_pool,
    default=[],
    help="Tu peux choisir 1 ou plusieurs Market Codes. Si tu ne choisis rien => on prend tout."
)

# Room Type (multi)
room_type_pool = sorted(df["Room Type"].astype(str).unique().tolist())
room_types = st.sidebar.selectbox(
    "Room Type",
    ["All"] + sorted(df["Room Type"].astype(str).unique().tolist())
)

# Source Code (multi)
sc_pool = sorted(df["Source Code"].astype(str).unique().tolist())
source_codes = st.sidebar.multiselect(
    "Source Code (multi)",
    options=sc_pool,
    default=[],
    help="Si vide => toutes les sources"
)

# Season (1 choix)
season = st.sidebar.selectbox("Season", ["All", "LOW", "HIGH"])

# Nationality (multi)
nat_pool = sorted(df["Nationality"].astype(str).unique().tolist())
nationalities = st.sidebar.multiselect(
    "Nationality (multi)",
    options=nat_pool,
    default=[]
)

# Lead Time (Days)
lt_col = "Lead Time(Days)"
lead_range = None
if lt_col in df.columns:
    lt_min = int(np.nanmin(df[lt_col]))
    lt_max = int(np.nanmax(df[lt_col]))
    lead_range = st.sidebar.slider("Lead Time (Days)", lt_min, lt_max, (lt_min, lt_max))
else:
    st.sidebar.info(f"Colonne '{lt_col}' non trouvée.")

# Date utilisée
date_field = st.sidebar.selectbox("Date utilisée", ["Arrival Date", "Reservation Date", "Departure Date"])


# Quick Date Range (avec All par défaut) + Date input manuel
# IMPORTANT : On met un label différent de ton screenshot pour être sûr.
dmin = pd.to_datetime(df[date_field]).min().date()
dmax = pd.to_datetime(df[date_field]).max().date()

quick_range = st.sidebar.selectbox(
    "Quick Date Range",
    options=["All", "Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "Past 2 Years"],
    index=0
)

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
else:
    start_default = max(dmin, (pd.to_datetime(dmax) - pd.Timedelta(days=730)).date())

date_range = st.sidebar.date_input(
    "Plage de dates",
    value=(start_default, end_default),
    min_value=dmin,
    max_value=dmax
)

# Sécurité : tuple ou date seule
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, date_range

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# our du mois
day_min, day_max = st.sidebar.slider("Jour du mois", 1, 31, (1, 31))

# Mois (multi)
months = st.sidebar.multiselect(
    "Mois (optionnel, multi)",
    options=list(range(1, 13)),
    format_func=lambda x: pd.Timestamp(year=2024, month=x, day=1).strftime("%B"),
    default=[]
)

# Courbe
y_mode = st.sidebar.selectbox("Axe Y", ["reservations", "nights", "revenue"])
bin_size = st.sidebar.slider("Bins ADR (€)", 5, 50, 15, 5)
reference_mode = st.sidebar.selectbox("Référence", ["median", "mean"])

# Debug
show_agg_table = st.sidebar.checkbox("Afficher table agg (debug)", value=False)
show_debug_details = st.sidebar.checkbox("Afficher debug prix (top bins)", value=False)


# Gestion Event
st.sidebar.header("Gestion Event")
event_mode = st.sidebar.selectbox("Traitement des events", ["Inclure", "Exclure", "Remplacer (moyenne autres années)"])

events_available = sorted([e for e in df["Event"].astype(str).unique().tolist() if e != "None"])
event_to_replace = None
if event_mode == "Remplacer (moyenne autres années)":
    event_to_replace = st.sidebar.selectbox("Quel event remplacer ?", options=events_available if events_available else ["(aucun event)"])


# 3) Application des filtres
filters = {
    "Property": None if property_ == "All" else property_,
    "Season": None if season == "All" else season,
}
df2 = apply_filters(df, filters)

# Segment / Market Codes :
# - Si user sélectionne des Market Codes => on filtre sur ces MC
# - Sinon :
#   * si Segment=All => rien à filtrer
#   * si Segment=X => on prend tous les codes du segment X
if market_codes:
    df2 = df2[df2["Market Code"].astype(str).isin([str(x) for x in market_codes])]
else:
    if segment != "All":
        allowed = SEGMENT_TO_MARKET_CODES.get(segment, [])
        df2 = df2[df2["Market Code"].astype(str).isin(allowed)]

# Room types
if room_types:
    df2 = df2[df2["Room Type"].astype(str).isin([str(x) for x in room_types])]

# Source codes
if source_codes:
    df2 = df2[df2["Source Code"].astype(str).isin([str(x) for x in source_codes])]

# Nationalities
if nationalities:
    df2 = df2[df2["Nationality"].astype(str).isin([str(x) for x in nationalities])]

# Lead time
if lead_range is not None and lt_col in df2.columns:
    df2 = df2[(df2[lt_col] >= lead_range[0]) & (df2[lt_col] <= lead_range[1])]

# Date range
df2 = df2[(df2[date_field] >= start_date) & (df2[date_field] <= end_date)]

# Jour du mois
df2 = df2[(df2[date_field].dt.day >= day_min) & (df2[date_field].dt.day <= day_max)]

# Mois
if months:
    df2 = df2[df2[date_field].dt.month.isin(months)]

# Event
df2["weight"] = 1.0
info_msg = None

if event_mode == "Exclure":
    df2 = df2[df2["Event"] == "None"].copy()
elif event_mode == "Remplacer (moyenne autres années)":
    if events_available and event_to_replace and "(aucun" not in str(event_to_replace):
        df2, info_msg = replace_event_period_with_other_years(df2, event_to_replace, date_col="Arrival Date")

st.write(f"Lignes après filtres : **{len(df2):,}**")
if info_msg:
    st.info(info_msg)

if len(df2) < 20:
    st.warning("Pas assez de données après filtres pour tracer une courbe.")
    st.dataframe(df2.head(200))
    st.stop()


# 4) Agrégation (bins ADR)
agg = aggregate_curve(df2, bin_size=bin_size, y_mode=y_mode)


# 5) Affichage : 3 tabs
tab1, tab2, tab3 = st.tabs(["Courbe sensibilité", "Top Nationalities", "Saisonnalité (heatmap)"])

with tab1:
    st.subheader("Courbe de sensibilité prix")

    # Zoom intelligent basé sur df2["ADR"] (quantiles)
    adr_raw = df2["ADR"].dropna().to_numpy(dtype=float)
    xmin = float(np.nanpercentile(adr_raw, 1))
    xmax = float(np.nanpercentile(adr_raw, 99))

    agg_plot = agg[(agg["ADR_mean"] >= xmin) & (agg["ADR_mean"] <= xmax)].copy()
    if len(agg_plot) < 5:
        agg_plot = agg.copy()

    fig = plt.figure(figsize=(9, 5))
    plt.plot(agg_plot["ADR_mean"], agg_plot["Y"], marker="o")
    plt.xlabel("ADR (euros)")
    plt.ylabel(agg_plot["Y_label"].iloc[0] if len(agg_plot) else "Y")
    plt.grid(True)
    plt.xlim(xmin, xmax)
    st.pyplot(fig)

    st.subheader("Prix optimal et comparaison au prix de référence")

    # IMPORTANT : Référence pondérée par nuitées (Nights) FIXE
    res = compute_reference_and_best(agg, reference=reference_mode, ref_weight="Nights")

    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix de référence", f"{res['ref_price']:.0f} euros")
        c2.metric("Meilleur prix (historique)", f"{res['best_price']:.0f} euros")
        c3.metric("Gain (bin)", f"{res['gain']:,.0f} euros", f"{res['gain_pct']:.1f}%")

        st.caption(
            "Si référence = meilleur prix : c'est parfois normal (bin typique = bin revenu max). "
            "Si ça arrive tout le temps, active le debug pour vérifier les bins."
        )

        if show_debug_details:
            st.markdown("### Debug prix (top bins revenue)")
            st.write("Référence pondérée par : **Nights**")
            st.write("Top 5 bins par Revenue :")
            st.dataframe(agg.sort_values("Revenue", ascending=False).head(5))
    else:
        st.info("Impossible de calculer la référence et l'optimum.")

    if show_agg_table:
        st.markdown("### Table agg (debug)")
        st.dataframe(agg)

with tab2:
    st.subheader("Top Nationalities")
    nat = df2["Nationality"].astype(str).value_counts().head(15)

    fig = plt.figure(figsize=(8, 6))
    plt.barh(nat.index[::-1], nat.values[::-1])
    plt.xlabel("Nombre de réservations")
    plt.ylabel("Nationality")
    plt.grid(True, axis="x")
    st.pyplot(fig)

with tab3:
    st.subheader("Saisonnalité : Mois x Jour de semaine (volume)")

    tmp = df2.copy()
    tmp["Weekday"] = tmp["Arrival Date"].dt.day_name()
    tmp["Month Number"] = tmp["Arrival Date"].dt.month

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
