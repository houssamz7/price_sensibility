# app/pages/1_Donnees_reelles.py
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

# Page : Données réelles
# - Utilise df_clean_final préparé depuis la page Home
# - Filtres principaux + gestion des events
# - Courbe de sensibilité + Top Nationalities + Saisonnalité (heatmap)


st.title("Données réelles — Dashboard")

# 0) Récupérer les données CLEAN (préparées depuis Home)
if "df_clean_final" not in st.session_state:
    st.warning("Aller d'abord sur la page Home pour uploader le fichier et générer le CLEAN.")
    st.stop()

df = st.session_state["df_clean_final"].copy()

# Sécurité : dates en datetime
for c in ["Arrival Date", "Reservation Date", "Departure Date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")


# 1) Sidebar - Filtres utilisateur
#   Modifications demandées :
#   - Suppression du filtre "Plage de dates"
#   - Conservation du filtre "Jour du mois"
#   - Conservation de "Top Nationalities" + Saisonnalité (heatmap)
#   - Suppression du tab "Fiabilité"
#   - Dans "Distributions", on garde uniquement Top Nationalities

st.sidebar.header("Filtres")

property_ = st.sidebar.selectbox("Property", ["All"] + sorted(df["Property"].astype(str).unique().tolist()))
mc = st.sidebar.selectbox("Market Code", ["All"] + sorted(df["Market Code"].astype(str).unique().tolist()))
rt = st.sidebar.selectbox("Room Type", ["All"] + sorted(df["Room Type"].astype(str).unique().tolist()))
sc = st.sidebar.selectbox("Source Code", ["All"] + sorted(df["Source Code"].astype(str).unique().tolist()))

season = st.sidebar.selectbox("Season", ["All", "LOW", "HIGH"])

nationalities = st.sidebar.multiselect(
    "Nationality",
    sorted(df["Nationality"].astype(str).unique().tolist())
)

# Lead time slider
lt_col = "Lead Time(Days)"
lead_range = None
if lt_col in df.columns:
    lt_min = int(np.nanmin(df[lt_col]))
    lt_max = int(np.nanmax(df[lt_col]))
    lead_range = st.sidebar.slider("Lead Time(Days)", lt_min, lt_max, (lt_min, lt_max))
else:
    st.sidebar.info("Colonne 'Lead Time(Days)' non trouvée dans le fichier.")

# Date utilisée (conservé, mais sans plage de dates)
date_field = st.sidebar.selectbox("Date utilisée", ["Arrival Date", "Reservation Date", "Departure Date"])

# Filtre "jour du mois" (conservé)
day_min, day_max = st.sidebar.slider(
    "Jour du mois (détails (ex: 1→13)",
    1, 31, (1, 31)
)

# Filtre "mois" (optionnel)
months = st.sidebar.multiselect(
    "Mois (optionnel)",
    options=list(range(1, 13)),
    format_func=lambda x: pd.Timestamp(year=2024, month=x, day=1).strftime("%B"),
    default=[]
)

# Options de courbe
y_mode = st.sidebar.selectbox("Axe Y", ["reservations", "nights", "revenue"])
bin_size = st.sidebar.slider("Bins ADR (€)", 5, 50, 15, 5)
reference_mode = st.sidebar.selectbox("Prix de référence", ["median", "mean"])


# 2) Gestion Event
st.sidebar.header("Gestion Event")

event_mode = st.sidebar.selectbox(
    "Traitement des events",
    [
        "Inclure (normal)",
        "Exclure (retirer les lignes event)",
        "Remplacer par moyenne autres années"
    ]
)

events_available = sorted([e for e in df["Event"].astype(str).unique().tolist() if e != "None"])
event_to_replace = None
if event_mode == "Remplacer par moyenne autres années":
    event_to_replace = st.sidebar.selectbox(
        "Quel event remplacer ?",
        options=events_available if events_available else ["(aucun event trouvé)"]
    )


# 3) Application des filtres

filters = {
    "Property": None if property_ == "All" else property_,
    "Market Code": None if mc == "All" else mc,
    "Source Code": None if sc == "All" else sc,
    "Season": None if season == "All" else season,
}
df2 = apply_filters(df, filters)

# Nationality
if len(nationalities) > 0:
    df2 = df2[df2["Nationality"].astype(str).isin(nationalities)]

# Lead time
if lead_range is not None:
    df2 = df2[(df2[lt_col] >= lead_range[0]) & (df2[lt_col] <= lead_range[1])]

# Jour du mois
df2 = df2[(df2[date_field].dt.day >= day_min) & (df2[date_field].dt.day <= day_max)]

# Mois (si sélectionné)
if len(months) > 0:
    df2 = df2[df2[date_field].dt.month.isin(months)]

# Poids par défaut
df2["weight"] = 1.0


# 4) Traitement des events selon le choix
info_msg = None

if event_mode == "Exclure (retirer les lignes event)":
    df2 = df2[df2["Event"] == "None"].copy()

elif event_mode == "Remplacer par moyenne autres années":
    if events_available and event_to_replace and "(aucun" not in event_to_replace:
        df2, info_msg = replace_event_period_with_other_years(
            df2,
            event_to_replace,
            date_col="Arrival Date"
        )
    else:
        st.warning("Aucun event à remplacer.")
        df2 = df2.copy()

st.write(f"Lignes après filtres : {len(df2):,}")
if info_msg:
    st.info(info_msg)

# 5) Condition minimum de données
if len(df2) < 20:
    st.warning("Pas assez de données après filtres pour tracer une courbe.")
    st.dataframe(df2.head(200))
    st.stop()

# 6) Courbe principale (sensibilité prix)
agg = aggregate_curve(df2, bin_size=bin_size, y_mode=y_mode)

# 7) Tabs (modifiés)
#   - Suppression du tab Fiabilité
#   - Distributions => uniquement Top Nationalities
#   - Conservation de Saisonnalité (heatmap)
tab1, tab3, tab4 = st.tabs([
    "Courbe sensibilité",
    "Top Nationalities",
    "Saisonnalité (heatmap)"
])


# TAB 1 : Courbe sensibilité
with tab1:
    st.subheader("Courbe de sensibilité prix")

    fig = plt.figure(figsize=(9, 5))
    plt.plot(agg["ADR_mean"], agg["Y"], marker="o")
    plt.xlabel("ADR (euros)")
    plt.ylabel(agg["Y_label"].iloc[0] if len(agg) else "Y")
    plt.grid(True)
    st.pyplot(fig)

    st.subheader("Prix optimal et comparaison au prix de référence")
    res = compute_reference_and_best(agg, reference=reference_mode)

    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix de référence", f"{res['ref_price']:.0f} euros")
        c2.metric("Meilleur prix (historique)", f"{res['best_price']:.0f} euros")
        c3.metric("Gain (bin)", f"{res['gain']:,.0f} euros", f"{res['gain_pct']:.1f}%")
    else:
        st.info("Pas assez de données pour calculer un optimum.")

    st.subheader("Table agrégée (bins ADR)")
    st.dataframe(agg)


# TAB 3 : Top Nationalities uniquement
with tab3:
    st.subheader("Top Nationalities")

    nat = (
        df2["Nationality"].astype(str)
        .value_counts()
        .head(15)
    )

    fig = plt.figure(figsize=(8, 6))
    plt.barh(nat.index[::-1], nat.values[::-1])
    plt.xlabel("Nombre de réservations")
    plt.ylabel("Nationality")
    plt.grid(True, axis="x")
    st.pyplot(fig)


# TAB 4 : Heatmap mois x weekday (saisonnalité)
with tab4:
    st.subheader("Saisonnalité : mois et jour de semaine")

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
