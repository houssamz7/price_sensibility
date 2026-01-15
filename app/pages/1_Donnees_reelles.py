# app/pages/1_Donnees_reelles.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.analytics import (
    apply_filters,
    aggregate_curve,
    compute_reference_and_best,
    replace_event_period_with_other_years
)

st.title("Données réelles — Dashboard")


# Récupérer les données CLEAN depuis la page principale
if "df_clean_final" not in st.session_state:
    st.warning("Va d'abord sur la page **Home** pour uploader le fichier et générer le CLEAN.")
    st.stop()

df = st.session_state["df_clean_final"].copy()

# Par sécurité : dates en datetime
df["Arrival Date"] = pd.to_datetime(df["Arrival Date"])
df["Reservation Date"] = pd.to_datetime(df["Reservation Date"])
df["Departure Date"] = pd.to_datetime(df["Departure Date"])


# Sidebar - Filtres
st.sidebar.header("Filtres")

property_ = st.sidebar.selectbox("Property", ["All"] + sorted(df["Property"].astype(str).unique().tolist()))
mc = st.sidebar.selectbox("Market Code", ["All"] + sorted(df["Market Code"].astype(str).unique().tolist()))
rt = st.sidebar.selectbox("Room Type", ["All"] + sorted(df["Room Type"].astype(str).unique().tolist()))
sc = st.sidebar.selectbox("Source Code", ["All"] + sorted(df["Source Code"].astype(str).unique().tolist()))

# une seule courbe : Season est un filtre
season = st.sidebar.selectbox("Season", ["All", "LOW", "HIGH"])

# Nationality (multi)
nationalities = st.sidebar.multiselect(
    "Nationality",
    sorted(df["Nationality"].astype(str).unique().tolist())
)

# Lead Time (Days)
lt_col = "Lead Time(Days)"
if lt_col in df.columns:
    lt_min = int(np.nanmin(df[lt_col]))
    lt_max = int(np.nanmax(df[lt_col]))
    lead_range = st.sidebar.slider("Lead Time(Days)", lt_min, lt_max, (lt_min, lt_max))
else:
    lead_range = None
    st.sidebar.info("Colonne 'Lead Time(Days)' non trouvée dans le fichier.")

# Filtre date (plage réelle)
date_field = st.sidebar.selectbox("Date utilisée", ["Arrival Date", "Reservation Date", "Departure Date"])
dmin = pd.to_datetime(df[date_field]).min().date()
dmax = pd.to_datetime(df[date_field]).max().date()
# date_range = st.sidebar.date_input("Plage de dates", (dmin, dmax))

# Filtre "jours du mois" (1->13, 25->30, etc.)
day_min, day_max = st.sidebar.slider("Jour du mois (ex: 1→13)", 1, 31, (1, 31))

# Filtre "mois" (pour faire 25→30 janvier)
months = st.sidebar.multiselect(
    "Mois (optionnel)",
    options=list(range(1, 13)),
    format_func=lambda x: pd.Timestamp(year=2024, month=x, day=1).strftime("%B"),
    default=[]
)

# Courbe options
y_mode = st.sidebar.selectbox("Axe Y", ["reservations", "nights", "revenue"])
bin_size = st.sidebar.slider("Bins ADR (€)", 5, 50, 15, 5)
reference_mode = st.sidebar.selectbox("Prix de référence", ["median", "mean"])


# Gestion Event (nouvelle logique)
st.sidebar.header("Gestion Event")

event_mode = st.sidebar.selectbox(
    "Traitement des events",
    [
        "Inclure (normal)",
        "Exclure (retirer les lignes event)",
        "Remplacer par moyenne autres années"
    ]
)

# Si remplacement: choisir quel event remplacer
events_available = sorted([e for e in df["Event"].astype(str).unique().tolist() if e != "None"])
event_to_replace = None
if event_mode == "Remplacer par moyenne autres années":
    event_to_replace = st.sidebar.selectbox(
        "Quel event remplacer ?",
        options=events_available if events_available else ["(aucun event trouvé)"]
    )


# Application des filtres
filters = {
    "Property": None if property_ == "All" else property_,
    "Market Code": None if mc == "All" else mc,
    "Room Type": None if mc == "All" else rt,
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

# Date range (sur date_field)
# start_date, end_date = date_range
# start_date = pd.to_datetime(start_date)
# end_date = pd.to_datetime(end_date)
# df2 = df2[(df2[date_field] >= start_date) & (df2[date_field] <= end_date)]

# Jour du mois
df2 = df2[(df2[date_field].dt.day >= day_min) & (df2[date_field].dt.day <= day_max)]

# Mois (si sélectionné)
if len(months) > 0:
    df2 = df2[df2[date_field].dt.month.isin(months)]

# On crée weight=1 (utile si plus tard tu veux pondérer)
df2["weight"] = 1.0


# Traitement events selon choix
info_msg = None
if event_mode == "Exclure (retirer les lignes event)":
    df2 = df2[df2["Event"] == "None"].copy()
elif event_mode == "Remplacer par moyenne autres années":
    if events_available and event_to_replace and "(aucun" not in event_to_replace:
        df2, info_msg = replace_event_period_with_other_years(df2, event_to_replace, date_col="Arrival Date")
    else:
        st.warning("Aucun event à remplacer.")
        df2 = df2.copy()

st.write(f" Lignes après filtres : **{len(df2):,}**")
if info_msg:
    st.info(info_msg)


# Affichage résultats
if len(df2) < 20:
    st.warning("Pas assez de données après filtres pour tracer une courbe.")
    st.dataframe(df2.head(200))
    st.stop()

agg = aggregate_curve(df2, bin_size=bin_size, y_mode=y_mode)

# Courbe (une seule)
st.subheader("Courbe (une seule)")
fig = plt.figure(figsize=(9, 5))
plt.plot(agg["ADR_mean"], agg["Y"], marker="o")
plt.xlabel("ADR (€)")
plt.ylabel(agg["Y_label"].iloc[0] if len(agg) else "Y")
plt.grid(True)
st.pyplot(fig)

# Table agrégée
st.subheader("Table agrégée")
st.dataframe(agg)

# Prix optimal + gain
st.subheader("Prix optimal & gain vs référence")
res = compute_reference_and_best(agg, reference=reference_mode)
if res:
    c1, c2, c3 = st.columns(3)
    c1.metric("Prix de référence", f"{res['ref_price']:.0f} €")
    c2.metric("Meilleur prix (hist.)", f"{res['best_price']:.0f} €")
    c3.metric("Gain (bin)", f"{res['gain']:,.0f} €", f"{res['gain_pct']:.1f}%")
else:
    st.info("Pas assez de données pour calculer un optimum.")
