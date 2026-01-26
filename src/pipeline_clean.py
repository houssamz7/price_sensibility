# src/pipeline_clean.py
import pandas as pd

# Les règles de nos hôtels pour Season (LOWn HIGH)
def get_season(month: int, day: int) -> str:
    """
    Détermine la saison LOW / HIGH selon la date d'arrivée.
    """
    if (
        month in [1, 2, 11, 12]
        or (month == 3 and day <= 15)
        or (month == 10 and day >= 30)
        or (month == 7 and day >= 19)
        or (month == 8 and day <= 30)
    ):
        return "LOW"
    return "HIGH"

# Event J.O 2024 hardcodé, les événements majeurs exceptionnels comptent un volume anoramle de réservations qui pourra fausser les données
def get_event(year: int, month: int, day: int) -> str:
    """
    JO 2024 : période étendue.
    """
    if year == 2024 and ((month == 7 and day >= 20) or (month == 8 and day <= 18)):
        return "Olympics_2024"
    return "None"

# Le volume de réservation est anormale pour une période étendue
def get_event_level(year: int, month: int, day: int) -> str:
    """
    Core = dates exactes de l'event
    Extended = 1 semaine avant / 1 semaine après
    """
    if year == 2024 and ((month == 7 and day >= 26) or (month == 8 and day <= 11)):
        return "Core"
    if year == 2024 and (
        (month == 7 and day >= 20 and day < 26)
        or (month == 8 and day <= 18 and day > 11)
    ):
        return "Extended"
    return "None"

# Pareil que build_clean_dataset(raw_path), mais à partir d'un DataFrame. Mais pour Streamlit (upload de fichier).
def build_clean_dataset_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Version 'Streamlit' : même logique mais à partir d'un DataFrame (upload).
    """
    # 1) Traitement des doublons :
    #    - on supprime les "mauvais doublons" (même confirmation mais autres colonnes différentes)
    cols_except_rate = df.columns.drop("Rate")
    check = df.groupby("Confirmation Number")[cols_except_rate].nunique()
    invalid = check[check.gt(1).any(axis=1)].index
    df = df[~df["Confirmation Number"].isin(invalid)]

    # 2) Doublons restants : on agrège (Rate => ADR = moyenne)
    df_clean = (
        df.groupby("Confirmation Number", as_index=False)
        .agg({**{c: "first" for c in cols_except_rate}, "Rate": "mean"})
        .rename(columns={"Rate": "ADR"})
    )

    # 3) Convertir dates en datetime
    for c in ["Reservation Date", "Arrival Date", "Departure Date"]:
        df_clean[c] = pd.to_datetime(df_clean[c], errors="coerce")

    # 4) Garder seulement les lignes cohérentes
    df_clean = df_clean[df_clean["Arrival Date"] >= df_clean["Reservation Date"]]
    df_clean = df_clean[df_clean["Arrival Date"] <= df_clean["Departure Date"]]

    # 5) Corrections métier
    df_clean["Number of Nights"] = df_clean["Number of Nights"].replace(0, 1)
    df_clean["Adults"] = df_clean["Adults"].replace(0, 1)

    # 6) Variables calendaires
    df_clean["Year"] = df_clean["Arrival Date"].dt.year
    df_clean["Month Number"] = df_clean["Arrival Date"].dt.month
    df_clean["Day"] = df_clean["Arrival Date"].dt.day
    df_clean["Weekday"] = df_clean["Arrival Date"].dt.day_name()
    df_clean["Month"] = df_clean["Arrival Date"].dt.month_name()

    # Saison selon tes règles
    df_clean["Season"] = df_clean.apply(lambda r: get_season(r["Month Number"], r["Day"]), axis=1)

    # 7) Filtre ADR minimum + revenue recalculé
    df_clean = df_clean[df_clean["ADR"] >= 40]
    df_clean["Room Revenue"] = df_clean["ADR"] * df_clean["Number of Nights"]

    # 8) Event JO hardcodé (toujours présent)
    df_clean["Event"] = df_clean.apply(lambda r: get_event(r["Year"], r["Month Number"], r["Day"]), axis=1)
    df_clean["Event Level"] = df_clean.apply(lambda r: get_event_level(r["Year"], r["Month Number"], r["Day"]), axis=1)

    return df_clean

def build_clean_dataset(raw_path: str) -> pd.DataFrame:
    """
    Version 'CLI' : lit un fichier brut depuis un chemin.
    """
    df = pd.read_csv(raw_path, sep="\t", encoding="utf-16")
    return build_clean_dataset_from_df(df)

def save_clean(df_clean: pd. DataFrame, out_path: str) -> None:
    df_clean.to_csv(out_path, sep="\t", index=False, encoding="utf-16")