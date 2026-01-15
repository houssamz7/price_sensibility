# app/streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline_clean import build_clean_dataset_from_df


st.set_page_config(page_title="Sensibilité Prix — Home", layout="wide")
st.title("Sensibilité au prix — Outil RM")

st.markdown("""
### Objectif
- Uploader un export brut R&A (CSV/TSV, séparation par tabulation, encodage utf-16)
- Le transformer en fichier CLEAN
- Ensuite aller dans les pages :
  - 1_Donnees_reelles
  - 2_IA (En cours de développement)
  - 3_Comparaison (En cours de développement)

Aucune courbe n’est affichée ici : les résultats sont dans les pages.
""")



# Lecture d’un fichier uploadé
def read_ra_file(uploaded_file) -> pd.DataFrame:
    """
    Lit un fichier export R&A uploadé via Streamlit (TSV/CSV encodé en utf-16).

    Point important : uploaded_file n’est pas un chemin de fichier sur disque.
    Streamlit fournit un objet "UploadedFile" contenant le contenu du fichier en mémoire.

    - uploaded_file.getvalue() renvoie le contenu complet du fichier sous forme d’octets (bytes).
    - pandas.read_csv attend soit :
        * un chemin de fichier (str)
        * soit un "file-like object" (un objet qui se comporte comme un fichier)
    - BytesIO permet de transformer des bytes en un objet "fichier en mémoire".
      On obtient ainsi un flux lisible par pandas comme si c’était un vrai fichier.
    """
    data = uploaded_file.getvalue()              # Contenu du fichier en bytes (octets)
    buffer = BytesIO(data)                       # Fichier virtuel en mémoire (flux binaire)
    return pd.read_csv(buffer, sep="\t", encoding="utf-16")  # Lecture TSV utf-16 via pandas


# Validation des événements saisis par l'utilisateur
def validate_events(events_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Valide le tableau d’événements saisi via data_editor.

    Colonnes attendues :
      - event_name
      - core_start
      - core_end

    Règles de validation :
      - Les colonnes doivent être présentes.
      - Les dates doivent être convertibles.
      - core_end doit être supérieur ou égal à core_start.
      - Format conseillé : YYYY-MM-DD
    """
    # Cas 1 : aucun événement saisi (table vide)
    # On considère que c'est valide, car il n'y a rien à appliquer.
    if events_df is None or len(events_df) == 0:
        return True, ""

    # Cas 2 : vérification de la présence des colonnes obligatoires
    # required est un set (ensemble) qui contient les noms requis.
    required = {"event_name", "core_start", "core_end"}

    # issubset vérifie que required est inclus dans la liste des colonnes existantes.
    # Exemple : {"a","b"} inclus dans {"a","b","c"} => True
    if not required.issubset(set(events_df.columns)):
        return False, "Colonnes attendues : event_name, core_start, core_end."

    # Copie de travail : évite de modifier events_df (bonne pratique)
    tmp = events_df.copy()

    # Conversion des dates.
    # errors="coerce" signifie :
    # - si une valeur est invalide (vide, texte incorrect), elle devient NaT (Not a Time).
    tmp["core_start"] = pd.to_datetime(tmp["core_start"], errors="coerce")
    tmp["core_end"] = pd.to_datetime(tmp["core_end"], errors="coerce")

    # Vérification de présence de NaT après conversion
    # isna().any() renvoie True si au moins une ligne contient une date manquante/invalide.
    if tmp["core_start"].isna().any() or tmp["core_end"].isna().any():
        return False, "Dates invalides ou vides. Format conseillé : YYYY-MM-DD."

    # Vérification de cohérence des dates :
    # core_end doit être >= core_start.
    # On sélectionne les lignes invalides (core_end < core_start).
    bad = tmp[tmp["core_end"] < tmp["core_start"]]

    # Si au moins une ligne est invalide, on retourne une erreur.
    if len(bad) > 0:
        # On prend la première ligne invalide pour afficher un message simple.
        r = bad.iloc[0]
        return False, (
            "Erreur : core_end (" + str(r["core_end"].date()) +
            ") avant core_start (" + str(r["core_start"].date()) + ")."
        )

    # Si toutes les vérifications passent, la table est valide.
    return True, ""



# Application des événements saisis par l'utilisateur sur le dataset CLEAN
def apply_user_events(df_clean: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des événements saisis par l'utilisateur au dataset CLEAN.

    Interprétation :
      - La période Core est définie par core_start -> core_end.
      - La période Extended est automatiquement calculée :
          7 jours avant core_start et 7 jours après core_end.

    Règle importante :
      - Les événements saisis ne doivent pas écraser des événements déjà présents.
      - L’écriture se fait uniquement sur les lignes où Event == "None".
    """
    # Copie de travail : évite de modifier df_clean en place
    out = df_clean.copy()

    # Cas 1 : aucun événement saisi => dataset inchangé
    if events_df is None or len(events_df) == 0:
        return out

    # Copie des événements saisis pour travailler dessus sans modifier events_df
    tmp = events_df.copy()

    # Conversion en datetime (sans errors="coerce" ici car validate_events est censée garantir la validité)
    tmp["core_start"] = pd.to_datetime(tmp["core_start"])
    tmp["core_end"] = pd.to_datetime(tmp["core_end"])

    # Conversion des dates d'arrivée du dataset (supporte comparaisons >= <=)
    arrival = pd.to_datetime(out["Arrival Date"])

    # Parcours des lignes d'événements saisis
    # iterrows renvoie (index, ligne) ; ici l'index n'est pas utilisé, d'où "_"
    for _, ev in tmp.iterrows():

        # Nettoyage du nom d'événement (strip enlève les espaces au début/fin)
        name = str(ev["event_name"]).strip()

        # Si le nom est vide, on ignore la ligne
        if name == "":
            continue

        # Définition de la période Core
        core_start = ev["core_start"]
        core_end = ev["core_end"]

        # Définition automatique de la période Extended autour du Core (± 7 jours)
        ext_start = core_start - pd.Timedelta(days=7)
        ext_end = core_end + pd.Timedelta(days=7)

        # Masques booléens :
        # - in_core : True pour les lignes dont Arrival Date est dans Core
        in_core = (arrival >= core_start) & (arrival <= core_end)

        # - in_ext : True pour Extended mais en excluant Core (~in_core)
        in_ext = (arrival >= ext_start) & (arrival <= ext_end) & (~in_core)

        # Masque d'autorisation d'écriture :
        # True uniquement si Event est actuellement "None" (aucun événement existant)
        can_write = (out["Event"] == "None")

        # Application des labels sur la partie Core
        out.loc[can_write & in_core, "Event"] = name
        out.loc[can_write & in_core, "Event Level"] = "Core"

        # Application des labels sur la partie Extended
        out.loc[can_write & in_ext, "Event"] = name
        out.loc[can_write & in_ext, "Event Level"] = "Extended"

    # Retour du dataset enrichi (Event / Event Level mis à jour)
    return out



# Interface : upload du fichier brut + nettoyage
st.sidebar.header("1) Upload fichier brut")
uploaded = st.sidebar.file_uploader("Export R&A (TSV utf-16)", type=["csv", "tsv"])

# Aucun fichier : on arrête l'exécution de l'application ici
if uploaded is None:
    st.info("Upload d’un fichier brut requis pour démarrer.")
    st.stop()

# Lecture du fichier brut uploadé
df_raw = read_ra_file(uploaded)

# Bouton : exécuter le nettoyage RAW -> CLEAN et stocker le résultat en mémoire Streamlit
# Streamlit relance le script à chaque interaction ; session_state permet de conserver un résultat.
if st.sidebar.button("Nettoyer (RAW -> CLEAN)"):
    st.session_state["df_clean"] = build_clean_dataset_from_df(df_raw)
    st.session_state["file_loaded"] = True

# Si le nettoyage n’a pas été lancé, on ne peut pas continuer
if "df_clean" not in st.session_state:
    st.warning("Lancer le nettoyage via le bouton 'Nettoyer (RAW -> CLEAN)' dans la sidebar.")
    st.stop()

# Dataset CLEAN issu du nettoyage
df_clean = st.session_state["df_clean"]



# Interface : saisie d’événements futurs (optionnel)
st.sidebar.header("2) Events futurs (optionnel)")
st.sidebar.caption("Saisie Core start/end uniquement. Extended calcule automatiquement ±7 jours.")

# Table par défaut (vide) pour la saisie utilisateur
default_events_user = pd.DataFrame([], columns=["event_name", "core_start", "core_end"])

# data_editor affiche un tableau modifiable ; num_rows="dynamic" permet d’ajouter/supprimer des lignes
events_user = st.sidebar.data_editor(
    st.session_state.get("events_user", default_events_user),
    num_rows="dynamic",
    use_container_width=True
)
st.session_state["events_user"] = events_user

# Validation des saisies utilisateur (cohérence dates)
ok, msg = validate_events(events_user)
if not ok:
    st.sidebar.error(msg)
    st.stop()

# Application des événements futurs sur le dataset CLEAN
df_clean2 = apply_user_events(df_clean, events_user)

# Stockage du dataset final dans session_state pour l’utilisation dans les autres pages
st.session_state["df_clean_final"] = df_clean2

# Message simple pour l’utilisateur : données prêtes
st.success("Données CLEAN prêtes. Utiliser la page 1_Donnees_reelles.")

# -------------------------------------------------------------------
# Téléchargement du dataset CLEAN final (optionnel)
# -------------------------------------------------------------------
st.download_button(
    "Télécharger le CLEAN",
    data=df_clean2.to_csv(sep="\t", index=False, encoding="utf-16").encode("utf-16"),
    file_name="Projet_Sensibilite_Prix_clean.csv",
    mime="text/tab-separated-values"
)