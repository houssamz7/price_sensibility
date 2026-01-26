# src/analytics.py
import numpy as np
import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Filters exemple:
      {"Property": "PARHDC", "Market Code": "3CNR", "Season": "LOW"}
    """
    out = df.copy()
    # On parcourt chaque filtre (colonne -> valeur choisie)
    for col, val in filters.items():
        # Pas de filtre demandé
        if val is None or val == "All":
            continue
        # Cas multi-sélection (ex: plusieurs nationalités)
        if isinstance(val, list):
            out = out[out[col].isin(val)]
        # Cas valeur unique
        else:
            out = out[out[col] == val]
    return out


def aggregate_curve(df: pd.DataFrame, bin_size: int, y_mode: str) -> pd.DataFrame:
    """
    Construit une courbe "Prix (ADR) -> Y" en regroupant le prix par tranches (bins).

    y_mode :
      - "reservations" : Y = nombre de réservations (pondérées)
      - "nights"       : Y = nombre de nuitées (pondérées)
      - "revenue"      : Y = revenue (pondéré)

    Correction RM importante :
    - L'ADR d'un bin ne doit PAS être une moyenne simple des ADR,
      mais un ADR "global" du bin :
        ADR_mean = (somme Revenue pondéré) / (somme Nights pondéré)

    Si df n'a pas de colonne "weight", on suppose weight = 1 partout.
    """
    out = df.copy()
    # Si pas de poids, on en crée un par défaut (toutes les lignes comptent 1)
    if "weight" not in out.columns:
        out["weight"] = 1.0

    # 1) Créer les bornes des bins de prix (ex: 50, 65, 80, 95, ...)
    # np.arange(start, stop, step) crée une suite de nombres
    bins = np.arange(out["ADR"].min(), out["ADR"].max() + bin_size, bin_size)
    # 2) Transformer chaque ADR en "catégorie de bin"
    # Exemple: ADR=72 est inclus dans bin [65, 80]
    out["ADR_bin"] = pd.cut(out["ADR"], bins=bins, include_lowest=True)

    # 3) Préparer des mesures pondérées
    # - w_res : 1 réservation = 1 ligne => on utilise le poids directement
    # - w_nights : nuitées pondérées = nights * weight
    # - w_rev : revenue pondéré = revenue * weight
    out["w_res"] = out["weight"]
    out["w_nights"] = out["Number of Nights"] * out["weight"]
    out["w_rev"] = out["Room Revenue"] * out["weight"]

    # 4) Agréger par bin de prix (une seule courbe)
    agg = (
        out.groupby(["ADR_bin"], as_index=False)
        .agg(
            Reservations=("w_res", "sum"),
            Nights=("w_nights", "sum"),
            Revenue=("w_rev", "sum"),
        )
    )

    # 5) ADR RM "global" du bin = Revenue / Nights (pondérés)
    agg["ADR_mean"] = agg["Revenue"] / agg["Nights"]

    # 6) Choisir la variable affichée sur l'axe Y
    # On crée une colonne unique "Y" pour simplifier le code du graphique
    if y_mode == "reservations":
        agg["Y"] = agg["Reservations"]
        agg["Y_label"] = "Réservations (pondérées)"
    elif y_mode == "nights":
        agg["Y"] = agg["Nights"]
        agg["Y_label"] = "Nuitées (pondérées)"
    else:
        agg["Y"] = agg["Revenue"]
        agg["Y_label"] = "Revenue (pondéré)"

    # 7) Nettoyage : enlever les divisions par 0 et les valeurs impossibles
    # - si Nights=0 => ADR_mean = inf (infini). On remplace par NaN.
    # - puis on supprime les lignes où ADR_mean ou Y est manquant
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["ADR_mean", "Y"])
    # 8) Trier par ADR_mean pour avoir une courbe "croissante" en X
    # reset_index(drop=True) remet l'index à 0..n-1 proprement (sans garder l'ancien index)
    agg = agg.sort_values("ADR_mean").reset_index(drop=True)
    return agg

import numpy as np
import pandas as pd


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """
    Calcule la médiane pondérée.

    Intuition :
    - Médiane classique : on trie x, puis on prend la valeur au "milieu" (50% à gauche / 50% à droite).
    - Médiane pondérée : on trie x, MAIS chaque valeur a un "poids" w.
      Exemple : si un bin a 200 réservations et un autre a 2 réservations,
      le bin à 200 "compte beaucoup plus" dans la médiane.

    Résultat :
    - On renvoie la première valeur x telle que la somme cumulée des poids atteigne 50% du poids total.
    """

    # 1) Convertir en tableaux numpy float (pour être sûr des types)
    x = np.asarray(x, dtype=float)  # valeurs (ex: ADR_mean)
    w = np.asarray(w, dtype=float)  # poids (ex: Nights ou Reservations)

    # 2) Filtrer les valeurs invalides
    # np.isfinite(...) = True si la valeur est un nombre valide (pas NaN, pas +inf, pas -inf)
    #
    # mask est un tableau de True/False de la même longueur que x et w.
    # On garde uniquement les lignes où :
    # - x est valide (finite)
    # - w est valide (finite)
    # - w > 0 (poids positif)
    #
    # Pourquoi w > 0 ?
    # - un poids 0 veut dire "ne compte pas du tout" (inutile pour la médiane)
    # - un poids négatif n'a pas de sens pour une pondération de volume/nuitées
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)

    # On applique le masque : on retire les éléments invalides
    x, w = x[mask], w[mask]

    # 3) Si après nettoyage il ne reste rien, on renvoie NaN
    if len(x) == 0:
        return np.nan

    # 4) Trier x et réordonner w exactement dans le même ordre
    # order = indices qui trient x
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]

    # 5) Somme cumulée des poids
    # Exemple :
    # x_sorted = [100, 200, 300]
    # w_sorted = [  1,   2,  10]
    # cum_w    = [  1,   3,  13]
    cum_w = np.cumsum(w_sorted)

    # 6) Le seuil "50% du poids total"
    # Si total poids = 13, cutoff = 6.5
    cutoff = 0.5 * np.sum(w_sorted)

    # 7) Trouver le premier index où cum_w >= cutoff
    # np.searchsorted renvoie l'endroit où on doit insérer cutoff dans cum_w
    # pour garder l'ordre croissant.
    idx = np.searchsorted(cum_w, cutoff)

    # 8) La médiane pondérée est alors x_sorted[idx]
    return float(x_sorted[idx])


def compute_reference_and_best(
    agg: pd.DataFrame,
    reference: str = "median",
    ref_weight: str = "Nights"
) -> dict:
    """
    Calcule :
    - Prix de référence : médiane ou moyenne PONDÉRÉE par 'Nights' ou 'Reservations'
    - Meilleur prix : le bin qui maximise 'Revenue'
    - Gain vs référence : différence entre revenue du meilleur bin et revenue du bin de référence

    Paramètres :
    - reference : "median" ou "mean"
    - ref_weight : "Nights" ou "Reservations" (colonne de agg)

    Remarque importante :
    - Ici, on calcule la référence à partir de la table agrégée (bins).
      Mais en pondérant par Nights/Reservations, on se rapproche d’une référence "réaliste RM".
    """

    # 0) Cas vide : pas de données
    if len(agg) == 0:
        return {}

    # 1) Choisir les poids (Nights / Reservations)
    # Si la colonne demandée n'existe pas, on fait un fallback (poids = 1)
    # => médiane non pondérée (moins bien, mais évite de casser le code)
    if ref_weight not in agg.columns:
        weights = np.ones(len(agg), dtype=float)
    else:
        weights = agg[ref_weight].to_numpy(dtype=float)

    # Valeurs de prix (ADR moyen par bin)
    prices = agg["ADR_mean"].to_numpy(dtype=float)

    # 2) Calcul du prix de référence (pondéré)
    if reference == "median":
        # médiane pondérée : "prix typique" avec l'importance des volumes
        ref_price = _weighted_median(prices, weights)
    else:
        # moyenne pondérée
        wsum = np.sum(weights)
        ref_price = float(np.sum(prices * weights) / wsum) if wsum > 0 else float(np.nanmean(prices))

    # 3) Meilleur prix = bin qui maximise le revenue
    best_row = agg.sort_values("Revenue", ascending=False).iloc[0]
    best_price = float(best_row["ADR_mean"])
    best_revenue = float(best_row["Revenue"])

    # 4) Revenue de référence :
    # On choisit le bin dont le prix est le plus proche de ref_price
    idx = (agg["ADR_mean"] - ref_price).abs().idxmin()
    ref_revenue = float(agg.loc[idx, "Revenue"])

    # 5) Gain : différence de revenue entre meilleur bin et référence
    gain = best_revenue - ref_revenue
    gain_pct = (gain / ref_revenue * 100) if ref_revenue > 0 else np.nan

    # 6) Résultat final
    return {
        "ref_price": ref_price,
        "ref_revenue": ref_revenue,
        "best_price": best_price,
        "best_revenue": best_revenue,
        "gain": gain,
        "gain_pct": gain_pct,
    }


def replace_event_period_with_other_years(
    df: pd.DataFrame,               # le DataFrame complet (déjà clean, déjà filtré éventuellement)
    event_name: str,                # le nom exact de l'event à remplacer (ex: "Olympics_2024")
    date_col: str = "Arrival Date", # nom de la colonne de date à utiliser (ex: "Arrival Date")
    year_col: str = "Year",         # nom de la colonne contenant l'année (ex: "Year")
    only_replace_if_event: bool = True  # si True: si event absent -> on renvoie un message d'info
) -> tuple[pd.DataFrame, str]:      # la fonction renvoie (nouveau_df, message)
    """
    Objectif:
      Si l'utilisateur choisit de "ne pas compter l'event", au lieu de supprimer les lignes,
      on les remplace par un baseline "plus réaliste" :
      -> les mêmes dates (même fenêtre en mois/jour) mais sur les autres années.
    """
    # On travaille sur une copie pour ne pas modifier df original
    out = df.copy()

    # 1) On s'assure que la colonne de date est bien au format datetime
    #    (sinon comparaison >= <= ne marche pas correctement)
    out[date_col] = pd.to_datetime(out[date_col])

    # 2) On extrait uniquement les lignes de l'event ciblé
    #    ex: toutes les réservations dont Event == "Olympics_2024"
    ev = out[out["Event"] == event_name].copy()

    # 3) Si l'event n'existe pas dans les données, on s'arrête
    if len(ev) == 0:
        # si l'utilisateur veut absolument remplacer un event mais il n'existe pas :
        if only_replace_if_event:
            return out, f"Aucun event '{event_name}' trouvé dans les données filtrées."
        # sinon on renvoie juste le df sans message
        return out, ""

    # 4) On récupère l'année principale de l'event (la plus fréquente)
    #    Exemple : si l'event est en 2024, la majorité des lignes event auront Year=2024
    #    mode() = valeur la plus fréquente, iloc[0] = première si plusieurs
    ev_year = int(ev[year_col].mode().iloc[0])

    # 5) On détecte la fenêtre de dates réelle de l'event dans nos données :
    #    - ev_start = première date (min)
    #    - ev_end   = dernière date (max)
    #    (sur la colonne choisie, par défaut Arrival Date)
    ev_start = ev[date_col].min()
    ev_end = ev[date_col].max()

    # 6) On récupère toutes les années présentes dans le dataset
    #    unique() -> valeurs distinctes
    #    dropna() -> enlever NaN
    #    tolist() -> convertir en liste python
    #    sorted() -> trier
    years = sorted(out[year_col].dropna().unique().tolist())

    # 7) On garde seulement les autres années (pas l'année de l'event)
    #    Exemple : si years=[2022,2023,2024], ev_year=2024 -> other_years=[2022,2023]
    other_years = [int(y) for y in years if int(y) != ev_year]

    # 8) On va construire une "baseline" en prenant, pour chaque autre année,
    #    les lignes dans la même fenêtre mois/jour
    baseline_parts = []    # liste des DataFrames baseline (un par année)
    per_year_counts = []   # nombre de lignes baseline trouvées par année

    # 9) Pour chaque année alternative
    for y in other_years:
        # 10) On reconstruit la fenêtre start/end mais dans l'année y
        #     Exemple: ev_start=2024-07-20 -> start_y=2023-07-20
        #             ev_end  =2024-08-18 -> end_y  =2023-08-18
        start_y = pd.Timestamp(year=y, month=ev_start.month, day=ev_start.day)
        end_y = pd.Timestamp(year=y, month=ev_end.month, day=ev_end.day)

        # 11) On récupère les lignes de out qui sont dans cette fenêtre,
        #     Mais seulement celles sans event (Event == "None") pour éviter
        #     de mélanger un autre event.
        tmp = out[
            (out[date_col] >= start_y) &
            (out[date_col] <= end_y) &
            (out["Event"] == "None")
        ].copy()

        # 12) Si on a trouvé des lignes, on les garde
        if len(tmp) > 0:
            baseline_parts.append(tmp)       # on stocke la baseline de cette année
            per_year_counts.append(len(tmp)) # on stocke combien de lignes on a trouvé

    # 13) Si aucune année alternative n'a fourni de baseline,
    #     on "fallback" (solution de secours) : on retire juste l'event
    if len(baseline_parts) == 0:
        out = out[out["Event"] != event_name].copy()
        return out, f"Pas de baseline sur autres années pour '{event_name}' -> event exclu."

    # 14) On concatène toutes les baselines (toutes années confondues) en un seul DataFrame
    baseline = pd.concat(baseline_parts, ignore_index=True)

    # 15) On calcule la moyenne du volume (nombre de lignes) par année
    #     Exemple: si 2022->500 lignes, 2023->700 lignes, moyenne = 600
    avg_count = float(np.mean(per_year_counts))

    # 16) Objectif : donner à la baseline un "poids total" proche de cette moyenne.
    #     Pour ça, il faut une colonne weight.
    #     Si elle n'existe pas, on la crée à 1 partout.
    if "weight" not in out.columns:
        out["weight"] = 1.0
    if "weight" not in baseline.columns:
        baseline["weight"] = 1.0

    # 17) baseline_total = volume total actuel de baseline (somme des poids)
    baseline_total = float(baseline["weight"].sum())

    # 18) On calcule un facteur scale pour ajuster les poids. On veut qu’elle pèse environ avg_count
    #     Exemple: baseline_total=1200 et avg_count=600 -> scale=0.5
    scale = avg_count / baseline_total if baseline_total > 0 else 1.0

    # 19) On applique ce facteur à toutes les lignes baseline
    baseline["weight"] = baseline["weight"] * scale

    # 20) On marque ces lignes baseline comme non-event (sinon confusion dans l'analyse)
    #     Important : on ne veut pas garder le label event, sinon ça re-filtrerait mal
    baseline["Event"] = "None"
    baseline["Event Level"] = "None"

    # 21) On retire les lignes event originales du dataset
    out = out[out["Event"] != event_name].copy()

    # 22) Puis on ajoute la baseline à la place
    out = pd.concat([out, baseline], ignore_index=True)

    # 23) Message explicatif pour l'utilisateur
    msg = "Option activée : les événements sont remplacés par des données comparables des autres années."

    # 24) On renvoie le nouveau df + le message
    return out, msg

