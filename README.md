# Sensibilité au Prix — Outil Revenue Management

## Objectif
Cet outil permet d’analyser la sensibilité des réservations au prix (ADR)
à partir de données réelles issues d’exports Revenue & Analytics.

Il est destiné à un usage métier (Revenue Management) et vise à :
- visualiser la relation Prix → Volume / Nuitées / Revenue
- comparer différents scénarios de filtrage
- identifier un prix optimal historique
- mesurer un gain potentiel par rapport à un prix de référence

---

## Utilisation (interface utilisateur)

L’outil est accessible via une application web Streamlit.

### Étapes principales
1. Uploader un fichier d’export R&A (format TSV, encodage UTF-16)
2. Nettoyer les données (RAW → CLEAN)
3. Définir des événements (optionnel)
4. Appliquer des filtres métier :
   - Property
   - Market Code
   - Source Code
   - Saison
   - Nationalité
   - Lead Time
   - Période calendaire
5. Visualiser :
   - une courbe unique Prix -> Y
   - un tableau agrégé
   - un prix optimal et un gain potentiel

Aucune donnée n’est stockée automatiquement : les résultats dépendent
uniquement des fichiers chargés par l’utilisateur.

---

## Données attendues
- Export brut R&A
- Séparateur : tabulation
- Encodage : UTF-16
- Colonnes métier standard (dates, ADR, nuitées, revenue, segments…)

Les fichiers de données ne sont pas versionnés dans ce dépôt.

---

## Structure du projet (vue d’ensemble)

- `app/` : application Streamlit
- `app/pages/` : pages fonctionnelles de l’interface
- `src/` : logique métier (nettoyage, agrégation, calculs RM)
- `requirements.txt` : dépendances Python

---

## Déploiement
L’application est conçue pour être déployée sur Streamlit Cloud
ou sur une infrastructure interne.

---

## Confidentialité
Ce dépôt est privé.
Les données utilisées dans l’application sont fournies par l’utilisateur
et ne sont pas incluses dans le code source.

---

## Auteur
Projet développé dans un cadre professionnel / analytique.
