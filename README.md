# K1_SA_DATABASE – Analyse Katas Premier League & Series A

Application **Streamlit** de consultation et d'analyse d'une base de données sur les compétitions de **Karate Kata** du circuit **K1 (Premier League)** et **SA (Series A)** en 2024-2025.

## Fonctionnalités

| Onglet | Description |
|--------|-------------|
| **Dataset** | Consultation, filtrage multi-critères (année, sexe, grade, compétition, recherche athlète) et export CSV |
| **Focus Athlète** | Fiche athlète, comparaison head-to-head, tour maximal par compétition, histogramme des katas, diagrammes Kiviat (notes par tour / kata), historique des rencontres |
| **Progression temporelle** | Évolution des notes moyennes au fil des compétitions, tendances par athlète ou groupe, interprétation automatique |
| **Score différentiel** | Analyse des marges de victoire par tour, reconstruction des matchs R/B, visualisation des écarts de notes |
| **Analyse continentale** | Répartition géographique des performances par continent / pays, comparaisons inter-régions |
| **Avancement par tour** | Funnel de progression des athlètes à travers les tours, probabilités d'avancement |
| **Diversité kata** | Analyse de la variété des katas utilisés par athlète, par tour ou par compétition |
| **Graphiques interactifs** | Générateur de graphiques avec choix des axes (numérique / catégoriel) et tests statistiques automatiques (normalité, ANOVA / Kruskal-Wallis, Chi² / Fisher, corrélation Pearson / Spearman) |
| **ACM** | Analyse des Correspondances Multiples sur Kata × Tour × Victoire avec interprétation automatique |
| **Probabilité de victoire** | Modèle de régression logistique avec shrinkage bayésien, recommandation Top 3 katas, métriques de cross-validation |

Interface bilingue FR / EN avec sélecteur de langue, glossaire contextuel et aide intégrée par onglet.

## Structure du projet

```
K1_SA_DATABASE/
├── app.py                      # Point d'entrée Streamlit
├── config.py                   # Chemins & constantes globales
├── constants/
│   ├── tours.py                # Mappings des tours K1/SA/Kiviat
│   └── styles.py               # Listes Shotokan / ShitoRyu
├── utils/
│   ├── ui.py                   # Composants UI partagés (CSS, footer, filtres)
│   ├── data_helpers.py         # Conversions Victoire, helpers données
│   ├── display.py              # Formatage d'affichage (tours, noms, colonnes)
│   ├── interpretations.py      # Aide contextuelle, indicateurs colorés, glossaire
│   ├── lang.py                 # Internationalisation FR / EN
│   └── stats.py                # Tests statistiques (normalité, ANOVA, Chi²…)
├── tabs/
│   ├── dataset_view.py         # Onglet Dataset
│   ├── athlete_focus/          # Onglet Focus Athlète (décomposé)
│   │   ├── __init__.py
│   │   ├── filters.py
│   │   ├── charts.py
│   │   └── history.py
│   ├── progression.py          # Onglet Progression temporelle
│   ├── score_differential.py   # Onglet Score différentiel
│   ├── continental.py          # Onglet Analyse continentale
│   ├── tour_advancement.py     # Onglet Avancement par tour
│   ├── kata_diversity.py       # Onglet Diversité kata
│   ├── graphs.py               # Onglet Graphiques interactifs
│   ├── acm.py                  # Onglet ACM
│   └── proba_victoire_kata.py  # Onglet Probabilité de victoire
├── data/
│   ├── Database_K1_SA.csv      # Base de données principale
│   └── script_correction.py    # Script de correction des styles
├── tests/
│   ├── test_data_helpers.py
│   └── test_stats.py
└── requirements.txt
```

## Installation

```bash
# Cloner le dépôt
cd K1_SA_DATABASE

# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Source des données

Les données proviennent de **SportData** et couvrent les compétitions K1 Premier League et Series A de 2024-2025.

## Auteur

**Alexis Vincent**
