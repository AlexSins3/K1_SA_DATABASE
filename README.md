# K1_SA_DATABASE – Analyse Katas Premier League & Series A

Application **Streamlit** de consultation et d'analyse d'une base de données sur les compétitions de **Karate Kata** du circuit **K1 (Premier League)** et **SA (Series A)** de 2024 à 2026.

## Fonctionnalités

| Onglet | Description |
|--------|-------------|
| 🥋 **Focus Athlète** | Fiche athlète, comparaison head-to-head, tour maximal par compétition, histogramme des katas, diagrammes Kiviat (notes par tour / kata), historique des rencontres |
| 🎯 **Probabilité de victoire** | Modèle de régression logistique avec shrinkage bayésien, recommandation Top 3 katas, métriques de cross-validation |
| 🆚 **Comparaison de katas** | Analyse comparative des performances par kata, écarts moyens, classement par tour |
| 🔍 **Tendances & Réalités** | Évolution des notes moyennes au fil des compétitions, tendances par athlète ou groupe, interprétation automatique |
| ⚔️ **Analyse des matchs** | Reconstruction des matchs R/B, analyse des marges de victoire par tour, système de drapeaux (2026+), visualisation des écarts |
| 🗺️ **ACM** | Analyse des Correspondances Multiples sur Kata × Tour × Victoire avec interprétation automatique |
| 📊 **Exploration libre** | Générateur de graphiques avec choix des axes (numérique / catégoriel) et tests statistiques automatiques (normalité, ANOVA / Kruskal-Wallis, Chi² / Fisher, corrélation Pearson / Spearman) |
| 📋 **Dataset** | Consultation, filtrage multi-critères (année, sexe, grade, compétition, recherche athlète) et export CSV |

Interface bilingue FR / EN avec sélecteur de langue, glossaire contextuel et aide intégrée par onglet.

## Structure du projet

```
K1_SA_DATABASE/
├── app.py                      # Point d'entrée Streamlit
├── config.py                   # Chemins & constantes globales
├── requirements.txt
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
│   ├── athlete_focus/          # Onglet Focus Athlète (décomposé)
│   │   ├── __init__.py
│   │   ├── filters.py
│   │   ├── charts.py
│   │   └── history.py
│   ├── proba_victoire_kata.py  # Onglet Probabilité de victoire
│   ├── kata_comparison.py      # Onglet Comparaison de katas
│   ├── tendances.py            # Onglet Tendances & Réalités
│   ├── match_analysis.py       # Onglet Analyse des matchs
│   ├── acm.py                  # Onglet ACM
│   ├── graphs.py               # Onglet Exploration libre
│   ├── dataset_view.py         # Onglet Dataset
│   ├── continental.py          # (non utilisé – analyse continentale)
│   ├── kata_diversity.py       # (non utilisé – diversité kata)
│   ├── score_differential.py   # (non utilisé – score différentiel)
│   └── tour_advancement.py     # (non utilisé – avancement par tour)
├── data/
│   ├── Database_K1_SA.csv      # Base de données principale
│   └── script_correction.py    # Script de correction des styles
├── up_to_date/
│   └── progression.py          # Module de progression (en développement)
├── tests/
│   ├── test_data_helpers.py
│   └── test_stats.py
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

Les données proviennent de **SportData** et couvrent les compétitions K1 Premier League et Series A de 2024 à 2026.

## Dépendances principales

- **streamlit** – Interface web interactive
- **pandas** / **numpy** – Manipulation de données
- **plotly** – Visualisations interactives
- **scikit-learn** – Modèle de régression logistique (proba de victoire)
- **scipy** – Tests statistiques
- **prince** – Analyse des Correspondances Multiples (ACM)
- **pytest** – Tests unitaires

## Auteur

**Alexis Vincent**
