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
├── app.py                       # Point d'entrée Streamlit
├── config.py                    # Config centralisée (DATA_PATH)
├── requirements.txt
├── .streamlit/config.toml
│
├── tabs/                        # Onglets Streamlit
│   ├── athlete_focus/           # Sous-module Focus Athlète
│   │   ├── filters.py
│   │   ├── charts.py
│   │   └── history.py
│   ├── proba_victoire_kata.py
│   ├── kata_comparison.py
│   ├── tendances.py
│   ├── match_analysis.py
│   ├── acm.py
│   ├── graphs.py
│   └── dataset_view.py
│
├── utils/                       # Utilitaires partagés
│   ├── ui.py                    # Composants UI (CSS, footer)
│   ├── data_helpers.py          # Conversions, helpers données
│   ├── display.py               # Formatage d'affichage
│   ├── interpretations.py       # Aide contextuelle, glossaire
│   ├── lang.py                  # Internationalisation FR/EN
│   └── stats.py                 # Tests statistiques
│
├── constants/                   # Constantes statiques
│   ├── tours.py                 # Mappings tours K1/SA
│   └── styles.py                # Listes Shotokan / ShitoRyu
│
├── data/                        # Données CSV
│   └── Database_K1_SA.csv
│
├── scripts/                     # CLI & maintenance
│   ├── data_correction.py       # Corrections de styles dans le CSV
│   └── progression.py           # Suivi de progression
│
├── tests/
│   ├── test_data_helpers.py
│   └── test_stats.py
│
└── app_publique/                # (Futur) Version publique avec authentification
    ├── api/                     # Backend FastAPI (auth + tracking)
    ├── auth/                    # Couche auth côté Streamlit
    ├── admin_tab.py             # Onglet admin
    ├── run_api.py               # Point d'entrée API
    └── manage_users.py          # Gestion des comptes CLI
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

L'application est également déployée sur **Streamlit Cloud**.

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

## À propos de `app_publique/`

Le dossier `app_publique/` contient un système complet d'authentification (FastAPI + JWT + SQLAlchemy) prévu pour une future version publique de l'application. Il inclut :

- **API REST** (FastAPI) avec login, register, changement de mot de passe
- **Tracking d'activité** (pages vues par utilisateur)
- **Panneau admin** (gestion des utilisateurs, historique d'utilisation)
- **Gestion CLI des comptes** (création, changement de mots de passe)

Cette version nécessite un hébergement capable de faire tourner deux processus (API + Streamlit), par exemple Render, Railway ou un VPS. Elle n'est pas compatible avec Streamlit Cloud seul.

## Auteur

**Alexis Vincent**

