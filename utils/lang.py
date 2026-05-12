"""Internationalization support – FR / EN language toggle."""

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# Translation dictionary: French (key) → English (value)
# ═══════════════════════════════════════════════════════════════════════════════

_TRANSLATIONS = {
    # ── App title & tabs ──
    "Suivi & Analyse des Katas – Premier League (K1) & Series A (SA)":
        "Kata Tracking & Analysis – Premier League (K1) & Series A (SA)",
    "Analyse Katas K1 / SA": "Kata Analysis K1 / SA",
    "Dataset": "Dataset",
    "Focus Athlète": "Athlete Focus",
    "Progression temporelle": "Score Progression",
    "Score différentiel": "Score Differential",
    "Analyse continentale": "Continental Analysis",
    "Avancement par tour": "Round Advancement",
    "Diversité kata": "Kata Diversity",
    "Graphiques interactifs": "Interactive Charts",
    "ACM": "MCA",
    "Probabilité de victoire": "Win Probability",

    # ── Common filter labels ──
    "Type de compétition": "Competition type",
    "Type compétition": "Competition type",
    "Tous": "All",
    "Sexe": "Gender",
    "Sexe des athlètes": "Athletes' gender",
    "Année(s)": "Year(s)",
    "Tours": "Rounds",
    "Compétitions": "Competitions",
    "Athlète principal": "Main athlete",
    "Comparer à": "Compare to",
    "Aucun": "None",
    "Niveau d'analyse": "Analysis level",
    "Min passages par athlète": "Min appearances per athlete",
    "Analyse": "Analysis",
    "Global": "Global",
    "Par athlète": "By athlete",
    "Par kata": "By kata",
    "Par style": "By style",
    "Athlète": "Athlete",
    "Kata": "Kata",
    "Style": "Style",
    "Athlète(s) à comparer": "Athlete(s) to compare",

    # ── Filter panel headings ──
    "### 🎛️ Filtres": "### 🎛️ Filters",
    "### 🎯 Filtres": "### 🎯 Targets",
    "### 🎯 Filtres ACM": "### 🎯 MCA Filters",
    "#### 🔎 Filtres avancés": "#### 🔎 Advanced filters",
    "#### Axes du graphique": "#### Chart axes",

    # ── Tab headers ──
    "Affichage du Dataset de Karaté": "Karate Dataset View",
    "Progression temporelle des notes": "Score Progression Over Time",
    "Analyse continentale & nationale": "Continental & National Analysis",
    "Avancement par tour – Funnel d'élimination": "Round Advancement – Elimination Funnel",
    "Analyse de la diversité kata": "Kata Diversity Analysis",
    "Générateur de Graphiques Interactifs": "Interactive Chart Generator",
    "Carte des associations Kata / Résultat (ACM)": "Kata / Result Association Map (MCA)",
    "Informations Athlète(s)": "Athlete Information",
    "Historique": "History",

    # ── Warnings & info ──
    "Aucun athlète disponible.": "No athlete available.",
    "Aucune donnée pour l'athlète sélectionné.": "No data for the selected athlete.",
    "Aucun sexe disponible dans les données filtrées.": "No gender available in filtered data.",
    "Aucun athlète disponible avec les filtres sélectionnés.": "No athlete available with selected filters.",
    "Sélectionnez un athlète pour afficher son profil.": "Select an athlete to display their profile.",
    "Aucune donnée dans ce périmètre.": "No data in this scope.",
    "Aucun historique trouvé pour cet athlète avec les filtres actuels.":
        "No history found for this athlete with current filters.",

    # ── Placeholders ──
    "Choisir un athlète...": "Choose an athlete...",

    # ── Athlete info ──
    "Non spécifié": "Not specified",
    "ans": "years old",

    # ── History ──
    "Historique des tours": "Round history",
    "Tour": "Round",
    "Note": "Score",
    "Victoire": "Victory",
    "Competition": "Competition",
    "Oui": "Yes",
    "Non": "No",
    "Inconnu": "Unknown",

    # ── Score differential ──
    "Marge de victoire / Score différentiel": "Victory Margin / Score Differential",

    # ── Charts/Graphs tab ──
    "Variable numérique (X)": "Numerical variable (X)",
    "Variable catégorielle (X)": "Categorical variable (X)",
    "Variable numérique (Y)": "Numerical variable (Y)",
    "Variable catégorielle (Y)": "Categorical variable (Y)",
    "Aucune": "None",

    # ── Proba tab ──
    "Probabilité de victoire par kata": "Win probability by kata",
    "Athlète A": "Athlete A",
    "Athlète B (adversaire)": "Athlete B (opponent)",
    "Tour simulé": "Simulated round",

    # ── Kata diversity ──
    "Spécialiste (1-2 katas)": "Specialist (1-2 katas)",
    "Modéré (3-4 katas)": "Moderate (3-4 katas)",
    "Polyvalent (5+ katas)": "Versatile (5+ katas)",

    # ── Continental ──
    "Nation": "Nation",
    "Continent": "Continent",
    "Region_monde": "World Region",

    # ── Tour display ──
    "1er tour Series A": "1st round Series A",
    "2ème tour Series A": "2nd round Series A",
    "3ème tour Series A": "3rd round Series A",
    "1er tour de poule K1": "1st pool round K1",
    "2ème tour de poule K1": "2nd pool round K1",
    "3ème tour de poule K1": "3rd pool round K1",
    "8ème de finale Series A": "Round of 16 Series A",
    "Quart de finale Series A": "Quarter-final Series A",
    "Demi-finale Series A": "Semi-final Series A",
    "Quart de finale K1": "Quarter-final K1",
    "Demi-finale K1": "Semi-final K1",
    "Match pour la médaille de bronze": "Bronze medal match",
    "Finale": "Final",
    "Poule": "Pool",
    "Quart de finale": "Quarter-final",
    "Demi finale": "Semi-final",
    "1er Tour": "1st Round",
    "2ème Tour": "2nd Round",
    "3ème Tour": "3rd Round",
    "Finale de poule": "Pool final",
    "Place de 3": "3rd place",
    "Bronze / Place de 3": "Bronze / 3rd place",

    # ── Tour type (ACM) ──
    "Match de médaille": "Medal match",
    "Quart/Demi/Finale de poule": "Quarter/Semi/Pool final",
    "Tour 1": "Round 1",
    "Tour 2": "Round 2",
    "Tour 3": "Round 3",
    "Type de Tour": "Round type",

    # ── Interpretations / color badges ──
    "✔ Bon": "✔ Good",
    "~ Moyen": "~ Average",
    "✘ Faible": "✘ Weak",

    # ── Glossaire ──
    "📖 Glossaire": "📖 Glossary",

    # ── Footer ──
    "Source : SportData": "Source: SportData",

    # ── Chart guides ──
    "📖 Comment lire ce graphique ?": "📖 How to read this chart?",
    "💡 Comment lire cet onglet ?": "💡 How to read this tab?",

    # ── Graph tab instructions ──
    "Comment utiliser les graphiques interactifs": "How to use interactive charts",

    # ── Athlete focus subheaders ──
    "Aucun athlète comparé sélectionné": "No comparison athlete selected",

    # ── Download ──
    "📥 Télécharger CSV": "📥 Download CSV",
    "Télécharger les données filtrées": "Download filtered data",

    # ── Progression tab ──
    "Moyenne mobile": "Moving average",
    "Note moyenne": "Average score",
    "Écart-type": "Standard deviation",
    "Tendance": "Trend",

    # ── Misc ──
    "Filtrer par": "Filter by",
    "Tours (N_Tour)": "Rounds (N_Tour)",
    "Résumé": "Summary",
    "Lignes": "Rows",
    "Athlètes": "Athletes",
    "Compétitions": "Competitions",
    "Katas distincts": "Distinct katas",
    "Nombre total de lignes dans le dataset filtré": "Total number of rows in the filtered dataset",
    "Nombre d'athlètes distincts dans la sélection": "Number of distinct athletes in the selection",
    "Nombre de compétitions différentes": "Number of different competitions",
    "Nombre de katas différents joués": "Number of different katas performed",
    "Colonnes à afficher": "Columns to display",
    "Télécharger le CSV filtré": "Download filtered CSV",
    "Rechercher un athlète": "Search for an athlete",
    "Compétition(s)": "Competition(s)",

    # ── Proba tab extra ──
    "Aucun athlète disponible avec ces filtres.": "No athlete available with these filters.",
    "Aucun adversaire compatible trouvé.": "No compatible opponent found.",
    "Aucun tour disponible.": "No round available.",
    "Tour simulé": "Simulated round",
    "Athlète A": "Athlete A",
    "Athlète B (adversaire)": "Athlete B (opponent)",
    "Comparaison sélectionnée": "Selected comparison",
    "Impossible de reconstruire des matchs avec notes.": "Unable to reconstruct matches with scores.",

    # ── Score differential extra ──
    "Score différentiel – Marge de victoire": "Score Differential – Victory Margin",
    "Matchs": "Matches",
    "Marge moy.": "Avg. margin",
    "Marge médiane": "Median margin",
    "% serrés (<0.5)": "% close (<0.5)",
    "Nombre total de matchs reconstruits": "Total number of reconstructed matches",
    "Écart moyen entre gagnant et perdant": "Average gap between winner and loser",
    "La marge du milieu": "The middle margin",
    "Proportion de matchs très disputés": "Proportion of highly contested matches",
    "Distribution de la marge de victoire": "Victory margin distribution",

    # ── Progression extra ──
    "Moyenne mobile (3 compétitions)": "Moving average (3 competitions)",
    "Sélectionnez au moins un athlète dans le panneau de filtres à gauche.":
        "Select at least one athlete in the filter panel on the left.",
    "Aucune donnée avec notes pour ces athlètes.": "No data with scores for these athletes.",
    "Évolution de la note moyenne par compétition": "Average score evolution by competition",
    "Compétition": "Competition",
    "Résumé par athlète": "Summary by athlete",
    "Distribution des notes": "Score distribution",
    "Distribution des notes par athlète": "Score distribution by athlete",

    # ── Glossary sidebar ──
    "📖 Glossaire des termes": "📖 Glossary of terms",

    # ── Score diff subheaders ──
    "Marge par tour": "Margin by round",
    "Type de victoire par tour": "Victory type by round",
    "Marge par compétition": "Margin by competition",

    # ── Athlete focus / charts ──
    "Tour maximal atteint par compétition": "Maximum round reached per competition",
    "Histogramme des Katas effectués": "Kata Performance Histogram",
    "Nombre de Katas effectués": "Number of Katas performed",
    "Nombre de fois": "Number of times",
    "Moyenne des notes par Tour": "Average score by Round",
    "Moyenne des notes par tour (tours réellement disputés)": "Average score by round (actually contested rounds)",
    "Moyenne des notes par Kata": "Average score by Kata",
    "Aucune donnée": "No data",
    "pour les athlètes sélectionnés.": "for the selected athletes.",
    "Aucun Kata à afficher pour les tours sélectionnés.": "No Kata to display for selected rounds.",
    "Aucune note disponible pour construire le diagramme par tour.": "No scores available to build the round diagram.",
    "Aucune note disponible pour construire le diagramme par Kata.": "No scores available to build the Kata diagram.",
    "Tour maximal": "Maximum round",

    # ── Score differential charts ──
    "Marge – K1": "Margin – K1",
    "Marge – SA": "Margin – SA",
    "Marge par tour – K1": "Margin by round – K1",
    "Marge par tour – SA": "Margin by round – SA",
    "Marge (|Note_R − Note_B|)": "Margin (|Score_R − Score_B|)",
    "Aucun match K1.": "No K1 match.",
    "Aucun match SA.": "No SA match.",
    "Serrée (<0.5)": "Close (<0.5)",
    "Nette (0.5-1.5)": "Clear (0.5-1.5)",
    "Dominante (>1.5)": "Dominant (>1.5)",
    "Type de victoire par tour – K1": "Victory type by round – K1",
    "Type de victoire par tour – SA": "Victory type by round – SA",
    "Marge moy. par compétition – K1": "Avg. margin by competition – K1",
    "Marge moy. par compétition – SA": "Avg. margin by competition – SA",
    "Type_Victoire": "Victory_Type",

    # ── Continental tab ──
    "Performances par": "Performance by",
    "Distribution des notes par": "Score distribution by",
    "Athlètes total": "Total athletes",
    "Note moy. globale": "Overall avg. score",
    "Nombre total d'athlètes distincts": "Total number of distinct athletes",
    "Note moyenne de tous les passages dans ce périmètre": "Average score of all appearances in this scope",
    "représentés": "represented",
    "Finalistes par": "Finalists by",
    "Aucune donnée de finale dans ce périmètre.": "No final data in this scope.",
    "Note moy.": "Avg. score",
    "Nombre de": "Number of",
    "différents dans la sélection": "different in the selection",
    "Le Top 3 et les graphiques excluent les": "Top 3 and charts exclude",
    "avec moins de": "with fewer than",
    "athlètes (non représentatifs).": "athletes (not representative).",
    "athlètes": "athletes",
    "par": "by",
    "Notes par": "Scores by",

    # ── Tour advancement ──
    "Funnel d'avancement": "Advancement funnel",
    "Note moyenne par tour": "Average score by round",
    "Note moy. par tour – K1": "Avg. score by round – K1",
    "Note moy. par tour – SA": "Avg. score by round – SA",
    "Aucune donnée K1.": "No K1 data.",
    "Aucune donnée SA.": "No SA data.",
    "Top athlètes – Tours avancés atteints": "Top athletes – Advanced rounds reached",
    "Type compétition": "Competition type",
    "Sexe (top)": "Gender (top)",
    "Tours (min atteint)": "Rounds (min reached)",
    "Aucune donnée avec ces filtres.": "No data with these filters.",
    "Nb athlètes": "Nb athletes",
    "Nb passages": "Nb appearances",

    # ── Kata diversity ──
    "Spécialiste (1-2)": "Specialist (1-2)",
    "Modéré (3-4)": "Moderate (3-4)",
    "Polyvalent (5+)": "Versatile (5+)",
    "Athlètes analysés": "Athletes analyzed",
    "Katas distincts moy.": "Avg. distinct katas",
    "Médiane katas": "Median katas",
    "Nombre d'athlètes ayant le minimum de passages requis": "Number of athletes with the minimum required appearances",
    "En moyenne, combien de katas différents chaque athlète utilise": "On average, how many different katas each athlete uses",
    "La moitié des athlètes utilisent plus de katas, l'autre moitié moins": "Half the athletes use more katas, the other half fewer",
    "Diversité kata vs Win rate": "Kata diversity vs Win rate",
    "Nb katas distincts vs Win rate": "Nb distinct katas vs Win rate",
    "Win rate par profil de diversité": "Win rate by diversity profile",
    "Spécialistes vs Polyvalents – Win rate": "Specialists vs Versatile – Win rate",
    "Note moyenne par profil de diversité": "Average score by diversity profile",
    "Spécialistes vs Polyvalents – Note moyenne": "Specialists vs Versatile – Average score",
    "Résumé par profil": "Summary by profile",
    "Classement des athlètes": "Athlete ranking",
    "Trier par": "Sort by",
    "Profil": "Profile",
    "Aucun athlète avec ces critères.": "No athlete with these criteria.",
    "Aucun athlète avec": "No athlete with",
    "Katas les plus utilisés": "Most used katas",
    "Top 20 katas – Nombre d'utilisations": "Top 20 katas – Number of uses",
    "Aucune donnée.": "No data.",

    # ── Graphs tab ──
    "Type de graphique généré": "Generated chart type",
    "Distribution de la variable": "Distribution of variable",
    "Filtre (optionnel)": "Filter (optional)",
    "Variable catégorielle pour filtrer": "Categorical variable to filter",
    "Modalités de": "Values of",
    "Aucune donnée disponible avec ces filtres.": "No data available with these filters.",
    "Moyenne": "Mean",
    "Médiane": "Median",
    "Test de normalité (automatique)": "Normality test (automatic)",
    "Effectuer le test de normalité": "Run normality test",
    "Distribution de": "Distribution of",
    "par rapport à chaque modalité de": "relative to each value of",
    "Test Statistique (automatique)": "Statistical Test (automatic)",
    "Effectuer le test statistique": "Run statistical test",
    "Histogramme des effectifs de chaque modalité de": "Count histogram of each value of",
    "Effectif": "Count",
    "Histogramme des proportions de chaque modalité de": "Proportion histogram of each value of",
    "Proportions des modalités de": "Proportions of values of",
    "en fonction de": "relative to",
    "Effectuer le test d'indépendance": "Run independence test",
    "Nuage de points entre": "Scatter plot between",
    "et": "and",
    "Test de corrélation (automatique)": "Correlation test (automatic)",
    "Effectuer le test de corrélation": "Run correlation test",
    "Veuillez sélectionner des combinaisons cohérentes de variables.": "Please select coherent variable combinations.",

    # ── ACM tab ──
    "Type(s) de Tour à inclure": "Round type(s) to include",
    "Aucun 'Type de Tour' sélectionné. Veuillez en sélectionner au moins un.": "No 'Round type' selected. Please select at least one.",
    "#### 🧍 Sexe": "#### 🧍 Gender",
    "Sexe à inclure": "Gender to include",
    "Aucune donnée disponible pour le sexe sélectionné.": "No data available for the selected gender.",
    "#### 🧬 Style de kata": "#### 🧬 Kata style",
    "Style(s) à inclure": "Style(s) to include",
    "Aucun style sélectionné, tous les styles sont inclus par défaut.": "No style selected, all styles included by default.",
    "Aucune donnée disponible après filtrage par style.": "No data available after style filtering.",
    "#### 🥋 Katas": "#### 🥋 Katas",
    "Katas à inclure dans l'ACM": "Katas to include in MCA",
    "Aucun kata disponible après filtrage.": "No kata available after filtering.",
    "Aucun kata sélectionné. Veuillez en sélectionner au moins un.": "No kata selected. Please select at least one.",
    "Aucune donnée disponible après filtrage par katas.": "No data available after kata filtering.",
    "#### 🎨 Options d'affichage": "#### 🎨 Display options",
    "Afficher les individus": "Show individuals",
    "Afficher les modalités des variables": "Show variable modalities",
    "Paramètres de l'ACM": "MCA Parameters",
    "Représentation de l'ACM": "MCA Representation",
    "Carte factorielle de l'ACM": "MCA Factor Map",
    "Interprétation de l'ACM": "MCA Interpretation",
    "Interpréter automatiquement l'ACM": "Automatically interpret MCA",
    "Pas de colonne 'Style' dans les données : filtrage par style désactivé.": "No 'Style' column in data: style filtering disabled.",
    "Aucune information de style disponible après filtrage.": "No style information available after filtering.",
    "Aucune donnée disponible après suppression des valeurs manquantes. Impossible de réaliser l'ACM.": "No data available after removing missing values. Unable to perform MCA.",

    # ── Progression tab ──
    "#### 💡 Interprétation": "#### 💡 Interpretation",

    # ── Proba tab detailed ──
    "### 🎯 Paramètres": "### 🎯 Parameters",
    "#### 🥋 Katas testés (pour A)": "#### 🥋 Tested katas (for A)",
    "#### 🏆 Top 3 katas à faire": "#### 🏆 Top 3 recommended katas",
    "🔮 Lancer la simulation": "🔮 Run simulation",
    "📊 Résultats de la simulation": "📊 Simulation results",
    "Probabilité de victoire de A": "Win probability of A",
    "Confiance": "Confidence",
    "Facteurs clés": "Key factors",
    "Avertissement": "Warning",

    # ── Proba victoire kata tab ──
    "### 🎯 Paramètres": "### 🎯 Parameters",
    "#### 🥋 Katas testés (pour A)": "#### 🥋 Katas tested (for A)",
    "#### 🏆 Top 3 katas à faire": "#### 🏆 Top 3 katas to perform",
    "🏆 Top 3": "🏆 Top 3",
    "🏆 Top 3 katas à faire": "🏆 Top 3 katas to perform",
    "Katas à tester": "Katas to test",
    "A – Note moy.": "A – Avg. score",
    "B – Note moy.": "B – Avg. score",
    "Note moyenne de l'athlète A sur tous ses passages": "Average score of athlete A across all performances",
    "Note moyenne de l'athlète B sur tous ses passages": "Average score of athlete B across all performances",
    "Ranking moy.": "Avg. ranking",
    "Tour": "Round",
    "tour": "round",
    "Périmètre": "Scope",
    "Historique scope": "Scope history",
    "matchs": "matches",
    "📊 Performance du modèle (cross-validation)": "📊 Model performance (cross-validation)",
    "Accuracy (CV) : données insuffisantes": "Accuracy (CV): insufficient data",
    "AUC (CV) : données insuffisantes": "AUC (CV): insufficient data",
    "##### Feature importances (coefficients)": "##### Feature importances (coefficients)",
    "Même nationalité": "Same nationality",
    "Même style de kata": "Same kata style",
    "Compétition K1 (vs SA)": "K1 competition (vs SA)",
    "Avantage face-à-face historique": "Historical head-to-head advantage",
    "Avantage win rate global": "Overall win rate advantage",
    "Win rate de A avec ce kata": "A's win rate with this kata",
    "Faiblesse de B face à ce kata": "B's weakness against this kata",
    "Efficacité du kata dans ce tour": "Kata effectiveness in this round",
    "Effet résiduel du kata": "Residual kata effect",
    "Différence de notes (A-B)": "Score difference (A-B)",
    "Avantage au classement mondial": "World ranking advantage",
    "Expérience de A avec ce kata": "A's experience with this kata",
    "Note de A avec ce kata (vs moyenne)": "A's score with this kata (vs avg)",
    "Différence de tendance de notes": "Score trend difference",
    "Différence de dynamique récente": "Recent momentum difference",
    "Différence d'âge": "Age difference",
    "A joue son kata favori": "A plays their favourite kata",
    "Différence de régularité": "Consistency difference",
    "Différence d'expérience (nb matchs)": "Experience difference (match count)",
    "Différence de diversité kata": "Kata diversity difference",
    "B a déjà vu ce kata": "B has seen this kata before",
    "A joue à domicile (continent)": "A plays at home (continent)",
    "Phase de compétition (poule → finale)": "Competition phase (pool → final)",
    "Explication": "Explanation",
    "Impossible d'extraire les coefficients.": "Unable to extract coefficients.",
    "Sélectionne au moins un kata à tester.": "Select at least one kata to test.",
    "Calculer les probabilités (katas sélectionnés)": "Calculate probabilities (selected katas)",
    "Résultats": "Results",
    "sur ta sélection": "from your selection",
    "Détail complet": "Full detail",
    "Probabilité de victoire estimée par kata (A) – sélection": "Estimated win probability by kata (A) – selection",
    "✅ Modèle v3 — 22 features : notes, ranking, H2H, tendance temporelle, "
    "momentum, âge, kata favori, consistance, expérience, diversité, "
    "familiarité adversaire, avantage géo, tour.":
        "✅ Model v3 — 22 features: scores, ranking, H2H, temporal trend, "
        "momentum, age, favourite kata, consistency, experience, diversity, "
        "opponent familiarity, geographic advantage, round.",
    "katas déjà joués par A": "katas already played by A",
    "tous les katas du style": "all style katas",
    "Impossible de proposer un Top 3 : aucun kata candidat trouvé.": "Unable to suggest a Top 3: no candidate kata found.",
    "Source candidats": "Candidate source",
    "Top katas recommandés (Top 10 affiché)": "Recommended top katas (Top 10 shown)",

    # ── Graphs tab – test interpretations ──
    "Test utilisé": "Test used",
    "Statistique": "Statistic",
    "Coefficient": "Coefficient",
    "Degrés de liberté": "Degrees of freedom",
    "La normalité **n'est pas rejetée** pour": "Normality is **not rejected** for",
    "La normalité est **rejetée** pour": "Normality is **rejected** for",
    "Différence **significative** de": "**Significant** difference in",
    "entre les groupes de": "between groups of",
    "Aucune différence significative de": "No significant difference in",
    "Association **significative** entre": "**Significant** association between",
    "et": "and",
    "Aucune association significative entre": "No significant association between",
    "Corrélation **significative**": "**Significant** correlation",
    "Pas de corrélation significative": "No significant correlation",

    # ── utils/stats.py comments ──
    "Effectif insuffisant pour un test de normalité (n < 3).":
        "Insufficient sample size for normality test (n < 3).",
    "Test adapté aux petits échantillons (n < 20). Il est assez sensible aux déviations.":
        "Test suited for small samples (n < 20). Fairly sensitive to deviations.",
    "Test basé sur l'asymétrie et l'aplatissement, adapté aux échantillons ≥ 20. "
    "Avec des échantillons très grands, il peut détecter de très faibles déviations.":
        "Test based on skewness and kurtosis, suited for samples ≥ 20. "
        "With very large samples, it can detect very small deviations.",
    "Comparaison de moyennes": "Mean comparison",
    "Pas assez de groupes avec au moins 3 observations.":
        "Not enough groups with at least 3 observations.",
    "Tous les groupes ont n ≥ 30. ANOVA est robuste via le théorème central limite, "
    "même si la normalité n'est pas parfaite.":
        "All groups have n ≥ 30. ANOVA is robust via the central limit theorem, "
        "even if normality is not perfect.",
    "Taille d'échantillon modérée et normalité non rejetée dans chaque groupe : "
    "ANOVA est appropriée.":
        "Moderate sample size and normality not rejected in each group: "
        "ANOVA is appropriate.",
    "Normalité incertaine ou rejetée dans au moins un groupe et/ou effectif faible : "
    "on utilise le test non-paramétrique de Kruskal-Wallis.":
        "Normality uncertain or rejected in at least one group and/or small sample: "
        "using the non-parametric Kruskal-Wallis test.",
    "Test d'indépendance": "Independence test",
    "Table de contingence vide, impossible de tester.":
        "Empty contingency table, cannot test.",
    "Tableau 2x2 avec au moins une fréquence attendue < 5 : "
    "le test exact de Fisher est plus approprié que le Chi².":
        "2x2 table with at least one expected frequency < 5: "
        "Fisher's exact test is more appropriate than Chi².",
    "Conditions du Chi² vérifiées (fréquences attendues suffisantes).":
        "Chi² conditions met (expected frequencies sufficient).",
    "Effectif insuffisant pour un test de corrélation (n < 3).":
        "Insufficient sample size for correlation test (n < 3).",
    "Petits effectifs (n < 20) et normalité non rejetée pour X et Y : "
    "corrélation de Pearson appropriée.":
        "Small sample (n < 20) and normality not rejected for X and Y: "
        "Pearson correlation appropriate.",
    "Petits effectifs (n < 20) et normalité incertaine/rejetée : "
    "corrélation de Spearman (non-paramétrique) préférable.":
        "Small sample (n < 20) and normality uncertain/rejected: "
        "Spearman correlation (non-parametric) preferred.",
    "Effectif n ≥ 20 : la corrélation de Pearson est robuste même en cas de légère "
    "déviation à la normalité (TCL).":
        "Sample n ≥ 20: Pearson correlation is robust even with slight "
        "deviation from normality (CLT).",
    "Spearman mesure la corrélation monotone basée sur les rangs, "
    "utile en cas de relations non linéaires ou de valeurs extrêmes.":
        "Spearman measures monotonic rank-based correlation, "
        "useful for non-linear relationships or extreme values.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_lang() -> str:
    """Return current language code ('fr' or 'en')."""
    return st.session_state.get("lang", "fr")


def t(text: str) -> str:
    """Translate *text* based on the current language.

    In French mode the text is returned as-is.
    In English mode the translation dictionary is consulted.
    """
    if get_lang() == "fr":
        return text
    return _TRANSLATIONS.get(text, text)


def render_language_selector():
    """Render a compact flag toggle in the top-right corner of the page."""
    # Use columns to push selector to the right
    cols = st.columns([6, 1])
    with cols[1]:
        options = {"🇫🇷 FR": "fr", "🇬🇧 EN": "en"}
        current = get_lang()
        default_idx = 0 if current == "fr" else 1
        selected = st.selectbox(
            "Lang",
            list(options.keys()),
            index=default_idx,
            key="lang_selector",
            label_visibility="collapsed",
        )
        new_lang = options[selected]
        if new_lang != st.session_state.get("lang", "fr"):
            st.session_state["lang"] = new_lang
            st.rerun()
