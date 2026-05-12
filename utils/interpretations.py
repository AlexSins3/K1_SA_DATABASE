"""Aide contextuelle, indicateurs colorés et glossaire pour coachs."""

from __future__ import annotations

import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# Indicateurs colorés (vert / orange / rouge)
# ═══════════════════════════════════════════════════════════════════════════════

def _color_badge(text: str, color: str) -> str:
    """Return an HTML badge with the given color."""
    bg = {"green": "#d4edda", "orange": "#fff3cd", "red": "#f8d7da"}
    fg = {"green": "#155724", "orange": "#856404", "red": "#721c24"}
    return (
        f"<span style='background:{bg[color]};color:{fg[color]};"
        f"padding:2px 8px;border-radius:4px;font-weight:600;font-size:13px;'>"
        f"{text}</span>"
    )


def colored_metric(label: str, value, color: str, help_text: str | None = None):
    """Display a metric with a colored badge underneath."""
    st.metric(label, value, help=help_text)
    st.markdown(_color_badge(_interpret_color(color), color), unsafe_allow_html=True)


def _interpret_color(color: str) -> str:
    return {"green": "✔ Bon", "orange": "~ Moyen", "red": "✘ Faible"}.get(color, "")


# ── Seuils pour win rate ──

def winrate_color(wr: float) -> str:
    """Return color based on win rate (0-100 scale)."""
    if wr >= 55:
        return "green"
    elif wr >= 40:
        return "orange"
    return "red"


def winrate_badge(wr: float) -> str:
    color = winrate_color(wr)
    return _color_badge(f"{wr:.0f}%", color)


# ── Seuils pour écart-type ──

def std_color(std_val: float) -> str:
    if std_val <= 0.5:
        return "green"
    elif std_val <= 1.2:
        return "orange"
    return "red"


# ── Seuils pour marge de victoire ──

def margin_color(margin: float) -> str:
    if margin < 0.5:
        return "green"  # serrée = compétitif
    elif margin < 1.5:
        return "orange"
    return "red"


# ── Seuils pour probabilité ──

def proba_color(p: float) -> str:
    """p on 0-100 scale."""
    if p >= 60:
        return "green"
    elif p >= 45:
        return "orange"
    return "red"


def proba_bar_html(p: float) -> str:
    """Return an HTML progress bar colored by probability."""
    color = proba_color(p)
    bar_colors = {"green": "#28a745", "orange": "#ffc107", "red": "#dc3545"}
    return (
        f"<div style='background:#e9ecef;border-radius:4px;height:20px;width:100%;'>"
        f"<div style='background:{bar_colors[color]};width:{min(p, 100):.0f}%;height:100%;"
        f"border-radius:4px;text-align:center;color:white;font-size:12px;line-height:20px;'>"
        f"{p:.0f}%</div></div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Encadrés d'aide par onglet
# ═══════════════════════════════════════════════════════════════════════════════

def show_tab_help(tab_name: str):
    """Display a help expander at the top of a tab."""
    help_content = _TAB_HELP.get(tab_name)
    if help_content:
        with st.expander("💡 Comment lire cet onglet ?", expanded=False):
            st.markdown(help_content)


_TAB_HELP = {
    "dataset": """
**Objectif :** Explorer et filtrer la base de données brute des compétitions de kata.

**Comment l'utiliser :**
- Utilisez les **filtres à gauche** pour cibler un type de compétition, une année, un sexe, etc.
- Les **4 indicateurs en haut** résument les données filtrées (nombre de lignes, athlètes, compétitions, katas distincts).
- Vous pouvez **télécharger** le tableau filtré en CSV pour l'analyser dans Excel.

**Astuce coach :** Utilisez la recherche d'athlète pour retrouver rapidement les passages d'un compétiteur.
""",

    "athlete_focus": """
**Objectif :** Analyser en profondeur le profil et les performances d'un athlète (ou comparer deux athlètes).

**Ce que vous trouverez ici :**
- 📋 **Fiche athlète** : sexe, âge, ranking, nationalité, style
- 📊 **Tour maximal par compétition** : jusqu'où l'athlète est allé à chaque compétition (barres = tours atteints)
- 🥋 **Katas utilisés** : quels katas l'athlète a performé et combien de fois
- 🕸️ **Radar par tour** : note moyenne à chaque phase de compétition → forme grande et régulière = performances stables
- 🕸️ **Radar par kata** : note moyenne par kata → identifie les katas forts et faibles
- 📜 **Historique** : liste des matchs avec adversaires et résultats

**Astuce coach :** Comparez deux athlètes d'un même poids pour préparer une stratégie de confrontation.
""",

    "progression": """
**Objectif :** Suivre l'évolution des notes d'un athlète au fil des compétitions.

**Comment lire le graphique :**
- 📈 **Ligne montante** = l'athlète progresse
- 📉 **Ligne descendante** = l'athlète régresse
- 🔀 **Ligne en dents de scie** = performances irrégulières
- La **ligne pointillée** (moyenne mobile) lisse les variations pour voir la tendance de fond

**Le tableau résumé :**
- **Note moy.** : performance moyenne globale
- **Écart-type** : régularité → **plus c'est bas, plus l'athlète est constant**
- **Tendance ↗/↘** : compare les 3 dernières compétitions aux 3 premières

**Astuce coach :** Comparez 2-3 athlètes du même sexe pour voir qui est en forme montante avant une compétition.
""",

    "score_differential": """
**Objectif :** Comprendre les écarts de score entre les adversaires lors des matchs.

**Les 3 types de victoire :**
- 🟢 **Serrée (<0.5 pts)** : les deux athlètes étaient très proches en niveau
- 🟠 **Nette (0.5-1.5 pts)** : un avantage clair mais pas écrasant
- 🔴 **Dominante (>1.5 pts)** : un athlète nettement supérieur

**Comment lire les indicateurs :**
- **Marge moy.** : écart moyen entre gagnant et perdant → plus c'est bas, plus les niveaux sont homogènes
- **% serrés** : proportion de matchs très disputés → plus c'est haut, plus la compétition est relevée

**Comment lire les graphiques :**
- **Histogramme** : concentration des matchs autour de quelles marges
- **Boxplot par tour** : est-ce que les finales sont plus serrées que les poules ?
- **Barplot par compétition** : certaines compétitions sont-elles plus disputées que d'autres ?

**Astuce coach :** Un % élevé de matchs serrés signifie que chaque détail compte : la préparation du kata, la stratégie, la gestion du stress.
""",

    "continental": """
**Objectif :** Comparer les performances par zone géographique (nation, continent ou région du monde).

**Comment lire les graphiques :**
- **Barplot Win Rate** : quels pays/continents gagnent le plus → les barres hautes = les zones dominantes
- **Gradient de couleur** : la couleur reflète la note moyenne → couleur chaude = bonnes notes
- **Boxplot** : dispersion des notes par zone → une boîte haute = grande variété de niveau dans cette zone
- **Heatmap** : croisement zone × tour → les cases foncées = les zones fortes dans ce tour

**Comment lire un boxplot :**
- La **boîte** contient 50% des notes (du 25ᵉ au 75ᵉ percentile)
- La **ligne au milieu** = médiane (note "du milieu")
- Les **points isolés** = performances exceptionnelles ou contre-performances

**Astuce coach :** Identifiez les nations émergentes (bon win rate mais peu de passages) et les zones à surveiller.
""",

    "tour_advancement": """
**Objectif :** Visualiser combien d'athlètes sont éliminés à chaque phase de la compétition.

**Comment lire le funnel (entonnoir) :**
- En haut = **tous les participants** qui entrent dans la compétition
- Chaque niveau = un tour successif
- En bas = les **finalistes**
- Les pourcentages montrent la part restante par rapport au départ
- Plus le funnel se rétrécit vite, plus la **sélection est sévère**

**Comment lire la courbe de notes :**
- Si la note moyenne **augmente** avec les tours → les meilleurs scores avancent (logique)
- Si elle **stagne** → le niveau est homogène à tous les stades

**Astuce coach :** Utilisez le mode "Par athlète" pour voir le parcours typique d'un compétiteur. Le mode "Par kata" révèle quels katas mènent le plus souvent en finale.
""",

    "kata_diversity": """
**Objectif :** Analyser si les athlètes qui utilisent beaucoup de katas différents réussissent mieux que les spécialistes.

**Les 3 profils :**
- 🎯 **Spécialiste (1-2 katas)** : maîtrise profonde mais prévisible pour les adversaires
- ⚖️ **Modéré (3-4 katas)** : bon compromis entre maîtrise et versatilité
- 🌐 **Polyvalent (5+ katas)** : moins prévisible mais dilue peut-être la préparation

**Comment lire le scatter plot (nuage de points) :**
- Chaque **point = un athlète**
- **Axe X** : nombre de katas différents utilisés
- **Axe Y** : pourcentage de victoires
- **Taille du point** : nombre de passages (plus gros = plus d'expérience)
- **Couleur** : note moyenne (vert = bonne, rouge = faible)
- En haut à droite = diversifié ET gagnant

**Astuce coach :** Regardez si vos athlètes sont dans la zone optimale (diversité suffisante sans être trop dispersés). Le classement en bas permet de trier par win rate, note ou diversité.
""",

    "graphs": """
**Objectif :** Créer des graphiques personnalisés en croisant n'importe quelles variables de la base de données.

**Mode d'emploi :**
1. Choisissez une **variable Y** (la mesure qui vous intéresse)
2. Optionnellement une **variable X** (pour comparer des groupes)
3. L'outil choisit automatiquement le bon type de graphique et le bon test statistique

**5 combinaisons possibles :**
| Y | X | Graphique | Test |
|---|---|-----------|------|
| Numérique | — | Histogramme + boxplot | Normalité (Shapiro / D'Agostino) |
| Numérique | Catégorielle | Boxplot par groupe | ANOVA ou Kruskal-Wallis |
| Catégorielle | — | Barplot des effectifs | — |
| Catégorielle | Catégorielle | Barplot empilé | Chi² ou Fisher |
| Numérique | Numérique | Nuage de points | Corrélation (Pearson / Spearman) |

**Comprendre les résultats statistiques :**
- **p-value < 0.05** → la différence ou relation observée est **significative** (pas due au hasard)
- **p-value > 0.05** → pas de preuve suffisante d'une différence réelle

**Astuce coach :** Essayez Note (Y) × Style (X) pour voir si les Shitoryu ont des notes différentes des Shotokan. Ou Victoire (Y) × Kata (X) pour voir quels katas gagnent le plus souvent.
""",

    "acm": """
**Objectif :** Voir quels katas sont associés à la victoire ou à la défaite sur une carte visuelle.

**Comment lire la carte (en termes simples) :**
- Chaque **point** représente une catégorie (un kata, un tour, ou victoire/défaite)
- Les points **proches** ont tendance à apparaître **ensemble** dans les mêmes combats
- Si un kata est **proche de "Victoire = True"** → c'est un kata qui gagne souvent dans ce contexte
- Si un kata est **proche de "Victoire = False"** → c'est un kata qui perd souvent

**Ce que signifient les axes :**
- Les deux axes résument au mieux les associations entre toutes les variables
- Le **% d'information capturée** indique la qualité du résumé (plus c'est haut, mieux c'est)

**Important :**
- L'ACM montre des **associations**, pas des causalités : un kata "gagnant" ne garantit pas la victoire
- Les résultats dépendent fortement des **filtres** (type de compétition, sexe, style, tours sélectionnés)

**Astuce coach :** Filtrez par sexe et type de compétition, puis repérez les katas proches de "Victoire = True" → ce sont les katas les plus efficaces dans ce contexte.
""",

    "proba_victoire": """
**Objectif :** Estimer la probabilité de victoire d'un athlète A contre un adversaire B selon le kata choisi.

**Comment ça marche (en simple) :**
- Le modèle analyse **22 critères** historiques : notes, ranking, face-à-face, dynamique récente, âge, kata favori, expérience, etc.
- Il calcule une **probabilité de victoire** pour chaque kata que A pourrait jouer
- Un **indice de confiance** (0 à 1) indique la fiabilité : proche de 1 = beaucoup de données, proche de 0 = peu de données → la proba est ramenée vers 50%

**Comment lire les résultats :**
- 🟢 **> 60%** : A est favori avec ce kata
- 🟠 **45-60%** : match ouvert, difficile à prédire
- 🔴 **< 45%** : A est défavorisé avec ce kata

**Les 22 critères en langage coach :**
| Critère technique | Ce que ça veut dire |
|---|---|
| Win rate / H2H | Historique de victoires global et face-à-face |
| Note moyenne / Diff. notes | Niveau de notation et écart entre les deux athlètes |
| Ranking | Classement mondial relatif |
| Tendance de notes | L'athlète progresse ou régresse récemment ? |
| Momentum | Taux de victoire sur les 15 derniers matchs |
| Kata favori | A joue-t-il son kata le plus pratiqué ? |
| Consistance (écart-type) | L'athlète est-il régulier ou imprévisible ? |
| Expérience | Nombre total de matchs joués |
| Diversité kata | L'athlète est-il polyvalent ? |
| B connaît le kata | L'adversaire a-t-il déjà affronté ce kata ? |
| Avantage géographique | A joue-t-il "à domicile" (même continent) ? |
| Tour | Phase de la compétition (poule, finale, etc.) |

**Attention :** Le modèle se base sur l'**historique**. Il ne peut pas prévoir la forme du jour, les blessures, le stress, ni la stratégie surprise d'un adversaire.

**Astuce coach :** Utilisez le "Top 3 katas" pour obtenir une recommandation rapide. Comparez toujours avec votre connaissance de l'athlète et de l'adversaire.
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Guides de lecture par type de graphique
# ═══════════════════════════════════════════════════════════════════════════════

def show_chart_guide(chart_type: str):
    """Show a small expander explaining how to read a chart type."""
    guide = _CHART_GUIDES.get(chart_type)
    if guide:
        with st.expander(f"📖 Comment lire ce graphique ?", expanded=False):
            st.markdown(guide)


_CHART_GUIDES = {
    "boxplot": """
**Boxplot (boîte à moustaches) :**
- La **boîte** contient 50% des valeurs (du 25ᵉ au 75ᵉ percentile)
- La **ligne au milieu** de la boîte = la **médiane** (la valeur "du milieu")
- Les **moustaches** s'étendent aux valeurs extrêmes normales
- Les **points isolés** au-delà = performances exceptionnelles ou contre-performances
- Plus la boîte est **petite**, plus les performances sont **régulières**
""",

    "radar": """
**Diagramme radar (toile d'araignée) :**
- Chaque branche = un critère (tour ou kata)
- Plus la forme est **grande**, plus les notes sont **élevées**
- Une forme **régulière** (cercle) = performances homogènes sur tous les critères
- Une forme **aplatie** d'un côté = faiblesse dans cette catégorie
- En comparant deux athlètes, regardez où les formes se croisent
""",

    "funnel": """
**Funnel (entonnoir) :**
- En haut = **tous les participants**
- Chaque niveau = un tour de compétition successif
- En bas = les **finalistes** (les survivants)
- Les pourcentages montrent la part restante par rapport au départ
- Plus le funnel se rétrécit vite, plus la **sélection est rude**
""",

    "heatmap": """
**Heatmap (carte de chaleur) :**
- Chaque **case** représente une combinaison (ex: pays × tour)
- **Couleurs chaudes** (foncées/vertes) = bonnes valeurs
- **Couleurs froides** (claires/rouges) = valeurs basses
- Cherchez les **lignes ou colonnes** avec beaucoup de cases foncées = zones de force
""",

    "scatter": """
**Nuage de points :**
- Chaque **point** = un individu (athlète, match, etc.)
- La **position** montre les valeurs sur les deux axes
- Cherchez les **tendances** : si les points forment une ligne montante, les deux variables sont liées
- Les **points isolés** en dehors du groupe méritent attention
""",

    "histogram": """
**Histogramme :**
- Les **barres** montrent combien de valeurs tombent dans chaque intervalle
- La barre la plus haute = l'intervalle le plus fréquent
- La **ligne pointillée rouge** = la moyenne
- La **ligne pointillée verte** = la médiane
- Si les barres forment une cloche symétrique = distribution "normale" (classique)
""",

    "bar": """
**Diagramme en barres :**
- La **hauteur** de chaque barre = la valeur mesurée
- Les barres sont triées pour faciliter la comparaison
- Le **gradient de couleur** (si présent) ajoute une information supplémentaire (ex: note moyenne)
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Interprétations dynamiques contextuelles
# ═══════════════════════════════════════════════════════════════════════════════

def interpret_score_differential(m_df) -> str:
    """Generate a natural language interpretation for score differential data."""
    pct_serres = (m_df["Marge"] < 0.5).mean() * 100
    marge_moy = m_df["Marge"].mean()
    n_matchs = len(m_df)

    lines = []
    if pct_serres > 60:
        lines.append(
            f"🟢 **{pct_serres:.0f}% des matchs sont serrés** (<0.5 pts d'écart) "
            f"→ les niveaux sont très homogènes, chaque détail compte."
        )
    elif pct_serres > 40:
        lines.append(
            f"🟠 **{pct_serres:.0f}% des matchs sont serrés** "
            f"→ un mélange de matchs disputés et de victoires nettes."
        )
    else:
        lines.append(
            f"🔴 Seulement **{pct_serres:.0f}% de matchs serrés** "
            f"→ les écarts de niveau sont importants, les victoires sont souvent nettes."
        )

    if marge_moy < 0.5:
        lines.append(f"Marge moyenne de **{marge_moy:.2f} pts** → compétition très disputée.")
    elif marge_moy < 1.0:
        lines.append(f"Marge moyenne de **{marge_moy:.2f} pts** → écarts modérés.")
    else:
        lines.append(f"Marge moyenne de **{marge_moy:.2f} pts** → écarts importants entre les compétiteurs.")

    return "\n\n".join(lines)


def interpret_diversity(ag_df) -> str:
    """Generate interpretation for kata diversity profiles."""
    if ag_df.empty:
        return ""

    lines = []
    for profil in ["Spécialiste (1-2)", "Modéré (3-4)", "Polyvalent (5+)"]:
        sub = ag_df[ag_df["Profil"] == profil]
        if not sub.empty:
            wr = sub["Win_Rate"].mean()
            n = len(sub)
            color = "🟢" if wr >= 55 else "🟠" if wr >= 40 else "🔴"
            lines.append(f"{color} **{profil}** ({n} athlètes) : win rate moyen de **{wr:.1f}%**")

    if len(lines) >= 2:
        lines.append("")
        lines.append("*Comparez les profils pour voir si la spécialisation ou la polyvalence paie dans ce contexte.*")

    return "\n\n".join(lines)


def interpret_progression(summary_list: list[dict]) -> str:
    """Generate interpretation for athlete progression."""
    lines = []
    for s in summary_list:
        nom = s["Athlète"]
        tendance = s.get("Tendance", "—")
        std = s.get("Écart-type", 0)
        note_moy = s.get("Note moy.", 0)

        if tendance == "↗":
            lines.append(f"🟢 **{nom}** est en **phase ascendante** (tendance ↗)")
        elif tendance == "↘":
            lines.append(f"🔴 **{nom}** est en **régression** (tendance ↘)")
        else:
            lines.append(f"🟠 **{nom}** : données insuffisantes pour dégager une tendance")

        if std <= 0.5:
            lines.append(f"   → Très régulier (écart-type: {std:.2f})")
        elif std <= 1.2:
            lines.append(f"   → Régularité moyenne (écart-type: {std:.2f})")
        else:
            lines.append(f"   → Performances irrégulières (écart-type: {std:.2f})")

    return "\n\n".join(lines)


def interpret_tour_advancement(tc_df, filter_label: str) -> str:
    """Generate interpretation for tour advancement funnel."""
    if tc_df.empty:
        return ""

    first = tc_df.iloc[0]["Nb athlètes"]
    last = tc_df.iloc[-1]["Nb athlètes"]

    if first > 0:
        pct_final = last / first * 100
    else:
        pct_final = 0

    lines = []
    if pct_final < 15:
        lines.append(
            f"🔴 Seuls **{pct_final:.0f}%** des participants atteignent le dernier tour "
            f"→ la sélection est très sévère. Passer les poules est déjà un excellent résultat."
        )
    elif pct_final < 30:
        lines.append(
            f"🟠 **{pct_final:.0f}%** des participants atteignent le dernier tour "
            f"→ une sélection modérée."
        )
    else:
        lines.append(
            f"🟢 **{pct_final:.0f}%** des participants atteignent le dernier tour "
            f"→ relativement peu d'élimination."
        )

    # Check if notes increase with tour
    notes = tc_df.dropna(subset=["Note moy."])
    if len(notes) >= 3:
        first_note = notes.iloc[0]["Note moy."]
        last_note = notes.iloc[-1]["Note moy."]
        if last_note > first_note + 0.3:
            lines.append("📈 Les notes **augmentent** avec les tours → les meilleurs passeurs avancent logiquement.")
        elif abs(last_note - first_note) <= 0.3:
            lines.append("➡️ Les notes restent **stables** entre les tours → le niveau est homogène à tous les stades.")

    return "\n\n".join(lines)


def interpret_continental(grp_df, geo_col: str) -> str:
    """Generate interpretation for continental analysis."""
    if grp_df.empty:
        return ""

    top3 = grp_df.nlargest(3, "Win_Rate")
    lines = [f"**Top 3 {geo_col}s par win rate :**"]
    for _, row in top3.iterrows():
        wr = row["Win_Rate"]
        color = "🟢" if wr >= 55 else "🟠" if wr >= 40 else "🔴"
        lines.append(
            f"{color} **{row[geo_col]}** : {wr:.1f}% de victoires "
            f"({int(row['Athlètes'])} athlètes, note moy. {row['Note_Moy']:.2f})"
        )

    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Glossaire
# ═══════════════════════════════════════════════════════════════════════════════

GLOSSAIRE = """
### 📖 Glossaire – Termes utilisés dans l'application

| Terme | Explication |
|-------|-------------|
| **Win rate** | Pourcentage de matchs gagnés. Au-dessus de 55% = très bon niveau international. |
| **Note moyenne** | Moyenne des notes obtenues lors des passages. |
| **Écart-type** | Mesure la régularité : **petit = constant**, **grand = irrégulier**. |
| **Médiane** | La note "du milieu" : la moitié des scores sont au-dessus, l'autre en dessous. Plus fiable que la moyenne quand il y a des valeurs extrêmes. |
| **Marge de victoire** | Différence de points entre le gagnant et le perdant dans un match. |
| **Tendance ↗ / ↘** | Compare les 3 dernières compétitions aux 3 premières pour voir si l'athlète progresse ou régresse. |
| **Moyenne mobile** | Moyenne calculée sur les 3 dernières compétitions. Lisse les variations pour voir la tendance de fond. |
| **Profil Spécialiste** | Athlète qui utilise 1-2 katas → maîtrise profonde mais prévisible. |
| **Profil Modéré** | Athlète qui utilise 3-4 katas → bon compromis maîtrise/versatilité. |
| **Profil Polyvalent** | Athlète qui utilise 5+ katas → moins prévisible mais dilue peut-être la préparation. |
| **ACM** | Carte visuelle qui montre quels katas, tours et résultats "vont ensemble". Les points proches sont liés. |
| **p-value** | Indique si un résultat est dû au hasard. **< 0.05 = significatif** (pas dû au hasard), **> 0.05 = pas de preuve**. |
| **Shrinkage** | Technique qui ramène les probabilités vers 50% quand on a peu de données, pour éviter des prédictions trop extrêmes. |
| **Confiance (0-1)** | Indice de fiabilité de la prédiction. Proche de 1 = beaucoup de données, proche de 0 = peu de données. |
| **Ranking** | Classement mondial de l'athlète (plus le chiffre est bas, meilleur est le classement). |
| **Funnel** | Graphique en entonnoir : montre combien d'athlètes restent à chaque tour de la compétition. |
| **K1 / Premier League** | Le plus haut niveau de compétition de la WKF. |
| **SA / Series A** | Deuxième niveau de compétition de la WKF. |
"""


def show_glossaire():
    """Show the glossary in the sidebar."""
    with st.sidebar:
        with st.expander("📖 Glossaire des termes", expanded=False):
            st.markdown(GLOSSAIRE)
