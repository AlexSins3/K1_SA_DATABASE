"""Aide contextuelle, indicateurs colorés et glossaire pour coachs."""

from __future__ import annotations

import streamlit as st

from utils.lang import t, get_lang


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
    return {"green": t("✔ Bon"), "orange": t("~ Moyen"), "red": t("✘ Faible")}.get(color, "")


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
    if get_lang() == "en":
        help_content = _TAB_HELP_EN.get(tab_name, help_content)
    if help_content:
        with st.expander(t("💡 Comment lire cet onglet ?"), expanded=False):
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

    "tendances": """
**Objectif :** Tester les idées reçues du karaté avec des données réelles et repérer les grandes tendances.

**Ce que vous trouverez ici :**
- 🥋 **Avantage ceinture rouge ?** — Test statistique réel (la ceinture influence-t-elle le résultat ?)
- 📊 **Popularité des katas** — Quels katas montent ou descendent en fréquence d'utilisation
- 🎂 **Performance & âge** — À quel âge est-on au pic ? Le déclin est-il réel ?
- 📋 **Ranking vs réalité** — Le classement WKF reflète-t-il vraiment les performances en compétition ?
- 🌍 **Domination géographique** — Quelles nations/continents dominent réellement
- 🎯 **Spécialiste vs Polyvalent** — Faut-il maîtriser peu ou beaucoup de katas ?

**Astuce coach :** Ce sont des réponses factuelles aux questions que tout le monde se pose au bord du tatami. Utilisez-les pour affiner votre stratégie.
""",

    "match_analysis": """
**Objectif :** Analyser la structure des matchs (écarts de score, parcours d'élimination).

**Deux vues disponibles :**
- ⚔️ **Score différentiel** — Les matchs sont-ils serrés ou déséquilibrés ? Par tour, par athlète, par kata.
- 🏔️ **Avancement par tour (funnel)** — Combien d'athlètes passent chaque tour ? Quel est le "filtre" ?

**Le filtre athlète / kata (nouveau) :**
- Sélectionnez un **athlète** pour voir uniquement ses matchs et sa marge typique
- Sélectionnez un **kata** pour voir les marges quand ce kata est joué

**Les 3 types de victoire :**
- 🟢 **Serrée** (<0.5 pts ou 1 drapeau d'écart)
- 🟠 **Nette** (0.5-1.5 pts ou 2-3 drapeaux)
- 🔴 **Dominante** (>1.5 pts ou 4+ drapeaux)

**Astuce coach :** Filtrez sur votre athlète + ses adversaires potentiels pour voir si les matchs sont typiquement serrés ou non.
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Guides de lecture par type de graphique
# ═══════════════════════════════════════════════════════════════════════════════

def show_chart_guide(chart_type: str):
    """Show a small expander explaining how to read a chart type."""
    guide = _CHART_GUIDES.get(chart_type)
    if get_lang() == "en":
        guide = _CHART_GUIDES_EN.get(chart_type, guide)
    if guide:
        with st.expander(t("📖 Comment lire ce graphique ?"), expanded=False):
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
    if get_lang() == "en":
        if pct_serres > 60:
            lines.append(
                f"🟢 **{pct_serres:.0f}% of matches are close** (<0.5 pts gap) "
                f"→ levels are very homogeneous, every detail counts."
            )
        elif pct_serres > 40:
            lines.append(
                f"🟠 **{pct_serres:.0f}% of matches are close** "
                f"→ a mix of contested matches and clear victories."
            )
        else:
            lines.append(
                f"🔴 Only **{pct_serres:.0f}% of close matches** "
                f"→ skill gaps are significant, victories are often clear."
            )
        if marge_moy < 0.5:
            lines.append(f"Average margin of **{marge_moy:.2f} pts** → highly contested competition.")
        elif marge_moy < 1.0:
            lines.append(f"Average margin of **{marge_moy:.2f} pts** → moderate gaps.")
        else:
            lines.append(f"Average margin of **{marge_moy:.2f} pts** → significant gaps between competitors.")
    else:
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
    if get_lang() == "en":
        for profil in ag_df["Profil"].unique():
            sub = ag_df[ag_df["Profil"] == profil]
            if not sub.empty:
                wr = sub["Win_Rate"].mean()
                n = len(sub)
                color = "🟢" if wr >= 55 else "🟠" if wr >= 40 else "🔴"
                lines.append(f"{color} **{profil}** ({n} athletes): avg. win rate of **{wr:.1f}%**")
        if len(lines) >= 2:
            lines.append("")
            lines.append("*Compare profiles to see whether specialization or versatility pays off in this context.*")
    else:
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
    if get_lang() == "en":
        for s in summary_list:
            nom = s.get("Athlete", s.get("Athlète", s.get(t("Athlète"), "?")))
            tendance = s.get("Trend", s.get("Tendance", s.get(t("Tendance"), "—")))
            std = s.get("Standard deviation", s.get("Écart-type", s.get(t("Écart-type"), 0)))

            if tendance == "↗":
                lines.append(f"🟢 **{nom}** is in an **ascending phase** (trend ↗)")
            elif tendance == "↘":
                lines.append(f"🔴 **{nom}** is **declining** (trend ↘)")
            else:
                lines.append(f"🟠 **{nom}**: insufficient data to determine a trend")

            if std <= 0.5:
                lines.append(f"   → Very consistent (std dev: {std:.2f})")
            elif std <= 1.2:
                lines.append(f"   → Average consistency (std dev: {std:.2f})")
            else:
                lines.append(f"   → Inconsistent performances (std dev: {std:.2f})")
    else:
        for s in summary_list:
            nom = s.get("Athlète", s.get(t("Athlète"), "?"))
            tendance = s.get("Tendance", s.get(t("Tendance"), "—"))
            std = s.get("Écart-type", s.get(t("Écart-type"), 0))

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
    if get_lang() == "en":
        if pct_final < 15:
            lines.append(
                f"🔴 Only **{pct_final:.0f}%** of participants reach the last round "
                f"→ the selection is very severe. Getting past pools is already an excellent result."
            )
        elif pct_final < 30:
            lines.append(
                f"🟠 **{pct_final:.0f}%** of participants reach the last round "
                f"→ moderate selection."
            )
        else:
            lines.append(
                f"🟢 **{pct_final:.0f}%** of participants reach the last round "
                f"→ relatively little elimination."
            )
        notes = tc_df.dropna(subset=["Note moy."])
        if len(notes) >= 3:
            first_note = notes.iloc[0]["Note moy."]
            last_note = notes.iloc[-1]["Note moy."]
            if last_note > first_note + 0.3:
                lines.append("📈 Scores **increase** with rounds → the best performers advance logically.")
            elif abs(last_note - first_note) <= 0.3:
                lines.append("➡️ Scores remain **stable** across rounds → the level is homogeneous at all stages.")
    else:
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
    if get_lang() == "en":
        lines = [f"**Top 3 {geo_col}s by win rate:**"]
        for _, row in top3.iterrows():
            wr = row["Win_Rate"]
            color = "🟢" if wr >= 55 else "🟠" if wr >= 40 else "🔴"
            lines.append(
                f"{color} **{row[geo_col]}**: {wr:.1f}% victories "
                f"({int(row['Athlètes'])} athletes, avg. score {row['Note_Moy']:.2f})"
            )
    else:
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
        with st.expander(t("📖 Glossaire des termes"), expanded=False):
            st.markdown(GLOSSAIRE if get_lang() == "fr" else GLOSSAIRE_EN)


# ═══════════════════════════════════════════════════════════════════════════════
# English translations
# ═══════════════════════════════════════════════════════════════════════════════

GLOSSAIRE_EN = """
### 📖 Glossary – Terms used in this application

| Term | Explanation |
|------|-------------|
| **Win rate** | Percentage of matches won. Above 55% = very good international level. |
| **Average score** | Average scores obtained during performances. |
| **Standard deviation** | Measures consistency: **small = consistent**, **large = inconsistent**. |
| **Median** | The "middle" score: half the scores are above, the other half below. More reliable than the mean when extreme values exist. |
| **Victory margin** | Point difference between winner and loser in a match. |
| **Trend ↗ / ↘** | Compares the last 3 competitions to the first 3 to see if the athlete is improving or declining. |
| **Moving average** | Average calculated over the last 3 competitions. Smooths variations to show the underlying trend. |
| **Specialist profile** | Athlete using 1-2 katas → deep mastery but predictable. |
| **Moderate profile** | Athlete using 3-4 katas → good mastery/versatility balance. |
| **Versatile profile** | Athlete using 5+ katas → less predictable but may dilute preparation. |
| **MCA** | Visual map showing which katas, rounds and results "go together". Close points are related. |
| **p-value** | Indicates if a result is due to chance. **< 0.05 = significant** (not due to chance), **> 0.05 = no evidence**. |
| **Shrinkage** | Technique that pulls probabilities toward 50% when data is scarce, to avoid overly extreme predictions. |
| **Confidence (0-1)** | Prediction reliability index. Close to 1 = lots of data, close to 0 = little data. |
| **Ranking** | World ranking of the athlete (lower number = better ranking). |
| **Funnel** | Funnel chart: shows how many athletes remain at each competition round. |
| **K1 / Premier League** | The highest WKF competition level. |
| **SA / Series A** | Second WKF competition level. |
"""

_TAB_HELP_EN = {
    "dataset": """
**Purpose:** Explore and filter the raw karate competition database.

**How to use:**
- Use the **filters on the left** to target a competition type, year, gender, etc.
- The **4 indicators at the top** summarize the filtered data (row count, athletes, competitions, distinct katas).
- You can **download** the filtered table as CSV for analysis in Excel.

**Coach tip:** Use the athlete search to quickly find a competitor's appearances.
""",

    "athlete_focus": """
**Purpose:** In-depth analysis of an athlete's profile and performances (or compare two athletes).

**What you'll find here:**
- 📋 **Athlete card**: gender, age, ranking, nationality, style
- 📊 **Max round per competition**: how far the athlete went in each competition
- 🥋 **Katas used**: which katas the athlete performed and how often
- 🕸️ **Radar by round**: average score at each competition phase → large and regular shape = stable performances
- 🕸️ **Radar by kata**: average score per kata → identifies strong and weak katas
- 📜 **History**: list of matches with opponents and results

**Coach tip:** Compare two athletes in the same weight category to prepare a confrontation strategy.
""",

    "progression": """
**Purpose:** Track an athlete's score evolution across competitions.

**How to read the chart:**
- 📈 **Rising line** = the athlete is improving
- 📉 **Falling line** = the athlete is declining
- 🔀 **Zigzag line** = inconsistent performances
- The **dashed line** (moving average) smooths variations to show the underlying trend

**Summary table:**
- **Avg. score**: overall average performance
- **Std. dev.**: consistency → **lower = more consistent**
- **Trend ↗/↘**: compares the last 3 competitions to the first 3

**Coach tip:** Compare 2-3 athletes of the same gender to see who is on an upward trend before a competition.
""",

    "score_differential": """
**Purpose:** Understand score gaps between opponents during matches.

**3 types of victory:**
- 🟢 **Close (<0.5 pts)**: both athletes were very close in level
- 🟠 **Clear (0.5-1.5 pts)**: a clear advantage but not overwhelming
- 🔴 **Dominant (>1.5 pts)**: one athlete clearly superior

**How to read the indicators:**
- **Avg. margin**: average gap between winner and loser → lower = more homogeneous levels
- **% close**: proportion of highly contested matches → higher = more competitive

**Coach tip:** A high % of close matches means every detail counts: kata preparation, strategy, stress management.
""",

    "continental": """
**Purpose:** Compare performances by geographic area (nation, continent or world region).

**How to read the charts:**
- **Win Rate bar chart**: which countries/continents win the most → tall bars = dominant areas
- **Color gradient**: reflects the average score → warm color = good scores
- **Boxplot**: score distribution by area → tall box = wide level variety in that area

**Coach tip:** Identify emerging nations (good win rate but few appearances) and areas to watch.
""",

    "tour_advancement": """
**Purpose:** Visualize how many athletes are eliminated at each competition phase.

**How to read the funnel:**
- Top = **all participants** entering the competition
- Each level = a successive round
- Bottom = the **finalists**
- Percentages show the remaining share compared to the start
- The faster the funnel narrows, the **more severe the selection**

**Coach tip:** Use "By athlete" mode to see a competitor's typical path. "By kata" mode reveals which katas most often lead to finals.
""",

    "kata_diversity": """
**Purpose:** Analyze whether athletes using many different katas perform better than specialists.

**3 profiles:**
- 🎯 **Specialist (1-2 katas)**: deep mastery but predictable
- ⚖️ **Moderate (3-4 katas)**: good balance between mastery and versatility
- 🌐 **Versatile (5+ katas)**: less predictable but may dilute preparation

**How to read the scatter plot:**
- Each **dot = an athlete**
- **X axis**: number of different katas used
- **Y axis**: win percentage
- **Dot size**: number of appearances (bigger = more experience)
- **Color**: average score (green = good, red = weak)

**Coach tip:** Check if your athletes are in the optimal zone (sufficient diversity without being too scattered).
""",

    "graphs": """
**Purpose:** Create custom charts by crossing any variables in the database.

**How to use:**
1. Choose a **Y variable** (the measure you're interested in)
2. Optionally an **X variable** (to compare groups)
3. The tool automatically selects the right chart type and statistical test

**Understanding statistical results:**
- **p-value < 0.05** → the observed difference or relationship is **significant** (not due to chance)
- **p-value > 0.05** → no sufficient evidence of a real difference

**Coach tip:** Try Score (Y) × Style (X) to see if Shitoryu have different scores from Shotokan.
""",

    "acm": """
**Purpose:** See which katas are associated with victory or defeat on a visual map.

**How to read the map (in simple terms):**
- Each **dot** represents a category (a kata, a round, or victory/defeat)
- **Close** dots tend to appear **together** in the same matches
- If a kata is **close to "Victory = True"** → it's a kata that often wins in this context
- If a kata is **close to "Victory = False"** → it's a kata that often loses

**What the axes mean:**
- Both axes summarize the associations between all variables as best as possible
- The **% of captured information** indicates the quality of the summary (higher = better)

**Coach tip:** Filter by gender and competition type, then spot katas close to "Victory = True" → these are the most effective katas in this context.
""",

    "proba_victoire": """
**Purpose:** Estimate the win probability of athlete A against opponent B according to the chosen kata.

**How it works (simplified):**
- The model analyzes **22 historical criteria**: scores, ranking, head-to-head, recent momentum, age, favorite kata, experience, etc.
- It calculates a **win probability** for each kata A could perform
- A **confidence index** (0 to 1) indicates reliability: close to 1 = lots of data, close to 0 = little data → probability is pulled toward 50%

**How to read results:**
- 🟢 **> 60%**: A is the favorite with this kata
- 🟠 **45-60%**: open match, hard to predict
- 🔴 **< 45%**: A is at a disadvantage with this kata

**Warning:** The model is based on **history**. It cannot predict current form, injuries, stress, or a surprise strategy from an opponent.

**Coach tip:** Use "Top 3 katas" for a quick recommendation. Always compare with your knowledge of the athlete and opponent.
""",

    "tendances": """
**Purpose:** Test common karate beliefs with real data and spot major trends.

**What you'll find:**
- 🥋 **Red belt advantage?** — Real statistical test (does belt color influence results?)
- 📊 **Kata popularity** — Which katas are rising or falling in usage frequency
- 🎂 **Performance & age** — At what age is peak performance? Is the decline real?
- 📋 **Ranking vs reality** — Does the WKF ranking really reflect competition performance?
- 🌍 **Geographic dominance** — Which nations/continents truly dominate
- 🎯 **Specialist vs Polyvalent** — Is it better to master few or many katas?

**Coach tip:** These are factual answers to the questions everyone asks at the side of the tatami. Use them to refine your strategy.
""",

    "match_analysis": """
**Purpose:** Analyze match structure (score gaps, elimination paths).

**Two views available:**
- ⚔️ **Score differential** — Are matches tight or unbalanced? By round, athlete, or kata.
- 🏔️ **Round advancement (funnel)** — How many athletes pass each round? What's the filter?

**The athlete / kata filter (new):**
- Select an **athlete** to see only their matches and typical margin
- Select a **kata** to see margins when that kata is performed

**The 3 victory types:**
- 🟢 **Close** (<0.5 pts or 1 flag gap)
- 🟠 **Clear** (0.5-1.5 pts or 2-3 flags)
- 🔴 **Dominant** (>1.5 pts or 4+ flags)

**Coach tip:** Filter on your athlete + their potential opponents to see if matches are typically close or not.
""",
}

_CHART_GUIDES_EN = {
    "boxplot": """
**Boxplot (box and whiskers):**
- The **box** contains 50% of the values (from 25th to 75th percentile)
- The **line in the middle** = the **median** (the "middle" value)
- The **whiskers** extend to normal extreme values
- **Isolated dots** beyond = exceptional performances or counter-performances
- The **smaller** the box, the **more consistent** the performances
""",

    "radar": """
**Radar chart (spider web):**
- Each branch = a criterion (round or kata)
- The **larger** the shape, the **higher** the scores
- A **regular** shape (circle) = homogeneous performances across all criteria
- A **flattened** shape on one side = weakness in that category
- When comparing two athletes, look where the shapes cross
""",

    "funnel": """
**Funnel:**
- Top = **all participants**
- Each level = a successive competition round
- Bottom = the **finalists** (survivors)
- Percentages show the remaining share compared to the start
- The faster the funnel narrows, the **tougher the selection**
""",

    "heatmap": """
**Heatmap:**
- Each **cell** represents a combination (e.g. country × round)
- **Warm colors** (dark/green) = good values
- **Cool colors** (light/red) = low values
- Look for **rows or columns** with many dark cells = areas of strength
""",

    "scatter": """
**Scatter plot:**
- Each **dot** = an individual (athlete, match, etc.)
- **Position** shows values on both axes
- Look for **trends**: if dots form a rising line, the two variables are related
- **Isolated dots** outside the group deserve attention
""",

    "histogram": """
**Histogram:**
- **Bars** show how many values fall in each interval
- The tallest bar = the most frequent interval
- The **red dashed line** = the mean
- The **green dashed line** = the median
- If bars form a symmetric bell = "normal" distribution (classic)
""",

    "bar": """
**Bar chart:**
- The **height** of each bar = the measured value
- Bars are sorted for easy comparison
- The **color gradient** (if present) adds extra information (e.g. average score)
""",
}
