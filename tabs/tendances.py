# tabs/tendances.py — Tendances & Réalités : idées reçues, données sous-exploitées
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help, _color_badge
from utils.display import format_display_df
from utils.lang import t, get_lang


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _significance_badge(p: float) -> str:
    if p < 0.01:
        return _color_badge(t("Très significatif (p<0.01)"), "green")
    elif p < 0.05:
        return _color_badge(t("Significatif (p<0.05)"), "green")
    elif p < 0.10:
        return _color_badge(t("Tendance (p<0.10)"), "orange")
    return _color_badge(t("Non significatif"), "red")


def _compute_belt_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate par ceinture (R vs B)."""
    d = df[df["Ceinture"].isin(["R", "B"])].copy()
    d["Win"] = d["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)
    stats = d.groupby("Ceinture", observed=True).agg(
        Victoires=("Win", "sum"),
        Total=("Win", "count"),
    ).reset_index()
    stats["Win_Rate"] = (stats["Victoires"] / stats["Total"] * 100).round(1)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main tab
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment
def show_tendances_tab(data: pd.DataFrame) -> None:
    st.header(t("Tendances & Réalités"))
    show_tab_help("tendances")

    df = data.copy()
    for col in ["Note", "Year", "Age", "Ranking", "Drapeau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Victoire" in df.columns:
        df["Win"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()
        st.markdown(t("### 🎛️ Filtres"))

        type_opts = [t("Tous"), "K1", "SA"]
        sel_type = st.radio(t("Type compétition"), type_opts, key="tend_type")
        d = df.copy()
        if sel_type != t("Tous"):
            d = d[d["Type_Compet"] == sel_type]

        sexes = sorted(d["Sexe"].dropna().unique().tolist())
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexes, key="tend_sexe")
        if sel_sexe != t("Tous"):
            d = d[d["Sexe"] == sel_sexe]

        years_avail = sorted(d["Year"].dropna().unique().tolist())
        sel_years = st.multiselect(
            t("Année(s)"), [int(y) for y in years_avail],
            default=[int(y) for y in years_avail], key="tend_years"
        )
        if sel_years:
            d = d[d["Year"].isin([float(y) for y in sel_years])]

        filter_panel_close()

    with content_col:
        if d.empty:
            st.info(t("Aucune donnée dans ce périmètre."))
            return

        # ══════════════════════════════════════════════════════════════════════
        # Section 1 : Avantage ceinture bleue ?
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("🥋 " + t("Idée reçue : la ceinture bleue a un avantage"))

        belt_stats = _compute_belt_stats(d)
        if len(belt_stats) == 2:
            wr_r = belt_stats[belt_stats["Ceinture"] == "R"]["Win_Rate"].values[0]
            wr_b = belt_stats[belt_stats["Ceinture"] == "B"]["Win_Rate"].values[0]
            n_total = belt_stats["Total"].sum() // 2  # pairs

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 " + t("Win Rate Rouge"), f"{wr_r:.1f}%")
            col2.metric("🔵 " + t("Win Rate Bleu"), f"{wr_b:.1f}%")
            col3.metric(t("Matchs analysés"), f"{n_total}")

            # Test binomial
            from scipy.stats import binomtest
            n_b = int(belt_stats[belt_stats["Ceinture"] == "B"]["Total"].values[0])
            wins_b = int(belt_stats[belt_stats["Ceinture"] == "B"]["Victoires"].values[0])
            test = binomtest(wins_b, n_b, 0.5, alternative='two-sided')

            if wr_b > wr_r + 2:
                conclusion = t("La ceinture bleue a un avantage")
            elif wr_r > wr_b + 2:
                conclusion = t("La ceinture rouge a un avantage")
            else:
                conclusion = t("Aucun avantage significatif détecté")

            st.markdown(
                f"**{t('Verdict')}** : {conclusion} "
                f"({t('écart')} = {abs(wr_b - wr_r):.1f} pts, {_significance_badge(test.pvalue)})",
                unsafe_allow_html=True,
            )

            # Contexte important
            _ctx_k1 = t("En K1, la tête de série de chaque poule passe systématiquement en bleue. "
                        "En SA, seules 4 à 8 têtes de série sur 96-128 participants portent le bleu en 1er tour.")
            _ctx_biais = t("Ce biais de sélection peut expliquer une partie de l'avantage observé.")
            st.info(
                f"⚠️ **{t('Contexte')}** : {_ctx_k1} {_ctx_biais}"
            )

            # Évolution par année
            d_belt = d[d["Ceinture"].isin(["R", "B"])].copy()
            belt_year = d_belt.groupby(["Year", "Ceinture"], observed=True)["Win"].mean().reset_index()
            belt_year["Win_Rate"] = (belt_year["Win"] * 100).round(1)
            if len(belt_year["Year"].unique()) > 1:
                _color_map = {"R": "#dc3545", "B": "#0d6efd"}
                fig_belt = go.Figure()
                for ceinture in ["R", "B"]:
                    df_c = belt_year[belt_year["Ceinture"] == ceinture]
                    if not df_c.empty:
                        pairs = sorted(zip(df_c["Year"].tolist(), df_c["Win_Rate"].tolist()), key=lambda p: p[0])
                        fig_belt.add_trace(go.Scatter(
                            x=[p[0] for p in pairs], y=[p[1] for p in pairs],
                            mode="lines+markers", name=ceinture,
                            line=dict(color=_color_map[ceinture]),
                        ))
                fig_belt.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
                fig_belt.update_layout(
                    title=t("Évolution avantage ceinture par année"),
                    xaxis_title=t("Année"), yaxis_title="Win rate (%)",
                    xaxis={"dtick": 1, "tickformat": "d"},
                )
                st.plotly_chart(fig_belt, width="stretch", key="tend_belt_year")

        # ══════════════════════════════════════════════════════════════════════
        # Section 2 : Popularité des katas dans le temps
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 " + t("Évolution de la popularité des katas"))

        # Filtre style local
        styles_avail = sorted(d["Style"].dropna().unique().tolist()) if "Style" in d.columns else []
        sel_style = st.selectbox(
            t("Style"), [t("Tous")] + styles_avail, key="tend_kata_style"
        )
        d_kata = d.copy()
        if sel_style != t("Tous") and "Style" in d_kata.columns:
            d_kata = d_kata[d_kata["Style"] == sel_style]

        # Exclure les katas vides / NaN
        d_kata = d_kata[d_kata["Kata"].notna() & (d_kata["Kata"].astype(str).str.strip() != "")]

        # Mode affichage
        view_mode = st.radio(
            t("Affichage"), [t("Part relative (%)"), t("Win rate (%)")],
            horizontal=True, key="tend_kata_view"
        )

        kata_year = d_kata.groupby(["Year", "Kata"], observed=True).agg(
            Passages=("Win", "count"),
            Win_Rate=("Win", "mean"),
        ).reset_index()
        total_year = d_kata.groupby("Year", observed=True).size().reset_index(name="Total")
        kata_year = kata_year.merge(total_year, on="Year")
        kata_year["Part (%)"] = (kata_year["Passages"] / kata_year["Total"] * 100).round(1)
        kata_year["Win_Rate"] = (kata_year["Win_Rate"] * 100).round(1)

        # Top katas (au moins 2% de passages sur l'ensemble)
        top_katas_all = d_kata["Kata"].value_counts(normalize=True)
        top_katas = top_katas_all[top_katas_all >= 0.02].index.tolist()

        if top_katas and len(kata_year["Year"].unique()) > 1:
            kata_year_top = kata_year[kata_year["Kata"].isin(top_katas)]

            y_col = "Part (%)" if view_mode == t("Part relative (%)") else "Win_Rate"
            title = (t("Part relative des katas les plus joués par année")
                     if view_mode == t("Part relative (%)")
                     else t("Win rate des katas les plus joués par année"))
            y_label = (t("Part des passages (%)")
                       if view_mode == t("Part relative (%)")
                       else "Win rate (%)")

            fig_pop = go.Figure()
            for kata in top_katas:
                df_k = kata_year_top[kata_year_top["Kata"] == kata]
                if not df_k.empty:
                    pairs = sorted(zip(df_k["Year"].tolist(), df_k[y_col].tolist()), key=lambda p: p[0])
                    fig_pop.add_trace(go.Scatter(
                        x=[p[0] for p in pairs], y=[p[1] for p in pairs],
                        mode="lines+markers", name=kata,
                    ))
            if view_mode != t("Part relative (%)"):
                fig_pop.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
            fig_pop.update_layout(
                title=title,
                xaxis_title=t("Année"), yaxis_title=y_label,
                xaxis={"dtick": 1, "tickformat": "d"},
            )
            st.plotly_chart(fig_pop, width="stretch", key="tend_kata_pop")

            # Katas en hausse / en baisse (dynamique entre deux dernières années)
            if len(kata_year["Year"].unique()) >= 2:
                years_sorted = sorted(kata_year["Year"].unique())
                last_year = years_sorted[-1]
                prev_year = years_sorted[-2]

                current_part = kata_year[kata_year["Year"] == last_year].set_index("Kata")["Part (%)"]
                previous_part = kata_year[kata_year["Year"] == prev_year].set_index("Kata")["Part (%)"]
                diff = (current_part - previous_part).dropna().sort_values()

                current_wr = kata_year[kata_year["Year"] == last_year].set_index("Kata")["Win_Rate"]
                previous_wr = kata_year[kata_year["Year"] == prev_year].set_index("Kata")["Win_Rate"]
                diff_wr = (current_wr - previous_wr).dropna().sort_values()

                col_up, col_down = st.columns(2)
                with col_up:
                    rising = diff[diff > 0.3].sort_values(ascending=False).head(5)
                    if not rising.empty:
                        st.markdown(f"**📈 {t('Katas en hausse')}** ({int(prev_year)}→{int(last_year)})")
                        for kata, delta in rising.items():
                            wr_val = current_wr.get(kata, 0)
                            wr_delta = diff_wr.get(kata, 0)
                            wr_icon = "🟢" if wr_delta > 2 else ("🔴" if wr_delta < -2 else "⚪")
                            st.markdown(f"- **{kata}** : +{delta:.1f} pts (WR: {wr_val:.0f}%, {wr_icon} {wr_delta:+.1f})")
                    else:
                        st.markdown(f"*{t('Pas de hausse marquée')}*")
                with col_down:
                    falling = diff[diff < -0.3].sort_values().head(5)
                    if not falling.empty:
                        st.markdown(f"**📉 {t('Katas en baisse')}** ({int(prev_year)}→{int(last_year)})")
                        for kata, delta in falling.items():
                            wr_val = current_wr.get(kata, 0)
                            wr_delta = diff_wr.get(kata, 0)
                            wr_icon = "🟢" if wr_delta > 2 else ("🔴" if wr_delta < -2 else "⚪")
                            st.markdown(f"- **{kata}** : {delta:.1f} pts (WR: {wr_val:.0f}%, {wr_icon} {wr_delta:+.1f})")
                    else:
                        st.markdown(f"*{t('Pas de baisse marquée')}*")

        # ══════════════════════════════════════════════════════════════════════
        # Section 3 : Performance & âge
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🎂 " + t("Performance selon l'âge"))

        d_age = d.dropna(subset=["Age"]).copy()
        if not d_age.empty:
            d_age["Tranche_Age"] = pd.cut(
                d_age["Age"], bins=[14, 18, 22, 26, 30, 35, 50],
                labels=["15-18", "19-22", "23-26", "27-30", "31-35", "36+"],
            )
            age_stats = d_age.groupby("Tranche_Age", observed=True).agg(
                Win_Rate=("Win", "mean"),
                Athlètes=("Nom", "nunique"),
                Passages=("Win", "count"),
            ).reset_index()
            age_stats["Win_Rate"] = (age_stats["Win_Rate"] * 100).round(1)

            fig_age = px.bar(
                age_stats, x="Tranche_Age", y="Win_Rate",
                color="Win_Rate", color_continuous_scale="RdYlGn",
                text="Win_Rate",
                hover_data=["Athlètes", "Passages"],
                title=t("Win rate par tranche d'âge"),
                labels={"Tranche_Age": t("Tranche d'âge"), "Win_Rate": "Win rate (%)", "Athlètes": t("Athlètes"), "Passages": t("Passages")},
            )
            fig_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_age.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_age, width="stretch", key="tend_age_bar")

            # Note moyenne par âge (si disponible)
            d_age_notes = d_age.dropna(subset=["Note"])
            if not d_age_notes.empty:
                age_note = d_age_notes.groupby("Tranche_Age", observed=True)["Note"].mean().reset_index()
                age_note["Note"] = age_note["Note"].round(2)
                fig_age_note = px.line(
                    age_note, x="Tranche_Age", y="Note", markers=True,
                    title=t("Note moyenne par tranche d'âge"),
                    labels={"Note": t("Note moyenne"), "Tranche_Age": t("Tranche d'âge")},
                )
                st.plotly_chart(fig_age_note, width="stretch", key="tend_age_note")

            # Peak age
            best_bracket = age_stats.loc[age_stats["Win_Rate"].idxmax()]
            st.markdown(
                f"**{t('Pic de performance')}** : {t('tranche')} **{best_bracket['Tranche_Age']}** "
                f"(win rate {best_bracket['Win_Rate']:.1f}%, {int(best_bracket['Athlètes'])} {t('athlètes')})"
            )

        # ══════════════════════════════════════════════════════════════════════
        # Section 4 : Ranking vs performance réelle
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📋 " + t("Le ranking reflète-t-il la réalité ?"))

        d_rank = d.dropna(subset=["Ranking"]).copy()
        if not d_rank.empty and len(d_rank) > 20:
            # Granularité réglable
            granularity = st.radio(
                t("Granularité"), [t("Standard (6)"), t("Détaillé (8)")],
                horizontal=True, key="tend_rank_granularity"
            )
            if granularity == t("Détaillé (8)"):
                bins = [0, 10, 20, 30, 50, 100, 200, 400, 9999]
                labels = ["Top 10", "11-20", "21-30", "31-50", "51-100", "101-200", "201-400", "400+"]
            else:
                bins = [0, 20, 50, 100, 200, 400, 9999]
                labels = ["Top 20", "21-50", "51-100", "101-200", "201-400", "400+"]

            d_rank["Tranche_Ranking"] = pd.cut(
                d_rank["Ranking"], bins=bins, labels=labels,
            )
            rank_stats = d_rank.groupby("Tranche_Ranking", observed=True).agg(
                Win_Rate=("Win", "mean"),
                Athlètes=("Nom", "nunique"),
            ).reset_index()
            rank_stats["Win_Rate"] = (rank_stats["Win_Rate"] * 100).round(1)

            fig_rank = px.bar(
                rank_stats, x="Tranche_Ranking", y="Win_Rate",
                color="Win_Rate", color_continuous_scale="RdYlGn",
                text="Win_Rate",
                hover_data=["Athlètes"],
                title=t("Win rate par tranche de ranking WKF"),
                labels={"Tranche_Ranking": "Ranking", "Win_Rate": "Win rate (%)", "Athlètes": t("Athlètes")},
            )
            fig_rank.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_rank.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_rank, width="stretch", key="tend_rank_bar")

            # Interprétation dynamique du graphe ranking
            if len(rank_stats) >= 2:
                top_bracket = rank_stats.iloc[0]
                bottom_bracket = rank_stats.iloc[-1]
                spread = top_bracket["Win_Rate"] - bottom_bracket["Win_Rate"]
                _interp_rank = (
                    f"**{t('Interprétation')}** : **{top_bracket['Tranche_Ranking']}** "
                    f"{t('affichent un win rate de')} {top_bracket['Win_Rate']:.1f}% vs "
                    f"{bottom_bracket['Win_Rate']:.1f}% {t('pour les')} **{bottom_bracket['Tranche_Ranking']}** "
                    f"({t('écart')} {spread:.0f} pts). "
                )
                if spread > 20:
                    _interp_rank += t("Le ranking est un bon prédicteur du résultat.")
                elif spread > 10:
                    _interp_rank += t("Le ranking est un indicateur modéré.")
                else:
                    _interp_rank += t("Le ranking est un indicateur faible sur ce périmètre.")
                st.markdown(_interp_rank)

            # Corrélation ranking / win rate athlète
            ath_perf = d_rank.groupby("Nom", observed=True).agg(
                Ranking_Moy=("Ranking", "mean"),
                Win_Rate=("Win", "mean"),
                Passages=("Win", "count"),
            ).reset_index()
            ath_perf = ath_perf[ath_perf["Passages"] >= 5]
            ath_perf["Win_Rate"] = (ath_perf["Win_Rate"] * 100).round(1)

            if len(ath_perf) > 10:
                from scipy.stats import spearmanr
                corr, p_val = spearmanr(ath_perf["Ranking_Moy"], ath_perf["Win_Rate"])

                fig_corr = px.scatter(
                    ath_perf, x="Ranking_Moy", y="Win_Rate",
                    hover_data=["Nom", "Passages"],
                    title=t("Ranking moyen vs Win rate (athlètes avec 5+ passages)"),
                    labels={"Ranking_Moy": t("Ranking moyen"), "Win_Rate": "Win rate (%)", "Nom": t("Nom"), "Passages": t("Passages")},
                )
                st.plotly_chart(fig_corr, width="stretch", key="tend_rank_scatter")
                st.markdown(
                    f"**{t('Corrélation')}** : ρ = {corr:.3f} (Spearman), {_significance_badge(p_val)}. "
                    f"{'→ ' + t('Le ranking est un bon indicateur du niveau réel.') if abs(corr) > 0.3 else '→ ' + t('Le ranking est un indicateur imparfait.')}",
                    unsafe_allow_html=True,
                )

        # ══════════════════════════════════════════════════════════════════════
        # Section 5 : Domination géographique (condensé)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🌍 " + t("Domination géographique"))

        if "Continent" in d.columns:
            cont_stats = d.groupby("Continent").agg(
                Win_Rate=("Win", "mean"),
                Athlètes=("Nom", "nunique"),
                Passages=("Win", "count"),
            ).reset_index()
            cont_stats["Win_Rate"] = (cont_stats["Win_Rate"] * 100).round(1)
            cont_stats = cont_stats[cont_stats["Athlètes"] >= 3].sort_values("Win_Rate", ascending=False)

            if not cont_stats.empty:
                fig_cont = px.bar(
                    cont_stats, x="Continent", y="Win_Rate",
                    color="Win_Rate", color_continuous_scale="RdYlGn",
                    text="Win_Rate", hover_data=["Athlètes", "Passages"],
                    title=t("Win rate par continent"),
                    labels={"Win_Rate": "Win rate (%)", "Athlètes": t("Athlètes"), "Passages": t("Passages")},
                )
                fig_cont.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_cont.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
                st.plotly_chart(fig_cont, width="stretch", key="tend_cont")

            # Top nations (pondéré par nombre d'athlètes)
            nation_stats = d.groupby("Nation").agg(
                Win_Rate=("Win", "mean"),
                Athlètes=("Nom", "nunique"),
                Passages=("Win", "count"),
            ).reset_index()
            nation_stats["Win_Rate"] = (nation_stats["Win_Rate"] * 100).round(1)
            # Filtre: min 3 athlètes ET 15 passages pour être représentatif
            nation_stats = nation_stats[(nation_stats["Athlètes"] >= 3) & (nation_stats["Passages"] >= 15)]
            # Score pondéré: WR ajusté par log(nb athlètes) pour ne pas pénaliser les grosses nations
            nation_stats["Score"] = (nation_stats["Win_Rate"] * np.log1p(nation_stats["Athlètes"])).round(1)
            nation_stats = nation_stats.sort_values("Score", ascending=False)

            if not nation_stats.empty:
                st.markdown(f"**Top 10 {t('nations')}** ({t('pondéré par nombre d athlètes')}, {t('min 3 athlètes')}) :")
                display_cols = ["Nation", "Win_Rate", "Athlètes", "Passages"]
                st.dataframe(
                    nation_stats.head(10)[display_cols].rename(columns={
                        "Win_Rate": "Win Rate (%)",
                        "Athlètes": t("Athlètes"),
                        "Passages": t("Passages"),
                    }),
                    width="stretch", hide_index=True,
                )

        # ══════════════════════════════════════════════════════════════════════
        # Section 6 : Spécialiste vs Polyvalent (condensé)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🎯 " + t("Spécialiste vs Polyvalent : qui gagne ?"))

        # Filtre: uniquement athlètes ayant passé au moins 1 tour dans 2+ compétitions
        # (exclut ceux éliminés systématiquement au 1er tour)
        d_qual = d.copy()
        # Identifier les tours "avancés" (après T1/Pool_1)
        _first_tours = {"T1", "Pool_1"}
        d_qual["_is_advanced"] = ~d_qual["N_Tour"].astype(str).isin(_first_tours)
        # Compter les compétitions où l'athlète a atteint un tour avancé
        adv_compets = d_qual[d_qual["_is_advanced"]].groupby("Nom")["Competition"].nunique().reset_index()
        adv_compets.columns = ["Nom", "Compets_Avancees"]
        eligible_athletes = adv_compets[adv_compets["Compets_Avancees"] >= 2]["Nom"].tolist()

        athlete_div = d[d["Nom"].isin(eligible_athletes)].groupby("Nom").agg(
            Katas_Distincts=("Kata", "nunique"),
            Win_Rate=("Win", "mean"),
            Passages=("Win", "count"),
        ).reset_index()
        athlete_div["Win_Rate"] = (athlete_div["Win_Rate"] * 100).round(1)

        if not athlete_div.empty:
            athlete_div["Profil"] = pd.cut(
                athlete_div["Katas_Distincts"],
                bins=[0, 2, 4, 100],
                labels=[t("Spécialiste (1-2)"), t("Modéré (3-4)"), t("Polyvalent (5+)")],
            )

            profil_stats = athlete_div.groupby("Profil", observed=True).agg(
                Win_Rate_Moy=("Win_Rate", "mean"),
                Athlètes=("Nom", "count"),
            ).reset_index()
            profil_stats["Win_Rate_Moy"] = profil_stats["Win_Rate_Moy"].round(1)

            col1, col2, col3 = st.columns(3)
            for i, (_, row) in enumerate(profil_stats.iterrows()):
                col = [col1, col2, col3][i] if i < 3 else col3
                col.metric(str(row["Profil"]), f"{row['Win_Rate_Moy']:.1f}%", help=f"{int(row['Athlètes'])} {t('athlètes')}")

            fig_div = px.scatter(
                athlete_div, x="Katas_Distincts", y="Win_Rate",
                size="Passages", hover_data=["Nom"],
                title=t("Diversité kata vs Win rate"),
                labels={"Katas_Distincts": t("Katas distincts"), "Win_Rate": "Win rate (%)", "Nom": t("Nom"), "Passages": t("Passages")},
            )
            st.plotly_chart(fig_div, width="stretch", key="tend_div_scatter")

            # Verdict
            if len(profil_stats) >= 2:
                best = profil_stats.loc[profil_stats["Win_Rate_Moy"].idxmax()]
                st.markdown(
                    f"**{t('Verdict')}** : **{best['Profil']}** {t('ont le meilleur win rate moyen')} "
                    f"({best['Win_Rate_Moy']:.1f}%) {t('sur ce périmètre')}."
                )
