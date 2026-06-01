# tabs/kata_comparison.py — Onglet : Comparaison de katas
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.display import fmt_tour
from utils.lang import t


_TOUR_ORDER = [
    "T1", "T2", "T3",
    "Pool_1", "Pool_2", "Pool_3",
    "PW1", "PW2", "PW3",
    "RP1", "RP2", "RP3", "RP4",
    "R1", "R2",
    "Bronze", "Final",
]


def _tour_rank(tour) -> int:
    try:
        return _TOUR_ORDER.index(str(tour))
    except (ValueError, TypeError):
        return 999


@st.fragment
def show_kata_comparison_tab(data: pd.DataFrame) -> None:
    st.header(t("Comparaison de katas"))
    st.caption(t("Comparez deux ou plusieurs katas pour aider au choix stratégique selon le tour, le contexte et l'historique."))

    df = data.copy()
    for col in ["Note", "Year", "Drapeau", "Ranking"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Victoire" in df.columns:
        df["Win"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    # Exclure katas vides
    df = df[df["Kata"].notna() & (df["Kata"].astype(str).str.strip() != "")]

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        # Type compétition
        type_opts = [t("Tous"), "K1", "SA"]
        sel_type = st.radio(t("Type compétition"), type_opts, key="kcomp_type")
        d = df.copy()
        if sel_type != t("Tous"):
            d = d[d["Type_Compet"] == sel_type]

        # Sexe
        sexes = sorted(d["Sexe"].dropna().unique().tolist())
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexes, key="kcomp_sexe")
        if sel_sexe != t("Tous"):
            d = d[d["Sexe"] == sel_sexe]

        # Style
        styles = sorted(d["Style"].dropna().unique().tolist()) if "Style" in d.columns else []
        sel_style = st.selectbox(t("Style"), [t("Tous")] + styles, key="kcomp_style")
        if sel_style != t("Tous") and "Style" in d.columns:
            d = d[d["Style"] == sel_style]

        # Années
        years_avail = sorted(d["Year"].dropna().unique().tolist())
        years_opts = [int(y) for y in years_avail]
        # Garder les années déjà sélectionnées qui sont encore valides
        if "kcomp_years" in st.session_state:
            st.session_state["kcomp_years"] = [
                y for y in st.session_state["kcomp_years"] if y in years_opts
            ]
        sel_years = st.multiselect(
            t("Année(s)"), years_opts,
            default=years_opts, key="kcomp_years"
        )
        if sel_years:
            d = d[d["Year"].isin([float(y) for y in sel_years])]

        # Sélection des katas à comparer
        katas_available = sorted(d["Kata"].dropna().unique().tolist())
        # Garder les katas déjà sélectionnés qui sont encore disponibles
        if "kcomp_katas" in st.session_state:
            st.session_state["kcomp_katas"] = [
                k for k in st.session_state["kcomp_katas"] if k in katas_available
            ]
        sel_katas = st.multiselect(
            t("Katas à comparer"),
            katas_available,
            max_selections=6,
            key="kcomp_katas",
        )

        filter_panel_close()

    with content_col:
        if not sel_katas or len(sel_katas) < 2:
            st.info(t("Sélectionnez au moins 2 katas dans le panneau de filtres pour lancer la comparaison."))
            return

        sub = d[d["Kata"].isin(sel_katas)].copy()
        if sub.empty:
            st.warning(t("Aucune donnée pour ces katas."))
            return

        # ═══════════════════════════════════════════════════════════════
        # KPIs globaux par kata
        # ═══════════════════════════════════════════════════════════════
        st.subheader(t("Vue d'ensemble"))

        stats = sub.groupby("Kata", observed=True).agg(
            Passages=("Win", "count"),
            Victoires=("Win", "sum"),
            Win_Rate=("Win", "mean"),
            Athlètes=("Nom", "nunique"),
            Note_Moy=("Note", "mean"),
            Note_Max=("Note", "max"),
            Note_Std=("Note", "std"),
        ).reset_index()
        stats["Win_Rate"] = (stats["Win_Rate"] * 100).round(1)
        stats["Note_Moy"] = stats["Note_Moy"].round(2)
        stats["Note_Max"] = stats["Note_Max"].round(2)
        stats["Note_Std"] = stats["Note_Std"].round(2)

        # Ordre d'affichage = ordre de sélection
        stats = stats.set_index("Kata").reindex(sel_katas).reset_index()

        # Metrics row
        cols = st.columns(len(sel_katas))
        for i, (_, row) in enumerate(stats.iterrows()):
            with cols[i]:
                st.markdown(f"**{row['Kata']}**")
                st.metric(t("Win Rate"), f"{row['Win_Rate']:.1f}%")
                st.metric(t("Passages"), int(row["Passages"]))
                st.metric(t("Note moy."), f"{row['Note_Moy']:.2f}" if pd.notna(row["Note_Moy"]) else "—")
                st.caption(f"{int(row['Athlètes'])} {t('athlètes')}")

        # ═══════════════════════════════════════════════════════════════
        # Comparaison par tour
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader(t("Win rate par tour"))

        tours_present = sorted(sub["N_Tour"].dropna().astype(str).unique().tolist(), key=_tour_rank)
        tour_stats = sub.groupby(["Kata", "N_Tour"], observed=True).agg(
            Win_Rate=("Win", "mean"),
            Passages=("Win", "count"),
        ).reset_index()
        tour_stats["Win_Rate"] = (tour_stats["Win_Rate"] * 100).round(1)
        tour_stats["Tour_Display"] = tour_stats["N_Tour"].apply(fmt_tour)
        tour_order_display = [fmt_tour(t_val) for t_val in tours_present]

        fig_tour = px.bar(
            tour_stats, x="Tour_Display", y="Win_Rate", color="Kata",
            barmode="group", text="Win_Rate",
            hover_data=["Passages"],
            title=t("Win rate par kata et par tour"),
            labels={"Tour_Display": t("Tour"), "Win_Rate": "Win rate (%)"},
            category_orders={"Tour_Display": tour_order_display, "Kata": sel_katas},
        )
        fig_tour.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_tour.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig_tour.update_layout(height=450)
        st.plotly_chart(fig_tour, width="stretch", key="kcomp_tour_wr")

        # Interprétation
        # Identifier le meilleur kata par tour
        best_by_tour = tour_stats.loc[tour_stats.groupby("N_Tour", observed=True)["Win_Rate"].idxmax()]
        if len(best_by_tour) > 1:
            _interp_parts = []
            for _, r in best_by_tour.iterrows():
                if r["Passages"] >= 3:
                    _interp_parts.append(f"**{fmt_tour(r['N_Tour'])}** → {r['Kata']} ({r['Win_Rate']:.0f}%)")
            if _interp_parts:
                st.markdown(f"📊 **{t('Meilleur kata par tour')}** : " + " | ".join(_interp_parts[:6]))

        # ═══════════════════════════════════════════════════════════════
        # Notes moyennes par kata
        # ═══════════════════════════════════════════════════════════════
        sub_notes = sub[sub["Note"].notna()]
        if not sub_notes.empty:
            st.markdown("---")
            st.subheader(t("Distribution des notes"))

            fig_box = px.box(
                sub_notes, x="Kata", y="Note", color="Kata",
                title=t("Distribution des notes par kata"),
                category_orders={"Kata": sel_katas},
                points="outliers",
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, width="stretch", key="kcomp_box_notes")

            # Note par tour
            note_tour = sub_notes.groupby(["Kata", "N_Tour"], observed=True)["Note"].mean().reset_index()
            note_tour["Note"] = note_tour["Note"].round(2)
            note_tour["_rank"] = note_tour["N_Tour"].apply(_tour_rank)

            # Index séquentiel (espacement uniforme) pour tous les tours présents
            all_tours_in_data = sorted(note_tour["N_Tour"].unique(), key=_tour_rank)
            tour_to_pos = {tour: i for i, tour in enumerate(all_tours_in_data)}
            tour_labels = [fmt_tour(t_val) for t_val in all_tours_in_data]

            fig_note_tour = go.Figure()
            for kata in sel_katas:
                df_k = note_tour[note_tour["Kata"] == kata]
                if not df_k.empty:
                    # Construire x/y et trier par position (bypass pandas sort)
                    x_pos = df_k["N_Tour"].map(tour_to_pos).tolist()
                    y_vals = df_k["Note"].tolist()
                    labels = [fmt_tour(t) for t in df_k["N_Tour"].tolist()]
                    sorted_data = sorted(zip(x_pos, y_vals, labels), key=lambda p: p[0])
                    x_sorted = [p[0] for p in sorted_data]
                    y_sorted = [p[1] for p in sorted_data]
                    labels_sorted = [p[2] for p in sorted_data]
                    fig_note_tour.add_trace(go.Scatter(
                        x=x_sorted, y=y_sorted,
                        mode="lines+markers", name=kata,
                        text=labels_sorted,
                        hovertemplate="%{text}<br>Note: %{y:.2f}<extra>%{fullData.name}</extra>",
                    ))
            fig_note_tour.update_layout(
                title=t("Note moyenne par tour"),
                xaxis_title=t("Tour"),
                yaxis_title=t("Note moyenne"),
                xaxis={
                    "type": "linear",
                    "tickmode": "array",
                    "tickvals": list(range(len(all_tours_in_data))),
                    "ticktext": tour_labels,
                },
            )
            st.plotly_chart(fig_note_tour, use_container_width=True, key="kcomp_note_tour")

        # ═══════════════════════════════════════════════════════════════
        # Popularité (nombre d'utilisations) dans le temps
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader(t("Popularité dans le temps"))

        usage_year = sub.groupby(["Kata", "Year"], observed=True).size().reset_index(name="Passages")
        if len(usage_year["Year"].unique()) > 1:
            fig_usage = go.Figure()
            for kata in sel_katas:
                df_k = usage_year[usage_year["Kata"] == kata]
                if not df_k.empty:
                    pairs = sorted(zip(df_k["Year"].tolist(), df_k["Passages"].tolist()), key=lambda p: p[0])
                    fig_usage.add_trace(go.Scatter(
                        x=[p[0] for p in pairs], y=[p[1] for p in pairs],
                        mode="lines+markers", name=kata,
                    ))
            fig_usage.update_layout(
                title=t("Nombre de passages par année"),
                xaxis_title=t("Année"),
                yaxis_title=t("Passages"),
                xaxis={"dtick": 1, "tickformat": "d"},
            )
            st.plotly_chart(fig_usage, width="stretch", key="kcomp_usage_year")
        else:
            usage_display = usage_year[["Kata", "Passages"]].sort_values("Passages", ascending=False)
            st.dataframe(usage_display, width="stretch", hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # Profil des athlètes qui jouent chaque kata
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader(t("Profil des athlètes"))

        profil = sub.groupby("Kata", observed=True).agg(
            Ranking_Moy=("Ranking", "mean"),
            Age_Moy=("Age", "mean") if "Age" in sub.columns else ("Year", "count"),
            Top_Athletes=("Nom", lambda s: ", ".join(
                s.value_counts().head(3).index.tolist()
            )),
        ).reset_index()
        profil["Ranking_Moy"] = profil["Ranking_Moy"].round(0)
        if "Age" in sub.columns:
            profil["Age_Moy"] = profil["Age_Moy"].round(1)

        st.dataframe(
            profil.rename(columns={
                "Kata": t("Kata"),
                "Ranking_Moy": t("Ranking moyen"),
                "Age_Moy": t("Âge moyen"),
                "Top_Athletes": t("Top 3 utilisateurs"),
            }),
            width="stretch", hide_index=True,
        )

        # ═══════════════════════════════════════════════════════════════
        # Win rate par type de compétition (si filtre "Tous")
        # ═══════════════════════════════════════════════════════════════
        if sel_type == t("Tous") and sub["Type_Compet"].nunique() > 1:
            st.markdown("---")
            st.subheader(t("Win rate par circuit (K1 vs SA)"))

            wr_circuit = sub.groupby(["Kata", "Type_Compet"], observed=True).agg(
                Win_Rate=("Win", "mean"),
                Passages=("Win", "count"),
            ).reset_index()
            wr_circuit["Win_Rate"] = (wr_circuit["Win_Rate"] * 100).round(1)

            fig_circuit = px.bar(
                wr_circuit, x="Kata", y="Win_Rate", color="Type_Compet",
                barmode="group", text="Win_Rate",
                hover_data=["Passages"],
                title=t("Win rate K1 vs SA"),
                labels={"Win_Rate": "Win rate (%)", "Kata": t("Kata"), "Type_Compet": "Circuit"},
                category_orders={"Kata": sel_katas},
            )
            fig_circuit.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig_circuit.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_circuit, width="stretch", key="kcomp_circuit")

        # ═══════════════════════════════════════════════════════════════
        # Performance en drapeaux (2026+)
        # ═══════════════════════════════════════════════════════════════
        sub_flags = sub[sub["Drapeau"].notna()] if "Drapeau" in sub.columns else pd.DataFrame()
        if not sub_flags.empty and len(sub_flags) >= 5:
            st.markdown("---")
            st.subheader(t("Performance drapeaux (2026+)"))

            flag_stats = sub_flags.groupby("Kata", observed=True).agg(
                Drapeau_Moy=("Drapeau", "mean"),
                Win_Rate_Flag=("Win", "mean"),
                Passages=("Win", "count"),
            ).reset_index()
            flag_stats["Drapeau_Moy"] = flag_stats["Drapeau_Moy"].round(2)
            flag_stats["Win_Rate_Flag"] = (flag_stats["Win_Rate_Flag"] * 100).round(1)

            col1, col2 = st.columns(2)
            with col1:
                fig_flag = px.bar(
                    flag_stats, x="Kata", y="Drapeau_Moy", color="Kata",
                    text="Drapeau_Moy",
                    title=t("Drapeaux moyens obtenus"),
                    labels={"Drapeau_Moy": t("Drapeaux moy."), "Kata": t("Kata")},
                    category_orders={"Kata": sel_katas},
                )
                fig_flag.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_flag.update_layout(showlegend=False)
                st.plotly_chart(fig_flag, width="stretch", key="kcomp_flag_moy")
            with col2:
                fig_wr_flag = px.bar(
                    flag_stats, x="Kata", y="Win_Rate_Flag", color="Kata",
                    text="Win_Rate_Flag",
                    title=t("Win rate (ère drapeaux)"),
                    labels={"Win_Rate_Flag": "Win rate (%)", "Kata": t("Kata")},
                    category_orders={"Kata": sel_katas},
                )
                fig_wr_flag.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
                fig_wr_flag.update_layout(showlegend=False)
                fig_wr_flag.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_wr_flag, width="stretch", key="kcomp_flag_wr")

        # ═══════════════════════════════════════════════════════════════
        # Synthèse / recommandation
        # ═══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader(t("Synthèse comparative"))

        # Tableau récapitulatif complet
        synth = stats[["Kata", "Passages", "Win_Rate", "Note_Moy", "Note_Max", "Note_Std", "Athlètes"]].copy()
        synth = synth.rename(columns={
            "Win_Rate": "Win Rate (%)",
            "Note_Moy": t("Note moy."),
            "Note_Max": "Note max",
            "Note_Std": t("Écart-type"),
            "Athlètes": t("Athlètes"),
        })
        st.dataframe(synth, width="stretch", hide_index=True)

        # Verdict automatique
        if len(stats) >= 2:
            best_wr = stats.loc[stats["Win_Rate"].idxmax()]
            best_note = stats.loc[stats["Note_Moy"].idxmax()] if stats["Note_Moy"].notna().any() else None
            most_used = stats.loc[stats["Passages"].idxmax()]

            verdicts = []
            verdicts.append(f"🏆 **{t('Meilleur win rate')}** : {best_wr['Kata']} ({best_wr['Win_Rate']:.1f}%, {int(best_wr['Passages'])} passages)")
            if best_note is not None and pd.notna(best_note["Note_Moy"]):
                verdicts.append(f"📝 **{t('Meilleure note moy.')}** : {best_note['Kata']} ({best_note['Note_Moy']:.2f})")
            verdicts.append(f"📊 **{t('Plus joué')}** : {most_used['Kata']} ({int(most_used['Passages'])} passages, {int(most_used['Athlètes'])} athlètes)")

            st.markdown("\n".join(verdicts))
