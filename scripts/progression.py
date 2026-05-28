# tabs/progression.py — Onglet A : Progression temporelle des notes
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import safe_mode, is_flag_era
from utils.interpretations import show_tab_help, show_chart_guide, interpret_progression, std_color, _color_badge
from utils.display import fmt_tour
from utils.lang import t


@st.fragment
def show_progression_tab(data: pd.DataFrame) -> None:
    st.header(t("Progression temporelle des notes"))
    show_tab_help("progression")

    df = data.copy()
    for col in ["Note", "Year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        # Type compet filter
        type_options = [t("Tous"), "K1", "SA"]
        sel_type = st.radio(t("Type compétition"), type_options, key="prog_type")
        df_scope = df.copy()
        if sel_type != t("Tous"):
            df_scope = df_scope[df_scope["Type_Compet"] == sel_type]

        # Sexe filter
        sexes = sorted(df_scope["Sexe"].dropna().unique().tolist())
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexes, key="prog_sexe")
        if sel_sexe != t("Tous"):
            df_scope = df_scope[df_scope["Sexe"] == sel_sexe]

        # Tour filter
        all_tours = sorted(df_scope["N_Tour"].dropna().astype(str).unique().tolist())
        sel_tours = st.multiselect(t("Tours"), all_tours, default=all_tours, format_func=fmt_tour, key="prog_tours")
        if sel_tours:
            df_scope = df_scope[df_scope["N_Tour"].astype(str).isin(sel_tours)]

        athletes = sorted(df_scope["Nom"].dropna().unique().tolist())
        if not athletes:
            st.warning(t("Aucun athlète disponible."))
            filter_panel_close()
            return

        sel_athletes = st.multiselect(
            t("Athlète(s) à comparer"),
            athletes,
            default=[],
            max_selections=5,
            key="prog_athletes",
        )

        show_rolling = st.checkbox(t("Moyenne mobile (3 compétitions)"), value=True, key="prog_rolling")
        show_benchmark = st.checkbox(t("Afficher moyenne circuit"), value=False, key="prog_bench")

        filter_panel_close()

    with content_col:
        if not sel_athletes:
            st.info(t("Sélectionnez au moins un athlète dans le panneau de filtres à gauche."))
            return

        # Filter out placeholder rows
        df_scope = df_scope[df_scope["Nom"].notna() & (df_scope["Nom"].astype(str).str.strip() != "")]

        sub = df_scope[df_scope["Nom"].isin(sel_athletes)].copy()
        if sub.empty:
            st.warning(t("Aucune donnée pour ces athlètes."))
            return

        # Separate note-era and flag-era data
        sub_notes = sub[sub["Note"].notna()].copy()
        sub_flags = sub[sub.apply(lambda r: is_flag_era(r.get("Year")), axis=1)].copy()

        # Build a chronological order based on first appearance index in the full dataset
        _chrono_order = (
            df_scope.drop_duplicates(subset=["Competition", "Year"])
            .sort_index()[["Competition", "Year"]]
            .assign(Compet_Label=lambda x: x["Competition"].astype(str) + " " + x["Year"].astype(int).astype(str))
        )["Compet_Label"].tolist()

        sub["Compet_Label"] = sub["Competition"].astype(str) + " " + sub["Year"].astype(int).astype(str)

        # ═══ Note-era progression (2024-2025) ═══
        if not sub_notes.empty:
            sub_notes["Compet_Label"] = sub_notes["Competition"].astype(str) + " " + sub_notes["Year"].astype(int).astype(str)

            agg = sub_notes.groupby(["Nom", "Compet_Label", "Year"]).agg(
                Note_Moy=("Note", "mean"),
                Note_Max=("Note", "max"),
                Nb_Passages=("Note", "count"),
            ).reset_index()

            # Keep only competition labels that appear in global chrono order
            compet_order = [c for c in _chrono_order if c in agg["Compet_Label"].values]

            if show_rolling:
                agg["Note_Rolling"] = agg.groupby("Nom")["Note_Moy"].transform(
                    lambda s: s.rolling(3, min_periods=1).mean()
                )

            fig = px.line(
                agg, x="Compet_Label", y="Note_Moy", color="Nom",
                markers=True,
                hover_data=["Note_Max", "Nb_Passages"],
                title=t("Évolution de la note moyenne par compétition"),
                labels={"Note_Moy": t("Note moyenne"), "Compet_Label": t("Compétition")},
                category_orders={"Compet_Label": compet_order},
            )

            # ── Benchmark: moyenne du circuit (optionnel) ──
            if show_benchmark:
                circuit_avg = df_scope[df_scope["Note"].notna()].copy()
                circuit_avg["Compet_Label"] = circuit_avg["Competition"].astype(str) + " " + circuit_avg["Year"].astype(int).astype(str)
                circuit_avg = circuit_avg[circuit_avg["Compet_Label"].isin(compet_order)]
                bench = circuit_avg.groupby("Compet_Label")["Note"].mean().reset_index()
                bench = bench.set_index("Compet_Label").reindex(compet_order).reset_index()
                if not bench.empty:
                    fig.add_trace(go.Scatter(
                        x=bench["Compet_Label"], y=bench["Note"],
                        mode="lines", line=dict(dash="dot", width=2, color="rgba(128,128,128,0.6)"),
                        name=t("Moyenne circuit"),
                        showlegend=True,
                    ))

            if show_rolling:
                color_map = {}
                for trace in fig.data:
                    color_map[trace.name] = trace.line.color

                for nom in sel_athletes:
                    d_a = agg[agg["Nom"] == nom]
                    if d_a.empty:
                        continue
                    base_color = color_map.get(nom, None)
                    fig.add_trace(go.Scatter(
                        x=d_a["Compet_Label"], y=d_a["Note_Rolling"],
                        mode="lines", line=dict(dash="dash", width=1, color=base_color),
                        name=f"{nom} (moy. mobile)",
                        showlegend=False,
                        legendgroup=nom,
                    ))

            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, width="stretch", key="prog_line")

        # ═══ Flag-era progression (2026+) — Win Rate + Drapeaux ═══
        if not sub_flags.empty:
            st.markdown("---")
            st.subheader(t("Progression (système drapeaux 2026+)"))

            sub_flags["Compet_Label"] = sub_flags["Competition"].astype(str) + " " + sub_flags["Year"].astype(int).astype(str)
            sub_flags["_win"] = sub_flags["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

            agg_flags = sub_flags.groupby(["Nom", "Compet_Label", "Year"]).agg(
                Win_Rate=("_win", "mean"),
                Nb_Matchs=("_win", "count"),
            ).reset_index()

            compet_order_f = [c for c in _chrono_order if c in agg_flags["Compet_Label"].values]

            # Win Rate progression
            fig_f = px.line(
                agg_flags, x="Compet_Label", y="Win_Rate", color="Nom",
                markers=True,
                hover_data=["Nb_Matchs"],
                title=t("Taux de victoire par compétition (2026+)"),
                labels={"Win_Rate": t("Taux de victoire"), "Compet_Label": t("Compétition")},
                category_orders={"Compet_Label": compet_order_f},
            )
            fig_f.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
            fig_f.update_layout(xaxis_tickangle=-45, height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig_f, width="stretch", key="prog_line_flags")

            # Drapeaux moyens obtenus par match
            if "Drapeau" in sub_flags.columns and sub_flags["Drapeau"].notna().any():
                agg_drapeaux = sub_flags[sub_flags["Drapeau"].notna()].groupby(
                    ["Nom", "Compet_Label", "Year"]
                ).agg(
                    Drapeaux_Moy=("Drapeau", "mean"),
                    Nb_Matchs=("Drapeau", "count"),
                ).reset_index().sort_values(["Year", "Compet_Label"])

                fig_drapeaux = px.line(
                    agg_drapeaux, x="Compet_Label", y="Drapeaux_Moy", color="Nom",
                    markers=True,
                    hover_data=["Nb_Matchs"],
                    title=t("Drapeaux moyens obtenus par compétition (2026+)"),
                    labels={"Drapeaux_Moy": t("Drapeaux moyens"), "Compet_Label": t("Compétition")},
                    category_orders={"Compet_Label": compet_order_f},
                )
                fig_drapeaux.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_drapeaux, width="stretch", key="prog_line_drapeaux")

        # ── Stats summary ──
        st.subheader(t("Résumé par athlète"))
        summary = []
        for nom in sel_athletes:
            d_a = sub[sub["Nom"] == nom]
            if d_a.empty:
                continue
            d_a_notes = d_a[d_a["Note"].notna()]
            entry = {
                t("Athlète"): nom,
                t("Compétitions"): d_a["Competition"].nunique(),
                "Passages": len(d_a),
            }
            if not d_a_notes.empty:
                entry[t("Note moy.")] = round(d_a_notes["Note"].mean(), 2)
                entry["Note max"] = round(d_a_notes["Note"].max(), 2)
                entry["Note min"] = round(d_a_notes["Note"].min(), 2)
                entry[t("Écart-type")] = round(d_a_notes["Note"].std(), 2) if len(d_a_notes) > 1 else 0.0
                entry[t("Tendance")] = "↗" if len(d_a_notes) >= 3 and d_a_notes["Note"].iloc[-3:].mean() > d_a_notes["Note"].iloc[:3].mean() else "↘" if len(d_a_notes) >= 3 else "—"
            # Add win rate info
            d_a_wins = d_a["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"])
            entry["Win Rate"] = f"{d_a_wins.mean():.0%}"
            summary.append(entry)

        if summary:
            st.markdown("---")
            st.markdown(t("#### 💡 Interprétation"))
            st.markdown(interpret_progression(summary), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(summary), width="stretch")

        # ── Box plot per athlete ──
        if len(sel_athletes) >= 2 and not sub_notes.empty:
            st.subheader(t("Distribution des notes"))
            show_chart_guide("boxplot")
            fig_box = px.box(
                sub_notes, x="Nom", y="Note", color="Nom",
                title=t("Distribution des notes par athlète"),
                points="outliers",
            )
            st.plotly_chart(fig_box, width="stretch", key="prog_box")
