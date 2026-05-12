# tabs/progression.py — Onglet A : Progression temporelle des notes
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import safe_mode
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

        filter_panel_close()

    with content_col:
        if not sel_athletes:
            st.info(t("Sélectionnez au moins un athlète dans le panneau de filtres à gauche."))
            return

        sub = df_scope[df_scope["Nom"].isin(sel_athletes)].dropna(subset=["Note"]).copy()
        if sub.empty:
            st.warning(t("Aucune donnée avec notes pour ces athlètes."))
            return

        # Build a chronological order: Year + Competition
        sub["Compet_Label"] = sub["Competition"].astype(str) + " " + sub["Year"].astype(int).astype(str)
        # Order by year then competition name
        sub = sub.sort_values(["Year", "Competition", "N_Tour"])

        # Aggregate: mean note per athlete per competition
        agg = sub.groupby(["Nom", "Compet_Label", "Year"]).agg(
            Note_Moy=("Note", "mean"),
            Note_Max=("Note", "max"),
            Nb_Passages=("Note", "count"),
        ).reset_index().sort_values(["Year", "Compet_Label"])

        # Only keep competitions where selected athletes participated
        compets_athletes = set(agg["Compet_Label"].unique())
        agg = agg[agg["Compet_Label"].isin(compets_athletes)]

        # Order x-axis to only show competitions where athletes have data
        compet_order = agg.sort_values(["Year", "Compet_Label"])["Compet_Label"].unique().tolist()

        # Rolling average
        if show_rolling:
            agg["Note_Rolling"] = agg.groupby("Nom")["Note_Moy"].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )

        # ── Line chart ──
        fig = px.line(
            agg, x="Compet_Label", y="Note_Moy", color="Nom",
            markers=True,
            hover_data=["Note_Max", "Nb_Passages"],
            title=t("Évolution de la note moyenne par compétition"),
            labels={"Note_Moy": t("Note moyenne"), "Compet_Label": t("Compétition")},
            category_orders={"Compet_Label": compet_order},
        )

        if show_rolling:
            # Use matching colors from the main traces
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
                    name=f"{nom} (moy. mobile 3 compét.)",
                    showlegend=True,
                    legendgroup=nom,
                ))

        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True, key="prog_line")

        # ── Stats summary ──
        st.subheader(t("Résumé par athlète"))
        summary = []
        for nom in sel_athletes:
            d_a = sub[sub["Nom"] == nom]
            if d_a.empty:
                continue
            summary.append({
                t("Athlète"): nom,
                t("Compétitions"): d_a["Competition"].nunique(),
                "Passages": len(d_a),
                t("Note moy."): round(d_a["Note"].mean(), 2),
                "Note max": round(d_a["Note"].max(), 2),
                "Note min": round(d_a["Note"].min(), 2),
                t("Écart-type"): round(d_a["Note"].std(), 2) if len(d_a) > 1 else 0.0,
                t("Tendance"): "↗" if len(d_a) >= 3 and d_a["Note"].iloc[-3:].mean() > d_a["Note"].iloc[:3].mean() else "↘" if len(d_a) >= 3 else "—",
            })
        if summary:
            # Interprétation dynamique
            st.markdown("---")
            st.markdown(t("#### 💡 Interprétation"))
            st.markdown(interpret_progression(summary), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(summary), use_container_width=True)

        # ── Box plot per athlete ──
        if len(sel_athletes) >= 2:
            st.subheader(t("Distribution des notes"))
            show_chart_guide("boxplot")
            fig_box = px.box(
                sub, x="Nom", y="Note", color="Nom",
                title=t("Distribution des notes par athlète"),
                points="outliers",
            )
            st.plotly_chart(fig_box, use_container_width=True, key="prog_box")
