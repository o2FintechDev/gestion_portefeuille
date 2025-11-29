import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import altair as alt
from calculs_financiers.markowitz import Perf


# ==========================================================
# HELPERS GÉNÉRIQUES
# ==========================================================

def _flatten_index_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Aplatit un MultiIndex en colonnes simples."""
    if isinstance(df.columns[0], tuple):
        df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns]
    return df


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """S'assure que DataFrame contient une colonne Date."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    else:
        date_cols = [c for c in df.columns if "date" in str(c).lower()]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "Date"})
        else:
            raise ValueError("Impossible d’identifier la colonne Date.")
    return df


def _style_plot(fig):
    fig.update_layout(template="plotly_dark", legend_title_text="", height=500)
    return fig


# ==========================================================
# 1) Poids du portefeuille
# ==========================================================

def tracer_poids_portefeuille(pf: Perf, titre="Répartition du portefeuille"):
    if pf is None or pf.weights is None or pf.weights.empty:
        st.warning("Aucun poids de portefeuille à afficher.")
        return

    wdf = (
        pf.weights.reset_index()
        .rename(columns={"index": "Ticker", 0: "Poids"})
        .sort_values("Poids", ascending=False)
    )

    st.subheader(titre)
    st.dataframe(wdf.style.format({"Poids": "{:.2%}"}), width='content')

    # Bar chart Altair
    chart = (
        alt.Chart(wdf)
        .mark_bar()
        .encode(
            x=alt.X("Ticker:N", sort="-y"),
            y=alt.Y("Poids:Q"),
            color=alt.Color("Ticker:N", legend=None)
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width='content')


# ==========================================================
# 2) Évolution des cours
# ==========================================================

def tracer_cours(df: pd.DataFrame, titre="Évolution des cours"):
    if df is None or df.empty:
        st.warning("Aucune donnée à afficher.")
        return

    df_plot = _flatten_index_if_needed(df)
    df_plot = _ensure_date_column(df_plot)

    # Conversion valeurs numériques
    value_cols = [c for c in df_plot.columns if c != "Date"]
    df_plot[value_cols] = df_plot[value_cols].apply(pd.to_numeric, errors="coerce")

    df_long = df_plot.melt(id_vars="Date", var_name="Ticker", value_name="Prix").dropna()

    fig = px.line(df_long, x="Date", y="Prix", color="Ticker", title=titre)
    st.plotly_chart(_style_plot(fig), width='content')


# ==========================================================
# 3) Matrice de corrélation
# ==========================================================

def tracer_correlation(corr: pd.DataFrame, titre="Matrice de corrélation"):
    if corr is None or corr.empty or corr.shape[0] < 2:
        st.warning("Pas assez de données pour afficher la corrélation.")
        return

    z = corr.values
    show_values = corr.shape[0] <= 5

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1, zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="ρ"),
            text=np.round(z, 2) if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont=dict(size=10, color="black") if show_values else None
        )
    )

    fig.update_layout(title=titre, template="plotly_dark", height=520)
    st.plotly_chart(fig, width='content')


# ==========================================================
# 4) Frontière efficiente + CML + portefeuille tangent
# ==========================================================

def tracer_frontiere(df, rf_rate=None, optimal=None):
    if df is None or df.empty:
        raise ValueError("DataFrame vide")

    df = df.dropna().sort_values("Risque")

    fig = go.Figure()

    # === Frontière ===
    fig.add_scatter(
        x=df["Risque"], y=df["Rendement"],
        mode="lines+markers",
        name="Frontière efficiente",
        line=dict(color="rgb(100,160,220)", width=3),
        marker=dict(size=4, opacity=0.6)
    )

    # === CML & Tangent ===
    if rf_rate is not None and optimal is not None:
        mu_t = float(optimal["Rendement"])
        sig_t = float(optimal["Risque"])

        sharpe = (mu_t - rf_rate) / sig_t

        # CML
        x_max = max(df["Risque"].max(), sig_t) * 1.3
        xs = np.linspace(0, x_max, 300)
        ys = rf_rate + sharpe * xs

        fig.add_scatter(
            x=xs, y=ys,
            mode="lines",
            name=f"CML (Sharpe={sharpe:.3f})",
            line=dict(color="red", width=2, dash="dash")
        )

        # Point optimal
        fig.add_scatter(
            x=[sig_t], y=[mu_t],
            mode="markers+text",
            name="Portefeuille tangent",
            marker=dict(color="red", size=12, symbol="star"),
            text=[f"μ={mu_t:.2%}<br>σ={sig_t:.2%}"],
            textposition="top center",
            textfont=dict(size=10)
        )

    fig.update_layout(
        title="Frontière Efficiente & Capital Market Line",
        xaxis_title="Risque (Volatilité annuelle σ)",
        yaxis_title="Rendement espéré annuel μ",
        template="plotly_dark",
        height=600,
        legend=dict(x=0.02, y=0.98),
        hovermode="closest",
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(tickformat=".1%")
    )

    return fig


    

