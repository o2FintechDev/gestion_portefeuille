import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import altair as alt
from calculs_financiers.markowitz import Perf

# ==========================================================
# 1) Poids du portefeuille
# ==========================================================
def tracer_poids_portefeuille(pf: Perf, titre: str = "Répartition du portefeuille"):
    if pf is None or pf.weights is None or pf.weights.empty:
        st.warning("Aucun poids de portefeuille à afficher.")
        return

    w = pf.weights.copy()
    # Flatten MultiIndex éventuel
    if isinstance(w.index[0], tuple):
        w.index = ["_".join(map(str, idx)) for idx in w.index]

    df_weights = (
        w.reset_index()
         .rename(columns={"index": "Ticker", 0: "Poids"})
    )
    st.subheader(titre)
    st.dataframe(df_weights.style.format({"Poids": "{:.2%}"}), width='stretch')

    chart = (
        alt.Chart(df_weights)
        .mark_bar()
        .encode(
            x=alt.X("Ticker:N", sort="-y", title="Ticker"),
            y=alt.Y("Poids:Q", title="Poids du portefeuille"),
            color=alt.Color("Ticker:N", legend=None)
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width='stretch')

# ==========================================================
# 2) Évolution des cours
# ==========================================================
def tracer_cours(df: pd.DataFrame, titre: str = "Évolution des cours boursiers"):
    if df is None or df.empty:
        st.warning("Aucune donnée à afficher.")
        return

    df_plot = df.copy()

    # Flatten MultiIndex colonnes si nécessaire
    if isinstance(df_plot.columns[0], tuple):
        df_plot.columns = ['_'.join([str(x) for x in col if x]) for col in df_plot.columns]

    # Mettre l'index de dates dans une colonne "Date"
    if isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot = df_plot.reset_index().rename(columns={df_plot.index.name or "index": "Date"})
    else:
        # tente de détecter une colonne date
        cands = [c for c in df_plot.columns if "date" in str(c).lower()]
        if cands:
            df_plot = df_plot.rename(columns={cands[0]: "Date"})
        else:
            st.error("Impossible d’identifier la colonne de dates.")
            return

    # Cast en float pour éviter 'mixed types'
    value_cols = [c for c in df_plot.columns if c != "Date"]
    df_plot[value_cols] = df_plot[value_cols].apply(pd.to_numeric, errors="coerce")

    df_long = df_plot.melt(id_vars="Date", var_name="Ticker", value_name="Prix").dropna()
    fig = px.line(df_long, x="Date", y="Prix", color="Ticker", title=titre)
    fig.update_layout(template="plotly_dark", height=500, legend_title_text="")
    st.plotly_chart(fig, width='stretch')

# ==========================================================
# 3) Matrice de corrélation
# ==========================================================
def tracer_correlation(corr_matrix: pd.DataFrame, titre: str = "Matrice de corrélation"):
    if corr_matrix is None or corr_matrix.empty or corr_matrix.shape[0] < 2:
        st.warning("Pas assez de données pour afficher la corrélation.")
        return

    # Heatmap Plotly pour cohérence de thème
    z = corr_matrix.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=list(corr_matrix.columns), y=list(corr_matrix.index),
            zmin=-1, zmax=1, colorscale="RdBu", colorbar=dict(title="ρ")
        )
    )
    fig.update_layout(title=titre, template="plotly_dark", height=520)
    st.plotly_chart(fig, width='stretch')

# ==========================================================
# 4) Frontière efficiente + CML + point optimal
# ==========================================================
def tracer_frontiere(df_sim: pd.DataFrame, rf_rate: float | None = None, optimal: dict | None = None):
    """
    df_sim: DataFrame avec colonnes ['Risque','Rendement'] (annualisés, décimaux).
    rf_rate: taux sans risque annuel en décimal (ex 0.04). Si None → pas de CML.
    optimal: dict avec clés {'Risque','Rendement','Sharpe'} calculées dans le même espace.
    """
    # Validation
    if df_sim is None or df_sim.empty:
        raise ValueError("DataFrame vide")
    
    required = {"Risque", "Rendement"}
    if not required.issubset(df_sim.columns):
        raise ValueError(f"Colonnes manquantes: {required - set(df_sim.columns)}")
    
    dfp = df_sim.dropna().sort_values("Risque").copy()

    fig = go.Figure()
    
    # Frontière
    fig.add_scatter(
        x=dfp["Risque"], 
        y=dfp["Rendement"],
        mode="lines+markers",
        name="Frontière efficiente",
        line=dict(color="rgb(100,160,220)", width=3),
        marker=dict(size=4, opacity=0.6)
    )
    
    # CML et optimal
    if rf_rate is not None and optimal is not None:
        mu_m = float(optimal.get("Rendement", 0))
        sig_m = float(optimal.get("Risque", 0))
        
        if sig_m > 1e-6 and np.isfinite(mu_m) and np.isfinite(sig_m):
            # NOUVEAU : Validation de cohérence
            sharpe_calc = (mu_m - rf_rate) / sig_m
            sharpe_fourni = optimal.get("Sharpe", sharpe_calc)
            
            if abs(sharpe_calc - sharpe_fourni) > 0.01:
                print(f"⚠️  Incohérence Sharpe: calculé={sharpe_calc:.3f}, fourni={sharpe_fourni:.3f}")
            
            # CML
            x_max = max(dfp["Risque"].max(), sig_m) * 1.3
            x_cml = np.linspace(0, x_max, 300)
            y_cml = rf_rate + sharpe_calc * x_cml
            
            fig.add_scatter(
                x=x_cml, y=y_cml,
                mode="lines",
                name=f"CML (Sharpe={sharpe_calc:.3f})",
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Point optimal
            fig.add_scatter(
                x=[sig_m], y=[mu_m],
                mode="markers+text",
                name="Portefeuille tangent",
                marker=dict(color="red", size=12, symbol="star"),
                text=[f"μ={mu_m:.2%}<br>σ={sig_m:.2%}"],
                textposition="top center",
                textfont=dict(size=10, color="white")
            )
    
    # Axes avec labels clairs
    fig.update_layout(
        title="Frontière Efficiente & Capital Market Line",
        xaxis_title="Risque (Volatilité annuelle σ)",
        yaxis_title="Rendement espéré annuel μ",
        template="plotly_dark",
        height=600,
        legend=dict(x=0.02, y=0.98),
        hovermode="closest",
        # NOUVEAU : Formater les axes en pourcentage directement dans layout
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(tickformat=".1%")
    )
    
    return fig

    

