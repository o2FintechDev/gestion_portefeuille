# IMPORTS GLOBAUX
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from stock.data_cache import fetch_data_with_cache
from stock.symbol_resolver import suggere_tickers

from calculs_financiers.rendement_risque import calcul_rendements, matrices_risque_rendement
from calculs_financiers.markowitz import (
    min_variance, 
    tangency_portfolio,
    tangent_on_frontier,
    efficient_frontier,
)

from calculs_financiers.visualisations import tracer_cours, tracer_correlation, tracer_frontiere
from calculs_financiers.inflation_api import get_inflation_label, EU_COUNTRIES
from calculs_financiers.indicateurs_techniques import rsi, macd, moyennes_mobiles
from calculs_financiers.utils import statistiques_actifs, resume_portefeuille
from calculs_financiers.taux_sans_risque_api import get_risk_free_rate
from calculs_financiers.utils import detect_market_index
from calculs_financiers.capm import classification_beta, compute_portfolio_beta

# ==========================================================
# CONFIGURATION GLOBALE
# ==========================================================

st.set_page_config(page_title="📈 Gestion de Portefeuille", layout="wide")
def configure_app():
    """Configuration Streamlit : page, layout, titre."""
    
    st.title("💼 Application de Gestion de Portefeuille")
    st.caption("Analyse interactive Markowitz/Tobin, risques, indicateurs techniques.")

# =========================================================
# FONCTION PRINCIPALE
# =========================================================
def main():

    configure_app()
    # ==========================================================
    # INITIALISATION DES ÉTATS
    # ==========================================================
    if "tickers" not in st.session_state:
        st.session_state.tickers = []
    if "analysis_ready" not in st.session_state:
        st.session_state.analysis_ready = False
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "last_params" not in st.session_state:
        st.session_state.last_params = None

    # ==========================================================
    # BARRE LATÉRALE : PARAMÈTRES GLOBAUX
    # ==========================================================
    st.sidebar.header("⚙️ Paramètres")

    country = st.sidebar.selectbox("Pays (pour l'inflation)", EU_COUNTRIES)

    # --- Recherche d'actifs ---
    st.sidebar.subheader("📊 Sélection des actifs")
    query = st.sidebar.text_input("Nom ou indice (ex: Apple, CAC 40, S&P 500)")

    if query:
        suggestions = suggere_tickers(query)
        if suggestions:
            choix = st.sidebar.selectbox("Suggestions :", [s[0] for s in suggestions],
                                        format_func=lambda x: dict(suggestions)[x])
            if st.sidebar.button("➕ Ajouter la suggestion"):
                if choix and choix not in st.session_state.tickers:
                    st.session_state.tickers.append(choix)
                    st.session_state.analysis_ready = False
                    st.rerun()

    # --- Ajout manuel ---
    manual = st.sidebar.text_input("Ajout manuel (ex: AAPL, MSFT)")
    if st.sidebar.button("➕ Ajouter manuellement"):
        nouveaux = [t.strip().upper() for t in manual.split(",") if t.strip()]
        added = False
        for t in nouveaux:
            if t not in st.session_state.tickers:
                st.session_state.tickers.append(t)
                added = True
        if added:
            st.session_state.analysis_ready = False
            st.rerun()

    # ==========================================================
    # AFFICHAGE ET GESTION DES ACTIFS SÉLECTIONNÉS
    # ==========================================================
    st.markdown("### ✅ Actifs sélectionnés")

    if st.session_state.tickers:
        cols = st.columns(min(5, len(st.session_state.tickers)))
        
        for idx, ticker in enumerate(st.session_state.tickers):
            col_idx = idx % 5
            st.markdown('<div class="small-left-btn">', unsafe_allow_html=True)
            with cols[col_idx]:
                if st.button(f"{ticker} ❌", key=f"remove_{ticker}_{idx}", width='stretch'):
                    st.session_state.tickers.remove(ticker)
                    st.session_state.analysis_ready = False
                    st.rerun()
        
        if st.button("🗑️ Effacer tous les actifs", type="secondary"):
            st.session_state.tickers = []
            st.session_state.analysis_ready = False
            st.session_state.analysis_result = None
            st.rerun()
    else:
        st.info("Aucun actif sélectionné. Utilisez la barre latérale pour en ajouter.")

    # ==========================================================
    # PARAMÈTRES : périodes, contraintes, taux sans risque
    # ==========================================================

    # --- Contraintes de poids ---
    st.sidebar.subheader("⚖️ Contraintes de portefeuille")
    presets = {
        "10% — Diversification institutionnelle": 0.10,
        "20% — Gestion prudente": 0.20,
        "35% — Modéré": 0.35,
        "50% — Concentré": 0.50,
        "100% — Sans contrainte": 1.00,
        "Personnalisé": None,
    }

    poids = st.sidebar.selectbox(
        "Choisissez un preset :",
        list(presets.keys()),
        index=2  # preset par défaut = 35%
    )

    if presets[poids] is not None:
        w_max = presets[poids]
    else:
        w_max = st.sidebar.slider(
            "Poids maximal personnalisé :", 
            min_value=0.05, 
            max_value=1.0, 
            value=0.35, 
            step=0.01
        )

    st.sidebar.write(f"**Poids max appliqué : {w_max:.0%}**")
    # --- Période et taux sans risque ---
    st.sidebar.subheader("🗓️ Période d'analyse")
    periodes = {"1 an": 252, "3 ans": 756, "5 ans": 1260}
    periode_label = st.sidebar.selectbox("Durée :", list(periodes.keys()))
    nb_jours = periodes[periode_label]

    st.sidebar.subheader("💶 Taux sans risque")
    rf_opts = {
        "Aucun": None,
        "Euro Short-Term Composite": "euro_short_rate",
        "Fed Funds (USD)": "fed_funds",
        "T-Bill 3M (USD)": "tbill_3m",
        "SOFR (USD)": "sofr",
        "SONIA (GBP)": "sonia",
        "UK Gilt 3M": "uk_3m",
        "OAT 10 ans": "oat_10y",
        "Bund 10 ans": "bund_10y",
    }


    rf_label = st.sidebar.selectbox("Actif sans risque :", list(rf_opts.keys()))
    choice = rf_opts[rf_label]

    rf_rate = get_risk_free_rate(choice) if choice else None
    # ==========================================================
    # LANCEMENT DE L'ANALYSE : téléchargement + préparation
    # ==========================================================
    current_params = (tuple(st.session_state.tickers), periode_label, rf_label, country)

    if len(st.session_state.tickers) < 2:
        st.info("Au minimum deux actifs sont requis pour lancer l'analyse.")

    if st.session_state.last_params != current_params:
        st.session_state.analysis_ready = False
        st.session_state.analysis_result = None

    if st.sidebar.button("🚀 Lancer l'analyse", type="primary", width='stretch'):
        if len(st.session_state.tickers)<2:
            st.warning("⚠️ Veuillez sélectionner au moins deux actifs.")
            st.stop()

        with st.spinner("📡 Téléchargement et traitement des données..."):
            end_date = date.today()
            start_date = end_date - timedelta(days=nb_jours)
            dfs = {}
            failed_tickers = []

            progress_bar = st.progress(0)
            for i, t in enumerate(st.session_state.tickers):
                try:
                    # Utilise le cache intelligent
                    data = fetch_data_with_cache(t, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        dfs[t] = data
                    else:
                        failed_tickers.append(t)
                except Exception as e:
                    st.warning(f"⚠️ Erreur pour {t}: {str(e)}")
                    failed_tickers.append(t)
                
                progress_bar.progress((i + 1) / len(st.session_state.tickers))
            
            progress_bar.empty()

            if failed_tickers:
                st.warning(f"⚠️ Actifs non téléchargés : {', '.join(failed_tickers)}")

            if not dfs:
                st.error("❌ Aucune donnée téléchargée. Vérifiez les symboles.")
                st.stop()


            # Extraction des prix de clôture
            prices_series = {}
            for ticker, data in dfs.items():
                
                # Cas DataFrame (yfinance normal)
                if isinstance(data, pd.DataFrame):
                    # Trouver colonne Close
                    close_col = None
                    for col in data.columns:
                        if isinstance(col, tuple) and col[0] == 'Close':
                            close_col = col
                            break
                        elif col == 'Close':
                            close_col = col
                            break
                    
                    if close_col is None:
                        st.warning(f"⚠️ Colonne 'Close' introuvable pour {ticker}")
                        continue

                    serie = data[close_col].dropna()
                    if len(serie) < 5:
                        st.warning(f"⚠️ Données insuffisantes pour {ticker}. Ignoré.")
                        continue

                    prices_series[ticker] = serie

                # Cas Série
                elif isinstance(data, pd.Series):
                    data = data.dropna()
                    if len(data) < 5:
                        st.warning(f"⚠️ Données insuffisantes pour {ticker}. Ignoré.")
                        continue

                    prices_series[ticker] = data

                else:
                    st.warning(f"⚠️ Format inconnu pour {ticker}. Ignoré.")
                    continue
            
            if not prices_series:
                st.error("❌ Aucune donnée de prix extraite.")
                st.stop()
            
            df = pd.DataFrame(prices_series)
            df = df.dropna()
            
            if len(df) < 30:
                st.error(f"❌ Pas assez de données communes ({len(df)} jours).")
                st.stop()

            st.session_state.analysis_result = df
            st.session_state.analysis_ready = True
            st.session_state.last_params = current_params
            st.success(f"✅ Analyse prête ! {len(df)} jours de données pour {len(df.columns)} actifs.")

    # ==========================================================
    # ONGLETS PRINCIPAUX
    # ==========================================================
    tab1, tab2, tab3 = st.tabs([
        "📊 Cours & Corrélation",
        "📈 Analyse technique",
        "🚀 Portefeuille optimisé"
    ])

    # ==========================================================
    # COURS & CORRÉLATION
    # ==========================================================
    with tab1:
        st.header("📊 Cours & Corrélation")
        if st.session_state.analysis_ready:
            df = st.session_state.analysis_result
            rendements = calcul_rendements(df)
            
            # Exclure la colonne 'Taux_réel' si elle existe
            rendements_clean = rendements.drop(columns=['Taux_réel'], errors='ignore')
            
            _, _, corr_matrix = matrices_risque_rendement(rendements_clean)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 Évolution des cours boursiers")
                tracer_cours(df)
            with col2:
                st.subheader("🔗 Matrice de corrélation")
                tracer_correlation(corr_matrix)
            
            st.subheader("📊 Statistiques des rendements")
            stats = pd.DataFrame({
                "Rendement moyen": rendements_clean.mean() * 252,
                "Volatilité annuelle": rendements_clean.std() * (252 ** 0.5),
                "Rendement min": rendements_clean.min(),
                "Rendement max": rendements_clean.max()
            })
            st.dataframe(stats.style.format("{:.2%}"), width='stretch')

            stats_actifs = statistiques_actifs(rendements, rf_rate=rf_rate)
            st.subheader("Statistiques des actifs")
            st.dataframe(stats_actifs.style.format("{:.2%}"), width='stretch')
        else:
            st.info("👈 Lancez l'analyse depuis la barre latérale.")

    # ==========================================================
    # ANALYSE TECHNIQUE
    # ==========================================================
    with tab2:
        st.header("📈 Analyse technique individuelle")
        if st.session_state.analysis_ready:
            df = st.session_state.analysis_result
            
            actif = st.selectbox("Choisissez un actif :", df.columns.tolist())
            data = df[actif].dropna().astype(float)

            if len(data) < 30:
                st.warning("Pas assez de données pour une analyse technique.")
                st.stop()

            try :

                # === Sécurité 1 : tout convertir en FLOAT ===
                rsi_vals = rsi(data).astype(float)
                macd_df = macd(data).astype(float)
                sma_df = moyennes_mobiles(data).astype(float)

                # === Sécurité 2 : colonnes propres & homogènes ===
                # Uniformisation propre des noms de colonnes SMA
                clean_cols = []
                for col in sma_df.columns:
                    col_str = str(col).upper().replace("SMA", "").strip()
                    clean_cols.append(f"SMA {col_str}")

                sma_df.columns = clean_cols
                macd_df.columns = ["MACD", "Signal", "Histogramme"][:macd_df.shape[1]]

                # === Construction du DataFrame pour les prix + SMA ===
                prix_df = pd.DataFrame({actif: data})
                chart_df = pd.concat([prix_df, sma_df], axis=1)

                # === Vérification stricte : uniquement colonnes numériques ===
                chart_df = chart_df.select_dtypes(include=["float", "int"])

                # === Affichage ===
                st.subheader(f"📊 Prix et moyennes mobiles — {actif}")
                st.line_chart(chart_df)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📉 RSI")
                    st.line_chart(rsi_vals)
                    if rsi_vals.iloc[-1] > 70:
                        st.warning("⚠️ Suracheté (RSI > 70)")
                    elif rsi_vals.iloc[-1] < 30:
                        st.info("💡 Survendu (RSI < 30)")

                with col2:
                    st.subheader("📈 MACD")
                    st.line_chart(macd_df[["MACD", "Signal"]])
                    if macd_df["MACD"].iloc[-1] > macd_df["Signal"].iloc[-1]:
                        st.success("📈 Signal haussier")
                    else:
                        st.error("📉 Signal baissier")

            except Exception as e:
                st.error(f"❌ Erreur dans le calcul des indicateurs : {str(e)}")
        else:
            st.info("👈 Lancez l'analyse depuis la barre latérale.")

    # ==========================================================
    # PORTFEUILLE OPTIMISÉ 
    # ==========================================================
    with tab3:
        st.header("Portefeuille optimisé et comparaison à l'inflation")

        if st.session_state.analysis_ready:

            df = st.session_state.analysis_result
            
            # === 1) Rendements journaliers ===
            rendements = calcul_rendements(df)
            rendements_clean = rendements.drop(columns=['Taux_réel'], errors='ignore')

            # === 2) Rendements & cov annualisés ===
            mean_returns, cov_matrix, _ = matrices_risque_rendement(rendements_clean)

            # --- rf toujours un float, jamais None ---
            rf = rf_rate if rf_rate not in (None, "") else 0.0
            rf = float(rf)

            # === 3) Portefeuille de variance minimale ===
            min_port = min_variance(mean_returns, cov_matrix, short=False, w_max=w_max)

            # === 3 bis : Portefeuille tangent (projeté sur la frontière) ===
            opt_port = tangent_on_frontier(mean_returns, cov_matrix, rf, short=False, w_max=w_max)

            # === Index marché détecté ===
            market_index = detect_market_index(st.session_state.tickers)

            # === Bêta ===
            beta_port, betas_assets = None, None
            if market_index is not None:
                beta_port, betas_assets, rendements_marche = compute_portfolio_beta(
                    weights=opt_port.weights,
                    rendements_assets=rendements_clean,
                    market_index=market_index
                )

            # ======================================================
            #  AFFICHAGE : GMV vs OPTIMAL
            # ======================================================
            col1, col2 = st.columns(2)

            # --- GMV ---
            with col1:
                st.subheader("📉 Portefeuille à variance minimale")
                st.metric("Rendement annuel", f"{min_port.mu:.2%}")
                st.metric("Volatilité", f"{min_port.sigma:.2%}")
                
                min_port.weights.index = list(mean_returns.index)
                wdf = (
                    pd.DataFrame({"Actif": min_port.weights.index,
                                "Poids": min_port.weights.values})
                    .query("Poids > 0.001")
                    .sort_values("Poids", ascending=False)
                )

                st.write("**Répartition :**")
                for _, row in wdf.iterrows():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.progress(float(row["Poids"]), text=str(row["Actif"]))
                    with c2:
                        st.write(f"**{row['Poids']:.1%}**")

            # --- OPTIMAL ---
            with col2:
                st.subheader("🚀 Portefeuille optimal (Sharpe max)")
                st.metric("Rendement annuel", f"{opt_port.mu:.2%}")
                st.metric("Volatilité", f"{opt_port.sigma:.2%}")
                st.metric("Sharpe", f"{opt_port.sharpe:.3f}")

                opt_port.weights.index = list(mean_returns.index)
                wdf = (
                    pd.DataFrame({"Actif": opt_port.weights.index,
                                "Poids": opt_port.weights.values})
                    .query("Poids > 0.001")
                    .sort_values("Poids", ascending=False)
                )

                st.write("**Répartition :**")
                for _, row in wdf.iterrows():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.progress(float(row["Poids"]), text=str(row["Actif"]))
                    with c2:
                        st.write(f"**{row['Poids']:.1%}**")

            # ======================================================
            #   FRONTIÈRE + CML
            # ======================================================
            st.markdown("---")
            st.subheader("🔍 Frontière efficiente & CML")

            df_front = efficient_frontier(mean_returns, cov_matrix, points=150, w_max=w_max)
            df_front = df_front.rename(columns={"mu": "Rendement", "sigma": "Risque"})

            fig = tracer_frontiere(
                df_front,
                rf_rate=rf,
                optimal={
                    "Rendement": opt_port.mu,
                    "Risque": opt_port.sigma,
                    "Sharpe": opt_port.sharpe
                }
            )
            st.plotly_chart(fig, width='content')

            # ======================================================
            #  STATISTIQUES
            # ======================================================
            st.markdown("---")
            stats_min = resume_portefeuille(rendements_clean, min_port.weights.values, rf_rate=rf)
            stats_opt = resume_portefeuille(rendements_clean, opt_port.weights.values, rf_rate=rf)

            st.dataframe(
                pd.DataFrame([stats_min, stats_opt], index=["Min Var", "Optimal"])
                .round(4)
                .style.format("{:.2%}")
            )

            # ======================================================
            #  BÊTA CAPM
            # ======================================================
            if beta_port is not None:
                st.subheader("📉 Bêta du portefeuille (CAPM)")
                st.metric("β", f"{beta_port:.3f}")
                st.write(classification_beta(beta_port))

                with st.expander("Bêta des actifs du portefeuille"):
                    st.dataframe(betas_assets.to_frame("Beta"), width='content')

            # ======================================================
            #  INFLATION
            # ======================================================
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rendement optimal", f"{opt_port.mu:.2%}")
            with col2:
                st.info(f"Pays : {country}")
            with col3:
                if opt_port.mu > 0.03 : st.success("Rendement > inflation")
                else : st.warning("Rendement ≈ inflation")

            st.stop()


# =============================================================
# POINT D’ENTRÉE PRINCIPAL
# =============================================================
if __name__ == "__main__":
    main()