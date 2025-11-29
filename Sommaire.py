import streamlit as st

def page_accueil():

    st.title("💼 Application de Gestion de Portefeuille")
    st.subheader("Analyse financière, optimisation et visualisation des marchés")

    # ----------------------------------------------------
    # 1) INTRODUCTION
    # ----------------------------------------------------
    st.markdown("""
    Cette application permet d’explorer, analyser et optimiser un portefeuille financier 
    grâce à des méthodes quantitatives éprouvées (Markowitz, ratio de Sharpe, frontière efficiente, etc.).  
    Elle s’adresse aux étudiants, analystes financiers et investisseurs souhaitant disposer d’un outil 
    performant pour comprendre le risque, le rendement et la structure de leurs actifs.
    """)

    st.markdown("---")

    # ----------------------------------------------------
    # 2) OBJECTIFS DE L’APPLICATION
    # ----------------------------------------------------
    st.header("🎯 Objectifs")
    st.markdown("""
    - **Analyser** les performances historiques d’actifs financiers.  
    - **Mesurer** le risque et les corrélations via des matrices dédiées.  
    - **Optimiser** un portefeuille selon Markowitz (minimum variance, portefeuille tangent).  
    - **Comparer** les rendements à l’inflation ou à un taux sans risque.  
    - **Visualiser** les tendances grâce à des indicateurs techniques (RSI, MACD, moyennes mobiles).  
    - **Interagir** de manière intuitive via une interface dynamique et rapide.
    """)

    st.markdown("---")

    # ----------------------------------------------------
    # 3) COMMENT UTILISER L’APPLICATION ?
    # ----------------------------------------------------
    st.header("⚙️ Comment ça fonctionne ?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 1. Sélection des actifs  
        - Saisissez un ou plusieurs *tickers* (AAPL, MSFT, ^FCHI…).  
        - Le moteur de suggestion vous aide à trouver les symboles valides.  

        ### 2. Paramétrage  
        - Choisissez l’horizon temporel d’analyse.  
        - Sélectionnez un **taux sans risque** (fixe ou API).  
        - Indiquez votre pays pour intégrer **l’inflation locale**.

        ### 3. Téléchargement des données  
        - Les données sont collectées via *yfinance* avec:
          - rotation de proxys  
          - mécanismes anti-blocage  
          - système de cache pour éviter des re-téléchargements inutiles  
        """)

    with col2:
        st.markdown("""
        ### 4. Analyse des données  
        - Calcul des indicateurs clés : rendements, volatilité, covariance.  
        - Extraction des indicateurs techniques (RSI, MACD, SMA…).  

        ### 5. Optimisation du portefeuille  
        - Portefeuille **à variance minimale**  
        - Portefeuille **tangent (Sharpe max)**  
        - Affichage de la **frontière efficiente**  

        ### 6. Visualisations interactives  
        - Graphiques de prix  
        - Matrices de corrélation  
        - Comparaison au taux sans risque et à l’inflation  
        - Courbes de performance cumulée  
        """)

    st.markdown("---")

    # ----------------------------------------------------
    # 4) APERÇU DES FONCTIONNALITÉS
    # ----------------------------------------------------
    st.header("📊 Ce que vous pouvez faire ici")
    st.markdown("""
    - Explorer les performances historiques du portefeuille  
    - Identifier les actifs dominants et redondants  
    - Comparer vos rendements réels à l’inflation  
    - Tester différents scénarios d’allocation  
    - Exporter les données et graphiques  
    """)

    st.markdown("---")

    # ----------------------------------------------------
    # 5) MESSAGE D’ACCUEIL
    # ----------------------------------------------------
    st.info("""
    Commencez en ajoutant vos premiers tickers dans la barre latérale.
    """)

if __name__ == "__main__":
    page_accueil()