import pandas as pd
import numpy as np

# ==============================
# 1. Calcul des rendements / rentabilités
# ==============================
def calcul_rendements(df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Calcule les rendements (ou rentabilités) à partir des prix d’actifs.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Données de prix ajustés (index = dates, colonnes = tickers)
    inflation : float
        Taux d'inflation annuel (ex: 0.02 = 2%)
    method : str
        "log"  -> rendements logarithmiques (par défaut)
        "simple" -> rendements simples (pct_change)

    Retour
    ------
    pd.DataFrame
        Rendements journaliers avec une colonne 'Taux_réel' (moyenne journalière - inflation/252)
    """
    assert isinstance(df, pd.DataFrame), "❌ df doit être un DataFrame pandas."

    # Gestion des valeurs manquantes
    df = df.ffill().dropna()

    # Calcul des rendements
    if method == "log":
        rendements = np.log(df / df.shift(1))
    else:
        rendements = df.pct_change()

    rendements = rendements.dropna()

    return rendements


# ==============================
# 2. Risque individuel (volatilité annuelle par actif)
# ==============================
def risque_actifs(rendements: pd.DataFrame) -> pd.Series:
    """
    Calcule la volatilité annuelle de chaque actif.
    σ_i = std(r_i) * sqrt(252)
    """
    assert isinstance(rendements, pd.DataFrame), "❌ rendements doit être un DataFrame."
    return rendements.std() * np.sqrt(252)


# ==============================
# 3. Matrices de rendement, covariance et corrélation
# ==============================
def matrices_risque_rendement(rendements: pd.DataFrame, annualisation: int = 252):

    r = rendements.copy()
    r = r.select_dtypes(include=[np.number])
    r = r.drop(columns=[c for c in r.columns 
                        if c.lower() in {"taux_réel", "taux_reel"}],
               errors="ignore")

    # Nettoyage cohérent
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    if r.shape[1] < 2:
        raise ValueError("Au moins deux actifs valides sont requis.")

    mu = r.mean() * annualisation
    cov = r.cov() * annualisation
    corr = r.corr()

    return mu, cov, corr


# ==============================
# 4. Rendement réel (corrigé de l'inflation)
# ==============================
def rendement_reel(r_nominal: float, inflation: float) -> float:
    """
    Calcule le rendement réel à partir du rendement nominal et du taux d’inflation.
    
    Formule : (1 + r_nominal) / (1 + inflation) - 1
    """
    assert isinstance(r_nominal, (float, int)), "❌ r_nominal doit être un nombre."
    assert isinstance(inflation, (float, int)), "❌ inflation doit être un nombre."
    return (1 + r_nominal) / (1 + inflation) - 1


# ==============================
# 5. Résumé global des actifs (utile avant Markowitz)
# ==============================
def resume_risque_rendement(rendements: pd.DataFrame, inflation: float = 0.0) -> pd.DataFrame:
    """
    Fournit un tableau synthétique des statistiques clés par actif :
    - rendement annuel
    - risque (σ annuel)
    - rendement réel (corrigé de l’inflation)
    """
    mu, cov, corr = matrices_risque_rendement(rendements)
    sigma = risque_actifs(rendements)
    mu_reel = rendement_reel(mu, inflation)

    stats = pd.DataFrame({
        "Rendement annuel": mu,
        "Risque (σ)": sigma,
        "Rendement réel": mu_reel
    })

    return stats.round(4)
