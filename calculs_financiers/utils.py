import pandas as pd
import numpy as np
from scipy.stats import norm

# ==============================
# 1. Volatilité et rendement
# ==============================
def calcul_volatilite(rendements: pd.DataFrame) -> pd.Series:
    """Volatilité annuelle par actif."""
    return rendements.std() * np.sqrt(252)


def volatilite_portefeuille(rendements: pd.DataFrame, poids: np.ndarray) -> float:
    """Volatilité annuelle du portefeuille pondéré."""
    cov = rendements.cov() * 252
    return float(np.sqrt(np.dot(poids.T, np.dot(cov, poids))))


def rendement_portefeuille(rendements: pd.DataFrame, poids: np.ndarray) -> pd.Series:
    """Rendement quotidien pondéré du portefeuille."""
    return rendements.dot(poids)


# ==============================
# 2. Ratios de performance
# ==============================
def ratio_sharpe(rendements: pd.Series, rf_rate: float = 0.0) -> float:
    """Sharpe = (E[R] - Rf) / σ, rf_rate doit être en décimal (ex: 0.03)."""
    if rf_rate is None or isinstance(rf_rate, str):
        rf_rate = 0.0

    # Sécurise encore au cas où
    if rf_rate > 1:
        rf_rate = rf_rate / 100

    mu = rendements.mean() * 252
    sigma = rendements.std() * np.sqrt(252)

    if sigma == 0:
        return np.nan

    return (mu - rf_rate) / sigma



def ratio_sortino(rendements: pd.Series, rf_rate: float = 0.0) -> float:
    """Sortino = (E[R] - Rf) / σ_down."""
    if rf_rate is None or isinstance(rf_rate, str):
        rf_rate = 0.0

    # Sécurise encore au cas où
    if rf_rate > 1:
        rf_rate = rf_rate / 100

    downside = rendements[rendements < 0].std() * np.sqrt(252)
    exc = rendements.mean() * 252 - rf_rate
    return exc / downside if downside != 0 else np.nan


def ratio_treynor(rendements: pd.Series, beta: float, rf_rate: float = 0.0) -> float:
    """Treynor = (E[R] - Rf) / β."""
    
    if rf_rate is None or isinstance(rf_rate, str):
        rf_rate = 0.0

    # Sécurise encore au cas où
    if rf_rate > 1:
        rf_rate = rf_rate / 100

    exc = rendements.mean() * 252 - rf_rate
    return exc / beta if beta != 0 else np.nan


# ==============================
# 3. Risques extrêmes
# ==============================
def value_at_risk_annuel(mu_ann: float, sigma_ann: float, alpha: float = 0.05) -> float:
    """
    VaR annuelle paramétrique sous hypothèse normale.
    Retourne une valeur en décimal (ex : -0.18 = -18%).
    """
    z = norm.ppf(alpha)
    return mu_ann + z * sigma_ann


def expected_shortfall_annuel(mu_ann: float, sigma_ann: float, alpha: float = 0.05) -> float:
    """
    ES annuelle paramétrique sous hypothèse normale.
    Retourne une valeur en décimal (ex : -0.25 = -25%).
    """
    z = norm.ppf(alpha)
    phi = norm.pdf(z)
    return mu_ann - sigma_ann * (phi / alpha)

# ==============================
# 4. Statistiques globales par actif
# ==============================
def statistiques_actifs(rendements: pd.DataFrame, rf_rate: float = 0.0) -> pd.DataFrame:
    """Résumé statistique annuel pour chaque actif."""
    
    stats = pd.DataFrame(index=rendements.columns)

    # µ annuel & σ annuel
    stats["Rendement annuel"] = rendements.mean() * 252
    stats["Volatilité"] = rendements.std() * np.sqrt(252)

    # === Ratios (Sharpe, Sortino) ===
    stats["Sharpe"] = [
        ratio_sharpe(rendements[col], rf_rate)
        for col in rendements.columns
    ]

    stats["Sortino"] = [
        ratio_sortino(rendements[col], rf_rate)
        for col in rendements.columns
    ]

    # === VaR & ES ANNUELLES ===
    stats["VaR 5%"] = [
        value_at_risk_annuel(stats.loc[col, "Rendement annuel"],
                             stats.loc[col, "Volatilité"])
        for col in stats.index
    ]

    stats["ES 5%"] = [
        expected_shortfall_annuel(stats.loc[col, "Rendement annuel"],
                                  stats.loc[col, "Volatilité"])
        for col in stats.index
    ]

    return stats.round(4)


# ==============================
# 5. Statistiques du portefeuille global
# ==============================
def resume_portefeuille(rendements: pd.DataFrame, poids: np.ndarray, rf_rate: float = 0.0) -> dict:
    """
    Résumé global du portefeuille pondéré :
    - rendement annuel, volatilité annuelle, Sharpe, Sortino, VaR annuelle, ES annuelle.
    """

    # Rendement du portefeuille
    port_ret = rendement_portefeuille(rendements, poids)

    # Annualisation
    mu_ann = port_ret.mean() * 252
    sigma_ann = port_ret.std() * np.sqrt(252)

    # === Ratios à partir de TES fonctions ===
    sharpe = ratio_sharpe(port_ret, rf_rate)
    sortino = ratio_sortino(port_ret, rf_rate)

    # VaR & ES ANNUELLES
    var_ann = value_at_risk_annuel(mu_ann, sigma_ann)
    es_ann  = expected_shortfall_annuel(mu_ann, sigma_ann)

    return {
        "Rendement annuel": mu_ann,
        "Volatilité": sigma_ann,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "VaR 5%": var_ann,
        "ES 5%": es_ann,
    }

def diagnostiquer_donnees(mu, cov):
    """
    NOUVEAU : Fonction de diagnostic pour détecter les problèmes d'échelle
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC DES DONNÉES")
    print("="*60)
    
    print(f"\n📊 Rendements attendus (μ):")
    print(f"  Min  : {float(mu.min()):.4f} ({float(mu.min())*100:.2f}%)")
    print(f"  Max  : {float(mu.max()):.4f} ({float(mu.max())*100:.2f}%)")
    print(f"  Mean : {float(mu.mean()):.4f} ({float(mu.mean())*100:.2f}%)")
    
    volatilites = np.sqrt(np.diag(cov.values))
    print(f"\n📈 Volatilités (σ):")
    print(f"  Min  : {float(volatilites.min()):.4f} ({float(volatilites.min())*100:.2f}%)")
    print(f"  Max  : {float(volatilites.max()):.4f} ({float(volatilites.max())*100:.2f}%)")
    print(f"  Mean : {float(volatilites.mean()):.4f} ({float(volatilites.mean())*100:.2f}%)")
    
    print(f"\n🔗 Matrice de corrélation:")
    # Calculer la matrice de corrélation via pandas pour éviter les problèmes
    corr_matrix = pd.DataFrame(cov).corr()
    # Extraire uniquement la partie triangulaire inférieure (sans diagonale) pour stats
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_values = corr_matrix.mask(mask).values.flatten()
    corr_values = corr_values[~np.isnan(corr_values)]
    
    print(f"  Min  : {float(np.min(corr_values)):.4f}")
    print(f"  Max  : {float(np.max(corr_values)):.4f}")
    print(f"  Mean : {float(np.mean(corr_values)):.4f}")
    
    # Vérification des unités
    if float(mu.max()) > 1.0:
        print("\n⚠️  ALERTE : Rendements > 100% → Vérifier les unités!")
    if float(volatilites.max()) > 2.0:
        print("\n⚠️  ALERTE : Volatilité > 200% → Données suspectes!")
    
    print("="*60 + "\n")

def verifier_coherence_tangent(opt_port, mean_returns, cov_matrix, rf_rate):
    """
    Vérifie que le portefeuille tangent calculé est cohérent
    """
    print("\n" + "="*60)
    print("VÉRIFICATION DU PORTEFEUILLE TANGENT")
    print("="*60)
    
    # Recalculer les métriques à partir des poids
    w = opt_port.weights.values
    mu_recalc = float(w @ mean_returns.values)
    sigma_recalc = float(np.sqrt(w @ cov_matrix.values @ w))
    sharpe_recalc = (mu_recalc - rf_rate) / sigma_recalc if sigma_recalc > 0 else np.nan
    
    print(f"\n📊 Portefeuille tangent :")
    print(f"  Rendement (fourni)   : {opt_port.mu:.4f} ({opt_port.mu*100:.2f}%)")
    print(f"  Rendement (recalculé): {mu_recalc:.4f} ({mu_recalc*100:.2f}%)")
    print(f"  Écart                : {abs(opt_port.mu - mu_recalc):.6f}")
    
    print(f"\n  Risque (fourni)      : {opt_port.sigma:.4f} ({opt_port.sigma*100:.2f}%)")
    print(f"  Risque (recalculé)   : {sigma_recalc:.4f} ({sigma_recalc*100:.2f}%)")
    print(f"  Écart                : {abs(opt_port.sigma - sigma_recalc):.6f}")
    
    print(f"\n  Sharpe (fourni)      : {opt_port.sharpe:.4f}")
    print(f"  Sharpe (recalculé)   : {sharpe_recalc:.4f}")
    print(f"  Écart                : {abs(opt_port.sharpe - sharpe_recalc):.6f}")
    
    # Alertes
    if abs(opt_port.mu - mu_recalc) > 0.01:
        print("\n⚠️  ALERTE : Incohérence majeure sur le rendement!")
    if abs(opt_port.sigma - sigma_recalc) > 0.01:
        print("\n⚠️  ALERTE : Incohérence majeure sur le risque!")
    if abs(opt_port.sharpe - sharpe_recalc) > 0.1:
        print("\n⚠️  ALERTE : Incohérence majeure sur le Sharpe!")
    
    print("="*60 + "\n")
    
    return mu_recalc, sigma_recalc, sharpe_recalc

def detect_market_index(tickers: list[str]) -> str:
    """
    Détecte automatiquement l'indice de marché à partir de la liste des tickers.
    - Si un ticker commence par "^", on le prend comme indice de marché
    - Sinon on prend le premier ticker appartenant à INDICES_MARCHE
    - Sinon fallback = S&P500 (^GSPC)
    """

    # 1) Un ticker commence par ^
    for t in tickers:
        if t.startswith("^"):
            return t
    return None
