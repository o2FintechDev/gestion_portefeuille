# ===============================================================
#   MARKOWITZ — VERSION STABLE & CORRIGÉE
# ===============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class Perf:
    mu: float
    sigma: float
    sharpe: float
    weights: pd.Series
    method: str


def _to_vec(x):
    return np.asarray(x, dtype=float).reshape(-1)


def portfolio_perf(w, mu, cov, rf=0.0):
    w = _to_vec(w)
    mu_p = float(w @ mu.values)
    sigma_p = float(np.sqrt(w @ cov.values @ w))
    sharpe = (mu_p - rf) / sigma_p if sigma_p > 1e-12 else np.nan
    return mu_p, sigma_p, sharpe


def _bounds(n, short=False, w_min=0.0, w_max=0.5):
    if short:
        return [(-1.0, 1.0)] * n
    w_min = max(0.0, w_min)
    w_max = min(1.0, w_max)
    return [(w_min, w_max)] * n


def _solve(obj, x0, Aeq=None, beq=None, bnds=None):
    cons = []
    if Aeq is not None:
        for Arow, brow in zip(Aeq, beq):
            cons.append({
                "type": "eq",
                "fun": lambda w, A=Arow, b=brow: float(A @ w - b)
            })

    res = minimize(
        obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-9}
    )
    if not res.success:
        raise RuntimeError("Optimisation échouée: " + res.message)
    return res.x


# ===============================================================
#   MIN VARIANCE
# ===============================================================

def min_variance(mu, cov, short=False) -> Perf:
    n = len(mu)
    Aeq = np.ones((1, n))
    beq = np.array([1.0])
    bnds = _bounds(n, short)
    x0 = np.repeat(1.0 / n, n)

    def obj(w):
        return w @ cov.values @ w

    w = _solve(obj, x0, Aeq, beq, bnds)
    mu_p, sigma_p, sharpe = portfolio_perf(w, mu, cov)
    return Perf(mu_p, sigma_p, sharpe, pd.Series(w, index=mu.index), "Markowitz")


# ===============================================================
#   TANGENCY PORTFOLIO — CORRIGÉ
# ===============================================================

def tangency_portfolio(mu, cov, rf, short=False, w_min=0.0, w_max=0.5) -> Perf:
    n = len(mu)
    excess = mu.values - rf

    # Si rf ≈ 0, éviter instabilités
    if abs(rf) < 1e-6:
        rf = 0.0

    bnds = _bounds(n, short, w_min, w_max)
    x0 = np.repeat(1.0 / n, n)

    def neg_sharpe(w):
        w = _to_vec(w)
        num = excess @ w
        den = np.sqrt(w @ cov.values @ w)
        if den < 1e-12:
            return 1e6
        return -(num / den)

    Aeq = np.ones((1, n))
    beq = np.array([1.0])

    w = _solve(neg_sharpe, x0, Aeq, beq, bnds)

    mu_p, sigma_p, sharpe = portfolio_perf(w, mu, cov, rf)
    return Perf(mu_p, sigma_p, sharpe, pd.Series(w, index=mu.index), "Tobin")


# ===============================================================
#   TARGET RETURN
# ===============================================================

def target_return(mu, cov, target_mu, short=False, w_min=0.0, w_max=0.5) -> Perf:
    """
    Portefeuille Markowitz pour un rendement cible donné.
    Version stabilisée :
    - Régularisation covariance
    - Correction du piège lambda (contrainte correcte)
    - Plage μ réaliste
    - Point de départ robuste
    """
    n = len(mu)
    mu_vals = mu.values.astype(float)

    # ========== Stabilisation covariance ==========
    cov_vals = cov.values.astype(float)
    cov_vals = cov_vals + 1e-8 * np.eye(n)  # régularisation
    cov = pd.DataFrame(cov_vals, index=cov.index, columns=cov.columns)

    # ========== Plage de rendement réalisable ==========
    mu_min_port = float(mu_vals.min())
    mu_max_port = float(mu_vals.max())

    # µ cible ajusté
    t = float(target_mu)
    if t <= mu_min_port:
        t = mu_min_port * 1.0001
    if t >= mu_max_port:
        t = mu_max_port * 0.9999

    # ========== Bornes ==========
    if short:
        bnds = [(-1.0, 1.0)] * n
    else:
        w_min = max(0.0, w_min)
        w_max = min(1.0, w_max)
        bnds = [(w_min, w_max)] * n

    # ========== Contraintes ==========
    def cons_sum(w):
        return np.sum(w) - 1.0

    def cons_return(w):
        return float(w @ mu_vals - t)

    cons = [
        {"type": "eq", "fun": cons_sum},
        {"type": "eq", "fun": cons_return},
    ]

    # ========== Point de départ robuste ==========
    w0 = np.repeat(1.0 / n, n)
    w0 = np.clip(w0, w_min, w_max)

    # ========== Objectif ==========
    def obj(w):
        return float(w @ cov_vals @ w)

    # ========== Optimisation ==========
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 800, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError(
            f"Optimisation échouée pour μ={t:.4f}: {res.message}"
        )

    w = res.x

    # ========== Validation ==========
    mu_p = float(w @ mu_vals)
    sigma_p = float(np.sqrt(w @ cov_vals @ w))
    sharpe = mu_p / sigma_p if sigma_p > 0 else np.nan

    if not np.isfinite(mu_p) or not np.isfinite(sigma_p):
        raise ValueError("Perf invalide (NaN/Infs détectés)")

    if abs(mu_p - t) > 0.005:  # tolérance à 0.5%
        raise ValueError(
            f"Rendement cible non atteint (visé {t:.4f}, obtenu {mu_p:.4f})"
        )

    return Perf(
        mu_p,
        sigma_p,
        sharpe,
        pd.Series(w, index=mu.index),
        "Markowitz",
    )


# ===============================================================
#   FRONTIÈRE EFFICIENTE — CORRIGÉE
# ===============================================================

def efficient_frontier(mu, cov, points=80, short=False):
    """
    Frontière efficiente Markowitz stabilisée :
    - Plage μ réaliste (GMV → μ max atteignable)
    - Résolution robuste
    - Filtrage des échecs et outliers
    - Aucun print : retourne un diagnostic propre
    """

    # === 1. Portefeuille de variance minimale (GMV) ===
    gmv = min_variance(mu, cov, short=short)
    mu_gmv = float(gmv.mu)

    # Rendement max réel (pas mu.max brut)
    mu_max = float(mu.max()) * 0.999  # sécurité

    if mu_gmv >= mu_max:
        raise ValueError("Plage μ invalide (GMV >= μ max).")

    # === 2. Grille de rendements uniquement EFFICIENTE ===
    grid = np.linspace(mu_gmv * 1.0005, mu_max, points)

    results = []
    failures = []

    # === 3. Boucle d'optimisation robuste ===
    for t in grid:
        try:
            p = target_return(mu, cov, t, short)

            # Validation stricte
            if not np.isfinite(p.sigma) or p.sigma <= 0:
                failures.append((t, "sigma non valide"))
                continue

            # Le rendement doit correspondre à la cible
            if abs(p.mu - t) > 0.01:
                failures.append((t, f"rendement non atteint ({p.mu:.4f})"))
                continue

            results.append({"mu": p.mu, "sigma": p.sigma})

        except Exception as e:
            failures.append((t, str(e)))
            continue

    if not results:
        raise ValueError("Aucun portefeuille valable trouvé (optimisation impossible).")

    # === 4. Ajouter le GMV ===
    results.insert(0, {"mu": mu_gmv, "sigma": gmv.sigma})

    df = pd.DataFrame(results)

    # === 5. Nettoyage : tri + suppression des dupes + outliers ===
    df = df.sort_values("sigma").drop_duplicates(subset=["sigma"], keep="first")

    # Filtrage des points aberrants (volatilités extrêmes)
    q99 = df["sigma"].quantile(0.99)
    df = df[df["sigma"] <= q99]

    df = df.reset_index(drop=True)

    return df

# ===============================================================
#   CML — correct
# ===============================================================

def cml_line(rf, mu_m, sigma_m, sigmas=None):
    if sigmas is None:
        sigmas = np.linspace(0, sigma_m * 2, 50)
    slope = (mu_m - rf) / sigma_m
    mu_vals = rf + slope * sigmas
    return pd.DataFrame({"sigma": sigmas, "mu": mu_vals})
# ==============================
# Monte Carlo (diagnostic uniquement)
# ==============================
def monte_carlo_portfolios(
    mean_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    rf_rate: float | None = None,
    nb_portfolios: int = 5000
):
    """
    Simule des allocations aléatoires pour visualisation.
    À utiliser UNIQUEMENT pour diagnostiquer, pas pour optimiser !
    
    Retourne:
      - df: DataFrame avec colonnes ["Rendement","Risque","Sharpe","Poids"]
      - min_port: ligne df du risque minimal
      - opt_port: ligne df du Sharpe max si rf_rate fourni
    """
    results = []
    tickers = list(mean_returns.index)
    rf = rf_rate if rf_rate is not None else 0.0

    for _ in range(nb_portfolios):
        w = np.random.random(len(tickers))
        w /= np.sum(w)

        r = float(w @ mean_returns.values)
        s = float(np.sqrt(w @ cov_matrix.values @ w))
        sharpe = (r - rf) / s if s > 1e-10 else np.nan
        results.append([r, s, sharpe, w])

    df = pd.DataFrame(results, columns=["Rendement", "Risque", "Sharpe", "Poids"])

    min_port = df.loc[df["Risque"].idxmin()]
    opt_port = df.loc[df["Sharpe"].idxmax()] if rf_rate is not None else df.loc[df["Rendement"].idxmax()]
    return df, min_port, opt_port


# ==============================
# Export CSV
# ==============================
def exporter_resultats(resultats: pd.DataFrame, chemin: str = "resultats_portefeuille.csv"):
    """Export CSV simple des résultats."""
    assert isinstance(resultats, pd.DataFrame), "❌ resultats doit être un DataFrame."
    resultats.to_csv(chemin, index=False)