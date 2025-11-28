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


def _bounds(n, short=False, w_min=0.0, w_max=1.0):
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
#   TANGENCY PORTFOLIO 
# ===============================================================

def tangency_portfolio(mu, cov, rf, short=False, w_min=0.0, w_max=1.0) -> Perf:
    n = len(mu)
    rf = 0.0 if abs(rf) < 1e-9 else float(rf)
    excess = mu.values - rf
    bnds = _bounds(n, short, w_min, w_max)
    x0 = np.repeat(1.0/n, n)

    def neg_sharpe(w):
        w = _to_vec(w)
        den = np.sqrt(w @ cov.values @ w)
        if den < 1e-12: return 1e6
        return -(excess @ w / den)

    Aeq = np.ones((1, n)); beq = np.array([1.0])
    w = _solve(neg_sharpe, x0, Aeq, beq, bnds)
    mu_p, sigma_p, sharpe = portfolio_perf(w, mu, cov, rf)
    return Perf(mu_p, sigma_p, sharpe, pd.Series(w, index=mu.index), "Tobin")



# ===============================================================
#   TARGET RETURN
# ===============================================================

def target_return(mu, cov, target_mu, short=False, w_min=0.0, w_max=1.0, w0=None) -> Perf:
    n = len(mu)
    mu_vals = mu.values.astype(float)

    cov_vals = cov.values.astype(float)
    cov_vals = cov_vals + 1e-8 * np.eye(n)

    mu_min_port = float(mu_vals.min())
    mu_max_port = float(mu_vals.max())
    t = float(target_mu)
    t = max(min(t, mu_max_port * 0.9999), mu_min_port * 1.0001)

    # bornes
    bnds = [(-1.0, 1.0)] * n if short else [(max(0.0, w_min), min(1.0, w_max))] * n

    # contraintes
    def cons_sum(w): return np.sum(w) - 1.0
    def cons_return(w): return float(w @ mu_vals - t)
    cons = [{"type":"eq","fun":cons_sum},{"type":"eq","fun":cons_return}]

    # point de départ: solution précédente si fournie
    if w0 is None:
        w0 = np.repeat(1.0/n, n)
    w0 = np.clip(w0, bnds[0][0], bnds[0][1])

    def obj(w): return float(w @ cov_vals @ w)

    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter":1000,"ftol":1e-12})

    if not res.success:
        raise RuntimeError(f"Optimisation échouée pour μ={t:.4f}: {res.message}")

    w = res.x
    mu_p  = float(w @ mu_vals)
    sigma = float(np.sqrt(w @ cov_vals @ w))
    sharpe = mu_p / sigma if sigma > 0 else np.nan

    return Perf(mu_p, sigma, sharpe, pd.Series(w, index=mu.index), "Markowitz")


# ===============================================================
#   FRONTIÈRE EFFICIENTE 
# ===============================================================

def efficient_frontier(mu, cov, points=80, short=False, w_min=0.0, w_max=1.0):
    # GMV
    gmv = min_variance(mu, cov, short=short)
    mu_gmv = float(gmv.mu)
    mu_max = float(mu.max()) * 0.999
    if mu_gmv >= mu_max:
        raise ValueError("Plage μ invalide (GMV >= μ max).")

    grid = np.linspace(mu_gmv * 1.0005, mu_max, points)

    results = []
    w_prev = gmv.weights.values

    for t in grid:
        try:
            p = target_return(mu, cov, t, short=short, w_min=w_min, w_max=w_max, w0=w_prev)
            w_prev = p.weights.values
            if not np.isfinite(p.sigma) or p.sigma <= 0: 
                continue
            results.append({"mu": p.mu, "sigma": p.sigma})
        except Exception:
            continue

    if not results:
        raise ValueError("Aucun portefeuille valable trouvé.")

    df = pd.DataFrame(results)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("sigma").drop_duplicates(subset=["sigma"], keep="first").reset_index(drop=True)

    # retirer les points non efficaces (mu non croissant avec sigma)
    df["mu_cummax"] = df["mu"].cummax()
    df = df[df["mu"] >= df["mu_cummax"] - 1e-10].drop(columns="mu_cummax").reset_index(drop=True)

    # optionnel: lissage léger (évite dents résiduelles)
    # df["mu"] = df["mu"].rolling(3, center=True, min_periods=1).mean()

    # ajouter GMV en tête
    df.loc[-1] = {"mu": mu_gmv, "sigma": gmv.sigma}
    df = df.sort_values("sigma").reset_index(drop=True)
    return df

# ===============================================================
#   CML
# ===============================================================

def cml_line(rf, mu_m, sigma_m, sigmas=None):
    if sigmas is None:
        sigmas = np.linspace(0, sigma_m * 2, 50)
    slope = (mu_m - rf) / sigma_m
    mu_vals = rf + slope * sigmas
    return pd.DataFrame({"sigma": sigmas, "mu": mu_vals})

# ===============================================================
# Monte Carlo (diagnostic uniquement)
# ===============================================================
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