from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize

# ===================================================================
# STRUCTURE DE SORTIE
# ===================================================================

@dataclass
class Perf:
    mu: float
    sigma: float
    sharpe: float
    weights: pd.Series
    method: str


# ===================================================================
# UTILITAIRES
# ===================================================================

def _to_vec(x):
    return np.asarray(x, float).reshape(-1)


def _risk(w, cov):
    """Volatilité racine(wᵀΣw)."""
    return float(np.sqrt(w @ cov @ w))


def _bounds(n, short=False, w_min=0.0, w_max=1.0):
    return [(-1, 1)] * n if short else [(max(0, w_min), min(1, w_max))] * n


def _cons_sum(n):
    """Somme des poids = 1."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _cons_return(mu_vals, target_mu):
    """Rendement cible wᵀμ = t."""
    return {"type": "eq", "fun": lambda w: float(w @ mu_vals - target_mu)}


def _solve(obj, x0, bounds, constraints):
    res = minimize(obj, x0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def _perf_from_weights(w, mu_vals, cov_vals, rf, name):
    mu_p = float(w @ mu_vals)
    sigma = _risk(w, cov_vals)
    sharpe = (mu_p - rf) / sigma if sigma > 1e-12 else np.nan
    return Perf(mu_p, sigma, sharpe, pd.Series(w), name)


# ===================================================================
# PORTFEUILLES DE BASE
# ===================================================================

def min_variance(mu, cov, short=False,w_min=0.0, w_max=1.0) -> Perf:
    n = len(mu)
    mu_vals = mu.values
    cov_vals = cov.values
    x0 = np.full(n, 1/n)
    bounds = _bounds(n, short, w_min, w_max)

    obj = lambda w: float(w @ cov_vals @ w)
    w = _solve(obj, x0, bounds, [_cons_sum(n)])
    return _perf_from_weights(w, mu_vals, cov_vals, 0.0, "GMV")


def tangency_portfolio(mu, cov, rf=0.0, short=False, w_min=0.0, w_max=1.0) -> Perf:
    n = len(mu)
    mu_vals = mu.values
    cov_vals = cov.values
    x0 = np.full(n, 1/n)
    rf = float(rf)

    bounds = _bounds(n, short, w_min, w_max)
    excess = mu_vals - rf

    def neg_sharpe(w):
        w = _to_vec(w)
        sig = _risk(w, cov_vals)
        return -(excess @ w / sig) if sig > 1e-12 else 1e6

    w = _solve(neg_sharpe, x0, bounds, [_cons_sum(n)])
    return _perf_from_weights(w, mu_vals, cov_vals, rf, "Tangent")


def target_return(mu, cov, target_mu, short=False, w_min=0.0, w_max=1.0, w0=None) -> Perf:
    n = len(mu)
    mu_vals = mu.values.astype(float)
    cov_vals = cov.values.astype(float) + 1e-8 * np.eye(n)

    # bornes μ
    t = float(target_mu)
    t = min(max(t, mu_vals.min()*1.0001), mu_vals.max()*0.9999)

    bounds = _bounds(n, short, w_min, w_max)
    x0 = w0 if w0 is not None else np.full(n, 1/n)

    obj = lambda w: float(w @ cov_vals @ w)

    constraints = [_cons_sum(n), _cons_return(mu_vals, t)]
    w = _solve(obj, x0, bounds, constraints)
    return _perf_from_weights(w, mu_vals, cov_vals, 0.0, "TargetReturn")


# ===================================================================
# TANGENT PROJETÉ SUR FRONTIÈRE
# ===================================================================
def tangent_on_frontier(mu, cov, rf, short=False,  w_min=0.0, w_max=1.0):
    # 1) tangent brut
    raw = tangency_portfolio(mu, cov, rf, short=short, w_min=w_min, w_max=w_max)

    # 2) projection frontière
    proj = target_return(
        mu, cov,
        target_mu=raw.mu,      
        short=short,
        w_min=w_min,
        w_max=w_max,
        w0=raw.weights.values  # accélère convergence
    )

    # 3) recalcul du Sharpe sur le point projeté
    proj.sharpe = (proj.mu - rf) / proj.sigma

    return proj



# ===================================================================
# FRONTIÈRE EFFICIENTE
# ===================================================================
def efficient_frontier(mu, cov, points=100, short=False, w_min=0.0, w_max=1.0):
    gmv = min_variance(mu, cov, short, w_min, w_max)
    mu_gmv = gmv.mu

    mu_vals = mu.values
    mu_max = float(mu_vals.max()) * 0.999

    grid = np.linspace(mu_gmv * 1.001, mu_max, points)

    rows = []
    w_prev = gmv.weights.values

    for t in grid:
        try:
            p = target_return(mu, cov, t, short=short, w_min=w_min, w_max=w_max, w0=w_prev)
            rows.append((p.mu, p.sigma))
            w_prev = p.weights.values
        except:
            pass

    df = pd.DataFrame(rows, columns=["mu", "sigma"]).sort_values("sigma")

    # Enveloppe convexe supérieure
    df["mu_cummax"] = df["mu"].cummax()
    df = df[df["mu"] >= df["mu_cummax"] - 1e-12]
    df = df.drop(columns="mu_cummax")

    # Ajouter GMV
    df.loc[-1] = {"mu": mu_gmv, "sigma": gmv.sigma}
    df = df.sort_values("sigma").reset_index(drop=True)

    return df


# ===================================================================
# CML
# ===================================================================

def cml_line(rf, mu_t, sigma_t, n=100):
    sigmas = np.linspace(0, sigma_t*1.5, n)
    slope = (mu_t - rf)/sigma_t
    return pd.DataFrame({"sigma": sigmas,
                         "mu": rf + slope * sigmas})


# ===================================================================
# MONTE CARLO (diagnostic)
# ===================================================================

def monte_carlo_portfolios(mean_returns, cov_matrix, rf_rate=0.0, nb_portfolios=5000):
    n = len(mean_returns)
    results = []

    for _ in range(nb_portfolios):
        w = np.random.random(n)
        w /= w.sum()

        r = float(w @ mean_returns.values)
        s = float(np.sqrt(w @ cov_matrix.values @ w))
        sharpe = (r - rf_rate)/s if s > 1e-12 else np.nan
        results.append([r, s, sharpe, w])

    df = pd.DataFrame(results,
                      columns=["Rendement","Risque","Sharpe","Poids"])
    return df
