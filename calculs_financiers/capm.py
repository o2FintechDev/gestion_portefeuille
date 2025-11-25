from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

def beta(
    asset_ret: pd.Series,
    market_ret: pd.Series,
    method: Literal["cov", "reg"] = "cov"
) -> float:
    asset_ret, market_ret = asset_ret.align(market_ret, join="inner")
    asset = asset_ret.dropna()
    market = market_ret.reindex(asset.index).dropna()
    asset = asset.reindex(market.index)
    if len(market) < 3:
        return np.nan

    if method == "cov":
        cov = np.cov(asset.values, market.values, ddof=1)[0, 1]
        var = np.var(market.values, ddof=1)
        return float(cov / var) if var > 0 else np.nan
    else:  # OLS: asset = a + b*market + e
        X = np.vstack([np.ones(len(market)), market.values]).T
        y = asset.values
        # beta = (X'X)^-1 X'y
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(b[1])

def portfolio_beta(weights: pd.Series | np.ndarray, betas: pd.Series | np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    b = np.asarray(betas, dtype=float).reshape(-1)
    if w.shape != b.shape:
        raise ValueError("Dimensions incompatibles pour w et betas")
    return float(np.dot(w, b))

def capm_expected_return(rf: float, beta_val: float, market_return: float) -> float:
    """
    SML: E[R_i] = rf + beta_i * (E[R_m] - rf)
    """
    return rf + beta_val * (market_return - rf)

def sml_points(
    betas: np.ndarray | pd.Series,
    rf: float,
    market_return: float
) -> pd.DataFrame:
    betas = np.asarray(betas, dtype=float)
    exp = rf + betas * (market_return - rf)
    return pd.DataFrame({"beta": betas, "mu": exp})

def classification_beta(beta_value: float) -> str:
    if np.isnan(beta_value):
        return "Indéterminé"
    if beta_value < 0.8:
        return "Portefeuille défensif (β < 0.8)"
    if beta_value <= 1.2:
        return "Portefeuille neutre / market-like (0.8 ≤ β ≤ 1.2)"
    return "Portefeuille agressif (β > 1.2)"

import pandas as pd
import numpy as np
from calculs_financiers.capm import beta, portfolio_beta


def compute_portfolio_beta(
    weights: pd.Series,
    rendements_assets: pd.DataFrame,
    market_index: str
):
    """
    Version ultra propre :
    - Aucun téléchargement
    - Utilise uniquement les rendements calculés dans rendements_clean
    - Le marché doit être un ticker présent dans rendements_assets
    """

    # 1. Vérification
    if market_index not in rendements_assets.columns:
        raise ValueError(f"L'indice de marché {market_index} n'est pas dans les rendements.")

    # 2. Rendements du marché
    rendements_marche = rendements_assets[market_index].dropna()

    # 3. Calcul des bêtas individuels
    betas_actifs = {}
    for col in rendements_assets.columns:
        if col == market_index:
            continue  # on ne calcule pas beta(market, market)
        
        betas_actifs[col] = beta(
            asset_ret=rendements_assets[col],
            market_ret=rendements_marche,
            method="reg"
        )

    betas_actifs = pd.Series(betas_actifs)

    # 4. Bêta du portefeuille
    w = weights.drop(index=[market_index], errors="ignore")
    b = betas_actifs.loc[w.index]

    beta_portefeuille = float(np.dot(w.values, b.values))

    return beta_portefeuille, betas_actifs, rendements_marche
