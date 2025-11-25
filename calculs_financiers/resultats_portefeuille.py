import pandas as pd
import numpy as np

def construire_resultats_df(min_port: dict, opt_port: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Construit un DataFrame récapitulatif des portefeuilles optimaux.
    Accepte dicts contenant Rendement, Risque, Sharpe, Poids.
    """
    def normaliser_poids(poids, tickers):
        if isinstance(poids, (pd.Series, dict)):
            return dict(poids)
        elif isinstance(poids, np.ndarray):
            if tickers:
                return {f"Poids_{t}": w for t, w in zip(tickers, poids)}
            return {f"Actif_{i}": w for i, w in enumerate(poids)}
        return {}

    poids_min = normaliser_poids(min_port.get("Poids"), tickers)
    poids_opt = normaliser_poids(opt_port.get("Poids"), tickers)

    df = pd.DataFrame([
        {
            "Type": "Min Variance",
            "Rendement": min_port["Rendement"],
            "Risque": min_port["Risque"],
            "Sharpe": min_port.get("Sharpe", None),
            **poids_min
        },
        {
            "Type": "Optimal",
            "Rendement": opt_port["Rendement"],
            "Risque": opt_port["Risque"],
            "Sharpe": opt_port["Sharpe"],
            **poids_opt
        }
    ])

    return df
