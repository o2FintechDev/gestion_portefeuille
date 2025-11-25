import requests
import pandas as pd
from functools import lru_cache

# ===================================================================
#  CONFIG
# ===================================================================

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = "211a1d50ed6e4cc2ec71989a8002f86d"  # mettre dans .env idéalement


# ===================================================================
#  FRED GENERIC FETCH
# ===================================================================

def fred_fetch(series_id: str):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    r = requests.get(FRED_BASE, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df = df[df["value"] != "."]  # retire les NaN FRED
    df["value"] = df["value"].astype(float)

    return float(df["value"].iloc[-1]) / 100  # toutes en %


# ===================================================================
#  RISK-FREE FUNCTIONS
# ===================================================================

# ----------- EUR (alternatives disponibles FRED) --------------------

@lru_cache
def get_euro_short_rate():
    return fred_fetch("IRSTCI01EZM156N")   # Euro Short-Term Composite


# ----------- USD ----------------------------------------------------

@lru_cache
def get_fedfunds():
    return fred_fetch("DFF")

@lru_cache
def get_tbill_3m():
    return fred_fetch("TB3MS")

@lru_cache
def get_sofr():
    return fred_fetch("SOFR")


# ----------- UK -----------------------------------------------------

@lru_cache
def get_sonia():
    return fred_fetch("IUDSOIA")

@lru_cache
def get_uk_3m():
    return fred_fetch("IR3TIB01GBM156N")


# ----------- Long-term ----------------------------------------------

@lru_cache
def get_oat_10y():
    return fred_fetch("IRLTLT01FRM156N")

@lru_cache
def get_bund_10y():
    return fred_fetch("IRLTLT01DEM156N")


# ===================================================================
#  WRAPPER PRINCIPAL
# ===================================================================
def get_risk_free_rate(label: str):
    mapping = {
        # EUR
        "euro_short_rate": get_euro_short_rate,

        # USD
        "fed_funds": get_fedfunds,
        "tbill_3m": get_tbill_3m,
        "sofr": get_sofr,

        # UK
        "sonia": get_sonia,
        "uk_3m": get_uk_3m,

        # Long term
        "oat_10y": get_oat_10y,
        "bund_10y": get_bund_10y,
    }

    if label not in mapping:
        raise ValueError(f"Taux sans risque inconnu : {label}")

    return mapping[label]()

