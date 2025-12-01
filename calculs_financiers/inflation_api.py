import pandas as pd
import requests
from typing import List, Optional

EU_COUNTRIES = ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU",
    "IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"]

def get_hicp_yoy(countries: List[str]) -> Optional[pd.DataFrame]:
    """
    Récupère les taux mensuels HICP YoY depuis Eurostat pour plusieurs pays.
    """
    try:
        # Normalisation des codes pays (majuscules, suppression d'espaces)
        countries = [c.strip().upper() for c in countries if c.strip().upper() in EU_COUNTRIES]

        if not countries:
            print("Aucun code pays valide fourni. Codes valides :", EU_COUNTRIES)
            return None

        all_data = []

        for country in countries:
            url = (
                "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
                f"prc_hicp_manr?coicop=CP00&geo={country}"
            )
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            js = r.json()

            # Mapping position -> periode
            time_index = js["dimension"]["time"]["category"]["index"]
            pos_to_period = {v: k for k, v in time_index.items()}

            # Valeurs indexées
            values_dict = js["value"]

            # Reconstruction cohérente
            dates = []
            vals = []

            for pos, val in values_dict.items():
                pos = int(pos)
                if pos in pos_to_period:
                    period = pos_to_period[pos]
                    dates.append(period)
                    vals.append(val)

            # Construction DataFrame propre
            df_country = pd.DataFrame({
                "Date": pd.to_datetime(dates),
                country: vals
            }).sort_values("Date")

            all_data.append(df_country)

        # --- Fusion des DataFrames des pays ---
        df_final = all_data[0]
        for d in all_data[1:]:
            df_final = df_final.merge(d, on="Date", how="outer")

        return df_final.set_index("Date")

    except Exception as e:
        print("Erreur Eurostat:", e)
        return None


def get_inflation_label(country) -> str:
    country = country.strip().upper()  # ← normalisation
    df = get_hicp_yoy([country])
    if df is None or df.empty or country not in df.columns:
        return f"Taux d’inflation {country} indisponible"
    val = df[country].dropna().iloc[-1]
    period = df.index[-1].strftime("%Y-%m")
    return f"Taux d’inflation ({country}) — {period}: {val:.2f} % en glissement annuel"

def get_inflation_value(country: str) -> Optional[float]:
    """
    Retourne la valeur d’inflation YoY pour un pays (float en décimal).
    Exemple : 4.00 % → 0.04
    """
    country = country.strip().upper()
    df = get_hicp_yoy([country])

    if df is None or df.empty or country not in df.columns:
        return None

    val = df[country].dropna().iloc[-1]     # Ex: 4.00
    return float(val) / 100                 # Convertit en 0.04
