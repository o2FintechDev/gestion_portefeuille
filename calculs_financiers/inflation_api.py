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
        countries = [c.strip().upper() for c in countries]
        countries = [c for c in countries if c in EU_COUNTRIES]

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

            periods = list(js["dimension"]["time"]["category"]["index"].keys())
            values = list(js["value"].values())

            df_country = pd.DataFrame({
                "Date": pd.to_datetime(periods),
                country: values
            })
            all_data.append(df_country)

        df = all_data[0]
        for d in all_data[1:]:
            df = df.merge(d, on="Date", how="outer")

        df = df.sort_values("Date").set_index("Date")
        return df

    except Exception as e:
        print("Erreur Eurostat:", e)
        return None


def get_inflation_label(country="FR") -> str:
    country = country.strip().upper()  # ← normalisation
    df = get_hicp_yoy([country])
    if df is None or df.empty or country not in df.columns:
        return f"Taux d’inflation {country} indisponible"
    val = df[country].dropna().iloc[-1]
    period = df.index[-1].strftime("%Y-%m")
    return f"Taux d’inflation ({country}) — {period}: {val:.2f} % en glissement annuel"