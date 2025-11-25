import yfinance as yf
import pandas as pd
import yfinance.shared as yfs
import time
import random
import os
from proxies.proxies_headers import Proxies

# Récupère une liste de proxies valides
proxies_list = Proxies.proxies_selection()
user_agents = Proxies.get_browser_headers()
proxy_index = 0

def get_user_ticket():
    """Retourne uniquement le ticker. Utilisable en Streamlit."""
    ticker = input("Symbol du cours boursier : ").strip().upper()
    return ticker


def get_user_date():
    """Retourne uniquement la période. Non utilisé en Streamlit."""
    while True:
        try:
            start_year = int(input("Entrez une année de début : "))
            end_year = int(input("Entrez une année de fin : "))

            if start_year >= end_year:
                raise ValueError("L'année de début doit être inférieure à celle de fin.")

            return f"{start_year}-01-01", f"{end_year}-12-31"

        except ValueError as e:
            print("Erreur dans la saisie :", e)


def fetch_data(ticker, start_date, end_date, max_retries=5,wait_seconds=5):
    """Récupération des données via yfinance avec gestion des NaN"""

    global proxy_index

    if len(proxies_list)>0:
        for attempt in range(max_retries):
                headers = random.choice(user_agents)
                    
                # Rotation de proxy
                current_proxy = proxies_list[proxy_index % len(proxies_list)]
                proxy_index += 1

                proxies = {
                    "http": current_proxy,
                    "https": current_proxy
                    } if current_proxy else {}

                print(f"Tentative {attempt+1} avec proxy {current_proxy}")

                try:
                    # injection dans les requêtes utilisées par yfinance
                    yfs._requests_kwargs = {
                        "headers": headers,
                        "proxies": proxies,
                        "timeout": 5
                    }

                    # Téléchargement des données quotidiennes
                    print(f"Téléchargement des données pour {ticker} de {start_date} à {end_date}...")
                    data = yf.download(
                        tickers = ticker,
                        start=start_date,
                        end=end_date,
                        interval="1d",
                        auto_adjust=True,
                        progress=True,
                        threads=False
                        )
                    
                    # Vérification contenu
                    if data.empty:
                        print("Aucune donnée reçue.")
                        continue

                    # Gestion des NaN
                    nan_ratio = data.isna().mean().mean()
                    if nan_ratio > 0:
                        print(f"⚠️ {nan_ratio:.2%} de valeurs manquantes détectées pour {ticker}.")
                        data = data.fillna(method="ffill").dropna()
                        if data.empty:
                            print("⚠️ Toutes les données manquantes ont été supprimées, DataFrame vide.")
                            continue

                    print("Données téléchargées et nettoyées avec succès.")
                    return data

                except Exception as e:
                    print(f"Erreur : {e}")

                print(f"Attente de {wait_seconds}s avant prochaine tentative...")
                time.sleep(wait_seconds)

    elif (not proxies_list) or (attempt + 1 == max_retries):
        print("Aucun proxy disponible. Utilisation de l'adresse IP locale.")

        try:

            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=True,
                threads=False
            )

            if data.empty:
                print("Aucune donnée reçue.")
                return pd.DataFrame()
            # Nettoyage NaN
            nan_ratio = data.isna().mean().mean()
            if nan_ratio > 0:
                print(f"⚠️ {nan_ratio:.2%} de valeurs manquantes détectées pour {ticker}.")
                data = data.fillna(method="ffill").dropna()

            print("Données téléchargées avec succès via IP locale.")
            return data

        except Exception as e:
            print(f"Échec du téléchargement sans proxy : {e}")
            return pd.DataFrame()

def download_stock_data():
    ticker = get_user_ticket()
    start_date, end_date = get_user_date()

    # Dossier de stockage dynamique
    fichier_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(fichier_dir, exist_ok=True)

    # Chemin du fichier Excel
    xlsx_file = os.path.join(fichier_dir, f"{ticker}_{start_date[:4]}_{end_date[:4]}.xlsx")

    # Vérifier si le fichier existe déjà
    if os.path.exists(xlsx_file):
        print("Fichier déjà présent localement. Chargement depuis le disque...")
        try:
            df = pd.read_excel(xlsx_file)
            print("Aperçu des données :")
            print(df.head())
            return
        except Exception as e:
            print("Erreur lors de la lecture du fichier :", e)

    # Téléchargement des données
    data = fetch_data(ticker, start_date, end_date)
    if data.empty:
        print("Aucune donnée téléchargée. Vérifiez le symbole ou la période.")
        return

    # Sauvegarde directe en Excel
    try:
        # Supprimer les heures des dates
        data.index = data.index.date

        # Sauvegarde Excel avec largeur automatique
        with pd.ExcelWriter(xlsx_file, engine='xlsxwriter') as writer:
            data.to_excel(writer, sheet_name='Données')
            worksheet = writer.sheets['Données']
            worksheet.set_column('A:A', 20)  # Date
            worksheet.set_column('B:F', 15)  # Valeurs numériques
            
        print(f"Données sauvegardées dans : {xlsx_file}")
        print(data.head())
    except Exception as e:
        print("Erreur lors de la sauvegarde Excel :", e)

if __name__ =="__main__":
    download_stock_data()