import requests
import yaml
import pandas as pd
import os
import random
from io import StringIO
import threading
import socket

# ==========================================================
# Détection Streamlit Cloud (où Selenium N'EXISTE PAS)
# ==========================================================
IS_STREAMLIT = "STREAMLIT_RUNTIME" in os.environ

# ==========================================================
# Import Selenium et webdriver_manager uniquement en local
# ==========================================================
try:
    if not IS_STREAMLIT:
        from selenium import webdriver
        from selenium.webdriver.firefox.service import Service as FirefoxService
        from selenium.webdriver.firefox.options import Options
        from webdriver_manager.firefox import GeckoDriverManager
        SELENIUM_AVAILABLE = True
    else:
        SELENIUM_AVAILABLE = False
except Exception:
    SELENIUM_AVAILABLE = False


class Proxies:

    # url de vérification des proxies
    test_url = "https://httpbin.org/ip"

    @staticmethod
    def get_browser_headers():
        """Retourne une liste de headers complets tirés de tous les navigateurs"""
        file_path = os.path.join(os.path.dirname(__file__), "headers.yml")

        with open(file_path, "r") as f_headers:
            browser_headers = yaml.safe_load(f_headers)

        headers_list = []
        for browser, headers in browser_headers.items():
            formatted = {k.lower(): v for k, v in headers.items()}
            headers_list.append(formatted)

        return headers_list

    # Récupération de la liste des proxies par scrapping
    @staticmethod
    def scraping_proxies():
        """Récupération de proxies HTTPS via scraping, avec gestion d'erreur et sauvegarde locale"""

        # Si Streamlit → aucun proxy (pas de Selenium)
        if IS_STREAMLIT:
            return []

        url = "https://free-proxy-list.net"
        try:
            # Test DNS
            socket.gethostbyname("free-proxy-list.net")

            # Téléchargement du tableau HTML
            proxy_response = requests.get(url, timeout=10)
            proxy_response.raise_for_status()

            proxy_list = pd.read_html(StringIO(proxy_response.text))[0]
            proxy_list["url"] = "http://" + proxy_list["IP Address"] + ":" + proxy_list["Port"].astype(str)
            https_proxies = proxy_list[proxy_list["Https"].str.contains("yes", case=False)]
            proxies = https_proxies["url"].tolist()

            return proxies

        except Exception as e:
            print(f"Erreur de connexion ou de parsing : {e}")
            backup_file = os.path.join(os.path.dirname(__file__), "proxies_backup.txt")
            if os.path.exists(backup_file):
                print("→ Utilisation de la liste locale de secours.")
                with open(backup_file) as f:
                    return [line.strip() for line in f if line.strip()]
            else:
                print("Aucune source de proxy disponible.")
                return []

    @staticmethod
    def validate_proxy(proxy_url, headers, good_proxies):
        """ Vérification et validation des proxies supportant le HTTPS """
        proxies = {"http": proxy_url, "https": proxy_url}
        try:
            response = requests.get(Proxies.test_url, headers=headers, proxies=proxies, timeout=2)

            if response.status_code == 200 and "origin" in response.json():
                good_proxies.append(proxy_url)

        except Exception:
            pass

    @staticmethod
    def proxies_selection():
        """ Selection des proxies """

        # 🚫 Sur Streamlit → retour liste vide → pas de Selenium → pas de proxies
        if not SELENIUM_AVAILABLE:
            print("⚠️ Selenium indisponible → Pas de proxies (Streamlit Cloud).")
            return []

        # Initialisation
        good_proxies = []
        threads = []
        headers = random.choice(Proxies.get_browser_headers())
        https_proxies = Proxies.scraping_proxies()

        # Vérification multithread
        for proxy_url in https_proxies:
            thread = threading.Thread(target=Proxies.validate_proxy, args=(proxy_url, headers, good_proxies))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"{len(good_proxies)} proxies valides trouvés.")
        return good_proxies

    @staticmethod
    def using_proxies():
        """Utilisation des proxies valides avec Selenium"""

        if not SELENIUM_AVAILABLE:
            print("⚠️ Selenium indisponible (Streamlit Cloud) → using_proxies() désactivé.")
            return []

        good_proxies = Proxies.proxies_selection()

        if not good_proxies:
            print("Aucun proxy valide trouvé.")
            return []

        for proxy_url in good_proxies:
            options = Options()
            options.add_argument(f"--proxy-server={proxy_url}")

            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)

            try:
                driver.get(Proxies.test_url)
                print("Accès via proxy réussi avec :", proxy_url)

            except Exception as e:
                print("Erreur avec proxy :", proxy_url, e)

            finally:
                driver.quit()

        return good_proxies
