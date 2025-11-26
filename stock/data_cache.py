import pickle
import os
from datetime import datetime
import pandas as pd
import yfinance as yf
import streamlit as st
from stock.stock_data_yfinance import fetch_data

# ==========================================================
# DÉTECTION SELENIUM
# ==========================================================
try:
    import selenium
    SELENIUM_AVAILABLE = True
except ModuleNotFoundError:
    SELENIUM_AVAILABLE = False

def fetch_with_yfinance_direct(ticker, start_date, end_date):
    """Fallback sans proxy si Selenium indisponible."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Aucune donnée trouvée via yfinance pour {ticker}")
    return df

class StockDataCache:
    """Gestionnaire de cache intelligent pour données boursières avec gestion de périodes"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, ticker):
        """Retourne le chemin du fichier cache pour un ticker"""
        return os.path.join(self.cache_dir, f"{ticker}.pkl")
    
    def _get_metadata_path(self, ticker):
        """Retourne le chemin du fichier métadonnées"""
        return os.path.join(self.cache_dir, f"{ticker}_meta.pkl")
    
    def get_cached_data(self, ticker, start_date, end_date, silent=True):
        """
        Récupère les données du cache si disponibles pour la période demandée
        
        Args:
            silent: Si True, aucun message n'est affiché (mode silencieux)
        
        Returns:
            tuple: (data, needs_download)
            - data: DataFrame avec les données disponibles (peut être vide)
            - needs_download: tuple (new_start_date, new_end_date) ou None si tout est en cache
        """
        cache_path = self._get_cache_path(ticker)
        meta_path = self._get_metadata_path(ticker)
        
        # Conversion en datetime.date si nécessaire
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Si pas de cache, tout télécharger
        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return pd.DataFrame(), (start_date, end_date)
        
        try:
            # Charger les métadonnées
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            
            cached_start = metadata['start_date']
            cached_end = metadata['end_date']
            
            # Charger les données
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Cas 1: Le cache couvre entièrement la période demandée
            if cached_start <= start_date and cached_end >= end_date:
                # Filtrer les données pour la période exacte
                mask = (cached_data.index.date >= start_date) & (cached_data.index.date <= end_date)
                filtered_data = cached_data[mask]
                
                return filtered_data, None
            
            # Cas 2: Période demandée dépasse le cache - retélécharger tout
            elif start_date < cached_start or end_date > cached_end:
                # Calculer la nouvelle période à télécharger (plus large)
                new_start = min(start_date, cached_start)
                new_end = max(end_date, cached_end)
                
                return pd.DataFrame(), (new_start, new_end)
            
            # Cas 3: Autre situation - retélécharger
            else:
                return pd.DataFrame(), (start_date, end_date)
                
        except Exception:
            return pd.DataFrame(), (start_date, end_date)
    
    def save_to_cache(self, ticker, data, start_date, end_date, silent=True):
        """
        Sauvegarde les données dans le cache avec métadonnées
        
        Args:
            silent: Si True, aucun message n'est affiché (mode silencieux)
        """
        if data.empty:
            return
        
        cache_path = self._get_cache_path(ticker)
        meta_path = self._get_metadata_path(ticker)
        
        # Conversion en datetime.date si nécessaire
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        try:
            # Sauvegarder les données
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Sauvegarder les métadonnées
            metadata = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now(),
                'rows': len(data),
                'columns': list(data.columns) if isinstance(data, pd.DataFrame) else ['Close']
            }
            
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
        except Exception:
            pass  # Échec silencieux
    
    def clear_cache(self, ticker=None, silent=False):
        """
        Efface le cache (tout ou pour un ticker spécifique)
        
        Args:
            ticker: Si spécifié, efface seulement ce ticker
            silent: Si True, aucun message n'est affiché
        """
        if ticker:
            files = [
                self._get_cache_path(ticker),
                self._get_metadata_path(ticker)
            ]
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
            if not silent:
                print(f"🗑️ Cache effacé pour {ticker}")
        else:
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
            if not silent:
                print("🗑️ Tout le cache effacé")
    
    def get_cache_info(self):
        """Retourne les infos sur le cache existant"""
        info = []
        for f in os.listdir(self.cache_dir):
            if f.endswith('_meta.pkl'):
                meta_path = os.path.join(self.cache_dir, f)
                try:
                    with open(meta_path, 'rb') as file:
                        metadata = pickle.load(file)
                        info.append(metadata)
                except:
                    pass
        return info


# ==========================================================
# INSTANCE GLOBALE DU CACHE
# ==========================================================

cache_manager = StockDataCache()

# ==========================================================
# FETCH DATA AVEC SELENIUM → SINON FALLBACK YFINANCE
# ==========================================================

def fetch_data_with_cache(ticker, start_date, end_date, max_retries=5, wait_seconds=5):
    """
    Version améliorée de fetch_data avec gestion intelligente du cache (mode silencieux)
    """
    
    
    # Vérifier le cache (mode silencieux)
    cached_data, needs_download = cache_manager.get_cached_data(ticker, start_date, end_date, silent=True)
    
    # Si on a toutes les données en cache
    if needs_download is None:
        return cached_data
    
    # Sinon, télécharger les données manquantes
    download_start, download_end = needs_download
    # ======================================================
    # 1) Mode Selenium
    # ======================================================

    if SELENIUM_AVAILABLE:
        try:
            # Appel à de la fonction fetch_data originale
            new_data = fetch_data(ticker, download_start, download_end, max_retries, wait_seconds)
        except Exception:
            new_data = None
    else:
        new_data = None

    
    # ======================================================
    # 2) Fallback YFinance sans proxy
    # ======================================================
    if new_data is None or new_data.empty:
        try:
            new_data = fetch_with_yfinance_direct(
                ticker, download_start, download_end
            )
        except Exception:
            return pd.DataFrame()

    if new_data is not None and not new_data.empty:
        # Sauvegarder dans le cache (mode silencieux)
        cache_manager.save_to_cache(ticker, new_data, download_start, download_end, silent=True)
        
        # Filtrer pour la période demandée
        mask = (new_data.index.date >= start_date) & (new_data.index.date <= end_date)
        filtered_data = new_data[mask]
        
        return filtered_data
    else:
        return pd.DataFrame()


# ==========================================================
# FONCTION UTILITAIRE POUR STREAMLIT
# ==========================================================

def afficher_cache_info():
    """Affiche les infos du cache dans Streamlit"""
    
    cache_info = cache_manager.get_cache_info()
    
    if cache_info:
        st.subheader("📦 Données en cache")
        
        for info in cache_info:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**{info['ticker']}**")
            with col2:
                st.write(f"{info['start_date']} → {info['end_date']}")
            with col3:
                st.write(f"{info['rows']} jours")
            with col4:
                if st.button("🗑️", key=f"clear_{info['ticker']}"):
                    cache_manager.clear_cache(info['ticker'])
                    st.rerun()
    else:
        st.info("Aucune donnée en cache")
    
    if st.button("🗑️ Effacer tout le cache"):
        cache_manager.clear_cache()
        st.rerun()