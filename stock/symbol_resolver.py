import json
from pathlib import Path
import difflib

from stock.tickers.catalogue_static import CATALOGUE_TICKERS



# ================================================================
# CHARGEMENT & FUSION
# ================================================================

def load_auto_catalogue():
    path = Path(__file__).resolve().parent / "tickers" / "catalogue_auto.json"

    if not path.exists():
        return []

    try:
        content = path.read_text().strip()
        if not content:
            return []  # fichier vide → catalogue vide

        return json.loads(content)

    except Exception:
        # fichier corrompu → on ignore et on évite le crash
        return []



def load_full_catalogue():
    # Priorité au catalogue manuel (statique)
    auto = load_auto_catalogue()
    return CATALOGUE_TICKERS + auto


# ================================================================
# INDEXATION — UNE SEULE FOIS EN MÉMOIRE
# ================================================================

_FULL = load_full_catalogue()          # Liste complète
_ALL_KEYS = []                         # Toutes les clés de recherche
_SYMBOL_MAP = {}                       # symbol -> item

for item in _FULL:
    # Récupération des clés de recherche
    keys = [item["symbol"], item["name"]] + item.get("aliases", [])
    item["_keys"] = [k.lower() for k in keys if k]

    # Index symbol -> item
    _SYMBOL_MAP[item["symbol"]] = item

    # Ajout dans la liste globale pour fuzzy search
    _ALL_KEYS.extend(item["_keys"])


# ================================================================
# SUGGESTIONS
# ================================================================

def suggere_tickers(query: str, topk: int = 6):
    """
    Suggestions (symbol, label) basées sur:
    - Contains prioritaires
    - Fuzzy difflib en fallback
    """
    q = query.strip().lower()
    if not q:
        return []

    # 1) Recherche directe "contains"
    contains = []
    for item in _FULL:
        if any(q in key for key in item["_keys"]):
            contains.append(item)

    out = contains

    # 2) Si aucun match direct → fuzzy matching
    if not out:
        close_keys = difflib.get_close_matches(q, _ALL_KEYS, n=topk, cutoff=0.6)
        if close_keys:
            seen = set()
            for ck in close_keys:
                for item in _FULL:
                    if ck in item["_keys"] and item["symbol"] not in seen:
                        out.append(item)
                        seen.add(item["symbol"])

    # 3) Limite & formatage final
    out = out[:topk]
    return [(it["symbol"], f"{it['name']} — {it['symbol']}") for it in out]




