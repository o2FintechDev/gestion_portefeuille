import pandas as pd

def rsi(series, period=14):
    series = series.dropna()
    if len(series) < period + 1:
        return pd.Series([None] * len(series), index=series.index, name="RSI")

    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("RSI")


def macd(series, fast=12, slow=26, signal=9):
    series = series.dropna()
    if len(series) < slow + signal:
        return pd.DataFrame({
            "MACD": [None] * len(series),
            "Signal": [None] * len(series)
        }, index=series.index)

    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()

    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line})


def moyennes_mobiles(series):
    series = series.dropna()
    df = pd.DataFrame(index=series.index)
    
    for window in [20, 50, 200]:
        if len(series) < window:
            df[f"SMA{window}"] = [None] * len(series)
        else:
            df[f"SMA{window}"] = series.rolling(window).mean()

    return df
