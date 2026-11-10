import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

# Configure Telegram configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    """Sends formatted alerts directly to your Telegram Channel/Chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configurations missing. Message not sent.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram Alert sent successfully.")
        else:
            print(f"Failed to send Telegram Alert: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram Alert: {e}")

# --- Copy of Scanner Functions ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(window=period).mean()

def find_peaks_troughs(df, order=2):
    highs = df['High'].values
    lows = df['Low'].values
    peak_idx = argrelextrema(highs, np.greater, order=order)[0]
    trough_idx = argrelextrema(lows, np.less, order=order)[0]
    swings = []
    for p in peak_idx: swings.append({"idx": p, "price": float(highs[p]), "type": "Peak", "date": df.index[p]})
    for t in trough_idx: swings.append({"idx": t, "price": float(lows[t]), "type": "Trough", "date": df.index[t]})
    swings.sort(key=lambda x: x["idx"])
    return swings

def scan_harmonics_combinatorial(swings, error_allowed=0.18):
    patterns = []
    n = len(swings)
    if n < 5: return patterns
    seen_signals = set()
    for i in range(n - 4):
        for j in range(i + 1, min(i + 8, n - 3)):
            for k in range(j + 1, min(j + 8, n - 2)):
                for l in range(k + 1, min(k + 8, n - 1)):
                    for m in range(l + 1, min(l + 8, n)):
                        sX, sA, sB, sC, sD = swings[i], swings[j], swings[k], swings[l], swings[m]
                        if not ((sX["type"] == "Trough" and sA["type"] == "Peak" and sB["type"] == "Trough" and sC["type"] == "Peak" and sD["type"] == "Trough") or
                                (sX["type"] == "Peak" and sA["type"] == "Trough" and sB["type"] == "Peak" and sC["type"] == "Trough" and sD["type"] == "Peak")):
                            continue
                        X, A, B, C, D = sX["price"], sA["price"], sB["price"], sC["price"], sD["price"]
                        XA, AB, BC = A - X, B - A, C - B
                        if XA == 0 or AB == 0 or BC == 0: continue
                        ret_B, ret_D = abs(AB / XA), abs((D - X) / XA)
                        p_type = None
                        if abs(ret_B - 0.618) <= error_allowed and abs(ret_D - 0.786) <= error_allowed: p_type = "Gartley"
                        elif abs(ret_B - 0.382) <= error_allowed and abs(ret_D - 0.886) <= error_allowed: p_type = "Bat"
                        elif abs(ret_B - 0.786) <= error_allowed and abs(ret_D - 1.272) <= error_allowed: p_type = "Butterfly"
                        elif abs(ret_B - 0.382) <= error_allowed and abs(ret_D - 1.618) <= error_allowed: p_type = "Crab"
                        if p_type is None: continue
                        is_bullish = XA > 0 and D < C
                        pat_name = f"{'Bullish' if is_bullish else 'Bearish'} {p_type}"
                        sig_key = (sD["date"], pat_name)
                        if sig_key not in seen_signals:
                            seen_signals.add(sig_key)
                            patterns.append({"type": pat_name, "points": [sX, sA, sB, sC, sD]})
                            break
    return patterns

if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS"]
    print("Scanning for fresh signals to publish...")
    
    for ticker in tickers:
        try:
            # Pulling trailing 30 days to check the absolute latest patterns
            data = yf.download(ticker, period="30d", interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            if data.empty: continue
            
            data['ATR'] = calculate_atr(data, period=14)
            swings = find_peaks_troughs(data, order=2)
            detected_patterns = scan_harmonics_combinatorial(swings, error_allowed=0.18)
            
            if not detected_patterns:
                continue
                
            # Check if a new signal developed on the very last trading bar
            last_idx = len(data) - 1
            for pat in detected_patterns:
                sD = pat["points"][-1]
                if sD["idx"] == last_idx:
                    # Format and transmit Telegram alert
                    close_price = float(np.ravel(data['Close'].iloc[-1])[0])
                    atr = float(np.ravel(data['ATR'].iloc[-1])[0])
                    is_bullish = "Bullish" in pat["type"]
                    
                    sl = close_price - atr if is_bullish else close_price + atr
                    tp = close_price + (3.0 * atr) if is_bullish else close_price - (3.0 * atr)
                    
                    alert_msg = (
                        f"🚨 *NEW HARMONIC SIGNAL DETECTED* 🚨\n\n"
                        f"*Ticker:* `{ticker}`\n"
                        f"*Pattern:* {pat['type']}\n"
                        f"*Action:* {'🟢 BUY (Long)' if is_bullish else '🔴 SELL (Short)'}\n"
                        f"*Entry Price:* ₹{close_price:,.2f}\n"
                        f"*Stop Loss (SL):* ₹{sl:,.2f}\n"
                        f"*Target (TP):* ₹{tp:,.2f}\n"
                    )
                    print(f"New Signal detected for {ticker}! Sending...")
                    send_telegram_alert(alert_msg)
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
          
