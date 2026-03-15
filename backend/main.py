import os
import time
import json
import requests
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit
import concurrent.futures

from groq import Groq

# --- Setup & Configuration ---
app.add_middleware(
    CORSMiddleware,
    # Replace with your actual frontend URL once deployed
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "https://your-portfolio-site.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Supabase Setup
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = None

if url and key:
    try:
        supabase = create_client(url, key)
        print("Supabase Connected")
    except Exception as e:
        print(f"Supabase Connection Failed: {e}")

# Groq Setup
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = None
if groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
        print("Groq API Configured for Bulk Analysis")
    except Exception as e:
        print(f"Groq Client Error: {e}")
else:
    print("Groq API Key missing")

app = FastAPI(title="Crypto Value Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Config ---

TRACKED_CRYPTOS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", 
    "DOGE-USD", "LINK-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "BCH-USD", 
    "UNI-USD", "ATOM-USD", "XLM-USD", "ALGO-USD", "NEAR-USD", "AAVE-USD",
    "SNX-USD", "MKR-USD", "GRT-USD", "FTM-USD", "SAND-USD", "MANA-USD"
]

# --- GROQ CLOUD: Bulk AI Sentiment Fetcher ---

def fetch_bulk_ai_sentiment(symbols: list) -> dict:
    """Sends ONE prompt to Groq containing all requested coins."""
    if not groq_client:
        return {}
        
    try:
        # 🚨 NEW: Added 'analysis' to the strict JSON requirements
        prompt = f"""
        Analyze the current overall market sentiment for the following cryptocurrencies: {', '.join(symbols)}. 
        Return EXACTLY a valid JSON object.
        The JSON must follow this exact structure:
        {{
            "SYMBOL-USD": {{
                "sentiment": "good" | "ok" | "bad",
                "analysis": "A brief 1 to 2 sentence justification explaining why this sentiment was chosen based on current market conditions."
            }}
        }}
        Ensure every symbol in the list is included as a key.
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative crypto analyst. You only respond in strictly formatted JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2, 
            response_format={"type": "json_object"} 
        )
        
        text = chat_completion.choices[0].message.content.strip()
        
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
            
        return json.loads(text.strip())
        
    except Exception as e:
        print(f"Bulk Groq API Error: {e}")
        return {}

# --- Coinbase Data Fetcher ---

def fetch_coinbase_candles(symbol: str, days: int = 1460) -> pd.DataFrame:
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    granularity = 86400  
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    all_candles = []
    current_end = end_time
    
    while current_end > start_time:
        current_start = max(start_time, current_end - timedelta(days=200))
        
        params = {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
            "granularity": granularity
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if not data: break
            all_candles.extend(data)
        elif response.status_code == 429:
            time.sleep(1)
            continue
        else:
            break
            
        current_end = current_start - timedelta(seconds=1)
        time.sleep(0.1)
        
    if not all_candles: return pd.DataFrame()
        
    df = pd.DataFrame(all_candles, columns=['time', 'low', 'high', 'open', 'Close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    return df

# --- Core Algorithm ---

def calculate_hybrid_value(symbol: str, decay_weight=0.7, tech_weight=0.3, ai_data=None) -> dict:
    try:
        symbol = symbol.upper().strip()
        
        STABLECOINS = ["USDC-USD", "USDT-USD", "DAI-USD", "PYUSD-USD", "FDUSD-USD", "TUSD-USD", "USDD-USD"]
        is_stablecoin = symbol in STABLECOINS
        peg_target = 1.00
        peg_tolerance = 0.02 
        
        data = fetch_coinbase_candles(symbol, days=200)
        if data.empty or len(data) < 200: return {"error": f"Insufficient data for {symbol}"}
        
        close_data = data['Close']
        prices = close_data.iloc[:, 0] if isinstance(close_data, pd.DataFrame) else close_data
        prices_list = prices.tolist()
        current_price = float(prices_list[-1])
        dates = [d.strftime("%Y-%m-%d") for d in data.index]

        # Volatility & Stability Math
        log_returns = np.log(prices / prices.shift(1)).dropna()
        daily_volatility = log_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(365)
        volatility_pct = annualized_volatility * 100
        stability_score = 100.0 * np.exp(-annualized_volatility)

        # Decay Math
        max_date = datetime.strptime(dates[-1], "%Y-%m-%d")
        ages_days = np.array([(max_date - datetime.strptime(d, "%Y-%m-%d")).days for d in dates])
        exp_weights = np.exp(-1.0 * (ages_days / 365.25))
        weighted_avg = np.average(prices_list, weights=(exp_weights / exp_weights.sum()))
        
        # Z-Score Math
        price_std = np.std(prices_list)
        z_score = (current_price - weighted_avg) / price_std if price_std > 0 else 0.0
        decay_score = max(0.0, min(100.0, 50.0 - (z_score * 20.0)))

        # RSI Math
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        current_rsi = float((100 - (100 / (1 + rs))).iloc[-1])
        tech_score = 100 - current_rsi

        # --- AI & LINKS INJECTION LOGIC ---
        sentiment_word = "ok"
        sentiment_multiplier = 1.0
        ai_analysis = "Awaiting live AI analysis..." # Default fallback
        
        # 🚨 GUARANTEED LIVE LINKS (Kept safely in Python to prevent AI 404 errors)
        coin_ticker = symbol.split('-')[0]
        ai_links = [
            f"https://finance.yahoo.com/quote/{symbol}/news/",
            f"https://news.google.com/search?q={coin_ticker}+crypto+news"
        ]
        
        if ai_data and symbol in ai_data:
            coin_ai = ai_data[symbol]
            sentiment_word = coin_ai.get("sentiment", "ok").lower()
            # 🚨 Extract the blurb from Groq
            ai_analysis = coin_ai.get("analysis", "No detailed analysis available at this time.")
            
            if sentiment_word == "good": sentiment_multiplier = 1.1
            elif sentiment_word == "bad": sentiment_multiplier = 0.9

        # Final Score Math
        final_score = (decay_score * decay_weight) + (tech_score * tech_weight)
        final_score = max(0.0, min(100.0, final_score * sentiment_multiplier)) * 0.76
        
        # Signal Generation
        if is_stablecoin:
            if current_price < (peg_target - peg_tolerance): signal, final_score = "SELL (DE-PEG)", 0.0
            elif current_price > (peg_target + peg_tolerance): signal, final_score = "BUY (PREMIUM)", 100.0
            else: signal, final_score = "HOLD (PEG INTACT)", 50.0
        else:
            if final_score >= 75: signal = "STRONG BUY"
            elif final_score >= 60: signal = "BUY"
            elif final_score <= 25: signal = "STRONG SELL"
            elif final_score <= 50: signal = "SELL"
            else: signal = "HOLD"

        if supabase:
            try:
                supabase.table("cryptos").upsert({
                    "symbol": symbol, "current_price": round(current_price, 3), "final_score": round(final_score, 1),
                    "weighted_avg": round(weighted_avg, 3), "signal": signal, "margin": round(final_score - 50, 1),
                    "updated_at": datetime.now().isoformat()
                }).execute()
            except Exception: pass

        return {
            "symbol": symbol,
            "current_price": round(current_price, 3),
            "final_score": round(final_score, 1),
            "signal": signal,
            "weighted_avg": round(weighted_avg, 3),
            "base_threshold": 50.0,
            "weight_recent": 0.0, 
            "components": {
                "fundamental_value": f"${weighted_avg:.3f}",
                "fundamental_score": round(decay_score, 1),
                "technical_rsi": round(current_rsi, 1),
                "technical_score": round(tech_score, 1),
                "volatility_pct": round(volatility_pct, 2),
                "stability_score": round(stability_score, 1),
                "ai_sentiment": sentiment_word,
                "ai_multiplier": sentiment_multiplier,
                "ai_analysis": ai_analysis, # <- Blurb passed to React here
                "ai_links": ai_links 
            },
            "value_coefficient": round(z_score, 3), 
            "margin": round(final_score - 50, 1) 
        }
        
    except Exception as e:
        print(f"Error calculating {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}

# --- Scheduler Logic ---

def scheduled_analysis():
    print(f"[SCHEDULER] Starting daily analysis for {len(TRACKED_CRYPTOS)} cryptos...")
    bulk_ai_data = fetch_bulk_ai_sentiment(TRACKED_CRYPTOS)
    success_count, error_count = 0, 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(calculate_hybrid_value, symbol, 0.7, 0.3, bulk_ai_data): symbol for symbol in TRACKED_CRYPTOS}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if "error" not in result: success_count += 1
                else: error_count += 1
            except Exception: error_count += 1
    
    print(f"[SCHEDULER] Complete. {success_count} analyzed, {error_count} errors")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_analysis, CronTrigger(hour=0, minute=0))
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- API Endpoints ---

@app.get("/")
def root():
    return {"status": "Crypto Value Analyzer running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/cryptos")
def top_cryptos():
    if supabase:
        try:
            response = supabase.table("cryptos").select("*").order("final_score", desc=True).execute()
            if response.data and len(response.data) > 0:
                return {"cryptos": response.data, "total": len(response.data), "source": "database_cache"}
        except Exception: pass

    bulk_ai_data = fetch_bulk_ai_sentiment(TRACKED_CRYPTOS)
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(calculate_hybrid_value, symbol, 0.7, 0.3, bulk_ai_data): symbol for symbol in TRACKED_CRYPTOS}
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if "error" not in result:
                results.append(result)
    
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return {"cryptos": results, "total": len(results), "source": "real_time_computed"}

@app.get("/analyze/{symbol}")
def analyze_crypto(symbol: str):
    symbol = symbol.upper()
    if "-" not in symbol: symbol = f"{symbol}-USD"
    ai_data = fetch_bulk_ai_sentiment([symbol])
    return calculate_hybrid_value(symbol, 0.7, 0.3, ai_data)

@app.get("/history/{symbol}")
def get_crypto_history(symbol: str):
    try:
        symbol = symbol.upper()
        if "-" not in symbol: symbol = f"{symbol}-USD"
        hist = fetch_coinbase_candles(symbol, days=365)
        if hist.empty: return {"error": f"No data found"}
        chart_data = [{"date": date.strftime("%Y-%m-%d"), "price": round(row['Close'], 2)} for date, row in hist.iterrows()]
        return {"data": chart_data}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)