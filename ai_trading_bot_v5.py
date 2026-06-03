import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import sqlite3
import requests
import json
import os
import feedparser
from datetime import datetime, timezone, timedelta
import yfinance as yf

# ======================================================
# GUI FILTER BRIDGE  (reads toggles from gui_server.py)
# ======================================================

GUI_FILTER_FILE = "gui_filters.json"

_GUI_FILTER_DEFAULTS = {
    "killzone":       False,
    "calendar":       True,
    "spread":         True,
    "volatility":     True,
    "market_quality": True,
    "regime":         True,
    "mtf":            True,
    "telegram":       True,
}

def read_gui_filters() -> dict:
    try:
        with open(GUI_FILTER_FILE) as f:
            data = json.load(f)
        return {**_GUI_FILTER_DEFAULTS, **data}
    except Exception:
        return dict(_GUI_FILTER_DEFAULTS)

# ======================================================
# CONFIG
# ======================================================

BYTEZ_API_KEY   = "1b3f304bf815e59b9912905961c7a9d9"
TELEGRAM_TOKEN  = "8885360577:AAGDPeUn2drVU1RLNDGJZ91azqMzp0e3QUY"
TELEGRAM_CHAT_ID = "745002829"

# yfinance symbol format
SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "GC=F",      # Gold
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURGBP=X",
    "GBPJPY=X",
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "EURJPY=X",  # EUR/JPY
    "SI=F",      # Silver
    "CL=F",      # Crude Oil
]

# Internal display names (for DB / Telegram)
SYMBOL_DISPLAY = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY",
    "GC=F":     "GOLD",
    "AUDUSD=X": "AUDUSD",
    "USDCAD=X": "USDCAD",
    "USDCHF=X": "USDCHF",
    "NZDUSD=X": "NZDUSD",
    "EURGBP=X": "EURGBP",
    "GBPJPY=X": "GBPJPY",
    "BTC-USD":  "BTCUSD",
    "ETH-USD":  "ETHUSD",
    "EURJPY=X": "EURJPY",
    "SI=F":     "SILVER",
    "CL=F":     "OILUSD",
}

# Spread limits per symbol (in price units)
SPREAD_LIMITS = {
    "XAU/USD":  1.0,
    "BTC/USD":  50.0,   # BTC has wide spreads
    "GBP/JPY":  0.08,
    "ETH-USD":  5.0,    # ETH wide spread
    "SI=F":     0.10,   # Silver
    "CL=F":     0.10,   # Crude Oil
    "EURJPY=X": 0.08,
    "DEFAULT":  0.05,
}

TIMEFRAME        = "5min"
SIGNAL_FILE      = "signal_history.json"
PERFORMANCE_FILE = "performance_data.json"
DATABASE         = "ai_trading.db"

# ======================================================
# v5 UPGRADE CONFIG
# ======================================================

SIGNAL_QUALITY_MIN   = 65       # minimum quality score to send Telegram alert
BYTEZ_MODEL          = "meta-llama/Meta-Llama-3-8b-instruct"  # Bytez model for AI analysis
DAILY_REPORT_FILE    = "daily_report_state.json"
AVOID_RULES_KEY      = "avoid_rules"   # key inside learning_state.json

# ======================================================
# UPGRADE 1 — MARKET REGIME CONFIG
# ======================================================

REGIME_ADX_TREND    = 25
REGIME_ADX_VOLATILE = 40

REGIME_MULTIPLIERS = {
    "TRENDING": {"sl": 1.5, "tp": 4.0},
    "RANGING":  {"sl": 1.0, "tp": 1.5},
    "VOLATILE": {"sl": 2.0, "tp": 2.5},
}

# ======================================================
# UPGRADE 3 — SESSION KILLZONE CONFIG (UTC hours)
# ======================================================

KILLZONES = {
    "Asian":    (0,  3),
    "London":   (8,  10),
    "NY_Open":  (13, 15),
    "NY_Close": (19, 20),
}

# ======================================================
# UPGRADE 2 — ECONOMIC CALENDAR CONFIG
# ======================================================

CALENDAR_BLOCK_BEFORE_MIN = 30
CALENDAR_BLOCK_AFTER_MIN  = 15

HIGH_IMPACT_KEYWORDS = [
    "nfp", "non-farm", "interest rate", "rate decision",
    "fomc", "cpi", "inflation", "gdp", "boe", "ecb",
    "rba", "rbnz", "boj", "fed", "powell", "lagarde",
    "unemployment", "payroll"
]

# ======================================================
# UPGRADE 6 — SIGNAL TRACKER CONFIG
# ======================================================

SIGNAL_EXPIRY_HOURS = 24
OPEN_SIGNALS_FILE   = "open_signals.json"

# ======================================================
# ONE TRADE PER SYMBOL GUARD
# ======================================================
# Bot will NOT open a new signal on a symbol that already
# has an open/running trade. A new trade is only allowed
# once the previous one is fully CLOSED (WIN/LOSS/EXPIRED).

def symbol_has_open_trade(symbol_display, open_signals):
    """Return True if this symbol already has a live trade."""
    for sig in open_signals:
        if sig.get("symbol") == symbol_display and sig.get("signal") in ("BUY", "SELL"):
            return True
    return False

# ======================================================
# AUTO-LEARNING ENGINE CONFIG
# ======================================================
# Tracks per-symbol win/loss patterns and adjusts minimum
# probability threshold dynamically to avoid repeated losses.

LEARNING_FILE      = "learning_state.json"
MIN_PROB_DEFAULT   = 70    # minimum probability to take a trade
MIN_PROB_FLOOR     = 60    # never go below this
MIN_PROB_CEILING   = 90    # never go above this
LOSS_STREAK_LIMIT  = 3     # consecutive losses before raising threshold

def load_learning_state():
    try:
        with open(LEARNING_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_learning_state(state):
    with open(LEARNING_FILE, "w") as f:
        json.dump(state, f, indent=2)

def update_learning(symbol_display, result):
    """Update per-symbol learning after a trade closes."""
    state = load_learning_state()
    sym   = state.get(symbol_display, {
        "wins": 0, "losses": 0, "streak": 0,
        "min_prob": MIN_PROB_DEFAULT
    })

    if result == "WIN":
        sym["wins"]   += 1
        sym["streak"]  = max(0, sym["streak"] - 1)   # reduce caution on win
        # Slowly lower threshold back toward default on wins
        sym["min_prob"] = max(MIN_PROB_DEFAULT, sym["min_prob"] - 2)

    elif result == "LOSS":
        sym["losses"] += 1
        sym["streak"] += 1
        # Raise bar after each loss, harder after streak
        raise_by = 3 * sym["streak"]
        sym["min_prob"] = min(MIN_PROB_CEILING, sym["min_prob"] + raise_by)
        if sym["streak"] >= LOSS_STREAK_LIMIT:
            print(f"[AutoLearn] {symbol_display} — {sym['streak']} loss streak! "
                  f"Raising min prob to {sym['min_prob']}%")

    total = sym["wins"] + sym["losses"]
    wr    = round(sym["wins"] / total * 100, 1) if total else 0.0
    print(f"[AutoLearn] {symbol_display} → W:{sym['wins']} L:{sym['losses']} "
          f"WR:{wr}% | MinProb:{sym['min_prob']}%")

    state[symbol_display] = sym
    save_learning_state(state)
    return sym

def get_min_probability(symbol_display):
    """Return the current learned minimum probability for this symbol."""
    state = load_learning_state()
    return state.get(symbol_display, {}).get("min_prob", MIN_PROB_DEFAULT)

# ======================================================
# v5 UPGRADE 1 — SIGNAL QUALITY SCORE
# ======================================================

def calculate_quality_score(signal_direction, probability, bull_score, bear_score,
                             regime, killzone, mtf, structure, fvg, order_block):
    """
    Composite quality score 0-100. Only signals >= SIGNAL_QUALITY_MIN get alerted.
    Combines indicator agreement, regime suitability, and SMC confluence.
    """
    score = 0

    # Base probability contribution (0-40 pts)
    score += min(40, int(probability * 0.42))

    # Regime suitability (0-15 pts)
    if regime == "TRENDING":
        score += 15
    elif regime == "RANGING":
        score += 8
    else:  # VOLATILE
        score += 0

    # Killzone bonus (0-10 pts)
    if killzone in ("London", "NY_Open"):
        score += 10
    elif killzone in ("NY_Close",):
        score += 5
    elif killzone == "Asian":
        score += 3

    # MTF alignment (0-15 pts)
    if (signal_direction == "BUY" and mtf == "BULLISH") or \
       (signal_direction == "SELL" and mtf == "BEARISH"):
        score += 15
    elif mtf == "NEUTRAL":
        score += 5

    # SMC confluence: structure + FVG + OB (0-20 pts)
    smc = 0
    if (signal_direction == "BUY" and structure == "BULLISH_BOS") or \
       (signal_direction == "SELL" and structure == "BEARISH_BOS"):
        smc += 7
    if (signal_direction == "BUY" and fvg == "BULLISH_FVG") or \
       (signal_direction == "SELL" and fvg == "BEARISH_FVG"):
        smc += 7
    if (signal_direction == "BUY" and order_block == "BULLISH_ORDER_BLOCK") or \
       (signal_direction == "SELL" and order_block == "BEARISH_ORDER_BLOCK"):
        smc += 6
    score += min(20, smc)

    return min(100, score)


# ======================================================
# v5 UPGRADE 2 — ENTRY TYPE SUGGESTER
# ======================================================

def suggest_entry_type(signal_direction, entry, df, regime, fvg, order_block, atr):
    """
    Returns 'LIMIT' or 'MARKET' and a brief reason.
    LIMIT is preferred when price is near a key SMC level or regime is ranging.
    MARKET is preferred when momentum is strong and trending.
    """
    reason_parts = []

    # Strong SMC levels → prefer limit to get better fill
    near_ob = (signal_direction == "BUY" and order_block == "BULLISH_ORDER_BLOCK") or \
              (signal_direction == "SELL" and order_block == "BEARISH_ORDER_BLOCK")
    near_fvg = (signal_direction == "BUY" and fvg == "BULLISH_FVG") or \
               (signal_direction == "SELL" and fvg == "BEARISH_FVG")

    if near_ob:
        reason_parts.append("OB retest")
    if near_fvg:
        reason_parts.append("FVG fill")

    if near_ob or near_fvg:
        # Suggest a limit slightly inside the zone
        if signal_direction == "BUY":
            limit_price = round(entry - atr * 0.1, 5)
        else:
            limit_price = round(entry + atr * 0.1, 5)
        reason = " + ".join(reason_parts) + f" — set limit @ {limit_price}"
        return "LIMIT", reason

    # Strong trending regime → market order
    if regime == "TRENDING":
        reason_parts.append("trending regime, momentum entry")
        return "MARKET", " + ".join(reason_parts)

    # Default to market with a note
    return "MARKET", "no key SMC level nearby"


# ======================================================
# v5 UPGRADE 3 — BYTEZ CONFIRMATION FILTER
# ======================================================

def bytez_confirmation(sym_display, signal_direction, probability, regime,
                        killzone, mtf, trend, news, structure, fvg, order_block,
                        bull_score, bear_score):
    """
    Ask Bytez AI to validate the signal. Returns (confirmed: bool, reason: str).
    Falls back to True (allow) if Bytez call fails so the bot never stalls.
    """
    if signal_direction == "HOLD":
        return True, "HOLD — no Bytez check needed"

    kz_str = killzone if killzone else "Outside Killzone"
    prompt = (
        f"You are a professional forex/crypto trading analyst. "
        f"Evaluate this signal and reply ONLY with JSON: "
        f"{{\"confirmed\": true/false, \"confidence\": 0-100, \"reason\": \"one sentence\"}}\n\n"
        f"Symbol: {sym_display} | Direction: {signal_direction}\n"
        f"Probability: {probability}% | Regime: {regime} | Killzone: {kz_str}\n"
        f"MTF: {mtf} | Trend: {trend} | News: {news}\n"
        f"Structure: {structure} | FVG: {fvg} | OrderBlock: {order_block}\n"
        f"Bull score: {bull_score}/13 | Bear score: {bear_score}/13\n\n"
        f"Confirm if this is a HIGH QUALITY trade setup worth taking. "
        f"Be strict — reject if confluence is weak."
    )

    try:
        raw_text = _bytez_chat(prompt)

        # Strip markdown fences if present
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        parsed    = json.loads(raw_text)
        confirmed = bool(parsed.get("confirmed", True))
        reason    = str(parsed.get("reason", "No reason given"))
        confidence = int(parsed.get("confidence", 50))
        print(f"[Bytez] {sym_display} {signal_direction} → confirmed={confirmed} "
              f"confidence={confidence}% | {reason}")
        return confirmed, reason

    except Exception as e:
        print(f"[Bytez] Confirmation call failed ({e}) — allowing signal through")
        return True, "Bytez unavailable — fallback allow"


# ======================================================
# v5 UPGRADE 4 — POST-TRADE BYTEZ ANALYSIS
# ======================================================

def bytez_post_trade_analysis(sig, result, current_price, minutes):
    """
    After a trade closes, ask Bytez WHY it won or lost.
    Saves the insight as an avoid-rule (for losses) or reinforce-rule (for wins)
    into learning_state.json.
    """
    try:
        symbol    = sig.get("symbol", "?")
        direction = sig.get("signal", "?")
        entry     = sig.get("entry", 0)
        sl        = sig.get("sl", 0)
        tp        = sig.get("tp", 0)
        regime    = sig.get("regime", "?")
        killzone  = sig.get("killzone", "Outside")
        mtf       = sig.get("mtf_confirmation", "?")
        structure = sig.get("structure", "?")
        fvg       = sig.get("fvg", "?")
        ob        = sig.get("order_block", "?")
        news      = sig.get("news", "?")

        prompt = (
            f"You are a professional trading coach. A bot took this trade:\n"
            f"Symbol: {symbol} | Direction: {direction} | Result: {result}\n"
            f"Entry: {entry} | TP: {tp} | SL: {sl} | Close: {current_price:.5f}\n"
            f"Duration: {minutes:.0f} min | Regime: {regime} | Killzone: {killzone}\n"
            f"MTF: {mtf} | Structure: {structure} | FVG: {fvg} | OB: {ob} | News: {news}\n\n"
            f"Reply ONLY with JSON: "
            f"{{\"why\": \"one sentence reason\", "
            f"\"avoid_if\": \"brief condition to avoid in future or null\", "
            f"\"reinforce_if\": \"brief condition to repeat in future or null\"}}"
        )

        raw = _bytez_chat(prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        why         = parsed.get("why", "")
        avoid_if    = parsed.get("avoid_if")
        reinforce_if = parsed.get("reinforce_if")

        print(f"[Bytez Post-Trade] {symbol} {result}: {why}")

        # Persist to learning_state
        state = load_learning_state()
        rules = state.setdefault(AVOID_RULES_KEY, {})
        sym_rules = rules.setdefault(symbol, {"avoid": [], "reinforce": []})

        if result == "LOSS" and avoid_if:
            entry_rule = {
                "rule": avoid_if,
                "regime": regime, "killzone": killzone,
                "mtf": mtf, "added": str(datetime.now().date())
            }
            sym_rules["avoid"].append(entry_rule)
            # Keep last 10 avoid rules per symbol
            sym_rules["avoid"] = sym_rules["avoid"][-10:]
            print(f"[Bytez] Avoid rule saved: {avoid_if}")

        if result == "WIN" and reinforce_if:
            sym_rules["reinforce"].append({
                "rule": reinforce_if,
                "added": str(datetime.now().date())
            })
            sym_rules["reinforce"] = sym_rules["reinforce"][-10:]

        # Also save analysis summary for daily report
        day_key = str(datetime.now().date())
        daily   = state.setdefault("daily_analysis", {})
        day_rec = daily.setdefault(day_key, [])
        day_rec.append({
            "symbol": symbol, "direction": direction,
            "result": result, "why": why,
            "avoid_if": avoid_if, "reinforce_if": reinforce_if
        })
        daily[day_key] = day_rec[-50:]   # cap at 50 per day

        save_learning_state(state)

        # Telegram summary of the analysis
        emoji = "✅" if result == "WIN" else "❌"
        msg = (
            f"🧠 BYTEZ POST-TRADE ANALYSIS\n"
            f"{emoji} {symbol} {direction} → {result}\n"
            f"Why: {why}\n"
        )
        if result == "LOSS" and avoid_if:
            msg += f"⛔ Avoid if: {avoid_if}\n"
        if result == "WIN" and reinforce_if:
            msg += f"✅ Repeat if: {reinforce_if}\n"
        send_telegram(msg)

    except Exception as e:
        print(f"[Bytez Post-Trade] Analysis failed: {e}")


# ======================================================
# v5 UPGRADE 5 — PRE-TRADE AVOID-RULES CHECK
# ======================================================

def check_avoid_rules(sym_display, regime, killzone, mtf):
    """
    Check current setup against learned avoid rules for this symbol.
    Returns (blocked: bool, matched_rule: str).
    """
    try:
        state    = load_learning_state()
        rules    = state.get(AVOID_RULES_KEY, {})
        sym_rules = rules.get(sym_display, {})
        avoid_list = sym_rules.get("avoid", [])

        if not avoid_list:
            return False, ""

        # Simple keyword matching on regime / killzone / mtf context
        context_str = f"{regime} {killzone} {mtf}".lower()

        for rule_entry in avoid_list:
            rule_text = rule_entry.get("rule", "").lower()
            rule_regime   = rule_entry.get("regime", "").lower()
            rule_killzone = rule_entry.get("killzone", "").lower()
            rule_mtf      = rule_entry.get("mtf", "").lower()

            # Match if stored regime/killzone/mtf match current context
            match_score = 0
            if rule_regime   and rule_regime   in context_str: match_score += 1
            if rule_killzone and rule_killzone  in context_str: match_score += 1
            if rule_mtf      and rule_mtf       in context_str: match_score += 1

            # Block if 2+ contextual factors match
            if match_score >= 2:
                print(f"[AvoidRule] {sym_display} blocked by learned rule: {rule_text}")
                return True, rule_text

        return False, ""

    except Exception as e:
        print(f"[AvoidRule] Check error: {e}")
        return False, ""


# ======================================================
# v5 UPGRADE 6 — DAILY LEARNING REPORT
# ======================================================

def send_daily_learning_report():
    """
    Sends a Telegram summary of what the bot learned today.
    Runs once per day (tracked via daily_report_state.json).
    """
    try:
        today = str(datetime.now().date())

        # Check if already sent today
        try:
            with open(DAILY_REPORT_FILE) as f:
                dr_state = json.load(f)
        except Exception:
            dr_state = {}

        if dr_state.get("last_sent") == today:
            return   # already sent today

        state    = load_learning_state()
        day_data = state.get("daily_analysis", {}).get(today, [])

        wins_today   = [x for x in day_data if x.get("result") == "WIN"]
        losses_today = [x for x in day_data if x.get("result") == "LOSS"]

        # Per-symbol summary from learning_state
        sym_lines = ""
        for sym, sym_data in state.items():
            if sym in (AVOID_RULES_KEY, "daily_analysis"):
                continue
            if not isinstance(sym_data, dict):
                continue
            w  = sym_data.get("wins", 0)
            l  = sym_data.get("losses", 0)
            mp = sym_data.get("min_prob", MIN_PROB_DEFAULT)
            st = sym_data.get("streak", 0)
            total = w + l
            wr = round(w / total * 100, 1) if total else 0.0
            sym_lines += f"  {sym:<10} W:{w} L:{l} WR:{wr}% MinProb:{mp}% Streak:{st}\n"

        # Collect today's avoid rules
        avoid_lines = ""
        rules = state.get(AVOID_RULES_KEY, {})
        for sym, sym_rules in rules.items():
            for rule in sym_rules.get("avoid", []):
                if rule.get("added") == today:
                    avoid_lines += f"  ⛔ {sym}: {rule['rule']}\n"

        reinforce_lines = ""
        for sym, sym_rules in rules.items():
            for rule in sym_rules.get("reinforce", []):
                if rule.get("added") == today:
                    reinforce_lines += f"  ✅ {sym}: {rule['rule']}\n"

        report = (
            f"📚 DAILY LEARNING REPORT — {today}\n"
            f"{'─'*32}\n"
            f"Trades resolved today: {len(day_data)}\n"
            f"  ✅ Wins:   {len(wins_today)}\n"
            f"  ❌ Losses: {len(losses_today)}\n\n"
            f"📊 Symbol Intelligence:\n{sym_lines if sym_lines else '  No data yet\n'}\n"
        )

        if avoid_lines:
            report += f"⛔ New Avoid Rules (from losses):\n{avoid_lines}\n"
        if reinforce_lines:
            report += f"✅ Reinforced Patterns (from wins):\n{reinforce_lines}\n"

        if not avoid_lines and not reinforce_lines:
            report += "🤖 No new patterns learned today.\n"

        report += f"{'─'*32}"

        send_telegram(report)
        print(report)

        # Mark as sent
        with open(DAILY_REPORT_FILE, "w") as f:
            json.dump({"last_sent": today}, f)

    except Exception as e:
        print(f"[DailyReport] Error: {e}")

# ======================================================
# GLOBAL AI STATS
# ======================================================

wins   = 0
losses = 0

# ======================================================
# BYTEZ REST API (direct — no SDK, avoids numpy conflict)
# ======================================================

def _bytez_chat(prompt):
    """
    Call Bytez API directly via requests.
    Returns the assistant reply string, or raises on failure.
    Falls back gracefully in all callers via try/except.
    """
    headers = {
        "Authorization": f"Bearer {BYTEZ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": BYTEZ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
    resp = requests.post(
        "https://api.bytez.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ======================================================
# SQLITE DATABASE
# ======================================================

conn   = sqlite3.connect(DATABASE)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    signal        TEXT,
    probability   REAL,
    trend         TEXT,
    regime        TEXT,
    killzone      TEXT,
    entry         REAL,
    sl            REAL,
    tp            REAL,
    status        TEXT,
    result        TEXT,
    minutes_taken REAL,
    time          TEXT,
    closed_time   TEXT
)
""")
conn.commit()

# ======================================================
# TELEGRAM
# ======================================================

def _telegram_enabled() -> bool:
    try:
        with open(GUI_FILTER_FILE) as f:
            return json.load(f).get("telegram", True)
    except Exception:
        return True

def send_telegram(message):
    if not _telegram_enabled():
        print(f"[Telegram OFF] {message[:60]}...")
        return
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

# ======================================================
# TWELVE DATA — MARKET DATA (replaces MT5)
# ======================================================

# Simple in-memory price cache to avoid wasting API calls
_price_cache = {}   # symbol → {"bid": x, "ask": x, "time": datetime}

def get_market_data(symbol):
    """Fetch OHLCV candles from yfinance. Returns DataFrame or None."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="5m")
        if df is None or df.empty:
            print(f"No Data For {symbol}")
            return None
        df = df.rename(columns={
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        print(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"Market Data Error ({symbol}): {e}")
        return None


def get_live_price(symbol):
    """Get latest price from yfinance."""
    # Use cache if fresh (< 30 seconds old)
    cached = _price_cache.get(symbol)
    if cached:
        age = (datetime.now() - cached["time"]).total_seconds()
        if age < 30:
            return cached["bid"], cached["ask"]
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m")
        if df is None or df.empty:
            return None, None
        price = float(df["Close"].iloc[-1])
        spread_est = price * 0.0001   # estimate 1 pip spread
        bid = price - spread_est / 2
        ask = price + spread_est / 2
        _price_cache[symbol] = {"bid": bid, "ask": ask, "time": datetime.now()}
        return bid, ask
    except Exception as e:
        print(f"Price Error ({symbol}): {e}")
        return None, None


def spread_filter(symbol):
    """Check spread is within acceptable range per symbol."""
    bid, ask = get_live_price(symbol)
    if bid is None:
        return True   # allow through if price unavailable
    spread = abs(ask - bid)
    print(f"Spread ({symbol}): {round(spread, 5)}")
    limit = SPREAD_LIMITS.get(symbol, SPREAD_LIMITS["DEFAULT"])
    if spread > limit:
        return False
    return True

# ======================================================
# ADD INDICATORS
# ======================================================

def add_indicators(df):
    df['EMA20']       = ta.ema(df['close'], length=20)
    df['EMA50']       = ta.ema(df['close'], length=50)
    df['RSI']         = ta.rsi(df['close'], length=14)
    macd              = ta.macd(df['close'])
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
    df['ATR']         = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx_data          = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX']         = adx_data['ADX_14']
    return df

# ======================================================
# TREND ANALYSIS
# ======================================================

def get_trend(latest):
    if latest['EMA20'] > latest['EMA50']:
        return "BULLISH"
    elif latest['EMA20'] < latest['EMA50']:
        return "BEARISH"
    return "SIDEWAYS"

# ======================================================
# PROBABILITY ENGINE
# ======================================================

def calculate_probability(latest):
    score = 50

    # RSI scoring — stronger signals at extremes
    rsi = latest['RSI']
    if rsi > 70 or rsi < 30:
        score += 15   # strong overbought/oversold
    elif rsi > 60 or rsi < 40:
        score += 10   # moderate signal

    # EMA trend alignment
    ema_gap = abs(latest['EMA20'] - latest['EMA50'])
    if latest['EMA20'] > latest['EMA50']:
        score += 15
        if ema_gap > latest['EMA50'] * 0.001:   # meaningful gap
            score += 5
    elif latest['EMA20'] < latest['EMA50']:
        score += 5    # bearish trend still gets partial score

    # MACD momentum
    if latest['MACD'] > latest['MACD_SIGNAL']:
        score += 15
        if abs(latest['MACD'] - latest['MACD_SIGNAL']) > 0.0001:
            score += 5   # strong separation
    elif latest['MACD'] < latest['MACD_SIGNAL']:
        score += 5

    # ADX trend strength
    adx = latest.get('ADX', 0)
    if adx > 30:
        score += 10   # strong trend
    elif adx > 20:
        score += 5    # moderate trend

    return min(round(score), 95)

# ======================================================
# MARKET QUALITY FILTER
# ======================================================

def market_quality_filter(df):
    latest = df.iloc[-1]
    if 45 <= latest['RSI'] <= 55:
        return False
    return True

# ======================================================
# LIQUIDITY SWEEP DETECTION
# ======================================================

def liquidity_sweep(df):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    if latest['high'] > previous['high'] and latest['close'] < previous['high']:
        return "BUY_SIDE_LIQUIDITY"
    if latest['low'] < previous['low'] and latest['close'] > previous['low']:
        return "SELL_SIDE_LIQUIDITY"
    return "NO_SWEEP"

# ======================================================
# MARKET STRUCTURE DETECTION
# ======================================================

def market_structure(df):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    if latest['high'] > previous['high']:
        return "BULLISH_BOS"
    if latest['low'] < previous['low']:
        return "BEARISH_BOS"
    return "RANGING"

# ======================================================
# ORDER BLOCK DETECTION
# ======================================================

def detect_order_block(df):
    latest      = df.iloc[-2]
    candle_size = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    if total_range == 0:
        return "NO_ORDER_BLOCK"
    body_ratio = candle_size / total_range
    if latest['close'] > latest['open'] and body_ratio > 0.6:
        return "BULLISH_ORDER_BLOCK"
    if latest['close'] < latest['open'] and body_ratio > 0.6:
        return "BEARISH_ORDER_BLOCK"
    return "NO_ORDER_BLOCK"

# ======================================================
# FAIR VALUE GAP DETECTION
# ======================================================

def detect_fvg(df):
    candle1 = df.iloc[-3]
    candle3 = df.iloc[-1]
    if candle1['high'] < candle3['low']:
        return "BULLISH_FVG"
    if candle1['low'] > candle3['high']:
        return "BEARISH_FVG"
    return "NO_FVG"

# ======================================================
# LIVE FOREX NEWS SENTIMENT
# ======================================================

def news_sentiment():
    try:
        feed_url      = "https://www.forexfactory.com/ffcal_week_this.xml"
        feed          = feedparser.parse(feed_url)
        bullish_words = ["bullish", "rate hike", "strong", "growth"]
        bearish_words = ["bearish", "recession", "crash", "weak"]
        bullish_score = 0
        bearish_score = 0
        for entry in feed.entries[:10]:
            title = entry.title.lower()
            for word in bullish_words:
                if word in title:
                    bullish_score += 1
            for word in bearish_words:
                if word in title:
                    bearish_score += 1
        if bullish_score > bearish_score:
            return "BULLISH_NEWS"
        if bearish_score > bullish_score:
            return "BEARISH_NEWS"
        return "NEUTRAL_NEWS"
    except Exception as e:
        print("News Error:", e)
        return "NEUTRAL_NEWS"

# ======================================================
# SESSION FILTER
# ======================================================

def is_market_session_active():
    current_hour = datetime.now().hour
    if 0 <= current_hour <= 23:
        return True
    return False

# ======================================================
# VOLATILITY FILTER
# ======================================================

def volatility_filter(df):
    if df['ATR'].iloc[-1] <= 0:
        return False
    return True

# ======================================================
# WIN RATE LEARNING
# ======================================================

def update_winrate(result):
    global wins, losses
    if result == "WIN":
        wins += 1
    elif result == "LOSS":
        losses += 1
    total = wins + losses
    if total == 0:
        return 0
    winrate = (wins / total) * 100
    print(f"REAL WIN RATE: {round(winrate, 2)}%")
    return round(winrate, 2)

# ======================================================
# REINFORCEMENT LEARNING ENGINE
# ======================================================

def reinforcement_learning(winrate):
    global wins, losses
    total = wins + losses
    if total == 0:
        # No history yet — use NORMAL mode, don't penalise
        print("AI MODE: NORMAL (no history)")
        return 1.0
    if winrate >= 70:
        print("AI MODE: AGGRESSIVE")
        return 1.2
    elif winrate >= 50:
        print("AI MODE: NORMAL")
        return 1.0
    else:
        print("AI MODE: DEFENSIVE")
        return 0.9   # softened from 0.8 → 0.9

# ======================================================
# SESSION PERFORMANCE
# ======================================================

def session_performance(history):
    london  = 0
    newyork = 0
    asian   = 0
    for signal in history:
        hour = datetime.fromisoformat(signal['time']).hour
        if 7 <= hour <= 12:
            london += 1
        elif 13 <= hour <= 18:
            newyork += 1
        else:
            asian += 1
    print("\nSESSION PERFORMANCE\n")
    print(f"London Signals: {london}")
    print(f"New York Signals: {newyork}")
    print(f"Asian Signals: {asian}")

# ======================================================
# AUTO STRATEGY OPTIMIZER
# ======================================================

def strategy_optimizer(winrate):
    global wins, losses
    total = wins + losses
    if total == 0:
        # No history yet — use balanced mode
        print("OPTIMIZER: BALANCED MODE (no history)")
        return {"risk_multiplier": 1.0, "confidence_boost": 5}
    if winrate >= 75:
        print("OPTIMIZER: HIGH PERFORMANCE MODE")
        return {"risk_multiplier": 1.5, "confidence_boost": 10}
    elif winrate >= 50:
        print("OPTIMIZER: BALANCED MODE")
        return {"risk_multiplier": 1.0, "confidence_boost": 5}
    else:
        print("OPTIMIZER: SAFE MODE")
        return {"risk_multiplier": 0.8, "confidence_boost": 3}

# ======================================================
# UPGRADE 1 — MARKET REGIME DETECTION
# ======================================================

def detect_market_regime(df):
    latest = df.iloc[-1]
    adx    = latest.get('ADX', 0)
    if adx > REGIME_ADX_VOLATILE:
        regime = "VOLATILE"
    elif adx > REGIME_ADX_TREND:
        regime = "TRENDING"
    else:
        regime = "RANGING"
    print(f"Market Regime: {regime} (ADX {round(adx, 2)})")
    return regime

def should_skip_regime(regime):
    if regime == "VOLATILE":
        print("Regime Filter: VOLATILE market — skipping for safety")
        return True
    return False

# ======================================================
# UPGRADE 2 — ECONOMIC CALENDAR FILTER
# ======================================================

def get_economic_calendar():
    events = []
    try:
        feed_url = "https://www.forexfactory.com/ffcal_week_this.xml"
        feed     = feedparser.parse(feed_url)
        for entry in feed.entries:
            title   = entry.get("title", "").lower()
            is_high = any(kw in title for kw in HIGH_IMPACT_KEYWORDS)
            if is_high:
                events.append({
                    "title":    entry.get("title", ""),
                    "impact":   "HIGH",
                    "time_str": entry.get("ff_time_formatted", "")
                })
    except Exception as e:
        print("Calendar Fetch Error:", e)
    return events

def is_calendar_blocked():
    events = get_economic_calendar()
    now    = datetime.now(timezone.utc).replace(tzinfo=None)
    for event in events:
        time_str = event.get("time_str", "")
        if not time_str:
            continue
        try:
            event_time    = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        except Exception:
            continue
        delta_minutes = (event_time - now).total_seconds() / 60
        if -CALENDAR_BLOCK_AFTER_MIN <= delta_minutes <= CALENDAR_BLOCK_BEFORE_MIN:
            print(f"Calendar Block: '{event['title']}' in {round(delta_minutes)} min")
            return True
    return False

# ======================================================
# UPGRADE 3 — SESSION KILLZONE AI
# ======================================================

def get_active_killzone():
    now_hour = datetime.now(timezone.utc).hour
    for name, (start, end) in KILLZONES.items():
        if start <= now_hour < end:
            return name
    return None

def is_killzone_active():
    kz = get_active_killzone()
    if kz:
        print(f"Killzone Active: {kz}")
        return True
    next_kz, minutes_until = get_next_killzone()
    print(f"No Killzone Active. Next: {next_kz} in {minutes_until} min")
    return False

def get_next_killzone():
    now      = datetime.now(timezone.utc)
    now_mins = now.hour * 60 + now.minute
    best_name    = ""
    best_minutes = 9999
    for name, (start, end) in KILLZONES.items():
        start_mins = start * 60
        if start_mins > now_mins:
            diff = start_mins - now_mins
        else:
            diff = (24 * 60 - now_mins) + start_mins
        if diff < best_minutes:
            best_minutes = diff
            best_name    = name
    return best_name, best_minutes

# ======================================================
# UPGRADE 4 — DYNAMIC TP / SL AI
# ======================================================

def calculate_dynamic_tp_sl(signal_direction, entry, atr, regime):
    mults   = REGIME_MULTIPLIERS.get(regime, {"sl": 1.5, "tp": 3.0})
    sl_mult = mults["sl"]
    tp_mult = mults["tp"]
    if signal_direction == "BUY":
        sl = entry - (atr * sl_mult)
        tp = entry + (atr * tp_mult)
    elif signal_direction == "SELL":
        sl = entry + (atr * sl_mult)
        tp = entry - (atr * tp_mult)
    else:
        sl = 0
        tp = 0
    print(f"Dynamic TP/SL | Regime: {regime} | SL×{sl_mult} TP×{tp_mult}")
    return round(sl, 5), round(tp, 5)

# ======================================================
# UPGRADE 5 — DASHBOARD REPORTER
# ======================================================

def build_dashboard_report(symbol, signal, regime, killzone,
                            quality_score=None, entry_type=None,
                            entry_reason=None, bytez_reason=None):
    sym_display = SYMBOL_DISPLAY.get(symbol, symbol)
    kz_display  = killzone if killzone else "Outside Killzone"

    if signal['signal'] == "HOLD":
        return (
            f"╔══ AI SIGNAL ENGINE v5 ══╗\n"
            f"NO TRADE — {sym_display}\n"
            f"Regime   : {regime}\n"
            f"Killzone : {kz_display}\n"
            f"Trend    : {signal['trend']}\n"
            f"Prob     : {signal['probability']}%\n"
            f"Quality  : {quality_score or 'N/A'}%\n"
            f"╚═══════════════════════╝"
        )

    rr = abs(signal['tp'] - signal['entry']) / max(abs(signal['entry'] - signal['sl']), 0.00001)

    # Pip distances for context
    entry = signal['entry']
    sl    = signal['sl']
    tp    = signal['tp']
    sym   = sym_display

    if "BTC" in sym:
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        dist_unit = "USD"
    elif "JPY" in sym:
        sl_dist = round(abs(entry - sl) / 0.01, 1)
        tp_dist = round(abs(tp - entry) / 0.01, 1)
        dist_unit = "pips"
    elif "GOLD" in sym or "XAU" in sym:
        sl_dist = round(abs(entry - sl), 2)
        tp_dist = round(abs(tp - entry), 2)
        dist_unit = "USD"
    else:
        sl_dist = round(abs(entry - sl) / 0.0001, 1)
        tp_dist = round(abs(tp - entry) / 0.0001, 1)
        dist_unit = "pips"

    # Direction emoji
    dir_emoji = "🟢" if signal['signal'] == "BUY" else "🔴"

    report = (
        f"╔══ AI SIGNAL ENGINE v5 ══╗\n\n"
        f"{dir_emoji} {sym_display}  —  {signal['signal']}\n\n"
        f"── TRADE LEVELS ─────────────\n"
        f"Entry    : {signal['entry']}\n"
        f"Stop Loss: {signal['sl']}  ({sl_dist} {dist_unit})\n"
        f"Take Prof: {signal['tp']}  ({tp_dist} {dist_unit})\n"
        f"R:R Ratio: 1:{round(rr, 2)}\n\n"
        f"── ENTRY GUIDANCE ───────────\n"
        f"Order    : {entry_type or 'MARKET'}\n"
        f"Reason   : {entry_reason or '—'}\n\n"
        f"── SIGNAL QUALITY ───────────\n"
        f"Quality  : {quality_score or '?'}%\n"
        f"Prob     : {signal['probability']}%\n"
        f"AI Check : {bytez_reason or '—'}\n\n"
        f"── MARKET CONTEXT ───────────\n"
        f"Regime   : {regime}\n"
        f"Killzone : {kz_display}\n"
        f"Trend    : {signal['trend']}\n"
        f"MTF      : {signal['mtf_confirmation']}\n\n"
        f"── SMC LAYERS ───────────────\n"
        f"Sweep    : {signal['sweep']}\n"
        f"Structure: {signal['structure']}\n"
        f"OB       : {signal['order_block']}\n"
        f"FVG      : {signal['fvg']}\n"
        f"News     : {signal['news']}\n\n"
        f"📊 Tracking until TP/SL hit...\n"
        f"╚═══════════════════════╝"
    )
    return report

# ======================================================
# UPGRADE 6 — REAL SIGNAL TRACKER
# ======================================================

def check_open_signals(open_signals):
    still_open = []
    for sig in open_signals:
        symbol = sig['symbol']
        if sig['signal'] == "HOLD":
            continue
        try:
            # Use Twelve Data live price
            td_symbol = next((k for k, v in SYMBOL_DISPLAY.items() if v == symbol), symbol)
            bid, ask  = get_live_price(td_symbol)
            if bid is None:
                still_open.append(sig)
                continue

            current_price = bid
            entry         = sig['entry']
            sl            = sig['sl']
            tp            = sig['tp']
            direction     = sig['signal']
            open_time     = datetime.fromisoformat(sig['open_time'])
            now           = datetime.now()
            minutes       = round((now - open_time).total_seconds() / 60, 2)
            hours         = minutes / 60
            result        = "RUNNING"

            if direction == "BUY":
                if current_price >= tp:
                    result = "WIN"
                elif current_price <= sl:
                    result = "LOSS"
            elif direction == "SELL":
                if current_price <= tp:
                    result = "WIN"
                elif current_price >= sl:
                    result = "LOSS"

            if result == "RUNNING" and hours >= SIGNAL_EXPIRY_HOURS:
                result = "EXPIRED"

            if result != "RUNNING":
                cursor.execute("""
                    UPDATE signals
                    SET    status       = ?,
                           result       = ?,
                           minutes_taken = ?,
                           closed_time  = ?
                    WHERE  symbol = ?
                      AND  time   = ?
                """, (
                    "CLOSED", result, minutes, str(now), symbol, sig['time']
                ))
                conn.commit()
                update_winrate(result)
                # ── AUTO-LEARNING: update per-symbol intelligence ──
                if result in ("WIN", "LOSS"):
                    update_learning(symbol, result)
                    # ── v5: POST-TRADE BYTEZ ANALYSIS ──
                    bytez_post_trade_analysis(sig, result, current_price, minutes)

                emoji    = "✅" if result == "WIN" else ("❌" if result == "LOSS" else "⏰")
                if "BTC" in symbol or "BTCUSD" in symbol:
                    pnl_pips = round(abs(current_price - entry), 2)   # BTC in USD
                elif "JPY" in symbol:
                    pnl_pips = round(abs(current_price - entry) / 0.01, 1)  # JPY pairs
                elif "GOLD" in symbol or "XAU" in symbol:
                    pnl_pips = round(abs(current_price - entry), 2)   # Gold in USD
                else:
                    pnl_pips = round(abs(current_price - entry) / 0.0001, 1)  # Forex pips

                alert = (
                    f"{emoji} SIGNAL CLOSED — {result}\n\n"
                    f"Symbol   : {symbol}\n"
                    f"Direction: {direction}\n"
                    f"Entry    : {entry}\n"
                    f"Close    : {round(current_price, 5)}\n"
                    f"TP       : {tp}\n"
                    f"SL       : {sl}\n"
                    f"Pips     : {pnl_pips}\n"
                    f"Duration : {minutes} min\n"
                    f"Regime   : {sig.get('regime', 'N/A')}\n"
                    f"Killzone : {sig.get('killzone', 'N/A')}"
                )
                print(alert)
                send_telegram(alert)
                send_scoreboard()
            else:
                still_open.append(sig)

        except Exception as e:
            print(f"Tracker Error ({symbol}): {e}")
            still_open.append(sig)

    return still_open


def send_scoreboard():
    cursor.execute("SELECT result, COUNT(*) FROM signals WHERE signal != 'HOLD' GROUP BY result")
    rows    = cursor.fetchall()
    counts  = {row[0]: row[1] for row in rows}
    total_w = counts.get("WIN",     0)
    total_l = counts.get("LOSS",    0)
    total_r = counts.get("RUNNING", 0)
    expired = counts.get("EXPIRED", 0)
    total   = total_w + total_l
    winrate = round((total_w / total) * 100, 2) if total > 0 else 0.0

    cursor.execute("""
        SELECT symbol,
               SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,
               COUNT(*) AS total
        FROM   signals WHERE signal != 'HOLD'
        GROUP  BY symbol
    """)
    sym_rows  = cursor.fetchall()
    sym_lines = ""
    for row in sym_rows:
        sym, w, l, t = row
        wr  = round((w / t) * 100, 1) if t > 0 else 0.0
        sym_lines += f"  {sym:<10} W:{w}  L:{l}  WR:{wr}%\n"

    scoreboard = (
        f"📊 SIGNAL SCOREBOARD\n"
        f"{'─'*28}\n"
        f"✅ Wins    : {total_w}\n"
        f"❌ Losses  : {total_l}\n"
        f"🔄 Running : {total_r}\n"
        f"⏰ Expired : {expired}\n"
        f"🎯 Win Rate: {winrate}%\n"
        f"{'─'*28}\n"
        f"Per Symbol:\n"
        f"{sym_lines}"
        f"{'─'*28}"
    )
    print(scoreboard)
    send_telegram(scoreboard)


def get_tracker_summary():
    cursor.execute("SELECT result, COUNT(*) FROM signals WHERE signal != 'HOLD' GROUP BY result")
    rows   = cursor.fetchall()
    counts = {row[0]: row[1] for row in rows}
    w      = counts.get("WIN",     0)
    l      = counts.get("LOSS",    0)
    r      = counts.get("RUNNING", 0)
    total  = w + l
    wr     = round((w / total) * 100, 1) if total > 0 else 0.0
    return f"Tracker → W:{w} L:{l} Running:{r} | WinRate:{wr}%"

# ======================================================
# MULTI TIMEFRAME CONFIRMATION (Twelve Data)
# ======================================================

def multi_timeframe_confirmation(symbol):
    timeframes = ["5m", "15m", "1h"]
    bullish = 0
    bearish = 0
    for tf in timeframes:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval=tf)
            if df is None or df.empty:
                continue
            df["close"] = pd.to_numeric(df["Close"])
            df = df.reset_index(drop=True)
            df["EMA20"] = ta.ema(df["close"], length=20)
            df["EMA50"] = ta.ema(df["close"], length=50)
            latest = df.iloc[-1]
            if latest["EMA20"] > latest["EMA50"]:
                bullish += 1
            else:
                bearish += 1
            time.sleep(0.3)
        except Exception:
            continue

    if bullish > bearish:
        return "BULLISH"
    elif bearish > bullish:
        return "BEARISH"
    return "NEUTRAL"
# ======================================================
# AI SIGNAL GENERATOR
# ======================================================

def generate_signal(symbol, df, regime):
    latest           = df.iloc[-1]
    sym_display      = SYMBOL_DISPLAY.get(symbol, symbol)
    trend            = get_trend(latest)
    mtf_confirmation = multi_timeframe_confirmation(symbol)
    sweep            = liquidity_sweep(df)
    structure        = market_structure(df)
    order_block      = detect_order_block(df)
    fvg              = detect_fvg(df)
    news             = news_sentiment()
    probability      = calculate_probability(latest)

    current_winrate  = update_winrate("RUNNING")
    ai_weight        = reinforcement_learning(current_winrate)
    optimizer        = strategy_optimizer(current_winrate)

    probability = min(
        round((probability * ai_weight) + optimizer['confidence_boost']),
        95
    )

    # ── AUTO-LEARNING: enforce learned minimum probability ──
    min_prob = get_min_probability(sym_display)
    if probability < min_prob:
        print(f"[AutoLearn] {sym_display} probability {probability}% < learned min {min_prob}% — HOLD forced")
        return {
            "symbol": sym_display, "signal": "HOLD",
            "sweep": sweep, "structure": structure,
            "order_block": order_block, "fvg": fvg, "news": news,
            "mtf_confirmation": mtf_confirmation, "regime": regime,
            "open_price": latest['close'], "open_time": str(datetime.now()),
            "status": "HOLD", "entry": round(float(latest['close']), 5),
            "sl": 0, "tp": 0, "probability": probability,
            "trend": trend, "time": str(datetime.now()),
        }

    entry = latest['close']
    atr   = latest['ATR']

    # Load per-symbol learning summary for AI context
    learn_state = load_learning_state().get(sym_display, {})
    learn_wins   = learn_state.get("wins", 0)
    learn_losses = learn_state.get("losses", 0)
    learn_streak = learn_state.get("streak", 0)

    # ── SIGNAL DECISION ENGINE (indicator-based, reliable) ──────────────────
    rsi       = float(latest['RSI'])
    macd_v    = float(latest['MACD'])
    macd_sig  = float(latest['MACD_SIGNAL'])
    adx_v     = float(latest.get('ADX', 0))

    # Score bullish vs bearish signals
    bull_score = 0
    bear_score = 0

    # Trend (weight: 3)
    if trend == "BULLISH":
        bull_score += 3
    elif trend == "BEARISH":
        bear_score += 3

    # MTF confirmation (weight: 3)
    if mtf_confirmation == "BULLISH":
        bull_score += 3
    elif mtf_confirmation == "BEARISH":
        bear_score += 3

    # MACD crossover (weight: 2)
    if macd_v > macd_sig:
        bull_score += 2
    elif macd_v < macd_sig:
        bear_score += 2

    # RSI (weight: 1)
    if rsi < 45:
        bear_score += 1
    elif rsi > 55:
        bull_score += 1

    # Market structure (weight: 1)
    if structure == "BULLISH_BOS":
        bull_score += 1
    elif structure == "BEARISH_BOS":
        bear_score += 1

    # Order block (weight: 1)
    if order_block == "BULLISH_ORDER_BLOCK":
        bull_score += 1
    elif order_block == "BEARISH_ORDER_BLOCK":
        bear_score += 1

    # FVG (weight: 1)
    if fvg == "BULLISH_FVG":
        bull_score += 1
    elif fvg == "BEARISH_FVG":
        bear_score += 1

    # News (weight: 1)
    if news == "BULLISH_NEWS":
        bull_score += 1
    elif news == "BEARISH_NEWS":
        bear_score += 1

    # Minimum score threshold to take a trade (out of 13 max)
    # Conservative mode (loss streak >= 3) requires stronger agreement
    min_score = 7 if learn_streak >= 3 else 5

    if bull_score >= min_score and bull_score > bear_score:
        signal_direction = "BUY"
    elif bear_score >= min_score and bear_score > bull_score:
        signal_direction = "SELL"
    else:
        signal_direction = "HOLD"

    print(f"[Signal Engine] {sym_display} → {signal_direction} "
          f"(bull:{bull_score} bear:{bear_score} min:{min_score} prob:{probability}%)")
    sl, tp = calculate_dynamic_tp_sl(signal_direction, entry, atr, regime)

    # ── v5: SIGNAL QUALITY SCORE ──────────────────────────────────────────────
    active_kz = get_active_killzone()
    quality_score = calculate_quality_score(
        signal_direction, probability, bull_score, bear_score,
        regime, active_kz, mtf_confirmation,
        structure, fvg, order_block
    ) if signal_direction != "HOLD" else 0

    # ── v5: PRE-TRADE AVOID-RULES CHECK ──────────────────────────────────────
    if signal_direction != "HOLD":
        blocked, matched_rule = check_avoid_rules(
            sym_display, regime, active_kz or "Outside", mtf_confirmation
        )
        if blocked:
            print(f"[AvoidRule] {sym_display} signal BLOCKED — {matched_rule}")
            send_telegram(
                f"⛔ SIGNAL BLOCKED by learned avoid-rule\n"
                f"{sym_display} {signal_direction}\n"
                f"Rule: {matched_rule}"
            )
            signal_direction = "HOLD"
            quality_score    = 0

    # ── v5: BYTEZ CONFIRMATION FILTER ────────────────────────────────────────
    bytez_reason = "—"
    if signal_direction != "HOLD":
        bytez_ok, bytez_reason = bytez_confirmation(
            sym_display, signal_direction, probability, regime,
            active_kz, mtf_confirmation, trend, news,
            structure, fvg, order_block, bull_score, bear_score
        )
        if not bytez_ok:
            print(f"[Bytez] {sym_display} signal REJECTED — {bytez_reason}")
            send_telegram(
                f"🤖 BYTEZ REJECTED signal\n"
                f"{sym_display} {signal_direction}\n"
                f"Reason: {bytez_reason}"
            )
            signal_direction = "HOLD"
            quality_score    = 0

    # ── v5: ENTRY TYPE SUGGESTION ─────────────────────────────────────────────
    entry_type, entry_reason = ("—", "—")
    if signal_direction != "HOLD":
        entry_type, entry_reason = suggest_entry_type(
            signal_direction, entry, df, regime, fvg, order_block, atr
        )

    return {
        "symbol":           SYMBOL_DISPLAY.get(symbol, symbol),
        "signal":           signal_direction,
        "sweep":            sweep,
        "structure":        structure,
        "order_block":      order_block,
        "fvg":              fvg,
        "news":             news,
        "mtf_confirmation": mtf_confirmation,
        "regime":           regime,
        "open_price":       entry,
        "open_time":        str(datetime.now()),
        "status":           "RUNNING",
        "entry":            round(entry, 5),
        "sl":               sl,
        "tp":               tp,
        "probability":      probability,
        "trend":            trend,
        "time":             str(datetime.now()),
        "quality_score":    quality_score,
        "entry_type":       entry_type,
        "entry_reason":     entry_reason,
        "bytez_reason":     bytez_reason,
    }

# ======================================================
# SAVE SIGNAL TO DATABASE
# ======================================================

def save_to_database(signal, killzone=""):
    cursor.execute("""
    INSERT INTO signals (
        symbol, signal, probability, trend,
        regime, killzone, entry, sl, tp,
        status, result, minutes_taken, time, closed_time
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal['symbol'],
        signal['signal'],
        signal['probability'],
        signal['trend'],
        signal.get('regime', ''),
        killzone,
        signal.get('entry', 0),
        signal.get('sl', 0),
        signal.get('tp', 0),
        signal.get('status', 'RUNNING'),
        "RUNNING",
        signal.get('minutes_taken', 0),
        signal['time'],
        None
    ))
    conn.commit()

# ======================================================
# LOAD / SAVE HISTORY
# ======================================================

def load_history():
    try:
        with open(SIGNAL_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_history(data):
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_open_signals():
    try:
        with open(OPEN_SIGNALS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_open_signals(data):
    with open(OPEN_SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ======================================================
# LEARNING ENGINE
# ======================================================

def learning_engine(history):
    if len(history) < 10:
        return
    total            = len(history)
    buy_count        = 0
    sell_count       = 0
    high_probability = 0
    for signal in history[-20:]:
        if signal['signal'] == 'BUY':
            buy_count += 1
        if signal['signal'] == 'SELL':
            sell_count += 1
        if signal['probability'] >= 80:
            high_probability += 1
    confidence = (high_probability / min(total, 20)) * 100
    print(f"AI Confidence Score: {round(confidence, 2)}%")
    print(f"BUY Signals: {buy_count}")
    print(f"SELL Signals: {sell_count}")

# ======================================================
# DAILY REPORT
# ======================================================

def daily_report(history):
    total           = len(history)
    buys            = len([x for x in history if x['signal'] == 'BUY'])
    sells           = len([x for x in history if x['signal'] == 'SELL'])
    avg_probability = np.mean([x['probability'] for x in history])
    report = (
        f"DAILY AI REPORT\n\n"
        f"Total Signals: {total}\n"
        f"BUY Signals: {buys}\n"
        f"SELL Signals: {sells}\n"
        f"Average Probability: {round(avg_probability, 2)}%"
    )
    print(report)
    send_telegram(report)

# ======================================================
# MAIN LOOP  (v4 — Twelve Data, no MT5)
# ======================================================

history      = load_history()
open_signals = load_open_signals()

print("Advanced AI Smart Money Signal Engine v5 Started (yfinance)")
send_telegram(
    "🚀 AI Signal Engine v5 Started\n"
    "─── Core ───────────────────\n"
    "[+] yfinance · 11 Symbols · One-Trade Guard\n"
    "[+] Regime · Calendar · Killzone · Dynamic TP/SL\n"
    "[+] Auto-Learning · Signal Tracker\n"
    "─── v5 Upgrades ─────────────\n"
    "[+] Bytez Confirmation (pre-trade AI check)\n"
    "[+] Signal Quality Score (alerts ≥ 65% only)\n"
    "[+] LIMIT vs MARKET entry suggestion\n"
    "[+] Rich alerts: Entry/TP/SL/RR/Reason\n"
    "[+] Post-Trade Bytez Analysis\n"
    "[+] Pattern Memory + Avoid-Rules\n"
    "[+] Daily Learning Report"
)

send_scoreboard()

while True:

    try:

        gui = read_gui_filters()

        print("\n==============================")
        print("NEW AI MARKET SCAN STARTED")
        print(get_tracker_summary())
        print("==============================\n")

        if open_signals:
            print(f"Tracking {len(open_signals)} open signal(s)...")
            open_signals = check_open_signals(open_signals)
            save_open_signals(open_signals)

        active_killzone = get_active_killzone()

        if gui["killzone"] and not is_killzone_active():
            next_kz, mins_until = get_next_killzone()
            print(f"Outside Killzone. Next: {next_kz} in {mins_until} min. Sleeping 5 min.")
            time.sleep(300)
            continue

        if gui["calendar"] and is_calendar_blocked():
            block_msg = "HIGH IMPACT EVENT NEAR — Skipping scan to avoid slippage."
            print(block_msg)
            if gui["telegram"]:
                send_telegram(f"Calendar Block Active: {block_msg}")
            time.sleep(300)
            continue

        for symbol in SYMBOLS:

            try:

                sym_display = SYMBOL_DISPLAY.get(symbol, symbol)
                print(f"\nChecking {sym_display}...")

                # ── ONE TRADE PER SYMBOL GUARD ──
                if symbol_has_open_trade(sym_display, open_signals):
                    print(f"[Guard] {sym_display} already has an open trade — skipping until closed.")
                    continue

                df = get_market_data(symbol)

                if df is None:
                    print(f"No data for {sym_display}")
                    continue

                df = add_indicators(df)

                if gui["spread"] and not spread_filter(symbol):
                    print(f"High Spread Skipping {sym_display}")
                    continue

                if gui["volatility"] and not volatility_filter(df):
                    print(f"Low Volatility Skipping {sym_display}")
                    continue

                if gui["market_quality"] and not market_quality_filter(df):
                    print(f"Weak Market Skipping {sym_display}")
                    continue

                regime = detect_market_regime(df)

                if gui["regime"] and should_skip_regime(regime):
                    if gui["telegram"]:
                        send_telegram(f"VOLATILE Regime — {sym_display} skipped")
                    continue

                signal = generate_signal(symbol, df, regime)

                history.append(signal)
                save_to_database(signal, killzone=active_killzone or "")
                save_history(history)
                learning_engine(history)

                # ── v5: quality gate — only alert high-confidence trades ──
                q_score = signal.get("quality_score", 0)
                if signal['signal'] != "HOLD" and q_score < SIGNAL_QUALITY_MIN:
                    print(f"[QualityGate] {sym_display} score {q_score}% < {SIGNAL_QUALITY_MIN}% — suppressing alert")
                    continue

                dashboard = build_dashboard_report(
                    symbol, signal, regime, active_killzone,
                    quality_score=q_score,
                    entry_type=signal.get("entry_type"),
                    entry_reason=signal.get("entry_reason"),
                    bytez_reason=signal.get("bytez_reason"),
                )
                print(dashboard)

                if gui["telegram"]:
                    send_telegram(dashboard)

                if signal['signal'] in ("BUY", "SELL"):
                    signal['killzone'] = active_killzone or ""
                    open_signals.append(signal)
                    save_open_signals(open_signals)
                    print(f"Signal added to tracker. Total open: {len(open_signals)}")

                # Small delay between symbols to respect API rate limit
                time.sleep(1)

            except Exception as symbol_error:
                print(f"SYMBOL ERROR ({symbol}):", symbol_error)
                if gui["telegram"]:
                    send_telegram(f"Symbol Error {symbol}: {symbol_error}")
                continue

        print("\nWaiting 5 Minutes...\n")
        # ── v5: send daily learning report once per day ──
        send_daily_learning_report()
        time.sleep(300)

    except Exception as e:
        print("MAIN LOOP ERROR:", e)
        send_telegram(f"Main Loop Error: {e}")
        time.sleep(60)
