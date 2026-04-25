import math
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

FMP_API_KEY = st.secrets.get("FMP_API_KEY", "")
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Stock Trading Dashboard Pro",
    page_icon="📈",
    layout="wide"
)

# =========================
# HELPERS
# =========================
def to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def fmt_num(value, decimals=2):
    value = to_float(value)
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_large_number(value):
    value = to_float(value)
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:,.0f}"


def first_non_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def normalize_percent_like(value):
    value = to_float(value)
    if value is None:
        return None
    return value * 100 if abs(value) <= 1 else value


@st.cache_data(ttl=600)
def fetch_company_name_yahoo(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        name = info.get("shortName") or info.get("longName") or ticker

        suffixes = [
            " Corporation", " Corp.", " Corp", " Inc.", " Inc",
            " Ltd.", " Ltd", " Holdings", " Group", " PLC", " plc",
            " Company", " Co.", " Co"
        ]
        for s in suffixes:
            if name.endswith(s):
                name = name[:-len(s)].strip()

        return name
    except Exception:
        return ticker


@st.cache_data(ttl=600)
def fetch_earnings_date_yahoo(ticker):
    try:
        tk = yf.Ticker(ticker)

        cal = tk.calendar
        if cal is not None:
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                for idx in cal.index:
                    idx_str = str(idx).lower()
                    if "earn" in idx_str:
                        row = cal.loc[idx]
                        if isinstance(row, pd.Series) and len(row) > 0:
                            val = row.iloc[0]
                            if pd.notna(val):
                                return pd.to_datetime(val).strftime("%Y-%m-%d")
            elif isinstance(cal, dict):
                for k, v in cal.items():
                    if "earn" in str(k).lower():
                        if isinstance(v, (list, tuple)) and len(v) > 0 and pd.notna(v[0]):
                            return pd.to_datetime(v[0]).strftime("%Y-%m-%d")
                        if not isinstance(v, (list, tuple)) and pd.notna(v):
                            return pd.to_datetime(v).strftime("%Y-%m-%d")

        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                next_dt = ed.index[0]
                return pd.to_datetime(next_dt).strftime("%Y-%m-%d")
        except Exception:
            pass

        return "N/A"
    except Exception:
        return "N/A"


# =========================
# TECHNICAL INDICATORS
# =========================
def compute_indicators(df):
    df = df.copy()
    close = df["Close"]

    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Impulse_MACD"] = df["MACD"] - df["Signal_Line"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# =========================
# INTERPRETATION
# =========================
def classify_trend(close_price, ema50_value, ema200_value):
    if close_price > ema50_value and ema50_value > ema200_value * 1.01:
        return "Bullish Trend"
    if close_price < ema50_value and ema50_value < ema200_value * 0.99:
        return "Bearish Trend"
    return "Mixed / Transition"


def classify_rsi(rsi_value):
    rsi_value = to_float(rsi_value)
    if rsi_value is None:
        return "N/A"
    if rsi_value >= 70:
        return "Overbought"
    if rsi_value <= 30:
        return "Oversold"
    return "Neutral"


def classify_macd(macd_value, signal_value):
    macd_value = to_float(macd_value)
    signal_value = to_float(signal_value)
    if macd_value is None or signal_value is None:
        return "N/A"
    if macd_value > signal_value:
        return "Bullish Momentum"
    if macd_value < signal_value:
        return "Bearish Momentum"
    return "Neutral Momentum"


def build_setup_verdict(trend_state, rsi_state, macd_state, rsi_value):
    rsi_value = to_float(rsi_value)
    if trend_state == "Bullish Trend" and macd_state == "Bullish Momentum":
        if rsi_value is not None and rsi_value >= 70:
            return "Bullish but extended"
        return "Constructive bullish setup"
    if trend_state == "Bearish Trend" and macd_state == "Bullish Momentum":
        return "Bearish trend with improving momentum"
    if trend_state == "Bearish Trend" and macd_state == "Bearish Momentum":
        return "Weak setup"
    return "Mixed setup"


def valuation_verdict(trailing_pe):
    trailing_pe = to_float(trailing_pe)
    if trailing_pe is None:
        return "Unclear"
    if trailing_pe < 20:
        return "Attractive"
    if trailing_pe < 30:
        return "Fair"
    return "Expensive"


def smart_valuation_layer(trailing_pe, forward_pe, revenue_growth, rule_of_40, peg):
    trailing_pe = to_float(trailing_pe)
    forward_pe = to_float(forward_pe)
    revenue_growth = to_float(revenue_growth)
    rule_of_40 = to_float(rule_of_40)
    peg = to_float(peg)

    pe_used = forward_pe if forward_pe is not None else trailing_pe

    if pe_used is None and revenue_growth is None and peg is None:
        return {
            "valuation_style": "Insufficient data",
            "growth_adjusted_view": "Cannot judge valuation vs growth"
        }

    if peg is not None:
        if peg < 1:
            return {
                "valuation_style": "PEG-Attractive",
                "growth_adjusted_view": "Valuation looks attractive relative to growth"
            }
        if peg <= 2:
            return {
                "valuation_style": "PEG-Reasonable",
                "growth_adjusted_view": "Valuation looks reasonable relative to growth"
            }
        return {
            "valuation_style": "PEG-Expensive",
            "growth_adjusted_view": "Valuation looks expensive relative to growth"
        }

    if pe_used is not None and revenue_growth is not None:
        if pe_used >= 35:
            if revenue_growth >= 0.40 and (rule_of_40 is not None and rule_of_40 >= 60):
                return {
                    "valuation_style": "Growth-justified premium",
                    "growth_adjusted_view": "Premium valuation appears supported by strong growth and profitability"
                }
            if revenue_growth >= 0.20:
                return {
                    "valuation_style": "Rich but plausible",
                    "growth_adjusted_view": "Valuation is rich, but some growth support exists"
                }
            return {
                "valuation_style": "Overpriced vs growth",
                "growth_adjusted_view": "Valuation looks too high for the observed growth"
            }

        if 20 <= pe_used < 35:
            if revenue_growth >= 0.15:
                return {
                    "valuation_style": "Reasonably valued growth",
                    "growth_adjusted_view": "Valuation appears reasonable relative to growth"
                }
            return {
                "valuation_style": "Full valuation",
                "growth_adjusted_view": "Valuation is not cheap and growth support is limited"
            }

        if pe_used < 20:
            if revenue_growth >= 0.10:
                return {
                    "valuation_style": "Attractive growth valuation",
                    "growth_adjusted_view": "Valuation appears attractive for the growth profile"
                }
            return {
                "valuation_style": "Value / mature",
                "growth_adjusted_view": "Lower valuation likely reflects a slower-growth profile"
            }

    return {
        "valuation_style": "Unclear",
        "growth_adjusted_view": "Valuation cannot be judged with confidence"
    }


def entry_timing_score(trend_state, rsi_value, macd_value, signal_value):
    score = 0

    if trend_state == "Bullish Trend":
        score += 3
    elif trend_state == "Mixed / Transition":
        score += 1
    else:
        score -= 2

    rsi_value = to_float(rsi_value)
    if rsi_value is not None:
        if 45 <= rsi_value <= 65:
            score += 2
        elif rsi_value < 30:
            score += 1
        elif rsi_value > 75:
            score -= 2

    if to_float(macd_value) is not None and to_float(signal_value) is not None:
        if macd_value > signal_value:
            score += 2
        else:
            score -= 1

    if score >= 6:
        return score, "Strong"
    if score >= 3:
        return score, "Moderate"
    return score, "Weak"


def trade_decision(setup_verdict, smart_view, trend_state):
    style = smart_view.get("valuation_style", "")

    if setup_verdict in ["Constructive bullish setup", "Bullish but extended"]:
        if style in ["Growth-justified premium", "Reasonably valued growth", "Attractive growth valuation", "PEG-Attractive", "PEG-Reasonable"]:
            return "Buy shares, ITM call LEAPS, or bull put spread"
        if style in ["Rich but plausible", "Full valuation"]:
            return "Prefer bull put spread or buy on pullbacks"
        if style in ["Overpriced vs growth", "PEG-Expensive"]:
            return "Prefer defined-risk premium selling over outright call buying"

    if setup_verdict == "Bearish trend with improving momentum":
        return "Wait for confirmation or use small defined-risk bullish structures"

    if trend_state == "Bearish Trend":
        return "Avoid aggressive bullish entries"

    return "Wait for cleaner entry"


def options_idea(trend_state, smart_view, macd_state):
    style = smart_view.get("valuation_style", "")

    if trend_state == "Bullish Trend" and macd_state == "Bullish Momentum":
        if style in ["Growth-justified premium", "Rich but plausible", "PEG-Expensive", "Overpriced vs growth"]:
            return "Bull put spread preferred over outright call buying"
        if style in ["Reasonably valued growth", "Attractive growth valuation", "PEG-Attractive", "PEG-Reasonable"]:
            return "ITM call LEAPS or bull put spread"
        return "Bull put spread or wait for more clarity"

    if trend_state == "Bearish Trend" and macd_state == "Bullish Momentum":
        return "Small defined-risk bullish spread only after confirmation"

    if trend_state == "Bearish Trend":
        return "No aggressive bullish options setup"

    return "Neutral / wait"


# =========================
# ENTRY ZONES
# =========================
def entry_zones(df):
    recent = df.tail(20).copy()
    close = float(df["Close"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])

    recent_low = float(recent["Low"].min())
    recent_high = float(recent["High"].max())

    support_1 = min(close, ema50)
    support_2 = recent_low
    resistance_1 = recent_high
    resistance_2 = max(recent_high, ema200)

    buy_zone_low = min(support_1, support_2)
    buy_zone_high = max(support_1, support_2)

    return {
        "support_1": support_1,
        "support_2": support_2,
        "resistance_1": resistance_1,
        "resistance_2": resistance_2,
        "buy_zone_low": buy_zone_low,
        "buy_zone_high": buy_zone_high,
    }


# =========================
# OPTIONS OPTIMIZER
# =========================
def round_down_strike(price, step=5):
    return math.floor(price / step) * step


def options_optimizer(latest_close, trend_state, setup_verdict, timing_label):
    latest_close = to_float(latest_close)
    if latest_close is None:
        return {}

    sell_strike = round_down_strike(latest_close * 0.92, 5)
    buy_strike = round_down_strike(latest_close * 0.87, 5)

    if buy_strike >= sell_strike:
        buy_strike = sell_strike - 5

    if trend_state == "Bullish Trend":
        if setup_verdict == "Bullish but extended":
            dte = "30-45 DTE"
            idea = "Wait for pullback or use conservative bull put spread"
        elif timing_label == "Strong":
            dte = "30-45 DTE"
            idea = "Bull put spread or ITM LEAPS"
        else:
            dte = "30-45 DTE"
            idea = "Bull put spread preferred"
    else:
        dte = "Wait"
        idea = "No aggressive bullish structure"

    itm_leaps_strike = round_down_strike(latest_close * 0.80, 5)

    return {
        "spread_sell": sell_strike,
        "spread_buy": buy_strike,
        "spread_width": sell_strike - buy_strike,
        "dte": dte,
        "idea": idea,
        "leaps_strike": itm_leaps_strike,
    }


# =========================
# EV/EBITDA RELATIVE VIEW
# =========================
def ev_ebitda_relative_view(current_ev_ebitda):
    current_ev_ebitda = to_float(current_ev_ebitda)
    if current_ev_ebitda is None:
        return {"status": "Unavailable", "comparison": "Historical EV/EBITDA comparison unavailable"}

    baseline = 22.0
    premium_pct = ((current_ev_ebitda / baseline) - 1) * 100

    if premium_pct > 25:
        status = "Premium vs baseline"
    elif premium_pct < -15:
        status = "Discount vs baseline"
    else:
        status = "Near baseline"

    comparison = f"Current {current_ev_ebitda:.1f}x vs baseline {baseline:.1f}x ({premium_pct:+.1f}%)"
    return {"status": status, "comparison": comparison}


# =========================
# OPTIONS / IV HELPERS
# =========================
def classify_iv(iv):
    iv = to_float(iv)
    if iv is None:
        return "N/A"
    if iv < 0.25:
        return "Low IV"
    if iv < 0.45:
        return "Moderate IV"
    return "High IV"


def options_setup_score(trend_state, timing_score, iv_percentile_approx, iv_regime, options_view):
    score = 0

    if trend_state == "Bullish Trend":
        score += 3
    elif trend_state == "Mixed / Transition":
        score += 1
    else:
        score -= 2

    timing_score = to_float(timing_score)
    if timing_score is not None:
        score += timing_score

    iv_percentile_approx = to_float(iv_percentile_approx)
    if iv_percentile_approx is not None:
        if iv_percentile_approx >= 70:
            score += 3
        elif iv_percentile_approx <= 30:
            score += 2
        else:
            score += 1

    if iv_regime == "High IV":
        score += 2
    elif iv_regime == "Moderate IV":
        score += 1

    if isinstance(options_view, str):
        ov = options_view.lower()
        if "bull put spread preferred" in ov:
            score += 2
        elif "itm call leaps" in ov:
            score += 2
        elif "neutral" in ov or "wait" in ov:
            score -= 1
        elif "no aggressive bullish" in ov:
            score -= 3

    return score


def options_setup_label(iv_percentile_approx, iv_regime, options_view, trend_state):
    iv_percentile_approx = to_float(iv_percentile_approx)
    ov = (options_view or "").lower()

    if trend_state == "Bearish Trend":
        return "Wait / Weak Setup"

    if iv_percentile_approx is not None and iv_percentile_approx >= 70:
        return "Best for Premium Selling"

    if iv_percentile_approx is not None and iv_percentile_approx <= 30:
        return "Best for LEAPS"

    if "bull put spread preferred" in ov:
        return "Best for Premium Selling"

    if "itm call leaps" in ov:
        return "Best for LEAPS"

    return "Balanced / Mixed"


@st.cache_data(ttl=900)
def fetch_iv_data_yahoo(ticker):
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options

        if not expirations:
            return {
                "atm_iv": None,
                "iv_percentile_approx": None,
                "iv_regime": "N/A",
                "iv_note": "No option expirations available from Yahoo"
            }

        hist = tk.history(period="5d")
        if hist.empty:
            return {
                "atm_iv": None,
                "iv_percentile_approx": None,
                "iv_regime": "N/A",
                "iv_note": "No recent price available for IV calculation"
            }

        spot = float(hist["Close"].iloc[-1])

        nearest_exp = expirations[0]
        chain = tk.option_chain(nearest_exp)
        calls = chain.calls.copy()

        if calls.empty or "strike" not in calls.columns or "impliedVolatility" not in calls.columns:
            return {
                "atm_iv": None,
                "iv_percentile_approx": None,
                "iv_regime": "N/A",
                "iv_note": "No usable call IV data from Yahoo"
            }

        valid = calls.copy()
        valid = valid[
            valid["strike"].notna() &
            valid["impliedVolatility"].notna()
        ].copy()

        valid = valid[
            (valid["impliedVolatility"] >= 0.05) &
            (valid["impliedVolatility"] <= 3.00)
        ].copy()

        if valid.empty:
            return {
                "atm_iv": None,
                "iv_percentile_approx": None,
                "iv_regime": "N/A",
                "iv_note": "Yahoo returned no valid ATM IV rows"
            }

        valid["distance"] = (valid["strike"] - spot).abs()
        valid = valid.sort_values("distance")

        atm_row = valid.iloc[0]
        atm_iv = to_float(atm_row["impliedVolatility"])

        atm_ivs = []
        for exp in expirations[:8]:
            try:
                ch = tk.option_chain(exp)
                c = ch.calls.copy()
                if c.empty or "strike" not in c.columns or "impliedVolatility" not in c.columns:
                    continue

                c = c[
                    c["strike"].notna() &
                    c["impliedVolatility"].notna() &
                    (c["impliedVolatility"] >= 0.05) &
                    (c["impliedVolatility"] <= 3.00)
                ].copy()

                if c.empty:
                    continue

                c["distance"] = (c["strike"] - spot).abs()
                row = c.sort_values("distance").iloc[0]
                iv_val = to_float(row["impliedVolatility"])
                if iv_val is not None:
                    atm_ivs.append(iv_val)
            except Exception:
                continue

        iv_percentile_approx = None
        if atm_iv is not None and len(atm_ivs) >= 3:
            arr = np.array(sorted(atm_ivs))
            iv_percentile_approx = float((arr <= atm_iv).sum() / len(arr) * 100)

        return {
            "atm_iv": atm_iv,
            "iv_percentile_approx": iv_percentile_approx,
            "iv_regime": classify_iv(atm_iv),
            "iv_note": "IV percentile is experimental and based on available expirations, not 1-year historical IV"
        }

    except Exception:
        return {
            "atm_iv": None,
            "iv_percentile_approx": None,
            "iv_regime": "N/A",
            "iv_note": "Yahoo options data unavailable"
        }


# =========================
# DATA FETCHERS
# =========================
@st.cache_data(ttl=600)
def fetch_price_data_yahoo(ticker, period):
    for _ in range(2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df.dropna(subset=["Close"]).copy()
        except Exception:
            time.sleep(2)
    return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_yahoo_backup_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        fast_info = dict(yf.Ticker(ticker).fast_info)

        return {
            "source": "Yahoo Backup",
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "earningsGrowth": info.get("earningsGrowth"),
            "revenueGrowth": info.get("revenueGrowth"),
            "ebitdaMargins": info.get("ebitdaMargins"),
            "marketCap": first_non_none(info.get("marketCap"), fast_info.get("marketCap")),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=600)
def fetch_fmp_fundamentals(ticker, api_key):
    if not api_key:
        return {}

    try:
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        ratios_ttm_url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={api_key}"
        key_metrics_ttm_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={api_key}"

        profile = requests.get(profile_url, timeout=20).json()
        ratios = requests.get(ratios_ttm_url, timeout=20).json()
        metrics = requests.get(key_metrics_ttm_url, timeout=20).json()

        profile = profile[0] if isinstance(profile, list) and profile else {}
        ratios = ratios[0] if isinstance(ratios, list) and ratios else {}
        metrics = metrics[0] if isinstance(metrics, list) and metrics else {}

        return {
            "source": "FMP",
            "trailingPE": first_non_none(ratios.get("priceEarningsRatioTTM"), metrics.get("peRatioTTM")),
            "forwardPE": profile.get("priceEarningsRatio"),
            "earningsGrowth": None,
            "revenueGrowth": metrics.get("revenueGrowth"),
            "ebitdaMargins": metrics.get("ebitdaMargin"),
            "marketCap": profile.get("mktCap"),
            "beta": profile.get("beta"),
            "fiftyTwoWeekHigh": None,
            "fiftyTwoWeekLow": None,
            "enterpriseToEbitda": first_non_none(
                metrics.get("enterpriseValueOverEBITDATTM"),
                metrics.get("enterpriseValueOverEBITDA")
            ),
            "peg": metrics.get("pegRatio")
        }
    except Exception:
        return {}


@st.cache_data(ttl=600)
def fetch_finnhub_fundamentals(ticker, api_key):
    if not api_key:
        return {}

    try:
        url = "https://finnhub.io/api/v1/stock/metric"
        headers = {"X-Finnhub-Token": api_key}
        params = {"symbol": ticker, "metric": "all"}

        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        metric = data.get("metric", {})

        return {
            "source": "Finnhub",
            "trailingPE": metric.get("peTTM"),
            "forwardPE": None,
            "earningsGrowth": None,
            "revenueGrowth": first_non_none(
                metric.get("revenueGrowthTTM"),
                metric.get("revenueGrowthAnnual"),
                metric.get("revenueGrowth5Y")
            ),
            "ebitdaMargins": None if metric.get("ebitdaMarginTTM") is None else metric.get("ebitdaMarginTTM") / 100,
            "marketCap": None if metric.get("marketCapitalization") is None else metric.get("marketCapitalization") * 1_000_000,
            "beta": metric.get("beta"),
            "fiftyTwoWeekHigh": metric.get("52WeekHigh"),
            "fiftyTwoWeekLow": metric.get("52WeekLow"),
            "enterpriseToEbitda": metric.get("evToEbitda"),
            "peg": None
        }
    except Exception:
        return {}


def merge_fundamentals(fmp_data, finnhub_data, yahoo_backup):
    sources = []
    if fmp_data:
        sources.append("FMP")
    if finnhub_data:
        sources.append("Finnhub")
    if yahoo_backup:
        sources.append("Yahoo Backup")

    return {
        "source_used": " + ".join(sources) if sources else "None",
        "trailingPE": first_non_none(
            fmp_data.get("trailingPE") if fmp_data else None,
            finnhub_data.get("trailingPE") if finnhub_data else None,
            yahoo_backup.get("trailingPE") if yahoo_backup else None
        ),
        "forwardPE": first_non_none(
            fmp_data.get("forwardPE") if fmp_data else None,
            yahoo_backup.get("forwardPE") if yahoo_backup else None
        ),
        "earningsGrowth": first_non_none(
            fmp_data.get("earningsGrowth") if fmp_data else None,
            yahoo_backup.get("earningsGrowth") if yahoo_backup else None
        ),
        "revenueGrowth": first_non_none(
            fmp_data.get("revenueGrowth") if fmp_data else None,
            finnhub_data.get("revenueGrowth") if finnhub_data else None,
            yahoo_backup.get("revenueGrowth") if yahoo_backup else None
        ),
        "ebitdaMargins": first_non_none(
            fmp_data.get("ebitdaMargins") if fmp_data else None,
            finnhub_data.get("ebitdaMargins") if finnhub_data else None,
            yahoo_backup.get("ebitdaMargins") if yahoo_backup else None
        ),
        "marketCap": first_non_none(
            fmp_data.get("marketCap") if fmp_data else None,
            finnhub_data.get("marketCap") if finnhub_data else None,
            yahoo_backup.get("marketCap") if yahoo_backup else None
        ),
        "beta": first_non_none(
            fmp_data.get("beta") if fmp_data else None,
            finnhub_data.get("beta") if finnhub_data else None,
            yahoo_backup.get("beta") if yahoo_backup else None
        ),
        "fiftyTwoWeekHigh": first_non_none(
            fmp_data.get("fiftyTwoWeekHigh") if fmp_data else None,
            finnhub_data.get("fiftyTwoWeekHigh") if finnhub_data else None,
            yahoo_backup.get("fiftyTwoWeekHigh") if yahoo_backup else None
        ),
        "fiftyTwoWeekLow": first_non_none(
            fmp_data.get("fiftyTwoWeekLow") if fmp_data else None,
            finnhub_data.get("fiftyTwoWeekLow") if finnhub_data else None,
            yahoo_backup.get("fiftyTwoWeekLow") if yahoo_backup else None
        ),
        "enterpriseToEbitda": first_non_none(
            fmp_data.get("enterpriseToEbitda") if fmp_data else None,
            finnhub_data.get("enterpriseToEbitda") if finnhub_data else None,
            yahoo_backup.get("enterpriseToEbitda") if yahoo_backup else None
        ),
        "peg": fmp_data.get("peg") if fmp_data else None
    }


# =========================
# MAIN LOADER
# =========================
@st.cache_data(ttl=600)
def load_analysis(ticker, period, fmp_api_key, finnhub_api_key):
    stock_data = fetch_price_data_yahoo(ticker, period)
    if stock_data.empty:
        return None

    stock_data = compute_indicators(stock_data)
    company_name = fetch_company_name_yahoo(ticker)
    earnings_date = fetch_earnings_date_yahoo(ticker)

    fmp_data = fetch_fmp_fundamentals(ticker, fmp_api_key)
    finnhub_data = fetch_finnhub_fundamentals(ticker, finnhub_api_key)
    yahoo_backup = fetch_yahoo_backup_fundamentals(ticker)

    data = merge_fundamentals(fmp_data, finnhub_data, yahoo_backup)

    trailing_pe = data.get("trailingPE")
    forward_pe = data.get("forwardPE")
    earnings_growth = data.get("earningsGrowth")
    revenue_growth = data.get("revenueGrowth")
    ebitda_margin = data.get("ebitdaMargins")
    market_cap = data.get("marketCap")
    beta = data.get("beta")
    fifty_two_high = data.get("fiftyTwoWeekHigh")
    fifty_two_low = data.get("fiftyTwoWeekLow")
    ev_to_ebitda = data.get("enterpriseToEbitda")
    peg = data.get("peg")

    forward_pe_val = to_float(forward_pe)
    earnings_growth_val = to_float(earnings_growth)

    if peg is None and forward_pe_val is not None and earnings_growth_val not in [None, 0]:
        peg = forward_pe_val / (earnings_growth_val * 100)

    rg_pts = normalize_percent_like(revenue_growth)
    em_pts = normalize_percent_like(ebitda_margin)
    rule_of_40 = None if rg_pts is None or em_pts is None else rg_pts + em_pts

    last = stock_data.iloc[-1]
    latest_close = float(last["Close"])
    latest_ema50 = float(last["EMA50"])
    latest_ema200 = float(last["EMA200"])
    latest_rsi = to_float(last["RSI"])
    latest_macd = float(last["MACD"])
    latest_signal = float(last["Signal_Line"])

    trend_state = classify_trend(latest_close, latest_ema50, latest_ema200)
    rsi_state = classify_rsi(latest_rsi)
    macd_state = classify_macd(latest_macd, latest_signal)
    setup_verdict = build_setup_verdict(trend_state, rsi_state, macd_state, latest_rsi)

    valuation = valuation_verdict(trailing_pe)
    smart_view = smart_valuation_layer(trailing_pe, forward_pe, revenue_growth, rule_of_40, peg)
    timing_score, timing_label = entry_timing_score(trend_state, latest_rsi, latest_macd, latest_signal)
    trade_view = trade_decision(setup_verdict, smart_view, trend_state)
    options_view = options_idea(trend_state, smart_view, macd_state)

    zones = entry_zones(stock_data)
    opt = options_optimizer(latest_close, trend_state, setup_verdict, timing_label)
    ev_rel = ev_ebitda_relative_view(ev_to_ebitda)
    iv_data = fetch_iv_data_yahoo(ticker)

    fundamentals = pd.DataFrame([
        ["Data Source", data.get("source_used")],
        ["Next Earnings Date", earnings_date],
        ["Market Cap", fmt_large_number(market_cap)],
        ["Trailing P/E", fmt_num(trailing_pe)],
        ["Forward P/E", fmt_num(forward_pe)],
        ["EV / EBITDA", fmt_num(ev_to_ebitda)],
        ["PEG", fmt_num(peg)],
        ["Revenue Growth", "N/A" if rg_pts is None else f"{rg_pts:.1f}%"],
        ["EBITDA Margin", "N/A" if em_pts is None else f"{em_pts:.1f}%"],
        ["Rule of 40", fmt_num(rule_of_40)],
        ["52 Week High", fmt_num(fifty_two_high)],
        ["52 Week Low", fmt_num(fifty_two_low)],
        ["Beta", fmt_num(beta, 3)],
    ], columns=["Metric", "Value"])

    return {
        "ticker": ticker,
        "company_name": company_name,
        "earnings_date": earnings_date,
        "data": stock_data,
        "fundamentals": fundamentals,
        "latest_close": latest_close,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "ev_to_ebitda": ev_to_ebitda,
        "peg": peg,
        "rule_of_40": rule_of_40,
        "trend_state": trend_state,
        "rsi_state": rsi_state,
        "macd_state": macd_state,
        "setup_verdict": setup_verdict,
        "valuation": valuation,
        "smart_view": smart_view,
        "timing_score": timing_score,
        "timing_label": timing_label,
        "trade_view": trade_view,
        "options_view": options_view,
        "zones": zones,
        "opt": opt,
        "ev_rel": ev_rel,
        "source_used": data.get("source_used"),
        "atm_iv": iv_data.get("atm_iv"),
        "iv_percentile_approx": iv_data.get("iv_percentile_approx"),
        "iv_regime": iv_data.get("iv_regime"),
        "iv_note": iv_data.get("iv_note"),
    }


# =========================
# WATCHLIST / UNIVERSE SCANNER
# =========================
@st.cache_data(ttl=600)
def scan_watchlist(tickers, period, fmp_api_key, finnhub_api_key):
    rows = []

    for ticker in tickers:
        try:
            result = load_analysis(ticker, period, fmp_api_key, finnhub_api_key)
            if result is None:
                continue

            options_score = options_setup_score(
                trend_state=result["trend_state"],
                timing_score=result["timing_score"],
                iv_percentile_approx=result.get("iv_percentile_approx"),
                iv_regime=result.get("iv_regime"),
                options_view=result["options_view"]
            )

            setup_label = options_setup_label(
                iv_percentile_approx=result.get("iv_percentile_approx"),
                iv_regime=result.get("iv_regime"),
                options_view=result["options_view"],
                trend_state=result["trend_state"]
            )

            rows.append({
                "Ticker": ticker,
                "Last Close": to_float(result["latest_close"]),
                "Trend": result["trend_state"],
                "Timing Score": to_float(result["timing_score"]),
                "Timing": f'{result["timing_score"]} ({result["timing_label"]})',
                "ATM IV": None if result.get("atm_iv") is None else round(result["atm_iv"] * 100, 1),
                "IV %ile": None if result.get("iv_percentile_approx") is None else round(result["iv_percentile_approx"], 0),
                "IV Regime": result.get("iv_regime", "N/A"),
                "Setup Label": setup_label,
                "Options Score": options_score,
                "Valuation Style": result["smart_view"]["valuation_style"],
                "Trade Idea": result["options_view"],
                "Fwd P/E": to_float(result["forward_pe"]),
                "PEG": to_float(result["peg"]),
                "Rule of 40": to_float(result["rule_of_40"]),
                "Source": result["source_used"],
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        by=["Options Score", "Timing Score", "IV %ile"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    display_df = df.copy()
    for col in ["Last Close", "Fwd P/E", "PEG", "Rule of 40"]:
        display_df[col] = display_df[col].apply(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")

    display_df["ATM IV"] = display_df["ATM IV"].apply(lambda x: "N/A" if pd.isna(x) else f"{x:.1f}%")
    display_df["IV %ile"] = display_df["IV %ile"].apply(lambda x: "N/A" if pd.isna(x) else f"{int(x)}")
    display_df["Options Score"] = display_df["Options Score"].apply(lambda x: "N/A" if pd.isna(x) else f"{int(x)}")

    return display_df


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Controls")

watchlist = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AVGO", "MU", "NFLX", "ORCL"]
ticker = st.sidebar.text_input("Ticker", value="").upper().strip()
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

fmp_api_key = FMP_API_KEY
finnhub_api_key = FINNHUB_API_KEY

run = st.sidebar.button("Run Analysis", use_container_width=True)

# =========================
# MAIN UI
# =========================
st.title("📈 Stock Trading Dashboard Pro")
st.caption("Hybrid FMP + Finnhub + Yahoo Backup engine with technicals, valuation, entry zones, and options ideas.")

tab_overview, tab_technical, tab_valuation, tab_options, tab_scanner = st.tabs(
    ["Overview", "Technical", "Valuation", "Options", "Scanner"]
)

result = None
if run:
    if not ticker:
        st.warning("Please enter a ticker.")
    else:
        result = load_analysis(ticker, period, fmp_api_key, finnhub_api_key)
        if result is None:
            st.error(f"No data found for {ticker}.")

with tab_overview:
    if result is None:
        st.info("Enter a ticker in the sidebar and click Run Analysis.")
    else:
        st.markdown(f"### {result['company_name']}")
        st.write(f"**Next Earnings Date:** {result['earnings_date']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", fmt_num(result["latest_close"]))
        c2.metric("Trailing P/E", fmt_num(result["trailing_pe"]))
        c3.metric("Forward P/E", fmt_num(result["forward_pe"]))
        c4.metric("EV / EBITDA", fmt_num(result["ev_to_ebitda"]))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PEG", fmt_num(result["peg"]))
        c6.metric("Rule of 40", fmt_num(result["rule_of_40"]))
        c7.metric("Trend", result["trend_state"])
        c8.metric("Timing", f'{result["timing_score"]} ({result["timing_label"]})')

        st.subheader("Decision Panel")
        d1, d2 = st.columns(2)
        with d1:
            st.write(f"**Data Source:** {result['source_used']}")
            st.write(f"**Valuation Verdict:** {result['valuation']}")
            st.write(f"**Smart Valuation Style:** {result['smart_view']['valuation_style']}")
            st.write(f"**Growth-Adjusted View:** {result['smart_view']['growth_adjusted_view']}")
        with d2:
            st.write(f"**Setup Verdict:** {result['setup_verdict']}")
            st.write(f"**Trade Decision:** {result['trade_view']}")
            st.write(f"**Options Idea:** {result['options_view']}")

        st.subheader("Entry Zones")
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("Support 1", fmt_num(result["zones"]["support_1"]))
        z2.metric("Support 2", fmt_num(result["zones"]["support_2"]))
        z3.metric("Resistance 1", fmt_num(result["zones"]["resistance_1"]))
        z4.metric("Resistance 2", fmt_num(result["zones"]["resistance_2"]))
        st.write(f"**Buy Zone:** {fmt_num(result['zones']['buy_zone_low'])} - {fmt_num(result['zones']['buy_zone_high'])}")

with tab_technical:
    if result is None:
        st.info("Run Analysis to view charts.")
    else:
        df = result["data"]

        st.subheader("Price and EMAs")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df.index, df["Close"], label=f"{ticker} Close")
        ax1.plot(df.index, df["EMA50"], label="EMA 50")
        ax1.plot(df.index, df["EMA200"], label="EMA 200")
        ax1.axhspan(result["zones"]["buy_zone_low"], result["zones"]["buy_zone_high"], alpha=0.12)
        ax1.set_title(
            f"{ticker} | P/E: {fmt_num(result['trailing_pe'])} | "
            f"Fwd P/E: {fmt_num(result['forward_pe'])} | "
            f"EV/EBITDA: {fmt_num(result['ev_to_ebitda'])} | "
            f"PEG: {fmt_num(result['peg'])}"
        )
        ax1.set_ylabel("Price ($)")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("RSI")
        fig2, ax2 = plt.subplots(figsize=(12, 3.5))
        ax2.plot(df.index, df["RSI"], label="RSI")
        ax2.axhline(70, linestyle="--", alpha=0.6)
        ax2.axhline(30, linestyle="--", alpha=0.6)
        ax2.set_ylabel("RSI")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("MACD")
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(df.index, df["MACD"], label="MACD")
        ax3.plot(df.index, df["Signal_Line"], label="Signal Line")
        ax3.bar(df.index, df["Impulse_MACD"], label="Impulse MACD", alpha=0.5)
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)

with tab_valuation:
    if result is None:
        st.info("Run Analysis to view valuation.")
    else:
        st.subheader("Fundamentals Table")
        st.dataframe(result["fundamentals"], use_container_width=True, hide_index=True)

        st.subheader("EV / EBITDA Relative View")
        st.write(f"**Status:** {result['ev_rel']['status']}")
        st.write(f"**Comparison:** {result['ev_rel']['comparison']}")

with tab_options:
    if result is None:
        st.info("Run Analysis to view options ideas.")
    else:
        st.subheader("Options Optimizer")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Sell Put", str(result["opt"].get("spread_sell", "N/A")))
        o2.metric("Buy Put", str(result["opt"].get("spread_buy", "N/A")))
        o3.metric("Width", str(result["opt"].get("spread_width", "N/A")))
        o4.metric("DTE", result["opt"].get("dte", "N/A"))

        st.write(f"**Spread Idea:** {result['opt'].get('idea', 'N/A')}")
        st.write(f"**ITM LEAPS Reference Strike:** {result['opt'].get('leaps_strike', 'N/A')}")

        st.subheader("Implied Volatility")
        iv1, iv2, iv3 = st.columns(3)
        iv1.metric("ATM IV", "N/A" if result["atm_iv"] is None else f"{result['atm_iv'] * 100:.1f}%")
        iv2.metric("IV Rank (Exp.)", "N/A" if result["iv_percentile_approx"] is None else f"{result['iv_percentile_approx']:.0f}")
        iv3.metric("IV Regime", result["iv_regime"])

        st.caption(result["iv_note"])

        st.subheader("IV-Based Interpretation")
        atm_iv = result["atm_iv"]
        iv_pct = result["iv_percentile_approx"]

        if atm_iv is None:
            st.write("**IV Takeaway:** IV data unavailable for this ticker from Yahoo.")
        else:
            if iv_pct is not None:
                if iv_pct >= 70:
                    st.write("**IV Takeaway:** Volatility is relatively elevated. Better environment for premium selling strategies.")
                elif iv_pct <= 30:
                    st.write("**IV Takeaway:** Volatility is relatively low. Better environment for directional long options like ITM LEAPS.")
                else:
                    st.write("**IV Takeaway:** Volatility is in a middle range. Use balanced strategy selection.")
            else:
                if atm_iv >= 0.45:
                    st.write("**IV Takeaway:** High IV environment. Bull put spreads may be more attractive than outright call buying.")
                elif atm_iv <= 0.25:
                    st.write("**IV Takeaway:** Lower IV environment. Long calls / LEAPS are relatively more attractive.")
                else:
                    st.write("**IV Takeaway:** Moderate IV environment. Either defined-risk spreads or LEAPS can work depending on setup.")

with tab_scanner:
    st.subheader("Scanner")
    st.caption("Ranks names by trend, timing, and options attractiveness using IV and setup quality.")

    universe = st.selectbox(
        "Choose universe",
        ["Watchlist", "Dow 30", "NASDAQ-100", "Russell Filtered"],
        key="scanner_universe"
    )

    if universe == "Watchlist":
        scan_tickers = watchlist
    elif universe == "Dow 30":
        scan_tickers = [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","JPM","V","UNH",
            "XOM","PG","MA","HD","CVX","MRK","ABBV","KO","PEP","COST",
            "WMT","AVGO","ADBE","CRM","BAC","NFLX","AMD","ORCL","CSCO","MCD"
        ]
    elif universe == "NASDAQ-100":
        scan_tickers = [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","ADBE",
            "NFLX","AMD","INTC","QCOM","AMAT","CSCO","TXN","INTU","ISRG","BKNG",
            "MU","ORCL","ADP","ADI","LRCX","PANW","KLAC","SNPS","CDNS","MAR"
        ]
    else:
        scan_tickers = [
            "CELH","IOT","FROG","ONTO","QLYS","SAIA","BMI","FN","ALGM","SIMO",
            "CVCO","LTH","PAYO","PAGS","TMDX","INSP","BRZE","RXST","ACLS","UFPT"
        ]

    st.caption(f"{len(scan_tickers)} tickers selected for {universe}")

    run_universe_scan = st.button("Run Universe Scan", key="run_universe_scan")

    if run_universe_scan:
        universe_df = scan_watchlist(scan_tickers, period, fmp_api_key, finnhub_api_key)

        if universe_df is None or universe_df.empty:
            st.warning("No scan results returned.")
        else:
            preferred_cols = [
                "Ticker", "Last Close", "Trend", "Timing", "ATM IV", "IV %ile",
                "IV Regime", "Setup Label", "Options Score", "Valuation Style",
                "Trade Idea", "Fwd P/E", "PEG", "Rule of 40", "Source"
            ]
            existing_cols = [c for c in preferred_cols if c in universe_df.columns]
            st.dataframe(universe_df[existing_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Choose a universe and click Run Universe Scan.")
