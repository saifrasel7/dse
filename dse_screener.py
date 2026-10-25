#!/usr/bin/env python3
"""
DSE swing screener  -  fetches REAL daily prices, ranks the whole market, writes dse_data.js

Run:
    python dse_screener.py                  # downloads history straight from dse.com.bd
    python dse_screener.py --csv archive.csv  # OR use a Day End Archive CSV you downloaded from dsebd.org
    python dse_screener.py --full             # ignore saved history, download everything again
    python dse_screener.py --top 20 --days 400

Then open dse-screener.html (keep it in the SAME folder as the generated dse_data.js).

What it does
  1. Gets daily Open/High/Low/Close/Volume for every DSE stock.
  2. Drops illiquid / suspended / dead stocks.
  3. Scores each stock 0-100 for a ~2-week swing setup (trend, momentum, volume, breakout structure).
  4. Backtests the very same rule on the history it downloaded, so you can see how often it worked.
  5. Saves the top N + their candles to dse_data.js for the web page.

This is a ranking tool. Nothing here can promise a stock will rise.
"""
import argparse, datetime as dt, json, os, sys, time
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings (edit freely)
TOP_N = 20
HISTORY_DAYS = 800            # calendar days of history to pull (need 200+ trading days for MA200)
MIN_TURNOVER_BDT = 5_000_000  # median daily traded value over last 20 sessions (liquidity filter)
MIN_TRADED_DAYS_20 = 15       # must have traded on at least this many of the last 20 sessions
MAX_STALE_DAYS = 5            # skip stocks whose last candle is older than this vs the market's latest
FORWARD_DAYS = 10             # "2 weeks" = 10 trading sessions (DSE trades Sun-Thu)
BACKTEST_MIN_SCORE = 65
ROUND_TRIP_COST_PCT = 0.5     # rough brokerage + tax drag used in the "after costs" backtest figure
OUT_FILE = "dse_data.js"
W_TECH, W_FUND = 0.6, 0.4      # final score = 60% chart/technical + 40% fundamentals (2-week horizon leans on the chart)
MIN_TECH_FOR_PICK = 40         # a cheap stock with a broken chart is not a 2-week pick
TRACK_FILE = "track_record.json"   # every published top-20 list is saved here and checked again later
HZ_SESSIONS = {"short": 10, "medium": 60, "long": 250}   # trading sessions to hold for each horizon
CACHE_FILE = "dse_history.csv"  # saved history; later runs only download the new days
CACHE_KEEP_DAYS = 850
OVERLAP_DAYS = 5               # re-download the last few days each time to pick up any corrections
# ------------------------------------------------------------------


# ============================== data loading ==============================
def clean_bars(df):
    """df needs columns open, high, low, close, volume and a date index/column. Returns ascending clean frame."""
    df = df.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df["open"] = df["open"].where(df["open"] > 0, df["close"])
    df["high"] = df[["high", "open", "close"]].max(axis=1).where(df["high"] > 0, df[["open", "close"]].max(axis=1))
    df["low"] = df["low"].where(df["low"] > 0, df[["open", "close"]].min(axis=1))
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0)
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


DSE_API = "https://dse.com.bd/api/live/data-archive/day-end"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://dse.com.bd/",
}


class Blocked(Exception):
    pass


def _get_json(session, params, tries=4):
    import requests
    last = None
    for t in range(tries):
        try:
            r = session.get(DSE_API, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = r.status_code
            if r.status_code in (401, 403):
                time.sleep(1 + t)
                continue
        except requests.RequestException as e:
            last = str(e)
        time.sleep(1 + t)
    raise Blocked(f"HTTP {last}")


def _fetch_span(session, lo, hi):
    """All day-end rows (every instrument) between two ISO dates; splits the range if the API truncates it."""
    first = _get_json(session, {"from": lo, "to": hi})
    rows = list(first.get("rows") or [])
    size = first.get("pageSize") or len(rows) or 1
    total = first.get("total") or len(rows)
    if first.get("truncated") and lo < hi:
        d0, d1 = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
        mid = d0 + (d1 - d0) // 2
        return _fetch_span(session, lo, mid.isoformat()) + _fetch_span(session, (mid + dt.timedelta(days=1)).isoformat(), hi)
    pages = -(-total // size)
    for pg in range(2, pages + 1):
        rows += (_get_json(session, {"from": lo, "to": hi, "page": pg}).get("rows") or [])
        time.sleep(0.15)
    return rows


def load_from_dse_api(days):
    try:
        import requests
    except ImportError:
        sys.exit("Run first:  py -m pip install requests pandas numpy")
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    try:
        s.get("https://dse.com.bd/", timeout=30)   # pick up any cookies like a normal visitor
    except Exception:
        pass
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    spans, cur = [], start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=9), end)
        spans.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + dt.timedelta(days=1)
    print(f"Downloading DSE day-end history {start} -> {end} in {len(spans)} chunks (a few minutes) ...")
    rows = []
    try:
        for i, (lo, hi) in enumerate(spans, 1):
            rows += _fetch_span(s, lo, hi)
            if i % 5 == 0 or i == len(spans):
                print(f"  chunk {i}/{len(spans)} done  ({len(rows):,} rows so far)")
    except Blocked as e:
        sys.exit(
            f"\nDSE refused the download ({e}). The site is blocking scripts.\n"
            "Use the browser route instead (it takes 2 minutes):\n"
            "  1. Open https://dse.com.bd in Chrome\n"
            "  2. Press F12 > Console, paste the contents of dse_browser_export.js, press Enter\n"
            "  3. Wait for dse_archive.csv to download, put it in this folder\n"
            "  4. Run:  py dse_screener.py --csv dse_archive.csv\n")
    if not rows:
        sys.exit("DSE returned no rows for that date range.")
    raw = pd.DataFrame(rows)
    ren = {"tradingCode": "code", "openp": "open", "closep": "close"}
    raw = raw.rename(columns=ren)
    if "close" not in raw.columns:
        sys.exit(f"Unexpected data format from DSE. Columns received: {list(raw.columns)}")
    for alt, col in (("ltp", "close"), ("ycp", "close")):
        if alt in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw[col] = raw[col].where(raw[col] > 0, pd.to_numeric(raw[alt], errors="coerce"))
    out = {}
    for code, g in raw.groupby(raw["code"].astype(str).str.upper()):
        g = clean_bars(g[["date", "open", "high", "low", "close", "volume"]])
        if len(g):
            out[code] = g
    print(f"Got history for {len(out)} instruments.")
    return out


def load_from_csv(path):
    raw = pd.read_csv(path, sep=None, engine="python")
    norm = {c: "".join(ch for ch in str(c).lower() if ch.isalnum()) for c in raw.columns}
    raw = raw.rename(columns=norm)

    def pick(*names):
        for n in names:
            if n in raw.columns:
                return n
        return None

    cd, cc = pick("date"), pick("tradingcode", "code", "symbol", "ticker")
    co, ch, cl = pick("openp", "open"), pick("high"), pick("low")
    cclose, cy, cltp, cv = pick("closep", "close", "adjclose"), pick("ycp"), pick("ltp"), pick("volume", "vol")
    if not (cd and cc and ch and cl and (cclose or cltp)):
        sys.exit(f"CSV needs Date, Trading Code, High, Low, Close columns. Found: {list(raw.columns)}")
    d = pd.DataFrame({"date": raw[cd], "code": raw[cc].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)})
    num = lambda n: pd.to_numeric(raw[n].astype(str).str.replace(",", ""), errors="coerce") if n else np.nan
    d["close"] = num(cclose)
    if cltp:
        d["close"] = d["close"].where(d["close"] > 0, num(cltp))
    if cy:
        d["close"] = d["close"].where(d["close"] > 0, num(cy))
    d["open"] = num(co)
    if cy:
        d["open"] = d["open"].where(d["open"] > 0, num(cy))
    d["high"], d["low"], d["volume"] = num(ch), num(cl), (num(cv) if cv else 0)
    out = {}
    for code, g in d.groupby("code"):
        g = clean_bars(g.drop(columns=["code"]))
        if len(g):
            out[code] = g
    print(f"Read {len(out)} stocks from {path}")
    return out


# ============================== local history cache ==============================
def load_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        raw = pd.read_csv(path)
        out = {}
        for code, g in raw.groupby("code"):
            out[str(code)] = clean_bars(g[["date", "open", "high", "low", "close", "volume"]])
        return out
    except Exception as e:
        print(f"(Could not read {path}: {e} - starting fresh)")
        return {}


def save_cache(data, path):
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=CACHE_KEEP_DAYS)
    parts = []
    for code, df in data.items():
        d = df[df["date"] >= cutoff].copy()
        d.insert(0, "code", code)
        parts.append(d)
    if parts:
        pd.concat(parts).to_csv(path, index=False, date_format="%Y-%m-%d")


def merge_history(old, new):
    out = dict(old)
    for code, df in new.items():
        if code in out:
            df = pd.concat([out[code], df]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        out[code] = df
    return out


# ============================== fundamentals (TradingView screener, whole DSE in a few calls) ==============================
TV_URL = "https://scanner.tradingview.com/bangladesh/scan"
TV_FIELDS = {   # our name -> TradingView column names to try, in order
    "name":      ["description"],
    "sector":    ["sector"],
    "pe":        ["price_earnings_ttm"],
    "epsG":      ["earnings_per_share_diluted_yoy_growth_ttm", "earnings_per_share_basic_yoy_growth_ttm"],
    "roe":       ["return_on_equity", "return_on_equity_fq"],
    "div":       ["dividends_yield", "dividend_yield_recent", "dividends_yield_current"],
    "de":        ["debt_to_equity", "debt_to_equity_fq"],
    "netMargin": ["net_margin", "net_margin_ttm"],
    "mcap":      ["market_cap_basic"],
}


def fetch_fundamentals():
    """{code: {pe, epsG, roe, div, de, netMargin, mcap, sector, name}} for every DSE stock, or {} if TradingView is unreachable."""
    try:
        import requests
    except ImportError:
        return {}
    h = dict(BROWSER_HEADERS)
    h.update({"Content-Type": "application/json", "Origin": "https://www.tradingview.com", "Referer": "https://www.tradingview.com/"})

    def q(col):
        body = {"filter": [], "options": {"lang": "en"}, "markets": ["bangladesh"],
                "symbols": {"query": {"types": []}, "tickers": []}, "columns": ["name", col],
                "sort": {"sortBy": "name", "sortOrder": "asc"}, "range": [0, 1500]}
        r = requests.post(TV_URL, json=body, headers=h, timeout=30)
        if r.status_code != 200:
            return None
        return {row["s"].split(":")[-1].upper(): row["d"][1] for row in r.json().get("data", [])}

    out, got, missed = {}, [], []
    for field, cands in TV_FIELDS.items():
        res = None
        for c in cands:
            try:
                res = q(c)
            except Exception:
                res = None
            if res:
                break
        if not res:
            missed.append(field)
            continue
        got.append(field)
        for code, v in res.items():
            out.setdefault(code, {})[field] = v
    if not out:
        print("Fundamentals: TradingView not reachable from this computer - ranking by chart only.")
        return {}
    print(f"Fundamentals loaded for {len(out)} stocks (fields: {', '.join(got)}" + (f"; unavailable: {', '.join(missed)}" if missed else "") + ").")
    return out


def fund_score(f):
    """0-100 quality/value score + short reasons. Missing data earns partial credit, never a free pass."""
    pts, why = 0.0, []
    pe, g, roe, dv, de = f.get("pe"), f.get("epsG"), f.get("roe"), f.get("div"), f.get("de")
    fin = "financ" in str(f.get("sector") or "").lower()
    # valuation (30)
    if pe is None:
        pts += 10
    elif pe <= 0:
        why.append("loss-making")
    else:
        pts += 30 if pe <= 8 else 24 if pe <= 12 else 17 if pe <= 16 else 9 if pe <= 22 else 3
        if pe <= 12:
            why.append(f"P/E {pe:.1f}")
    # earnings growth (30)
    if g is None:
        pts += 10
    elif g > 150:
        pts += 15                                  # huge % growth is usually a tiny base year, not momentum
        why.append(f"EPS {g:+.0f}% (low base?)")
    else:
        pts += 30 if g >= 40 else 24 if g >= 20 else 17 if g >= 10 else 9 if g >= 0 else 0
        if g >= 15:
            why.append(f"EPS {g:+.0f}%")
        elif g < 0:
            why.append(f"EPS {g:+.0f}%")
    # quality (25)
    if roe is None:
        pts += 8
    else:
        pts += 25 if roe >= 20 else 20 if roe >= 15 else 13 if roe >= 10 else 6 if roe >= 5 else 0
        if roe >= 15:
            why.append(f"ROE {roe:.0f}%")
    # income (15)
    if dv is None:
        pts += 4
    else:
        pts += 15 if dv >= 6 else 11 if dv >= 4 else 7 if dv >= 2 else 4 if dv > 0 else 0
        if dv >= 4:
            why.append(f"yield {dv:.1f}%")
    # leverage penalty (not meaningful for banks/insurers: their "debt" includes deposits)
    if not fin and de is not None:
        if de > 200:
            pts -= 10; why.append("high debt")
        elif de > 100:
            pts -= 5
    return max(0.0, min(100.0, pts)), why


# ============================== indicators ==============================
def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100).where(au.notna())


def features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f["close"] = c
    f["ma20"], f["ma50"], f["ma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["rsi"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    f["vol5"], f["vol20"] = v.rolling(5).mean(), v.rolling(20).mean()
    f["vol_ratio"] = f["vol5"] / f["vol20"].replace(0, np.nan)
    f["hi20_prev"] = h.rolling(20).max().shift(1)
    f["hi60"] = h.rolling(60).max()
    f["ret5"] = c.pct_change(5)
    f["ext20"] = c / f["ma20"] - 1
    f["turnover20"] = (c * v).rolling(20).median()
    f["traded20"] = (v > 0).rolling(20).sum()
    f["brk_vol"] = v / v.rolling(20).mean().shift(1).replace(0, np.nan)
    sd = c.rolling(20).std()
    f["bb_up"], f["bb_lo"] = f["ma20"] + 2 * sd, f["ma20"] - 2 * sd
    f["bb_pctb"] = (c - f["bb_lo"]) / (f["bb_up"] - f["bb_lo"]).replace(0, np.nan)
    f["bb_w"] = (f["bb_up"] - f["bb_lo"]) / f["ma20"]
    f["bb_w_rank"] = f["bb_w"].rolling(120, min_periods=60).rank(pct=True)
    f["ret60"], f["ret250"] = c.pct_change(60), c.pct_change(250)
    f["hi250"] = h.rolling(250, min_periods=120).max()
    f["ma50_up"] = f["ma50"] > f["ma50"].shift(10)
    f["ma200_up"] = f["ma200"] > f["ma200"].shift(20)
    f["vol_ann"] = c.pct_change().rolling(250, min_periods=120).std() * np.sqrt(250)
    return f


def score_frame(f):
    """Vectorised 0-100 chart score.
    Re-weighted after testing on real DSE history (--calibrate): strength and nearness to highs kept outperforming,
    while volume surges, the 'healthy RSI' zone and overextension penalties added nothing, so they carry no weight."""
    c = f["close"]
    trend = (8 * (c > f["ma20"]) + 8 * (c > f["ma50"]) + 6 * (f["ma20"] > f["ma50"]) + 8 * (f["ma50"] > f["ma50"].shift(10))).astype(float)
    r = f["rsi"]
    mom = (22 * (r >= 55) + 9 * ((r >= 45) & (r < 55)) + 8 * (f["macd_hist"] > 0)).astype(float)
    near20 = c >= f["hi20_prev"] * 0.97          # at, near or through the prior 20-day high
    near60 = c >= f["hi60"] * 0.97
    struct = (28 * near20 + 12 * near60).astype(float)
    total = (trend + mom + struct).clip(0, 100)
    valid = f[["ma50", "rsi", "vol20", "hi20_prev"]].notna().all(axis=1)
    total = total.where(valid)
    zero = total * 0
    return total, {"trend": trend, "momentum": mom, "volume": zero, "structure": struct, "penalty": zero,
                   "breakout": c > f["hi20_prev"], "near_high": near20 & ~(c > f["hi20_prev"])}


def calibrate(scored, out_path):
    """Daily top-20 test: pick the top 20 by chart score each day, see what they did over the next FORWARD_DAYS sessions."""
    rows = []
    for code, (df, f, total) in scored.items():
        fwd = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
        m = total.notna() & fwd.notna() & (f["turnover20"] >= MIN_TURNOVER_BDT) & (f["traded20"] >= MIN_TRADED_DAYS_20)
        rows.append(pd.DataFrame({"date": df["date"][m], "score": total[m], "fwd": fwd[m]}))
    a = pd.concat(rows)
    a["ex"] = a["fwd"] - a.groupby("date")["fwd"].transform("mean")
    out = [f"Liquid stock-days: {len(a)}, {(a['fwd'] > 0).mean() * 100:.1f}% ended higher after {FORWARD_DAYS} sessions", "",
           "DAILY TOP-20 TEST (top 20 by chart score each day)"]
    d = [(k, g.nlargest(20, "score")["fwd"].mean() * 100, g.nlargest(20, "score")["ex"].mean() * 100) for k, g in a.groupby("date") if len(g) >= 40]
    dd = pd.DataFrame(d, columns=["date", "fwd", "ex"])
    h = len(dd) // 2
    for lbl, part in [("whole period", dd), ("first half", dd.iloc[:h]), ("second half", dd.iloc[h:])]:
        if len(part):
            out.append(f"  {lbl:<13} avg {part['fwd'].mean():+.2f}%, {part['ex'].mean():+.2f}% vs average stock, beat market on {(part['ex'] > 0).mean() * 100:.0f}% of days")
    out += ["", "SCORE BANDS (vs market, %up)"]
    for lo, hi in [(0, 40), (40, 60), (60, 75), (75, 90), (90, 101)]:
        g = a[(a["score"] >= lo) & (a["score"] < hi)]
        if len(g):
            out.append(f"  {lo:>3}-{hi - 1:<3} n={len(g):>6}  {g['ex'].mean() * 100:+.2f}%  up {(g['fwd'] > 0).mean() * 100:.1f}%")
    txt = "\n".join(out)
    print(txt)
    open(out_path, "w", encoding="utf-8").write(txt)


def _n(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)


def sector_pts(sinfo, n_sectors):
    if not sinfo or n_sectors < 3:
        return 7
    r = (sinfo["rank"] - 1) / max(n_sectors - 1, 1)
    return 15 if r <= 0.34 else 9 if r <= 0.67 else 3


def score_medium(fl, rs, sinfo, n_sec):
    """~3-month view: trend + relative strength vs the market + nearness to highs + momentum + sector strength (0-100)."""
    c = fl["close"]
    trend = 7 * (c > fl["ma50"]) + 6 * (fl["ma20"] > fl["ma50"]) + 6 * bool(fl["ma50_up"]) + (3 if _n(fl["ma200"]) else 6 * (c > fl["ma200"]))
    rel = 10 if rs is None else 25 if rs >= 0.10 else 20 if rs >= 0.05 else 15 if rs >= 0.02 else 10 if rs >= 0 else 5 if rs >= -0.03 else 0
    near = (12 if c >= fl["hi60"] * 0.95 else 6 if c >= fl["hi60"] * 0.90 else 0) + (8 if c >= fl["hi20_prev"] * 0.98 else 0)
    mom = (8 if 50 <= fl["rsi"] < 80 else 0) + (7 if fl["macd_hist"] > 0 else 0)
    sec = sector_pts(sinfo, n_sec)
    parts = [["Trend", float(trend), 25], ["Strength vs market", float(rel), 25], ["Near highs", float(near), 20], ["Momentum", float(mom), 15], ["Sector", float(sec), 15]]
    return float(min(100, sum(x[1] for x in parts))), parts


def score_long(fl, rs, beta, sinfo, n_sec):
    """1-year+ view: long-term trend, stability (low volatility/beta), drawdown from the 52-week high, sector strength (0-100)."""
    c = fl["close"]
    trend = (6 if _n(fl["ma200"]) else 12 * (c > fl["ma200"])) + (5 if _n(fl["ma200"]) else 10 * bool(fl["ma200_up"])) + (4 if _n(fl["ma200"]) else 8 * (fl["ma50"] > fl["ma200"]))
    va = fl["vol_ann"]
    stab = (6 if _n(va) else 15 if va < 0.30 else 9 if va < 0.45 else 3) + (6 if beta is None else 15 if beta <= 0.9 else 9 if beta <= 1.2 else 3)
    dd = (c / fl["hi250"] - 1) if not _n(fl["hi250"]) else None
    strength = (6 if dd is None else 15 if dd >= -0.15 else 9 if dd >= -0.30 else 2) + (4 if rs is None else 10 if rs > 0 else 5 if rs > -0.10 else 0)
    sec = sector_pts(sinfo, n_sec)
    parts = [["Long-term trend", float(trend), 30], ["Stability", float(stab), 30], ["Strength", float(strength), 25], ["Sector", float(sec), 15]]
    return float(min(100, sum(x[1] for x in parts))), parts


def fund_score_long(f):
    """Quality + value + income score for holding a year or more (0-100)."""
    pts, why = 0.0, []
    pe, g, roe, dv, de = f.get("pe"), f.get("epsG"), f.get("roe"), f.get("div"), f.get("de")
    fin = "financ" in str(f.get("sector") or "").lower()
    if pe is None: pts += 10
    elif pe <= 0: why.append("loss-making")
    else:
        pts += 30 if pe <= 10 else 22 if pe <= 15 else 14 if pe <= 20 else 7 if pe <= 28 else 2
        if pe <= 15: why.append(f"P/E {pe:.1f}")
    if roe is None: pts += 10
    else:
        pts += 30 if roe >= 20 else 24 if roe >= 15 else 16 if roe >= 10 else 8 if roe >= 5 else 0
        if roe >= 15: why.append(f"ROE {roe:.0f}%")
    if dv is None: pts += 5
    else:
        pts += 20 if dv >= 6 else 15 if dv >= 4 else 10 if dv >= 2 else 5 if dv > 0 else 0
        if dv >= 4: why.append(f"yield {dv:.1f}%")
    if g is None: pts += 5
    elif g > 150: pts += 8
    else:
        pts += 15 if g >= 15 else 11 if g >= 5 else 6 if g >= 0 else 0
        if g >= 10: why.append(f"EPS {g:+.0f}%")
    if not fin and de is not None:
        if de > 150: pts -= 10; why.append("high debt")
        elif de > 80: pts -= 5
    return max(0.0, min(100.0, pts)), why


def update_track(path, lists_today, date_str):
    try:
        tr = json.load(open(path, encoding="utf-8"))
    except Exception:
        tr = {"lists": []}
    have = {(l["date"], l["hz"]) for l in tr["lists"]}
    for hz, picks in lists_today.items():
        if picks and (date_str, hz) not in have:
            tr["lists"].append({"date": date_str, "hz": hz, "picks": [{"code": q["code"], "price": q["price"]} for q in picks]})
    tr["lists"] = tr["lists"][-400:]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tr, fh, separators=(",", ":"))
    return tr


def evaluate_track(tr, scored, midx):
    """For each saved list: buy every pick at the close of the list date, hold N sessions (or until today if it is too early)."""
    out = {}
    cost = ROUND_TRIP_COST_PCT
    for hz, N in HZ_SESSIONS.items():
        rows = []
        for l in sorted([x for x in tr["lists"] if x["hz"] == hz], key=lambda x: x["date"]):
            d0 = pd.Timestamp(l["date"])
            m0 = midx.asof(d0)
            rets, exc, els = [], [], []
            for pk in l["picks"]:
                if pk["code"] not in scored or not pk["price"]:
                    continue
                df = scored[pk["code"]][0]
                idx = int(df["date"].searchsorted(d0, side="right")) - 1
                if idx < 0:
                    continue
                el = len(df) - 1 - idx
                use = min(el, N)
                r = float(df["close"].iloc[idx + use]) / pk["price"] - 1
                m1 = midx.asof(df["date"].iloc[idx + use])
                rets.append(r); exc.append(r - (float(m1) / float(m0) - 1)); els.append(el)
            if rets and max(els) > 0:
                rows.append({"date": l["date"], "sessions": int(min(max(els), N)), "done": bool(min(els) >= N), "n": len(rets),
                             "avgPct": round(float(np.mean(rets)) * 100, 2), "excessPct": round(float(np.mean(exc)) * 100, 2),
                             "upPct": round(float(np.mean([r > 0 for r in rets])) * 100)})
        dn = [r for r in rows if r["done"]]
        summ = None
        if dn:
            summ = {"lists": len(dn), "picks": int(sum(r["n"] for r in dn)), "avgPct": round(float(np.mean([r["avgPct"] for r in dn])), 2),
                    "afterCostsPct": round(float(np.mean([r["avgPct"] for r in dn])) - cost, 2),
                    "excessPct": round(float(np.mean([r["excessPct"] for r in dn])), 2),
                    "upPct": round(float(np.mean([r["upPct"] for r in dn]))),
                    "beatMarketPct": round(float(np.mean([r["excessPct"] > 0 for r in dn])) * 100)}
        out[hz] = {"days": N, "summary": summ, "lists": rows[-40:], "saved": len([x for x in tr["lists"] if x["hz"] == hz])}
    return out


def explain(f_last, parts_last):
    out = []
    if f_last["close"] > f_last["ma20"] and f_last["close"] > f_last["ma50"]:
        out.append("above MA20 & MA50")
    if f_last["ma20"] > f_last["ma50"]:
        out.append("MA20 > MA50")
    if f_last["rsi"] >= 55:
        out.append(f"RSI {f_last['rsi']:.0f} strong")
    if f_last["macd_hist"] > 0:
        out.append("MACD positive")
    if parts_last["breakout"]:
        out.append("through 20-day high")
    elif parts_last["near_high"]:
        out.append("within 3% of 20-day high")
    if f_last["vol_ratio"] >= 1.5:
        out.append(f"volume {f_last['vol_ratio']:.1f}x avg")
    return ", ".join(out[:5]) or "mixed signals"


# ============================== backtest ==============================
def backtest(all_scored):
    rows = []
    for code, (df, f, total) in all_scored.items():
        fwd = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
        m = total.notna() & fwd.notna() & (f["turnover20"] >= MIN_TURNOVER_BDT) & (f["traded20"] >= MIN_TRADED_DAYS_20)
        rows.append(pd.DataFrame({"score": total[m], "fwd": fwd[m]}))
    if not rows:
        return None
    b = pd.concat(rows)
    sig, base = b[b["score"] >= BACKTEST_MIN_SCORE]["fwd"], b["fwd"]
    if len(sig) < 30:
        return None
    cost = ROUND_TRIP_COST_PCT / 100
    return {
        "signals": int(len(sig)), "minScore": BACKTEST_MIN_SCORE, "forwardDays": FORWARD_DAYS,
        "hitRatePct": round(float((sig > 0).mean() * 100), 1),
        "hitRateAfterCostsPct": round(float((sig - cost > 0).mean() * 100), 1),
        "avgReturnPct": round(float(sig.mean() * 100), 2),
        "medianReturnPct": round(float(sig.median() * 100), 2),
        "avgAfterCostsPct": round(float((sig - cost).mean() * 100), 2),
        "baselineHitRatePct": round(float((base > 0).mean() * 100), 1),
        "baselineAvgReturnPct": round(float(base.mean() * 100), 2),
    }


# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Day End Archive CSV from dsebd.org instead of downloading")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--days", type=int, default=HISTORY_DAYS)
    ap.add_argument("--out", default=OUT_FILE)
    ap.add_argument("--calibrate", action="store_true", help="test which signals really worked on DSE history, then exit")
    ap.add_argument("--full", action="store_true", help="ignore the saved history and download everything again")
    a = ap.parse_args()

    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CACHE_FILE)
    cache = {} if a.full else load_cache(cache_path)
    if cache and (pd.Timestamp.today() - min(df["date"].iloc[0] for df in cache.values())).days < HISTORY_DAYS - 60:
        print("Saved history is shorter than the 3-month/1-year views need - downloading the longer history once (takes longer) ...")
        cache = {}
    if a.csv:
        data = merge_history(cache, load_from_csv(a.csv))
    elif cache:
        last_saved = max(df["date"].iloc[-1] for df in cache.values())
        gap = (pd.Timestamp.today().normalize() - last_saved).days
        days = max(gap + OVERLAP_DAYS, 7)
        print(f"Saved history found (up to {last_saved.date()}). Downloading only the last {days} days ...")
        data = merge_history(cache, load_from_dse_api(days))
    else:
        data = load_from_dse_api(a.days)
    save_cache(data, cache_path)
    market_last = max(df["date"].iloc[-1] for df in data.values())
    print(f"Latest market session in data: {market_last.date()}")

    scored, ranked = {}, []
    adv = dec = flat = 0
    a50 = n50 = a200 = n200 = 0
    for code, df in data.items():
        if len(df) < 80:
            continue
        f = features(df)
        total, parts = score_frame(f)
        scored[code] = (df, f, total)
        last = len(df) - 1
        if (market_last - df["date"].iloc[-1]).days > MAX_STALE_DAYS:
            continue
        fl = f.iloc[last]
        if not (fl["turnover20"] >= MIN_TURNOVER_BDT and fl["traded20"] >= MIN_TRADED_DAYS_20):
            continue
        chg = df["close"].iloc[last] - df["close"].iloc[last - 1]
        adv, dec, flat = adv + (chg > 0), dec + (chg < 0), flat + (chg == 0)
        if not pd.isna(fl["ma50"]):
            n50 += 1; a50 += int(fl["close"] > fl["ma50"])
        if not pd.isna(fl["ma200"]):
            n200 += 1; a200 += int(fl["close"] > fl["ma200"])
        if pd.isna(total.iloc[last]):
            continue
        pl = {k: v.iloc[last] for k, v in parts.items()}
        ranked.append((code, float(total.iloc[last]), df, f, fl, pl))

    print(f"Scanned {len(scored)} stocks, {len(ranked)} passed the liquidity/activity filters.")
    if a.calibrate:
        calibrate(scored, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.txt"))
        return
    bt = backtest(scored)
    if bt:
        print(f"Backtest of the CHART score alone ({bt['signals']} signals >= {bt['minScore']}, {bt['forwardDays']}-session hold): "
              f"{bt['hitRatePct']}% ended higher (all days: {bt['baselineHitRatePct']}%), avg {bt['avgReturnPct']}% "
              f"(after ~{ROUND_TRIP_COST_PCT}% costs: {bt['avgAfterCostsPct']}%).")

    fund = fetch_fundamentals()

    # ---- market index (equal-weighted), beta and sector strength ----
    R = pd.DataFrame({code: df.set_index("date")["close"].pct_change() for code, (df, f, t) in scored.items()}).sort_index()
    mret = R.clip(-0.1, 0.1).mean(axis=1)
    midx = (1 + mret.fillna(0)).cumprod()
    mk60 = float(midx.iloc[-1] / midx.iloc[-61] - 1) if len(midx) > 61 else None
    mk250 = float(midx.iloc[-1] / midx.iloc[-251] - 1) if len(midx) > 251 else None

    def beta_of(code):
        sr = R[code].tail(250)
        m = mret.reindex(sr.index)
        ok = sr.notna() & m.notna()
        if ok.sum() < 120:
            return None
        sr, m = sr[ok], m[ok]
        v = float(m.var())
        return float(np.cov(sr, m)[0, 1] / v) if v else None

    sec_rows = {}
    for (code, sc, df, f, fl, pl) in ranked:
        sct = (fund.get(code) or {}).get("sector")
        if sct and not pd.isna(fl["ret60"]):
            sec_rows.setdefault(sct, []).append((float(fl["ret60"]), 0 if pd.isna(fl["ma50"]) else int(fl["close"] > fl["ma50"])))
    sectors = [{"sector": k, "n": len(v), "ret3mPct": round(float(np.median([x[0] for x in v])) * 100, 1),
                "aboveMa50Pct": round(100 * sum(x[1] for x in v) / len(v))} for k, v in sec_rows.items() if len(v) >= 3]
    sectors.sort(key=lambda z: z["ret3mPct"], reverse=True)
    for k, z in enumerate(sectors, 1):
        z["rank"] = k
    SEC = {z["sector"]: z for z in sectors}

    def rel(v, mk):
        return None if _n(v) or mk is None else float(v) - mk

    def fund_pick(code, fn):
        fd = fund.get(code)
        if not fund:
            return None, None, [], False
        if fd is None:
            return None, 35.0, ["no fundamentals data"], True
        if fd.get("pe") is not None and fd["pe"] <= 0:
            return fd, None, [], False
        fs, why = fn(fd)
        return fd, fs, why, True

    # ---- three horizons ----
    short, medium, long_ = [], [], []
    for (code, sc, df, f, fl, pl) in ranked:
        sinfo = SEC.get((fund.get(code) or {}).get("sector"))
        beta = beta_of(code)
        if sc >= MIN_TECH_FOR_PICK:
            fd, fs, fwhy, ok = fund_pick(code, fund_score)
            if ok:
                comb = W_TECH * sc + W_FUND * fs if fund else sc
                plist = [["Trend", float(pl["trend"]), 30], ["Momentum", float(pl["momentum"]), 30], ["Near highs", float(pl["structure"]), 40]]
                short.append((code, comb, sc, fs, fd, fwhy, df, f, fl, plist, explain(fl, pl), {k: float(pl[k]) for k in ["trend", "momentum", "volume", "structure", "penalty"]}, beta, sinfo))
        if not pd.isna(fl["ma200"]) or len(df) >= 120:
            ms, mparts = score_medium(fl, rel(fl["ret60"], mk60), sinfo, len(sectors))
            if ms >= 45:
                fd, fs, fwhy, ok = fund_pick(code, fund_score)
                if ok:
                    comb = 0.5 * ms + 0.5 * fs if fund else ms
                    why = []
                    r3 = rel(fl["ret60"], mk60)
                    if r3 is not None and r3 > 0: why.append(f"beating market by {r3 * 100:+.0f}% in 3 mo")
                    if fl["close"] > fl["ma50"] and fl["ma50_up"]: why.append("above rising MA50")
                    if sinfo and sinfo["rank"] <= 3: why.append(f"strong sector (#{sinfo['rank']})")
                    medium.append((code, comb, ms, fs, fd, fwhy, df, f, fl, mparts, ", ".join(why[:3]) or "mixed signals", {}, beta, sinfo))
            ls, lparts = score_long(fl, rel(fl["ret250"], mk250), beta, sinfo, len(sectors))
            if ls >= 40 and len(df) >= 150:
                fd, fs, fwhy, ok = fund_pick(code, fund_score_long)
                if ok and (fs is None or fs >= 45):
                    comb = 0.3 * ls + 0.7 * fs if fund else ls
                    why = []
                    if beta is not None and beta <= 0.9: why.append(f"beta {beta:.2f}")
                    if not pd.isna(fl["hi250"]) and fl["close"] >= fl["hi250"] * 0.85: why.append("near 52-wk high")
                    if not pd.isna(fl["ma200"]) and fl["close"] > fl["ma200"]: why.append("above MA200")
                    long_.append((code, comb, ls, fs, fd, fwhy, df, f, fl, lparts, ", ".join(why[:3]) or "mixed signals", {}, beta, sinfo))
    for lst in (short, medium, long_):
        lst.sort(key=lambda x: x[1], reverse=True)

    def rnd(v, d=2):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v) else round(float(v), d)

    used = set()

    def mk(picks):
        out = []
        for rank, (code, comb, sc, fs, fd, fwhy, df, f, fl, plist, reason, parts, beta, sinfo) in enumerate(picks[: a.top], 1):
            last, prev = df.iloc[-1], df.iloc[-2]
            if fwhy:
                reason += " | " + ", ".join(fwhy[:4])
            used.add(code)
            out.append({
                "rank": rank, "code": code, "name": (fd or {}).get("name") or code, "sector": (fd or {}).get("sector") or "DSE",
                "price": round(float(last["close"]), 2),
                "changePct": round(float((last["close"] / prev["close"] - 1) * 100), 2),
                "score": round(comb, 1), "techScore": round(sc, 1), "fundScore": None if fs is None else round(fs, 1),
                "rec": "Strong pick" if comb >= 72 else ("Good pick" if comb >= 60 else "Watch"),
                "reason": reason,
                "fund": None if not fd else {"pe": rnd(fd.get("pe")), "epsG": rnd(fd.get("epsG"), 1), "roe": rnd(fd.get("roe"), 1),
                                             "div": rnd(fd.get("div")), "de": rnd(fd.get("de"), 0), "netMargin": rnd(fd.get("netMargin"), 1),
                                             "mcap": rnd(fd.get("mcap"), 0)},
                "rsi": round(float(fl["rsi"]), 1),
                "ma50": rnd(fl["ma50"]), "ma200": rnd(fl["ma200"]),
                "volRatio": rnd(fl["vol_ratio"]),
                "ret5Pct": rnd(fl["ret5"] * 100, 1),
                "turnoverM": round(float(fl["turnover20"]) / 1e6, 1),
                "parts": {k: round(v, 0) for k, v in parts.items()},
                "partsList": [[n, round(v), m] for n, v, m in plist],
                "bb": {"pctB": rnd(fl["bb_pctb"], 2), "widthPct": rnd(fl["bb_w"] * 100, 1),
                       "squeeze": bool(fl["bb_w_rank"] <= 0.2) if not pd.isna(fl["bb_w_rank"]) else False,
                       "upper": rnd(fl["bb_up"]), "lower": rnd(fl["bb_lo"])},
                "beta": rnd(beta, 2), "ret3mPct": rnd(fl["ret60"] * 100, 1), "ret1yPct": rnd(fl["ret250"] * 100, 1),
                "fromHighPct": rnd((fl["close"] / fl["hi250"] - 1) * 100, 1), "volAnnPct": rnd(fl["vol_ann"] * 100, 0),
                "sectorInfo": sinfo,
            })
        return out

    h_short, h_med, h_long = mk(short), mk(medium), mk(long_)
    stocks = h_short
    bars = {}
    for code in used:
        df = scored[code][0]
        bars[code] = [[r.date.strftime("%Y-%m-%d"), round(r.open, 2), round(r.high, 2), round(r.low, 2), round(r.close, 2), int(r.volume)]
                      for r in df.tail(250).itertuples()]

    tr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACK_FILE)
    track = update_track(tr_path, {"short": h_short, "medium": h_med, "long": h_long}, market_last.strftime("%Y-%m-%d"))
    track_eval = evaluate_track(track, scored, midx)
    payload = {
        "track": track_eval,
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "latestDate": market_last.strftime("%Y-%m-%d"),
        "scanned": len(scored), "passed": len(ranked), "backtest": bt, "stocks": stocks, "hasFundamentals": bool(fund), "weights": {"tech": W_TECH, "fund": W_FUND},
        "horizons": {"short": h_short, "medium": h_med, "long": h_long}, "bars": bars, "sectors": sectors,
        "hzWeights": {"short": [0.6, 0.4], "medium": [0.5, 0.5], "long": [0.3, 0.7]},
        "marketRet": {"m3Pct": None if mk60 is None else round(mk60 * 100, 1), "y1Pct": None if mk250 is None else round(mk250 * 100, 1)},
        "market": {"advancers": int(adv), "decliners": int(dec), "unchanged": int(flat),
                   "aboveMa50Pct": round(100 * a50 / n50, 1) if n50 else None,
                   "aboveMa200Pct": round(100 * a200 / n200, 1) if n200 else None},
    }
    js = "window.DSE_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(js)
    # Self-contained copy for your phone: site/index.html has the data inside it (upload this folder to a web host)
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = os.path.join(here, "dse-screener.html")
    if os.path.exists(tpl):
        html = open(tpl, encoding="utf-8").read()
        tag = '<script src="dse_data.js" onerror="void 0"></script>'
        if tag in html:
            os.makedirs(os.path.join(here, "site"), exist_ok=True)
            with open(os.path.join(here, "site", "index.html"), "w", encoding="utf-8") as fh:
                fh.write(html.replace(tag, "<script>" + js + "</script>"))
            print("Phone version saved: site/index.html")
    print(f"\nTop {len(stocks)} saved to {os.path.abspath(a.out)}")
    for s in stocks:
        print(f"  {s['rank']:>2}. {s['code']:<12} {s['score']:>5} (chart {s['techScore']}, fund {s['fundScore']})  {s['rec']:<11} {s['reason']}")
    print("\nNow open dse-screener.html (same folder). Research tool only - not financial advice.")


if __name__ == "__main__":
    main()
