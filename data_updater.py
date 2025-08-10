
import pandas as pd
import yfinance as yf
import requests, math, json
from pathlib import Path
import datetime as dt

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_tw_tickers():
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
    ]
    frames = []
    for u in urls:
        try:
            html = requests.get(u, timeout=30).text
            tables = pd.read_html(html)
            if tables:
                frames.append(tables[0])
        except Exception as e:
            print(f"[WARN] 代碼來源失敗：{u}: {e}")
    if not frames:
        return pd.DataFrame(columns=["code","name","market","yahoo"])
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    code_name_col = [c for c in df.columns if "有價證券代號" in str(c)]
    market_col = [c for c in df.columns if "市場別" in str(c)]
    if not code_name_col or not market_col:
        return pd.DataFrame(columns=["code","name","market","yahoo"])
    code_name_col = code_name_col[0]
    market_col = market_col[0]
    parsed = df[code_name_col].astype(str).str.extract(r"^(\d{4,6})\s+(.+)$")
    parsed.columns = ["code","name"]
    out = pd.concat([parsed, df[market_col].rename("market")], axis=1).dropna(subset=["code"])
    out = out[out["code"].str.match(r"^\d{4}$")]
    def suffix(m):
        m = str(m)
        if "上櫃" in m:
            return ".TWO"
        return ".TW"
    out["yahoo"] = out["code"] + out["market"].apply(suffix)
    return out[["code","name","market","yahoo"]].drop_duplicates()

def pull_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info, fast = {}, {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        fast = t.fast_info or {}
    except Exception:
        fast = {}
    return {
        "Ticker": ticker,
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap") or fast.get("market_cap"),
        "currency": info.get("currency") or fast.get("currency"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "returnOnEquity": info.get("returnOnEquity"),
        "grossMargins": info.get("grossMargins"),
        "operatingMargins": info.get("operatingMargins"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
        "lastPrice": fast.get("last_price") or info.get("currentPrice"),
    }

def fetch_history(ticker: str, days: int = 400) -> pd.DataFrame:
    df = yf.download(
        ticker, period=f"{int(days)}d", interval="1d", auto_adjust=True, progress=False, threads=False, group_by="ticker"
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=0, drop_level=False)
            df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
        except Exception:
            df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("Date").reset_index()
    return df.dropna(subset=["Close"])

def compute_snr(df: pd.DataFrame, window_days: int = 120):
    if df.empty or "Close" not in df.columns:
        return None
    close = df["Close"]
    roll = close.rolling(window_days, min_periods=max(20, window_days//4))
    out = pd.DataFrame({
        "Date": df["Date"],
        "Close": df["Close"],
        "S": roll.quantile(0.20),
        "N": roll.quantile(0.50),
        "R": roll.quantile(0.80),
    })
    return out

def suggest(last_close, S, R, near=0.03):
    import pandas as pd, math
    if any(pd.isna(x) for x in (last_close, S, R)):
        return "資料不足"
    dS = (last_close - S) / S if S else math.inf
    dR = (R - last_close) / R if R else math.inf
    if dS <= near: return "接近支撐：偏多、可分批佈局"
    if dR <= near: return "接近壓力：保守、等待回檔"
    if last_close < (S + R) / 2: return "區間下半：偏多但勿追高"
    return "區間上半：觀望或逢高減碼"

def main():
    today = dt.date.today().isoformat()
    tickers_df = fetch_tw_tickers()
    tickers = tickers_df["yahoo"].tolist()
    print(f"[INFO] 全市場代碼數：{len(tickers)}")

    # Fundamentals
    rows = []
    for i, t in enumerate(tickers, 1):
        rows.append(pull_fundamentals(t))
        if i % 50 == 0: print(f"[INFO] 基本面 {i}/{len(tickers)}")
    fund_df = pd.DataFrame(rows)
    fund_df["asOfDate"] = today
    fund_df.to_parquet(DATA_DIR / "fundamentals.parquet", index=False)
    fund_df.to_csv(DATA_DIR / f"fundamentals_{today}.csv", index=False, encoding="utf-8-sig")

    # SNR summary (僅存最後一日的 S/N/R)
    snr_rows = []
    for i, t in enumerate(tickers, 1):
        hist = fetch_history(t, 400)
        if hist.empty: continue
        snr = compute_snr(hist, 120)
        if snr is None or snr.dropna().empty: continue
        last = snr.dropna(subset=["Close"]).iloc[-1]
        snr_rows.append({
            "Ticker": t,
            "LastDate": pd.to_datetime(last["Date"]).date(),
            "Close": float(last["Close"]),
            "S": float(last["S"]) if pd.notna(last["S"]) else None,
            "N": float(last["N"]) if pd.notna(last["N"]) else None,
            "R": float(last["R"]) if pd.notna(last["R"]) else None,
            "Suggestion": suggest(last["Close"], last["S"], last["R"], 0.03)
        })
        if i % 50 == 0: print(f"[INFO] SNR {i}/{len(tickers)}")
    snr_df = pd.DataFrame(snr_rows)
    snr_df["asOfDate"] = today
    snr_df.to_parquet(DATA_DIR / "snr_summary.parquet", index=False)
    snr_df.to_csv(DATA_DIR / f"snr_summary_{today}.csv", index=False, encoding="utf-8-sig")

    print("[OK] 更新完成")

if __name__ == "__main__":
    main()
