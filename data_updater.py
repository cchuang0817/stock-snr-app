# data_updater.py
import pandas as pd
import yfinance as yf
import requests, time, os, glob
from pathlib import Path
import datetime as dt

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

def log(msg): print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---------- 1) 台股代碼（上市/上櫃） ----------
def fetch_tw_tickers(retries=3, sleep_sec=3) -> pd.DataFrame:
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",  # 上市
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",  # 上櫃
    ]
    frames = []
    for u in urls:
        ok = False
        for _ in range(retries):
            try:
                html = requests.get(u, headers=HEADERS, timeout=30).text
                tables = pd.read_html(html)
                if tables:
                    frames.append(tables[0]); ok = True; break
            except Exception as e:
                log(f"WARN 代碼來源失敗一次：{u} ({e})"); time.sleep(sleep_sec)
        if not ok: log(f"WARN 放棄該來源：{u}")

    if not frames:
        return pd.DataFrame(columns=["code","name","market","yahoo"])

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)

    code_name_col = next((c for c in df.columns if "代號" in str(c)), None)
    market_col    = next((c for c in df.columns if "市場別" in str(c)), None)
    if not code_name_col or not market_col:
        log("ERROR 找不到必要欄位（代號/市場別）")
        return pd.DataFrame(columns=["code","name","market","yahoo"])

    parsed = df[code_name_col].astype(str).str.extract(r"^(\d{4,6})\s+(.+)$")
    parsed.columns = ["code","name"]
    out = pd.concat([parsed, df[market_col].rename("market")], axis=1).dropna(subset=["code"])
    out = out[out["code"].str.match(r"^\d{4}$")]  # 僅 4 位數個股

    def suffix(m): return ".TWO" if "上櫃" in str(m) else ".TW"
    out["yahoo"] = out["code"] + out["market"].apply(suffix)
    out = out[["code","name","market","yahoo"]].drop_duplicates()

    today = dt.date.today().isoformat()
    out.to_csv(DATA_DIR / f"tickers_{today}.csv", index=False, encoding="utf-8-sig")
    log(f"INFO 台股代碼數：{len(out)}")
    return out

# ---------- 2) yfinance 幫手 ----------
def pull_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info, fast = {}, {}
    try: info = t.info or {}
    except Exception: info = {}
    try: fast = t.fast_info or {}
    except Exception: fast = {}
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

def fetch_history(ticker: str, days: int = 420) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{int(days)}d", interval="1d",
                     auto_adjust=True, progress=False, threads=False, group_by="ticker")
    if df is None or df.empty: return pd.DataFrame()
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
    if df.empty or "Close" not in df.columns: return None
    roll = df["Close"].rolling(window_days, min_periods=max(20, window_days//4))
    out = pd.DataFrame({
        "Date": df["Date"],
        "Close": df["Close"],
        "S": roll.quantile(0.20),
        "N": roll.quantile(0.50),
        "R": roll.quantile(0.80),
    })
    return out

# ---------- 3) 主流程（自動用最近交易日；抓不到就沿用上一版） ----------
def main():
    tickers_df = fetch_tw_tickers()
    if tickers_df.empty or len(tickers_df) < 100:
        log("ERROR 代碼清單為空或過少；改用上一版資料做備援。")
        use_previous_as_fallback()
        return

    tickers = tickers_df["yahoo"].tolist()

    # ---- 基本面 ----
    fund_rows = []
    for i, t in enumerate(tickers, 1):
        try: fund_rows.append(pull_fundamentals(t))
        except Exception as e: log(f"WARN fundamentals {t} 失敗：{e}")
        if i % 60 == 0: time.sleep(1)
    fund_df = pd.DataFrame(fund_rows)
    if fund_df.empty:
        log("ERROR 基本面為空；改用上一版資料做備援。")
        use_previous_as_fallback()
        return

    # ---- SNR & 最近交易日 ----
    snr_rows, last_dates = [], []
    for i, t in enumerate(tickers, 1):
        try:
            hist = fetch_history(t, 420)
            if hist.empty: 
                continue
            last_dates.append(pd.to_datetime(hist["Date"].iloc[-1]).date())
            snr = compute_snr(hist, 120)
            if snr is None or snr.dropna().empty:
                continue
            last = snr.dropna(subset=["Close"]).iloc[-1]
            snr_rows.append({
                "Ticker": t,
                "LastDate": pd.to_datetime(last["Date"]).date(),
                "Close": float(last["Close"]),
                "S": float(last["S"]) if pd.notna(last["S"]) else None,
                "N": float(last["N"]) if pd.notna(last["N"]) else None,
                "R": float(last["R"]) if pd.notna(last["R"]) else None,
            })
        except Exception as e:
            log(f"WARN snr {t} 失敗：{e}")
        if i % 50 == 0: time.sleep(1)

    if not last_dates or not snr_rows:
        log("ERROR 今天抓不到任何 SNR/歷史日線；改用上一版資料做備援。")
        use_previous_as_fallback()
        return

    file_date = max(last_dates).isoformat()   # 最近交易日
    log(f"INFO 最近交易日：{file_date}")

    # 寫檔（覆蓋 parquet、另存對應日期 csv）
    fund_df["asOfDate"] = file_date
    fund_df.to_parquet(DATA_DIR / "fundamentals.parquet", index=False)
    fund_df.to_csv(DATA_DIR / f"fundamentals_{file_date}.csv", index=False, encoding="utf-8-sig")
    log(f"OK fundamentals：{len(fund_df)} 列")

    snr_df = pd.DataFrame(snr_rows)
    snr_df["asOfDate"] = file_date
    snr_df.to_parquet(DATA_DIR / "snr_summary.parquet", index=False)
    snr_df.to_csv(DATA_DIR / f"snr_summary_{file_date}.csv", index=False, encoding="utf-8-sig")
    log(f"OK snr_summary：{len(snr_df)} 列")

def use_previous_as_fallback():
    """抓不到新資料時：沿用 data 目錄裡最新的一組檔案，重新覆蓋 parquet，保證前端可用。"""
    prev_f = sorted(glob.glob("data/fundamentals_*.csv"))
    prev_s = sorted(glob.glob("data/snr_summary_*.csv"))
    if not prev_f or not prev_s:
        log("FATAL 沒有可用的舊檔可沿用，結束（前端會顯示即時模式）。")
        return
    fund_csv = prev_f[-1]; snr_csv = prev_s[-1]
    fund_df = pd.read_csv(fund_csv); snr_df = pd.read_csv(snr_csv)
    # 從檔名取日期
    try:
        file_date = os.path.basename(snr_csv).split("_")[-1].replace(".csv","")
    except Exception:
        file_date = dt.date.today().isoformat()
    fund_df["asOfDate"] = file_date
    snr_df["asOfDate"] = file_date
    fund_df.to_parquet(DATA_DIR / "fundamentals.parquet", index=False)
    snr_df.to_parquet(DATA_DIR / "snr_summary.parquet", index=False)
    log(f"OK 使用上一版檔案覆蓋 parquet：{file_date}（fund={len(fund_df)}、snr={len(snr_df)}）")

if __name__ == "__main__":
    main()
