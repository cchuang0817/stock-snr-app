import pandas as pd
import yfinance as yf
import requests, time, math, sys
from pathlib import Path
import datetime as dt

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

def fetch_tw_tickers(retries=3, sleep_sec=3) -> pd.DataFrame:
    """TWSE/TPEx ISIN：上市/上櫃個股清單，失敗會重試。"""
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
            except Exception:
                time.sleep(sleep_sec)
        if not ok:
            print(f"[WARN] 代碼載入失敗：{u}")
    if not frames:
        return pd.DataFrame(columns=["code","name","market","yahoo"])

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)

    code_name_col = next((c for c in df.columns if "代號" in str(c)), None)
    market_col    = next((c for c in df.columns if "市場別" in str(c)), None)
    if not code_name_col or not market_col:
        print("[ERROR] 找不到必要欄位（代號/市場別）")
        return pd.DataFrame(columns=["code","name","market","yahoo"])

    parsed = df[code_name_col].astype(str).str.extract(r"^(\d{4,6})\s+(.+)$")
    parsed.columns = ["code","name"]
    out = pd.concat([parsed, df[market_col].rename("market")], axis=1).dropna(subset=["code"])
    out = out[out["code"].str.match(r"^\d{4}$")]  # 僅 4 位數個股

    def suffix(m):
        return ".TWO" if "上櫃" in str(m) else ".TW"

    out["yahoo"] = out["code"] + out["market"].apply(suffix)
    out = out[["code","name","market","yahoo"]].drop_duplicates()

    # 存一份代碼清單方便除錯
    today = dt.date.today().isoformat()
    out.to_csv(DATA_DIR / f"tickers_{today}.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] 台股代碼數：{len(out)}")
    return out

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

def fetch_history(ticker: str, days: int = 400) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{int(days)}d", interval="1d",
                     auto_adjust=True, progress=False, threads=False, group_by="ticker")
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
    roll = df["Close"].rolling(window_days, min_periods=max(20, window_days//4))
    out = pd.DataFrame({
        "Date": df["Date"],
        "Close": df["Close"],
        "S": roll.quantile(0.20),
        "N": roll.quantile(0.50),
        "R": roll.quantile(0.80),
    })
    return out

def main():
    tickers_df = fetch_tw_tickers()
    if tickers_df.empty or len(tickers_df) < 100:
        print("[ERROR] 代碼清單為空或過少，終止。")
        sys.exit(1)
    tickers = tickers_df["yahoo"].tolist()

    # ------- 基本面 -------
    fund_rows = []
    for i, t in enumerate(tickers, 1):
        try:
            fund_rows.append(pull_fundamentals(t))
        except Exception as e:
            print(f"[WARN] fundamentals {t} 失敗：{e}")
        if i % 60 == 0: time.sleep(1)
    fund_df = pd.DataFrame(fund_rows)
    if fund_df.empty:
        print("[ERROR] 基本面為空，終止。")
        sys.exit(1)

    # ------- SNR（用來決定「最近交易日」）-------
    snr_rows = []
    last_dates = []
    for i, t in enumerate(tickers, 1):
        try:
            hist = fetch_history(t, 400)
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
            print(f"[WARN] snr {t} 失敗：{e}")
        if i % 50 == 0: time.sleep(1)

    if not last_dates:
        print("[ERROR] 無任何標的有歷史日線，終止。")
        sys.exit(1)

    # 以「所有成功抓到的標的」之中最大的日期 = 最近交易日
    file_date = max(last_dates).isoformat()
    print(f"[INFO] 最近交易日：{file_date}")

    # 寫檔（用最近交易日命名；同時覆蓋最新 parquet）
    fund_df["asOfDate"] = file_date
    fund_df.to_parquet(DATA_DIR / "fundamentals.parquet", index=False)
    fund_df.to_csv(DATA_DIR / f"fundamentals_{file_date}.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] fundamentals：{len(fund_df)} 列")

    snr_df = pd.DataFrame(snr_rows)
    if snr_df.empty:
        print("[ERROR] SNR 為空，終止。")
        sys.exit(1)
    snr_df["asOfDate"] = file_date
    snr_df.to_parquet(DATA_DIR / "snr_summary.parquet", index=False)
    snr_df.to_csv(DATA_DIR / f"snr_summary_{file_date}.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] snr_summary：{len(snr_df)} 列（最後日期：{file_date}）")

if __name__ == "__main__":
    main()
