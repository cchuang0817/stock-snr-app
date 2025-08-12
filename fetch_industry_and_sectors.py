# fetch_industry_and_sectors.py
# 目的：
# (A) 從 MOPS 取得「公司代號 -> 產業類別」對照（抓不到就先產生空表，後續標「未分類」）
# (B) 根據全市場個股的「市值加權單日報酬」計算各產業相對於大盤（優/相/劣）

from __future__ import annotations
import pandas as pd
from pathlib import Path
import requests, time, os, re, glob
from typing import Optional, Tuple

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

# -----------------------------
# A) 產業對照：抓 MOPS 上市/上櫃公司彙總表
# -----------------------------
def _mops_try_fetch(url: str, method: str = "GET", data: Optional[dict] = None, retries: int = 3, sleep: float = 1.0) -> Optional[str]:
    for i in range(retries):
        try:
            if method == "POST":
                r = requests.post(url, headers=HEADERS, data=data, timeout=30)
            else:
                r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.text:
                return r.text
        except Exception:
            time.sleep(sleep * (i + 1))
    return None

def _parse_tables(html: str) -> list[pd.DataFrame]:
    # 先用 lxml，再用 bs4/html5lib
    try:
        return pd.read_html(html)
    except Exception:
        try:
            return pd.read_html(html, flavor="bs4")
        except Exception:
            return []

def _normalize_industry_map(df: pd.DataFrame) -> pd.DataFrame:
    # 期待含「公司代號」「產業類別」欄位；不同版型容錯處理
    cols = {c: str(c) for c in df.columns}
    df.columns = [cols[c].strip() for c in df.columns]
    code_col = next((c for c in df.columns if "公司代號" in c or "代號" in c), None)
    ind_col  = next((c for c in df.columns if "產業" in c), None)
    if code_col is None or ind_col is None:
        return pd.DataFrame(columns=["code", "twse_industry"])

    out = df[[code_col, ind_col]].copy()
    out.columns = ["code", "twse_industry"]
    out["code"] = out["code"].astype(str).str.extract(r"(\d{4})")[0]
    out = out.dropna(subset=["code"]).drop_duplicates(subset=["code"])
    out["twse_industry"] = out["twse_industry"].astype(str).str.strip()
    out.loc[out["twse_industry"].eq("") | out["twse_industry"].isna(), "twse_industry"] = "未分類"
    return out[["code", "twse_industry"]]

def update_industry_map() -> Path:
    """
    嘗試自 MOPS 抓上市(sii)與上櫃(otc)公司基本資料表，建立 code->產業 類別對照。
    抓不到時，仍會輸出一個空表（後續在 join 時會被填成「未分類」）。
    """
    # 常見的 MOPS 入口（頁面可能更動；我們同時嘗試多個）
    endpoints = [
        # t51sb01：公司基本資料（上市/上櫃）
        ("https://mops.twse.com.tw/mops/web/t51sb01", "GET", None),
        ("https://mops.twse.com.tw/mops/web/ajax_t51sb01", "POST", {"encodeURIComponent": "1", "step": "1", "firstin": "1", "TYPEK": "sii"}),
        ("https://mops.twse.com.tw/mops/web/ajax_t51sb01", "POST", {"encodeURIComponent": "1", "step": "1", "firstin": "1", "TYPEK": "otc"}),
        # 備援頁
        ("https://mops.twse.com.tw/mops/web/index", "GET", None),
    ]

    frames = []
    for url, method, data in endpoints:
        html = _mops_try_fetch(url, method, data)
        if not html:
            continue
        tables = _parse_tables(html)
        for t in tables:
            norm = _normalize_industry_map(t)
            if not norm.empty:
                frames.append(norm)

    if frames:
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"])
    else:
        merged = pd.DataFrame(columns=["code", "twse_industry"])

    # 輸出
    out_path = DATA_DIR / "industry_map.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

def attach_industry_to_fundamentals(fund_df: pd.DataFrame, industry_csv: Path) -> pd.DataFrame:
    """
    把 industry_map.csv 併到 fund_df（Ticker 為 2330.TW/2454.TWO）
    缺少分類的一律填「未分類」。
    """
    if not industry_csv.exists():
        # 沒有 map 時，全部未分類
        fund_df = fund_df.copy()
        fund_df["twse_industry"] = "未分類"
        return fund_df

    map_df = pd.read_csv(industry_csv, dtype={"code": str})
    fund_df = fund_df.copy()
    fund_df["code"] = fund_df["Ticker"].astype(str).str.extract(r"(\d{4})")[0]
    merged = fund_df.merge(map_df, on="code", how="left")
    merged["twse_industry"] = merged["twse_industry"].fillna("未分類")
    merged = merged.drop(columns=["code"])
    return merged

# -----------------------------
# B) 產業相對表現
# -----------------------------
def _weighted_daily_return(frame: pd.DataFrame, ret_col: str, mcap_col: str) -> float:
    df = frame[[ret_col, mcap_col]].dropna()
    if df.empty: return float("nan")
    w = df[mcap_col].clip(lower=0)
    if w.sum() == 0: return float("nan")
    return (df[ret_col] * w).sum() / w.sum()

def compute_sector_performance(fund_df_with_ind: pd.DataFrame, snr_df: pd.DataFrame,
                               ret_col: str = "DailyReturn", mcap_col: str = "marketCap",
                               threshold: float = 0.003) -> pd.DataFrame:
    """
    以個股「市值加權『單日報酬』」代表該產業表現，與全市場加權報酬比較：
      diff >= +0.3% → 優於大盤
      diff <= -0.3% → 劣於大盤
      其餘 → 相似
    """
    if fund_df_with_ind is None or fund_df_with_ind.empty or snr_df is None or snr_df.empty:
        return pd.DataFrame(columns=["twse_industry", "n", "industry_ret", "market_ret", "diff", "relation"])

    # 準備合併：snr_df 需要 DailyReturn（你在 updater 會補上）
    snr_use = snr_df[["Ticker", ret_col]].dropna()
    base = fund_df_with_ind.merge(snr_use, on="Ticker", how="inner")

    # 大盤（全市場加權）報酬
    market_ret = _weighted_daily_return(base, ret_col=ret_col, mcap_col=mcap_col)

    # 產業分組
    grp = base.groupby("twse_industry", dropna=False)
    rows = []
    for ind, g in grp:
        ind_ret = _weighted_daily_return(g, ret_col=ret_col, mcap_col=mcap_col)
        diff = ind_ret - market_ret if pd.notna(ind_ret) and pd.notna(market_ret) else float("nan")
        if pd.isna(diff):
            relation = "相似"
        elif diff >= threshold:
            relation = "優於大盤"
        elif diff <= -threshold:
            relation = "劣於大盤"
        else:
            relation = "相似"
        rows.append({
            "twse_industry": ind or "未分類",
            "n": len(g),
            "industry_ret": ind_ret,
            "market_ret": market_ret,
            "diff": diff,
            "relation": relation
        })
    out = pd.DataFrame(rows).sort_values(["relation", "diff"], ascending=[True, False])
    return out

def save_sector_performance(perf_df: pd.DataFrame, as_of_date: str) -> Path:
    out_path = DATA_DIR / "sectors_daily.parquet"
    perf = perf_df.copy()
    perf["asOfDate"] = as_of_date
    perf.to_parquet(out_path, index=False)
    return out_path
