import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math, os, glob, datetime as dt

st.set_page_config(page_title="看股空間｜全市場 + 產業篩選（每日資料庫）", layout="wide")

DATA_FUND = "data/fundamentals.parquet"
DATA_SNR  = "data/snr_summary.parquet"

# ---------- 資料庫讀取（Parquet 優先；失敗/為空則改讀最新 CSV） ----------
@st.cache_data(ttl=60*30)
def load_db():
    fund_df = None
    snr_df  = None
    used_files = []
    # 先讀 Parquet
    if os.path.exists(DATA_FUND):
        try:
            fund_df = pd.read_parquet(DATA_FUND)
            used_files.append("fundamentals.parquet")
        except Exception as e:
            used_files.append(f"fundamentals.parquet 讀取失敗：{e}")
    if os.path.exists(DATA_SNR):
        try:
            snr_df = pd.read_parquet(DATA_SNR)
            used_files.append("snr_summary.parquet")
        except Exception as e:
            used_files.append(f"snr_summary.parquet 讀取失敗：{e}")

    # 後援：CSV（挑最新一個）
    if fund_df is None or fund_df.empty:
        cand = sorted(glob.glob("data/fundamentals_*.csv"))
        if cand:
            fund_df = pd.read_csv(cand[-1])
            used_files.append(os.path.basename(cand[-1]))
    if snr_df is None or snr_df.empty:
        cand = sorted(glob.glob("data/snr_summary_*.csv"))
        if cand:
            snr_df = pd.read_csv(cand[-1])
            used_files.append(os.path.basename(cand[-1]))

    # 推導資料日期：優先用欄位 asOfDate；否則從 CSV 檔名取日期；都沒有就 None
    data_date = None
    def pick_date(df, csv_pattern):
        if df is not None and not df.empty and "asOfDate" in df.columns:
            try:
                return str(pd.to_datetime(df["asOfDate"].iloc[0]).date())
            except Exception:
                pass
        cand = sorted(glob.glob(csv_pattern))
        if cand:
            # 檔名形如 data/snr_summary_YYYY-MM-DD.csv
            name = os.path.basename(cand[-1])
            try:
                return name.split("_")[-1].replace(".csv","")
            except Exception:
                return None
        return None

    # 以 fundamentals 為主，沒有就看 snr_summary
    data_date = pick_date(fund_df, "data/fundamentals_*.csv") or pick_date(snr_df, "data/snr_summary_*.csv")
    return fund_df, snr_df, used_files, data_date

# ---------- SNR 與工具 ----------
@st.cache_data(ttl=60*30)
def fetch_history(ticker: str, days: int = 365) -> pd.DataFrame:
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
    df = df.rename_axis("日期").reset_index()
    return df.dropna(subset=["Close"])

def compute_snr(df: pd.DataFrame, window: int = 120):
    if df.empty or "Close" not in df.columns:
        return df.assign(支撐=pd.NA, 中位=pd.NA, 壓力=pd.NA)
    roll = df["Close"].rolling(window, min_periods=max(20, window//4))
    out = df.copy()
    out["支撐"] = roll.quantile(0.20)
    out["中位"] = roll.quantile(0.50)
    out["壓力"] = roll.quantile(0.80)
    return out

def suggest(last_close, S, R, near=0.03):
    import pandas as pd
    if any(pd.isna(x) for x in (last_close, S, R)):
        return "資料不足"
    dS = (last_close - S) / S if S else math.inf
    dR = (R - last_close) / R if R else math.inf
    if dS <= near: return "接近支撐：偏多、可分批佈局"
    if dR <= near: return "接近壓力：保守、等待回檔"
    if last_close < (S + R) / 2: return "區間下半：偏多但勿追高"
    return "區間上半：觀望或逢高減碼"

def score_frame(df: pd.DataFrame):
    def rank(s: pd.Series, asc=True):
        r = s.rank(method="average", ascending=asc, na_option="keep")
        if r.isna().all(): return pd.Series([float("nan")] * len(s), index=s.index)
        rmin, rmax = r.min(), r.max()
        if rmax == rmin: return pd.Series([0.5] * len(s), index=s.index)
        return (rmax - r) / (rmax - rmin)
    val = pd.DataFrame(index=df.index)
    val["PE"] = rank(df["trailingPE"], asc=True)
    val["PB"] = rank(df["priceToBook"], asc=True)
    val["估值分數"] = val[["PE","PB"]].mean(axis=1, skipna=True)
    qua = pd.DataFrame(index=df.index)
    qua["ROE"] = rank(df["returnOnEquity"], asc=False)
    qua["毛利"] = rank(df["grossMargins"], asc=False)
    qua["營益"] = rank(df["operatingMargins"], asc=False)
    qua["品質分數"] = qua[["ROE","毛利","營益"]].mean(axis=1, skipna=True)
    gro = pd.DataFrame(index=df.index)
    gro["營收"] = rank(df["revenueGrowth"], asc=False)
    gro["獲利"] = rank(df["earningsGrowth"], asc=False)
    gro["成長分數"] = gro[["營收","獲利"]].mean(axis=1, skipna=True)
    score = pd.concat([val["估值分數"], qua["品質分數"], gro["成長分數"]], axis=1)
    score["total_score"] = 0.40*score["估值分數"] + 0.35*score["品質分數"] + 0.25*score["成長分數"]
    return pd.concat([df, score], axis=1)

# ---------- UI ----------
st.title("看股空間｜全市場 + 產業篩選（每日資料庫）")
st.caption("每日台北時間 18:30 由 GitHub Actions 更新資料庫。若遇假日，採用最近交易日資料。")

fund_df, snr_df, used_files, data_date = load_db()

# 資料來源提示
if used_files:
    st.info("讀取來源：" + "、".join(used_files))
if data_date:
    st.success(f"資料日期：{data_date}")

colL, colR = st.columns([3,2], gap="large")

# -------- 左：Top-N 與圖 --------
with colL:
    if fund_df is None or fund_df.empty:
        st.warning("資料庫不存在或為空。請先在 GitHub Actions 手動執行一次更新，或稍後再試。")
    else:
        # 產業類別清單（可能混中英）
        sectors    = sorted(set(str(x) for x in fund_df["sector"].dropna().unique()))
        industries = sorted(set(str(x) for x in fund_df["industry"].dropna().unique()))
        st.subheader("篩選條件")
        pick = st.selectbox("選擇產業（不選代表全部）", ["全部"] + sectors + industries, index=0)
        kw   = st.text_input("或輸入關鍵字（例如：半導體 / Semiconductor）", "")
        topn = st.number_input("Top-N（精選數量）", min_value=1, max_value=100, value=5, step=1)
        win  = st.slider("SNR 視窗（天）", min_value=60, max_value=240, value=120, step=10)
        near = st.slider("接近支撐/壓力判定（%）", min_value=1, max_value=10, value=3, step=1) / 100.0

        df = fund_df.copy()
        if pick != "全部":
            mask = df[["sector","industry","shortName"]].astype(str).apply(lambda s: s.str.contains(pick, case=False, na=False))
            df = df[mask.any(axis=1)]
        if kw.strip():
            mask = df[["sector","industry","shortName"]].astype(str).apply(lambda s: s.str.contains(kw, case=False, na=False))
            df = df[mask.any(axis=1)]

        if df.empty:
            st.warning("沒有符合條件的股票。請更換產業或關鍵字。")
        else:
            scored = score_frame(df).sort_values("total_score", ascending=False)
            show_cols = ["Ticker","shortName","sector","industry","trailingPE","priceToBook",
                         "returnOnEquity","grossMargins","operatingMargins","revenueGrowth","earningsGrowth",
                         "total_score"]
            st.markdown("**排名表（依總分）**")
            st.dataframe(scored[show_cols], use_container_width=True, height=360)

            picks = scored.head(topn)

            st.markdown("---")
            st.markdown("### Top-N 的 SNR 圖與建議")
            for t in picks["Ticker"].tolist():
                # 若資料庫有 SNR 概況，先顯示
                if snr_df is not None and not snr_df.empty:
                    row = snr_df[snr_df["Ticker"] == t].tail(1)
                    if not row.empty:
                        st.markdown(
                            f"**{t}**｜{row['LastDate'].iloc[0]}｜現價：{round(float(row['Close'].iloc[0]),2)}"
                        )
                        # 依近距再給建議（以現算為準更準確）
                # 畫 SNR 圖（即時算 Top-N，不會太慢）
                hist = fetch_history(t, 365)
                if hist.empty:
                    st.warning(f"{t} 無法取得歷史資料")
                    continue
                snr = compute_snr(hist, win)
                last = snr.dropna(subset=["Close"]).iloc[-1]
                advice = suggest(last["Close"], last.get("支撐"), last.get("壓力"), near)
                st.write(f"建議：{advice}")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(snr["日期"], snr["Close"], label="收盤")
                ax.plot(snr["日期"], snr["支撐"], linestyle="--", label="支撐(20%)")
                ax.plot(snr["日期"], snr["中位"], linestyle="--", label="中位(50%)")
                ax.plot(snr["日期"], snr["壓力"], linestyle="--", label="壓力(80%)")
                ax.set_title(f"{t} 的 SNR"); ax.set_xlabel("日期"); ax.set_ylabel("價格"); ax.legend()
                st.pyplot(fig)

# -------- 右：單檔查詢 --------
with colR:
    st.subheader("單檔查詢（即時）")
    q = st.text_input("輸入股票代碼（台股 .TW/.TWO，例如 2330.TW）", "2330.TW")
    if st.button("查詢"):
        hist = fetch_history(q, 365)
        if hist.empty:
            st.error("抓不到資料，請檢查代碼或稍後再試。")
        else:
            snr = compute_snr(hist, 120)
            last = snr.dropna(subset=["Close"]).iloc[-1]
            st.markdown(f"**{q}**｜{last['日期'].date()}｜現價：{round(float(last['Close']),2)}")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(snr["日期"], snr["Close"], label="收盤")
            ax.plot(snr["日期"], snr["支撐"], linestyle="--", label="支撐(20%)")
            ax.plot(snr["日期"], snr["中位"], linestyle="--", label="中位(50%)")
            ax.plot(snr["日期"], snr["壓力"], linestyle="--", label="壓力(80%)")
            ax.set_title(f"{q} 的 SNR"); ax.set_xlabel("日期"); ax.set_ylabel("價格"); ax.legend()
            st.pyplot(fig)

st.caption("免責聲明：本工具僅供研究與教育用途，非投資建議。")
