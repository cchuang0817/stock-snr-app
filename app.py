
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math, os

st.set_page_config(page_title="看股空間｜全市場 + 產業篩選（繁中）", layout="wide")

DATA_FUND = "data/fundamentals.parquet"
DATA_SNR = "data/snr_summary.parquet"

# -------- 基本工具 --------
@st.cache_data(ttl=60*30)
def 取得歷史價(ticker: str, days: int = 365) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{int(days)}d", interval="1d", auto_adjust=True, progress=False, threads=False, group_by="ticker")
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

def 計算SNR(df: pd.DataFrame, 視窗天數: int = 120):
    if df.empty or "Close" not in df.columns:
        return df.assign(支撐=pd.NA, 中位=pd.NA, 壓力=pd.NA)
    close = df["Close"]
    roll = close.rolling(視窗天數, min_periods=max(20, 視窗天數//4))
    out = df.copy()
    out["支撐"] = roll.quantile(0.20)
    out["中位"] = roll.quantile(0.50)
    out["壓力"] = roll.quantile(0.80)
    return out

def 建議文字(收盤: float, 支撐: float, 壓力: float, 近距比例: float = 0.03):
    if any(pd.isna(x) for x in (收盤, 支撐, 壓力)):
        return "資料不足"
    距支撐 = (收盤 - 經濟 if False else (收盤 - 支撐)) / 支撐 if 支撐 else math.inf
    距壓力 = (壓力 - 收盤) / 壓力 if 壓力 else math.inf
    if 距支撐 <= 近距比例:
        return "接近支撐：偏多、可分批佈局"
    if 距壓力 <= 近距比例:
        return "接近壓力：保守、等待回檔"
    if 收盤 < (支撐 + 壓力) / 2:
        return "區間下半：偏多但勿追高"
    else:
        return "區間上半：觀望或逢高減碼"

def 加總評分(df: pd.DataFrame):
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

# -------- 介面 --------
st.title("看股空間｜全市場 + 產業篩選（每日資料庫）")
st.caption("每日台北時間 18:30 自動更新資料庫（GitHub Actions）。頁面優先讀資料庫，無庫時改即時計算。")

# 載入資料庫
fund_df, snr_df = None, None
if os.path.exists(DATA_FUND):
    fund_df = pd.read_parquet(DATA_FUND)
if os.path.exists(DATA_SNR):
    snr_df = pd.read_parquet(DATA_SNR)

左, 右 = st.columns([3,2], gap="large")

with 左:
    if fund_df is not None and not fund_df.empty:
        sectors = sorted(set(str(x) for x in fund_df["sector"].dropna().unique()))
        industries = sorted(set(str(x) for x in fund_df["industry"].dropna().unique()))
        st.subheader("篩選條件")
        產業類別 = st.selectbox("選擇產業（不選代表全部）", ["全部"] + sectors + industries, index=0)
        TopN = st.number_input("Top-N（精選數量）", min_value=1, max_value=100, value=5, step=1)
        SNR視窗 = st.slider("SNR 視窗（天）", min_value=60, max_value=240, value=120, step=10)
        近距百分比 = st.slider("接近支撐/壓力判定（%）", min_value=1, max_value=10, value=3, step=1) / 100.0

        df = fund_df.copy()
        if 產業類別 != "全部":
            mask = df[["sector","industry","shortName"]].astype(str).apply(lambda s: s.str.contains(產業類別, case=False, na=False))
            df = df[mask.any(axis=1)]
        if df.empty:
            st.warning("沒有符合條件的股票。")
        else:
            scored = 加總評分(df).sort_values("total_score", ascending=False)
            st.markdown("**排名表（依總分）**")
            show_cols = ["Ticker","shortName","sector","industry","trailingPE","priceToBook","returnOnEquity","grossMargins","operatingMargins","revenueGrowth","earningsGrowth","total_score"]
            st.dataframe(scored[show_cols], use_container_width=True, height=360)
            picks = scored.head(TopN)

            st.markdown("---")
            st.markdown("### Top-N 的 SNR 圖與建議")
            for t in picks["Ticker"].tolist():
                # 若有 SNR 資料庫，可直接顯示近況；同時現算圖
                if snr_df is not None and not snr_df.empty:
                    row = snr_df[snr_df["Ticker"] == t].tail(1)
                    if not row.empty:
                        last_date = row["LastDate"].iloc[0]
                        close = row["Close"].iloc[0]
                        s = row["S"].iloc[0]; r = row["R"].iloc[0]
                        st.markdown(f"**{t}**｜{last_date}｜現價：{round(float(close),2)}｜建議：{row['Suggestion'].iloc[0]}")

                # 畫 SNR 圖（僅 Top-N 即時計算，不會太慢）
                hist = 取得歷史價(t, 365)
                if hist.empty:
                    st.warning(f"{t} 無法取得歷史資料")
                    continue
                snr = 計算SNR(hist, SNR視窗)
                # 畫圖
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(snr["日期"], snr["Close"], label="收盤")
                ax.plot(snr["日期"], snr["支撐"], linestyle="--", label="支撐(20%)")
                ax.plot(snr["日期"], snr["中位"], linestyle="--", label="中位(50%)")
                ax.plot(snr["日期"], snr["壓力"], linestyle="--", label="壓力(80%)")
                ax.set_title(f"{t} 的 SNR")
                ax.set_xlabel("日期"); ax.set_ylabel("價格"); ax.legend()
                st.pyplot(fig)
    else:
        st.info("資料庫不存在，將使用即時模式。請稍後或先於 GitHub Actions 建立資料庫。")

with 右:
    st.subheader("單檔查詢（即時）")
    q = st.text_input("輸入股票代碼（台股 .TW/.TWO，例如 2330.TW）", "2330.TW")
    if st.button("查詢"):
        hist = 取得歷史價(q, 365)
        if hist.empty:
            st.error("抓不到資料，請檢查代碼或稍後再試。")
        else:
            snr = 計算SNR(hist, 120)
            last = snr.dropna(subset=["Close"]).iloc[-1]
            # 建議
            dS = (last["Close"] - last.get("支撐")) / last.get("支撐") if pd.notna(last.get("支撐")) else None
            dR = (last.get("壓力") - last["Close"]) / last.get("壓力") if pd.notna(last.get("壓力")) else None
            st.markdown(f"**{q}**｜{last['日期'].date()}｜現價：{round(float(last['Close']),2)}")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(snr["日期"], snr["Close"], label="收盤")
            ax.plot(snr["日期"], snr["支撐"], linestyle="--", label="支撐(20%)")
            ax.plot(snr["日期"], snr["中位"], linestyle="--", label="中位(50%)")
            ax.plot(snr["日期"], snr["壓力"], linestyle="--", label="壓力(80%)")
            ax.set_title(f"{q} 的 SNR")
            ax.set_xlabel("日期"); ax.set_ylabel("價格"); ax.legend()
            st.pyplot(fig)

st.caption("免責聲明：本工具僅供研究與教育用途，非投資建議。")
