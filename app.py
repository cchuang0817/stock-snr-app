
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import math

st.set_page_config(page_title="看股空間｜SNR + 基本面（繁中）", layout="wide")

# -------- 基礎工具 --------
@st.cache_data(ttl=60*30)  # 快取 30 分鐘
def 取得歷史價(ticker: str, days: int = 365) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{int(days)}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="ticker"
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
    距支撐 = (收盤 - 支撐) / 支撐 if 支撐 else math.inf
    距壓力 = (壓力 - 收盤) / 壓力 if 壓力 else math.inf
    if 距支撐 <= 近距比例:
        return "接近支撐：偏多、可分批佈局"
    if 距壓力 <= 近距比例:
        return "接近壓力：保守、等待回檔"
    if 收盤 < (支撐 + 壓力) / 2:
        return "區間下半：偏多但勿追高"
    else:
        return "區間上半：觀望或逢高減碼"

@st.cache_data(ttl=60*60)
def 取得基本面(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    fast = {}
    try:
        fast = t.fast_info or {}
    except Exception:
        fast = {}
    return {
        "代碼": ticker,
        "名稱": info.get("shortName"),
        "產業": info.get("sector"),
        "子產業": info.get("industry"),
        "市值": info.get("marketCap") or fast.get("market_cap"),
        "本益比(近12月)": info.get("trailingPE"),
        "本益比(預估)": info.get("forwardPE"),
        "股價淨值比": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "毛利率": info.get("grossMargins"),
        "營益率": info.get("operatingMargins"),
        "營收成長": info.get("revenueGrowth"),
        "獲利成長": info.get("earningsGrowth"),
        "最新價": fast.get("last_price") or info.get("currentPrice"),
        "幣別": info.get("currency") or fast.get("currency"),
    }

def 序列排名(s: pd.Series, 越小越好=True):
    r = s.rank(method="average", ascending=越小越好, na_option="keep")
    if r.isna().all():
        return pd.Series([float("nan")] * len(s), index=s.index)
    rmin, rmax = r.min(), r.max()
    if rmax == rmin:
        return pd.Series([0.5] * len(s), index=s.index)
    # 轉 0~1，1 代表較佳
    return (rmax - r) / (rmax - rmin)

def 加總評分(df: pd.DataFrame):
    val = pd.DataFrame(index=df.index)
    val["PE"] = 序列排名(df["本益比(近12月)"], 越小越好=True)
    val["PB"] = 序列排名(df["股價淨值比"], 越小越好=True)
    val["估值分數"] = val[["PE","PB"]].mean(axis=1, skipna=True)

    qua = pd.DataFrame(index=df.index)
    qua["ROE"] = 序列排名(df["ROE"], 越小越好=False)
    qua["毛利"] = 序列排名(df["毛利率"], 越小越好=False)
    qua["營益"] = 序列排名(df["營益率"], 越小越好=False)
    qua["品質分數"] = qua[["ROE","毛利","營益"]].mean(axis=1, skipna=True)

    gro = pd.DataFrame(index=df.index)
    gro["營收"] = 序列排名(df["營收成長"], 越小越好=False)
    gro["獲利"] = 序列排名(df["獲利成長"], 越小越好=False)
    gro["成長分數"] = gro[["營收","獲利"]].mean(axis=1, skipna=True)

    score = pd.concat([val["估值分數"], qua["品質分數"], gro["成長分數"]], axis=1)
    score["總分"] = 0.40*score["估值分數"] + 0.35*score["品質分數"] + 0.25*score["成長分數"]
    return pd.concat([df, score], axis=1)

def 畫SNR(df_snr: pd.DataFrame, 標題: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_snr["日期"], df_snr["Close"], label="收盤")
    ax.plot(df_snr["日期"], df_snr["支撐"], linestyle="--", label="支撐(20%)")
    ax.plot(df_snr["日期"], df_snr["中位"], linestyle="--", label="中位(50%)")
    ax.plot(df_snr["日期"], df_snr["壓力"], linestyle="--", label="壓力(80%)")
    ax.set_title(標題)
    ax.set_xlabel("日期"); ax.set_ylabel("價格"); ax.legend()
    st.pyplot(fig)

# -------- 側邊欄設定 --------
st.sidebar.title("設定（免費版）")
預設股票池 = ["2330.TW","2454.TW","2317.TW","2379.TW","3008.TW","NVDA","AAPL","MSFT"]
uni_text = st.sidebar.text_area("股票池（以逗號分隔；台股請加 .TW）", ",".join(預設股票池))
股票池 = [u.strip() for u in uni_text.split(",") if u.strip()]
TopN = st.sidebar.number_input("Top-N（每日精選數量）", min_value=1, max_value=50, value=5, step=1)
SNR視窗 = st.sidebar.slider("SNR 視窗（天）", min_value=60, max_value=240, value=120, step=10)
近距百分比 = st.sidebar.slider("接近支撐/壓力判定（%）", min_value=1, max_value=10, value=3, step=1) / 100.0

st.title("看股空間（繁體中文）｜SNR + 基本面挑選")
st.caption("資料來源：yfinance（免費、最佳努力；可能有缺值/延遲）。建議收盤後使用。")

分頁1, 分頁2, 分頁3, 分頁4 = st.tabs(["每日 Top-N", "單檔查詢", "產業/族群 Top-N", "Top-N 擴充"])

# -------- 分頁1：每日 Top-N --------
with 分頁1:
    st.subheader("每日 Top-N（估值40% + 品質35% + 成長25%）")
    rows = [取得基本面(t) for t in 股票池]
    全部 = pd.DataFrame(rows)
    已評分 = 加總評分(全部).sort_values("總分", ascending=False)
    顯示欄 = ["代碼","名稱","產業","子產業","本益比(近12月)","股價淨值比","ROE","毛利率","營益率","營收成長","獲利成長","總分"]
    st.dataframe(已評分[顯示欄], use_container_width=True)
    精選 = 已評分.head(TopN)

    st.markdown("**本日精選 Top-N**")
    st.dataframe(精選[["代碼","名稱","總分"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### Top-N 的 SNR 圖與建議")
    for t in 精選["代碼"].tolist():
        hist = 取得歷史價(t, 365)
        if hist.empty:
            st.warning(f"{t} 無法取得歷史資料")
            continue
        snr = 計算SNR(hist, SNR視窗)
        last = snr.dropna(subset=["Close"]).iloc[-1]
        建議 = 建議文字(last["Close"], last.get("支撐"), last.get("壓力"), 近距百分比)
        st.markdown(f"**{t}**｜{last['日期'].date()}｜現價：{round(float(last['Close']),2)}｜建議：{建議}")
        畫SNR(snr, f"{t} 的 SNR")

# -------- 分頁2：單檔查詢 --------
with 分頁2:
    st.subheader("單檔查詢")
    q = st.text_input("輸入股票代碼（台股請加 .TW，例如 2330.TW）", "2330.TW")
    if st.button("查詢", key="lookup"):
        hist = 取得歷史價(q, 365)
        if hist.empty:
            st.error("抓不到資料，請檢查代碼或稍後再試。")
        else:
            snr = 計算SNR(hist, SNR視窗)
            last = snr.dropna(subset=["Close"]).iloc[-1]
            建議 = 建議文字(last["Close"], last.get("支撐"), last.get("壓力"), 近距百分比)
            st.markdown(f"**{q}**｜{last['日期'].date()}｜現價：{round(float(last['Close']),2)}｜建議：{建議}")
            畫SNR(snr, f"{q} 的 SNR")
            meta = 取得基本面(q)
            st.json(meta)

# -------- 分頁3：產業/族群 Top-N --------
with 分頁3:
    st.subheader("產業/族群 Top-N")
    kw = st.text_input("輸入產業/關鍵字（如 Semiconductor / 半導體）", "Semiconductor")
    rows = [取得基本面(t) for t in 股票池]
    df = pd.DataFrame(rows)
    mask = df[["產業","子產業","名稱"]].astype(str).apply(lambda s: s.str.contains(kw, case=False, na=False))
    篩選 = df[mask.any(axis=1)]
    if 篩選.empty:
        st.info("在股票池中找不到符合該產業/關鍵字的標的。")
    else:
        得分 = 加總評分(篩選).sort_values("總分", ascending=False)
        k = st.number_input("取前 N 名", min_value=1, max_value=50, value=min(TopN, len(得分)), step=1)
        st.dataframe(得分.head(k)[["代碼","名稱","產業","子產業","總分"]], use_container_width=True)

# -------- 分頁4：Top-N 擴充 --------
with 分頁4:
    st.subheader("Top-N 擴充檢視")
    rows = [取得基本面(t) for t in 股票池]
    得分 = 加總評分(pd.DataFrame(rows)).sort_values("總分", ascending=False)
    k = st.slider("顯示前幾名", min_value=TopN, max_value=min(50, len(得分)), value=min(10, len(得分)), step=1)
    show_cols = ["代碼","名稱","產業","子產業","本益比(近12月)","股價淨值比","ROE","毛利率","營益率","營收成長","獲利成長","總分"]
    st.dataframe(得分.head(k)[show_cols], use_container_width=True)

st.caption("免責聲明：本工具僅供研究與教育用途，非投資建議。")
