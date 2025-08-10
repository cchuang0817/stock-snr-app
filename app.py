import streamlit as st
import pandas as pd
import os
import glob

# 偵錯用資訊
st.sidebar.markdown("### 偵錯資訊")
st.sidebar.write("工作目錄：", os.getcwd())
st.sidebar.write("data 資料夾存在：", os.path.isdir("data"))
st.sidebar.write("data 資料夾檔案：", [os.path.basename(p) for p in glob.glob("data/*")])

# 資料載入函數
def load_data():
    if os.path.exists("data/fundamentals.parquet") and os.path.exists("data/snr_summary.parquet"):
        fundamentals = pd.read_parquet("data/fundamentals.parquet")
        snr_summary = pd.read_parquet("data/snr_summary.parquet")
        return fundamentals, snr_summary
    else:
        st.warning("資料庫不存在，將使用即時模式。")
        return None, None

# 主程式
st.title("看股空間｜全市場 + 產業篩選（每日資料庫）")

fundamentals, snr_summary = load_data()

if fundamentals is not None and snr_summary is not None:
    st.success("已讀取每日資料庫")
    st.dataframe(snr_summary.head(10))
else:
    st.info("目前為即時模式，請先建立資料庫或檢查檔案位置。")
