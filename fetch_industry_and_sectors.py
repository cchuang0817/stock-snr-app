import pandas as pd
import requests
from io import StringIO
from datetime import datetime

def update_industry_map():
    urls = {
        "上市": "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv",
        "上櫃": "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
    }
    dfs = []
    for name, url in urls.items():
        resp = requests.get(url)
        resp.encoding = "utf-8"
        df = pd.read_csv(StringIO(resp.text))
        if "公司代號" in df.columns and "產業別" in df.columns:
            df = df[["公司代號", "產業別"]].copy()
            df["公司代號"] = df["公司代號"].astype(str).str.zfill(4) + ".TW"
            df["產業別"] = df["產業別"].fillna("未分類")
            dfs.append(df)
    if dfs:
        final_df = pd.concat(dfs).drop_duplicates(subset=["公司代號"])
        final_df.to_csv("data/industry_map.csv", index=False, encoding="utf-8-sig")
        print(f"產業對照表已更新，{len(final_df)} 筆")

def update_sector_performance():
    today = datetime.now().strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={today}&type=IND"
    resp = requests.get(url)
    resp.encoding = "big5"
    content = resp.text.split("\n")
    data_lines = [line for line in content if len(line.split("","")) > 5]
    csv_data = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_data.replace("=", "")))
    df = df.rename(columns=lambda x: x.strip())
    # 取加權指數漲跌幅
    taiex_row = df[df["指數名稱"].str.contains("發行量加權股價指數")]
    if taiex_row.empty:
        print("未取得加權指數資料")
        return
    taiex_change = float(taiex_row["漲跌點數"].values[0]) / (float(taiex_row["收盤指數"].values[0]) - float(taiex_row["漲跌點數"].values[0])) * 100
    results = []
    for _, row in df.iterrows():
        name = row["指數名稱"]
        try:
            change_pct = float(row["漲跌點數"]) / (float(row["收盤指數"]) - float(row["漲跌點數"])) * 100
            diff = change_pct - taiex_change
            if diff >= 0.3:
                relation = "優於大盤"
            elif diff <= -0.3:
                relation = "劣於大盤"
            else:
                relation = "相似"
            results.append({"產業名稱": name, "漲跌幅%": round(change_pct, 2), "相對表現": relation})
        except:
            continue
    pd.DataFrame(results).to_csv("data/sectors_daily.csv", index=False, encoding="utf-8-sig")
    print("產業相對表現已更新")
