
# 看股空間（全市場 + 產業篩選 + 每日資料庫）

## 每日自動更新
- GitHub Actions：`.github/workflows/daily.yml` 會在 **台北時間 18:30** 執行 `data_updater.py`。
- 產出資料庫：`data/fundamentals.parquet`、`data/snr_summary.parquet`，以及當天 CSV 快照。

## App 使用
- `app.py` 會優先讀取 `data/*.parquet`，可在「產業」下拉選擇（不選＝全部）。
- Top-N 直接從資料庫即時計算，僅對 Top-N 畫 SNR 圖（快速）。
