import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import glob
import chardet
import io
import pyperclip

# ==== 15 分鐘乘數表（模擬用）====
volume_multiplier_15m = {
    "09:15": 8.00, "09:30": 5.00, "09:45": 4.00, "10:00": 3.00,
    "10:15": 2.50, "10:30": 2.20, "10:45": 2.10, "11:00": 1.90,
    "11:15": 1.70, "11:30": 1.60, "11:45": 1.50, "12:00": 1.45,
    "12:15": 1.38, "12:30": 1.32, "12:45": 1.25, "13:00": 1.17,
    "13:15": 1.10, "13:30": 1.00
}

# ==== 停損 / 停利 ====
def stop_loss(entry_price: float, rate: float = 0.025) -> float:
    return round(entry_price * (1 - rate), 2)

def take_profit(entry_price: float, rate: float = 0.02) -> float:
    return round(entry_price * (1 + rate), 2)

# ==== 下載歷史資料 ====
def fetch_history(symbol: str, period: str = "1mo", interval: str = "1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df

# ==== 大盤濾網 ====
def fetch_index(period="6mo"):
    idx = yf.Ticker("^TWII").history(period=period, interval="1d")
    idx["MA20"] = idx["Close"].rolling(20).mean()
    return idx

# ==== 技術指標 ====
def calc_indicators(df):
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    return df

# ==== 股本 & 營收資料篩選 ====
def load_capital(over25=True):
    if over25:
        df = pd.read_csv("capital_over25.csv", encoding="utf-8")
    else:
        df = pd.read_csv("capital_under25.csv", encoding="utf-8")
    df["symbol"] = df["公司代號"].astype(str) + ".TW"
    return df

def load_revenue_trend():
    files = sorted(glob.glob("上市公司114年*.csv"))
    df_all = {}
    for f in files:
        with open(f, "rb") as fin:
            raw = fin.read(4096)
            encoding = chardet.detect(raw)["encoding"]
        df = pd.read_csv(f, encoding=encoding)
        month = f.split("114年")[1].split("月份")[0]
        df_all[month] = df.rename(columns={
            "公司代號": "代號",
            "營業收入-當月營收": f"{month}月營收",
            "營業收入-去年當月營收": f"{month}月去年營收",
            "累計營業收入-當月累計營收": f"{month}月累計",
            "累計營業收入-去年累計營收": f"{month}月去年累計"
        })[["代號", f"{month}月營收", f"{month}月去年營收", f"{month}月累計", f"{month}月去年累計"]]

    revenue_df = df_all["6"].merge(df_all["7"], on="代號").merge(df_all["8"], on="代號")
    cond_trend = (revenue_df["6月營收"] < revenue_df["7月營收"]) & (revenue_df["7月營收"] < revenue_df["8月營收"])
    cond_cum = revenue_df["8月累計"] > revenue_df["8月去年累計"]

    filtered = revenue_df[cond_trend & cond_cum]
    filtered["symbol"] = filtered["代號"].astype(str) + ".TW"
    return filtered

# ==== 分析 ====
def analyze(symbol: str, period="1mo", tw_name_map=None):
    df = fetch_history(symbol, period=period, interval="1d")
    if df.empty:
        return None

    df["Vol_5"] = df["Volume"].rolling(5).mean()
    df["Vol_20"] = df["Volume"].rolling(20).mean()

    yesterday_close = df["Close"].iloc[-2]
    yesterday_vol = df["Volume"].iloc[-2]

    today_close = df["Close"].iloc[-1]
    today_vol = df["Volume"].iloc[-1]

    price_change = (today_close - yesterday_close) / yesterday_close * 100
    est_volume = today_vol * volume_multiplier_15m["09:30"]
    explosive = est_volume > 2 * yesterday_vol

    pos_5 = "高於 5 日均線" if today_close > df["Vol_5"].iloc[-1] else "低於 5 日均線"
    pos_20 = "高於 20 日均線" if today_close > df["Vol_20"].iloc[-1] else "低於 20 日均線"
    ma_position = f"{pos_5}、{pos_20}"

    buy_explosive = "爆量" if explosive else "正常"
    sell_vol_ratio = round(today_vol / df["Vol_5"].iloc[-1], 2) if df["Vol_5"].iloc[-1] else None

    stock_name_zh = tw_name_map.get(symbol, "") if tw_name_map else ""
    result = {
        "股票": f"{symbol} {stock_name_zh}",
        "昨日收盤": yesterday_close,
        "今日收盤": today_close,
        "漲跌幅%": round(price_change, 2),
        "昨日量": yesterday_vol,
        "今日量": today_vol,
        "5日均量": round(df["Vol_5"].iloc[-1], 0),
        "20日均量": round(df["Vol_20"].iloc[-1], 0),
        "模擬預估量": round(est_volume, 0),
        "爆量": explosive,
        "停損價": stop_loss(today_close),
        "停利價": take_profit(today_close),
        "方向": "多" if price_change > 2 else ("空" if price_change < -2 else "觀望"),
        "均線位置": ma_position,
        "成交量狀況": buy_explosive,
        "賣出量/5日均量": sell_vol_ratio,
    }
    return result

def plot_stock(symbol, df, result):
    fig = go.Figure()
    # X軸轉成字串日期
    x_dates = df.index.strftime("%Y-%m-%d")
    fig.add_trace(go.Candlestick(
        x=x_dates,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="K線",
        increasing_line_color="red",   # 漲用紅色
        decreasing_line_color="green"  # 跌用綠色
    ))
    fig.add_trace(go.Scatter(
        x=x_dates, y=df["Volume"], name="成交量", yaxis="y2", line=dict(color="orange")
    ))
    if result["方向"] == "多":
        fig.add_trace(go.Scatter(
            x=[x_dates[-1]], y=[df["Close"].iloc[-1]],
            mode="markers", name="買入點",
            marker=dict(color="green", size=12, symbol="star")
        ))
    fig.update_layout(
        title=f"{symbol} K線與爆量",
        yaxis_title="股價",
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False,
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),  # yyyy-mm-dd格式
        height=500
    )
    return fig

# ==== Streamlit 主程式 ====
st.set_page_config(page_title="批次掃描 (財報+技術)", layout="wide")
st.title("批次掃描：股本 + 營收快篩 + 技術分析")

st.markdown("""
**選股策略說明：**
- 先依「股本」篩選（可選 ≥25億或 <25億）
- 再依「營收連續成長」條件過濾（目前使用 6~8 月資料）
- 技術面：爆量且漲幅 ≥2% 判斷為「多頭」，只顯示多頭股票
- 停利 / 停損：預設停利 3%，停損 1.5%（可於介面調整）
- 強制出場天數：程式預設持股 3 天即出場，屬於短線操作推薦
- 其他條件如均線位置、成交量等皆自動計算
- 點擊「開始掃描」即可執行批次分析，並可下載 Excel 結果

> **提醒：目前營收資料僅涵蓋 6~8 月，每個月初請務必更新最新營收 csv，否則分析結果會落後！**
""")

# 資料區間直接寫死
period = "3mo"

# 股本篩選加「所有公司」
capital_filter = st.radio(
    "股本篩選",
    ["所有公司", "大公司 (股本 ≥25 億)", "小公司 (股本 <25 億)"],
    index=0
)
if capital_filter == "所有公司":
    use_over25 = None
elif capital_filter == "大公司 (股本 ≥25 億)":
    use_over25 = True
else:
    use_over25 = False

run = st.button("開始掃描")

df_twse = pd.read_csv("twse_list.csv", encoding="utf-8")
df_twse["symbol"] = df_twse["code"].astype(str) + ".TW"
tw_name_map = dict(zip(df_twse["symbol"], df_twse["name"]))

if run:
    # 股本篩選
    if use_over25 is None:
        cap_df = pd.concat([
            load_capital(over25=True),
            load_capital(over25=False)
        ], ignore_index=True)
    else:
        cap_df = load_capital(over25=use_over25)
    rev_df = load_revenue_trend()
    good_list = set(cap_df["symbol"]) & set(rev_df["symbol"])
    all_symbols = list(good_list)

    st.write(f"篩選後共 {len(all_symbols)} 檔股票")

    st.markdown("""
    **選股策略說明：**
    - 只顯示「多頭」股票的K線圖，代表技術面明確、勝率較高
    - 其他股票僅顯示數據，屬於觀望或空頭，不建議操作
    """)

    # 加入防呆，避免 yfinance RateLimitError
    try:
        idx_df = fetch_index(period=period)
    except Exception as e:
        st.warning(f"大盤資料取得失敗（可能被限制存取頻率），不影響個股分析。")
        idx_df = None

    results = []
    for sym in all_symbols:
        df = fetch_history(sym, period=period, interval="1d")
        if df.empty:
            continue
        res = analyze(sym, period=period, tw_name_map=tw_name_map)
        if res:
            results.append(res)

    df_result = pd.DataFrame(results)
    df_result_long = df_result[df_result["方向"] == "多"]
    st.session_state["df_result_long"] = df_result_long  # 保存到 session

# 只顯示多頭表格
df_result_long = st.session_state.get("df_result_long", pd.DataFrame())
st.subheader("分析結果（只顯示多頭）")
st.dataframe(df_result_long, use_container_width=True)

# 股票互動選單
if not df_result_long.empty:
    stock_choices = df_result_long["股票"].tolist()
    selected_stock = st.selectbox("點選股票顯示K線", stock_choices)
    selected_symbol = selected_stock.split()[0]
    df_selected = fetch_history(selected_symbol, period=period, interval="1d")
    res_selected = analyze(selected_symbol, period=period, tw_name_map=tw_name_map)
    if res_selected:
        fig_selected = plot_stock(selected_symbol, df_selected, res_selected)
        st.plotly_chart(fig_selected, use_container_width=True, key=f"{selected_symbol}_selected")

    # 判斷資料日期
    last_date = df_selected.index[-1]
    today = pd.Timestamp("today").normalize().tz_localize(last_date.tz)
    if last_date < today:
        st.warning(f"最新資料日期：{last_date.date()}，今日可能為休市日或尚未更新。")

    if not df_result_long.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_result_long.to_excel(writer, index=False)
        output.seek(0)
        st.download_button(
            "下載 Excel",
            data=output.getvalue(),
            file_name="history_analysis_long.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 新增：複製多頭股票代碼按鈕
        codes = [s.split()[0] for s in df_result_long["股票"].tolist()]
        codes_str = "\n".join(codes)
        st.text_area("複製分析結果（只顯示多頭）所有股票代碼", codes_str, height=100, key="codes_area")

        if st.button("複製到剪貼簿"):
            try:
                pyperclip.copy(codes_str)
                st.success("已複製到剪貼簿！")
            except Exception:
                st.warning("複製失敗，請手動複製上方內容。")
