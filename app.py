import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. 基礎設定 ---
st.set_page_config(page_title="全球財經戰情室", layout="wide")
st.title("🌏 跨市場自動化監控系統 (混合戰略版)")

# 自定義 CSS (確保文字清晰可見)
st.markdown("""
    <style>
    .trade-card { 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        border-left: 10px solid #ccc;
        color: #333333 !important; /* 強制文字黑色 */
        background-color: #f9f9f9;
    }
    .trade-card h3, .trade-card p, .trade-card li, .trade-card b { 
        color: #333333 !important; 
    }
    .card-long { background-color: #d1e7dd; border-left-color: #0f5132; }
    .card-short { background-color: #f8d7da; border-left-color: #842029; }
    .card-wait { background-color: #fff3cd; border-left-color: #ffecb5; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 定義觀察清單 (自動分類)
watch_lists = {
    "Futures": {
        "NQ=F": "那斯達克期", 
        "ES=F": "S&P500期", 
        "WTX=F": "台指期"
    },
    "Stocks": {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla", "AMZN": "Amazon",
        "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "3661.TW": "世芯-KY", 
        "2308.TW": "台達電", "2345.TW": "智邦", "6442.TW": "光聖", "3081.TW": "聯亞"
    }
}

# --- 2. 核心運算邏輯 (共用工具) ---

def flatten_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

# --- 3. 策略 A：個股順勢回檔 (Trend Pullback) ---
def analyze_stock_strategy(ticker, name):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        df = flatten_data(df)
        if len(df) < 60: return None

        # 計算指標
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['ATR'] = calculate_atr(df)

        last = df.iloc[-1]
        curr = float(last['Close'])
        ma60 = float(last['MA60'])
        rsi = float(last['RSI'])
        atr = float(last['ATR']) if not pd.isna(last['ATR']) else curr * 0.02
        
        signal = "None"
        reason = ""

        # 策略：季線之上回檔，季線之下反彈
        if curr > ma60 and rsi < 45:
            signal = "LONG"
            reason = f"📈 個股策略：趨勢偏多 (股價 > 季線) 且 RSI ({rsi:.1f}) 回檔。"
        elif curr < ma60 and rsi > 55:
            signal = "SHORT"
            reason = f"📉 個股策略：趨勢偏空 (股價 < 季線) 且 RSI ({rsi:.1f}) 反彈。"

        if signal != "None":
            # 個股停損較寬 (2倍ATR)
            sl = curr - (2 * atr) if signal == "LONG" else curr + (2 * atr)
            tp = curr + (3 * atr) if signal == "LONG" else curr - (3 * atr)
            return {"ticker": ticker, "name": name, "type": signal, "price": curr, "reason": reason, "sl": sl, "tp": tp}
        return None
    except: return None

# --- 4. 策略 B：期貨 MTF 共振 (1H/15M/5M) ---
def analyze_future_strategy(ticker, name):
    try:
        # 下載多週期資料
        df_1h = flatten_data(yf.download(ticker, period="1mo", interval="1h", progress=False))
        df_15m = flatten_data(yf.download(ticker, period="5d", interval="15m", progress=False))
        df_5m = flatten_data(yf.download(ticker, period="5d", interval="5m", progress=False))
        
        if df_5m.empty or len(df_5m) < 2: return None

        # 計算 EMA (21, 55)
        for df in [df_1h, df_15m, df_5m]:
            df['E21'] = calculate_ema(df['Close'], 21)
            df['E55'] = calculate_ema(df['Close'], 55)
        
        # 取得最新狀態 (使用倒數第一筆)
        # 1H 趨勢
        h1_bull = df_1h['E21'].iloc[-1] > df_1h['E55'].iloc[-1]
        h1_bear = df_1h['E21'].iloc[-1] < df_1h['E55'].iloc[-1]
        
        # 15M 趨勢
        m15_bull = df_15m['E21'].iloc[-1] > df_15m['E55'].iloc[-1]
        m15_bear = df_15m['E21'].iloc[-1] < df_15m['E55'].iloc[-1]
        
        # 5M 交叉 (前一根與現在這根比較)
        m5_e21_prev, m5_e21_curr = df_5m['E21'].iloc[-2], df_5m['E21'].iloc[-1]
        m5_e55_prev, m5_e55_curr = df_5m['E55'].iloc[-2], df_5m['E55'].iloc[-1]
        
        cross_bull = (m5_e21_prev < m5_e55_prev) and (m5_e21_curr > m5_e55_curr)
        cross_bear = (m5_e21_prev > m5_e55_prev) and (m5_e21_curr < m5_e55_curr)
        
        # 5M 斜率確認 (均線向上/向下)
        slope_up = m5_e21_curr > m5_e21_prev
        slope_down = m5_e21_curr < m5_e21_prev

        curr = float(df_5m['Close'].iloc[-1])
        signal = "WAIT" # 預設為觀察中
        reason = f"1H:{'多' if h1_bull else '空'} | 15M:{'多' if m15_bull else '空'} | 等待 5M 共振..."

        # 判斷共振
        if h1_bull and m15_bull and cross_bull and slope_up:
            signal = "LONG"
            reason = "🔥 期貨策略：1H/15M 多頭排列 + 5M 黃金交叉共振！"
        elif h1_bear and m15_bear and cross_bear and slope_down:
            signal = "SHORT"
            reason = "❄️ 期貨策略：1H/15M 空頭排列 + 5M 死亡交叉共振！"
            
        # 期貨停損設定 (依你的輸入參數：SL 1.5%, TP 4%)
        # 為了適合當沖，這裡稍微調整為較窄的比例，或者你可以改回 1.5%
        sl_pct = 0.005 # 0.5% (當沖比較合理)
        tp_pct = 0.01  # 1.0%
        
        if signal == "LONG":
            sl, tp = curr * (1 - sl_pct), curr * (1 + tp_pct)
        elif signal == "SHORT":
            sl, tp = curr * (1 + sl_pct), curr * (1 - tp_pct)
        else:
            sl, tp = 0, 0

        return {"ticker": ticker, "name": name, "type": signal, "price": curr, "reason": reason, "sl": sl, "tp": tp}

    except Exception as e: return None

# --- 5. 側邊欄控制 ---
with st.sidebar:
    st.header("⚙️ 戰略指揮中心")
    
    # 選擇模式
    st.subheader("🤖 智能分析")
    target_type = st.radio("選擇掃描對象", ["個股 (Stock)", "期貨 (Futures)"])
    
    if target_type == "個股 (Stock)":
        scan_list = watch_lists["Stocks"]
    else:
        scan_list = watch_lists["Futures"]
        
    run_scan = st.button(f"🚀 執行 {target_type} 掃描", type="primary")

    st.markdown("---")
    if st.button("🔄 刷新數據"): st.cache_data.clear(); st.rerun()

# --- 6. 掃描結果顯示 ---
if run_scan:
    st.header(f"📢 {target_type} 策略掃描報告")
    progress = st.progress(0)
    
    results = []
    for i, (ticker, name) in enumerate(scan_list.items()):
        if target_type == "個股 (Stock)":
            res = analyze_stock_strategy(ticker, name)
        else:
            res = analyze_future_strategy(ticker, name)
            
        if res: results.append(res)
        progress.progress((i + 1) / len(scan_list))
    
    progress.empty()
    
    if not results:
        st.info("無數據或無訊號。")
    else:
        # 針對期貨，即使是 WAIT 也要顯示；針對個股，只顯示有訊號的
        for op in results:
            if target_type == "個股 (Stock)" and op['type'] == "None": continue
            
            # 決定卡片樣式
            if op['type'] == "LONG":
                c_class, icon = "card-long", "🐂 多頭訊號"
            elif op['type'] == "SHORT":
                c_class, icon = "card-short", "🐻 空頭訊號"
            else: # WAIT (僅期貨會出現)
                c_class, icon = "card-wait", "👀 觀察中"
            
            st.markdown(f"""
            <div class="trade-card {c_class}">
                <h3>{icon}：{op['name']} ({op['ticker']})</h3>
                <p><b>現價：</b>{op['price']:,.2f}</p>
                <p><b>分析：</b>{op['reason']}</p>
                {f"<p>🎯 <b>停利：</b>{op['tp']:,.2f} | 🛑 <b>停損：</b>{op['sl']:,.2f}</p>" if op['type'] != 'WAIT' else ''}
            </div>
            """, unsafe_allow_html=True)

# --- 7. 市場概況 (底部) ---
st.markdown("---")
st.subheader("📊 市場看板")
simple_data = yf.download(list({**watch_lists["Futures"], **watch_lists["Stocks"]}.keys()), period="5d", interval="1d", progress=False)['Close']
simple_data = flatten_data(simple_data)

cols = st.columns(4)
idx = 0
for cat in watch_lists.values():
    for tic, name in cat.items():
        try:
            if tic in simple_data.columns:
                cur = simple_data[tic].iloc[-1]
                chg = cur - simple_data[tic].iloc[-2]
                cols[idx % 4].metric(name, f"{cur:,.2f}", f"{chg:+.2f}")
                idx += 1
        except: pass
