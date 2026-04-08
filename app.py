import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
@st.cache_data 
def get_clean_data():
    nifty = yf.download("^NSEI", start="2022-01-01")
    if nifty.columns.nlevels > 1:
        nifty.columns = nifty.columns.get_level_values(0)
    
    df = nifty[['Open', 'High', 'Low', 'Close']].copy()
    df['Daily_Return_Percentage'] = df['Close'].pct_change() * 100
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    macro = yf.download(['INR=X', 'CL=F', '^INDIAVIX'], start="2022-01-01")['Close']
    df = df.join(macro)
    df.rename(columns={'INR=X': 'USD_INR', 'CL=F': 'Crude_Oil', '^INDIAVIX': 'Fear_Checker'}, inplace=True)
    
    df.ffill(inplace=True)
    return df.dropna()

# website
st.set_page_config(page_title="Nifty 50 AI Predictor", layout="wide")
st.title("🚀 Nifty 50 Real-Time AI Dashboard")

# Refresh Button
if st.sidebar.button('🔄 Refresh Data'):
    st.cache_data.clear()

df = get_clean_data()

# Show Live Price
current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
price_diff = current_price - prev_price
st.metric("Nifty 50 Live (Approx)", f"₹{current_price:,.2f}", f"{price_diff:,.2f}")

# train model
features = ['Daily_Return_Percentage', 'SMA_20', 'RSI_14', 'USD_INR', 'Crude_Oil', 'Fear_Checker']
X = df[features]
y = df['Target']

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
model.fit(X.iloc[:-1], y.iloc[:-1])

# Predict for today
current_features = X.tail(1)
prediction = model.predict(current_features)[0]

# Show Prediction
st.subheader("AI Prediction for Tomorrow")
if prediction == 1:
    st.success("✅ BULLISH: The AI predicts the market will go UP.")
else:
    st.error("🔻 BEARISH: The AI predicts the market will go DOWN.")

# candle graph for last 60 days
st.subheader("Recent Market Trend")
chart_df = df.tail(60) # Last 60 days
fig = go.Figure(data=[go.Candlestick(
    x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], 
    low=chart_df['Low'], close=chart_df['Close'], name='Nifty 50'
)])
fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
# determining feature importance
st.subheader("Why did the AI make this choice?")
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)

# Create a horizontal bar chart
fig_imp = go.Figure(go.Bar(
    x=feat_importances.values,
    y=feat_importances.index,
    orientation='h',
    marker_color='royalblue'
))
fig_imp.update_layout(title="Importance of each Data Point", template='plotly_dark', height=400)
st.plotly_chart(fig_imp, use_container_width=True)

st.info("This chart shows which factor (RSI, Fear, Oil, etc.) influenced the AI's decision the most today.")
