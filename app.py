import streamlit as st
import yfinance as yf
import plotly.express as px

st.title("Mi cartera automática")

# Acciones
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"]

# Descargar datos
df = yf.download(tickers, start="2020-01-01")["Close"]

# Crear gráfico
fig = px.line(df, title="Precios históricos")

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)
