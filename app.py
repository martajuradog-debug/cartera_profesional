import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Portfolio Manager")

st.sidebar.header("Configuración")

tickers = st.sidebar.multiselect(
    "Selecciona activos",
    ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
    default=["AAPL", "MSFT", "NVDA", "AMZN"]
)

if len(tickers) == 0:
    st.stop()

datos = yf.download(tickers, start="2020-01-01")["Close"]
retornos = datos.pct_change().dropna()

col1, col2, col3 = st.columns(3)

col1.metric("Activos", len(tickers))
col2.metric("Rentabilidad anual", f"{retornos.mean().mean() * 252:.2%}")
col3.metric("Volatilidad anual", f"{retornos.std().mean() * np.sqrt(252):.2%}")

tab1, tab2, tab3 = st.tabs(["Mercado", "Rentabilidad", "Datos"])

with tab1:
    fig = px.line(datos, x=datos.index, y=datos.columns, title="Precios históricos")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    acumulado = (1 + retornos).cumprod()
    fig2 = px.line(acumulado, x=acumulado.index, y=acumulado.columns, title="Rentabilidad acumulada")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.dataframe(datos.tail(30))
