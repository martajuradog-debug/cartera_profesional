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
st.caption("Dashboard cuantitativo de precios, riesgo, momentum y benchmark.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuración")

tickers = st.sidebar.multiselect(
    "Selecciona activos",
    ["AAPL", "MSFT", "NVDA", "AMZN", "SPY", "TSLA"],
    default=["AAPL", "MSFT", "NVDA", "AMZN"]
)

benchmark = st.sidebar.selectbox(
    "Benchmark",
    ["SPY"],
    index=0
)

start_date = st.sidebar.date_input("Fecha inicial")
lookback = st.sidebar.slider(
    "Ventana Momentum (días bursátiles)",
    min_value=20,
    max_value=252,
    value=126
)

if len(tickers) == 0:
    st.warning("Selecciona al menos un activo.")
    st.stop()

# -----------------------------
# Data download
# -----------------------------
all_tickers = list(dict.fromkeys(tickers + [benchmark]))
raw = yf.download(all_tickers, start=start_date, progress=False)

if raw.empty:
    st.error("No se han podido descargar datos. Revisa los tickers o la conexión.")
    st.stop()

prices = raw["Close"].copy()

# If only one ticker returns a Series
if isinstance(prices, pd.Series):
    prices = prices.to_frame()

# Keep selected tickers + benchmark only, in case yfinance returns extra columns
prices = prices.loc[:, [c for c in prices.columns if c in all_tickers]]

if benchmark not in prices.columns:
    st.error(f"No se ha podido cargar el benchmark {benchmark}.")
    st.stop()

returns = prices.pct_change().dropna()

if returns.empty:
    st.error("No hay suficientes datos para calcular retornos.")
    st.stop()

# Separate portfolio assets from benchmark
asset_cols = [c for c in tickers if c in prices.columns]
benchmark_col = benchmark

if len(asset_cols) == 0:
    st.error("No hay activos válidos seleccionados.")
    st.stop()

asset_prices = prices[asset_cols]
asset_returns = returns[asset_cols]
benchmark_returns = returns[benchmark_col]

# -----------------------------
# Portfolio metrics
# -----------------------------
equal_weights = np.array([1 / len(asset_cols)] * len(asset_cols))
portfolio_returns = asset_returns @ equal_weights

ann_return = portfolio_returns.mean() * 252
ann_vol = portfolio_returns.std() * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

benchmark_ann_return = benchmark_returns.mean() * 252
benchmark_ann_vol = benchmark_returns.std() * np.sqrt(252)

best_asset = ((asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1).sort_values(ascending=False).index[0]
top_asset_return = ((asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1).max()

# Momentum
if len(asset_prices) > lookback:
    momentum = (asset_prices.iloc[-1] / asset_prices.iloc[-lookback] - 1).sort_values(ascending=False)
else:
    momentum = ((asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1).sort_values(ascending=False)

momentum_weights = momentum.clip(lower=0)
if momentum_weights.sum() > 0:
    momentum_weights = momentum_weights / momentum_weights.sum()
else:
    momentum_weights = pd.Series(
        [1 / len(momentum.index)] * len(momentum.index),
        index=momentum.index
    )

mom_portfolio_returns = asset_returns[momentum_weights.index] @ momentum_weights.values
mom_ann_return = mom_portfolio_returns.mean() * 252
mom_ann_vol = mom_portfolio_returns.std() * np.sqrt(252)
mom_sharpe = mom_ann_return / mom_ann_vol if mom_ann_vol != 0 else np.nan

# -----------------------------
# Top KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Activos", len(asset_cols))
col2.metric("Rentabilidad anual", f"{ann_return:.2%}")
col3.metric("Volatilidad anual", f"{ann_vol:.2%}")
col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Benchmark", benchmark)
col6.metric("Rent. Benchmark", f"{benchmark_ann_return:.2%}")
col7.metric("Mejor activo", best_asset)
col8.metric("Momentum Sharpe", f"{mom_sharpe:.2f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Resumen", "Mercado", "Modelo", "Riesgo", "Datos"]
)

with tab1:
    st.subheader("Comparación cartera vs benchmark")

    portfolio_cum = (1 + portfolio_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    compare = pd.DataFrame({
        "Cartera igual ponderada": portfolio_cum,
        benchmark: benchmark_cum
    })

    fig = px.line(
        compare,
        x=compare.index,
        y=compare.columns,
        title="Evolución acumulada"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Métricas del modelo")
    metrics_df = pd.DataFrame({
        "Rentabilidad anual": [ann_return, benchmark_ann_return, mom_ann_return],
        "Volatilidad anual": [ann_vol, benchmark_ann_vol, mom_ann_vol],
        "Sharpe": [sharpe, benchmark_ann_return / benchmark_ann_vol if benchmark_ann_vol != 0 else np.nan, mom_sharpe]
    }, index=["Cartera igual ponderada", benchmark, "Cartera Momentum"])

    st.dataframe(metrics_df.style.format("{:.2%}", subset=["Rentabilidad anual", "Volatilidad anual"]).format("{:.2f}", subset=["Sharpe"]), use_container_width=True)

with tab2:
    st.subheader("Precios históricos")

    fig_prices = px.line(
        asset_prices,
        x=asset_prices.index,
        y=asset_prices.columns,
        title="Precios históricos de los activos"
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    st.subheader("Rentabilidad acumulada")
    asset_cum = (1 + asset_returns).cumprod()

    fig_cum = px.line(
        asset_cum,
        x=asset_cum.index,
        y=asset_cum.columns,
        title="Rentabilidad acumulada por activo"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

with tab3:
    st.subheader("Señal Momentum")
    st.write(f"Ventana usada: **{lookback}** días bursátiles")

    momentum_df = momentum.rename("Momentum").to_frame()
    momentum_df["Peso"] = pd.Series(momentum_weights)
    momentum_df = momentum_df.sort_values("Momentum", ascending=False)

    fig_mom = px.bar(
        momentum_df.reset_index(),
        x="index",
        y="Momentum",
        title="Ranking Momentum",
        labels={"index": "Activo"}
    )
    st.plotly_chart(fig_mom, use_container_width=True)

    st.markdown("### Pesos de la cartera Momentum")
    st.dataframe(momentum_df.style.format({"Momentum": "{:.2%}", "Peso": "{:.2%}"}), use_container_width=True)

    st.markdown("### Pesos iguales vs Momentum")
    weights_compare = pd.DataFrame({
        "Equal Weight": pd.Series(equal_weights, index=asset_cols),
        "Momentum Weight": momentum_weights.reindex(asset_cols).fillna(0)
    })
    st.dataframe(weights_compare.style.format("{:.2%}"), use_container_width=True)

with tab4:
    st.subheader("Correlación entre activos")
    corr = asset_returns.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Matriz de correlación"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Riesgo histórico")
    risk_df = pd.DataFrame({
        "Retorno medio diario": asset_returns.mean(),
        "Volatilidad diaria": asset_returns.std(),
        "Retorno anual": asset_returns.mean() * 252,
        "Volatilidad anual": asset_returns.std() * np.sqrt(252),
    })

    st.dataframe(
        risk_df.style.format({
            "Retorno medio diario": "{:.4f}",
            "Volatilidad diaria": "{:.4f}",
            "Retorno anual": "{:.2%}",
            "Volatilidad anual": "{:.2%}",
        }),
        use_container_width=True
    )

with tab5:
    st.subheader("Últimas filas de datos")
    st.dataframe(prices.tail(30), use_container_width=True)

    st.subheader("Últimos retornos")
    st.dataframe(returns.tail(30), use_container_width=True)
