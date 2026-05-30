import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import datetime
from io import BytesIO

# Excel engine fallback
EXCEL_ENGINE = "xlsxwriter"
try:
    import xlsxwriter  # noqa: F401
except Exception:
    EXCEL_ENGINE = "openpyxl"

# Optional Black-Litterman / portfolio optimization
try:
    from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models, expected_returns
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Portfolio Manager")
st.caption("Dashboard cuantitativo de precios, riesgo, momentum, benchmark, drawdown y análisis de cartera.")

st.markdown("""
### Investment Dashboard
Sistema cuantitativo de análisis de carteras basado en:
- Momentum
- Risk Metrics
- Benchmark Comparison
- Portfolio Construction
- Drawdown Analysis
- Attribution
- Forecast heurístico

Datos obtenidos en tiempo real mediante Yahoo Finance.
""")

# -----------------------------
# Universe
# -----------------------------
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "NFLX",
    "JPM", "BAC", "XOM", "CVX", "UNH", "KO", "PEP", "COST",
    "SPY", "QQQ", "IWM", "VEA", "EEM", "GLD", "TLT", "AGG"
]

BENCHMARK_MAP = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Russell 2000 (IWM)": "IWM",
    "Developed Markets (VEA)": "VEA",
    "Emerging Markets (EEM)": "EEM",
    "US Bonds (AGG)": "AGG"
}

TICKER_TO_BENCHMARK_NAME = {v: k for k, v in BENCHMARK_MAP.items()}

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AMZN": "Consumer Discretionary",
    "META": "Technology", "GOOGL": "Technology", "TSLA": "Consumer Discretionary", "NFLX": "Communication",
    "JPM": "Financials", "BAC": "Financials", "XOM": "Energy", "CVX": "Energy",
    "UNH": "Healthcare", "KO": "Consumer Staples", "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "VEA": "Index", "EEM": "Index", "GLD": "Commodity",
    "TLT": "Bond", "AGG": "Bond"
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuración")

tickers = st.sidebar.multiselect(
    "Selecciona activos",
    UNIVERSE,
    default=["AAPL", "MSFT", "NVDA", "AMZN"]
)

benchmark_name = st.sidebar.selectbox(
    "Benchmark principal",
    list(BENCHMARK_MAP.keys()),
    index=0
)
benchmark = BENCHMARK_MAP[benchmark_name]

default_extra_benchmark_names = ["Nasdaq 100 (QQQ)", "Russell 2000 (IWM)"] if benchmark == "SPY" else []
extra_benchmark_names = st.sidebar.multiselect(
    "Benchmarks adicionales",
    [name for name in BENCHMARK_MAP.keys() if name != benchmark_name],
    default=default_extra_benchmark_names
)
extra_benchmarks = [BENCHMARK_MAP[name] for name in extra_benchmark_names]

start_date = st.sidebar.date_input(
    "Fecha inicial",
    value=datetime.date(2020, 1, 1)
)

lookback = st.sidebar.slider(
    "Ventana Momentum (días bursátiles)",
    min_value=20,
    max_value=252,
    value=126
)

rebalance_frequency = st.sidebar.selectbox(
    "Frecuencia de rebalanceo",
    ["Mensual", "Trimestral", "Semestral", "Anual"],
    index=0
)

forecast_horizon = st.sidebar.selectbox(
    "Horizonte de forecast",
    ["1 mes", "3 meses", "6 meses"],
    index=0
)

if len(tickers) == 0:
    st.warning("Selecciona al menos un activo.")
    st.stop()

# -----------------------------
# Data download
# -----------------------------
all_tickers = list(dict.fromkeys(tickers + [benchmark] + extra_benchmarks))
raw = yf.download(all_tickers, start=start_date, progress=False)

if raw.empty:
    st.error("No se han podido descargar datos. Revisa los tickers o la conexión.")
    st.stop()

prices = raw["Close"].copy()

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

prices = prices.loc[:, [c for c in prices.columns if c in all_tickers]]

if benchmark not in prices.columns:
    st.error(f"No se ha podido cargar el benchmark {benchmark}.")
    st.stop()

if len(prices) < 2:
    st.error("No hay suficientes datos para calcular retornos. Selecciona una fecha inicial más antigua.")
    st.stop()

returns = prices.pct_change().dropna()

if returns.empty:
    st.error("No hay suficientes datos para calcular retornos. Selecciona una fecha inicial más antigua.")
    st.stop()

# Separate portfolio assets from benchmarks
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
benchmark_sharpe = benchmark_ann_return / benchmark_ann_vol if benchmark_ann_vol != 0 else np.nan

best_asset = ((asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1).sort_values(ascending=False).index[0]

# Momentum
if len(asset_prices) > lookback:
    momentum = (asset_prices.iloc[-1] / asset_prices.iloc[-lookback] - 1).sort_values(ascending=False)
else:
    momentum = ((asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1).sort_values(ascending=False)

momentum_weights = momentum.clip(lower=0)
if momentum_weights.sum() > 0:
    momentum_weights = momentum_weights / momentum_weights.sum()
else:
    momentum_weights = pd.Series([1 / len(momentum.index)] * len(momentum.index), index=momentum.index)

mom_portfolio_returns = asset_returns[momentum_weights.index] @ momentum_weights.values
mom_ann_return = mom_portfolio_returns.mean() * 252
mom_ann_vol = mom_portfolio_returns.std() * np.sqrt(252)
mom_sharpe = mom_ann_return / mom_ann_vol if mom_ann_vol != 0 else np.nan

# -----------------------------
# Drawdown
# -----------------------------
portfolio_cum = (1 + portfolio_returns).cumprod()
benchmark_cum = (1 + benchmark_returns).cumprod()
mom_cum = (1 + mom_portfolio_returns).cumprod()

portfolio_drawdown = portfolio_cum / portfolio_cum.cummax() - 1
benchmark_drawdown = benchmark_cum / benchmark_cum.cummax() - 1
mom_drawdown = mom_cum / mom_cum.cummax() - 1

max_dd_portfolio = portfolio_drawdown.min()
max_dd_benchmark = benchmark_drawdown.min()
max_dd_momentum = mom_drawdown.min()

# -----------------------------
# Attribution
# -----------------------------
asset_total_returns = (asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1
equal_contribution = pd.Series(equal_weights, index=asset_cols) * asset_total_returns
equal_contribution_pct = equal_contribution / equal_contribution.sum() if equal_contribution.sum() != 0 else equal_contribution

mom_weights_aligned = momentum_weights.reindex(asset_cols).fillna(0)
momentum_contribution = mom_weights_aligned * asset_total_returns
momentum_contribution_pct = momentum_contribution / momentum_contribution.sum() if momentum_contribution.sum() != 0 else momentum_contribution

# -----------------------------
# Forecast heurístico
# -----------------------------
horizon_map = {"1 mes": 21, "3 meses": 63, "6 meses": 126}
forecast_days = horizon_map[forecast_horizon]

forecast_returns = asset_returns.tail(min(lookback, len(asset_returns))).mean() * forecast_days
forecast_df = forecast_returns.sort_values(ascending=False).rename("Expected Return").to_frame()

# -----------------------------
# Factor exposure (simple, manual)
# -----------------------------
factor_df = pd.DataFrame({
    "Activo": asset_cols,
    "Sector": [SECTOR_MAP.get(t, "Other") for t in asset_cols],
    "Peso Equal": equal_weights,
    "Peso Momentum": mom_weights_aligned.reindex(asset_cols).values
})

# -----------------------------
# Black-Litterman (optional)
# -----------------------------
bl_available = PYPFOPT_AVAILABLE and len(asset_cols) >= 2

if bl_available:
    try:
        mu = expected_returns.mean_historical_return(asset_prices)
        S = risk_models.sample_cov(asset_prices)

        top_assets = list(momentum.head(min(2, len(momentum))).index)
        bottom_assets = list(momentum.tail(min(2, len(momentum))).index)

        Q = []
        P = []

        for t in top_assets:
            view = np.zeros(len(asset_cols))
            if t in asset_cols:
                view[asset_cols.index(t)] = 1.0
                P.append(view)
                Q.append(float(momentum.loc[t]))

        for t in bottom_assets:
            view = np.zeros(len(asset_cols))
            if t in asset_cols:
                view[asset_cols.index(t)] = 1.0
                P.append(view)
                Q.append(float(momentum.loc[t]))

        bl_weights = None
        if len(P) > 0:
            bl = BlackLittermanModel(S, pi=mu, P=np.array(P), Q=np.array(Q))
            bl_mu = bl.bl_returns()
            ef = EfficientFrontier(bl_mu, S)
            bl_weights = ef.max_sharpe()
            bl_weights = pd.Series(bl_weights).reindex(asset_cols).fillna(0)
    except Exception:
        bl_weights = None
else:
    bl_weights = None

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Activos", len(asset_cols))
col2.metric("Rentabilidad anual", f"{ann_return:.2%}")
col3.metric("Volatilidad anual", f"{ann_vol:.2%}")
col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
col5.metric("Max Drawdown", f"{max_dd_portfolio:.2%}")

col6, col7, col8, col9 = st.columns(4)
col6.metric("Benchmark", benchmark_name)
col7.metric("Rent. Benchmark", f"{benchmark_ann_return:.2%}")
col8.metric("Mejor activo", best_asset)
col9.metric("Momentum Sharpe", f"{mom_sharpe:.2f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Resumen", "Mercado", "Modelo", "Riesgo", "Atribución", "Factores", "Forecast", "Datos"]
)

with tab1:
    st.subheader("Comparación cartera vs benchmark")
    compare = pd.DataFrame({
        "Cartera igual ponderada": portfolio_cum,
        benchmark_name: benchmark_cum
    })

    fig = px.line(compare, x=compare.index, y=compare.columns, title="Evolución acumulada")
    st.plotly_chart(fig, use_container_width=True)

    for b in extra_benchmarks:
        if b in prices.columns:
            extra_cum = (1 + returns[b].dropna()).cumprod()
            st.markdown(f"### Benchmark adicional: {TICKER_TO_BENCHMARK_NAME.get(b, b)}")
            st.line_chart(extra_cum)

    st.markdown("### Métricas del modelo")
    metrics_df = pd.DataFrame({
        "Rentabilidad anual": [ann_return, benchmark_ann_return, mom_ann_return],
        "Volatilidad anual": [ann_vol, benchmark_ann_vol, mom_ann_vol],
        "Sharpe": [sharpe, benchmark_sharpe, mom_sharpe]
    }, index=["Cartera igual ponderada", benchmark_name, "Cartera Momentum"])

    st.dataframe(
        metrics_df.style.format({
            "Rentabilidad anual": "{:.2%}",
            "Volatilidad anual": "{:.2%}",
            "Sharpe": "{:.2f}"
        }),
        use_container_width=True
    )

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
    st.subheader("Construcción de cartera")

    momentum_df = momentum.rename("Momentum").to_frame().reset_index()
    momentum_df.columns = ["Activo", "Momentum"]
    momentum_df["Peso Momentum"] = momentum_weights.reindex(momentum_df["Activo"]).values
    momentum_df["Peso Equal"] = pd.Series(equal_weights, index=asset_cols).reindex(momentum_df["Activo"]).values
    momentum_df = momentum_df.sort_values("Momentum", ascending=False)

    st.markdown("### Ranking Momentum")
    fig_mom = px.bar(momentum_df, x="Activo", y="Momentum", title="Ranking Momentum")
    st.plotly_chart(fig_mom, use_container_width=True)

    st.markdown("### Pesos de cartera")
    st.dataframe(
        momentum_df.style.format({
            "Momentum": "{:.2%}",
            "Peso Momentum": "{:.2%}",
            "Peso Equal": "{:.2%}"
        }),
        use_container_width=True
    )

    st.markdown("### Comparación Equal Weight vs Momentum")
    weights_compare = pd.DataFrame({
        "Equal Weight": pd.Series(equal_weights, index=asset_cols),
        "Momentum Weight": momentum_weights.reindex(asset_cols).fillna(0)
    })
    st.dataframe(weights_compare.style.format("{:.2%}"), use_container_width=True)

    if bl_weights is not None:
        st.markdown("### Black-Litterman (si está disponible)")
        bl_df = pd.DataFrame({
            "Black-Litterman Weight": bl_weights.reindex(asset_cols).fillna(0)
        })
        st.dataframe(bl_df.style.format("{:.2%}"), use_container_width=True)
    else:
        st.info("Black-Litterman no está disponible con la configuración actual o no se pudo calcular.")

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

    st.subheader("Drawdown")
    drawdown_df = pd.DataFrame({
        "Cartera igual ponderada": portfolio_drawdown,
        benchmark_name: benchmark_drawdown,
        "Cartera Momentum": mom_drawdown
    })

    fig_dd = px.line(
        drawdown_df,
        x=drawdown_df.index,
        y=drawdown_df.columns,
        title="Drawdown histórico"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    dd_summary = pd.DataFrame({
        "Max Drawdown": [
            max_dd_portfolio,
            max_dd_benchmark,
            max_dd_momentum
        ]
    }, index=[
        "Cartera igual ponderada",
        benchmark_name,
        "Cartera Momentum"
    ])

    st.dataframe(
        dd_summary.style.format({"Max Drawdown": "{:.2%}"}),
        use_container_width=True
    )

    st.subheader("VaR simple")
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)

    var_df = pd.DataFrame({
        "VaR 95%": [var_95],
        "VaR 99%": [var_99]
    }, index=["Cartera igual ponderada"])

    st.dataframe(var_df.style.format("{:.2%}"), use_container_width=True)

with tab5:
    st.subheader("Atribución de rentabilidad")
    attrib_df = pd.DataFrame({
        "Activo": asset_cols,
        "Contribución Equal": equal_contribution.reindex(asset_cols).values,
        "Contribución Momentum": momentum_contribution.reindex(asset_cols).values,
        "Peso Equal": pd.Series(equal_weights, index=asset_cols).values,
        "Peso Momentum": momentum_weights.reindex(asset_cols).fillna(0).values
    })

    attrib_fig = px.bar(
        attrib_df.sort_values("Contribución Equal", ascending=False),
        x="Activo",
        y="Contribución Equal",
        title="Atribución de rentabilidad (Equal Weight)"
    )
    st.plotly_chart(attrib_fig, use_container_width=True)

    st.dataframe(
        attrib_df.style.format({
            "Contribución Equal": "{:.2%}",
            "Contribución Momentum": "{:.2%}",
            "Peso Equal": "{:.2%}",
            "Peso Momentum": "{:.2%}"
        }),
        use_container_width=True
    )

    st.markdown("### Atribución porcentual")
    attrib_pct_df = pd.DataFrame({
        "Contribución % Equal": equal_contribution_pct.reindex(asset_cols).values,
        "Contribución % Momentum": momentum_contribution_pct.reindex(asset_cols).values
    }, index=asset_cols)

    st.dataframe(attrib_pct_df.style.format("{:.2%}"), use_container_width=True)

with tab6:
    st.subheader("Exposición por factor / sector")
    sector_exposure = factor_df.groupby("Sector")[["Peso Equal", "Peso Momentum"]].sum().reset_index()

    fig_sector = px.bar(
        sector_exposure.melt(id_vars="Sector", var_name="Estrategia", value_name="Peso"),
        x="Sector",
        y="Peso",
        color="Estrategia",
        barmode="group",
        title="Exposición sectorial"
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    st.dataframe(
        factor_df.style.format({
            "Peso Equal": "{:.2%}",
            "Peso Momentum": "{:.2%}"
        }),
        use_container_width=True
    )

with tab7:
    st.subheader("Forecast heurístico")
    st.caption("Estimación simple basada en retornos históricos recientes. No es una predicción garantizada.")

    forecast_table = forecast_df.reset_index()
    forecast_table.columns = ["Activo", "Expected Return"]

    forecast_fig = px.bar(
        forecast_table,
        x="Activo",
        y="Expected Return",
        title=f"Expected Return ({forecast_horizon})"
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    st.dataframe(
        forecast_table.style.format({
            "Expected Return": "{:.2%}"
        }),
        use_container_width=True
    )

with tab8:
    st.subheader("Últimas filas de datos")
    st.dataframe(prices.tail(30), use_container_width=True)

    st.subheader("Últimos retornos")
    st.dataframe(returns.tail(30), use_container_width=True)

    st.subheader("Descargar informe Excel")

    def build_excel() -> bytes:
        output = BytesIO()
        with pd.ExcelWriter(output, engine=EXCEL_ENGINE) as writer:
            summary = pd.DataFrame({
                "Metric": [
                    "Activos", "Rentabilidad anual", "Volatilidad anual", "Sharpe Ratio",
                    "Max Drawdown", "Benchmark", "Rent. Benchmark", "Momentum Sharpe"
                ],
                "Value": [
                    len(asset_cols), ann_return, ann_vol, sharpe,
                    max_dd_portfolio, benchmark_name, benchmark_ann_return, mom_sharpe
                ]
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)
            prices.to_excel(writer, sheet_name="Prices")
            returns.to_excel(writer, sheet_name="Returns")
            momentum_df.to_excel(writer, sheet_name="Momentum", index=False)
            risk_df.to_excel(writer, sheet_name="Risk")
            dd_summary.to_excel(writer, sheet_name="Drawdown")
            attrib_df.to_excel(writer, sheet_name="Attribution", index=False)
            factor_df.to_excel(writer, sheet_name="Factors", index=False)
            forecast_df.to_excel(writer, sheet_name="Forecast")

        return output.getvalue()

    excel_data = build_excel()

    st.download_button(
        label="Download Portfolio Report (.xlsx)",
        data=excel_data,
        file_name="portfolio_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.info(f"Frecuencia de rebalanceo seleccionada: {rebalance_frequency}")
