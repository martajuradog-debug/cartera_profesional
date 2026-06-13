"Peso Equal": equal_weights,
"Peso Momentum": mom_weights_aligned.reindex(asset_cols).values
})
    sector_metrics = pd.DataFrame()

# -----------------------------
    # -----------------------------
# Sector analysis
# -----------------------------
sector_map_series = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in asset_cols})

sector_returns = pd.DataFrame(index=asset_returns.index)

for sector in sorted(sector_map_series.unique()):
sector_assets = sector_map_series[sector_map_series == sector].index.tolist()
    if sector_assets:

    if len(sector_assets) > 0:
sector_returns[sector] = asset_returns[sector_assets].mean(axis=1)

if not sector_returns.empty:
sector_cum = (1 + sector_returns).cumprod()

sector_metrics = pd.DataFrame({
"Rentabilidad anual": sector_returns.mean() * 252,
        "Volatilidad anual": sector_returns.std() * np.sqrt(252),
        "Volatilidad anual": sector_returns.std() * np.sqrt(252)
})
    sector_metrics["Sharpe"] = np.where(
        sector_metrics["Volatilidad anual"] != 0,
        sector_metrics["Rentabilidad anual"] / sector_metrics["Volatilidad anual"],
        np.nan
    )

    sector_metrics["Sharpe"] = sector_metrics["Rentabilidad anual"] / sector_metrics["Volatilidad anual"]
else:
sector_cum = pd.DataFrame()
sector_metrics = pd.DataFrame()
