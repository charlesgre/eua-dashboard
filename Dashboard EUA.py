import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Gas Dashboard", layout="wide")
st.title("\U0001F4CA EUA Analytics Dashboard")

file_path = "Gas storages.xlsx"

tabs = st.tabs(["\U0001F4E6 Stocks", "\U0001F4B0 Prix (EUA/TTF)", "\U0001F4C8 Stratégies RSI / StochRSI"])

# === 1. Onglet STOCKS ===
with tabs[0]:
    st.header("Stockages de gaz - par pays")

    start_year = 2020
    end_year = 2025

    columns_mapping = [
        'Date', 'Europe Gas Storage (TWh)', 'US DOE estimated storage',
        'UK Gas Storage (TWh)', 'Germany Gas Storage (TWh)', 'Netherlands Gas Storage (TWh)'
    ]

    colors = {
        2020: 'blue', 2021: 'orange', 2022: 'purple',
        2023: 'yellow', 2024: 'green', 2025: 'red'
    }


    if st.button("🔄 Forcer la mise à jour des données"):
        st.cache_data.clear()
        st.experimental_rerun()

    @st.cache_data
    def load_stock_data():
        df = pd.read_excel(file_path, sheet_name="Stocks", header=None, skiprows=6)
        df = df.iloc[:, :6]
        df.columns = columns_mapping
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.dropna()

    df_stock = load_stock_data()
    country_map = {
        'Europe Gas Storage (TWh)': 'Europe',
        'US DOE estimated storage': 'US',
        'UK Gas Storage (TWh)': 'UK',
        'Germany Gas Storage (TWh)': 'Germany',
        'Netherlands Gas Storage (TWh)': 'Netherlands'
    }
    selected_country = st.selectbox("Choisir un pays :", list(country_map.keys()))

    series = df_stock[['Date', selected_country]].dropna()
    series['Value'] = pd.to_numeric(series[selected_country], errors='coerce')
    series = series[series['Date'].dt.year >= start_year].dropna()

    range_data = series[series['Date'].dt.year <= 2024].copy()
    range_data['DOY'] = range_data['Date'].dt.dayofyear

    all_years = []
    for year in range(2020, 2025):
        yearly = range_data[range_data['Date'].dt.year == year].copy()
        yearly = yearly.groupby('DOY')['Value'].mean().reindex(np.arange(1, 367)).interpolate()
        all_years.append(yearly.values)

    all_years_array = np.vstack(all_years)
    min_vals = np.nanmin(all_years_array, axis=0)
    max_vals = np.nanmax(all_years_array, axis=0)
    mean_vals = np.nanmean(all_years_array, axis=0)

    full_doy = np.arange(1, 367)
    mois = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mois_jours = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=full_doy,
        y=min_vals,
        mode='lines',
        line=dict(color='lightgray'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=full_doy,
        y=max_vals,
        mode='lines',
        fill='tonexty',
        line=dict(color='lightgray'),
        name='Min-Max 2020–2024',
        fillcolor='rgba(128,128,128,0.3)'
    ))
    fig.add_trace(go.Scatter(
        x=full_doy,
        y=mean_vals,
        mode='lines',
        name='Moyenne 2020–2024',
        line=dict(color='black', dash='dash')
    ))

    for year in range(start_year, end_year + 1):
        yearly = series[series['Date'].dt.year == year].copy()
        if not yearly.empty:
            yearly['DOY'] = yearly['Date'].dt.dayofyear
            fig.add_trace(go.Scatter(
                x=yearly['DOY'],
                y=yearly['Value'],
                mode='lines',
                name=str(year),
                line=dict(width=2 if year >= 2023 else 1),
                opacity=1.0 if year >= 2023 else 0.4
            ))

    fig.update_layout(
        title=f"{country_map[selected_country]} - Stockage de gaz (TWh)",
        xaxis=dict(
            title="Mois",
            tickmode='array',
            tickvals=mois_jours,
            ticktext=mois
        ),
        yaxis_title="TWh",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# === 2. Onglet PRIX ===
with tabs[1]:
    st.header("Prix du marché - EUA & TTF")
    df_prices = pd.read_excel(file_path, sheet_name="Prices", skiprows=6)
    df_prices.columns = ['Date', 'EUA', 'TTF']
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], errors='coerce')
    df_prices = df_prices.dropna(subset=['Date'])
    df_prices['Year'] = df_prices['Date'].dt.year
    df_prices = df_prices[df_prices['Year'].between(2021, 2025)]
    df_prices['DayOfYear'] = df_prices['Date'].dt.dayofyear

    def seasonal_price_plotly(df, col, ylabel, exclude=None):
        fig = go.Figure()
        for year in sorted(df['Year'].unique()):
            if exclude and year in exclude:
                continue
            data = df[df['Year'] == year]
            fig.add_trace(go.Scatter(
                x=data['DayOfYear'],
                y=data[col],
                mode='lines',
                name=str(year),
                opacity=1.0 if year >= 2023 else 0.3
            ))
        ticks = [pd.Timestamp(2022, m, 1).dayofyear for m in range(1, 13)]
        labels = [calendar.month_abbr[m] for m in range(1, 13)]
        fig.update_layout(
            title=f"{col} - Seasonal Daily Pattern",
            xaxis=dict(title="Month", tickmode='array', tickvals=ticks, ticktext=labels),
            yaxis_title=ylabel,
            legend_title="Année",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    seasonal_price_plotly(df_prices, 'EUA', "Price (€/tCO2)")
    seasonal_price_plotly(df_prices, 'TTF', "Price (€/MWh)", exclude=[2021, 2022])

# === 3. STRATÉGIES RSI ===
with tabs[2]:
    st.header("Stratégies techniques sur le marché EUA")

    df = pd.read_excel(file_path, sheet_name="Prices", skiprows=6, usecols="A,B")
    df.columns = ['Date', 'EUA']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['EUA'] = pd.to_numeric(df['EUA'], errors='coerce')
    df = df.dropna().set_index('Date')

    delta = df['EUA'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi = df['RSI']
    stochrsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    df['StochRSI'] = stochrsi

    def run_strategy(df, long_cond, short_cond):
        trades = []
        position = None
        for date, row in df.iterrows():
            if not (2021 <= date.year <= 2025 and 4 <= date.month <= 9):
                continue
            price = row['EUA']
            if position is None:
                if long_cond(row):
                    position = {'type': 'long', 'entry_price': price}
                elif short_cond(row):
                    position = {'type': 'short', 'entry_price': price}
            elif position:
                entry = position['entry_price']
                if position['type'] == 'long':
                    if price >= entry + 2:
                        trades.append({'date': date, 'pnl': 2, 'type': 'long'})
                        position = None
                    elif price <= entry - 1:
                        trades.append({'date': date, 'pnl': -1, 'type': 'long'})
                        position = None
                elif position['type'] == 'short':
                    if price <= entry - 2:
                        trades.append({'date': date, 'pnl': 2, 'type': 'short'})
                        position = None
                    elif price >= entry + 1:
                        trades.append({'date': date, 'pnl': -1, 'type': 'short'})
                        position = None
        tdf = pd.DataFrame(trades).set_index('date').sort_index()
        tdf['PnL €'] = tdf['pnl'] * 100000
        tdf['Cumulative PnL'] = tdf['PnL €'].cumsum()
        tdf['Year'] = tdf.index.year
        return tdf

    trades_rsi = run_strategy(df, lambda r: r['RSI'] < 30, lambda r: r['RSI'] > 70)
    trades_stoch = run_strategy(df, lambda r: r['StochRSI'] < 0.2, lambda r: r['StochRSI'] > 0.8)

    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("Stochastic RSI (14)")
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df['StochRSI'], mode='lines', name='StochRSI', line_color='orange'))
    fig_stoch.add_hline(y=0.8, line_dash="dash", line_color="red")
    fig_stoch.add_hline(y=0.2, line_dash="dash", line_color="green")
    st.plotly_chart(fig_stoch, use_container_width=True)

    st.subheader("Cumulative PnL des stratégies")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=trades_rsi.index, y=trades_rsi['Cumulative PnL'], name='RSI Strategy'))
    fig_pnl.add_trace(go.Scatter(x=trades_stoch.index, y=trades_stoch['Cumulative PnL'], name='StochRSI Strategy'))
    fig_pnl.update_layout(yaxis_title="Cumulative PnL (€)")
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.subheader("PnL Annuel par stratégie")
    annual_rsi = trades_rsi.groupby('Year')['PnL €'].sum()
    annual_stoch = trades_stoch.groupby('Year')['PnL €'].sum()

    x = np.arange(len(annual_rsi.index))
    bar_width = 0.35
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=annual_rsi.index - 0.2, y=annual_rsi.values, name='RSI Strategy'))
    fig_bar.add_trace(go.Bar(x=annual_stoch.index + 0.2, y=annual_stoch.values, name='StochRSI Strategy'))
    fig_bar.update_layout(barmode='group', xaxis_title='Année', yaxis_title='PnL (€)')
    st.plotly_chart(fig_bar, use_container_width=True)
