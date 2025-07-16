import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
import statsmodels.api as sm
import matplotlib.ticker as ticker
from datetime import datetime

st.set_page_config(page_title="Gas Dashboard", layout="wide")
st.title("📊 EUA Analytics Dashboard")

file_path = r"\\gvaps1\USR6\CHGE\desktop\Gas storages.xlsx"

tabs = st.tabs(["📦 Stocks", "💰 Prix (EUA/TTF)", "🔁 Corrélation EUA vs Stocks", "📈 Stratégies RSI / StochRSI"])

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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(full_doy, min_vals, max_vals, color='grey', alpha=0.3, label='Min-Max 2020–2024')
    ax.plot(full_doy, mean_vals, color='black', linestyle='--', linewidth=2, label='Moyenne 2020–2024')

    for year in range(start_year, end_year + 1):
        yearly = series[series['Date'].dt.year == year].copy()
        if not yearly.empty:
            yearly['DOY'] = yearly['Date'].dt.dayofyear
            alpha_val = 1.0 if year >= 2023 else 0.4
            lw = 2 if year >= 2023 else 1
            ax.plot(yearly['DOY'], yearly['Value'], label=str(year),
                    color=colors.get(year, 'gray'), alpha=alpha_val, linewidth=lw)
    
    mois = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mois_jours = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
    ax.set_xticks(mois_jours)
    ax.set_xticklabels(mois)
    ax.set_title(f"{country_map[selected_country]} - Stockage de gaz (TWh)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("TWh")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

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

    year_styles = {
        2021: {'color': 'blue', 'alpha': 0.3},
        2022: {'color': 'orange', 'alpha': 0.3},
        2023: {'color': 'yellow', 'alpha': 1.0},
        2024: {'color': 'green', 'alpha': 1.0},
        2025: {'color': 'red', 'alpha': 1.0}
    }

    def seasonal_price_plot(df, col, ylabel, exclude=None):
        fig, ax = plt.subplots(figsize=(12, 5))
        for year in sorted(df['Year'].unique()):
            if exclude and year in exclude:
                continue
            data = df[df['Year'] == year]
            style = year_styles.get(year, {'color': 'gray', 'alpha': 0.5})
            ax.plot(data['DayOfYear'], data[col], label=str(year), color=style['color'], alpha=style['alpha'], linewidth=2)
        ticks = [pd.Timestamp(2022, m, 1).dayofyear for m in range(1, 13)]
        labels = [calendar.month_abbr[m] for m in range(1, 13)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_title(f"{col} - Seasonal Daily Pattern")
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    seasonal_price_plot(df_prices, 'EUA', "Price (€/tCO2)")
    seasonal_price_plot(df_prices, 'TTF', "Price (€/MWh)", exclude=[2021, 2022])

# === 3. Onglet CORRELATION ===
with tabs[2]:
    st.header("Corrélation EUA vs Stock Gaz")
    df_corr = pd.read_excel(file_path, sheet_name="Correl EUA vs stocks", header=6)
    df_corr.columns = ['Date', 'EUA_Price', 'Gas_Storage_TWh', 'Ignore']
    df_corr = df_corr[['Date', 'EUA_Price', 'Gas_Storage_TWh']]
    df_corr['Date'] = pd.to_datetime(df_corr['Date'], errors='coerce')
    df_corr = df_corr.dropna()
    df_corr.set_index('Date', inplace=True)
    df_corr['EUA_smooth'] = df_corr['EUA_Price'].rolling(7).mean()
    df_corr['Gas_smooth'] = df_corr['Gas_Storage_TWh'].rolling(7).mean()
    df_corr = df_corr.dropna()

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_corr.index, df_corr['Gas_smooth'], label="Stock Gaz (TWh)", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(df_corr.index, df_corr['EUA_smooth'], label="Prix EUA", color="green")
    ax1.set_ylabel("Stock (TWh)", color="blue")
    ax2.set_ylabel("Prix EUA (€/tCO2)", color="green")
    ax1.set_xlabel("Date")
    fig1.suptitle("Corrélation Stock Gaz vs Prix EUA - Moyenne glissante 7 jours")
    fig1.tight_layout()
    st.pyplot(fig1)

    lags = range(-60, 61)
    corrs = [df_corr['EUA_smooth'].corr(df_corr['Gas_smooth'].shift(lag)) for lag in lags]
    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lags, corrs, marker='o')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title("Corrélation croisée entre Stock Gaz et Prix EUA")
    ax.set_xlabel("Décalage (jours)")
    ax.set_ylabel("Corrélation")
    ax.grid(True)
    st.pyplot(fig2)

    max_lag = lags[np.argmax(corrs)]
    st.info(f"🔁 Décalage avec corrélation maximale : {max_lag} jours  | Corrélation : {max(corrs):.3f}")

    # Régression linéaire
    X = df_corr['Gas_smooth']
    y = df_corr['EUA_smooth']
    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()

    fig3, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.4, label='Données')
    ax.plot(X, model.predict(X_), color='red', label='Régression')
    ax.set_title("Régression Linéaire: Prix EUA vs Stock Gaz")
    ax.set_xlabel("Stock Gaz (TWh)")
    ax.set_ylabel("Prix EUA (€/tCO2)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig3)

    st.subheader("Résumé du modèle de régression")
    st.text(model.summary())

# === 4. STRATÉGIES RSI ===
with tabs[3]:
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
    fig_rsi, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df['RSI'], color='blue')
    ax.axhline(70, color='red', linestyle='--')
    ax.axhline(30, color='green', linestyle='--')
    ax.grid(True)
    st.pyplot(fig_rsi)

    st.subheader("Stochastic RSI (14)")
    fig_stoch, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df['StochRSI'], color='orange')
    ax.axhline(0.8, color='red', linestyle='--')
    ax.axhline(0.2, color='green', linestyle='--')
    ax.grid(True)
    st.pyplot(fig_stoch)

    st.subheader("Cumulative PnL des stratégies")
    fig_pnl, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trades_rsi.index, trades_rsi['Cumulative PnL'], label='RSI Strategy', color='green')
    ax.plot(trades_stoch.index, trades_stoch['Cumulative PnL'], label='StochRSI Strategy', color='orange')
    ax.legend()
    ax.set_ylabel("Cumulative PnL (€)")
    ax.grid(True)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f} €"))
    st.pyplot(fig_pnl)

    st.subheader("PnL Annuel par stratégie")
    annual_rsi = trades_rsi.groupby('Year')['PnL €'].sum()
    annual_stoch = trades_stoch.groupby('Year')['PnL €'].sum()

    x = np.arange(len(annual_rsi.index))
    bar_width = 0.35
    fig_bar, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - bar_width/2, annual_rsi.values, width=bar_width, label='RSI Strategy', color='green')
    ax.bar(x + bar_width/2, annual_stoch.values, width=bar_width, label='StochRSI Strategy', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(annual_rsi.index)
    ax.set_ylabel("PnL (€)")
    ax.legend()
    ax.grid(True, axis='y')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f} €"))
    st.pyplot(fig_bar)
