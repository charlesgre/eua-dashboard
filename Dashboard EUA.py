import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(page_title="Gas Dashboard", layout="wide")
st.title("\U0001F4CA EUA Analytics Dashboard")

# --- chemins robustes ---
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()

file_path = APP_DIR / "Gas storages.xlsx"   # data gaz
eua_oi_path = APP_DIR / "EUA & OI forward.xlsx"  # OI fichier principal
if not eua_oi_path.exists():
    alt = APP_DIR / "EUA OI & forward.xlsx"     # fallback ancien nom
    if alt.exists():
        eua_oi_path = alt

tabs = st.tabs([
    "\U0001F4E6 Stocks",
    "\U0001F4B0 Prix (EUA/TTF)",
    "\U0001F4C8 Stratégies RSI / StochRSI",
    "\U0001F4C9 EUA Open Interest"
])

# ===================== 1) STOCKS =====================
with tabs[0]:
    st.header("Stockages de gaz - par pays")

    # Vérif fichier
    if not file_path.exists():
        st.error(f"Fichier introuvable : **{file_path.name}**. Place-le dans : {APP_DIR}")
        st.stop()

    start_year = 2020
    end_year = 2025

    columns_mapping = [
        'Date', 'Europe Gas Storage (TWh)', 'US DOE estimated storage',
        'UK Gas Storage (TWh)', 'Germany Gas Storage (TWh)', 'Netherlands Gas Storage (TWh)'
    ]

    country_map = {
        'Europe Gas Storage (TWh)': 'Europe',
        'US DOE estimated storage': 'US',
        'UK Gas Storage (TWh)': 'UK',
        'Germany Gas Storage (TWh)': 'Germany',
        'Netherlands Gas Storage (TWh)': 'Netherlands'
    }

    if st.button("🔄 Forcer la mise à jour des données", key="refresh_stocks"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data(show_spinner=False)
    def load_stock_data(xlsx_path: Path, file_version: float):
        df = pd.read_excel(xlsx_path, sheet_name="Stocks", header=None, skiprows=6)
        df = df.iloc[:, :len(columns_mapping)]
        df.columns = columns_mapping
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for c in columns_mapping[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Date'])
        return df.dropna()

    file_mtime = os.path.getmtime(file_path)
    df_stock = load_stock_data(file_path, file_mtime)

    st.caption(f"Dernière date lue : **{df_stock['Date'].max().date()}**  (mtime: {int(file_mtime)})")

    selected_country = st.selectbox(
        "Choisir un pays :",
        list(country_map.keys()),
        key="country_select"
    )

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
    mois = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    mois_jours = [15,45,75,105,135,165,195,225,255,285,315,345]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_doy, y=min_vals, mode='lines', line=dict(color='lightgray'), showlegend=False))
    fig.add_trace(go.Scatter(x=full_doy, y=max_vals, mode='lines', fill='tonexty',
                             line=dict(color='lightgray'), name='Min-Max 2020–2024',
                             fillcolor='rgba(128,128,128,0.3)'))
    fig.add_trace(go.Scatter(x=full_doy, y=mean_vals, mode='lines', name='Moyenne 2020–2024',
                             line=dict(color='black', dash='dash')))

    for year in range(start_year, end_year + 1):
        yearly = series[series['Date'].dt.year == year].copy()
        if not yearly.empty:
            yearly['DOY'] = yearly['Date'].dt.dayofyear
            fig.add_trace(go.Scatter(
                x=yearly['DOY'], y=yearly['Value'], mode='lines', name=str(year),
                line=dict(width=2 if year >= 2023 else 1), opacity=1.0 if year >= 2023 else 0.4
            ))

    fig.update_layout(
        title=f"{country_map[selected_country]} - Stockage de gaz (TWh)",
        xaxis=dict(title="Mois", tickmode='array', tickvals=mois_jours, ticktext=mois),
        yaxis_title="TWh", legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=50, b=40), height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================== 2) PRIX =====================
with tabs[1]:
    st.header("Prix du marché - EUA & TTF")

    if not file_path.exists():
        st.error(f"Fichier introuvable : **{file_path.name}**.")
        st.stop()

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
                x=data['DayOfYear'], y=data[col], mode='lines', name=str(year),
                opacity=1.0 if year >= 2023 else 0.3
            ))
        ticks = [pd.Timestamp(2022, m, 1).dayofyear for m in range(1, 13)]
        labels = [calendar.month_abbr[m] for m in range(1, 13)]
        fig.update_layout(
            title=f"{col} - Seasonal Daily Pattern",
            xaxis=dict(title="Month", tickmode='array', tickvals=ticks, ticktext=labels),
            yaxis_title=ylabel, legend_title="Année",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    seasonal_price_plotly(df_prices, 'EUA', "Price (€/tCO2)")
    seasonal_price_plotly(df_prices, 'TTF', "Price (€/MWh)", exclude=[2021, 2022])

# ===================== 3) STRATÉGIES RSI =====================
with tabs[2]:
    st.header("Stratégies techniques sur le marché EUA")

    if not file_path.exists():
        st.error(f"Fichier introuvable : **{file_path.name}**.")
        st.stop()

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
            else:
                entry = position['entry_price']
                if position['type'] == 'long':
                    if price >= entry + 2:
                        trades.append({'date': date, 'pnl': 2, 'type': 'long'})
                        position = None
                    elif price <= entry - 1:
                        trades.append({'date': date, 'pnl': -1, 'type': 'long'})
                        position = None
                else:
                    if price <= entry - 2:
                        trades.append({'date': date, 'pnl': 2, 'type': 'short'})
                        position = None
                    elif price >= entry + 1:
                        trades.append({'date': date, 'pnl': -1, 'type': 'short'})
                        position = None
        tdf = pd.DataFrame(trades).set_index('date').sort_index()
        if not tdf.empty:
            tdf['PnL €'] = tdf['pnl'] * 100000
            tdf['Cumulative PnL'] = tdf['PnL €'].cumsum()
            tdf['Year'] = tdf.index.year
        else:
            tdf = pd.DataFrame(columns=['pnl', 'PnL €', 'Cumulative PnL', 'Year'])
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
    fig_pnl.add_trace(go.Scatter(x=trades_rsi.index, y=trades_rsi.get('Cumulative PnL', pd.Series(dtype=float)), name='RSI Strategy'))
    fig_pnl.add_trace(go.Scatter(x=trades_stoch.index, y=trades_stoch.get('Cumulative PnL', pd.Series(dtype=float)), name='StochRSI Strategy'))
    fig_pnl.update_layout(yaxis_title="Cumulative PnL (€)")
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.subheader("PnL Annuel par stratégie")
    annual_rsi = trades_rsi.groupby('Year')['PnL €'].sum() if not trades_rsi.empty else pd.Series(dtype=float)
    annual_stoch = trades_stoch.groupby('Year')['PnL €'].sum() if not trades_stoch.empty else pd.Series(dtype=float)

    fig_bar = go.Figure()
    if not annual_rsi.empty:
        fig_bar.add_trace(go.Bar(x=annual_rsi.index, y=annual_rsi.values, name='RSI Strategy'))
    if not annual_stoch.empty:
        fig_bar.add_trace(go.Bar(x=annual_stoch.index, y=annual_stoch.values, name='StochRSI Strategy'))
    fig_bar.update_layout(barmode='group', xaxis_title='Année', yaxis_title='PnL (€)')
    st.plotly_chart(fig_bar, use_container_width=True)

# ===================== 4) EUA OPEN INTEREST =====================
with tabs[3]:
    st.header("EUA Futures - Open Interest (feuille 2)")

    if not eua_oi_path.exists():
        st.error(f"Fichier introuvable : **{eua_oi_path.name}**. Place-le dans : {APP_DIR}")
        st.stop()

    if st.button("🔄 Recharger l'OI EUA", key="refresh_oi"):
        st.cache_data.clear()
        st.rerun()

    def _pick_sheet2(xlsx_path: Path) -> str:
        xls = pd.ExcelFile(xlsx_path)
        names = xls.sheet_names
        for n in names:
            if n.lower() == "sheet2":
                return n
        if len(names) >= 2:
            return names[1]
        return names[0]

    @st.cache_data(show_spinner=False)
    def load_eua_oi(xlsx_path: Path, file_version: float):
        sheet_name = _pick_sheet2(xlsx_path)
        titles_row = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=1)
        titles = titles_row.iloc[0, 1:].dropna().astype(str).tolist()
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, skiprows=4)
        df = df.iloc[:, : (1 + len(titles))]
        df.columns = ["Date"] + titles
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df, titles, sheet_name

    try:
        eua_mtime = eua_oi_path.stat().st_mtime
        df_oi, contract_titles, used_sheet = load_eua_oi(eua_oi_path, eua_mtime)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.caption(
        f"Fichier: **{eua_oi_path.name}** — Feuille utilisée: **{used_sheet}** — "
        f"Dernière date: **{df_oi['Date'].max().date()}** (mtime: {int(eua_mtime)})"
    )

    left, right = st.columns([2, 1])
    with left:
        sel_contracts = st.multiselect(
            "Contrats à superposer (facultatif, sinon tous) :",
            options=contract_titles,
            default=contract_titles,
            key="oi_multiselect"
        )
    with right:
        single_contract = st.selectbox(
            "Graphique individuel :",
            options=contract_titles,
            index=0,
            key="oi_single_select"
        )

    st.subheader("Superposé")
    fig_super = go.Figure()
    to_plot = sel_contracts if sel_contracts else contract_titles
    for col in to_plot:
        fig_super.add_trace(go.Scatter(
            x=df_oi["Date"], y=df_oi[col],
            mode="lines", name=col,
            hovertemplate="%{x|%Y-%m-%d} — %{y:,}<extra>" + col + "</extra>"
        ))
    fig_super.update_layout(
        title="Historique des contrats (Open Interest) - Superposé",
        xaxis_title="Date",
        yaxis_title="Open Interest",
        yaxis=dict(tickformat=","),  # séparateur des milliers
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520
    )
    st.plotly_chart(fig_super, use_container_width=True)

    st.subheader("Individuel")
    fig_single = go.Figure()
    fig_single.add_trace(go.Scatter(
        x=df_oi["Date"], y=df_oi[single_contract],
        mode="lines", name=single_contract,
        hovertemplate="%{x|%Y-%m-%d} — %{y:,}<extra>" + single_contract + "</extra>"
    ))
    fig_single.update_layout(
        title=f"Historique Open Interest - {single_contract}",
        xaxis_title="Date",
        yaxis_title="Open Interest",
        yaxis=dict(tickformat=","),  # séparateur des milliers
        margin=dict(l=40, r=40, t=50, b=40),
        height=460
    )
    st.plotly_chart(fig_single, use_container_width=True)

    with st.expander("Export"):
        st.download_button(
            label="📥 Télécharger les données OI (CSV)",
            data=df_oi.to_csv(index=False).encode("utf-8"),
            file_name="eua_open_interest.csv",
            mime="text/csv"
        )
