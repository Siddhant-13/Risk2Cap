import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

print("🔄 Generating synthetic data for Capital & Funding Analytics Platform...")

# ============= COUNTERPARTIES =============
counterparties = pd.DataFrame({
    'CounterpartyID': [f'CP{i:03d}' for i in range(1, 26)],
    'CounterpartyName': [
        'JPMorgan', 'Goldman Sachs', 'Morgan Stanley', 'Citi', 'Bank of America',
        'Deutsche Bank', 'Barclays', 'UBS', 'Credit Suisse', 'HSBC',
        'BNP Paribas', 'Societe Generale', 'Nomura', 'Mizuho', 'MUFG',
        'Wells Fargo', 'RBC', 'TD Bank', 'Scotiabank', 'BMO',
        'Citadel', 'Millennium', 'Bridgewater', 'AQR', 'Two Sigma'
    ],
    'CreditRating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB'], size=25, p=[0.08, 0.24, 0.36, 0.24, 0.08]),
    'Region': np.random.choice(['North America', 'Europe', 'Asia'], size=25, p=[0.5, 0.3, 0.2]),
    'Sector': np.random.choice(['Bank', 'Hedge Fund', 'Asset Manager'], size=25, p=[0.6, 0.3, 0.1]),
    'ExposureLimit': np.random.randint(5_000_000, 100_000_000, size=25)
})

print(f"✅ Generated {len(counterparties)} counterparties")

# ============= TRADES =============
n_trades = 400
trade_dates = [datetime.now() - timedelta(days=np.random.randint(30, 365)) for _ in range(n_trades)]
maturity_dates = [td + timedelta(days=np.random.randint(180, 1825)) for td in trade_dates]

trades = []

# Interest Rate Swaps (40%)
print("🔄 Generating Interest Rate Swaps...")
for i in range(160):
    trades.append({
        'TradeID': f'IRS{i:04d}',
        'CounterpartyID': np.random.choice(counterparties['CounterpartyID']),
        'ProductType': 'IRS',
        'Notional': np.random.randint(5, 50) * 1_000_000,
        'FixedRate': np.random.uniform(0.02, 0.06),
        'FloatingRate': 0.045,  # Current market rate
        'TradeDate': trade_dates[i],
        'MaturityDate': maturity_dates[i],
        'PayReceive': np.random.choice(['Receive_Fixed', 'Pay_Fixed']),
        'Currency': 'USD'
    })

# FX Forwards (35%)
print("🔄 Generating FX Forwards...")
for i in range(160, 300):
    buy_ccy = np.random.choice(['EUR', 'GBP', 'JPY'])
    spot_rates = {'EUR': 1.10, 'GBP': 1.27, 'JPY': 0.0067}
    spot = spot_rates[buy_ccy]
    forward = spot * np.random.uniform(0.98, 1.02)  # Small premium/discount
    
    trades.append({
        'TradeID': f'FXF{i:04d}',
        'CounterpartyID': np.random.choice(counterparties['CounterpartyID']),
        'ProductType': 'FX_Forward',
        'Notional': np.random.randint(2, 30) * 1_000_000,
        'BuyCurrency': buy_ccy,
        'SellCurrency': 'USD',
        'ForwardRate': forward,
        'SpotRate': spot,
        'TradeDate': trade_dates[i],
        'MaturityDate': maturity_dates[i]
    })

# Equity Options (25%)
print("🔄 Generating Equity Options...")
for i in range(300, 400):
    strike = np.random.randint(100, 300)
    current = strike * np.random.uniform(0.9, 1.1)  # ATM to slightly ITM/OTM
    
    trades.append({
        'TradeID': f'OPT{i:04d}',
        'CounterpartyID': np.random.choice(counterparties['CounterpartyID']),
        'ProductType': 'Equity_Option',
        'OptionType': np.random.choice(['Call', 'Put']),
        'Underlying': np.random.choice(['SPX', 'NDX', 'AAPL', 'MSFT', 'GOOGL']),
        'Strike': strike,
        'CurrentPrice': current,
        'Notional': np.random.randint(1, 10) * 1_000_000,
        'TradeDate': trade_dates[i],
        'Expiry': maturity_dates[i],
        'Volatility': np.random.uniform(0.15, 0.35)
    })

trades_df = pd.DataFrame(trades)
print(f"✅ Generated {len(trades_df)} trades")

# ============= MARKET DATA =============
market_data = {
    'Date': datetime.now(),
    'FloatingRate_USD': 0.045,
    'SpotRate_EUR': 1.10,
    'SpotRate_GBP': 1.27,
    'SpotRate_JPY': 0.0067,
    'StockPrice_SPX': 6000,
    'StockPrice_NDX': 20000,
    'StockPrice_AAPL': 180,
    'StockPrice_MSFT': 420,
    'StockPrice_GOOGL': 175,
    'RiskFreeRate': 0.05,
    'FundingRate': 0.05  # Repo/funding rate
}

print("✅ Generated market data snapshot")

# ============= SAVE =============
import os
os.makedirs('data', exist_ok=True)

counterparties.to_csv('data/counterparties.csv', index=False)
trades_df.to_csv('data/trades.csv', index=False)
pd.DataFrame([market_data]).to_csv('data/market_data.csv', index=False)

print("\n" + "="*60)
print("✅ DATA GENERATION COMPLETE!")
print("="*60)
print(f"📁 Files created in 'data/' directory:")
print(f"   - counterparties.csv ({len(counterparties)} rows)")
print(f"   - trades.csv ({len(trades_df)} rows)")
print(f"   - market_data.csv (1 row)")
print("\n💡 Next step: Run capital calculations")
print("   python capital_calculator.py")