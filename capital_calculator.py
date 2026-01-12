import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CapitalCalculator:
    """
    Capital & Funding Analytics Calculator for OTC Derivatives
    
    Calculates:
    1. Mark-to-Market (MTM)
    2. Potential Future Exposure (PFE) via Monte Carlo
    3. Risk-Weighted Assets (RWA) - Basel III
    4. Leverage Ratio Exposure
    5. Initial Margin & Funding Costs
    6. Capital Efficiency Score
    """
    
    def __init__(self, trades_df, market_data, counterparties_df):
        self.trades = trades_df
        self.market = market_data
        self.counterparties = counterparties_df
        
    # ==================== MTM CALCULATIONS ====================
    
    def calculate_mtm(self, trade):
        """Calculate Mark-to-Market for a trade"""
        if trade['ProductType'] == 'IRS':
            return self._mtm_irs(trade)
        elif trade['ProductType'] == 'FX_Forward':
            return self._mtm_fx_forward(trade)
        elif trade['ProductType'] == 'Equity_Option':
            return self._mtm_option(trade)
        return 0
    
    def _mtm_irs(self, trade):
        """MTM for Interest Rate Swap"""
        notional = trade['Notional']
        fixed_rate = trade['FixedRate']
        floating_rate = self.market['FloatingRate_USD']
        
        # Years to maturity
        years_remaining = (pd.to_datetime(trade['MaturityDate']) - datetime.now()).days / 365.25
        years_remaining = max(years_remaining, 0.1)
        
        # Simplified duration
        duration = years_remaining * 0.9
        
        # Rate differential
        if trade['PayReceive'] == 'Receive_Fixed':
            rate_diff = fixed_rate - floating_rate
        else:
            rate_diff = floating_rate - fixed_rate
        
        mtm = notional * rate_diff * duration
        return mtm
    
    def _mtm_fx_forward(self, trade):
        """MTM for FX Forward"""
        notional = trade['Notional']
        forward_rate = trade['ForwardRate']
        spot_rate = self.market[f"SpotRate_{trade['BuyCurrency']}"]
        
        years_to_maturity = (pd.to_datetime(trade['MaturityDate']) - datetime.now()).days / 365.25
        years_to_maturity = max(years_to_maturity, 0.01)
        
        discount_factor = np.exp(-self.market['RiskFreeRate'] * years_to_maturity)
        
        mtm = notional * (spot_rate - forward_rate) * discount_factor
        return mtm
    
    def _mtm_option(self, trade):
        """MTM for Equity Option using Black-Scholes"""
        S = self.market[f"StockPrice_{trade['Underlying']}"]
        K = trade['Strike']
        
        T = (pd.to_datetime(trade['Expiry']) - datetime.now()).days / 365.25
        T = max(T, 0.01)
        
        r = self.market['RiskFreeRate']
        sigma = trade['Volatility']
        
        # Black-Scholes
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if trade['OptionType'] == 'Call':
            option_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Scale by notional
        mtm = option_value * (trade['Notional'] / (S * 100))
        return mtm
    
    # ==================== PFE CALCULATION ====================
    
    def calculate_pfe(self, trade, num_simulations=1000):
        """Calculate Potential Future Exposure using Monte Carlo"""
        
        T = (pd.to_datetime(trade.get('MaturityDate') or trade.get('Expiry')) - datetime.now()).days / 365.25
        T = max(T, 0.1)
        
        # Product-specific volatility
        if trade['ProductType'] == 'IRS':
            volatility = 0.01  # 100 bps
            current_param = self.market['FloatingRate_USD']
            param_key = 'FloatingRate_USD'
        elif trade['ProductType'] == 'FX_Forward':
            volatility = 0.10  # 10%
            current_param = self.market[f"SpotRate_{trade['BuyCurrency']}"]
            param_key = f"SpotRate_{trade['BuyCurrency']}"
        else:  # Option
            volatility = trade['Volatility']
            current_param = self.market[f"StockPrice_{trade['Underlying']}"]
            param_key = f"StockPrice_{trade['Underlying']}"
        
        # Monte Carlo simulation
        random_shocks = np.random.normal(0, volatility * np.sqrt(T), num_simulations)
        simulated_params = current_param * np.exp(random_shocks)
        
        future_mtms = []
        for sim_param in simulated_params:
            # Create simulated market
            sim_market = self.market.copy()
            sim_market[param_key] = sim_param
            
            # Temporarily update market
            old_market = self.market
            self.market = sim_market
            
            # Calculate MTM
            future_mtm = self.calculate_mtm(trade)
            future_mtms.append(max(future_mtm, 0))  # Only positive exposures
            
            # Restore market
            self.market = old_market
        
        # PFE = 95th percentile
        pfe = np.percentile(future_mtms, 95)
        return pfe
    
    # ==================== RWA CALCULATION (Basel III) ====================
    
    def calculate_rwa(self, trade, pfe):
        """
        Calculate Risk-Weighted Assets (RWA) for capital requirement
        
        Basel III SA-CCR (simplified):
        RWA = 1.4 × EAD × Risk Weight
        
        where EAD (Exposure at Default) ≈ PFE
        """
        
        # Get counterparty credit rating
        cp_id = trade['CounterpartyID']
        counterparty = self.counterparties[self.counterparties['CounterpartyID'] == cp_id].iloc[0]
        credit_rating = counterparty['CreditRating']
        
        # Risk weights by credit rating (Basel III standardized approach)
        risk_weights = {
            'AAA': 0.20,
            'AA': 0.20,
            'A': 0.50,
            'BBB': 1.00,
            'BB': 2.00,
            'B': 3.00
        }
        
        risk_weight = risk_weights.get(credit_rating, 1.00)
        
        # Simplified RWA calculation
        # EAD = Exposure at Default (we use PFE as proxy)
        ead = pfe
        
        # RWA = 1.4 × EAD × Risk Weight (1.4 is alpha factor in SA-CCR)
        rwa = 1.4 * ead * risk_weight
        
        # Capital requirement = 8% of RWA (Basel III minimum)
        capital_required = 0.08 * rwa
        
        return {
            'RWA': rwa,
            'CapitalRequired': capital_required,
            'RiskWeight': risk_weight
        }
    
    # ==================== LEVERAGE RATIO ====================
    
    def calculate_leverage_exposure(self, trade):
        """
        Calculate Leverage Ratio exposure (Basel III)
        
        Leverage Ratio = Tier 1 Capital / Total Exposure
        
        For derivatives: Exposure = Replacement Cost + Add-On
        """
        
        # Replacement Cost = max(MTM, 0)
        mtm = self.calculate_mtm(trade)
        replacement_cost = max(mtm, 0)
        
        # Add-on based on notional and maturity
        notional = trade['Notional']
        
        if trade['ProductType'] == 'IRS':
            maturity_bucket = self._get_maturity_bucket(trade)
            # Supervisory factors for IR derivatives
            add_on_factor = {'<1yr': 0.0, '1-5yr': 0.005, '>5yr': 0.015}
            add_on = notional * add_on_factor.get(maturity_bucket, 0.005)
        
        elif trade['ProductType'] == 'FX_Forward':
            maturity_bucket = self._get_maturity_bucket(trade)
            add_on_factor = {'<1yr': 0.01, '1-5yr': 0.05, '>5yr': 0.075}
            add_on = notional * add_on_factor.get(maturity_bucket, 0.05)
        
        else:  # Equity Option
            add_on = notional * 0.06  # 6% for equity derivatives
        
        # Total leverage exposure
        leverage_exposure = replacement_cost + add_on
        
        return leverage_exposure
    
    def _get_maturity_bucket(self, trade):
        """Helper to categorize maturity"""
        maturity_date = pd.to_datetime(trade.get('MaturityDate') or trade.get('Expiry'))
        years_to_maturity = (maturity_date - datetime.now()).days / 365.25
        
        if years_to_maturity < 1:
            return '<1yr'
        elif years_to_maturity < 5:
            return '1-5yr'
        else:
            return '>5yr'
    
    # ==================== FUNDING COST ====================
    
    def calculate_initial_margin(self, trade, pfe):
        """
        Calculate Initial Margin (IM) requirement and funding cost
        
        IM is the collateral needed to cover potential future losses
        Typically IM ≈ 1.5 × PFE (regulatory buffer)
        """
        
        # For uncleared derivatives, IM is typically based on SIMM
        # Simplified: IM = multiplier × PFE
        im_multiplier = 1.5
        
        initial_margin = im_multiplier * pfe
        
        # Funding cost = IM × Funding Rate × Time
        funding_rate = self.market['FundingRate']  # Cost of borrowing cash for collateral
        
        # Calculate time to maturity
        maturity_date = pd.to_datetime(trade.get('MaturityDate') or trade.get('Expiry'))
        years_to_maturity = (maturity_date - datetime.now()).days / 365.25
        years_to_maturity = max(years_to_maturity, 0.1)
        
        # Annualized funding cost
        funding_cost = initial_margin * funding_rate * years_to_maturity
        
        return {
            'InitialMargin': initial_margin,
            'FundingCost': funding_cost,
            'FundingRate': funding_rate
        }
    
    # ==================== CAPITAL EFFICIENCY ====================
    
    def calculate_capital_efficiency(self, trade, capital_required, funding_cost):
        """
        Calculate Capital Efficiency Score
        
        Efficiency = Expected Revenue / Total Capital Cost
        
        Higher score = better use of balance sheet
        """
        
        # Estimate revenue (simplified: assume bid-offer spread)
        notional = trade['Notional']
        
        # Revenue assumptions by product type
        if trade['ProductType'] == 'IRS':
            revenue_bps = 0.025  # 2.5 bps
        elif trade['ProductType'] == 'FX_Forward':
            revenue_bps = 0.05  # 5 bps
        else:  # Equity Option
            revenue_bps = 0.10  # 10 bps
        
        expected_revenue = notional * revenue_bps / 100
        
        # Total capital cost
        total_capital_cost = capital_required + funding_cost
        
        # Efficiency score
        if total_capital_cost > 0:
            efficiency = expected_revenue / total_capital_cost
        else:
            efficiency = 0
        
        return {
            'ExpectedRevenue': expected_revenue,
            'TotalCapitalCost': total_capital_cost,
            'CapitalEfficiency': efficiency
        }
    
    # ==================== MAIN CALCULATION ====================
    
    def calculate_all_exposures(self):
        """Calculate all metrics for all trades"""
        results = []
        
        print(f"🔄 Processing {len(self.trades)} trades...")
        
        for idx, trade in self.trades.iterrows():
            if idx % 50 == 0:
                print(f"   Processed {idx}/{len(self.trades)} trades...")
            
            # Basic exposure metrics
            mtm = self.calculate_mtm(trade)
            pfe = self.calculate_pfe(trade)
            
            # Capital metrics
            rwa_data = self.calculate_rwa(trade, pfe)
            leverage_exp = self.calculate_leverage_exposure(trade)
            funding_data = self.calculate_initial_margin(trade, pfe)
            efficiency_data = self.calculate_capital_efficiency(
                trade, 
                rwa_data['CapitalRequired'], 
                funding_data['FundingCost']
            )
            
            results.append({
                'TradeID': trade['TradeID'],
                'CounterpartyID': trade['CounterpartyID'],
                'ProductType': trade['ProductType'],
                'Notional': trade['Notional'],
                
                # Exposure
                'MTM': mtm,
                'PFE': pfe,
                'CurrentExposure': max(mtm, 0),
                'TotalExposure': max(mtm, 0) + pfe,
                
                # Capital
                'RWA': rwa_data['RWA'],
                'CapitalRequired': rwa_data['CapitalRequired'],
                'RiskWeight': rwa_data['RiskWeight'],
                'LeverageExposure': leverage_exp,
                
                # Funding
                'InitialMargin': funding_data['InitialMargin'],
                'FundingCost': funding_data['FundingCost'],
                
                # Efficiency
                'ExpectedRevenue': efficiency_data['ExpectedRevenue'],
                'TotalCapitalCost': efficiency_data['TotalCapitalCost'],
                'CapitalEfficiency': efficiency_data['CapitalEfficiency']
            })
        
        print(f"✅ Processed all {len(self.trades)} trades")
        
        exposures_df = pd.DataFrame(results)
        
        # Add counterparty details
        exposures_df = exposures_df.merge(
            self.counterparties[['CounterpartyID', 'CounterpartyName', 'Region', 'CreditRating', 'ExposureLimit', 'Sector']], 
            on='CounterpartyID'
        )
        
        return exposures_df
    
    def aggregate_by_counterparty(self, exposures_df):
        """Aggregate all metrics by counterparty"""
        
        print("🔄 Aggregating by counterparty...")
        
        agg = exposures_df.groupby('CounterpartyID').agg({
            'CounterpartyName': 'first',
            'Region': 'first',
            'CreditRating': 'first',
            'ExposureLimit': 'first',
            'Sector': 'first',
            
            # Exposure metrics
            'MTM': 'sum',
            'CurrentExposure': 'sum',
            'PFE': 'sum',
            'TotalExposure': 'sum',
            
            # Capital metrics
            'RWA': 'sum',
            'CapitalRequired': 'sum',
            'LeverageExposure': 'sum',
            
            # Funding
            'InitialMargin': 'sum',
            'FundingCost': 'sum',
            
            # Efficiency
            'ExpectedRevenue': 'sum',
            'TotalCapitalCost': 'sum',
            
            'TradeID': 'count',
            'Notional': 'sum'
        }).reset_index()
        
        agg.rename(columns={'TradeID': 'NumTrades'}, inplace=True)
        
        # Calculate metrics
        agg['Utilization'] = agg['TotalExposure'] / agg['ExposureLimit']
        agg['RWADensity'] = agg['RWA'] / agg['Notional']  # RWA as % of notional
        agg['CapitalEfficiency'] = agg['ExpectedRevenue'] / agg['TotalCapitalCost']
        
        # Status flags
        agg['Status'] = 'OK'
        agg.loc[agg['Utilization'] > 0.9, 'Status'] = 'Warning'
        agg.loc[agg['Utilization'] > 1.0, 'Status'] = 'Breach'
        agg.loc[agg['Utilization'] > 1.2, 'Status'] = 'Critical'
        
        print("✅ Aggregation complete")
        
        return agg

# ==================== HELPER FUNCTION ====================

def load_and_calculate():
    """Load data and calculate all exposures"""
    
    print("\n" + "="*60)
    print("CAPITAL & FUNDING ANALYTICS CALCULATOR")
    print("="*60 + "\n")
    
    print("📁 Loading data files...")
    trades = pd.read_csv('data/trades.csv')
    counterparties = pd.read_csv('data/counterparties.csv')
    market_data = pd.read_csv('data/market_data.csv').iloc[0].to_dict()
    print("✅ Data loaded successfully\n")
    
    calc = CapitalCalculator(trades, market_data, counterparties)
    
    exposures = calc.calculate_all_exposures()
    print()
    summary = calc.aggregate_by_counterparty(exposures)
    
    print("\n" + "="*60)
    print("CALCULATION SUMMARY")
    print("="*60)
    print(f"Total Trades: {len(exposures)}")
    print(f"Total Counterparties: {len(summary)}")
    print(f"Total Notional: ${exposures['Notional'].sum()/1e6:.1f}M")
    print(f"Total RWA: ${summary['RWA'].sum()/1e6:.1f}M")
    print(f"Total Capital Required: ${summary['CapitalRequired'].sum()/1e6:.2f}M")
    print(f"Total Funding Cost: ${summary['FundingCost'].sum()/1e6:.2f}M")
    print(f"Average Capital Efficiency: {summary['CapitalEfficiency'].mean():.2f}x")
    print("="*60 + "\n")
    
    return exposures, summary, counterparties

# ==================== STANDALONE EXECUTION ====================

if __name__ == "__main__":
    exposures, summary, counterparties = load_and_calculate()
    
    print("💡 Calculations complete! Run the dashboard:")
    print("   streamlit run dashboard.py")