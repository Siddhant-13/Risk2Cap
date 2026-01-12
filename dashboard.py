import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from capital_calculator import load_and_calculate
import numpy as np

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Capital & Funding Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================

@st.cache_data
def get_data():
    return load_and_calculate()

with st.spinner('🔄 Loading and calculating capital metrics...'):
    exposures, summary, counterparties = get_data()

# ==================== SIDEBAR ====================

st.sidebar.title("💰 Capital Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🏦 Capital Analysis", "💵 Funding Analysis", "📈 Trade Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

region_filter = st.sidebar.multiselect(
    "Region",
    options=summary['Region'].unique(),
    default=None
)

product_filter = st.sidebar.multiselect(
    "Product Type",
    options=exposures['ProductType'].unique(),
    default=None
)

# Apply filters
if region_filter:
    summary = summary[summary['Region'].isin(region_filter)]
    exposures = exposures[exposures['Region'].isin(region_filter)]

if product_filter:
    exposures = exposures[exposures['ProductType'].isin(product_filter)]
    from capital_calculator import CapitalCalculator
    trades = pd.read_csv('data/trades.csv')
    market_data = pd.read_csv('data/market_data.csv').iloc[0].to_dict()
    calc = CapitalCalculator(trades, market_data, counterparties)
    summary = calc.aggregate_by_counterparty(exposures)

st.sidebar.markdown("---")
st.sidebar.info(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

# ==================== PAGE 1: OVERVIEW ====================

if page == "📊 Overview":
    
    st.markdown('<p class="main-header">💰 Capital & Funding Analytics Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time capital optimization for OTC derivatives trading</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_rwa = summary['RWA'].sum()
        st.metric("Total RWA", f"${total_rwa/1e6:.1f}M")
    
    with col2:
        total_capital = summary['CapitalRequired'].sum()
        st.metric("Capital Required", f"${total_capital/1e6:.2f}M")
    
    with col3:
        total_funding = summary['FundingCost'].sum()
        st.metric("Funding Cost", f"${total_funding/1e6:.2f}M")
    
    with col4:
        avg_efficiency = summary['CapitalEfficiency'].mean()
        st.metric("Avg Efficiency", f"{avg_efficiency:.2f}x")
    
    with col5:
        breaches = len(summary[summary['Status'].isin(['Breach', 'Critical'])])
        st.metric("Limit Breaches", breaches, delta=f"{breaches} active", delta_color="inverse")
    
    st.markdown("---")
    
    breached = summary[summary['Status'].isin(['Breach', 'Critical'])].sort_values('Utilization', ascending=False)
    
    if len(breached) > 0:
        st.error(f"⚠️ **{len(breached)} EXPOSURE LIMIT BREACHES DETECTED**")
        
        breach_display = breached[['CounterpartyName', 'TotalExposure', 'ExposureLimit', 'Utilization', 'Status']].copy()
        breach_display['TotalExposure'] = breach_display['TotalExposure'].apply(lambda x: f"${x/1e6:.2f}M")
        breach_display['ExposureLimit'] = breach_display['ExposureLimit'].apply(lambda x: f"${x/1e6:.2f}M")
        breach_display['Utilization'] = breach_display['Utilization'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(breach_display, hide_index=True, use_container_width=True)
    else:
        st.success("✅ **No exposure limit breaches** - All exposures within approved limits")
    
    st.markdown("---")
    
    inefficient_trades = exposures[exposures['CapitalEfficiency'] < 1.0]
    
    if len(inefficient_trades) > 0:
        st.warning(f"⚠️ **{len(inefficient_trades)} trades have negative capital ROI** (Efficiency < 1.0x)")
        st.caption("These trades cost more in capital than they generate in revenue. Consider unwinding or hedging.")
    
    st.markdown("---")
    
    st.subheader("📊 Top 10 Counterparties by RWA Consumption")
    
    top_10_rwa = summary.nlargest(10, 'RWA')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_10_rwa['CounterpartyName'],
        y=top_10_rwa['RWA'] / 1e6,
        name='RWA',
        marker_color='#FF6B6B',
        text=top_10_rwa['RWA'].apply(lambda x: f'${x/1e6:.1f}M'),
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis_title="Counterparty",
        yaxis_title="RWA ($M)",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 RWA by Product Type")
        
        product_rwa = exposures.groupby('ProductType')['RWA'].sum().reset_index()
        product_rwa['Product'] = product_rwa['ProductType'].map({
            'IRS': 'Interest Rate Swaps',
            'FX_Forward': 'FX Forwards',
            'Equity_Option': 'Equity Options'
        })
        
        fig = px.pie(
            product_rwa,
            values='RWA',
            names='Product',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Capital Required by Region")
        
        region_capital = summary.groupby('Region')['CapitalRequired'].sum().reset_index()
        
        fig = px.bar(
            region_capital,
            x='Region',
            y='CapitalRequired',
            color='Region',
            text=region_capital['CapitalRequired'].apply(lambda x: f'${x/1e6:.2f}M'),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Capital Required ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Capital Efficiency Analysis")
    
    fig = px.scatter(
        summary,
        x='RWA',
        y='CapitalEfficiency',
        size='NumTrades',
        color='CreditRating',
        hover_data=['CounterpartyName', 'Region'],
        labels={'RWA': 'Risk-Weighted Assets ($)', 'CapitalEfficiency': 'Capital Efficiency (Revenue/Cost)'},
        title="Capital Efficiency vs. RWA by Counterparty"
    )
    
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even (1.0x)")
    
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("💡 **Insight:** Counterparties above the red line generate positive capital ROI. Below = capital-inefficient.")

elif page == "🏦 Capital Analysis":
    
    st.title("🏦 Capital Analysis & Optimization")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_rwa = summary['RWA'].sum()
        rwa_density = total_rwa / summary['Notional'].sum()
        st.metric("Total RWA", f"${total_rwa/1e6:.1f}M", delta=f"{rwa_density:.2%} of notional")
    
    with col2:
        total_capital = summary['CapitalRequired'].sum()
        st.metric("Capital Requirement", f"${total_capital/1e6:.2f}M", delta="8% of RWA")
    
    with col3:
        total_leverage = summary['LeverageExposure'].sum()
        st.metric("Leverage Exposure", f"${total_leverage/1e6:.1f}M")
    
    with col4:
        avg_rwa_density = summary['RWADensity'].mean()
        st.metric("Avg RWA Density", f"{avg_rwa_density:.2%}")
    
    st.markdown("---")
    
    st.subheader("⚠️ Capital-Inefficient Trades (Bottom 10)")
    
    inefficient = exposures.nsmallest(10, 'CapitalEfficiency')[
        ['TradeID', 'CounterpartyName', 'ProductType', 'Notional', 
         'RWA', 'CapitalRequired', 'FundingCost', 'ExpectedRevenue', 'TotalCapitalCost', 'CapitalEfficiency']
    ].copy()
    
    inefficient['Notional'] = inefficient['Notional'].apply(lambda x: f"${x/1e6:.2f}M")
    inefficient['RWA'] = inefficient['RWA'].apply(lambda x: f"${x/1e6:.2f}M")
    inefficient['CapitalRequired'] = inefficient['CapitalRequired'].apply(lambda x: f"${x/1e3:.1f}K")
    inefficient['FundingCost'] = inefficient['FundingCost'].apply(lambda x: f"${x/1e3:.1f}K")
    inefficient['ExpectedRevenue'] = inefficient['ExpectedRevenue'].apply(lambda x: f"${x/1e3:.1f}K")
    inefficient['TotalCapitalCost'] = inefficient['TotalCapitalCost'].apply(lambda x: f"${x/1e3:.1f}K")
    inefficient['CapitalEfficiency'] = inefficient['CapitalEfficiency'].apply(lambda x: f"{x:.2f}x")
    
    st.dataframe(inefficient, hide_index=True, use_container_width=True)
    
    st.info("🎯 **Recommendation:** These trades consume disproportionate capital relative to revenue. Consider unwinding, hedging, or repricing.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 RWA by Product Type")
        
        rwa_by_product = exposures.groupby('ProductType').agg({
            'RWA': 'sum',
            'Notional': 'sum',
            'TradeID': 'count'
        }).reset_index()
        
        rwa_by_product['RWADensity'] = rwa_by_product['RWA'] / rwa_by_product['Notional']
        rwa_by_product.columns = ['ProductType', 'RWA', 'Notional', 'NumTrades', 'RWADensity']
        
        rwa_by_product['Product'] = rwa_by_product['ProductType'].map({
            'IRS': 'Interest Rate Swaps',
            'FX_Forward': 'FX Forwards',
            'Equity_Option': 'Equity Options'
        })
        
        fig = px.bar(
            rwa_by_product,
            x='Product',
            y='RWA',
            color='Product',
            text=rwa_by_product['RWA'].apply(lambda x: f'${x/1e6:.1f}M'),
            title="Which products consume most capital?"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 RWA Density by Credit Rating")
        
        rwa_by_rating = summary.groupby('CreditRating').agg({
            'RWA': 'sum',
            'Notional': 'sum'
        }).reset_index()
        
        rwa_by_rating['RWADensity'] = rwa_by_rating['RWA'] / rwa_by_rating['Notional']
        
        fig = px.bar(
            rwa_by_rating,
            x='CreditRating',
            y='RWADensity',
            color='CreditRating',
            text=rwa_by_rating['RWADensity'].apply(lambda x: f'{x:.1%}'),
            title="Capital intensity by counterparty rating"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="RWA Density (% of Notional)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔥 RWA Density Heatmap")
    
    heatmap_data = summary.groupby(['Region', 'CreditRating'])['RWADensity'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Region', columns='CreditRating', values='RWADensity').fillna(0)
    
    fig = px.imshow(
        heatmap_pivot * 100,
        labels=dict(x="Credit Rating", y="Region", color="RWA Density %"),
        color_continuous_scale='Reds',
        aspect="auto",
        text_auto='.1f'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

elif page == "💵 Funding Analysis":
    
    st.title("💵 Funding & Liquidity Analysis")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_im = summary['InitialMargin'].sum()
        st.metric("Total Initial Margin", f"${total_im/1e6:.1f}M")
    
    with col2:
        total_funding = summary['FundingCost'].sum()
        st.metric("Total Funding Cost", f"${total_funding/1e6:.2f}M")
    
    with col3:
        avg_funding_rate = 5.0
        st.metric("Funding Rate", f"{avg_funding_rate:.2f}%")
    
    with col4:
        funding_as_pct_revenue = (total_funding / summary['ExpectedRevenue'].sum()) * 100
        st.metric("Funding as % Revenue", f"{funding_as_pct_revenue:.1f}%")
    
    st.markdown("---")
    
    st.subheader("💸 Highest Funding Cost Trades")
    
    high_funding = exposures.nlargest(10, 'FundingCost')[
        ['TradeID', 'CounterpartyName', 'ProductType', 'Notional', 
         'InitialMargin', 'FundingCost', 'ExpectedRevenue']
    ].copy()
    
    high_funding['Notional'] = high_funding['Notional'].apply(lambda x: f"${x/1e6:.2f}M")
    high_funding['InitialMargin'] = high_funding['InitialMargin'].apply(lambda x: f"${x/1e6:.2f}M")
    high_funding['FundingCost'] = high_funding['FundingCost'].apply(lambda x: f"${x/1e3:.1f}K")
    high_funding['ExpectedRevenue'] = high_funding['ExpectedRevenue'].apply(lambda x: f"${x/1e3:.1f}K")
    
    st.dataframe(high_funding, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Initial Margin by Product")
        
        im_by_product = exposures.groupby('ProductType')['InitialMargin'].sum().reset_index()
        im_by_product['Product'] = im_by_product['ProductType'].map({
            'IRS': 'Interest Rate Swaps',
            'FX_Forward': 'FX Forwards',
            'Equity_Option': 'Equity Options'
        })
        
        fig = px.pie(
            im_by_product,
            values='InitialMargin',
            names='Product',
            title="Collateral requirements by product type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Funding Cost by Region")
        
        funding_by_region = summary.groupby('Region')['FundingCost'].sum().reset_index()
        
        fig = px.bar(
            funding_by_region,
            x='Region',
            y='FundingCost',
            color='Region',
            text=funding_by_region['FundingCost'].apply(lambda x: f'${x/1e6:.2f}M')
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Funding Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Trade Explorer":
    
    st.title("📈 Trade-Level Analytics")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_notional = st.slider("Min Notional ($M)", 0, 50, 0)
    
    with col2:
        min_efficiency = st.slider("Min Capital Efficiency", 0.0, 5.0, 0.0)
    
    with col3:
        selected_cp = st.multiselect("Counterparty", exposures['CounterpartyName'].unique())
    
    filtered = exposures.copy()
    filtered = filtered[filtered['Notional'] >= min_notional * 1e6]
    filtered = filtered[filtered['CapitalEfficiency'] >= min_efficiency]
    if selected_cp:
        filtered = filtered[filtered['CounterpartyName'].isin(selected_cp)]
    
    st.markdown(f"**Showing {len(filtered)} of {len(exposures)} trades**")
    st.markdown("---")
    
    display_df = filtered[[
        'TradeID', 'CounterpartyName', 'ProductType', 'CreditRating',
        'Notional', 'MTM', 'PFE', 'RWA', 'CapitalRequired', 
        'FundingCost', 'ExpectedRevenue', 'CapitalEfficiency'
    ]].copy()
    
    display_df['Notional'] = display_df['Notional'].apply(lambda x: f"${x/1e6:.2f}M")
    display_df['MTM'] = display_df['MTM'].apply(lambda x: f"${x/1e3:.0f}K")
    display_df['PFE'] = display_df['PFE'].apply(lambda x: f"${x/1e6:.2f}M")
    display_df['RWA'] = display_df['RWA'].apply(lambda x: f"${x/1e6:.2f}M")
    display_df['CapitalRequired'] = display_df['CapitalRequired'].apply(lambda x: f"${x/1e3:.1f}K")
    display_df['FundingCost'] = display_df['FundingCost'].apply(lambda x: f"${x/1e3:.1f}K")
    display_df['ExpectedRevenue'] = display_df['ExpectedRevenue'].apply(lambda x: f"${x/1e3:.1f}K")
    display_df['CapitalEfficiency'] = display_df['CapitalEfficiency'].apply(lambda x: f"{x:.2f}x")
    
    st.dataframe(display_df, hide_index=True, use_container_width=True, height=500)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Trade-Level Data (CSV)",
            data=exposures.to_csv(index=False),
            file_name=f"trade_analytics_{pd.Timestamp.now().date()}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📥 Download Counterparty Summary (CSV)",
            data=summary.to_csv(index=False),
            file_name=f"counterparty_summary_{pd.Timestamp.now().date()}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Built with Python • Streamlit • Monte Carlo • Basel III • Black-Scholes")
st.caption(f"📊 {len(exposures)} trades | 💰 ${summary['RWA'].sum()/1e6:.1f}M RWA | ⏰ Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")