# Investment Portfolio Analysis

This project provides a structured quantitative portfolio analysis across S&P 500 (United States), IPC (Mexico), and Ibovespa (Brazil).  
It covers universe construction, cleaning, price validation, covariance modeling, portfolio optimization, efficient frontier visualization, and historical performance benchmarking.  

---

## Quick Access
- Dashboard: Streamlit App (to be deployed)  
- Notebook: [Main Jupyter Notebook](./notebooks/Efficient_Frontier_Analysis.ipynb)  
- PDF Report: To be added  

---

## Highlights
- Full pipeline: from raw index constituents to cleaning, validation, price fetching, and analysis.  
- Optimizations: Minimum Variance and Maximum Sharpe portfolios.  
- Performance Ratios: Sharpe, Sortino, and Treynor included for deeper evaluation.  
- Efficient Frontier: thousands of simulated allocations visualized against volatility/return trade-offs.  
- Historical Performance: comparison of individual assets versus optimized portfolios, with ROI and a zero-return baseline.  
- Cross-markets: works with equity universes from the United States, Mexico, and Brazil.  

---

## Repository Structure
- `data/universes/` — Validated tickers with Yahoo Finance status.  
- `notebook.py` — Jupyter notebooks with full analysis.  
- `app.py` — Code for the Streamlit dashboard (to be added).  
- `requirements.txt` — Project dependencies.  
- `README.md` — Project documentation.

---

## Data
- Sources:  
  - Wikipedia (S&P 500, IPC), TradingView (Ibovespa), and Yahoo Finance.  
- Content:  
  - Constituents with validation flags (`ok` or `no_price_data`).  
  - Historical daily price data fetched via Yahoo Finance API.  
- Note: Price data is not stored in this repository. The dashboard fetches live prices at runtime.  

---

## How to Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/luisesantamaria/investment-portfolio-analysis.git
cd investment-portfolio-analysis

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/Efficient_Frontier_Analysis.ipynb

