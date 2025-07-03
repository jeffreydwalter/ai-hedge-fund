import datetime
import os
import pandas as pd
import requests
import time
import yfinance as yf

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or yfinance."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return []

    prices = [
        Price(
            open=float(row["Open"]),
            close=float(row["Close"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            volume=int(row["Volume"]),
            time=index.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        for index, row in df.iterrows()
    ]

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    t = yf.Ticker(ticker)
    info = t.info

    metrics = FinancialMetrics(
        ticker=ticker,
        report_period=end_date,
        period=period,
        currency=info.get("currency", "USD"),
        market_cap=info.get("marketCap"),
        enterprise_value=info.get("enterpriseValue"),
        price_to_earnings_ratio=info.get("trailingPE"),
        price_to_book_ratio=info.get("priceToBook"),
        price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
        return_on_equity=info.get("returnOnEquity"),
        return_on_assets=info.get("returnOnAssets"),
    )

    financial_metrics = [metrics]
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items using yfinance financial statements."""
    cache_key = f"{ticker}_{period}_{end_date}_{'_'.join(line_items)}_{limit}"
    if cached := _cache.get_line_items(cache_key):
        return [LineItem(**li) for li in cached]

    t = yf.Ticker(ticker)
    freq = "annual" if period == "annual" else "quarterly"
    income = t.income_stmt if freq == "annual" else t.quarterly_income_stmt
    balance = t.balance_sheet if freq == "annual" else t.quarterly_balance_sheet
    cash = t.cashflow if freq == "annual" else t.quarterly_cashflow
    info = t.info
    currency = info.get("currency", "USD")

    mapping = {
        "net_income": "Net Income",
        "earnings_per_share": "Diluted EPS",
        "ebit": "EBIT",
        "interest_expense": "Interest Expense",
        "operating_income": "Operating Income",
        "revenue": "Total Revenue",
        "gross_margin": "Gross Profit",
        "operating_margin": "Operating Income",
        "total_assets": "Total Assets",
        "total_liabilities": "Total Liabilities Net Minority Interest",
        "current_assets": "Current Assets",
        "current_liabilities": "Current Liabilities",
        "free_cash_flow": "Free Cash Flow",
        "capital_expenditure": "Capital Expenditure",
        "cash_and_equivalents": "Cash Cash Equivalents And Short Term Investments",
        "total_debt": "Total Debt",
        "shareholders_equity": "Stockholders Equity",
        "outstanding_shares": "Ordinary Shares Number",
        "dividends_and_other_cash_distributions": "Cash Dividends Paid",
        "issuance_or_purchase_of_equity_shares": "Net Common Stock Issuance",
        "research_and_development": "Research And Development",
        "goodwill_and_intangible_assets": "Goodwill And Other Intangible Assets",
    }

    dfs = [income, balance, cash]
    results = []
    if income is None or income.empty:
        income = pd.DataFrame()
    if balance is None or balance.empty:
        balance = pd.DataFrame()
    if cash is None or cash.empty:
        cash = pd.DataFrame()

    # Combine columns from all statements
    cols = sorted(set(income.columns) | set(balance.columns) | set(cash.columns), reverse=True)
    for col in cols:
        if len(results) >= limit:
            break
        period_data = {
            "ticker": ticker,
            "report_period": str(col)[:10],
            "period": freq,
            "currency": currency,
        }
        for item in line_items:
            df_val = None
            target = mapping.get(item)
            for df in dfs:
                if target in df.index and col in df.columns:
                    df_val = df.loc[target, col]
                    break
            if item == "gross_margin" and df_val is not None and "Total Revenue" in income.index:
                revenue_val = income.loc["Total Revenue", col]
                df_val = df_val / revenue_val if revenue_val else None
            if item == "operating_margin" and "Operating Income" in income.index and "Total Revenue" in income.index:
                op = income.loc["Operating Income", col]
                rev = income.loc["Total Revenue", col]
                df_val = op / rev if rev else None
            period_data[item] = None if df_val is None or pd.isna(df_val) else float(df_val)

        results.append(LineItem(**period_data))

    _cache.set_line_items(cache_key, [r.model_dump() for r in results])
    return results


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades. Uses SEC data if available."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # Free alternatives are limited; return empty list for now
    _cache.set_insider_trades(cache_key, [])
    return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news using free sources (yfinance)."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # yfinance provides limited news, which may not align exactly with the previous API
    _cache.set_company_news(cache_key, [])
    return []


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap using yfinance."""
    t = yf.Ticker(ticker)
    info = t.info
    mc = info.get("marketCap")
    if mc:
        return float(mc)
    financial_metrics = get_financial_metrics(ticker, end_date)
    if financial_metrics:
        return financial_metrics[0].market_cap
    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
