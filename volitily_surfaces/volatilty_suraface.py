import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from zoneinfo import ZoneInfo

tickers = ['SPY','AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
#load the first stock
y_ticker =  yf.Ticker(tickers[0])

#get the expriton date
expiration_dates = y_ticker.options

print("Expiration Dates: ", expiration_dates)

calls_chain = []
puts_chain = []
#loop through the expiration dates
for expiration_date in expiration_dates:
    #get the options chain for the expiration date
    options_chain = y_ticker.option_chain(expiration_date)
    calls = options_chain.calls
    puts = options_chain.puts

    #add the expiration date to the calls and puts dataframes
    calls['expirationDate'] = expiration_date
    puts['expirationDate'] = expiration_date

    #append the calls and puts dataframes to the lists
    calls_chain.append(calls)
    puts_chain.append(puts)
    

#get the current price of the stock
current_price = y_ticker.history(period='1d')['Close'].iloc[-1]
print("Current Price: ", current_price)
global risk_free_rate 
risk_free_rate = 0.043

def binomial_tree(S, K, T, r, sigma, n, option_type='call'):
    """
    S: current stock price
    K: strike price
    T: time to expiration in years
    r: risk-free interest rate
    sigma: volatility
    n: number of steps in the binomial tree
    """
    dt = T / n  # time step
    u = np.exp(sigma * np.sqrt(dt))  # up factor
    d = 1 / u  # down factor
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    # Initialize asset prices at maturity
    n = int(n)
    asset_prices = np.zeros(n + 1)
    for i in range(n + 1):
        asset_prices[i] = S * (u ** (i)) * (d ** (n-i))

    # Initialize option values at maturity
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)  # Call option payoff
    elif option_type == 'put':
        option_values = np.maximum(0, K - asset_prices)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    # Backward induction to calculate option price
    for j in range(n-1, -1, -1):
        for i in range(j + 1):
            option_values[i] = (p * option_values[i + 1] + (1 - p) * option_values[i]) * np.exp(-r * dt)
            if option_type == 'call':
                intrinsic_value = np.maximum(0, S*(u**i)*(d**(j-i)) - K)
            elif option_type == 'put':
                intrinsic_value = np.maximum(0, K - S*(u**i)*(d**(j-i)))
            # handle early exercise for American options
            option_values[i] = np.maximum(option_values[i], intrinsic_value)
    return option_values[0]  # Return the option price at time 0


# Example parameters
# S = current_price  # Current stock price
# K = 200  # Strike price
# T = 30 / 365  # Time to expiration in years (30 days)
# r = 0.05  # Risk-free interest rate (5%)
# sigma = 0.32  # Volatility (20%)
# n = 1000  # Number of steps in the binomial tree
# # Calculate call and put option prices
# call_price = binomial_tree(S, K, T, r, sigma, n, option_type='call')
# put_price = binomial_tree(S, K, T, r, sigma, n, option_type='put')
# print(f"Call Option Price: {call_price:.2f}")
# print(f"Put Option Price: {put_price:.2f}")



calls_chain_df = pd.concat(calls_chain)
puts_chain_df = pd.concat(puts_chain)

from scipy.optimize import brentq

def implied_vol_binomial_tree(S, K, T, r, market_price, N, option_type='call'):
    
    def objective(sigma):
        price = binomial_tree(S, K, T, r, sigma, N, option_type)
        return price - market_price

    # Brent's method requires a bracket where the function changes sign
    try:
        implied_vol = brentq(objective, 1e-3, 30.0)  # 0.001% to 300% volatility
        return implied_vol
    except ValueError:
        print("No root found in the interval.")
        return None  # No root found in the interval


def calculate_implied_volatility(row):
    # Extract the market price from the options chain
    T = ((datetime.fromisoformat(row['expirationDate']).replace(tzinfo=ZoneInfo("America/New_York")) - datetime.now(ZoneInfo("America/New_York"))).days+1) / 365.0  # Time to expiration in years
    K = row['strike']
    S=row['currentPrice']
    r = risk_free_rate
    market_price = row['lastPrice']
    n = min(T* 365*20,500)
    option_type = row['optionType'].lower()  

    IV = implied_vol_binomial_tree(S, K, T, r, market_price, n, option_type)
    row['impliedVolatility_cal'] = IV
    return IV

#get appl current price
calls_chain_df = calls_chain_df[(calls_chain_df['lastTradeDate'] >= '2025-05-07') & (calls_chain_df['volume'] > 100) & (calls_chain_df['expirationDate'] > '2025-05-12')]
from tqdm import tqdm
calls_chain_df['optionType'] = 'call'
calls_chain_df['currentPrice'] = float(current_price)

calls_chain_df['daysToExpiration'] = calls_chain_df.apply(lambda x:((datetime.fromisoformat(x['expirationDate']).replace(tzinfo=ZoneInfo("America/New_York")) - datetime.now(ZoneInfo("America/New_York"))).days+1), axis=1)

def safe_calculate_iv(row):
    try:
        return calculate_implied_volatility(row)
    except Exception as e:
        print(f"Error calculating IV for row {row.name}: {e}")
        return np.nan
    
from multiprocessing.dummy import Pool as ThreadPool  # thread-based pool
from tqdm import tqdm

# Use list of rows
rows = [row for _, row in calls_chain_df.iterrows()]

# Run with threads
with ThreadPool(8) as pool:  # Or use os.cpu_count()
    results = list(tqdm(pool.imap(safe_calculate_iv, rows), total=len(rows)))

calls_chain_df['impliedVolatility_cal'] = results

calls_chain_df['daysToExpiration'] = calls_chain_df['daysToExpiration']/365
calls_chain_df['impliedVolatility_cal'] = calls_chain_df['impliedVolatility_cal']*100
calls_chain_df['strike'] = calls_chain_df['strike']/current_price


#plot a 3d graph with the x axis as the strike price/current price, y axis as the days to expiration and z axis as the implied volatility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot the implied volatility sarface 
fig = plt.figure(figsize=(10, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(calls_chain_df['strike'], calls_chain_df['daysToExpiration'], calls_chain_df['impliedVolatility_cal'], cmap='viridis',
                       linewidth=0.1, antialiased=True,
                       alpha=0.8)
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar.set_label('Implied Volatility', rotation=270, labelpad=15)
ax.set_xlabel('Strike Price')
ax.set_ylabel('Days to Expiration')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')
plt.show()