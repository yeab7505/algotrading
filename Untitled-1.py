# %%
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go

# %%
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

def supertrend(df, periods=14, multiplier=3.0):
    """
    Calculate Supertrend using pandas_ta library and generate signals.
    
    Returns a DataFrame with:
      - supertrend: The Supertrend line
      - buy_signal: Boolean series where True indicates a buy signal
      - sell_signal: Boolean series where True indicates a sell signal
      - trend: The raw trend direction from Supertrend indicator
      - up / dn: The bands from the indicator (for reference)
      - atr and tr for further filtering or analysis.
    """
    st = ta.supertrend(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        length=periods,
        multiplier=multiplier
    )
    
    supertrend_col = f"SUPERT_{periods}_{multiplier}"
    direction_col = f"SUPERTd_{periods}_{multiplier}"
    lower_band_col = f"SUPERTl_{periods}_{multiplier}"
    upper_band_col = f"SUPERTs_{periods}_{multiplier}"

    # Create initial raw signals: buy when direction flips positive and sell when it flips negative
    buy_signal = (st[direction_col] > 0) & (st[direction_col].shift(1) < 0)
    sell_signal = (st[direction_col] < 0) & (st[direction_col].shift(1) > 0)

    atr = ta.atr(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        length=periods,
        mamode='rma'
    )
    tr = ta.true_range(
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )

    return pd.DataFrame({
        'supertrend': st[supertrend_col],
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'trend': st[direction_col],
        'up': st[lower_band_col],  # Lower band (or “up” line)
        'dn': st[upper_band_col],  # Upper band (or “dn” line)
        'atr': atr,
        'tr': tr
    }, index=df.index)

def filter_false_signals(df, signals, confirmation_period=1):
    """
    Example filter to reduce false signals.
    
    Confirms a buy signal only if the closing price stays above 
    the Supertrend for the next 'confirmation_period' bars (and similarly for sell signals).
    """
    filtered_buy = signals['buy_signal'].copy()
    filtered_sell = signals['sell_signal'].copy()
    
    for i in range(len(signals) - confirmation_period):
        if signals['buy_signal'].iloc[i]:
            if not all(df['Close'].iloc[i+1:i+1+confirmation_period] > signals['supertrend'].iloc[i+1:i+1+confirmation_period]):
                filtered_buy.iloc[i] = False
        if signals['sell_signal'].iloc[i]:
            if not all(df['Close'].iloc[i+1:i+1+confirmation_period] < signals['supertrend'].iloc[i+1:i+1+confirmation_period]):
                filtered_sell.iloc[i] = False

    signals['buy_signal'] = filtered_buy
    signals['sell_signal'] = filtered_sell
    return signals

def calculate_dynamic_trade_levels(entry_price, atr, direction, 
                                   historical_data=None, 
                                   risk_reward_ratio=2.0, 
                                   atr_sl_multiplier=1.5, 
                                   atr_tp_multiplier=3.0, 
                                   trailing_stop_enabled=False, 
                                   trailing_multiplier=1.0, 
                                   lookback=10):
    """
    Calculate dynamic take profit, stop loss, and optional trailing stop levels.
    
    Blends ATR-based volatility, recent price swings, and a desired risk-reward ratio.
    
    Parameters:
        entry_price (float): Price at which the trade is entered.
        atr (float): ATR value at entry.
        direction (str): 'long' or 'short'.
        historical_data (pd.DataFrame or pd.Series, optional): Recent price data used to refine SL.
        risk_reward_ratio (float): Desired risk-reward ratio.
        atr_sl_multiplier (float): ATR multiplier for initial stop loss.
        atr_tp_multiplier (float): ATR multiplier for initial take profit.
        trailing_stop_enabled (bool): If True, calculate a trailing stop distance.
        trailing_multiplier (float): ATR multiplier for trailing stop.
        lookback (int): Number of periods to look back for swing levels.
    
    Returns:
        dict: { 'take_profit': ..., 'stop_loss': ..., 'trailing_stop_distance': ... }
    """
    # Step 1: ATR-based initialization
    if direction.lower() == 'long':
        initial_sl = entry_price - atr * atr_sl_multiplier
        initial_tp = entry_price + atr * atr_tp_multiplier
    elif direction.lower() == 'short':
        initial_sl = entry_price + atr * atr_sl_multiplier
        initial_tp = entry_price - atr * atr_tp_multiplier
    else:
        raise ValueError("Direction must be 'long' or 'short'")
    
    # Step 2: Adjust SL using recent swing levels if historical data is provided
    if historical_data is not None:
        if isinstance(historical_data, pd.DataFrame):
            if direction.lower() == 'long' and 'Low' in historical_data.columns:
                recent_swing_low = historical_data['Low'].tail(lookback).min()
                adjusted_sl = recent_swing_low if recent_swing_low > initial_sl else initial_sl
            elif direction.lower() == 'short' and 'High' in historical_data.columns:
                recent_swing_high = historical_data['High'].tail(lookback).max()
                adjusted_sl = recent_swing_high if recent_swing_high < initial_sl else initial_sl
            else:
                adjusted_sl = initial_sl
        else:
            if direction.lower() == 'long':
                recent_swing_low = historical_data.tail(lookback).min()
                adjusted_sl = recent_swing_low if recent_swing_low > initial_sl else initial_sl
            elif direction.lower() == 'short':
                recent_swing_high = historical_data.tail(lookback).max()
                adjusted_sl = recent_swing_high if recent_swing_high < initial_sl else initial_sl
    else:
        adjusted_sl = initial_sl

    # Step 3: Recalculate TP based on risk-reward ratio
    if direction.lower() == 'long':
        recalculated_tp = entry_price + (entry_price - adjusted_sl) * risk_reward_ratio
        final_tp = min(initial_tp, recalculated_tp)
    else:
        recalculated_tp = entry_price - (adjusted_sl - entry_price) * risk_reward_ratio
        final_tp = max(initial_tp, recalculated_tp)
    
    # Step 4: Calculate trailing stop distance if enabled
    trailing_distance = atr * trailing_multiplier if trailing_stop_enabled else None

    return {
        'take_profit': final_tp,
        'stop_loss': adjusted_sl,
        'trailing_stop_distance': trailing_distance
    }

def backtest_dynamic_levels(df, signals, risk_reward_ratio=2.0, atr_sl_multiplier=1.5, 
                            atr_tp_multiplier=3.0, lookback=10):
    """
    Backtests the strategy using dynamic TP/SL levels.
    
    When a buy signal occurs, a long position is opened. At entry the ATR and historical data 
    (up to that point) are used to compute dynamic stop loss and take profit levels. Then, for each 
    subsequent bar the trade is checked:
      - For a long trade:
          * If the bar's Low goes below the SL, exit at the SL price.
          * If the bar's High goes above the TP, exit at the TP price.
          * Additionally, if a sell signal is generated, exit at the bar's close.
    
    If the trade is still open at the end, it is closed at the last available price.
    
    Returns a list of trade details.
    """
    trades = []
    position = None  # None if no open trade
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    for i in range(len(df)):
        if position is None:
            # Open a long trade on a confirmed buy signal
            if signals['buy_signal'].iloc[i]:
                entry_price = df['Close'].iloc[i]
                atr_value = signals['atr'].iloc[i]
                # Use historical data up to (and including) current bar for swing level calculation
                historical = df.iloc[:i+1]
                levels = calculate_dynamic_trade_levels(entry_price, atr_value, 'long',
                                                        historical_data=historical,
                                                        risk_reward_ratio=risk_reward_ratio,
                                                        atr_sl_multiplier=atr_sl_multiplier,
                                                        atr_tp_multiplier=atr_tp_multiplier,
                                                        trailing_stop_enabled=False,
                                                        lookback=lookback)
                position = {
                    'entry_index': i,
                    'entry_date': df['Datetime'].iloc[i],
                    'entry_price': entry_price,
                    'stop_loss': levels['stop_loss'],
                    'take_profit': levels['take_profit'],
                    'atr': atr_value
                }
        else:
            # Process the open trade on each new bar
            current_bar = df.iloc[i]
            exit_price = None
            exit_reason = None
            
            # For a long position: check if the bar's low or high triggers SL/TP.
            if current_bar['Low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_reason = 'stop_loss'
            elif current_bar['High'] >= position['take_profit']:
                exit_price = position['take_profit']
                exit_reason = 'take_profit'
            
            # Also exit on a contrary sell signal (if not already exited)
            if signals['sell_signal'].iloc[i] and exit_price is None:
                exit_price = current_bar['Close']
                exit_reason = 'signal_exit'
            
            if exit_price is not None:
                trade = {
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': current_bar['Datetime'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'profit': exit_price - position['entry_price']
                }
                trades.append(trade)
                position = None
                
    if position is not None:
        last_price = df['Close'].iloc[-1]
        trade = {
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': df['Datetime'].iloc[-1],
            'exit_price': last_price,
            'exit_reason': 'end_of_data',
            'profit': last_price - position['entry_price']
        }
        trades.append(trade)
    
    return trades

def backtest_cumulative_returns(df, signals):
    """
    Calculates the cumulative returns of the strategy compared to a buy & hold strategy.
    """
    df = df.copy()
    df['position'] = 0
    in_position = 0
    for i in range(len(df)):
        if signals['buy_signal'].iloc[i]:
            in_position = 1
        elif signals['sell_signal'].iloc[i]:
            in_position = 0
        df.loc[df.index[i], 'position'] = in_position
    df['daily_return'] = df['Close'].pct_change().fillna(0)
    df['strategy_return'] = df['daily_return'] * df['position'].shift(1).fillna(0)
    df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()
    df['cumulative_buy_hold_return'] = (1 + df['daily_return']).cumprod()
    return df

def plot_supertrend(df, supertrend_df):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if 'Open' not in df.columns:
        df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    price_trace = go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )
    supertrend_line = go.Scatter(
        x=df['Datetime'],
        y=supertrend_df['supertrend'],
        mode='lines',
        name='Supertrend',
        line=dict(color='blue')
    )
    buy_signals = supertrend_df[supertrend_df['buy_signal']]
    buy_markers = go.Scatter(
        x=df.loc[buy_signals.index, 'Datetime'],
        y=buy_signals['supertrend'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10),
        name='Buy Signal'
    )
    sell_signals = supertrend_df[supertrend_df['sell_signal']]
    sell_markers = go.Scatter(
        x=df.loc[sell_signals.index, 'Datetime'],
        y=sell_signals['supertrend'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10),
        name='Sell Signal'
    )
    fig = go.Figure(data=[price_trace, supertrend_line, buy_markers, sell_markers])
    fig.update_layout(
        title='Supertrend Indicator',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[df['Low'].min() * 0.9, df['High'].max() * 1.1])
    )
    fig.show()

def plot_performance(df_backtest):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_backtest['Datetime'],
        y=df_backtest['cumulative_strategy_return'],
        mode='lines',
        name='Strategy Return'
    ))
    fig.add_trace(go.Scatter(
        x=df_backtest['Datetime'],
        y=df_backtest['cumulative_buy_hold_return'],
        mode='lines',
        name='Buy & Hold Return'
    ))
    fig.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        xaxis_rangeslider_visible=False
    )
    fig.show()
import numpy as np
import pandas as pd

def optimize_strategy(df,
                      st_periods_range,
                      st_multiplier_range,
                      confirmation_period_range,
                      risk_reward_ratio_range,
                      atr_sl_multiplier_range,
                      atr_tp_multiplier_range,
                      lookback_range):
    """
    Optimize key strategy parameters using grid search.
    
    Parameters:
      df: DataFrame with price data.
      st_periods_range: List of values for the Supertrend period.
      st_multiplier_range: List of values for the Supertrend multiplier.
      confirmation_period_range: List of confirmation periods for filtering signals.
      risk_reward_ratio_range: List of risk-reward ratios.
      atr_sl_multiplier_range: List of ATR multipliers for stop loss.
      atr_tp_multiplier_range: List of ATR multipliers for take profit.
      lookback_range: List of lookback values for dynamic level adjustment.
      
    Returns:
      best_params: Dictionary of the best parameter combination.
      all_results: List of dictionaries with the parameters and the final cumulative return.
    """
    best_metric = -np.inf
    best_params = None
    all_results = []
    
    # Iterate through all combinations of parameters
    for periods in st_periods_range:
        for multiplier in st_multiplier_range:
            # Generate initial signals using the Supertrend function
            signals = supertrend(df, periods=periods, multiplier=multiplier)
            for confirmation in confirmation_period_range:
                # Filter out false signals
                signals_filtered = filter_false_signals(df, signals.copy(), confirmation_period=confirmation)
                for risk_reward_ratio in risk_reward_ratio_range:
                    for atr_sl_multiplier in atr_sl_multiplier_range:
                        for atr_tp_multiplier in atr_tp_multiplier_range:
                            for lookback in lookback_range:
                                # Run backtest using the dynamic TP/SL backtesting function
                                trades = backtest_dynamic_levels(df, signals_filtered,
                                                                 risk_reward_ratio=risk_reward_ratio,
                                                                 atr_sl_multiplier=atr_sl_multiplier,
                                                                 atr_tp_multiplier=atr_tp_multiplier,
                                                                 lookback=lookback)
                                # Use backtest cumulative returns as a performance metric.
                                # If no trades were taken, default performance is 1.0 (i.e. no gain/loss).
                                if trades:
                                    df_backtest = backtest_cumulative_returns(df, signals_filtered)
                                    final_return = df_backtest['cumulative_strategy_return'].iloc[-1]
                                else:
                                    final_return = 1.0
                                
                                params = {
                                    'periods': periods,
                                    'multiplier': multiplier,
                                    'confirmation': confirmation,
                                    'risk_reward_ratio': risk_reward_ratio,
                                    'atr_sl_multiplier': atr_sl_multiplier,
                                    'atr_tp_multiplier': atr_tp_multiplier,
                                    'lookback': lookback,
                                    'final_return': final_return,
                                    'num_trades': len(trades)
                                }
                                all_results.append(params)
                                if final_return > best_metric:
                                    best_metric = final_return
                                    best_params = params
    return best_params, all_results

# === Example usage ===
if __name__ == "__main__":
    # Load your data (make sure the CSV file exists in your working directory)
    df = pd.read_csv('bitcoin45days30.csv')
    
    # Define parameter ranges (these are examples; adjust ranges as needed)
    st_periods_range = [10, 14, 20]
    st_multiplier_range = [2.0, 3.0, 4.0]
    confirmation_period_range = [1, 3, 5]
    risk_reward_ratio_range = [1.5, 2.0, 3.0]
    atr_sl_multiplier_range = [1.0, 1.5, 2.0]
    atr_tp_multiplier_range = [2.0, 3.0, 4.0]
    lookback_range = [5, 10, 15]
    
    best_params, results = optimize_strategy(df,
                                             st_periods_range,
                                             st_multiplier_range,
                                             confirmation_period_range,
                                             risk_reward_ratio_range,
                                             atr_sl_multiplier_range,
                                             atr_tp_multiplier_range,
                                             lookback_range)
    
    print("Best Parameters Found:")
    print(best_params)
    
    # (Optional) Further processing: Convert results to a DataFrame for analysis
    results_df = pd.DataFrame(results)
    print("\nAll Optimization Results:")
    print(results_df.sort_values(by='final_return', ascending=False).head())


# %%
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Supertrend:
    def __init__(self, data, period, multiplier):
        self.data = data
        self.period = period
        self.multiplier = multiplier
        self.df = None  # Initialize df as None
        self.ADX = None  # Initialize ADX as None

    def prepare_data(self):  # Renamed to follow Python conventions
        if isinstance(self.data, str):
            self.df = pd.read_csv(self.data)
        elif isinstance(self.data, pd.DataFrame):
            self.df = self.data.copy()
        self.df['return'] = self.df['Close'].pct_change()
        # Ensure Date column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Datetime'])
    def calculate_double_ema(self):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        self.df['DEMA'] = ta.dema(self.df['Close'], 200)
    
    def calculate_ADX(self):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        self.ADX = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], 14)
        return self.ADX

    def supertrend(self):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        df = self.df
        st = ta.supertrend(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            length=self.period,
            multiplier=self.multiplier
        )

        supertrend_col = f"SUPERT_{self.period}_{self.multiplier:.1f}"
        direction_col = f"SUPERTd_{self.period}_{self.multiplier:.1f}"
        lower_band_col = f"SUPERTl_{self.period}_{self.multiplier:.1f}"
        upper_band_col = f"SUPERTs_{self.period}_{self.multiplier:.1f}"

        df[supertrend_col] = st[supertrend_col]
        df[lower_band_col] = st[lower_band_col]
        df[upper_band_col] = st[upper_band_col]

        # Align the ADX DataFrame with the main DataFrame
        self.ADX, df = self.ADX.align(df, axis=0, copy=False)

        # Use the specific ADX column for the condition
        df['buy'] = (st[direction_col] > 0) & (st[direction_col].shift(1) < 0) & (self.ADX['ADX_14'] > 0)
        df['close_buy']=(st[direction_col] < 0) & (st[direction_col].shift(1) > 0) 
        df['sell'] = (st[direction_col] < 0) & (st[direction_col].shift(1) > 0) & (self.ADX['ADX_14'] > 0)
        df['close_sell']=(st[direction_col] > 0) & (st[direction_col].shift(1) < 0)
        
        return df

    def plot(self, supertrend_df):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        if 'buy' not in supertrend_df.columns or 'sell' not in supertrend_df.columns:
            raise ValueError("Run supertrend() first to generate signals.")

        df = self.df
        price_trace = go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )
        double_ema_trace = go.Scatter(
            x=df['Datetime'],
            y=df['DEMA'],
            mode='lines',
            name='DEMA',
            line=dict(color='orange')
        )

        supertrend_col = f"SUPERT_{self.period}_{self.multiplier:.1f}"
        supertrend_line = go.Scatter(
            x=df['Datetime'],
            y=supertrend_df[supertrend_col],
            mode='lines',
            name='Supertrend',
            line=dict(color='blue')
        )

        buy_signals = supertrend_df[supertrend_df['buy']]
        buy_markers = go.Scatter(
            x=df.loc[buy_signals.index, 'Datetime'],
            y=buy_signals[supertrend_col],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy Signal'
        )

        close_buy_signals = supertrend_df[supertrend_df['close_buy']]
        close_buy_markers = go.Scatter(
            x=df.loc[close_buy_signals.index, 'Datetime'],
            y=close_buy_signals[supertrend_col],
            mode='markers',
            marker=dict(symbol='x', color='green', size=10),
            name='Close Buy Signal'
        )

        sell_signals = supertrend_df[supertrend_df['sell']]
        sell_markers = go.Scatter(
            x=df.loc[sell_signals.index, 'Datetime'],
            y=sell_signals[supertrend_col],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell Signal'
        )

        close_sell_signals = supertrend_df[supertrend_df['close_sell']]
        close_sell_markers = go.Scatter(
            x=df.loc[close_sell_signals.index, 'Datetime'],
            y=close_sell_signals[supertrend_col],
            mode='markers',
            marker=dict(symbol='x', color='red', size=10),
            name='Close Sell Signal'
        )

        fig = go.Figure(data=[price_trace, supertrend_line, buy_markers, sell_markers,double_ema_trace ])

        
        fig.update_layout(
            title='Supertrend Indicator',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            yaxis=dict(range=[df['Low'].min() * 0.9, df['High'].max() * 1.1])  # Set dynamic range
        )
        fig.show()

# Example usage with your data
df = pd.read_csv('bitcoin30days.csv')

# Initialize and run Supertrend
st_instance = Supertrend(df, 14, 3)  # Pass DataFrame directly
st_instance.prepare_data()
st_instance.calculate_ADX()
st_instance.calculate_double_ema()
result_df = st_instance.supertrend()
st_instance.plot(result_df)  # Pass the result DataFrame to the plot method

# Calculate counts
buy_true = result_df['buy'].sum()
sell_true = result_df['sell'].sum()
buy_false = (~result_df['buy']).sum()
sell_false = (~result_df['sell']).sum()

print(f"Buy Signals (True): {buy_true}")
print(f"Sell Signals (True): {sell_true}")
print(f"Buy Signals (False): {buy_false}")
print(f"Sell Signals (False): {sell_false}")

# Optional: Print rows where either buy or sell is True
signals = result_df[result_df['buy'] | result_df['sell']]
print("\nRows with signals:")
print(signals[['Datetime', 'Close', 'buy', 'sell']])

# %%
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product

class Supertrend:
    def __init__(self, data, period, multiplier, capital=100, tc=0.0005):
        self.data = data
        self.period = period
        self.multiplier = multiplier
        self.df = None
        self.ADX = None
        self.capital = capital
        self.tc = tc

    def prepare_data(self):
        if isinstance(self.data, str):
            self.df = pd.read_csv(self.data)
        elif isinstance(self.data, pd.DataFrame):
            self.df = self.data.copy()
        self.df['return'] = self.df['Close'].pct_change()
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])

    def calculate_ADX(self):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        self.ADX = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], 14)
        return self.ADX

    def supertrend(self):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        df = self.df
        st = ta.supertrend(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            length=self.period,
            multiplier=self.multiplier
        )

        supertrend_col = f"SUPERT_{self.period}_{self.multiplier:.1f}"
        direction_col = f"SUPERTd_{self.period}_{self.multiplier:.1f}"
        lower_band_col = f"SUPERTl_{self.period}_{self.multiplier:.1f}"
        upper_band_col = f"SUPERTs_{self.period}_{self.multiplier:.1f}"

        df[supertrend_col] = st[supertrend_col]
        df[lower_band_col] = st[lower_band_col]
        df[upper_band_col] = st[upper_band_col]

        if self.ADX is None:
            self.calculate_ADX()
        self.ADX, df = self.ADX.align(df, axis=0, copy=False)

        df['position'] = 0
        df['buy'] = False
        df['close_buy'] = False
        df['sell'] = False
        df['close_sell'] = False
        df['entry_price'] = 0.0
        df['exit_price'] = 0.0
        df['trade_return'] = 0.0

        for i in range(1, len(df)):
            prev_pos = df.at[i-1, 'position']
            trend_up = (st.at[i, direction_col] > 0) and (st.at[i-1, direction_col] < 0)
            trend_down = (st.at[i, direction_col] < 0) and (st.at[i-1, direction_col] > 0)
            adx_condition = self.ADX.at[i, 'ADX_14'] > 20

            # Buy from no position
            if trend_up and adx_condition and prev_pos == 0:
                df.at[i, 'buy'] = True
                df.at[i, 'position'] = 1
                df.at[i, 'entry_price'] = df.at[i, 'Close']
            # Sell from no position
            elif trend_down and adx_condition and prev_pos == 0:
                df.at[i, 'sell'] = True
                df.at[i, 'position'] = -1
                df.at[i, 'entry_price'] = df.at[i, 'Close']
            # Close buy and reverse to short
            elif prev_pos == 1 and trend_down:
                df.at[i, 'close_buy'] = True
                df.at[i, 'exit_price'] = df.at[i, 'Close']
                df.at[i, 'trade_return'] = ((df.at[i, 'exit_price'] / df.at[i-1, 'entry_price']) - 1) - (2 * self.tc)
                df.at[i, 'sell'] = True
                df.at[i, 'position'] = -1
                df.at[i, 'entry_price'] = df.at[i, 'Close']
            # Close sell and reverse to long
            elif prev_pos == -1 and trend_up:
                df.at[i, 'close_sell'] = True
                df.at[i, 'exit_price'] = df.at[i, 'Close']
                df.at[i, 'trade_return'] = ((df.at[i-1, 'entry_price'] / df.at[i, 'exit_price']) - 1) - (2 * self.tc)
                df.at[i, 'buy'] = True
                df.at[i, 'position'] = 1
                df.at[i, 'entry_price'] = df.at[i, 'Close']
            # Carry forward
            else:
                df.at[i, 'position'] = prev_pos
                if prev_pos != 0:
                    df.at[i, 'entry_price'] = df.at[i-1, 'entry_price']

        return df

    def calculate_returns(self, supertrend_df):
        trades = []
        df = supertrend_df
        entry_time = None
        entry_price = 0.0
        position_type = None

        for i in range(len(df)):
            if df.at[i, 'buy'] and not df.at[i, 'close_sell']:  # New long position from no position
                entry_time = df.at[i, 'Datetime']
                entry_price = df.at[i, 'entry_price']
                position_type = 'Long'
            elif df.at[i, 'sell'] and not df.at[i, 'close_buy']:  # New short position from no position
                entry_time = df.at[i, 'Datetime']
                entry_price = df.at[i, 'entry_price']
                position_type = 'Short'
            elif df.at[i, 'close_buy'] or df.at[i, 'close_sell']:  # Trade completed
                if entry_time is not None:  # Ensure there’s an open trade
                    exit_time = df.at[i, 'Datetime']
                    exit_price = df.at[i, 'exit_price']
                    trade_return = df.at[i, 'trade_return']
                    trades.append({
                        'entry_date': entry_time,
                        'exit_date': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'type': position_type,
                        'return': trade_return
                    })
                    # Set new entry for the reversed position
                    entry_time = df.at[i, 'Datetime']
                    entry_price = df.at[i, 'entry_price']
                    position_type = 'Short' if df.at[i, 'close_buy'] else 'Long'

        trades_df = pd.DataFrame(trades)
        return trades_df

    def plot(self, supertrend_df):
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        if 'buy' not in supertrend_df.columns:
            raise ValueError("Run supertrend() first to generate signals.")

        df = supertrend_df
        supertrend_col = f"SUPERT_{self.period}_{self.multiplier:.1f}"

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df['Datetime'],
                y=df[supertrend_col],
                mode='lines',
                name='Supertrend',
                line=dict(color='blue')
            )
        )
        buy_signals = df[df['buy']]
        fig.add_trace(
            go.Scatter(
                x=buy_signals['Datetime'],
                y=buy_signals[supertrend_col],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy Signal'
            )
        )
        close_buy_signals = df[df['close_buy']]
        fig.add_trace(
            go.Scatter(
                x=close_buy_signals['Datetime'],
                y=close_buy_signals[supertrend_col],
                mode='markers',
                marker=dict(symbol='x', color='green', size=10),
                name='Close Buy Signal'
            )
        )
        sell_signals = df[df['sell']]
        fig.add_trace(
            go.Scatter(
                x=sell_signals['Datetime'],
                y=sell_signals[supertrend_col],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell Signal'
            )
        )
        close_sell_signals = df[df['close_sell']]
        fig.add_trace(
            go.Scatter(
                x=close_sell_signals['Datetime'],
                y=close_sell_signals[supertrend_col],
                mode='markers',
                marker=dict(symbol='x', color='red', size=10),
                name='Close Sell Signal'
            )
        )
        fig.update_layout(
            title=f'Supertrend Indicator (Period={self.period}, Multiplier={self.multiplier})',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            yaxis=dict(range=[df['Low'].min() * 0.9, df['High'].max() * 1.1]),
            height=600,
            width=1000
        )
        fig.show()

    def optimize(self, atr_range, multiplier_range, period_range):
        combinations = list(product(atr_range, multiplier_range, period_range))
        results = []
        for combo in combinations:
            self.period = combo[2]
            self.multiplier = combo[1]
            self.atr = combo[0]
            self.prepare_data()
            self.calculate_ADX()
            result_df = self.supertrend()
            trades_df = self.calculate_returns(result_df)
            results.append({'params': combo, 'trades': trades_df})
        return results

# Example usage
df = pd.read_csv('bitcoin30days.csv')
st_instance = Supertrend(df, 10, 4, capital=100, tc=0.0005)
st_instance.prepare_data()
st_instance.calculate_ADX()
result_df = st_instance.supertrend()
st_instance.plot(result_df)
trades_df = st_instance.calculate_returns(result_df)

# Calculate counts
buy_true = result_df['buy'].sum()
sell_true = result_df['sell'].sum()
close_buy_true = result_df['close_buy'].sum()
close_sell_true = result_df['close_sell'].sum()

print(f"Buy Signals (True): {buy_true}")
print(f"Sell Signals (True): {sell_true}")
print(f"Close Buy Signals (True): {close_buy_true}")
print(f"Close Sell Signals (True): {close_sell_true}")

print("\nTrade Results:")
print(trades_df)

signals = result_df[result_df['buy'] | result_df['sell'] | result_df['close_buy'] | result_df['close_sell']]
print("\nRows with signals:")
print(signals[['Datetime', 'Close', 'buy', 'sell', 'close_buy', 'close_sell', 'position', 'entry_price', 'exit_price', 'trade_return']])

# %%
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

def supertrend(df, periods=14, multiplier=3.0):
    """
    Calculate Supertrend using pandas_ta library.
    
    Parameters:
    - df: pandas DataFrame with 'High', 'Low', 'Close'
    - periods: ATR period (default=14)
    - multiplier: ATR multiplier (default=3.0)
    
    Returns:
    - DataFrame with Supertrend values, signals, and bands
    """
    # Calculate Supertrend using pandas_ta
    st = ta.supertrend(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        length=periods,
        multiplier=multiplier
    )
    
    # Generate column names based on parameters
    supertrend_col = f"SUPERT_{periods}_{multiplier}"
    direction_col = f"SUPERTd_{periods}_{multiplier}"
    lower_band_col = f"SUPERTl_{periods}_{multiplier}"
    upper_band_col = f"SUPERTs_{periods}_{multiplier}"

    # Create signals
    buy_signal = (st[direction_col] > 0) & (st[direction_col].shift(1) < 0)
    sell_signal = (st[direction_col] < 0) & (st[direction_col].shift(1) > 0)

    # Calculate ATR and TR using pandas_ta
    atr = ta.atr(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        length=periods,
        mamode='rma'
    )
    tr = ta.true_range(
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )

    # Compile results
    return pd.DataFrame({
        'supertrend': st[supertrend_col],
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'trend': st[direction_col],
        'up': st[lower_band_col],  # Lower band in pandas_ta is 'up' in original
        'dn': st[upper_band_col],  # Upper band in pandas_ta is 'dn' in original
        'atr': atr,
        'tr': tr
    }, index=df.index)

# The plotting function remains unchanged
def plot_supertrend(df, supertrend_df):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Ensure candlestick plotting by generating Open if missing
    if 'Open' not in df.columns:
        df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    
    price_trace = go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )

    # Rest of the plotting code remains the same...
    supertrend_line = go.Scatter(
        x=df['Datetime'],
        y=supertrend_df['supertrend'],
        mode='lines',
        name='Supertrend',
        line=dict(color='blue')
    )

    buy_signals = supertrend_df[supertrend_df['buy_signal']]
    buy_markers = go.Scatter(
        x=df.loc[buy_signals.index, 'Datetime'],
        y=buy_signals['supertrend'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10),
        name='Buy Signal'
    )

    sell_signals = supertrend_df[supertrend_df['sell_signal']]
    sell_markers = go.Scatter(
        x=df.loc[sell_signals.index, 'Datetime'],
        y=sell_signals['supertrend'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10),
        name='Sell Signal'
    )

    fig = go.Figure(data=[price_trace, supertrend_line, buy_markers, sell_markers])
    fig.update_layout(
        title='Supertrend Indicator',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[df['Low'].min() * 0.9, df['High'].max() * 1.1])  # Set dynamic range
    )
    fig.show()

if __name__ == "__main__":
    df = pd.read_csv('bitcoin30days.csv')
    print("Price range:", df['Close'].min(), "to", df['Close'].max())  # Debug print
    result = supertrend(df, periods=14, multiplier=3.0)
    plot_supertrend(df, result)

# %% [markdown]
# in my dataset the columns are named as High,Low,Close,Open but in this code it is in small letter can u fix that 

# %%


# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def calculate_supertrend(df, period=10, multiplier=3.0, changeATR=True):
    """
    Calculates the Supertrend indicator.
    
    Parameters:
      df: DataFrame with columns 'Open', 'High', 'Low', 'Close'
      period: ATR period (default 10)
      multiplier: ATR multiplier (default 3.0)
      changeATR: if True, use Wilder's ATR smoothing; if False, use simple moving average of TR
      
    Returns:
      DataFrame with added columns:
        - hl2: average of high and low
        - tr: true range
        - atr: the chosen ATR
        - basic_ub / basic_lb: preliminary upper/lower bands
        - final_ub / final_lb: adjusted bands (up and dn in PineScript)
        - trend: 1 for uptrend, -1 for downtrend
        - buy_signal: True when a buy condition occurs
        - sell_signal: True when a sell condition occurs
    """
    df = df.copy()
    df['hl2'] = (df['High'] + df['Low']) / 2.0

    # Calculate True Range (TR)
    df['prev_close'] = df['Close'].shift(1)
    df['tr'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(np.abs(df['High'] - df['prev_close']),
                                     np.abs(df['Low'] - df['prev_close'])))
    # Set the first TR value as High - Low
    df.loc[df.index[0], 'tr'] = df.loc[df.index[0], 'High'] - df.loc[df.index[0], 'Low']

    # Calculate ATR: either Wilder ATR or SMA of TR based on changeATR flag
    atr = np.full(len(df), np.nan)
    if changeATR:
        # Wilder ATR smoothing: first valid ATR is the SMA of the first 'period' TR values
        for i in range(len(df)):
            if i == period - 1:
                atr[i] = df['tr'].iloc[:period].mean()
            elif i >= period:
                atr[i] = (atr[i-1] * (period - 1) + df['tr'].iloc[i]) / period
    else:
        # Use simple moving average of TR over the period (for each index, average of available values)
        atr = df['tr'].rolling(window=period, min_periods=1).mean().values
    df['atr'] = atr

    # Calculate the basic upper and lower bands (like the "up" and "dn" lines before recursive adjustments)
    df['basic_ub'] = df['hl2'] - multiplier * df['atr']
    df['basic_lb'] = df['hl2'] + multiplier * df['atr']

    # Initialize final bands (the recursive adjustment)
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            final_ub[i] = df['basic_ub'].iloc[i]
            final_lb[i] = df['basic_lb'].iloc[i]
        else:
            prev_final_ub = final_ub[i-1]
            prev_final_lb = final_lb[i-1]
            # For upper band: if previous close > previous final_ub then take the maximum; otherwise, use the basic value
            if df['Close'].iloc[i-1] > prev_final_ub:
                final_ub[i] = max(df['basic_ub'].iloc[i], prev_final_ub)
            else:
                final_ub[i] = df['basic_ub'].iloc[i]
            # For lower band: if previous close < previous final_lb then take the minimum; otherwise, use the basic value
            if df['Close'].iloc[i-1] < prev_final_lb:
                final_lb[i] = min(df['basic_lb'].iloc[i], prev_final_lb)
            else:
                final_lb[i] = df['basic_lb'].iloc[i]
    df['final_ub'] = final_ub  # corresponds to the "up" line in PineScript
    df['final_lb'] = final_lb  # corresponds to the "dn" line in PineScript

    # Determine trend direction using the recursive logic:
    # If the previous trend was down (-1) and current close > previous final_lb, then trend becomes up (1)
    # Else if previous trend was up (1) and current close < previous final_ub, then trend becomes down (-1)
    trend = np.zeros(len(df))
    trend[0] = 1  # initialize the first value as uptrend
    for i in range(1, len(df)):
        prev_trend = trend[i-1]
        if prev_trend == -1 and df['Close'].iloc[i] > final_lb[i-1]:
            trend[i] = 1
        elif prev_trend == 1 and df['Close'].iloc[i] < final_ub[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev_trend
    df['trend'] = trend

    # Generate buy/sell signals
    df['buy_signal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
    df['sell_signal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)
    
    return df

def plot_supertrend(df, showsignals=True, highlighting=True):
    """
    Plots the price candlesticks with the Supertrend indicator using Plotly.
    
    Parameters:
      df: DataFrame that includes columns: Open, High, Low, Close, final_ub, final_lb, trend, buy_signal, sell_signal
      showsignals: if True, markers for Buy/Sell signals are displayed
      highlighting: if True, colored vertical bands are added based on the trend direction
    """
    fig = go.Figure()

    # Candlestick price chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    # Supertrend lines:
    # Up trend: when trend==1, we plot final_ub; Down trend: when trend==-1, plot final_lb.
    up_line = np.where(df['trend'] == 1, df['final_ub'], np.nan)
    dn_line = np.where(df['trend'] == -1, df['final_lb'], np.nan)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=up_line,
        mode='lines',
        line=dict(color='green', width=2),
        name='Up Trend'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=dn_line,
        mode='lines',
        line=dict(color='red', width=2),
        name='Down Trend'
    ))

    # Plot Buy/Sell signals as markers
    if showsignals:
        buy_signals = df[df['buy_signal']]
        sell_signals = df[df['sell_signal']]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers+text',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            text=['Buy'] * len(buy_signals),
            textposition='bottom center',
            name='Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers+text',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            text=['Sell'] * len(sell_signals),
            textposition='top center',
            name='Sell Signal'
        ))

    # Optional highlighting: add vertical bands showing the trend (green for uptrend, red for downtrend)
    if highlighting:
        # Identify regions where the trend is constant
        trend_changes = df['trend'].diff().fillna(0) != 0
        start_idx = df.index[0]
        current_trend = df['trend'].iloc[0]
        for i in range(1, len(df)):
            if trend_changes.iloc[i]:
                end_idx = df.index[i-1]
                color = 'rgba(0, 255, 0, 0.1)' if current_trend == 1 else 'rgba(255, 0, 0, 0.1)'
                fig.add_vrect(x0=start_idx, x1=end_idx, fillcolor=color, opacity=0.3, line_width=0)
                start_idx = df.index[i]
                current_trend = df['trend'].iloc[i]
        # Add the final region
        end_idx = df.index[-1]
        color = 'rgba(0, 255, 0, 0.1)' if current_trend == 1 else 'rgba(255, 0, 0, 0.1)'
        fig.add_vrect(x0=start_idx, x1=end_idx, fillcolor=color, opacity=0.3, line_width=0)

    fig.update_layout(
        title='Supertrend Indicator',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()

# -------------------------------
# Example usage:
# -------------------------------
if __name__ == "__main__":
    # Replace this section with your own data input.
    # For example, load data from a CSV with columns: Date, Open, High, Low, Close
    # df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')
    
    # For demonstration, we generate synthetic OHLC data.
    df=pd.read_csv('bitcoin30days.csv')
    
    # Calculate the Supertrend indicator
    df_st = calculate_supertrend(df, period=10, multiplier=3.0, changeATR=True)
    
    # Plot the candlestick chart with the Supertrend overlay, buy/sell signals, and background highlighting.
    plot_supertrend(df_st, showsignals=True, highlighting=True)


# %%



