import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def market_structure_plotly(df, len_period=50, short_len_period=3,
                             bull_color='#089981', bear_color='#ff5252',
                             idm_color='gray', sweeps_color='gray',
                             show_choch=True, show_bos=True,
                             show_idm=True, show_sweeps=True,
                             show_swings=True):
    """
    Detects market structure elements (CHoCH, BOS, IDM, Sweeps) and plots them using Plotly.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        len_period (int): Period for CHoCH detection.
        short_len_period (int): Period for IDM detection.
        bull_color (str): Color for bullish elements.
        bear_color (str): Color for bearish elements.
        idm_color (str): Color for inducement elements.
        sweeps_color (str): Color for sweeps.
        show_choch (bool): Show Change of Character.
        show_bos (bool): Show Break of Structure.
        show_idm (bool): Show Inducements.
        show_sweeps (bool): Show Sweeps.
        show_swings (bool): Show Swing Highs/Lows.

    Returns:
        plotly.graph_objects.Figure: Plotly figure with market structure elements.
    """

    # Create numeric index for calculations
    n = pd.Series(range(len(df)), index=df.index)

    def swings(length):
        os = 0
        topx = [None] * len(df)
        btmx = [None] * len(df)
        top = [None] * len(df)
        btm = [None] * len(df)

        upper = df['High'].rolling(window=length).max()
        lower = df['Low'].rolling(window=length).min()

        os_values = [0] * len(df)
        for i in range(length, len(df)):
            if df['High'].iloc[i] > upper.iloc[i]:
                os_values[i] = 0
            elif df['Low'].iloc[i] < lower.iloc[i]:
                os_values[i] = 1
            else:
                os_values[i] = os_values[i-1]
            os = os_values[i]

            if os == 0 and os_values[i-1] != 0:
                top[i] = df['High'].iloc[i]
                topx[i] = n.iloc[i]
            if os == 1 and os_values[i-1] != 1:
                btm[i] = df['Low'].iloc[i]
                btmx[i] = n.iloc[i]
        return top, topx, btm, btmx

    top, topx, btm, btmx = swings(len_period)
    stop, stopx, sbtm, sbtmx = swings(short_len_period)

    os = 0
    top_crossed = False
    btm_crossed = False

    max_price = None
    min_price = None

    max_x1 = None
    min_x1 = None

    topy = None
    btmy = None
    stop_crossed_short = False
    sbtm_crossed_short = False

    market_structure_events = []

    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_n = n.iloc[i]

        prev_os = os
        prev_top_crossed = top_crossed
        prev_btm_crossed = btm_crossed

        if top[i] is not None:
            topy = top[i]
            top_crossed = False
        if btm[i] is not None:
            btmy = btm[i]
            btm_crossed = False

        # Test for CHoCH
        if topy is not None and current_close > topy and not top_crossed:
            os = 1
            top_crossed = True
            market_structure_events.append({'type': 'CHoCH', 'price': topy, 'time': i, 'direction': 'bullish'})
        if btmy is not None and current_close < btmy and not btm_crossed:
            os = 0
            btm_crossed = True
            market_structure_events.append({'type': 'CHoCH', 'price': btmy, 'time': i, 'direction': 'bearish'})

        if os != prev_os:
            max_price = df['High'].iloc[i]
            min_price = df['Low'].iloc[i]
            max_x1 = current_n
            min_x1 = current_n
            stop_crossed_short = False
            sbtm_crossed_short = False

        stopy_val = stop[i] if stop[i] is not None else (stop[i-1] if i > 0 else None)
        sbtmy_val = sbtm[i] if sbtm[i] is not None else (sbtm[i-1] if i > 0 else None)
        btmy_long = btm[i-len_period] if i >= len_period and btm[i-len_period] is not None else btmy
        topy_long = top[i-len_period] if i >= len_period and top[i-len_period] is not None else topy

        # Bullish BOS
        if sbtmy_val is not None and current_low < sbtmy_val and not sbtm_crossed_short and os == 1 and sbtmy_val != btmy_long:
            market_structure_events.append({'type': 'IDM', 'price': sbtmy_val, 'time': i, 'direction': 'bullish'})
            sbtm_crossed_short = True

        if max_price is not None and current_close > max_price and sbtm_crossed_short and os == 1:
            market_structure_events.append({'type': 'BOS', 'price': max_price, 'time': i, 'direction': 'bullish'})
            sbtm_crossed_short = False

        # Bearish BOS
        if stopy_val is not None and current_high > stopy_val and not stop_crossed_short and os == 0 and stopy_val != topy_long:
            market_structure_events.append({'type': 'IDM', 'price': stopy_val, 'time': i, 'direction': 'bearish'})
            stop_crossed_short = True

        if min_price is not None and current_close < min_price and stop_crossed_short and os == 0:
            market_structure_events.append({'type': 'BOS', 'price': min_price, 'time': i, 'direction': 'bearish'})
            stop_crossed_short = False

        # Sweeps
        if max_price is not None and current_high > max_price and current_close < max_price and os == 1 and current_n - max_x1 > 1 and show_sweeps:
            market_structure_events.append({'type': 'Sweep', 'price': max_price, 'time': i, 'direction': 'bearish'})

        if min_price is not None and current_low < min_price and current_close > min_price and os == 0 and current_n - min_x1 > 1 and show_sweeps:
            market_structure_events.append({'type': 'Sweep', 'price': min_price, 'time': i, 'direction': 'bullish'})

        # Trailing max/min
        if max_price is None or current_high > max_price:
            max_price = current_high
            max_x1 = current_n
        if min_price is None or current_low < min_price:
            min_price = current_low
            min_x1 = current_n

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # Candlestick trace
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'), row=1, col=1)

    # Plot Swing Highs
    if show_swings:
        swing_high_indices = [i for i, val in enumerate(top) if val is not None]
        swing_high_prices = [top[i] for i in swing_high_indices]
        swing_high_times = df.index[swing_high_indices]
        fig.add_trace(go.Scatter(x=swing_high_times, y=swing_high_prices, mode='markers',
                                 marker=dict(symbol='circle', size=10, color=bear_color, opacity=0.5),
                                 name='Swing High'), row=1, col=1)

    # Plot Swing Lows
    if show_swings:
        swing_low_indices = [i for i, val in enumerate(btm) if val is not None]
        swing_low_prices = [btm[i] for i in swing_low_indices]
        swing_low_times = df.index[swing_low_indices]
        fig.add_trace(go.Scatter(x=swing_low_times, y=swing_low_prices, mode='markers',
                                 marker=dict(symbol='circle', size=10, color=bull_color, opacity=0.5),
                                 name='Swing Low'), row=1, col=1)

    # Plot Market Structure Events
    for event in market_structure_events:
        time = df.index[event['time']]
        price = event['price']
        event_type = event['type']
        direction = event['direction']

        if event_type == 'CHoCH' and show_choch:
            color = bull_color if direction == 'bullish' else bear_color
            text = 'CHoCH'
            y_offset = -10 if direction == 'bullish' else 10
            fig.add_annotation(x=time, y=price, text=text, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                               arrowcolor=color, ax=0, ay=y_offset, font=dict(color=color, size=10))
            # Draw dashed line at CHoCH level (simplified extension)
            fig.add_shape(type="line", x0=time, y0=price, x1=df.index[-1], y1=price,
                          line=dict(color=color, width=1, dash='dash'))

        elif event_type == 'BOS' and show_bos:
            color = bull_color if direction == 'bullish' else bear_color
            text = 'BOS'
            y_offset = -10 if direction == 'bullish' else 10
            fig.add_annotation(x=time, y=price, text=text, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                               arrowcolor=color, ax=0, ay=y_offset, font=dict(color=color, size=10))
            # Draw solid line at BOS level (simplified extension)
            fig.add_shape(type="line", x0=time, y0=price, x1=df.index[-1], y1=price,
                          line=dict(color=color, width=2))

        elif event_type == 'IDM' and show_idm:
            color = idm_color
            text = 'IDM'
            y_offset = 10 if direction == 'bullish' else -10
            fig.add_annotation(x=time, y=price, text=text, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                               arrowcolor=color, ax=0, ay=y_offset, font=dict(color=color, size=10))
            # Draw dotted line at IDM level (simplified extension)
            fig.add_shape(type="line", x0=time, y0=price, x1=df.index[-1], y1=price,
                          line=dict(color=color, width=1, dash='dot'))

        elif event_type == 'Sweep' and show_sweeps:
            color = sweeps_color
            text = 'x'
            y_offset = 10 if direction == 'bullish' else -10
            fig.add_annotation(x=time, y=price, text=text, showarrow=False, font=dict(color=color, size=12))
            # Draw dotted line at Sweep level (simplified extension)
            fig.add_shape(type="line", x0=time, y0=price, x1=df.index[-1], y1=price,
                          line=dict(color=color, width=1, dash='dot'))

    fig.update_layout(title='Market Structure with Inducements & Sweeps',
                      xaxis_title='Time',
                      yaxis_title='Price')

    return fig

if __name__ == '__main__':
    # Load your dataset from a CSV file
    try:
        df = pd.read_csv('ETHUSDT3Month5min.csv', index_col=0, parse_dates=True)
        # Ensure your CSV has 'Open', 'High', 'Low', 'Close' columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            raise ValueError("CSV file must contain 'Open', 'High', 'Low', and 'Close' columns.")

        # Example usage of the function
        fig = market_structure_plotly(df)
        fig.show()

    except FileNotFoundError:
        print("Error: 'ETHUSDT3Month5min.csv' not found. Please replace with your actual file path.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}") 