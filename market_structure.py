import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

def calculate_swings(data, length):
    """Calculate swing highs and lows"""
    highs = data['High'].rolling(window=length).max()
    lows = data['Low'].rolling(window=length).min()
    
    # Initialize arrays for swing points
    top = pd.Series(index=data.index, dtype=float)
    topx = pd.Series(index=data.index, dtype=int)
    btm = pd.Series(index=data.index, dtype=float)
    btmx = pd.Series(index=data.index, dtype=int)
    
    # Calculate swing points
    os = 0
    os_prev = 0  # Initialize os_prev
    for i in range(length, len(data)):
        if data['High'].iloc[i-length] > highs.iloc[i-1]:
            os = 0
        elif data['Low'].iloc[i-length] < lows.iloc[i-1]:
            os = 1
            
        if os == 0 and os_prev != 0:
            top.iloc[i] = data['High'].iloc[i-length]
            topx.iloc[i] = i-length
        elif os == 1 and os_prev != 1:
            btm.iloc[i] = data['Low'].iloc[i-length]
            btmx.iloc[i] = i-length
            
        os_prev = os
    
    return top, topx, btm, btmx

def detect_market_structure(data, len_choch=50, len_idm=3):
    """Detect market structure including CHoCH, BOS, and IDM"""
    # Calculate swings
    top, topx, btm, btmx = calculate_swings(data, len_choch)
    stop, stopx, sbtm, sbtmx = calculate_swings(data, len_idm)
    
    # Initialize variables
    os = pd.Series(0, index=data.index)
    top_crossed = pd.Series(False, index=data.index)
    btm_crossed = pd.Series(False, index=data.index)
    max_price = pd.Series(index=data.index, dtype=float)
    min_price = pd.Series(index=data.index, dtype=float)
    max_x1 = pd.Series(index=data.index, dtype=int)
    min_x1 = pd.Series(index=data.index, dtype=int)
    topy = pd.Series(index=data.index, dtype=float)
    btmy = pd.Series(index=data.index, dtype=float)
    stop_crossed = pd.Series(False, index=data.index)
    sbtm_crossed = pd.Series(False, index=data.index)
    
    # Calculate market structure
    for i in range(len_choch, len(data)):
        # Update topy and btmy
        if pd.notna(top.iloc[i]):
            topy.iloc[i] = top.iloc[i]
            top_crossed.iloc[i] = False
        if pd.notna(btm.iloc[i]):
            btmy.iloc[i] = btm.iloc[i]
            btm_crossed.iloc[i] = False
            
        # Test for CHoCH
        if data['Close'].iloc[i] > topy.iloc[i] and not top_crossed.iloc[i]:
            os.iloc[i] = 1
            top_crossed.iloc[i] = True
        if data['Close'].iloc[i] < btmy.iloc[i] and not btm_crossed.iloc[i]:
            os.iloc[i] = 0
            btm_crossed.iloc[i] = True
            
        # Update max/min
        if os.iloc[i] != os.iloc[i-1]:
            max_price.iloc[i] = data['High'].iloc[i]
            min_price.iloc[i] = data['Low'].iloc[i]
            max_x1.iloc[i] = i
            min_x1.iloc[i] = i
            stop_crossed.iloc[i] = False
            sbtm_crossed.iloc[i] = False
            
        # Update trailing max/min
        if i > 0:
            max_price.iloc[i] = max(data['High'].iloc[i], max_price.iloc[i-1])
            min_price.iloc[i] = min(data['Low'].iloc[i], min_price.iloc[i-1])
            
            if max_price.iloc[i] > max_price.iloc[i-1]:
                max_x1.iloc[i] = i
            if min_price.iloc[i] < min_price.iloc[i-1]:
                min_x1.iloc[i] = i
    
    return {
        'os': os,
        'top': top,
        'btm': btm,
        'stop': stop,
        'sbtm': sbtm,
        'max_price': max_price,
        'min_price': min_price,
        'max_x1': max_x1,
        'min_x1': min_x1,
        'topy': topy,
        'btmy': btmy,
        'stop_crossed': stop_crossed,
        'sbtm_crossed': sbtm_crossed
    }

def plot_market_structure(data, structure, show_choch=True, show_bos=True, show_idm=True, show_sweeps=True):
    """Plot market structure using Plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    
    # Add CHoCH lines
    if show_choch:
        for i in range(len(data)):
            if pd.notna(structure['topy'].iloc[i]):
                fig.add_trace(go.Scatter(
                    x=[data.index[i], data.index[i]],
                    y=[structure['topy'].iloc[i], structure['topy'].iloc[i]],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='CHoCH High',
                    showlegend=True
                ))
            if pd.notna(structure['btmy'].iloc[i]):
                fig.add_trace(go.Scatter(
                    x=[data.index[i], data.index[i]],
                    y=[structure['btmy'].iloc[i], structure['btmy'].iloc[i]],
                    mode='lines',
                    line=dict(color='green', dash='dash', width=2),
                    name='CHoCH Low',
                    showlegend=True
                ))
    
    # Add BOS lines
    if show_bos:
        for i in range(len(data)):
            if pd.notna(structure['max_price'].iloc[i]):
                fig.add_trace(go.Scatter(
                    x=[data.index[i], data.index[i]],
                    y=[structure['max_price'].iloc[i], structure['max_price'].iloc[i]],
                    mode='lines',
                    line=dict(color='purple', width=1),
                    name='BOS High',
                    showlegend=True
                ))
            if pd.notna(structure['min_price'].iloc[i]):
                fig.add_trace(go.Scatter(
                    x=[data.index[i], data.index[i]],
                    y=[structure['min_price'].iloc[i], structure['min_price'].iloc[i]],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='BOS Low',
                    showlegend=True
                ))
    
    # Update layout
    fig.update_layout(
        title='Market Structure Analysis',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def main():
    # Read and prepare the CSV data
    data = pd.read_csv('ETHUSDT3Month5min.csv')
    
    # Convert 'Open Time' to datetime and set as index
    data['Open Time'] = pd.to_datetime(data['Open Time'])
    data = data.set_index('Open Time')
    
    # Rename columns if necessary
    if 'Open' not in data.columns:
        data = data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        })
    
    # Calculate market structure with adjusted parameters
    structure = detect_market_structure(data, len_choch=30, len_idm=3)  # Adjusted length for better visualization
    
    # Plot results
    fig = plot_market_structure(data, structure)
    fig.show()
    
    # Save the plot
    fig.write_html("market_structure.html")

if __name__ == "__main__":
    main() 