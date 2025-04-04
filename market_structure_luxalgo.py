import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class MarketStructure:
    def __init__(self, df, len=50, short_len=3):
        """
        Initialize Market Structure indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns: High, Low, Close, Open
        len : int
            CHoCH Detection Period
        short_len : int
            IDM Detection Period
        """
        self.df = df.copy()
        self.len = len
        self.short_len = short_len
        self.n = np.arange(len(df))
        
    def swings(self, length):
        """Calculate swings"""
        os = np.zeros(len(self.df))
        topx = np.full(len(self.df), np.nan)
        btmx = np.full(len(self.df), np.nan)
        top = np.full(len(self.df), np.nan)
        btm = np.full(len(self.df), np.nan)
        
        for i in range(length, len(self.df)):
            upper = self.df['High'].iloc[i-length:i].max()
            lower = self.df['Low'].iloc[i-length:i].min()
            
            if self.df['High'].iloc[i-length] > upper:
                os[i] = 0
            elif self.df['Low'].iloc[i-length] < lower:
                os[i] = 1
            else:
                os[i] = os[i-1]
                
            if os[i] == 0 and os[i-1] != 0:
                top[i] = self.df['High'].iloc[i-length]
                topx[i] = self.n[i-length]
            if os[i] == 1 and os[i-1] != 1:
                btm[i] = self.df['Low'].iloc[i-length]
                btmx[i] = self.n[i-length]
                
        return top, topx, btm, btmx
    
    def calculate(self):
        """Calculate market structure"""
        # Calculate swings
        top, topx, btm, btmx = self.swings(self.len)
        stop, stopx, sbtm, sbtmx = self.swings(self.short_len)
        
        # Initialize variables
        os = np.zeros(len(self.df))
        top_crossed = np.zeros(len(self.df), dtype=bool)
        btm_crossed = np.zeros(len(self.df), dtype=bool)
        max_price = np.full(len(self.df), np.nan)
        min_price = np.full(len(self.df), np.nan)
        max_x1 = np.full(len(self.df), np.nan)
        min_x1 = np.full(len(self.df), np.nan)
        topy = np.full(len(self.df), np.nan)
        btmy = np.full(len(self.df), np.nan)
        stop_crossed = np.zeros(len(self.df), dtype=bool)
        sbtm_crossed = np.zeros(len(self.df), dtype=bool)
        
        # Calculate CHoCH and BOS
        for i in range(self.len, len(self.df)):
            if not np.isnan(top[i]):
                topy[i] = top[i]
                top_crossed[i] = False
            if not np.isnan(btm[i]):
                btmy[i] = btm[i]
                btm_crossed[i] = False
                
            if self.df['Close'].iloc[i] > topy[i] and not top_crossed[i]:
                os[i] = 1
                top_crossed[i] = True
            if self.df['Close'].iloc[i] < btmy[i] and not btm_crossed[i]:
                os[i] = 0
                btm_crossed[i] = True
                
            if os[i] != os[i-1]:
                max_price[i] = self.df['High'].iloc[i]
                min_price[i] = self.df['Low'].iloc[i]
                max_x1[i] = self.n[i]
                min_x1[i] = self.n[i]
                stop_crossed[i] = False
                sbtm_crossed[i] = False
                
            # Update max/min
            if i > 0:
                max_price[i] = max(self.df['High'].iloc[i], max_price[i-1])
                min_price[i] = min(self.df['Low'].iloc[i], min_price[i-1])
                
                if max_price[i] > max_price[i-1]:
                    max_x1[i] = self.n[i]
                if min_price[i] < min_price[i-1]:
                    min_x1[i] = self.n[i]
                    
        return {
            'os': os,
            'top': top,
            'btm': btm,
            'topy': topy,
            'btmy': btmy,
            'max_price': max_price,
            'min_price': min_price,
            'max_x1': max_x1,
            'min_x1': min_x1,
            'stop_crossed': stop_crossed,
            'sbtm_crossed': sbtm_crossed
        }
    
    def plot(self, results):
        """Plot the market structure"""
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='OHLC'
        ))
        
        # Add CHoCH lines
        for i in range(self.len, len(self.df)):
            if results['os'][i] != results['os'][i-1]:
                if results['os'][i] == 1:
                    fig.add_trace(go.Scatter(
                        x=[self.df.index[int(results['max_x1'][i])], self.df.index[i]],
                        y=[results['max_price'][i], results['max_price'][i]],
                        mode='lines',
                        line=dict(color='#089981', dash='dash'),
                        name='CHoCH'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[self.df.index[int(results['min_x1'][i])], self.df.index[i]],
                        y=[results['min_price'][i], results['min_price'][i]],
                        mode='lines',
                        line=dict(color='#ff5252', dash='dash'),
                        name='CHoCH'
                    ))
        
        # Add swing points
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=results['top'],
            mode='markers',
            marker=dict(color='#ff5252', size=8, symbol='circle'),
            name='Swing High'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=results['btm'],
            mode='markers',
            marker=dict(color='#089981', size=8, symbol='circle'),
            name='Swing Low'
        ))
        
        # Update layout
        fig.update_layout(
            title='Market Structure with Inducements & Sweeps [LuxAlgo]',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            showlegend=True
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_data.csv')
    # df['Date'] = pd.to_datetime(df['Date'])
    # df.set_index('Date', inplace=True)
    
    # Create MarketStructure instance
    # ms = MarketStructure(df)
    
    # Calculate results
    # results = ms.calculate()
    
    # Plot results
    # fig = ms.plot(results)
    # fig.show() 