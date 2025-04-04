# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# %%
df=pd.read_csv('BTC-USD1d.csv').dropna()
df.drop(index=0, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df

# %%
DS=1.003
#inorder to have storng confitmation we need to multiply by the DS
def is_support(df, i):
        support = df['Low'][i]*DS < df['Low'][i-1] and df['Low'][i]*DS < df['Low'][i+1] and \
                  df['Low'][i+1]*DS < df['Low'][i+2] and df['Low'][i-1]*DS < df['Low'][i-2]
        return support

def is_resistance(df, i):
        resistance = df['High'][i] > df['High'][i-1]*DS and df['High'][i] > df['High'][i+1]*DS and \
                     df['High'][i+1] > df['High'][i+2]*DS and df['High'][i-1] > df['High'][i-2]*DS
        return resistance



# %%
# Reset the index so that the rows are numbered consecutively
df.reset_index(drop=True, inplace=True)

support=[]
resistance=[]

for i in range(2,len(df['Low'])-2):
    if is_support(df,i):
        # to avoid repitition
        if not support or abs(df['Low'][i] - support[-1][1]) > df['Low'][i]*0.02:
            support.append((i,df['Low'][i]))

    elif is_resistance(df,i):
        if not resistance or abs(df['High'][i] - resistance[-1][1]) > df['High'][i]*0.02:
            resistance.append((i,df['High'][i]))

# Bundle close support and resistance levels
def bundle_levels(levels, tolerance):
    bundled_levels = []
    i = 0
    while i < len(levels):
        current_level = levels[i]
        bundle = [current_level]
        j = i + 1
        while j < len(levels) and abs(levels[j][1] - current_level[1]) <= tolerance:
            bundle.append(levels[j])
            j += 1
        
        # Calculate average value for the bundle
        avg_value = sum(level[1] for level in bundle) / len(bundle)
        
        # Use the index of the first level in the bundle
        bundled_levels.append((bundle[0][0], avg_value))
        
        i = j
    return bundled_levels

# Define a tolerance for bundling (e.g., 2% of the price)
price_range = df['High'].max() - df['Low'].min()
tolerance_percentage = 0.05
tolerance = price_range * tolerance_percentage

bundled_support = bundle_levels(support, tolerance)
bundled_resistance = bundle_levels(resistance, tolerance)

len(support), len(bundled_support), len(resistance), len(bundled_resistance)

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Convert 'Price', 'Close', 'High', 'Low', 'Open' to numeric, coercing errors to NaN
for col in ['Close', 'High', 'Low', 'Open']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values introduced by the conversion
df = df.dropna()

# Convert 'Price' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Create subplots
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_width=[1], column_width=[1])

# Add candlestick trace
fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)

# Add support and resistance lines
for res in bundled_resistance:
    fig.add_trace(go.Scatter(x=[df['Date'].iloc[res[0]], df['Date'].iloc[-1]], y=[res[1], res[1]], mode='lines', line=dict(color='red'), name='Resistance'))

for sup in bundled_support:
    fig.add_trace(go.Scatter(x=[df['Date'].iloc[sup[0]], df['Date'].iloc[-1]], y=[sup[1], sup[1]], mode='lines', line=dict(color='green'), name='Support'))

fig.update_layout(
    title='BTC-USD Candlestick Chart with Support and Resistance',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

fig.show()


# %%



