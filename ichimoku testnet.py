"""Note next time update the leverage amount based on the choppy amount if the choppy amount is lower than 38 leverage should double than the specified amount 
   also add a checker method inorder to check if there is enough margin to trade and also to put stop loss and make sure the trade that is going to be in is in the same trend as btc  """ 

import pandas as pd
import pandas_ta as ta
import numpy as np
from binance.client import Client
import time
import os
import threading
from datetime import datetime, timedelta, UTC
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tools import measure
from binance import ThreadedWebsocketManager

# Add scikit-learn imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

# Set pandas option to handle future behavior
pd.set_option('future.no_silent_downcasting', True)

# --- Configuration ---
API_KEY = "iOgcObLOw4UIFSvvEPXLFP1vgwp1wzyHYfw57vd1vrg19Xt6SXCE4RywDi5QoM28"
API_SECRET = "bz1m4UlthzklqXlWoqAqXZiJE35jjT0g5uJ5cQ43vwDNnsIpPYS5OqevfBVz84iK"

#these will be used to make asset specifice decsion on tp and sl multi]plier
multiplier_set={'ETHUSDT':[1,3],'BTCUSDT':[1,3],'SOLUSDT':[1,3.5],'XRPUSDT':[3.5,3],'BNBUSDT':[2.5,3],'TONUSDT':[1,3.5],'DOGEUSDT':[1,3.5],'TRXUSDT':[3.5,3],'LTCUSDT':[1,3.5],'GUNUSDT':[3,3.5],'TUTUSDT':[1,3.5],'ADAUSDT':[1,3.5],'XLMUSDT':[1,3.5],'VETUSDT':[1.5,3.5],'HBARUSDT':[1,3.5],'SANDUSDT':[3.5,3.5],'GALAUSDT':[1,3.5],'FETUSDT':[1,3.5],"GRTUSDT":[1,3.5]}
adx_limit= {'ETHUSDT':80,'BTCUSDT':40,'SOLUSDT':30,'XRPUSDT':40,"TONUSDT":30,'DOGEUSDT':40,'TRXUSDT':70,'LTCUSDT':50,'ADAUSDT':50,'XLMUSDT':60,'VETUSDT':40,'HBARUSDT':40,'SANDUSDT':50,'1000PEPEUSDT':50,'1000BONKUSDT':70,'GALAUSDT':60,'FETUSDT':50,'GRTUSDT':60,'1000SHIBUSDT':40,'DOTUSDT':70,'LINKUSDT':50,'AVAXUSDT':40,'SUIUSDT':80}

orders_lock = threading.Lock()
orders = pd.DataFrame(columns=['symbol', 'side', 'entry_price', 'tp_price', 'sl_price','choppy'])

tradereport= pd.DataFrame(columns=['symbol', 'side', 'entry_price', 'tp_price', 'sl_price', 'exit_price','choppy'])

# --- Strategy & Trading Parameters ---
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK_PERIODS = 100
TP_MULTIPLIER = 1.8
SP_MULTIPLIER = 2.4
LEVERAGE = 1
TC = 0.0005
SIMULATED_INITIAL_BALANCE = 100
ASSETS = ['ETHUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT','SOLUSDT',"TONUSDT",'DOGEUSDT','TRXUSDT','SHIBUSDT','GUNUSDT','TUTUSDT','ADAUSDT','XLMUSDT','VETUSDT','HBARUSDT','SANDUSDT','1000PEPEUSDT','1000BONKUSDT','GALAUSDT','FETUSDT','GRTUSDT']

MAX_TRENDING_ASSETS = 0  # Maximum number of trending assets to trade

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("binance").setLevel(logging.INFO)

# --- Helper function ---
def interval_to_milliseconds(interval):
    seconds_per_unit = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            multiplier = int(interval[:-1])
            return seconds_per_unit[unit] * multiplier * 1000
        except ValueError:
            pass
    logging.error(f"Interval format {interval} not fully supported for ms conversion.")
    return None

def get_all_usdt_pairs(client):
    """Get all available USDT trading pairs from Binance."""

    try:
     
        exchange_info = client.futures_exchange_info()
        usdt_pairs = []
        for symbol_data in exchange_info['symbols']:
            if symbol_data['symbol'].endswith('USDT') and symbol_data['status'] == 'TRADING':
                usdt_pairs.append(symbol_data['symbol'])

        usdt_pairs.remove('DEFIUSDT')
        logging.info(f"Found {len(usdt_pairs)} USDT trading pairs.")
        return usdt_pairs
    except Exception as e:
        # Log the full traceback to understand where the error originated
        logging.error(f"Error fetching USDT pairs: {e}")
        return []

# Cache for trend scores to avoid recalculating
@lru_cache(maxsize=100)
def get_cached_klines(client, symbol, interval, limit):
    """Cached version of get_klines to reduce API calls"""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines or len(klines) == 0:
            logging.warning(f"No klines returned for {symbol}")
            return None
        return klines
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol}: {str(e)}")
        return None

def calculate_technical_features(df):
    """Calculate comprehensive technical indicators using multiple timeframes"""
    try:
        if df is None or df.empty:
            return None

        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Price momentum features
        df['returns'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(periods=5)
        df['momentum_10'] = df['close'].pct_change(periods=10)
        df['momentum_20'] = df['close'].pct_change(periods=20)
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_std_5'] = df['volume'].rolling(window=5).std()
        df['volume_zscore'] = (df['volume'] - df['volume_ma_5']) / df['volume_std_5'].replace(0, np.nan)
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Trend strength features
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_data is not None and 'ADX_14' in adx_data.columns:
            df['adx'] = adx_data['ADX_14']
            df['di_plus'] = adx_data['DMP_14']
            df['di_minus'] = adx_data['DMN_14']
        else:
            df['adx'] = np.nan
            df['di_plus'] = np.nan
            df['di_minus'] = np.nan
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_ratio'] = df['body_size'] / (df['upper_shadow'] + df['lower_shadow']).replace(0, np.nan)
        
        # Additional trend features
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_200'] = ta.ema(df['close'], length=200)
        df['ema_cross'] = (df['ema_20'] > df['ema_50']).astype(int)
        
        # RSI for overbought/oversold conditions
        df['rsi_14'] = ta.rsi(df['close'], length=14)  
        
        # Replace any remaining inf/-inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0 and ensure numeric type for all feature columns
        feature_columns = [
            'momentum_5', 'momentum_10', 'momentum_20',
            'volume_zscore', 'volume_ma_5', 'volume_ma_10',
            'volatility_5', 'volatility_10', 'atr_14',
            'adx', 'di_plus', 'di_minus',
            'body_size', 'upper_shadow', 'lower_shadow', 'candle_ratio',
            'ema_cross', 'rsi_14'
        ]
        
        for col in feature_columns:
            if col in df.columns:
                # Use infer_objects() to handle type inference properly
                df[col] = df[col].fillna(0).infer_objects(copy=False)
        
        return df

    except Exception as e:
        logging.error(f"Error in calculate_technical_features: {str(e)}")
        return None

def calculate_trend_score(df):
    """
    Calculate trend score using multiple sophisticated methods:
    1. PCA for dimensionality reduction
    2. K-means clustering for pattern recognition
    3. Isolation Forest for anomaly detection
    4. 24-hour price change
    5. Ensemble scoring combining all methods
    """
    try:
        if df is None:
            return 0.0

        # Select features for scoring
        features = [
            'momentum_5', 'momentum_10', 'momentum_20',
            'volume_zscore', 'volume_ma_5', 'volume_ma_10',
            'volatility_5', 'volatility_10', 'atr_14',
            'adx', 'di_plus', 'di_minus',
            'body_size', 'upper_shadow', 'lower_shadow', 'candle_ratio',
            'ema_cross', 'rsi_14'
        ]
        
        # Check if we have all required features
        if not all(feature in df.columns for feature in features):
            missing_features = [f for f in features if f not in df.columns]
            logging.error(f"Missing required features: {missing_features}")
            return 0.0

        # Get feature data and handle missing values more efficiently
        feature_data = df[features].copy()
        
        # Replace inf and -inf with NaN first
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0 and ensure numeric type
        feature_data = feature_data.fillna(0).infer_objects(copy=False)
        
        if feature_data.empty:
            logging.error("Empty feature data for trend scoring")
            return 0.0

        # Calculate 24-hour price change
        if len(df) >= 96:  # Ensure we have enough data for 24 hours (assuming 15min candles)
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-96] - 1) * 100
        else:
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # Normalize 24h price change to [-1, 1] range
        # Using tanh to cap extreme values while preserving direction
        price_change_score = np.tanh(price_change_24h / 50)  # Divide by 50 to scale down large changes
        
        # Standardize features before analysis
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # 1. PCA for dimensionality reduction
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(scaled_features)
        pca_score = float(pca_features[-1].sum())  # Sum of last row's components
        
        # 2. K-means clustering for pattern recognition
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        silhouette = silhouette_score(scaled_features, clusters)
        
        # 3. Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(scaled_features)
        anomaly_score = float(np.mean(anomaly_scores[-5:]))  # Average of last 5 scores
        
        # 4. Ensemble scoring
        # Normalize individual scores
        pca_score_norm = np.tanh(pca_score / 10)  # Scale down PCA score
        silhouette_norm = np.tanh(silhouette * 10)  # Scale up silhouette score
        anomaly_score_norm = np.tanh(anomaly_score * 2)  # Scale anomaly score
        
        # Combine scores with weights, giving significant weight to 24h price change
        final_score = (
            0.45 * price_change_score +  # 24h price change (most important)
            0.40 * pca_score_norm +      # PCA captures main trend components
            0.15 * silhouette_norm +      # Clustering identifies pattern strength
            0.0 * anomaly_score_norm     # Anomaly detection finds unusual movements
        )
        
        return float(final_score)

    except Exception as e:
        logging.error(f"Error calculating trend score: {str(e)}")
        return 0.0

def process_symbol(client, symbol, interval=Client.KLINE_INTERVAL_1DAY, lookback=7):
    """Process a single symbol with enhanced trend detection"""
    try:
        # Get cached klines
        klines = get_cached_klines(client, symbol, interval, lookback)
        if not klines:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 
                                         'tb_quote_av', 'ignore'])
        
        if df.empty:
            return None
            
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Calculate features
        df = calculate_technical_features(df)
        if df is None:
            return None
        
        # Calculate trend score
        score = calculate_trend_score(df)
        
        # Calculate 24h price change for display
        if len(df) >= 96:  # Ensure we have enough data for 24 hours (assuming 15min candles)
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-96] - 1) * 100
        else:
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # Only return if we have valid data
        if score != 0.0 and not pd.isna(score):
            return {
                'symbol': symbol,
                'score': score,
                'price_change_24h': price_change_24h,  # Added 24h price change
                'price_change': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                'avg_volume': df['volume'].mean(),
                'volatility': df['volatility_5'].iloc[-1] * 100 if 'volatility_5' in df else 0,
                'adx': df['adx'].iloc[-1] if 'adx' in df else 0,
                'rsi': df['rsi_14'].iloc[-1] if 'rsi_14' in df else 0
            }
        return None

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")
        return None

def get_trending_assets(client, max_assets=5):
    """Identify trending assets using parallel processing and advanced scoring"""
    logging.info("Identifying trending assets...")
    
    # Get all USDT pairs
    usdt_pairs = get_all_usdt_pairs(client)
    if not usdt_pairs:
        logging.error("Failed to get USDT pairs")
        return []
    
    # Filter out already specified assets and limit initial pairs for testing
    usdt_pairs = [pair for pair in usdt_pairs if pair not in ASSETS][:50]  # Limit to 50 pairs for testing
    
    # Process symbols in parallel with reduced concurrency
    trend_scores = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced from 10 to 5
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol, client, symbol): symbol 
            for symbol in usdt_pairs
        }
        
        # Process results as they complete
        for future in as_completed(future_to_symbol):
            try:
                result = future.result()
                if result:  # Only add non-None results
                    trend_scores.append(result)
            except Exception as e:
                symbol = future_to_symbol[future]
                logging.error(f"Error processing future for {symbol}: {str(e)}")
    
    # Sort by score and get top pairs
    if trend_scores:
        trend_scores.sort(key=lambda x: x['score'], reverse=True)
        top_pairs = [score['symbol'] for score in trend_scores[:max_assets]]
        logging.info(f"Selected trending assets: {top_pairs}")
        return top_pairs
    else:
        logging.warning("No valid trend scores calculated")
        return []

def print_trading_status():
    """Print current orders and trade reports with formatting."""
    # Clear some space before printing
    print("\n\n")
    
    # Print current trades
    print("="*100)
    print("CURRENT ACTIVE TRADES:".center(100))
    print("="*100)
    if len(orders) > 0:
        # Format the DataFrame output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)
        pd.set_option('display.max_rows', None)
        print(orders.to_string(index=True))
    else:
        print("No active trades".center(100))
    
    # Add spacing between sections
    print("\n" + "="*100)
    print("COMPLETED TRADES:".center(100))
    print("="*100)
    if len(tradereport) > 0:
        # Format the DataFrame output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)
        pd.set_option('display.max_rows', None)
        print(tradereport.to_string(index=True))
    else:
        print("No completed trades".center(100))
    print("="*100)
    
    # Add extra newlines after printing
    print("\n\n")
    
    # Flush the output to ensure it's displayed immediately
    sys.stdout.flush()

# Add file lock for thread-safe CSV operations
file_lock = threading.Lock()

def save_trade_to_csv(trade_data: dict) -> None:
    """Save trade data to CSV file in a thread-safe manner."""
    try:
        # Create trades directory if it doesn't exist
        os.makedirs('trades', exist_ok=True)
        
        # Define CSV file path
        csv_file = 'trades/trade_history_unpicked.csv'
        
        # Convert trade data to DataFrame
        trade_df = pd.DataFrame([trade_data])
        
        # Add timestamp for when the trade was saved
        trade_df['saved_at'] = datetime.utcnow()
        
        # Use file lock to ensure thread-safe file operations
        with file_lock:
            # Check if file exists
            if os.path.exists(csv_file):
                # Append to existing file
                trade_df.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                # Create new file with header
                trade_df.to_csv(csv_file, index=False)
                
    except Exception as e:
        logging.error(f"Error saving trade to CSV: {e}", exc_info=True)

def BTC_trend_identification(client):
    now = datetime.now()
    before = datetime.now() - timedelta(days=1)
    
    # Convert datetime objects to string format that Binance API expects
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    before_str = before.strftime('%Y-%m-%d %H:%M:%S')
    
    klines = client.get_historical_klines(
        symbol='BTCUSDT', 
        interval=Client.KLINE_INTERVAL_15MINUTE, 
        start_str=before_str, 
        end_str=now_str
    )
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    if df['close'].iloc[-1] > df['close'].iloc[0]:
        return 'up'
    else:
        return 'down'

class ForwardIchimokuTrader:
    def __init__(self, symbol: str, interval: str, lookback: int,
                 tp_multiplier: float = 2.5, sp_multiplier: float = 2,
                 leverage: int = 1, tc: float = 0.0005, initial_balance: float = 1000) -> None:
        self.symbol = symbol
        self.interval = interval
        self.interval_5m = Client.KLINE_INTERVAL_15MINUTE
        self.lookback = lookback
        self.required_lookback = max(lookback, 52, 27)
        if lookback < self.required_lookback:
            logging.warning(f"Initial lookback {lookback} increased to {self.required_lookback} for indicator calculations.")
            self.lookback = self.required_lookback

        self.interval_LTF = Client.KLINE_INTERVAL_1HOUR
        self.interval_HTF = Client.KLINE_INTERVAL_2HOUR
        self.tp_multiplier = tp_multiplier
        self.sp_multiplier = sp_multiplier
        self.leverage = leverage
        self.tc = tc
        self.initial_balance = initial_balance
        self.simulated_balance = initial_balance

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{symbol}")
        self.logger.info("Initializing Trader...")

        try:
            self.client = Client(API_KEY, API_SECRET)
            self.client.ping()
            self.logger.info("Binance client initialized and connection tested.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}", exc_info=True)
            raise

        self.df = pd.DataFrame()
        self.position = 0
        self.entry_price = 0.0
        self.tp_level = None
        self.sl_level = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.last_trade_time = None

        self.ms_interval = interval_to_milliseconds(self.interval)
        if not self.ms_interval:
            raise ValueError("Invalid interval for millisecond conversion")
        self.logger.info(f"Interval: {self.interval}, Milliseconds: {self.ms_interval}")

    def _get_server_time(self):
        try:
            server_time = self.client.get_server_time()
            server_time_ms = server_time['serverTime']
            return pd.to_datetime(server_time_ms, unit='ms')
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            return datetime.utcnow()
    def _fetch_HTF_data(self, limit=100) -> pd.DataFrame:
        self.logger.debug(f"Fetching latest {limit} HFT klines for {self.symbol}...")
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval_HTF, limit=limit)
            if not klines:
                self.logger.warning("Could not fetch HFT klines (empty list).")
                return None

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']
            data = pd.DataFrame(klines, columns=cols + ['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
            data = data[cols]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                self.logger.warning("NaN values in HFT OHLC data after conversion.")
                data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms')
            data.set_index('Datetime', inplace=True)
            df_HTF = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            if len(df_HTF) < 27:
                self.logger.warning(f"Insufficient HFT data fetched ({len(df_HTF)} rows). Need at least 27.")
                return None

            return df_HTF
        except Exception as e:
            self.logger.error(f"Error fetching HFT data: {e}", exc_info=True)
            return None
    def _calculate_HTF_indicators(self, df_HTF: pd.DataFrame) -> pd.DataFrame:
        if df_HTF is None or len(df_HTF) < 52:
            self.logger.warning("Insufficient data for HFT Ichimoku calculation.")
            return None

        try:
            ichimoku_data = ta.ichimoku(df_HTF['High'], df_HTF['Low'], df_HTF['Close'])
            if ichimoku_data is None or not isinstance(ichimoku_data, tuple) or len(ichimoku_data) < 1 or ichimoku_data[0].empty:
                self.logger.warning("HFT Ichimoku calculation returned unexpected/empty data.")
                return None

            temp_df_ichi_HTF = ichimoku_data[0].rename(columns={
                'ISA_9': 'leading Span A', 'ISB_26': 'leading Span B',
                'ITS_9': 'conversion line', 'IKS_26': 'base line',
                'ICS_26': 'lagging Span'
            })
            temp_df_ichi_HTF.index = df_HTF.index[-len(temp_df_ichi_HTF):]
            df_HTF = df_HTF.join(temp_df_ichi_HTF)
            return df_HTF
        except Exception as e:
            self.logger.error(f"Error calculating HFT indicators: {e}", exc_info=True)
            return None

    def _check_HTF_confirmation(self, direction: str) -> bool:
        df_HTF = self._fetch_HTF_data(limit=100)
        if df_HTF is None:
            return False

        df_HTF = self._calculate_HTF_indicators(df_HTF)
        if df_HTF is None:
            return False

        if len(df_HTF) < 27:
            self.logger.warning("Not enough HTF data for signal check.")
            return False

        last_HTF = df_HTF.iloc[-1]
        close_t_minus_26_5m = df_HTF['Close'].iloc[-27]
        current_close = df_HTF['Close'].iloc[-1]
        leading_span_A_shifted=df_HTF['leading Span A'].iloc[-26]

        if pd.isna(close_t_minus_26_5m) or last_HTF[['Close', 'conversion line', 'base line']].isnull().any():
            self.logger.warning("NaN values in 5m data for signal check.")
            return False

        buy_cond_HTF =  (last_HTF['conversion line'] > last_HTF['base line']  and (current_close > leading_span_A_shifted))
        sell_cond_HTF = (last_HTF['conversion line'] < last_HTF['base line'] and  (current_close < leading_span_A_shifted))

        if direction == 'buy':
            self.logger.debug(f"HTF Buy confirmation: {buy_cond_HTF}")
            return buy_cond_HTF
        elif direction == 'sell':
            self.logger.debug(f"HTF Sell confirmation: {sell_cond_HTF}")
            return sell_cond_HTF
        else:
            self.logger.error(f"Invalid direction {direction} in _check_HTF_confirmation.")
            return False
    
    def _fetch_LTF_data(self, limit=100) -> pd.DataFrame:
        self.logger.debug(f"Fetching latest {limit} LTF klines for {self.symbol}...")
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval_LTF, limit=limit)
            if not klines:
                self.logger.warning("Could not fetch LTF klines (empty list).")
                return None

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']
            data = pd.DataFrame(klines, columns=cols + ['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
            data = data[cols]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                self.logger.warning("NaN values in LTF OHLC data after conversion.")
                data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms')
            data.set_index('Datetime', inplace=True)
            df_LTF = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            if len(df_LTF) < 27:
                self.logger.warning(f"Insufficient LTF data fetched ({len(df_LTF)} rows). Need at least 27.")
                return None

            return df_LTF
        except Exception as e:
            self.logger.error(f"Error fetching LTF data: {e}", exc_info=True)
            return None

    def _calculate_LTF_indicators(self, df_LTF: pd.DataFrame) -> pd.DataFrame:
        if df_LTF is None or len(df_LTF) < 52:
            self.logger.warning("Insufficient data for LTF Ichimoku calculation.")
            return None

        try:
            ichimoku_data = ta.ichimoku(df_LTF['High'], df_LTF['Low'], df_LTF['Close'])
            if ichimoku_data is None or not isinstance(ichimoku_data, tuple) or len(ichimoku_data) < 1 or ichimoku_data[0].empty:
                self.logger.warning("LTF Ichimoku calculation returned unexpected/empty data.")
                return None

            temp_df_ichi = ichimoku_data[0].rename(columns={
                'ISA_9': 'leading Span A', 'ISB_26': 'leading Span B',
                'ITS_9': 'conversion line', 'IKS_26': 'base line',
                'ICS_26': 'lagging Span'
            })
            temp_df_ichi.index = df_LTF.index[-len(temp_df_ichi):]
            df_LTF = df_LTF.join(temp_df_ichi)
            return df_LTF
        except Exception as e:
            self.logger.error(f"Error calculating 5m indicators: {e}", exc_info=True)
            return None

    def _check_LTF_confirmation(self, direction: str) -> bool:
        df_LTF = self._fetch_LTF_data(limit=100)
        if df_LTF is None:
            return False

        df_LTF = self._calculate_LTF_indicators(df_LTF)
        if df_LTF is None:
            return False

        if len(df_LTF) < 27:
            self.logger.warning("Not enough LTF data for signal check.")
            return False

        last_LTF = df_LTF.iloc[-1]
        close_t_minus_26_5m = df_LTF['Close'].iloc[-27]
        current_close = last_LTF['Close']
        leading_span_A_shifted=df_LTF['leading Span A'].iloc[-26]

        if pd.isna(close_t_minus_26_5m) or last_LTF[['Close', 'conversion line', 'base line']].isnull().any():
            self.logger.warning("NaN values in 5m data for signal check.")
            return False

        buy_cond_LTF =  (last_LTF['conversion line'] > last_LTF['base line'] and (current_close > leading_span_A_shifted))
        sell_cond_LTF =  (last_LTF['conversion line'] < last_LTF['base line'] and  (current_close < leading_span_A_shifted))

        if direction == 'buy':
            self.logger.debug(f"5m Buy confirmation: {buy_cond_LTF}")
            return buy_cond_LTF
        elif direction == 'sell':
            self.logger.debug(f"5m Sell confirmation: {sell_cond_LTF}")
            return sell_cond_LTF
        else:
            self.logger.error(f"Invalid direction {direction} in _check_5m_confirmation.")
            return False
        
    def _ensure_dataframe_integrity(self) -> bool:
        """Ensure DataFrame has unique, sorted index and valid data"""
        try:
            if self.df.empty:
                self.logger.warning("DataFrame is empty")
                return False
            
            # Check for duplicate timestamps
            if not self.df.index.is_unique:
                self.logger.warning("Duplicate timestamps found. Removing duplicates...")
                self.df = self.df[~self.df.index.duplicated(keep='last')]
                self.logger.info(f"After removing duplicates: {len(self.df)} rows")
            
            # Ensure index is sorted
            if not self.df.index.is_monotonic_increasing:
                self.logger.warning("Index not monotonic. Sorting...")
                self.df.sort_index(inplace=True)
            
            # Check for NaN values in critical columns
            critical_cols = ['Open', 'High', 'Low', 'Close']
            if self.df[critical_cols].isnull().any().any():
                self.logger.warning("NaN values found in critical columns. Removing rows with NaN...")
                self.df.dropna(subset=critical_cols, inplace=True)
                self.logger.info(f"After removing NaN rows: {len(self.df)} rows")
            
            return True
        except Exception as e:
            self.logger.error(f"Error ensuring DataFrame integrity: {e}")
            return False

    def _fetch_initial_data(self) -> bool:
        self.logger.info(f"Fetching initial {self.lookback + 50} klines for {self.symbol}...")
        try:
            start=str(self._get_server_time()-timedelta(minutes=1600))
            klines = self.client.get_historical_klines(symbol=self.symbol, interval=self.interval,start_str=start)
            klines=klines[:-1]
            if not klines:
                self.logger.error("Could not fetch initial klines (received empty list).")
                return False

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']
            data = pd.DataFrame(klines, columns=cols + ['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
            data = data[cols]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                self.logger.warning("NaN values found in OHLC data after numeric conversion during initial fetch.")
                data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms')
            data.set_index('Datetime', inplace=True)
            self.df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Ensure DataFrame integrity
            if not self._ensure_dataframe_integrity():
                self.logger.error("Failed to ensure DataFrame integrity after initial data fetch")
                return False
            
            #print(self.df.iloc[-1]) for debugging 
            if len(self.df) < self.required_lookback:
                self.logger.error(f"Insufficient valid initial data fetched ({len(self.df)} rows). Need at least {self.required_lookback}.")
                return False

            self.logger.info(f"Successfully fetched and processed {len(self.df)} initial candles. Last candle time: {self.df.index[-1]}")
            return True
        except Exception as e:
            self.logger.error(f"Error fetching initial data: {e}", exc_info=True)
            return False

    def _fetch_latest_candle(self) -> bool:
        self.logger.debug("Attempting to fetch latest candle...")
        try:
            start=str(self._get_server_time()-timedelta(minutes=60))
            klines = self.client.get_historical_klines(symbol=self.symbol, interval=self.interval,start_str=start )
            klines=klines[:-1]
            if not klines or len(klines) < 2:
                self.logger.warning("Could not fetch latest klines or not enough data yet (< 2).")
                return False

            # Ensure DataFrame integrity before processing new candles
            if not self.df.empty and not self._ensure_dataframe_integrity():
                self.logger.error("Failed to ensure DataFrame integrity before fetching latest candle")
                return False

            latest_kline = klines[-1]
            close_time_ms = latest_kline[6]
            latest_dt = pd.to_datetime(close_time_ms, unit='ms')

            if self.df.empty or latest_dt <= self.df.index[-1]:
                return False

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']
            new_data = pd.DataFrame([latest_kline], columns=cols + ['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
            new_data = new_data[cols]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            if new_data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                self.logger.error(f"NaN value detected in OHLC for new candle at {latest_dt}. Skipping append.")
                return False

            new_data['Datetime'] = latest_dt
            new_data.set_index('Datetime', inplace=True)
            new_row = new_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[0]
            #print(new_row) for debuggging 

            if latest_dt in self.df.index:
                self.logger.debug(f"Duplicate candle at {latest_dt}, skipping append.")
                return False

            # Use a safer concatenation approach
            try:
                # Create a temporary DataFrame with the new row
                temp_df = new_row.to_frame().T
                
                # Ensure the new DataFrame has a unique index
                if temp_df.index.is_unique:
                    # Use ignore_index=False to preserve the datetime index
                    self.df = pd.concat([self.df, temp_df], ignore_index=False)
                    self.logger.info(f"New candle appended: {latest_dt}, Close: {new_row['Close']:.4f}, DF rows: {len(self.df)}")
                else:
                    self.logger.warning(f"New candle has duplicate index: {latest_dt}")
                    return False
            except Exception as concat_error:
                self.logger.error(f"Error during concatenation for {latest_dt}: {concat_error}")
                return False

            # Ensure DataFrame integrity after appending
            if not self._ensure_dataframe_integrity():
                self.logger.error("Failed to ensure DataFrame integrity after appending new candle")
                return False

            max_len = self.lookback + 100
            if len(self.df) > max_len:
                self.logger.debug(f"Trimming DataFrame from {len(self.df)} to {max_len} rows.")
                self.df = self.df.iloc[-max_len:]

            return True
        except Exception as e:
            self.logger.error(f"Error fetching/appending latest candle: {e}", exc_info=True)
            return False

    def _calculate_indicators(self) -> bool:
        self.logger.debug(f"Calculating indicators for {len(self.df)} rows...")
        min_data_needed = self.required_lookback
        if len(self.df) < min_data_needed:
            self.logger.warning(f"Not enough data ({len(self.df)}, need {min_data_needed}) for all indicator calculations.")
            return False
        try:
            # Ensure DataFrame integrity before calculations
            if not self._ensure_dataframe_integrity():
                self.logger.error("Failed to ensure DataFrame integrity before indicator calculation")
                return False
            
            ichimoku_data = ta.ichimoku(self.df['High'], self.df['Low'], self.df['Close'])
            if ichimoku_data is None or not isinstance(ichimoku_data, tuple) or len(ichimoku_data) < 1 or ichimoku_data[0].empty:
                self.logger.warning("Ichimoku calculation returned unexpected/empty data.")
                return False

            temp_df_ichi = ichimoku_data[0].rename(columns={
                'ISA_9': 'leading Span A', 'ISB_26': 'leading Span B',
                'ITS_9': 'conversion line', 'IKS_26': 'base line',
                'ICS_26': 'lagging Span'
            })
            temp_df_ichi.index = self.df.index[-len(temp_df_ichi):]

            if len(self.df) >= 15:
                self.df['atr'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
                self.df['adx'] = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)
                self.df['choppy'] = ta.chop(self.df['High'],self.df['Low'],self.df['Close'])
                psar = ta.psar(self.df['High'],self.df['Low'],self.df['Close'])
                
                # Drop any existing PSAR columns first to avoid overlap
                psar_cols_to_drop = ['PSAR_Long', 'PSAR_Short', 'PSAR_Reversal', 'PSARaf_0.02_0.2', 
                                   'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARr_0.02_0.2']
                existing_cols_to_drop = [col for col in psar_cols_to_drop if col in self.df.columns]
                if existing_cols_to_drop:
                    self.df = self.df.drop(columns=existing_cols_to_drop)
                    self.logger.debug(f"Dropped existing PSAR columns: {existing_cols_to_drop}")
                
                # Rename PSAR columns to our standard names
                psar = psar.rename(columns={'PSARl_0.02_0.2':'PSAR_Long',"PSARs_0.02_0.2":'PSAR_Short','PSARr_0.02_0.2':"PSAR_Reversal"})
                
                # Ensure PSAR has the same index as the main DataFrame
                if not psar.empty:
                    psar.index = self.df.index[-len(psar):]
                    
                    # Use join instead of concat to avoid index issues
                    self.df = self.df.join(psar)
                else:
                    self.logger.warning("PSAR calculation returned empty DataFrame")
            else:
                self.logger.warning(f"Not enough data ({len(self.df)}) for ATR(14) calculation.")
                self.df['atr'] = np.nan

            ichimoku_cols_to_drop = [col for col in temp_df_ichi.columns if col in self.df.columns]
            if ichimoku_cols_to_drop:
                self.df = self.df.drop(columns=ichimoku_cols_to_drop)

            self.df = self.df.join(temp_df_ichi)
            
            # Ensure DataFrame integrity after calculations
            if not self._ensure_dataframe_integrity():
                self.logger.error("Failed to ensure DataFrame integrity after indicator calculation")
                return False
            
            last_row = self.df.iloc[-1]
            if last_row[['conversion line', 'base line', 'atr']].isnull().any():
                self.logger.warning(f"Last row has NaN indicators: {last_row[['conversion line', 'base line', 'atr']]}")
            else:
                self.logger.debug("Last row indicators are valid.")
            return True
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return False

    def _check_signals(self) -> tuple[bool, bool]:
        self.logger.debug("Checking signals...")
        current_length = len(self.df)
        if current_length < 27:
            self.logger.warning(f"DataFrame length ({current_length}) is less than 27. Cannot perform lagging Span comparison yet.")
            return False, False
        last3=self.df.iloc[-3]
        last = self.df.iloc[-1]
        last_index_name = last.name

        required_cols = ['Close', 'conversion line', 'base line','leading Span A','leading Span B']
        if last[required_cols].isnull().any():
            self.logger.warning(f"Signal check skipped: NaN in required columns at {last_index_name}.")
            return False, False

        try:
            target_past_index_pos = -1 - 26
            target_past_index = self.df.index[target_past_index_pos]
            close_t_minus_26 = self.df.loc[target_past_index, 'Close']
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error accessing T-26 Close: {e}. Length: {current_length}")
            return False, False

        if pd.isna(close_t_minus_26):
            self.logger.warning(f"Signal check skipped: Close at {target_past_index} is NaN.")
            return False, False

        conversion_line3=last3['conversion line']
        current_close = last['Close']
        conversion_line = last['conversion line']
        base_line = last['base line']
        leading_Span_A = last['leading Span A']
        leading_Span_A_shifted=self.df['leading Span A'].iloc[-26]
        leading_Span_B = last['leading Span B']
        choppy = last['choppy']
        psar_long = last['PSAR_Long']
        psar_short = last['PSAR_Short']
        adx=last['adx']

        
        if isinstance(psar_long, pd.Series):
            psar_long = psar_long.iloc[-1]
        if hasattr(psar_long, 'item'):
            try:
                psar_long = psar_long.item()
            except Exception:
                pass
        psar_short = last['PSAR_Short']
        if isinstance(psar_short, pd.Series):
            psar_short = psar_short.iloc[-1]
        if hasattr(psar_short, 'item'):
            try:
                psar_short = psar_short.item()
            except Exception:
                pass

        

        buy_signal = (current_close > close_t_minus_26) and (conversion_line >= base_line) and (leading_Span_A > leading_Span_B) and (current_close > leading_Span_A_shifted)  and (psar_long < current_close) and (adx > adx_limit[self.symbol])
        sell_signal = (current_close < close_t_minus_26) and (conversion_line <= base_line) and (leading_Span_B > leading_Span_A) and (current_close < leading_Span_A_shifted)  and (psar_short > current_close) and (adx > adx_limit[self.symbol])
        
        
        self.logger.debug(f"15min. Signals @ {last_index_name}: Buy={buy_signal}, Sell={sell_signal}")
        return buy_signal, sell_signal

    def _manage_position(self) -> None:
        if self.df.empty or 'atr' not in self.df.columns:
            self.logger.warning("DataFrame not ready for position management (empty or no ATR).")
            return

        try:
            last_row = self.df.iloc[-1]
            current_price_high = last_row['High']
            current_price_low = last_row['Low']
            current_price = last_row['Close']
            atr = last_row['atr']
            """print(current_price)
            print(current_price_high)
            print(current_price_low)"""#for debugging purpose
            
            
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error accessing data in _manage_position: {e}")
            return

        if pd.isna(current_price) or pd.isna(atr) or atr <= 0:
            self.logger.warning(f"Invalid price ({current_price}) or ATR ({atr}) at {last_row.name}. Skipping.")
            return

        if self.position == 1:
            if self.tp_level is None or self.sl_level is None or self.highest_price_since_entry is None:
                self.logger.error(f"Inconsistent state for long position @ {last_row.name}. Resetting.")
                self._reset_position_state()
                return
            self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
            self.sl_level = self.highest_price_since_entry - (multiplier_set[self.symbol][1] * atr)
            orders.loc[self.order_id, 'sl_price'] = self.sl_level # Update SL in orders DataFrame

            if current_price_high >= self.tp_level:
                self._close_position(self.tp_level, f"Take Profit hit at {self.tp_level:.4f}")
            elif current_price_low <= self.sl_level:
                self._close_position(self.sl_level, f"Stop Loss hit at {self.sl_level:.4f}")
        elif self.position == -1:
            if self.tp_level is None or self.sl_level is None or self.lowest_price_since_entry is None:
                self.logger.error(f"Inconsistent state for short position @ {last_row.name}. Resetting.")
                self._reset_position_state()
                return
            self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)
            self.sl_level = self.lowest_price_since_entry + (multiplier_set[self.symbol][1] * atr)
            orders.loc[self.order_id, 'sl_price'] = self.sl_level # Update SL in orders DataFrame
            
            
            if current_price_low <= self.tp_level:
                self._close_position(self.tp_level, f"Take Profit hit at {self.tp_level:.4f}")
            elif current_price >= self.sl_level:
                self._close_position(self.sl_level, f"Stop Loss hit at {self.sl_level:.4f}")

        if self.position == 0:
            buy_signal, sell_signal = self._check_signals()
            buy_cond_LFT,sell_cond_LFT = self._check_LTF_confirmation('buy'),self._check_LTF_confirmation('sell')
            buy_cond_HTF,sell_cond_HTF = self._check_HTF_confirmation('buy'),self._check_HTF_confirmation('sell')
            self.logger.debug(f"Entry check: Buy signal={buy_signal}, Sell signal={sell_signal}")
            trend=BTC_trend_identification(self.client)
            if buy_signal:
                
                    self._enter_position(1, current_price, atr)
            elif sell_signal :
                
                    self._enter_position(-1, current_price, atr)
                    
    def _enter_position(self, direction: int, entry_price: float, atr: float) -> None:
        global orders # Needed because we assign to the global 'orders' variable via pd.concat

        if pd.isna(atr) or atr <= 0:
            self.logger.error(f"Attempted to enter position with invalid ATR: {atr}. Aborting entry.")
            return
        if self.position != 0:
            self.logger.warning(f"Attempted to enter {'LONG' if direction == 1 else 'SHORT'} while already in position {self.position}. Aborting.")
            return

        self.position = direction
        self.entry_price = entry_price
        entry_cost = self.simulated_balance * self.tc # Assume cost applies on entry
        self.simulated_balance -= entry_cost

        log_msg_base = f"PAPER TRADE: Entered {'LONG' if direction == 1 else 'SHORT'} {self.symbol} @ {entry_price:.4f}"
        try:
            if direction == 1: # Long
                self.tp_level = entry_price + (atr * multiplier_set[self.symbol][0])
                self.sl_level = entry_price - (atr * multiplier_set[self.symbol][1])
                self.highest_price_since_entry = entry_price # Initialize high tracker
                self.lowest_price_since_entry = None # Not used for long
                side = 'BUY'
                log_msg = f"{log_msg_base} | TP: {self.tp_level:.4f}, SL: {self.sl_level:.4f} | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
                
                """ available_balance=client.get_asset_balance(asset='USDT')
                if available_balance>0:
                    quantity=available_balance/self.entry_price - (0.001*available_balance/self.entry_price)
                    order=client.create_order(symbol=self.symbol,side=side,type='LIMIT',quantity=quantity)
                    bought_asset=client.get_asset_balance(asset=self.symbol)
                    tp=client.create_order(symbol=self.symbol,side="SELL",type='TAKE_PROFIT_LIMIT',quantity=quantity,price=self.tp_level)
                    sl=client.create_order(symbol=self.symbol,side="SELL",type='STOP_LOSS_LIMIT',quantity=quantity,price=self.sl_level)
                    
                else:
                    self.logger.warning(f"balance insufficent to enter trade")"""
                
            elif direction == -1: # Short
                self.tp_level = entry_price - (atr * multiplier_set[self.symbol][0])
                self.sl_level = entry_price + (atr * multiplier_set[self.symbol][1])
                self.lowest_price_since_entry = entry_price # Initialize low tracker
                self.highest_price_since_entry = None # Not used for short
                side = 'SELL'
                log_msg = f"{log_msg_base} | TP: {self.tp_level:.4f}, SL: {self.sl_level:.4f} | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"

                """available_balance=client.get_asset_balance(asset='USDT')
                if available_balance>0:
                    order=client.create_order(symbol=self.symbol,side=side,type='MARKET',quantity=available_balance)
                else:
                    self.logger.warning(f"balance insufficent to enter trade")""" 
            else:
                self.logger.error(f"Invalid direction {direction} passed to _enter_position.")
                self._reset_position_state(source_error="Invalid direction on entry")
                return

            # Create new order data as a dictionary
            entry_time_now = datetime.utcnow()
            self.entry_time = entry_time_now
            new_order_data = {
                'entry_time': entry_time_now,
                'symbol': self.symbol,
                'side': side,
                'entry_price': self.entry_price,
                'tp_price': self.tp_level,
                'sl_price': self.sl_level,
                'choppy':self.df['choppy'].iloc[-1]
                
            }

            # Create a DataFrame for the new row
            new_order_df = pd.DataFrame([new_order_data])

            # Add order safely using pd.concat within the lock
            with orders_lock:
                orders = pd.concat([orders, new_order_df], ignore_index=True)
                # Store the index (order_id) of the newly added order
                self.order_id = orders.index[-1]
                new_order_data['order_id'] = self.order_id # Add order_id for logging if needed

            # Include order_id in log
            self.logger.info(log_msg + f" | OrderID: {self.order_id}")


        except Exception as e:
            self.logger.error(f"Error calculating TP/SL or adding order during entry: {e}. Position may be invalid.", exc_info=True)
            self._reset_position_state(source_error="Exception during entry") # Reset state on error


    def _close_position(self, exit_price: float, reason: str) -> None:
        global orders, tradereport

        if self.position == 0:
            self.logger.warning("Attempted to close position while already flat.")
            return

        # Calculate PnL and update balance
        pnl_percentage = 0
        if self.position == 1:
            pnl_percentage = (exit_price / self.entry_price - 1) if self.entry_price != 0 else 0
        elif self.position == -1:
            pnl_percentage = (self.entry_price / exit_price - 1) if exit_price != 0 else 0

        effective_pnl = pnl_percentage * self.leverage
        exit_cost = self.simulated_balance * (1 + effective_pnl) * self.tc
        final_pnl_factor = (1 + effective_pnl) * (1 - self.tc)

        effective_pnl_perc = pnl_percentage * self.leverage

        # --- Calculate Balance Change & Apply Exit Cost ---
        balance_before_close = self.simulated_balance
        pnl_amount_gross = balance_before_close * effective_pnl_perc
        balance_after_gross_pnl = balance_before_close + pnl_amount_gross
        exit_cost = balance_after_gross_pnl * self.tc
        self.simulated_balance = balance_after_gross_pnl - exit_cost
        balance_change = self.simulated_balance - balance_before_close

        exit_time = datetime.utcnow()
        closed_direction = 'LONG' if self.position == 1 else 'SHORT'

        # --- Prepare trade data for CSV ---
        trade_data = {
            'symbol': self.symbol,
            'direction': closed_direction,
            'entry_time': self.entry_time,
            'exit_time': exit_time,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'tp_price': self.tp_level,
            'final_sl': self.sl_level,
            'pnl_percentage': effective_pnl_perc,
            'pnl_amount': balance_change,
            'exit_reason': reason,
            'choppy': self.df['choppy'].iloc[-1],
            'leverage': self.leverage,
            'initial_balance': self.initial_balance,
            'final_balance': self.simulated_balance
        }

        # Save trade to CSV
        save_trade_to_csv(trade_data)
        profit=0

        # --- Add to Trade Report (Thread-Safe) ---
        trade_report_data = {
            'symbol': self.symbol,
            'side': closed_direction,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'tp_price': self.tp_level,
            'final_sl': self.sl_level,
            'pnl_amount': balance_change,
            'exit_reason': reason,
            'choppy': self.df['choppy'].iloc[-1]
        }
        new_report_df = pd.DataFrame([trade_report_data])
        profit+=balance_change
        self.logger.info(f"=============Total profit: {profit:.4f}=============")
        with orders_lock:
            tradereport = pd.concat([tradereport, new_report_df], ignore_index=True)

        closed_direction = 'LONG' if self.position == 1 else 'SHORT'
        balance_before_close = self.simulated_balance
        self.simulated_balance *= final_pnl_factor

        # Log the closure
        self.logger.info(f"PAPER TRADE: Closed {closed_direction} @ {exit_price:.4f} | Entry: {self.entry_price:.4f} | Reason: {reason}")
        self.logger.info(f"PnL: {effective_pnl:.4%} (before exit cost) | Exit Cost: {exit_cost:.4f} | Balance Change: {self.simulated_balance - balance_before_close:.4f} | New Balance: {self.simulated_balance:.2f}")

        # Remove the order with thread safety
        with orders_lock:
            symbol_orders = orders[orders['symbol'] == self.symbol]
            if not symbol_orders.empty:
                last_order_index = symbol_orders.index[-1]
                orders.drop(last_order_index, inplace=True)
                self.logger.info(f"Removed order for {self.symbol} from orders DataFrame.")
            else:
                self.logger.warning(f"No orders found for {self.symbol} to remove in orders DataFrame.")
        

        # Reset position state
        self._reset_position_state()
        self.last_trade_time = datetime.utcnow()

    def _reset_position_state(self):
        self.logger.debug("Resetting position state.")
        self.position = 0
        self.entry_price = 0.0
        self.tp_level = None
        self.sl_level = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None

    def run(self) -> None:
        self.logger.info(f"Starting trader for {self.symbol}")
        if not self._fetch_initial_data():
            self.logger.error("Failed to fetch initial data. Stopping.")
            return

        if not self._calculate_indicators():
            self.logger.warning("Initial indicator calculation failed. Will retry on next candle.")
        else:
            self._manage_position()

        # Print status after first run for last asset
        if self.symbol == ASSETS[-1]:
            print_trading_status()

        loop_count = 0
        while True:
            try:
                loop_count += 1
                server_now = self._get_server_time()
                
                # Validate DataFrame state
                if self.df.empty or len(self.df) < 2:
                    self.logger.error("DataFrame is empty or has insufficient data. Attempting to refetch initial data.")
                    if not self._fetch_initial_data():
                        self.logger.error("Failed to refetch initial data. Sleeping for 30 seconds.")
                        time.sleep(30)
                        continue

                # Use the last candle's timestamp (self.df.index[-1]) instead of [-2]
                last_candle_time_utc = self.df.index[-1].tz_localize(None)
                self.logger.debug(f"Last candle time: {last_candle_time_utc}, Server time: {server_now}")

                # Calculate next candle time
                next_candle_time_utc = last_candle_time_utc + pd.Timedelta(milliseconds=self.ms_interval)
                wait_seconds = (next_candle_time_utc - server_now).total_seconds() + 1

                # Validate wait time
                if wait_seconds < -self.ms_interval / 1000:
                    self.logger.warning(f"Calculated wait time is significantly negative ({wait_seconds:.1f}s). Possible timestamp mismatch. Last candle: {last_candle_time_utc}, Server: {server_now}")
                    wait_seconds = 0
                elif wait_seconds > self.ms_interval / 1000 + 5:
                    self.logger.warning(f"Calculated wait time ({wait_seconds:.1f}s) exceeds expected interval. Clamping to interval + 5s.")
                    wait_seconds = self.ms_interval / 1000 + 5

                if wait_seconds > 0:
                    self.logger.debug(f"Waiting {wait_seconds:.2f} seconds until next expected candle time ({next_candle_time_utc})...")
                    time.sleep(wait_seconds)

                if loop_count % 10 == 0:
                    self.logger.debug(f"Current DF shape: {self.df.shape}, Last candle: {self.df.index[-1]}, Close: {self.df['Close'].iloc[-1]:.4f}")
                    self.logger.debug(f"Current Position: {self.position}, Balance: {self.simulated_balance:.2f}")
                    self.logger.debug(f"---------------------------------")
    
                new_candle_fetched = self._fetch_latest_candle()
                if new_candle_fetched:
                    """global trending_assets
                    trending_assets=[]
                    get_trending_assets(client, MAX_TRENDING_ASSETS)
                    trending_assets=trending_assets[:12]"""
                    indicators_ok = self._calculate_indicators()
                    if indicators_ok:
                        self._manage_position()
                        self.logger.debug("Position management completed successfully.")
                        
                        # Print status after each candle for last asset
                        if self.symbol == ASSETS[-1]:
                            print_trading_status()
                    else:
                        self.logger.warning("Skipping position management due to indicator calculation issues on new candle.")
                else:
                    self.logger.debug("No new candle fetched in this cycle.")
                    time.sleep(min(15, self.ms_interval / 1000 / 4))

            except KeyboardInterrupt:
                self.logger.info("Forward testing stopped by user.")
                break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                self.logger.info("Attempting to continue after 30 seconds...")
                time.sleep(30)
        
        self.logger.info("Forward testing loop finished.")
        self.logger.info(f"Final Simulated Balance: ${self.simulated_balance:.2f}")
        performance = (self.simulated_balance / self.initial_balance - 1) * 100 if self.initial_balance else 0
        self.logger.info(f"Total Performance: {performance:.2f}%")
        tradereport.to_csv('trader_report.csv', index=False)

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logging.critical("Binance API Key/Secret not found.")
    else:
        client = Client(API_KEY, API_SECRET)
        
        # Get trending assets
        trending_assets = get_trending_assets(client, MAX_TRENDING_ASSETS)
        
        # Combine specified and trending assets
        all_assets = list(set(ASSETS + trending_assets))  # Using set to remove any duplicates
        
        threads = []
        for asset in all_assets:
            try:
                trader = ForwardIchimokuTrader(
                    symbol=asset,
                    interval=INTERVAL,
                    lookback=LOOKBACK_PERIODS,
                    tp_multiplier=TP_MULTIPLIER,
                    sp_multiplier=SP_MULTIPLIER,
                    leverage=LEVERAGE,
                    tc=TC,
                    initial_balance=SIMULATED_INITIAL_BALANCE
                )
                thread = threading.Thread(target=trader.run)
                threads.append(thread)
                thread.start()
                logging.info(f"Started trading thread for {asset}")
            except Exception as e:
                logging.critical(f"Failed to initialize trader for {asset}: {e}", exc_info=True)

        # Optionally wait for all threads to finish (though they run indefinitely unless stopped)
        for thread in threads:
            thread.join()
