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
from functools import lru_cache
# from tools import measure  # Not used, removed to fix import error
from dotenv import load_dotenv
from decimal import Decimal
import requests
import json
from telegram_reporter import setup_telegram_logging_from_env, TelegramReporter
from gemini_market_analyzer import GeminiMarketAnalyzer


 

multiplier_set={'ETHUSDT':[3.5,3],'BTCUSDT':[2,3],'SOLUSDT':[2,3.5],'XRPUSDT':[3.5,3],'BNBUSDT':[2.5,3],'TONUSDT':[1,3.5],'DOGEUSDT':[2,3.5],'TRXUSDT':[2,3],'LTCUSDT':[1.5,3.5],'GUNUSDT':[3,3.5],'TUTUSDT':[1,3.5],'ADAUSDT':[2.5,3.5],'XLMUSDT':[1,3.5],'VETUSDT':[3.5,3.5],'HBARUSDT':[1.5,3.5],'SANDUSDT':[3.5,3.5],'1000PEPEUSDT':[2,3.5],'1000BONKUSDT':[1.5,3],'GALAUSDT':[2.5,3.5],'FETUSDT':[0.5,3.5],"GRTUSDT":[3.5,3.5],'1000SHIBUSDT':[2.5,3.5],'DOTUSDT':[1,3.5],'LINKUSDT':[2.5,3],'AVAXUSDT':[1.5,3.5],'SUIUSDT':[1,3.5]}

adx_limit= {'ETHUSDT':80,'BTCUSDT':40,'SOLUSDT':30,'XRPUSDT':40,"TONUSDT":30,'DOGEUSDT':40,'TRXUSDT':70,'LTCUSDT':50,'ADAUSDT':50,'XLMUSDT':60,'VETUSDT':40,'HBARUSDT':40,'SANDUSDT':50,'1000PEPEUSDT':50,'1000BONKUSDT':70,'GALAUSDT':60,'FETUSDT':50,'GRTUSDT':60,'1000SHIBUSDT':40,'DOTUSDT':70,'LINKUSDT':50,'AVAXUSDT':40,'SUIUSDT':80}
#these will be used to make asset specifice decsion on tp and sl multiplier

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
API_KEY = "iOgcObLOw4UIFSvvEPXLFP1vgwp1wzyHYfw57vd1vrg19Xt6SXCE4RywDi5QoM28"
API_SECRET = "bz1m4UlthzklqXlWoqAqXZiJE35jjT0g5uJ5cQ43vwDNnsIpPYS5OqevfBVz84iK"


if not API_KEY or not API_SECRET:
    raise ValueError("Binance API credentials not found in environment variables. Please set BINANCE_API_KEY and BINANCE_API_SECRET.")

orders_lock = threading.Lock()
orders = pd.DataFrame(columns=['symbol', 'side', 'entry_price', 'tp_price', 'sl_price', 'choppy'])

tradereport = pd.DataFrame(columns=['symbol', 'side', 'entry_price', 'tp_price', 'sl_price', 'exit_price', 'choppy', 'profit'])

# Trading parameters
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK_PERIODS = 100
TP_MULTIPLIER = 1
SP_MULTIPLIER = 3.5
LEVERAGE = 2
TC = 0.0005

ASSETS = ['SANDUSDT','1000BONKUSDT']
MAX_TRENDING_ASSETS = 0  # Maximum number of trending assets to trade
MAX_CONCURRENT_TRADES = 2  # Set this to the desired number of concurrent trades
active_trades = []
active_trades_lock = threading.Lock()
ALLOCATION_MODE = 'dynamic_remaining'  # Options: 'fixed_per_slot', 'dynamic_remaining', 'full_available'

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading.log')  # Add file handler
    ]
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("binance").setLevel(logging.INFO)
telegram_reporter = setup_telegram_logging_from_env()

def create_binance_client_with_failover(api_key: str, api_secret: str) -> Client:
    """Create a Binance Client with higher timeout and endpoint failover.

    Works around connect timeouts to `api.binance.com` by retrying and attempting
    alternative API hosts. It temporarily overrides the Client.API_URL class attr
    during construction because the library pings on __init__.
    """
    requests_params = {'timeout': 30}
    # First try default endpoint with longer timeout, a couple retries
    last_exc = None
    for _ in range(2):
        try:
            client = Client(api_key, api_secret, requests_params=requests_params)
            client.futures_ping()
            return client
        except Exception as e:
            last_exc = e
            logging.info(f"Default Binance endpoint init failed: {e}")
            time.sleep(1)

    # Try known alternate endpoints
    alternate_endpoints = [
        'https://api1.binance.com',
        'https://api2.binance.com',
        'https://api3.binance.com',
        'https://api-gcp.binance.com',
    ]
    original_api_url = getattr(Client, 'API_URL', 'https://api.binance.com')
    for ep in alternate_endpoints:
        try:
            # Temporarily point the class-level API URL and construct
            Client.API_URL = ep
            client = Client(api_key, api_secret, requests_params=requests_params)
            client.futures_ping()
            logging.info(f"Initialized Binance client via alternate endpoint: {ep}")
            return client
        except Exception as e:
            last_exc = e
            logging.info(f"Alternate endpoint init failed on {ep}: {e}")
            time.sleep(1)
        finally:
            # Restore for next attempt/normal use
            Client.API_URL = original_api_url

    raise last_exc if last_exc else RuntimeError("Unable to initialize Binance client")

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
@lru_cache(maxsize=1)  # Cache last result
def BTC_trend_identification(client):
    """Identify BTC trend using cached data to reduce API calls"""
    try:
        # Get the last 2 hours of data (8 15-min candles)
        now = datetime.now()
        before = now - timedelta(hours=12)
        
        klines = client.futures_klines(
            symbol='ETHUSDT',
            interval=Client.KLINE_INTERVAL_15MINUTE,
            limit=48
        )
        
        if not klines:
            return None
            
        first_close = float(klines[0][4])  # Close price of first candle
        last_close = float(klines[-1][4])  # Close price of last candle
        
        return 'up' if last_close > first_close else 'down'
    except Exception as e:
        logging.error(f"Error in BTC trend identification: {e}")
        return None

# Global signal pool and lock for best trade selection
signal_pool = []
signal_pool_lock = threading.Lock()
trade_in_progress = threading.Event()

# Global Gemini analysis queue for batching API calls
gemini_analysis_queue = []
gemini_queue_lock = threading.Lock()
gemini_last_batch_time = None
gemini_batch_interval = 30  # Process queue every 30 seconds
gemini_results_cache = {}  # Cache to store results by (symbol, df_hash)

def get_df_hash(df):
    """Get a simple hash of the dataframe to identify if it's changed"""
    return hash(str(df.index[-1]) + str(df['Close'].iloc[-1]))

def add_to_gemini_queue(trader, df, signal_type, callback_func, **kwargs):
    """Add a Gemini analysis request to the queue instead of calling immediately"""
    global gemini_analysis_queue
    
    with gemini_queue_lock:
        gemini_analysis_queue.append({
            'trader': trader,
            'df': df,
            'symbol': trader.symbol,
            'signal_type': signal_type,
            'callback': callback_func,
            'kwargs': kwargs,
            'timestamp': datetime.now(UTC)
        })
        logging.debug(f"Added {trader.symbol} to Gemini queue. Queue size: {len(gemini_analysis_queue)}")

def process_gemini_queue():
    """Process the Gemini analysis queue in batches to save API calls"""
    global gemini_analysis_queue, gemini_last_batch_time, gemini_batch_interval
    
    try:
        with gemini_queue_lock:
            if not gemini_analysis_queue:
                return
            
            # Check if enough time has passed or queue is full
            now = datetime.now(UTC)
            time_since_last_batch = (now - gemini_last_batch_time).total_seconds() if gemini_last_batch_time else float('inf')
            
            # Process if interval passed or queue has 2+ items
            if time_since_last_batch >= gemini_batch_interval or len(gemini_analysis_queue) >= 2:
                pending_analyses = gemini_analysis_queue.copy()
                gemini_analysis_queue.clear()
                gemini_last_batch_time = now
            else:
                return
        
        # Group by trader (should only have one per trader, but safe to group)
        trader_batches = {}
        for item in pending_analyses:
            trader = item['trader']
            if trader not in trader_batches:
                trader_batches[trader] = []
            trader_batches[trader].append(item)
        
        # Get the first trader that has a gemini client
        processing_trader = None
        for trader in trader_batches.keys():
            if hasattr(trader, 'gemini') and trader.gemini is not None:
                processing_trader = trader
                break
        
        if not processing_trader or not hasattr(processing_trader, 'gemini') or processing_trader.gemini is None:
            logging.warning("No trader with Gemini client available. Requeuing items.")
            with gemini_queue_lock:
                gemini_analysis_queue.extend(pending_analyses)
            return
        
        # Prepare batch analyses for Gemini
        batch_analyses = []
        pending_items = []
        
        for item in pending_analyses:
            batch_analyses.append({
                'df': item['df'],
                'symbol': item['symbol']
            })
            pending_items.append(item)
        
        if not batch_analyses:
            return
        
        # Call Gemini batch analysis (ONE API call for multiple symbols)
        logging.info(f"Processing Gemini batch for {len(batch_analyses)} symbols: {[a['symbol'] for a in batch_analyses]}")
        try:
            results = processing_trader.gemini.analyze_multiple_consolidations(batch_analyses)
            
            # Process results and call callbacks
            for item in pending_items:
                symbol = item['symbol']
                if symbol in results:
                    is_consolidating, reasoning = results[symbol]
                    # Call the callback with the result
                    kwargs = item.get('kwargs', {})
                    item['callback'](is_consolidating, reasoning, item['signal_type'], **kwargs)
                else:
                    logging.error(f"No result returned for {symbol} from batch analysis")
                    # Default to allowing trade if analysis fails
                    kwargs = item.get('kwargs', {})
                    item['callback'](False, "Analysis failed - defaulting to allow", item['signal_type'], **kwargs)
                    
        except Exception as e:
            logging.error(f"Error in batch Gemini analysis: {e}", exc_info=True)
            # Requeue items on error
            with gemini_queue_lock:
                gemini_analysis_queue.extend(pending_items)
            
    except Exception as e:
        logging.error(f"Error processing Gemini queue: {e}", exc_info=True)

# Trading inhibition system
trading_inhibited = False
inhibition_start_time = None
inhibition_trigger_trade_count = 0  # Track how many trades existed when inhibition was triggered
INHIBITION_DURATION_MINUTES = 30

# News-based trading inhibition system removed per user request


def check_last_two_trades_and_manage_inhibition():
    """Check if last 2 trades were losses and manage trading inhibition"""
    global trading_inhibited, inhibition_start_time, inhibition_trigger_trade_count
    
    try:
        # Check if we have at least 2 completed trades
        if len(tradereport) < 2:
            return True  # Allow trading if less than 2 trades
        
        # If currently inhibited, check if time has expired
        if trading_inhibited:
            if inhibition_start_time:
                time_elapsed = datetime.now(UTC) - inhibition_start_time
                if time_elapsed.total_seconds() >= INHIBITION_DURATION_MINUTES * 60:
                    # Check if there are new trades since inhibition started
                    current_trade_count = len(tradereport)
                    if current_trade_count > inhibition_trigger_trade_count+2:
                        # There are new trades, check if they are losses
                        new_trades = tradereport.iloc[inhibition_trigger_trade_count:]
                        if len(new_trades) >= 2:
                            # Check if the last 2 new trades are losses
                            last_two_new_trades = new_trades.tail(2)
                            both_losses = all(trade['profit'] < 0 for _, trade in last_two_new_trades.iterrows())
                            if both_losses:
                                # Extend inhibition for another 30 minutes
                                inhibition_start_time = datetime.now(UTC)
                                inhibition_trigger_trade_count = current_trade_count
                                logging.warning(f"Last 2 new trades were also losses. Extending inhibition for another {INHIBITION_DURATION_MINUTES} minutes.")
                                logging.warning(f"New trades: {last_two_new_trades[['symbol', 'profit']].to_dict('records')}")
                                return False
                    
                    # No new trades or new trades are not both losses, resume trading
                    trading_inhibited = False
                    inhibition_start_time = None
                    inhibition_trigger_trade_count = 0
                    logging.info("Trading inhibition period expired. Resuming normal trading.")
                    return True
                else:
                    remaining_minutes = INHIBITION_DURATION_MINUTES - (time_elapsed.total_seconds() / 60)
                    logging.debug(f"Trading still inhibited. {remaining_minutes:.1f} minutes remaining.")
                    return False
            else:
                # Reset if start time is missing
                trading_inhibited = False
                inhibition_trigger_trade_count = 0
                return True
        
        # Not currently inhibited, check if we should start inhibition
        # Get the last 2 trades
        last_two_trades = tradereport.tail(2)
        
        # Check if both were losses
        both_losses = all(trade['profit'] < 0 for _, trade in last_two_trades.iterrows())
        
        if both_losses:
            # Start inhibition period
            trading_inhibited = True
            inhibition_start_time = datetime.now(UTC)
            inhibition_trigger_trade_count = len(tradereport)
            logging.warning(f"Last 2 trades were losses. Inhibiting trading for {INHIBITION_DURATION_MINUTES} minutes.")
            logging.warning(f"Last 2 trades: {last_two_trades[['symbol', 'profit']].to_dict('records')}")
            return False
        
        return True  # Allow trading if not both losses
        
    except Exception as e:
        logging.error(f"Error checking last two trades: {e}")
        return True  # Allow trading on error

def close_all_existing_trades():
    # News system removed; function retained only if referenced elsewhere
    return False

def print_trading_status():
    """Print current orders and trade reports with formatting."""
    # Clear some space before printing
    print("\n\n")

    # Print current trades
    print("="*100)
    print('BALANCE:'.center(100))
    print('='*100)
    available_balance = client.futures_account_balance()
    usdt_balance = 0
    for balance in available_balance:
        if balance['asset'] == 'USDT':
            usdt_balance = float(balance['availableBalance'])  
    print(f"Available balance: {usdt_balance} USDT".center(100))
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
    
    # Add profit summary section
    if len(tradereport) > 0 and 'profit' in tradereport.columns:
        print("\n" + "="*100)
        print("PROFIT SUMMARY:".center(100))
        print("="*100)
        
        # Calculate profit statistics
        total_profit = tradereport['profit'].sum()
        winning_trades = len(tradereport[tradereport['profit'] > 0])
        losing_trades = len(tradereport[tradereport['profit'] < 0])
        total_trades = len(tradereport)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"Total Profit/Loss: {total_profit:.4f} USDT".center(100))
        print(f"Winning Trades: {winning_trades}/{total_trades} ({win_rate:.1f}%)".center(100))
        print(f"Losing Trades: {losing_trades}/{total_trades}".center(100))
        
        if winning_trades > 0:
            avg_win = tradereport[tradereport['profit'] > 0]['profit'].mean()
            print(f"Average Win: {avg_win:.4f} USDT".center(100))
        
        if losing_trades > 0:
            avg_loss = tradereport[tradereport['profit'] < 0]['profit'].mean()
            print(f"Average Loss: {avg_loss:.4f} USDT".center(100))
        
        print("="*100)
    
    # Add trading inhibition status
    print("\n" + "="*100)
    print("TRADING STATUS:".center(100))
    print("="*100)
    
    # Check inhibition (news system removed)
    loss_inhibited = trading_inhibited
    if loss_inhibited:
        if inhibition_start_time:
            time_elapsed = datetime.now(UTC) - inhibition_start_time
            remaining_minutes = INHIBITION_DURATION_MINUTES - (time_elapsed.total_seconds() / 60)
            print(f"TRADING INHIBITED - Losses ({remaining_minutes:.1f} min remaining)".center(100))
        else:
            print("TRADING INHIBITED - Losses (Time unknown)".center(100))
    else:
        print("TRADING ACTIVE".center(100))
    
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
        csv_file = 'trades/trade_history_live.csv'
        
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

# Add after other global/threading variables
signal_pool = []
signal_pool_lock = threading.Lock()
trade_in_progress = threading.Event()

class ForwardIchimokuTrader:
    def __init__(self, symbol: str, interval: str, lookback: int,
                 tp_multiplier: float = 1, sp_multiplier: float = 2.5,
                 leverage: int = 1, tc: float = 0.0005) -> None:
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
        

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{symbol}")
        self.logger.info("Initializing Trader...")

        # Initialize Gemini AI analyzer for consolidation detection
        try:
            self.gemini = GeminiMarketAnalyzer(api_key='AIzaSyAdyo9u1iFoyoqSXCG3h38ADtuZHmc85vg')
            self.logger.info("Gemini AI analyzer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini AI: {e}")
            self.gemini = None

        try:
            self.client = create_binance_client_with_failover(API_KEY, API_SECRET)
            # Test futures endpoint specifically
            self.client.futures_ping()
            # Set initial futures mode to one-way position mode
            try:
                self.client.futures_change_position_mode(dualSidePosition=False)
            except Exception as e:
                if "No need to change position side" not in str(e):
                    raise
            
            # Clean any existing orders during initialization
            self._cleaning_existing_order()
            
            self.logger.info("Binance Futures client initialized and connection tested.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance Futures client: {e}", exc_info=True)
            raise

        self.df = pd.DataFrame()
        self.position = 0
        self.entry_price = 0.0
        self.tp_level = None
        self.sl_level = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.last_trade_time = None
        self.position_size = 0.0  # Store the actual position size
        self.orderid = None  # Stop loss order ID
        self.tp_orderid = None  # Take profit order ID
        self.processed_orders = set()  # Track processed orders to avoid duplicates

        self.ms_interval = interval_to_milliseconds(self.interval)
        if not self.ms_interval:
            raise ValueError("Invalid interval for millisecond conversion")
        self.logger.info(f"Interval: {self.interval}, Milliseconds: {self.ms_interval}")

    def __eq__(self, other):
        if not isinstance(other, ForwardIchimokuTrader):
            return False
        return self.symbol == other.symbol

    def __hash__(self):
        return hash(self.symbol)

    def _get_server_time(self):
        try:
            server_time = self.client.futures_time()
            server_time_ms = server_time['serverTime']
            return pd.to_datetime(server_time_ms, unit='ms', utc=True)  # Make sure it's UTC timezone-aware
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            return datetime.now(UTC)  # Return timezone-aware datetime
    
    def _fetch_initial_data(self) -> bool:
        self.logger.info(f"Fetching initial {self.lookback + 50} klines for {self.symbol}...")
        try:
            start=str(self._get_server_time()-timedelta(minutes=1600))
            klines = self.client.futures_historical_klines(symbol=self.symbol, interval=self.interval, start_str=start)
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

            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms', utc=True)  # Make timezone-aware
            data.set_index('Datetime', inplace=True)
            self.df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            #print(self.df.iloc[-1]) for debugging 
            if len(self.df) < self.required_lookback:
                self.logger.error(f"Insufficient valid initial data fetched ({len(self.df)} rows). Need at least {self.required_lookback}.")
                return False

            if not self.df.index.is_monotonic_increasing:
                self.logger.warning("Initial DataFrame index is not monotonic increasing. Sorting...")
                self.df.sort_index(inplace=True)

            self.logger.info(f"Successfully fetched and processed {len(self.df)} initial candles. Last candle time: {self.df.index[-1]}")
            return True
        except Exception as e:
            self.logger.error(f"Error fetching initial data: {e}", exc_info=True)
            return False

    def _fetch_latest_candle(self) -> bool:
        self.logger.debug("Attempting to fetch latest candle...")
        try:
            start=str(self._get_server_time()-timedelta(minutes=60))
            klines = self.client.futures_historical_klines(symbol=self.symbol, interval=self.interval, start_str=start)
            klines=klines[:-1]
            if not klines or len(klines) < 2:
                self.logger.warning("Could not fetch latest klines or not enough data yet (< 2).")
                return False

            latest_kline = klines[-1]
            close_time_ms = latest_kline[6]
            latest_dt = pd.to_datetime(close_time_ms, unit='ms', utc=True)  # Make timezone-aware

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

            self.df = pd.concat([self.df, new_row.to_frame().T])
            if not self.df.index.is_monotonic_increasing:
                self.logger.warning(f"Index became non-monotonic after adding candle {latest_dt}. Sorting...")
                self.df.sort_index(inplace=True)

            max_len = self.lookback + 100
            if len(self.df) > max_len:
                self.logger.debug(f"Trimming DataFrame from {len(self.df)} to {max_len} rows.")
                self.df = self.df.iloc[-max_len:]

            self.logger.info(f"New candle appended: {self.df.index[-1]}, Close: {self.df['Close'].iloc[-1]:.4f}, DF rows: {len(self.df)}")
            return True
        except Exception as e:
            self.logger.error(f"Error fetching/appending latest candle: {e}", exc_info=True)
            return False

    def _calculate_future_kumo(self, df: pd.DataFrame) -> tuple[float, float]:
        """Calculate future Kumo (cloud) values for the period 26 bars ahead."""
        try:
            # Get the current conversion and base lines
            current_conversion = df['conversion line'].iloc[-1]
            current_base = df['base line'].iloc[-1]
            
            # Calculate future Senkou Span A (26 periods ahead)
            # Future Span A = (Conversion Line + Base Line) / 2 shifted 26 periods forward
            future_span_a = (current_conversion + current_base) / 2
            
            # Calculate future Senkou Span B (26 periods ahead)
            # Future Span B = (52-period high + 52-period low) / 2 shifted 26 periods forward
            period_52_high = self.df['High'].rolling(window=52).max().iloc[-1]
            period_52_low = self.df['Low'].rolling(window=52).min().iloc[-1]
            future_span_b = (period_52_high + period_52_low) / 2
            
            return future_span_a, future_span_b
        except Exception as e:
            self.logger.error(f"Error calculating future Kumo: {e}")
            return None, None

    def _calculate_indicators(self) -> bool:
        self.logger.debug(f"Calculating indicators for {len(self.df)} rows...")
        min_data_needed = self.required_lookback
        if len(self.df) < min_data_needed:
            self.logger.warning(f"Not enough data ({len(self.df)}, need {min_data_needed}) for all indicator calculations.")
            return False
        try:
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
            
            # Calculate future Kumo values
            future_span_a, future_span_b = self._calculate_future_kumo(temp_df_ichi)
            if future_span_a is not None and future_span_b is not None:
                temp_df_ichi['future_span_a'] = future_span_a
                temp_df_ichi['future_span_b'] = future_span_b

            if len(self.df) >= 15:
                self.df['atr'] = ta.atr(self.df['High'],self.df['Low'],self.df['Close'], length=14)
                self.df['choppy'] = ta.chop(self.df['High'],self.df['Low'],self.df['Close'])
                adx=ta.adx(self.df['High'],self.df['Low'],self.df['Close'])
                self.df['ADX_14'] = adx['ADX_14']
                psar_data = ta.psar(self.df['High'],self.df['Low'],self.df['Close'])

                psar_cols_to_drop = ['PSAR_Long', 'PSAR_Short', 'PSAR_Reversal', 'PSARaf_0.02_0.2', 
                                   'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARr_0.02_0.2']
                existing_cols_to_drop = [col for col in psar_cols_to_drop if col in self.df.columns]
                if existing_cols_to_drop:
                    self.df = self.df.drop(columns=existing_cols_to_drop)
                    self.logger.debug(f"Dropped existing PSAR columns: {existing_cols_to_drop}")
                
                # Rename PSAR columns to our standard names
                psar_data = psar_data.rename(columns={'PSARl_0.02_0.2':'PSAR_Long',"PSARs_0.02_0.2":'PSAR_Short','PSARr_0.02_0.2':"PSAR_Reversal"})
                
                # Ensure PSAR has the same index as the main DataFrame
                if not psar_data.empty:
                    psar_data.index = self.df.index[-len(psar_data):]
                    
                    # Use join instead of concat to avoid index issues
                    self.df = self.df.join(psar_data)
            else:
                self.logger.warning(f"Not enough data ({len(self.df)}) for ATR(14) calculation.")
                self.df['atr'] = np.nan

            ichimoku_cols_to_drop = [col for col in temp_df_ichi.columns if col in self.df.columns]
            if ichimoku_cols_to_drop:
                self.df = self.df.drop(columns=ichimoku_cols_to_drop)

            self.df = self.df.join(temp_df_ichi)
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
            self.logger.warning(f"DataFrame length ({current_length}) is less than 27. Cannot perform lagging span comparison yet.")
            return False, False

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

        current_close = last['Close']
        conversion_line = last['conversion line']
        base_line = last['base line']
        leading_span_A = last['leading Span A']
        leading_span_B = last['leading Span B']
        choppy = last['choppy']
        leading_Span_A_shifted = self.df['leading Span A'].iloc[-26]
        adx= last['ADX_14']

        

        # Check if PSAR columns exist and get values
        psar_long = None
        psar_short = None
        
        if 'PSAR_Long' in self.df.columns and 'PSAR_Short' in self.df.columns:
            psar_long = last['PSAR_Long']
            psar_short = last['PSAR_Short']
            
            # Handle Series objects
            if isinstance(psar_long, pd.Series):
                psar_long = psar_long.iloc[-1]
            if isinstance(psar_short, pd.Series):
                psar_short = psar_short.iloc[-1]
                
            # Handle numpy/pandas objects
            if hasattr(psar_long, 'item'):
                try:
                    psar_long = psar_long.item()
                except Exception:
                    pass
            if hasattr(psar_short, 'item'):
                try:
                    psar_short = psar_short.item()
                except Exception:
                    pass
        

        # Get future Kumo values
        future_span_a = last.get('future_span_a', None)
        future_span_b = last.get('future_span_b', None)
        
        trend = BTC_trend_identification(self.client)

        # Check if future Kumo is bullish or bearish
        future_kumo_bullish = (future_span_a is not None and future_span_b is not None and future_span_a >= future_span_b)
        future_kumo_bearish = (future_span_a is not None and future_span_b is not None and future_span_a <= future_span_b)

        buy_signal = (
            (current_close > close_t_minus_26)
            and (conversion_line >= base_line)
            
            and (current_close > leading_span_A)
            and (current_close > leading_span_B)
            and future_kumo_bullish  # Add future Kumo check
            and (not(pd.isna(psar_long)))  # Only checks if PSAR exists, not its value
            
            and adx < adx_limit[self.symbol]
            
        )
        sell_signal = (
            (current_close < close_t_minus_26) 
            and (conversion_line <= base_line) 
            
            and (current_close < leading_span_A)
            and (current_close < leading_span_B)
            and future_kumo_bearish  # Add future Kumo check
            and (not(pd.isna(psar_short)))
              # Only checks if PSAR exists, not its value
            and adx < adx_limit[self.symbol]
        )

        # Store signals for potential use in callbacks
        self._pending_buy_signal = buy_signal
        self._pending_sell_signal = sell_signal
        
        # Gemini AI consolidation check - blocks trades in choppy markets
        if buy_signal or sell_signal:
            try:
                if hasattr(self, 'gemini') and self.gemini is not None:
                    # Check if we're already waiting for a Gemini result
                    if not hasattr(self, '_gemini_pending') or not self._gemini_pending:
                        # Queue Gemini analysis instead of calling directly
                        signal_type = "BUY" if buy_signal else "SELL"
                        self._gemini_pending = True
                        add_to_gemini_queue(
                            trader=self,
                            df=self.df,
                            signal_type=signal_type,
                            callback_func=self._handle_gemini_result,
                            buy_signal=buy_signal,
                            sell_signal=sell_signal
                        )
                    # Return False to block signal until Gemini approves
                    return False, False
            except Exception as e:
                self.logger.error(f"Error queueing Gemini check for {self.symbol}: {e}")
                # Continue with original signals if Gemini check fails

        self.logger.debug(f"15min. Signals @ {last_index_name}: Buy={buy_signal}, Sell={sell_signal}")
        self.logger.debug(f"Signal components - Close: {current_close:.4f}, T-26: {close_t_minus_26:.4f}, Conv: {conversion_line:.4f}, Base: {base_line:.4f}")
        self.logger.debug(f"Current Kumo - Span A: {leading_span_A:.4f}, Span B: {leading_span_B:.4f}")
        self.logger.debug(f"Future Kumo - Span A: {str(future_span_a) if future_span_a is not None else 'None'}, Span B: {str(future_span_b) if future_span_b is not None else 'None'}")
        self.logger.debug(f"PSAR - Long: {str(psar_long) if psar_long is not None else 'None'}, Short: {str(psar_short) if psar_short is not None else 'None'}")
        return buy_signal, sell_signal
    
    def _handle_gemini_result(self, is_consolidating, reasoning, signal_type, buy_signal, sell_signal):
        """Callback to handle Gemini analysis result and execute trade if approved"""
        # Clear pending flag
        self._gemini_pending = False
        
        try:
            if is_consolidating:
                self.logger.info(
                    f"🧠 Gemini detected consolidation for {self.symbol}: {reasoning}. "
                    f"Trade BLOCKED despite signal."
                )
                # Send Telegram notification about blocked trade
                try:
                    if 'telegram_reporter' in globals() and telegram_reporter:
                        signal_type_text = "BUY" if buy_signal else "SELL"
                        price = self.df['Close'].iloc[-1]
                        telegram_reporter.send(
                            f"⚠️ <b>Trade Blocked</b> - <code>{self.symbol}</code>\n"
                            f"Signal: {signal_type_text} @ ${price:.4f}\n"
                            f"Reason: Market is consolidating\n"
                            f"💡 {reasoning}"
                        )
                except Exception as e:
                    self.logger.error(f"Failed to send Telegram notification: {e}")
            else:
                self.logger.info(
                    f"🧠 Gemini confirmed trending market for {self.symbol}: {reasoning}. "
                    f"Trade ALLOWED - submitting signal."
                )
                # Send Telegram notification about allowed trade
                try:
                     if 'telegram_reporter' in globals() and telegram_reporter:
                         signal_type_text = "BUY" if buy_signal else "SELL"
                         price = self.df['Close'].iloc[-1]
                         telegram_reporter.send(
                             f"✅ <b>Trade Allowed</b> - <code>{self.symbol}</code>\n"
                             f"Signal: {signal_type_text} @ ${price:.4f}\n"
                             f"Status: Market is trending\n"
                             f"💡 {reasoning}"
                         )
                except Exception as e:
                     self.logger.error(f"Failed to send Telegram notification: {e}")
                
                # Get latest data and submit signal if still valid
                try:
                    last_row = self.df.iloc[-1]
                    current_price = last_row['Close']
                    atr = last_row['atr']
                    choppy = last_row['choppy']
                    
                    # Submit signal directly without re-checking (avoid recursive Gemini calls)
                    if buy_signal:
                        self.logger.info(f"Submitting BUY signal for {self.symbol} after Gemini approval")
                        self._submit_signal('buy', current_price, atr, choppy)
                    elif sell_signal:
                        self.logger.info(f"Submitting SELL signal for {self.symbol} after Gemini approval")
                        self._submit_signal('sell', current_price, atr, choppy)
                except Exception as e:
                    self.logger.error(f"Error submitting signal after Gemini approval for {self.symbol}: {e}", exc_info=True)
                    
        except Exception as e:
            self.logger.error(f"Error handling Gemini result for {self.symbol}: {e}", exc_info=True)
    
    def _check_tp_sl_hits(self):
        """Check for TP/SL hits and add them to trade report"""
        try:
            # Get orders for the current symbol first to check for TP/SL hits
            symbol_orders = self.client.futures_get_all_orders(symbol=self.symbol)
            
            # Check for TP/SL hit orders and add to tradereport
            self.logger.debug(f"Checking {len(symbol_orders)} orders for {self.symbol} for TP/SL hits...")
            self.logger.debug(f"Current position state: {self.position}, entry_price: {self.entry_price}, position_size: {self.position_size}")
            hit_detected_via_orders = False
            for order in symbol_orders:
                try:
                    order_type = order.get('type')
                    order_status = order.get('status')
                    order_id = order.get('orderId')
                    is_reduce_only = bool(order.get('reduceOnly', False))
                except Exception:
                    # Skip malformed orders
                    continue

                # Only consider filled reduce-only TP/SL orders
                if (
                    order_status == 'FILLED' and
                    order_type in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'STOP_LOSS', 'TAKE_PROFIT'] and
                    is_reduce_only
                ):
                    # Check if we've already processed this order
                    if order_id in self.processed_orders:
                        continue
                    
                    self.logger.info(f"Detected FILLED {order_type} (reduce-only) order {order_id} for {self.symbol} - processing trade closure")
                    self.processed_orders.add(order_id)
                    hit_detected_via_orders = True
                    
                    # Derive a reliable exit price
                    exit_price = 0.0
                    avg_price_str = order.get('avgPrice')
                    price_str = order.get('price')
                    stop_price_str = order.get('stopPrice') if 'stopPrice' in order else None
                    try:
                        if avg_price_str and float(avg_price_str) > 0:
                            exit_price = float(avg_price_str)
                        elif price_str and float(price_str) > 0:
                            exit_price = float(price_str)
                        elif stop_price_str and float(stop_price_str) > 0:
                            exit_price = float(stop_price_str)
                    except Exception:
                        # Fall back quietly; will remain 0.0 if not derivable
                        pass
                        
                        # Determine original position side based on current position state
                        # Use the actual position side from the trader's state
                        if self.position == 1:
                            original_position_side = 'BUY'  # Long position
                        elif self.position == -1:
                            original_position_side = 'SELL'  # Short position
                        else:
                            # Fallback: try to determine from order side (TP/SL orders are opposite to position)
                            original_position_side = 'SELL' if order['side'] == 'BUY' else 'BUY'
                        
                        # Calculate profit/loss
                        if self.position_size > 0:
                            if original_position_side == 'BUY':  # Long position closed
                                profit = (exit_price - self.entry_price) * self.position_size
                            else:  # Short position closed
                                profit = (self.entry_price - exit_price) * self.position_size
                            self.logger.info(f"PnL calculation: {original_position_side} position, entry: {self.entry_price}, exit: {exit_price}, size: {self.position_size}, profit: {profit}")
                        else:
                            # Fallback: try to get position size from the order
                            order_qty = float(order['executedQty']) if 'executedQty' in order else 0
                            if order_qty > 0:
                                if original_position_side == 'BUY':  # Long position closed
                                    profit = (exit_price - self.entry_price) * order_qty
                                else:  # Short position closed
                                    profit = (self.entry_price - exit_price) * order_qty
                                self.logger.info(f"PnL calculation (fallback): {original_position_side} position, entry: {self.entry_price}, exit: {exit_price}, size: {order_qty}, profit: {profit}")
                            else:
                                profit = 0.0  # Cannot calculate profit without position size
                                self.logger.warning(f"Cannot calculate PnL: no position size available")
                        
                        trade_data = {
                            'symbol': self.symbol,
                            'side': original_position_side,  # Original position side
                            'entry_price': self.entry_price,
                            'tp_price': self.tp_level,
                            'sl_price': self.sl_level,
                            'exit_price': exit_price,
                            'choppy': 'N/A',  # Not applicable for TP/SL hits
                            'profit': round(profit, 4)
                        }
                        
                        # Add to tradereport
                        global tradereport
                        with orders_lock:
                            tradereport = pd.concat([tradereport, pd.DataFrame([trade_data])], ignore_index=True)
                        self.logger.info(f"TP/SL hit recorded for {self.symbol}: {order_type} at {trade_data['exit_price']}, profit: {trade_data['profit']}")
                        self.logger.info(f"Trade data added: {trade_data}")
                        try:
                            if 'telegram_reporter' in globals() and telegram_reporter:
                                side_text = trade_data['side']
                                exit_price_text = f"{trade_data['exit_price']:.4f}"
                                pnl_text = f"{trade_data['profit']:.4f} USDT"
                                telegram_reporter.send(
                                    f"<b>Closed</b> <code>{self.symbol}</code> {side_text} @ {exit_price_text}\n"
                                    f"PnL: {pnl_text}"
                                )
                                self.logger.info(f"Telegram PnL notification sent for {self.symbol}")
                        except Exception as e:
                            self.logger.error(f"Failed to send Telegram PnL notification: {e}")
                        # Reset position state after TP/SL hit
                        self.position = 0
                        self.entry_price = 0.0
                        self.position_size = 0.0
                        self.tp_level = None
                        self.sl_level = None
                        self.highest_price_since_entry = None
                        self.lowest_price_since_entry = None
                        self.orderid = None
                        self.tp_orderid = None
                        
                        # Remove from active trades
                        with active_trades_lock:
                            if self in active_trades:
                                active_trades.remove(self)
                                self.logger.info(f"Removed {self.symbol} from active trades after TP/SL hit")
                        
                        # Immediately display updated completed trades and persist
                        try:
                            print_trading_status()
                            self.logger.info("Trading status printed successfully")
                        except Exception as e:
                            self.logger.error(f"Could not print trading status: {e}")
            
            # Check if position is still active on exchange (after processing orders)
            if self.position != 0 and not hit_detected_via_orders:
                positions = self.client.futures_position_information()
                active_position = next((p for p in positions if p['symbol'] == self.symbol and float(p['positionAmt']) != 0), None)
                
                if not active_position:
                    # Position was closed but we couldn't detect it through orders or price-cross
                    self.logger.info(f"Position for {self.symbol} was closed but not detected through orders or price-cross. Resetting state.")
                    self._reset_position_state()
                    return
            
            # Fallback: price-based TP/SL cross detection when orders do not show fills
            if not hit_detected_via_orders and self.position != 0 and self.tp_level is not None and self.sl_level is not None:
                try:
                    last_high = float(self.df['High'].iloc[-1])
                    last_low = float(self.df['Low'].iloc[-1])
                    exit_reason = None
                    exit_price = None
                    original_position_side = 'BUY' if self.position == 1 else 'SELL'

                    if self.position == 1:
                        if last_high >= self.tp_level:
                            exit_reason = 'TP'
                            exit_price = self.tp_level
                        elif last_low <= self.sl_level:
                            exit_reason = 'SL'
                            exit_price = self.sl_level
                    elif self.position == -1:
                        if last_low <= self.tp_level:
                            exit_reason = 'TP'
                            exit_price = self.tp_level
                        elif last_high >= self.sl_level:
                            exit_reason = 'SL'
                            exit_price = self.sl_level

                    if exit_reason is not None and exit_price is not None:
                        # Extra visibility for price-cross path
                        self.logger.info(
                            f"Price-cross {exit_reason} for {self.symbol}: last_high={last_high}, last_low={last_low}, tp={self.tp_level}, sl={self.sl_level}, exit={exit_price}"
                        )
                        # Record as a price-cross detected closure   
                        if self.position_size > 0:
                            if original_position_side == 'BUY':
                                profit = (exit_price - self.entry_price) * self.position_size
                            else:
                                profit = (self.entry_price - exit_price) * self.position_size
                        else:
                            profit = 0.0

                        trade_data = {
                            'symbol': self.symbol,
                            'side': original_position_side,
                            'entry_price': self.entry_price,
                            'tp_price': self.tp_level,
                            'sl_price': self.sl_level,
                            'exit_price': exit_price,
                            'choppy': 'N/A',
                            'profit': round(profit, 4)
                        }

                        self.logger.info(f"Price-cross {exit_reason} detected for {self.symbol} at {exit_price}. Recording trade and resetting state.")
                        with orders_lock:
                            tradereport = pd.concat([tradereport, pd.DataFrame([trade_data])], ignore_index=True)

                        try:
                            if 'telegram_reporter' in globals() and telegram_reporter:
                                side_text = trade_data['side']
                                exit_price_text = f"{trade_data['exit_price']:.4f}"
                                pnl_text = f"{trade_data['profit']:.4f} USDT"
                                telegram_reporter.send(
                                    f"<b>Closed</b> <code>{self.symbol}</code> {side_text} @ {exit_price_text}\n"
                                    f"PnL: {pnl_text}"
                                )
                                self.logger.info(f"Telegram PnL notification sent for {self.symbol} (price-cross)")
                        except Exception as e:
                            self.logger.error(f"Failed to send Telegram PnL notification (price-cross): {e}")

                        # Attempt to clean any remaining reduce-only orders
                        try:
                            self._cleaning_existing_order()
                        except Exception as e:
                            self.logger.error(f"Failed cleaning orders after price-cross closure: {e}")

                        # Reset internal state
                        self.position = 0
                        self.entry_price = 0.0
                        self.position_size = 0.0
                        self.tp_level = None
                        self.sl_level = None
                        self.highest_price_since_entry = None
                        self.lowest_price_since_entry = None
                        self.orderid = None
                        self.tp_orderid = None

                        with active_trades_lock:
                            if self in active_trades:
                                active_trades.remove(self)
                                self.logger.info(f"Removed {self.symbol} from active trades after price-cross {exit_reason}")

                        try:
                            print_trading_status()
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.error(f"Error during price-cross TP/SL detection: {e}")
        except Exception as e:
            self.logger.error(f"Error checking TP/SL hits: {e}", exc_info=True)

    def _update_sl(self, new_sl_price):
        try:
            if not hasattr(self, '_price_precision') or self._price_precision is None:
                exchange_info = self.client.futures_exchange_info()
                symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == self.symbol)
                self._price_precision = 2
                for f in symbol_info['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size_str = f['tickSize']
                        if Decimal(tick_size_str) < 1:
                            self._price_precision = abs(Decimal(tick_size_str).as_tuple().exponent)
                        break
            
            self.sl_level = round(new_sl_price, self._price_precision)
            
            # Cancel existing SL orders first
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            for order in open_orders:
                if order['type'] == 'STOP_MARKET' and order['reduceOnly']:
                    self.client.futures_cancel_order(symbol=self.symbol, orderId=order['orderId'])
                    self.logger.info(f"Cancelled existing stop-loss order {order['orderId']}")
            
            time.sleep(0.2) # Give time for cancellation to process

            # Create new SL order
            side = Client.SIDE_SELL if self.position == 1 else Client.SIDE_BUY
            sl_params = {
                'symbol': self.symbol, 'side': side, 'type': 'STOP_MARKET',
                'quantity': self.position_size, 'stopPrice': self.sl_level, 'reduceOnly': True
            }
            sl_order = self.client.futures_create_order(**sl_params)
            self.orderid = sl_order['orderId']
            self.logger.info(f"SUCCESS: Updated trailing stop. New SL order ID: {self.orderid} at {self.sl_level}")

        except Exception as e:
            self.logger.error(f"FAILED to update SL order: {e}", exc_info=True)
            self.orderid = None

    


    def manage_position(self) -> None:

        try:
            # Always check for TP/SL hits first - this should run regardless of position state
            self._check_tp_sl_hits()
            
            # Always get the real-time position status from the exchange
            positions = self.client.futures_position_information()
            active_position = next((p for p in positions if p['symbol'] == self.symbol and float(p['positionAmt']) != 0), None)

            # If the bot thinks it's in a position, but the exchange says it's not, reset.
            if self.position != 0 and not active_position:
                self.logger.info(f"Position for {self.symbol} is closed on the exchange. Resetting state.")
                self._reset_position_state()
                return

            # If there's no active position, check for new signals.
            if not active_position:
                last_row = self.df.iloc[-1]
                current_price = last_row['Close']
                atr = last_row['atr']
                buy_signal, sell_signal = self._check_signals()
                self.logger.info(f"Signal check for {self.symbol}:  buy_signal={buy_signal},  sell_signal={sell_signal}")
                if buy_signal:
                    self.logger.info(f"Submitting BUY signal for {self.symbol}")
                    self._submit_signal('buy', current_price, atr, last_row['choppy'])
                elif sell_signal:
                    self.logger.info(f"Submitting SELL signal for {self.symbol}")
                    self._submit_signal('sell', current_price, atr, last_row['choppy'])
                return

            # If we are here, there is an active position to manage.
            # Sync the bot's internal state with the exchange.
            self.position = 1 if float(active_position['positionAmt']) > 0 else -1
            self.entry_price = float(active_position['entryPrice'])
            self.position_size = abs(float(active_position['positionAmt']))

            # Now, manage the trailing stop.
            last_row = self.df.iloc[-1]
            current_price = last_row['Close']
            atr = last_row['atr']
            
            if self.position == 1:
                self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
                new_sl_level = self.highest_price_since_entry - (multiplier_set[self.symbol][1] * atr)
                if self.sl_level is None or new_sl_level > self.sl_level:
                    self._update_sl(new_sl_level)
            elif self.position == -1:
                self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)
                new_sl_level = self.lowest_price_since_entry + (multiplier_set[self.symbol][1] * atr)
                if self.sl_level is None or new_sl_level < self.sl_level:
                    self._update_sl(new_sl_level)

        except Exception as e:
            self.logger.error(f"Error in manage_position: {e}", exc_info=True)


    def _cleaning_existing_order(self):

        # First try to cancel all open orders
        try:
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            for order in open_orders:
                try:
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=order['orderId']
                    )
                    self.logger.info(f"Cancelled order {order['orderId']} for {self.symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order['orderId']}: {e}")
                    # Don't return here, continue trying to cancel other orders
                    continue
            
            if open_orders:  # If we had any orders to cancel, wait a moment
                time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Error checking open orders: {e}")
            

    def enter_position(self, direction, entry_price, atr, concurrent_slot_count: int = 1) -> None:
        self._cleaning_existing_order()
        self.logger.info('existing orders are cleaned')
        global orders
        self.logger.info(f"ENTER_POSITION called for {self.symbol}: direction={direction}, price={entry_price:.4f}, atr={atr:.4f}")

        try:
            positions = self.client.futures_position_information()
            existing_position = next((p for p in positions if p['symbol'] == self.symbol and float(p['positionAmt']) != 0), None)

            if existing_position:
                self.logger.warning(f"Position for {self.symbol} already exists on the exchange. Syncing state and skipping new entry.")
                self.position = 1 if float(existing_position['positionAmt']) > 0 else -1
                self.entry_price = float(existing_position['entryPrice'])
                self.position_size = abs(float(existing_position['positionAmt']))
                
                with active_trades_lock:
                    if self not in active_trades:
                        active_trades.append(self)
                return
        except Exception as e:
            self.logger.error(f"Error checking for existing positions for {self.symbol}: {e}")
            return

        # Atomically check and reserve a trade slot
        with active_trades_lock:
            if self.position != 0:
                self.logger.warning(f"Attempted to enter position for {self.symbol}, but already in position {self.position}.")
                return
            
            if len(active_trades) >= MAX_CONCURRENT_TRADES:
                self.logger.info(f"Cannot enter trade for {self.symbol}, max concurrent trades of {MAX_CONCURRENT_TRADES} reached.")
                return
            
            # Reserve our spot
            active_trades.append(self)
            self.logger.info(f"Reserved trade slot for {self.symbol}. Active trades: {[t.symbol for t in active_trades]}")
        
        try:
            # --- Start of actual trade execution ---
            self.logger.info(f"Step 1: Starting position entry process")
            
            # Get available balance from futures account
            self.logger.info(f"Step 5: Getting futures account balance...")
            available_balance = self.client.futures_account_balance()
            usdt_balance = 0
            for balance in available_balance:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['availableBalance'])  # Use availableBalance instead of balance
                    break
            self.logger.info(f"Step 6: Account balance retrieved: {usdt_balance} USDT")
            
            if usdt_balance < 1:
                self.logger.warning(f"Insufficient balance ({usdt_balance} USDT) for trading. Skipping entry.")
                self._reset_position_state() # Release our reserved slot
                return

            # Set position state first, but entry price will be updated later
            self.position = direction
            
            self.logger.info(f"Step 11: Setting leverage...")
            try:
                choppy_value = self.df['choppy'].iloc[-1] if 'choppy' in self.df.columns else 50
                leverage_to_set = self.leverage * 2 if choppy_value < 38 else self.leverage
                self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage_to_set)
                self.logger.info(f"Set leverage to {leverage_to_set}x for {self.symbol}")
            except Exception as e:
                if "leverage not modified" not in str(e):
                    self.logger.warning(f"Could not set leverage for {self.symbol}: {e}")

            self.logger.info(f"Step 12: Setting margin type...")
            try:
                self.client.futures_change_margin_type(symbol=self.symbol, marginType='CROSSED')
                self.logger.debug(f"Set margin type to CROSSED for {self.symbol}")
            except Exception as e:
                if "No need to change margin type" not in str(e):
                    self.logger.debug(f"Could not set margin type for {self.symbol}: {e}")
            
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == self.symbol), None)
            
            if not symbol_info or symbol_info['status'] != 'TRADING':
                self.logger.error(f"Symbol {self.symbol} not found or not trading.")
                self._reset_position_state()
                return
                    
            self.logger.info(f"Step 14: Calculating position size...")
            # Determine trade capital based on allocation mode
            active_count_before = max(len(active_trades) - 1, 0)
            if ALLOCATION_MODE == 'fixed_per_slot':
                denominator = max(int(MAX_CONCURRENT_TRADES), 1)
            elif ALLOCATION_MODE == 'dynamic_remaining':
                # Divide by remaining slots given currently active trades BEFORE this entry
                denominator = max(int(MAX_CONCURRENT_TRADES) - active_count_before, 1)
            else:  # 'full_available'
                denominator = 1

            trade_capital = usdt_balance / denominator

            if leverage_to_set==1:
                margin_to_use = trade_capital * 0.95
            else:
                margin_to_use = trade_capital * 0.95
            self.logger.info(f"Capital per trade: {trade_capital:.2f} USDT. Margin to use: {margin_to_use:.2f} USDT")
            
            # Calculate the notional value of the position using the correct leverage
            notional_value = margin_to_use * leverage_to_set

            min_notional = 5.0
            if notional_value < min_notional:
                self.logger.warning(f"Position notional value (${notional_value:.2f}) is below minimum (${min_notional}). Skipping trade.")
                self._reset_position_state()
                return

            # Calculate position size based on signal price
            position_size = notional_value / entry_price
            
            # Get quantity precision from symbol info
            qty_precision = 0
            min_qty = 0.0
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size_str = f['stepSize']
                    min_qty = float(f['minQty'])
                    if Decimal(step_size_str) < 1:
                        qty_precision = abs(Decimal(step_size_str).as_tuple().exponent)
                    break
            
            # Round to the correct precision and ensure it meets minimum quantity
            position_size = max(round(position_size, qty_precision), min_qty)
            
            if position_size * entry_price < min_notional:
                self.logger.warning(f"Notional value below minimum after adjustments. Skipping.")
                self._reset_position_state()
                return

            if position_size <= 0:
                self.logger.error(f"Invalid position size after calculations: {position_size}")
                self._reset_position_state()
                return
            
            self.position_size = position_size
            # Place entry order
            side = Client.SIDE_BUY if direction == 1 else Client.SIDE_SELL
            self.logger.info(f"Placing market {side} order for {self.symbol}: quantity={position_size}")
            
            try:
                order_params = {
                    'symbol': self.symbol,
                    'side': side,
                    'type': Client.ORDER_TYPE_MARKET,
                    'quantity': self.position_size
                }
                entry_order = self.client.futures_create_order(**order_params)
                self.logger.info(f"SUCCESS: Entry order placed. Full response: {entry_order}")

                actual_entry_price = float(entry_order.get('avgPrice', 0.0))

                if actual_entry_price == 0.0:
                    time.sleep(0.5)  # Wait for fill
                    order_details = self.client.futures_get_order(symbol=self.symbol, orderId=entry_order['orderId'])
                    actual_entry_price = float(order_details.get('avgPrice', 0.0))

                if actual_entry_price == 0.0:
                    self.logger.error("Could not determine actual entry price. Closing position.")
                    self._reset_position_state()
                    return
                
                self.entry_price = actual_entry_price
                
                try:
                    # Short Telegram notification on successful entry
                    if 'telegram_reporter' in globals() and telegram_reporter:
                        side_text = 'BUY' if direction == 1 else 'SELL'
                        qty_text = f"{self.position_size}"
                        price_text = f"{self.entry_price:.4f}"
                        telegram_reporter.send(
                            f"<b>Entered</b> <code>{self.symbol}</code> {side_text} qty {qty_text} @ {price_text}"
                        )
                except Exception:
                    pass

            except Exception as e:
                self.logger.error(f"FAILED TO PLACE ENTRY ORDER. Error: {e}", exc_info=True)
                self._reset_position_state()
                return

            price_precision = 0
            for f in symbol_info['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    tick_size_str = f['tickSize']
                    if Decimal(tick_size_str) < 1:
                        price_precision = abs(Decimal(tick_size_str).as_tuple().exponent)
                    break
            
            # Calculate TP/SL with actual entry price
            if direction == 1:
                self.tp_level = round(self.entry_price + (atr * multiplier_set[self.symbol][0]), price_precision)
                self.sl_level = round(self.entry_price - (atr * multiplier_set[self.symbol][1]), price_precision)
            else: # direction == -1
                self.tp_level = round(self.entry_price - (atr * multiplier_set[self.symbol][0]), price_precision)
                self.sl_level = round(self.entry_price + (atr * multiplier_set[self.symbol][1]), price_precision)

            # Place SL and TP orders
            sl_side = Client.SIDE_SELL if direction == 1 else Client.SIDE_BUY
            tp_side = sl_side

            try:
                sl_params = {
                    'symbol': self.symbol, 'side': sl_side, 'type': 'STOP_MARKET',
                    'quantity': self.position_size, 'stopPrice': self.sl_level, 'reduceOnly': True
                }
                sl_order = self.client.futures_create_order(**sl_params)
                self.orderid = sl_order['orderId']
                self.logger.info(f"SUCCESS: Stop loss order placed. ID: {self.orderid}")
            except Exception as e:
                self.logger.error(f"FAILED TO PLACE SL ORDER. Error: {e}", exc_info=True)

            try:
                tp_params = {
                    'symbol': self.symbol, 'side': tp_side, 'type': 'TAKE_PROFIT_MARKET',
                    'quantity': self.position_size, 'stopPrice': self.tp_level, 'reduceOnly': True
                }
                tp_order = self.client.futures_create_order(**tp_params)
                self.tp_orderid = tp_order['orderId']
                self.logger.info(f"SUCCESS: Take profit order placed. ID: {self.tp_orderid}")
            except Exception as e:
                self.logger.error(f"FAILED TO PLACE TP ORDER. Error: {e}", exc_info=True)

            # Initialize tracking variables
            self.highest_price_since_entry = self.entry_price
            self.lowest_price_since_entry = self.entry_price
            self.processed_orders.clear()  # Clear processed orders for new position
            
        except Exception as e:
            self.logger.error(f"Error in position entry for {self.symbol}: {e}", exc_info=True)
            self._reset_position_state() # Ensure we release our slot on any error
            return
            
        # Update order tracking
        new_order_data = {
            'entry_time': datetime.now(UTC),
            'symbol': self.symbol,
            'side': 'BUY' if direction == 1 else 'SELL',
            'entry_price': self.entry_price,
            'tp_price': self.tp_level,
            'sl_price': self.sl_level,
            'choppy': self.df['choppy'].iloc[-1] if 'choppy' in self.df.columns else 0
        }

        with orders_lock:
            new_order_df = pd.DataFrame([new_order_data])
            orders = pd.concat([orders, new_order_df], ignore_index=True)
            self.order_id = orders.index[-1]
            new_order_data['order_id'] = self.order_id

        self.logger.info(f"New position entered | OrderID: {self.order_id}")

    
    def _reset_position_state(self):
        """Reset position state and clean up orders/positions"""
        self.logger.info(f"Resetting position state for {self.symbol}")
        
        # Clean existing orders
        self._cleaning_existing_order()
        
        # Get orders for the current symbol
        symbol_orders = self.client.futures_get_all_orders(symbol=self.symbol)
        
        # Check for TP/SL hit orders and add to tradereport
        self.logger.info(f"Checking {len(symbol_orders)} orders for {self.symbol} for TP/SL hits...")
        for order in symbol_orders:
            if order['status'] == 'FILLED':
                if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'STOP_LOSS', 'TAKE_PROFIT']:
                    self.logger.info(f"Found TP/SL hit: {order['type']} order {order['orderId']} for {self.symbol}")
                    # Create trade report entry for TP/SL hit
                    exit_price = float(order['avgPrice']) if order['avgPrice'] else float(order['price'])
                    
                    # Determine original position side based on current position state
                    # Use the actual position side from the trader's state
                    if self.position == 1:
                        original_position_side = 'BUY'  # Long position
                    elif self.position == -1:
                        original_position_side = 'SELL'  # Short position
                    else:
                        # Fallback: try to determine from order side (TP/SL orders are opposite to position)
                        original_position_side = 'SELL' if order['side'] == 'BUY' else 'BUY'
                    
                    # Calculate profit/loss
                    if self.position_size > 0:
                        if original_position_side == 'BUY':  # Long position closed
                            profit = (exit_price - self.entry_price) * self.position_size
                        else:  # Short position closed
                            profit = (self.entry_price - exit_price) * self.position_size
                        self.logger.info(f"PnL calculation (reset): {original_position_side} position, entry: {self.entry_price}, exit: {exit_price}, size: {self.position_size}, profit: {profit}")
                    else:
                        # Fallback: try to get position size from the order
                        order_qty = float(order['executedQty']) if 'executedQty' in order else 0
                        if order_qty > 0:
                            if original_position_side == 'BUY':  # Long position closed
                                profit = (exit_price - self.entry_price) * order_qty
                            else:  # Short position closed
                                profit = (self.entry_price - exit_price) * order_qty
                            self.logger.info(f"PnL calculation (reset fallback): {original_position_side} position, entry: {self.entry_price}, exit: {exit_price}, size: {order_qty}, profit: {profit}")
                        else:
                            profit = 0.0  # Cannot calculate profit without position size
                            self.logger.warning(f"Cannot calculate PnL (reset): no position size available")
                    
                    trade_data = {
                        'symbol': self.symbol,
                        'side': original_position_side,  # Original position side
                        'entry_price': self.entry_price,
                        'tp_price': self.tp_level,
                        'sl_price': self.sl_level,
                        'exit_price': exit_price,
                        'choppy': 'N/A',  # Not applicable for TP/SL hits
                        'profit': round(profit, 4)
                    }
                    
                    # Add to tradereport
                    global tradereport
                    with orders_lock:
                        tradereport = pd.concat([tradereport, pd.DataFrame([trade_data])], ignore_index=True)
                    self.logger.info(f"TP/SL hit recorded for {self.symbol}: {order['type']} at {trade_data['exit_price']}, profit: {trade_data['profit']}")
                    self.logger.info(f"Trade data added: {trade_data}")
                    
                    # Send Telegram PnL notification
                    try:
                        if 'telegram_reporter' in globals() and telegram_reporter:
                            side_text = trade_data['side']
                            exit_price_text = f"{trade_data['exit_price']:.4f}"
                            pnl_text = f"{trade_data['profit']:.4f} USDT"
                            telegram_reporter.send(
                                f"<b>Closed</b> <code>{self.symbol}</code> {side_text} @ {exit_price_text}\n"
                                f"PnL: {pnl_text}"
                            )
                    except Exception as e:
                        self.logger.error(f"Failed to send Telegram PnL notification: {e}")
                    
                    # Immediately display updated completed trades and persist
                    try:
                        print_trading_status()
                        self.logger.info("Trading status printed successfully")
                    except Exception as e:
                        self.logger.error(f"Could not print trading status: {e}")
                    
                    
        # Close any open position and log completion
        try:
            position = self.client.futures_position_information(symbol=self.symbol)
            if position and float(position[0]['positionAmt']) != 0:
                self.client.futures_create_order(
                    symbol=self.symbol,
                    type='MARKET',
                    side='SELL' if float(position[0]['positionAmt']) > 0 else 'BUY',
                    quantity=abs(float(position[0]['positionAmt'])),
                    reduceOnly=True
                )
                self.logger.info(f"Closed open position for {self.symbol}")
                try:
                    print_trading_status()
                except Exception as e:
                    self.logger.debug(f"Could not print trading status after closure: {e}")
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

        # Reset state variables
        self.position = 0
        self.entry_price = 0.0
        self.tp_level = None
        self.sl_level = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.position_size = 0.0
        self.orderid = None
        self.tp_orderid = None
        self.processed_orders.clear()

        # Remove from active trades
        with active_trades_lock:
            if self in active_trades:
                active_trades.remove(self)
                self.logger.info(f"Removed {self.symbol} from active trades. Active trades: {len(active_trades)}")
        
        trade_in_progress.clear()
    def run(self) -> None:
        self.logger.info(f"Starting trader for {self.symbol}") 
        if not self._fetch_initial_data():
            self.logger.error("Failed to fetch initial data. Stopping.")
            return

        if not self._calculate_indicators():
            self.logger.warning("Initial indicator calculation failed. Will retry on next candle.")
        else:
            self.manage_position()

        # Print status after first run for last asset
        if self.symbol == ASSETS[-1]:
            print_trading_status()

        loop_count = 0
        while True:
            try:
                loop_count += 1
                server_now = self._get_server_time()  # This is now timezone-aware
                
                # Validate DataFrame state
                if self.df.empty or len(self.df) < 2:
                    self.logger.error("DataFrame is empty or has insufficient data. Attempting to refetch initial data.")
                    if not self._fetch_initial_data():
                        self.logger.error("Failed to refetch initial data. Sleeping for 30 seconds.")
                        time.sleep(30)
                        continue

                # Use the last candle's timestamp (self.df.index[-1]) instead of [-2]
                last_candle_time_utc = self.df.index[-1]  # Already timezone-aware from _fetch_initial_data
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
                    self.logger.debug(f"Current Position: {self.position}")
                    self.logger.debug(f"---------------------------------")

                new_candle_fetched = self._fetch_latest_candle()
                if new_candle_fetched:
                    indicators_ok = self._calculate_indicators()
                    if indicators_ok:
                        self.manage_position()
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
        
        tradereport.to_csv('trader_report.csv', index=False)

    def _submit_signal(self, direction, price, atr, choppy):
        # Check if trading is inhibited due to losses
        if not check_last_two_trades_and_manage_inhibition():
            self.logger.info(f"Signal submission blocked for {self.symbol} due to trading inhibition (last 2 trades were losses)")
            return
        
        # News inhibition removed
        
        signal = {
            'symbol': self.symbol,
            'direction': direction,
            'price': price,
            'atr': atr,
            'choppy': choppy,
            'trader': self,
            'tp_to_sl_ratio': multiplier_set[self.symbol][0] / multiplier_set[self.symbol][1]
        }
        with signal_pool_lock:
            signal_pool.append(signal)
            logging.info(f"Signal added to pool: {self.symbol} {direction} at {price:.4f}, pool size: {len(signal_pool)}")

    @staticmethod
    def pick_and_execute_best_trade():
        # Check if trading is inhibited due to losses
        if not check_last_two_trades_and_manage_inhibition():
            logging.info("Trade execution blocked due to trading inhibition (last 2 trades were losses)")
            return
        
        # News inhibition removed
        
        signals_to_execute = []
        with signal_pool_lock:
            if not signal_pool:
                return

            available_slots = 0
            with active_trades_lock:
                # Actively sync active_trades with exchange to avoid stale entries after SL/TP
                still_active = []
                for t in list(active_trades):
                    try:
                        pos = t.client.futures_position_information(symbol=t.symbol)
                        if pos and float(pos[0]['positionAmt']) != 0:
                            still_active.append(t)
                        else:
                            t.position = 0
                            logging.info(f"Detected closed position for {t.symbol}. Removing from active trades.")
                    except Exception as e:
                        # On error, keep previous state to avoid accidental over-allocation
                        logging.warning(f"Could not sync position for {t.symbol}: {e}")
                        if t.position != 0:
                            still_active.append(t)
                active_trades[:] = still_active

                available_slots = MAX_CONCURRENT_TRADES - len(active_trades)
                logging.info(f"Signal pool size: {len(signal_pool)}, Available slots: {available_slots}, Active trades: {[t.symbol for t in active_trades]}")
            
            if available_slots <= 0:
                if signal_pool:
                    logging.info("No available trade slots.")
                    signal_pool.clear()
                return

            sorted_signals = sorted(signal_pool, key=lambda s: (s['choppy'], -s['atr'], -s['tp_to_sl_ratio']))
            signals_to_execute = sorted_signals[:available_slots]
            logging.info(f"Selected {len(signals_to_execute)} signals: {[(s['symbol'], s['direction']) for s in signals_to_execute]}")
            signal_pool.clear()

    # Execute trades without holding any locks from this method
        for signal in signals_to_execute:
            trader = signal['trader']
            logging.info(f"Attempting to execute trade for {trader.symbol}...")
            try:
                if signal['direction'] == 'buy':
                    trader.enter_position(1, signal['price'], signal['atr'], concurrent_slot_count=len(signals_to_execute))
                elif signal['direction'] == 'sell':
                    trader.enter_position(-1, signal['price'], signal['atr'], concurrent_slot_count=len(signals_to_execute))
            except Exception as e:
                logging.error(f"Error while calling enter_position for {trader.symbol}: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        if telegram_reporter:
            telegram_reporter.send("<b>Trading bot starting</b> — environment configured and logger attached.")
        # Initialize Binance client
        client = create_binance_client_with_failover(API_KEY, API_SECRET)
        
        # Test connection
        try:
            client.futures_ping()
            logging.info("Successfully connected to Binance Futures API")
        except Exception as e:
            logging.critical(f"Failed to connect to Binance API: {e}")
            sys.exit(1)
        
        # Check account balance
        try:
            account_balance = client.futures_account_balance()
            usdt_balance = 0
            for balance in account_balance:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['balance'])
                    break
            
            effective_buying_power = usdt_balance * LEVERAGE
            logging.info(f"Current USDT futures balance: ${usdt_balance:.2f}")
            logging.info(f"Effective buying power with {LEVERAGE}x leverage: ${effective_buying_power:.2f}")
            if telegram_reporter:
                telegram_reporter.send(f"<b>Available balance</b>: ${usdt_balance:.2f} USDT")
            
            min_margin_for_trade = 5.0 / LEVERAGE  # Minimum margin needed for $5 trade
            if usdt_balance < min_margin_for_trade:
                logging.critical(f"Insufficient balance: ${usdt_balance:.2f}. Minimum required for $5 trade with {LEVERAGE}x leverage: ${min_margin_for_trade:.2f}")
                sys.exit(1)
            elif usdt_balance < 2.0:  # Minimum recommended for multiple assets
                logging.warning(f"Low balance warning: ${usdt_balance:.2f} may be insufficient for trading multiple assets simultaneously")
                
        except Exception as e:
            logging.error(f"Error checking account balance: {e}")
            # Continue anyway, individual trades will check balance
        
        # Get trending assets
        try:
            
            logging.info(f"Found none trending assets")
        except Exception as e:
            logging.error(f"Error getting trending assets: {e}")
            trending_assets = []
        
        # Validate the specified assets list
        
        
        # Combine validated and trending assets
        all_assets = ASSETS  # Using set to remove any duplicates
        logging.info(f"Total assets to trade: {len(all_assets)}")
        
        # Setup leverage for all symbols
        """try:
            successful_symbols = setup_leverage_for_symbols(client, all_assets, LEVERAGE)
            if not successful_symbols:
                logging.error("Failed to setup leverage for any symbols. Exiting.")
                sys.exit(1)
            # Only trade symbols where leverage was successfully set
            all_assets = successful_symbols
            logging.info(f"Ready to trade {len(all_assets)} symbols with {LEVERAGE}x leverage")
        except Exception as e:
            logging.error(f"Error setting up leverage: {e}")
            # Continue anyway, individual traders will try to set leverage
        """
        # Initialize and start trading threads
        threads = []
        traders = []
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
                    
                )
                traders.append(trader)
                thread = threading.Thread(target=trader.run)
                threads.append(thread)
                thread.start()
                logging.info(f"Started trading thread for {asset}")
            except Exception as e:
                logging.critical(f"Failed to initialize trader for {asset}: {e}", exc_info=True)

        # Add a thread to periodically pick and execute the best trade
        def trade_picker_loop():
            while True:
                ForwardIchimokuTrader.pick_and_execute_best_trade()
                time.sleep(5)  # Check every 5 seconds

        picker_thread = threading.Thread(target=trade_picker_loop, daemon=True)
        picker_thread.start()
        
        # Add a thread to periodically process Gemini analysis queue
        def gemini_queue_processor_loop():
            while True:
                try:
                    process_gemini_queue()
                except Exception as e:
                    logging.error(f"Error in Gemini queue processor: {e}", exc_info=True)
                time.sleep(5)  # Check every 5 seconds
        
        gemini_processor_thread = threading.Thread(target=gemini_queue_processor_loop, daemon=True)
        gemini_processor_thread.start()
        
        # News checker removed

        # Wait for all threads to finish
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt. Shutting down gracefully...")
            # Add any cleanup code here if needed
        except Exception as e:
            logging.error(f"Error in main thread: {e}", exc_info=True)
        finally:
            # Save final trade report
            try:
                tradereport.to_csv('trader_report_live.csv', index=False)
                logging.info("Trade report saved successfully")
            except Exception as e:
                logging.error(f"Failed to save trade report: {e}")
            
            logging.info("Trading bot shutdown complete")
            if telegram_reporter:
                telegram_reporter.send("<b>Trading bot shutdown complete</b>")
                telegram_reporter.stop(drain=True)
            
    except Exception as e:
        logging.critical(f"Fatal error in main execution: {e}", exc_info=True)
        sys.exit(1)    
