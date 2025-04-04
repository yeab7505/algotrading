import pandas as pd
import pandas_ta as ta
import numpy as np
from binance.client import Client
import time
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import sys

# --- Configuration ---s
API_KEY = "iOgcObLOw4UIFSvvEPXLFP1vgwp1wzyHYfw57vd1vrg19Xt6SXCE4RywDi5QoM28"
API_SECRET = "bz1m4UlthzklqXlWoqAqXZiJE35jjT0g5uJ5cQ43vwDNnsIpPYS5OqevfBVz84iK"

# --- Strategy & Trading Parameters ---
SYMBOL = 'ETHUSDT'  # Change to your desired symbol
INTERVAL = Client.KLINE_INTERVAL_1MINUTE # e.g., 1m, 5m, 15m, 1h, 4h, 1d
LOOKBACK_PERIODS = 100 # Number of initial candles (ensure >= longest SMA/ADX/Supertrend period)

# Strategy Parameters from your backtest (adjust as needed)
SMA_SHORT = 19
SMA_LONG = 32
ADX_THRESHOLD = 16
ADX_PERIOD = 20 # Default ADX period used in your backtest
SUPERTREND_PERIOD = 14 # Default Supertrend period
SUPERTREND_MULTIPLIER = 3.0 # Default Supertrend multiplier

# Position Sizing & Risk Management
STOP_MULTIPLIER = 1.75 # ATR multiplier for trailing stop loss
TP_MULTIPLIER = 2.0    # ATR multiplier for initial take profit (trailing logic uses SL mult)
LEVERAGE = 1           # Keep leverage = 1 for paper trading spot simulation
TC = 0.0005            # Simulated transaction cost (e.g., 0.05%)
SIMULATED_INITIAL_BALANCE = 1000 # Starting virtual capital
ATR_PERIOD = 14        # Default ATR period

# --- Logging Setup ---
# Set to DEBUG for detailed info, INFO for less noise
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
# Reduce verbosity of libraries if needed
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

class ForwardSmaAdxSuperTrader:
    def __init__(self, symbol: str, interval: str, lookback: int,
                 sma_short: int, sma_long: int, adx_threshold: float, adx_period: int,
                 st_period: int, st_multiplier: float, atr_period: int,
                 tp_multiplier: float, sl_multiplier: float,
                 leverage: int = 1, tc: float = 0.0005, initial_balance: float = 1000) -> None:
        """Initialize the SMA/ADX/Supertrend forward tester."""
        self.symbol = symbol
        self.interval = interval

        # Strategy Params
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.adx_threshold = adx_threshold
        self.adx_period = adx_period
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.atr_period = atr_period

        # Risk Params
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier # Renamed from stop_multiplier for clarity
        self.leverage = leverage
        self.tc = tc

        # Account State
        self.initial_balance = initial_balance
        self.simulated_balance = initial_balance

        # Ensure lookback is sufficient for all indicators
        self.required_lookback = max(lookback, sma_long + 1, adx_period * 2, st_period + 1, atr_period + 1) # Rough estimate, ADX needs more
        self.lookback = self.required_lookback
        if lookback < self.required_lookback:
             logging.warning(f"Initial lookback {lookback} increased to {self.required_lookback} for indicator calculations.")

        # Use a specific logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trader...")
        self.logger.info(f"Params: SMA={sma_short}/{sma_long}, ADX > {adx_threshold} (P:{adx_period}), "
                         f"SuperT={st_period}/{st_multiplier}, ATR={atr_period}, "
                         f"SL={sl_multiplier}*ATR, TP={tp_multiplier}*ATR")


        try:
            self.client = Client(API_KEY, API_SECRET)
            self.client.ping() # Test connection
            self.logger.info("Binance client initialized and connection tested.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}", exc_info=True)
            raise

        self.df = pd.DataFrame()

        # Position State Attributes
        self.position = 0  # -1: Short, 0: Flat, 1: Long
        self.entry_price = 0.0
        self.tp_level = None # Initial TP based on entry ATR
        self.sl_level = None # Trailing SL based on ATR
        self.highest_price_since_entry = None # For trailing SL (long)
        self.lowest_price_since_entry = None # For trailing SL (short)
        self.last_trade_time = None # To potentially avoid rapid signals

        self.ms_interval = interval_to_milliseconds(self.interval)
        if not self.ms_interval:
            raise ValueError("Invalid interval for millisecond conversion")
        self.logger.info(f"Interval: {self.interval}, Milliseconds: {self.ms_interval}")

    # --- Data Fetching Methods (_fetch_initial_data, _fetch_latest_candle) ---
    #     These can be reused directly from the ForwardIchimokuTrader class
    #     as they are generic data handling functions.
    #     (Copy them here for completeness, ensuring self.logger is used)

    def _fetch_initial_data(self) -> bool:
        self.logger.info(f"Fetching initial {self.lookback + 50} klines for {self.symbol}...")
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=self.lookback + 50)
            if not klines:
                self.logger.error("Could not fetch initial klines (received empty list).")
                return False

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                    'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']
            data = pd.DataFrame(klines, columns=cols)
            if data.empty:
                 self.logger.error("Initial klines resulted in empty DataFrame.")
                 return False

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                 self.logger.warning("NaN values found in OHLC data after numeric conversion during initial fetch.")
                 initial_len = len(data)
                 data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                 self.logger.warning(f"Dropped {initial_len - len(data)} rows with NaN OHLC.")

            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms')
            data.set_index('Datetime', inplace=True)
            self.df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            if len(self.df) < self.required_lookback:
                 self.logger.error(f"Insufficient valid initial data fetched ({len(self.df)} rows) after processing. Need at least {self.required_lookback}.")
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
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=2)
            if not klines or len(klines) < 2:
                self.logger.warning("Could not fetch latest klines or not enough data yet (< 2).")
                return False

            latest_kline = klines[-2]
            close_time_ms = latest_kline[6]
            latest_dt = pd.to_datetime(close_time_ms, unit='ms')

            if self.df.empty:
                 self.logger.warning("Main DataFrame is empty, cannot compare timestamps.")
                 return False
            if latest_dt <= self.df.index[-1]:
                 return False # No new candle

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

            if not isinstance(self.df.index, pd.DatetimeIndex):
                 self.logger.error("Main DataFrame index is not DatetimeIndex. Cannot append.")
                 return False

            self.df = pd.concat([self.df, new_row.to_frame().T])
            if not self.df.index.is_monotonic_increasing:
                 self.logger.warning(f"Index became non-monotonic after adding candle {latest_dt}. Sorting...")
                 self.df.sort_index(inplace=True)
                 if not self.df.index.is_monotonic_increasing:
                      self.logger.error("Index still non-monotonic after sorting. Halting.")
                      raise ValueError("DataFrame index could not be kept monotonic.")

            max_len = self.lookback + 200
            if len(self.df) > max_len:
                 self.logger.debug(f"Trimming DataFrame from {len(self.df)} to {max_len} rows.")
                 self.df = self.df.iloc[-max_len:]

            self.logger.info(f"New candle appended: {self.df.index[-1]}, Close: {self.df['Close'].iloc[-1]:.4f}, DF rows: {len(self.df)}")
            return True
        except Exception as e:
            self.logger.error(f"Error fetching/appending latest candle: {e}", exc_info=True)
            return False

    # --- Indicator Calculation ---
    def _calculate_indicators(self) -> bool:
        """Calculates SMA, ADX, Supertrend, and ATR."""
        self.logger.debug(f"Calculating indicators for {len(self.df)} rows...")
        if len(self.df) < self.required_lookback:
            self.logger.warning(f"Not enough data ({len(self.df)}, need {self.required_lookback}) for all indicator calculations.")
            return False
        try:
            # --- Pre-calculation Check ---
            if self.df[['High', 'Low', 'Close']].isnull().values.any():
                 self.logger.warning("NaN values found in High, Low, or Close columns BEFORE indicator calculation.")
                 # Potentially drop last row if it has NaN OHLC? Or just warn.
                 # self.df.drop(self.df.tail(1).index, inplace=True) # Risky if source data has gaps

            # Store calculations temporarily
            indicators = pd.DataFrame(index=self.df.index)

            # 1. SMAs
            indicators[f'sma_{self.sma_short}'] = ta.sma(self.df['Close'], length=self.sma_short)
            indicators[f'sma_{self.sma_long}'] = ta.sma(self.df['Close'], length=self.sma_long)

            # 2. ATR (for signals and TP/SL)
            indicators['atr'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=self.atr_period)

            # 3. ADX
            adx_data = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=self.adx_period)
            if adx_data is not None and not adx_data.empty:
                adx_col_name = f'ADX_{self.adx_period}'
                if adx_col_name in adx_data.columns:
                     indicators['adx'] = adx_data[adx_col_name]
                else:
                     self.logger.warning(f"Column {adx_col_name} not found in ADX result. Columns: {adx_data.columns}")
                     indicators['adx'] = np.nan
            else:
                self.logger.warning("ADX calculation returned None or empty DataFrame.")
                indicators['adx'] = np.nan


            # 4. Supertrend
            st_data = ta.supertrend(
                high=self.df['High'], low=self.df['Low'], close=self.df['Close'],
                length=self.st_period, multiplier=self.st_multiplier
            )
            # Example column names: SUPERT_14_3.0, SUPERTd_14_3.0, SUPERTl_14_3.0, SUPERTs_14_3.0
            st_dir_col = f"SUPERTd_{self.st_period}_{self.st_multiplier:.1f}" # pandas-ta uses 1 decimal for multiplier in name
            if st_data is not None and st_dir_col in st_data.columns:
                indicators['st_dir'] = st_data[st_dir_col] # Get the direction column (-1 or 1)
            else:
                self.logger.warning(f"Supertrend direction column '{st_dir_col}' not found in calculation result. Columns: {st_data.columns if st_data is not None else 'None'}")
                indicators['st_dir'] = np.nan


            # --- Update Main DataFrame ---
            # Define columns to potentially drop/update
            indicator_cols = list(indicators.columns)
            cols_present_in_df = [col for col in indicator_cols if col in self.df.columns]
            if cols_present_in_df:
                self.logger.debug(f"Dropping existing indicator columns: {cols_present_in_df}")
                self.df = self.df.drop(columns=cols_present_in_df)

            # Join the newly calculated indicators
            self.df = self.df.join(indicators)
            self.logger.debug("Indicators calculated and joined to DataFrame.")

            # --- Post-calculation Check (Last Row) ---
            last_row_indicators = self.df.iloc[-1][indicator_cols]
            if last_row_indicators.isnull().any():
                 self.logger.warning(f"NaN values found in indicators on the latest row ({self.df.index[-1]}):")
                 self.logger.warning(f"{last_row_indicators[last_row_indicators.isnull()]}")
                 # Decide if this should return False
                 # return False

            return True

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return False

    # --- Signal Checking ---
    def _check_signals(self) -> tuple[bool, bool]:
        """Checks for combined buy/sell entry signals."""
        self.logger.debug("Checking signals...")
        # Define required columns for signal generation based on calculation step
        sma_short_col = f'sma_{self.sma_short}'
        sma_long_col = f'sma_{self.sma_long}'
        required_cols = [sma_short_col, sma_long_col, 'adx', 'st_dir']

        # Need at least 2 rows to check for Supertrend flip
        if len(self.df) < 2:
             self.logger.debug("Need at least 2 rows for signal check (Supertrend flip).")
             return False, False

        # Check if all required columns exist
        if not all(col in self.df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for signal check. Needed: {required_cols}, Have: {self.df.columns}")
            return False, False

        try:
            # Get the last two rows for checking conditions and flips
            last = self.df.iloc[-1]
            prev = self.df.iloc[-2]
            last_index_name = last.name
        except IndexError:
             self.logger.error("Cannot access last 2 rows for signal check.")
             return False, False

        # Check for NaNs in the latest row's required indicators
        if last[required_cols].isnull().any():
             self.logger.warning(f"Signal check skipped: NaN value found in required indicators for latest candle {last_index_name}.")
             self.logger.debug(f"NaN details: {last[required_cols].isnull()}")
             return False, False
        # Also check previous Supertrend direction is not NaN
        if pd.isna(prev['st_dir']):
             self.logger.warning(f"Signal check skipped: Previous Supertrend direction is NaN at {prev.name}.")
             return False, False


        # --- Define Signal Conditions ---
        # 1. SMA Crossover
        sma_cross_bull = last[sma_short_col] > last[sma_long_col]
        sma_cross_bear = last[sma_short_col] < last[sma_long_col]

        # 2. ADX Filter
        adx_strong = last['adx'] > self.adx_threshold

        # 3. Supertrend Confirmation (using the flip condition from backtest)
        st_buy_flip = last['st_dir'] > 0 and prev['st_dir'] < 0
        st_sell_flip = last['st_dir'] < 0 and prev['st_dir'] > 0

        # Combine conditions for final signals
        buy_signal = sma_cross_bull and adx_strong and st_buy_flip
        sell_signal = sma_cross_bear and adx_strong and st_sell_flip

        self.logger.debug(f"Signal Conditions @ {last_index_name}: "
                          f"SMA_Cross(B/S)=({sma_cross_bull}/{sma_cross_bear}), "
                          f"ADX_OK={adx_strong}({last['adx']:.2f}>{self.adx_threshold}), "
                          f"ST_Flip(B/S)=({st_buy_flip}/{st_sell_flip}) (Dir:{last['st_dir']:.0f}<-{prev['st_dir']:.0f})")
        self.logger.debug(f"Result: Buy Signal={buy_signal}, Sell Signal={sell_signal}")


        # Optional: Anti-whipsaw (debounce) check
        # if self.last_trade_time and (datetime.utcnow() - self.last_trade_time) < timedelta(minutes=INTERVAL_MINUTES * 2): # Example: wait 2 intervals
        #     if buy_signal or sell_signal:
        #        self.logger.debug("Signal suppressed due to recent trade.")
        #        buy_signal = False
        #        sell_signal = False

        return buy_signal, sell_signal

    # --- Position Management Methods ---
    #     _manage_position, _enter_position, _close_position, _reset_position_state
    #     can be reused from the previous example (ForwardIchimokuTrader),
    #     as the TP/SL logic (ATR trailing stop) is the same based on your backtest.
    #     Ensure they use self.logger and self.sl_multiplier.
    #     (Copy them here, ensure logger and sl_multiplier are used)

    def _manage_position(self) -> None:
        """Manages exits (TP/SL) and entries based on signals."""
        if self.df.empty or 'atr' not in self.df.columns:
            self.logger.warning("DataFrame not ready for position management (empty or no ATR).")
            return

        try:
            last_row = self.df.iloc[-1]
            current_price = last_row['Close']
            current_price_high=last_row['High']
            current_price_low=last_row['Low']
            atr = last_row['atr']
        except IndexError:
             self.logger.error("Cannot access last row in _manage_position.")
             return
        except KeyError as e:
             self.logger.error(f"Missing column in _manage_position: {e}")
             return

        if pd.isna(current_price):
             self.logger.warning(f"Current price is NaN ({last_row.name}). Skipping position management.")
             return
        if pd.isna(atr) or atr <= 0: # Added check for non-positive ATR
            self.logger.warning(f"Invalid ATR ({atr:.4f}) at {last_row.name}. Skipping TP/SL update and potential entry.")
            # If in position, should we still check fixed TP/SL? For now, skip if ATR bad.
            return

        # --- Check Exits First ---
        position_closed_this_step = False
        if self.position == 1: # Long position active
            if self.tp_level is None or self.sl_level is None or self.highest_price_since_entry is None:
                 self.logger.error(f"Inconsistent state for long position @ {last_row.name}. TP/SL/HighestPrice not set. Resetting.")
                 self._reset_position_state()
                 return

            self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
            # Recalculate trailing SL based on current ATR and highest price
            trailing_sl = self.highest_price_since_entry - (self.sl_multiplier * atr)
            # SL only moves up
            self.sl_level = max(self.sl_level, trailing_sl)
            self.logger.debug(f"Long active: Price={current_price:.4f}, TP={self.tp_level:.4f}, TrailSL={self.sl_level:.4f} (based on High={self.highest_price_since_entry:.4f}, ATR={atr:.4f})")


            exit_reason = None
            # Check TP first (as per backtest logic)
            if current_price >= self.tp_level:
                exit_reason = f"Take Profit hit at {self.tp_level:.4f}"
            elif current_price <= self.sl_level:
                exit_reason = f"Trailing Stop Loss hit at {self.sl_level:.4f}"

            if exit_reason:
                self._close_position(current_price, exit_reason)
                position_closed_this_step = True # Mark position as closed

        elif self.position == -1: # Short position active
            if self.tp_level is None or self.sl_level is None or self.lowest_price_since_entry is None:
                 self.logger.error(f"Inconsistent state for short position @ {last_row.name}. TP/SL/LowestPrice not set. Resetting.")
                 self._reset_position_state()
                 return

            self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)
             # Recalculate trailing SL based on current ATR and lowest price
            trailing_sl = self.lowest_price_since_entry + (self.sl_multiplier * atr)
            # SL only moves down
            self.sl_level = min(self.sl_level, trailing_sl)
            self.logger.debug(f"Short active: Price={current_price:.4f}, TP={self.tp_level:.4f}, TrailSL={self.sl_level:.4f} (based on Low={self.lowest_price_since_entry:.4f}, ATR={atr:.4f})")

            exit_reason = None
             # Check TP first (as per backtest logic)
            if current_price_high <= self.tp_level:
                exit_reason = f"Take Profit hit at {self.tp_level:.4f}"
            elif current_price_low >= self.sl_level:
                exit_reason = f"Trailing Stop Loss hit at {self.sl_level:.4f}"

            if exit_reason:
                self._close_position(current_price, exit_reason)
                position_closed_this_step = True # Mark position as closed

        # --- Check Entries (Only if flat *and* position wasn't closed just now) ---
        if self.position == 0 and not position_closed_this_step:
             # Ensure ATR is valid before considering entry
             if pd.isna(atr) or atr <= 0:
                 self.logger.debug(f"Skipping entry check due to invalid ATR ({atr:.4f}) at {last_row.name}")
                 return

             buy_signal, sell_signal = self._check_signals()

             if buy_signal:
                 self._enter_position(1, current_price, atr)
             elif sell_signal:
                 self._enter_position(-1, current_price, atr)

    def _enter_position(self, direction: int, entry_price: float, atr: float) -> None:
        """Simulates entering a position."""
        if pd.isna(atr) or atr <= 0:
             self.logger.error(f"Attempted to enter position with invalid ATR: {atr:.4f}. Aborting entry.")
             return
        if self.position != 0:
             self.logger.warning(f"Attempted to enter {direction} while already in position {self.position}. Ignoring.")
             return


        self.position = direction
        self.entry_price = entry_price
        entry_cost = self.simulated_balance * self.tc
        self.simulated_balance -= entry_cost # Simulate entry cost

        log_msg_base = f"PAPER TRADE: Entered {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.4f}"

        try:
            if direction == 1: # Long
                # Initial TP/SL calculation based on entry ATR
                self.tp_level = entry_price + (atr * self.tp_multiplier)
                self.sl_level = entry_price - (atr * self.sl_multiplier) # Initial SL uses sl_multiplier
                self.highest_price_since_entry = entry_price
                self.lowest_price_since_entry = None # Reset for long
                log_msg = f"{log_msg_base} | Init TP: {self.tp_level:.4f}, Init SL: {self.sl_level:.4f} (ATR={atr:.4f}) | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
            elif direction == -1: # Short
                self.tp_level = entry_price - (atr * self.tp_multiplier)
                self.sl_level = entry_price + (atr * self.sl_multiplier) # Initial SL uses sl_multiplier
                self.lowest_price_since_entry = entry_price
                self.highest_price_since_entry = None # Reset for short
                log_msg = f"{log_msg_base} | Init TP: {self.tp_level:.4f}, Init SL: {self.sl_level:.4f} (ATR={atr:.4f}) | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
            else: # Should not happen
                 self.logger.error(f"Invalid direction {direction} passed to _enter_position.")
                 self._reset_position_state()
                 return

            self.logger.info(log_msg)

        except Exception as e:
             self.logger.error(f"Error calculating TP/SL during entry: {e}. Position may be invalid.", exc_info=True)
             self._reset_position_state()


    def _close_position(self, exit_price: float, reason: str) -> None:
        """Simulates closing the current position and updates balance."""
        if self.position == 0:
             self.logger.warning("Attempted to close position while already flat.")
             return
        if self.entry_price == 0: # Safety check
             self.logger.error("Cannot calculate PnL: Entry price is zero. Resetting state.")
             self._reset_position_state()
             return

        pnl_percentage = 0
        if self.position == 1: # Closing Long
            pnl_percentage = (exit_price / self.entry_price - 1)
        elif self.position == -1: # Closing Short
            pnl_percentage = (self.entry_price / exit_price - 1) if exit_price != 0 else -1 # Avoid div by zero, assume 100% loss

        # Apply leverage effect and transaction cost for exit
        effective_pnl = pnl_percentage * self.leverage
        # Calculate exit cost based on value *after* PnL is applied, then apply TC
        value_after_pnl = self.simulated_balance * (1 + effective_pnl)
        exit_cost = value_after_pnl * self.tc
        final_pnl_factor = (1 + effective_pnl) * (1 - self.tc) # Combined factor for balance update


        # Store position before resetting
        closed_direction = 'LONG' if self.position == 1 else 'SHORT'
        balance_before_close = self.simulated_balance

        # Update balance
        self.simulated_balance *= final_pnl_factor # Apply PnL and exit cost


        self.logger.info(f"PAPER TRADE: Closed {closed_direction} @ {exit_price:.4f} | Entry: {self.entry_price:.4f} | Reason: {reason}")
        self.logger.info(f"PnL: {effective_pnl:.4%} (before exit cost) | Exit Cost: {exit_cost:.4f} | Balance Change: {self.simulated_balance - balance_before_close:.4f} | New Balance: {self.simulated_balance:.2f}")

        # Reset position state AFTER logging
        self._reset_position_state()
        self.last_trade_time = datetime.utcnow() # Use UTC


    def _reset_position_state(self):
        """Helper function to safely reset all position-related attributes."""
        self.logger.debug("Resetting position state.")
        self.position = 0
        self.entry_price = 0.0
        self.tp_level = None
        self.sl_level = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        # Do not reset last_trade_time here, it's set on actual close


    # --- Main Execution Loop ---
    def run(self) -> None:
        """Main execution loop for the forward tester."""
        self.logger.info(f"Starting forward test for {self.symbol} on {self.interval} interval.")
        self.logger.info(f"Initial Balance: ${self.initial_balance:.2f}")

        if not self._fetch_initial_data():
            self.logger.error("Failed to fetch initial data. Stopping.")
            return

        if not self._calculate_indicators():
             self.logger.warning("Initial indicator calculation failed. Will retry on next candle.")

        loop_count = 0
        while True:
            try:
                #
                loop_count += 1
                now = datetime.utcnow()

                # --- Calculate Wait Time ---
                if self.df.empty or not isinstance(self.df.index, pd.DatetimeIndex):
                     self.logger.error("DataFrame missing or index invalid. Cannot calculate wait time. Stopping.")
                     break
                last_candle_time_utc = self.df.index[-1].tz_localize(None) # Ensure timezone naive
                next_candle_time_utc = last_candle_time_utc + pd.Timedelta(milliseconds=self.ms_interval)
                wait_seconds = (next_candle_time_utc - now).total_seconds() + 2 # Wait till ~2 seconds past close

                # Basic sanity checks for wait time
                max_wait = self.ms_interval / 1000 * 1.5 # Allow 50% extra time max
                if wait_seconds > max_wait:
                     self.logger.warning(f"Calculated wait time ({wait_seconds:.1f}s) > 1.5x interval. Clamping to {max_wait:.1f}s.")
                     wait_seconds = max_wait
                if wait_seconds < -10: # If more than 10s behind schedule
                     self.logger.warning(f"System lagging significantly ({abs(wait_seconds):.1f}s behind). Processing immediately.")
                     wait_seconds = 0 # Process immediately

                if wait_seconds > 0:
                     self.logger.debug(f"Waiting {wait_seconds:.2f} seconds until next expected candle time ({next_candle_time_utc})...")
                     time.sleep(wait_seconds)


                # --- Fetch, Calculate, Trade ---
                new_candle_fetched = self._fetch_latest_candle()

                if new_candle_fetched:
                    indicators_ok = self._calculate_indicators()
                    if indicators_ok:
                        self._manage_position()
                    else:
                        self.logger.warning("Skipping position management due to indicator calculation issues on new candle.")
                else:
                    self.logger.debug("No new candle fetched in this cycle.")
                    # Add a small sleep to prevent tight loop if fetch consistently fails or during normal wait
                    time.sleep(min(10, self.ms_interval / 1000 / 5)) # Sleep briefly


            except KeyboardInterrupt:
                self.logger.info("Forward testing stopped by user.")
                break # Exit the loop cleanly
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                self.logger.info("Attempting to continue after 30 seconds...")
                time.sleep(30)

        # --- End of Run Summary ---
        self.logger.info("Forward testing loop finished.")
        self.logger.info(f"Final Simulated Balance: ${self.simulated_balance:.2f}")
        performance = (self.simulated_balance / self.initial_balance - 1) * 100 if self.initial_balance else 0
        self.logger.info(f"Total Performance: {performance:.2f}%")


# --- Main Execution ---
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logging.critical("Binance API Key/Secret not found. Set environment variables BINANCE_API_KEY and BINANCE_SECRET_KEY.")
    else:
        try:
            trader = ForwardSmaAdxSuperTrader(
                symbol=SYMBOL,
                interval=INTERVAL,
                lookback=LOOKBACK_PERIODS,
                sma_short=SMA_SHORT,
                sma_long=SMA_LONG,
                adx_threshold=ADX_THRESHOLD,
                adx_period=ADX_PERIOD,
                st_period=SUPERTREND_PERIOD,
                st_multiplier=SUPERTREND_MULTIPLIER,
                atr_period=ATR_PERIOD,
                tp_multiplier=TP_MULTIPLIER,
                sl_multiplier=STOP_MULTIPLIER, # Use the name from your params
                leverage=LEVERAGE,
                tc=TC,
                initial_balance=SIMULATED_INITIAL_BALANCE
            )
            trader.run()
        except ValueError as ve:
             logging.critical(f"Configuration error: {ve}")
        except Exception as main_e:
             logging.critical(f"Failed to initialize or run trader: {main_e}", exc_info=True)