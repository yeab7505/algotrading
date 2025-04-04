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

# --- Configuration ---
load_dotenv()
API_KEY = "iOgcObLOw4UIFSvvEPXLFP1vgwp1wzyHYfw57vd1vrg19Xt6SXCE4RywDi5QoM28"
API_SECRET = "bz1m4UlthzklqXlWoqAqXZiJE35jjT0g5uJ5cQ43vwDNnsIpPYS5OqevfBVz84iK"

# --- Strategy & Trading Parameters ---
SYMBOL = 'ETHUSDT'
INTERVAL = Client.KLINE_INTERVAL_5MINUTE # Make sure this matches your expectation
LOOKBACK_PERIODS = 100 # Min 52 for Ichimoku, 27 for Lagging Span signal. 100 is usually safe.
TP_MULTIPLIER = 2
SP_MULTIPLIER = 1.5
LEVERAGE = 1
TC = 0.0005
SIMULATED_INITIAL_BALANCE = 1000

# --- Logging Setup ---
# Set to DEBUG to see detailed logs
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
# Reduce verbosity of libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("binance").setLevel(logging.INFO) # Show basic binance client info

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

class ForwardIchimokuTrader:
    def __init__(self, symbol: str, interval: str, lookback: int,
                 tp_multiplier: float = 2, sp_multiplier: float = 1.5,
                 leverage: int = 3, tc: float = 0.0005, initial_balance: float = 1000) -> None:
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        # Ensure lookback is sufficient
        self.required_lookback = max(lookback, 52, 27) # Need at least 52 for Ichi B, 27 for lag span signal
        if lookback < self.required_lookback:
             logging.warning(f"Initial lookback {lookback} increased to {self.required_lookback} for indicator calculations.")
             self.lookback = self.required_lookback

        self.tp_multiplier = tp_multiplier
        self.sp_multiplier = sp_multiplier
        self.leverage = leverage
        self.tc = tc
        self.initial_balance = initial_balance
        self.simulated_balance = initial_balance

        # Use a specific logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trader...")

        try:
            self.client = Client(API_KEY, API_SECRET)
            self.client.ping() # Test connection
            self.logger.info("Binance client initialized and connection tested.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}", exc_info=True)
            raise

        self.df = pd.DataFrame()
        # self.df_ichi = pd.DataFrame() # We calculate and join directly now

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


    def _fetch_initial_data(self) -> bool:
        self.logger.info(f"Fetching initial {self.lookback + 50} klines for {self.symbol}...")
        try:
            # Fetch slightly more initially
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=self.lookback + 50) # Fetch ample buffer
            if not klines:
                self.logger.error("Could not fetch initial klines (received empty list).")
                return False

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                    'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']
            data = pd.DataFrame(klines, columns=cols)

            # --- Data Validation ---
            if data.empty:
                 self.logger.error("Initial klines resulted in empty DataFrame.")
                 return False

            # Convert relevant columns to numeric, coerce errors to NaN
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Check for NaNs introduced by conversion
            if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                 self.logger.warning("NaN values found in OHLC data after numeric conversion during initial fetch.")
                 # Optionally drop rows with NaN OHLC, but log it
                 initial_len = len(data)
                 data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                 self.logger.warning(f"Dropped {initial_len - len(data)} rows with NaN OHLC.")

            # Use Close_time for indexing
            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms')
            data.set_index('Datetime', inplace=True)

            # Keep only necessary columns
            self.df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # --- Post-Fetch Validation ---
            if len(self.df) < self.required_lookback:
                 self.logger.error(f"Insufficient valid initial data fetched ({len(self.df)} rows) after processing. Need at least {self.required_lookback}.")
                 return False

            # Check if index is monotonic increasing
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

            latest_kline = klines[-2] # Latest fully closed candle
            close_time_ms = latest_kline[6]
            latest_dt = pd.to_datetime(close_time_ms, unit='ms')

            if self.df.empty:
                 self.logger.warning("Main DataFrame is empty, cannot compare timestamps. Re-fetching initial data might be needed.")
                 return False # Or trigger re-init

            if latest_dt <= self.df.index[-1]:
                 # self.logger.debug(f"Candle {latest_dt} already processed (last known: {self.df.index[-1]}).")
                 return False # No new candle

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time'] # Reduced columns needed
            new_data = pd.DataFrame([latest_kline], columns=cols + ['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
            new_data = new_data[cols] # Keep only needed

            # Convert to numeric, check for NaN
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            if new_data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                 self.logger.error(f"NaN value detected in OHLC for new candle at {latest_dt}. Skipping append.")
                 return False # Don't append bad data

            new_data['Datetime'] = latest_dt
            new_data.set_index('Datetime', inplace=True)

            new_row = new_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[0]

            # --- Append and Validate ---
            # Ensure index is compatible before concatenating
            if not isinstance(self.df.index, pd.DatetimeIndex):
                 self.logger.error("Main DataFrame index is not DatetimeIndex. Cannot append.")
                 # Attempt recovery or raise error
                 return False

            self.df = pd.concat([self.df, new_row.to_frame().T])

            # Verify index monotonicity after append
            if not self.df.index.is_monotonic_increasing:
                 self.logger.warning(f"Index became non-monotonic after adding candle {latest_dt}. Sorting...")
                 self.df.sort_index(inplace=True)
                 # Check again
                 if not self.df.index.is_monotonic_increasing:
                      self.logger.error("Index still non-monotonic after sorting. Halting.")
                      raise ValueError("DataFrame index could not be kept monotonic.")


            # Keep DataFrame size manageable - Trim **older** data
            # Ensure we always keep enough for lookbacks
            max_len = self.lookback + 200 # Keep a larger buffer
            if len(self.df) > max_len:
                 self.logger.debug(f"Trimming DataFrame from {len(self.df)} to {max_len} rows.")
                 self.df = self.df.iloc[-max_len:]

            self.logger.info(f"New candle appended: {self.df.index[-1]}, Close: {self.df['Close'].iloc[-1]:.4f}, DF rows: {len(self.df)}")
            return True

        except Exception as e:
            self.logger.error(f"Error fetching/appending latest candle: {e}", exc_info=True)
            # Potentially add logic here to check connection or retry after a delay
            return False


    def _calculate_indicators(self) -> bool:
        self.logger.debug(f"Calculating indicators for {len(self.df)} rows...")
        min_data_needed = self.required_lookback # Use the calculated required lookback
        if len(self.df) < min_data_needed:
            self.logger.warning(f"Not enough data ({len(self.df)}, need {min_data_needed}) for all indicator calculations.")
            return False
        try:
            # --- Pre-calculation Check ---
            if self.df[['High', 'Low', 'Close']].isnull().any().any():
                 self.logger.warning("NaN values found in High, Low, or Close columns BEFORE indicator calculation. Results may be inaccurate.")
                 # You might choose to return False here if this is critical
                 # return False

            # Calculate Ichimoku
            ichimoku_data = ta.ichimoku(self.df['High'], self.df['Low'], self.df['Close'])
            if ichimoku_data is None or not isinstance(ichimoku_data, tuple) or len(ichimoku_data) < 1 or ichimoku_data[0].empty:
                 self.logger.warning("Ichimoku calculation returned unexpected/empty data.")
                 return False

            temp_df_ichi = ichimoku_data[0].rename(columns={
                'ISA_9': 'leading Span A', 'ISB_26': 'leading Span B',
                'ITS_9': 'conversion line', 'IKS_26': 'base line',
                'ICS_26': 'lagging Span' # This will have NaNs at the end, it's ok
            })
            # Align index explicitly to avoid issues if calculation returns slightly different index
            temp_df_ichi.index = self.df.index[-len(temp_df_ichi):]


            # Calculate ATR
            # Ensure enough data for ATR length (e.g., 14 + 1 = 15)
            if len(self.df) >= 15:
                 self.df['atr'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
            else:
                 self.logger.warning(f"Not enough data ({len(self.df)}) for ATR(14) calculation.")
                 self.df['atr'] = np.nan # Assign NaN if not enough data

            # Remove old Ichimoku columns before joining new ones
            ichimoku_cols_to_drop = list(temp_df_ichi.columns) # Get names from the temp df
            cols_present_in_df = [col for col in ichimoku_cols_to_drop if col in self.df.columns]
            if cols_present_in_df:
                self.logger.debug(f"Dropping existing columns: {cols_present_in_df}")
                self.df = self.df.drop(columns=cols_present_in_df)

            # Join the newly calculated Ichimoku columns
            self.df = self.df.join(temp_df_ichi)
            self.logger.debug("Indicators calculated and joined to DataFrame.")

            # --- Post-calculation Check (Optional but recommended) ---
            # Check the *last* row for NaNs in indicators needed for signals
            last_row_indicators = self.df.iloc[-1][['conversion line', 'base line', 'atr']] # Check these for signal/SL/TP logic
            if last_row_indicators.isnull().any():
                 self.logger.warning(f"NaN values found in key indicators on the latest row ({self.df.index[-1]}): {last_row_indicators.isnull()}")
                 # Decide if this should prevent trading
                 # return False # Uncomment if a NaN in these specific indicators should halt signals
            print(f"Indicators calculated for {self.symbol} on {self.interval} interval. Latest indicators: {last_row_indicators}")
            return True

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return False


    def _check_signals(self) -> tuple[bool, bool]:
        self.logger.debug("Checking signals...")
        current_length = len(self.df)

        # Check for NaN in 'Close' column
        if self.df['Close'].isnull().any():
            self.logger.warning(f"NaN values found in 'Close' column. Total NaNs: {self.df['Close'].isnull().sum()}")
            self.logger.debug(f"Last 5 rows of 'Close':\n{self.df['Close'].tail(5)}")

        if current_length < 27:
            self.logger.warning(f"DataFrame length ({current_length}) is less than 27. Cannot perform lagging span comparison yet.")
            return False, False

        last = self.df.iloc[-1]
        last_index_name = last.name

        required_cols = ['Close', 'conversion line', 'base line', 'leading Span A', 'leading Span B']
        if last[required_cols].isnull().any():
            self.logger.warning(f"Signal check skipped: NaN in required columns at {last_index_name}.")
            return False, False

        try:
            target_past_index_pos = -1 - 26
            target_past_index = self.df.index[target_past_index_pos]
            close_t_minus_26 = self.df.loc[target_past_index, 'Close']
            self.logger.debug(f"Lagging Span Check: Close[{last_index_name}]={last['Close']:.4f}, Close[{target_past_index}]={close_t_minus_26}")
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error accessing T-26 Close: {e}. Length: {current_length}")
            return False, False

        if pd.isna(close_t_minus_26):
            self.logger.warning(f"Signal check skipped: Close at {target_past_index} is NaN.")
            self.logger.debug(f"Data around T-26:\n{self.df.loc[target_past_index-2:target_past_index+2, 'Close']}")
            return False, False

        current_close = last['Close']
        conversion_line = last['conversion line']
        base_line = last['base line']
        leading_span_A = last['leading Span A']
        leading_span_B = last['leading Span B']
        lagging_span = close_t_minus_26

        buy_cond1 = current_close > close_t_minus_26
        buy_cond2 = conversion_line > base_line
        buy_cond3 = leading_span_A > leading_span_B
        sell_cond1 = current_close < close_t_minus_26
        sell_cond2 = conversion_line < base_line
        sell_cond3 = leading_span_A < leading_span_B

        buy_signal = buy_cond1 and buy_cond2 and buy_cond3
        sell_signal = sell_cond1 and sell_cond2 and sell_cond3

        self.logger.debug(f"Signals @ {last_index_name}: Buy={buy_signal}, Sell={sell_signal}")
        return buy_signal, sell_signal


    def _manage_position(self) -> None:
        # (Keep the _manage_position, _enter_position, _close_position methods as they were,
        # ensuring they use self.logger for logging)
        # ... (Add self.logger.info/debug calls inside these methods) ...
        """Manages exits (TP/SL) and entries based on signals."""
        if self.df.empty or 'atr' not in self.df.columns:
            self.logger.warning("DataFrame not ready for position management (empty or no ATR).")
            return

        try:
            last_row = self.df.iloc[-1]
            current_price_high = last_row['High']
            current_price_low = last_row['Low']
            current_price = last_row['Close']
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
            self.logger.warning(f"Invalid ATR ({atr}) at {last_row.name}. Skipping position management.")
            # We need ATR for TP/SL, so we usually can't proceed without it.
            # If we are in a position, maybe only check SL/TP if they were set previously?
            # For simplicity now, we just skip if ATR is bad.
            return

        # --- Check Exits First ---
        if self.position == 1: # Long position active
            if self.tp_level is None or self.sl_level is None or self.highest_price_since_entry is None:
                 self.logger.error(f"Inconsistent state for long position @ {last_row.name}. TP/SL/HighestPrice not set. Resetting position state.")
                 self._reset_position_state() # Add a helper function to reset state safely
                 return

            self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
            trailing_sl = self.highest_price_since_entry - (self.sp_multiplier * atr)
            self.sl_level = max(self.sl_level, trailing_sl) # SL can only move up
            winning_ = 0
            losing_ = 0
            exit_reason = None
            if current_price_high >= self.tp_level:
                exit_reason = f"Take Profit hit at {self.tp_level:.4f}"
                winning_ += 1
                print(winning_)
            elif current_price_low <= self.sl_level:
                exit_reason = f"Stop Loss hit at {self.sl_level:.4f}"
                losing_ += 1
                print(losing_)
            if exit_reason:
                self._close_position(current_price, exit_reason)

        elif self.position == -1: # Short position active
            if self.tp_level is None or self.sl_level is None or self.lowest_price_since_entry is None:
                 self.logger.error(f"Inconsistent state for short position @ {last_row.name}. TP/SL/LowestPrice not set. Resetting position state.")
                 self._reset_position_state()
                 return

            self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)
            trailing_sl = self.lowest_price_since_entry + (self.sp_multiplier * atr)
            self.sl_level = min(self.sl_level, trailing_sl) # SL can only move down

            exit_reason = None
            if current_price <= self.tp_level:
                exit_reason = f"Take Profit hit at {self.tp_level:.4f}"
            elif current_price >= self.sl_level:
                exit_reason = f"Stop Loss hit at {self.sl_level:.4f}"

            if exit_reason:
                self._close_position(current_price, exit_reason)


        # --- Check Entries (Only if flat) ---
        if self.position == 0:
            # Ensure ATR is valid before considering entry (as it's needed for TP/SL calc)
             if pd.isna(atr) or atr <= 0:
                 self.logger.debug(f"Skipping entry check due to invalid ATR ({atr}) at {last_row.name}")
                 return

             buy_signal, sell_signal = self._check_signals()

             if buy_signal:
                 self._enter_position(1, current_price, atr)
             elif sell_signal:
                 self._enter_position(-1, current_price, atr)

    def _enter_position(self, direction: int, entry_price: float, atr: float) -> None:
        """Simulates entering a position."""
        if pd.isna(atr) or atr <= 0:
             self.logger.error(f"Attempted to enter position with invalid ATR: {atr}. Aborting entry.")
             return

        self.position = direction
        self.entry_price = entry_price
        entry_cost = self.simulated_balance * self.tc
        self.simulated_balance -= entry_cost # Simulate entry cost

        log_msg_base = f"PAPER TRADE: Entered {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.4f}"

        try:
            if direction == 1: # Long
                self.tp_level = entry_price + (atr * self.tp_multiplier)
                self.sl_level = entry_price - (atr * self.sp_multiplier)
                self.highest_price_since_entry = entry_price
                self.lowest_price_since_entry = None # Reset for long
                log_msg = f"{log_msg_base} | TP: {self.tp_level:.4f}, SL: {self.sl_level:.4f} | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
            elif direction == -1: # Short
                self.tp_level = entry_price - (atr * self.tp_multiplier)
                self.sl_level = entry_price + (atr * self.sp_multiplier)
                self.lowest_price_since_entry = entry_price
                self.highest_price_since_entry = None # Reset for short
                log_msg = f"{log_msg_base} | TP: {self.tp_level:.4f}, SL: {self.sl_level:.4f} | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
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

        pnl_percentage = 0
        if self.position == 1: # Closing Long
            pnl_percentage = (exit_price / self.entry_price - 1) if self.entry_price != 0 else 0
        elif self.position == -1: # Closing Short
            pnl_percentage = (self.entry_price / exit_price - 1) if exit_price != 0 else 0

        # Apply leverage effect and transaction cost for exit
        effective_pnl = pnl_percentage * self.leverage
        exit_cost = self.simulated_balance * (1 + effective_pnl) * self.tc # Cost based on value after PnL
        final_pnl_factor = (1 + effective_pnl) * (1 - self.tc)

        # Store position before resetting
        closed_direction = 'LONG' if self.position == 1 else 'SHORT'

        # Update balance
        balance_before_close = self.simulated_balance
        self.simulated_balance *= final_pnl_factor

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


    def run(self) -> None:
        """Main execution loop for the forward tester."""
        self.logger.info(f"Starting forward test for {self.symbol} on {self.interval} interval.")
        self.logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        self.logger.info(f"Parameters: TP Multi={self.tp_multiplier}, SL Multi={self.sp_multiplier}, TC={self.tc}")

        if not self._fetch_initial_data():
            self.logger.error("Failed to fetch initial data. Stopping.")
            return

        if not self._calculate_indicators():
             self.logger.warning("Initial indicator calculation failed. Will retry on next candle.")

        loop_count = 0
        while True:
            try:
                loop_count += 1
                now = datetime.utcnow()
                last_candle_time_utc = self.df.index[-2].tz_localize(None) # Ensure timezone naive for comparison

                # --- Calculate Wait Time ---
                next_candle_time_utc = last_candle_time_utc + pd.Timedelta(milliseconds=self.ms_interval)
                wait_seconds = (next_candle_time_utc - now).total_seconds() + 2 # Wait till ~2 seconds past candle close

                if wait_seconds > self.ms_interval / 1000 + 5: # Sanity check if wait time is too long
                     self.logger.warning(f"Calculated wait time ({wait_seconds:.1f}s) seems excessive. Clamping to interval + 5s.")
                     wait_seconds = self.ms_interval / 1000 + 5
                elif wait_seconds < 0 and abs(wait_seconds) > 10: # If we are significantly behind
                     self.logger.warning(f"System seems to be lagging. Behind schedule by {abs(wait_seconds):.1f}s.")
                     # Don't sleep if we're already behind, try to catch up.
                     wait_seconds = 0 # Process immediately

                if wait_seconds > 0:
                     self.logger.debug(f"Waiting {wait_seconds:.2f} seconds until next expected candle time ({next_candle_time_utc})...")
                     time.sleep(wait_seconds)

                # --- Periodic DataFrame Check (Every N loops) ---
                if loop_count % 10 == 0: # Check every 10 loops
                    self.logger.debug(f"--- Loop {loop_count} Periodic Check ---")
                    self.logger.debug(f"Current DF shape: {self.df.shape}")
                    self.logger.debug(f"DF index is monotonic? {self.df.index.is_monotonic_increasing}")
                    self.logger.debug(f"DF tail:\n{self.df.tail(3)}")
                    self.logger.debug(f"Current Position: {self.position}, Balance: {self.simulated_balance:.2f}")
                    self.logger.debug(f"---------------------------------")


                # --- Fetch, Calculate, Trade ---
                new_candle_fetched = self._fetch_latest_candle()

                if new_candle_fetched:
                    indicators_ok = self._calculate_indicators()
                    if indicators_ok:
                        self._manage_position()
                    else:
                        self.logger.warning("Skipping position management due to indicator calculation issues on new candle.")
                else:
                    # No new candle, maybe check API status or just wait for next cycle
                    self.logger.debug("No new candle fetched in this cycle.")
                    # Add a small sleep to prevent tight loop if fetch consistently fails
                    time.sleep(min(15, self.ms_interval / 1000 / 4))


            except KeyboardInterrupt:
                self.logger.info("Forward testing stopped by user.")
                break # Exit the loop cleanly
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                self.logger.info("Attempting to continue after 30 seconds...")
                time.sleep(30) # Wait before retrying after a major error

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
            trader = ForwardIchimokuTrader(
                symbol=SYMBOL,
                interval=INTERVAL,
                lookback=LOOKBACK_PERIODS,
                tp_multiplier=TP_MULTIPLIER,
                sp_multiplier=SP_MULTIPLIER,
                leverage=LEVERAGE,
                tc=TC,
                initial_balance=SIMULATED_INITIAL_BALANCE
            )
            trader.run()
        except ValueError as ve:
             logging.critical(f"Configuration error: {ve}")
        except Exception as main_e:
             logging.critical(f"Failed to initialize or run trader: {main_e}", exc_info=True)