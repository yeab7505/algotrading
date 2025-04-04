import pandas as pd
import pandas_ta as ta
import numpy as np
from binance.client import Client
import time
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta, UTC
import sys
import threading

# --- Configuration ---
load_dotenv()
API_KEY = "iOgcObLOw4UIFSvvEPXLFP1vgwp1wzyHYfw57vd1vrg19Xt6SXCE4RywDi5QoM28"
API_SECRET = "bz1m4UlthzklqXlWoqAqXZiJE35jjT0g5uJ5cQ43vwDNnsIpPYS5OqevfBVz84iK"

orders = pd.DataFrame(columns=['symbol', 'side', 'entry_price', 'tp_price', 'sl_price' ])
# --- Strategy & Trading Parameters ---
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK_PERIODS = 100
TP_MULTIPLIER = 2
SP_MULTIPLIER = 1.5
LEVERAGE = 1
TC = 0.0005
SIMULATED_INITIAL_BALANCE = 1000
ASSETS = ['ETHUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT','SOLUSDT',"TONUSDT",'DOGEUSDT','TRXUSDT','SHIBUSDT','GUNUSDT','TUTUSDT']

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

class ForwardIchimokuTrader:
    def __init__(self, symbol: str, interval: str, lookback: int,
                 tp_multiplier: float = 2, sp_multiplier: float = 1.5,
                 leverage: int = 1, tc: float = 0.0005, initial_balance: float = 1000) -> None:
        self.symbol = symbol
        self.interval = interval
        self.interval_5m = Client.KLINE_INTERVAL_5MINUTE
        self.lookback = lookback
        self.required_lookback = max(lookback, 52, 27)
        if lookback < self.required_lookback:
            logging.warning(f"Initial lookback {lookback} increased to {self.required_lookback} for indicator calculations.")
            self.lookback = self.required_lookback

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

    def _fetch_initial_data(self) -> bool:
        self.logger.info(f"Fetching initial {self.lookback + 50} klines for {self.symbol}...")
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=self.lookback + 50)
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
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=2)
            if not klines or len(klines) < 2:
                self.logger.warning("Could not fetch latest klines or not enough data yet (< 2).")
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

            if len(self.df) >= 15:
                self.df['atr'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
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
        

        buy_signal = (current_close > close_t_minus_26) and (conversion_line > base_line) and (leading_span_A > leading_span_B) and (current_close > leading_span_B)
        sell_signal = (current_close < close_t_minus_26) and (conversion_line < base_line) and (leading_span_B > leading_span_A) and (current_close < leading_span_A)

        self.logger.debug(f"15min. Signals @ {last_index_name}: Buy={buy_signal}, Sell={sell_signal}")
        return buy_signal, sell_signal

    def _fetch_5m_data(self, limit=100) -> pd.DataFrame:
        self.logger.debug(f"Fetching latest {limit} 5m klines for {self.symbol}...")
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval_5m, limit=limit)
            if not klines:
                self.logger.warning("Could not fetch 5m klines (empty list).")
                return None

            cols = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']
            data = pd.DataFrame(klines, columns=cols + ['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
            data = data[cols]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
                self.logger.warning("NaN values in 5m OHLC data after conversion.")
                data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            data['Datetime'] = pd.to_datetime(data['Close_time'], unit='ms')
            data.set_index('Datetime', inplace=True)
            df_5m = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            if len(df_5m) < 27:
                self.logger.warning(f"Insufficient 5m data fetched ({len(df_5m)} rows). Need at least 27.")
                return None

            return df_5m
        except Exception as e:
            self.logger.error(f"Error fetching 5m data: {e}", exc_info=True)
            return None

    def _calculate_5m_indicators(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        if df_5m is None or len(df_5m) < 52:
            self.logger.warning("Insufficient data for 5m Ichimoku calculation.")
            return None

        try:
            ichimoku_data = ta.ichimoku(df_5m['High'], df_5m['Low'], df_5m['Close'])
            if ichimoku_data is None or not isinstance(ichimoku_data, tuple) or len(ichimoku_data) < 1 or ichimoku_data[0].empty:
                self.logger.warning("5m Ichimoku calculation returned unexpected/empty data.")
                return None

            temp_df_ichi = ichimoku_data[0].rename(columns={
                'ISA_9': 'leading Span A', 'ISB_26': 'leading Span B',
                'ITS_9': 'conversion line', 'IKS_26': 'base line',
                'ICS_26': 'lagging Span'
            })
            temp_df_ichi.index = df_5m.index[-len(temp_df_ichi):]
            df_5m = df_5m.join(temp_df_ichi)
            return df_5m
        except Exception as e:
            self.logger.error(f"Error calculating 5m indicators: {e}", exc_info=True)
            return None

    def _check_5m_confirmation(self, direction: str) -> bool:
        df_5m = self._fetch_5m_data(limit=100)
        if df_5m is None:
            return False

        df_5m = self._calculate_5m_indicators(df_5m)
        if df_5m is None:
            return False

        if len(df_5m) < 27:
            self.logger.warning("Not enough 5m data for signal check.")
            return False

        last_5m = df_5m.iloc[-1]
        close_t_minus_26_5m = df_5m['Close'].iloc[-27]

        if pd.isna(close_t_minus_26_5m) or last_5m[['Close', 'conversion line', 'base line']].isnull().any():
            self.logger.warning("NaN values in 5m data for signal check.")
            return False

        buy_cond_5m = (last_5m['Close'] > close_t_minus_26_5m) and (last_5m['conversion line'] > last_5m['base line'])
        sell_cond_5m = (last_5m['Close'] < close_t_minus_26_5m) and (last_5m['conversion line'] < last_5m['base line'])

        if direction == 'buy':
            self.logger.debug(f"5m Buy confirmation: {buy_cond_5m}")
            return buy_cond_5m
        elif direction == 'sell':
            self.logger.debug(f"5m Sell confirmation: {sell_cond_5m}")
            return sell_cond_5m
        else:
            self.logger.error(f"Invalid direction {direction} in _check_5m_confirmation.")
            return False

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
            self.sl_level = self.highest_price_since_entry - (self.sp_multiplier * atr)
            if current_price_high >= self.tp_level:
                self._close_position(current_price, f"Take Profit hit at {self.tp_level:.4f}")
            elif current_price_low <= self.sl_level:
                self._close_position(current_price, f"Stop Loss hit at {self.sl_level:.4f}")
        elif self.position == -1:
            if self.tp_level is None or self.sl_level is None or self.lowest_price_since_entry is None:
                self.logger.error(f"Inconsistent state for short position @ {last_row.name}. Resetting.")
                self._reset_position_state()
                return
            self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)
            self.sl_level = self.lowest_price_since_entry + (self.sp_multiplier * atr)
            if current_price <= self.tp_level:
                self._close_position(current_price, f"Take Profit hit at {self.tp_level:.4f}")
            elif current_price >= self.sl_level:
                self._close_position(current_price, f"Stop Loss hit at {self.sl_level:.4f}")

        if self.position == 0:
            buy_signal, sell_signal = self._check_signals()
            self.logger.debug(f"Entry check: Buy signal={buy_signal}, Sell signal={sell_signal}")
            if buy_signal:
                confirmation = self._check_5m_confirmation('buy')
                self.logger.debug(f"5m confirmation for buy: {confirmation}")
                if confirmation:
                    self._enter_position(1, current_price, atr)
            elif sell_signal:
                confirmation = self._check_5m_confirmation('sell')
                self.logger.debug(f"5m confirmation for sell: {confirmation}")
                if confirmation:
                    self._enter_position(-1, current_price, atr)

    def _check_accidental_exit(self):
        try:
            last_row = self.df.iloc[-1]
            current_price = last_row['Close']
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error accessing data in _check_accidental_exit: {e}")
            return

        if self.position == 1:
            if self._check_signals()[0] and self._check_5m_confirmation('buy'):
                self.logger.info("Accidental exit check: Long position still valid.")
            else:
                self.logger.info("Accidental exit check: Long position invalid. Closing position.")
                self._close_position(current_price, "Accidental exit due to signal change.")
        elif self.position == -1:
            if self._check_signals()[1] and self._check_5m_confirmation('sell'):
                self.logger.info("Accidental exit check: Short position still valid.")
            else:
                self.logger.info("Accidental exit check: Short position invalid. Closing position.")
                self._close_position(current_price, "Accidental exit due to signal change.")

    def _enter_position(self, direction: int, entry_price: float, atr: float) -> None:
        if pd.isna(atr) or atr <= 0:
            self.logger.error(f"Attempted to enter position with invalid ATR: {atr}. Aborting entry.")
            return

        self.position = direction
        self.entry_price = entry_price
        entry_cost = self.simulated_balance * self.tc
        self.simulated_balance -= entry_cost

        log_msg_base = f"PAPER TRADE: Entered {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.4f}"
        try:
            if direction == 1:
                self.tp_level = entry_price + (atr * self.tp_multiplier)
                self.sl_level = entry_price - (atr * self.sp_multiplier)
                self.highest_price_since_entry = entry_price
                self.lowest_price_since_entry = None
                log_msg = f"{log_msg_base} | TP: {self.tp_level:.4f}, SL: {self.sl_level:.4f} | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
                orders.loc[len(orders)] = [self.symbol, 'BUY', entry_price, self.tp_level, self.sl_level]
            elif direction == -1:
                self.tp_level = entry_price - (atr * self.tp_multiplier)
                self.sl_level = entry_price + (atr * self.sp_multiplier)
                self.lowest_price_since_entry = entry_price
                self.highest_price_since_entry = None
                log_msg = f"{log_msg_base} | TP: {self.tp_level:.4f}, SL: {self.sl_level:.4f} | Entry Cost: {entry_cost:.4f} | Bal: {self.simulated_balance:.2f}"
                orders.loc[len(orders)] = [self.symbol, 'SELL', entry_price, self.tp_level, self.sl_level]
            else:
                self.logger.error(f"Invalid direction {direction} passed to _enter_position.")
                self._reset_position_state()
                return
            self.logger.info(log_msg)
        except Exception as e:
            self.logger.error(f"Error calculating TP/SL during entry: {e}. Position may be invalid.", exc_info=True)
            self._reset_position_state()

    def _close_position(self, exit_price: float, reason: str) -> None:
        if self.position == 0:
            self.logger.warning("Attempted to close position while already flat.")
            return

        pnl_percentage = 0
        if self.position == 1:
            pnl_percentage = (exit_price / self.entry_price - 1) if self.entry_price != 0 else 0
            orders.drop(orders[orders['symbol'] == self.symbol].index[-1], inplace=True)
        elif self.position == -1:
            pnl_percentage = (self.entry_price / exit_price - 1) if exit_price != 0 else 0
            orders.drop(orders[orders['symbol'] == self.symbol].index[-1], inplace=True)

        effective_pnl = pnl_percentage * self.leverage
        exit_cost = self.simulated_balance * (1 + effective_pnl) * self.tc
        final_pnl_factor = (1 + effective_pnl) * (1 - self.tc)

        closed_direction = 'LONG' if self.position == 1 else 'SHORT'
        balance_before_close = self.simulated_balance
        self.simulated_balance *= final_pnl_factor

        self.logger.info(f"PAPER TRADE: Closed {closed_direction} @ {exit_price:.4f} | Entry: {self.entry_price:.4f} | Reason: {reason}")
        self.logger.info(f"PnL: {effective_pnl:.4%} (before exit cost) | Exit Cost: {exit_cost:.4f} | Balance Change: {self.simulated_balance - balance_before_close:.4f} | New Balance: {self.simulated_balance:.2f}")

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
        

        loop_count = 0
        while True:
            try:
                loop_count += 1
                server_now = self._get_server_time()
                last_candle_time_utc = self.df.index[-2].tz_localize(None)

                next_candle_time_utc = last_candle_time_utc + pd.Timedelta(milliseconds=self.ms_interval)
                wait_seconds = (next_candle_time_utc - server_now).total_seconds() + 2

                if wait_seconds < 0:
                    self.logger.warning(f"Calculated wait time is negative ({wait_seconds:.1f}s). Adjusting to 0.")
                    wait_seconds = 0
                elif wait_seconds > self.ms_interval / 1000 + 5:
                    self.logger.warning(f"Calculated wait time ({wait_seconds:.1f}s) exceeds expected interval. Clamping to interval + 5s.")
                    wait_seconds = self.ms_interval / 1000 + 5

                if wait_seconds > 0:
                    self.logger.debug(f"Waiting {wait_seconds:.2f} seconds until next expected candle time ({next_candle_time_utc})...")
                    time.sleep(wait_seconds)

                if loop_count % 10 == 0:
                    self.logger.debug(f"Current DF shape: {self.df.shape}")
                    self.logger.debug(f"Current Position: {self.position}, Balance: {self.simulated_balance:.2f}")
                    self.logger.debug(f"---------------------------------")

                new_candle_fetched = self._fetch_latest_candle()
                if new_candle_fetched: 
                    indicators_ok = self._calculate_indicators()
                    if indicators_ok:
                        self._manage_position()
                        self.logger.debug("Position management completed successfully.")
                        print(orders)
                        
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

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logging.critical("Binance API Key/Secret not found.")
    else:
        threads = []
        for asset in ASSETS:
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
            except Exception as e:
                logging.critical(f"Failed to initialize trader for {asset}: {e}", exc_info=True)

        # Optionally wait for all threads to finish (though they run indefinitely unless stopped)
        for thread in threads:
            thread.join()