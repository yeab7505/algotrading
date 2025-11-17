"""
Gemini AI Market Consolidation Detector
This module integrates with Google's Gemini AI to analyze market conditions
and determine if the market is consolidating or trending.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import date
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Daily request limit
DAILY_REQUEST_LIMIT = 250

class GeminiMarketAnalyzer:
    """
    Uses Google's Gemini AI to analyze market data and determine consolidation status.
    """
    
    def __init__(self, api_key: Optional[str] = None, daily_limit: int = DAILY_REQUEST_LIMIT):
        """
        Initialize the Gemini Market Analyzer.
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from environment.
            daily_limit: Daily request limit (default: 250)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.daily_limit = daily_limit
        self.request_count = 0
        self.last_reset_date = date.today()
        self._reset_if_new_day()
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
            logger.warning("Set it in your .env file or pass it during initialization.")
            self.client = None
        else:
            genai.configure(api_key=self.api_key)
            # Use the latest Gemini 2.5 models (Flash is faster and cheaper)
            try:
                self.client = genai.GenerativeModel('gemini-2.5-pro')
                logger.info("Gemini AI initialized successfully with gemini-2.5-flash")
            except Exception as e:
                logger.warning(f"Failed to initialize gemini-2.5-flash: {e}, trying gemini-2.5-pro...")
                try:
                    self.client = genai.GenerativeModel('gemini-2.5-flash')
                    logger.info("Gemini AI initialized successfully with gemini-2.5-pro")
                except Exception as e2:
                    logger.warning(f"Failed to initialize gemini-2.5-pro: {e2}, trying gemini-1.5-flash")
                    try:
                        self.client = genai.GenerativeModel('gemini-1.5-flash')
                        logger.info("Gemini AI initialized successfully with gemini-1.5-flash")
                    except Exception as e3:
                        self.client = genai.GenerativeModel('gemini-1.5-pro')
                        logger.info("Gemini AI initialized successfully with gemini-1.5-pro")
        
        logger.info(f"Request tracking initialized: {self.get_remaining_requests()} requests remaining today")
    
    def _reset_if_new_day(self):
        """Reset request count if it's a new day."""
        today = date.today()
        if today != self.last_reset_date:
            self.request_count = 0
            self.last_reset_date = today
            logger.info(f"Daily request counter reset. New day: {today}")
    
    def get_remaining_requests(self) -> int:
        """Get the number of remaining API requests for today."""
        self._reset_if_new_day()
        remaining = max(0, self.daily_limit - self.request_count)
        return remaining
    
    def _increment_request_count(self):
        """Increment request count and log remaining requests."""
        self._reset_if_new_day()
        self.request_count += 1
        remaining = self.get_remaining_requests()
        logger.info(f"API Request #{self.request_count}/{self.daily_limit} used. {remaining} requests remaining today")
        
        if remaining <= 10:
            logger.warning(f"Low API quota: Only {remaining} requests remaining today!")
        elif remaining <= 0:
            logger.error("API quota exhausted! No more requests available today.")
    
    def _prepare_market_summary(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Prepare a human-readable summary of market data for Gemini analysis.
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Formatted string with market data summary
        """
        if df.empty or len(df) < 10:
            return "Insufficient data for analysis"
        
        last_row = df.iloc[-1]
        recent_data = df.iloc[-20:] if len(df) >= 20 else df
        
        # Calculate basic statistics
        price_range = df['Close'].max() - df['Close'].min()
        price_range_pct = (price_range / df['Close'].min()) * 100 if df['Close'].min() > 0 else 0
        
        recent_volatility = recent_data['Close'].std()
        overall_volatility = df['Close'].std()
        volatility_ratio = recent_volatility / overall_volatility if overall_volatility > 0 else 0
        
        # Trend analysis
        current_price = last_row['Close']
        price_20_ago = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
        price_change_pct = ((current_price - price_20_ago) / price_20_ago * 100) if price_20_ago > 0 else 0
        
        # Indicator values
        has_choppy = 'choppy' in last_row and not pd.isna(last_row['choppy'])
        choppy_value = last_row['choppy'] if has_choppy else None
        
        has_atr = 'atr' in last_row and not pd.isna(last_row['atr'])
        atr_value = last_row['atr'] if has_atr else None
        
        has_adx = 'ADX_14' in last_row and not pd.isna(last_row['ADX_14'])
        adx_value = last_row['ADX_14'] if has_adx else None
        
        # Compute ATR if not available and derive structure metrics
        computed_atr = None
        try:
            highs = df['High']
            lows = df['Low']
            closes = df['Close']
            prev_close = closes.shift(1)
            tr = pd.concat([
                highs - lows,
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ], axis=1).max(axis=1)
            computed_atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
        except Exception:
            computed_atr = None
        effective_atr = float(atr_value) if atr_value is not None else (float(computed_atr) if computed_atr is not None else 0.0)
        
        # Structural metrics over recent window
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        recent_range = recent_high - recent_low
        recent_range_pct = (recent_range / current_price * 100) if current_price else 0.0
        range_to_atr = (recent_range / effective_atr) if effective_atr > 0 else 0.0
        
        # Linear trend slope over recent window (percent over window)
        try:
            y = recent_data['Close'].values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            trend_slope_pct = (slope * len(y)) / y[0] * 100 if y[0] != 0 else 0.0
        except Exception:
            trend_slope_pct = 0.0
        
        # Count structure: higher highs/lows vs lower highs/lows across last 10 candles
        window_for_structure = min(len(df), 10)
        hh = hl = lh = ll = 0
        if window_for_structure >= 3:
            highs_series = df['High'].iloc[-window_for_structure:]
            lows_series = df['Low'].iloc[-window_for_structure:]
            for i in range(2, window_for_structure):
                prev_high = float(highs_series.iloc[i-1])
                prev_low = float(lows_series.iloc[i-1])
                cur_high = float(highs_series.iloc[i])
                cur_low = float(lows_series.iloc[i])
                if cur_high > prev_high:
                    hh += 1
                if cur_low > prev_low:
                    hl += 1
                if cur_high < prev_high:
                    lh += 1
                if cur_low < prev_low:
                    ll += 1
        
        # Average candle body vs ATR
        body_sizes = (df['Close'] - df['Open']).abs().iloc[-20:]
        avg_body = body_sizes.mean() if len(body_sizes) else 0.0
        body_to_atr = (avg_body / effective_atr) if effective_atr > 0 else 0.0
        
        # Volume analysis
        recent_volume_avg = recent_data['Volume'].mean()
        overall_volume_avg = df['Volume'].mean()
        volume_ratio = recent_volume_avg / overall_volume_avg if overall_volume_avg > 0 else 0
        
        summary = f"""
MARKET DATA FOR: {symbol}
====================
CURRENT PRICE: ${current_price:.4f}
PRICE ACTION: Change over last 20 candles = {price_change_pct:.2f}%

MARKET CONDITIONS:
- Price Range: ${price_range:.4f} ({price_range_pct:.2f}% of min price)
- Volatility: {volatility_ratio:.2f}x recent vs overall
- Volume Trend: {volume_ratio:.2f}x (1.0=normal, >1.0=increasing, <1.0=decreasing)

STRUCTURAL METRICS (last ~20 candles):
- Trend Slope: {trend_slope_pct:.2f}% over window
- Recent Range: ${recent_range:.4f} ({recent_range_pct:.2f}% of price)
- Range to ATR: {range_to_atr:.2f}x
- Avg Body to ATR: {body_to_atr:.2f}x
- Structure counts: HH={hh}, HL={hl}, LH={lh}, LL={ll}
- ADX_14: {adx_value if adx_value is not None else 'N/A'}
- Choppy: {choppy_value if choppy_value is not None else 'N/A'}

Recent Price Action (Last 5 candles):
"""
        
        for i in range(-5, 0):
            row = df.iloc[i]
            summary += f"Candle {i+6}: O:{row['Open']:.4f} H:{row['High']:.4f} L:{row['Low']:.4f} C:{row['Close']:.4f} V:{row['Volume']:.0f}\n"
        
        return summary
    
    def _prepare_batch_summary(self, analyses: List[Tuple[pd.DataFrame, str, Optional[str]]]) -> str:
        """
        Prepare a batch summary of multiple markets for single API call.
        
        Args:
            analyses: List of tuples (df, symbol, context)
            
        Returns:
            Formatted string with all market summaries
        """
        batch_summary = "BATCH MARKET ANALYSIS - Multiple Symbols\n"
        batch_summary += "=" * 50 + "\n\n"
        
        for idx, (df, symbol, context) in enumerate(analyses, 1):
            summary = self._prepare_market_summary(df, symbol)
            if context:
                summary += f"\nAdditional Context: {context}\n"
            batch_summary += f"\n{'='*50}\n"
            batch_summary += f"SYMBOL {idx}/{len(analyses)}: {symbol}\n"
            batch_summary += f"{'='*50}\n"
            batch_summary += summary + "\n"
        
        return batch_summary
    
    def analyze_consolidation(self, df: pd.DataFrame, symbol: str, 
                             context: Optional[str] = None) -> Tuple[bool, str]:
        """
        Use Gemini AI to determine if the market is consolidating.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Trading symbol
            context: Additional context or constraints
            
        Returns:
            Tuple of (is_consolidating: bool, reasoning: str)
        """
        if self.client is None:
            logger.error("Gemini client not initialized. Cannot analyze consolidation.")
            return False, "Gemini client not initialized. Check API key."
        
        if self.get_remaining_requests() <= 0:
            logger.error("API quota exhausted. Cannot make request.")
            return True, "API quota exhausted. Please try again tomorrow."
        
        try:
            # Prepare market summary
            market_summary = self._prepare_market_summary(df, symbol)
            
            # Create optimized prompt for Gemini
            prompt = f"""You are an expert crypto market analyst. Determine if the market is in CONSOLIDATION or TRENDING.

Evaluate: NEWS/EVENTS, CORRELATION (BTC/ETH leadership), MARKET STRUCTURE (support/resistance, volume), SENTIMENT.

Definitions:
- CONSOLIDATION = range-bound, low conviction, mixed structure, sideways relative to typical volatility.
- TRENDING = clear direction, strong conviction, consistent HH/HL (uptrend) or LH/LL (downtrend), breakouts with momentum.

Rules:
- Use staircase structure to identify trend.
- Ignore small counter-trend candles unless they break prior swing structure.
- Call consolidation only if structure is mixed AND momentum is weak AND price stays within a range.
- Consider news, correlations, and macro context when technicals conflict.

Input Data:
{market_summary}

Additional Context:
{context or 'None'}

Respond ONLY in this JSON format:
{{
  "is_consolidating": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation referencing news, correlations, structure",
  "key_factors": ["factor1", "factor2", ...]
}}
"""
            
            # Query Gemini
            response = self.client.generate_content(prompt)
            self._increment_request_count()
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_str = response_text
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1]
            
            json_str = json_str.strip()
            
            # Parse response
            result = json.loads(json_str)
            
            is_consolidating = bool(result.get('is_consolidating', True))
            reasoning = result.get('reasoning', 'No reasoning provided')
            key_factors = result.get('key_factors', [])
            confidence = result.get('confidence', 0.5)
            
            logger.info(f"Gemini analysis for {symbol}: Consolidating={is_consolidating}, Confidence={confidence:.2f}, Key Factors={key_factors}")
            
            return is_consolidating, reasoning
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis for {symbol}: {e}", exc_info=True)
            return True, f"Error in analysis: {str(e)}"
    
    def is_market_consolidating(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Simplified interface to check if market is consolidating.
        
        Args:
            df: DataFrame with market data
            symbol: Trading symbol
            
        Returns:
            bool: True if market is consolidating, False otherwise
        """
        is_consolidating, _ = self.analyze_consolidation(df, symbol)
        return is_consolidating

    def analyze_multiple_consolidations(self, analyses) -> Dict[str, Tuple[bool, str]]:
        """
        Batch analyze multiple markets for consolidation using Gemini in a SINGLE API call.
        This is optimized to save API requests - all symbols are analyzed together.
        
        Accepts a list of items, where each item can be either:
        - dict with keys: 'df' (pd.DataFrame), 'symbol' (str), optional 'context' (str)
        - tuple in the form: (df, symbol) or (df, symbol, context)
        
        Returns a dict mapping symbol -> (is_consolidating, reasoning).
        Any analysis errors are captured and returned as (True, "Error: <msg>").
        """
        results: Dict[str, Tuple[bool, str]] = {}
        if analyses is None or len(analyses) == 0:
            return results
        
        if self.client is None:
            logger.error("Gemini client not initialized. Cannot analyze consolidation.")
            return {str(i): (True, "Gemini client not initialized") for i in range(len(analyses))}
        
        if self.get_remaining_requests() <= 0:
            logger.error("API quota exhausted. Cannot make batch request.")
            return {str(i): (True, "API quota exhausted") for i in range(len(analyses))}
        
        # Parse all analyses into standardized format
        parsed_analyses: List[Tuple[pd.DataFrame, str, Optional[str]]] = []
        symbol_order: List[str] = []
        
        for item in analyses:
            try:
                df: Optional[pd.DataFrame] = None
                symbol: Optional[str] = None
                context: Optional[str] = None
                
                if isinstance(item, dict):
                    df = item.get('df')
                    symbol = item.get('symbol')
                    context = item.get('context')
                elif isinstance(item, (list, tuple)):
                    if len(item) >= 2:
                        df = item[0]
                        symbol = item[1]
                    if len(item) >= 3:
                        context = item[2]
                else:
                    raise ValueError("Unsupported analysis item type; expected dict or tuple")
                
                if df is None or symbol is None:
                    raise ValueError("Each analysis item must include df and symbol")
                
                parsed_analyses.append((df, str(symbol), context))
                symbol_order.append(str(symbol))
            except Exception as e:
                key = str(symbol) if symbol is not None else f"item_{len(symbol_order)}"
                symbol_order.append(key)
                results[key] = (True, f"Error parsing analysis item: {e}")
        
        if len(parsed_analyses) == 0:
            return results
        
        # Batch process all symbols in a single API call
        try:
            logger.info(f"Batch analyzing {len(parsed_analyses)} symbols in a single API request (saves {len(parsed_analyses)-1} requests)")
            
            batch_summary = self._prepare_batch_summary(parsed_analyses)
            symbols_list = ", ".join([symbol for _, symbol, _ in parsed_analyses])
            
            prompt = f"""You are an expert crypto market analyst. Analyze MULTIPLE markets and determine if each is in CONSOLIDATION or TRENDING.

Evaluate for EACH symbol: NEWS/EVENTS, CORRELATION (BTC/ETH leadership), MARKET STRUCTURE (support/resistance, volume), SENTIMENT.

Definitions:
- CONSOLIDATION = range-bound, low conviction, mixed structure, sideways relative to typical volatility.
- TRENDING = clear direction, strong conviction, consistent HH/HL (uptrend) or LH/LL (downtrend), breakouts with momentum.

Rules:
- Use staircase structure to identify trend.
- Ignore small counter-trend candles unless they break prior swing structure.
- Call consolidation only if structure is mixed AND momentum is weak AND price stays within a range.
- Consider news, correlations, and macro context when technicals conflict.

Input Data (Multiple Symbols):
{batch_summary}

Respond ONLY in this JSON format with results for ALL symbols:
{{
  "results": [
    {{
      "symbol": "SYMBOL1",
      "is_consolidating": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation",
      "key_factors": ["factor1", "factor2"]
    }},
    {{
      "symbol": "SYMBOL2",
      "is_consolidating": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation",
      "key_factors": ["factor1", "factor2"]
    }}
  ]
}}

IMPORTANT: Include results for ALL symbols: {symbols_list}
"""
            
            response = self.client.generate_content(prompt)
            self._increment_request_count()
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_str = response_text
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1]
            
            json_str = json_str.strip()
            
            # Parse batch response
            batch_result = json.loads(json_str)
            batch_results = batch_result.get('results', [])
            
            # Map results back to symbols
            result_map = {r.get('symbol'): r for r in batch_results}
            
            for df, symbol, context in parsed_analyses:
                if symbol in result_map:
                    r = result_map[symbol]
                    is_consolidating = bool(r.get('is_consolidating', True))
                    reasoning = r.get('reasoning', 'No reasoning provided')
                    confidence = r.get('confidence', 0.5)
                    key_factors = r.get('key_factors', [])
                    
                    logger.info(f"Batch analysis for {symbol}: Consolidating={is_consolidating}, Confidence={confidence:.2f}")
                    results[symbol] = (is_consolidating, reasoning)
                else:
                    logger.warning(f"No result found for {symbol} in batch response")
                    results[symbol] = (True, "Symbol not found in batch response")
            
            logger.info(f"Batch analysis completed: {len(results)}/{len(parsed_analyses)} symbols analyzed in 1 API request")
            
        except Exception as e:
            logger.error(f"Error in batch Gemini analysis: {e}", exc_info=True)
            # Fallback: mark all as error
            for df, symbol, context in parsed_analyses:
                if symbol not in results:
                    results[symbol] = (True, f"Error in batch analysis: {str(e)}")
        
        return results


def check_market_consolidation(df: pd.DataFrame, symbol: str, 
                               api_key: Optional[str] = None) -> bool:
    """
    Convenience function to check if a market is consolidating using Gemini AI.
    
    Args:
        df: DataFrame with market data and indicators
        symbol: Trading symbol
        api_key: Optional Gemini API key
        
    Returns:
        bool: True if market is consolidating
    """
    # Use provided key or default hardcoded key
    key = api_key or 'AIzaSyAdyo9u1iFoyoqSXCG3h38ADtuZHmc85vg'
    analyzer = GeminiMarketAnalyzer(api_key=key)
    return analyzer.is_market_consolidating(df, symbol)
