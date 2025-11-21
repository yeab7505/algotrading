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
import time
from typing import Dict, Tuple, Optional, List
from datetime import date
import google.generativeai as genai
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Daily request limit (free tier: 50 per model per day)
DAILY_REQUEST_LIMIT = 200

# Available models in order of preference (Flash models are faster/cheaper)
# Free tier: 50 requests per day per model
AVAILABLE_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro', # Fastest, cheapest
    'gemini-2.0-flash-lite',  # Experimental flash
    'gemini-2.5-flash-lite',        # More capable
    'gemini-2.0-flash',      # Latest flash
           # Latest pro (most expensive)
]

class GeminiMarketAnalyzer:
    """
    Uses Google's Gemini AI to analyze market data and determine consolidation status.
    """
    
    def __init__(self, api_key: Optional[str] = None, daily_limit: int = DAILY_REQUEST_LIMIT):
        """
        Initialize the Gemini Market Analyzer.
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from environment.
            daily_limit: Daily request limit per model (default: 50 for free tier)
        """
        self.api_key = 'AIzaSyBbWrXelA5WLJiedbqjp7j5M9QzmwA6tPk'
        self.daily_limit = daily_limit
        self.request_count = 0
        self.last_reset_date = date.today()
        self._reset_if_new_day()
        
        # Track model usage and exhausted models
        self.current_model_index = 0
        self.exhausted_models: Dict[str, date] = {}
        self.model_request_counts: Dict[str, int] = {}
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
            logger.warning("Set it in your .env file or pass it during initialization.")
            self.client = None
            self.current_model = None
        else:
            genai.configure(api_key=self.api_key)
            self._initialize_model()
        
        logger.info(f"Request tracking initialized: {self.get_remaining_requests()} requests remaining today")
    
    def _initialize_model(self):
        """Initialize the first available model."""
        self._reset_exhausted_models()
        for idx, model_name in enumerate(AVAILABLE_MODELS):
            try:
                self.client = genai.GenerativeModel(model_name)
                self.current_model = model_name
                self.current_model_index = idx
                if model_name not in self.model_request_counts:
                    self.model_request_counts[model_name] = 0
                logger.info(f"Gemini AI initialized successfully with {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
                continue
        
        logger.error("Failed to initialize any Gemini model")
        self.client = None
        self.current_model = None
    
    def _reset_exhausted_models(self):
        """Reset exhausted models if it's a new day."""
        today = date.today()
        models_to_remove = []
        for model_name, exhausted_date in self.exhausted_models.items():
            if exhausted_date < today:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            del self.exhausted_models[model_name]
            if model_name in self.model_request_counts:
                self.model_request_counts[model_name] = 0
            logger.info(f"Reset exhausted model: {model_name}")
    
    def _switch_to_next_model(self) -> bool:
        """Switch to the next available model. Returns True if successful."""
        self._reset_exhausted_models()
        
        # Try next models in order
        for idx in range(self.current_model_index + 1, len(AVAILABLE_MODELS)):
            model_name = AVAILABLE_MODELS[idx]
            
            # Skip if model is exhausted today
            if model_name in self.exhausted_models:
                continue
            
            try:
                self.client = genai.GenerativeModel(model_name)
                self.current_model = model_name
                self.current_model_index = idx
                if model_name not in self.model_request_counts:
                    self.model_request_counts[model_name] = 0
                logger.info(f"Switched to model: {model_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to switch to {model_name}: {e}")
                continue
        
        logger.error("No available models remaining")
        return False
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit (429) error."""
        error_str = str(error)
        if '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower():
            return True
        
        # Check for Google API rate limit exceptions
        if hasattr(error, 'status_code') and error.status_code == 429:
            return True
        
        if google_exceptions and isinstance(error, google_exceptions.ResourceExhausted):
            return True
        
        return False
    
    def _extract_retry_delay(self, error: Exception) -> float:
        """Extract retry delay from error message. Returns delay in seconds."""
        error_str = str(error)
        try:
            # Try to extract retry delay from error message
            if 'retry_delay' in error_str or 'retry in' in error_str.lower():
                import re
                # Look for patterns like "retry in 39.11264135s" or "seconds: 39"
                match = re.search(r'(\d+\.?\d*)\s*(?:seconds?|s)', error_str, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception:
            pass
        
        # Default exponential backoff: start with 5 seconds
        return 5.0
    
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
    
    def _increment_request_count(self, model_name: Optional[str] = None):
        """Increment request count and log remaining requests."""
        self._reset_if_new_day()
        self.request_count += 1
        
        # Track per-model counts
        if model_name:
            if model_name not in self.model_request_counts:
                self.model_request_counts[model_name] = 0
            self.model_request_counts[model_name] += 1
            
            remaining = self.daily_limit - self.model_request_counts[model_name]
            logger.info(f"API Request for {model_name}: #{self.model_request_counts[model_name]}/{self.daily_limit} used. {remaining} requests remaining today")
            
            if remaining <= 10:
                logger.warning(f"Low API quota for {model_name}: Only {remaining} requests remaining today!")
            elif remaining <= 0:
                logger.warning(f"API quota exhausted for {model_name}! Marking as exhausted.")
                self.exhausted_models[model_name] = date.today()
        else:
            remaining = self.get_remaining_requests()
            logger.info(f"API Request #{self.request_count}/{self.daily_limit} used. {remaining} requests remaining today")
    
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

    def analyze_multi_timeframe_consolidation(self, 
                                             df_ltf: pd.DataFrame, 
                                             df_htf: pd.DataFrame, 
                                             symbol: str, 
                                             max_retries: int = 3) -> Tuple[bool, str]:
        """
        Analyze consolidation across both Lower Timeframe (LTF) and Higher Timeframe (HTF) in a SINGLE request.
        
        Args:
            df_ltf: Lower timeframe DataFrame (e.g., 15m)
            df_htf: Higher timeframe DataFrame (e.g., 4H)
            symbol: Trading symbol
            max_retries: Max retry attempts
            
        Returns:
            Tuple of (is_consolidating: bool, reasoning: str)
            Returns True (is consolidating) only if BOTH timeframes show issues or overall market is unsafe.
            Returns False (is trending) if the market is suitable for trading.
        """
        if self.client is None:
            logger.error("Gemini client not initialized. Cannot analyze consolidation.")
            return False, "Gemini client not initialized. Check API key."
        
        # Prepare market summaries
        ltf_summary = self._prepare_market_summary(df_ltf, symbol)
        htf_summary = self._prepare_market_summary(df_htf, symbol)
        
        prompt = f"""You are an expert crypto market analyst. Analyze the market structure across TWO timeframes to determine if the asset is TRENDING or CONSOLIDATING.

SYMBOL: {symbol}

DATA 1: LOWER TIMEFRAME (15m) - Entry Timing
{ltf_summary}

DATA 2: HIGHER TIMEFRAME (4H) - Trend Direction
{htf_summary}

Definitions:
- CONSOLIDATION = range-bound, choppy, mixed structure, low momentum.
- TRENDING = clear direction (uptrend/downtrend), consistent structure (HH/HL or LH/LL), strong momentum.

Task:
Synthesize both timeframes to decide if a trade is safe. 
- If HTF is trending strongly but LTF is consolidating (bull flag), this might be safe (Trending) but still needs to be confirmed by the LTF, allow the trade if you are very very confident in the HFT.
- If HTF is consolidating/choppy, the market is unsafe regardless of LTF (Consolidating).
- If both are choppy, it is definitely Consolidating.

Respond ONLY in this JSON format:
{{
  "is_consolidating": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Concise explanation synthesizing both timeframes (e.g., 'HTF uptrend strong, LTF breakout confirmed')",
  "key_factors": ["factor1", "factor2"]
}}
"""
        
        # Retry logic with model switching (same as single analysis)
        for attempt in range(max_retries + 1):
            if self.client is None:
                return True, "No available models."
            
            current_model = self.current_model
            model_count = self.model_request_counts.get(current_model, 0) if current_model else 0
            
            if current_model and model_count >= self.daily_limit:
                logger.warning(f"Model {current_model} has reached daily limit. Switching...")
                if not self._switch_to_next_model():
                    return True, "All models have reached daily quota limits."
                continue
            
            try:
                response = self.client.generate_content(prompt)
                self._increment_request_count(current_model)
                response_text = response.text.strip()
                
                # Extract JSON
                json_str = response_text
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1]
                json_str = json_str.strip()
                
                result = json.loads(json_str)
                is_consolidating = bool(result.get('is_consolidating', True))
                reasoning = result.get('reasoning', 'No reasoning provided')
                confidence = result.get('confidence', 0.5)
                
                logger.info(f"Multi-TF Analysis for {symbol} ({current_model}): Consolidating={is_consolidating}, Confidence={confidence:.2f}")
                return is_consolidating, reasoning
                
            except Exception as e:
                if self._is_rate_limit_error(e):
                    logger.warning(f"Rate limit error for {current_model}: {e}")
                    if current_model: self.exhausted_models[current_model] = date.today()
                    if attempt < max_retries:
                         if self._switch_to_next_model(): continue
                    return True, f"Rate limit exceeded: {e}"
                else:
                    logger.error(f"Error in Multi-TF analysis: {e}")
                    return True, f"Error in analysis: {str(e)}"
        
        return True, "Failed after all retry attempts"

    def analyze_consolidation(self, df: pd.DataFrame, symbol: str, 
                             context: Optional[str] = None, max_retries: int = 3) -> Tuple[bool, str]:
        """
        Use Gemini AI to determine if the market is consolidating.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Trading symbol
            context: Additional context or constraints
            max_retries: Maximum number of retries with different models
            
        Returns:
            Tuple of (is_consolidating: bool, reasoning: str)
        """
        if self.client is None:
            logger.error("Gemini client not initialized. Cannot analyze consolidation.")
            return False, "Gemini client not initialized. Check API key."
        
        # Prepare market summary once
        market_summary = self._prepare_market_summary(df, symbol)
        
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
        
        # Retry logic with model switching
        for attempt in range(max_retries + 1):
            if self.client is None:
                logger.error("No available models. Cannot analyze consolidation.")
                return True, "No available models. All models exhausted."
            
            current_model = self.current_model
            model_count = self.model_request_counts.get(current_model, 0) if current_model else 0
            
            if current_model and model_count >= self.daily_limit:
                logger.warning(f"Model {current_model} has reached daily limit. Switching...")
                if not self._switch_to_next_model():
                    return True, "All models have reached daily quota limits."
                continue
            
            try:
                # Query Gemini
                response = self.client.generate_content(prompt)
                self._increment_request_count(current_model)
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
                
                logger.info(f"Gemini analysis for {symbol} ({current_model}): Consolidating={is_consolidating}, Confidence={confidence:.2f}, Key Factors={key_factors}")
                
                return is_consolidating, reasoning
                
            except Exception as e:
                if self._is_rate_limit_error(e):
                    logger.warning(f"Rate limit error for {current_model}: {e}")
                    
                    # Mark current model as exhausted
                    if current_model:
                        self.exhausted_models[current_model] = date.today()
                        logger.warning(f"Marked {current_model} as exhausted due to rate limit")
                    
                    # Try to switch to next model
                    if attempt < max_retries:
                        retry_delay = self._extract_retry_delay(e)
                        logger.info(f"Waiting {retry_delay:.1f}s before switching models...")
                        time.sleep(min(retry_delay, 10.0))  # Cap delay at 10 seconds
                        
                        if self._switch_to_next_model():
                            logger.info(f"Switched to {self.current_model} for retry attempt {attempt + 1}")
                            continue
                        else:
                            logger.error("No more models available after rate limit error")
                            return True, f"Rate limit exceeded. All models exhausted: {str(e)}"
                    else:
                        logger.error(f"Max retries reached. Rate limit error: {e}")
                        return True, f"Rate limit exceeded after {max_retries} retries: {str(e)}"
                else:
                    # Non-rate-limit error
                    logger.error(f"Error in Gemini analysis for {symbol}: {e}", exc_info=True)
                    return True, f"Error in analysis: {str(e)}"
        
        return True, "Failed after all retry attempts"


def check_market_consolidation(df: pd.DataFrame, symbol: str, 
                               api_key: Optional[str] = None) -> bool:
    """
    Convenience function to check if a market is consolidating using Gemini AI.
    
    Args:
        df: DataFrame with market data and indicators
        symbol: Trading symbol
        api_key: Optional Gemini API key (if None, uses default)
        
    Returns:
        bool: True if market is consolidating
    """
    analyzer = GeminiMarketAnalyzer(api_key=api_key)
    return analyzer.is_market_consolidating(df, symbol)
