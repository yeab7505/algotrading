"""
AI Market Consolidation Detector
This module integrates with OpenRouter's Tongyi 30B model to analyze market conditions
and determine if the market is consolidating or trending.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import time
import requests
from typing import Dict, Tuple, Optional, List
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Daily request limit (OpenRouter has generous limits)
DAILY_REQUEST_LIMIT = 10000
# Cap for response tokens; configurable via env, clamped to 2k-16k
MAX_RESPONSE_TOKENS = max(2000, min(16000, int(os.getenv("OPENROUTER_MAX_TOKENS", "20000"))))

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available models in order of preference
AVAILABLE_MODELS = [
    'alibaba/tongyi-deepresearch-30b-a3b:free',  # Tongyi 30B model
    'alibaba/tongyi-pro',  # Fallback to Tongyi Pro if available
]

class GeminiMarketAnalyzer:
    """
    Uses OpenRouter's Tongyi 30B model to analyze market data and determine consolidation status.
    Class name kept for backward compatibility with existing code.
    """
    
    def __init__(self, api_key: Optional[str] = None, daily_limit: int = DAILY_REQUEST_LIMIT):
        """
        Initialize the Market Analyzer using OpenRouter.
        
        Args:
            api_key: OpenRouter API key. If None, will try to load from environment.
            daily_limit: Daily request limit (default: 10000 for OpenRouter)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.daily_limit = daily_limit
        self.request_count = 0
        self.last_reset_date = date.today()
        self._reset_if_new_day()
        
        # Track model usage and exhausted models
        self.current_model_index = 0
        self.exhausted_models: Dict[str, date] = {}
        self.model_request_counts: Dict[str, int] = {}
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables.")
            logger.warning("Set it in your .env file or pass it during initialization.")
            self.current_model = None
        else:
            self._initialize_model()
        
        logger.info(f"OpenRouter Market Analyzer initialized. Model: {self.current_model}. Request tracking initialized: {self.get_remaining_requests()} requests remaining today")
    
    def _initialize_model(self):
        """Initialize the first available model."""
        self._reset_exhausted_models()
        for idx, model_name in enumerate(AVAILABLE_MODELS):
            try:
                # Test the model with a simple request
                # Use ample max_tokens to avoid truncation during init
                test_response = self._make_api_request(
                    model=model_name,
                    messages=[{"role": "user", "content": "Say 'OK' if you can process requests."}],
                    max_tokens=200
                )
                # If successful, set as current model
                self.current_model = model_name
                self.current_model_index = idx
                if model_name not in self.model_request_counts:
                    self.model_request_counts[model_name] = 0
                logger.info(f"OpenRouter model initialized successfully: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
                continue
        
        logger.error("Failed to initialize any OpenRouter model")
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
                # Test the model
                # Use ample max_tokens to avoid truncation during switching
                test_response = self._make_api_request(
                    model=model_name,
                    messages=[{"role": "user", "content": "Say 'OK' if you can process requests."}],
                    max_tokens=200
                )
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
    
    def _make_api_request(self, model: str, messages: List[Dict], max_tokens: int = MAX_RESPONSE_TOKENS, temperature: float = 0.3) -> str:
        """
        Make an API request to OpenRouter.
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            
        Returns:
            Response text
            
        Raises:
            Exception: If API request fails or response is invalid
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/your-repo',  # Optional
            'X-Title': 'Trading Bot'  # Optional
        }
        
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Log full response for debugging (first time only)
            if not hasattr(self, '_logged_response_format'):
                logger.debug(f"OpenRouter API response structure: {list(result.keys())}")
                self._logged_response_format = True
            
            # Validate response structure
            if 'choices' not in result or len(result['choices']) == 0:
                error_msg = result.get('error', {}).get('message', 'No choices in response')
                error_data = result.get('error', {})
                logger.error(f"OpenRouter API error: {error_msg}. Error data: {error_data}. Full response keys: {list(result.keys())}")
                raise Exception(f"Invalid API response: {error_msg}")
            
            # Extract content - handle different possible response formats
            choice = result['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            # Some providers return reasoning text separate from content
            if not content and message.get('reasoning'):
                content = message.get('reasoning')
            
            # Check finish_reason to understand why content might be empty
            finish_reason = choice.get('finish_reason', '')
            
            # Alternative: check if content is in choice directly
            if not content and 'text' in choice:
                content = choice['text']
            
            # Alternative: check if content is in delta (streaming responses)
            if not content and 'delta' in choice:
                delta = choice.get('delta', {})
                content = delta.get('content', '')
            
            if not content:
                # Check if it's a length limit issue
                if finish_reason == 'length':
                    usage = result.get('usage', {})
                    completion_tokens = usage.get('completion_tokens', 0)
                    max_tokens_used = usage.get('total_tokens', 0)
                    logger.warning(f"Response truncated due to max_tokens limit. finish_reason='length', completion_tokens={completion_tokens}")
                    raise Exception(f"Response truncated: max_tokens ({max_tokens}) too low. Generated {completion_tokens} tokens before cutoff.")
                else:
                    logger.error(f"Empty content in OpenRouter response.")
                    logger.error(f"finish_reason: {finish_reason}")
                    logger.error(f"Response structure: choices={len(result.get('choices', []))}, first choice keys: {list(choice.keys()) if result.get('choices') else 'no choices'}")
                    logger.error(f"Full response (first 2000 chars): {str(result)[:2000]}")
                    raise Exception(f"Empty content in API response (finish_reason: {finish_reason})")
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"API error details: {error_detail}")
                except:
                    logger.error(f"API error response: {e.response.text}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response structure from OpenRouter: {e}")
            logger.error(f"Response: {result if 'result' in locals() else 'No response'}")
            raise Exception(f"Unexpected response format: {e}")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit (429) error."""
        error_str = str(error)
        if '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower():
            return True
        
        # Check for HTTP status code
        if hasattr(error, 'status_code') and error.status_code == 429:
            return True
        
        # Check for requests library exceptions
        if isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response.status_code == 429:
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
        Prepare a human-readable summary of market data for AI analysis.
        
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
    
    def analyze_multi_timeframe_consolidation(self,
                                              df_ltf: pd.DataFrame,
                                              df_htf: pd.DataFrame,
                                              symbol: str,
                                              trade_side: str) -> Tuple[bool, str]:
        """
        Analyze consolidation across multiple timeframes for a single symbol.
        
        Args:
            df_ltf: Lower timeframe DataFrame (e.g., 15m)
            df_htf: Higher timeframe DataFrame (e.g., 1H)
            symbol: Trading symbol
            trade_side: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_consolidating/unsafe: bool, reasoning: str)
        """
        signals = [{
            'symbol': symbol,
            'trade_side': trade_side,
            'df_ltf': df_ltf,
            'df_htf': df_htf
        }]
        
        results = self.analyze_batch_multi_timeframe_consolidation(signals)
        
        if symbol in results:
            return results[symbol]
        return True, "Analysis failed - no result returned"
    
    def analyze_batch_multi_timeframe_consolidation(self,
                                                   signals: List[Dict],
                                                   max_retries: int = 3) -> Dict[str, Tuple[bool, str]]:
        """
        Analyze consolidation for multiple symbols in a SINGLE API call.
        This is much more efficient than calling analyze_multi_timeframe_consolidation multiple times.
        
        Args:
            signals: List of signal dicts, each containing:
                - 'symbol': Trading symbol
                - 'trade_side': 'BUY' or 'SELL'
                - 'df_ltf': Lower timeframe DataFrame
                - 'df_htf': Higher timeframe DataFrame
            max_retries: Max retry attempts
            
        Returns:
            Dict mapping symbol to (is_unsafe: bool, reasoning: str)
        """
        if not self.api_key or self.current_model is None:
            logger.error("OpenRouter client not initialized. Cannot analyze consolidation.")
            return {s['symbol']: (True, "OpenRouter client not initialized") for s in signals}
        
        if not signals:
            return {}
        
        # Prepare batch prompt with all symbols
        batch_prompts = []
        for signal in signals:
            symbol = signal['symbol']
            trade_side = signal['trade_side']
            df_ltf = signal['df_ltf']
            df_htf = signal['df_htf']
            
            ltf_summary = self._prepare_market_summary(df_ltf, symbol)
            htf_summary = self._prepare_market_summary(df_htf, symbol)
            
            batch_prompts.append(f"""
SYMBOL: {symbol}
PROPOSED TRADE: {trade_side}

DATA 1: LOWER TIMEFRAME (15m) - Entry Timing
{ltf_summary}

DATA 2: HIGHER TIMEFRAME (1H) - Trend Direction
{htf_summary}
""")
        
        combined_prompt = f"""You are an expert crypto market analyst. Analyze the market structure across TWO timeframes for MULTIPLE symbols to determine if each trade is safe.

Analyze {len(signals)} symbols in this batch:

{''.join(batch_prompts)}

Definitions:
- CONSOLIDATION = range-bound, choppy, mixed structure, low momentum.
- TRENDING = clear direction (uptrend/downtrend), consistent structure (HH/HL or LH/LL), strong momentum.

For EACH symbol, synthesize both timeframes to decide if the proposed trade is safe:

1. CHECK ALIGNMENT: Is the 15m signal aligned with the 1H trend?
   - If 1H is Uptrend and trade is SELL -> UNSAFE (Counter-trend).
   - If 1H is Downtrend and trade is BUY -> UNSAFE (Counter-trend).

2. CHECK CONSOLIDATION:
   - If 1H is consolidating/choppy -> UNSAFE.
   - If 1H is trending strongly but 15m is consolidating (flag) -> WAIT for breakout (UNSAFE until clear).
   - If both are choppy -> UNSAFE.

BUT IF THERE IS VERY STRONG TREND REVERSAL IN THE 15M TIMEFRAME, THEN IT IS SAFE TO TRADE.
When UNSAFE, provide a concise, specific reason (2-4 sentences) citing alignment, consolidation, and volatility cues.

One thing to consider is the fact that the data provided is fully close candle so in 1 hour data you might not see the candles and the volume spike you see in the 15min data if the time is between hours like HH:15, HH:30, HH:45.

What is trend reversal and its characteristics?
- Trend reversal is a change in the direction of the trend.

- A. Break of Market Structure (BMS)
    In an uptrend, price forms higher highs (HH) and higher lows (HL).
    If price breaks below the last higher low, the structure flips -> Downtrend begins.

- B. Momentum Shift (Candle Strength)
    Look for:
    - Large opposite-direction candles
    - Strong volume spike during opposite candles
    - Weak continuation candles during the old trend

- C. Supply & Demand Zones
    Reversal probability is high if:
    - Uptrend hits a strong supply zone -> reversal down
    - Downtrend hits a strong demand zone -> reversal up

- D. Volume Confirmation
    Reversal = volume shift:
    - Rising volume in the opposite direction
    - Falling volume in the previous direction

Signs of pullback (NOT reversal):
- A. Pullback Has Weak Momentum
    - Signs:
        - Small candles
        - Low volume
        - Slow movement
        - Wicky candles
        - Reversal = strong opposite momentum
        - Pullback = weak opposite momentum

- B. Price Does NOT Break Structure
    - In an uptrend:
        - Pullback will not break the previous HL
        - Instead, it forms a new HL before continuing up
    - In a downtrend:
        - Pullback stays below previous LH
        - If structure is not broken -> pullback, not reversal.

- E. Pullback Ends at a Trendline
    If trendline is respected -> pullback
    If trendline breaks -> early sign of reversal

For EACH symbol, respond ONLY in this JSON format. Include ALL symbols in the response:
{{
  "{signals[0]['symbol']}": {{
    "is_unsafe": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Concise explanation. Mention trend alignment (e.g., '1H Uptrend supports BUY' or '1H Downtrend contradicts BUY').",
    "key_factors": ["factor1", "factor2"]
  }}{','.join([f',\n  "{s["symbol"]}": {{\n    "is_unsafe": true/false,\n    "confidence": 0.0-1.0,\n    "reasoning": "Concise explanation. Mention trend alignment (e.g., \'1H Uptrend supports BUY\' or \'1H Downtrend contradicts BUY\').",\n    "key_factors": ["factor1", "factor2"]\n  }}' for s in signals[1:]])}
}}
"""
        
        # Retry logic with model switching
        for attempt in range(max_retries + 1):
            if self.current_model is None:
                return {s['symbol']: (True, "No available models") for s in signals}
            
            current_model = self.current_model
            model_count = self.model_request_counts.get(current_model, 0) if current_model else 0
            
            if current_model and model_count >= self.daily_limit:
                logger.warning(f"Model {current_model} has reached daily limit. Switching...")
                if not self._switch_to_next_model():
                    return {s['symbol']: (True, "All models exhausted") for s in signals}
                continue
            
            try:
                # Use OpenRouter API format
                # Calculate max_tokens based on number of symbols (more symbols = more tokens needed)
                # Base: 1000 tokens per symbol for JSON response, min 4000, capped by MAX_RESPONSE_TOKENS
                estimated_tokens = max(4000, min(MAX_RESPONSE_TOKENS, len(signals) * 1000))
                
                response_text = self._make_api_request(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": "You are an expert crypto market analyst. Analyze market structure and provide JSON responses. Always respond with valid JSON only, no markdown formatting. Do not include code fences. Output must be a single JSON object exactly matching the requested schema. If you cannot comply, return a JSON object with is_unsafe=true and a concise reasoning."},
                        {"role": "user", "content": combined_prompt}
                    ],
                    max_tokens=estimated_tokens,
                    temperature=0.3
                )
                self._increment_request_count(current_model)
                
                if not response_text:
                    logger.error("Received empty response from OpenRouter API")
                    raise ValueError("Empty response from API")
                
                response_text = response_text.strip()
                logger.debug(f"Raw API response (first 500 chars): {response_text[:500]}")
                
                # Extract JSON - try multiple methods
                json_str = response_text
                
                # Method 1: Check for markdown code blocks
                if '```json' in json_str:
                    parts = json_str.split('```json')
                    if len(parts) > 1:
                        json_str = parts[1].split('```')[0].strip()
                elif '```' in json_str:
                    parts = json_str.split('```')
                    if len(parts) > 1:
                        # Take the content between first set of backticks
                        json_str = parts[1].strip()
                
                # Method 2: Try to find JSON object boundaries
                if not json_str or json_str[0] != '{':
                    first_brace = json_str.find('{')
                    last_brace = json_str.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = json_str[first_brace:last_brace + 1]
                
                json_str = json_str.strip()
                
                # If still not JSON-like, fallback: block trades with reasoning from text
                if not json_str or not json_str.startswith('{'):
                    logger.error("Model returned non-JSON response after extraction; blocking for safety.")
                    brief = (response_text[:180] + '...') if len(response_text) > 200 else response_text
                    return {s['symbol']: (True, f"Model returned non-JSON response: {brief}") for s in signals}
                
                # Try to parse JSON
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
                    logger.error(f"Attempted to parse (first 1000 chars): {json_str[:1000]}")
                    logger.error(f"Full response text (first 2000 chars): {response_text[:2000]}")
                    
                    # Try to fix common JSON issues
                    import re
                    json_str_fixed = re.sub(r',\s*}', '}', json_str)
                    json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
                    
                    try:
                        result = json.loads(json_str_fixed)
                        logger.info("Successfully parsed JSON after fixing trailing commas")
                    except json.JSONDecodeError:
                        # If still invalid, return a safe fallback marking all symbols unsafe
                        logger.error("Model response is not valid JSON after fixes; blocking trades for safety.")
                        return {s['symbol']: (True, f"Model returned invalid JSON; blocked trade. Error: {e.msg} at {e.pos}") for s in signals}
                
                # Parse results for each symbol
                results = {}
                for signal in signals:
                    symbol = signal['symbol']
                    if symbol in result:
                        symbol_result = result[symbol]
                        is_unsafe = bool(symbol_result.get('is_unsafe', True))
                        reasoning = symbol_result.get('reasoning', 'No reasoning provided')
                        results[symbol] = (is_unsafe, reasoning)
                    else:
                        # Fallback if symbol not in response
                        results[symbol] = (True, "Symbol not found in batch response")
                
                logger.info(f"Batch Multi-TF Analysis ({current_model}): Processed {len(signals)} symbols")
                return results
                
            except Exception as e:
                error_str = str(e)
                # Check if it's a max_tokens truncation issue
                if 'truncated' in error_str.lower() or 'max_tokens' in error_str.lower() or 'finish_reason' in error_str.lower():
                    logger.warning(f"Response truncated for {current_model}: {e}")
                    # Retry with increased max_tokens if we haven't exhausted retries
                    if attempt < max_retries:
                        # Double the token limit for next attempt
                        estimated_tokens = min(MAX_RESPONSE_TOKENS, estimated_tokens * 2)
                        logger.info(f"Retrying with increased max_tokens: {estimated_tokens}")
                        continue
                    else:
                        logger.error(f"Failed after retrying with increased tokens: {e}")
                        return {s['symbol']: (True, f"Response truncated: {e}") for s in signals}
                elif self._is_rate_limit_error(e):
                    logger.warning(f"Rate limit error for {current_model}: {e}")
                    if current_model:
                        self.exhausted_models[current_model] = date.today()
                    if attempt < max_retries:
                        if self._switch_to_next_model():
                            continue
                    return {s['symbol']: (True, f"Rate limit exceeded: {e}") for s in signals}
                else:
                    logger.error(f"Error in Batch Multi-TF analysis: {e}")
                    return {s['symbol']: (True, f"Error in analysis: {str(e)}") for s in signals}
        
        return {s['symbol']: (True, "Failed after all retry attempts") for s in signals}
