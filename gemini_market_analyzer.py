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
from typing import Dict, Tuple, Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class GeminiMarketAnalyzer:
    """
    Uses Google's Gemini AI to analyze market data and determine consolidation status.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Market Analyzer.
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from environment.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
            logger.warning("Set it in your .env file or pass it during initialization.")
            self.client = None
        else:
            genai.configure(api_key=self.api_key)
            # Use the latest Gemini 2.5 models (Flash is faster and cheaper)
            try:
                self.client = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini AI initialized successfully with gemini-2.5-flash")
            except Exception as e:
                logger.warning(f"Failed to initialize gemini-2.5-flash: {e}, trying gemini-2.5-pro...")
                try:
                    self.client = genai.GenerativeModel('gemini-2.5-pro')
                    logger.info("Gemini AI initialized successfully with gemini-2.5-flash")
                except Exception as e2:
                    logger.warning(f"Failed to initialize gemini-2.5-pro: {e2}, trying gemini-1.5-flash")
                    try:
                        self.client = genai.GenerativeModel('gemini-1.5-pro')
                        logger.info("Gemini AI initialized successfully with gemini-1.5-pro")
                    except Exception as e3:
                        self.client = genai.GenerativeModel('gemini-1.5-pro')
                        logger.info("Gemini AI initialized successfully with gemini-1.5-pro")
    
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

Recent Price Action (Last 5 candles):
"""
        
        for i in range(-5, 0):
            row = df.iloc[i]
            summary += f"Candle {i+6}: O:{row['Open']:.4f} H:{row['High']:.4f} L:{row['Low']:.4f} C:{row['Close']:.4f} V:{row['Volume']:.0f}\n"
        
        return summary
    
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
        
        try:
            # Prepare market summary
            market_summary = self._prepare_market_summary(df, symbol)
            
            # Create prompt for Gemini
            prompt = f"""You are an expert cryptocurrency market analyst specializing in identifying market consolidation patterns.

Your task is to analyze market data and determine if the market is currently in a CONSOLIDATING/CHOPPY state or in a TRENDING state.

IMPORTANT: Consider ALL relevant factors, not just technical indicators:

1. MARKET NEWS & EVENTS:
   - Recent major announcements (partnerships, listings, regulations)
   - Macro-economic events affecting crypto markets
   - Adoption news or institutional developments
   - Regulatory news or government policies
   - Exchange or protocol updates

2. MAJOR CRYPTOCURRENCY CORRELATIONS:
   - Bitcoin (BTC) movements (market leader, often sets trend)
   - Ethereum (ETH) correlation (if it's altcoin)
   - Overall crypto market sentiment
   - Whether major cryptos are trending or consolidating
   - Market-wide fear and greed index

3. MARKET STRUCTURE & VOLUME:
   - Support and resistance levels being tested
   - Price rejection patterns at key levels
   - Volume patterns (increasing/decreasing/neutral)
   - Market depth and liquidity
   - Order book imbalances

4. MARKET SENTIMENT:
   - Social media sentiment
   - Trading volume patterns
   - Whether price action aligns with fundamentals
   - Overall crypto market momentum

CONSOLIDATION characteristics:
- Price oscillating in a bounded range
- Lack of directional conviction
- Low volatility and sideways movement
- Market uncertain about direction
- Waiting for catalyst (news, event, Bitcoin move)

TRENDING characteristics:
- Clear directional bias (up or down)
- Consistent price movement with momentum
- Strong market conviction
- Breaking out of ranges or following trend
- Correlation with major market leaders (BTC/ETH)

Market Data Provided:
{market_summary}

Additional Context:
{context if context else 'Analyze current market conditions holistically'}

Based on ALL the above factors, determine if the market is CONSOLIDATING or TRENDING.
Consider news cycles, market events, major crypto correlations, and overall market sentiment - not just the technical indicators.

Respond in the following JSON format:
{{
    "is_consolidating": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation considering news, correlations, and market factors",
    "key_factors": ["factor1", "factor2", ...]
}}

Only respond with the JSON object, no additional text."""
            
            # Query Gemini
            response = self.client.generate_content(prompt)
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
            
            is_consolidating = bool(result.get('is_consolidating', False))
            reasoning = result.get('reasoning', 'No reasoning provided')
            confidence = result.get('confidence', 0.5)
            
            logger.info(f"Gemini analysis for {symbol}: Consolidating={is_consolidating}, Confidence={confidence:.2f}")
            
            return is_consolidating, reasoning
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis for {symbol}: {e}", exc_info=True)
            return False, f"Error in analysis: {str(e)}"
    
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

