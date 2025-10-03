# smart_money_integration.py
"""
Integration module to connect Smart Money Radar with the main FOREX AI Backend
"""

from forex_smart_money_radar import ForexSmartMoneyRadar
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

class SmartMoneyIntegration:
    """
    Integrates the Smart Money Radar with the main trading system
    """
    
    def __init__(self):
        self.radar = ForexSmartMoneyRadar()
        self.last_signals = {}
        self.arbitrage_history = []
        
    async def get_smart_money_analysis(self) -> Dict:
        """
        Get current smart money analysis
        """
        # Get mock prices (in production, this would come from real data)
        self.radar.get_mock_prices()
        
        # Check for arbitrage
        arbitrage_opportunities = self.radar.check_triangular_arbitrage()
        
        # Get smart money flow
        smart_flow = self.radar.detect_smart_money_flow()
        
        # Calculate currency strength
        currency_strength = self.radar.calculate_currency_strength()
        
        # Prepare response
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "arbitrage_opportunities": arbitrage_opportunities,
            "smart_money_flow": {
                "strongest_currencies": smart_flow['strongest'],
                "weakest_currencies": smart_flow['weakest'],
                "recommended_pair": smart_flow['recommendation']
            },
            "currency_strength": currency_strength,
            "key_prices": {
                pair: price for pair, price in self.radar.prices.items() 
                if pair in ['EUR/USD', 'GBP/USD', 'USD/JPY', 'EUR/GBP', 'EUR/JPY']
            }
        }
        
        # Store in history
        if arbitrage_opportunities:
            self.arbitrage_history.extend(arbitrage_opportunities)
            # Keep only last 100 entries
            self.arbitrage_history = self.arbitrage_history[-100:]
        
        return analysis
    
    async def get_trading_recommendation(self, pair: str = "EUR/USD") -> Dict:
        """
        Get trading recommendation based on smart money analysis
        """
        analysis = await self.get_smart_money_analysis()
        
        # Check if there's manipulation detected
        manipulation_detected = False
        for opp in analysis['arbitrage_opportunities']:
            if opp['type'] == 'MANIPULATION':
                manipulation_detected = True
                break
        
        # Get currency strength for the pair
        base, quote = pair.split('/')
        base_strength = analysis['currency_strength'].get(base, 100)
        quote_strength = analysis['currency_strength'].get(quote, 100)
        
        # Determine signal
        if manipulation_detected:
            signal = "HOLD"
            confidence = 30
            reason = "Market manipulation detected - avoid trading"
        elif base_strength > quote_strength * 1.02:  # Base is 2% stronger
            signal = "BUY"
            confidence = min(90, (base_strength / quote_strength - 1) * 100)
            reason = f"{base} is stronger than {quote}"
        elif quote_strength > base_strength * 1.02:  # Quote is 2% stronger
            signal = "SELL"
            confidence = min(90, (quote_strength / base_strength - 1) * 100)
            reason = f"{quote} is stronger than {base}"
        else:
            signal = "HOLD"
            confidence = 50
            reason = "No clear direction"
        
        return {
            "pair": pair,
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "currency_strength": {
                base: base_strength,
                quote: quote_strength
            },
            "smart_money_recommendation": analysis['smart_money_flow']['recommended_pair']
        }
    
    def get_arbitrage_history(self) -> List[Dict]:
        """
        Get recent arbitrage opportunities
        """
        return self.arbitrage_history

# FastAPI endpoints to add to forex_backend.py
"""
Add these endpoints to your forex_backend.py file:

@app.get("/api/smart-money/analysis")
async def get_smart_money_analysis():
    integration = SmartMoneyIntegration()
    return await integration.get_smart_money_analysis()

@app.get("/api/smart-money/recommendation/{pair}")
async def get_trading_recommendation(pair: str = "EUR/USD"):
    integration = SmartMoneyIntegration()
    pair_formatted = pair.replace("-", "/")
    return await integration.get_trading_recommendation(pair_formatted)

@app.get("/api/smart-money/arbitrage-history")
async def get_arbitrage_history():
    integration = SmartMoneyIntegration()
    return integration.get_arbitrage_history()
"""

# Test the integration
if __name__ == "__main__":
    async def test():
        print("Testing Smart Money Integration...")
        integration = SmartMoneyIntegration()
        
        # Test analysis
        print("\n1. Getting Smart Money Analysis...")
        analysis = await integration.get_smart_money_analysis()
        print(f"   Strongest: {analysis['smart_money_flow']['strongest_currencies'][:2]}")
        print(f"   Weakest: {analysis['smart_money_flow']['weakest_currencies'][:2]}")
        print(f"   Arbitrage: {len(analysis['arbitrage_opportunities'])} opportunities found")
        
        # Test recommendations
        print("\n2. Getting Trading Recommendations...")
        for pair in ["EUR/USD", "GBP/USD", "EUR/JPY"]:
            rec = await integration.get_trading_recommendation(pair)
            print(f"   {pair}: {rec['signal']} (confidence: {rec['confidence']}%)")
            print(f"           Reason: {rec['reason']}")
        
        print("\n✅ Integration test completed!")
    
    # Run the test
    asyncio.run(test())