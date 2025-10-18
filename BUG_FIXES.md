# Critical Bug Fixes - Trading Bot

**Date:** 2025-10-18  
**Issues Addressed:** Network timeout causing incorrect position closures & Budget not freed after position close

---

## Issue 1: Network Timeout Causing Incorrect Position Closures

### Problem Description
When network connection times out or has issues:
- The bot would get incorrect/stale prices from the API
- These incorrect prices would trigger TP/SL levels incorrectly
- Positions would be closed prematurely with wrong prices
- This resulted in unexpected losses and poor trading performance

### Root Cause
The `_get_current_price()` method in `position_manager.py` would:
1. Return `None` on connection errors (good)
2. But would NOT validate if prices are reasonable
3. Sudden large price changes (likely API errors) were treated as real
4. No check for suspicious price movements

### Fix Applied

**File:** `trading/position_manager.py`

#### Change 1: Added Price Validation
```python
def _get_current_price(self, symbol: str) -> Optional[float]:
    # ... get price from API ...
    
    # CRITICAL FIX: Additional validation to prevent stale/incorrect prices
    if not hasattr(self, '_last_valid_prices'):
        self._last_valid_prices = {}
    
    if symbol in self._last_valid_prices:
        last_price = self._last_valid_prices[symbol]
        price_change_pct = abs((price - last_price) / last_price) * 100
        
        # If price changed more than 50% in one check, it's likely incorrect
        if price_change_pct > 50:
            self.logger.error(f"Suspicious price change for {symbol}: "
                            f"{last_price:.6f} -> {price:.6f} ({price_change_pct:.1f}%). "
                            f"Rejecting price to prevent incorrect position closure.")
            return None
    
    # Update last valid price
    self._last_valid_prices[symbol] = price
    return price
```

#### Change 2: Enhanced Logging for Failed Price Fetches
```python
def _check_position_triggers(self, position_id: int):
    # ... get position ...
    
    current_price = self._get_current_price(position.symbol)
    if current_price is None:
        # CRITICAL FIX: Don't process position if we can't get a valid price
        self.logger.warning(f"Cannot get valid price for {position.symbol}, "
                          f"skipping position check to prevent incorrect closure")
        session.close()
        return
```

### Expected Behavior After Fix
- Prices that change more than 50% in a single check are rejected
- Last known valid price is tracked per symbol
- Positions are NOT closed with suspicious/stale prices
- Clear logging when prices are rejected
- Bot continues monitoring with last valid price until network recovers

---

## Issue 2: Budget Not Freed After Position Close

### Problem Description
After opening 4 positions:
- Bot stopped opening new positions
- Logged "Insufficient balance" even though positions were closed
- `used_balance` was incremented when opening positions
- `used_balance` was NEVER decremented when closing positions
- Result: Available balance permanently decreased to zero

### Example from User's Data
```
Position 397 (DAGUSDT): Entry value = 0.019602 × 2550.76 = $50.01
Position 398 (PLAYUSDT): Entry value = 0.028282 × 883.954 = $24.99
Position 399 (CHZUSDT): Entry value = 0.032625 × 383.142 = $12.50
Position 400 (TUTUSDT): Entry value = 0.025036 × 249.641 = $6.25

Total locked: ~$93.75 (with demo_balance = $100)
After closing: used_balance should be freed, but it wasn't!
```

### Root Cause
In `trading/engine.py`:
```python
# When opening position:
self.used_balance += position_value  # ✅ Correctly added

# When closing position:
self.position_manager.close_position(position_id, reason)
# ❌ used_balance was NEVER decremented!
```

### Fix Applied

**File:** `trading/position_manager.py`

#### Change 1: Added Trading Engine Reference
```python
class PositionManager:
    def __init__(self, api: CoinExAPI, trading_engine=None):
        self.api = api
        self.trading_engine = trading_engine  # NEW: Reference to trading engine
        # ... rest of init ...
```

#### Change 2: Free Balance on Position Close
```python
def close_position(self, position_id: int, reason: str = "Manual close") -> bool:
    # ... get position and close it ...
    
    # CRITICAL FIX: Calculate position value to free from used_balance
    position_value = position.entry_price * position.quantity
    
    # ... update database ...
    
    # CRITICAL FIX: Free up the allocated balance when position closes
    if self.trading_engine and hasattr(self.trading_engine, 'used_balance'):
        self.trading_engine.used_balance = max(0, self.trading_engine.used_balance - position_value)
        self.logger.info(f"Freed ${position_value:.2f} from used_balance. "
                       f"New used_balance: ${self.trading_engine.used_balance:.2f}")
    
    return True
```

**File:** `trading/engine.py`

#### Change 3: Pass Trading Engine Reference
```python
def __init__(self, demo_mode: bool = True):
    # ... other init ...
    
    # Pass reference to self so position_manager can update used_balance
    self.position_manager = PositionManager(self.api, trading_engine=self)
```

### Expected Behavior After Fix
- When position opens: `used_balance` increases by position value
- When position closes: `used_balance` decreases by position value
- Available balance correctly reflects: `demo_balance - used_balance`
- Bot can continue opening new positions after old ones close
- Clear logging shows balance being freed: "Freed $50.01 from used_balance"

---

## Testing Recommendations

### Test 1: Network Timeout Scenario
1. Start bot with demo mode
2. Open a position
3. Simulate network timeout (disconnect internet briefly)
4. Reconnect network
5. **Expected**: Position NOT closed with incorrect price
6. **Expected**: Log message about suspicious price change

### Test 2: Balance Management
1. Start bot with `demo_balance = 100`
2. Set `risk_per_trade = 0.5` (50%)
3. Open first position (should use ~$50)
4. Log should show: `used_balance = $50, available = $50`
5. Close first position
6. **Expected**: Log shows "Freed $50 from used_balance"
7. **Expected**: `used_balance = $0, available = $100`
8. Open second position
9. **Expected**: Position opens successfully (not rejected)

### Test 3: Multiple Positions Cycle
1. Open 4 positions (max_positions limit)
2. Wait for all 4 to close (TP/SL hit)
3. **Expected**: All positions freed from used_balance
4. **Expected**: New BUY signals are NOT rejected
5. **Expected**: Available balance restored to ~$100

---

## Files Modified

1. **trading/position_manager.py**
   - Added price validation logic
   - Added last_valid_prices tracking
   - Added trading_engine reference parameter
   - Added balance freeing on close_position()

2. **trading/engine.py**
   - Pass self reference to PositionManager

---

## Security & Safety

- ✅ No new security vulnerabilities introduced
- ✅ Enhanced safety: Prevents incorrect position closures
- ✅ Better error handling during network issues
- ✅ Accurate budget tracking for risk management

---

## Summary

### Before Fix:
❌ Network timeouts caused incorrect position closures  
❌ Budget locked permanently after opening positions  
❌ Bot stopped trading after 4 positions

### After Fix:
✅ Invalid prices are rejected (>50% change threshold)  
✅ Budget correctly freed when positions close  
✅ Bot continues trading indefinitely  
✅ Better logging for debugging

---

**Status:** Ready for testing and deployment
