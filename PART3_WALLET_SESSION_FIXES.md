# Part 3: Wallet/Transaction/Position Fixes

## Overview

Part 3 addresses critical issues with SQLAlchemy session management, wallet balance reconciliation, TP/SL logic, and transaction atomicity.

## Problems Fixed

### 1. SQLAlchemy "not bound to a Session" Error

**Problem:**
- Threads/workers accessing detached database objects
- Session lifecycle not properly managed
- Objects becoming stale across thread boundaries

**Solution:**
- Implemented `scoped_session` for thread-safe session management
- Added `expire_on_commit=False` to prevent object expiration
- Created transaction context manager (`get_transaction_session()`)
- Use `session.merge()` or fresh queries for cross-thread objects

**Implementation:**
```python
# database/connection.py
session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
self.Session = scoped_session(session_factory)

@contextmanager
def get_transaction_session(self):
    """Context manager for atomic transactions"""
    session = self.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
```

### 2. TP/SL Position Closure Logic

**Problem:**
- TP3 not closing positions immediately
- Only flagging TP3_hit without closing
- Positions continuing beyond TP3

**Solution:**
- TP1 hit: Flag only, trail SL to breakeven
- TP2 hit: Flag only, trail SL to TP1
- **TP3 hit: CLOSE POSITION IMMEDIATELY**

**Implementation:**
```python
def _check_long_position_triggers(self, position, current_price):
    # ... TP1 and TP2 logic ...
    
    elif position.tp2_hit and not position.tp3_hit and current_price >= position.tp3_price:
        # TP3 hit (+10%) - CLOSE POSITION IMMEDIATELY
        position.tp3_hit = True
        self.logger.info(f"TP3 hit for position {position.id} at +10%. Closing position immediately.")
        self.close_position(position.id, "TP3 (+10%) reached")
```

### 3. Atomic Wallet/Position Updates

**Problem:**
- Position updates and wallet updates in separate transactions
- Race conditions possible
- Inconsistent state if one fails

**Solution:**
- All operations in single atomic transaction
- Row-level locking with `with_for_update()`
- Position + Wallet + Transaction in same commit

**Implementation:**
```python
def close_position(self, position_id, reason):
    with db_connection.get_transaction_session() as session:
        # Lock position and wallet
        position = session.query(Position).filter_by(id=position_id).with_for_update().first()
        wallet = session.query(Wallet).filter_by(account_type=account_type).with_for_update().first()
        
        # Update position
        position.status = 'CLOSED'
        position.closed_at = datetime.now()
        
        # Update wallet atomically
        wallet.locked_balance -= position_value
        wallet.available_balance += position_value + pnl
        
        # Record transaction
        transaction = WalletTransaction(...)
        session.add(transaction)
        
        # Single commit for all changes
```

### 4. Balance Consistency

**Problem:**
- `available_balance` could become negative
- No validation before deducting
- Drift between `total`, `available`, and `locked`

**Solution:**
- Formula enforced: `available_balance = total_balance - locked_balance - used_balance`
- Validation before position open
- Clamp to 0 if would go negative (with error log)

**Implementation:**
```python
# Before opening position
if wallet.available_balance < position_value:
    raise ValueError("Insufficient available balance")

# After closing position
wallet.available_balance += position_value + pnl
if wallet.available_balance < 0:
    self.logger.error(f"Balance would be negative, clamping to 0")
    wallet.available_balance = 0
```

### 5. Wallet Transaction Recording

**Problem:**
- Incomplete transaction records
- Balance snapshots not recorded
- PnL components not tracked

**Solution:**
- Record every position open/close
- Store `balance_before` and `balance_after`
- Include detailed description with PnL breakdown

**Implementation:**
```python
# Position open
transaction = WalletTransaction(
    account_type=account_type,
    transaction_type='position_open',
    amount=-position_value,  # Negative = locked
    balance_before=balance_before,
    balance_after=balance_after,
    position_id=position_id,
    description=f"Opened {side} position for {symbol}: {quantity} @ ${entry_price:.6f}"
)

# Position close
transaction = WalletTransaction(
    account_type=account_type,
    transaction_type='position_close',
    amount=pnl,  # Net PnL
    balance_before=balance_before,
    balance_after=balance_after,
    position_id=position_id,
    description=f"Closed {side} {symbol}: {reason}, Net PnL=${pnl:.2f} (Gross=${gross_pnl:.2f}, Costs=${costs:.2f})"
)
```

### 6. Restart Safety

**Problem:**
- `used_balance` not restored on restart
- Wallet state inconsistent with open positions
- Memory state vs database state mismatch

**Solution:**
- Restore wallet from database on startup
- Calculate `used_balance` from all open positions
- Reconcile locked balance with position values

**Implementation:**
```python
def _initialize_wallet(self):
    wallet = session.query(Wallet).filter_by(account_type=account_type).first()
    
    if wallet:
        # Restore balances
        self.demo_balance = wallet.total_balance
        
        # Restore used_balance from open positions
        open_positions = session.query(Position).filter_by(status='OPEN').all()
        self.used_balance = sum(pos.entry_price * pos.quantity for pos in open_positions)
        
        # Update wallet locked balance
        wallet.locked_balance = self.used_balance
        wallet.available_balance = wallet.total_balance - self.used_balance
        session.commit()
```

### 7. Idempotency Protection

**Problem:**
- Same position could be closed multiple times
- Race conditions in monitoring thread
- Duplicate transactions

**Solution:**
- Idempotency key per close operation
- Track closed positions in memory
- Check before processing

**Implementation:**
```python
def close_position(self, position_id, reason):
    # Idempotency check
    if not hasattr(self, '_closed_positions'):
        self._closed_positions = set()
    
    if position_id in self._closed_positions:
        self.logger.warning(f"Position {position_id} already closed, skipping")
        return False
    
    # ... close logic ...
    
    # Mark as closed
    self._closed_positions.add(position_id)
```

### 8. Wallet Health Check

**Problem:**
- No way to verify wallet consistency
- Silent drift over time
- No reconciliation mechanism

**Solution:**
- Health check method validates:
  - `sum(transactions) + initial_balance == total_balance`
  - `total_balance - locked == available_balance`
  - Open positions match locked balance

**Implementation:**
```python
def wallet_health_check(self):
    wallet = session.query(Wallet).filter_by(account_type=account_type).first()
    transactions = session.query(WalletTransaction).filter_by(account_type=account_type).all()
    
    initial_balance = TRADING_CONFIG['demo_balance']
    transaction_sum = sum(t.amount for t in transactions)
    expected_balance = initial_balance + transaction_sum
    
    balance_match = abs(expected_balance - wallet.total_balance) < 0.01
    
    return {
        'status': 'HEALTHY' if balance_match else 'MISMATCH',
        'expected': expected_balance,
        'actual': wallet.total_balance,
        'difference': wallet.total_balance - expected_balance
    }
```

## Testing Scenarios

### Scenario 1: TP1 → TP2 → TP3 Progression

**Steps:**
1. Open LONG position at $100
2. Price reaches TP1 ($103) - Position stays open, SL to breakeven
3. Price reaches TP2 ($106) - Position stays open, SL to TP1
4. Price reaches TP3 ($110) - **Position closes immediately**

**Expected:**
- Position status = CLOSED
- `closed_at` timestamp set
- Net PnL calculated with all fees
- `used_balance` freed
- Wallet transaction recorded
- `available_balance` increased by (position_value + net_pnl)
- Balance never negative

### Scenario 2: Network Disconnect During Position

**Steps:**
1. Open position
2. Network timeout occurs
3. Invalid price data received

**Expected:**
- Price validation rejects suspicious changes (>50%)
- Position NOT closed with bad price
- System paused if `trading_paused_by_network = True`
- Position checks skipped during pause
- Normal operation resumes after reconnect

### Scenario 3: Server Restart with Open Positions

**Steps:**
1. Open 3 positions ($25 each = $75 locked)
2. Server stops
3. Server restarts

**Expected:**
- Wallet restored from database
- `used_balance = $75` calculated from open positions
- `available_balance = total - 75`
- Position monitoring resumes
- No duplicate transactions

### Scenario 4: Concurrent Position Operations

**Steps:**
1. Position manager monitoring in thread 1
2. Trading engine opening position in thread 2
3. Both access database simultaneously

**Expected:**
- Scoped sessions prevent conflicts
- Row-level locks prevent race conditions
- All transactions atomic
- No "not bound to Session" errors

## Error Resolution

### "Instance is not bound to a Session"

**Before:**
```python
position = session.query(Position).get(position_id)
session.close()
# Later in different thread
position.status = 'CLOSED'  # ERROR!
```

**After:**
```python
with db_connection.get_transaction_session() as session:
    position = session.query(Position).filter_by(id=position_id).first()
    position.status = 'CLOSED'
    # Automatic commit
```

### Available Balance Negative

**Before:**
```python
wallet.available_balance -= position_value  # Can go negative!
```

**After:**
```python
if wallet.available_balance < position_value:
    raise ValueError("Insufficient balance")

wallet.available_balance -= position_value
if wallet.available_balance < 0:
    self.logger.error("Balance negative, clamping")
    wallet.available_balance = 0
```

## Logging Examples

### Position Open
```
✓ Position opened successfully: ID 401
Wallet updated: Locked $25.00, Available: $100.00 → $75.00
```

### TP3 Close
```
TP3 hit for position 401 at +10%. Closing position immediately.
PnL Calculation: Gross PnL=$2.50 (+10%), Entry Fee=$0.065, Exit Fee=$0.0715, Spread=$0.0275, Slippage=$0.01375, Total Costs=$0.178, Net PnL=$2.322 (+9.29%)
Wallet updated: Freed $25.00, PnL=$2.32, Available: $75.00 → $102.32
✓ Position 401 closed: TP3 (+10%) reached, Net PnL: $2.3220 (+9.29%)
```

### Health Check Pass
```
✓ Wallet health check PASSED: Total=$102.32, Available=$102.32
```

### Health Check Fail
```
⚠ Wallet health check FAILED: {
  'status': 'MISMATCH',
  'expected': 102.50,
  'actual': 102.32,
  'difference': -0.18
}
```

## Acceptance Criteria

✅ No "not bound to Session" errors in logs  
✅ TP3 hit closes position immediately  
✅ Position + Wallet + Transaction in atomic commits  
✅ `available_balance` never negative  
✅ Wallet health check passes: `sum(transactions) + initial == total`  
✅ Restart safety: wallet and positions restored correctly  
✅ Concurrent operations don't cause race conditions  
✅ All balance changes have corresponding transactions  

## Integration

All fixes are integrated into:
- `database/connection.py` - Session management
- `trading/position_manager.py` - TP/SL logic, atomic transactions
- `trading/engine.py` - Wallet initialization

No configuration changes required. Existing code will automatically use new transaction system.

## Performance Impact

- **Minimal overhead** from row-level locking (microseconds)
- **Improved reliability** from atomic operations
- **Better concurrency** from scoped sessions
- **Easier debugging** from comprehensive logging

## Migration

No manual migration needed. On next restart:
1. Scoped sessions automatically used
2. Wallet restored from database
3. Open positions rehydrated
4. Health check can be run: `position_manager.wallet_health_check()`
