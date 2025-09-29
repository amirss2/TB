#!/usr/bin/env python3
"""
Import 1m OHLCV CSV data into candles table.

Unlimited mode:
  --no-limit   (یا)  --limit 0
Examples:
  python scripts/import_minute_csv.py --dir data/1m --no-limit
  python scripts/import_minute_csv.py --file data/1m/BTCUSDT_1m.csv --symbol BTCUSDT --limit 0
"""

import os, sys, csv, argparse
from typing import List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import db_connection
from database.models import Candle
from sqlalchemy import and_

BATCH_SIZE = 5000  # بزرگ‌تر برای سرعت بهتر

def normalize_ts(t):
    try:
        v = int(float(t))
        if v > 10_000_000_000: v //= 1000
        return v
    except: return None

def detect_symbol_from_filename(p): return os.path.basename(p).split('_')[0].upper()

def read_csv_stream(path):
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            yield row

def fetch_existing_timestamps(session, symbol: str, ts_list: List[int]) -> set:
    if not ts_list: return set()
    rows = session.query(Candle.timestamp).filter(
        and_(Candle.symbol == symbol, Candle.timestamp.in_(ts_list))
    ).all()
    return {r[0] for r in rows}

def insert_candles(session, symbol: str, rows: List[dict]):
    session.bulk_save_objects([
        Candle(symbol=symbol,
               timestamp=r["timestamp"],
               open=r["open"],
               high=r["high"],
               low=r["low"],
               close=r["close"],
               volume=r["volume"]) for r in rows
    ])

def import_file(path: str, symbol: Optional[str], limit: int):
    unlimited = (limit <= 0)
    if not symbol: symbol = detect_symbol_from_filename(path)
    symbol = symbol.upper()
    session = db_connection.get_session()
    inserted = 0; scanned = 0; batch = []
    try:
        for row in read_csv_stream(path):
            if (not unlimited) and inserted >= limit:
                break
            ts = normalize_ts(row.get('timestamp') or row.get('time') or row.get('date'))
            if ts is None: continue
            try:
                o=float(row['open']);h=float(row['high']);l=float(row['low']);c=float(row['close']);v=float(row['volume'])
            except: continue
            batch.append({"timestamp": ts,"open": o,"high": h,"low": l,"close": c,"volume": v})
            scanned += 1
            # پر شدن بچ
            if len(batch) >= BATCH_SIZE:
                inserted += flush_batch(session, symbol, batch, limit, unlimited)
                batch.clear()
                if (not unlimited) and inserted >= limit: break
        # باقی‌مانده
        if batch and ((unlimited) or inserted < limit):
            inserted += flush_batch(session, symbol, batch, limit, unlimited)
        print(f"[{symbol}] Imported={inserted} scanned={scanned} (mode={'unlimited' if unlimited else limit}) file={os.path.basename(path)}")
    except Exception as e:
        session.rollback()
        print(f"[{symbol}] ERROR: {e}")
        raise
    finally:
        session.close()

def flush_batch(session, symbol, batch, limit, unlimited):
    candidate_ts = [b["timestamp"] for b in batch]
    existing = fetch_existing_timestamps(session, symbol, candidate_ts)
    final_rows = [b for b in batch if b["timestamp"] not in existing]
    if not final_rows: return 0
    if not unlimited:
        space = limit - session.query(Candle).filter(Candle.symbol==symbol).count()
        if space <= 0: return 0
        final_rows = final_rows[:space]
    insert_candles(session, symbol, final_rows)
    session.commit()
    return len(final_rows)

def collect_files(args):
    pairs=[]
    if args.file:
        if not os.path.isfile(args.file): raise FileNotFoundError(args.file)
        pairs.append((args.file, args.symbol))
    elif args.dir:
        for name in os.listdir(args.dir):
            if name.lower().endswith('.csv'):
                full = os.path.join(args.dir, name)
                if args.symbols:
                    sym = detect_symbol_from_filename(full)
                    if sym.upper() not in [s.upper() for s in args.symbols]: continue
                pairs.append((full, None))
    else:
        raise ValueError("Provide --file or --dir")
    return pairs

def main():
    p = argparse.ArgumentParser(description="Import 1m OHLCV CSV data")
    p.add_argument('--file')
    p.add_argument('--dir')
    p.add_argument('--symbol')
    p.add_argument('--symbols', nargs='+')
    p.add_argument('--limit', type=int, default=10000, help='Max rows per symbol (0/unset=unlimited with --no-limit)')
    p.add_argument('--no-limit', action='store_true', help='Ignore limit (import all)')
    args = p.parse_args()
    if args.no_limit: args.limit = 0
    files = collect_files(args)
    print(f"Import start: files={len(files)} limit={'UNLIMITED' if args.limit<=0 else args.limit}")
    for path, sym in files:
        import_file(path, sym or args.symbol, args.limit)
    print("Done.")
if __name__ == '__main__':
    main()