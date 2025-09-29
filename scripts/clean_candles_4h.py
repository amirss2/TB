#!/usr/bin/env python3
"""
Clean Candles Script - Remove non-4h-aligned timestamps from database

This script removes candle records that are not aligned to 4-hour boundaries
to ensure clean data for 4h timeframe trading.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import db_connection
from database.models import Candle
from config.settings import TRADING_CONFIG

def is_aligned_4h(timestamp: int) -> bool:
    """Check if timestamp is aligned to 4-hour boundary"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        # 4h aligned means hour is divisible by 4 and minute/second are 0
        return dt.hour % 4 == 0 and dt.minute == 0 and dt.second == 0
    except Exception:
        return False

def clean_non_aligned_candles(symbols: list = None, dry_run: bool = True):
    """
    Remove non-4h-aligned candles from database
    
    Args:
        symbols: List of symbols to clean (default: all configured symbols)
        dry_run: If True, only report what would be deleted without actual deletion
    """
    if symbols is None:
        symbols = TRADING_CONFIG['symbols']
    
    print(f"{'DRY RUN: ' if dry_run else ''}Cleaning non-4h-aligned candles for symbols: {symbols}")
    
    session = db_connection.get_session()
    
    total_removed = 0
    total_checked = 0
    
    try:
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            
            # Get all candles for this symbol
            candles = session.query(Candle).filter(
                Candle.symbol == symbol
            ).order_by(Candle.timestamp.asc()).all()
            
            removed_count = 0
            symbol_total = len(candles)
            
            for candle in candles:
                total_checked += 1
                
                if not is_aligned_4h(candle.timestamp):
                    dt = datetime.fromtimestamp(candle.timestamp)
                    print(f"  {'Would remove' if dry_run else 'Removing'} non-aligned candle: {dt} ({candle.timestamp})")
                    
                    if not dry_run:
                        session.delete(candle)
                    
                    removed_count += 1
            
            total_removed += removed_count
            aligned_remaining = symbol_total - removed_count
            
            print(f"  {symbol}: {removed_count} non-aligned candles {'would be' if dry_run else ''} removed")
            print(f"  {symbol}: {aligned_remaining} aligned candles remaining")
    
        if not dry_run:
            session.commit()
            print(f"\nCommitted changes to database")
        else:
            print(f"\nDRY RUN completed - no changes made")
        
        print(f"\nSummary:")
        print(f"  Total candles checked: {total_checked}")
        print(f"  Non-aligned candles {'would be' if dry_run else ''} removed: {total_removed}")
        print(f"  Aligned candles remaining: {total_checked - total_removed}")
        
    except Exception as e:
        print(f"Error during cleaning: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean non-4h-aligned candles from database')
    parser.add_argument('--symbols', nargs='+', help='Symbols to clean (default: all configured)')
    parser.add_argument('--execute', action='store_true', help='Actually perform deletion (default: dry run)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("4H CANDLE CLEANING UTILITY")
    print("=" * 60)
    
    if not args.execute:
        print("WARNING: This is a DRY RUN. Use --execute to actually remove data.")
        print("-" * 60)
    
    # Show current timeframe
    print(f"Current timeframe: {TRADING_CONFIG['timeframe']}")
    
    if TRADING_CONFIG['timeframe'] != '4h':
        print("WARNING: Current timeframe is not 4h. Are you sure you want to clean for 4h alignment?")
        if args.execute:
            confirm = input("Type 'yes' to continue: ")
            if confirm.lower() != 'yes':
                print("Cancelled.")
                return
    
    clean_non_aligned_candles(
        symbols=args.symbols,
        dry_run=not args.execute
    )
    
    print("\nCleaning completed.")

if __name__ == '__main__':
    main()