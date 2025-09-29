#!/usr/bin/env python3
"""
Trading Log Analyzer - Fixed Version

This script analyzes trading bot logs to extract trading signals, positions,
profit/loss statistics, and overall performance metrics.

Usage:
    python analyze_trading_logs_fixed.py path/to/log_file.txt
"""

import re
import sys
import datetime
from collections import defaultdict, Counter
import pandas as pd
from typing import Dict, List, Tuple, Any

class TradingLogAnalyzer:
    def __init__(self, log_file_path: str):
        """Initialize with path to log file."""
        self.log_file_path = log_file_path
        self.signals = []
        self.positions_opened = []
        self.positions_closed = []
        self.timeframe = "unknown"
        self.symbols = []
        self.confidence_threshold = 0
        self.model_info = {}
        self.start_time = None
        self.end_time = None
        
        # Statistics counters
        self.signal_counts = defaultdict(lambda: defaultdict(int))
        self.signal_confidence = defaultdict(list)
        self.position_outcomes = defaultdict(int)
        self.pnl_by_symbol = defaultdict(list)
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        # Line count for progress reporting
        self.total_lines = 0
        self.lines_processed = 0
        
    def count_lines(self):
        """Count total lines in file for progress reporting."""
        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.total_lines = sum(1 for _ in f)
        
    def parse_log(self):
        """Parse the log file and extract relevant information."""
        print(f"Analyzing log file: {self.log_file_path}")
        self.count_lines()
        
        # Regular expressions for parsing different log entries
        timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
        signal_pattern = re.compile(r'TRADING SIGNAL for (\w+): (\w+) \(confidence: ([\d.]+)\)')
        probabilities_pattern = re.compile(r'Probabilities: BUY=([\d.]+), SELL=([\d.]+), HOLD=([\d.]+)')
        position_open_pattern = re.compile(r'Opening (\w+) position for (\w+): ([\d.e-]+) @ ([\d.]+)')
        position_id_pattern = re.compile(r'Position opened successfully: ID (\d+)')
        position_close_pattern = re.compile(r'Position (\d+) closed: (.*), PnL: ([+-]?[\d.]+) \(([-+]?[\d.]+)%\)')
        timeframe_pattern = re.compile(r'\[CONFIG\] timeframe=(\w+)')
        threshold_pattern = re.compile(r'Confidence threshold set to ([\d.]+)')
        model_trained_pattern = re.compile(r'Model trained successfully. Accuracy: ([\d.]+)')
        
        # Process the file line by line
        try:
            with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                current_timestamp = None
                current_symbol = None
                current_action = None
                current_confidence = None
                current_probabilities = None
                looking_for_probabilities = False
                position_id_pending = False
                last_position_details = {}
                
                for i, line in enumerate(f):
                    # Update progress every 100,000 lines
                    if i % 100000 == 0:
                        self.lines_processed = i
                        progress = (i / self.total_lines) * 100 if self.total_lines > 0 else 0
                        print(f"Progress: {progress:.1f}% ({i:,}/{self.total_lines:,} lines)")
                    
                    # Extract timestamp from log line
                    timestamp_match = timestamp_pattern.match(line)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        try:
                            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                            current_timestamp = timestamp
                            
                            # Update start/end time
                            if not self.start_time or timestamp < self.start_time:
                                self.start_time = timestamp
                            if not self.end_time or timestamp > self.end_time:
                                self.end_time = timestamp
                        except ValueError:
                            pass
                    
                    # Extract timeframe
                    timeframe_match = timeframe_pattern.search(line)
                    if timeframe_match:
                        self.timeframe = timeframe_match.group(1)
                    
                    # Extract confidence threshold
                    threshold_match = threshold_pattern.search(line)
                    if threshold_match:
                        self.confidence_threshold = float(threshold_match.group(1))
                    
                    # Extract model information
                    model_trained_match = model_trained_pattern.search(line)
                    if model_trained_match:
                        self.model_info['accuracy'] = float(model_trained_match.group(1))
                    
                    if "Model training completed" in line and ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            model_id = parts[1].strip()
                            self.model_info['model_id'] = model_id
                    
                    # Extract signals
                    signal_match = signal_pattern.search(line)
                    if signal_match:
                        symbol, action, confidence = signal_match.groups()
                        confidence = float(confidence)
                        
                        current_symbol = symbol
                        current_action = action
                        current_confidence = confidence
                        looking_for_probabilities = True
                        
                        # Add symbol to list if not present
                        if symbol not in self.symbols:
                            self.symbols.append(symbol)
                            
                    # Look for probabilities in the next line after a signal
                    elif looking_for_probabilities:
                        prob_match = probabilities_pattern.search(line)
                        if prob_match:
                            buy_prob, sell_prob, hold_prob = map(float, prob_match.groups())
                            current_probabilities = {
                                'buy': buy_prob,
                                'sell': sell_prob,
                                'hold': hold_prob
                            }
                            
                            # Now we have both signal and probabilities, add to list
                            signal = {
                                'timestamp': current_timestamp,
                                'symbol': current_symbol,
                                'action': current_action,
                                'confidence': current_confidence,
                                'probabilities': current_probabilities
                            }
                            
                            self.signals.append(signal)
                            
                            # Update signal statistics
                            self.signal_counts[current_symbol][current_action] += 1
                            self.signal_confidence[current_action].append(current_confidence)
                            
                            # Reset
                            looking_for_probabilities = False
                            current_symbol = None
                            current_action = None
                            current_confidence = None
                            current_probabilities = None
                    
                    # Extract positions opened
                    position_open_match = position_open_pattern.search(line)
                    if position_open_match:
                        position_type, symbol, amount, price = position_open_match.groups()
                        
                        last_position_details = {
                            'timestamp': current_timestamp,
                            'symbol': symbol,
                            'type': position_type,
                            'amount': float(amount),
                            'price': float(price),
                            'value': float(amount) * float(price)
                        }
                        position_id_pending = True
                        
                    # Look for position ID in the next line
                    elif position_id_pending:
                        position_id_match = position_id_pattern.search(line)
                        if position_id_match:
                            position_id = int(position_id_match.group(1))
                            last_position_details['id'] = position_id
                            self.positions_opened.append(last_position_details)
                            position_id_pending = False
                    
                    # Extract positions closed
                    position_close_match = position_close_pattern.search(line)
                    if position_close_match:
                        position_id, reason, pnl, pnl_percent = position_close_match.groups()
                        
                        position = {
                            'id': int(position_id),
                            'timestamp': current_timestamp,
                            'reason': reason,
                            'pnl': float(pnl),
                            'pnl_percent': float(pnl_percent)
                        }
                        
                        self.positions_closed.append(position)
                        
                        # Update PnL statistics
                        self.total_pnl += float(pnl)
                        
                        # Find corresponding opened position to get symbol
                        for opened_pos in self.positions_opened:
                            if opened_pos['id'] == int(position_id):
                                symbol = opened_pos['symbol']
                                self.pnl_by_symbol[symbol].append(float(pnl))
                                break
                        
                        # Update win/loss count
                        if float(pnl) > 0:
                            self.win_count += 1
                            self.position_outcomes['win'] += 1
                        else:
                            self.loss_count += 1
                            self.position_outcomes['loss'] += 1
                            
                        if "Stop Loss" in reason:
                            self.position_outcomes['stop_loss'] += 1
                        elif "Take Profit" in reason:
                            self.position_outcomes['take_profit'] += 1
                        elif "Manual" in reason:
                            self.position_outcomes['manual'] += 1
            
            self.lines_processed = self.total_lines
            print(f"Log analysis completed: processed {self.lines_processed:,} lines")
            
        except Exception as e:
            print(f"Error analyzing log file: {e}")
            import traceback
            traceback.print_exc()
            
    def match_positions(self):
        """Match opened and closed positions to calculate duration and complete metrics."""
        positions = []
        
        # Create a lookup for closed positions by ID
        closed_positions_dict = {p['id']: p for p in self.positions_closed}
        
        for opened in self.positions_opened:
            position_id = opened['id']
            position = opened.copy()
            
            # If the position was closed, add closing information
            if position_id in closed_positions_dict:
                closed = closed_positions_dict[position_id]
                position.update({
                    'close_timestamp': closed['timestamp'],
                    'pnl': closed['pnl'],
                    'pnl_percent': closed['pnl_percent'],
                    'reason': closed['reason'],
                    'status': 'closed'
                })
                
                # Calculate position duration if timestamps are valid
                try:
                    open_time = opened['timestamp']
                    close_time = closed['timestamp']
                    duration = close_time - open_time
                    position['duration_seconds'] = duration.total_seconds()
                    position['duration'] = str(duration)
                except (ValueError, KeyError, TypeError):
                    position['duration_seconds'] = None
                    position['duration'] = "unknown"
            else:
                position.update({
                    'status': 'open',
                    'pnl': None,
                    'pnl_percent': None,
                    'reason': None
                })
                
            positions.append(position)
            
        return positions
            
    def generate_report(self):
        """Generate a comprehensive trading report."""
        # Match opened and closed positions
        complete_positions = self.match_positions()
        
        # Calculate trading period
        trading_period = (self.end_time - self.start_time) if self.start_time and self.end_time else datetime.timedelta(0)
        trading_hours = trading_period.total_seconds() / 3600 if trading_period else 0
        
        # Calculate win rate
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Average trade PnL
        avg_pnl = self.total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate signals by confidence level
        high_confidence = sum(1 for s in self.signals if s['confidence'] >= 0.9)
        medium_confidence = sum(1 for s in self.signals if 0.8 <= s['confidence'] < 0.9)
        low_confidence = sum(1 for s in self.signals if self.confidence_threshold <= s['confidence'] < 0.8)
        below_threshold = sum(1 for s in self.signals if s['confidence'] < self.confidence_threshold)
        
        # Print report header
        print("\n" + "="*80)
        print(f"📊 TRADING BOT PERFORMANCE REPORT - {self.timeframe.upper()} TIMEFRAME")
        print("="*80)
        
        # General statistics
        print("\n🕒 TRADING PERIOD")
        print(f"Start Time: {self.start_time}")
        print(f"End Time: {self.end_time}")
        print(f"Duration: {trading_period} ({trading_hours:.1f} hours)")
        
        # Configuration
        print("\n⚙️ CONFIGURATION")
        print(f"Timeframe: {self.timeframe}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Model ID: {self.model_info.get('model_id', 'Unknown')}")
        print(f"Model Accuracy: {self.model_info.get('accuracy', 0):.4f}")
        
        # Signal statistics
        print("\n📈 SIGNAL STATISTICS")
        print(f"Total Signals: {len(self.signals)}")
        print(f"  - High Confidence (≥90%): {high_confidence}")
        print(f"  - Medium Confidence (80-89%): {medium_confidence}")
        print(f"  - Low Confidence ({int(self.confidence_threshold*100)}-79%): {low_confidence}")
        print(f"  - Below Threshold (<{int(self.confidence_threshold*100)}%): {below_threshold}")
        
        print("\nSignals by Action:")
        all_actions = set()
        for symbol_actions in self.signal_counts.values():
            all_actions.update(symbol_actions.keys())
            
        for action in sorted(all_actions):
            total_action = sum(counts[action] for counts in self.signal_counts.values())
            avg_confidence = sum(self.signal_confidence[action]) / len(self.signal_confidence[action]) if self.signal_confidence[action] else 0
            print(f"  - {action}: {total_action} signals (Avg Confidence: {avg_confidence:.2f})")
        
        print("\nSignals by Symbol:")
        for symbol in sorted(self.signal_counts.keys()):
            symbol_total = sum(self.signal_counts[symbol].values())
            print(f"  - {symbol}: {symbol_total} signals")
            for action, count in sorted(self.signal_counts[symbol].items()):
                print(f"      {action}: {count}")
        
        # Trading statistics
        print("\n💰 TRADING STATISTICS")
        print(f"Total Trades: {len(self.positions_opened)}")
        print(f"Completed Trades: {total_trades}")
        print(f"Winning Trades: {self.win_count}")
        print(f"Losing Trades: {self.loss_count}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: {self.total_pnl:.4f}")
        print(f"Average P&L per Trade: {avg_pnl:.4f}")
        print(f"Positions Closed by Stop Loss: {self.position_outcomes['stop_loss']}")
        print(f"Positions Closed by Take Profit: {self.position_outcomes['take_profit']}")
        print(f"Positions Closed Manually: {self.position_outcomes['manual']}")
        
        # P&L by Symbol
        print("\nP&L by Symbol:")
        for symbol, pnl_list in sorted(self.pnl_by_symbol.items()):
            total_symbol_pnl = sum(pnl_list)
            avg_symbol_pnl = total_symbol_pnl / len(pnl_list) if pnl_list else 0
            win_count = sum(1 for p in pnl_list if p > 0)
            loss_count = sum(1 for p in pnl_list if p <= 0)
            win_rate = (win_count / len(pnl_list) * 100) if pnl_list else 0
            print(f"  - {symbol}: Total: {total_symbol_pnl:.4f}, Avg: {avg_symbol_pnl:.4f}, "
                  f"Win Rate: {win_rate:.2f}% ({win_count}/{len(pnl_list)})")
        
        # Position duration statistics
        durations = [p.get('duration_seconds') for p in complete_positions 
                    if p['status'] == 'closed' and p.get('duration_seconds') is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"\nAverage Position Duration: {datetime.timedelta(seconds=avg_duration)}")
            
            min_duration = min(durations)
            max_duration = max(durations)
            print(f"Shortest Position: {datetime.timedelta(seconds=min_duration)}")
            print(f"Longest Position: {datetime.timedelta(seconds=max_duration)}")
        
        # Most profitable and least profitable trades
        if complete_positions:
            closed_positions = [p for p in complete_positions if p['status'] == 'closed' and p.get('pnl') is not None]
            if closed_positions:
                most_profitable = max(closed_positions, key=lambda p: p['pnl'])
                least_profitable = min(closed_positions, key=lambda p: p['pnl'])
                
                print("\n🏆 MOST PROFITABLE TRADE")
                print(f"Symbol: {most_profitable['symbol']}")
                print(f"P&L: {most_profitable['pnl']:.4f} ({most_profitable['pnl_percent']:.2f}%)")
                print(f"Entry: {most_profitable['price']}")
                print(f"Time: {most_profitable['timestamp']}")
                print(f"Duration: {most_profitable.get('duration', 'Unknown')}")
                
                print("\n📉 LEAST PROFITABLE TRADE")
                print(f"Symbol: {least_profitable['symbol']}")
                print(f"P&L: {least_profitable['pnl']:.4f} ({least_profitable['pnl_percent']:.2f}%)")
                print(f"Entry: {least_profitable['price']}")
                print(f"Time: {least_profitable['timestamp']}")
                print(f"Duration: {least_profitable.get('duration', 'Unknown')}")
        
        # Trading frequency
        if total_trades > 0 and trading_hours > 0:
            trades_per_hour = total_trades / trading_hours
            print(f"\nTrading Frequency: {trades_per_hour:.2f} trades/hour")
            if trades_per_hour > 0:
                avg_hours_between_trades = 1 / trades_per_hour
                print(f"Average time between trades: {avg_hours_between_trades:.2f} hours")
        
        # Export data to CSV files
        self.export_data(complete_positions)
        
        print("\n" + "="*80)
        print("Report Complete!")
        print("="*80)
        
    def export_data(self, complete_positions):
        """Export trading data to CSV files."""
        try:
            # Export signals
            signals_df = pd.DataFrame([
                {
                    'timestamp': str(s['timestamp']),
                    'symbol': s['symbol'],
                    'action': s['action'],
                    'confidence': s['confidence'],
                    'buy_prob': s['probabilities']['buy'] if 'probabilities' in s else None,
                    'sell_prob': s['probabilities']['sell'] if 'probabilities' in s else None,
                    'hold_prob': s['probabilities']['hold'] if 'probabilities' in s else None
                } for s in self.signals
            ])
            
            if not signals_df.empty:
                signals_df.to_csv('trading_signals.csv', index=False)
                print("\n📄 Data exported to trading_signals.csv")
            
            # Export positions
            if complete_positions:
                positions_df = pd.DataFrame([
                    {k: (str(v) if isinstance(v, datetime.datetime) else v) 
                     for k, v in p.items()}
                    for p in complete_positions
                ])
                positions_df.to_csv('trading_positions.csv', index=False)
                print("📄 Data exported to trading_positions.csv")
            
            # Create visualizations
            self.create_visualizations(signals_df, pd.DataFrame(complete_positions) if complete_positions else None)
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def create_visualizations(self, signals_df, positions_df):
        """Create basic visualizations from the trading data."""
        try:
            # Ensure matplotlib is available
            import matplotlib.pyplot as plt
            import os
            
            # Create output directory if it doesn't exist
            os.makedirs('trading_analysis', exist_ok=True)
            
            if not signals_df.empty:
                # 1. Signal confidence distribution
                plt.figure(figsize=(10, 6))
                plt.hist(signals_df['confidence'], bins=20, alpha=0.7, color='skyblue')
                plt.title('Trading Signal Confidence Distribution')
                plt.xlabel('Confidence Level')
                plt.ylabel('Number of Signals')
                plt.grid(True, alpha=0.3)
                plt.savefig('trading_analysis/signal_confidence_distribution.png')
                
                # 2. Signals by action
                action_counts = signals_df['action'].value_counts()
                if len(action_counts) > 1:  # Only create pie chart if there are multiple actions
                    plt.figure(figsize=(8, 8))
                    plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', 
                            startangle=140, colors=['green', 'red', 'gray'])
                    plt.title('Trading Signals by Action')
                    plt.savefig('trading_analysis/signals_by_action.png')
            
            if positions_df is not None and not positions_df.empty:
                # 3. Win/Loss ratio
                if 'pnl' in positions_df.columns:
                    closed_positions = positions_df[positions_df['status'] == 'closed'].copy()
                    if not closed_positions.empty and 'pnl' in closed_positions:
                        # Convert pnl to numeric in case it's stored as string
                        closed_positions['pnl'] = pd.to_numeric(closed_positions['pnl'], errors='coerce')
                        
                        closed_positions['outcome'] = closed_positions['pnl'].apply(
                            lambda x: 'Win' if x > 0 else 'Loss')
                        outcome_counts = closed_positions['outcome'].value_counts()
                        
                        if len(outcome_counts) > 0:
                            plt.figure(figsize=(8, 8))
                            plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%',
                                    startangle=140, colors=['green', 'red'])
                            plt.title('Trading Outcomes (Win/Loss Ratio)')
                            plt.savefig('trading_analysis/win_loss_ratio.png')
                
                # 4. P&L by symbol
                if 'pnl' in positions_df.columns and 'symbol' in positions_df.columns:
                    closed_positions = positions_df[positions_df['status'] == 'closed'].copy()
                    if not closed_positions.empty:
                        # Convert columns to appropriate types
                        closed_positions['pnl'] = pd.to_numeric(closed_positions['pnl'], errors='coerce')
                        
                        pnl_by_symbol = closed_positions.groupby('symbol')['pnl'].sum()
                        
                        if not pnl_by_symbol.empty:
                            plt.figure(figsize=(10, 6))
                            bars = plt.bar(pnl_by_symbol.index, pnl_by_symbol, 
                                          color=['green' if x > 0 else 'red' for x in pnl_by_symbol])
                            plt.title('Total P&L by Symbol')
                            plt.xlabel('Symbol')
                            plt.ylabel('Total P&L')
                            plt.grid(True, alpha=0.3)
                            
                            # Add values on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                plt.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.2f}',
                                        ha='center', va='bottom')
                                        
                            plt.savefig('trading_analysis/pnl_by_symbol.png')
            
            print("📊 Visualizations saved to trading_analysis/ directory")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")

def main():
    if len(sys.argv) < 2:
        print("Please provide the path to the log file")
        print("Usage: python analyze_trading_logs_fixed.py path/to/log_file.txt")
        sys.exit(1)
        
    log_file_path = sys.argv[1]
    analyzer = TradingLogAnalyzer(log_file_path)
    analyzer.parse_log()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
