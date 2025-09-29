#!/usr/bin/env python3
"""
Log Summary Utility for TB Trading Bot
=====================================

Parses LOG.txt and extracts basic metrics for diagnostic purposes.
Handles both JSON and semi-structured/plain text log formats.

Usage:
    python scripts/log_summary.py --input LOG.txt --out-json reports/diagnostic_stats.json --out-md reports/diagnostic_stats.md
"""

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional


class LogAnalyzer:
    """Analyzes log files and extracts diagnostic metrics."""
    
    def __init__(self):
        self.total_lines = 0
        self.counts_by_level = {'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'DEBUG': 0}
        self.api_error_lines = 0
        self.signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        self.latency_values = []
        self.unique_symbols = set()
        self.timestamps = []
        
        # Regex patterns for parsing
        self.timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3})')
        self.iso_timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)')
        self.level_pattern = re.compile(r' - (INFO|WARNING|ERROR|DEBUG) - ')
        self.symbol_pattern = re.compile(r'(BTC|ETH|SOL|DOGE|USDT|BTCUSDT|ETHUSDT|SOLUSDT|DOGEUSDT)')
        self.signal_pattern = re.compile(r'\b(BUY|SELL|HOLD)\b')
        self.latency_pattern = re.compile(r'(?:latency|time)=([0-9\.]+)(?:ms|s)')
        self.api_error_pattern = re.compile(r'(?:404|timeout|error|failed|connection.*error)', re.IGNORECASE)
    
    def parse_line(self, line: str) -> None:
        """Parse a single log line and extract relevant information."""
        self.total_lines += 1
        line = line.strip()
        
        if not line:
            return
            
        # Try to parse as JSON first
        if line.startswith('{') and line.endswith('}'):
            try:
                log_data = json.loads(line)
                self._parse_json_log(log_data)
                return
            except json.JSONDecodeError:
                pass
        
        # Parse as structured text log
        self._parse_text_log(line)
    
    def _parse_json_log(self, log_data: Dict[str, Any]) -> None:
        """Parse JSON-formatted log entry."""
        # Extract timestamp
        if 'timestamp' in log_data:
            self._add_timestamp(log_data['timestamp'])
        
        # Extract log level
        if 'level' in log_data:
            level = log_data['level'].upper()
            if level in self.counts_by_level:
                self.counts_by_level[level] += 1
        
        # Extract message and analyze content
        message = str(log_data.get('message', ''))
        self._analyze_message_content(message)
        
        # Check for extra_data
        if 'extra_data' in log_data:
            extra = log_data['extra_data']
            if isinstance(extra, dict):
                # Extract symbol from extra data
                if 'symbol' in extra:
                    self.unique_symbols.add(extra['symbol'])
                
                # Extract event type for signals
                if 'event_type' in extra and 'direction' in extra:
                    direction = extra['direction'].upper()
                    if direction in self.signal_counts:
                        self.signal_counts[direction] += 1
    
    def _parse_text_log(self, line: str) -> None:
        """Parse text-formatted log entry."""
        # Extract timestamp
        timestamp_match = self.timestamp_pattern.search(line)
        if timestamp_match:
            self._add_timestamp(timestamp_match.group(1))
        else:
            # Try ISO format
            iso_match = self.iso_timestamp_pattern.search(line)
            if iso_match:
                self._add_timestamp(iso_match.group(1))
        
        # Extract log level
        level_match = self.level_pattern.search(line)
        if level_match:
            level = level_match.group(1).upper()
            if level in self.counts_by_level:
                self.counts_by_level[level] += 1
        
        # Analyze message content
        self._analyze_message_content(line)
    
    def _analyze_message_content(self, content: str) -> None:
        """Analyze message content for patterns."""
        # Check for API errors
        if self.api_error_pattern.search(content):
            self.api_error_lines += 1
        
        # Extract symbols
        symbol_matches = self.symbol_pattern.findall(content)
        for symbol in symbol_matches:
            self.unique_symbols.add(symbol)
        
        # Extract signals
        signal_matches = self.signal_pattern.findall(content)
        for signal in signal_matches:
            if signal in self.signal_counts:
                self.signal_counts[signal] += 1
        
        # Extract latency values
        latency_matches = self.latency_pattern.findall(content)
        for latency_str in latency_matches:
            try:
                latency = float(latency_str)
                # Convert to milliseconds if needed
                if 'time=' in content and 's' in content:
                    latency *= 1000  # Convert seconds to milliseconds
                self.latency_values.append(latency)
            except ValueError:
                pass
    
    def _add_timestamp(self, timestamp_str: str) -> None:
        """Add timestamp to collection for analysis."""
        try:
            # Try different timestamp formats
            for fmt in ['%Y-%m-%d %H:%M:%S,%f', '%Y-%m-%d %H:%M:%S.%f', 
                       '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
                       '%Y-%m-%dT%H:%M:%S.%f']:
                try:
                    dt = datetime.strptime(timestamp_str.replace('Z', ''), fmt.replace('Z', ''))
                    self.timestamps.append(dt)
                    break
                except ValueError:
                    continue
        except Exception:
            pass  # Silently skip unparsable timestamps
    
    def get_statistics(self) -> Dict[str, Any]:
        """Generate diagnostic statistics."""
        stats = {
            'total_lines': self.total_lines,
            'counts_by_level': self.counts_by_level,
            'api_error_lines': self.api_error_lines,
            'signal_counts': self.signal_counts,
            'unique_symbols': sorted(list(self.unique_symbols)),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Latency statistics
        if self.latency_values:
            stats['average_latency_ms'] = round(statistics.mean(self.latency_values), 2)
            stats['p95_latency_ms'] = round(sorted(self.latency_values)[int(len(self.latency_values) * 0.95)], 2)
            stats['max_latency_ms'] = round(max(self.latency_values), 2)
            stats['min_latency_ms'] = round(min(self.latency_values), 2)
        else:
            stats['average_latency_ms'] = None
            stats['p95_latency_ms'] = None
            stats['max_latency_ms'] = None
            stats['min_latency_ms'] = None
        
        # Timestamp analysis
        if self.timestamps:
            stats['earliest_timestamp'] = min(self.timestamps).isoformat()
            stats['latest_timestamp'] = max(self.timestamps).isoformat()
            stats['log_duration_hours'] = round((max(self.timestamps) - min(self.timestamps)).total_seconds() / 3600, 2)
        else:
            stats['earliest_timestamp'] = None
            stats['latest_timestamp'] = None
            stats['log_duration_hours'] = None
        
        return stats


def write_json_output(stats: Dict[str, Any], output_path: str) -> None:
    """Write statistics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def write_markdown_output(stats: Dict[str, Any], output_path: str) -> None:
    """Write statistics to Markdown file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    md_content = f"""# Log Analysis Summary

Generated on: {stats['analysis_timestamp']}

## Overview

- **Total Lines**: {stats['total_lines']:,}
- **Log Duration**: {stats['log_duration_hours']} hours
- **Earliest Entry**: {stats['earliest_timestamp']}
- **Latest Entry**: {stats['latest_timestamp']}

## Log Levels

| Level | Count |
|-------|-------|
"""
    
    for level, count in stats['counts_by_level'].items():
        md_content += f"| {level} | {count:,} |\n"
    
    md_content += f"""
## Error Analysis

- **API Error Lines**: {stats['api_error_lines']:,}

## Trading Signals

| Signal Type | Count |
|-------------|-------|
"""
    
    for signal, count in stats['signal_counts'].items():
        md_content += f"| {signal} | {count:,} |\n"
    
    md_content += f"""
## Symbols Detected

{', '.join(stats['unique_symbols']) if stats['unique_symbols'] else 'None detected'}

## Performance Metrics

- **Average Latency**: {stats['average_latency_ms']} ms
- **95th Percentile Latency**: {stats['p95_latency_ms']} ms
- **Min Latency**: {stats['min_latency_ms']} ms
- **Max Latency**: {stats['max_latency_ms']} ms

## Summary

This analysis covers {stats['total_lines']:,} log lines spanning {stats['log_duration_hours']} hours of trading bot activity.
The system processed {sum(stats['signal_counts'].values())} trading signals across {len(stats['unique_symbols'])} cryptocurrency symbols.

Generated by TB Log Summary Utility v1.0
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Parse TB trading bot logs and generate diagnostic statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/log_summary.py --input LOG.txt --out-json reports/diagnostic_stats.json --out-md reports/diagnostic_stats.md
  python scripts/log_summary.py --input LOG.txt --out-json stats.json
        """
    )
    
    parser.add_argument('--input', required=True, help='Input log file path (e.g., LOG.txt)')
    parser.add_argument('--out-json', required=True, help='Output JSON file path')
    parser.add_argument('--out-md', help='Output Markdown file path (optional)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    print(f"Analyzing log file: {args.input}")
    
    # Initialize analyzer
    analyzer = LogAnalyzer()
    
    # Process log file
    try:
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                analyzer.parse_line(line)
                if line_num % 1000 == 0:
                    print(f"Processed {line_num:,} lines...")
    except Exception as e:
        print(f"Error reading log file: {e}")
        return 1
    
    # Generate statistics
    stats = analyzer.get_statistics()
    
    # Write outputs
    try:
        write_json_output(stats, args.out_json)
        print(f"JSON output written to: {args.out_json}")
        
        if args.out_md:
            write_markdown_output(stats, args.out_md)
            print(f"Markdown output written to: {args.out_md}")
        
        print(f"\nSummary: Analyzed {stats['total_lines']:,} lines, found {sum(stats['signal_counts'].values())} signals")
        
    except Exception as e:
        print(f"Error writing output files: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())