#!/usr/bin/env python3
"""
Extract Match Timings from Tournament Logs
===========================================

This script parses tournament logs to extract timing information for each match
and saves it to a CSV file.
"""

import re
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def parse_log_file(log_file: Path) -> List[Dict]:
    """Parse tournament log file and extract match timing information."""
    match_timings = []
    
    if not log_file.exists():
        print(f"Warning: Log file not found: {log_file}")
        return match_timings
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern to match match results with timing
    # Look for patterns like "Pair X/Y: AI1 vs AI2" followed by timing info
    pair_pattern = r'Pair (\d+)/(\d+):\s+(\w+)\s+vs\s+(\w+)'
    result_pattern = r'Result:\s+(\w+)\s+(\d+)-(\d+)\s+(\w+)\s+\(draws:\s+(\d+)\)'
    
    # Find all pairs
    pairs = re.finditer(pair_pattern, content)
    results = re.finditer(result_pattern, content)
    
    # Create dictionaries for pairs and results
    pairs_dict = {}
    for match in pairs:
        pair_num, total_pairs, ai_left, ai_right = match.groups()
        pairs_dict[int(pair_num)] = {
            'pair_num': int(pair_num),
            'total_pairs': int(total_pairs),
            'ai_left': ai_left,
            'ai_right': ai_right,
            'start_pos': match.start()
        }
    
    # Match results with pairs (find the closest pair before each result)
    results_list = []
    for match in results:
        ai_left, left_wins, right_wins, ai_right, draws = match.groups()
        results_list.append({
            'ai_left': ai_left,
            'ai_right': ai_right,
            'left_wins': int(left_wins),
            'right_wins': int(right_wins),
            'draws': int(draws),
            'position': match.start()
        })
    
    # Try to extract timing from log entries between pair start and result
    for pair_num, pair_info in pairs_dict.items():
        # Find the result for this pair
        pair_result = None
        for result in results_list:
            if (result['ai_left'] == pair_info['ai_left'] and 
                result['ai_right'] == pair_info['ai_right']):
                pair_result = result
                break
        
        if not pair_result:
            continue
        
        # Try to extract timing between pair start and result
        # Look for timestamps around this section
        start_pos = pair_info['start_pos']
        end_pos = pair_result['position']
        section = content[start_pos:end_pos]
        
        # Look for timing patterns (HH:MM:SS or seconds)
        time_patterns = [
            r'(\d{2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d+\.?\d*)\s*seconds?',
            r'took\s+(\d+\.?\d*)\s*(?:seconds?|s)',
        ]
        
        match_time = None
        for pattern in time_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(int, match.groups())
                    match_time = hours * 3600 + minutes * 60 + seconds
                else:  # seconds
                    match_time = float(match.group(1))
                break
        
        # If no explicit timing found, estimate from log entry times
        if match_time is None:
            # Extract all timestamps in the section
            timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
            timestamps = re.findall(timestamp_pattern, section)
            if len(timestamps) >= 2:
                try:
                    start_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S')
                    match_time = (end_time - start_time).total_seconds()
                except:
                    pass
        
        match_timings.append({
            'pair_number': pair_info['pair_num'],
            'total_pairs': pair_info['total_pairs'],
            'ai_left': pair_info['ai_left'],
            'ai_right': pair_info['ai_right'],
            'left_wins': pair_result['left_wins'],
            'right_wins': pair_result['right_wins'],
            'draws': pair_result['draws'],
            'total_games': pair_result['left_wins'] + pair_result['right_wins'] + pair_result['draws'],
            'match_time_seconds': match_time if match_time else None,
            'match_time_formatted': format_time(match_time) if match_time else 'N/A'
        })
    
    return match_timings


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    if seconds is None:
        return 'N/A'
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def extract_timings_from_stdout(stdout_file: Path) -> List[Dict]:
    """Extract timing information from stdout log file."""
    match_timings = []
    
    if not stdout_file.exists():
        print(f"Warning: Stdout file not found: {stdout_file}")
        return match_timings
    
    # Read the stdout file
    with open(stdout_file, 'r') as f:
        lines = f.readlines()
    
    # Track matches as we go
    current_match = None
    match_start_time = None
    
    for i, line in enumerate(lines):
        # Look for match start: "Pair X/Y: AI1 vs AI2"
        pair_match = re.search(r'Pair (\d+)/(\d+):\s+(\w+)\s+vs\s+(\w+)', line)
        if pair_match:
            pair_num, total_pairs, ai_left, ai_right = pair_match.groups()
            current_match = {
                'pair_number': int(pair_num),
                'total_pairs': int(total_pairs),
                'ai_left': ai_left,
                'ai_right': ai_right,
            }
            # Extract timestamp if present
            ts_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
            if ts_match:
                try:
                    match_start_time = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                except:
                    pass
        
        # Look for match result
        if current_match:
            result_match = re.search(
                r'Result:\s+(\w+)\s+(\d+)-(\d+)\s+(\w+)\s+\(draws:\s+(\d+)\)', 
                line
            )
            if result_match:
                ai_left_result, left_wins, right_wins, ai_right_result, draws = result_match.groups()
                
                # Calculate timing
                match_time = None
                ts_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
                if ts_match and match_start_time:
                    try:
                        match_end_time = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                        match_time = (match_end_time - match_start_time).total_seconds()
                    except:
                        pass
                
                match_timings.append({
                    'pair_number': current_match['pair_number'],
                    'total_pairs': current_match['total_pairs'],
                    'ai_left': current_match['ai_left'],
                    'ai_right': current_match['ai_right'],
                    'left_wins': int(left_wins),
                    'right_wins': int(right_wins),
                    'draws': int(draws),
                    'total_games': int(left_wins) + int(right_wins) + int(draws),
                    'match_time_seconds': match_time,
                    'match_time_formatted': format_time(match_time) if match_time else 'N/A'
                })
                current_match = None
                match_start_time = None
    
    return match_timings


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_match_timings.py <stdout_log_file> <output_csv_file>")
        sys.exit(1)
    
    stdout_file = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])
    
    print(f"Extracting match timings from: {stdout_file}")
    
    # Extract timings from stdout file
    match_timings = extract_timings_from_stdout(stdout_file)
    
    if not match_timings:
        print("Warning: No match timings found in log file")
        # Try alternative parsing
        match_timings = parse_log_file(stdout_file)
    
    # Write to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        if match_timings:
            fieldnames = [
                'pair_number', 'total_pairs', 'ai_left', 'ai_right',
                'left_wins', 'right_wins', 'draws', 'total_games',
                'match_time_seconds', 'match_time_formatted'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(match_timings)
            print(f"Extracted {len(match_timings)} match timings")
        else:
            # Write header even if no data
            writer = csv.DictWriter(f, fieldnames=[
                'pair_number', 'total_pairs', 'ai_left', 'ai_right',
                'left_wins', 'right_wins', 'draws', 'total_games',
                'match_time_seconds', 'match_time_formatted'
            ])
            writer.writeheader()
            print("No match timings found - created empty CSV with headers")
    
    print(f"Match timings saved to: {output_csv}")


if __name__ == "__main__":
    main()

