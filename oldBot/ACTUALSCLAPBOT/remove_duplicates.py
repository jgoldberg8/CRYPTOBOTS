import json
import re
from typing import Dict
import os
from tempfile import NamedTemporaryFile
from datetime import datetime

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")

def deduplicate_logs(file_path: str) -> Dict:
    """
    Remove duplicate log entries based on tokenAddress field, keeping the latest entry.
    
    Args:
        file_path (str): Path to the log file to deduplicate
        
    Returns:
        Dict: Statistics about the deduplication process
    """
    # Dictionary to store entries by token address, with timestamp as key for sorting
    token_entries: Dict[str, Dict] = {}
    
    # Compile regex pattern to extract timestamp and JSON content
    log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z) \[FINAL-ANALYSIS\]: (.+)')
    
    # Statistics
    stats = {
        'total_records': 0,
        'unique_tokens': 0,
        'duplicates_removed': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stats['total_records'] += 1
                
                match = log_pattern.match(line.strip())
                if not match:
                    continue
                    
                timestamp_str, json_str = match.groups()
                
                try:
                    json_obj = json.loads(json_str)
                    token_address = json_obj.get('tokenAddress')
                    
                    if token_address:
                        # Parse timestamp for comparison
                        timestamp = parse_timestamp(timestamp_str)
                        
                        # If we haven't seen this token before, or if this entry is newer
                        if token_address not in token_entries or \
                           parse_timestamp(token_entries[token_address]['timestamp']) < timestamp:
                            token_entries[token_address] = {
                                'timestamp': timestamp_str,
                                'line': line
                            }
                except json.JSONDecodeError:
                    continue
        
        # Create temporary file with UTF-8 encoding
        with NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            # Sort entries by timestamp before writing
            sorted_entries = sorted(
                [entry['line'] for entry in token_entries.values()],
                key=lambda x: parse_timestamp(log_pattern.match(x.strip()).group(1))
            )
            temp_file.writelines(sorted_entries)
            temp_file_path = temp_file.name
        
        # Replace original file with deduplicated content
        os.replace(temp_file_path, file_path)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except IOError as e:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)  # Clean up temp file if it exists
        raise IOError(f"Error processing file {file_path}: {str(e)}")
    
    stats['unique_tokens'] = len(token_entries)
    stats['duplicates_removed'] = stats['total_records'] - stats['unique_tokens']
    return stats

def main():
    file_path = 'logs/final-analysis-2025-01-06.log'
    
    try:
        stats = deduplicate_logs(file_path)
        print(f"Log deduplication complete:")
        print(f"Total records processed: {stats['total_records']}")
        print(f"Unique token addresses found: {stats['unique_tokens']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"File has been deduplicated in place: {file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()