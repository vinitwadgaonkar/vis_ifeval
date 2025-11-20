#!/usr/bin/env python3
"""
Manual script to track a repository visit.
This can be called from webhooks or manually.
"""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
import sys

def track_visit(ip_address=None, user_agent=None, referrer=None):
    """Track a single visit to the repository."""
    
    # Get repository root
    repo_root = Path(__file__).parent.parent.parent
    data_dir = repo_root / 'docs' / 'analytics' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_file = data_dir / 'raw_visits.jsonl'
    
    # Create visit record
    visit = {
        'timestamp': datetime.utcnow().isoformat(),
        'ip': ip_address or 'unknown',
        'user_agent': user_agent or 'unknown',
        'referrer': referrer or '',
        'hash': hashlib.md5(
            (ip_address or 'unknown' + user_agent or 'unknown').encode()
        ).hexdigest()
    }
    
    # Append to raw data file
    with open(raw_data_file, 'a') as f:
        f.write(json.dumps(visit) + '\n')
    
    print(f"Visit tracked: {visit['timestamp']}")
    return visit

if __name__ == '__main__':
    # Get command line arguments
    ip = sys.argv[1] if len(sys.argv) > 1 else None
    ua = sys.argv[2] if len(sys.argv) > 2 else None
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    
    track_visit(ip, ua, ref)

