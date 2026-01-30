"""
Sahayog Setu - Hash Chain Utility
Implements the immutable ledger logic for the Fairness Ledger.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any

def generate_hash(data: str) -> str:
    """Generate SHA-256 hash of a string."""
    return hashlib.sha256(data.encode()).hexdigest()

def create_ledger_entry(
    payload: Dict[str, Any],
    prev_hash: str,
    action: str,
    actor_id: str
) -> Dict[str, Any]:
    """
    Create a cryptographically linked ledger entry.
    
    Structure:
    Hash = SHA256( previous_hash + timestamp + action + json_payload )
    """
    timestamp = datetime.utcnow().isoformat()
    
    # Sort keys to ensure consistent hashing
    payload_str = json.dumps(payload, sort_keys=True)
    
    # Construct the content block to hash
    # Format: [PREV_HASH]|[TIMESTAMP]|[ACTION]|[ACTOR]|[PAYLOAD]
    content_block = f"{prev_hash}|{timestamp}|{action}|{actor_id}|{payload_str}"
    
    # Generate new hash
    new_hash = generate_hash(content_block)
    
    return {
        "hash": new_hash,
        "prev_hash": prev_hash,
        "action": action,
        "payload": payload,
        "timestamp": timestamp,
        "actor_id": actor_id
    }

def verify_chain_integrity(entry: Dict[str, Any], prev_entry: Dict[str, Any]) -> bool:
    """
    Verify if an entry is valid and strictly linked to the previous one.
    """
    # 1. Check if prev_hash matches
    if entry["prev_hash"] != prev_entry["hash"]:
        return False
        
    # 2. Re-calculate hash to verify content hasn't been tampered
    payload_str = json.dumps(entry["payload"], sort_keys=True)
    content_block = f"{entry['prev_hash']}|{entry['timestamp']}|{entry['action']}|{entry['actor_id']}|{payload_str}"
    
    calculated = generate_hash(content_block)
    
    return calculated == entry["hash"]
