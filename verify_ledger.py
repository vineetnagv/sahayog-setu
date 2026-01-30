"""
Sahayog Setu - Ledger Verification Script
Tests the integrity of the Fairness Ledger (Hash Chain).
1. Fetches audit logs.
2. Verifies cryptographic links between entries.
3. Checks for any broken chains.
"""

import requests
import json
import hashlib

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def verify_ledger():
    print_section("[START] Verifying Fairness Ledger Integrity")
    
    # 1. Fetch Audit Log
    print("[INFO] Fetching latest audit logs...")
    try:
        res = requests.get(f"{BASE_URL}{API_PREFIX}/ledger/audit-log?limit=10")
        if res.status_code != 200:
            print(f"[FAIL] Could not fetch logs: {res.text}")
            return
            
        logs = res.json()
        print(f"[INFO] Found {len(logs)} ledger entries.")
        
        if not logs:
            print("[WARN] Ledger is empty. Run match/allocation tests first.")
            return

    except Exception as e:
        print(f"[FAIL] Network error: {e}")
        return

    # 2. Verify Chain Integrity (Client-Side Check)
    print("\n[INFO] verifying cryptographic links (Client-Side)...")
    
    # Note: logs are returned DESC (newest first).
    # We need to verify that log[i]['prev_hash'] == log[i+1]['hash']
    
    all_valid = True
    
    for i in range(len(logs) - 1):
        current_entry = logs[i]
        prev_entry = logs[i+1]
        
        expected_prev = prev_entry['hash']
        actual_prev = current_entry['prev_hash']
        
        if expected_prev == actual_prev:
            print(f"   [OK] Entry {current_entry['id'][:8]}... links correctly to {prev_entry['id'][:8]}...")
        else:
            print(f"   [FAIL] BROKEN LINK at {current_entry['id']}!")
            print(f"          Expected prev: {expected_prev}")
            print(f"          Actual prev:   {actual_prev}")
            all_valid = False
            
    if all_valid:
        print("\n[SUCCESS] Local chain verification passed for recent entries.")
    else:
        print("\n[FAIL] Chain integrity check failed.")

    # 3. Verify specific entry integration via API (Server-Side Check)
    latest_id = logs[0]['id']
    print(f"\n[INFO] requesting Server-Side verification for {latest_id}...")
    
    try:
        res = requests.get(f"{BASE_URL}{API_PREFIX}/ledger/verify/{latest_id}")
        result = res.json()
        
        print("   Server Response:")
        print(json.dumps(result, indent=2))
        
        if result.get("verified") is True:
             print("[SUCCESS] Server verified the entry integrity.")
        else:
             print("[FAIL] Server failed to verify entry.")
             
    except Exception as e:
        print(f"[FAIL] Verification API failed: {e}")

    print_section("[OK] Ledger Verification Complete")

if __name__ == "__main__":
    verify_ledger()
