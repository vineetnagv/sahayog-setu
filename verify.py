"""
Sahayog Setu - End-to-End Verification Script
Tests the full 'Harvest Hero' flow:
1. Setup specific scenario (Worker + Government Pause)
2. Test Matching Engine (Should find Private Job)
3. Test Allocation (Should record hash)
4. Test Bias Check (Should trigger if we skip a needy worker)
"""

import requests
import json
import random
import string
import time

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def random_string(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def run_test():
    print_section("[START] Starting Sahayog Setu Verification")
    
    # 0. Health Check
    try:
        health = requests.get(f"{BASE_URL}/health").json()
        db_health = health.get("checks", {}).get("database", {})
        print(f"[INFO] Health Check: API={health.get('status')} DB={db_health.get('status')}")
        if db_health.get("status") != "healthy":
            print(f"[ERROR] DB Error: {db_health.get('message')}")
            return
    except Exception as e:
        print(f"[FAIL] Health check failed: {e}")
        return

    # ---------------------------------------------------------
    # 1. SETUP: Create Village & Workers
    # ---------------------------------------------------------
    village_code = f"VILL_{random_string()}"
    print(f"[INFO] Created Test Village: {village_code}")
    
    # Create Needy Worker (Score ~80)
    print("\n[1] Creating 'Needy Worker' (Ramesh)...")
    res = requests.post(f"{BASE_URL}{API_PREFIX}/workers", json={
        "aadhaar_number": f"1234{random_string(8)}",
        "name": "Ramesh (Needy)",
        "phone": "9998887776",
        "village_code": village_code,
        "family_size": 6,          # High need
        "land_owned_acres": 0.0    # High need
    })
    if res.status_code != 201:
        print(f"[FAIL] Failed to create worker: {res.text}")
        return
    needy_worker = res.json()
    needy_id = needy_worker["worker_id"]
    print(f"   [OK] Created. ID: {needy_id}. Score: {needy_worker['need_score']}")

    # Create Less Needy Worker (Score ~30)
    print("\n[2] Creating 'Less Needy Worker' (Suresh)...")
    res = requests.post(f"{BASE_URL}{API_PREFIX}/workers", json={
        "aadhaar_number": f"5678{random_string(8)}",
        "name": "Suresh (Okay)",
        "phone": "9998887775",
        "village_code": village_code,
        "family_size": 2,          # Low need
        "land_owned_acres": 5.0    # Low need
    })
    less_needy_worker = res.json()
    less_needy_id = less_needy_worker["worker_id"]
    print(f"   [OK] Created. ID: {less_needy_id}. Score: {less_needy_worker['need_score']}")

    # ---------------------------------------------------------
    # 2. SETUP: Create Government Project (PAUSED)
    # ---------------------------------------------------------
    print("\n[3] Creating PAUSED Government Project...")
    res = requests.post(f"{BASE_URL}{API_PREFIX}/government/projects", json={
        "project_code": f"MGNREGA_{random_string()}",
        "village_code": village_code,
        "panchayat_code": "P1",
        "title": "Road Construction",
        "budget_allocated": 100000,
        "start_date": "2025-01-01"
    })
    proj_id = res.json()["project_id"]
    
    # Pause it
    requests.put(f"{BASE_URL}{API_PREFIX}/government/projects/{proj_id}/status", 
                 params={"status": "PAUSED", "pause_reason": "Harvest Season"})
    print("   [OK] Project Created and set to PAUSED.")

    # ---------------------------------------------------------
    # 3. SETUP: Create Private Demand
    # ---------------------------------------------------------
    print("\n[4] Creating Private Farmer Demand...")
    # Register Farmer
    f_res = requests.post(f"{BASE_URL}{API_PREFIX}/farmers", json={
        "name": "Farmer Singh",
        "phone": "9876543210",
        "village_code": village_code,
        "land_area_acres": 10.0
    })
    farmer_id = f_res.json()["farmer_id"]
    
    # Create Demand
    d_res = requests.post(f"{BASE_URL}{API_PREFIX}/private-demand", json={
        "farmer_id": farmer_id,
        "description": "Wheat Harvest",
        "work_type": "harvest",
        "workers_needed": 5,
        "daily_wage": 400.0,
        "work_date": "2025-10-15"
    })
    demand_id = d_res.json()["demand_id"]
    print(f"   [OK] Demand Created. ID: {demand_id}")

    # ---------------------------------------------------------
    # 4. TEST: Matching Engine (Harvest Hero)
    # ---------------------------------------------------------
    print_section("[TEST] Testing Matching Engine")
    print("\n[5] Checking matches for Ramesh...")
    res = requests.get(f"{BASE_URL}{API_PREFIX}/matching/worker/{needy_id}")
    match_data = res.json()
    
    print(f"   Result: {json.dumps(match_data, indent=2)}")
    
    if match_data["match_type"] == "PRIVATE":
        print("   [SUCCESS] System correctly routed to Private Job (Harvest Hero)!")
    else:
        print("   [FAIL] Did not route to private job.")

    # ---------------------------------------------------------
    # 5. TEST: Bias Checker (Simulate Unfairness)
    # ---------------------------------------------------------
    print_section("[TEST] Testing Bias Checker")
    print("Attempting to allocate job to Suresh (Less Needy) instead of Ramesh (Needy)...")
    
    res = requests.post(f"{BASE_URL}{API_PREFIX}/matching/allocate", json={
        "worker_id": less_needy_id,
        "job_type": "PRIVATE",
        "job_id": demand_id
    })
    
    result = res.json()
    print("\nAllocation Result:")
    print(json.dumps(result, indent=2))
    
    if "bias_alerts" in result and len(result["bias_alerts"]) > 0:
        print("\n[SUCCESS] Bias Checker CAUGHT the unfair allocation!")
        print(f"   Alert Reason: {result['bias_alerts'][0]['reason']}")
    else:
        print("\n[FAIL] Bias Checker did not catch the unfairness.")

    print_section("[OK] Verification Complete")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n[FAIL] Script Failed: {e}")
        print("Make sure the server is running on port 8000!")
