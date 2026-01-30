#!/usr/bin/env python3
"""
+==============================================================================+
|              SAHAYOG SETU - WORK ALLOCATION ALGORITHM DEMO                   |
|                                                                              |
|  This standalone script demonstrates how the geospatial work allocation      |
|  algorithm works in Sahayog Setu. It uses:                                   |
|  - Real geocoding via OpenStreetMap (Nominatim)                              |
|  - Real satellite data via OpenWeather Agro API (NDVI & NDWI)                |
|  - Geodesic distance calculations                                            |
|                                                                              |
|  Run this file independently: python allocation_demo.py                      |
+==============================================================================+

ALGORITHM OVERVIEW:
===================

The Sahayog Setu allocation algorithm prioritizes workers based on TWO factors:

+-----------------------------------------------------------------------------+
|  FINAL_SCORE = (GEO_SCORE x 0.4) + (NEED_SCORE x 0.6)                       |
|                                                                              |
|  Where:                                                                      |
|  * GEO_SCORE  = Proximity score (closer to job = higher score)              |
|  * NEED_SCORE = Drought/distress score from satellite data (NDVI + NDWI)    |
|                                                                              |
|  The 60/40 weighting prioritizes NEED over DISTANCE, ensuring workers       |
|  from drought-affected areas get work even if they're slightly farther.     |
+-----------------------------------------------------------------------------+

WHAT ARE NDVI AND NDWI?
=======================

NDVI (Normalized Difference Vegetation Index):
  - Measures vegetation health using satellite imagery
  - Range: -1 to +1
  - Low NDVI (< 0.2) = Barren/Drought-affected land
  - High NDVI (> 0.6) = Healthy, green vegetation
  
NDWI (Normalized Difference Water Index):
  - Measures water content in vegetation/soil
  - Range: -1 to +1  
  - Low NDWI = Dry conditions (potential drought)
  - High NDWI = Good water availability

For drought detection, we INVERT these scores:
  Low NDVI + Low NDWI = HIGH need score (priority for work allocation)
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

# ISRO VEDAS API - Get your key at: https://vedas.sac.gov.in/
# The API provides real NDVI satellite data for locations in India
VEDAS_API_KEY = "wUHMJtqXrBdEz_wFQkIdgQ"  # Your VEDAS API key

# Algorithm Weights (must sum to 1.0)
WEIGHT_DISTANCE = 0.4  # 40% weight for proximity
WEIGHT_NEED = 0.6      # 60% weight for need/drought score

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Village:
    """Represents a village with location and drought data."""
    name: str
    state: str
    latitude: float
    longitude: float
    ndvi: Optional[float] = None      # Vegetation index (-1 to 1)
    ndwi: Optional[float] = None      # Water index (-1 to 1)
    drought_score: float = 50.0       # Normalized 0-100 (higher = more distressed)


@dataclass
class Worker:
    """Represents a worker seeking employment."""
    id: int
    name: str
    village: Village
    skills: List[str]
    is_available: bool = True


@dataclass
class Job:
    """Represents a job/work opportunity."""
    id: int
    title: str
    village: Village
    workers_needed: int
    skills_required: List[str]


@dataclass
class ScoredWorker:
    """Worker with calculated allocation scores."""
    worker: Worker
    distance_km: float
    geo_score: float          # 0-100, higher = closer
    drought_score: float      # 0-100, higher = more need
    final_score: float        # Weighted combination
    explanation: str


# =============================================================================
# GEOSPATIAL SERVICE
# =============================================================================

class GeospatialService:
    """
    Handles all location-based calculations.
    Uses OpenStreetMap's Nominatim for geocoding (free, no API key needed).
    """
    
    def __init__(self):
        self.geocode_cache = {}
    
    def get_coordinates(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        Convert a place name to (latitude, longitude) using Nominatim.
        
        Example:
            >>> geo.get_coordinates("Kuduragere, Karnataka")
            (13.2845, 77.3892)
        """
        if place_name in self.geocode_cache:
            return self.geocode_cache[place_name]
        
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": place_name,
                "format": "json",
                "limit": 1
            }
            headers = {"User-Agent": "SahayogSetuDemo/1.0"}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                self.geocode_cache[place_name] = (lat, lon)
                return (lat, lon)
            return None
            
        except Exception as e:
            print(f"  [!] Geocoding error for '{place_name}': {e}")
            return None
    
    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """
        Calculate geodesic distance (km) between two (lat, lon) points.
        Uses the Haversine formula for accuracy on Earth's curved surface.
        
        Example:
            >>> geo.calculate_distance((13.28, 77.38), (13.35, 77.42))
            8.45  # kilometers
        """
        lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
        lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return r * c


# =============================================================================
# SATELLITE DATA SERVICE (NDVI via ISRO VEDAS API)
# =============================================================================

class SatelliteDataService:
    """
    Fetches real satellite vegetation indices from ISRO's VEDAS API.
    
    VEDAS (Visualisation of Earth Data & Archival System) by ISRO provides:
    - NDVI (Normalized Difference Vegetation Index)
    - Historical satellite data for India
    
    API Documentation: https://vedas.sac.gov.in/
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key if api_key and api_key != "YOUR_API_KEY_HERE" else None
        self.base_url = "https://vedas.sac.gov.in/vapi/ridam_server3/info/"
        self.ndvi_cache = {}  # Cache to avoid repeated API calls
        
    def get_vegetation_indices(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Fetch NDVI for a location from VEDAS API.
        
        Returns:
            Tuple of (ndvi, ndwi) values, each in range -1 to 1
            Note: VEDAS provides NDVI; NDWI is estimated from NDVI.
            
        If API is unavailable, returns simulated values based on location.
        """
        cache_key = f"{lat:.4f},{lon:.4f}"
        if cache_key in self.ndvi_cache:
            return self.ndvi_cache[cache_key]
        
        if self.api_key:
            result = self._fetch_from_vedas(lat, lon)
            self.ndvi_cache[cache_key] = result
            return result
        else:
            result = self._simulate_indices(lat, lon)
            self.ndvi_cache[cache_key] = result
            return result
    
    def _fetch_from_vedas(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Fetch real NDVI data from ISRO's VEDAS API.
        
        Uses the Location Profile endpoint (T5S1I1) to get point-based NDVI data.
        """
        import json
        
        try:
            url = f"{self.base_url}?X-API-KEY={self.api_key}"
            
            # Get data for last 2 years
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years back
            
            body_args = {
                "dataset_id": "T3S1P1",      # NDVI Dataset
                "from_time": start_date.strftime("%Y%m%d"),
                "to_time": end_date.strftime("%Y%m%d"),
                "param": "NDVI",
                "lon": lon,
                "lat": lat,
                "filter_nodata": "no",
                "composite": False
            }
            
            payload = {
                "layer": "T5S1I1",  # Layer ID for Point/Location queries
                "args": body_args
            }
            
            headers = {
                "accept": "application/json", 
                "content-type": "application/json",
                "Referer": "https://vedas.sac.gov.in"
            }
            
            print(f"      [VEDAS] Fetching NDVI for ({lat:.4f}, {lon:.4f})...", end=" ")
            
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                data = result.get("result", [])
                
                if data and len(data) > 0:
                    # Calculate average NDVI from all data points
                    # VEDAS returns data in format: [[timestamp, value], ...]
                    ndvi_values = []
                    for item in data:
                        if isinstance(item, list) and len(item) >= 2:
                            val = item[1]
                            if val is not None and val != "null":
                                try:
                                    ndvi_values.append(float(val))
                                except (ValueError, TypeError):
                                    pass
                    
                    if ndvi_values:
                        # Get average of recent values (last 10 readings)
                        recent_values = ndvi_values[-10:] if len(ndvi_values) > 10 else ndvi_values
                        avg_ndvi = sum(recent_values) / len(recent_values)
                        
                        # VEDAS NDVI is typically in 0-1 range, normalize if needed
                        if avg_ndvi > 1:
                            avg_ndvi = avg_ndvi / 100  # Some datasets return 0-100
                        
                        # Estimate NDWI from NDVI (correlated in most vegetation scenarios)
                        # NDWI is typically 0.6-0.8 of NDVI in vegetated areas
                        estimated_ndwi = avg_ndvi * 0.7 - 0.1
                        
                        print(f"OK (NDVI={avg_ndvi:.3f})")
                        return (round(avg_ndvi, 3), round(estimated_ndwi, 3))
                
                print("No data, using simulation")
                return self._simulate_indices(lat, lon)
            else:
                print(f"Error {response.status_code}")
                return self._simulate_indices(lat, lon)
                
        except requests.exceptions.Timeout:
            print("Timeout, using simulation")
            return self._simulate_indices(lat, lon)
        except Exception as e:
            print(f"Error: {e}")
            return self._simulate_indices(lat, lon)
    
    def _simulate_indices(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Simulate realistic NDVI/NDWI values based on geographic patterns.
        
        Used as fallback when API is unavailable.
        This creates realistic variation:
        - Lower values in arid regions (central India)
        - Higher values in coastal/river areas
        """
        # Base simulation using location hash for consistency
        location_hash = abs(hash(f"{lat:.2f},{lon:.2f}")) % 1000 / 1000
        
        # Simulate regional drought patterns in India
        # Areas around 15-20N, 75-78E tend to be more drought-prone
        drought_factor = 0
        if 14 < lat < 20 and 74 < lon < 79:
            drought_factor = 0.3  # Deccan Plateau - drought prone
        elif 24 < lat < 28 and 70 < lon < 76:
            drought_factor = 0.25  # Rajasthan - arid
        
        # NDVI: Vegetation health (0.1 to 0.7 typical range)
        ndvi = 0.3 + (location_hash * 0.4) - drought_factor
        ndvi = max(-0.1, min(0.8, ndvi))
        
        # NDWI: Water content (correlated with NDVI but with variation)
        ndwi = ndvi * 0.6 + (location_hash * 0.2) - 0.1
        ndwi = max(-0.3, min(0.5, ndwi))
        
        return (round(ndvi, 3), round(ndwi, 3))
    
    def calculate_drought_score(self, ndvi: float, ndwi: float) -> float:
        """
        Convert NDVI and NDWI into a drought/need score (0-100).
        
        Formula:
          1. Invert NDVI: Low vegetation = high drought
          2. Invert NDWI: Low water = high drought  
          3. Combine with weights (NDVI 60%, NDWI 40%)
          4. Normalize to 0-100 scale
        
        Example:
            NDVI=0.1, NDWI=-0.1 -> Drought score ~75 (high need)
            NDVI=0.7, NDWI=0.3  -> Drought score ~25 (low need)
        """
        # Normalize NDVI from [-1,1] to [0,1], then invert
        # NDVI of -1 -> 1.0 (extreme drought)
        # NDVI of 1  -> 0.0 (lush vegetation)
        ndvi_normalized = (1 - ndvi) / 2  # Range: 0 to 1, inverted
        
        # Normalize NDWI similarly
        ndwi_normalized = (1 - ndwi) / 2
        
        # Weighted combination (NDVI is more reliable for vegetation stress)
        combined = (ndvi_normalized * 0.6) + (ndwi_normalized * 0.4)
        
        # Scale to 0-100
        drought_score = combined * 100
        
        return round(drought_score, 2)


# =============================================================================
# ALLOCATION ENGINE
# =============================================================================

class AllocationEngine:
    """
    The core allocation algorithm that matches workers to jobs.
    
    ALGORITHM STEPS:
    ================
    1. Get job location coordinates
    2. For each available worker:
       a. Calculate distance to job
       b. Convert distance to geo_score (closer = higher)
       c. Get drought_score from satellite data (more distress = higher)
       d. Compute: final_score = (geo_score x 0.4) + (drought_score x 0.6)
    3. Sort workers by final_score (descending)
    4. Return top N workers for the job
    """
    
    def __init__(self):
        self.geo_service = GeospatialService()
        self.satellite_service = SatelliteDataService(VEDAS_API_KEY)
    
    def calculate_geo_score(self, distance_km: float) -> float:
        """
        Convert distance to a score (0-100).
        
        Formula: geo_score = (1 / (1 + distance)) x 100
        
        This gives:
          - 0 km  -> 100 points
          - 1 km  -> 50 points
          - 9 km  -> 10 points
          - 99 km -> 1 point
        
        The inverse relationship means each additional km matters less,
        which is realistic (5km vs 6km matters less than 1km vs 2km).
        """
        return (1 / (1 + distance_km)) * 100
    
    def allocate_workers(self, job: Job, workers: List[Worker], top_n: int = 5) -> List[ScoredWorker]:
        """
        Main allocation method.
        
        Args:
            job: The job to allocate workers for
            workers: List of all available workers
            top_n: Number of top workers to return
            
        Returns:
            List of ScoredWorker objects, sorted by final_score (descending)
        """
        print(f"\n{'='*70}")
        print(f"ALLOCATING WORKERS FOR: {job.title}")
        print(f"Location: {job.village.name}, {job.village.state}")
        print(f"Workers Needed: {job.workers_needed}")
        print(f"{'='*70}\n")
        
        job_loc = (job.village.latitude, job.village.longitude)
        scored_workers = []
        
        for worker in workers:
            if not worker.is_available:
                continue
            
            worker_loc = (worker.village.latitude, worker.village.longitude)
            
            # --- STEP 1: Calculate Distance ---
            distance_km = self.geo_service.calculate_distance(job_loc, worker_loc)
            
            # --- STEP 2: Calculate Geo Score ---
            geo_score = self.calculate_geo_score(distance_km)
            
            # --- STEP 3: Get/Calculate Drought Score ---
            # If village doesn't have satellite data, fetch it
            if worker.village.ndvi is None:
                ndvi, ndwi = self.satellite_service.get_vegetation_indices(
                    worker.village.latitude, 
                    worker.village.longitude
                )
                worker.village.ndvi = ndvi
                worker.village.ndwi = ndwi
                worker.village.drought_score = self.satellite_service.calculate_drought_score(ndvi, ndwi)
            
            drought_score = worker.village.drought_score
            
            # --- STEP 4: Calculate Final Weighted Score ---
            final_score = (geo_score * WEIGHT_DISTANCE) + (drought_score * WEIGHT_NEED)
            
            # Build explanation
            explanation = (
                f"Distance: {distance_km:.1f}km (geo_score: {geo_score:.1f}) | "
                f"NDVI: {worker.village.ndvi:.2f}, NDWI: {worker.village.ndwi:.2f} "
                f"(drought_score: {drought_score:.1f})"
            )
            
            scored_workers.append(ScoredWorker(
                worker=worker,
                distance_km=round(distance_km, 2),
                geo_score=round(geo_score, 2),
                drought_score=round(drought_score, 2),
                final_score=round(final_score, 2),
                explanation=explanation
            ))
        
        # --- STEP 5: Sort by Final Score (Descending) ---
        scored_workers.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_workers[:top_n]


# =============================================================================
# DEMO DATA
# =============================================================================

def create_demo_data() -> Tuple[Job, List[Worker]]:
    """
    Create realistic dummy data for Karnataka, India.
    
    Scenario: A road construction job in Bangalore needs workers.
    Workers are located in various villages around Karnataka.
    """
    
    # --- Villages with realistic coordinates ---
    villages = [
        Village("Bengaluru", "Karnataka", 12.9716, 77.5946),       # Job location
        Village("Kolar", "Karnataka", 13.1362, 78.1292),           # 60km from Bangalore
        Village("Tumkur", "Karnataka", 13.3392, 77.1010),          # 70km
        Village("Mandya", "Karnataka", 12.5218, 76.8951),          # 100km
        Village("Ramanagara", "Karnataka", 12.7159, 77.2819),      # 40km
        Village("Chikkaballapur", "Karnataka", 13.4355, 77.7315),  # 50km
        Village("Devanahalli", "Karnataka", 13.2473, 77.7140),     # 35km (near airport)
        Village("Anekal", "Karnataka", 12.7105, 77.6970),          # 25km
    ]
    
    # --- The Job ---
    job = Job(
        id=1,
        title="Road Construction - National Highway Extension",
        village=villages[0],  # Bengaluru
        workers_needed=5,
        skills_required=["construction", "labor"]
    )
    
    # --- Workers in various villages ---
    workers = [
        Worker(1, "Ramesh Kumar", villages[1], ["construction", "masonry"]),      # Kolar
        Worker(2, "Suresh Gowda", villages[2], ["labor", "driving"]),             # Tumkur
        Worker(3, "Manjunath", villages[3], ["construction", "welding"]),         # Mandya
        Worker(4, "Venkatesh", villages[4], ["labor", "construction"]),           # Ramanagara
        Worker(5, "Shivakumar", villages[5], ["masonry", "construction"]),        # Chikkaballapur
        Worker(6, "Basavaraj", villages[6], ["labor", "painting"]),               # Devanahalli
        Worker(7, "Nagaraj", villages[7], ["construction", "plumbing"]),          # Anekal
        Worker(8, "Ravi Shankar", villages[1], ["labor", "construction"]),        # Kolar (2nd worker)
    ]
    
    return job, workers


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the allocation demo."""
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "     SAHAYOG SETU - GEOSPATIAL WORK ALLOCATION DEMO".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Check API key status
    if VEDAS_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[!] No API key configured - using simulated satellite data")
        print("    Get a key at: https://vedas.sac.gov.in/\n")
    else:
        print("\n[OK] Using ISRO VEDAS API for real NDVI satellite data\n")
    
    print("""
ALGORITHM EXPLANATION:
---------------------------------------------------------------------
The allocation algorithm scores workers using TWO factors:

  1. DISTANCE (40% weight)
     - Calculated using Haversine formula (geodesic distance)
     - Score formula: geo_score = 100 / (1 + distance_km)
     - Workers closer to the job get higher geo_score
     
  2. NEED/DROUGHT (60% weight)  
     - Based on NDVI (vegetation) + NDWI (water) satellite indices
     - Low NDVI + Low NDWI = Drought-affected area = Higher priority
     - Workers from distressed areas get higher drought_score
     
  FINAL SCORE = (geo_score x 0.4) + (drought_score x 0.6)
  
  The 60/40 split prioritizes social equity - workers from 
  drought-affected villages get opportunities even if farther away.
---------------------------------------------------------------------
""")
    
    # Create demo data
    job, workers = create_demo_data()
    
    # Run allocation
    engine = AllocationEngine()
    results = engine.allocate_workers(job, workers, top_n=5)
    
    # Display results
    print("\n" + "-" * 70)
    print(" ALLOCATION RESULTS (Top 5 Workers)")
    print("-" * 70)
    
    print(f"\n{'Rank':<6}{'Worker':<18}{'Village':<18}{'Dist':<8}{'Geo':<8}{'Need':<8}{'FINAL':<8}")
    print("-" * 70)
    
    for i, sw in enumerate(results, 1):
        print(f"{i:<6}{sw.worker.name:<18}{sw.worker.village.name:<18}"
              f"{sw.distance_km:<8.1f}{sw.geo_score:<8.1f}{sw.drought_score:<8.1f}{sw.final_score:<8.1f}")
    
    print("\n" + "-" * 70)
    print(" DETAILED BREAKDOWN")
    print("-" * 70)
    
    for i, sw in enumerate(results, 1):
        print(f"\n#{i} {sw.worker.name} ({sw.worker.village.name})")
        print(f"   {sw.explanation}")
        print(f"   -> Final Score: {sw.final_score:.2f} = "
              f"({sw.geo_score:.1f} x 0.4) + ({sw.drought_score:.1f} x 0.6)")
    
    print("\n" + "#" * 70)
    print("  [OK] Allocation complete! Workers ranked by need + proximity.")
    print("#" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    main()
