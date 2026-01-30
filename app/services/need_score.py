"""
Sahayog Setu - Need Score Calculator
Logic to calculate worker priority based on poverty indicators.
"""

from typing import Optional

def calculate_need_score(
    days_since_last_work: int,
    family_size: int,
    land_owned_acres: float,
    earnings_last_30_days: Optional[float] = 0.0
) -> float:
    """
    Calculate a Need Score (0-100) for a worker.
    Higher Score = Higher Priority for work allocation.
    
    Weights:
    - Income Urgency (Days since work): 40%
    - Dependency Burden (Family Size): 30%
    - Asset Vulnerability (Land): 30%
    """
    
    # 1. Income Urgency (0-40 points)
    # 30+ days without work gets max points
    income_score = min(days_since_last_work, 30) / 30 * 40
    
    # 2. Dependency Burden (0-30 points)
    # Family of 6+ gets max points
    # Formula: (size / 6) * 30
    family_score = min(family_size, 6) / 6 * 30
    
    # 3. Asset Vulnerability (0-30 points)
    # 0 acres = 30 points (Landless)
    # 5+ acres = 0 points
    if land_owned_acres >= 5:
        asset_score = 0
    else:
        # Linear decay: 5 acres -> 0 pts, 0 acres -> 30 pts
        asset_score = (5 - land_owned_acres) / 5 * 30
        
    total_score = income_score + family_score + asset_score
    
    # Bonus: If earnings are very low (< ₹1000/month), add 5 points buffer
    if earnings_last_30_days < 1000:
        total_score = min(total_score + 5, 100)
        
    return round(total_score, 2)
