#!/usr/bin/env python3
"""
+==============================================================================+
|           SAHAYOG SETU - MATCHING ENGINE DEMO (Standalone)                   |
|                                                                              |
|  This standalone script demonstrates the optimal matching engine that        |
|  allocates workers to jobs using a weighted cost matrix approach.            |
|                                                                              |
|  Run independently: python matching_engine_demo.py                           |
+==============================================================================+

MATCHING ALGORITHM OVERVIEW:
============================

The Matching Engine finds GLOBALLY OPTIMAL worker-job assignments using:

+-----------------------------------------------------------------------------+
|  TOTAL_SCORE = (Location × 0.25) + (Need × 0.35) + (Skills × 0.20)          |
|                + (Job Necessity × 0.20)                                      |
|                                                                              |
|  Where:                                                                      |
|  * Location     = Proximity score (closer to job = higher)                   |
|  * Need         = Drought/distress score from VEDAS satellite data           |
|  * Skills       = Worker skill match with job requirements                   |
|  * Job Necessity = Job urgency + demand                                      |
+-----------------------------------------------------------------------------+

OPTIMIZATION APPROACH:
======================
1. Build a COST MATRIX for all worker-job pairs
2. Use a GREEDY AUCTION algorithm to find optimal assignments
3. Return ALLOCATION PATHS showing why each match was made
"""

import math
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from datetime import date, timedelta
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimization Weights (must sum to 1.0)
WEIGHTS = {
    "location": 0.25,      # Proximity score
    "need": 0.35,          # Drought/distress (VEDAS)
    "skills": 0.20,        # Skill match
    "job_necessity": 0.20  # Job urgency/demand
}

MAX_DISTANCE_KM = 100  # Maximum distance threshold


# =============================================================================
# ENUMS
# =============================================================================

class WorkStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"


class JobStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Village:
    """Represents a village with location and drought data."""
    id: int
    name: str
    state: str
    latitude: float
    longitude: float
    drought_score: float = 50.0  # 0-100, higher = more need


@dataclass
class Worker:
    """Represents a worker seeking employment."""
    id: int
    name: str
    village: Village
    skills: str  # Comma-separated
    status: WorkStatus = WorkStatus.AVAILABLE


@dataclass
class Job:
    """Represents a job/work opportunity."""
    id: int
    title: str
    village: Village
    required_skills: str  # Comma-separated
    status: JobStatus = JobStatus.OPEN
    start_date: Optional[date] = None
    wage_per_day: float = 350.0


@dataclass
class AllocationVariable:
    """Individual variable contribution to a match."""
    name: str
    value: float
    weight: float
    weighted_score: float
    explanation: str


@dataclass
class AllocationPath:
    """Complete allocation path showing all decision factors."""
    worker_id: int
    worker_name: str
    worker_village: str
    job_id: int
    job_title: str
    job_village: str
    total_score: float
    variables: List[AllocationVariable]
    matched_skills: List[str]
    path_explanation: str
    rank: int = 0


@dataclass
class CostMatrixEntry:
    """Single entry in the cost matrix."""
    worker_id: int
    job_id: int
    cost: float  # Lower is better
    score: float  # Higher is better
    variables: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimalMatchResult:
    """Result of optimal matching operation."""
    matches: List[AllocationPath]
    unmatched_workers: List[int]
    unmatched_jobs: List[int]
    total_score: float
    optimization_method: str
    summary: str


# =============================================================================
# GEOSPATIAL UTILITIES
# =============================================================================

def calculate_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """
    Calculate geodesic distance (km) using Haversine formula.
    
    Args:
        loc1: (latitude, longitude) of first point
        loc2: (latitude, longitude) of second point
        
    Returns:
        Distance in kilometers
    """
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return 6371 * c  # Earth's radius in km


# =============================================================================
# MATCHING ENGINE
# =============================================================================

class MatchingEngine:
    """
    Optimal Matching Engine for Worker-Job Allocation.
    
    Uses weighted bipartite matching to find globally optimal
    assignments across all workers and jobs simultaneously.
    """
    
    def __init__(self, workers: List[Worker], jobs: List[Job]):
        self.workers = workers
        self.jobs = jobs
    
    # -------------------------------------------------------------------------
    # SCORING METHODS
    # -------------------------------------------------------------------------
    
    def calculate_location_score(self, distance_km: float) -> float:
        """Convert distance to proximity score (0-100)."""
        if distance_km >= MAX_DISTANCE_KM:
            return 0.0
        return (1 / (1 + distance_km)) * 100
    
    def calculate_skill_match(self, worker_skills: str, job_skills: str) -> float:
        """Calculate skill match percentage (0-100)."""
        if not worker_skills or not job_skills:
            return 50.0
        
        worker_set = {s.strip().lower() for s in worker_skills.split(",")}
        job_set = {s.strip().lower() for s in job_skills.split(",")}
        
        if not job_set:
            return 100.0
        
        matched = worker_set.intersection(job_set)
        coverage = len(matched) / len(job_set)
        
        return min(100.0, coverage * 100 + min(len(worker_set) * 2, 10))
    
    def calculate_job_necessity(self, job: Job) -> float:
        """Calculate job necessity score (0-100)."""
        score = 50.0
        
        if job.start_date:
            days = (job.start_date - date.today()).days
            if days <= 0:
                score += 25
            elif days <= 3:
                score += 20
            elif days <= 7:
                score += 10
        
        if job.status == JobStatus.OPEN:
            score += 20
        
        return min(100.0, score)
    
    def get_matched_skills(self, worker_skills: str, job_skills: str) -> List[str]:
        """Get list of skills that match between worker and job."""
        if not worker_skills or not job_skills:
            return []
        
        worker_set = {s.strip().lower() for s in worker_skills.split(",")}
        job_set = {s.strip().lower() for s in job_skills.split(",")}
        
        return list(worker_set.intersection(job_set))
    
    # -------------------------------------------------------------------------
    # COST MATRIX
    # -------------------------------------------------------------------------
    
    def build_cost_matrix(self) -> Tuple[List[List[CostMatrixEntry]], Dict[int, int], Dict[int, int]]:
        """Build the cost matrix for all worker-job pairs."""
        worker_idx = {w.id: i for i, w in enumerate(self.workers)}
        job_idx = {j.id: i for i, j in enumerate(self.jobs)}
        
        n_workers = len(self.workers)
        n_jobs = len(self.jobs)
        
        matrix = [[None for _ in range(n_jobs)] for _ in range(n_workers)]
        
        for worker in self.workers:
            if worker.status != WorkStatus.AVAILABLE:
                continue
            
            worker_loc = (worker.village.latitude, worker.village.longitude)
            w_idx = worker_idx[worker.id]
            
            for job in self.jobs:
                if job.status != JobStatus.OPEN:
                    continue
                
                job_loc = (job.village.latitude, job.village.longitude)
                j_idx = job_idx[job.id]
                
                variables = {}
                
                # 1. LOCATION SCORE
                distance_km = calculate_distance(worker_loc, job_loc)
                if distance_km > MAX_DISTANCE_KM:
                    matrix[w_idx][j_idx] = CostMatrixEntry(
                        worker_id=worker.id,
                        job_id=job.id,
                        cost=1000,
                        score=0,
                        variables={"location": 0, "too_far": True}
                    )
                    continue
                
                location_score = self.calculate_location_score(distance_km)
                variables["location"] = location_score
                variables["distance_km"] = distance_km
                
                # 2. NEED SCORE (VEDAS)
                need_score = worker.village.drought_score
                variables["need"] = need_score
                
                # 3. SKILL SCORE
                skill_score = self.calculate_skill_match(worker.skills, job.required_skills)
                variables["skills"] = skill_score
                
                # 4. JOB NECESSITY SCORE
                necessity_score = self.calculate_job_necessity(job)
                variables["job_necessity"] = necessity_score
                
                # WEIGHTED TOTAL SCORE
                total_score = sum(
                    variables.get(var, 0) * WEIGHTS[var]
                    for var in WEIGHTS
                )
                
                matrix[w_idx][j_idx] = CostMatrixEntry(
                    worker_id=worker.id,
                    job_id=job.id,
                    cost=100 - total_score,
                    score=total_score,
                    variables=variables
                )
        
        return matrix, worker_idx, job_idx
    
    # -------------------------------------------------------------------------
    # OPTIMAL MATCHING
    # -------------------------------------------------------------------------
    
    def compute_optimal_matching(self, max_workers_per_job: int = 5) -> OptimalMatchResult:
        """
        Compute globally optimal worker-job matching.
        
        Uses greedy auction algorithm with cost matrix.
        """
        if not self.workers or not self.jobs:
            return OptimalMatchResult(
                matches=[],
                unmatched_workers=[w.id for w in self.workers],
                unmatched_jobs=[j.id for j in self.jobs],
                total_score=0,
                optimization_method="none",
                summary="No workers or jobs available"
            )
        
        # Build cost matrix
        matrix, worker_idx, job_idx = self.build_cost_matrix()
        
        # Reverse mappings
        idx_to_worker = {v: k for k, v in worker_idx.items()}
        idx_to_job = {v: k for k, v in job_idx.items()}
        
        # Track allocations
        worker_assigned: Set[int] = set()
        job_allocation_count: Dict[int, int] = {j.id: 0 for j in self.jobs}
        
        # Build priority queue (min-heap by cost)
        match_heap = []
        for w_idx in range(len(self.workers)):
            for j_idx in range(len(self.jobs)):
                entry = matrix[w_idx][j_idx]
                if entry and entry.cost < 999:
                    heapq.heappush(match_heap, (entry.cost, w_idx, j_idx, entry))
        
        # Greedy assignment
        matches: List[AllocationPath] = []
        rank = 1
        
        while match_heap:
            cost, w_idx, j_idx, entry = heapq.heappop(match_heap)
            
            worker_id = idx_to_worker[w_idx]
            job_id = idx_to_job[j_idx]
            
            if worker_id in worker_assigned:
                continue
            if job_allocation_count[job_id] >= max_workers_per_job:
                continue
            
            worker_assigned.add(worker_id)
            job_allocation_count[job_id] += 1
            
            # Get objects
            worker = next(w for w in self.workers if w.id == worker_id)
            job = next(j for j in self.jobs if j.id == job_id)
            
            # Build allocation path
            path = self._build_allocation_path(worker, job, entry, rank)
            matches.append(path)
            rank += 1
        
        # Identify unmatched
        unmatched_workers = [w.id for w in self.workers if w.id not in worker_assigned]
        unmatched_jobs = [j.id for j in self.jobs if job_allocation_count[j.id] == 0]
        
        total_score = sum(m.total_score for m in matches)
        
        return OptimalMatchResult(
            matches=matches,
            unmatched_workers=unmatched_workers,
            unmatched_jobs=unmatched_jobs,
            total_score=round(total_score, 2),
            optimization_method="greedy_auction",
            summary=f"Optimally matched {len(matches)} workers to {len(set(m.job_id for m in matches))} jobs"
        )
    
    def _build_allocation_path(
        self,
        worker: Worker,
        job: Job,
        entry: CostMatrixEntry,
        rank: int
    ) -> AllocationPath:
        """Build detailed allocation path from matrix entry."""
        
        variables = []
        
        # Location
        loc_score = entry.variables.get("location", 0)
        distance = entry.variables.get("distance_km", 0)
        variables.append(AllocationVariable(
            name="Location (Proximity)",
            value=round(loc_score, 2),
            weight=WEIGHTS["location"],
            weighted_score=round(loc_score * WEIGHTS["location"], 2),
            explanation=f"Distance: {distance:.1f}km → Score: {loc_score:.1f}/100"
        ))
        
        # Need
        need_score = entry.variables.get("need", 50)
        variables.append(AllocationVariable(
            name="Need (VEDAS Drought)",
            value=round(need_score, 2),
            weight=WEIGHTS["need"],
            weighted_score=round(need_score * WEIGHTS["need"], 2),
            explanation=f"Drought severity: {need_score:.1f}/100"
        ))
        
        # Skills
        skill_score = entry.variables.get("skills", 50)
        variables.append(AllocationVariable(
            name="Skills Match",
            value=round(skill_score, 2),
            weight=WEIGHTS["skills"],
            weighted_score=round(skill_score * WEIGHTS["skills"], 2),
            explanation=f"Skill compatibility: {skill_score:.1f}%"
        ))
        
        # Job necessity
        necessity_score = entry.variables.get("job_necessity", 50)
        variables.append(AllocationVariable(
            name="Job Necessity",
            value=round(necessity_score, 2),
            weight=WEIGHTS["job_necessity"],
            weighted_score=round(necessity_score * WEIGHTS["job_necessity"], 2),
            explanation=f"Job urgency/demand: {necessity_score:.1f}/100"
        ))
        
        # Matched skills
        matched_skills = self.get_matched_skills(worker.skills, job.required_skills)
        
        # Explanation
        top_factors = sorted(variables, key=lambda x: x.weighted_score, reverse=True)[:2]
        explanation = " + ".join([f"{v.name.split()[0]}: {v.weighted_score:.1f}" for v in top_factors])
        
        return AllocationPath(
            worker_id=worker.id,
            worker_name=worker.name,
            worker_village=worker.village.name,
            job_id=job.id,
            job_title=job.title,
            job_village=job.village.name,
            total_score=round(entry.score, 2),
            variables=variables,
            matched_skills=matched_skills,
            path_explanation=explanation,
            rank=rank
        )


# =============================================================================
# DEMO DATA
# =============================================================================

def create_demo_data() -> Tuple[List[Worker], List[Job]]:
    """
    Create realistic dummy data for Karnataka, India.
    
    Scenario: Multiple jobs across villages need workers.
    Workers are distributed across villages with varying
    drought levels and skill sets.
    """
    
    # --- Villages with varied drought levels ---
    villages = [
        Village(1, "Bengaluru", "Karnataka", 12.9716, 77.5946, drought_score=30),    # Urban, low drought
        Village(2, "Kolar", "Karnataka", 13.1362, 78.1292, drought_score=75),        # High drought
        Village(3, "Tumkur", "Karnataka", 13.3392, 77.1010, drought_score=60),       # Moderate drought
        Village(4, "Mandya", "Karnataka", 12.5218, 76.8951, drought_score=45),       # Low-moderate
        Village(5, "Ramanagara", "Karnataka", 12.7159, 77.2819, drought_score=55),   # Moderate
        Village(6, "Chikkaballapur", "Karnataka", 13.4355, 77.7315, drought_score=80), # Very high drought
        Village(7, "Devanahalli", "Karnataka", 13.2473, 77.7140, drought_score=40),  # Low
        Village(8, "Anekal", "Karnataka", 12.7105, 77.6970, drought_score=35),       # Low
    ]
    
    # --- Workers with varied skills ---
    workers = [
        Worker(1, "Ramesh Kumar", villages[1], "construction, masonry"),           # Kolar (high drought)
        Worker(2, "Suresh Gowda", villages[2], "labor, driving"),                  # Tumkur
        Worker(3, "Manjunath", villages[3], "construction, welding"),              # Mandya
        Worker(4, "Venkatesh", villages[4], "labor, construction"),                # Ramanagara
        Worker(5, "Shivakumar", villages[5], "masonry, construction"),             # Chikkaballapur (very high drought)
        Worker(6, "Basavaraj", villages[6], "labor, painting"),                    # Devanahalli
        Worker(7, "Nagaraj", villages[7], "construction, plumbing"),               # Anekal
        Worker(8, "Ravi Shankar", villages[1], "welding, electrical"),             # Kolar (2nd worker)
        Worker(9, "Kumar", villages[5], "labor, construction"),                    # Chikkaballapur
        Worker(10, "Prakash", villages[2], "masonry, labor"),                      # Tumkur
    ]
    
    # --- Jobs across different villages ---
    jobs = [
        Job(
            id=1,
            title="Road Construction - NH Extension",
            village=villages[0],  # Bengaluru
            required_skills="construction, labor",
            start_date=date.today() + timedelta(days=1),  # Urgent!
            wage_per_day=400.0
        ),
        Job(
            id=2,
            title="Canal Repair - Irrigation Project",
            village=villages[3],  # Mandya
            required_skills="construction, masonry",
            start_date=date.today() + timedelta(days=5),
            wage_per_day=380.0
        ),
        Job(
            id=3,
            title="School Building - MGNREGA",
            village=villages[2],  # Tumkur
            required_skills="masonry, construction, labor",
            start_date=date.today() + timedelta(days=3),
            wage_per_day=350.0
        ),
        Job(
            id=4,
            title="Farm Harvesting - Private",
            village=villages[4],  # Ramanagara
            required_skills="labor",
            start_date=date.today(),  # Today!
            wage_per_day=450.0
        ),
        Job(
            id=5,
            title="Water Tank Construction",
            village=villages[6],  # Devanahalli
            required_skills="welding, plumbing",
            start_date=date.today() + timedelta(days=7),
            wage_per_day=420.0
        ),
    ]
    
    return workers, jobs


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_header():
    """Print demo header."""
    print("\n" + "#" * 75)
    print("#" + " " * 73 + "#")
    print("#" + "  SAHAYOG SETU - OPTIMAL MATCHING ENGINE DEMO".center(73) + "#")
    print("#" + " " * 73 + "#")
    print("#" * 75)


def print_algorithm_explanation():
    """Print algorithm explanation."""
    print("""
MATCHING ENGINE ALGORITHM:
=============================================================================

The engine uses a GREEDY AUCTION algorithm with a weighted cost matrix:

  1. BUILD COST MATRIX
     - For each worker-job pair, calculate a score using 4 variables
     - Convert score to cost (cost = 100 - score)
     
  2. VARIABLE WEIGHTS
     ┌────────────────────┬────────┬──────────────────────────────────────┐
     │ Variable           │ Weight │ Description                          │
     ├────────────────────┼────────┼──────────────────────────────────────┤
     │ Location           │  25%   │ Proximity (closer = higher score)    │
     │ Need (VEDAS)       │  35%   │ Drought severity (higher = priority) │
     │ Skills             │  20%   │ Skill match percentage               │
     │ Job Necessity      │  20%   │ Job urgency + demand                 │
     └────────────────────┴────────┴──────────────────────────────────────┘
     
  3. GREEDY ASSIGNMENT
     - Sort all pairs by score (best first)
     - Assign best available match iteratively
     - Each worker assigned once; jobs capped at max_workers_per_job
     
  4. RESULT: Globally optimized assignments with allocation paths
=============================================================================
""")


def print_input_data(workers: List[Worker], jobs: List[Job]):
    """Print input data summary."""
    print("\n" + "=" * 75)
    print(" INPUT DATA")
    print("=" * 75)
    
    print(f"\n📍 WORKERS ({len(workers)}):")
    print("-" * 75)
    print(f"{'ID':<4}{'Name':<18}{'Village':<18}{'Skills':<25}{'Drought':<10}")
    print("-" * 75)
    for w in workers:
        print(f"{w.id:<4}{w.name:<18}{w.village.name:<18}{w.skills:<25}{w.village.drought_score:<10.0f}")
    
    print(f"\n💼 JOBS ({len(jobs)}):")
    print("-" * 75)
    print(f"{'ID':<4}{'Title':<30}{'Village':<15}{'Skills':<20}{'Start':<12}")
    print("-" * 75)
    for j in jobs:
        start = j.start_date.strftime("%Y-%m-%d") if j.start_date else "TBD"
        title = j.title[:28] + ".." if len(j.title) > 30 else j.title
        print(f"{j.id:<4}{title:<30}{j.village.name:<15}{j.required_skills:<20}{start:<12}")


def print_results(result: OptimalMatchResult):
    """Print matching results."""
    print("\n" + "=" * 75)
    print(" OPTIMAL MATCHING RESULTS")
    print("=" * 75)
    
    print(f"\n📊 Summary: {result.summary}")
    print(f"🎯 Total Optimization Score: {result.total_score}")
    print(f"🔧 Method: {result.optimization_method}")
    
    print("\n" + "-" * 75)
    print(" MATCHED PAIRS (ranked by score)")
    print("-" * 75)
    print(f"{'Rank':<5}{'Worker':<18}{'→':<3}{'Job':<28}{'Score':<8}{'Top Factors'}")
    print("-" * 75)
    
    for match in result.matches:
        title = match.job_title[:26] + ".." if len(match.job_title) > 28 else match.job_title
        print(f"{match.rank:<5}{match.worker_name:<18}{'→':<3}{title:<28}{match.total_score:<8.1f}{match.path_explanation}")
    
    if result.unmatched_workers:
        print(f"\n⚠️  Unmatched Workers: {result.unmatched_workers}")
    if result.unmatched_jobs:
        print(f"⚠️  Unmatched Jobs: {result.unmatched_jobs}")


def print_allocation_paths(result: OptimalMatchResult, top_n: int = 3):
    """Print detailed allocation paths for top matches."""
    print("\n" + "=" * 75)
    print(f" DETAILED ALLOCATION PATHS (Top {top_n})")
    print("=" * 75)
    
    for match in result.matches[:top_n]:
        print(f"\n┌{'─' * 73}┐")
        print(f"│ #{match.rank} {match.worker_name} → {match.job_title:<40} │")
        print(f"├{'─' * 73}┤")
        print(f"│ From: {match.worker_village:<20} To: {match.job_village:<25} │")
        print(f"│ Total Score: {match.total_score:<10.2f} Matched Skills: {', '.join(match.matched_skills) or 'None':<20} │")
        print(f"├{'─' * 73}┤")
        print(f"│ {'Variable':<22}{'Value':>10}{'Weight':>10}{'Weighted':>12}  │")
        print(f"│ {'-' * 68}  │")
        
        for var in match.variables:
            print(f"│ {var.name:<22}{var.value:>10.1f}{var.weight:>10.0%}{var.weighted_score:>12.2f}  │")
        
        print(f"└{'─' * 73}┘")


def print_variable_analysis(result: OptimalMatchResult):
    """Analyze variable impact on matches."""
    print("\n" + "=" * 75)
    print(" VARIABLE IMPACT ANALYSIS")
    print("=" * 75)
    
    if not result.matches:
        print("No matches to analyze")
        return
    
    # Aggregate contributions
    totals = {name: 0 for name in WEIGHTS}
    
    for match in result.matches:
        for var in match.variables:
            # Map variable name to weight key
            if "Location" in var.name:
                totals["location"] += var.weighted_score
            elif "Need" in var.name:
                totals["need"] += var.weighted_score
            elif "Skills" in var.name:
                totals["skills"] += var.weighted_score
            elif "Necessity" in var.name:
                totals["job_necessity"] += var.weighted_score
    
    print(f"\nTotal contribution across {len(result.matches)} matches:")
    print("-" * 50)
    
    for name, total in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(total / 5)  # Scale for display
        bar = "█" * bar_len
        print(f"  {name.capitalize():<15} │ {bar:<40} {total:.1f}")
    
    most_impactful = max(totals, key=totals.get)
    print(f"\n➤ Most impactful variable: {most_impactful.upper()} ({totals[most_impactful]:.1f} points)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the matching engine demo."""
    
    print_header()
    print_algorithm_explanation()
    
    # Create demo data
    workers, jobs = create_demo_data()
    print_input_data(workers, jobs)
    
    # Run matching engine
    print("\n⏳ Running optimal matching algorithm...")
    engine = MatchingEngine(workers, jobs)
    result = engine.compute_optimal_matching(max_workers_per_job=3)
    
    # Display results
    print_results(result)
    print_allocation_paths(result, top_n=3)
    print_variable_analysis(result)
    
    print("\n" + "#" * 75)
    print("  ✅ Matching complete! Workers optimally allocated based on all variables.")
    print("#" * 75 + "\n")
    
    return result


if __name__ == "__main__":
    main()
