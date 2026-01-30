# Part 1: Product Requirements Document (PRD)

## 1. Project Overview

* 
**Project Title:** SAHAYOG SETU 


* 
**Subtitle:** A 360° Livelihood Grid Bridging the Gap from Rights to Assets in the New Era of VB-G RAM G.


* 
**Vision:** To transition India’s rural employment landscape from the rights-based MGNREGA era to the asset-focused Viksit Bharat era by creating a livelihood grid that ensures no worker is left behind due to digital or policy gaps.



## 2. Problem Statement

The transition to the **VB-G RAM G Bill 2025** shifts focus from demand-driven work to outcome-driven asset creation. While visionary, this introduces three critical structural gaps ("The Villains"):

1. 
**The "Agricultural Pause" Gap (The Income Cliff):** The bill mandates a 60-day work pause during harvest seasons, leaving landless laborers with zero income.


2. 
**The "Normative Allocation" Gap (The Budget Wall):** Fixed budget caps mean that if funds are spent too fast, they dry up by December, excluding the poorest who seek work late in the year.


3. 
**The "Viksit Plan" Gap (The Capability Void):** Work approval requires complex "Viksit Gram Panchayat Plans" (VGPP), which village heads (Sarpanchs) often lack the technical expertise to design, leading to rejections.



## 3. User Personas

* **Ramesh (The Vulnerable Worker):** A 45-year-old semi-literate landless laborer with a feature phone. He faces poverty during the 60-day government work pause and struggles to find private farm work.


* **Sarpanch Devi (The Overwhelmed Leader):** A newly elected village head who wants to help but is confused by digital planning rules. She fears exhausting her capped budget prematurely.


* **The Block Officer (The Auditor):** A government official managing 50 villages. They struggle to physically verify sites for "Ghost Works" or monitor for caste-based exclusion.



## 4. Functional Requirements

### Module A: "Mazdoor Mitra" (Worker Facing)

* 
**Objective:** Solve the "Income Cliff" using offline access and private-public matching.


* 
**Voice-First Interface:** Users must be able to speak in local dialects (e.g., Bhojpuri/Hindi) via a toll-free number to request work without touching a screen.


* **"Harvest Hero" Feature:**
* System must detect when government work is paused (Status = PAUSED).


* System must automatically switch the worker's profile to a "Private Labor Marketplace".


* System must match workers with local private farmers needing harvest help and convey wages/location via Voice/SMS.





### Module B: "Gram Sahayak" (Leader Facing)

* 
**Objective:** Solve the "Capability Void" via automation.


* **"Smart-Plan Auto-Pilot":**
* Allow leaders to input simple requests (e.g., "We need water storage").


* Utilize Generative AI and Satellite GIS data to scan for optimal locations (e.g., low-lying areas).


* Auto-generate a compliant VGPP Plan aligned with PM Gati Shakti rules to unlock funding.





### Module C: "Drishti Dashboard" (System Facing)

* 
**Objective:** Solve the "Budget Wall" and ensure fairness.


* **"Fund Flow Forecaster":**
* Use predictive analytics to monitor spending velocity.


* Alert administration if spending is too fast and suggest rationing strategies to ensure funds last the full year.




* **"Fairness Ledger":**
* Record every job allocation on an immutable ledger.


* Flag "Bias Alerts" if a worker with a higher "Need Score" is skipped in favor of others (e.g., the Sarpanch's nephew).





## 5. Success Metrics

* 
**Worker Impact:** Continuous income security (365 days) regardless of government work pauses.


* 
**Government Impact:** 100% policy compliance with VB-G RAM G and prevention of budget exhaustion.


* 
**Economic Impact:** Efficient allocation of labor during critical harvest windows to solve farmer labor shortages.