# Part 2: Technical Design Document (SDD)

## 1. System Architecture

The system follows a **3-Layered Architecture** designed to bridge the digital divide and ensure data integrity.

### High-Level Components

1. **Interface Layer (Telephony/IVR)**
2. **Application Logic Layer (FastAPI Backend)**
3. **Data & Intelligence Layer (AI Models & Database)**

## 2. Component Design

### 2.1 Interface Layer

* 
**Technology:** Twilio / Exotel.


* **Function:** Handles inbound calls from feature phones.
* **Workflow:**
1. User calls toll-free number.
2. Voice audio is streamed to the Language AI service.
3. System responds with synthesized speech (Text-to-Speech) in the local dialect.



### 2.2 Intelligence & AI Services

* **Speech Processing (ASR):**
* 
**Tech:** Vosk (Offline ASR) or Bhashini.


* 
**Role:** Converts local dialects (Speech) to Text for intent analysis.




* **Generative AI & GIS:**
* 
**Role:** Analyzes satellite data to identify terrain features (low-lying areas) and generates compliant infrastructure plans.




* **Predictive Analytics:**
* 
**Tech:** Facebook Prophet / ARIMA.


* 
**Role:** Forecasts budget exhaustion dates based on current spending velocity.





### 2.3 Backend Logic

* 
**Tech:** Python (FastAPI).


* **Core Logic:**
* 
**Matching Engine:** Matches worker availability with either Government projects or Private Farmer demand based on the "Harvest Pause" status.


* 
**Bias Check:** Compares "Need Scores" of applicants against allocation results to trigger alerts.





### 2.4 Data Storage & Integrity

* 
**Primary Database:** PostgreSQL.


* 
**Audit Trail:** Hash Chain / Blockchain Ledger.


* 
**Role:** Stores data with immutable audit trails to prevent "Ghost Works" and ensure the fairness of job allocation.





## 3. Data Flow: The "Harvest Hero" Scenario

This flow describes the system behavior during the 60-day government work pause (October).

1. 
**Input:** Ramesh calls the system and asks, "Sarkar, is there work?".


2. **Processing (Mazdoor Mitra):**
* ASR converts speech to text.
* Backend checks `Government_Status`. Result: `PAUSED`.


* Backend checks `Private_Demand`. Result: `HIGH`.




3. 
**Matching:** The engine identifies a local private farmer (Farmer Singh) needing harvest help at ₹350/day.


4. 
**Response:** Voice AI responds, "Government work is paused... Farmer Singh needs 3 people... Press 1 to accept".


5. **Transaction:**
* Ramesh presses 1.
* System sends SMS with location details.


* Transaction is hashed and logged on the **Fairness Ledger** for the Block Officer to view.





4. Technical Stack Summary 

| Component | Technology | Role |
| --- | --- | --- |
| **Interface** | Twilio / Exotel | Voice IVR (Telephony) |
| **Language AI** | Vosk (Offline ASR) / Bhashini | Speech-to-Text for local dialects |
| **Backend Logic** | Python (FastAPI) | Matching algorithms & Business Logic |
| **Forecasting** | Facebook Prophet | Predicting budget exhaustion dates |
| **Database** | PostgreSQL + Hash Chain | Storing data with immutable audit trails |