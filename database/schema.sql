-- ============================================
-- SAHAYOG SETU - Database Schema for Supabase
-- ============================================
-- Run this in the Supabase SQL Editor
-- ============================================

-- Enable UUID extension (usually already enabled in Supabase)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- ENUM TYPES
-- ============================================

-- Government job status
CREATE TYPE government_job_status AS ENUM ('ACTIVE', 'PAUSED', 'COMPLETED');

-- Private demand status
CREATE TYPE private_demand_status AS ENUM ('OPEN', 'FILLED', 'CANCELLED');

-- Job allocation type
CREATE TYPE job_type AS ENUM ('GOVERNMENT', 'PRIVATE');

-- ============================================
-- TABLES
-- ============================================

-- 1. WORKERS (Mazdoor) - The vulnerable laborers like Ramesh
CREATE TABLE workers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    aadhaar_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 hash of Aadhaar for privacy
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(15) NOT NULL,
    village_code VARCHAR(20) NOT NULL,
    dialect VARCHAR(50) DEFAULT 'hindi', -- e.g., 'bhojpuri', 'hindi', 'maithili'
    need_score DECIMAL(5,2) DEFAULT 50.00, -- 0-100 scale, higher = more need
    is_available BOOLEAN DEFAULT TRUE,
    family_size INTEGER DEFAULT 1,
    land_owned_acres DECIMAL(6,2) DEFAULT 0.00,
    days_since_last_work INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast village-based queries
CREATE INDEX idx_workers_village ON workers(village_code);
CREATE INDEX idx_workers_available ON workers(is_available) WHERE is_available = TRUE;
CREATE INDEX idx_workers_need_score ON workers(need_score DESC);

-- 2. FARMERS - Private farmers who post work demands
CREATE TABLE farmers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(15) NOT NULL,
    village_code VARCHAR(20) NOT NULL,
    land_area_acres DECIMAL(8,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_farmers_village ON farmers(village_code);

-- 3. GOVERNMENT_JOBS - MGNREGA / VB-G RAM G projects
CREATE TABLE government_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_code VARCHAR(50) UNIQUE NOT NULL,
    village_code VARCHAR(20) NOT NULL,
    panchayat_code VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    status government_job_status DEFAULT 'ACTIVE',
    daily_wage DECIMAL(10,2) NOT NULL DEFAULT 350.00,
    workers_needed INTEGER NOT NULL DEFAULT 10,
    workers_allocated INTEGER DEFAULT 0,
    budget_allocated DECIMAL(15,2) NOT NULL,
    budget_spent DECIMAL(15,2) DEFAULT 0.00,
    start_date DATE NOT NULL,
    end_date DATE,
    pause_reason TEXT, -- Reason for PAUSED status (e.g., "Harvest Season")
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_govt_jobs_village ON government_jobs(village_code);
CREATE INDEX idx_govt_jobs_status ON government_jobs(status);
CREATE INDEX idx_govt_jobs_panchayat ON government_jobs(panchayat_code);

-- 4. PRIVATE_DEMANDS - "Harvest Hero" feature - private farmer work requests
CREATE TABLE private_demands (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id UUID NOT NULL REFERENCES farmers(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    work_type VARCHAR(100) NOT NULL, -- e.g., 'harvesting', 'sowing', 'irrigation'
    workers_needed INTEGER NOT NULL,
    workers_allocated INTEGER DEFAULT 0,
    daily_wage DECIMAL(10,2) NOT NULL,
    work_date DATE NOT NULL,
    work_duration_days INTEGER DEFAULT 1,
    location_gps VARCHAR(50), -- "lat,lng" format
    location_description TEXT, -- Human readable location
    status private_demand_status DEFAULT 'OPEN',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_private_demands_farmer ON private_demands(farmer_id);
CREATE INDEX idx_private_demands_status ON private_demands(status) WHERE status = 'OPEN';
CREATE INDEX idx_private_demands_date ON private_demands(work_date);

-- 5. JOB_ALLOCATIONS - Records of worker-job matching (Fairness Ledger entries)
CREATE TABLE job_allocations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    worker_id UUID NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
    government_job_id UUID REFERENCES government_jobs(id) ON DELETE SET NULL,
    private_demand_id UUID REFERENCES private_demands(id) ON DELETE SET NULL,
    job_type job_type NOT NULL,
    worker_need_score_at_allocation DECIMAL(5,2) NOT NULL, -- Snapshot for audit
    daily_wage DECIMAL(10,2) NOT NULL,
    work_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'ALLOCATED', -- ALLOCATED, COMPLETED, CANCELLED
    -- Hash chain fields for immutable audit trail
    hash VARCHAR(64) NOT NULL,
    prev_hash VARCHAR(64) NOT NULL,
    allocated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Ensure only one job type is linked
    CONSTRAINT chk_job_reference CHECK (
        (government_job_id IS NOT NULL AND private_demand_id IS NULL) OR
        (government_job_id IS NULL AND private_demand_id IS NOT NULL)
    )
);

CREATE INDEX idx_allocations_worker ON job_allocations(worker_id);
CREATE INDEX idx_allocations_govt_job ON job_allocations(government_job_id);
CREATE INDEX idx_allocations_private ON job_allocations(private_demand_id);
CREATE INDEX idx_allocations_date ON job_allocations(work_date);
CREATE INDEX idx_allocations_hash ON job_allocations(hash);

-- 6. AUDIT_LOG - General audit trail for all system actions
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    action VARCHAR(100) NOT NULL, -- e.g., 'JOB_ALLOCATED', 'STATUS_CHANGED', 'BIAS_ALERT'
    entity_type VARCHAR(50) NOT NULL, -- e.g., 'worker', 'job_allocation', 'government_job'
    entity_id UUID NOT NULL,
    actor_type VARCHAR(50), -- 'system', 'worker', 'sarpanch', 'block_officer'
    actor_id UUID,
    payload JSONB NOT NULL, -- Detailed action data
    -- Hash chain fields
    hash VARCHAR(64) NOT NULL,
    prev_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_action ON audit_log(action);
CREATE INDEX idx_audit_created ON audit_log(created_at DESC);

-- 7. BIAS_ALERTS - Flagged allocations for Block Officer review
CREATE TABLE bias_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    allocation_id UUID NOT NULL REFERENCES job_allocations(id) ON DELETE CASCADE,
    skipped_worker_id UUID NOT NULL REFERENCES workers(id),
    allocated_worker_id UUID NOT NULL REFERENCES workers(id),
    skipped_need_score DECIMAL(5,2) NOT NULL,
    allocated_need_score DECIMAL(5,2) NOT NULL,
    score_difference DECIMAL(5,2) NOT NULL, -- How much higher the skipped worker's score was
    alert_reason TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, REVIEWED, DISMISSED, CONFIRMED
    reviewed_by UUID,
    reviewed_at TIMESTAMPTZ,
    review_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_bias_alerts_status ON bias_alerts(status) WHERE status = 'PENDING';
CREATE INDEX idx_bias_alerts_allocation ON bias_alerts(allocation_id);

-- 8. BUDGET_TRACKING - For Fund Flow Forecaster
CREATE TABLE budget_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    panchayat_code VARCHAR(20) NOT NULL,
    fiscal_year VARCHAR(9) NOT NULL, -- e.g., '2025-2026'
    total_budget DECIMAL(15,2) NOT NULL,
    budget_spent DECIMAL(15,2) DEFAULT 0.00,
    budget_remaining DECIMAL(15,2) GENERATED ALWAYS AS (total_budget - budget_spent) STORED,
    last_spending_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(panchayat_code, fiscal_year)
);

CREATE INDEX idx_budget_panchayat ON budget_tracking(panchayat_code);

-- 9. DAILY_SPENDING - For forecasting (Fund Flow Forecaster data)
CREATE TABLE daily_spending (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    panchayat_code VARCHAR(20) NOT NULL,
    spending_date DATE NOT NULL,
    amount_spent DECIMAL(15,2) NOT NULL,
    workers_paid INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(panchayat_code, spending_date)
);

CREATE INDEX idx_spending_panchayat_date ON daily_spending(panchayat_code, spending_date DESC);

-- ============================================
-- GENESIS RECORD FOR HASH CHAIN
-- ============================================

-- Insert genesis record for audit log hash chain
INSERT INTO audit_log (id, action, entity_type, entity_id, payload, hash, prev_hash)
VALUES (
    '00000000-0000-0000-0000-000000000000',
    'GENESIS',
    'system',
    '00000000-0000-0000-0000-000000000000',
    '{"message": "Sahayog Setu Fairness Ledger Genesis Block", "version": "1.0.0"}',
    'genesis_hash_sahayog_setu_2025',
    '0000000000000000000000000000000000000000000000000000000000000000'
);

-- ============================================
-- ROW LEVEL SECURITY (RLS) - Supabase specific
-- ============================================

-- Enable RLS on all tables
ALTER TABLE workers ENABLE ROW LEVEL SECURITY;
ALTER TABLE farmers ENABLE ROW LEVEL SECURITY;
ALTER TABLE government_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE private_demands ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE bias_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE budget_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_spending ENABLE ROW LEVEL SECURITY;

-- For now, allow all operations (you can restrict these later based on auth)
-- These policies allow the service role to access everything
CREATE POLICY "Allow all for service role" ON workers FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON farmers FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON government_jobs FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON private_demands FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON job_allocations FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON audit_log FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON bias_alerts FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON budget_tracking FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON daily_spending FOR ALL USING (true);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_workers_updated_at BEFORE UPDATE ON workers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_farmers_updated_at BEFORE UPDATE ON farmers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_government_jobs_updated_at BEFORE UPDATE ON government_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_private_demands_updated_at BEFORE UPDATE ON private_demands
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_budget_tracking_updated_at BEFORE UPDATE ON budget_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

-- View: Available workers with high need scores
CREATE VIEW available_workers_by_need AS
SELECT 
    id,
    name,
    phone,
    village_code,
    dialect,
    need_score,
    days_since_last_work,
    family_size
FROM workers
WHERE is_available = TRUE
ORDER BY need_score DESC;

-- View: Open private demands with farmer info
CREATE VIEW open_private_demands AS
SELECT 
    pd.id,
    pd.description,
    pd.work_type,
    pd.workers_needed,
    pd.workers_allocated,
    (pd.workers_needed - pd.workers_allocated) AS workers_remaining,
    pd.daily_wage,
    pd.work_date,
    pd.location_description,
    f.name AS farmer_name,
    f.phone AS farmer_phone,
    f.village_code
FROM private_demands pd
JOIN farmers f ON pd.farmer_id = f.id
WHERE pd.status = 'OPEN'
ORDER BY pd.work_date ASC;

-- View: Government work status by village
CREATE VIEW village_work_status AS
SELECT 
    village_code,
    COUNT(*) FILTER (WHERE status = 'ACTIVE') AS active_projects,
    COUNT(*) FILTER (WHERE status = 'PAUSED') AS paused_projects,
    SUM(workers_needed) FILTER (WHERE status = 'ACTIVE') AS workers_needed,
    CASE 
        WHEN COUNT(*) FILTER (WHERE status = 'ACTIVE') > 0 THEN 'ACTIVE'
        WHEN COUNT(*) FILTER (WHERE status = 'PAUSED') > 0 THEN 'PAUSED'
        ELSE 'NO_PROJECTS'
    END AS overall_status
FROM government_jobs
WHERE end_date IS NULL OR end_date >= CURRENT_DATE
GROUP BY village_code;

-- ============================================
-- SUCCESS MESSAGE
-- ============================================
-- Schema created successfully! 🎉
-- Tables: 9 | Views: 3 | Functions: 1 | Triggers: 5
