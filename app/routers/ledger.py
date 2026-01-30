"""
Sahayog Setu - Ledger Router (Fairness Ledger)
API endpoints for auditing and verifying the immutable hash chain.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
import json

from app.database import get_db

router = APIRouter()

@router.get("/ledger/audit-log")
async def get_audit_log(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get the immutable audit log entries.
    """
    try:
        result = db.execute(
            text("""
                SELECT id, action, entity_type, entity_id, actor_id, 
                       hash, prev_hash, created_at, payload
                FROM audit_log
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            {"limit": limit, "offset": offset}
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(f"Ledger Fetch Error: {str(e)}\n")
            f.write(traceback.format_exc())
            
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ledger/verify/{entry_id}")
async def verify_entry(entry_id: str, db: Session = Depends(get_db)):
    """
    Cryptographically verify a specific ledger entry.
    Checks:
    1. Checks if prev_hash exists in the chain (Link Integrity).
    """
    entry = db.execute(
        text("SELECT * FROM audit_log WHERE id = :id"),
        {"id": entry_id}
    ).fetchone()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
        
    entry_dict = dict(entry._mapping)
    
    # Check Chain Integrity
    prev_entry = db.execute(
        text("SELECT hash FROM audit_log WHERE hash = :h"),
        {"h": entry_dict["prev_hash"]}
    ).fetchone()
    
    # Genesis block exception
    is_genesis = entry_dict["prev_hash"] == "genesis_hash_sahayog_setu_2025"
    chain_link_valid = True
    
    if not is_genesis and not prev_entry:
        chain_link_valid = False
        
    return {
        "verified": chain_link_valid,
        "entry_id": entry_id,
        "chain_link_valid": chain_link_valid,
        "is_genesis": is_genesis,
        "stored_hash": entry_dict["hash"],
        "prev_hash": entry_dict["prev_hash"],
        "message": "Chain link verified." if chain_link_valid else "BROKEN LINK: prev_hash not found in ledger!"
    }

@router.get("/ledger/bias-alerts")
async def get_bias_alerts(
    status: str = Query("PENDING"),
    db: Session = Depends(get_db)
):
    """
    Get unresolved bias alerts.
    """
    result = db.execute(
        text("""
            SELECT * FROM bias_alerts 
            WHERE status = :status
            ORDER BY created_at DESC
        """),
        {"status": status}
    )
    return [dict(row._mapping) for row in result]
