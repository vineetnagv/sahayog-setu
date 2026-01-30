
import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app.models.models import Village, Worker, Job, WorkStatus, JobStatus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Seeder")

def seed_data():
    logger.info("creating tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if data exists
        if db.query(Village).count() > 0:
            logger.info("Data already exists. Skipping seed.")
            return

        logger.info("Seeding Villages...")
        # 1. Villages (Mix of High Need and Low Need)
        # Coordinates are roughly around Bangalore/Karnataka
        v1 = Village(name="Kuduragere", latitude=13.1500, longitude=77.5800, drought_score=85.0, district="Bangalore Rural") # High Distress
        v2 = Village(name="Whitefield", latitude=12.9698, longitude=77.7500, drought_score=20.0, district="Bangalore Urban") # Low Distress
        v3 = Village(name="Nelamangala", latitude=13.0953, longitude=77.3964, drought_score=60.0, district="Bangalore Rural") # Medium
        v4 = Village(name="Kanakapura", latitude=12.5462, longitude=77.4197, drought_score=90.0, district="Ramanagara") # Very High Distress

        db.add_all([v1, v2, v3, v4])
        db.commit()

        logger.info("Seeding Workers...")
        # 2. Workers
        workers = [
            # High Need Village (Kuduragere) - Should be prioritized if job is close
            Worker(name="Ramesh Kumar", phone_number="9900112233", skills="Farming, Digging", village_id=v1.id),
            Worker(name="Suresh B", phone_number="9900112234", skills="Masonry", village_id=v1.id),
            
            # Low Need Village (Whitefield) - Should be lower priority
            Worker(name="Anil Tech", phone_number="8800112233", skills="Driver", village_id=v2.id),
            
            # Medium Need (Nelamangala)
            Worker(name="Ganesh M", phone_number="7700112233", skills="Carpenter", village_id=v3.id),
            
            # Very High Need (Kanakapura) - Far away but high need
            Worker(name="Mahesh D", phone_number="6600112233", skills="Farming", village_id=v4.id),
        ]
        db.add_all(workers)
        db.commit()

        logger.info("Seeding Jobs...")
        # 3. Jobs
        # Job in Peenya (Industrial area, roughly between them)
        j1 = Job(
            title="Road Construction Peenya",
            description="Laying concrete road in Peenya industrial area",
            required_skills="Digging, Masonry",
            village_id=v3.id, # Located in Nelamangala (for simplicity/proximity)
            status=JobStatus.OPEN,
            wage_per_day=500.0
        )
        db.add(j1)
        db.commit()
        
        logger.info(f"Seeding Complete! Created Job ID: {j1.id} at {v3.name}")

    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
