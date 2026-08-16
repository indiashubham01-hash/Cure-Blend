"""
CureBlend — Database Manager & Offline Data Store
===================================================
Hybrid data store:
  1. Primary: Async MongoDB connection using Motor driver.
  2. Fallback: Local SQLite database (`data/offline_history.db`) for seamless offline persistence.

If MongoDB is unreachable, the system automatically falls back to local SQLite so assessment
history is NEVER lost.
"""

import sys
import json
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Ensure project root (cureblend-backend) is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_COLLECTION_HISTORY, DATA_DIR


# ── Local SQLite Offline Database Path ─────────────────────────
SQLITE_DB_PATH = DATA_DIR / "offline_history.db"


def _get_sqlite_conn():
    """Get a working SQLite connection, recovering automatically if database is malformed."""
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        # Test integrity
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        return conn
    except sqlite3.DatabaseError as e:
        print(f"  [!] SQLite corrupted ({e}). Recreating database file...")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        try:
            if SQLITE_DB_PATH.exists():
                SQLITE_DB_PATH.unlink()
        except Exception:
            pass
        _init_sqlite_db()
        return sqlite3.connect(SQLITE_DB_PATH)


def _init_sqlite_db():
    """Initialize local SQLite database table for offline fallback."""
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_history (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                top_prediction TEXT,
                confidence REAL,
                severity_level TEXT,
                severity_score INTEGER,
                is_emergency INTEGER,
                record_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except sqlite3.DatabaseError as e:
        print(f"  [!] Corrupt SQLite file detected on init ({e}). Rebuilding...")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        try:
            if SQLITE_DB_PATH.exists():
                SQLITE_DB_PATH.unlink()
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessment_history (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    top_prediction TEXT,
                    confidence REAL,
                    severity_level TEXT,
                    severity_score INTEGER,
                    is_emergency INTEGER,
                    record_json TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as ex:
            print(f"  [!] Local SQLite init warning: {ex}")
    except Exception as e:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        print(f"  [!] Local SQLite init warning: {e}")


# Initialize SQLite database file on module import
_init_sqlite_db()


# ══════════════════════════════════════════════════════════════
#  DATABASE CONNECTION MANAGER
# ══════════════════════════════════════════════════════════════

class DatabaseManager:
    """Manages async MongoDB connection lifecycle with local SQLite fallback."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        self._connected = False
    
    async def connect(self):
        """Establish connection to MongoDB Atlas / local MongoDB."""
        try:
            self.client = AsyncIOMotorClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=3000,  # 3 second timeout
                connectTimeoutMS=3000
            )
            
            # Verify connection with a ping
            await self.client.admin.command("ping")
            
            self.db = self.client[MONGODB_DB_NAME]
            self.collection = self.db[MONGODB_COLLECTION_HISTORY]
            
            # Create indexes for efficient queries
            await self.collection.create_index("timestamp", background=True)
            await self.collection.create_index("request_id", unique=True, background=True)
            
            self._connected = True
            print(f"  [OK] MongoDB connected: {MONGODB_DB_NAME}.{MONGODB_COLLECTION_HISTORY}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"  [!] MongoDB connection unavailable: {e}")
            print(f"  [OK] Using local offline data store: {SQLITE_DB_PATH}")
            self._connected = False
        except Exception as e:
            print(f"  [!] MongoDB unexpected error: {e}")
            print(f"  [OK] Using local offline data store: {SQLITE_DB_PATH}")
            self._connected = False
    
    async def disconnect(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            print("  [OK] MongoDB disconnected.")
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# Global singleton
db_manager = DatabaseManager()


# ══════════════════════════════════════════════════════════════
#  CRUD OPERATIONS (MongoDB + Local SQLite Fallback)
# ══════════════════════════════════════════════════════════════

async def save_assessment(record: dict) -> bool:
    """
    Save an assessment record to MongoDB (if online) or local SQLite (if offline).
    """
    # Ensure timestamp is present
    if "timestamp" not in record:
        record["timestamp"] = datetime.now(timezone.utc)
    
    ts = record["timestamp"]
    if isinstance(ts, datetime):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    # 1. Primary: MongoDB (if connected)
    if db_manager.is_connected:
        try:
            db_record = dict(record)
            if isinstance(db_record.get("timestamp"), str):
                db_record["timestamp"] = datetime.fromisoformat(db_record["timestamp"])
            result = await db_manager.collection.insert_one(db_record)
            return result.acknowledged
        except Exception as e:
            print(f"  [!] MongoDB save failed: {e}. Falling back to SQLite.")

    # 2. Fallback: Local SQLite Store
    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        
        # Serialize datetime for JSON
        serializable_record = dict(record)
        serializable_record["timestamp"] = ts_str

        cursor.execute(
            """
            INSERT OR REPLACE INTO assessment_history 
            (request_id, timestamp, top_prediction, confidence, severity_level, severity_score, is_emergency, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("request_id", ""),
                ts_str,
                record.get("top_prediction", ""),
                float(record.get("confidence", 0.0)),
                record.get("severity_level", ""),
                int(record.get("severity_score", 0)),
                1 if record.get("is_emergency") else 0,
                json.dumps(serializable_record)
            )
        )
        conn.commit()
        conn.close()
        print(f"  [OK] Saved to local offline database: {record.get('request_id')}")
        return True

    except Exception as e:
        print(f"  [!] SQLite save failed: {e}")
        return False


async def get_history(
    limit: int = 50,
    skip: int = 0
) -> tuple[list[dict], int]:
    """
    Retrieve assessment history from MongoDB (if online) or local SQLite (if offline).
    """
    # 1. Primary: MongoDB (if connected)
    if db_manager.is_connected:
        try:
            total = await db_manager.collection.count_documents({})
            cursor = db_manager.collection.find({}, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit)
            records = await cursor.to_list(length=limit)
            for r in records:
                if isinstance(r.get("timestamp"), datetime):
                    r["timestamp"] = r["timestamp"].isoformat()
            return records, total
        except Exception as e:
            print(f"  [!] MongoDB fetch failed ({e}). Falling back to SQLite.")

    # 2. Fallback: Local SQLite Store
    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM assessment_history")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT record_json FROM assessment_history ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, skip)
        )
        rows = cursor.fetchall()
        conn.close()

        records = [json.loads(row[0]) for row in rows]
        return records, total

    except Exception as e:
        print(f"  [!] SQLite fetch failed: {e}")
        return [], 0


async def get_assessment_by_id(request_id: str) -> Optional[dict]:
    """Retrieve a single assessment by ID from MongoDB or SQLite."""
    if db_manager.is_connected:
        try:
            record = await db_manager.collection.find_one({"request_id": request_id}, {"_id": 0})
            if record and isinstance(record.get("timestamp"), datetime):
                record["timestamp"] = record["timestamp"].isoformat()
            return record
        except Exception:
            pass

    # Fallback SQLite
    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT record_json FROM assessment_history WHERE request_id = ?", (request_id,))
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


async def delete_history() -> int:
    """Delete all assessment history from both stores."""
    count = 0
    if db_manager.is_connected:
        try:
            res = await db_manager.collection.delete_many({})
            count += res.deleted_count
        except Exception:
            pass

    try:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM assessment_history")
        count += cursor.rowcount
        conn.commit()
        conn.close()
    except Exception:
        pass

    return count


# ══════════════════════════════════════════════════════════════
#  STANDALONE DIAGNOSTIC TEST RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def _test():
        print("=" * 60)
        print("  CureBlend — Database Diagnostic Test")
        print("=" * 60)
        print(f"  Target URI: {MONGODB_URL}")
        print(f"  Database  : {MONGODB_DB_NAME}")
        print(f"  Collection: {MONGODB_COLLECTION_HISTORY}")
        print(f"  SQLite DB : {SQLITE_DB_PATH}")
        print("-" * 60)
        
        print("\n[1] Testing connection...")
        await db_manager.connect()
        
        if db_manager.is_connected:
            print("  -> Status: [CONNECTED] Successfully connected to live MongoDB!")
        else:
            print("  -> Status: [OFFLINE] MongoDB not reachable. Using local SQLite store.")
        
        print("\n[2] Testing save record...")
        test_rec = {
            "request_id": "diag-test-001",
            "timestamp": datetime.now(timezone.utc),
            "input_symptoms": ["headache", "fever"],
            "top_prediction": "Common Cold",
            "confidence": 0.88,
            "severity_level": "Low",
            "severity_score": 25,
            "is_emergency": False,
            "recommendations_summary": "Diagnostic test record"
        }
        res = await save_assessment(test_rec)
        print(f"  -> Save successful: {res}")
        
        print("\n[3] Testing retrieve history...")
        history, total = await get_history(limit=5)
        print(f"  -> Total records in store: {total}")
        print(f"  -> Retrieved records count: {len(history)}")
        
        await db_manager.disconnect()
        print("\n" + "=" * 60)
        print("  Diagnostic check completed.")
        print("=" * 60)

    asyncio.run(_test())

